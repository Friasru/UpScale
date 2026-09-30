"""The recorder installed behind `hooks.emit`: serialize in the caller (cheap), write in a
background thread (never blocking production), de-duplicate repeats.

* Nothing is recorded while a replay clock is frozen (`services.clock.frozen_now`):
  historical replay must never write "evidence" about the past.
* The queue is bounded: when it is full a record is dropped and counted, production is
  never slowed down.
* De-duplication: a record identical (same fingerprint: payload without volatile
  timestamps) to the latest one of its key (kind, asset, pool, provider) within the kind's
  window is skipped; any change of state is stored, and an unchanged state is stored again
  once the window has passed (so persistence of a state stays visible in the history).
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from upscale.services.clock import frozen_time
from upscale.services.evidence_archive import payloads
from upscale.services.evidence_archive.store import EvidenceStore, Kind, PendingRecord

logger = logging.getLogger("upscale.evidence")

DEDUPE_SECONDS: dict[Kind, float] = {
    "market": 60.0,
    "dex_market": 60.0,
    "safety": 600.0,  # the safety service re-analyzes its cached chain data (120 s) often
    "social": 300.0,
    "scout": 0.0,  # one record per ranking run
    "decision": 0.0,  # every Analyze
}


@dataclass
class RecorderStats:
    queued: int = 0
    written: int = 0
    deduplicated: int = 0
    dropped: int = 0
    refused_replay: int = 0
    errors: int = 0
    last_error: str | None = None
    last_write_at: datetime | None = None
    by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "queued": self.queued,
            "written": self.written,
            "deduplicated": self.deduplicated,
            "dropped_queue_full": self.dropped,
            "refused_during_replay": self.refused_replay,
            "write_errors": self.errors,
            "last_error": self.last_error,
            "last_successful_write": self.last_write_at.isoformat() if self.last_write_at else None,
            "written_by_kind": dict(self.by_kind),
        }


class EvidenceRecorder:
    def __init__(
        self,
        store: EvidenceStore,
        versions: Callable[[], dict[str, str]] = dict,
        capabilities: Callable[[], list[str]] = list,
        max_queue: int = 10_000,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.store = store
        self._versions = versions
        self._capabilities = capabilities
        self._queue: queue.Queue[PendingRecord] = queue.Queue(maxsize=max_queue)
        self._now = now
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._versions_cache: dict[str, str] | None = None
        self.stats = RecorderStats()

    # --- the hook side (caller's thread) --------------------------------------------------------

    def submit(self, kind: str, obj: Any, component: str, extra: dict[str, Any]) -> None:
        if frozen_time() is not None:
            self.stats.refused_replay += 1
            return
        records = self._convert(kind, obj, extra)
        if not records:
            return
        versions = self.versions()
        for r in records:
            r.component = component
            r.versions = versions
            try:
                self._queue.put_nowait(r)
                self.stats.queued += 1
            except queue.Full:
                self.stats.dropped += 1
        self._ensure_thread()

    def versions(self) -> dict[str, str]:
        if self._versions_cache is None:
            self._versions_cache = dict(self._versions())
        return self._versions_cache

    def _convert(self, kind: str, obj: Any, extra: dict[str, Any]) -> list[PendingRecord]:
        if kind == "market":
            return payloads.market(obj)
        if kind == "dex_market":
            return payloads.dex_market(obj)
        if kind == "safety":
            return payloads.safety(obj, list(extra.get("pools") or []))
        if kind == "safety_failed":
            return payloads.safety_failed(
                extra.get("chain", "solana"),
                extra["address"],
                extra.get("provider"),
                obj,
                self._now(),
            )
        if kind == "social":
            return payloads.social(obj)
        if kind == "scout":
            return payloads.scout(obj, self._capabilities())
        if kind == "decision":
            return payloads.decision(
                extra["request"], obj, extra.get("observation"), extra.get("at") or self._now(),
                self._capabilities(),
            )  # fmt: skip
        raise ValueError(f"unknown evidence kind {kind!r}")

    # --- the writer thread ------------------------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._work, name="upscale-evidence-writer", daemon=True
                )
                self._thread.start()

    def _work(self) -> None:
        while True:
            r = self._queue.get()
            try:
                if self.store.append(r, DEDUPE_SECONDS.get(r.kind, 0.0)):
                    self.stats.written += 1
                    self.stats.by_kind[r.kind] = self.stats.by_kind.get(r.kind, 0) + 1
                    self.stats.last_write_at = datetime.now(UTC)
                else:
                    self.stats.deduplicated += 1
            except Exception as exc:  # a bad record never stops the writer
                self.stats.errors += 1
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("evidence archive: write failed")
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until everything queued so far is written (tests, shutdown)."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.005)
        return True
