"""Background Shadow: the shadow engine, run on a schedule while the API is alive.

Off unless ``UPSCALE_SHADOW=1``. It makes no provider request at all (it only reads the
Evidence Archive and writes the shadow database), and it is still the lowest-priority
background work: a step is deferred while Analyze is active or a Scout scan (manual or
background) is running, so it never competes with them for CPU or disk. A failing step is
logged and retried at the next slot; nothing it does can raise into Scout or Analyze.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from upscale.services.evidence_archive.store import EvidenceStore
from upscale.services.shadow.config import ShadowSettings, default_evidence_db, default_shadow_db
from upscale.services.shadow.engine import ShadowEngine
from upscale.services.shadow.store import ShadowStore

logger = logging.getLogger("upscale.shadow")


def step(
    settings: ShadowSettings,
    shadow_db: str,
    evidence_db: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """One processing step with its own connections (runs in a worker thread)."""
    store = ShadowStore(shadow_db)
    evidence = EvidenceStore(evidence_db, read_only=True)
    try:
        engine = ShadowEngine(store, evidence, now=now)
        engine.init_baselines()  # idempotent: registers v1 baselines, never changes them
        engine.ensure_run(settings.run_id, settings.since, args={"source": "background"})
        return engine.run(settings.run_id)
    finally:
        store.close()
        evidence.close()


class BackgroundShadow:
    def __init__(
        self,
        settings: ShadowSettings,
        defer: Callable[[], str | None],
        shadow_db: Callable[[], str] = default_shadow_db,
        evidence_db: Callable[[], str] = default_evidence_db,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.settings = settings
        self._defer = defer
        self._shadow_db = shadow_db
        self._evidence_db = evidence_db
        self.now = now
        self.running = False
        self.last_result: dict[str, Any] | None = None
        self.next_run: datetime | None = None
        self.counts = {"completed": 0, "deferred": 0, "failed": 0}
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        s = self.settings
        return {
            "enabled": s.enabled,
            "run_id": s.run_id,
            "since": s.since.isoformat(),
            "interval_minutes": s.interval_minutes,
            "running": self.running,
            "next_run": self.next_run.isoformat() if self.next_run and self.active else None,
            "last_result": self.last_result,
            "runs_completed": self.counts["completed"],
            "runs_deferred": self.counts["deferred"],
            "runs_failed": self.counts["failed"],
        }

    def _record(self, status: str, at: datetime, **extra: Any) -> dict[str, Any]:
        self.counts[status] += 1
        self.last_result = {
            "status": status,
            "attempted_at": at.isoformat(),
            "finished_at": self.now().isoformat(),
            **extra,
        }
        return self.last_result

    async def run_once(self) -> dict[str, Any]:
        at = self.now()
        if self.running:
            return self._record("deferred", at, defer_reason="a Shadow step is already running")
        try:
            reason = self._defer()
        except Exception:
            logger.exception("background Shadow could not check priorities")
            reason = "priority check failed"
        if reason is not None:
            logger.info("background Shadow deferred: %s", reason)
            return self._record("deferred", at, defer_reason=reason)
        self.running = True
        try:
            report = await asyncio.to_thread(
                step, self.settings, self._shadow_db(), self._evidence_db()
            )
        except Exception as exc:  # never takes the app (or the scheduler) down
            logger.exception("background Shadow failed")
            return self._record("failed", at, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.running = False
        summary = {
            "events": report.get("events"),
            "actions": report.get("actions"),
            "fills": report.get("fills"),
            "processed_until": str(report.get("until")),
        }
        logger.info("background Shadow completed: %s", summary)
        return self._record("completed", at, **summary)

    async def run_forever(self) -> None:
        s = self.settings
        self.next_run = self.now() + timedelta(minutes=s.startup_delay_minutes)
        while not self._stop.is_set():
            delay = max(0.0, (self.next_run - self.now()).total_seconds())
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except TimeoutError:
                pass
            result = await self.run_once()
            wait = s.retry_minutes if result["status"] == "deferred" else s.interval_minutes
            self.next_run = self.now() + timedelta(minutes=wait)

    def start(self) -> None:
        """Start the schedule (no-op unless enabled; never started twice)."""
        if not self.settings.enabled or self.active:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self.run_forever())

    async def stop(self, grace_seconds: float = 15.0) -> None:
        self._stop.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), grace_seconds)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except Exception:
            logger.exception("background Shadow stopped with an error")
        self._task = None
        self.next_run = None
