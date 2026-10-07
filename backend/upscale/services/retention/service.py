"""Background retention: one bounded pass every `interval_hours` while the API is alive.

Off unless ``UPSCALE_RETENTION_ENABLED`` is set (``dry-run`` only logs; ``1`` deletes). It
is the lowest-priority background work: a pass waits while Analyze, a Scout scan, a Shadow
step, the held-position watch or safety enrichment is running, runs in a worker thread
with its own connections, holds the write lock for one small batch at a time, and stops
after `max_seconds` per database (the rest waits for the next pass). A failing pass is
logged and retried at the next slot; nothing it does can raise into Scout, the Evidence
Archive, Shadow or the API.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from upscale.services.retention import engine
from upscale.services.retention.config import RetentionSettings

logger = logging.getLogger("upscale.retention")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": report["mode"],
        **report["totals"],
        "tables": {
            f"{t['database']}.{t['table']}": {
                k: t[k] for k in ("cutoff", "eligible", "deleted", "skipped", "incomplete", "error")
            }
            for d in report["databases"].values()
            for t in d.get("tables", [])
        },
    }


class BackgroundRetention:
    def __init__(
        self,
        settings: RetentionSettings,
        defer: Callable[[], str | None],
        scout_db: Callable[[], str],
        evidence_db: Callable[[], str],
        shadow_db: Callable[[], str],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.settings = settings
        self._defer = defer
        self._scout_db = scout_db
        self._evidence_db = evidence_db
        self._shadow_db = shadow_db
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
        return {
            "mode": self.settings.mode,
            "policy": self.settings.policy(),
            "running": self.running,
            "next_run": self.next_run.isoformat() if self.next_run and self.active else None,
            "last_result": self.last_result,
            **{f"runs_{k}": v for k, v in self.counts.items()},
        }

    def _pass(self) -> dict[str, Any]:
        return engine.run(
            self.settings,
            self._scout_db(),
            self._evidence_db(),
            self._shadow_db(),
            dry_run=self.settings.mode != "on",
            now=self.now(),
            budget_seconds=self.settings.max_seconds,
        )

    def _record(self, status: str, at: datetime, **extra: Any) -> dict[str, Any]:
        self.counts[status] += 1
        self.last_result = {"status": status, "attempted_at": at.isoformat(),
                            "finished_at": self.now().isoformat(), **extra}  # fmt: skip
        return self.last_result

    async def run_once(self) -> dict[str, Any]:
        at = self.now()
        if self.running:
            return self._record("deferred", at, defer_reason="a retention pass is running")
        try:
            reason = self._defer()
        except Exception:
            logger.exception("retention could not check priorities")
            reason = "priority check failed"
        if reason is not None:
            logger.info("retention deferred: %s", reason)
            return self._record("deferred", at, defer_reason=reason)
        self.running = True
        try:
            report = await asyncio.to_thread(self._pass)
        except Exception as exc:  # never takes the app (or the scheduler) down
            logger.exception("retention pass failed")
            return self._record("failed", at, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.running = False
        summary = _summary(report)
        logger.info("retention %s: %s", report["mode"], summary)
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
            try:
                result = await self.run_once()
            except Exception:  # run_once never raises; belt and braces for the loop
                logger.exception("retention scheduler error")
                result = {"status": "failed"}
            wait = (
                timedelta(minutes=s.retry_minutes)
                if result["status"] == "deferred"
                else timedelta(hours=s.interval_hours)
            )
            self.next_run = self.now() + wait

    def start(self) -> None:
        """Start the schedule (no-op when off; never started twice)."""
        if self.settings.mode == "off" or self.active:
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
            logger.exception("retention stopped with an error")
        self._task = None
        self.next_run = None
