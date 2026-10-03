"""Background Shadow: the shadow engine, run on a schedule while the API is alive.

Off unless ``UPSCALE_SHADOW=1``. It makes no provider request at all (it only reads the
Evidence Archive and writes the shadow database), and it is still the lowest-priority
background work: a step is deferred while Analyze is active or a Scout scan (manual or
background) is running, so it never competes with them for CPU or disk. A failing step is
logged and retried at the next slot; nothing it does can raise into Scout or Analyze.

Runs: single-run mode (``UPSCALE_SHADOW_RUN``, unchanged) creates the run on first use and
advances it. Multi-run mode (``UPSCALE_SHADOW_RUNS=continuous-v2,continuous-v2-realistic``)
advances every listed EXISTING run, one after the other in the listed order, each with its
own connections, checkpoint and frozen settings (execution model, availability policy,
start); it never creates or redefines a run, and one run failing never stops the others.
The engine refuses to advance a run another step is already advancing (in this process,
and across processes through the checkpoint's cursor).

Diagnostics storage follows ``UPSCALE_SHADOW_DIAGNOSTICS_DETAIL`` (default ``sampled``:
exact aggregate counters plus a bounded sample of rejection rows, never one row per
rejected candidate) and the opt-in ``UPSCALE_SHADOW_REJECTION_RETENTION_DAYS``.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from upscale.services.evidence_archive.store import EvidenceStore
from upscale.services.shadow.config import ShadowSettings, default_evidence_db, default_shadow_db
from upscale.services.shadow.engine import ShadowEngine, ShadowError
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


def advance(
    run_id: str,
    shadow_db: str,
    evidence_db: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Multi-run mode: one processing step of an EXISTING run (never created here, so its
    execution model, availability policy and start can never change silently)."""
    store = ShadowStore(shadow_db)
    evidence = EvidenceStore(evidence_db, read_only=True)
    try:
        if store.run(run_id) is None:
            raise ShadowError(
                f"run {run_id} does not exist: create it with `python -m "
                "upscale.services.shadow run` first (multi-run mode never creates a run)"
            )
        return ShadowEngine(store, evidence, now=now).run(run_id)
    finally:
        store.close()
        evidence.close()


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "events": report.get("events"),
        "actions": report.get("actions"),
        "fills": report.get("fills"),
        "diagnostics_detail": report.get("diagnostics_detail"),
        "rejection_rows_stored": report.get("rejection_rows_stored"),
        "rejection_rows_expired": report.get("rejection_rows_expired"),
        "processed_until": str(report.get("until")),
    }


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
        self.counts = {"completed": 0, "deferred": 0, "failed": 0, "partial": 0}
        self.runs: dict[str, dict[str, Any]] = {
            r: self._blank() for r in (settings.run_ids or (settings.run_id,))
        }
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    def _blank() -> dict[str, Any]:
        return {"last_success_at": None, "last_failure_at": None, "last_error": None,
                "last_result": None, "advances": 0, "failures": 0}  # fmt: skip

    @property
    def multi(self) -> bool:
        return bool(self.settings.run_ids)

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        s = self.settings
        return {
            "enabled": s.enabled,
            "mode": "multi-run" if self.multi else "single-run",
            "run_id": s.run_id if not self.multi else None,
            "runs_configured": list(s.run_ids or (s.run_id,)),
            "ignored_run_ids": list(s.ignored_run_ids),
            "runs": {r: dict(v) for r, v in self.runs.items()},
            "since": s.since.isoformat() if not self.multi else None,
            "interval_minutes": s.interval_minutes,
            "running": self.running,
            "next_run": self.next_run.isoformat() if self.next_run and self.active else None,
            "last_result": self.last_result,
            "runs_completed": self.counts["completed"],
            "runs_deferred": self.counts["deferred"],
            "runs_failed": self.counts["failed"],
        }

    def _run_result(
        self, run_id: str, ok: bool, at: datetime, detail: dict[str, Any] | str
    ) -> None:
        r = self.runs.setdefault(run_id, self._blank())
        if ok:
            r["last_success_at"] = at.isoformat()
            r["last_result"] = detail
            r["advances"] += 1
        else:
            r["last_failure_at"] = at.isoformat()
            r["last_error"] = detail
            r["failures"] += 1

    def _advance_all(self) -> list[tuple[str, dict[str, Any] | None, str | None]]:
        """Every configured run in order, each isolated (runs in a worker thread)."""
        out: list[tuple[str, dict[str, Any] | None, str | None]] = []
        for run_id in self.settings.run_ids:
            try:
                report = advance(run_id, self._shadow_db(), self._evidence_db(), now=self.now)
                out.append((run_id, report, None))
            except Exception as exc:  # one run never stops the others
                logger.exception("background Shadow run %s failed", run_id)
                out.append((run_id, None, f"{type(exc).__name__}: {exc}"))
        return out

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
        if self.multi:
            try:
                results = await asyncio.to_thread(self._advance_all)
            finally:
                self.running = False
            per_run: dict[str, Any] = {}
            for run_id, report, error in results:
                if report is not None:
                    per_run[run_id] = {"status": "completed", **_summary(report)}
                    self._run_result(run_id, True, at, per_run[run_id])
                else:
                    per_run[run_id] = {"status": "failed", "error": error}
                    self._run_result(run_id, False, at, error or "")
            failed = sum(1 for _, r, _ in results if r is None)
            status = (
                "completed" if not failed else "failed" if failed == len(results) else "partial"
            )
            logger.info("background Shadow %s: %s", status, per_run)
            return self._record(status, at, runs=per_run)
        try:
            report = await asyncio.to_thread(
                step, self.settings, self._shadow_db(), self._evidence_db()
            )
        except Exception as exc:  # never takes the app (or the scheduler) down
            logger.exception("background Shadow failed")
            self._run_result(self.settings.run_id, False, at, f"{type(exc).__name__}: {exc}")
            return self._record("failed", at, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.running = False
        summary = _summary(report)
        logger.info("background Shadow completed: %s", summary)
        self._run_result(self.settings.run_id, True, at, summary)
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
