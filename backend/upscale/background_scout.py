"""Background Scout: the existing Scout scan, run on a schedule while the API is alive.

It adds no discovery logic of its own: each run is exactly the scan a manual Scout refresh
performs (`ScoutFeed.refresh` in `upscale.main`: discovery, tracked-token refresh,
ranking, persistence and the outcome-anchor policy), so observations are anchored only
when the existing policy says so.

Background Scout is the lowest priority user of every shared provider quota:

1. Interactive Analyze
2. Due outcome collection
3. Manual Scout refresh
4. Background Scout

Before each run `defer_reason` is asked why not to run now; any reason defers the run by
`retry_minutes` (never a busy loop). A run is never started while another is running
(single-flight), and a scan that fails is logged and retried at the next scheduled time:
the last good Scout results are kept (`ScoutFeed` never drops them on failure).

Scheduling: the first run waits `startup_delay_minutes` after startup (no request burst
as the backend starts), then every `interval_minutes` from the previous run's start.
"""

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from upscale.services.outcomes.collector import CycleReport
from upscale.services.quota import DEFAULT_LANE
from upscale.services.scout.gate import RequestGate

logger = logging.getLogger("upscale.scout.background")

DEFAULT_INTERVAL_MINUTES = 30.0
# Shorter intervals from the environment are raised to this (a scan is a provider burst).
MIN_INTERVAL_MINUTES = 5.0
STARTUP_DELAY_MINUTES = 5.0
RETRY_MINUTES = 5.0
# A provider that answered "rate limited" this recently is left alone.
RATE_LIMIT_COOLDOWN_MINUTES = 10.0
# On shutdown, a running scan gets this long to finish before it is cancelled.
SHUTDOWN_GRACE_SECONDS = 15.0

_OFF = ("0", "false", "off", "no")


@dataclass(frozen=True)
class BackgroundScoutSettings:
    enabled: bool = True
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES
    startup_delay_minutes: float = STARTUP_DELAY_MINUTES
    retry_minutes: float = RETRY_MINUTES
    rate_limit_cooldown_minutes: float = RATE_LIMIT_COOLDOWN_MINUTES
    shutdown_grace_seconds: float = SHUTDOWN_GRACE_SECONDS


def load_settings(enabled: str | None, interval_minutes: str | None) -> BackgroundScoutSettings:
    """From `UPSCALE_BACKGROUND_SCOUT` (default on; 0 / false / off disables it) and
    `UPSCALE_BACKGROUND_SCOUT_INTERVAL_MINUTES` (default 30, at least 5)."""
    on = (enabled or "1").strip().lower() not in _OFF
    interval = DEFAULT_INTERVAL_MINUTES
    if interval_minutes is not None and interval_minutes.strip():
        try:
            interval = float(interval_minutes)
        except ValueError:
            interval = math.nan
        if not math.isfinite(interval) or interval <= 0:
            logger.warning(
                "invalid background Scout interval %r; using %g minutes",
                interval_minutes,
                DEFAULT_INTERVAL_MINUTES,
            )
            interval = DEFAULT_INTERVAL_MINUTES
        elif interval < MIN_INTERVAL_MINUTES:
            logger.warning(
                "background Scout interval %g minutes is below the minimum; using %g",
                interval,
                MIN_INTERVAL_MINUTES,
            )
            interval = MIN_INTERVAL_MINUTES
    return BackgroundScoutSettings(enabled=on, interval_minutes=interval)


# --- Priority ------------------------------------------------------------------------------


def outcome_backlog(report: CycleReport | None, now: datetime, max_age: timedelta) -> str | None:
    """Due outcome work the collector's latest cycle (if recent) could not measure for
    provider access: Scout must not take that capacity first."""
    if report is None or now - report.at > max_age:
        return None
    if report.deferred_for_quota:
        return (
            f"outcome collection has {report.deferred_for_quota} due measurement(s) "
            "deferred for provider quota"
        )
    if report.network_skipped:
        return "outcome collection has due measurements waiting for provider access"
    return None


def provider_pressure(gates: Sequence[RequestGate], cooldown_seconds: float) -> str | None:
    """A discovery provider that recently answered "rate limited", or whose shared quota
    can't give discovery its full unreserved share right now (other work is using it)."""
    for gate in gates:
        if gate.rate_limited_within(cooldown_seconds):
            return f"{gate.name} rate limiting was detected recently"
    for gate in gates:
        share = gate.limits.calls_per_minute - sum(gate.limits.reservations.values())
        if gate.available(DEFAULT_LANE) < share:
            return f"{gate.name} capacity is in use by other work"
    return None


def defer_reason(
    *,
    analyze_active: bool,
    outcome_backlog: str | None,
    scout_running: bool,
    provider_pressure: str | None,
) -> str | None:
    """Why background Scout should not run now, in priority order (None: run)."""
    if analyze_active:
        return "Analyze is active"
    if outcome_backlog is not None:
        return outcome_backlog
    if scout_running:
        return "a Scout refresh is already running"
    return provider_pressure


# --- The scheduler ----------------------------------------------------------------------------


class ScanSummary(BaseModel):
    """What one completed scan measured (None: not measurable for this run)."""

    candidates_discovered: int | None = None
    candidates_ranked: int | None = None
    new_outcome_anchors: int | None = None


class ScanDeferred(Exception):
    """Raised by a run that decided not to scan after all (e.g. one just finished)."""


class ScanFailed(Exception):
    """Raised by a run whose scan failed (the previous Scout results are kept)."""


class BackgroundRunResult(BaseModel):
    status: Literal["completed", "deferred", "failed"]
    attempted_at: datetime
    finished_at: datetime
    candidates_discovered: int | None = None
    candidates_ranked: int | None = None
    new_outcome_anchors: int | None = None
    deferred: bool = False
    defer_reason: str | None = None
    error: str | None = None


class BackgroundScoutStatus(BaseModel):
    enabled: bool
    interval_minutes: float
    running: bool
    last_run: datetime | None  # when the last scan (not a deferral) started
    next_run: datetime | None
    last_result: BackgroundRunResult | None
    runs_completed: int
    runs_deferred: int
    runs_failed: int


class BackgroundScout:
    """`run`: one scan (raises `ScanDeferred` / `ScanFailed`); `defer`: why not to run
    now (None to run)."""

    def __init__(
        self,
        settings: BackgroundScoutSettings,
        run: Callable[[], Awaitable[ScanSummary]],
        defer: Callable[[], str | None],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.settings = settings
        self._run = run
        self._defer = defer
        self.now = now
        self.running = False
        self.last_run: datetime | None = None
        self.next_run: datetime | None = None
        self.last_result: BackgroundRunResult | None = None
        self.counts = {"completed": 0, "deferred": 0, "failed": 0}
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def status(self) -> BackgroundScoutStatus:
        return BackgroundScoutStatus(
            enabled=self.settings.enabled,
            interval_minutes=self.settings.interval_minutes,
            running=self.running,
            last_run=self.last_run,
            next_run=self.next_run if self.active else None,
            last_result=self.last_result,
            runs_completed=self.counts["completed"],
            runs_deferred=self.counts["deferred"],
            runs_failed=self.counts["failed"],
        )

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def _record(self, result: BackgroundRunResult) -> BackgroundRunResult:
        self.last_result = result
        self.counts[result.status] += 1
        return result

    def _deferred(self, at: datetime, reason: str) -> BackgroundRunResult:
        logger.info("background Scout deferred: %s", reason)
        return self._record(
            BackgroundRunResult(
                status="deferred",
                attempted_at=at,
                finished_at=self.now(),
                deferred=True,
                defer_reason=reason,
            )
        )

    async def run_once(self) -> BackgroundRunResult:
        """One attempt: deferred if anything outranks it, else one scan. Never raises."""
        at = self.now()
        if self.running:  # single-flight: never two background runs at once
            return self._deferred(at, "a background Scout run is already in progress")
        try:
            reason = self._defer()
        except Exception:
            logger.exception("background Scout could not check priorities")
            reason = "priority check failed"
        if reason is not None:
            return self._deferred(at, reason)
        self.running = True
        previous, self.last_run = self.last_run, at
        logger.info("background Scout started")
        try:
            summary = await self._run()
        except ScanDeferred as exc:
            self.last_run = previous  # no scan ran
            return self._deferred(at, str(exc))
        except Exception as exc:  # never takes the app (or the scheduler) down
            error = str(exc) if isinstance(exc, ScanFailed) else type(exc).__name__
            if isinstance(exc, ScanFailed):
                logger.warning("background Scout failed: %s", error)
            else:
                logger.exception("background Scout failed")
            return self._record(
                BackgroundRunResult(
                    status="failed", attempted_at=at, finished_at=self.now(), error=error
                )
            )
        finally:
            self.running = False
        logger.info(
            "background Scout completed: %s discovered, %s ranked, %s new outcome anchor(s)",
            summary.candidates_discovered,
            summary.candidates_ranked,
            summary.new_outcome_anchors,
        )
        return self._record(
            BackgroundRunResult(
                status="completed", attempted_at=at, finished_at=self.now(), **summary.model_dump()
            )
        )

    async def run_forever(self) -> None:
        s = self.settings
        self.next_run = self.now() + timedelta(minutes=s.startup_delay_minutes)
        logger.info(
            "background Scout scheduled every %g minutes; first run at %s",
            s.interval_minutes,
            self.next_run.isoformat(timespec="seconds"),
        )
        while not self._stop.is_set():
            delay = max(0.0, (self.next_run - self.now()).total_seconds())
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break  # stopped
            except TimeoutError:
                pass
            result = await self.run_once()
            after = self.now()
            retry = after + timedelta(minutes=s.retry_minutes)
            if result.status == "deferred":
                self.next_run = retry
            else:  # completed or failed: the next scheduled slot
                self.next_run = max(
                    result.attempted_at + timedelta(minutes=s.interval_minutes), retry
                )

    def start(self) -> None:
        """Start the schedule (no-op when disabled or already started)."""
        if not self.settings.enabled or self.active:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        """Stop the schedule: a running scan gets a short grace to finish, then is
        cancelled (Scout's own writes are individually committed)."""
        self._stop.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), self.settings.shutdown_grace_seconds)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except Exception:
            logger.exception("background Scout stopped with an error")
        self._task = None
        self.next_run = None
