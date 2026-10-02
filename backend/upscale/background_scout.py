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

Before each run the hard reasons (`hard_defer_reason`: Analyze active, a Scout scan already
running) and then the soft ones (`defer_reason`: due outcome work waiting for quota, provider
pressure) are asked why not to run now; any reason defers the run by `retry_minutes` (never
a busy loop). Deferral is bounded: the first soft deferral since the last completed scan
opens a deferral window (`deferred_since`); repeated soft deferrals never move it (they are
counted by a stable reason code, not by their message, whose counts change every cycle),
hard deferrals neither open nor reset it, and only a completed scan closes it. Once the
window is `max_deferral_minutes` old (default 60) the scan is overdue: only hard reasons
can still defer it, and the next attempt they allow runs the scan (a failed scan keeps the
window open, so the next scheduled attempt is overdue too). The window is kept in memory: a
restart opens a new one. Every provider gate
stays hard, outcome work keeps outranking Scout on the shared GeckoTerminal quota, and the
"interactive" reservation is never touched, so a scan under pressure just runs fewer
discovery feeds. Without this bound, a steady outcome backlog on GeckoTerminal (6 calls a
minute, shared) deferred every attempt for hours. A run is never started while another is running
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
from typing import Literal, Self

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
# After this long past the interval without a completed scan, only hard reasons defer it.
MAX_DEFERRAL_MINUTES = 60.0
# On shutdown, a running scan gets this long to finish before it is cancelled.
SHUTDOWN_GRACE_SECONDS = 15.0

_OFF = ("0", "false", "off", "no")

DeferKind = Literal["hard", "soft"]


class DeferReason(str):
    """A human-readable defer message (it may carry counts that change every cycle) with a
    stable `code`: deferral diagnostics count by code."""

    code: str

    def __new__(cls, message: str, code: str) -> Self:
        reason = super().__new__(cls, message)
        reason.code = code
        return reason


def reason_code(reason: str) -> str:
    """The stable code of a defer reason (a plain message is its own code)."""
    return reason.code if isinstance(reason, DeferReason) else reason


ANALYZE_ACTIVE = DeferReason("Analyze is active", "analyze_active")
SCOUT_RUNNING = DeferReason("a Scout refresh is already running", "scout_running")
BACKGROUND_RUNNING = DeferReason(
    "a background Scout run is already in progress", "background_run_in_progress"
)
# Codes that only ever mean "someone is waiting" or "a scan is running": they defer even an
# overdue scan, whichever check reported them.
HARD_CODES = frozenset({ANALYZE_ACTIVE.code, SCOUT_RUNNING.code, BACKGROUND_RUNNING.code})


@dataclass(frozen=True)
class BackgroundScoutSettings:
    enabled: bool = True
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES
    startup_delay_minutes: float = STARTUP_DELAY_MINUTES
    retry_minutes: float = RETRY_MINUTES
    rate_limit_cooldown_minutes: float = RATE_LIMIT_COOLDOWN_MINUTES
    shutdown_grace_seconds: float = SHUTDOWN_GRACE_SECONDS
    max_deferral_minutes: float = MAX_DEFERRAL_MINUTES


def load_settings(
    enabled: str | None, interval_minutes: str | None, max_deferral_minutes: str | None = None
) -> BackgroundScoutSettings:
    """From `UPSCALE_BACKGROUND_SCOUT` (default on; 0 / false / off disables it),
    `UPSCALE_BACKGROUND_SCOUT_INTERVAL_MINUTES` (default 30, at least 5) and
    `UPSCALE_BACKGROUND_SCOUT_MAX_DEFERRAL_MINUTES` (default 60, at least 5)."""
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
    max_deferral = MAX_DEFERRAL_MINUTES
    if max_deferral_minutes is not None and max_deferral_minutes.strip():
        try:
            max_deferral = max(float(max_deferral_minutes), MIN_INTERVAL_MINUTES)
        except ValueError:
            max_deferral = MAX_DEFERRAL_MINUTES
        if not math.isfinite(max_deferral):
            max_deferral = MAX_DEFERRAL_MINUTES
    return BackgroundScoutSettings(
        enabled=on, interval_minutes=interval, max_deferral_minutes=max_deferral
    )


# --- Priority ------------------------------------------------------------------------------


def outcome_backlog(
    report: CycleReport | None, now: datetime, max_age: timedelta
) -> DeferReason | None:
    """Due outcome work the collector's latest cycle (if recent) could not measure for
    provider access: Scout must not take that capacity first."""
    if report is None or now - report.at > max_age:
        return None
    if report.deferred_for_quota:
        return DeferReason(
            f"outcome collection has {report.deferred_for_quota} due measurement(s) "
            "deferred for provider quota",
            "outcome_backlog_quota",
        )
    if report.network_skipped:
        return DeferReason(
            "outcome collection has due measurements waiting for provider access",
            "outcome_backlog_waiting",
        )
    return None


def provider_pressure(gates: Sequence[RequestGate], cooldown_seconds: float) -> DeferReason | None:
    """A discovery provider that recently answered "rate limited", or whose shared quota
    can't give discovery its full unreserved share right now (other work is using it)."""
    for gate in gates:
        if gate.rate_limited_within(cooldown_seconds):
            return DeferReason(
                f"{gate.name} rate limiting was detected recently",
                f"provider_rate_limited:{gate.name}",
            )
    for gate in gates:
        share = gate.limits.calls_per_minute - sum(gate.limits.reservations.values())
        if gate.available(DEFAULT_LANE) < share:
            return DeferReason(
                f"{gate.name} capacity is in use by other work", f"provider_capacity:{gate.name}"
            )
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
        return ANALYZE_ACTIVE
    if outcome_backlog is not None:
        return outcome_backlog
    if scout_running:
        return SCOUT_RUNNING
    return provider_pressure


def hard_defer_reason(*, analyze_active: bool, scout_running: bool) -> str | None:
    """The reasons that still defer an overdue scan: someone is waiting, or a scan runs."""
    if analyze_active:
        return ANALYZE_ACTIVE
    if scout_running:
        return SCOUT_RUNNING
    return None


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
    overdue: bool = False  # past the deferral bound: only hard reasons were asked
    candidates_discovered: int | None = None
    candidates_ranked: int | None = None
    new_outcome_anchors: int | None = None
    deferred: bool = False
    defer_reason: str | None = None  # human-readable (may carry changing counts)
    defer_code: str | None = None  # stable code (what diagnostics count by)
    defer_kind: DeferKind | None = None  # hard: defers even an overdue scan
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
    # Reliability diagnostics (in memory: they restart with the process).
    process_started_at: datetime | None = None
    background_scans_started: int = 0
    background_scans_completed: int = 0
    background_scans_failed: int = 0
    last_started_at: datetime | None = None
    last_completed_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_failure_reason: str | None = None
    last_scan_duration_seconds: float | None = None
    longest_scan_duration_seconds: float | None = None
    next_due_at: datetime | None = None
    deferrals_by_reason: dict[str, int] = {}  # by stable reason code
    consecutive_deferrals: int = 0
    # Start of the deferral window: the first soft deferral since the last completed scan
    # (repeated soft and any hard deferrals keep it; a completed scan clears it).
    deferred_since: datetime | None = None
    overdue_at: datetime | None = None  # deferred_since + max_deferral_minutes
    overdue: bool = False  # now past overdue_at: only hard reasons can still defer
    last_defer_kind: DeferKind | None = None
    last_defer_code: str | None = None
    longest_gap_between_completed_minutes: float | None = None
    overdue_runs: int = 0  # forced (overdue) scans actually started
    max_deferral_minutes: float = MAX_DEFERRAL_MINUTES


class BackgroundScout:
    """`run`: one scan (raises `ScanDeferred` / `ScanFailed`); `defer`: why not to run
    now (None to run)."""

    def __init__(
        self,
        settings: BackgroundScoutSettings,
        run: Callable[[], Awaitable[ScanSummary]],
        defer: Callable[[], str | None],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        hard_defer: Callable[[], str | None] | None = None,
    ):
        self.settings = settings
        self._run = run
        self._defer = defer
        self._hard_defer = hard_defer
        self.now = now
        self.started_at = now()
        self.started = 0
        self.last_completed_at: datetime | None = None
        self.last_failure_at: datetime | None = None
        self.last_failure_reason: str | None = None
        self.last_duration: float | None = None
        self.longest_duration: float | None = None
        self.longest_gap: float | None = None
        self.deferrals: dict[str, int] = {}
        self.consecutive_deferrals = 0
        self.deferred_since: datetime | None = None
        self.last_defer_kind: DeferKind | None = None
        self.last_defer_code: str | None = None
        self.overdue_runs = 0
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
            process_started_at=self.started_at,
            background_scans_started=self.started,
            background_scans_completed=self.counts["completed"],
            background_scans_failed=self.counts["failed"],
            last_started_at=self.last_run,
            last_completed_at=self.last_completed_at,
            last_failure_at=self.last_failure_at,
            last_failure_reason=self.last_failure_reason,
            last_scan_duration_seconds=self.last_duration,
            longest_scan_duration_seconds=self.longest_duration,
            next_due_at=self.next_run if self.active else None,
            deferrals_by_reason=dict(self.deferrals),
            consecutive_deferrals=self.consecutive_deferrals,
            deferred_since=self.deferred_since,
            overdue_at=self._overdue_at(),
            overdue=self.overdue(self.now()),
            last_defer_kind=self.last_defer_kind,
            last_defer_code=self.last_defer_code,
            longest_gap_between_completed_minutes=self.longest_gap,
            overdue_runs=self.overdue_runs,
            max_deferral_minutes=self.settings.max_deferral_minutes,
        )

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def _record(self, result: BackgroundRunResult) -> BackgroundRunResult:
        self.last_result = result
        self.counts[result.status] += 1
        if result.status == "deferred":
            self.consecutive_deferrals += 1
            self.last_defer_kind, self.last_defer_code = result.defer_kind, result.defer_code
            if result.defer_kind == "soft" and self.deferred_since is None:
                self.deferred_since = result.attempted_at  # opens the deferral window
        else:  # a scan started
            self.consecutive_deferrals = 0
            self.last_defer_kind = self.last_defer_code = None
            if result.status == "completed":  # only a completed scan closes the window
                self.deferred_since = None
        return result

    def _overdue_at(self) -> datetime | None:
        if self._hard_defer is None or self.deferred_since is None:
            return None  # no bound without a hard check; no window without a soft deferral
        return self.deferred_since + timedelta(minutes=self.settings.max_deferral_minutes)

    def overdue(self, at: datetime) -> bool:
        """Soft reasons have deferred the scan for max_deferral_minutes (counted from the
        first soft deferral since the last completed scan)."""
        deadline = self._overdue_at()
        return deadline is not None and at >= deadline

    def _deferred(
        self, at: datetime, reason: str, kind: DeferKind, overdue: bool = False
    ) -> BackgroundRunResult:
        code = reason_code(reason)
        if code in HARD_CODES:
            kind = "hard"
        self.deferrals[code] = self.deferrals.get(code, 0) + 1
        logger.info("background Scout deferred (%s): %s", kind, reason)
        return self._record(
            BackgroundRunResult(
                status="deferred",
                attempted_at=at,
                finished_at=self.now(),
                overdue=overdue,
                deferred=True,
                defer_reason=str(reason),
                defer_code=code,
                defer_kind=kind,
            )
        )

    def _check(self, at: datetime) -> tuple[str, DeferKind] | None:
        """Why not to scan now: hard reasons always; soft ones until the scan is overdue."""
        try:
            hard = self._hard_defer() if self._hard_defer is not None else None
        except Exception:
            logger.exception("background Scout could not check hard priorities")
            return DeferReason("priority check failed", "priority_check_failed"), "hard"
        if hard is not None:
            return hard, "hard"
        try:
            soft = self._defer()
        except Exception:
            logger.exception("background Scout could not check priorities")
            soft = DeferReason("priority check failed", "priority_check_failed")
        if soft is None:
            return None
        if self.overdue(at):
            logger.warning(
                "background Scout overdue (soft deferrals since %s): bypassing %s",
                self.deferred_since.isoformat() if self.deferred_since else None,
                soft,
            )
            return None
        return soft, "soft"

    async def run_once(self) -> BackgroundRunResult:
        """One attempt: deferred if anything outranks it, else one scan. Never raises."""
        at = self.now()
        overdue = self.overdue(at)
        if self.running:  # single-flight: never two background runs at once
            return self._deferred(at, BACKGROUND_RUNNING, "hard", overdue)
        check = self._check(at)
        if check is not None:
            return self._deferred(at, *check, overdue)
        if overdue:
            self.overdue_runs += 1  # a forced scan actually starts
        self.running = True
        self.started += 1
        previous, self.last_run = self.last_run, at
        logger.info("background Scout started")
        try:
            summary = await self._run()
        except ScanDeferred as exc:
            self.last_run = previous  # no scan ran
            self.started -= 1
            if overdue:
                self.overdue_runs -= 1
            return self._deferred(at, str(exc), "hard", overdue)
        except Exception as exc:  # never takes the app (or the scheduler) down
            error = str(exc) if isinstance(exc, ScanFailed) else type(exc).__name__
            if isinstance(exc, ScanFailed):
                logger.warning("background Scout failed: %s", error)
            else:
                logger.exception("background Scout failed")
            self.last_failure_at, self.last_failure_reason = self.now(), error
            self._duration(at)
            return self._record(
                BackgroundRunResult(
                    status="failed",
                    attempted_at=at,
                    finished_at=self.now(),
                    error=error,
                    overdue=overdue,
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
        done = self.now()
        self._duration(at)
        if self.last_completed_at is not None:
            gap = (done - self.last_completed_at).total_seconds() / 60
            self.longest_gap = max(self.longest_gap or 0.0, gap)
        self.last_completed_at = done
        return self._record(
            BackgroundRunResult(
                status="completed",
                attempted_at=at,
                finished_at=done,
                overdue=overdue,
                **summary.model_dump(),
            )
        )

    def _duration(self, started: datetime) -> None:
        self.last_duration = (self.now() - started).total_seconds()
        self.longest_duration = max(self.longest_duration or 0.0, self.last_duration)

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
