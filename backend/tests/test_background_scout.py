"""Background Scout: the existing Scout scan on a schedule, lowest priority on every quota.

Offline: scans are fakes or the synthetic rankings of test_scout_api; outcome anchors go to
the temporary store conftest installs (never ~/.upscale/scout.sqlite3).
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

import upscale.main
import upscale.services as services
from tests.test_outcomes import later_run
from tests.test_scout_api import ranking
from upscale.background_scout import (
    ANALYZE_ACTIVE,
    DEFAULT_INTERVAL_MINUTES,
    MIN_INTERVAL_MINUTES,
    SCOUT_RUNNING,
    STARTUP_DELAY_MINUTES,
    BackgroundScout,
    BackgroundScoutSettings,
    DeferReason,
    ScanDeferred,
    ScanFailed,
    ScanSummary,
    defer_reason,
    load_settings,
    outcome_backlog,
    provider_pressure,
    reason_code,
)
from upscale.scout_api import ScoutFeed
from upscale.services.geckoterminal import DexCandleService
from upscale.services.market_data import MarketDataUnavailableError, ProviderRateLimitedError
from upscale.services.outcomes.collector import CycleReport
from upscale.services.quota import LaneLimiter
from upscale.services.scout.config import ScoutConfig, ScoutProviderLimits
from upscale.services.scout.gate import RateLimitReachedError, RequestGate
from upscale.services.scout.growth.models import GrowthScoutResult
from upscale.services.scout.providers import GeckoTerminalDiscoveryProvider

T0 = datetime(2026, 9, 1, 12, tzinfo=UTC)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def fast(**changes: Any) -> BackgroundScoutSettings:
    """Settings in fractions of a second (minutes / 60 -> seconds)."""
    base: dict[str, Any] = {
        "startup_delay_minutes": 0.05 / 60,
        "interval_minutes": 0.1 / 60,
        "retry_minutes": 0.05 / 60,
        "shutdown_grace_seconds": 0.5,
    }
    base.update(changes)
    return BackgroundScoutSettings(**base)


class FakeScan:
    """A scan that counts calls; `outcomes` (per call) may be exceptions; `gate` blocks."""

    def __init__(self, *outcomes: ScanSummary | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def __call__(self) -> ScanSummary:
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        outcome = self.outcomes.pop(0) if self.outcomes else ScanSummary()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def until(check: Callable[[], bool], timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not check():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def gate(
    per_minute: int = 6, reservations: dict[str, int] | None = None, clock: Any = None
) -> RequestGate:
    limits = ScoutProviderLimits(calls_per_minute=per_minute, reservations=reservations or {})
    return RequestGate("GeckoTerminal", limits, clock=clock or (lambda: 0.0))


# --- Settings (environment) -----------------------------------------------------------------


def test_enabled_by_default_with_a_30_minute_interval() -> None:
    s = load_settings(None, None)
    assert s.enabled is True and s.interval_minutes == DEFAULT_INTERVAL_MINUTES == 30
    assert load_settings("1", "").enabled is True
    assert BackgroundScoutSettings().startup_delay_minutes == STARTUP_DELAY_MINUTES >= 2


@pytest.mark.parametrize("raw", ["0", "false", "OFF", " no "])
def test_disabled_by_environment(raw: str) -> None:
    assert load_settings(raw, None).enabled is False


def test_custom_interval_and_its_safety_floor(caplog: pytest.LogCaptureFixture) -> None:
    assert load_settings("1", "45").interval_minutes == 45
    assert load_settings("1", "7.5").interval_minutes == 7.5
    with caplog.at_level(logging.WARNING):
        assert load_settings("1", "1").interval_minutes == MIN_INTERVAL_MINUTES
        assert load_settings("1", "abc").interval_minutes == DEFAULT_INTERVAL_MINUTES
        assert load_settings("1", "-3").interval_minutes == DEFAULT_INTERVAL_MINUTES
        assert load_settings("1", "nan").interval_minutes == DEFAULT_INTERVAL_MINUTES
    assert len(caplog.records) == 4


# --- Scheduling --------------------------------------------------------------------------------


def test_disabled_scheduler_never_runs() -> None:
    async def go() -> None:
        scan = FakeScan()
        bg = BackgroundScout(fast(enabled=False), scan, lambda: None)
        bg.start()
        await asyncio.sleep(0.3)
        assert scan.calls == 0 and not bg.active
        status = bg.status()
        assert status.enabled is False and status.next_run is None and status.last_run is None
        await bg.stop()

    run(go())


def test_enabled_scheduler_runs_on_its_interval() -> None:
    async def go() -> None:
        scan = FakeScan()
        bg = BackgroundScout(fast(), scan, lambda: None)
        bg.start()
        await until(lambda: scan.calls >= 3)
        await bg.stop()
        assert bg.counts["completed"] >= 3 and bg.counts["failed"] == 0
        assert not bg.active

    run(go())


def test_first_run_waits_for_the_startup_delay() -> None:
    async def go() -> None:
        scan = FakeScan()
        start = datetime.now(UTC)
        bg = BackgroundScout(BackgroundScoutSettings(), scan, lambda: None)
        bg.start()
        await asyncio.sleep(0.2)  # the default delay is minutes: nothing may run yet
        assert scan.calls == 0
        assert bg.next_run is not None
        wait = (bg.next_run - start).total_seconds()
        assert STARTUP_DELAY_MINUTES * 60 - 1 <= wait <= STARTUP_DELAY_MINUTES * 60 + 1
        await bg.stop()

        scan = FakeScan()
        bg = BackgroundScout(fast(startup_delay_minutes=0.3 / 60), scan, lambda: None)
        bg.start()
        await asyncio.sleep(0.15)
        assert scan.calls == 0  # still within the (short) startup delay
        await until(lambda: scan.calls == 1)
        await bg.stop()

    run(go())


def test_next_run_is_one_interval_after_the_previous_start() -> None:
    async def go() -> None:
        scan = FakeScan()
        bg = BackgroundScout(fast(interval_minutes=10), scan, lambda: None)
        bg.start()
        await until(lambda: scan.calls == 1 and bg.last_result is not None)
        assert bg.last_run is not None and bg.next_run is not None
        assert bg.next_run - bg.last_run == timedelta(minutes=10)
        await bg.stop()

    run(go())


def test_deferred_run_retries_later_without_busy_looping() -> None:
    async def go() -> None:
        scan = FakeScan()
        reasons: list[str | None] = ["Analyze is active"] * 3
        checks = {"n": 0}

        def defer() -> str | None:
            checks["n"] += 1
            return reasons.pop(0) if reasons else None

        bg = BackgroundScout(fast(retry_minutes=0.05 / 60), scan, defer)
        bg.start()
        await until(lambda: scan.calls == 1)
        await bg.stop()
        assert checks["n"] == 4 and bg.counts["deferred"] == 3

        # A realistic retry delay: one check, then nothing for minutes.
        checks["n"] = 0
        reasons[:] = ["Analyze is active"] * 100
        bg = BackgroundScout(fast(retry_minutes=5), scan, defer)
        bg.start()
        await asyncio.sleep(0.3)
        assert checks["n"] == 1
        assert bg.next_run is not None
        assert bg.next_run - datetime.now(UTC) > timedelta(minutes=4)
        await bg.stop()

    run(go())


def test_single_flight_never_two_background_runs() -> None:
    async def go() -> None:
        scan = FakeScan()
        scan.gate = asyncio.Event()
        bg = BackgroundScout(fast(), scan, lambda: None)
        first = asyncio.create_task(bg.run_once())
        await scan.started.wait()
        second = await bg.run_once()
        assert second.deferred and second.defer_reason is not None
        assert "already in progress" in second.defer_reason
        assert bg.status().running is True
        scan.gate.set()
        assert (await first).status == "completed"
        assert scan.calls == 1 and bg.status().running is False
        bg.start()
        bg.start()  # a second start never creates a second loop
        tasks = [t for t in asyncio.all_tasks() if "run_forever" in repr(t.get_coro())]
        assert len(tasks) == 1
        await bg.stop()

    run(go())


# --- Failure safety ---------------------------------------------------------------------------


def test_scheduler_continues_after_a_failed_run(caplog: pytest.LogCaptureFixture) -> None:
    async def go() -> None:
        scan = FakeScan(RuntimeError("boom"), ScanFailed("Scout refresh failed"), ScanSummary())
        bg = BackgroundScout(fast(), scan, lambda: None)
        bg.start()
        await until(lambda: bg.counts["completed"] >= 1)
        await bg.stop()
        assert bg.counts["failed"] == 2 and scan.calls >= 3

    with caplog.at_level(logging.INFO, logger="upscale.scout.background"):
        run(go())
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("background Scout scheduled") for m in messages)
    assert any(m == "background Scout started" for m in messages)
    assert any(m.startswith("background Scout failed") for m in messages)
    assert any(m.startswith("background Scout completed") for m in messages)


def test_a_broken_priority_check_defers_instead_of_crashing() -> None:
    def defer() -> str | None:
        raise RuntimeError("bug")

    scan = FakeScan()
    result = run(BackgroundScout(fast(), scan, defer).run_once())
    assert result.deferred and scan.calls == 0


# --- Priority --------------------------------------------------------------------------------


def test_priority_order_analyze_then_outcomes_then_manual_refresh_then_providers() -> None:
    full = {
        "analyze_active": True,
        "outcome_backlog": "outcome backlog",
        "scout_running": True,
        "provider_pressure": "GeckoTerminal busy",
    }
    assert defer_reason(**full) == "Analyze is active"
    assert defer_reason(**{**full, "analyze_active": False}) == "outcome backlog"
    assert defer_reason(**{**full, "analyze_active": False, "outcome_backlog": None}) == (
        "a Scout refresh is already running"
    )
    only_providers = {**full, "analyze_active": False, "outcome_backlog": None}
    assert defer_reason(**{**only_providers, "scout_running": False}) == "GeckoTerminal busy"
    assert defer_reason(analyze_active=False, outcome_backlog=None, scout_running=False,
                        provider_pressure=None) is None  # fmt: skip


def test_outcome_backlog_deferral() -> None:
    now = T0
    age = timedelta(minutes=30)
    assert outcome_backlog(None, now, age) is None
    assert outcome_backlog(CycleReport(at=now, due=5), now, age) is None  # all measured
    quota = outcome_backlog(CycleReport(at=now, due=12, deferred_for_quota=9), now, age)
    assert quota is not None and "9 due measurement(s) deferred for provider quota" in quota
    waiting = outcome_backlog(CycleReport(at=now, due=3, network_skipped=True), now, age)
    assert waiting is not None and "waiting for provider access" in waiting
    # A cycle long ago no longer describes the backlog.
    old = CycleReport(at=now - timedelta(hours=2), deferred_for_quota=9)
    assert outcome_backlog(old, now, age) is None


def test_provider_quota_and_rate_limit_deferral() -> None:
    clock = {"t": 0.0}
    g = gate(6, {"refresh": 2, "interactive": 2}, clock=lambda: clock["t"])
    assert provider_pressure([g], 600) is None  # idle: discovery gets its full share
    assert g._limiter.try_acquire("outcomes")
    assert g._limiter.try_acquire("outcomes")
    reason = provider_pressure([g], 600)
    assert reason == "GeckoTerminal capacity is in use by other work"
    clock["t"] = 61  # the window passes
    assert provider_pressure([g], 600) is None
    g.note_rate_limited()
    assert provider_pressure([g], 600) == "GeckoTerminal rate limiting was detected recently"
    clock["t"] += 599
    assert provider_pressure([g], 600) is not None
    clock["t"] += 2
    assert provider_pressure([g], 600) is None


def test_provider_429s_are_recorded_on_the_shared_quota() -> None:
    """A real 429 from Scout discovery or from Analyze's / outcomes' pool candles marks the
    one shared GeckoTerminal quota (no second limiter)."""
    shared = LaneLimiter(6, 60.0, {"refresh": 2, "interactive": 2})
    g = RequestGate("GeckoTerminal", ScoutConfig().geckoterminal, limiter=shared)
    transport = httpx2.MockTransport(lambda _: httpx2.Response(429))
    provider = GeckoTerminalDiscoveryProvider(gate=g, transport=transport)
    with pytest.raises(MarketDataUnavailableError, match="rate limit"):
        run(provider.discover_new_tokens("solana", 10))
    assert shared.rate_limited_within(60) and g.rate_limited_within(60)

    class Limited:
        name = "GeckoTerminal"

        async def fetch_pool_candles(self, *_: Any, **__: Any) -> Any:
            raise ProviderRateLimitedError("GeckoTerminal rate limit reached")

    shared.reset()
    assert not shared.rate_limited_within(60)
    candles = DexCandleService(Limited(), limiter=shared)  # type: ignore[arg-type]
    with pytest.raises(ProviderRateLimitedError):
        run(
            candles.get_candles(
                "solana", "pool", "1m", 10, symbol="X", canonical_id=None, token="T"
            )
        )
    assert shared.rate_limited_within(60)


# --- The app's wiring -------------------------------------------------------------------------


@pytest.fixture
def app_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """No interactive work, no outcome backlog, idle providers."""
    monkeypatch.setattr(upscale.main, "_interactive", 0)
    monkeypatch.setattr(upscale.main, "_last_analyze", -1e9)
    monkeypatch.setattr(services.outcome_collector, "last_cycle", None)
    for g in services.scout_service.gates():
        g.reset()


def test_app_defers_for_analyze(app_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    assert upscale.main._background_defer_reason() is None
    monkeypatch.setattr(upscale.main, "_interactive", 1)
    assert upscale.main._background_defer_reason() == "Analyze is active"
    monkeypatch.setattr(upscale.main, "_interactive", 0)
    monkeypatch.setattr(upscale.main, "_last_analyze", upscale.main.time.monotonic())
    assert upscale.main._background_defer_reason() == "Analyze is active"  # just finished


def test_app_defers_for_the_outcome_backlog(
    app_state: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = CycleReport(at=datetime.now(UTC), due=40, deferred_for_quota=37)
    monkeypatch.setattr(services.outcome_collector, "last_cycle", report)
    reason = upscale.main._background_defer_reason()
    assert reason is not None and "deferred for provider quota" in reason


def test_app_defers_for_provider_rate_limiting(app_state: None) -> None:
    gt = next(g for g in services.scout_service.gates() if g.name == "GeckoTerminal")
    gt.note_rate_limited()
    assert upscale.main._background_defer_reason() == (
        "GeckoTerminal rate limiting was detected recently"
    )
    gt.reset()
    assert upscale.main._background_defer_reason() is None


def test_manual_refresh_outranks_background_and_jobs_never_overlap(
    app_state: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = ranking(tmp_path)

    async def go() -> None:
        release = asyncio.Event()
        scans = {"n": 0, "active": 0, "max_active": 0}

        async def scan() -> GrowthScoutResult:
            scans["n"] += 1
            scans["active"] += 1
            scans["max_active"] = max(scans["max_active"], scans["active"])
            await release.wait()
            scans["active"] -= 1
            return result

        feed = ScoutFeed(scan, min_refresh_seconds=0)
        monkeypatch.setattr(upscale.main, "scout_feed", feed)
        bg = BackgroundScout(fast(), upscale.main._background_scan,
                             upscale.main._background_defer_reason)  # fmt: skip
        # A manual refresh is running: background defers.
        manual = asyncio.create_task(feed.refresh())
        await until(lambda: feed.refreshing)
        deferred = await bg.run_once()
        assert deferred.defer_reason == "a Scout refresh is already running"
        release.set()
        await manual
        # A background scan is running: a manual refresh joins it (one scan, not two).
        release.clear()
        background = asyncio.create_task(bg.run_once())
        await until(lambda: feed.refreshing)
        joined = asyncio.create_task(feed.refresh())
        await asyncio.sleep(0.05)
        release.set()
        assert await joined is True
        assert (await background).status == "completed"
        assert scans["n"] == 2 and scans["max_active"] == 1

    run(go())


# --- Real scans: counts, anchors, failure recovery ------------------------------------------


def test_background_scan_counts_and_no_duplicate_anchors(
    app_state: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Background runs go through the app's own scan, so outcome anchors follow the
    existing policy: the same ranking 30 minutes later anchors nothing new."""
    first = ranking(tmp_path)
    results = [first, later_run(first, 30)]

    async def scan(*_: Any, **__: Any) -> GrowthScoutResult:
        return results.pop(0)

    monkeypatch.setattr(services.growth_scout_service, "scan", scan)
    monkeypatch.setattr(services.outcome_collector, "wake", lambda: None)
    feed = ScoutFeed(upscale.main._scan, min_refresh_seconds=0)
    monkeypatch.setattr(upscale.main, "scout_feed", feed)
    bg = BackgroundScout(fast(), upscale.main._background_scan, lambda: None)

    one = run(bg.run_once())
    assert one.status == "completed"
    assert one.candidates_discovered == first.universe.discovered  # type: ignore[union-attr]
    assert one.candidates_ranked == first.eligible
    anchored = run(services.outcome_store.scout_observations())
    assert one.new_outcome_anchors == len(anchored) > 0

    two = run(bg.run_once())
    assert two.status == "completed" and two.new_outcome_anchors == 0
    assert len(run(services.outcome_store.scout_observations())) == len(anchored)


def test_recovery_after_a_provider_failure_keeps_the_last_results(
    app_state: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    good = ranking(tmp_path)
    outcomes: list[GrowthScoutResult | Exception] = [
        good,
        MarketDataUnavailableError("GeckoTerminal request timed out"),
        later_run(good, 30),
    ]

    async def scan(*_: Any, **__: Any) -> GrowthScoutResult:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(services.growth_scout_service, "scan", scan)
    monkeypatch.setattr(services.outcome_collector, "wake", lambda: None)
    feed = ScoutFeed(upscale.main._scan, min_refresh_seconds=0)
    monkeypatch.setattr(upscale.main, "scout_feed", feed)
    bg = BackgroundScout(fast(), upscale.main._background_scan, lambda: None)

    assert run(bg.run_once()).status == "completed"
    failed = run(bg.run_once())
    assert failed.status == "failed" and failed.error is not None
    assert "MarketDataUnavailableError" in failed.error
    assert feed.result is good  # the last good Scout results stay
    recovered = run(bg.run_once())
    assert recovered.status == "completed" and feed.error is None
    assert bg.counts == {"completed": 2, "deferred": 0, "failed": 1}


def test_a_scan_that_just_finished_defers_the_background_run(
    app_state: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = ranking(tmp_path)
    calls = {"n": 0}

    async def scan() -> GrowthScoutResult:
        calls["n"] += 1
        return result

    feed = ScoutFeed(scan, min_refresh_seconds=60)
    monkeypatch.setattr(upscale.main, "scout_feed", feed)
    bg = BackgroundScout(fast(), upscale.main._background_scan, lambda: None)
    run(feed.refresh())  # a manual refresh just now
    later = run(bg.run_once())
    assert later.deferred and later.defer_reason == "a Scout scan finished moments ago"
    assert calls["n"] == 1 and bg.last_run is None


def test_scan_deferred_is_reported_as_a_deferral() -> None:
    async def scan() -> ScanSummary:
        raise ScanDeferred("because")

    result = run(BackgroundScout(fast(), scan, lambda: None).run_once())
    assert result.status == "deferred" and result.defer_reason == "because"


# --- Status endpoint & lifecycle ------------------------------------------------------------


def test_status_endpoint_reports_the_real_scheduler_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = client.get("/scout/background/status").json()
    assert body["enabled"] is True and body["interval_minutes"] == 30
    assert body["running"] is False and body["next_run"] is None  # not started (no lifespan)

    scan = FakeScan(ScanSummary(candidates_discovered=12, candidates_ranked=9,
                                new_outcome_anchors=3))  # fmt: skip
    bg = BackgroundScout(fast(interval_minutes=45), scan, lambda: None)
    run(bg.run_once())
    monkeypatch.setattr(upscale.main, "background_scout", bg)
    body = client.get("/scout/background/status").json()
    assert body["interval_minutes"] == 45 and body["last_run"] is not None
    last = body["last_result"]
    assert last["status"] == "completed" and last["deferred"] is False
    assert last["defer_reason"] is None and last["error"] is None
    assert (last["candidates_discovered"], last["candidates_ranked"],
            last["new_outcome_anchors"]) == (12, 9, 3)  # fmt: skip
    assert body["runs_completed"] == 1

    deferring = BackgroundScout(fast(), scan, lambda: "Analyze is active")
    run(deferring.run_once())
    monkeypatch.setattr(upscale.main, "background_scout", deferring)
    last = client.get("/scout/background/status").json()["last_result"]
    assert last["deferred"] is True and last["defer_reason"] == "Analyze is active"
    assert last["candidates_discovered"] is None  # nothing measured: never invented


def test_clean_shutdown_even_during_a_scan() -> None:
    async def go() -> None:
        idle = BackgroundScout(BackgroundScoutSettings(), FakeScan(), lambda: None)
        idle.start()
        assert idle.active
        await asyncio.wait_for(idle.stop(), 1)
        assert not idle.active and idle.status().next_run is None

        stuck = FakeScan()
        stuck.gate = asyncio.Event()  # never set: the scan hangs
        busy = BackgroundScout(fast(shutdown_grace_seconds=0.1), stuck, lambda: None)
        busy.start()
        await stuck.started.wait()
        await asyncio.wait_for(busy.stop(), 2)
        assert not busy.active and busy.running is False

    run(go())


def test_app_lifespan_starts_and_stops_the_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    scan = FakeScan()
    bg = BackgroundScout(BackgroundScoutSettings(), scan, lambda: None)
    monkeypatch.setattr(upscale.main, "background_scout", bg)
    monkeypatch.setattr(upscale.main, "OUTCOMES_COLLECTOR", False)
    with TestClient(upscale.main.app) as client:
        body = client.get("/scout/background/status").json()
        assert body["enabled"] is True and body["next_run"] is not None
        assert bg.active
    assert not bg.active and scan.calls == 0  # startup delay: no burst at startup

    off = BackgroundScout(BackgroundScoutSettings(enabled=False), scan, lambda: None)
    monkeypatch.setattr(upscale.main, "background_scout", off)
    with TestClient(upscale.main.app) as client:
        body = client.get("/scout/background/status").json()
        assert body["enabled"] is False and body["next_run"] is None
    assert scan.calls == 0


# --- bounded deferral and reliability diagnostics ------------------------------------------------


class Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def backlog(due: int) -> DeferReason | None:
    """The production soft reason, with its per-cycle count."""
    return outcome_backlog(CycleReport(at=T0, due=due, deferred_for_quota=due), T0,
                           timedelta(days=365))  # fmt: skip


def production_scout(
    clock: Clock, scan: FakeScan, soft: list[str | None], hard: list[str | None]
) -> BackgroundScout:
    """The production schedule: 30-minute interval, retries every 5, 60-minute bound."""
    return BackgroundScout(
        BackgroundScoutSettings(interval_minutes=30, max_deferral_minutes=60), scan,
        lambda: soft[0], now=clock, hard_defer=lambda: hard[0],
    )  # fmt: skip


def test_soft_deferrals_are_bounded_from_the_first_soft_deferral() -> None:
    """Production (2026-10-02): started 01:30:53, deferred since 01:35:53 for an outcome
    backlog whose count changed every cycle; at 02:45:53 (70 min deferred) still
    `overdue: false`, because the bound was interval + max_deferral (90 min) since startup.
    Now the first attempt 60 min after the first soft deferral runs the scan."""
    started = datetime(2026, 10, 2, 1, 30, 53, tzinfo=UTC)
    clock = Clock(started)
    scan = FakeScan()
    soft: list[str | None] = [None]
    hard: list[str | None] = [None]
    bg = production_scout(clock, scan, soft, hard)
    first = started + timedelta(minutes=5)  # after the startup delay
    counts = [93, 81, 146, 120, 99, 146, 77, 88, 101, 146, 64, 146]
    for i, due in enumerate(counts):  # 01:35:53 .. 02:30:53: soft, not yet overdue
        clock.at = first + timedelta(minutes=5 * i)
        soft[0] = backlog(due)
        result = run(bg.run_once())
        assert result.status == "deferred" and not result.overdue
        assert result.defer_kind == "soft" and result.defer_code == "outcome_backlog_quota"
        assert result.defer_reason == (
            f"outcome collection has {due} due measurement(s) deferred for provider quota"
        )
        s = bg.status()
        assert s.deferred_since == first  # changing text / counts never reset the window
        assert s.overdue_at == first + timedelta(minutes=60) and not s.overdue
    assert scan.calls == 0
    s = bg.status()
    assert s.consecutive_deferrals == 12 == s.runs_deferred
    assert s.deferrals_by_reason == {"outcome_backlog_quota": 12}  # one stable code
    assert (s.last_defer_kind, s.last_defer_code) == ("soft", "outcome_backlog_quota")

    clock.at = first + timedelta(minutes=60)  # 02:35:53: the bound is reached
    assert bg.status().overdue
    soft[0] = backlog(146)
    result = run(bg.run_once())
    assert result.status == "completed" and result.overdue and scan.calls == 1
    s = bg.status()
    assert s.overdue_runs == 1 and s.background_scans_started == 1
    assert s.background_scans_completed == 1 and s.runs_deferred == 12  # counters kept
    # A completed scan closes the window.
    assert s.deferred_since is None and s.overdue_at is None and not s.overdue
    assert s.consecutive_deferrals == 0 and s.last_defer_kind is None

    clock.at += timedelta(minutes=30)  # the next slot: soft reasons defer again, anew
    result = run(bg.run_once())
    assert result.status == "deferred" and not result.overdue and scan.calls == 1
    assert bg.status().deferred_since == clock.at


def test_a_late_attempt_past_the_bound_runs_at_once() -> None:
    clock = Clock(T0)
    scan = FakeScan()
    bg = production_scout(clock, scan, [backlog(5)], [None])
    run(bg.run_once())
    clock.at = T0 + timedelta(minutes=61, seconds=7)  # any attempt at/after the bound
    assert run(bg.run_once()).status == "completed" and bg.overdue_runs == 1


def test_hard_reasons_still_defer_an_overdue_scan_then_it_runs_immediately() -> None:
    clock = Clock(T0)
    scan = FakeScan()
    soft: list[str | None] = [backlog(93)]
    hard: list[str | None] = [None]
    bg = production_scout(clock, scan, soft, hard)
    run(bg.run_once())  # opens the window at T0
    clock.at = T0 + timedelta(minutes=30)
    hard[0] = ANALYZE_ACTIVE  # hard before the bound: neither opens nor resets the window
    result = run(bg.run_once())
    assert (result.defer_kind, result.defer_code) == ("hard", "analyze_active")
    assert bg.status().deferred_since == T0
    soft[0] = backlog(81)
    for minutes in (60, 65):  # at the deadline a hard reason still defers
        clock.at = T0 + timedelta(minutes=minutes)
        hard[0] = ANALYZE_ACTIVE if minutes == 60 else SCOUT_RUNNING
        result = run(bg.run_once())
        assert result.status == "deferred" and result.overdue and result.defer_kind == "hard"
        s = bg.status()
        assert s.overdue and s.deferred_since == T0 and s.overdue_runs == 0
    assert scan.calls == 0
    assert bg.status().deferrals_by_reason == {
        "outcome_backlog_quota": 1, "analyze_active": 2, "scout_running": 1,
    }  # fmt: skip
    clock.at = T0 + timedelta(minutes=70)
    hard[0] = None  # cleared: the next attempt runs, overdue, despite the soft reason
    result = run(bg.run_once())
    assert result.status == "completed" and result.overdue and bg.overdue_runs == 1


def test_a_failed_overdue_scan_keeps_the_window_a_completed_one_closes_it() -> None:
    clock = Clock(T0)
    scan = FakeScan(ScanFailed("no feeds answered"), ScanSummary())
    bg = production_scout(clock, scan, [backlog(146)], [None])
    run(bg.run_once())
    clock.at = T0 + timedelta(minutes=60)
    assert run(bg.run_once()).status == "failed"
    s = bg.status()
    assert s.deferred_since == T0 and s.overdue and s.overdue_runs == 1
    clock.at = T0 + timedelta(minutes=90)  # the next slot is still overdue
    result = run(bg.run_once())
    assert result.status == "completed" and result.overdue and bg.overdue_runs == 2
    assert bg.status().deferred_since is None


def test_a_scan_that_defers_itself_is_not_counted_as_an_overdue_run() -> None:
    clock = Clock(T0)
    scan = FakeScan(ScanDeferred("a Scout scan finished moments ago"))
    bg = production_scout(clock, scan, [backlog(146)], [None])
    run(bg.run_once())
    clock.at = T0 + timedelta(minutes=60)
    result = run(bg.run_once())
    assert result.status == "deferred" and result.defer_kind == "hard" and result.overdue
    s = bg.status()
    assert s.overdue_runs == 0 and s.background_scans_started == 0 and s.deferred_since == T0


def test_hard_deferrals_alone_open_no_window() -> None:
    clock = Clock(T0)
    scan = FakeScan()
    bg = production_scout(clock, scan, [backlog(5)], [ANALYZE_ACTIVE])
    for minutes in range(0, 300, 5):
        clock.at = T0 + timedelta(minutes=minutes)
        assert run(bg.run_once()).defer_kind == "hard"
    s = bg.status()
    assert s.deferred_since is None and not s.overdue and scan.calls == 0


def test_soft_reason_codes_are_stable() -> None:
    assert reason_code(backlog(93) or "") == reason_code(backlog(146) or "")
    waiting = outcome_backlog(CycleReport(at=T0, due=3, network_skipped=True), T0,
                              timedelta(hours=1))  # fmt: skip
    assert waiting is not None and reason_code(waiting) == "outcome_backlog_waiting"
    assert reason_code("plain message") == "plain message"


def test_a_forced_scan_still_respects_provider_budgets() -> None:
    """Overdue only bypasses soft *deferral*: inside the scan every gate still enforces its
    budget and keeps the "interactive" reservation, so the scan runs fewer feeds."""
    clock = Clock(T0)
    g = gate(6, {"interactive": 2})
    assert g._limiter.try_acquire("outcomes") and g._limiter.try_acquire("outcomes")
    fetched: list[str] = []
    refused: list[str] = []

    async def scan() -> ScanSummary:
        for feed in ("trending", "new", "top", "gainers"):

            async def fetch(feed: str = feed) -> str:
                return feed

            try:
                fetched.append(await g.run(feed, fetch))
            except RateLimitReachedError:
                refused.append(feed)
        return ScanSummary(candidates_discovered=len(fetched))

    pressure = provider_pressure([g], 600)
    assert pressure is not None and reason_code(pressure) == "provider_capacity:GeckoTerminal"
    bg = BackgroundScout(
        BackgroundScoutSettings(interval_minutes=30, max_deferral_minutes=60), scan,
        lambda: provider_pressure([g], 600), now=clock, hard_defer=lambda: None,
    )  # fmt: skip
    assert run(bg.run_once()).defer_kind == "soft"
    clock.at = T0 + timedelta(minutes=60)
    result = run(bg.run_once())
    assert result.status == "completed" and result.overdue
    # 6 a minute, 2 reserved for interactive, 2 already used by outcomes: 2 feeds run.
    assert fetched == ["trending", "new"] and refused == ["top", "gainers"]
    assert g.available("interactive") == 2  # the reservation is untouched


def test_without_a_hard_defer_callback_the_old_behaviour_holds() -> None:
    clock = Clock(T0)
    scan = FakeScan()
    bg = BackgroundScout(BackgroundScoutSettings(), scan, lambda: "busy", now=clock)
    clock.at = T0 + timedelta(hours=10)
    assert run(bg.run_once()).status == "deferred" and scan.calls == 0


def test_reliability_diagnostics() -> None:
    clock = Clock(T0)
    scan = FakeScan(ScanSummary(), ScanFailed("Scout scan produced no result"), ScanSummary())
    bg = BackgroundScout(BackgroundScoutSettings(), scan, lambda: None, now=clock)
    run(bg.run_once())
    clock.at = T0 + timedelta(minutes=40)
    run(bg.run_once())
    clock.at = T0 + timedelta(minutes=200)
    run(bg.run_once())
    s = bg.status()
    assert (s.background_scans_started, s.background_scans_completed,
            s.background_scans_failed) == (3, 2, 1)  # fmt: skip
    assert s.last_failure_at == T0 + timedelta(minutes=40)
    assert s.last_failure_reason == "Scout scan produced no result"
    assert s.last_started_at == T0 + timedelta(minutes=200)
    assert s.longest_gap_between_completed_minutes == 200.0
    assert s.last_scan_duration_seconds == 0.0 and s.process_started_at == T0


def test_max_deferral_setting() -> None:
    assert load_settings(None, None).max_deferral_minutes == 60.0
    assert load_settings(None, None, "120").max_deferral_minutes == 120.0
    assert load_settings(None, None, "1").max_deferral_minutes == MIN_INTERVAL_MINUTES
    assert load_settings(None, None, "x").max_deferral_minutes == 60.0


def test_production_hard_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    main = upscale.main
    monkeypatch.setattr(main, "_last_analyze", -1e9)
    monkeypatch.setattr(main, "_interactive", 0)
    assert main._background_hard_defer_reason() is None
    monkeypatch.setattr(main, "_interactive", 1)
    assert main._background_hard_defer_reason() == "Analyze is active"
    assert main.background_scout._hard_defer is main._background_hard_defer_reason
