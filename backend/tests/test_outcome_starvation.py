"""Outcome collection can't starve on the shared GeckoTerminal quota (the live failure).

Live: 41 due 24h horizons stayed PENDING for 150+ cycles. DEX Screener lookups worked, no
GeckoTerminal 429 was ever seen, yet every candle request was "deferred: background quota
in use": of GeckoTerminal's 6 / minute, 2 are held for Analyze and 2 for Scout refresh
(held from startup until a scan releases them, and background Scout never scanned because
outcome work was backlogged), and outcome work kept 1 more free for Scout discovery. The
one request left each cycle went to the highest-priority row: a decision without a
reference price, whose candles can never measure anything, so it never finalized and
took that request again every cycle.

Here a fake GeckoTerminal *server* enforces the real limit (a 7th request within 60 s
gets HTTP 429) on a clock shared with UpScale's quota.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

import upscale.services
from tests.test_outcomes import (
    Clock,
    FakeCandles,
    FakePools,
    anchor,
    assessment,
    chat,
    collector,
    dex_pool,
    horizon,
)
from tests.test_scout_growth import MINTS, NOW, run
from upscale.background_scout import defer_reason, outcome_backlog
from upscale.services.geckoterminal import DexCandleService, GeckoTerminalProvider
from upscale.services.market_data import CandleSeries, MarketDataUnavailableError, Timeframe
from upscale.services.outcomes import (
    OUTCOME_LANE,
    OutcomeConfig,
    OutcomeStore,
    ProviderCandles,
    record_decision,
)
from upscale.services.outcomes.collector import CycleReport
from upscale.services.outcomes.config import CollectorConfig, HorizonSpec
from upscale.services.outcomes.models import PriceRef, ScoutObservation
from upscale.services.quota import DEFAULT_LANE, INTERACTIVE_LANE, LaneLimiter
from upscale.services.scout.service import REFRESH_LANE
from upscale.services.scout.store import ScoutSnapshotStore

from .conftest import FakeGeckoTerminal

MINUTE = timedelta(minutes=1)
H24 = HorizonSpec(label="24h", minutes=1440, candles="15m", retry_minutes=1440)
ONLY_24H = OutcomeConfig(horizons=(H24,))
LIMIT = 6  # GeckoTerminal's free tier, measured live


class RealGeckoTerminal:
    """GeckoTerminal's own sliding-window limit, on UpScale's clock."""

    def __init__(self, fake: FakeGeckoTerminal, clock: Clock):
        self.clock = clock
        self.served: list[datetime] = []
        self.refused = 0
        inner = fake.handler

        def handler(request: httpx2.Request) -> httpx2.Response:
            now = self.clock()
            if sum(1 for t in self.served if now - t < MINUTE) >= LIMIT:
                self.refused += 1
                return httpx2.Response(429, json={"status": "429"})
            self.served.append(now)
            return inner(request)

        fake.handler = handler

    def worst_window(self) -> int:
        return max(
            (sum(1 for t in self.served if 0 <= (t - s).total_seconds() < 60) for s in self.served),
            default=0,
        )


def production_quota(clock: Clock) -> LaneLimiter:
    """GeckoTerminal's process-wide quota exactly as the app configures it."""
    app = upscale.services.geckoterminal_quota
    return LaneLimiter(
        app.max_calls, app.period, app.reservations, lambda: clock().timestamp(), app.outranks
    )


def candles_15m(start: datetime, price: float, count: int = 100) -> list[list[float]]:
    step = 15 * MINUTE
    return [
        [(start + i * step).timestamp(), price, price * 1.02, price * 0.98, price, 50.0]
        for i in range(count)
    ]


def backlog(
    tmp_path: Path, fake: FakeGeckoTerminal, count: int
) -> tuple[OutcomeStore, list[ScoutObservation], ScoutSnapshotStore, FakePools]:
    """`count` Scout anchors of distinct pools, 24h horizon only, all at the same time."""
    _, obs, scout = anchor(tmp_path / "template")
    store = OutcomeStore(tmp_path / "live.sqlite3")
    copies = [
        obs.model_copy(update={
            "id": None, "canonical_id": f"solana:token{i}", "address": f"token{i}",
            "pool_address": f"pool{i}",
        })
        for i in range(count)
    ]  # fmt: skip
    stored = run(store.add_scout_observations(copies, [H24]))
    assert len(stored) == count
    pools = FakePools(name=obs.market_provider)
    p0 = obs.market.price_usd or 1.0
    for o in stored:
        pools.by_token[o.canonical_id] = [dex_pool("A", o.pool_address)]
        pools.by_token[o.canonical_id][0].base.address = o.address
        fake.candles[o.pool_address] = candles_15m(o.observed_at, p0)
    return store, stored, scout, pools


def unmeasurable_decision(store: OutcomeStore, at: datetime) -> int:
    """An Analyze decision with neither a live price nor a last close."""
    request, response = chat(
        assessment("wait", live_price=None, last_close=None), pool="pool-decision"
    )
    d = run(record_decision(store, request, response, at, ONLY_24H))
    assert d is not None and d.reference_price is None and d.id is not None
    oid: int = d.id
    return oid


def scout_defer(report: Any, now: datetime) -> str | None:
    """Background Scout's decision with Analyze idle, no scan running, no 429."""
    return defer_reason(
        analyze_active=False,
        outcome_backlog=outcome_backlog(report, now, timedelta(minutes=30)),
        scout_running=False,
        provider_pressure=None,
    )


# --- The live case -----------------------------------------------------------------------------


def test_live_starvation_recovers_without_breaking_the_provider_limit(
    tmp_path: Path, fake_geckoterminal: FakeGeckoTerminal
) -> None:
    count = 20
    store, stored, scout, pools = backlog(tmp_path, fake_geckoterminal, count)
    start = stored[0].observed_at
    decision = unmeasurable_decision(store, start)
    analyze_pools = FakePools(name="DEX Screener")  # Analyze's pools come from DEX Screener
    analyze_pools.by_token[f"solana:{MINTS['A']}"] = [dex_pool("A", "pool-decision")]
    clock = Clock(start + timedelta(hours=24, minutes=5))
    server = RealGeckoTerminal(fake_geckoterminal, clock)
    quota = production_quota(clock)
    dex = DexCandleService(
        GeckoTerminalProvider(transport=fake_geckoterminal.transport()),
        limiter=quota, now=clock, clock=lambda: clock().timestamp(),
    )  # fmt: skip
    candles = ProviderCandles(
        dex, upscale.services.market_data_service, CollectorConfig().min_free_calls
    )
    c = collector(store, clock, scout, candles, [pools, analyze_pools], config=ONLY_24H)

    # A Scout scan just used the window (discovery + refresh): nothing left for outcomes.
    for lane in (DEFAULT_LANE, DEFAULT_LANE, REFRESH_LANE, REFRESH_LANE):
        assert quota.try_acquire(lane)
    assert quota.available(OUTCOME_LANE) == 0
    first = run(c.collect())
    assert first.due == count + 1 and fake_geckoterminal.requests == []
    assert first.deferred_by_reason == {"quota": count}  # UpScale's quota, not a 429
    assert first.candle_headroom == {"GeckoTerminal": 0}
    assert scout_defer(first, clock()) is not None  # background Scout waits
    # The decision needs no candles: it finalizes at once, market state only.
    d = horizon(store, decision, "24h", "decision")
    assert d.status == "PARTIAL" and d.price is None
    assert any("price outcome not measurable" in m for m in d.missing)

    cycles, reports = 0, [first]
    while run(store.counts())["scout"].get("PENDING", 0):
        cycles += 1
        assert cycles <= count, "due outcome work is starving"
        clock.advance(seconds=CollectorConfig().retry_seconds)
        before = len(server.served)
        report = run(c.collect())
        reports.append(report)
        assert len(server.served) > before  # bounded progress every cycle
        assert quota.available(INTERACTIVE_LANE) >= 2  # Analyze's reservation untouched
        if run(store.counts())["scout"].get("PENDING", 0):
            assert scout_defer(report, clock()) is not None
    # 3 requests per cycle (the per-cycle budget), each finishing one horizon.
    assert cycles == -(-count // 3)
    assert server.refused == 0 and server.worst_window() <= LIMIT
    assert "pool-decision" not in {r.url.path.split("/pools/")[1].split("/")[0]
                                   for r in fake_geckoterminal.requests}  # fmt: skip
    assert run(store.counts())["scout"] == {"observations": count, "COMPLETE": count}
    assert all(r.deferred_by_reason.keys() <= {"quota"} for r in reports)
    # The backlog is clear: background Scout may run again.
    assert reports[-1].deferred_for_quota == 0 and scout_defer(reports[-1], clock()) is None


def test_unmeasurable_decision_without_market_state_ends_unavailable(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    decision = unmeasurable_decision(store, obs.observed_at)
    candles = FakeCandles()
    tolerance = timedelta(seconds=0.1 * 1440 * 60)
    clock = Clock(obs.observed_at + timedelta(hours=24, minutes=5))
    c = collector(store, clock, None, candles, FakePools(headroom=0), config=ONLY_24H)
    run(c.collect())
    assert horizon(store, decision, "24h", "decision").status == "PENDING"  # market may come
    clock.at = obs.observed_at + timedelta(hours=24) + tolerance + MINUTE
    run(c.collect())
    h = horizon(store, decision, "24h", "decision")
    assert h.status == "UNAVAILABLE" and h.price is None and h.market is None
    assert all(call[0] != "pool-decision" for call in candles.calls)  # nothing to measure


# --- Fairness, cooldown and quota precedence ------------------------------------------------------


class FailingPool(FakeCandles):
    """Candles for every pool but one, whose provider always answers HTTP 502."""

    def __init__(self, pool: str, headroom: int):
        super().__init__(headroom)
        self.pool = pool

    async def window(
        self, ref: PriceRef, timeframe: Timeframe, start: datetime, end: datetime, now: datetime
    ) -> CandleSeries:
        if ref.pool_address == self.pool:
            self.calls.append((self.pool, timeframe, start, end))
            raise MarketDataUnavailableError("GeckoTerminal returned HTTP 502")
        return await super().window(ref, timeframe, start, end, now)


def test_longest_deferred_work_gets_capacity_first(tmp_path: Path) -> None:
    """One request per cycle and a best-ranked market whose candles always fail: without
    aging it would take that request every cycle and the others would never get one."""
    fake = FakeGeckoTerminal()
    store, stored, scout, _ = backlog(tmp_path, fake, 4)
    clock = Clock(stored[0].observed_at + timedelta(hours=24, minutes=5))
    failing = min(stored, key=lambda o: o.rank).pool_address
    candles = FailingPool(failing, headroom=1)
    budget = OutcomeConfig(
        horizons=(H24,), collector=CollectorConfig(request_budgets={"GeckoTerminal": 1})
    )
    c = collector(store, clock, scout, candles, FakePools(headroom=0), config=budget)
    served: list[str] = []
    for _ in range(len(stored)):
        before = len(candles.calls)
        report = run(c.collect())
        served += [call[0] for call in candles.calls[before:]]
        assert report.longest_candle_deferral < len(stored)
        clock.advance(minutes=2)
    # One request per cycle, and within as many cycles as there are markets everyone got
    # one: the failing market went back behind the work that had waited longer.
    assert sorted(served) == sorted(o.pool_address for o in stored)
    assert served[0] == failing


def test_real_429_cooldown_is_distinguished_from_quota_deferral(
    tmp_path: Path, fake_geckoterminal: FakeGeckoTerminal
) -> None:
    store, stored, scout, pools = backlog(tmp_path, fake_geckoterminal, 2)
    clock = Clock(stored[0].observed_at + timedelta(hours=24, minutes=5))
    quota = production_quota(clock)
    dex = DexCandleService(
        GeckoTerminalProvider(transport=fake_geckoterminal.transport()),
        limiter=quota, now=clock, clock=lambda: clock().timestamp(),
    )  # fmt: skip
    candles = ProviderCandles(dex, upscale.services.market_data_service, 1)
    c = collector(store, clock, scout, candles, pools, config=ONLY_24H)
    quota.note_rate_limited()  # e.g. Analyze just got an HTTP 429
    report = run(c.collect())
    assert report.deferred_by_reason == {"429_cooldown": 2} and fake_geckoterminal.requests == []
    assert any("HTTP 429" in m for m in horizon(store, stored[0].id or 0, "24h").missing)
    clock.advance(seconds=CollectorConfig().rate_limit_cooldown_seconds)
    report = run(c.collect())
    assert report.deferred_by_reason == {} and len(fake_geckoterminal.requests) == 2


def test_outcomes_outrank_scout_refresh_but_never_analyze() -> None:
    now = [0.0]
    quota = LaneLimiter(6, 60.0, {"refresh": 2, "interactive": 2}, lambda: now[0],
                        {OUTCOME_LANE: {REFRESH_LANE}})  # fmt: skip
    # Refresh reservation held and idle: outcome work may use it, never Analyze's.
    assert quota.available(OUTCOME_LANE) == 4 and quota.available(DEFAULT_LANE) == 2
    assert all(quota.try_acquire(OUTCOME_LANE) for _ in range(4))
    assert not quota.try_acquire(OUTCOME_LANE)
    assert quota.available(INTERACTIVE_LANE) == 2  # still there for a user
    assert quota.available(DEFAULT_LANE) == 0 and quota.available(REFRESH_LANE) == 0
    assert quota.try_acquire(INTERACTIVE_LANE) and quota.try_acquire(INTERACTIVE_LANE)
    assert quota.available(INTERACTIVE_LANE) == 0  # hard total: 6
    now[0] = 60.0
    assert quota.available(OUTCOME_LANE) == 4  # the window moved on


def test_app_quota_lets_outcomes_outrank_refresh_only() -> None:
    quota = upscale.services.geckoterminal_quota
    assert quota.outranks == {OUTCOME_LANE: frozenset({REFRESH_LANE})}
    assert quota.reservations[INTERACTIVE_LANE] >= 2 and quota.max_calls == LIMIT


def test_status_makes_starvation_diagnosable(
    client: TestClient, isolated_outcome_store: OutcomeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        upscale.services.outcome_collector,
        "last_cycle",
        CycleReport(at=NOW, due=41, deferred_for_quota=41, deferred_by_reason={"quota": 41},
                    longest_candle_deferral=150, candle_headroom={"GeckoTerminal": 0}),
    )  # fmt: skip
    status = client.get("/outcomes/status").json()
    assert status["last_cycle"]["deferred_by_reason"] == {"quota": 41}
    assert status["last_cycle"]["longest_candle_deferral"] == 150
    gt = status["quota"]["GeckoTerminal"]
    assert gt["limit"] == LIMIT and gt["available"][OUTCOME_LANE] >= 0
    assert gt["provider_429_cooldown_active"] is False and gt["provider_429_seconds_ago"] is None
    assert gt["outranks"] == {OUTCOME_LANE: [REFRESH_LANE]}
