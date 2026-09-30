"""Outcome tracking: immutable Scout / decision observations, the observation policy, fixed
horizons, MFE / MAE, trigger touches, terminal and missing outcomes, the background
collector's quota discipline, aggregates with small-sample protection, the schema
migration, the read-only replay and the developer API.

All offline: Growth Scout results come from the synthetic markets of test_scout_growth,
candles and pool lookups from fakes (or the fake GeckoTerminal / DEX Screener transports).
"""

import asyncio
import hashlib
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

import upscale.main
import upscale.services
from tests.test_scout_growth import (
    MINTS,
    NOW,
    SOL,
    accelerating,
    fading,
    flat,
    run,
    service,
)
from upscale.schemas import (
    AgentResult,
    Analysis,
    AssetRef,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Uncertainty,
)
from upscale.services.geckoterminal import DexCandleService, GeckoTerminalProvider
from upscale.services.market_data import (
    Candle,
    CandleSeries,
    MarketDataUnavailableError,
    ProviderRateLimitedError,
    Timeframe,
)
from upscale.services.opportunity import Invalidation, OpportunityAssessment, Trigger
from upscale.services.outcomes import (
    OUTCOME_LANE,
    OutcomeCollector,
    OutcomeConfig,
    OutcomeStore,
    ProviderCandles,
    record_decision,
    record_scout_run,
)
from upscale.services.outcomes.analytics import cohort, summarize_scout
from upscale.services.outcomes.config import (
    AnalyticsConfig,
    CollectorConfig,
    HorizonSpec,
    ObservationPolicy,
)
from upscale.services.outcomes.metrics import (
    distribution,
    first_event,
    median,
    path_from_candles,
    percentile,
    trigger_outcome,
)
from upscale.services.outcomes.models import (
    HorizonOutcome,
    PricePath,
    PriceRef,
    ScoutObservation,
)
from upscale.services.outcomes.replay import REPLAY_LABEL, replay
from upscale.services.outcomes.store import HorizonUpdate, OutcomeStoreError
from upscale.services.quota import LaneLimiter
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.scout.models import ScoutMarketMetrics, ScoutSnapshot, ScoutWindow
from upscale.services.scout.providers import DexScreenerDiscoveryProvider
from upscale.services.scout.social.store import SocialStore
from upscale.services.scout.store import ScoutSnapshotStore
from upscale.services.solana_dex import DexPool, TokenRef, WindowStats

MINUTE = timedelta(minutes=1)
CFG = OutcomeConfig()
# Most collector tests measure one horizon at a time: candles are fetched as soon as it
# ends (the default shares one request across same-timeframe horizons; tested below).
IMMEDIATE = OutcomeConfig(collector=CollectorConfig(coalesce_candles=False))


class Clock:
    def __init__(self, at: datetime = NOW):
        self.at = at

    def __call__(self) -> datetime:
        return self.at

    def advance(self, **kw: float) -> None:
        self.at += timedelta(**kw)


# --- Fakes ---------------------------------------------------------------------------------


class FakeCandles:
    """Candles per pool (or exchange symbol); records every request."""

    def __init__(self, headroom: int = 10):
        self.by_pool: dict[str, list[Candle]] = {}
        self.calls: list[tuple[str, str, datetime, datetime]] = []
        self.error: Exception | None = None
        self._headroom = headroom

    def provider(self, ref: PriceRef) -> str:
        return "GeckoTerminal" if ref.kind == "dex_pool" else ref.provider or "exchange"

    def headroom(self, ref: PriceRef) -> int:
        return self._headroom

    def rate_limited(self, ref: PriceRef, seconds: float) -> bool:
        return False

    async def window(
        self, ref: PriceRef, timeframe: Timeframe, start: datetime, end: datetime, now: datetime
    ) -> CandleSeries:
        key = ref.pool_address or ref.symbol or ""
        self.calls.append((key, timeframe, start, end))
        if self.error is not None:
            raise self.error
        return CandleSeries(
            symbol=key, provider="GeckoTerminal", provider_id=key, timeframe=timeframe,
            candles=self.by_pool.get(key, []), volume_available=True, fetched_at=now,
        )  # fmt: skip


class FakePools:
    """Exact pools per token (the test anchors are priced by GeckoTerminal, so the default
    fake is authoritative for them)."""

    def __init__(self, headroom: int = 10, name: str = "GeckoTerminal"):
        self.name = name
        self.by_token: dict[str, list[DexPool]] = {}
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []
        self.error: Exception | None = None
        self._headroom = headroom

    def headroom(self, chain: str) -> int:
        return self._headroom

    def requests(self, count: int) -> int:
        return -(-count // 30)

    async def pools(
        self, chain: str, keys: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], DexPool | None]:
        self.calls.append((chain, list(keys)))
        if self.error is not None:
            raise self.error
        out: dict[tuple[str, str], DexPool | None] = {}
        for token, pool in keys:
            listed = self.by_token.get(f"{chain}:{token}", [])
            out[(token, pool)] = next((p for p in listed if p.pair_address == pool), None)
        return out


def candle(at: datetime, o: float, h: float, lo: float, c: float) -> Candle:
    return Candle(timestamp=at, open=o, high=h, low=lo, close=c, volume=100.0)


def minute_candles(
    start: datetime, prices: list[tuple[float, float, float, float]]
) -> list[Candle]:
    return [candle(start + i * MINUTE, *p) for i, p in enumerate(prices)]


def dex_pool(key: str, pool: str, *, liquidity: float = 80_000, price: float = 0.0011) -> DexPool:
    mint = MINTS.get(key, key)
    return DexPool(
        chain="solana", dex="raydium", pair_address=pool,
        base=TokenRef(address=mint, symbol=key), quote=TokenRef(address=SOL, symbol="SOL"),
        price_usd=price, liquidity_usd=liquidity, market_cap_usd=450_000, fdv_usd=450_000,
        windows=[WindowStats(window="h1", buys=60, sells=40, volume_usd=30_000),
                 WindowStats(window="h24", buys=600, sells=400, volume_usd=300_000)],
    )  # fmt: skip


# --- Harness ---------------------------------------------------------------------------------


def ranked(
    tmp_path: Path, scenarios: list[Any] | None = None
) -> tuple[GrowthScoutResult, ScoutSnapshotStore]:
    svc = service(tmp_path)
    scenarios = scenarios or [accelerating("A"), fading("D"), flat("E")]

    async def go() -> GrowthScoutResult:
        for c, snaps in scenarios:
            await svc.store.record_seen(c)
            await svc.store.save_snapshot(
                ScoutSnapshot(canonical_id=c.canonical_id, observed_at=c.observed_at,
                              provider=c.market_provider, pool_address=c.pool.address,
                              dex=c.pool.dex, metrics=c.metrics), 0)  # fmt: skip
            for s in snaps:
                await svc.store.save_snapshot(s, 0)
        return await svc.rank([c for c, _ in scenarios], limit=500)

    return run(go()), svc.store


def outcome_store(tmp_path: Path) -> OutcomeStore:
    return OutcomeStore(tmp_path / "scout.sqlite3")


def later_run(
    result: GrowthScoutResult, minutes: float, **changes: dict[str, Any]
) -> GrowthScoutResult:
    """The same ranking `minutes` later, with per-token updates (`observed_at` moves on
    unless a change says otherwise)."""
    at = result.computed_at + timedelta(minutes=minutes)
    candidates = []
    for g in result.candidates:
        key = next(k for k, m in MINTS.items() if g.canonical_id.endswith(m))
        update: dict[str, Any] = {"observed_at": at}
        update.update(changes.get(key, {}))
        candidates.append(g.model_copy(update=update, deep=True))
    return result.model_copy(update={"computed_at": at, "candidates": candidates})


def with_score(g: GrowthCandidate, score: float) -> dict[str, Any]:
    return {"scout_momentum": g.scout_momentum.model_copy(update={"score": score})}


def collector(
    store: OutcomeStore,
    clock: Clock,
    scout_store: ScoutSnapshotStore | None = None,
    candles: Any = None,
    pools: Any = None,
    config: OutcomeConfig = IMMEDIATE,
    busy: Any = lambda: False,
) -> OutcomeCollector:
    sources = [] if pools is None else pools if isinstance(pools, list) else [pools]
    return OutcomeCollector(store, config, scout_store, candles, sources, busy=busy, now=clock)


def horizon(store: OutcomeStore, oid: int, label: str, kind: str = "scout") -> HorizonOutcome:
    rows = run(store.horizons(kind, [oid]))[oid]  # type: ignore[arg-type]
    return next(h for h in rows if h.horizon == label)


def anchor(
    tmp_path: Path, key: str = "A"
) -> tuple[OutcomeStore, ScoutObservation, ScoutSnapshotStore]:
    result, scout = ranked(tmp_path)
    store = outcome_store(tmp_path)
    stored = run(record_scout_run(store, result, CFG))
    obs = next(o for o in stored if o.canonical_id == f"solana:{MINTS[key]}")
    return store, obs, scout


def raw(store: OutcomeStore) -> sqlite3.Connection:
    return sqlite3.connect(store.path)


# --- A / B / C: observations and the observation policy --------------------------------------


def test_a_ranked_candidate_creates_an_immutable_observation(tmp_path: Path) -> None:
    result, _ = ranked(tmp_path)
    store = outcome_store(tmp_path)
    stored = run(record_scout_run(store, result, CFG))
    assert len(stored) == len(result.candidates) and all(o.id for o in stored)
    top = result.candidates[0]
    o = next(x for x in stored if x.canonical_id == top.canonical_id)
    assert (o.chain, o.address, o.pool_address) == (
        top.chain,
        top.address,
        top.market.selected_pool.address,
    )
    assert o.rank == 1 and o.in_top10 and o.in_top20 and o.anchor_reason == "FIRST_RANKED"
    assert o.stage == top.stage and o.score == pytest.approx(top.score)
    assert o.components.market_activity is not None and o.components.base == pytest.approx(
        top.scout_momentum.base
    )
    assert o.market.price_usd == top.market.price_usd and o.market.volume_h1_usd is not None
    assert o.market.txns_h24 is not None and o.market.buy_share_h1 is not None
    assert (
        o.social.status == top.momentum.social_status
        and o.safety.status == top.quality.safety_status
    )
    assert o.reasons == top.reasons_surfaced and o.observed_at == top.observed_at
    assert o.run_id == result.computed_at.isoformat()
    # Every fixed horizon is created PENDING at its due time.
    rows = run(store.horizons("scout", [o.id or 0]))[o.id or 0]
    assert [(h.horizon, h.status) for h in rows] == [
        ("5m", "PENDING"), ("15m", "PENDING"), ("1h", "PENDING"), ("4h", "PENDING"), ("24h", "PENDING")
    ]  # fmt: skip
    assert rows[-1].due_at == o.observed_at + timedelta(hours=24)
    # The database itself refuses to rewrite or delete it.
    db = raw(store)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute("UPDATE scout_outcome_observations SET stage = 'FADING' WHERE id = ?", (o.id,))
    with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
        db.execute("DELETE FROM scout_outcome_observations WHERE id = ?", (o.id,))
    assert run(store.scout_observation(o.id or 0)) == o


def test_b_identical_rerun_creates_no_duplicate_anchor(tmp_path: Path) -> None:
    result, _ = ranked(tmp_path)
    store = outcome_store(tmp_path)
    first = run(record_scout_run(store, result, CFG))
    assert run(record_scout_run(store, result, CFG)) == []  # the same run again
    # A minute later, newer evidence, same stages and scores: nothing material changed.
    assert run(record_scout_run(store, later_run(result, 1), CFG)) == []
    # The same evidence carried into a later run (e.g. STALE_CARRIED) is never re-anchored.
    same = later_run(result, 2, **{k: {"observed_at": NOW} for k in MINTS})
    assert run(record_scout_run(store, same, CFG)) == []
    assert run(store.counts())["scout"]["observations"] == len(first)


def test_c_material_changes_create_new_anchors(tmp_path: Path) -> None:
    result, _ = ranked(tmp_path)
    store = outcome_store(tmp_path)
    run(record_scout_run(store, result, CFG))
    a = next(g for g in result.candidates if g.canonical_id.endswith(MINTS["A"]))
    new_stage = "CROWDED" if a.stage != "CROWDED" else "FADING"
    stage_run = later_run(result, 3, A={"stage": new_stage})
    [changed] = run(record_scout_run(store, stage_run, CFG))
    assert changed.anchor_reason == "STAGE_CHANGE" and changed.stage == new_stage
    original = run(store.scout_observations(canonical_id=a.canonical_id))[-1]
    assert original.stage == a.stage  # never rewritten
    assert changed.previous_observation_id == original.id

    score_run = later_run(stage_run, 3, A={"stage": new_stage} | with_score(a, a.score - 15))
    [moved] = run(record_scout_run(store, score_run, CFG))
    assert moved.anchor_reason == "SCORE_CHANGE"

    elapsed = later_run(score_run, 6 * 60 + 1, A={"stage": new_stage} | with_score(a, a.score - 15))
    anchored = {o.canonical_id: o.anchor_reason for o in run(record_scout_run(store, elapsed, CFG))}
    assert anchored[a.canonical_id] == "TIME_ELAPSED"


def test_c_reentry_after_leaving_the_ranking(tmp_path: Path) -> None:
    result, _ = ranked(tmp_path)
    store = outcome_store(tmp_path)
    run(record_scout_run(store, result, CFG))
    a_id = f"solana:{MINTS['A']}"
    without_a = later_run(result, 20)
    without_a.candidates = [g for g in without_a.candidates if g.canonical_id != a_id]
    run(record_scout_run(store, without_a, CFG))
    back = later_run(result, 45)
    anchored = {o.canonical_id: o.anchor_reason for o in run(record_scout_run(store, back, CFG))}
    assert anchored == {a_id: "REENTRY"}  # the others never left
    # A token that missed a run, but only briefly, is not a re-entry.
    policy = OutcomeConfig(observation=ObservationPolicy(reentry_gap_minutes=120))
    store2 = OutcomeStore(tmp_path / "other.sqlite3")
    run(record_scout_run(store2, result, policy))
    run(record_scout_run(store2, without_a, policy))
    assert run(record_scout_run(store2, back, policy)) == []


def test_max_rank_limits_anchors(tmp_path: Path) -> None:
    result, _ = ranked(tmp_path)
    store = outcome_store(tmp_path)
    config = OutcomeConfig(observation=ObservationPolicy(max_rank=1))
    stored = run(record_scout_run(store, result, config))
    assert [o.rank for o in stored] == [1]
    # Every ranked token still counts as surfaced.
    history = run(store.surfacing([g.canonical_id for g in result.candidates]))
    assert len(history) == len(result.candidates)


# --- D / E / F: fixed horizons -------------------------------------------------------------


def test_d_5m_horizon_stays_pending_before_5m(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    clock = Clock(obs.observed_at + 4 * MINUTE)
    candles = FakeCandles()
    report = run(collector(store, clock, scout, candles, FakePools()).collect())
    assert report.due == 0 and candles.calls == []
    clock.at = obs.observed_at + 5 * MINUTE + 30 * timedelta(seconds=1)  # before settling
    assert run(collector(store, clock, scout, candles, FakePools()).collect()).due == 0
    assert horizon(store, obs.id or 0, "5m").status == "PENDING"


def test_e_5m_completes_with_real_evidence(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    candles = FakeCandles()
    p0 = obs.market.price_usd or 0
    candles.by_pool[obs.pool_address] = minute_candles(
        start, [(p0, p0 * 1.05, p0 * 0.99, p0 * 1.03)] * 5
    )
    pools = FakePools()
    pools.by_token[obs.canonical_id] = [dex_pool("A", obs.pool_address, liquidity=90_000)]
    clock = Clock(start + 5 * MINUTE + CFG.collector.settle_seconds * timedelta(seconds=1))
    report = run(collector(store, clock, scout, candles, pools).collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.status == "COMPLETE" and h.market_status == "ACTIVE"
    assert h.price is not None and h.price.source == "candles" and h.price.points == 5
    assert h.price.return_pct == pytest.approx(3.0)
    assert h.price.mfe_pct == pytest.approx(5.0) and h.price.mae_pct == pytest.approx(-1.0)
    assert h.market is not None and h.market.source == "pool_lookup"
    assert h.market.liquidity_change_pct == pytest.approx(
        (90_000 / (obs.market.liquidity_usd or 1) - 1) * 100
    )
    # One batched pool lookup for every token due, then candles best-ranked first within
    # GeckoTerminal's per-cycle budget (3): the last pool waits for a later cycle.
    assert report.requests == {"GeckoTerminal": 3} and report.deferred_for_quota == 1
    assert report.finalized["COMPLETE"] == 1 and report.still_pending == 1
    assert h.future_stage is None  # no Growth Scout run near 12:05 in this store


def test_f_24h_cannot_complete_at_4h(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    p0 = obs.market.price_usd or 0
    candles = FakeCandles()
    candles.by_pool[obs.pool_address] = [
        candle(start + i * 5 * MINUTE, p0, p0 * 1.1, p0 * 0.9, p0) for i in range(60)
    ]
    clock = Clock(start + timedelta(hours=4, minutes=2))
    run(collector(store, clock, scout, candles, FakePools()).collect())
    assert horizon(store, obs.id or 0, "24h").status == "PENDING"
    assert horizon(store, obs.id or 0, "24h").attempts == 0
    assert horizon(store, obs.id or 0, "4h").price is not None


def test_short_horizons_share_one_candle_request_with_the_1h_horizon(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    p0 = obs.market.price_usd or 0
    candles = FakeCandles()
    candles.by_pool[obs.pool_address] = minute_candles(start, [(p0, p0 * 1.1, p0 * 0.9, p0)] * 70)
    pools = FakePools()
    pools.by_token[obs.canonical_id] = [dex_pool("A", obs.pool_address)]
    clock = Clock(start + 7 * MINUTE)
    c = collector(store, clock, scout, candles, pools, config=CFG)
    report = run(c.collect())
    h5 = horizon(store, obs.id or 0, "5m")
    # The horizon-end market state is captured now; the price path waits for the 1h request.
    assert h5.status == "PENDING" and h5.market is not None and h5.price is None
    assert any("wait for the 1h horizon" in m for m in h5.missing)
    assert candles.calls == [] and report.waiting_to_share_candles == 3
    for minutes in (17, 62):  # the 15m horizon ends; then the 1h horizon ends
        clock.at = start + minutes * MINUTE
        run(c.collect())
    rows = {h.horizon: h for h in run(store.horizons("scout", [obs.id or 0]))[obs.id or 0]}
    assert {k: rows[k].status for k in ("5m", "15m", "1h")} == {
        "5m": "COMPLETE", "15m": "COMPLETE", "1h": "COMPLETE"
    }  # fmt: skip
    assert [x[0] for x in candles.calls].count(obs.pool_address) == 1  # one request for all three
    assert rows["5m"].price is not None and rows["5m"].price.points == 5
    assert rows["1h"].price is not None and rows["1h"].price.points == 60
    assert rows["4h"].status == "PENDING"


def test_stored_parts_are_kept_across_attempts(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    p0 = obs.market.price_usd or 0
    pools = FakePools()
    pools.by_token[obs.canonical_id] = [dex_pool("A", obs.pool_address, liquidity=70_000)]
    candles = FakeCandles(headroom=0)
    clock = Clock(start + 7 * MINUTE)
    c = collector(store, clock, scout, candles, pools)
    run(c.collect())  # market state captured; candles deferred
    first = horizon(store, obs.id or 0, "5m")
    assert first.status == "PENDING" and first.market is not None
    candles._headroom = 10
    candles.by_pool[obs.pool_address] = minute_candles(start, [(p0, p0 * 1.01, p0, p0 * 1.01)] * 5)
    clock.at = start + 30 * MINUTE  # long after the horizon end: no new lookup possible
    run(c.collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.status == "COMPLETE" and h.market == first.market
    assert not any("market state not observed" in m for m in h.missing)


# --- G / H: identity ---------------------------------------------------------------------------


def test_g_exact_identity_and_pool_are_preserved(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    # Scout later observed the same token on another pool: never used for this outcome.
    other = ScoutSnapshot(canonical_id=obs.canonical_id, observed_at=start + 5 * MINUTE,
                          provider="GeckoTerminal", pool_address="another-pool", dex="orca",
                          metrics=ScoutMarketMetrics(price_usd=9.0, liquidity_usd=1.0))  # fmt: skip
    run(scout.save_snapshot(other, 0))
    candles = FakeCandles()
    candles.by_pool["another-pool"] = minute_candles(start, [(9, 9, 9, 9)] * 5)
    clock = Clock(start + 7 * MINUTE)
    run(collector(store, clock, scout, candles, FakePools(headroom=0)).collect())
    requested = {c[0] for c in candles.calls}
    assert obs.pool_address in requested and "another-pool" not in requested
    h = horizon(store, obs.id or 0, "5m")
    assert h.market is None or h.market.price_usd != 9.0
    assert any("other pool" in m for m in h.missing)


def test_h_same_ticker_different_contract_cannot_contaminate(tmp_path: Path) -> None:
    impostor_mint = MINTS["F"]
    result, scout = ranked(tmp_path, [accelerating("A", symbol="PEPE"), flat("F", symbol="PEPE")])
    store = outcome_store(tmp_path)
    stored = {o.address: o for o in run(record_scout_run(store, result, CFG))}
    real, impostor = stored[MINTS["A"]], stored[impostor_mint]
    assert real.symbol == impostor.symbol == "PEPE" and real.canonical_id != impostor.canonical_id
    start = real.observed_at
    candles = FakeCandles()
    p0 = real.market.price_usd or 0
    candles.by_pool[real.pool_address] = minute_candles(start, [(p0, p0 * 1.02, p0, p0)] * 5)
    candles.by_pool[impostor.pool_address] = minute_candles(start, [(1, 50, 0.01, 40)] * 5)
    pools = FakePools()
    # A lookup of the real token that also lists the impostor's pool (same ticker).
    pools.by_token[real.canonical_id] = [
        dex_pool("A", real.pool_address),
        dex_pool("F", impostor.pool_address, price=40),
    ]
    run(collector(store, Clock(start + 7 * MINUTE), scout, candles, pools).collect())
    h = horizon(store, real.id or 0, "5m")
    assert h.price is not None and h.price.mfe_pct == pytest.approx(2.0)
    assert h.market is not None and h.market.price_usd == pytest.approx(0.0011)


# --- I / J: MFE / MAE -------------------------------------------------------------------------


def test_i_j_mfe_mae_from_the_spec_example() -> None:
    start = NOW
    interval = timedelta(minutes=15)
    rows = [
        (0.0010, 0.00105, 0.00082, 0.00090),  # the low: -18% at 0:00
        (0.00090, 0.0012, 0.00088, 0.0011),
        (0.0011, 0.0019, 0.0011, 0.0016),  # the high: +90% at 0:30
        (0.0016, 0.0017, 0.0013, 0.0014),  # ends at 0.0014: +40%
    ]
    candles = [candle(start + i * interval, *r) for i, r in enumerate(rows)]
    candles.insert(0, candle(start - interval, 0.001, 0.005, 0.0001, 0.001))  # before: ignored
    path = path_from_candles(candles, interval, 0.0010, start, start + 4 * interval,
                             provider="x", timeframe="15m", price_drop_pct=90)  # fmt: skip
    assert path.points == 4
    assert path.return_pct == pytest.approx(40.0)
    assert path.mfe_pct == pytest.approx(90.0) and path.highest_price == 0.0019
    assert path.mae_pct == pytest.approx(-18.0) and path.lowest_price == 0.00082
    assert path.time_to_mfe_minutes == 30 and path.time_to_mae_minutes == 0
    # Drawdown: the 0:45 low (0.0013) against the 0.0019 peak reached before it.
    assert path.max_drawdown_pct == pytest.approx((0.0013 / 0.0019 - 1) * 100)
    assert path.price_collapsed is False


def test_mfe_mae_are_zero_bounded_and_empty_windows_are_not_measured() -> None:
    interval = MINUTE
    up = [candle(NOW + i * MINUTE, 1.1, 1.2, 1.05, 1.15) for i in range(3)]
    path = path_from_candles(up, interval, 1.0, NOW, NOW + 3 * MINUTE, provider="x",
                             timeframe="1m", price_drop_pct=90)  # fmt: skip
    assert (
        path.mae_pct == 0 and path.time_to_mae_minutes is None and path.mfe_pct == pytest.approx(20)
    )
    empty = path_from_candles([], interval, 1.0, NOW, NOW + 3 * MINUTE, provider="x",
                              timeframe="1m", price_drop_pct=90)  # fmt: skip
    assert empty.points == 0 and empty.return_pct is None and empty.mfe_pct is None


# --- K / L: terminal and missing outcomes -----------------------------------------------------


def test_k_liquidity_collapse_stays_in_the_dataset(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    p0 = obs.market.price_usd or 0
    candles = FakeCandles()
    candles.by_pool[obs.pool_address] = minute_candles(start, [(p0, p0, p0 * 0.05, p0 * 0.06)] * 5)
    pools = FakePools()
    pools.by_token[obs.canonical_id] = [
        dex_pool("A", obs.pool_address, liquidity=400, price=p0 * 0.06)
    ]
    run(collector(store, Clock(start + 7 * MINUTE), scout, candles, pools).collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.status == "COMPLETE" and h.market_status == "LIQUIDITY_COLLAPSE"
    assert h.market is not None and h.market.liquidity_collapsed
    assert (
        h.price is not None
        and h.price.price_collapsed
        and h.price.return_pct == pytest.approx(-94.0)
    )
    summary = summarize_scout(run(store.all_scout()), "all", "5m", CFG.analytics)
    [c] = summary.cohorts
    assert c.liquidity_collapses == 1 and c.price_collapses == 1 and c.complete == 1


def test_pool_gone_is_recorded_not_dropped(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    candles = FakeCandles()  # GeckoTerminal no longer has candles either
    pools = FakePools()
    pools.by_token[obs.canonical_id] = [dex_pool("A", "some-new-pool")]
    run(collector(store, Clock(start + 7 * MINUTE), scout, candles, pools).collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.status == "PARTIAL" and h.market_status == "POOL_GONE"
    assert h.market is not None and not h.market.pool_found


def test_a_pool_missing_from_another_source_is_not_called_gone(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)  # priced by GeckoTerminal
    dexscreener = FakePools(name="DEX Screener")  # knows the token, not this pool
    dexscreener.by_token[obs.canonical_id] = [dex_pool("A", "a-different-pool")]
    candles = FakeCandles()
    run(
        collector(store, Clock(obs.observed_at + 7 * MINUTE), scout, candles, dexscreener).collect()
    )
    h = horizon(store, obs.id or 0, "5m")
    assert h.market_status != "POOL_GONE" and h.market is None
    assert any("does not list this pool (priced by GeckoTerminal)" in m for m in h.missing)


def test_l_provider_unavailable_is_partial_or_unavailable_never_deleted(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    candles, pools = FakeCandles(), FakePools()
    candles.error = MarketDataUnavailableError("GeckoTerminal returned HTTP 503")
    pools.error = MarketDataUnavailableError("could not reach DEX Screener")
    clock = Clock(start + 7 * MINUTE)
    c = collector(store, clock, scout, candles, pools)
    run(c.collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.status == "PENDING" and h.attempts == 1 and h.market_status == "PROVIDER_UNAVAILABLE"
    assert any("HTTP 503" in m for m in h.missing) and any("DEX Screener" in m for m in h.missing)
    # Retried after `retry_seconds`, then finalized with its reasons after `retry_minutes`.
    clock.at = start + timedelta(minutes=5 + 121)
    run(c.collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.status == "UNAVAILABLE" and h.finalized_at == clock.at
    assert any("not collected within" in m for m in h.missing)
    assert any("market state not observed" in m for m in h.missing)
    assert run(store.scout_observation(obs.id or 0)) is not None


def test_snapshot_only_evidence_is_partial_with_lower_bound_note(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    p0 = obs.market.price_usd or 0
    for minutes, price in ((2, p0 * 1.3), (5, p0 * 1.1)):
        snap = ScoutSnapshot(
            canonical_id=obs.canonical_id, observed_at=start + minutes * MINUTE,
            provider="GeckoTerminal", pool_address=obs.pool_address, dex=obs.pool_dex,
            metrics=ScoutMarketMetrics(price_usd=price, liquidity_usd=85_000, windows=[
                ScoutWindow(window="h1", volume_usd=1.0, buys=1, sells=1)]),
        )  # fmt: skip
        run(scout.save_snapshot(snap, 0))
    candles = FakeCandles(headroom=0)  # the background quota never allows a candle request
    clock = Clock(start + 7 * MINUTE)
    c = collector(store, clock, scout, candles, FakePools())
    report = run(c.collect())
    h = horizon(store, obs.id or 0, "5m")
    assert report.reused_snapshots == 1 and h.status == "PENDING" and candles.calls == []
    assert h.market is not None and h.market.source == "scout_snapshot"
    clock.at = start + timedelta(hours=3)
    run(c.collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.status == "PARTIAL" and h.price is not None and h.price.source == "snapshots"
    assert h.price.return_pct == pytest.approx(10.0) and h.price.mfe_pct == pytest.approx(30.0)
    assert "lower bounds" in h.price.notes[0]


# --- M: idempotence ----------------------------------------------------------------------------


def test_m_rerunning_the_collector_is_idempotent(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    p0 = obs.market.price_usd or 0
    candles = FakeCandles()
    candles.by_pool[obs.pool_address] = minute_candles(start, [(p0, p0 * 1.1, p0, p0)] * 70)
    pools = FakePools()
    pools.by_token[obs.canonical_id] = [dex_pool("A", obs.pool_address)]
    clock = Clock(start + 62 * MINUTE)
    c = collector(store, clock, scout, candles, pools)
    run(c.collect())
    statuses = {
        h.horizon: h.status for h in run(store.horizons("scout", [obs.id or 0]))[obs.id or 0]
    }
    # 1h: candles + a fresh pool lookup. 5m / 15m ended too long ago for a lookup to
    # describe their horizon end: price measured, market state missing (with its reason).
    assert statuses == {"5m": "PARTIAL", "15m": "PARTIAL", "1h": "COMPLETE", "4h": "PENDING",
                        "24h": "PENDING"}  # fmt: skip
    before = run(store.horizons("scout", [obs.id or 0]))
    calls = len(candles.calls)
    run(c.collect())
    # Finalized horizons are never measured again (only still-pending ones of other tokens).
    assert [x for x in candles.calls[calls:] if x[0] == obs.pool_address] == []
    after = run(store.horizons("scout", [obs.id or 0]))[obs.id or 0]
    assert [h for h in after if h.status != "PENDING"] == [
        h for h in before[obs.id or 0] if h.status != "PENDING"
    ]
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
        raw(store).execute(
            "UPDATE scout_outcome_horizons SET return_pct = 1 WHERE observation_id = ? AND horizon = '5m'",
            (obs.id,),
        )
    assert (
        run(
            store.update_horizon(
                "scout", obs.id or 0, "5m", HorizonUpdate(finalize="UNAVAILABLE"), clock.at
            )
        )
        is False
    )
    # One candle request served the three 1m horizons due together.
    assert len([x for x in candles.calls if x[0] == obs.pool_address]) == 1


# --- N / O / P / Q: decisions ----------------------------------------------------------------


def assessment(action: str, **kw: Any) -> OpportunityAssessment:
    fields: dict[str, Any] = dict(
        asset="TOKA", timeframe="1h", requested_timeframe=None, action=action, confidence="medium",
        confirmed=action != "wait", summary=f"{action} because of synthetic evidence",
        bullish_score=5, bearish_score=1, bullish_evidence=[], bearish_evidence=[],
        blocking_factors=[], cautions=[], bullish_trigger=None, bearish_trigger=None,
        risk_level="medium", uncertainty_level="medium", missing_evidence=[], live_price=1.0,
        last_close=0.99, position="none", intent="enter_long" if action == "buy" else "wait",
    )  # fmt: skip
    fields.update(kw)
    return OpportunityAssessment(**fields)


def chat(
    a: OpportunityAssessment, pool: str = "pool-A", scout: bool = True
) -> tuple[ChatRequest, ChatResponse]:
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Analyze")],
        asset=AssetRef(chain="solana", address=MINTS["A"], symbol="A", pool_address=pool)
        if scout
        else None,
    )
    results = [
        AgentResult(agent="opportunity", mock=False, summary="x", findings=a.model_dump(mode="json")),
        AgentResult(agent="technical_analysis", mock=False, summary="t",
                    findings={"provider": "GeckoTerminal", "symbol": "A",
                              "pools": {"market_pool": {"address": pool}, "technical_pool": {"address": pool}}}),
        AgentResult(agent="risk", mock=False, summary="Overall risk for A: medium."),
    ]  # fmt: skip
    analysis = Analysis(mock=False, summary="s", uncertainty=Uncertainty(level="medium"),
                        agent_results=results, disclaimer="d")  # fmt: skip
    return request, ChatResponse(
        message=ChatMessage(role="assistant", content="x"), analysis=analysis
    )


def buy_trigger(price: float) -> Trigger:
    return Trigger(action="buy", condition=f"1h close above ~${price}", price=price,
                   basis="resistance_zone_upper", confirmed=True)  # fmt: skip


def test_n_decision_observation_is_immutable_and_linked_to_scout(tmp_path: Path) -> None:
    store, obs, _ = anchor(tmp_path)
    a = assessment("buy", bullish_trigger=buy_trigger(1.1),
                   invalidation=Invalidation(condition="1h close below ~$0.95", price=0.95,
                                             basis="support_zone_lower", timeframe="1h"))  # fmt: skip
    request, response = chat(a, pool=obs.pool_address)
    at = obs.observed_at + 10 * MINUTE
    d = run(record_decision(store, request, response, at, CFG))
    assert d is not None and d.id and d.source == "scout" and d.scout_observation_id == obs.id
    assert d.action == "buy" and d.asset_id == obs.canonical_id and d.timeframe == "1h"
    assert d.buy_trigger is not None and d.buy_trigger.price == 1.1
    assert d.invalidation is not None and d.invalidation.direction == "below"
    assert d.reference_price == 1.0 and d.reference_basis == "live_price"
    assert d.price_ref.kind == "dex_pool" and d.price_ref.pool_address == obs.pool_address
    assert d.risk.summary == "Overall risk for A: medium."
    # A double click is one decision; the stored one can't be rewritten.
    assert run(record_decision(store, request, response, at + timedelta(seconds=5), CFG)) is None
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        raw(store).execute("UPDATE decision_observations SET action = 'sell' WHERE id = ?", (d.id,))
    assert run(store.decision(d.id or 0)) == d


def test_decision_without_a_measurable_market_is_not_recorded(tmp_path: Path) -> None:
    store = outcome_store(tmp_path)
    request, response = chat(assessment("wait"), scout=False)
    response.analysis.agent_results = response.analysis.agent_results[:1]  # type: ignore[union-attr]
    assert run(record_decision(store, request, response, NOW, CFG)) is None


def decision_outcome(
    tmp_path: Path, a: OpportunityAssessment, prices: list[tuple[float, float, float, float]]
) -> HorizonOutcome:
    store = outcome_store(tmp_path)
    request, response = chat(a)
    d = run(record_decision(store, request, response, NOW, CFG))
    assert d is not None
    candles = FakeCandles()
    candles.by_pool["pool-A"] = minute_candles(NOW, prices)
    pools = FakePools(name="DEX Screener")  # Analyze's pools come from DEX Screener
    pools.by_token[f"solana:{MINTS['A']}"] = [dex_pool("A", "pool-A")]
    run(collector(store, Clock(NOW + 17 * MINUTE), None, candles, pools).collect())
    return horizon(store, d.id or 0, "15m", "decision")


def test_o_buy_trigger_reached(tmp_path: Path) -> None:
    prices = (
        [(1.0, 1.02, 0.99, 1.01)] * 6 + [(1.01, 1.12, 1.0, 1.1)] + [(1.1, 1.11, 1.05, 1.08)] * 8
    )
    h = decision_outcome(tmp_path, assessment("buy", bullish_trigger=buy_trigger(1.1)), prices)
    assert h.status == "COMPLETE"
    t = h.triggers["buy_trigger"]
    assert (
        t.reached and t.first_reached_at == NOW + 6 * MINUTE and t.already_beyond_at_start is False
    )
    assert h.first_trigger_event == "buy_trigger" and h.price_position == "below buy trigger"


def test_p_invalidation_reached_first(tmp_path: Path) -> None:
    a = assessment("buy", bullish_trigger=buy_trigger(1.1),
                   invalidation=Invalidation(condition="1h close below ~$0.95", price=0.95,
                                             basis="support_zone_lower", timeframe="1h"))  # fmt: skip
    prices = [(1.0, 1.0, 0.99, 0.99), (0.99, 0.99, 0.94, 0.96)] + [(0.96, 1.0, 0.96, 1.0)] * 8
    prices += [(1.0, 1.15, 1.0, 1.12)] * 5
    h = decision_outcome(tmp_path, a, prices)
    assert h.triggers["invalidation"].first_reached_at == NOW + MINUTE
    assert h.triggers["buy_trigger"].first_reached_at == NOW + 10 * MINUTE
    assert h.first_trigger_event == "invalidation"
    assert h.price is not None and h.price.mae_pct == pytest.approx(-6.0)


def test_q_wait_decision_still_gets_factual_market_outcome(tmp_path: Path) -> None:
    prices = [(1.0, 1.04, 0.97, 1.02)] * 15
    h = decision_outcome(tmp_path, assessment("wait"), prices)
    assert h.status == "COMPLETE" and h.triggers == {} and h.first_trigger_event is None
    assert h.price is not None
    assert (h.price.return_pct, h.price.mfe_pct, h.price.mae_pct) == pytest.approx((2.0, 4.0, -3.0))
    assert h.market is not None and h.market.pool_found


def test_first_event_same_candle_is_ambiguous() -> None:
    c = [candle(NOW, 1.0, 1.2, 0.8, 1.0)]
    triggers = {"buy_trigger": trigger_outcome(c, 1.1, "above", 1.0),
                "invalidation": trigger_outcome(c, 0.9, "below", 1.0)}  # fmt: skip
    assert first_event(triggers) == "same_candle:buy_trigger+invalidation"
    assert first_event({"buy_trigger": trigger_outcome(c, 5, "above", 1.0)}) == "none"


# --- R / S: aggregates -------------------------------------------------------------------------


def measured(value: float, stage_after: str = "EARLY") -> tuple[str | None, HorizonOutcome]:
    price = PricePath(source="candles", points=5, window_start=NOW, window_end=NOW + MINUTE,
                      reference_price=1.0, return_pct=value, mfe_pct=max(value, 0) + 5,
                      mae_pct=min(value, 0) - 5)  # fmt: skip
    return "ACCELERATING", HorizonOutcome(
        observation_id=1, horizon="1h", horizon_minutes=60, due_at=NOW, status="COMPLETE",
        market_status="ACTIVE", price=price, future_stage=stage_after,
    )  # fmt: skip


def test_r_small_sample_reports_insufficient_sample() -> None:
    rows = [measured(float(v)) for v in range(9)]
    c = cohort("ACCELERATING", "1h", rows, AnalyticsConfig())
    assert c.sample_status == "INSUFFICIENT_SAMPLE" and c.measured == 9
    assert c.return_pct is None and c.mfe_pct is None and c.return_distribution is None
    assert c.liquidity_collapse_rate is None
    assert c.note is not None and "n = 9" in c.note and "insufficient sample" in c.note
    assert c.future_stage_counts == {"ACCELERATING -> EARLY": 9}  # raw counts stay visible
    assert "accura" not in c.model_dump_json().lower() and "win" not in c.model_dump_json().lower()


def test_s_median_and_percentiles() -> None:
    values = [float(v) for v in range(1, 11)]
    assert median(values) == 5.5
    assert percentile(values, 25) == pytest.approx(3.25)
    assert percentile(values, 90) == pytest.approx(9.1)
    assert percentile([4.0], 90) == 4.0 and percentile([], 50) is None
    assert distribution([-60, -10, 0, 0, 30, 150], [-50, -5, 5, 50]) == {
        "<-50": 1, "-50..-5": 1, "-5..5": 2, "5..50": 1, ">=50": 1
    }  # fmt: skip
    rows = [measured(float(v)) for v in range(25)] + [(None, None)]
    c = cohort("all", "1h", rows, AnalyticsConfig())
    assert c.sample_status == "SUFFICIENT" and c.observations == 26 and c.pending == 1
    assert c.return_pct is not None and c.return_pct.median == 12 and c.return_pct.mean == 12
    assert c.return_pct.p25 == 6 and c.return_pct.p10 is None  # outer percentiles need 50
    assert c.mfe_pct is not None and c.mfe_pct.median == 17


def test_aggregates_group_by_every_dimension(tmp_path: Path) -> None:
    store, obs, _ = anchor(tmp_path)
    records = run(store.all_scout())
    for dim in ("stage", "score_band", "chain", "age_band", "liquidity_band", "market_cap_band",
                "social_state", "social_support", "safety_status", "risk_flag", "rank_band",
                "discovery_status", "ranking_mode", "anchor_reason"):  # fmt: skip
        summary = summarize_scout(records, dim, "1h", CFG.analytics)
        assert sum(c.observations for c in summary.cohorts) >= summary.total_observations == 3
        assert all(c.pending == c.observations for c in summary.cohorts)
    bands = {c.group for c in summarize_scout(records, "rank_band", "1h", CFG.analytics).cohorts}
    assert bands == {"1-10"}
    support = {
        c.group for c in summarize_scout(records, "social_support", "1h", CFG.analytics).cohorts
    }
    assert support == {"social_unavailable"}  # unavailable is never "market only"


# --- T: quota priority -------------------------------------------------------------------------


def gt_rows(start: datetime, count: int, price: float = 1.0) -> list[list[float]]:
    return [
        [(start + i * MINUTE).timestamp(), price, price * 1.1, price * 0.9, price, 5.0]
        for i in range(count)
    ]


def test_t_collector_respects_provider_quota_priority(
    tmp_path: Path, fake_geckoterminal: Any
) -> None:
    store, obs, scout = anchor(tmp_path)
    clock = Clock(obs.observed_at + 7 * MINUTE)
    # Priced like the observed market (the token's own price, not another token's).
    fake_geckoterminal.candles[obs.pool_address] = gt_rows(
        obs.observed_at, 10, obs.market.price_usd or 1.0
    )
    quota = LaneLimiter(6, 60.0, {"refresh": 2, "interactive": 2})
    dex = DexCandleService(GeckoTerminalProvider(transport=fake_geckoterminal.transport()),
                           limiter=quota, now=clock)  # fmt: skip
    candles = ProviderCandles(dex, upscale.services.market_data_service, min_free=0)
    # Scout discovery used the unreserved capacity; refresh and interactive are held.
    assert quota.try_acquire("default") and quota.try_acquire("default")
    assert quota.available(OUTCOME_LANE) == 0
    c = collector(store, clock, scout, candles, FakePools())
    run(c.collect())
    assert (
        fake_geckoterminal.requests == [] and horizon(store, obs.id or 0, "5m").status == "PENDING"
    )
    assert any("deferred" in m for m in horizon(store, obs.id or 0, "5m").missing)
    # Scout's refresh finished: its unused reservation is free, but "interactive" never is.
    quota.release("refresh")
    clock.advance(minutes=3)
    run(c.collect())
    # Exactly the 2 freed calls were used (one per pool, best-ranked first); never more.
    assert len(fake_geckoterminal.requests) == 2 and quota.available(OUTCOME_LANE) == 0
    request = fake_geckoterminal.requests[0]
    assert f"/pools/{obs.pool_address}/" in request.url.path
    assert "before_timestamp" in request.url.params and request.url.path.endswith("/ohlcv/minute")
    assert quota.available("interactive") == 2 and quota.used(OUTCOME_LANE) == 2
    assert horizon(store, obs.id or 0, "5m").price is not None
    # `min_free_calls` keeps a margin for Analyze only on quotas without an "interactive"
    # reservation; this one holds it (and outcome work outranks Scout), so none is kept.
    polite = ProviderCandles(dex, upscale.services.market_data_service, min_free=2)
    assert polite.headroom(obs.price_ref) == quota.available(OUTCOME_LANE)
    unreserved = LaneLimiter(6, 60.0, clock=lambda: 0.0)
    plain = DexCandleService(GeckoTerminalProvider(), limiter=unreserved, now=clock)
    polite = ProviderCandles(plain, upscale.services.market_data_service, min_free=2)
    assert polite.headroom(obs.price_ref) == unreserved.available(OUTCOME_LANE) - 2 == 4


def test_t_collector_makes_no_requests_while_scout_or_analyze_runs(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    candles, pools = FakeCandles(), FakePools()
    c = collector(
        store, Clock(obs.observed_at + 7 * MINUTE), scout, candles, pools, busy=lambda: True
    )
    report = run(c.collect())
    assert report.network_skipped and candles.calls == [] and pools.calls == []
    assert horizon(store, obs.id or 0, "5m").status == "PENDING"


def test_rate_limited_provider_is_not_hammered(tmp_path: Path) -> None:
    result, scout = ranked(tmp_path)
    store = outcome_store(tmp_path)
    stored = run(record_scout_run(store, result, CFG))
    candles = FakeCandles()
    candles.error = ProviderRateLimitedError("429")
    c = collector(store, Clock(NOW + 7 * MINUTE), scout, candles, FakePools())
    run(c.collect())
    assert len(candles.calls) == 1  # the first 429 stops candle requests for this cycle
    assert all(
        any("rate limited" in m for m in horizon(store, o.id or 0, "5m").missing) for o in stored
    )


def test_scheduler_sleeps_until_the_next_due_horizon(tmp_path: Path) -> None:
    store, obs, _ = anchor(tmp_path)
    cc = CFG.collector
    wake = run(store.next_wake(cc.settle_seconds, cc.retry_seconds))
    assert wake == obs.observed_at + timedelta(minutes=5, seconds=cc.settle_seconds)


def test_run_forever_collects_and_stops(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    candles = FakeCandles()
    c = collector(store, Clock(obs.observed_at + 7 * MINUTE), scout, candles, FakePools(headroom=0))

    async def go() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(c.run_forever(stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, 2)

    run(go())
    assert c.last_cycle is not None and c.last_cycle.due >= 1


# --- U: migration ------------------------------------------------------------------------------


def test_u_migration_from_the_current_scout_schema(tmp_path: Path) -> None:
    path = tmp_path / "scout.sqlite3"
    result, scout = ranked(tmp_path)  # a v2 Scout store with tokens, snapshots
    run(scout.record_stages(NOW, {g.canonical_id: g.stage for g in result.candidates}))
    social = SocialStore(path)
    run(social.author_salt())
    db = sqlite3.connect(path)
    before = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("scout_tokens", "scout_snapshots", "scout_growth_stages", "scout_meta")}  # fmt: skip
    store = OutcomeStore(path)
    run(record_scout_run(store, result, CFG))
    assert db.execute("PRAGMA user_version").fetchone()[0] == 2  # Scout's version untouched
    assert {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in before} == before
    assert db.execute(
        "SELECT value FROM outcome_meta WHERE key = 'outcome_schema_version'"
    ).fetchone() == ("1",)
    # The Scout stores keep working on the same file.
    assert run(scout.first_seen(result.candidates[0].canonical_id)) is not None
    store.close()
    reopened = OutcomeStore(path)
    assert run(reopened.counts())["scout"]["observations"] == 3
    # A horizon added later (e.g. 7d) is created for existing observations; nothing else moves.
    week = HorizonSpec(label="7d", minutes=7 * 1440, candles="1h", retry_minutes=2880)
    assert run(reopened.ensure_horizons([*CFG.horizons, week])) == 3
    assert run(reopened.ensure_horizons([*CFG.horizons, week])) == 0
    db.execute("UPDATE outcome_meta SET value = '99' WHERE key = 'outcome_schema_version'")
    db.commit()
    with pytest.raises(OutcomeStoreError, match="v99"):
        run(OutcomeStore(path).counts())


# --- Replay ------------------------------------------------------------------------------------


def test_replay_is_read_only_and_labeled(tmp_path: Path) -> None:
    result, scout = ranked(tmp_path)
    ranks = {g.canonical_id: (g.rank, g.score) for g in result.candidates}
    run(scout.record_stages(NOW, {g.canonical_id: g.stage for g in result.candidates}, ranks))
    a = result.candidates[0]
    for minutes, price in ((5, 0.0012), (60, 0.0009)):
        run(scout.save_snapshot(
            ScoutSnapshot(canonical_id=a.canonical_id, observed_at=NOW + minutes * MINUTE,
                          provider="GeckoTerminal", pool_address=a.market.selected_pool.address,
                          dex="raydium", metrics=ScoutMarketMetrics(price_usd=price)), 0))  # fmt: skip
    scout.close()
    path = tmp_path / "scout.sqlite3"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = replay(path)
    assert report.label == REPLAY_LABEL and "not a backtest" in report.label
    top = next(o for o in report.observations if o.canonical_id == a.canonical_id)
    by = {h.horizon: h for h in top.horizons}
    assert by["5m"].status == "MEASURED" and by["5m"].return_pct == pytest.approx(
        (0.0012 / (a.market.price_usd or 1) - 1) * 100
    )
    assert by["1h"].mae_lower_bound_pct is not None and by["4h"].status == "NOT_ELAPSED"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


# --- Providers ---------------------------------------------------------------------------------


def test_gt_window_keeps_gaps_and_sends_before_timestamp(fake_geckoterminal: Any) -> None:
    rows = gt_rows(NOW, 3) + [[(NOW + 10 * MINUTE).timestamp(), 1, 1, 1, 1, 1]]  # a gap
    fake_geckoterminal.candles["pool-x"] = rows
    dex = DexCandleService(GeckoTerminalProvider(transport=fake_geckoterminal.transport()),
                           now=lambda: NOW + timedelta(hours=1))  # fmt: skip
    series = run(dex.get_window("solana", "pool-x", "1m", 20, before=NOW + 15 * MINUTE,
                                lane=OUTCOME_LANE, canonical_id="solana:x"))  # fmt: skip
    assert [c.timestamp for c in series.candles][-1] == NOW + 10 * MINUTE and len(
        series.candles
    ) == 4
    params = fake_geckoterminal.requests[0].url.params
    assert params["before_timestamp"] == str(int((NOW + 15 * MINUTE).timestamp()))


def test_geckoterminal_exact_pools_check_the_base_token() -> None:
    from upscale.services.outcomes import GeckoTerminalPools
    from upscale.services.scout.providers import GeckoTerminalDiscoveryProvider

    requests: list[httpx2.Request] = []

    def pool(address: str, base: str) -> dict[str, Any]:
        return {"type": "pool", "id": f"solana_{address}",
                "attributes": {"address": address, "base_token_price_usd": "0.002",
                               "reserve_in_usd": "12345"},
                "relationships": {"base_token": {"data": {"id": f"solana_{base}"}},
                                  "quote_token": {"data": {"id": f"solana_{SOL}"}},
                                  "dex": {"data": {"id": "raydium"}}}}  # fmt: skip

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200, json={"data": [pool("pool-A", MINTS["A"]), pool("pool-X", MINTS["F"])]}
        )

    source = GeckoTerminalPools(
        GeckoTerminalDiscoveryProvider(transport=httpx2.MockTransport(handle)), min_free=0
    )
    found = run(source.pools("solana", [(MINTS["A"], "pool-A"), (MINTS["A"], "pool-X")]))
    assert (
        found[(MINTS["A"], "pool-A")] is not None
        and found[(MINTS["A"], "pool-A")].liquidity_usd == 12345
    )  # type: ignore[union-attr]
    assert found[(MINTS["A"], "pool-X")] is None  # another token's pool: never this token's market
    assert len(requests) == 1 and "/pools/multi/pool-A,pool-X" in requests[0].url.path


def test_dexscreener_token_pools_are_exact_and_batched() -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        pair = {"chainId": "solana", "dexId": "raydium", "pairAddress": "pool-A",
                "baseToken": {"address": MINTS["A"], "symbol": "PEPE"},
                "quoteToken": {"address": SOL, "symbol": "SOL"}, "priceUsd": "0.001",
                "liquidity": {"usd": 5000}}  # fmt: skip
        impostor = pair | {
            "pairAddress": "pool-F",
            "baseToken": {"address": MINTS["F"], "symbol": "PEPE"},
        }
        return httpx2.Response(200, json=[pair, impostor])

    provider = DexScreenerDiscoveryProvider(transport=httpx2.MockTransport(handle))
    pools = run(provider.token_pools("solana", [MINTS["A"], MINTS["B"]]))
    assert [p.pair_address for p in pools[f"solana:{MINTS['A']}"]] == ["pool-A"]
    assert pools[f"solana:{MINTS['B']}"] == [] and f"solana:{MINTS['F']}" not in pools
    assert len(requests) == 1


# --- API and app wiring --------------------------------------------------------------------------


@pytest.fixture
def api_store(tmp_path: Path, isolated_outcome_store: OutcomeStore) -> OutcomeStore:
    result, _ = ranked(tmp_path)
    run(record_scout_run(isolated_outcome_store, result, CFG))
    return isolated_outcome_store


def test_outcome_api(client: TestClient, api_store: OutcomeStore) -> None:
    listed = client.get("/outcomes/scout").json()
    assert len(listed) == 3 and len(listed[0]["horizons"]) == 5
    one = client.get(f"/outcomes/scout/{listed[0]['observation']['id']}").json()
    assert one["observation"]["canonical_id"] == listed[0]["observation"]["canonical_id"]
    assert client.get("/outcomes/scout/9999").status_code == 404
    summary = client.get("/outcomes/summary?group_by=stage&horizon=1h").json()
    assert summary["total_observations"] == 3 and "not a win rate" in summary["disclaimer"]
    assert all(c["sample_status"] == "NO_DATA" for c in summary["cohorts"])
    assert client.get("/outcomes/summary?group_by=nonsense").status_code == 422
    assert client.get("/outcomes/summary?horizon=3m").status_code == 422
    assert client.get("/outcomes/decisions").json() == []
    assert client.get("/outcomes/decisions/1").status_code == 404
    status = client.get("/outcomes/status").json()
    assert status["counts"]["scout"] == {"observations": 3, "PENDING": 15}
    assert client.get("/outcomes/summary?kind=decision&group_by=action").status_code == 200


def test_chat_records_a_decision_in_the_background_store(
    client: TestClient, isolated_outcome_store: OutcomeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, response = chat(assessment("wait"))

    async def respond(_: ChatRequest) -> ChatResponse:
        return response

    monkeypatch.setattr(upscale.main.orchestrator, "respond", respond)
    reply = client.post("/chat", json=request.model_dump(mode="json"))
    assert reply.status_code == 200
    [d] = run(isolated_outcome_store.decisions())
    assert d.action == "wait" and d.source == "scout" and d.asset_id == f"solana:{MINTS['A']}"


def test_scout_view_shows_surfacing_history(tmp_path: Path, api_store: OutcomeStore) -> None:
    from upscale.scout_api import ScoutFilters, build_view

    result, _ = ranked(tmp_path / "again")
    history = run(api_store.surfacing([g.canonical_id for g in result.candidates]))
    view = build_view(result, 10, ScoutFilters(), NOW, history=history)
    rows = {r.label: r.value for s in view.candidates[0].details for r in s.rows}
    assert rows["Previously surfaced"] == "first time" and rows["First surfaced"] == "this run"
    later = later_run(result, 120)
    run(record_scout_run(api_store, later, CFG))
    history = run(api_store.surfacing([g.canonical_id for g in later.candidates]))
    view = build_view(later, 10, ScoutFilters(), NOW, history=history)
    rows = {r.label: r.value for s in view.candidates[0].details for r in s.rows}
    assert rows["Previously surfaced"] == "1 time" and rows["First surfaced"] == "2.0 h ago"


def test_app_collector_is_quiet_during_and_just_after_interactive_work(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upscale.main, "_last_activity", -1e9)
    assert upscale.main._busy() is False
    request, response = chat(assessment("wait"))

    async def respond(_: ChatRequest) -> ChatResponse:
        assert upscale.main._busy() is True  # a user is waiting
        return response

    monkeypatch.setattr(upscale.main.orchestrator, "respond", respond)
    client.post("/chat", json=request.model_dump(mode="json"))
    assert upscale.main._busy() is True  # just finished: still quiet
    monkeypatch.setattr(upscale.main, "_last_activity", -1e9)
    assert upscale.main._busy() is False


def test_decisions_are_not_starved_by_staggered_scout_work(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    request, response = chat(assessment("wait"), pool="pool-decision")
    d = run(record_decision(store, request, response, NOW + 30 * timedelta(seconds=1), CFG))
    assert d is not None
    clock = Clock(NOW + 7 * MINUTE)
    busy = [True]
    candles = FakeCandles(headroom=1)
    c = collector(store, clock, scout, candles, FakePools(), busy=lambda: busy[0])
    run(c.collect())  # every due row attempted (network skipped), attempts staggered below
    clock.advance(seconds=60)  # within retry spacing of that attempt
    busy[0] = False
    candles._headroom = 1
    run(c.collect())
    assert candles.calls and candles.calls[0][0] == "pool-decision"  # highest priority first


def test_unlisted_pool_that_still_trades_is_not_called_gone(tmp_path: Path) -> None:
    store, obs, scout = anchor(tmp_path)
    start = obs.observed_at
    p0 = obs.market.price_usd or 0
    candles = FakeCandles()
    candles.by_pool[obs.pool_address] = minute_candles(start, [(p0, p0 * 1.01, p0, p0)] * 5)
    pools = FakePools()  # the source that priced the pool no longer lists it...
    pools.by_token[obs.canonical_id] = [dex_pool("A", "a-different-pool")]
    clock = Clock(start + 7 * MINUTE)
    c = collector(store, clock, scout, candles, pools)
    run(c.collect())
    h = horizon(store, obs.id or 0, "5m")  # ...but it traded up to the horizon end
    assert h.market_status == "ACTIVE" and h.status == "PENDING"  # a snapshot may still come
    clock.at = start + 9 * MINUTE
    run(c.collect())
    h = horizon(store, obs.id or 0, "5m")
    assert h.market_status == "ACTIVE" and h.status == "PARTIAL"
    assert any("not treated as gone" in m for m in h.missing)
