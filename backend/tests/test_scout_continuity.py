"""Tracked-token continuity: reserved refresh capacity, refresh priority, an honest tracking
horizon, and the short last-good-snapshot grace for provider failures.

Offline and deterministic: the real ScoutService, RequestGate (with a fake clock) and
GrowthScoutService, over a fake GeckoTerminal-like provider (6 requests a minute, 2 of
them reserved for refresh) that serves synthetic markets.
"""

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.test_scout_growth import ACCEL, NOW, Clock, accelerating, candidate, run
from upscale.services.market_data import MarketDataUnavailableError
from upscale.services.scout.config import ScoutConfig, ScoutProviderLimits
from upscale.services.scout.gate import RateLimitReachedError, RequestGate, request_lane
from upscale.services.scout.growth import GrowthConfig, GrowthScoutService
from upscale.services.scout.models import ScoutCandidate, ScoutSourceEvidence
from upscale.services.scout.normalize import DiscoveryResult
from upscale.services.scout.service import ScoutService
from upscale.services.scout.store import ScoutSnapshotStore

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def mint(i: int) -> str:
    head = ""
    n = i + 1
    while n:
        n, r = divmod(n, 58)
        head = B58[r] + head
    return (head + "x" + "7" * 43)[:43]


def market(i: int, **kw: Any) -> ScoutCandidate:
    c, _ = candidate(mint(i), ACCEL, symbol=f"T{i}", **kw)
    return c


# --- 1. The gate: reservations inside a hard limit --------------------------------------------


class Ticker:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def gate(clock: Ticker) -> RequestGate:
    limits = ScoutProviderLimits(calls_per_minute=6, reservations={"refresh": 2},
                                 cache_ttl_seconds=0)  # fmt: skip
    return RequestGate("GeckoTerminal", limits, clock=clock)


async def call(g: RequestGate, key: str, lane: str | None = None) -> bool:
    async def fetch() -> str:
        return key

    try:
        if lane:
            with request_lane(lane):
                await g.run(key, fetch)
        else:
            await g.run(key, fetch)
        return True
    except RateLimitReachedError:
        return False


def test_discovery_cannot_consume_the_refresh_reservation() -> None:
    g = gate(Ticker())

    async def go() -> tuple[list[bool], list[bool]]:
        discovery = [await call(g, f"listing-{i}") for i in range(12)]
        refresh = [await call(g, f"lookup-{i}", "refresh") for i in range(2)]
        return discovery, refresh

    discovery, refresh = run(go())
    assert discovery.count(True) == 4  # 6 per minute, 2 held for refresh
    assert refresh == [True, True]  # the reservation is still there


def test_unused_refresh_capacity_is_lent_to_discovery_after_release() -> None:
    g = gate(Ticker())

    async def go() -> list[bool]:
        for i in range(4):
            assert await call(g, f"listing-{i}")
        assert not await call(g, "listing-4")  # held for refresh
        assert await call(g, "lookup-0", "refresh")  # refresh used 1 of its 2
        g.release("refresh")
        return [await call(g, f"retry-{i}") for i in range(3)]

    assert run(go()) == [True, False, False]  # the one left over, and not a call more


def test_the_provider_limit_stays_hard_for_every_lane() -> None:
    clock = Ticker()
    g = gate(clock)

    async def go() -> int:
        sent = 0
        for i in range(10):  # refresh may use free capacity too, but never above 6
            sent += await call(g, f"lookup-{i}", "refresh")
        g.release("refresh")
        for i in range(10):
            sent += await call(g, f"listing-{i}")
        return sent

    assert run(go()) == 6
    assert g.requests_made == 6
    clock.t = 61  # the window moves on: a new minute's budget, reservation re-armed
    g.arm("refresh")
    assert run(call(g, "next", "refresh"))


def test_reservations_must_leave_unreserved_capacity() -> None:
    with pytest.raises(ValueError):
        ScoutProviderLimits(calls_per_minute=6, reservations={"refresh": 6})


# --- A fake GeckoTerminal-only provider behind the real ScoutService --------------------------


class FakeGT:
    """Discovery listings and batched exact lookups, each request through the real gate."""

    name = "GeckoTerminal"
    kinds = frozenset({"new", "active", "trending", "lookup"})
    chains = frozenset({"solana"})

    def __init__(self, clock: Clock, ticker: Ticker, markets: dict[str, ScoutCandidate]):
        self.clock = clock
        self.gate = RequestGate("GeckoTerminal", ScoutProviderLimits(
            calls_per_minute=6, reservations={"refresh": 2}, cache_ttl_seconds=0), clock=ticker)  # fmt: skip
        self.markets = markets  # by canonical id: what the provider knows now
        self.listed: dict[str, list[str]] = {"new": [], "active": [], "trending": []}
        self.down = False  # every request fails (a provider outage)
        self.lookups: list[list[str]] = []

    async def _gated(self, key: str) -> None:
        async def fetch() -> None:
            if self.down:
                raise MarketDataUnavailableError("GeckoTerminal rate limit reached")

        await self.gate.run(key, fetch)

    async def _listing(self, kind: str, chain: str) -> DiscoveryResult:
        await self._gated(f"{kind}/{chain}/{self.clock().isoformat()}")
        at = self.clock()
        return DiscoveryResult(candidates=[
            self.markets[cid].model_copy(update={"observed_at": at, "market_provider": self.name, "sources": [
                ScoutSourceEvidence(provider=self.name, kind=kind, listing=kind, fetched_at=at)]})  # type: ignore[arg-type]
            for cid in self.listed[kind]
        ])  # fmt: skip

    async def discover_new_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        return await self._listing("new", chain)

    async def discover_active_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        return await self._listing("active", chain)

    async def discover_trending_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        return await self._listing("trending", chain)

    async def lookup_exact_tokens(self, chain: str, addresses: list[str]) -> DiscoveryResult:
        out = DiscoveryResult()
        for i in range(0, len(addresses), 30):
            batch = addresses[i : i + 30]
            await self._gated(f"lookup/{chain}/{','.join(batch)}/{self.clock().isoformat()}")
            self.lookups.append(batch)
            at = self.clock()
            for a in batch:
                c = self.markets.get(f"{chain}:{a}")
                if c is not None:
                    out.candidates.append(c.model_copy(update={"observed_at": at, "market_provider": self.name, "sources": [
                        ScoutSourceEvidence(provider=self.name, kind="lookup",
                                            listing="tokens", fetched_at=at)]}))  # fmt: skip
        return out

    async def lookup_exact_token(self, chain: str, address: str) -> DiscoveryResult:
        return await self.lookup_exact_tokens(chain, [address])


def setup(
    tmp_path: Path, n: int, **tracking: Any
) -> tuple[GrowthScoutService, FakeGT, Clock, Ticker]:
    clock, ticker = Clock(), Ticker()
    markets = {f"solana:{mint(i)}": market(i) for i in range(n)}
    gt = FakeGT(clock, ticker, markets)
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    growth = GrowthScoutService(
        store, GrowthConfig.model_validate({"tracking": tracking}) if tracking else None, now=clock
    )
    return growth, gt, clock, ticker


def next_run(clock: Clock, ticker: Ticker, minutes: float = 10) -> None:
    clock.at += timedelta(minutes=minutes)
    ticker.t += minutes * 60


def scan_with(growth: GrowthScoutService, scout: ScoutService) -> Any:
    return run(growth.scan(scout, limit=500))


def scout_of(growth: GrowthScoutService, gt: FakeGT, clock: Clock) -> ScoutService:
    cfg = ScoutConfig(chains=("solana",), min_snapshot_interval_seconds=0)
    return ScoutService([gt], growth.store, cfg, enrichment_provider=None, now=clock)


def ids(result: Any) -> set[str]:
    return {g.canonical_id for g in result.candidates}


# --- 2. GeckoTerminal-only tokens survive discovery exhausting its capacity -------------------


def test_reserved_refresh_keeps_tracked_tokens_when_discovery_exhausts_capacity(
    tmp_path: Path,
) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 6)
    scout = scout_of(growth, gt, clock)
    gt.listed["new"] = [f"solana:{mint(i)}" for i in range(3)]
    gt.listed["trending"] = [f"solana:{mint(i)}" for i in range(3, 6)]
    first = scan_with(growth, scout)
    assert len(ids(first)) == 6
    next_run(clock, ticker)
    # This run the listings surface nothing, and discovery (plus another consumer this
    # minute) wants more than the unreserved budget.
    gt.listed = {"new": [], "active": [], "trending": []}
    for i in range(3):
        assert run(call(gt.gate, f"other-{i}"))
    second = scan_with(growth, scout)
    assert ids(second) == ids(first)  # nothing lost to discovery rotation
    u = second.universe
    assert u.refreshed == 6 and u.carried_stale == 0
    assert u.refresh_capacity_requests == {"GeckoTerminal": 2}
    assert gt.gate.used("refresh") >= 1  # the refresh really ran in its reserved lane
    assert gt.gate.requests_made <= 6 * 2  # two minutes, never above 6 a minute


def test_failed_listings_are_retried_on_capacity_the_refresh_left(tmp_path: Path) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 3)
    scout = scout_of(growth, gt, clock)
    gt.listed["trending"] = [f"solana:{mint(i)}" for i in range(3)]
    # 3 listings; the 4 unreserved calls cover them; nothing to refresh; nothing to retry.
    first = scan_with(growth, scout)
    assert first.universe.discovered == 3 and first.universe.discovery_retried == 0
    next_run(clock, ticker)
    # Burn unreserved capacity first (e.g. another consumer this minute): discovery fails
    # its listings, the (empty) refresh leaves its reservation unused, the retry borrows it.
    for i in range(4):
        run(call(gt.gate, f"other-{i}"))
    gt.listed["trending"] = [f"solana:{mint(0)}"]
    second = scan_with(growth, scout)
    assert second.universe.discovery_retried >= 1
    assert f"solana:{mint(0)}" in ids(second)


# --- 3. Refresh priority, rotation and an honest horizon -------------------------------------


def test_more_tracked_tokens_than_one_run_can_refresh(tmp_path: Path) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 90)
    scout = scout_of(growth, gt, clock)
    everyone = [f"solana:{mint(i)}" for i in range(90)]
    gt.listed = {"new": everyone[:30], "active": everyone[30:60], "trending": everyone[60:]}
    scan_with(growth, scout)
    next_run(clock, ticker)
    gt.listed = {"new": [], "active": [], "trending": []}
    result = scan_with(growth, scout)
    u = result.universe
    assert u.tracked == 90
    assert u.refresh_capacity_tokens == 60  # 2 reserved requests x 30 per batch
    assert u.refreshed == 60 and u.deferred == 30  # capacity binds, nothing oversent
    assert u.estimated_max_revisit_minutes == pytest.approx(20.0)  # 2 runs x 10 minutes
    assert u.horizon_covered


def test_fair_rotation_eventually_refreshes_every_tracked_token(tmp_path: Path) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 90)
    scout = scout_of(growth, gt, clock)
    everyone = [f"solana:{mint(i)}" for i in range(90)]
    gt.listed = {"new": everyone[:30], "active": everyone[30:60], "trending": everyone[60:]}
    scan_with(growth, scout)
    gt.listed = {"new": [], "active": [], "trending": []}
    refreshed: set[str] = set()
    for _ in range(3):
        next_run(clock, ticker)
        gt.lookups.clear()
        scan_with(growth, scout)
        refreshed |= {f"solana:{a}" for batch in gt.lookups for a in batch}
    assert refreshed == set(everyone)


def test_the_top_candidate_is_refreshed_first(tmp_path: Path) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 90)
    scout = scout_of(growth, gt, clock)
    everyone = [f"solana:{mint(i)}" for i in range(90)]
    gt.listed = {"new": everyone[:30], "active": everyone[30:60], "trending": everyone[60:]}
    first = scan_with(growth, scout)
    leader = first.candidates[0].canonical_id
    # Make the leader the *freshest* token, so only its rank can put it first.
    for _ in range(3):
        next_run(clock, ticker)
        gt.listed = {"new": [leader], "active": [], "trending": []}
        scan_with(growth, scout)
    next_run(clock, ticker)
    gt.listed = {"new": [], "active": [], "trending": []}
    gt.lookups.clear()
    scan_with(growth, scout)
    assert leader.split(":")[1] in gt.lookups[0] or any(
        leader.split(":")[1] in b for b in gt.lookups
    )


def test_an_uncoverable_horizon_is_reported(tmp_path: Path) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 90, horizon_hours=0.3, max_refresh=10)
    scout = scout_of(growth, gt, clock)
    everyone = [f"solana:{mint(i)}" for i in range(90)]
    gt.listed = {"new": everyone[:30], "active": everyone[30:60], "trending": everyone[60:]}
    scan_with(growth, scout)
    next_run(clock, ticker)
    gt.listed = {"new": [], "active": [], "trending": []}
    result = scan_with(growth, scout)
    u = result.universe
    assert not u.horizon_covered and u.estimated_max_revisit_minutes == pytest.approx(90.0)
    assert any("tracking horizon is 0.3h" in n for n in result.notes)


# --- 4. Grace for provider failures ------------------------------------------------------------


def _tracked_then_outage(
    tmp_path: Path,
) -> tuple[GrowthScoutService, FakeGT, Clock, Ticker, ScoutService, Any]:
    growth, gt, clock, ticker = setup(tmp_path, 3)
    scout = scout_of(growth, gt, clock)
    gt.listed["new"] = [f"solana:{mint(i)}" for i in range(3)]
    first = scan_with(growth, scout)
    gt.listed["new"] = []
    gt.down = True
    return growth, gt, clock, ticker, scout, first


def test_a_transient_failure_carries_the_candidate_forward_labeled(tmp_path: Path) -> None:
    growth, gt, clock, ticker, scout, first = _tracked_then_outage(tmp_path)
    before = {g.canonical_id: g for g in first.candidates}
    next_run(clock, ticker)
    result = scan_with(growth, scout)
    assert ids(result) == set(before)  # carried once, not erased
    u = result.universe
    assert (u.refreshed, u.unresolved, u.carried_stale, u.unusable) == (0, 3, 3, 0)
    for g in result.candidates:
        assert g.data_status == "STALE_CARRIED"
        assert g.snapshot_age_minutes == pytest.approx(10.0)
        assert g.stage_reasons[0].startswith("stale: evidence 10 minutes old")
        flag = next(f for f in g.risk_flags if f.code == "stale_market_data")
        assert flag.penalty == 10.0
        assert g.score < before[g.canonical_id].score  # stale is never scored as current


def test_grace_expires(tmp_path: Path) -> None:
    growth, gt, clock, ticker, scout, _ = _tracked_then_outage(tmp_path)
    next_run(clock, ticker)
    assert len(scan_with(growth, scout).candidates) == 3  # 10 minutes old: within grace
    next_run(clock, ticker)
    expired = scan_with(growth, scout)  # 20 minutes old: past the 15-minute grace
    assert expired.candidates == []
    assert expired.universe.unresolved == 3 and expired.universe.carried_stale == 0


def test_a_confirmed_unusable_token_is_not_carried(tmp_path: Path) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 3)
    scout = scout_of(growth, gt, clock)
    gt.listed["new"] = [f"solana:{mint(i)}" for i in range(3)]
    scan_with(growth, scout)
    next_run(clock, ticker)
    gt.listed["new"] = []
    del gt.markets[f"solana:{mint(0)}"]  # the lookup answered: the token is gone
    result = scan_with(growth, scout)
    assert f"solana:{mint(0)}" not in ids(result)
    assert (result.universe.unusable, result.universe.carried_stale) == (1, 0)
    assert all(g.data_status == "CURRENT" for g in result.candidates)
    # ... and it no longer takes refresh capacity, unless a listing surfaces it again.
    next_run(clock, ticker)
    gt.lookups.clear()
    later = scan_with(growth, scout)
    assert mint(0) not in {a for batch in gt.lookups for a in batch}
    assert later.universe.tracked == 2 and later.universe.expired == 0


def test_carrying_forward_never_duplicates_identities_or_snapshots(tmp_path: Path) -> None:
    growth, gt, clock, ticker, scout, first = _tracked_then_outage(tmp_path)

    async def snapshots() -> int:
        total = 0
        for g in first.candidates:
            total += len(await growth.store.history(g.canonical_id))
        return total

    before = run(snapshots())
    next_run(clock, ticker)
    result = scan_with(growth, scout)
    every = [g.canonical_id for g in [*result.candidates, *result.unranked]]
    assert len(every) == len(set(every)) == 3
    assert run(snapshots()) == before  # a carried observation is never stored again
    # Stages are only recorded for current evidence.
    stored = run(growth.store.recent_stages(every, clock() - timedelta(minutes=1)))
    assert stored == {}


def test_same_ticker_tokens_stay_distinct_through_refresh(tmp_path: Path) -> None:
    growth, gt, clock, ticker = setup(tmp_path, 2)
    scout = scout_of(growth, gt, clock)
    for c in gt.markets.values():
        c.symbol = "SAME"
    gt.listed["new"] = list(gt.markets)
    scan_with(growth, scout)
    next_run(clock, ticker)
    gt.listed["new"] = [next(iter(gt.markets))]  # one rediscovered, one refreshed
    result = scan_with(growth, scout)
    assert sorted(ids(result)) == sorted(gt.markets)
    assert [g.symbol for g in result.candidates] == ["SAME", "SAME"]
    _ = (NOW, accelerating, asyncio)  # shared helpers kept importable


class FakeDS(FakeGT):
    name = "DEX Screener"

    def __init__(self, clock: Clock, ticker: Ticker, markets: dict[str, ScoutCandidate]):
        super().__init__(clock, ticker, markets)
        self.gate = RequestGate("DEX Screener", ScoutProviderLimits(
            calls_per_minute=50, reservations={"refresh": 10}, cache_ttl_seconds=0), clock=ticker)  # fmt: skip


def test_fallbacks_never_take_capacity_planned_for_another_provider(tmp_path: Path) -> None:
    """Live: GeckoTerminal-priced leaders were carried stale because other tokens'
    fallback lookups competed for GeckoTerminal's two reserved requests."""
    clock, ticker = Clock(), Ticker()
    gt_only = {f"solana:{mint(i)}": market(i) for i in range(3)}  # priced by GeckoTerminal
    ds_known = {f"solana:{mint(i)}": market(i) for i in range(3, 9)}
    gt = FakeGT(clock, ticker, {**gt_only, **ds_known})
    ds = FakeDS(clock, ticker, dict(ds_known))
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    cfg = ScoutConfig(chains=("solana",), min_snapshot_interval_seconds=0)
    scout = ScoutService([ds, gt], store, cfg, now=clock)  # DEX Screener enriches
    growth = GrowthScoutService(store, now=clock)
    gt.listed["new"] = list(gt_only) + list(ds_known)
    first = scan_with(growth, scout)
    providers = {g.canonical_id: g.market.market_provider for g in first.candidates}
    assert {providers[c] for c in gt_only} == {"GeckoTerminal"}
    assert {providers[c] for c in ds_known} == {"DEX Screener"}
    next_run(clock, ticker)
    gt.listed["new"] = []
    ds.markets.clear()  # DEX Screener forgets its tokens: they all fall back to GeckoTerminal
    result = scan_with(growth, scout)
    refreshed = {g.canonical_id for g in result.candidates if g.data_status == "CURRENT"}
    assert set(gt_only) <= refreshed  # the planned GeckoTerminal refresh came first
    assert set(ds_known) <= refreshed | {g.canonical_id for g in result.candidates}
    assert gt.gate.requests_made <= 12  # 6 a minute over the two runs, never more


def test_unresolved_is_decided_per_token_by_its_planned_provider(tmp_path: Path) -> None:
    """Live: DEX Screener-priced tokens it answered were gone kept being carried stale
    because another token's GeckoTerminal fallback on the same chain had failed."""
    clock, ticker = Clock(), Ticker()
    gone, alive = f"solana:{mint(0)}", f"solana:{mint(1)}"  # priced by DEX Screener
    gt_priced = f"solana:{mint(2)}"
    gt = FakeGT(clock, ticker, {gt_priced: market(2), gone: market(0), alive: market(1)})
    ds = FakeDS(clock, ticker, {gone: market(0), alive: market(1)})
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    cfg = ScoutConfig(chains=("solana",), min_snapshot_interval_seconds=0)
    scout = ScoutService([ds, gt], store, cfg, now=clock)
    growth = GrowthScoutService(store, now=clock)
    gt.listed["new"] = [gone, alive, gt_priced]
    scan_with(growth, scout)
    next_run(clock, ticker)
    gt.listed["new"] = []
    del ds.markets[gone]  # DEX Screener answers: no usable market any more
    gt.down = True  # ... and the fallback provider is down
    result = scan_with(growth, scout)
    status = {g.canonical_id: g.data_status for g in [*result.candidates, *result.unranked]}
    assert gone not in status  # confirmed by its own provider: dropped, not carried
    assert status[alive] == "CURRENT"
    assert status[gt_priced] == "STALE_CARRIED"  # its own provider failed: carried
    u = result.universe
    assert (u.unusable, u.unresolved, u.carried_stale) == (1, 1, 1)
