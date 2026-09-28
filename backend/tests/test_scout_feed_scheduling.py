"""Fair discovery feed scheduling: when a provider's discovery capacity can't run every
feed (listing kind x chain) in one scan, feeds take turns by weight, the rotation is
persisted, and no feed starves. Offline: the real ScoutService, RequestGate and store,
over a fake provider shaped like GeckoTerminal (6 requests a minute, 2 held for refresh).
"""

from collections import Counter
from datetime import timedelta
from pathlib import Path

from tests.test_scout_continuity import Ticker
from tests.test_scout_growth import Clock, run
from upscale.services.scout.config import ScoutConfig, ScoutProviderLimits
from upscale.services.scout.gate import RequestGate
from upscale.services.scout.growth import GrowthScoutService
from upscale.services.scout.normalize import DiscoveryResult
from upscale.services.scout.service import ScoutService
from upscale.services.scout.store import ScoutSnapshotStore

CHAINS = ("solana", "ethereum", "base", "bsc")


class Listings:
    """Every listing kind on four chains; each request through the real gate."""

    name = "GeckoTerminal"
    kinds = frozenset({"new", "active", "trending", "lookup"})
    chains = frozenset(CHAINS)

    def __init__(self, ticker: Ticker):
        self.gate = RequestGate("GeckoTerminal", ScoutProviderLimits(
            calls_per_minute=6, reservations={"refresh": 2}, cache_ttl_seconds=0), clock=ticker)  # fmt: skip
        self.calls: list[str] = []

    async def _listing(self, kind: str, chain: str) -> DiscoveryResult:
        async def fetch() -> DiscoveryResult:
            self.calls.append(f"{kind}:{chain}")
            return DiscoveryResult()

        return await self.gate.run(f"{kind}:{chain}:{len(self.calls)}", fetch)

    async def discover_new_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        return await self._listing("new", chain)

    async def discover_active_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        return await self._listing("active", chain)

    async def discover_trending_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        return await self._listing("trending", chain)

    async def lookup_exact_tokens(self, chain: str, addresses: list[str]) -> DiscoveryResult:
        return DiscoveryResult()

    async def lookup_exact_token(self, chain: str, address: str) -> DiscoveryResult:
        return DiscoveryResult()


def scout(tmp_path: Path, provider: Listings, clock: Clock, **cfg: object) -> ScoutService:
    config = ScoutConfig.model_validate({"chains": list(CHAINS), **cfg})
    return ScoutService([provider], ScoutSnapshotStore(tmp_path / "scout.sqlite3"), config,
                        enrichment_provider=None, now=clock)  # fmt: skip


def runs(tmp_path: Path, n: int, **cfg: object) -> tuple[list[list[str]], Listings, ScoutService]:
    """`n` discovery passes 10 minutes apart, the refresh reservation held (as in a scan)."""
    clock, ticker = Clock(), Ticker()
    provider = Listings(ticker)
    s = scout(tmp_path, provider, clock, **cfg)
    executed = []
    for _ in range(n):
        provider.gate.arm("refresh")
        result = run(s.discover())
        [report] = result.feeds
        assert report.executed == provider.calls[-len(report.executed) :] or not report.executed
        executed.append(report.executed)
        clock.at += timedelta(minutes=10)
        ticker.t += 600
    return executed, provider, s


def test_a_small_budget_never_starves_a_feed(tmp_path: Path) -> None:
    executed, provider, _ = runs(tmp_path, 8)
    assert all(len(e) == 4 for e in executed)  # 6 a minute, 2 held for refresh
    counts = Counter(f for e in executed for f in e)
    assert set(counts) == {f"{k}:{c}" for k in ("new", "active", "trending") for c in CHAINS}


def test_new_pools_are_covered_often_and_active_trending_still_come_round(
    tmp_path: Path,
) -> None:
    executed, _, _ = runs(tmp_path, 8)
    assert all(e[0].startswith("new:") for e in executed[:1])  # new pools first at the start
    counts = Counter(f.split(":")[0] for e in executed for f in e)
    assert counts["new"] == 2 * counts["active"] == 2 * counts["trending"]  # weight 2 : 1 : 1
    for chain in CHAINS:  # every new-pool feed at least every other run
        ran = [i for i, e in enumerate(executed) if f"new:{chain}" in e]
        assert max(b - a for a, b in zip(ran, ran[1:], strict=False)) <= 2
    first_active = min(i for i, e in enumerate(executed) if any(f.startswith("active") for f in e))
    first_trending = min(
        i for i, e in enumerate(executed) if any(f.startswith("trending") for f in e)
    )
    assert first_active <= 2 and first_trending <= 3


def test_rotation_persists_across_service_instances(tmp_path: Path) -> None:
    executed, _, _ = runs(tmp_path, 2)
    clock, ticker = Clock(), Ticker()
    provider = Listings(ticker)
    fresh = scout(tmp_path, provider, clock)  # a restart: same store, new service
    provider.gate.arm("refresh")
    [report] = run(fresh.discover()).feeds
    assert set(report.executed).isdisjoint(executed[0]) or report.executed != executed[0]
    straight, _, _ = runs(tmp_path / "straight", 3)
    assert report.executed == straight[2]  # exactly where the rotation left off


def test_input_order_does_not_change_the_schedule(tmp_path: Path) -> None:
    forward, _, _ = runs(tmp_path / "a", 6)
    backward, _, _ = runs(tmp_path / "b", 6, chains=list(reversed(CHAINS)),
                          kinds=["trending", "active", "new"])  # fmt: skip
    assert [sorted(e) for e in forward] == [sorted(e) for e in backward]


def test_hard_limit_and_refresh_reservation_are_untouched(tmp_path: Path) -> None:
    clock, ticker = Clock(), Ticker()
    provider = Listings(ticker)
    s = scout(tmp_path, provider, clock)
    provider.gate.arm("refresh")
    [report] = run(s.discover()).feeds
    assert report.requests == 4 and len(report.deferred) == 8 and report.failed == []
    assert provider.gate.available("refresh") == 2  # discovery never touched it
    assert provider.gate.available() == 0
    assert report.next_scheduled and len(report.next_scheduled) == 4
    assert provider.gate.requests_made == 4 <= 6


def test_unused_refresh_capacity_runs_deferred_feeds_after_the_refresh(tmp_path: Path) -> None:
    """In a scan: discovery runs 4 feeds, the refresh has nothing to do, its unused
    reservation is released, and the deferred feeds next in line use it."""
    clock, ticker = Clock(), Ticker()
    provider = Listings(ticker)
    s = scout(tmp_path, provider, clock)
    growth = GrowthScoutService(s.store, now=clock)
    result = run(growth.scan(s))
    [report] = result.universe.feeds
    assert report.requests == 6 and len(report.executed) == 6  # 4 + the 2 left by refresh
    assert len(report.deferred) == 6 and report.failed == []
    assert provider.gate.requests_made == 6  # never above the hard limit
    assert result.universe.discovery_retried == 2  # what was really sent, not offered
