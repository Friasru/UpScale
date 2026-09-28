"""One shared GeckoTerminal quota for Scout and Analyze (the live Scout -> Analyze failure).

Live: a Scout scan used GeckoTerminal's real ~6 requests / minute; the user pressed Analyze;
the candle request got HTTP 429 (Scout and Analyze each counted against their own limit)
and technical analysis failed. Here a fake GeckoTerminal *server* enforces that real limit
across every request it gets, from Scout's gate and Analyze's candle service alike.
"""

import asyncio
from pathlib import Path
from typing import Any

import httpx2
import pytest

import upscale.services
from tests.test_onchain_safety import POOL
from tests.test_scout_continuity import Ticker
from tests.test_scout_feed_scheduling import CHAINS, Listings
from tests.test_scout_growth import Clock, run
from tests.test_solana_dex import MINT, pair
from tests.test_trading_pipeline import gt_rows
from upscale.orchestrator import Orchestrator
from upscale.schemas import AssetRef, ChatMessage, ChatRequest
from upscale.services.capabilities import ProviderRegistry
from upscale.services.geckoterminal import DexCandleService, GeckoTerminalProvider
from upscale.services.market_data import CandleSeries, ProviderRateLimitedError
from upscale.services.quota import INTERACTIVE_LANE, LaneLimiter
from upscale.services.scout.config import ScoutConfig, ScoutProviderLimits
from upscale.services.scout.gate import RateLimitReachedError, RequestGate, request_lane
from upscale.services.scout.service import ScoutService
from upscale.services.scout.store import ScoutSnapshotStore

from .conftest import FakeDexScreener, FakeGeckoTerminal

LIMITS = ScoutProviderLimits(calls_per_minute=6, reservations={"refresh": 2, "interactive": 2},
                             cache_ttl_seconds=0)  # fmt: skip


class RealLimit:
    """GeckoTerminal's own limit: more than 6 requests in the window gets HTTP 429."""

    def __init__(self, fake: FakeGeckoTerminal, limit: int = 6):
        self.fake, self.limit, self.served, self.refused = fake, limit, 0, 0
        inner = fake.handler

        def handler(request: httpx2.Request) -> httpx2.Response:
            return self.hit() or inner(request)

        fake.handler = handler

    def hit(self) -> httpx2.Response | None:
        if self.served >= self.limit:
            self.refused += 1
            return httpx2.Response(429, json={"status": "429"})
        self.served += 1
        return None


def app_scout_gate() -> RequestGate:
    gate = next(
        p.gate for p in upscale.services.scout_service.providers if p.name == "GeckoTerminal"
    )
    assert isinstance(gate, RequestGate)
    return gate


def scout_scan_traffic(gate: RequestGate, server: RealLimit, attempts: int = 12) -> int:
    """A Scout scan: discovery listings (default lane), then tracked-token refresh (refresh
    lane), then retries on what refresh left. Returns the requests that were sent."""

    async def call(key: str) -> None:
        async def fetch() -> None:
            if server.hit() is not None:
                raise ProviderRateLimitedError("GeckoTerminal rate limit reached")

        await gate.run(key, fetch)

    async def go() -> int:
        sent = 0
        gate.arm("refresh")
        for i in range(attempts):
            try:
                await call(f"listing-{i}")
                sent += 1
            except RateLimitReachedError:
                pass
        with request_lane("refresh"):
            for i in range(attempts):
                try:
                    await call(f"refresh-{i}")
                    sent += 1
                except RateLimitReachedError:
                    pass
        gate.release("refresh")
        for i in range(attempts):
            try:
                await call(f"retry-{i}")
                sent += 1
            except RateLimitReachedError:
                pass
        return sent

    return run(go())


def analyze_mint(fake_dexscreener: FakeDexScreener) -> Any:
    fake_dexscreener.pairs[MINT] = [pair(symbol="NEWT", liquidity=5e6)]
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=f"Analyze NEWT: {MINT} on Solana 5m")],
        asset=AssetRef(chain="solana", address=MINT, symbol="NEWT", pool_address=POOL),
    )
    response = asyncio.run(Orchestrator().respond(request))
    return {r.agent: r for r in response.analysis.agent_results}


# --- F. One quota, really shared ------------------------------------------------------------


def test_scout_and_analyze_share_one_geckoterminal_quota() -> None:
    shared = upscale.services.geckoterminal_quota
    assert upscale.services.dex_candle_service.limiter is shared
    assert app_scout_gate()._limiter is shared
    assert shared.max_calls == 6
    assert shared.reservations == {"refresh": 2, "interactive": 2}
    assert upscale.services.dex_candle_service.lane == INTERACTIVE_LANE


# --- A. The live failure: Scout first, then Analyze -----------------------------------------


def test_analyze_right_after_a_scout_scan_still_gets_candles(
    fake_dexscreener: FakeDexScreener, fake_geckoterminal: FakeGeckoTerminal
) -> None:
    server = RealLimit(fake_geckoterminal)
    sent = scout_scan_traffic(app_scout_gate(), server)
    assert sent == 4  # Scout used everything it may: 2 discovery + 2 refresh
    assert server.refused == 0  # and never went over the real limit
    fake_geckoterminal.candles[POOL] = gt_rows(120)
    by_agent = analyze_mint(fake_dexscreener)
    technical = by_agent["technical_analysis"]
    assert technical.status == "ok", technical.error
    assert technical.findings["provider"] == "GeckoTerminal"
    assert server.refused == 0  # no 429: the interactive capacity was still there
    assert by_agent["opportunity"].findings["asset_profile"]["canonical_id"] == f"solana:{MINT}"


# --- B. Several Analyze requests never exceed the hard limit -----------------------------------


def test_repeated_analyze_never_exceeds_the_real_limit(
    fake_geckoterminal: FakeGeckoTerminal,
) -> None:
    server = RealLimit(fake_geckoterminal)
    scout_scan_traffic(app_scout_gate(), server)
    candles = upscale.services.dex_candle_service
    pools = [f"{POOL[:-2]}{i:02d}" for i in range(10)]
    for pool in pools:
        fake_geckoterminal.candles[pool] = gt_rows(120)
    outcomes = []
    for pool in pools:  # different pools: nothing comes from the cache
        try:
            run(candles.get_candles("solana", pool, "5m", 100, symbol="X", canonical_id=None))
            outcomes.append("ok")
        except ProviderRateLimitedError:
            outcomes.append("limited")
    assert server.served == 6 and server.refused == 0  # never above the real limit
    assert outcomes[:2] == ["ok", "ok"] and set(outcomes[2:]) == {"limited"}


# --- C / G. Scout keeps its share, feed scheduling and refresh reservations hold --------------


def test_scout_keeps_its_capacity_while_analyze_is_idle(tmp_path: Path) -> None:
    ticker = Ticker()
    shared = LaneLimiter(6, 60.0, {"refresh": 2, "interactive": 2}, clock=ticker)
    provider = Listings(ticker)
    provider.gate = RequestGate("GeckoTerminal", LIMITS, clock=ticker, limiter=shared)
    config = ScoutConfig.model_validate({"chains": list(CHAINS)})
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    scout = ScoutService([provider], store, config, enrichment_provider=None, now=Clock())
    executed = []
    for _ in range(6):
        scout.hold_reservations()
        [report] = run(scout.discover()).feeds
        executed.append(report.executed)
        assert shared.available("refresh") == 2  # the refresh reservation is still held
        assert shared.available("interactive") >= 2  # and so is Analyze's
        scout.release_reservations()
        retries = 0
        while shared.try_acquire("default"):  # discovery retries borrow what refresh left,
            retries += 1
        assert retries == 2
        assert shared.available("interactive") == 2  # never Analyze's 2
        ticker.t += 600  # the next scan, ten minutes later: Scout resumes normally
    assert all(len(e) == 2 for e in executed)  # discovery's own share every run
    ran = {f for e in executed for f in e}
    assert any(f.startswith("active") for f in ran) and any(f.startswith("trending") for f in ran)
    assert sum(f.startswith("new") for e in executed for f in e) >= 6  # new pools still favored


def test_interactive_priority_does_not_starve_scout() -> None:
    ticker = Ticker()
    shared = LaneLimiter(6, 60.0, {"refresh": 2, "interactive": 2}, clock=ticker)
    for _ in range(6):  # a burst of Analyze: it may use free capacity too...
        shared.try_acquire(INTERACTIVE_LANE)
    assert shared.available("default") == 0  # ...so Scout defers this minute
    ticker.t += 61
    assert shared.available("default") == 2  # and has its share back the next


# --- D. An honest reason when every exact candle source is rate-limited ----------------------


def test_rate_limited_candles_give_a_provider_reason_not_missing_history(
    fake_dexscreener: FakeDexScreener, fake_geckoterminal: FakeGeckoTerminal
) -> None:
    fake_geckoterminal.candles[POOL] = gt_rows(120)
    shared = upscale.services.geckoterminal_quota
    while shared.try_acquire(INTERACTIVE_LANE):
        pass  # Analyze's capacity is used up this minute
    by_agent = analyze_mint(fake_dexscreener)
    technical = by_agent["technical_analysis"]
    assert technical.status == "error"
    assert technical.error == (
        "technical market data temporarily rate-limited; try again in about a minute"
    )
    assert technical.findings["unavailable"]["kind"] == "provider_rate_limited"
    assert technical.findings["unavailable"]["provider"] == "GeckoTerminal"
    assert fake_geckoterminal.requests == []  # nothing sent over the limit
    decision = by_agent["opportunity"].findings
    assert decision["action"] == "wait"  # the pipeline still decides WAIT...
    reasons = " ".join(f["reason"] for f in decision["blocking_factors"])
    assert "temporarily rate-limited" in reasons and "not missing market history" in reasons
    assert "no price candles for this exact mint" not in reasons  # ...for the honest reason


def test_a_token_without_candles_still_says_so(
    fake_dexscreener: FakeDexScreener, fake_geckoterminal: FakeGeckoTerminal
) -> None:
    by_agent = analyze_mint(fake_dexscreener)  # GeckoTerminal knows no candles for the pool
    assert "unavailable" not in by_agent["technical_analysis"].findings
    reasons = " ".join(f["reason"] for f in by_agent["opportunity"].findings["blocking_factors"])
    assert "rate-limited" not in reasons


# --- E. Provider fallback, never identity fallback --------------------------------------------


class ExactSource:
    """Another provider of candles for the same exact pool (records what it was asked)."""

    provider_name = "OtherExact"
    supported_timeframes = frozenset({"5m"})

    def __init__(self) -> None:
        self.asked: list[tuple[str, str, str | None]] = []

    def covers(self, chain: str) -> bool:
        return chain == "solana"

    async def get_candles(self, chain: str, pool: str, timeframe: Any, limit: int, *,
                          symbol: str, canonical_id: str | None) -> CandleSeries:  # fmt: skip
        self.asked.append((chain, pool, canonical_id))
        raise ProviderRateLimitedError("OtherExact rate limit reached")


def registry_with(fallback: ExactSource, limited: bool = True) -> tuple[ProviderRegistry, Any]:
    gt = GeckoTerminalProvider(transport=httpx2.MockTransport(
        lambda r: httpx2.Response(429 if limited else 404, json={})))  # fmt: skip
    candles = DexCandleService(gt, limiter=LaneLimiter(6, 60.0, {"interactive": 2}))
    base = upscale.services.provider_registry
    registry = ProviderRegistry(market_data=base.market_data, dex=base.dex, dex_candles=candles,
                                pool_candle_fallbacks=[fallback])  # fmt: skip
    return registry, candles


def test_fallback_is_another_provider_of_the_same_exact_pool() -> None:
    other = ExactSource()
    registry, _ = registry_with(other)
    with pytest.raises(ProviderRateLimitedError) as err:
        run(registry.pool_candles("solana", POOL, "5m", 100, symbol="PEPE",
                                  canonical_id=f"solana:{MINT}"))  # fmt: skip
    assert other.asked == [("solana", POOL, f"solana:{MINT}")]  # same pool, same token
    assert "GeckoTerminal rate limit reached" in str(err.value)
    assert "OtherExact rate limit reached" in str(err.value)


def test_a_real_candle_problem_is_not_reported_as_a_rate_limit() -> None:
    other = ExactSource()
    registry, _ = registry_with(other, limited=False)  # GeckoTerminal: unknown pool (404)
    with pytest.raises(Exception) as err:
        run(registry.pool_candles("solana", POOL, "5m", 100, symbol="PEPE", canonical_id=None))
    assert not isinstance(err.value, ProviderRateLimitedError)
    assert "doesn't know this pool" in str(err.value)
