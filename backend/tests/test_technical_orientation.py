"""Technical analysis prices its pool candles for the exact token it analyzes: a provider
may orient a pool the other way round (GeckoTerminal "DOGE / GOAT" where the token is GOAT),
and its default "base" would then be the other token. Identity / orientation only: the
indicator math is unchanged."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest

from upscale.agents import TechnicalAnalysisAgent
from upscale.agents.base import AgentContext
from upscale.services.asset_profile import AssetIdentity
from upscale.services.geckoterminal import DexCandleService, GeckoTerminalProvider
from upscale.services.market_data import CandleSeries, MarketDataUnavailableError
from upscale.services.replay_lab.candles import PointInTimeCandles, PointInTimeCandleSource
from upscale.services.replay_lab.clock import HistoricalClock
from upscale.services.solana_dex import SolanaDexSnapshot
from upscale.services.technical_analysis import TechnicalAnalysisConfig, analyze_series

from .conftest import FakeGeckoTerminal
from .test_solana_dex import MINT
from .test_trading_pipeline import NOW, POOL, dex_prior, gt_rows, results

OTHER_TOKEN = "DoGEV7LASBkQbibMc5k5vKnTZoMg423GpJ5QtJEGfm7R"  # the pool's other token


def scaled(rows: list[list[float]], factor: float) -> list[list[float]]:
    return [[r[0], *(v * factor for v in r[1:5]), r[5]] for r in rows]


def oriented(
    fake: FakeGeckoTerminal, reversed_pool: bool, mint_available: bool = True
) -> list[list[float]]:
    """Serve the pool's candles by the requested `token`: the exact mint gets the token's
    own prices; "base" gets them too for a normal pool, or the OTHER token's for a reversed
    one."""
    own = scaled(gt_rows(120), 0.001)  # the token trades around $0.001
    other = scaled(gt_rows(120), 1250.0)  # the other token around $1,250

    def handle(request: httpx2.Request) -> httpx2.Response:
        token = request.url.params.get("token")
        if token == MINT:
            if not mint_available:
                return httpx2.Response(404, json={"errors": [{"status": "404"}]})
            rows = own
        elif token == "base":
            rows = other if reversed_pool else own
        elif token == OTHER_TOKEN:
            rows = other
        else:
            return httpx2.Response(404, json={"errors": [{"status": "404"}]})
        body = {"data": {"id": "x", "type": "ohlcv_request_response",
                         "attributes": {"ohlcv_list": sorted(rows, key=lambda r: -r[0])}}}  # fmt: skip
        return httpx2.Response(200, json=body)

    fake.handler = handle
    return own


def technical(fake_dexscreener: Any) -> Any:
    dex = dex_prior(fake_dexscreener)
    ctx = AgentContext(query="", assets=["NEWT"], asset_identity=AssetIdentity(chain="solana", address=MINT),
                       prior_results=results(dex), timeframe="5m")  # fmt: skip
    return asyncio.run(TechnicalAnalysisAgent().run(ctx))


def tokens(fake: FakeGeckoTerminal) -> list[str | None]:
    return [r.url.params.get("token") for r in fake.requests]


def test_normal_pool_orientation_indicators_unchanged(
    fake_dexscreener: Any, fake_geckoterminal: Any
) -> None:
    own = oriented(fake_geckoterminal, reversed_pool=False)
    r = technical(fake_dexscreener)
    assert r.status == "ok" and tokens(fake_geckoterminal) == [MINT]
    # Exactly what the pre-fix "base" request gave for a correctly oriented pool.
    limit = TechnicalAnalysisConfig().candles_to_fetch
    base = asyncio.run(GeckoTerminalProvider(transport=fake_geckoterminal.transport()).fetch_pool_candles(
        "solana", POOL, "5m", limit, symbol="NEWT", canonical_id=f"solana:{MINT}", now=NOW))  # fmt: skip
    assert base.candles[-1].close < 0.01 and len(own) == 120  # the token's own prices
    expected = analyze_series(base.model_copy(update={"symbol": "NEWT"}), TechnicalAnalysisConfig())
    for name in ("RSI 14", "MACD 12/26/9", "SMA 20", "EMA 50"):
        got = next(i for i in r.findings["indicators"] if i["name"] == name)
        want = expected.indicator(name)
        assert want is not None and got["value"] == pytest.approx(want.value), name
    assert r.findings["trend"] == expected.trend.model_dump(mode="json")
    assert r.findings["last_close"] == pytest.approx(expected.last_close)


def test_reversed_pool_orientation_analyzes_the_exact_token(
    fake_dexscreener: Any, fake_geckoterminal: Any
) -> None:
    oriented(fake_geckoterminal, reversed_pool=True)
    r = technical(fake_dexscreener)
    assert r.status == "ok"
    assert r.findings["last_close"] < 0.01  # the token (~$0.001), not the other token (~$1,250)
    assert tokens(fake_geckoterminal) == [MINT]  # "base" (the other token) is never requested


def test_unavailable_exact_token_candles_never_fall_back(
    fake_dexscreener: Any, fake_geckoterminal: Any
) -> None:
    oriented(fake_geckoterminal, reversed_pool=True, mint_available=False)
    r = technical(fake_dexscreener)
    assert r.status == "error"  # Technical evidence unavailable
    assert set(tokens(fake_geckoterminal)) == {MINT}  # no "base" / other-token substitute


def test_same_pool_opposite_token_is_a_separate_cache_entry(fake_geckoterminal: Any) -> None:
    oriented(fake_geckoterminal, reversed_pool=True)
    dex = DexCandleService(
        GeckoTerminalProvider(transport=fake_geckoterminal.transport()), now=lambda: NOW
    )
    mine = asyncio.run(
        dex.get_candles("solana", POOL, "5m", 100, symbol="NEWT", canonical_id=None, token=MINT)
    )
    other = asyncio.run(
        dex.get_candles("solana", POOL, "5m", 100, symbol="X", canonical_id=None, token=OTHER_TOKEN)
    )
    again = asyncio.run(
        dex.get_candles("solana", POOL, "5m", 100, symbol="NEWT", canonical_id=None, token=MINT)
    )
    assert tokens(fake_geckoterminal) == [MINT, OTHER_TOKEN]  # the repeat came from the cache
    assert mine.candles[-1].close < 1 < other.candles[-1].close
    assert again.candles == mine.candles


def test_exact_evm_contract_is_normalized_and_passed() -> None:
    contract = "0xAbCdEf0123456789aBcDeF0123456789AbCdEf01"

    class Registry:
        def __init__(self) -> None:
            self.asked: list[str] = []
            self.dex_candles = self

        provider_name = "GeckoTerminal"
        supported_timeframes = frozenset({"5m", "4h"})

        def covers(self, chain: str) -> bool:
            return True

        async def pool_candles(self, chain: str, pool: str, timeframe: Any, limit: int, *, symbol: str,
                               canonical_id: str | None, token: str) -> CandleSeries:  # fmt: skip
            self.asked.append(token)
            raise MarketDataUnavailableError("no candles in this test")

    snap = SolanaDexSnapshot(
        canonical_id=f"base:{contract.lower()}", mint=contract, symbol="PEPE", name=None,
        provider="DEX Screener", dex="uniswap", pair_address="0xpool", pair_url=None, quote_symbol="WETH",
        quote_address="0x4200000000000000000000000000000000000006", quote_kind="WETH", price_usd=1.0,
        price_native=None, liquidity_usd=1e5, market_cap_usd=None, fdv_usd=None, pair_created_at=None,
        pool_age_hours=None, first_pool_created_at=None, windows=[], fetched_at=datetime.now(UTC),
        candidates=[], primary_clear=True, chain="base",
    )  # fmt: skip
    registry = Registry()
    agent = TechnicalAnalysisAgent(registry=registry)  # type: ignore[arg-type]
    ctx = AgentContext(
        query="", assets=["PEPE"], asset_identity=AssetIdentity(chain="base", address=contract)
    )
    r = asyncio.run(agent._run_pool(ctx, "PEPE", snap, None))
    assert r.status == "error" and registry.asked == [
        contract.lower()
    ]  # exact contract, lowercased


def test_replay_serves_only_the_samples_exact_token() -> None:
    clock = HistoricalClock(NOW, __import__("datetime").timedelta(hours=26))
    candles = PointInTimeCandles(clock, "solana", MINT, POOL)
    source = PointInTimeCandleSource(candles)
    with pytest.raises(MarketDataUnavailableError, match="exact token"):
        asyncio.run(
            source.get_candles(
                "solana", POOL, "5m", 10, symbol="X", canonical_id=None, token=OTHER_TOKEN
            )
        )
    with pytest.raises(
        MarketDataUnavailableError, match="not loaded"
    ):  # the right token passes the gate
        asyncio.run(
            source.get_candles("solana", POOL, "5m", 10, symbol="X", canonical_id=None, token=MINT)
        )
