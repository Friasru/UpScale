"""The consolidated trading pipeline: asset resolver, trade context, capability routing,
pool candles, adaptive decisions, position intents, freshness, follow-ups and latency.

All offline: DEX Screener, GeckoTerminal, CoinGecko and the Solana RPC are fakes.
"""

import asyncio
from datetime import timedelta
from typing import Any

import httpx2
import pytest

from upscale.agents import AgentContext, TechnicalAnalysisAgent
from upscale.orchestrator import Orchestrator
from upscale.routing import route
from upscale.schemas import AgentResult, ChatMessage, ChatRequest
from upscale.services import opportunity as opportunity_service
from upscale.services import provider_registry
from upscale.services import risk as risk_service
from upscale.services.asset_profile import AssetIdentity, build_profile
from upscale.services.asset_resolver import AssetResolver, ResolverConfig, ticker_candidates
from upscale.services.capabilities import CapabilityUnavailableError
from upscale.services.dexscreener import DexScreenerProvider
from upscale.services.geckoterminal import GeckoTerminalProvider, parse_pool_candles
from upscale.services.market_data import (
    AssetNotFoundError,
    MarketDataUnavailableError,
    UnsupportedTimeframeError,
)
from upscale.services.risk import candle_freshness_factors, collect_inputs, observations
from upscale.services.solana_dex import SolanaDexService
from upscale.services.strategy import DEFAULT_STRATEGY, STRATEGIES, strategy_for
from upscale.services.technical_analysis import TechnicalAnalysis
from upscale.services.trade_context import (
    build_trade_context,
    leverage_from_text,
    position_from_text,
    requested_action,
    trader_context,
)

from .conftest import DEX_NOW, FakeDexScreener, FakeGeckoTerminal
from .test_onchain_safety import POOL, setup_token, wallet
from .test_opportunity import bear_ta, bullish_news, full_buy, risk, ta
from .test_risk_agent import market
from .test_solana_dex import BONK_MINT, MINT, OTHER_MINT, pair

NOW = DEX_NOW
BASE_WETH = "0x4200000000000000000000000000000000000006"
EVM_TOKEN = "0x1111111111111111111111111111111111111111"
EVM_TOKEN_2 = "0x2222222222222222222222222222222222222222"


# --- Helpers --------------------------------------------------------------------------------


def evm_pair(address: str = EVM_TOKEN, chain: str = "base", **kw: Any) -> dict[str, Any]:
    row = pair(f"0x{'ab' * 20}", mint=address, quote=BASE_WETH, chain=chain, **kw)
    row["quoteToken"] = {"address": BASE_WETH, "symbol": "WETH", "name": "Wrapped Ether"}
    return row


def resolver(fake: FakeDexScreener, **cfg: Any) -> AssetResolver:
    svc = SolanaDexService(DexScreenerProvider(transport=fake.transport()), now=lambda: NOW)
    return AssetResolver(search=svc, config=ResolverConfig(**cfg) if cfg else None)


def resolve(fake: FakeDexScreener, text: str, history: list[str] | None = None) -> Any:
    return asyncio.run(resolver(fake).resolve(text, history or []))


def ask(*messages: str) -> Any:
    history = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=m)
        for i, m in enumerate(messages)
    ]
    return asyncio.run(Orchestrator().respond(ChatRequest(messages=history)))


def gt_rows(
    count: int, *, step_s: int = 300, end_open: Any = None, start: float = 1.0, drift: float = 0.004
) -> list[list[float]]:
    """Consecutive closed 5m candles in a steady uptrend ending just before NOW."""
    end = end_open or (NOW - timedelta(seconds=step_s))
    rows = []
    price = start
    for i in range(count):
        opened = end - timedelta(seconds=step_s * (count - 1 - i))
        nxt = price * (1 + drift) if i % 4 else price * (1 - drift / 2)
        rows.append(
            [
                opened.timestamp(),
                price,
                max(price, nxt) * 1.001,
                min(price, nxt) * 0.999,
                nxt,
                1000.0 + i,
            ]
        )
        price = nxt
    return rows


def results(*rs: AgentResult) -> dict[Any, AgentResult]:
    return {r.agent: r for r in rs}


# --- Asset resolver -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "label", "registered"),
    [
        ("Should I buy BTC right now?", "BTC", True),
        ("what about ethereum", "ETH", True),
        ("Should I sell my SOL?", "SOL", True),
        ("Analyze BONK on 5m", "BONK", True),
        ("thoughts on dogwifhat", "WIF", True),
        ("Should I buy PEPE?", "PEPE", True),
    ],
)
def test_registry_assets_resolve_without_network(fake_dexscreener, text, label, registered) -> None:
    r = resolve(fake_dexscreener, text)
    assert r.primary is not None and r.primary.label == label
    assert r.primary.registered is registered and not r.primary.is_contract
    assert fake_dexscreener.requests == []


def test_exact_solana_mint_is_an_exact_identity(fake_dexscreener) -> None:
    r = resolve(fake_dexscreener, f"Should I buy {MINT}?")
    a = r.primary
    assert a.is_contract and a.confidence == "exact"
    assert (a.identity.chain, a.identity.address) == ("solana", MINT)


def test_registered_mint_resolves_to_the_registered_asset(fake_dexscreener) -> None:
    a = resolve(fake_dexscreener, f"check {BONK_MINT}").primary
    assert a.label == "BONK" and a.registered and not a.is_contract


def test_evm_address_with_a_chain_mention(fake_dexscreener) -> None:
    a = resolve(fake_dexscreener, f"Should I buy {EVM_TOKEN} on base?").primary
    assert (a.identity.chain, a.identity.address) == ("base", EVM_TOKEN)
    assert fake_dexscreener.requests == []  # the text named the chain


def test_evm_address_chain_is_discovered_from_its_pools(fake_dexscreener) -> None:
    fake_dexscreener.search[EVM_TOKEN] = [evm_pair()]
    a = resolve(fake_dexscreener, f"Should I buy {EVM_TOKEN}?").primary
    assert a.identity.chain == "base" and a.is_contract


def test_evm_address_on_several_chains_asks_which(fake_dexscreener) -> None:
    fake_dexscreener.search[EVM_TOKEN] = [evm_pair(chain="base"), evm_pair(chain="ethereum")]
    r = resolve(fake_dexscreener, f"Should I buy {EVM_TOKEN}?")
    assert r.assets == [] and "several chains" in r.clarification
    assert "Base" in r.clarification and "Ethereum" in r.clarification


def test_unknown_evm_address_asks_for_the_chain(fake_dexscreener) -> None:
    r = resolve(fake_dexscreener, f"Should I buy {EVM_TOKEN}?")
    assert r.assets == [] and "Which chain" in r.clarification


def test_dominant_ticker_is_discovered(fake_dexscreener) -> None:
    fake_dexscreener.search["USELESS"] = [
        pair(symbol="USELESS", liquidity=5e6),
        pair(
            "Tiny1111111111111111111111111111111111111111",
            mint=OTHER_MINT,
            symbol="USELESS",
            liquidity=20_000.0,
            volume={"h24": 1_000.0},
        ),
    ]
    r = resolve(fake_dexscreener, "Should I buy $USELESS?")
    a = r.primary
    assert a.source == "discovered" and a.label == "USELESS"
    assert (a.identity.chain, a.identity.address) == ("solana", MINT)
    assert "most of this ticker's DEX trading" in a.note


def test_ambiguous_ticker_asks_instead_of_guessing(fake_dexscreener) -> None:
    fake_dexscreener.search["DUP"] = [
        pair(symbol="DUP", liquidity=900_000.0),
        pair(
            "Two11111111111111111111111111111111111111111",
            mint=OTHER_MINT,
            symbol="DUP",
            liquidity=600_000.0,
        ),
        evm_pair(symbol="DUP", liquidity=400_000.0),
    ]
    r = resolve(fake_dexscreener, "should I buy $DUP")
    assert r.assets == []
    assert "Several tokens use the ticker $DUP" in r.clarification
    assert (
        r.clarification.count("\n1. ")
        + r.clarification.count("\n2. ")
        + r.clarification.count("\n3. ")
        == 3
    )
    assert "Paste the contract or mint address" in r.clarification


def test_thin_single_token_is_not_treated_as_established(fake_dexscreener) -> None:
    fake_dexscreener.search["THIN"] = [pair(symbol="THIN", liquidity=5_000.0)]
    r = resolve(fake_dexscreener, "Should I hold THIN?")
    assert r.assets == [] and r.clarification is not None


def test_uppercase_word_ticker_is_discovered(fake_dexscreener) -> None:
    fake_dexscreener.search["TRUMP"] = [pair(symbol="TRUMP", liquidity=50e6)]
    assert resolve(fake_dexscreener, "Should I hold TRUMP?").primary.label == "TRUMP"


def test_common_words_are_not_tickers() -> None:
    assert ticker_candidates("Should I BUY now? RSI and MACD on 5M look OK") == []
    assert ticker_candidates("what about $wif") == []  # registered: resolved elsewhere


def test_unknown_ticker_with_no_market_is_reported(fake_dexscreener) -> None:
    r = resolve(fake_dexscreener, "Should I buy $NOPE?")
    assert r.assets == [] and r.clarification is None
    assert "No DEX market" in r.notes[0]


def test_discovery_outage_is_reported_not_guessed(fake_dexscreener) -> None:
    fake_dexscreener.handler = lambda r: httpx2.Response(503)
    r = resolve(fake_dexscreener, "Should I buy $NOPE?")
    assert r.assets == [] and "Couldn't look up" in r.notes[0]


def test_follow_up_reuses_asset_and_timeframe(fake_dexscreener) -> None:
    r = resolve(fake_dexscreener, "Should I sell?", ["Analyze SOL 5m"])
    assert r.primary.label == "SOL" and r.primary.source == "history"
    assert r.timeframe == "5m"


def test_follow_up_with_a_new_timeframe_keeps_the_asset(fake_dexscreener) -> None:
    r = resolve(fake_dexscreener, "and on 1h?", ["Analyze SOL 5m"])
    assert (r.primary.label, r.timeframe) == ("SOL", "1h")


# --- Trader context -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "position", "action"),
    [
        ("Should I buy BTC right now?", "none", "buy"),
        ("Should I sell my SOL?", "long", "sell"),
        ("Should I hold TRUMP?", "long", "hold"),
        ("Analyze PEPE on 5m", "unknown", "analyze"),
        ("Should I buy or wait on ETH?", "none", "buy_or_wait"),
        ("I'm short BTC, should I cover my short?", "short", "buy"),
        ("I hold some BONK, thoughts?", "long", "analyze"),
    ],
)
def test_position_and_intent_from_wording(text, position, action) -> None:
    assert position_from_text(text) == position
    assert requested_action(text) == action


def test_position_carries_over_from_the_conversation() -> None:
    t = trader_context("what now?", ["I bought SOL yesterday"], "5m")
    assert (t.position, t.position_source, t.horizon) == ("long", "history", "scalp")


def test_leverage_is_recorded() -> None:
    assert leverage_from_text("I'm long BTC with 10x leverage") == 10.0
    assert leverage_from_text("should I buy BTC") is None


def test_trade_context_for_a_mint() -> None:
    profile = build_profile(AssetIdentity(chain="solana", address=MINT, metadata=None), now=NOW)
    trade = build_trade_context(profile, "exact", trader_context("buy?", [], "5m"), "5m")
    assert trade.identity.canonical_id == f"solana:{MINT}"
    assert trade.identity.confidence == "exact"
    caps = {d.capability: d.status for d in trade.data}
    assert set(caps) >= {"candles", "market_snapshot", "dex", "onchain", "derivatives", "social"}
    assert caps["derivatives"] == caps["social"] == "unavailable"


# --- Capability routing ---------------------------------------------------------------------


def test_providers_are_chosen_by_capability_and_chain() -> None:
    assert provider_registry.providers_for("dex_market", "solana") == ["DEX Screener"]
    assert provider_registry.providers_for("dex_market", "base") == ["DEX Screener"]
    assert provider_registry.providers_for("dex_market", "bitcoin") == []
    assert "GeckoTerminal" in provider_registry.providers_for("candles", "solana")
    assert "GeckoTerminal" not in provider_registry.providers_for("candles", "bitcoin")
    assert provider_registry.providers_for("onchain_safety", "base") == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: provider_registry.onchain_safety("base", EVM_TOKEN),
        lambda: provider_registry.onchain_safety("solana", MINT),  # no RPC configured
        lambda: provider_registry.derivatives("BTC"),
        lambda: provider_registry.social("BTC"),
        lambda: provider_registry.pool_candles(
            "bitcoin", "x", "5m", 10, symbol="X", canonical_id=None
        ),
        lambda: provider_registry.dex_market("bitcoin", "x"),
    ],
)
def test_unavailable_capabilities_say_so(call) -> None:
    with pytest.raises(CapabilityUnavailableError):
        asyncio.run(call())


# --- Pool candles (GeckoTerminal) -----------------------------------------------------------


def test_pool_candles_drop_the_open_candle_and_never_fill_gaps() -> None:
    rows = gt_rows(10)
    rows.append([NOW.timestamp(), 2.0, 2.1, 1.9, 2.0, 5.0])  # in progress: dropped
    gap_start = NOW - timedelta(minutes=5 * 30)
    rows += [
        [(gap_start - timedelta(minutes=5 * i)).timestamp(), 1.0, 1.1, 0.9, 1.0, 1.0]
        for i in range(3)
    ]
    candles, notes = parse_pool_candles(rows, "5m", NOW, "GeckoTerminal")
    assert len(candles) == 10 and candles[-1].timestamp == NOW - timedelta(minutes=5)
    assert notes and "gaps are never filled in" in notes[0]


@pytest.mark.parametrize(
    "rows",
    [
        [["x", 1, 1, 1, 1, 1]],
        [[NOW.timestamp(), 1, 0.5, 2, 1, 1]],
        [[NOW.timestamp() + 7, 1, 1, 1, 1, 1]],
    ],
)
def test_malformed_pool_candles_are_rejected(rows) -> None:
    with pytest.raises(MarketDataUnavailableError):
        parse_pool_candles(rows, "5m", NOW + timedelta(hours=1), "GeckoTerminal")


def test_pool_candles_request_and_normalization(fake_geckoterminal: FakeGeckoTerminal) -> None:
    fake_geckoterminal.candles[POOL] = gt_rows(120)
    series = asyncio.run(
        provider_registry.pool_candles(
            "solana", POOL, "5m", 100, symbol="NEWT", canonical_id=f"solana:{MINT}"
        )
    )
    req = fake_geckoterminal.requests[0]
    assert req.url.path == f"/api/v2/networks/solana/pools/{POOL}/ohlcv/minute"
    assert req.url.params["aggregate"] == "5" and req.url.params["currency"] == "usd"
    assert len(series.candles) == 100 and series.provider == "GeckoTerminal"
    assert series.canonical_id == f"solana:{MINT}" and series.volume_unit == "USD"
    assert series.as_of == NOW
    # Cached: a second read makes no request.
    asyncio.run(
        provider_registry.pool_candles("solana", POOL, "5m", 100, symbol="NEWT", canonical_id=None)
    )
    assert len(fake_geckoterminal.requests) == 1


def test_thirty_minute_pool_candles_are_not_offered() -> None:
    provider = GeckoTerminalProvider(transport=httpx2.MockTransport(lambda r: httpx2.Response(500)))
    with pytest.raises(UnsupportedTimeframeError):
        asyncio.run(
            provider.fetch_pool_candles(
                "solana", POOL, "30m", 10, symbol="X", canonical_id=None, now=NOW
            )
        )


@pytest.mark.parametrize(
    ("status", "error"), [(404, AssetNotFoundError), (429, MarketDataUnavailableError)]
)
def test_pool_candle_failures(fake_geckoterminal, status, error) -> None:
    fake_geckoterminal.handler = lambda r: httpx2.Response(status)
    with pytest.raises(error):
        asyncio.run(
            provider_registry.pool_candles("solana", POOL, "5m", 10, symbol="X", canonical_id=None)
        )


# --- Technical on pool candles --------------------------------------------------------------


def dex_prior(fake_dex: FakeDexScreener, mint: str = MINT, **kw: Any) -> AgentResult:
    from upscale.agents import DexMarketAgent

    fake_dex.pairs[mint] = [pair(mint=mint, **kw)]
    ctx = AgentContext(
        query="", assets=[mint], asset_identity=AssetIdentity(chain="solana", address=mint)
    )
    return asyncio.run(DexMarketAgent().run(ctx))


def technical_for_mint(fake_dex, fake_gt, count: int = 120, **kw: Any) -> AgentResult:
    dex = dex_prior(fake_dex, **kw)
    fake_gt.candles[POOL] = gt_rows(count)
    ctx = AgentContext(
        query="",
        assets=["NEWT"],
        asset_identity=AssetIdentity(chain="solana", address=MINT),
        prior_results=results(dex),
        timeframe="5m",
    )
    return asyncio.run(TechnicalAnalysisAgent().run(ctx))


def test_contract_token_technical_uses_its_own_pool(fake_dexscreener, fake_geckoterminal) -> None:
    r = technical_for_mint(fake_dexscreener, fake_geckoterminal)
    assert r.status == "ok"
    assert r.findings["provider"] == "GeckoTerminal"
    assert r.findings["canonical_id"] == f"solana:{MINT}"
    assert r.findings["provider_id"] == POOL and r.findings["timeframe"] == "5m"


def test_contract_token_without_a_pool_gets_no_candles(fake_geckoterminal) -> None:
    ctx = AgentContext(
        query="", assets=["NEWT"], asset_identity=AssetIdentity(chain="solana", address=MINT)
    )
    r = asyncio.run(TechnicalAnalysisAgent().run(ctx))
    assert r.status == "error" and "no usable DEX pool" in r.error
    assert fake_geckoterminal.requests == []


def test_another_tokens_pool_is_never_used(fake_dexscreener, fake_geckoterminal) -> None:
    dex = dex_prior(fake_dexscreener, mint=OTHER_MINT)
    ctx = AgentContext(
        query="",
        assets=["NEWT"],
        asset_identity=AssetIdentity(chain="solana", address=MINT),
        prior_results=results(dex),
    )
    r = asyncio.run(TechnicalAnalysisAgent().run(ctx))
    assert r.status == "error" and fake_geckoterminal.requests == []


def test_unsupported_timeframe_is_recorded_not_silent(fake_dexscreener, fake_geckoterminal) -> None:
    dex = dex_prior(fake_dexscreener)
    fake_geckoterminal.candles[POOL] = [
        r for r in gt_rows(60, step_s=14400, end_open=NOW - timedelta(hours=4))
    ]
    ctx = AgentContext(
        query="",
        assets=["NEWT"],
        asset_identity=AssetIdentity(chain="solana", address=MINT),
        prior_results=results(dex),
        timeframe="30m",
    )
    r = asyncio.run(TechnicalAnalysisAgent().run(ctx))
    assert r.findings["timeframe"] == "4h"
    assert "30m candles aren't available for DEX pools" in r.findings["timeframe_note"]


def test_listed_token_falls_back_to_its_pool_when_exchange_candles_fail(
    fake_dexscreener, fake_geckoterminal
) -> None:
    dex = dex_prior(fake_dexscreener, mint=BONK_MINT, symbol="Bonk")
    fake_geckoterminal.candles[POOL] = gt_rows(120, step_s=14400, end_open=NOW - timedelta(hours=4))
    ctx = AgentContext(query="", assets=["BONK"], prior_results=results(dex))
    r = asyncio.run(TechnicalAnalysisAgent().run(ctx))  # fake CoinGecko doesn't know BONK
    assert r.status == "ok" and r.findings["provider"] == "GeckoTerminal"
    assert "Exchange candles were unavailable" in r.findings["timeframe_note"]
    assert r.findings["canonical_id"] == f"solana:{BONK_MINT}"


# --- Adaptive decisions for a new DEX token -------------------------------------------------


def _token_prior(fake_rpc, fake_dex, candles: int) -> dict[Any, AgentResult]:
    from upscale.agents import OnchainSafetyAgent

    from .test_onchain_safety import holding, service

    setup_token(fake_rpc)
    for i in range(300):
        holding(fake_rpc, f"Sma11{i:04d}".ljust(43, "1"), 0.001, n=2000 + i)
    dex = dex_prior(
        fake_dex,
        liquidity=5e6,
        age=timedelta(days=20),
        change={"m5": 0.1, "h1": 1.0},
        price="84445",  # consistent with the candle fixture's prices
    )
    ctx = AgentContext(
        query="",
        assets=[MINT],
        asset_identity=AssetIdentity(chain="solana", address=MINT),
        prior_results=results(dex),
    )
    chain = asyncio.run(OnchainSafetyAgent(service(fake_rpc)).run(ctx))
    tech = ta(symbol="NEWT")
    tech.findings["canonical_id"] = f"solana:{MINT}"
    tech.findings["provider"] = "GeckoTerminal"  # pool candles
    tech.findings["candle_count"] = candles
    return results(dex, chain, tech)


def _decide(prior: dict[Any, AgentResult], position: str = "unknown") -> Any:
    ident = AssetIdentity(chain="solana", address=MINT)
    p = build_profile(
        ident,
        observations(collect_inputs(prior, "NEWT")),
        now=NOW,
        integrated=frozenset({"candles", "market_snapshot", "news", "dex", "onchain"}),
    )
    r = risk_service.assess(prior, "NEWT", profile=p)
    rr = AgentResult(agent="risk", mock=False, summary="r", findings=r.model_dump(mode="json"))
    return opportunity_service.assess(prior | {"risk": rr}, "NEWT", profile=p, position=position), p


def test_new_token_with_safety_liquidity_and_candles_can_act(
    fake_solana_rpc, fake_dexscreener
) -> None:
    a, p = _decide(_token_prior(fake_solana_rpc, fake_dexscreener, candles=180))
    assert p.category == "new_dex_token" and p.unavailable_critical() == []
    assert p.evidence_status("technical_structure").status == "available"
    assert a.action == "buy"  # same strict rules as BTC; every safety gate passed


def test_new_token_with_too_few_candles_waits(fake_solana_rpc, fake_dexscreener) -> None:
    a, p = _decide(_token_prior(fake_solana_rpc, fake_dexscreener, candles=30))
    assert p.evidence_status("technical_structure").status == "insufficient"
    assert a.action == "wait" and a.bullish_evidence == []


def test_ticker_candles_are_never_attributed_to_a_mint(fake_solana_rpc, fake_dexscreener) -> None:
    prior = _token_prior(fake_solana_rpc, fake_dexscreener, candles=180)
    prior["technical_analysis"].findings["canonical_id"] = None  # exchange candles by ticker
    prior["technical_analysis"].findings["provider"] = "Kraken"
    a, _ = _decide(prior)
    assert a.action == "wait" and a.bullish_evidence == []


# --- Position intents -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "position", "action", "intent"),
    [
        ("buy", "none", "buy", "enter_long"),
        ("sell", "none", "wait", "wait"),
        ("buy", "long", "wait", "hold_long"),
        ("sell", "long", "sell", "exit_long"),
        ("buy", "short", "buy", "cover_short"),
        ("sell", "short", "wait", "hold_short"),
        ("buy", "unknown", "buy", "enter_long"),
        ("sell", "unknown", "sell", "exit_long"),
    ],
)
def test_position_changes_what_the_action_means(setup, position, action, intent) -> None:
    prior = results(
        *(full_buy() if setup == "buy" else [bear_ta(), market(price=84_335.0), risk()])
    )
    a = opportunity_service.assess(prior, "BTC", position=position)
    assert (a.setup_action, a.action, a.intent) == (setup, action, intent)
    assert a.action_meaning


def test_weakening_long_is_reduced_and_a_blocked_bullish_long_is_held() -> None:
    # A bearish setup that high risk keeps from being a confirmed SELL.
    blocked = results(bear_ta(), market(price=84_335.0), risk("high"))
    base = opportunity_service.assess(blocked, "BTC")
    assert base.action == "wait" and base.bearish_score - base.bullish_score >= 3
    a = opportunity_service.assess(blocked, "BTC", position="long")
    assert (a.intent, a.action) == ("reduce_long", "sell")
    assert "consider reducing" in a.summary
    calm = results(ta(trend="mixed"), market(), risk())
    held = opportunity_service.assess(calm, "BTC", position="long")
    assert (held.intent, held.action) == ("hold_long", "wait")


def test_unknown_position_keeps_btc_decisions_identical() -> None:
    for prior in (full_buy(), [bear_ta(), market(price=84_335.0), bullish_news(), risk()]):
        a = opportunity_service.assess(results(*prior), "BTC")
        assert a.action == a.setup_action


# --- Strategy, freshness --------------------------------------------------------------------


def test_every_category_has_a_strategy_equal_to_todays_defaults() -> None:
    assert set(STRATEGIES) >= {
        "major_crypto",
        "large_cap_alt",
        "established_memecoin",
        "new_dex_token",
        "stablecoin",
    }
    assert all(s == DEFAULT_STRATEGY for s in STRATEGIES.values())
    assert strategy_for(build_profile("BTC")) == DEFAULT_STRATEGY


def test_stale_pool_candles_raise_uncertainty() -> None:
    t = TechnicalAnalysis.model_validate(ta(symbol="NEWT").findings)
    fresh = t.model_copy(update={"as_of": t.last_candle_at + timedelta(minutes=10)})
    stale = t.model_copy(update={"as_of": t.last_candle_at + timedelta(hours=2)})
    cfg = DEFAULT_STRATEGY.risk
    assert candle_freshness_factors(fresh, cfg) == []
    f = candle_freshness_factors(stale, cfg)[0]
    assert f.id == "stale_candles" and f.affects == "uncertainty"
    assert candle_freshness_factors(t, cfg) == []  # exchange candles: no as_of, always current


# --- End to end -----------------------------------------------------------------------------


def test_ambiguous_ticker_gets_a_clarification_not_an_analysis(fake_dexscreener) -> None:
    fake_dexscreener.search["DUP"] = [
        pair(symbol="DUP", liquidity=900_000.0),
        pair(
            "Two11111111111111111111111111111111111111111",
            mint=OTHER_MINT,
            symbol="DUP",
            liquidity=600_000.0,
        ),
    ]
    response = ask("Should I buy $DUP?")
    assert response.message.content.startswith("Several tokens use the ticker $DUP")
    assert response.analysis.agent_results == []


def test_discovered_token_end_to_end(fake_dexscreener, fake_geckoterminal) -> None:
    fake_dexscreener.search["USELESS"] = [pair(symbol="USELESS", liquidity=5e6)]
    fake_dexscreener.pairs[MINT] = [pair(symbol="USELESS", liquidity=5e6)]
    fake_geckoterminal.candles[POOL] = gt_rows(120)
    response = ask("Should I buy $USELESS on 5m?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert list(by_agent) == ["dex_market", "technical_analysis", "risk", "opportunity"]
    assert by_agent["technical_analysis"].findings["provider"] == "GeckoTerminal"
    decision = by_agent["opportunity"].findings
    assert decision["asset_profile"]["canonical_id"] == f"solana:{MINT}"
    assert decision["position"] == "none"
    assert decision["action"] == "wait"  # no on-chain safety evidence: required for new tokens
    assert "most of this ticker's DEX trading" in response.analysis.uncertainty.notes[0]
    assert "asset_resolution" in response.analysis.timings and "total" in response.analysis.timings


def test_evm_memecoin_end_to_end(fake_dexscreener, fake_geckoterminal) -> None:
    fake_dexscreener.pairs[EVM_TOKEN] = [evm_pair(symbol="BMEME")]
    response = ask(f"Should I buy {EVM_TOKEN} on base?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert "onchain_safety" not in by_agent  # no EVM on-chain provider yet
    assert by_agent["dex_market"].findings["chain"] == "base"
    assert by_agent["dex_market"].findings["snapshot"]["quote_kind"] == "WETH"
    decision = by_agent["opportunity"].findings
    assert decision["action"] == "wait"
    assert decision["asset_profile"]["canonical_id"] == f"base:{EVM_TOKEN}"
    onchain = next(
        c for c in decision["asset_profile"]["capabilities"] if c["capability"] == "onchain"
    )
    assert "integrated for Base" in onchain["reason"]


def test_follow_up_end_to_end_reuses_sol_and_5m(fake_coingecko) -> None:
    response = ask("Analyze SOL 5m", "SOL 5m: ...", "Should I sell?")
    assert response.analysis.assets == ["SOL"]
    opp = next(r for r in response.analysis.agent_results if r.agent == "opportunity")
    assert opp.findings["position"] == "long"
    tech = next(r for r in response.analysis.agent_results if r.agent == "technical_analysis")
    assert tech.findings.get("requested_timeframe") == "5m" or tech.status == "error"


def test_education_is_not_hijacked_by_follow_up() -> None:
    response = ask("Analyze SOL 5m", "SOL 5m: ...", "What is RSI?")
    assert response.analysis.agents_used == ["education"]


def test_prefetch_and_agents_share_one_on_chain_read(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90))
    fake_dexscreener.pairs[MINT] = [pair()]
    ask(f"Should I buy {MINT}?")
    assert sum(r["method"] == "getAccountInfo" for r in fake_solana_rpc.requests) == 1
    assert len([r for r in fake_dexscreener.requests if "token-pairs" in r.url.path]) == 1


def test_dex_candle_outage_is_isolated(fake_dexscreener, fake_geckoterminal) -> None:
    fake_dexscreener.pairs[MINT] = [pair()]
    fake_geckoterminal.handler = lambda r: httpx2.Response(503)
    response = ask(f"Should I buy {MINT}?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert by_agent["technical_analysis"].status == "error"
    assert (
        by_agent["opportunity"].status == "ok"
        and by_agent["opportunity"].findings["action"] == "wait"
    )


def test_btc_route_and_agents_are_unchanged() -> None:
    assert route("Should I buy BTC right now?", has_images=False).agents == [
        "technical_analysis",
        "market",
        "news_sentiment",
        "risk",
        "opportunity",
    ]


# --- Real DEX Screener search responses (recorded live, 2026-09-25) -------------------------

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _serve_fixture(fake: FakeDexScreener, query: str) -> None:
    body = __import__("json").loads(
        (FIXTURES / f"dexscreener_search_{query.lower()}.json").read_text()
    )
    fake.search[query] = body["pairs"]


def test_real_useless_search_picks_the_traded_token_not_the_inflated_pool(fake_dexscreener) -> None:
    # A copycat's single pool reports $247M liquidity with 20 trades a day; the real Useless
    # Coin trades ~$9M a day across ~20 pools. Ranking by reported liquidity picked the
    # copycat; ranking by real trading must not.
    _serve_fixture(fake_dexscreener, "USELESS")
    a = resolve(fake_dexscreener, "Should I buy $USELESS right now?").primary
    assert a is not None
    assert a.identity.address == "Dz9mQ9NzkBcCsuGPFJ3r1bS4wgqKMHBPiVuniW8Mbonk"
    assert a.identity.chain == "solana"


def test_real_trump_search_picks_official_trump_over_zero_volume_pools(fake_dexscreener) -> None:
    _serve_fixture(fake_dexscreener, "TRUMP")
    a = resolve(fake_dexscreener, "Should I hold TRUMP?").primary
    assert a is not None
    assert a.identity.address == "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN"


def test_liquidity_without_trading_is_never_established(fake_dexscreener) -> None:
    fake_dexscreener.search["GHOST"] = [
        pair(symbol="GHOST", liquidity=50e6, volume={"h24": 0.0}, txns={"h24": (1, 1)})
    ]
    r = resolve(fake_dexscreener, "Should I buy $GHOST?")
    assert r.assets == [] and r.clarification is not None
