"""DEX contract safety policy (venue-based, not category-based), market pool vs technical
pool, and venue parsing. All offline."""

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from upscale.agents import AgentContext, TechnicalAnalysisAgent
from upscale.orchestrator import Orchestrator
from upscale.schemas import AgentResult, ChatMessage, ChatRequest
from upscale.services import opportunity as opportunity_service
from upscale.services import risk as risk_service
from upscale.services.asset_profile import AssetIdentity, build_profile
from upscale.services.asset_registry import AssetMetadata
from upscale.services.risk import collect_inputs, observations
from upscale.services.solana_dex import SolanaDexSnapshot, build_snapshot
from upscale.services.technical_pool import TechnicalPoolConfig, alternate_pools
from upscale.services.trade_context import (
    VenueRequest,
    build_trade_context,
    requested_venue,
    trader_context,
)

from .conftest import DEX_NOW, FakeDexScreener, FakeGeckoTerminal
from .test_opportunity import full_buy, ta
from .test_solana_dex import MINT, OTHER_MINT, pair
from .test_trading_pipeline import BASE_WETH, gt_rows

NOW = DEX_NOW
PEPE = "0x6982508145454ce325ddbe47a25d4ec3d2311933"
MARKET_POOL = "MktPoo1111111111111111111111111111111111111"
ALT_POOL = "A1tPoo1111111111111111111111111111111111111"
ALT_POOL_2 = "A1tPoo2111111111111111111111111111111111111"
ALL = frozenset({"candles", "market_snapshot", "news", "dex", "onchain"})


def results(*rs: AgentResult) -> dict[Any, AgentResult]:
    return {r.agent: r for r in rs}


def old_untagged_token(**meta: Any) -> AssetIdentity:
    """A Solana token traded on DEXes for over a year, with no meme tag."""
    fields: dict[str, Any] = {
        "symbol": "OLDT",
        "chain": "solana",
        "dex_listed": True,
        "pool_created_at": NOW - timedelta(days=400),
        "liquidity_usd": 5e6,
    }
    return AssetIdentity(chain="solana", address=MINT, metadata=AssetMetadata(**(fields | meta)))


def bullish_pool_technical(canonical: str) -> AgentResult:
    t = ta(symbol="OLDT")
    t.findings["canonical_id"] = canonical
    t.findings["provider"] = "GeckoTerminal"
    return t


# --- 1. DEX contract safety policy ----------------------------------------------------------


@pytest.mark.parametrize("tags", [frozenset(), frozenset({"meme"})])
def test_old_dex_token_requires_safety_with_or_without_a_meme_tag(tags) -> None:
    p = build_profile(old_untagged_token(tags=tags), now=NOW, integrated=ALL)
    assert p is not None and p.market_policy.kind == "dex_contract"
    assert p.category != "new_dex_token"  # over 90 days old
    assert {
        "token_authorities",
        "holder_concentration",
        "dex_liquidity",
        "technical_structure",
    } <= set(p.decision_critical_evidence)


def test_old_untagged_dex_token_waits_without_on_chain_safety() -> None:
    # A bullish setup from the token's own pool, but no on-chain safety evidence.
    prior = results(bullish_pool_technical(f"solana:{MINT}"))
    p = build_profile(
        old_untagged_token(), observations(collect_inputs(prior, "OLDT")), now=NOW, integrated=ALL
    )
    assert p.category == "unknown_crypto"
    a = opportunity_service.assess(prior, "OLDT", profile=p)
    assert a.action == "wait"
    reasons = " ".join(f.reason for f in a.blocking_factors)
    assert "token mint/freeze authorities" in reasons and "holder concentration" in reasons


def test_established_memecoin_on_an_exchange_keeps_category_rules() -> None:
    cex = build_profile("PEPE", venue="cex")
    assert cex is not None and cex.market_policy.kind == "exchange"
    assert cex.decision_critical_evidence == ["technical_structure"]  # no on-chain gate


def test_same_asset_on_a_dex_requires_chain_safety() -> None:
    ident = AssetIdentity(symbol="PEPE", chain="ethereum", address=PEPE)
    dex = build_profile(ident, venue="dex", integrated=ALL)
    assert dex is not None and dex.market_policy.kind == "dex_contract"
    assert dex.canonical_id == "coingecko:pepe"  # same asset, different market
    missing = {e.evidence for e in dex.unavailable_critical()}
    assert {"token_authorities", "holder_concentration"} <= missing
    assert "integrated for Ethereum" in dex.evidence_status("token_authorities").reason


def test_plain_ticker_and_major_coins_are_not_dex_trades() -> None:
    for symbol in ("BTC", "SOL", "PEPE", "BONK"):
        p = build_profile(symbol)
        assert p is not None and p.market_policy.kind != "dex_contract"


def test_candles_from_the_tokens_pool_make_it_a_dex_trade() -> None:
    prior = results(bullish_pool_technical("coingecko:bonk"))
    prior["technical_analysis"].findings["symbol"] = "BONK"
    p = risk_service.profile_from_results(prior, "BONK")
    assert p is not None and p.market_policy.kind == "dex_contract"
    assert "DEX pool" in p.market_policy.reason


# --- PEPE on Kraken vs PEPE on Uniswap, end to end ------------------------------------------


def ask(text: str) -> Any:
    return asyncio.run(
        Orchestrator().respond(ChatRequest(messages=[ChatMessage(role="user", content=text)]))
    )


def uniswap_pepe(fake_dex: FakeDexScreener, fake_gt: FakeGeckoTerminal) -> None:
    pool = "0x" + "cd" * 20
    row = pair(pool, mint=PEPE, symbol="PEPE", chain="ethereum", dex="uniswap", quote=BASE_WETH)
    row["quoteToken"] = {
        "address": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        "symbol": "WETH",
        "name": "WETH",
    }
    fake_dex.pairs[PEPE] = [row]
    fake_gt.candles[pool] = gt_rows(120, step_s=14400, end_open=NOW - timedelta(hours=4))


def test_pepe_on_kraken_uses_the_exchange_path(fake_dexscreener, fake_geckoterminal) -> None:
    uniswap_pepe(fake_dexscreener, fake_geckoterminal)
    response = ask("Should I buy PEPE/USD on Kraken?")
    agents = [r.agent for r in response.analysis.agent_results]
    assert "dex_market" not in agents
    opp = response.analysis.agent_results[-1].findings
    assert opp["asset_profile"]["market_policy"]["kind"] == "exchange"
    assert fake_geckoterminal.requests == []


def test_pepe_on_uniswap_is_a_dex_contract_trade(fake_dexscreener, fake_geckoterminal) -> None:
    uniswap_pepe(fake_dexscreener, fake_geckoterminal)
    response = ask("Should I buy PEPE on Uniswap?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert by_agent["dex_market"].findings["snapshot"]["dex"] == "uniswap"
    assert by_agent["technical_analysis"].findings["provider"] == "GeckoTerminal"
    opp = by_agent["opportunity"].findings
    assert opp["asset_profile"]["market_policy"]["kind"] == "dex_contract"
    assert opp["action"] == "wait"  # no EVM on-chain safety provider yet
    assert any("token mint/freeze authorities" in b["reason"] for b in opp["blocking_factors"])


def test_pasted_contract_of_a_registered_token_is_a_dex_trade(
    fake_dexscreener, fake_geckoterminal
) -> None:
    uniswap_pepe(fake_dexscreener, fake_geckoterminal)
    response = ask(f"Should I buy {PEPE}?")
    opp = response.analysis.agent_results[-1].findings
    assert opp["asset_profile"]["canonical_id"] == "coingecko:pepe"
    assert opp["asset_profile"]["market_policy"]["kind"] == "dex_contract"
    assert opp["action"] == "wait"


# --- 3. Market pool vs technical pool -------------------------------------------------------


def jup_like(
    fake_dex: FakeDexScreener, fake_gt: FakeGeckoTerminal, alt_price: str = "0.0123"
) -> None:
    """A deep but gappy market pool, plus an active pool with full 5m history."""
    fake_dex.pairs[MINT] = [
        pair(MARKET_POOL, liquidity=2.2e6, txns={"h24": (2000, 2000)}),
        pair(
            ALT_POOL,
            dex="orca",
            liquidity=780_000.0,
            txns={"h24": (12_000, 11_000)},
            price=alt_price,
        ),
    ]
    fake_gt.candles[MARKET_POOL] = gt_rows(2)  # only 2 consecutive closed candles
    fake_gt.candles[ALT_POOL] = gt_rows(180)


def dex_result(mint: str = MINT) -> AgentResult:
    from upscale.agents import DexMarketAgent

    ctx = AgentContext(
        query="", assets=[mint], asset_identity=AssetIdentity(chain="solana", address=mint)
    )
    return asyncio.run(DexMarketAgent().run(ctx))


def technical(dex: AgentResult, venue: VenueRequest | None = None) -> AgentResult:
    ident = AssetIdentity(chain="solana", address=MINT)
    profile = build_profile(ident, now=NOW, integrated=ALL)
    trade = build_trade_context(
        profile, "exact", trader_context("buy?", [], "5m"), "5m", venue=venue
    )
    ctx = AgentContext(
        query="",
        assets=["NEWT"],
        asset_identity=ident,
        prior_results=results(dex),
        timeframe="5m",
        trade=trade,
    )
    return asyncio.run(TechnicalAnalysisAgent().run(ctx))


def test_gappy_market_pool_falls_back_to_an_active_pool_of_the_same_token(
    fake_dexscreener, fake_geckoterminal
) -> None:
    jup_like(fake_dexscreener, fake_geckoterminal)
    dex = dex_result()
    assert dex.findings["snapshot"]["pair_address"] == MARKET_POOL  # market pool unchanged
    r = technical(dex)
    pools = r.findings["pools"]
    assert pools["market_pool"]["address"] == MARKET_POOL
    assert pools["technical_pool"]["address"] == ALT_POOL
    assert "only 2 consecutive closed 5m candles" in pools["fallback_reason"]
    assert r.findings["candle_count"] == 180 and r.findings["provider_id"] == ALT_POOL
    assert r.findings["canonical_id"] == f"solana:{MINT}"
    assert any(line.startswith("Market pool:") for line in r.evidence)


def test_explicit_venue_prevents_the_fallback(fake_dexscreener, fake_geckoterminal) -> None:
    jup_like(fake_dexscreener, fake_geckoterminal)
    r = technical(dex_result(), VenueRequest(kind="dex", dex="raydium"))
    pools = r.findings["pools"]
    assert pools["technical_pool"]["address"] == MARKET_POOL and pools["fallback_reason"] is None
    assert r.findings["candle_count"] == 2
    assert "trader asked for this DEX/pool" in pools["rejected"][0]
    assert all(ALT_POOL not in str(req.url) for req in fake_geckoterminal.requests)


def test_fallback_is_rejected_when_pools_disagree_on_price(
    fake_dexscreener, fake_geckoterminal
) -> None:
    jup_like(fake_dexscreener, fake_geckoterminal, alt_price="0.0140")  # ~14% apart
    dex = dex_result()
    r = technical(dex)
    pools = r.findings["pools"]
    assert pools["technical_pool"]["address"] == MARKET_POOL
    assert pools["price_rejected"] and "differs" in pools["price_rejected"][0]
    review = risk_service.assess(results(dex, r), "NEWT")
    assert any(f.id == "pool_price_divergence" for f in review.factors)


def test_fallback_cannot_cross_to_another_token(fake_dexscreener, fake_geckoterminal) -> None:
    # A pool of a different token with the same ticker is never a candidate.
    snapshot = build_snapshot(
        MINT,
        [
            *(
                __import__("upscale.services.dexscreener", fromlist=["parse_pair"]).parse_pair(p)
                for p in [
                    pair(MARKET_POOL, liquidity=2.2e6),
                    pair(
                        ALT_POOL, mint=OTHER_MINT, liquidity=900_000.0, txns={"h24": (9000, 9000)}
                    ),
                ]
            )
        ],
        "DEX Screener",
        NOW,
    )
    alternates = alternate_pools(snapshot, TechnicalPoolConfig())
    assert [c.pair_address for c in snapshot.candidates] == [MARKET_POOL]
    assert alternates.pools == []


def test_alternate_ordering_is_deterministic() -> None:
    parse = __import__("upscale.services.dexscreener", fromlist=["parse_pair"]).parse_pair
    snapshot: SolanaDexSnapshot = build_snapshot(
        MINT,
        [
            parse(pair(MARKET_POOL, liquidity=3e6)),
            parse(pair(ALT_POOL, liquidity=400_000.0, txns={"h24": (500, 500)})),
            parse(pair(ALT_POOL_2, liquidity=300_000.0, txns={"h24": (900, 900)})),
        ],
        "DEX Screener",
        NOW,
    )
    order = [c.pair_address for c in alternate_pools(snapshot, TechnicalPoolConfig()).pools]
    assert order == [ALT_POOL_2, ALT_POOL]  # most trades first


def test_technical_pool_config_is_validated_and_tunable() -> None:
    with pytest.raises(ValueError):
        TechnicalPoolConfig(max_price_divergence_pct=0)
    with pytest.raises(ValueError):
        TechnicalPoolConfig(min_candles=0)
    parse = __import__("upscale.services.dexscreener", fromlist=["parse_pair"]).parse_pair
    snapshot = build_snapshot(
        MINT,
        [
            parse(pair(MARKET_POOL, liquidity=3e6)),
            parse(pair(ALT_POOL, liquidity=400_000.0, price="0.0126")),
        ],
        "DEX Screener",
        NOW,
    )
    assert alternate_pools(snapshot, TechnicalPoolConfig()).pools == []  # 2.4% > 2%
    assert (
        len(alternate_pools(snapshot, TechnicalPoolConfig(max_price_divergence_pct=3.0)).pools) == 1
    )


def test_requested_dex_restricts_the_market_pool(fake_dexscreener) -> None:
    from upscale.services import solana_dex_service

    fake_dexscreener.pairs[MINT] = [
        pair(MARKET_POOL, liquidity=2e6),
        pair(ALT_POOL, dex="orca", liquidity=500_000.0),
    ]
    s = asyncio.run(solana_dex_service.get_snapshot(MINT, dex="orca"))
    assert s.pair_address == ALT_POOL and s.requested_venue == "orca"
    with pytest.raises(Exception, match="no pumpfun pool"):
        asyncio.run(solana_dex_service.get_snapshot(MINT, dex="pumpfun"))
    assert len(fake_dexscreener.requests) == 1  # raw pools cached across venues


# --- Venue parsing --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "dex"),
    [
        ("Should I buy PEPE/USD on Kraken?", "cex", None),
        ("Should I buy PEPE on Uniswap?", "dex", "uniswap"),
        ("analyze BONK on orca", "dex", "orca"),
        ("is the pump.fun price ok", "dex", "pumpfun"),
        ("what about BONK on the DEX", "dex", None),
        ("Should I buy BTC right now?", "unknown", None),
    ],
)
def test_venue_from_the_request(text, kind, dex) -> None:
    v = requested_venue(text)
    assert (v.kind, v.dex) == (kind, dex)


def test_a_second_address_is_a_requested_pool() -> None:
    v = requested_venue(f"buy {MINT} in pool {MARKET_POOL}", token_address=MINT)
    assert (v.kind, v.pool) == ("dex", MARKET_POOL)


# --- BTC / SOL unchanged --------------------------------------------------------------------


@pytest.mark.parametrize("symbol", ["BTC", "SOL"])
def test_btc_and_sol_decisions_are_unchanged_by_the_policy(symbol) -> None:
    prior = results(*full_buy(symbol=symbol))
    plain = opportunity_service.assess(prior, symbol)
    profiled = opportunity_service.assess(
        prior, symbol, profile=risk_service.profile_from_results(prior, symbol)
    )
    assert profiled.action == plain.action == "buy"
    p = risk_service.profile_from_results(prior, symbol)
    assert p is not None and p.market_policy.kind == "exchange"
