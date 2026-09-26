"""Solana DEX market data: DEX Screener parsing, pool selection, the caching service, the
DEX agent, routing, and how the asset profile, Risk and Opportunity use DEX evidence.

All offline: DEX Screener is replaced by `FakeDexScreener` / MockTransport.
"""

import asyncio
from datetime import timedelta
from typing import Any

import httpx2
import pytest

from upscale.agents import AgentContext, DexMarketAgent
from upscale.orchestrator import Orchestrator, with_profile_agents
from upscale.routing import detect_solana_mint, route
from upscale.schemas import AgentResult, ChatMessage, ChatRequest
from upscale.services import opportunity as opportunity_service
from upscale.services import risk as risk_service
from upscale.services.asset_profile import AssetIdentity, build_profile
from upscale.services.asset_registry import AssetMetadata, AssetRegistry
from upscale.services.dexscreener import DexScreenerProvider, parse_pair
from upscale.services.market_data import (
    AssetNotFoundError,
    InvalidRequestError,
    MarketDataUnavailableError,
)
from upscale.services.risk import DexRiskConfig, collect_inputs, observations
from upscale.services.solana_dex import (
    NoUsablePoolError,
    PoolSelectionConfig,
    SolanaDexService,
    select_primary_pool,
)

from .conftest import DEX_NOW, FakeDexScreener
from .test_opportunity import full_buy
from .test_risk_agent import market

NOW = DEX_NOW
MINT = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
OTHER_MINT = "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
FAKE_QUOTE = "FakeQuote1111111111111111111111111111111111"
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
QUOTES = {SOL: ("SOL", "Wrapped SOL"), USDC: ("USDC", "USD Coin"), FAKE_QUOTE: ("FAKE", "Fake")}
Txns = dict[str, tuple[int, int]]


# --- Builders -------------------------------------------------------------------------------


def pair(
    address: str = "PoolSo1111111111111111111111111111111111111",
    *,
    mint: str = MINT,
    symbol: str = "NEWT",
    quote: str = SOL,
    dex: str = "raydium",
    chain: str = "solana",
    price: str | None = "0.0123",
    native: str | None = "0.0000812",
    liquidity: float | None = 250_000.0,
    fdv: float | None = 1_200_000.0,
    mcap: float | None = 1_000_000.0,
    age: timedelta | None = timedelta(days=2),
    txns: Txns | None = None,
    volume: dict[str, float] | None = None,
    change: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One pair object shaped like DEX Screener's /token-pairs response."""
    qsym, qname = QUOTES.get(quote, ("Q", "Quote"))
    row: dict[str, Any] = {
        "chainId": chain,
        "dexId": dex,
        "url": f"https://dexscreener.com/solana/{address.lower()}",
        "pairAddress": address,
        "labels": ["CLMM"],
        "baseToken": {"address": mint, "name": "New Token", "symbol": symbol},
        "quoteToken": {"address": quote, "name": qname, "symbol": qsym},
        "txns": {
            w: {"buys": b, "sells": s}
            for w, (b, s) in (
                txns or {"m5": (12, 8), "h1": (120, 80), "h6": (600, 500), "h24": (2000, 1500)}
            ).items()
        },
        "volume": volume or {"m5": 1_500.0, "h1": 18_000.0, "h6": 90_000.0, "h24": 350_000.0},
        "priceChange": change or {"m5": 1.2, "h1": -3.4, "h6": 8.9, "h24": 21.5},
    }
    if price is not None:
        row["priceUsd"] = price
    if native is not None:
        row["priceNative"] = native
    if liquidity is not None:
        row["liquidity"] = {"usd": liquidity, "base": 1_000_000.0, "quote": 800.0}
    if fdv is not None:
        row["fdv"] = fdv
    if mcap is not None:
        row["marketCap"] = mcap
    if age is not None:
        row["pairCreatedAt"] = int((NOW - age).timestamp() * 1000)
    return row


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def service(fake: FakeDexScreener, clock: Clock | None = None, **kw: Any) -> SolanaDexService:
    provider = DexScreenerProvider(transport=fake.transport())
    return SolanaDexService(provider, clock=clock or Clock(), now=lambda: NOW, **kw)


def snapshot(fake: FakeDexScreener, *pairs: dict[str, Any], mint: str = MINT) -> Any:
    fake.pairs[mint] = list(pairs)
    return asyncio.run(service(fake).get_snapshot(mint))


def pools(*rows: dict[str, Any]) -> list[Any]:
    parsed = [parse_pair(r) for r in rows]
    assert all(p is not None for p in parsed)
    return parsed


def dex_result(fake: FakeDexScreener, *pairs: dict[str, Any], mint: str = MINT) -> AgentResult:
    fake.pairs[mint] = list(pairs)
    agent = DexMarketAgent(service(fake))
    context = AgentContext(
        query="", assets=[mint], asset_identity=AssetIdentity(chain="solana", address=mint)
    )
    return asyncio.run(agent.run(context))


def identity(mint: str = MINT) -> AssetIdentity:
    return AssetIdentity(chain="solana", address=mint)


def results(*rs: AgentResult) -> dict[Any, AgentResult]:
    return {r.agent: r for r in rs}


def profile_for(prior: dict[Any, AgentResult], asset: str, mint: str = MINT, **kw: Any) -> Any:
    p = build_profile(identity(mint), observations(collect_inputs(prior, asset)), now=NOW, **kw)
    assert p is not None
    return p


def review(prior: dict[Any, AgentResult], asset: str = "NEWT", **kw: Any) -> Any:
    return risk_service.assess(prior, asset, profile=profile_for(prior, asset), **kw)


def factor(r: Any, fid: str) -> Any:
    return next((f for f in r.factors if f.id == fid), None)


# --- DEX Screener provider ------------------------------------------------------------------


def test_uses_the_official_token_pairs_endpoint_by_exact_mint(fake_dexscreener) -> None:
    snapshot(fake_dexscreener, pair())
    request = fake_dexscreener.requests[0]
    assert request.url.host == "api.dexscreener.com"
    assert request.url.path == f"/token-pairs/v1/solana/{MINT}"


def test_snapshot_normalizes_every_reported_field(fake_dexscreener) -> None:
    s = snapshot(fake_dexscreener, pair())
    assert s.canonical_id == f"solana:{MINT}" and s.mint == MINT
    assert (s.symbol, s.name) == ("NEWT", "New Token")
    assert s.provider == "DEX Screener" and s.dex == "raydium"
    assert s.pair_address.startswith("PoolSo")
    assert (s.quote_symbol, s.quote_address, s.quote_kind) == ("SOL", SOL, "SOL")
    assert s.price_usd == 0.0123 and s.price_native == 0.0000812
    assert s.liquidity_usd == 250_000.0
    assert s.market_cap_usd == 1_000_000.0 and s.fdv_usd == 1_200_000.0
    assert s.pair_created_at == NOW - timedelta(days=2)
    assert s.pool_age_hours == pytest.approx(48.0)
    assert s.fetched_at == NOW
    h24, m5 = s.window("h24"), s.window("m5")
    assert (h24.buys, h24.sells, h24.volume_usd, h24.price_change_pct) == (
        2000,
        1500,
        350_000.0,
        21.5,
    )
    assert (m5.buys, m5.sells, m5.volume_usd, m5.price_change_pct) == (12, 8, 1_500.0, 1.2)
    assert [w.window for w in s.windows] == ["m5", "h1", "h6", "h24"]
    assert s.primary_clear and len(s.candidates) == 1 and s.candidates[0].primary


def test_usdc_quoted_pool(fake_dexscreener) -> None:
    s = snapshot(fake_dexscreener, pair(quote=USDC, native="0.0123"))
    assert (s.quote_symbol, s.quote_kind) == ("USDC", "USDC")
    assert s.price_native == 0.0123


def test_missing_fields_stay_missing(fake_dexscreener) -> None:
    row = pair(fdv=None, mcap=None, age=None, native=None)
    del row["priceChange"]
    row["txns"] = {"h24": {"buys": 40, "sells": 30}}
    row["volume"] = {"h24": 5_000.0}
    s = snapshot(fake_dexscreener, row)
    assert s.market_cap_usd is None and s.fdv_usd is None
    assert s.pair_created_at is None and s.pool_age_hours is None
    assert s.price_native is None
    assert [w.window for w in s.windows] == ["h24"]
    assert s.window("h24").price_change_pct is None
    assert s.window("m5") is None


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx2.Response(200, content=b'{"pairs": []}'), "unexpected response"),
        (httpx2.Response(200, content=b"not json"), "invalid JSON"),
        (httpx2.Response(200, content=b'[{"chainId": 1}, "junk"]'), "malformed pair data"),
        (httpx2.Response(500), "HTTP 500"),
        (httpx2.Response(429), "rate limit"),
    ],
)
def test_malformed_or_failed_responses_are_errors(fake_dexscreener, response, message) -> None:
    fake_dexscreener.handler = lambda r: response
    with pytest.raises(MarketDataUnavailableError, match=message):
        asyncio.run(service(fake_dexscreener).get_snapshot(MINT))


def test_unparseable_values_become_none_not_guesses() -> None:
    row = pair(price="abc", liquidity=None)
    row["liquidity"] = {"usd": "NaN"}
    row["txns"]["h1"] = {"buys": -3, "sells": True}
    p = parse_pair(row)
    assert p is not None
    assert p.price_usd is None and p.liquidity_usd is None
    assert (p.window("h1").buys, p.window("h1").sells) == (None, None)


def test_unknown_mint_is_not_found_and_remembered(fake_dexscreener) -> None:
    svc = service(fake_dexscreener)
    for _ in range(2):
        with pytest.raises(AssetNotFoundError):
            asyncio.run(svc.get_snapshot(MINT))
    assert len(fake_dexscreener.requests) == 1


def test_invalid_mint_is_rejected_without_a_request(fake_dexscreener) -> None:
    with pytest.raises(InvalidRequestError):
        asyncio.run(service(fake_dexscreener).get_snapshot("not-a-mint"))
    assert fake_dexscreener.requests == []


# --- Pool selection -------------------------------------------------------------------------


def test_most_liquid_pool_is_primary_regardless_of_order() -> None:
    a = pair("PoolA111111111111111111111111111111111111111", liquidity=120_000.0)
    b = pair("PoolB111111111111111111111111111111111111111", liquidity=900_000.0, dex="orca")
    c = pair("PoolC111111111111111111111111111111111111111", liquidity=40_000.0, quote=USDC)
    for order in ([a, b, c], [c, b, a]):
        sel = select_primary_pool(pools(*order), MINT)
        assert sel.primary is not None and sel.primary.pair_address.startswith("PoolB")
        assert [c.pair_address[:5] for c in sel.candidates] == ["PoolB", "PoolA", "PoolC"]
        assert sel.clear


def test_equal_liquidity_is_broken_by_volume_then_address() -> None:
    a = pair("PoolA111111111111111111111111111111111111111", volume={"h24": 10.0})
    b = pair("PoolB111111111111111111111111111111111111111", volume={"h24": 99.0})
    assert select_primary_pool(pools(a, b), MINT).primary.pair_address.startswith("PoolB")
    c = pair("PoolC111111111111111111111111111111111111111", volume={"h24": 10.0})
    assert select_primary_pool(pools(c, a), MINT).primary.pair_address.startswith("PoolA")


def test_tiny_spoof_pool_never_becomes_the_market(fake_dexscreener) -> None:
    tiny = pair("Tiny1111111111111111111111111111111111111111", liquidity=300.0, price="5.0")
    real = pair("Real1111111111111111111111111111111111111111", liquidity=400_000.0)
    s = snapshot(fake_dexscreener, tiny, real)
    assert s.pair_address.startswith("Real") and s.price_usd == 0.0123
    rejected = next(c for c in s.candidates if c.pair_address.startswith("Tiny"))
    assert not rejected.eligible and "below $1,000" in rejected.rejected_because[0]


def test_unrecognized_quote_cannot_outrank_a_real_quote_pool() -> None:
    fake_q = pair("FakeQ111111111111111111111111111111111111111", quote=FAKE_QUOTE, liquidity=50e6)
    real = pair("Real1111111111111111111111111111111111111111", liquidity=400_000.0)
    sel = select_primary_pool(pools(fake_q, real), MINT)
    assert sel.primary.pair_address.startswith("Real")
    fq = next(c for c in sel.candidates if c.pair_address.startswith("FakeQ"))
    assert fq.eligible and "SOL, USDC or USDT" in fq.rejected_because[0]
    # With no recognized-quote pool, the other quote is used rather than nothing.
    assert select_primary_pool(pools(fake_q), MINT).primary is not None


def test_inactive_unpriced_and_foreign_pools_are_not_eligible() -> None:
    dead = pair("Dead1111111111111111111111111111111111111111", txns={"h24": (0, 0)})
    unpriced = pair("NoPx1111111111111111111111111111111111111111", price=None)
    other_chain = pair("Eth11111111111111111111111111111111111111111", chain="ethereum")
    quoted = pair("Quot1111111111111111111111111111111111111111", mint=OTHER_MINT)
    quoted["quoteToken"] = {"address": MINT, "symbol": "NEWT", "name": "New Token"}
    sel = select_primary_pool(pools(dead, unpriced, other_chain, quoted), MINT)
    assert sel.primary is None
    reasons = {c.pair_address[:4]: c.rejected_because for c in sel.candidates}
    assert set(reasons) == {"Dead", "NoPx"}  # foreign chain / mint-as-quote aren't candidates
    assert "inactive" in reasons["Dead"][0] and "no USD price" in reasons["NoPx"][0]


def test_no_usable_pool_keeps_the_candidates(fake_dexscreener) -> None:
    fake_dexscreener.pairs[MINT] = [pair(liquidity=200.0)]
    with pytest.raises(NoUsablePoolError) as exc:
        asyncio.run(service(fake_dexscreener).get_snapshot(MINT))
    assert len(exc.value.candidates) == 1 and not exc.value.candidates[0].eligible


def test_competing_pools_make_the_primary_market_unclear() -> None:
    a = pair("PoolA111111111111111111111111111111111111111", liquidity=400_000.0)
    b = pair("PoolB111111111111111111111111111111111111111", liquidity=300_000.0, quote=USDC)
    sel = select_primary_pool(pools(a, b), MINT)
    assert not sel.clear and "75%" in sel.ambiguity[0]


def test_diverging_pool_prices_make_the_primary_market_unclear() -> None:
    a = pair("PoolA111111111111111111111111111111111111111", liquidity=400_000.0)
    b = pair("PoolB111111111111111111111111111111111111111", liquidity=20_000.0, price="0.02")
    sel = select_primary_pool(pools(a, b), MINT)
    assert sel.primary.pair_address.startswith("PoolA")
    assert not sel.clear and "prices differ" in sel.ambiguity[0]


def test_selection_thresholds_are_configurable() -> None:
    p = pair(liquidity=5_000.0)
    assert select_primary_pool(pools(p), MINT).primary is not None
    strict = PoolSelectionConfig(min_liquidity_usd=10_000.0)
    assert select_primary_pool(pools(p), MINT, strict).primary is None


def test_same_ticker_pool_for_another_mint_is_never_used(fake_dexscreener) -> None:
    spoof = pair("Spoof111111111111111111111111111111111111111", mint=OTHER_MINT, liquidity=9e6)
    real = pair("Real1111111111111111111111111111111111111111", liquidity=100_000.0)
    s = snapshot(fake_dexscreener, spoof, real)
    assert s.pair_address.startswith("Real")
    assert all(not c.pair_address.startswith("Spoof") for c in s.candidates)


# --- Service: caching, dedup, rate limits ---------------------------------------------------


def test_snapshots_are_cached_until_the_ttl_expires(fake_dexscreener) -> None:
    clock = Clock()
    svc = service(fake_dexscreener, clock, cache_ttl=30)
    fake_dexscreener.pairs[MINT] = [pair()]
    asyncio.run(svc.get_snapshot(MINT))
    asyncio.run(svc.get_snapshot(MINT))
    assert len(fake_dexscreener.requests) == 1
    clock.t += 31
    asyncio.run(svc.get_snapshot(MINT))
    assert len(fake_dexscreener.requests) == 2


def test_concurrent_requests_are_deduplicated(fake_dexscreener) -> None:
    svc = service(fake_dexscreener)
    fake_dexscreener.pairs[MINT] = [pair()]

    async def many() -> list[Any]:
        return await asyncio.gather(*(svc.get_snapshot(MINT) for _ in range(5)))

    snaps = asyncio.run(many())
    assert len(fake_dexscreener.requests) == 1
    assert len({s.pair_address for s in snaps}) == 1


def test_failures_are_not_cached(fake_dexscreener) -> None:
    svc = service(fake_dexscreener)
    fake_dexscreener.handler = lambda r: httpx2.Response(429)
    with pytest.raises(MarketDataUnavailableError):
        asyncio.run(svc.get_snapshot(MINT))
    fake_dexscreener.handler = fake_dexscreener.token_pairs
    fake_dexscreener.pairs[MINT] = [pair()]
    assert asyncio.run(svc.get_snapshot(MINT)).mint == MINT  # retried, not cached
    assert len(fake_dexscreener.requests) == 2


def test_upscale_rate_limits_its_own_requests(fake_dexscreener) -> None:
    clock = Clock()
    svc = service(fake_dexscreener, clock, max_calls_per_minute=1)
    fake_dexscreener.pairs[MINT] = [pair()]
    fake_dexscreener.pairs[OTHER_MINT] = [pair(mint=OTHER_MINT)]
    asyncio.run(svc.get_snapshot(MINT))
    with pytest.raises(MarketDataUnavailableError, match="request limit"):
        asyncio.run(svc.get_snapshot(OTHER_MINT))
    assert asyncio.run(svc.get_snapshot(MINT)).mint == MINT  # cache still served
    clock.t += 61
    assert asyncio.run(svc.get_snapshot(OTHER_MINT)).mint == OTHER_MINT
    assert len(fake_dexscreener.requests) == 2


def test_two_mints_with_the_same_ticker_are_separate(fake_dexscreener) -> None:
    fake_dexscreener.pairs[MINT] = [pair(symbol="PEPE", price="0.5")]
    fake_dexscreener.pairs[OTHER_MINT] = [pair(mint=OTHER_MINT, symbol="PEPE", price="0.001")]
    svc = service(fake_dexscreener)
    a, b = asyncio.run(svc.get_snapshot(MINT)), asyncio.run(svc.get_snapshot(OTHER_MINT))
    assert a.symbol == b.symbol == "PEPE"
    assert a.canonical_id != b.canonical_id and a.price_usd != b.price_usd


# --- Agent ----------------------------------------------------------------------------------


def test_agent_reports_the_primary_pool(fake_dexscreener) -> None:
    r = dex_result(fake_dexscreener, pair())
    assert r.status == "ok" and not r.mock
    assert r.findings["snapshot"]["mint"] == MINT
    assert r.findings["canonical_id"] == f"solana:{MINT}"
    assert len(r.findings["candidates"]) == 1
    text = " ".join([r.summary, *r.evidence])
    assert "NEWT on raydium (SOL pool): $0.0123" in r.summary
    assert "2,000 buys / 1,500 sells" in text
    assert "not used as identity or size proof" in text
    assert "market data only" in text


def test_agent_without_a_mint_requests_nothing(fake_dexscreener) -> None:
    r = asyncio.run(
        DexMarketAgent(service(fake_dexscreener)).run(AgentContext(query="", assets=["BTC"]))
    )
    assert r.status == "ok" and r.findings["snapshot"] is None
    assert fake_dexscreener.requests == []


def test_agent_uses_the_registry_mint_for_a_registered_solana_token(fake_dexscreener) -> None:
    fake_dexscreener.pairs[BONK_MINT] = [pair(mint=BONK_MINT, symbol="Bonk")]
    agent = DexMarketAgent(service(fake_dexscreener))
    r = asyncio.run(agent.run(AgentContext(query="", assets=["BONK"])))
    assert r.findings["mint"] == BONK_MINT and r.findings["snapshot"] is not None


def test_agent_outage_is_an_isolated_error(fake_dexscreener) -> None:
    fake_dexscreener.handler = lambda r: httpx2.Response(503)
    r = dex_result(fake_dexscreener)
    assert r.status == "error" and "HTTP 503" in (r.error or "")


def test_agent_no_usable_pool_reports_candidates(fake_dexscreener) -> None:
    r = dex_result(fake_dexscreener, pair(liquidity=100.0))
    assert r.status == "ok" and r.findings["snapshot"] is None
    assert len(r.findings["candidates"]) == 1 and "rejected" in r.evidence[0]


# --- Routing and orchestration --------------------------------------------------------------


def test_mint_in_the_request_routes_to_dex_without_ticker_agents() -> None:
    d = route(f"Should I buy {MINT}?", has_images=False)
    assert d.token_address == MINT and d.assets == [MINT]
    assert d.agents == ["dex_market", "risk", "opportunity"]


def test_a_ticker_next_to_a_mint_does_not_trigger_ticker_lookups() -> None:
    d = route(f"is PEPE {MINT} a good entry?", has_images=False)
    assert "technical_analysis" not in d.agents and "market" not in d.agents
    assert d.assets == [MINT]


def test_evm_addresses_and_normal_text_are_not_mints() -> None:
    assert detect_solana_mint("0x6982508145454Ce325dDbE47a25d4ec3d2311933") is None
    assert detect_solana_mint("Should I buy BTC or ETH this week?") is None
    assert detect_solana_mint(f"mint: {MINT}.") == MINT


@pytest.mark.parametrize(
    ("query", "has_dex"),
    [
        ("Should I buy BTC?", False),
        ("Should I buy SOL?", False),
        ("Should I buy DOGE?", False),  # memecoin, but no Solana mint
        ("Should I buy PEPE?", True),  # established memecoin with an Ethereum contract
        ("Should I buy BONK?", True),  # established Solana memecoin with a mint
    ],
)
def test_dex_agent_is_selected_only_when_the_profile_calls_for_it(query, has_dex) -> None:
    d = with_profile_agents(route(query, has_images=False), None)
    assert ("dex_market" in d.agents) is has_dex


def ask(content: str) -> Any:
    request = ChatRequest(messages=[ChatMessage(role="user", content=content)])
    return asyncio.run(Orchestrator().respond(request))


def test_mint_query_end_to_end_stays_wait(fake_dexscreener) -> None:
    fake_dexscreener.pairs[MINT] = [pair()]
    response = ask(f"Should I buy {MINT}?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert list(by_agent) == ["dex_market", "technical_analysis", "risk", "opportunity"]
    assert response.analysis.assets == ["NEWT"]
    decision = by_agent["opportunity"].findings
    assert decision["action"] == "wait"
    assert decision["asset_profile"]["canonical_id"] == f"solana:{MINT}"
    assert decision["asset_profile"]["category"] == "new_dex_token"
    blockers = {f["id"] for f in decision["blocking_factors"]}
    assert "profile_evidence_missing" in blockers
    assert by_agent["risk"].findings["asset_profile"]["capabilities"][3] == {
        "capability": "dex",
        "status": "available",
        "verified": True,
        "reason": "DEX Screener reported 1 pool(s) for this mint.",
    }
    assert len(fake_dexscreener.requests) == 1


def test_dex_outage_does_not_break_the_response(fake_dexscreener) -> None:
    fake_dexscreener.handler = lambda r: httpx2.Response(503)
    response = ask(f"Should I buy {MINT}?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert by_agent["dex_market"].status == "error"
    assert by_agent["risk"].status == by_agent["opportunity"].status == "ok"
    assert by_agent["opportunity"].findings["action"] == "wait"


def test_btc_query_never_calls_dex_screener(fake_dexscreener) -> None:
    ask("Should I buy BTC?")
    assert fake_dexscreener.requests == []


# --- Profile integration --------------------------------------------------------------------


def test_dex_data_fills_the_profile(fake_dexscreener) -> None:
    prior = results(dex_result(fake_dexscreener, pair()))
    p = profile_for(prior, "NEWT")
    assert p.category == "new_dex_token"
    # The mint is the identity; the DEX-reported symbol is only its label.
    assert (p.symbol, p.name, p.canonical_id) == ("NEWT", "New Token", f"solana:{MINT}")
    dex = p.capability("dex")
    assert dex.status == "available" and dex.verified
    assert p.dex_available is True and p.market_type == "dex_spot"
    assert p.pool_age_days == 2
    assert p.liquidity_class == "low" and "DEX Screener primary pool" in p.liquidity_basis
    for e in ("dex_liquidity", "buy_sell_flow", "pool_age"):
        assert p.evidence_status(e).status == "available"
    # A DEX-reported market cap is context only: never a size tier or identity proof.
    assert p.market_cap_usd == 1_000_000.0
    assert "not used to classify" in p.market_cap_basis
    assert p.classification_cap_tier is None


def test_huge_dex_market_cap_does_not_make_a_token_major(fake_dexscreener) -> None:
    prior = results(dex_result(fake_dexscreener, pair(mcap=900e9, age=timedelta(days=400))))
    p = profile_for(prior, "NEWT")
    assert p.category == "unknown_crypto"
    assert "not a new DEX token" in p.category_reasons[0]


def test_old_liquid_dex_memecoin_is_refined_to_established(fake_dexscreener) -> None:
    registry = AssetRegistry(
        (AssetMetadata(symbol="OLDM", chain="solana", address=MINT, tags=frozenset({"meme"})),)
    )
    old = pair(symbol="OLDM", liquidity=3e6, age=timedelta(days=400))
    prior = results(dex_result(fake_dexscreener, old))
    p = profile_for(prior, "OLDM", registry=registry)
    assert p.category == "established_memecoin"
    assert "$3,000,000 DEX pool liquidity" in p.category_reasons[2]


def test_pool_age_uses_the_oldest_pool_for_token_age(fake_dexscreener) -> None:
    # The main pool is new (migrated), but the token has traded for 200 days elsewhere.
    new_main = pair(
        "Main1111111111111111111111111111111111111111", liquidity=2e6, age=timedelta(days=3)
    )
    old_pool = pair(
        "Old11111111111111111111111111111111111111111", liquidity=5_000.0, age=timedelta(days=200)
    )
    prior = results(dex_result(fake_dexscreener, new_main, old_pool))
    p = profile_for(prior, "NEWT")
    assert p.pool_age_days == 3
    assert p.category != "new_dex_token"


def test_dex_data_for_another_mint_is_ignored(fake_dexscreener) -> None:
    prior = results(dex_result(fake_dexscreener, pair(mint=OTHER_MINT), mint=OTHER_MINT))
    p = profile_for(prior, "NEWT", mint=MINT)
    assert p.capability("dex").verified is False
    r = risk_service.assess(prior, "NEWT", profile=p)
    assert not any(f.category == "dex" for f in r.factors)
    assert {i.agent: i.status for i in r.inputs}["dex_market"] == "no_data"


# --- Risk integration -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("liquidity", "severity"), [(20_000.0, "high"), (60_000.0, "medium"), (200_000.0, None)]
)
def test_low_liquidity(fake_dexscreener, liquidity, severity) -> None:
    f = factor(
        review(results(dex_result(fake_dexscreener, pair(liquidity=liquidity)))),
        "dex_low_liquidity",
    )
    assert (f.severity if f else None) == severity
    if f:
        assert "not rug-pull detection" in f.explanation


@pytest.mark.parametrize(
    ("age", "severity"),
    [(timedelta(hours=10), "high"), (timedelta(hours=50), "medium"), (timedelta(days=5), None)],
)
def test_very_new_pool(fake_dexscreener, age, severity) -> None:
    f = factor(review(results(dex_result(fake_dexscreener, pair(age=age)))), "dex_new_pool")
    assert (f.severity if f else None) == severity


@pytest.mark.parametrize(
    ("change", "severity"),
    [
        ({"m5": 35.0, "h1": 5.0}, "high"),
        ({"m5": -16.0, "h1": 5.0}, "medium"),
        ({"m5": 2.0, "h1": 55.0}, "high"),
        ({"m5": 2.0, "h1": 12.0}, None),  # above BTC's 10% "high" move, normal here
    ],
)
def test_extreme_short_window_moves(fake_dexscreener, change, severity) -> None:
    r = review(results(dex_result(fake_dexscreener, pair(change=change))))
    f = factor(r, "dex_extreme_move")
    assert (f.severity if f else None) == severity
    assert factor(r, "large_recent_move") is None  # BTC thresholds never apply to DEX data


@pytest.mark.parametrize(
    ("h1", "h24", "severity"),
    [
        ((95, 5), (1000, 900), "high"),
        ((17, 83), (1000, 900), "medium"),
        ((55, 45), (1000, 900), None),
        ((5, 0), (950, 50), "high"),  # too few 1h trades: the 24h window decides
    ],
)
def test_buy_sell_imbalance(fake_dexscreener, h1, h24, severity) -> None:
    txns = {"h1": h1, "h24": h24}
    f = factor(review(results(dex_result(fake_dexscreener, pair(txns=txns)))), "dex_flow_imbalance")
    assert (f.severity if f else None) == severity


def test_competing_pools_raise_uncertainty(fake_dexscreener) -> None:
    a = pair("PoolA111111111111111111111111111111111111111", liquidity=400_000.0)
    b = pair("PoolB111111111111111111111111111111111111111", liquidity=350_000.0, quote=USDC)
    f = factor(review(results(dex_result(fake_dexscreener, a, b))), "dex_competing_pools")
    assert f.affects == "uncertainty" and f.severity == "medium"


def test_missing_liquidity_everywhere(fake_dexscreener) -> None:
    # No usable pool means no symbol was learned: the asset is still the mint.
    r = review(results(dex_result(fake_dexscreener, pair(liquidity=None))), asset=MINT)
    f = factor(r, "dex_liquidity_missing")
    assert f.severity == "high" and f.affects == "uncertainty"


def test_only_tiny_pools_is_a_liquidity_risk(fake_dexscreener) -> None:
    r = review(results(dex_result(fake_dexscreener, pair(liquidity=400.0))), asset=MINT)
    assert factor(r, "dex_low_liquidity").severity == "high"


def test_calm_liquid_pool_adds_no_dex_risk(fake_dexscreener) -> None:
    calm = pair(liquidity=5e6, age=timedelta(days=20), change={"m5": 0.1, "h1": 1.0})
    r = review(results(dex_result(fake_dexscreener, calm)))
    assert not any(f.category == "dex" for f in r.factors)
    assert r.overall_risk == "low"
    assert r.uncertainty_level == "high"  # token safety evidence is still missing


def test_dex_risk_thresholds_are_configurable_and_validated(fake_dexscreener) -> None:
    prior = results(dex_result(fake_dexscreener, pair(liquidity=200_000.0)))
    loose = review(prior)
    strict = review(
        prior,
        dex_config=DexRiskConfig(liquidity_high_usd=300_000.0, liquidity_medium_usd=500_000.0),
    )
    assert factor(loose, "dex_low_liquidity") is None
    assert factor(strict, "dex_low_liquidity").severity == "high"
    with pytest.raises(ValueError):
        DexRiskConfig(liquidity_high_usd=10.0, liquidity_medium_usd=5.0)
    with pytest.raises(ValueError):
        DexRiskConfig(imbalance_medium_share=0.4)


def test_dangerous_dex_data_makes_overall_risk_high(fake_dexscreener) -> None:
    hot = pair(liquidity=10_000.0, age=timedelta(hours=3))
    assert review(results(dex_result(fake_dexscreener, hot))).overall_risk == "high"


# --- Opportunity ----------------------------------------------------------------------------


def _decide(prior: dict[Any, AgentResult], asset: str = "NEWT") -> Any:
    p = profile_for(prior, asset)
    r = risk_service.assess(prior, asset, profile=p)
    prior = prior | {
        "risk": AgentResult(
            agent="risk", mock=False, summary="r", findings=r.model_dump(mode="json")
        )
    }
    return opportunity_service.assess(prior, asset, profile=p)


def test_new_token_with_good_dex_data_still_waits_for_onchain_safety(fake_dexscreener) -> None:
    calm = pair(liquidity=5e6, age=timedelta(days=20), change={"m5": 0.1, "h1": 1.0})
    a = _decide(results(dex_result(fake_dexscreener, calm)))
    assert a.action == "wait"
    reasons = " ".join(f.reason for f in a.blocking_factors)
    assert "token mint/freeze authorities" in reasons and "holder concentration" in reasons
    assert "DEX pool liquidity" not in reasons  # now available, so no longer a blocker
    assert a.live_price == 0.0123
    assert any("DEX price $0.0123" in c for c in a.context)


def test_new_token_waits_even_with_a_bullish_ticker_setup(fake_dexscreener) -> None:
    # Ticker-keyed candles for "NEWT" can't be attributed to the mint and never unblock BUY.
    prior = results(dex_result(fake_dexscreener, pair()), *full_buy(symbol="NEWT")[:1])
    a = _decide(prior)
    assert a.action == "wait" and a.bullish_evidence == []


def test_dex_risks_show_up_as_blockers_and_cautions(fake_dexscreener) -> None:
    risky = pair(liquidity=15_000.0, age=timedelta(hours=50))
    a = _decide(results(dex_result(fake_dexscreener, risky)))
    blockers = {f.id for f in a.blocking_factors}
    cautions = {f.id for f in a.cautions}
    assert "dex_low_liquidity" in blockers  # high severity
    assert "dex_new_pool" in cautions  # medium severity
    assert any(f.reason.startswith("DEX market:") for f in a.blocking_factors)


# --- Existing BTC / SOL behavior -------------------------------------------------------------


@pytest.mark.parametrize("symbol", ["BTC", "SOL"])
def test_unrelated_dex_results_leave_btc_and_sol_unchanged(fake_dexscreener, symbol) -> None:
    base = {r.agent: r for r in full_buy(symbol=symbol)}
    base["market"] = market(symbol=symbol, price=84_445.0)
    with_dex = base | results(dex_result(fake_dexscreener, pair()))
    for prior in (base, with_dex):
        p = risk_service.profile_from_results(prior, symbol)
        assert p is not None and p.category in ("major_crypto", "large_cap_alt")
    before = opportunity_service.assess(
        base, symbol, profile=risk_service.profile_from_results(base, symbol)
    )
    after = opportunity_service.assess(
        with_dex, symbol, profile=risk_service.profile_from_results(with_dex, symbol)
    )
    assert after.model_dump(exclude={"asset_profile", "inputs"}) == before.model_dump(
        exclude={"asset_profile", "inputs"}
    )
    assert before.action == "buy"
    r_before = risk_service.assess(base, symbol)
    r_after = risk_service.assess(with_dex, symbol)
    assert r_after.factors == r_before.factors and r_after.overall_risk == r_before.overall_risk


def test_btc_route_is_unchanged() -> None:
    assert with_profile_agents(route("What about BTC?", has_images=False), None).agents == [
        "technical_analysis",
        "market",
        "news_sentiment",
        "risk",
        "opportunity",
    ]
