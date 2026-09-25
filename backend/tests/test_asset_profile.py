"""Crypto asset profiles: identity, category, capabilities, evidence, and how Risk and
Opportunity use them. All offline: no network, no model calls."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from upscale.agents import AgentContext, OpportunityAgent, RiskAgent
from upscale.routing import AMBIGUOUS_TICKERS, ASSET_ALIASES, route
from upscale.schemas import AgentResult
from upscale.services import opportunity as opportunity_service
from upscale.services import risk as risk_service
from upscale.services.asset_profile import (
    CATEGORY_PROFILES,
    MIN_TECHNICAL_CANDLES,
    AssetIdentity,
    CategoryProfile,
    CryptoAssetProfile,
    EvidenceRule,
    build_profile,
    planned_agents,
)
from upscale.services.asset_registry import (
    DEFAULT_REGISTRY,
    KNOWN_ASSETS,
    AssetMetadata,
    AssetRegistry,
)
from upscale.services.coingecko import COINGECKO_IDS
from upscale.services.risk import (
    RiskConfig,
    collect_inputs,
    observations,
    profile_from_results,
    risk_config_for,
)

from .test_opportunity import (
    bear_ta,
    bullish_news,
    full_buy,
    full_sell,
    risk,
    ta,
)
from .test_risk_agent import market, news, story, technical

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
MINT_A = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
MINT_B = "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"
PEPE_ETH = "0x6982508145454Ce325dDbE47a25d4ec3d2311933"


def profile(identity: AssetIdentity | str, **kwargs: Any) -> CryptoAssetProfile:
    p = build_profile(identity, now=NOW, **kwargs)
    assert p is not None
    return p


def new_solana_token(
    symbol: str = "NEWT", mint: str = MINT_A, *, candles: int | None = 30, **meta: Any
) -> AssetIdentity:
    """A four-day-old Solana memecoin traded only in a DEX pool."""
    fields: dict[str, Any] = {
        "symbol": symbol,
        "chain": "solana",
        "tags": frozenset({"meme"}),
        "dex_listed": True,
        "cex_listed": False,
        "pool_created_at": NOW - timedelta(days=4),
        "liquidity_usd": 300_000.0,
        "candle_history": candles,
        "source": "test DEX provider",
    }
    return AssetIdentity(
        symbol=symbol,
        chain="solana",
        address=mint,
        metadata=AssetMetadata(**(fields | meta)),
    )


def prior(*results: AgentResult) -> dict[Any, AgentResult]:
    return {r.agent: r for r in results}


def status(p: CryptoAssetProfile, evidence: str) -> str:
    e = p.evidence_status(evidence)  # type: ignore[arg-type]
    assert e is not None
    return e.status


def step(p: CryptoAssetProfile, evidence: str) -> Any:
    return next(s for s in p.analysis_plan if s.evidence == evidence)


# --- Categories ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "category"),
    [
        ("BTC", "major_crypto"),
        ("ETH", "major_crypto"),
        ("SOL", "large_cap_alt"),
        ("XRP", "large_cap_alt"),
        ("AVAX", "large_cap_alt"),  # mid cap, CEX-listed, years of history
        ("DOGE", "established_memecoin"),
        ("PEPE", "established_memecoin"),
        ("USDT", "stablecoin"),
        ("USDC", "stablecoin"),
    ],
)
def test_registry_assets_are_classified_by_characteristics(symbol: str, category: str) -> None:
    p = profile(symbol)
    assert p.category == category
    assert p.category_reasons
    assert not p.identity_ambiguous and p.ticker_data_attributable


def test_btc_profile() -> None:
    p = profile("BTC")
    assert p.canonical_id == "coingecko:bitcoin"
    assert (p.symbol, p.name, p.chain, p.address) == ("BTC", "Bitcoin", "bitcoin", None)
    assert p.classification_cap_tier == "mega" and p.market_type == "cex_spot"
    # The registry tier is static metadata: no live market cap is claimed without a provider.
    assert p.market_cap_usd is None and p.market_cap_class is None
    assert "not live market data" in (p.classification_cap_tier_basis or "")
    assert p.cex_available is True and p.dex_available is None
    assert p.asset_age_days is not None and p.asset_age_days > 6000
    assert p.risk_thresholds_calibrated
    assert p.required_evidence == ["technical_structure", "market_snapshot", "news"]
    assert p.planned_evidence == ["derivatives"]
    assert p.evidence_weights["technical_structure"] == "primary"


def test_sol_profile_is_a_large_cap_alt_with_ecosystem_evidence() -> None:
    p = profile("sol")
    assert p.canonical_id == "coingecko:solana"
    assert p.category == "large_cap_alt"
    assert {"ecosystem", "onchain_activity"} <= set(p.optional_evidence)
    assert status(p, "ecosystem") == "unavailable"  # no on-chain provider yet
    assert p.risk_thresholds_calibrated


def test_established_memecoin_from_characteristics_not_ticker() -> None:
    # Not in the registry: classified from provider facts alone.
    identity = AssetIdentity(
        symbol="OLDMEME",
        chain="solana",
        address=MINT_B,
        metadata=AssetMetadata(
            symbol="OLDMEME",
            chain="solana",
            tags=frozenset({"meme"}),
            launched=(NOW - timedelta(days=500)).date(),
            market_cap_usd=1.5e9,
            dex_listed=True,
            source="test provider",
        ),
    )
    p = profile(identity)
    assert p.category == "established_memecoin"
    assert p.market_cap_class == "mid"
    assert {"dex_liquidity", "holder_concentration", "social"} <= set(p.optional_evidence)
    assert not p.risk_thresholds_calibrated
    assert any("BTC" in c for c in p.risk_characteristics)


def test_young_or_small_memecoin_is_not_established() -> None:
    young = AssetMetadata(
        symbol="YNG",
        chain="solana",
        tags=frozenset({"meme"}),
        launched=(NOW - timedelta(days=100)).date(),
        cex_listed=True,
    )
    p = profile(AssetIdentity(symbol="YNG", chain="solana", address=MINT_B, metadata=young))
    assert p.category == "unknown_crypto"
    assert "180 days" in p.category_reasons[0]


def test_new_solana_dex_token() -> None:
    p = profile(new_solana_token())
    assert p.category == "new_dex_token"
    assert p.canonical_id == f"solana:{MINT_A}"
    assert p.identity_basis == "contract"
    assert p.market_type == "dex_spot"
    assert p.pool_age_days == 4
    assert p.liquidity_class == "low"
    assert "300,000" in (p.liquidity_basis or "")
    assert p.required_evidence == [
        "dex_liquidity",
        "token_authorities",
        "holder_concentration",
        "buy_sell_flow",
        "pool_age",
    ]
    assert set(p.decision_critical_evidence) == {
        "dex_liquidity",
        "token_authorities",
        "holder_concentration",
    }
    assert p.evidence_weights["dex_liquidity"] == "primary"
    assert p.evidence_weights["technical_structure"] == "context"


def test_new_dex_token_of_unknown_age_is_still_treated_as_new() -> None:
    p = profile(new_solana_token(pool_created_at=None))
    assert p.category == "new_dex_token"
    assert "age unknown" in p.category_reasons[1]


def test_stablecoin_does_not_use_technical_indicators_as_a_decision_source() -> None:
    p = profile("USDT")
    assert p.category == "stablecoin"
    assert set(p.decision_critical_evidence) == {"peg_stability", "issuer_risk"}
    assert p.evidence_weights["technical_structure"] == "not_used"
    assert step(p, "technical_structure").status == "skip"
    assert "technical_analysis" not in planned_agents(p)
    # CoinGecko prices exist, but no peg rule has been written yet: said so, not faked.
    assert status(p, "peg_stability") == "not_analyzed"
    assert status(p, "issuer_risk") == "unavailable"


def test_stablecoin_from_metadata_peg() -> None:
    meta = AssetMetadata(symbol="XUSD", chain="solana", peg="USD", dex_listed=True)
    p = profile(AssetIdentity(symbol="XUSD", chain="solana", address=MINT_B, metadata=meta))
    assert p.category == "stablecoin"


def test_unknown_ticker_is_ambiguous_unknown_crypto() -> None:
    p = profile("XYZT")
    assert p.category == "unknown_crypto"
    assert p.canonical_id == "symbol:XYZT"
    assert p.identity_basis == "symbol_only"
    assert p.identity_ambiguous
    assert p.capability("kraken_ohlcv").status == "unknown"
    assert p.capability("coingecko_snapshot").status == "unknown"


def test_unknown_ticker_stays_unknown_even_with_a_large_live_market_cap() -> None:
    snap = market(symbol="XYZT", price=2.0).findings["snapshots"][0]
    snap["market_cap_usd"] = 500e9
    obs = observations(collect_inputs(prior(_market(snap)), "XYZT"))
    p = profile("XYZT", observed=obs)
    assert p.category == "unknown_crypto"
    assert p.market_cap_class == "mega"
    assert "ticker-only" in (p.market_cap_basis or "")


def _market(snapshot: dict[str, Any]) -> AgentResult:
    return AgentResult(
        agent="market",
        mock=False,
        summary="m",
        findings={"provider": "CoinGecko", "snapshots": [snapshot], "unavailable": []},
    )


# --- Identity --------------------------------------------------------------------------------


def test_duplicate_tickers_never_share_an_identity() -> None:
    registry_pepe = profile("PEPE")
    same_by_contract = profile(AssetIdentity(chain="ethereum", address=PEPE_ETH.upper()))
    solana_a = profile(new_solana_token("PEPE", MINT_A))
    solana_b = profile(new_solana_token("PEPE", MINT_B))

    # The registry's PEPE is the same asset by ticker or by (case-insensitive) contract.
    assert registry_pepe.canonical_id == same_by_contract.canonical_id == "coingecko:pepe"
    ids = {registry_pepe.canonical_id, solana_a.canonical_id, solana_b.canonical_id}
    assert len(ids) == 3
    # Ticker-keyed data (Kraken, CoinGecko, news) can't be attributed to the Solana tokens.
    for p in (solana_a, solana_b):
        assert not p.ticker_data_attributable
        for cap in ("kraken_ohlcv", "coingecko_snapshot", "news"):
            assert p.capability(cap).status == "unavailable"  # type: ignore[arg-type]
            assert "ticker" in p.capability(cap).reason  # type: ignore[arg-type]


def test_same_ticker_contract_token_does_not_receive_the_registry_assets_evidence() -> None:
    results = prior(
        ta(symbol="PEPE"), market(symbol="PEPE"), news([story("bullish", "high")], "PEPE")
    )
    identity = new_solana_token("PEPE", MINT_A)

    p = risk_service.profile_from_results(results, "PEPE", identity)
    assert p is not None and p.canonical_id == f"solana:{MINT_A}"
    review = risk_service.assess(results, "PEPE", profile=p)
    states = {i.agent: i.status for i in review.inputs}
    assert states["technical_analysis"] == states["market"] == states["news_sentiment"] == "no_data"
    assert review.overall_risk == "unknown"
    assert not any(f.category in ("technical", "market", "news") for f in review.factors)

    decision = opportunity_service.assess(results, "PEPE", profile=p)
    assert decision.action == "wait"
    assert decision.bullish_evidence == decision.bearish_evidence == []
    assert decision.last_close is None and decision.live_price is None


def test_registry_ticker_matches_only_unique_symbols() -> None:
    a = AssetMetadata(symbol="DUP", chain="solana", address=MINT_A)
    b = AssetMetadata(symbol="DUP", chain="ethereum", address="0xabc")
    registry = AssetRegistry((a, b))
    assert registry.by_symbol("DUP") is None
    assert registry.by_address("solana", MINT_A) == a
    p = profile("DUP", registry=registry)
    assert p.identity_ambiguous and p.category == "unknown_crypto"


def test_cex_pair_identity() -> None:
    assert profile(AssetIdentity(kraken_pair="XBTUSD")).canonical_id == "coingecko:bitcoin"
    other = profile(AssetIdentity(kraken_pair="ZZZUSD", symbol="ZZZ"))
    assert other.canonical_id == "kraken:ZZZUSD"
    assert other.identity_basis == "cex_pair" and not other.identity_ambiguous


def test_registry_is_consistent_with_routing_and_providers() -> None:
    symbols = [e.symbol for e in KNOWN_ASSETS]
    assert len(symbols) == len(set(symbols))
    routable = set(ASSET_ALIASES.values()) | AMBIGUOUS_TICKERS
    assert routable <= set(symbols)
    for e in KNOWN_ASSETS:
        if e.symbol in COINGECKO_IDS:
            assert e.coingecko_id == COINGECKO_IDS[e.symbol]
        assert DEFAULT_REGISTRY.by_symbol(e.symbol) == e


# --- Missing metadata ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identity",
    [None, AssetIdentity(), AssetIdentity(address=MINT_A)],  # an address needs its chain
)
def test_no_usable_identity_gives_no_profile(identity: AssetIdentity | None) -> None:
    assert build_profile(identity, now=NOW) is None


def test_bare_contract_with_no_metadata_is_unknown_and_lists_what_is_missing() -> None:
    p = profile(AssetIdentity(chain="solana", address=MINT_B))
    assert p.category == "unknown_crypto"
    assert p.symbol == "?"
    assert {"name", "market cap", "launch / pool creation date"} <= set(p.missing_metadata)
    assert p.market_type == "unknown"
    assert p.classification_cap_tier is p.market_cap_class is None
    assert p.volatility_class is p.liquidity_class is None
    assert "unknown: market cap, age" in p.category_reasons[0]


def test_volatility_and_liquidity_only_from_evidence() -> None:
    assert profile("BTC").volatility_class is None  # no live data: not guessed
    obs = observations(collect_inputs(prior(market(price=100, high=112, low=100)), "BTC"))
    p = profile("BTC", observed=obs)
    assert p.volatility_class == "high"  # 12% range
    assert p.liquidity_class == "deep"  # $1B 24h volume
    assert "one day only" in (p.volatility_basis or "")


# --- Capabilities ----------------------------------------------------------------------------


def test_capabilities_are_explicit_for_every_asset() -> None:
    p = profile("BTC")
    caps = {c.capability: c for c in p.capabilities}
    assert set(caps) == {
        "kraken_ohlcv",
        "coingecko_snapshot",
        "news",
        "dex",
        "onchain",
        "social",
        "derivatives",
    }
    assert caps["kraken_ohlcv"].status == "available" and not caps["kraken_ohlcv"].verified
    assert caps["coingecko_snapshot"].status == "available"
    assert caps["news"].status == "available"
    for missing in ("onchain", "social", "derivatives"):
        assert caps[missing].status == "unavailable"  # type: ignore[index]
        assert "No provider" in caps[missing].reason  # type: ignore[index]
    # DEX data is integrated, but only for Solana tokens identified by mint.
    assert caps["dex"].status == "unavailable"
    assert "Solana mint" in caps["dex"].reason


def test_capabilities_are_verified_by_this_turns_results() -> None:
    results = prior(ta(), market(), news([story("bullish", "low")]))
    p = profile_from_results(results, "BTC")
    assert p is not None
    assert p.capability("kraken_ohlcv").verified
    assert p.capability("coingecko_snapshot").verified
    assert p.capability("news").verified
    assert status(p, "technical_structure") == "available"


def test_failed_agent_makes_its_capability_unavailable() -> None:
    failed = AgentResult(
        agent="technical_analysis", status="error", mock=False, summary="x", error="Kraken down"
    )
    p = profile_from_results(prior(failed), "BTC")
    assert p is not None
    assert p.capability("kraken_ohlcv").status == "unavailable"
    assert "Kraken down" in p.capability("kraken_ohlcv").reason


def test_unconfirmed_kraken_listing_is_unknown_not_assumed() -> None:
    assert profile("BNB").capability("kraken_ohlcv").status == "unknown"


def test_integrating_a_provider_changes_capabilities_and_plan() -> None:
    without = profile(
        new_solana_token(), integrated=frozenset({"kraken_ohlcv", "coingecko_snapshot", "news"})
    )
    default = profile(new_solana_token())
    with_onchain = profile(
        new_solana_token(),
        integrated=frozenset({"kraken_ohlcv", "coingecko_snapshot", "news", "dex", "onchain"}),
    )
    assert without.capability("dex").status == "unavailable"
    assert status(without, "dex_liquidity") == "unavailable"
    assert default.capability("dex").status == "available"  # looked up by mint
    assert status(default, "dex_liquidity") == "expected"
    assert with_onchain.capability("onchain").status == "available"
    # Data exists, but no agent analyzes it yet: reported, not faked.
    assert status(with_onchain, "token_authorities") == "not_analyzed"


# --- Required evidence and plan --------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "required"),
    [
        ("BTC", {"technical_structure", "market_snapshot", "news"}),
        ("SOL", {"technical_structure", "market_snapshot", "news"}),
        ("DOGE", {"technical_structure", "market_snapshot"}),
        ("USDT", {"peg_stability", "issuer_risk", "market_snapshot"}),
        ("XYZT", {"technical_structure", "market_snapshot"}),
    ],
)
def test_required_evidence_per_category(symbol: str, required: set[str]) -> None:
    assert set(profile(symbol).required_evidence) == required


def test_every_category_has_a_consistent_profile() -> None:
    for category, spec in CATEGORY_PROFILES.items():
        assert spec.category == category
        kinds = [r.evidence for r in spec.evidence]
        assert len(kinds) == len(set(kinds))
        assert any(r.decision_critical for r in spec.evidence)
        assert all(r.requirement == "required" for r in spec.evidence if r.decision_critical)
        assert spec.risk_characteristics


@pytest.mark.parametrize("symbol", ["BTC", "ETH", "SOL"])
def test_major_and_large_cap_plans_match_the_current_pipeline(symbol: str) -> None:
    current = route(f"What about {symbol}?", has_images=False).agents
    assert planned_agents(profile(symbol)) == current


def test_new_dex_token_plan_prioritizes_dex_and_safety_evidence() -> None:
    p = profile(new_solana_token())
    assert [s.evidence for s in p.analysis_plan[:5]] == p.required_evidence
    assert step(p, "dex_liquidity").agent == "dex_market"
    assert {step(p, e).status for e in ("dex_liquidity", "buy_sell_flow", "pool_age")} == {"run"}
    assert step(p, "token_authorities").status == "unavailable"
    assert step(p, "holder_concentration").status == "unavailable"
    assert step(p, "technical_structure").status == "skip"
    assert planned_agents(p) == ["dex_market", "risk", "opportunity"]


def _registered_new_token() -> AssetRegistry:
    """A new DEX token that is in the registry, so its ticker data is attributable."""
    token = AssetMetadata(
        symbol="NEWT",
        chain="solana",
        address=MINT_A,
        tags=frozenset({"meme"}),
        dex_listed=True,
        cex_listed=False,
        pool_created_at=NOW - timedelta(days=4),
    )
    return AssetRegistry((token,))


def test_technical_is_conditional_when_history_is_unknown() -> None:
    p = profile("NEWT", registry=_registered_new_token())
    assert p.category == "new_dex_token"
    assert step(p, "technical_structure").status == "conditional"
    assert str(MIN_TECHNICAL_CANDLES) in step(p, "technical_structure").reason


def test_new_dex_token_without_enough_history_does_not_fabricate_technical_evidence() -> None:
    # Registered so its ticker data is attributable; only 30 candles exist.
    short = ta(symbol="NEWT", close=1.0, support=((0.9, 0.95),), resistance=((1.2, 1.3),))
    short.findings["candle_count"] = 30
    results = prior(short, risk(asset="NEWT"))
    obs = observations(collect_inputs(results, "NEWT"))
    p = profile("NEWT", observed=obs, registry=_registered_new_token())

    assert p.category == "new_dex_token"
    assert p.ticker_data_attributable
    assert status(p, "technical_structure") == "insufficient"
    assert not p.technical_usable

    review = risk_service.assess(results, "NEWT", profile=p)
    assert not any(f.category == "technical" for f in review.factors)
    assert {i.agent: i.status for i in review.inputs}["technical_analysis"] == "no_data"

    decision = opportunity_service.assess(results, "NEWT", profile=p)
    assert decision.action == "wait"
    assert decision.timeframe is None and decision.last_close is None
    assert decision.bullish_evidence == decision.bearish_evidence == []
    assert decision.bullish_trigger is None and decision.bearish_trigger is None
    ids = {f.id for f in decision.blocking_factors}
    assert "profile_evidence_missing" in ids
    assert "no_technical" not in ids  # technical isn't decision-critical for this category


# --- Risk integration ------------------------------------------------------------------------


def test_risk_flags_missing_essential_evidence_for_a_new_dex_token() -> None:
    p = profile(new_solana_token())
    review = risk_service.assess({}, "NEWT", profile=p)
    factor = next(f for f in review.factors if f.id == "profile_evidence_unavailable")
    assert factor.severity == "high" and factor.affects == "uncertainty"
    assert review.uncertainty_level == "high"
    assert review.overall_risk == "unknown"  # missing evidence is not high risk
    assert any("mint/freeze authorities" in m for m in review.missing_evidence)
    assert any("holder concentration" in m.lower() for m in review.missing_evidence)
    assert review.profile_notes  # thresholds not calibrated for this kind of asset


def test_risk_notes_uncalibrated_thresholds_for_memecoins_without_changing_severity() -> None:
    results = prior(market(symbol="DOGE", change=6.0), technical(symbol="DOGE"))
    plain = risk_service.assess(results, "DOGE")
    profiled = risk_service.assess(results, "DOGE", profile=profile("DOGE"))
    assert [f.model_dump() for f in plain.factors] == [f.model_dump() for f in profiled.factors]
    assert profiled.overall_risk == plain.overall_risk
    assert "established memecoin" in profiled.profile_notes[0]


def test_risk_overrides_come_from_explicit_category_config() -> None:
    assert risk_config_for(profile("BTC")) == RiskConfig()
    custom = dict(CATEGORY_PROFILES)
    custom["established_memecoin"] = CategoryProfile(
        category="established_memecoin",
        description="test",
        evidence=(EvidenceRule("technical_structure", "required", "primary", True),),
        risk_characteristics=("test",),
        risk_thresholds_calibrated=True,
        risk_overrides={"move_medium_pct": 12.0, "move_high_pct": 25.0},
    )
    doge = profile("DOGE", categories=custom)
    cfg = risk_config_for(doge)
    assert (cfg.move_medium_pct, cfg.move_high_pct) == (12.0, 25.0)
    review = risk_service.assess(prior(market(symbol="DOGE", change=6.0)), "DOGE", profile=doge)
    assert not any(f.id == "large_recent_move" for f in review.factors)


def test_invalid_risk_override_fails_loudly() -> None:
    p = profile("BTC").model_copy(update={"risk_overrides": {"not_a_threshold": 1.0}})
    with pytest.raises(TypeError):
        risk_config_for(p)


# --- Current BTC / SOL behavior is unchanged -------------------------------------------------

_SCENARIOS = {
    "buy": lambda s: full_buy(symbol=s),
    "sell": lambda s: full_sell(symbol=s),
    "mixed": lambda s: [ta(symbol=s, trend="mixed"), market(symbol=s), risk(asset=s)],
    "no_news": lambda s: [ta(symbol=s), market(symbol=s), risk(asset=s)],
    "high_risk": lambda s: [ta(symbol=s), market(symbol=s), bullish_news(), risk("high", asset=s)],
    "no_risk": lambda s: [bear_ta(symbol=s), market(symbol=s)],
    "nothing": lambda s: [],
}


def _fix_symbols(results: list[AgentResult], symbol: str) -> list[AgentResult]:
    out = []
    for r in results:
        if r.agent == "market":
            r = market(symbol=symbol, price=r.findings["snapshots"][0]["price_usd"])
        if r.agent == "news_sentiment":
            r = news([story("bullish", "medium"), story("bullish", "low")], symbol)
        if r.agent == "risk":
            r = r.model_copy(update={"findings": r.findings | {"asset": symbol}})
        out.append(r)
    return out


@pytest.mark.parametrize("symbol", ["BTC", "SOL"])
@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_opportunity_decisions_are_unchanged_by_the_profile(symbol: str, scenario: str) -> None:
    results = prior(*_fix_symbols(_SCENARIOS[scenario](symbol), symbol))
    before = opportunity_service.assess(results, symbol)
    p = profile_from_results(results, symbol)
    assert p is not None and p.category in ("major_crypto", "large_cap_alt")
    after = opportunity_service.assess(results, symbol, profile=p)
    assert after.asset_profile == p
    assert after.model_dump(exclude={"asset_profile"}) == before.model_dump(
        exclude={"asset_profile"}
    )


@pytest.mark.parametrize("symbol", ["BTC", "SOL"])
@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_risk_reviews_are_unchanged_by_the_profile(symbol: str, scenario: str) -> None:
    results = prior(
        *[r for r in _fix_symbols(_SCENARIOS[scenario](symbol), symbol) if r.agent != "risk"]
    )
    before = risk_service.assess(results, symbol)
    after = risk_service.assess(results, symbol, profile=profile_from_results(results, symbol))
    assert after.profile_notes == []
    assert after.model_dump(exclude={"asset_profile"}) == before.model_dump(
        exclude={"asset_profile"}
    )


@pytest.mark.parametrize("symbol", ["BTC", "SOL"])
def test_agents_render_the_same_text_with_the_profile(symbol: str) -> None:
    results = prior(*_fix_symbols(full_buy(symbol=symbol), symbol))
    context = AgentContext(query="", assets=[symbol], prior_results=results)
    decision = asyncio.run(OpportunityAgent().run(context))
    expected = opportunity_service.assess(results, symbol)
    assert decision.findings["action"] == expected.action == "buy"
    assert decision.findings["confidence"] == expected.confidence
    assert decision.findings["asset_profile"]["canonical_id"].startswith("coingecko:")
    review = asyncio.run(RiskAgent().run(context))
    assert not any(line.startswith("Asset profile:") for line in review.evidence)
    assert review.findings["asset_profile"]["category"] in ("major_crypto", "large_cap_alt")


# --- Opportunity integration -----------------------------------------------------------------


def test_stablecoin_never_gets_buy_or_sell_from_indicators() -> None:
    results = prior(
        *_fix_symbols(
            full_buy(
                symbol="USDT",
                close=1.0,
                prev=0.999,
                support=((0.9985, 0.999),),
                resistance=((1.01, 1.02),),
                atr=0.0001,
            ),
            "USDT",
        )
    )
    plain = opportunity_service.assess(results, "USDT")
    p = profile_from_results(results, "USDT")
    decision = opportunity_service.assess(results, "USDT", profile=p)
    assert plain.action == "buy"  # what the generic rules would have said
    assert decision.action == "wait"
    reasons = " ".join(f.reason for f in decision.blocking_factors)
    assert "peg stability" in reasons and "issuer" in reasons


def test_ticker_only_unknown_asset_lowers_confidence() -> None:
    results = prior(*_fix_symbols(full_buy(symbol="XYZT"), "XYZT"))
    p = profile_from_results(results, "XYZT")
    decision = opportunity_service.assess(results, "XYZT", profile=p)
    caution = next(f for f in decision.cautions if f.id == "asset_profile_unknown")
    assert "ticker only" in caution.reason


def test_agent_uses_an_explicit_identity_from_the_context() -> None:
    results = prior(ta(symbol="PEPE"), market(symbol="PEPE"))
    context = AgentContext(
        query="",
        assets=["PEPE"],
        prior_results=results,
        asset_identity=new_solana_token("PEPE", MINT_A),
    )
    decision = asyncio.run(OpportunityAgent().run(context))
    assert decision.findings["asset_profile"]["canonical_id"] == f"solana:{MINT_A}"
    assert decision.findings["action"] == "wait"
    assert decision.findings["bullish_evidence"] == []


def test_registry_holds_static_classification_metadata_only() -> None:
    for e in KNOWN_ASSETS:
        assert e.market_cap_usd is e.liquidity_usd is e.candle_history is None
    live = AssetMetadata(symbol="X", classification_cap_tier="mid", market_cap_usd=1e9)
    with pytest.raises(ValueError, match="static classification metadata"):
        AssetRegistry((live,))


def test_live_market_cap_comes_only_from_a_provider_and_never_replaces_the_static_tier() -> None:
    snap = market(symbol="SOL", price=100.0).findings["snapshots"][0]
    snap["market_cap_usd"] = 900e9  # a live value above the mega band
    obs = observations(collect_inputs(prior(_market(snap)), "SOL"))
    p = profile("SOL", observed=obs)
    assert p.market_cap_usd == 900e9 and p.market_cap_class == "mega"
    assert "live CoinGecko" in (p.market_cap_basis or "")
    assert p.classification_cap_tier == "large"  # static tier keeps SOL's category stable
    assert p.category == "large_cap_alt"
