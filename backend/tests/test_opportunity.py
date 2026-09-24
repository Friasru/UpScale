"""Opportunity agent: deterministic BUY / SELL / WAIT from the other agents' evidence.

Technical inputs are real `TechnicalAnalysis` models (some computed from candles by the
real analysis code); news, market, vision and risk inputs reuse the builders the risk
tests use, so shapes can't drift. No test touches the network or calls a model.
"""

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest

from upscale.agents import AgentContext, OpportunityAgent
from upscale.orchestrator import Orchestrator
from upscale.routing import route
from upscale.schemas import AgentResult, ChatMessage, ChatRequest, ImageAttachment
from upscale.services import opportunity as opportunity_module
from upscale.services.market_data import TIMEFRAMES
from upscale.services.opportunity import OpportunityAssessment, OpportunityConfig, assess
from upscale.services.risk import RiskAssessment
from upscale.services.technical_analysis import (
    Indicator,
    RecentRange,
    SupportResistance,
    TechnicalAnalysis,
    TechnicalAnalysisConfig,
    Trend,
    VolumeAnalysis,
    analyze_series,
)
from upscale.services.technical_analysis import Level as PriceLevel

from .conftest import CHART_PNG
from .ta_helpers import swing_series
from .test_risk_agent import failed, market, news, story, vision
from .test_technical_agent import serve_wave

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
Zone = tuple[float, float]

# --- Builders -------------------------------------------------------------------------------


def _level(bounds: Zone, close: float) -> PriceLevel:
    lower, upper = bounds
    mean = (lower + upper) / 2
    return PriceLevel(
        price=mean,
        lower=lower,
        upper=upper,
        touches=2,
        last_touched=NOW - timedelta(hours=1),
        distance_pct=100 * (mean - close) / close,
    )


def ta(
    *,
    symbol: str = "BTC",
    tf: str = "5m",
    close: float = 84_440.0,
    prev: float | None = 84_420.0,
    trend: str | None = "uptrend",
    macd: tuple[float, float] | None = (12.0, 8.0),
    rsi: float = 60.0,
    support: tuple[Zone, ...] = ((84_418.0, 84_429.0),),
    resistance: tuple[Zone, ...] = ((84_700.0, 84_720.0),),
    containing: Zone | None = None,
    atr: float | None = 20.0,
    volume: tuple[float, float, float] | None = (1.8, 70.0, 30.0),  # relative, up %, down %
    levels: bool = True,
    requested: str | None = None,
    sma20: float | None = None,
) -> AgentResult:
    """A technical agent result shaped exactly like `TechnicalAnalysisAgent`'s findings.

    Defaults: a 5m uptrend whose last close broke above the zone $84,418–$84,429.
    """
    sma20 = sma20 if sma20 is not None else close * (0.999 if trend != "downtrend" else 1.001)
    sma50 = close * (0.998 if trend != "downtrend" else 1.002)
    indicators = [
        Indicator(name="SMA 20", kind="sma", params={"period": 20}, available=True, value=sma20),
        Indicator(name="SMA 50", kind="sma", params={"period": 50}, available=True, value=sma50),
        Indicator(name="RSI 14", kind="rsi", params={"period": 14}, available=True, value=rsi),
    ]
    if macd is None:
        indicators.append(
            Indicator(
                name="MACD 12/26/9",
                kind="macd",
                params={},
                available=False,
                unavailable_reason="needs 34 candles, only 20 available",
            )
        )
    else:
        indicators.append(
            Indicator(
                name="MACD 12/26/9",
                kind="macd",
                params={"fast": 12, "slow": 26, "signal": 9},
                available=True,
                value=macd[0],
                signal=macd[1],
                histogram=macd[0] - macd[1],
            )
        )
    sr = (
        SupportResistance(
            method="m",
            lookback=100,
            available=True,
            atr_period=14,
            atr=atr,
            zone_tolerance=atr,
            support=[_level(z, close) for z in support],
            resistance=[_level(z, close) for z in resistance],
            containing=_level(containing, close) if containing else None,
        )
        if levels
        else SupportResistance(
            method="m", lookback=100, available=False, unavailable_reason="needs 100 candles"
        )
    )
    if volume is None:
        vol = VolumeAnalysis(
            lookback=20, available=False, unavailable_reason="no per-candle volume"
        )
    else:
        rel, up, down = volume
        vol = VolumeAnalysis(
            lookback=20,
            available=True,
            unit=symbol,
            last_volume=10.0 * rel,
            average_volume=10.0,
            relative_volume=rel,
            up_volume_pct=up,
            down_volume_pct=down,
        )
    analysis = TechnicalAnalysis(
        symbol=symbol,
        timeframe=tf,
        provider="Kraken",
        provider_id=f"{symbol}USD",
        pair=f"{symbol}/USD",
        candle_count=180,
        first_candle_at=NOW - timedelta(hours=15),
        last_candle_at=NOW - timedelta(minutes=5),
        last_close=close,
        previous_close=prev,
        volume_available=volume is not None,
        volume_note="note",
        volume=vol,
        indicators=indicators,
        recent_range=RecentRange(lookback=50, available=False, unavailable_reason="n/a"),
        trend=Trend(method="m", available=True, label=trend, reasons=["r"])
        if trend
        else Trend(method="m", available=False, unavailable_reason="needs 50 candles"),
        levels=sr,
    )
    return _technical_result(analysis, requested)


def _technical_result(analysis: TechnicalAnalysis, requested: str | None = None) -> Any:
    findings = {
        **analysis.model_dump(mode="json"),
        "requested_timeframe": requested,
        "requested_timeframe_source": "user" if requested else None,
        "timeframe_note": None,
        "not_analyzed": [],
    }
    return AgentResult(agent="technical_analysis", mock=False, summary="ta", findings=findings)


def bear_ta(**overrides: Any) -> Any:
    """A 5m downtrend whose last close broke below the zone $84,349–$84,360."""
    params: dict[str, Any] = {
        "close": 84_340.0,
        "prev": 84_360.0,
        "trend": "downtrend",
        "macd": (-12.0, -8.0),
        "rsi": 40.0,
        "support": ((84_000.0, 84_020.0),),
        "resistance": ((84_349.0, 84_360.0),),
        "volume": (1.8, 30.0, 70.0),
    }
    return ta(**(params | overrides))


def risk(
    overall: str = "low", uncertainty: str = "low", *, asset: str = "BTC", why: str = "x"
) -> Any:
    review = RiskAssessment(
        asset=asset,
        overall_risk=overall,
        overall_reasons=[f"{why} (risk)"],
        uncertainty_level=uncertainty,
        uncertainty_reasons=[f"{why} (uncertainty)"] if uncertainty != "low" else [],
        missing_evidence=[],
        factors=[],
        invalidation_conditions=[],
        conflicting_evidence=[],
        data_quality=[],
        inputs=[],
        rules="r",
    )
    findings = {"reviewed_agents": ["market", "technical_analysis"], **review.model_dump()}
    return AgentResult(agent="risk", mock=False, summary="risk", findings=findings)


def btc_market(price: float = 84_445.0) -> Any:
    return market(price=price)


def bullish_news() -> Any:
    return news([story("bullish", "medium"), story("bullish", "low")])


def bearish_news() -> Any:
    return news([story("bearish", "medium"), story("bearish", "low")])


def decide(*results: Any, asset: str | None = "BTC", **cfg: Any) -> OpportunityAssessment:
    return assess({r.agent: r for r in results}, asset, OpportunityConfig(**cfg) if cfg else None)


def run_agent(*results: Any, asset: str | None = "BTC") -> Any:
    context = AgentContext(
        query="", assets=[asset] if asset else [], prior_results={r.agent: r for r in results}
    )
    return asyncio.run(OpportunityAgent().run(context))


def blocker_ids(a: OpportunityAssessment) -> set[str]:
    return {f.id for f in a.blocking_factors}


def caution_ids(a: OpportunityAssessment) -> set[str]:
    return {f.id for f in a.cautions}


def all_text(result: Any) -> str:
    return " ".join([result.summary, *result.evidence, *(r.description for r in result.risks)])


def full_buy(**overrides: Any) -> list[Any]:
    return [ta(**overrides), btc_market(), bullish_news(), risk()]


def full_sell(**overrides: Any) -> list[Any]:
    return [bear_ta(**overrides), btc_market(84_335.0), bearish_news(), risk()]


# --- Strong setups --------------------------------------------------------------------------


def test_strong_buy_setup():
    a = decide(*full_buy())
    assert a.action == "buy"
    assert a.confirmed is True and a.setup == "breakout"
    assert a.confidence == "high"
    assert a.timeframe == "5m" and a.asset == "BTC"
    assert {s.id for s in a.bullish_evidence} >= {
        "uptrend",
        "macd",
        "rsi",
        "break_above",
        "volume_bias",
        "volume_surge",
        "news_tone",
    }
    assert a.bearish_evidence == [] and a.blocking_factors == []
    assert a.bullish_score > a.bearish_score
    assert a.risk_level == "low" and a.uncertainty_level == "low"
    assert a.missing_evidence == []


def test_strong_sell_setup():
    a = decide(*full_sell())
    assert a.action == "sell"
    assert a.confirmed is True and a.setup == "breakdown"
    assert a.confidence == "high"
    assert {s.id for s in a.bearish_evidence} >= {
        "downtrend",
        "macd",
        "rsi",
        "break_below",
        "volume_bias",
        "volume_surge",
        "news_tone",
    }
    assert a.bullish_evidence == []


def test_buy_after_confirmed_resistance_break():
    a = decide(*full_buy())
    assert a.bullish_trigger is not None and a.bullish_trigger.confirmed
    assert a.bullish_trigger.price == 84_429.0  # the broken zone's upper bound
    assert a.bullish_trigger.condition.startswith(
        "confirmed: 5m close $84,440.00 above ~$84,429.00"
    )
    assert a.entry_zone is not None
    assert (a.entry_zone.lower, a.entry_zone.upper) == (84_418.0, 84_429.0)
    assert a.entry_basis == "retest of the broken zone"


def test_sell_after_confirmed_support_break():
    a = decide(*full_sell())
    assert a.bearish_trigger is not None and a.bearish_trigger.confirmed
    assert a.bearish_trigger.price == 84_349.0  # the broken zone's lower bound
    assert a.entry_zone is not None
    assert (a.entry_zone.lower, a.entry_zone.upper) == (84_349.0, 84_360.0)
    assert a.entry_basis == "retest of the broken zone, to exit longs"
    assert "Exit zone: ~$84,349.00–$84,360.00" in run_agent(*full_sell()).summary


def test_trend_continuation_buy_without_a_break():
    a = decide(
        ta(prev=84_435.0, support=((84_000.0, 84_020.0),), resistance=((84_900.0, 84_920.0),)),
        btc_market(),
        bullish_news(),
        risk(),
    )
    assert a.action == "buy" and a.setup == "trend_continuation"
    assert a.entry_zone is None  # no evidence-based entry zone is manufactured
    assert a.bullish_trigger is not None and a.bullish_trigger.basis == "last_close"
    assert a.invalidation is not None and a.invalidation.price == 84_000.0


# --- WAIT ------------------------------------------------------------------------------------


def test_wait_due_to_mixed_trend():
    a = decide(*full_buy(trend="mixed"))
    assert a.action == "wait"
    assert "mixed_trend" in blocker_ids(a)
    assert a.summary.startswith("BTC 5m: the 5m trend is mixed")
    assert a.invalidation is None and a.entry_zone is None and a.setup is None


def test_wait_inside_containing_zone():
    a = decide(
        *full_buy(
            close=84_425.0,
            prev=84_424.0,
            containing=(84_418.0, 84_429.0),
            support=((84_300.0, 84_320.0),),
        )
    )
    assert a.action == "wait"
    assert "inside_zone" in blocker_ids(a)
    assert a.bullish_trigger is not None and a.bearish_trigger is not None
    assert (
        a.bullish_trigger.price == 84_429.0 and a.bullish_trigger.basis == "containing_zone_upper"
    )
    assert (
        a.bearish_trigger.price == 84_418.0 and a.bearish_trigger.basis == "containing_zone_lower"
    )


def test_wait_because_breakout_has_not_confirmed():
    # Uptrend pressing into resistance $84,418–$84,429 (8 below it, ATR 20): no close above yet.
    a = decide(
        *full_buy(
            close=84_410.0,
            prev=84_405.0,
            support=((84_300.0, 84_320.0),),
            resistance=((84_418.0, 84_429.0),),
        )
    )
    assert a.action == "wait"
    assert "breakout_not_confirmed" in blocker_ids(a)
    assert a.bullish_trigger is not None
    assert a.bullish_trigger.confirmed is False
    assert a.bullish_trigger.price == 84_429.0
    assert a.bullish_trigger.condition.startswith(
        "5m close above ~$84,429.00 (upper bound of the resistance zone ~$84,418.00–$84,429.00)"
    )
    assert a.bearish_trigger is not None and a.bearish_trigger.price == 84_300.0


def test_uptrend_with_bearish_macd_conflict_waits():
    a = decide(*full_buy(macd=(8.0, 12.0)))
    assert a.action == "wait"
    assert "macd_against_trend" in blocker_ids(a)
    assert any("against the uptrend" in f.reason for f in a.blocking_factors)


def test_downtrend_with_bullish_macd_conflict_waits():
    a = decide(*full_sell(macd=(-8.0, -12.0)))
    assert a.action == "wait"
    assert "macd_against_trend" in blocker_ids(a)
    assert a.bearish_trigger is not None and not a.bearish_trigger.confirmed


def test_weak_volume_blocks_a_breakout():
    a = decide(*full_buy(volume=(0.5, 50.0, 50.0)))
    assert a.action == "wait"
    assert "weak_breakout_volume" in blocker_ids(a)


def test_weak_volume_lowers_confidence_on_a_continuation():
    base = {"prev": 84_435.0, "support": ((84_000.0, 84_020.0),)}
    strong = decide(*full_buy(**base, volume=(1.2, 70.0, 30.0)))
    weak = decide(*full_buy(**base, volume=(0.5, 70.0, 30.0)))
    assert strong.action == weak.action == "buy"
    assert "weak_volume" in caution_ids(weak)
    assert ["low", "medium", "high"].index(weak.confidence) < ["low", "medium", "high"].index(
        strong.confidence
    )


def test_strong_confirming_volume_adds_evidence():
    with_surge = decide(*full_buy(volume=(1.8, 70.0, 30.0)))
    flat = decide(*full_buy(volume=(1.0, 50.0, 50.0)))
    assert with_surge.bullish_score - flat.bullish_score == 2
    assert any(s.id == "volume_surge" for s in with_surge.bullish_evidence)
    assert not any(s.category == "volume" for s in flat.bullish_evidence)


def test_rsi_extreme_blocks_chasing():
    a = decide(*full_buy(rsi=85.0))
    assert a.action == "wait" and "rsi_extreme" in blocker_ids(a)
    stretched = decide(*full_buy(rsi=74.0))
    assert "rsi_stretched" in caution_ids(stretched)


def test_bullish_technical_with_bearish_news_lowers_confidence():
    a = decide(ta(), btc_market(), bearish_news(), risk())
    assert a.action == "buy"
    assert "news_against" in caution_ids(a)
    assert a.confidence != "high"
    assert any(s.id == "news_tone" for s in a.bearish_evidence)


def test_bullish_technical_with_high_impact_bearish_news_waits():
    headline = "Exchange hack drains hot wallets"
    a = decide(ta(), btc_market(), news([story("bearish", "high", title=headline)]), risk())
    assert a.action == "wait"
    assert "news_against" in blocker_ids(a)
    assert headline in a.summary


def test_bearish_technical_with_high_impact_bullish_news_waits():
    a = decide(bear_ta(), btc_market(84_335.0), news([story("bullish", "high")]), risk())
    assert a.action == "wait"
    assert "news_against" in blocker_ids(a)


def test_conflicting_high_impact_news_waits():
    both = news([story("bullish", "high"), story("bearish", "high")])
    a = decide(ta(), btc_market(), both, risk())
    assert a.action == "wait" and "news_conflict" in blocker_ids(a)


def test_mixed_news_is_a_caution_not_a_signal():
    mixed = news([story("bullish", "medium"), story("bearish", "medium")])
    assert mixed.findings["reports"][0]["overall_news_sentiment"] == "mixed"
    a = decide(ta(), btc_market(), mixed, risk())
    clean = decide(ta(), btc_market(), bullish_news(), risk())
    assert "news_mixed" in caution_ids(a)
    assert not any(s.category == "news" for s in a.bullish_evidence + a.bearish_evidence)
    assert a.action == "buy"  # a strong confirmed setup survives mixed news...
    assert a.confidence == "medium" and clean.confidence == "high"  # ...with less confidence


def test_news_alone_never_creates_an_action():
    a = decide(*full_buy(trend="mixed", macd=None), news([story("bullish", "high")]))
    assert a.action == "wait"


def test_stale_news_is_not_scored():
    a = decide(ta(), btc_market(), news([story("bearish", "high", stale=True)]), risk())
    assert not any(s.category == "news" for s in a.bearish_evidence)
    assert "Fresh classified news for this asset" in a.missing_evidence


# --- Risk -----------------------------------------------------------------------------------


def test_high_risk_forces_wait():
    a = decide(ta(), btc_market(), bullish_news(), risk("high", why="large 24h move"))
    assert a.action == "wait"
    assert "risk_high" in blocker_ids(a)
    assert a.summary.startswith("BTC 5m: overall risk is high (large 24h move (risk))")
    assert a.risk_level == "high"


def test_high_uncertainty_forces_wait():
    a = decide(ta(), btc_market(), bullish_news(), risk("low", "high", why="stale data"))
    assert a.action == "wait" and "uncertainty_high" in blocker_ids(a)
    assert a.uncertainty_level == "high"


def test_medium_risk_caps_confidence():
    a = decide(ta(), btc_market(), bullish_news(), risk("medium"))
    assert a.action == "buy" and a.confidence == "medium"


def test_low_risk_does_not_automatically_create_buy():
    assert decide(*full_buy(trend="mixed")).action == "wait"
    # Trend + MACD alone (RSI stretched, no break, no volume, no news) isn't enough.
    thin = decide(
        ta(rsi=72.0, prev=84_435.0, support=((84_000.0, 84_020.0),), volume=None),
        btc_market(),
        risk("low"),
    )
    assert thin.action == "wait" and "not_enough_evidence" in blocker_ids(thin)


# --- Partial data ---------------------------------------------------------------------------


def test_missing_news_still_decides_with_lower_confidence():
    a = decide(ta(), btc_market(), risk())
    assert a.action == "buy"
    assert a.confidence == "medium"
    assert "News & sentiment for this asset" in a.missing_evidence


def test_missing_market_still_decides_with_lower_confidence():
    a = decide(ta(), bullish_news(), risk())
    assert a.action == "buy" and a.confidence == "medium"
    assert "Live market snapshot" in a.missing_evidence
    assert a.live_price is None


def test_missing_risk_is_more_conservative():
    minimal = {
        "prev": 84_435.0,
        "support": ((84_000.0, 84_020.0),),
        "resistance": ((84_900.0, 84_920.0),),
        "volume": None,
    }
    with_risk = decide(ta(**minimal), btc_market(), risk())
    without = decide(ta(**minimal), btc_market())
    assert with_risk.action == "buy"
    assert without.action == "wait" and "not_enough_evidence" in blocker_ids(without)
    strong_without = decide(ta(), btc_market(), bullish_news())
    assert strong_without.action == "buy" and strong_without.confidence == "low"
    assert strong_without.risk_level == "unavailable"
    assert "Risk review" in strong_without.missing_evidence


def test_failed_risk_review_counts_as_missing():
    a = decide(ta(), btc_market(), bullish_news(), failed("risk"))
    assert a.risk_level == "unavailable" and a.confidence == "low"


@pytest.mark.parametrize("technical", [None, failed("technical_analysis")])
def test_missing_technical_waits(technical):
    results = [btc_market(), bullish_news(), risk()]
    if technical is not None:
        results.append(technical)
    a = decide(*results)
    assert a.action == "wait" and a.confidence == "low"
    assert blocker_ids(a) == {"no_technical"}
    assert a.bullish_trigger is None and a.bearish_trigger is None
    assert "Live technical analysis for this asset" in a.missing_evidence


def test_no_asset_waits():
    a = decide(ta(), btc_market(), asset=None)
    assert a.action == "wait" and blocker_ids(a) == {"no_asset"}


def test_unavailable_trend_or_macd_waits():
    assert "trend_unavailable" in blocker_ids(decide(*full_buy(trend=None)))
    assert "macd_unavailable" in blocker_ids(decide(*full_buy(macd=None)))


def test_timeframe_fallback_is_flagged_and_caps_confidence():
    a = decide(ta(tf="4h", requested="1m"), btc_market(), bullish_news(), risk())
    assert "timeframe_fallback" in caution_ids(a)
    assert a.timeframe == "4h" and a.requested_timeframe == "1m"
    assert a.confidence == "low"


def test_live_price_beyond_invalidation_waits():
    a = decide(ta(), btc_market(84_400.0), bullish_news(), risk())
    assert a.action == "wait"
    assert "live_beyond_invalidation" in blocker_ids(a)


def test_evidence_from_wrong_asset_is_ignored():
    a = decide(
        ta(symbol="ETH"),
        market(symbol="ETH", price=3_100.0),
        news([story("bullish", "high")], asset="ETH"),
        risk(asset="ETH"),
    )
    assert a.action == "wait" and blocker_ids(a) == {"no_technical"}
    assert a.risk_level == "unavailable"
    assert {
        "Live technical analysis for this asset",
        "Live market snapshot",
        "News & sentiment for this asset",
        "Risk review",
    } <= set(a.missing_evidence)
    # Right-asset technicals with another asset's news: news is not used.
    mixed = decide(ta(), btc_market(), news([story("bearish", "high")], asset="ETH"), risk())
    assert mixed.action == "buy"
    assert not any(s.category == "news" for s in mixed.bearish_evidence)


def test_screenshot_of_another_asset_is_not_used():
    a = decide(*full_buy(), vision(symbol="ETH"))
    assert any("shows ETH, not BTC" in c for c in a.context)


def test_screenshot_is_supporting_context_only():
    with_shot = decide(*full_buy(), vision(price=84_000.0, timeframe="5m", normalized="5m"))
    without = decide(*full_buy())
    assert with_shot.action == without.action
    assert with_shot.bullish_score == without.bullish_score
    assert any("supporting context only" in c for c in with_shot.context)


# --- Triggers and invalidation --------------------------------------------------------------


def _real(swings: list[float], close: float) -> Any:
    """Technical findings computed by the real analysis from BTC-like 1m candles."""
    series = swing_series(84_000.0, 28.0, swings, last_close=close)
    config = TechnicalAnalysisConfig(sr_lookback=len(series.candles), sr_pivot_window=2)
    return _technical_result(analyze_series(series, config))


def test_zone_boundary_triggers_use_real_zone_bounds():
    technical = _real([83_950.00, 83_972.00, 84_020.00, 84_035.00, 84_044.00], 84_000.0)
    [support] = technical.findings["levels"]["support"]
    [resistance] = technical.findings["levels"]["resistance"]
    a = decide(technical, btc_market(84_000.0), risk())
    assert a.action == "wait"
    assert a.timeframe == "1m"
    assert a.bullish_trigger is not None and a.bearish_trigger is not None
    assert a.bullish_trigger.price == resistance["upper"] == 84_044.0
    assert a.bearish_trigger.price == support["lower"] == 83_950.0
    assert a.bullish_trigger.condition.startswith("1m close above ~$84,044.00")
    assert a.bearish_trigger.condition.startswith("1m close below ~$83,950.00")


def test_real_containing_zone_gives_both_bounds_as_triggers():
    technical = _real([83_950.00, 84_016.80, 84_025.20], 84_022.40)
    a = decide(technical, btc_market(84_022.4), risk())
    assert "inside_zone" in blocker_ids(a)
    assert a.bullish_trigger is not None and a.bearish_trigger is not None
    assert (a.bullish_trigger.price, a.bearish_trigger.price) == (84_025.2, 84_016.8)


def test_invalidation_uses_real_zone_bounds():
    buy = decide(*full_buy())
    assert buy.invalidation is not None
    assert buy.invalidation.price == 84_418.0 and buy.invalidation.basis == "broken_zone_lower"
    assert buy.invalidation.condition == (
        "5m close back below ~$84,418.00 (lower bound of the broken zone ~$84,418.00–$84,429.00)"
    )
    sell = decide(*full_sell())
    assert sell.invalidation is not None
    assert sell.invalidation.price == 84_360.0 and sell.invalidation.basis == "broken_zone_upper"
    # The trend classification ending is a secondary invalidation at the real SMA value.
    assert [i.basis for i in buy.other_invalidations] == ["trend_average"]


def test_continuation_without_support_is_invalidated_by_the_trend_average():
    a = decide(
        ta(prev=84_435.0, support=(), resistance=((84_900.0, 84_920.0),), sma20=84_390.0),
        btc_market(),
        bullish_news(),
        risk(),
    )
    assert a.action == "buy"
    assert a.invalidation is not None
    assert a.invalidation.basis == "trend_average" and a.invalidation.price == 84_390.0


def test_no_arbitrary_stop_percentages():
    for a in (decide(*full_buy()), decide(*full_sell())):
        assert a.invalidation is not None
        allowed = {84_418.0, 84_429.0, 84_349.0, 84_360.0, 84_000.0, 84_020.0}
        assert a.invalidation.price in allowed
        texts = [a.invalidation.condition, *(i.condition for i in a.other_invalidations)]
        assert not any("%" in t for t in texts)


def _closes_on(tf: str, text: str) -> bool:
    return re.search(rf"(?<!\d){tf} close", text) is not None


def test_triggers_and_invalidation_stay_on_the_analyzed_timeframe():
    a = decide(*full_buy(tf="15m"))
    texts = [a.invalidation.condition if a.invalidation else ""]
    texts += [t.condition for t in (a.bullish_trigger, a.bearish_trigger) if t]
    for text in texts:
        assert "15m close" in text
        assert not any(_closes_on(other, text) for other in TIMEFRAMES if other != "15m")


# --- Output ---------------------------------------------------------------------------------


def test_buy_summary_is_short_and_structured():
    result = run_agent(*full_buy())
    lines = result.summary.splitlines()
    assert lines[0] == "BUY"
    assert lines[1].startswith("BTC 5m ")
    assert any(line.startswith("Trigger: confirmed: 5m close") for line in lines)
    assert any(line.startswith("Entry zone: ~$84,418.00–$84,429.00") for line in lines)
    assert any(line.startswith("Invalidation: 5m close back below ~$84,418.00") for line in lines)
    assert lines[-1] == "Risk: Low · Confidence: High"
    assert len(lines) <= 7
    assert result.findings["action"] == "buy"


def test_wait_summary_shows_both_triggers():
    result = run_agent(
        *full_buy(
            close=84_410.0,
            prev=84_405.0,
            support=((84_300.0, 84_320.0),),
            resistance=((84_418.0, 84_429.0),),
        )
    )
    lines = result.summary.splitlines()
    assert lines[0] == "WAIT"
    assert any(line.startswith("Buy trigger: 5m close above ~$84,429.00") for line in lines)
    assert any(line.startswith("Sell trigger: 5m close below ~$84,300.00") for line in lines)
    assert lines[-1].startswith("Risk: Low · Confidence: ")


def test_sell_does_not_imply_shorting():
    a = decide(*full_sell())
    assert a.action == "sell"
    assert a.sell_meaning == (
        "SELL means reduce or exit a long position. It is not a signal to open a short position."
    )
    result = run_agent(*full_sell())
    assert "Note: SELL means reduce or exit a long position" in result.summary
    text = all_text(result).lower()
    assert "go short" not in text and "open a short position." in text
    assert text.count("short") == 1  # only in the explanation of what SELL means


FAKE_PRECISION = re.compile(
    r"probab|chance|odds|likel|guarantee|\d+(?:\.\d+)?\s*%\s*(?:stop|chance|probab)|target|"
    r"profit",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "results",
    [
        full_buy(),
        full_sell(),
        full_buy(trend="mixed"),
        [ta(), btc_market(), bearish_news()],
        [btc_market()],
    ],
    ids=["buy", "sell", "wait", "no-risk", "no-technical"],
)
def test_no_numerical_probabilities_guarantees_or_targets(results):
    result = run_agent(*results)
    assert not FAKE_PRECISION.search(all_text(result))
    assert result.confidence is None  # no numeric confidence either
    assert result.findings["confidence"] in ("low", "medium", "high")


def test_assessment_is_deterministic():
    assert decide(*full_buy()) == decide(*full_buy())


def test_thresholds_are_configurable():
    assert decide(*full_buy(), min_score=12).action == "wait"
    with pytest.raises(ValueError):
        OpportunityConfig(weak_relative_volume=1.2)


def test_offline_and_model_free(monkeypatch):
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(httpx2.AsyncClient, "send", refuse)
    monkeypatch.setattr(httpx2.Client, "send", refuse)
    assert run_agent(*full_buy()).findings["action"] == "buy"
    source = open(opportunity_module.__file__, encoding="utf-8").read()
    assert "httpx" not in source and "anthropic" not in source


# --- Routing and orchestration -------------------------------------------------------------

DECISION_QUERIES = [
    "Should I buy BTC right now?",
    "Should I sell BTC?",
    "What's the best move on BTC?",
    "Buy or wait on BTC?",
    "Is BTC a good entry here?",
]


@pytest.mark.parametrize("query", DECISION_QUERIES)
def test_decision_questions_run_the_full_pipeline(query):
    assert route(query, has_images=False).agents == [
        "technical_analysis",
        "market",
        "news_sentiment",
        "risk",
        "opportunity",
    ]


@pytest.mark.parametrize("query", ["What should I do?", "Should I sell?", "What's the best move?"])
def test_decision_questions_without_a_coin_still_route_to_opportunity(query):
    agents = route(query, has_images=False).agents
    assert agents[-2:] == ["risk", "opportunity"] and "news_sentiment" in agents


@pytest.mark.parametrize(
    ("query", "news"),
    [
        ("Analyze this chart.", False),
        ("", False),
        ("Is this a good entry?", True),
        ("Analyze this chart and tell me what to do.", True),
    ],
)
def test_screenshot_routing(query, news):
    agents = route(query, has_images=True).agents
    expected = ["vision", "technical_analysis", "market"]
    expected += ["news_sentiment"] if news else []
    assert agents == [*expected, "risk", "opportunity"]


def test_technical_only_questions_do_not_require_news_or_a_decision():
    assert route("BTC RSI and support levels", has_images=False).agents == [
        "technical_analysis",
        "risk",
    ]


def test_opportunity_always_brings_its_evidence():
    agents = route("BTC outlook?", has_images=False).agents
    assert {"technical_analysis", "market", "risk", "opportunity"} <= set(agents)


def test_non_crypto_requests_are_not_routed():
    assert route("where can I buy a car", has_images=False).agents == []
    assert route("what's the weather", has_images=False).agents == []


def ask(content: str, images: int = 0) -> Any:
    attachments = [
        ImageAttachment(name=f"c{i}.png", media_type="image/png", data=CHART_PNG)
        for i in range(images)
    ]
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=content, attachments=attachments)]
    )
    return asyncio.run(Orchestrator().respond(request))


def test_orchestrator_runs_opportunity_after_risk_and_its_inputs(monkeypatch, fake_coingecko):
    serve_wave(fake_coingecko)
    orchestrator = Orchestrator()
    order: list[str] = []
    for agent in orchestrator.agents.values():
        original = agent.run

        async def recorded(context, _agent=agent, _run=original):
            result = await _run(context)
            order.append(_agent.name)
            return result

        monkeypatch.setattr(agent, "run", recorded)
    request = ChatRequest(messages=[ChatMessage(role="user", content="Should I buy BTC?")])
    response = asyncio.run(orchestrator.respond(request))
    assert order[-1] == "opportunity" and order[-2] == "risk"
    assert set(order[:3]) == {"technical_analysis", "market", "news_sentiment"}
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    decision = by_agent["opportunity"]
    assert decision.status == "ok" and decision.mock is False
    assert decision.findings["risk_level"] == by_agent["risk"].findings["overall_risk"]
    assert {i["agent"]: i["status"] for i in decision.findings["inputs"]}[
        "technical_analysis"
    ] == "ok"
    # The reply leads with the decision block.
    assert response.message.content.startswith(decision.summary)
    assert decision.summary.splitlines()[0] in ("BUY", "SELL", "WAIT")


def test_screenshot_decision_request(fake_coingecko):
    serve_wave(fake_coingecko)
    response = ask("Analyze this chart and tell me what to do", images=1)
    analysis = response.analysis
    assert analysis.agents_used == [
        "vision",
        "technical_analysis",
        "market",
        "news_sentiment",
        "risk",
        "opportunity",
    ]
    decision = next(r for r in analysis.agent_results if r.agent == "opportunity")
    assert decision.findings["asset"] == "BTC"  # from the screenshot
    assert decision.findings["timeframe"] == "4h"
    assert {i["agent"]: i["status"] for i in decision.findings["inputs"]}["vision"] == "ok"
    assert any("Screenshot (4h chart)" in c for c in decision.findings["context"])


def test_orchestrator_decision_survives_evidence_failures(fake_coingecko):
    fake_coingecko.handler = lambda r: httpx2.Response(503)
    decision = next(
        r for r in ask("Should I buy BTC?").analysis.agent_results if r.agent == "opportunity"
    )
    assert decision.status == "ok"
    assert decision.findings["action"] == "wait"
    assert decision.findings["confidence"] == "low"


@pytest.mark.parametrize("tf", ["1m", "5m", "15m"])
def test_kraken_intraday_decision_uses_the_requested_timeframe(fake_kraken, tf):
    response = ask(f"Should I buy BTC on the {tf} chart?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    technical = by_agent["technical_analysis"]
    assert technical.status == "ok"
    assert technical.findings["provider"] == "Kraken" and technical.findings["timeframe"] == tf
    decision = by_agent["opportunity"].findings
    assert decision["timeframe"] == tf and decision["requested_timeframe"] == tf
    assert decision["action"] in ("buy", "sell", "wait")
    conditions = [
        t["condition"] for t in (decision["bullish_trigger"], decision["bearish_trigger"]) if t
    ]
    if decision["invalidation"]:
        conditions.append(decision["invalidation"]["condition"])
    assert conditions  # Kraken history is long enough for zones
    for text in conditions:
        assert f"{tf} close" in text
        assert not any(_closes_on(other, text) for other in TIMEFRAMES if other != tf)
    zones = technical.findings["levels"]
    bounds = {
        z[k]
        for z in zones["support"] + zones["resistance"] + [zones["containing"] or {}]
        for k in ("lower", "upper")
        if k in z
    }
    for trigger in (decision["bullish_trigger"], decision["bearish_trigger"]):
        if trigger and not trigger["confirmed"]:
            assert trigger["price"] in bounds
