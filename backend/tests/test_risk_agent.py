"""Risk agent: deterministic risk factors, overall risk, uncertainty and invalidation.

Inputs are built with the same models (and the news agent's own findings builder) the real
agents use, so the shapes can't drift. No test touches the network or calls a model.
"""

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest

from upscale.agents import AgentContext, RiskAgent
from upscale.agents.news_sentiment import _report_findings
from upscale.formatting import usd, usd_zone
from upscale.orchestrator import Orchestrator
from upscale.routing import route
from upscale.schemas import AgentName, AgentResult, ChatMessage, ChatRequest, ImageAttachment
from upscale.services.market_data import MarketSnapshot
from upscale.services.news import NewsReport, Story, aggregate_sentiment, find_conflicts
from upscale.services.risk import RiskAssessment, RiskConfig, assess
from upscale.services.technical_analysis import (
    Indicator,
    RecentRange,
    SupportResistance,
    TechnicalAnalysis,
    TechnicalAnalysisConfig,
    Trend,
    support_resistance,
)
from upscale.services.technical_analysis import Level as PriceLevel
from upscale.services.vision import ChartReading, ChartVision

from .conftest import CHART_PNG, chart_reading_data
from .ta_helpers import swing_series
from .test_technical_agent import serve_wave

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
BUY_SELL_WORDS = ("buy", "sell", "go long", "go short", "entry", "take profit", "stop loss")
FAKE_PRECISION = re.compile(
    r"probab|chance|odds|likelihood|\blikely\b|liquidation level|order[- ]book|"
    r"implied volatility|volatility index|\bATR\b",
    re.IGNORECASE,
)


# --- Builders -------------------------------------------------------------------------------


def _level(close: float, pct: float) -> PriceLevel:
    price = close * (1 + pct / 100)
    return PriceLevel(
        price=price,
        lower=price * 0.998,
        upper=price * 1.002,
        touches=2,
        last_touched=NOW - timedelta(days=2),
        distance_pct=pct,
    )


def technical(
    *,
    symbol: str = "BTC",
    timeframe: str = "4h",
    close: float = 100.0,
    trend: str | None = "uptrend",
    support_pct: float | None = -6.0,
    resistance_pct: float | None = 6.0,
    rsi: float = 55.0,
    macd: tuple[float, float] = (1.0, 0.5),
    volume: bool = False,
    short_history: bool = False,
    note: str | None = None,
    requested: str | None = None,
    source: str | None = None,
    levels: SupportResistance | None = None,
) -> AgentResult:
    """A technical agent result, shaped exactly like `TechnicalAnalysisAgent`'s findings."""

    def ind(name: str, kind: str, value: float, ok: bool = True, **extra: Any) -> Indicator:
        if not ok:
            return Indicator(
                name=name,
                kind=kind,
                params={},
                available=False,
                unavailable_reason="needs 50 candles, only 30 available",
            )
        return Indicator(name=name, kind=kind, params={}, available=True, value=value, **extra)

    long_ok = not short_history
    indicators = [
        ind("SMA 20", "sma", close * 0.97),
        ind("SMA 50", "sma", close * 0.94, long_ok),
        ind("EMA 20", "ema", close * 0.975),
        ind("EMA 50", "ema", close * 0.95, long_ok),
        ind("RSI 14", "rsi", rsi),
        ind("MACD 12/26/9", "macd", macd[0], long_ok, signal=macd[1], histogram=macd[0] - macd[1]),
    ]
    trend_model = (
        Trend(method="m", available=True, label=trend, reasons=["Close is above SMA 20."])
        if trend and long_ok
        else Trend(method="m", available=False, unavailable_reason="needs 50 candles")
    )
    levels = levels or (
        SupportResistance(
            method="m",
            lookback=100,
            available=True,
            support=[_level(close, support_pct)] if support_pct is not None else [],
            resistance=[_level(close, resistance_pct)] if resistance_pct is not None else [],
        )
        if long_ok
        else SupportResistance(
            method="m", lookback=100, available=False, unavailable_reason="needs 100 candles"
        )
    )
    analysis = TechnicalAnalysis(
        symbol=symbol,
        timeframe=timeframe,
        provider="TestProvider",
        provider_id=symbol.lower(),
        candle_count=30 if short_history else 180,
        first_candle_at=NOW - timedelta(days=30),
        last_candle_at=NOW - timedelta(hours=4),
        last_close=close,
        volume_available=volume,
        volume_note="volume note",
        indicators=indicators,
        recent_range=RecentRange(lookback=50, available=False, unavailable_reason="n/a"),
        trend=trend_model,
        levels=levels,
    )
    findings = {
        **analysis.model_dump(mode="json"),
        "requested_timeframe": requested,
        "requested_timeframe_source": source,
        "timeframe_note": note,
        "not_analyzed": [],
    }
    return AgentResult(agent="technical_analysis", mock=False, summary="ta", findings=findings)


def market(
    *,
    symbol: str = "BTC",
    price: float = 100.0,
    change: float | None = 1.0,
    high: float | None = None,
    low: float | None = None,
    volume: float | None = 1e9,
    lag_minutes: float = 1,
) -> AgentResult:
    snap = MarketSnapshot(
        symbol=symbol,
        name="Bitcoin",
        provider="CoinGecko",
        provider_id="bitcoin",
        price_usd=price,
        change_24h_pct=change,
        high_24h_usd=high if high is not None else price * 1.02,
        low_24h_usd=low if low is not None else price * 0.98,
        volume_24h_usd=volume,
        last_updated=NOW - timedelta(minutes=lag_minutes),
        fetched_at=NOW,
    )
    findings = {
        "provider": "CoinGecko",
        "quote_currency": "USD",
        "snapshots": [snap.model_dump(mode="json")],
        "unavailable": [],
    }
    return AgentResult(agent="market", mock=False, summary="market", findings=findings)


_ids = iter(range(10_000))


def story(
    sentiment: str | None = "neutral",
    impact: str | None = "low",
    *,
    title: str | None = None,
    stale: bool = False,
    scope: str = "asset",
) -> Story:
    n = next(_ids)
    return Story(
        id=f"n{n}",
        title=title or f"Routine crypto story {n}",
        source="CoinDesk",
        url=f"https://example.com/{n}",
        published_at=NOW - timedelta(hours=60 if stale else 2),
        summary=None,
        tickers=["BTC"],
        also_reported_by=[],
        scope=scope,
        relevance="headline",
        age_hours=60.0 if stale else 2.0,
        stale=stale,
        model_use="headline",
        sentiment=sentiment,
        impact=impact,
    )


def news(stories: list[Story], asset: str | None = "BTC") -> AgentResult:
    focus = [s for s in stories if s.scope == ("asset" if asset else "market")]
    report = NewsReport(
        asset=asset,
        provider="RSS",
        sources=["CoinDesk"],
        failed_sources=[],
        retrieved_at=NOW,
        analyzed_at=NOW,
        model="fake",
        stories=stories,
        asset_story_count=sum(s.scope == "asset" for s in stories),
        overall_sentiment=aggregate_sentiment(focus),
        market_news_sentiment=aggregate_sentiment([s for s in stories if s.scope == "market"]),
        sentiment_counts={},
        conflicts=find_conflicts(focus),
        uncertainties=[],
        excluded=[],
        discarded=[],
    )
    findings = {"provider": "RSS", "reports": [_report_findings(report)], "unavailable": []}
    return AgentResult(agent="news_sentiment", mock=False, summary="news", findings=findings)


def calm_news() -> AgentResult:
    return news([story("neutral", "low"), story("bullish", "low"), story("neutral", "medium")])


def vision(
    *,
    price: float | None = 100.0,
    timeframe: str = "4h",
    normalized: str | None = "4h",
    symbol: str = "BTC",
    clean: bool = True,
    **overrides: Any,
) -> AgentResult:
    data = chart_reading_data(
        asset={
            "symbol": symbol,
            "pair": f"{symbol}/USDT",
            "exchange": None,
            "basis": "visible",
            "evidence": "header",
        },
        timeframe={"label": timeframe, "basis": "visible", "evidence": "toolbar"},
        displayed_price={
            "value": price,
            "basis": "visible" if price else "unknown",
            "evidence": "tag",
        },
        **overrides,
    )
    if clean:
        data |= {
            "uncertainties": [],
            "support_levels": [
                {
                    "price": 95.0,
                    "label": None,
                    "source": "drawn_line",
                    "basis": "visible",
                    "evidence": "line",
                }
            ],
            "resistance_levels": [
                {
                    "price": 105.0,
                    "label": None,
                    "source": "drawn_line",
                    "basis": "visible",
                    "evidence": "line",
                }
            ],
        }
    chart = ChartVision(
        image_name="chart.png",
        media_type="image/png",
        size_bytes=100,
        width=320,
        height=200,
        model="fake-vision",
        reading=ChartReading.model_validate(data),
        normalized_timeframe=normalized,
        discarded=[],
        analyzed_at=NOW,
    )
    findings = {
        "charts": [chart.model_dump(mode="json")],
        "failed": [],
        "detected_asset": symbol,
        "detected_timeframe": normalized,
        "detected_timeframe_label": timeframe,
    }
    return AgentResult(agent="vision", mock=False, summary="vision", findings=findings)


def failed(name: AgentName, reason: str = "provider down") -> AgentResult:
    return AgentResult(agent=name, status="error", mock=False, summary="failed", error=reason)


def review(*results: AgentResult, asset: str | None = "BTC", **cfg: float) -> RiskAssessment:
    return assess({r.agent: r for r in results}, asset, RiskConfig(**cfg) if cfg else None)


def run_agent(*results: AgentResult, asset: str | None = "BTC") -> AgentResult:
    context = AgentContext(
        query="",
        assets=[asset] if asset else [],
        prior_results={r.agent: r for r in results},
    )
    return asyncio.run(RiskAgent().run(context))


def ids(a: RiskAssessment) -> set[str]:
    return {f.id for f in a.factors}


def factor(a: RiskAssessment, factor_id: str):
    return next(f for f in a.factors if f.id == factor_id)


def all_text(result: AgentResult) -> str:
    return " ".join([result.summary, *result.evidence, *(r.description for r in result.risks)])


# --- Input combinations ---------------------------------------------------------------------


def test_all_four_inputs_with_calm_evidence_is_low_risk_low_uncertainty():
    a = review(vision(), technical(), market(), calm_news())
    assert a.overall_risk == "low"
    assert a.uncertainty_level == "low"
    assert {i.agent: i.status for i in a.inputs} == {
        "vision": "ok",
        "technical_analysis": "ok",
        "market": "ok",
        "news_sentiment": "ok",
    }
    assert all(f.severity == "low" for f in a.factors)
    assert a.overall_reasons == [
        "No medium or high risk factors were found in the available evidence."
    ]


def test_market_only():
    a = review(market())
    assert a.overall_risk == "low"
    assert a.uncertainty_level == "high"  # no technical structure and no news
    assert any(m.startswith("Technical structure") for m in a.missing_evidence)
    assert any(m.startswith("Classified recent news") for m in a.missing_evidence)


def test_technical_only():
    a = review(technical(trend="mixed"))
    assert a.overall_risk == "medium"
    assert "mixed_trend" in ids(a)
    assert a.uncertainty_level == "medium"  # news missing
    assert "Live market snapshot: 24h change and range (not part of this request)" in (
        a.missing_evidence
    )


def test_news_only():
    a = review(news([story("bullish", "high"), story("bearish", "high")]))
    assert "strong_news_conflict" in ids(a)
    assert a.overall_risk == "medium"  # one uncorroborated high factor
    assert a.uncertainty_level == "high"
    assert any(m.startswith("Live price data") for m in a.missing_evidence)


def test_screenshot_only_is_unknown_risk_with_high_uncertainty():
    result = run_agent(vision())
    a = result.findings
    assert a["overall_risk"] == "unknown"
    assert a["uncertainty_level"] == "high"
    assert "A live price to check the screenshot against" in a["missing_evidence"]
    assert "insufficient_evidence" in {f["id"] for f in a["factors"]}
    assert result.summary.startswith("Overall risk for BTC: unknown.")
    assert "Missing:" in result.summary


def test_no_inputs_is_unknown_and_says_what_is_missing():
    a = review()
    assert a.overall_risk == "unknown"
    assert a.uncertainty_level == "high"
    assert len(a.missing_evidence) == 3
    assert {i.status for i in a.inputs} == {"not_run"}


def test_partial_agent_failure_raises_uncertainty_not_risk():
    a = review(failed("market"), technical(), calm_news())
    assert a.overall_risk == "low"
    assert "market_failed" in ids(a)
    assert factor(a, "market_failed").affects == "uncertainty"
    assert any("Live market data failed (provider down)" in d for d in a.data_quality)
    assert a.uncertainty_level == "medium"


def test_everything_failed_is_unknown_not_high():
    a = review(failed("vision"), failed("technical_analysis"), failed("market"))
    assert a.overall_risk == "unknown"
    assert a.uncertainty_level == "high"
    assert {"vision_failed", "technical_analysis_failed", "market_failed"} <= ids(a)


def test_mock_and_unreadable_inputs_are_not_used():
    mock = AgentResult(agent="market", mock=True, summary="[Mock]", findings={"snapshots": [1]})
    broken = AgentResult(
        agent="technical_analysis", mock=False, summary="x", findings={"candle_count": "many"}
    )
    a = review(mock, broken)
    states = {i.agent: i.status for i in a.inputs}
    assert states["market"] == "not_run"
    assert states["technical_analysis"] == "unreadable"
    assert a.overall_risk == "unknown"


def test_other_assets_data_is_not_used_for_the_primary_asset():
    a = review(market(symbol="ETH"), technical(symbol="ETH"))
    assert a.overall_risk == "unknown"
    assert {i.agent: i.status for i in a.inputs}["market"] == "no_data"


# --- Individual rules -------------------------------------------------------------------------


def test_mixed_technical_signals():
    a = review(technical(trend="uptrend", macd=(0.2, 0.8)), market(), calm_news())
    f = factor(a, "indicator_conflict")
    assert f.severity == "medium" and f.category == "technical"
    assert "MACD line is below its signal line" in f.explanation
    assert f.explanation in a.conflicting_evidence
    assert a.overall_risk == "medium"


def test_strong_conflicting_news_with_a_technical_risk_is_high():
    a = review(
        technical(resistance_pct=0.5),
        market(),
        news([story("bullish", "high"), story("bearish", "high")]),
    )
    assert factor(a, "strong_news_conflict").severity == "high"
    assert factor(a, "near_resistance").severity == "medium"
    assert a.overall_risk == "high"  # high news factor corroborated by a technical one


def test_news_tone_against_technical_trend_is_a_conflict():
    a = review(technical(trend="uptrend"), news([story("bearish", "medium")] * 2))
    assert "news_vs_trend" in ids(a)
    assert any("does not line up" in c for c in a.conflicting_evidence)


def test_high_impact_and_regulatory_news():
    a = review(
        news(
            [
                story("neutral", "high", title="SEC sues major exchange over staking"),
                story("neutral", "low"),
            ]
        )
    )
    assert factor(a, "high_impact_news").severity == "medium"
    assert "SEC sues major exchange" in factor(a, "macro_regulatory_event").explanation


def test_stale_news_raises_uncertainty_and_is_not_treated_as_fresh_impact():
    a = review(
        technical(),
        market(),
        news([story("bearish", "high", stale=True), story("neutral", "low", stale=True)]),
    )
    assert factor(a, "stale_news").affects == "uncertainty"
    assert "high_impact_news" not in ids(a)
    assert a.uncertainty_level == "medium"


def test_insufficient_news():
    a = review(technical(), market(), news([story("neutral", "low", scope="market")]))
    assert "insufficient_news" in ids(a)
    assert any(m.startswith("Classified recent news") for m in a.missing_evidence)


@pytest.mark.parametrize(
    ("screenshot_price", "severity", "uncertainty"),
    [(100.5, None, "low"), (103.0, "medium", "medium"), (112.0, "high", "high")],
)
def test_screenshot_vs_live_price(screenshot_price, severity, uncertainty):
    a = review(vision(price=screenshot_price), technical(), market(), calm_news())
    if severity is None:
        assert "screenshot_price_discrepancy" not in ids(a)
    else:
        f = factor(a, "screenshot_price_discrepancy")
        assert f.severity == severity and f.affects == "uncertainty"
        assert "may be stale" in f.explanation and "live data takes precedence" in f.explanation
        assert f.explanation in a.conflicting_evidence
    assert a.overall_risk == "low"  # a stale screenshot is uncertainty, not market risk
    assert a.uncertainty_level == uncertainty


def test_screenshot_of_another_asset_is_not_compared_on_price():
    a = review(vision(symbol="ETH", price=3100.0), technical(), market(), calm_news())
    assert "screenshot_asset_mismatch" in ids(a)
    assert "screenshot_price_discrepancy" not in ids(a)


def test_screenshot_timeframe_mismatch_from_provider_fallback():
    note = "The screenshot's timeframe is 1m, but 1m candles are not available; analyzed 4h."
    a = review(
        vision(timeframe="1m", normalized="1m"),
        technical(note=note, requested="1m", source="screenshot"),
        market(),
        calm_news(),
    )
    f = factor(a, "timeframe_mismatch")
    assert f.severity == "medium" and f.affects == "uncertainty"
    assert "1m chart" in f.explanation and "4h candles" in f.explanation
    assert "unsupported_timeframe" not in ids(a)  # reported once, as the mismatch
    assert a.uncertainty_level == "medium"


def test_screenshot_timeframe_differs_from_user_requested_timeframe():
    a = review(vision(timeframe="1D", normalized="1d"), technical(timeframe="4h"), market())
    assert "live analysis used 4h" in factor(a, "timeframe_mismatch").headline


def test_unsupported_timeframe_requested_by_user():
    note = "1m candles are not available from the configured market data provider; analyzed 4h."
    a = review(technical(note=note, requested="1m", source="user"), market(), calm_news())
    f = factor(a, "unsupported_timeframe")
    assert f.severity == "medium" and f.category == "data_quality"
    assert "requested 1m timeframe could only be analyzed using 4h" in f.headline


def test_missing_volume():
    a = review(technical(volume=False), market(volume=None), calm_news())
    assert factor(a, "volume_unavailable").severity == "low"
    assert "market_volume_missing" in ids(a)
    assert "Per-candle volume (not supplied by the candle provider)" in a.missing_evidence
    assert (
        review(technical(volume=True)).missing_evidence.count(
            "Per-candle volume (not supplied by the candle provider)"
        )
        == 0
    )


def test_insufficient_candle_history():
    a = review(technical(short_history=True), market(), calm_news())
    f = factor(a, "insufficient_history")
    assert f.severity == "medium" and f.affects == "uncertainty"
    assert "SMA 50" in f.explanation and "trend" in f.explanation
    assert not any("classification" in c.condition for c in a.invalidation_conditions)


def test_stale_market_data_and_disagreeing_live_sources():
    a = review(technical(close=100.0), market(price=120.0, lag_minutes=90))
    assert "stale_market_data" in ids(a)
    assert "live_sources_disagree" in ids(a)
    assert a.uncertainty_level == "high"


def test_screenshot_inferred_and_unreadable_parts():
    a = review(vision(clean=False), technical(), market())
    assert "screenshot_inferred" in ids(a)
    assert "Volume pane is cropped" in factor(a, "screenshot_unreadable").explanation


# --- Overall risk and uncertainty levels ----------------------------------------------------


def test_medium_risk_evidence():
    a = review(technical(support_pct=-0.8), market(), calm_news())
    assert factor(a, "near_support").severity == "medium"
    assert a.overall_risk == "medium"


def test_high_risk_evidence():
    a = review(technical(trend="mixed"), market(change=-12.0, low=85.0, high=102.0), calm_news())
    assert factor(a, "large_recent_move").severity == "high"
    assert a.overall_risk == "high"
    assert "BTC moved -12.0% in 24h" in a.overall_reasons[0]


def test_single_uncorroborated_high_factor_is_medium_not_high():
    a = review(technical(), market(change=12.0), calm_news())
    assert factor(a, "large_recent_move").severity == "high"
    assert a.overall_risk == "medium"


def test_two_medium_factors_in_one_category_stay_medium():
    a = review(technical(trend="mixed", resistance_pct=0.5), market(), calm_news())
    assert a.overall_risk == "medium"


@pytest.mark.parametrize(
    ("results", "level"),
    [
        ((technical(), market(), calm_news()), "low"),
        ((technical(), market()), "medium"),  # news missing
        ((technical(), failed("market"), calm_news()), "medium"),  # one agent failed
        ((market(),), "high"),  # two evidence types missing
        ((failed("technical_analysis"), failed("market"), calm_news()), "high"),
    ],
)
def test_uncertainty_levels(results, level):
    assert review(*results).uncertainty_level == level


@pytest.mark.parametrize(
    ("change", "severity"),
    [(4.99, None), (5.0, "medium"), (-9.99, "medium"), (10.0, "high"), (None, None)],
)
def test_severity_thresholds_are_deterministic(change, severity):
    a = review(market(change=change, high=100.5, low=99.5))
    got = next((f.severity for f in a.factors if f.id == "large_recent_move"), None)
    assert got == severity


def test_level_proximity_thresholds():
    near = review(technical(resistance_pct=1.0))
    close = review(technical(resistance_pct=2.5))
    far = review(technical(resistance_pct=2.6))
    assert factor(near, "near_resistance").severity == "medium"
    assert factor(close, "near_resistance").severity == "low"
    assert "near_resistance" not in ids(far)


def test_assessment_is_deterministic():
    inputs = (vision(price=104.0), technical(trend="mixed"), market(change=7.0), calm_news())
    assert review(*inputs) == review(*inputs)
    assert review(*reversed(inputs)) == review(*inputs)


def test_thresholds_are_configurable():
    a = review(market(change=3.0, high=100.5, low=99.5), move_medium_pct=2.0)
    assert factor(a, "large_recent_move").severity == "medium"


# --- Invalidation ---------------------------------------------------------------------------


def _zones(swings: list[float], close: float = 84_000.0) -> SupportResistance:
    """Zones from BTC-like 1m candles around $84,000 with an ATR of ~$28."""
    series = swing_series(84_000.0, 28.0, swings, last_close=close)
    config = TechnicalAnalysisConfig(sr_lookback=len(series.candles), sr_pivot_window=2)
    return support_resistance(series, config)


# Resistance zone $84,020.00-$84,044.00 (mean $84,033.00); support $83,950.00-$83,972.00.
TWO_ZONES = [83_950.00, 83_972.00, 84_020.00, 84_035.00, 84_044.00]


def test_level_invalidation_names_the_zone_and_its_breaking_bound():
    levels = _zones(TWO_ZONES)
    a = review(technical(timeframe="1m", close=84_000.0, levels=levels))
    conditions = [c.condition for c in a.invalidation_conditions]
    assert (
        "Resistance zone ~$84,020.00–$84,044.00; a 1m close above ~$84,044.00 would break it."
        in conditions
    )
    assert (
        "Support zone ~$83,950.00–$83,972.00; a 1m close below ~$83,950.00 would break it."
        in conditions
    )


def test_level_invalidation_threshold_is_always_the_zone_bound():
    levels = _zones(TWO_ZONES)
    a = review(technical(timeframe="1m", close=84_000.0, levels=levels))
    text = " ".join(c.condition for c in a.invalidation_conditions)
    [support], [resistance] = levels.support, levels.resistance
    assert levels.zone_tolerance is not None
    assert resistance.upper - resistance.lower <= levels.zone_tolerance
    assert f"close above ~{usd(resistance.upper)} would break it" in text
    assert f"close below ~{usd(support.lower)} would break it" in text
    # The mean is never presented as the level that gets broken.
    assert usd(resistance.price) not in text and usd(support.price) not in text


def test_price_inside_a_zone_is_not_called_support_or_resistance():
    # Zone mean $84,021.00 is below the $84,022.40 close; a mean-based rule would call it support.
    close = 84_022.40
    levels = _zones([83_950.00, 84_016.80, 84_025.20], close=close)
    a = review(technical(timeframe="1m", close=close, levels=levels))
    conditions = [c.condition for c in a.invalidation_conditions]
    assert not any("Resistance zone" in c for c in conditions)
    assert "Support zone ~$83,950.00; a 1m close below ~$83,950.00 would break it." in conditions
    assert (
        "Price is inside the swing zone ~$84,016.80–$84,025.20; a 1m close above ~$84,025.20 "
        "or below ~$84,016.80 would take it out of that zone." in conditions
    )
    inside = factor(a, "inside_level_zone")
    assert inside.severity == "medium"
    assert "~$84,016.80–$84,025.20" in inside.headline


def test_near_level_factor_states_the_zone_and_its_breaking_bound():
    a = review(technical(timeframe="1m", close=84_000.0, levels=_zones(TWO_ZONES)))
    near = factor(a, "near_resistance")
    assert "zone ~$84,020.00–$84,044.00 (mean ~$84,033.00)" in near.explanation
    assert near.weakens_if == (
        "Resistance zone ~$84,020.00–$84,044.00; a 1m close above ~$84,044.00 would break it."
    )


def test_single_swing_zone_is_shown_as_one_price():
    assert usd_zone(82_000.0, 82_000.0) == "~$82,000.00"
    assert usd_zone(84_020.0, 84_044.0) == "~$84,020.00–$84,044.00"


def test_invalidation_conditions_come_from_evidence():
    a = review(
        vision(price=100.0),
        technical(trend="uptrend"),
        market(),
        news([story("bullish", "high", title="ETF inflows hit a record")]),
    )
    conditions = [c.condition for c in a.invalidation_conditions]
    assert conditions[0] == (
        "A 4h close below SMA 20 (~$97.00) would end the current rule-based uptrend classification."
    )
    assert "Support zone ~$93.81–$94.19; a 4h close below ~$93.81 would break it." in conditions
    assert (
        "Resistance zone ~$105.79–$106.21; a 4h close above ~$106.21 would break it." in conditions
    )
    assert any(c.startswith("A move outside the 24h range ($98.00 – $102.00)") for c in conditions)
    assert any("diverges more than 2% from the screenshot's ~$100.00" in c for c in conditions)
    assert any("'ETF inflows hit a record' (CoinDesk) develops further" in c for c in conditions)
    assert any("New high-impact bearish coverage would weaken" in c for c in conditions)
    assert {c.basis for c in a.invalidation_conditions} == {"live_data", "screenshot", "news"}


def test_downtrend_invalidation_is_a_close_above_the_fast_average():
    a = review(technical(trend="downtrend", macd=(-1.0, -0.5)))
    assert a.invalidation_conditions[0].condition.startswith("A 4h close above SMA 20")


def test_screenshot_levels_are_used_only_without_live_levels():
    with_live = review(vision(), technical())
    without_live = review(vision(), technical(short_history=True))
    assert not any(
        "marked on the screenshot" in c.condition for c in with_live.invalidation_conditions
    )
    marked = [
        c for c in without_live.invalidation_conditions if "marked on the screenshot" in c.condition
    ]
    assert [c.basis for c in marked] == ["screenshot", "screenshot"]
    assert "read from the screenshot, not live data" in marked[0].condition


def test_every_factor_says_what_would_weaken_the_analysis():
    a = review(
        vision(price=112.0, clean=False),
        technical(
            trend="uptrend", macd=(0.1, 0.9), rsi=82, resistance_pct=0.4, short_history=False
        ),
        market(change=12.0, volume=None, lag_minutes=90),
        news([story("bullish", "high"), story("bearish", "high"), story(None, None)]),
    )
    assert len(a.factors) >= 10
    assert all(f.weakens_if for f in a.factors)


# --- Output safety --------------------------------------------------------------------------

SCENARIOS = [
    (),
    (vision(),),
    (market(change=15.0),),
    (technical(trend="mixed", rsi=85),),
    (vision(price=130.0), technical(), market(), calm_news()),
    (
        technical(resistance_pct=0.3),
        market(),
        news([story("bullish", "high"), story("bearish", "high")]),
    ),
]


@pytest.mark.parametrize("results", SCENARIOS)
def test_no_fake_probabilities_or_recommendations(results):
    text = all_text(run_agent(*results))
    assert not FAKE_PRECISION.search(text), FAKE_PRECISION.search(text)
    lowered = text.lower()
    assert not any(word in lowered for word in BUY_SELL_WORDS)


def test_no_invented_prices():
    """Every dollar amount in the output comes from an input value."""
    tech = technical(trend="uptrend")
    mkt = market()
    shot = vision(price=101.0)
    result = run_agent(shot, tech, mkt, calm_news())
    known: set[str] = set()
    ta = tech.findings
    for ind in ta["indicators"]:
        if ind["value"] is not None:
            known.add(usd(ind["value"]))
    for lv in [*ta["levels"]["support"], *ta["levels"]["resistance"]]:
        known |= {usd(lv["price"]), usd(lv["lower"]), usd(lv["upper"])}
    snap = mkt.findings["snapshots"][0]
    known |= {usd(snap[k]) for k in ("price_usd", "high_24h_usd", "low_24h_usd")}
    known |= {usd(ta["last_close"]), usd(101.0)}
    amounts = set(re.findall(r"\$[\d,]+\.\d{2}", all_text(result)))
    assert amounts and amounts <= known


def test_no_volume_or_market_numbers_without_market_data():
    text = all_text(run_agent(technical()))
    assert "Live market snapshot: 24h change and range (not part of this request)" in text
    assert not re.search(r"moved|24h range \(\$|24h (high|low) \(", text)
    assert not re.search(r"volume (of|was|is) \$?\d", text, re.IGNORECASE)


# --- Agent output ---------------------------------------------------------------------------


def test_agent_summary_explains_the_level():
    result = run_agent(
        technical(trend="mixed", resistance_pct=0.6, note="n", requested="1m", source="user"),
        market(),
        news([story("bullish", "medium"), story("bearish", "medium")]),
    )
    assert result.mock is False and result.status == "ok"
    assert result.summary.startswith("Overall risk for BTC: medium. The technical trend is mixed;")
    assert "BTC is near resistance" in result.summary
    assert "Uncertainty: medium because the requested 1m timeframe could only be analyzed" in (
        result.summary
    )
    f = result.findings
    for key in (
        "overall_risk",
        "uncertainty_level",
        "uncertainty_reasons",
        "missing_evidence",
        "factors",
        "invalidation_conditions",
        "conflicting_evidence",
        "data_quality",
        "rules",
    ):
        assert key in f
    assert all(r.source for r in result.risks)
    assert {r.severity for r in result.risks} <= {"low", "medium", "high"}


def test_agent_declares_input_dependencies():
    assert set(RiskAgent.depends_on) == {"vision", "technical_analysis", "market", "news_sentiment"}


# --- Orchestrator and routing ---------------------------------------------------------------


def ask(content: str, images: int = 0):
    attachments = [
        ImageAttachment(name=f"c{i}.png", media_type="image/png", data=CHART_PNG)
        for i in range(images)
    ]
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=content, attachments=attachments)]
    )
    return asyncio.run(Orchestrator().respond(request))


def test_orchestrator_runs_risk_after_its_inputs(fake_coingecko):
    serve_wave(fake_coingecko)
    response = ask("What's up with BTC?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    risk = by_agent["risk"]
    assert risk.status == "ok" and risk.mock is False
    assert {"technical_analysis", "market", "news_sentiment"} <= set(
        risk.findings["reviewed_agents"]
    )
    assert risk.findings["overall_risk"] in ("low", "medium", "high")
    assert any(r.source == "news_sentiment" for r in response.analysis.risks)
    assert any(n.startswith("Risk review:") for n in response.analysis.uncertainty.notes)
    assert "Overall risk for BTC:" in response.message.content


def test_orchestrator_risk_survives_input_failures(fake_coingecko):
    fake_coingecko.handler = lambda r: httpx2.Response(503)
    risk = next(r for r in ask("What's up with BTC?").analysis.agent_results if r.agent == "risk")
    assert risk.status == "ok"
    ids_ = {f["id"] for f in risk.findings["factors"]}
    assert {"market_failed", "technical_analysis_failed"} <= ids_
    assert risk.findings["uncertainty_level"] == "high"


def test_orchestrator_screenshot_flow_feeds_risk(fake_coingecko):
    serve_wave(fake_coingecko)
    response = ask("", images=1)
    risk = next(r for r in response.analysis.agent_results if r.agent == "risk")
    assert risk.findings["reviewed_agents"] == ["market", "technical_analysis", "vision"]
    assert {i["agent"]: i["status"] for i in risk.findings["inputs"]}["vision"] == "ok"
    assert response.analysis.uncertainty.level == "high"


def test_uncertainty_from_risk_review_can_raise_the_analysis_level():
    orchestrator = Orchestrator(agents=[RiskAgent()])
    response = asyncio.run(
        orchestrator.respond(ChatRequest(messages=[ChatMessage(role="user", content="BTC news?")]))
    )
    assert response.analysis.agents_used == ["risk"]
    assert response.analysis.uncertainty.level == "high"  # no evidence at all


@pytest.mark.parametrize(
    ("query", "images"),
    [("Any news on bitcoin?", 0), ("BTC RSI", 0), ("", 1), ("What about ETH?", 0)],
)
def test_risk_is_routed_whenever_other_agents_run(query, images):
    decision = route(query, has_images=bool(images))
    assert decision.agents[-1] == "risk"
    assert len(decision.agents) > 1


def test_risk_is_not_routed_for_non_crypto_requests():
    assert "risk" not in route("what's the weather", has_images=False).agents
