from upscale.agents.base import Agent, AgentContext
from upscale.formatting import usd
from upscale.schemas import AgentResult, Risk, Scenario
from upscale.services import technical_analysis_service
from upscale.services.market_data import MarketDataError
from upscale.services.technical_analysis import (
    InvalidCandleDataError,
    TechnicalAnalysis,
    TechnicalAnalysisService,
)

TIME_FORMAT = "%Y-%m-%d %H:%M UTC"


class TechnicalAnalysisAgent(Agent):
    """Deterministic technical read (moving averages, RSI, MACD, range, trend, levels).

    All numbers come from `TechnicalAnalysisService`; this agent only turns them into
    evidence and conditional scenarios. It never recommends buying or selling.
    """

    name = "technical_analysis"
    description = "SMA/EMA, RSI, MACD, recent range, rule-based trend and approximate levels."
    depends_on = ("vision",)

    def __init__(self, service: TechnicalAnalysisService | None = None):
        self.service = service or technical_analysis_service

    async def run(self, context: AgentContext) -> AgentResult:
        symbol = context.primary_asset
        if symbol is None:
            note = "No specific cryptocurrency was identified, so no technical analysis was run."
            if context.attachments:
                note += " Screenshot contents are not read yet; name the coin to analyze it."
            return AgentResult(agent=self.name, mock=False, summary=note, findings={})

        try:
            timeframe, fallback_note = self.service.resolve_timeframe(context.timeframe)
            if fallback_note and context.timeframe_source == "screenshot":
                fallback_note = (
                    f"The screenshot's timeframe is {context.timeframe}, but {fallback_note}"
                )
            analysis = await self.service.analyze(symbol, timeframe)
        except (MarketDataError, InvalidCandleDataError) as exc:
            return AgentResult(
                agent=self.name,
                status="error",
                mock=False,
                summary=f"Technical analysis could not be performed for {symbol}: {exc}.",
                findings={"symbol": symbol},
                error=str(exc),
            )

        others = context.assets[1:]
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=_summary(analysis),
            findings={
                **analysis.model_dump(mode="json"),
                "requested_timeframe": context.timeframe,
                "requested_timeframe_source": context.timeframe_source,
                "timeframe_note": fallback_note,
                "not_analyzed": others,
            },
            evidence=_evidence(analysis, fallback_note, others),
            scenarios=_scenarios(analysis),
            risks=_risks(analysis, fallback_note),
        )


def _label(analysis: TechnicalAnalysis) -> str:
    return f"{analysis.symbol} {analysis.timeframe}"


def _summary(a: TechnicalAnalysis) -> str:
    trend = f"trend: {a.trend.label}" if a.trend.label else "trend unavailable"
    return (
        f"Technical analysis of {_label(a)} from {a.candle_count} {a.provider} candles ({trend})."
    )


def _evidence(a: TechnicalAnalysis, fallback_note: str | None, others: list[str]) -> list[str]:
    lines = [
        f"{_label(a)}: {a.candle_count} candles from {a.provider}, "
        f"{a.first_candle_at.strftime(TIME_FORMAT)} to {a.last_candle_at.strftime(TIME_FORMAT)} "
        f"(candle open times); last close {usd(a.last_close)}."
    ]
    if fallback_note:
        lines.append(fallback_note)

    for ind in a.indicators:
        if not ind.available or ind.value is None:
            lines.append(f"{ind.name} unavailable: {ind.unavailable_reason}.")
        elif ind.kind in ("sma", "ema"):
            side = (
                "above"
                if a.last_close > ind.value
                else "below"
                if a.last_close < ind.value
                else "at"
            )
            lines.append(f"{ind.name}: {usd(ind.value)} (last close is {side} it).")
        elif ind.kind == "rsi":
            zone = (
                "above 70, a zone commonly described as overbought"
                if ind.value >= 70
                else "below 30, a zone commonly described as oversold"
                if ind.value <= 30
                else "between 30 and 70"
            )
            lines.append(f"{ind.name}: {ind.value:.1f} ({zone}).")
        elif ind.kind == "macd" and ind.signal is not None and ind.histogram is not None:
            side = (
                "above" if ind.value > ind.signal else "below" if ind.value < ind.signal else "at"
            )
            lines.append(
                f"{ind.name}: line {ind.value:.4g}, signal {ind.signal:.4g}, "
                f"histogram {ind.histogram:+.4g} (line {side} signal)."
            )

    r = a.recent_range
    if r.available and r.high is not None and r.low is not None and r.high_at and r.low_at:
        position = (
            f"; last close at {r.close_position_pct:.0f}% of the range"
            if r.close_position_pct is not None
            else ""
        )
        lines.append(
            f"{r.lookback}-candle range: low {usd(r.low)} ({r.low_at.strftime(TIME_FORMAT)}) "
            f"to high {usd(r.high)} ({r.high_at.strftime(TIME_FORMAT)}){position}."
        )
    else:
        lines.append(f"{r.lookback}-candle high/low unavailable: {r.unavailable_reason}.")

    if a.trend.available:
        lines.append(f"Trend (rule-based): {a.trend.label}. " + " ".join(a.trend.reasons))
    else:
        lines.append(f"Trend unavailable: {a.trend.unavailable_reason}.")

    lv = a.levels
    if lv.available:
        support = ", ".join(f"~{usd(x.price)} ({x.touches} touch(es))" for x in lv.support)
        resistance = ", ".join(f"~{usd(x.price)} ({x.touches} touch(es))" for x in lv.resistance)
        lines.append(
            f"Approximate support: {support or 'none below the last close in the window'}; "
            f"approximate resistance: {resistance or 'none above the last close in the window'}."
        )
    else:
        lines.append(f"Support/resistance unavailable: {lv.unavailable_reason}.")

    lines.append(a.volume_note)
    if others:
        lines.append(f"Only {a.symbol} was analyzed; not analyzed: {', '.join(others)}.")
    return lines


def _scenarios(a: TechnicalAnalysis) -> list[Scenario]:
    """Conditional descriptions built from the computed levels. Not recommendations."""
    support = a.levels.support[0] if a.levels.support else None
    resistance = a.levels.resistance[0] if a.levels.resistance else None
    tf = a.timeframe
    scenarios = []
    if support and resistance:
        scenarios.append(
            Scenario(
                name="Range holds",
                description=(
                    f"{a.symbol} keeps trading between approximate support ~{usd(support.price)} "
                    f"and resistance ~{usd(resistance.price)}."
                ),
                conditions=[f"{tf} closes stay between the two levels"],
                invalidation=f"A {tf} close outside the {usd(support.low)}–{usd(resistance.high)} zone.",
            )
        )
    if resistance:
        scenarios.append(
            Scenario(
                name="Break above resistance",
                description=f"{a.symbol} closes above the ~{usd(resistance.price)} resistance zone.",
                conditions=[
                    f"A {tf} close above {usd(resistance.high)}",
                    "MACD line staying above its signal line would be consistent with this",
                ],
                invalidation=f"Price falls back below {usd(resistance.low)}.",
            )
        )
    if support:
        scenarios.append(
            Scenario(
                name="Break below support",
                description=f"{a.symbol} closes below the ~{usd(support.price)} support zone.",
                conditions=[
                    f"A {tf} close below {usd(support.low)}",
                    "MACD line staying below its signal line would be consistent with this",
                ],
                invalidation=f"Price recovers above {usd(support.high)}.",
            )
        )
    return scenarios


def _risks(a: TechnicalAnalysis, fallback_note: str | None) -> list[Risk]:
    risks = [
        Risk(
            description="Indicators describe past prices and lag; they do not predict moves.",
            severity="medium",
        ),
    ]
    if a.levels.available:
        risks.append(
            Risk(
                description="Support/resistance levels are approximate zones from recent swing points.",
                severity="low",
            )
        )
    if not a.volume_available:
        risks.append(
            Risk(
                description=f"No per-candle volume from {a.provider}; moves can't be checked against volume.",
                severity="low",
            )
        )
    unavailable = [i.name for i in a.indicators if not i.available]
    if unavailable:
        risks.append(
            Risk(
                description=f"Not enough history for: {', '.join(unavailable)}.",
                severity="medium",
            )
        )
    if fallback_note:
        risks.append(Risk(description=fallback_note, severity="medium"))
    return risks
