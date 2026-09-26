from typing import Any

import upscale.services as services
from upscale.agents.base import Agent, AgentContext
from upscale.formatting import usd, usd_zone
from upscale.schemas import AgentResult, Risk, Scenario
from upscale.services import technical_analysis_service
from upscale.services.chains import same_address
from upscale.services.market_data import TIMEFRAMES, MarketDataError, Timeframe
from upscale.services.solana_dex import SolanaDexSnapshot
from upscale.services.strategy import strategy_for
from upscale.services.technical_analysis import (
    InvalidCandleDataError,
    Level,
    TechnicalAnalysis,
    TechnicalAnalysisService,
    analyze_series,
)
from upscale.services.technical_pool import alternate_pools

TIME_FORMAT = "%Y-%m-%d %H:%M UTC"


class TechnicalAnalysisAgent(Agent):
    """Deterministic technical read (moving averages, RSI, MACD, range, trend, levels).

    All numbers come from `TechnicalAnalysisService`; this agent only turns them into
    evidence and conditional scenarios. It never recommends buying or selling.
    """

    name = "technical_analysis"
    description = (
        "SMA/EMA, RSI, MACD, recent range, relative volume, rule-based trend and approximate "
        "levels."
    )
    # Vision can name the asset; DEX data names the pool whose candles a token trades on.
    depends_on = ("vision", "dex_market")

    def __init__(self, service: TechnicalAnalysisService | None = None):
        self.service = service or technical_analysis_service

    async def run(self, context: AgentContext) -> AgentResult:
        symbol = context.primary_asset
        if symbol is None:
            note = "No specific cryptocurrency was identified, so no technical analysis was run."
            if context.attachments:
                note += " Screenshot contents are not read yet; name the coin to analyze it."
            return AgentResult(agent=self.name, mock=False, summary=note, findings={})
        pool = _pool(context)
        identity = context.asset_identity
        if identity is not None and identity.address and pool is not None:
            if (
                not same_address(pool.chain, pool.mint, identity.address)
                or pool.chain != identity.chain
            ):
                pool = None  # another token's pool: never used
        if identity is not None and identity.address:
            # A contract token: only its own pool's candles describe it, never a ticker's.
            if pool is None:
                reason = "no usable DEX pool was found, so there are no candles for this token"
                return self._failure(symbol, reason)
            return await self._run_pool(context, symbol, pool, None)

        try:
            timeframe, fallback_note = self.service.resolve_timeframe(context.timeframe)
            if fallback_note and context.timeframe_source == "screenshot":
                fallback_note = (
                    f"The screenshot's timeframe is {context.timeframe}, but {fallback_note}"
                )
            analysis = await self.service.analyze(symbol, timeframe)
        except (MarketDataError, InvalidCandleDataError) as exc:
            if pool is not None:
                # Fallback for a listed token that also trades in a DEX pool: that exact
                # pool's candles, recorded as such (never another token's).
                note = f"Exchange candles were unavailable ({exc}); used DEX pool candles."
                return await self._run_pool(context, symbol, pool, note)
            return self._failure(symbol, str(exc))
        return self._result(context, analysis, fallback_note)

    def _failure(self, symbol: str, reason: str) -> AgentResult:
        return AgentResult(
            agent=self.name,
            status="error",
            mock=False,
            summary=f"Technical analysis could not be performed for {symbol}: {reason}.",
            findings={"symbol": symbol},
            error=reason,
        )

    async def _run_pool(
        self, context: AgentContext, symbol: str, pool: SolanaDexSnapshot, note: str | None
    ) -> AgentResult:
        registry = services.provider_registry
        strategy = strategy_for(context.trade.profile if context.trade else None)
        cfg = strategy.technical
        supported = [tf for tf in TIMEFRAMES if tf in registry.dex_candles.supported_timeframes]
        requested = context.timeframe
        timeframe: Timeframe = next(
            (tf for tf in supported if tf == requested),
            cfg.default_timeframe if cfg.default_timeframe in supported else supported[0],
        )
        fallback_note = note
        if requested is not None and requested != timeframe:
            fallback_note = (
                f"{requested} candles aren't available for DEX pools; analyzed {timeframe} "
                f"instead (available: {', '.join(supported)})."
            )
        pool_cfg = strategy.technical_pool
        market = context.trade.market if context.trade else None
        explicit = bool(market and (market.requested_dex or market.requested_pool))

        async def analyze_pool(address: str) -> tuple[TechnicalAnalysis | None, str | None]:
            try:
                series = await registry.pool_candles(
                    pool.chain,
                    address,
                    timeframe,
                    cfg.candles_to_fetch,
                    symbol=symbol,
                    canonical_id=pool.canonical_id,  # every candidate pool is this token's
                )
                return analyze_series(series, cfg), None
            except (MarketDataError, InvalidCandleDataError) as exc:
                return None, str(exc)

        analysis, error = await analyze_pool(pool.pair_address)
        pools: dict[str, Any] = {
            "market_pool": {"address": pool.pair_address, "dex": pool.dex},
            "technical_pool": {"address": pool.pair_address, "dex": pool.dex},
            "fallback_reason": None,
            "rejected": [],
            "price_rejected": [],
            "explicit_venue": explicit,
        }
        have = analysis.candle_count if analysis else 0
        if have < pool_cfg.min_candles and explicit:
            pools["rejected"].append(
                "no other pool was considered: the trader asked for this DEX/pool"
            )
        elif have < pool_cfg.min_candles:
            alternates = alternate_pools(pool, pool_cfg)
            pools["rejected"] += alternates.rejected
            pools["price_rejected"] = alternates.price_rejected
            for alt in alternates.pools[: pool_cfg.max_alternates]:
                alt_analysis, alt_error = await analyze_pool(alt.pair_address)
                alt_have = alt_analysis.candle_count if alt_analysis else 0
                if alt_analysis is None or alt_have < pool_cfg.min_candles:
                    pools["rejected"].append(
                        f"{alt.dex} pool {alt.pair_address}: "
                        + (alt_error or f"only {alt_have} consecutive closed {timeframe} candles")
                    )
                    continue
                pools["technical_pool"] = {"address": alt.pair_address, "dex": alt.dex}
                pools["fallback_reason"] = (
                    f"the market pool ({pool.dex} {pool.pair_address}) has only {have} "
                    f"consecutive closed {timeframe} candles (needs {pool_cfg.min_candles}); "
                    f"used {alt.dex} pool {alt.pair_address}: same token, "
                    f"${alt.liquidity_usd or 0:,.0f} liquidity, {alt.txns_24h} trades in 24h, "
                    f"price within {pool_cfg.max_price_divergence_pct:g}% of the market pool"
                )
                analysis = alt_analysis
                break
        if analysis is None:
            return self._failure(symbol, error or "no candles")
        result = self._result(context, analysis, fallback_note)
        result.findings["pools"] = pools
        result.evidence.insert(1, _pool_line(pools))
        return result

    def _result(
        self, context: AgentContext, analysis: TechnicalAnalysis, fallback_note: str | None
    ) -> AgentResult:
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
                "candle_source": {
                    "provider": analysis.provider,
                    "pair": analysis.pair,
                    "requested_timeframe": context.timeframe,
                    "actual_timeframe": analysis.timeframe,
                    "candle_count": analysis.candle_count,
                    "volume_available": analysis.volume_available,
                    "fallback_notes": analysis.fallback_notes,
                },
                "not_analyzed": others,
            },
            evidence=_evidence(analysis, fallback_note, others),
            scenarios=_scenarios(analysis),
            risks=_risks(analysis, fallback_note),
        )


def _pool_line(pools: dict[str, Any]) -> str:
    market, technical = pools["market_pool"], pools["technical_pool"]
    if pools["fallback_reason"]:
        return (
            f"Market pool: {market['dex']} {market['address']}. Technical candles: "
            f"{technical['dex']} {technical['address']}, because {pools['fallback_reason']}."
        )
    line = f"Market and technical pool: {market['dex']} {market['address']}."
    if pools["rejected"]:
        line += " Other pools not used: " + "; ".join(pools["rejected"]) + "."
    return line


def _pool(context: AgentContext) -> SolanaDexSnapshot | None:
    """The primary DEX pool the DEX agent selected for this asset, if any."""
    dex = context.prior_results.get("dex_market")
    if dex is None or dex.status != "ok" or dex.mock:
        return None
    raw = dex.findings.get("snapshot")
    if not isinstance(raw, dict):
        return None
    try:
        return SolanaDexSnapshot.model_validate(raw)
    except ValueError:
        return None


def _label(analysis: TechnicalAnalysis) -> str:
    return f"{analysis.symbol} {analysis.timeframe}"


def _summary(a: TechnicalAnalysis) -> str:
    trend = f"trend: {a.trend.label}" if a.trend.label else "trend unavailable"
    return (
        f"Technical analysis of {_label(a)} from {a.candle_count} {a.provider} candles ({trend})."
    )


def _evidence(a: TechnicalAnalysis, fallback_note: str | None, others: list[str]) -> list[str]:
    pair = f" ({a.pair})" if a.pair else ""
    volume = "with" if a.volume_available else "without"
    lines = [
        f"{_label(a)}: {a.candle_count} candles from {a.provider}{pair} {volume} per-candle "
        f"volume, {a.first_candle_at.strftime(TIME_FORMAT)} to "
        f"{a.last_candle_at.strftime(TIME_FORMAT)} (candle open times); last close "
        f"{usd(a.last_close)}."
    ]
    lines += [f"{note}; used {a.provider} instead." for note in a.fallback_notes]
    lines += [f"{note}." for note in a.notes]
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
        support = ", ".join(_zone(x) for x in lv.support)
        resistance = ", ".join(_zone(x) for x in lv.resistance)
        lines.append(
            f"Approximate support zones: {support or 'none below the last close in the window'}; "
            f"approximate resistance zones: {resistance or 'none above the last close in the window'}."
        )
        if lv.containing is not None:
            lines.append(
                f"The last close ({usd(a.last_close)}) is inside the swing zone "
                f"{_zone(lv.containing)}, so that zone is neither support nor resistance."
            )
    else:
        lines.append(f"Support/resistance unavailable: {lv.unavailable_reason}.")

    lines.append(a.volume_note)
    v = a.volume
    if (
        v.available
        and v.last_volume is not None
        and v.average_volume is not None
        and v.relative_volume is not None
        and v.up_volume_pct is not None
        and v.down_volume_pct is not None
    ):
        lines.append(
            f"Last {a.timeframe} candle volume {v.last_volume:,.4g} {v.unit} is "
            f"{v.relative_volume:.2f}x the {v.lookback}-candle average ({v.average_volume:,.4g} "
            f"{v.unit}); over the last {v.lookback + 1} candles {v.up_volume_pct:.0f}% of "
            f"volume traded on candles that closed up and {v.down_volume_pct:.0f}% on candles "
            "that closed down."
        )
    if others:
        lines.append(f"Only {a.symbol} was analyzed; not analyzed: {', '.join(others)}.")
    return lines


def _zone(lv: Level) -> str:
    zone = usd_zone(lv.lower, lv.upper)
    if zone == f"~{usd(lv.price)}":  # a single swing price: the mean adds nothing
        return f"{zone} ({lv.touches} touch(es))"
    return f"{zone} (mean ~{usd(lv.price)}, {lv.touches} touch(es))"


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
                    f"{a.symbol} keeps trading between the support zone "
                    f"{usd_zone(support.lower, support.upper)} and the resistance zone "
                    f"{usd_zone(resistance.lower, resistance.upper)}."
                ),
                conditions=[f"{tf} closes stay between the two zones"],
                invalidation=(
                    f"A {tf} close below ~{usd(support.lower)} (support zone's lower bound) or "
                    f"above ~{usd(resistance.upper)} (resistance zone's upper bound)."
                ),
            )
        )
    if resistance:
        scenarios.append(
            Scenario(
                name="Break above resistance",
                description=(
                    f"{a.symbol} closes above the resistance zone "
                    f"{usd_zone(resistance.lower, resistance.upper)}."
                ),
                conditions=[
                    f"A {tf} close above ~{usd(resistance.upper)} (the zone's upper bound)",
                    "MACD line staying above its signal line would be consistent with this",
                ],
                invalidation=f"Price falls back below ~{usd(resistance.lower)} (the zone's lower bound).",
            )
        )
    if support:
        scenarios.append(
            Scenario(
                name="Break below support",
                description=(
                    f"{a.symbol} closes below the support zone "
                    f"{usd_zone(support.lower, support.upper)}."
                ),
                conditions=[
                    f"A {tf} close below ~{usd(support.lower)} (the zone's lower bound)",
                    "MACD line staying below its signal line would be consistent with this",
                ],
                invalidation=f"Price recovers above ~{usd(support.upper)} (the zone's upper bound).",
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
    if a.volume.available:
        risks.append(
            Risk(
                description=(
                    f"Volume is {a.provider}'s alone, not market-wide; other venues may differ."
                ),
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
