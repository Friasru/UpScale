"""Deterministic risk review of the evidence other agents already produced.

No network I/O and no model calls: every risk factor, severity, the overall risk level and
the uncertainty level come from the rules below, applied to the structured findings of the
Vision, Technical Analysis, Market and News & Sentiment agents. Nothing is estimated that
those agents didn't report (no probabilities, volatility models, liquidation levels, order
books or volume analysis).

Two kinds of factor are kept apart:

* **risk** factors describe the market/setup itself (large moves, mixed trend, price near
  a level, conflicting or high-impact news). Only these set the overall risk level.
* **uncertainty** factors describe how reliable the evidence is (failed agents, stale or
  missing data, screenshot vs live discrepancies, timeframe fallbacks). They raise
  uncertainty, never the risk level, so missing evidence is not mistaken for high risk.

Overall risk (`overall_risk`):

* ``unknown`` when there is no live substantive evidence: no market snapshot, no technical
  analysis and no classified news for the asset (a screenshot alone is not live evidence).
* ``high`` when two or more risk factors are high, or one is high and a medium-or-high
  factor from a *different* category corroborates it.
* ``medium`` when any risk factor is high or medium.
* ``low`` otherwise (only low-severity factors, or none).

Uncertainty (`uncertainty_level`), from the evidence types price, technical structure and
news:

* ``high`` when overall risk is unknown, any uncertainty factor is high, two or more are
  medium, or two or more evidence types are missing.
* ``medium`` when one uncertainty factor is medium or one evidence type is missing.
* ``low`` otherwise.
"""

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from upscale.formatting import usd, usd_zone
from upscale.schemas import AgentName, AgentResult, Level
from upscale.services.asset_profile import (
    EVIDENCE_LABELS,
    AssetIdentity,
    CryptoAssetProfile,
    Observations,
    build_profile,
    upper_first,
    usable,
)
from upscale.services.asset_registry import DEFAULT_REGISTRY
from upscale.services.market_data import MarketSnapshot
from upscale.services.news import Story
from upscale.services.solana_dex import PoolCandidate, SolanaDexSnapshot, Window
from upscale.services.technical_analysis import Level as PriceLevel
from upscale.services.technical_analysis import TechnicalAnalysis
from upscale.services.vision import ChartVision

RiskCategory = Literal["market", "technical", "news", "screenshot", "data_quality", "dex"]
Affects = Literal["risk", "uncertainty"]
OverallRisk = Literal["low", "medium", "high", "unknown"]
InputStatus = Literal["ok", "no_data", "failed", "not_run", "unreadable"]

INPUT_AGENTS: tuple[AgentName, ...] = ("vision", "technical_analysis", "market", "news_sentiment")
AGENT_LABELS: dict[AgentName, str] = {
    "vision": "Screenshot reading",
    "technical_analysis": "Technical analysis",
    "market": "Live market data",
    "dex_market": "Solana DEX market data",
    "news_sentiment": "News & sentiment",
    "opportunity": "Opportunity decision",
    "risk": "Risk review",
}
_SEVERITY_RANK: dict[Level, int] = {"low": 0, "medium": 1, "high": 2}
_CATEGORY_LABELS: dict[RiskCategory, str] = {
    "market": "market",
    "technical": "technical",
    "news": "news/event",
    "screenshot": "screenshot",
    "data_quality": "data quality",
    "dex": "DEX market",
}
# Macro and regulatory topics, matched on headlines of fresh medium/high-impact stories.
_EVENT_WORDS = re.compile(
    r"\b(?:sec|cftc|regulat\w*|lawsuit|court|ban(?:s|ned)?|sanction\w*|legislat\w*|bill|"
    r"fed|fomc|interest rates?|rate (?:cut|hike)s?|inflation|cpi|tariff\w*|treasury|"
    r"hack(?:ed|s)?|exploit\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RiskConfig:
    """Every threshold used by the rules. Change these, not the logic."""

    move_medium_pct: float = 5.0  # |24h change| at or above this is a medium market risk
    move_high_pct: float = 10.0
    range_medium_pct: float = 8.0  # 24h high-low span, as % of the 24h low
    range_high_pct: float = 15.0
    range_edge_pct: float = 1.0  # price this close to the 24h high/low is flagged (low)
    level_near_pct: float = 1.0  # nearest support/resistance within this: medium
    level_close_pct: float = 2.5  # ... within this: low
    rsi_stretched: float = 70.0  # RSI >= this (or <= 100 - this): low
    rsi_extreme: float = 80.0  # RSI >= this (or <= 100 - this): medium
    screenshot_diff_medium_pct: float = 2.0  # screenshot vs live price
    screenshot_diff_high_pct: float = 10.0
    live_sources_diff_pct: float = 5.0  # last candle close vs market snapshot price
    market_stale_minutes: float = 30.0  # provider's last update older than this at fetch time


class RiskFactor(BaseModel):
    id: str  # stable rule id, e.g. "near_resistance"
    category: RiskCategory
    affects: Affects
    severity: Level
    headline: str  # short clause for the summary
    explanation: str
    source: AgentName  # agent whose evidence triggered the rule
    evidence: list[str] = Field(default_factory=list)
    # What would invalidate or weaken the analysis in light of this factor, when known.
    weakens_if: str | None = None


class InvalidationCondition(BaseModel):
    condition: str
    source: AgentName
    basis: Literal["live_data", "screenshot", "news"]


class InputState(BaseModel):
    agent: AgentName
    status: InputStatus
    detail: str | None = None


class RiskAssessment(BaseModel):
    asset: str | None
    overall_risk: OverallRisk
    overall_reasons: list[str]
    uncertainty_level: Level
    uncertainty_reasons: list[str]
    missing_evidence: list[str]
    factors: list[RiskFactor]
    invalidation_conditions: list[InvalidationCondition]
    conflicting_evidence: list[str]
    data_quality: list[str]
    inputs: list[InputState]
    rules: str
    # What kind of asset this is and which evidence matters for it (None: no profile).
    asset_profile: CryptoAssetProfile | None = None
    # How the profile qualifies this review, e.g. thresholds not calibrated for the asset.
    profile_notes: list[str] = Field(default_factory=list)


# What would invalidate or weaken the analysis, per rule (factors may set a specific one).
WEAKENS: dict[str, str] = {
    "large_recent_move": "Another swing of similar size would quickly outdate the levels described.",
    "at_24h_high": "A move beyond the 24h high takes price outside the recently observed range.",
    "at_24h_low": "A move beyond the 24h low takes price outside the recently observed range.",
    "stale_market_data": "If the current price differs from the reported one, price-based readings shift.",
    "market_volume_missing": "Moves can't be confirmed or questioned against 24h volume.",
    "market_partially_unavailable": "Assets without live data can't be assessed.",
    "mixed_trend": "Without a clear trend, either break of the range can set the direction.",
    "indicator_conflict": "If momentum keeps diverging, the trend classification may change.",
    "rsi_stretched": "A momentum reversal from a stretched reading would weaken the current move.",
    "insufficient_history": "Indicators missing for lack of history can't confirm or contradict the read.",
    "volume_unavailable": "Breakouts or breakdowns can't be checked against volume.",
    "unsupported_timeframe": "Signals on the requested timeframe may differ from those analyzed.",
    "strong_news_conflict": "Whichever story develops further may dominate the backdrop.",
    "news_conflict": "Further coverage may tip the balance either way.",
    "high_impact_news": "Follow-up developments in these stories can change the backdrop.",
    "macro_regulatory_event": "An official decision or data release can shift the whole market.",
    "news_vs_trend": "If the news tone persists, it can work against the current trend.",
    "stale_news": "Newer developments may not be reflected in the news read.",
    "some_stale_news": "Newer developments may not be reflected in the news read.",
    "insufficient_news": "Unreported or very recent events aren't reflected.",
    "unclassified_news": "Unclassified articles could shift the overall news tone.",
    "news_sources_failed": "Coverage from unread sources isn't reflected.",
    "screenshot_asset_mismatch": "Screenshot levels don't apply to the asset being analyzed.",
    "timeframe_mismatch": "Signals on the screenshot's timeframe may differ from those analyzed.",
    "screenshot_inferred": "Inferred readings may be wrong; visible values are more reliable.",
    "screenshot_unreadable": "Unread parts of the chart may contain relevant information.",
    "screenshot_not_chart": "Nothing from the screenshot could be used.",
    "screenshot_partially_read": "Unread screenshots may contain relevant information.",
    "dex_low_liquidity": "A liquidity withdrawal or large sell can move the price sharply.",
    "dex_new_pool": "Price discovery in a new pool is unstable; early levels may not hold.",
    "dex_extreme_move": "A move this fast can reverse just as quickly.",
    "dex_flow_imbalance": "One-sided flow can flip abruptly when early buyers or sellers exit.",
    "dex_liquidity_missing": "Without reported liquidity, exit cost can't be judged.",
    "dex_competing_pools": "Prices may differ between pools; the chosen primary may not be the real market.",
}

RULES = (
    "Overall risk is unknown without live evidence (market data, technical analysis or "
    "classified news); high with two high risk factors, or one high factor corroborated by "
    "a medium/high factor in another category; medium with any medium or high risk factor; "
    "otherwise low. Data-quality and screenshot issues raise uncertainty, not risk."
)


# --- Parsing the other agents' findings ----------------------------------------------------


@dataclass
class NewsInput:
    asset: str | None
    overall: str
    stories: list[Story]
    asset_story_count: int
    conflicts: list[str]
    failed_sources: list[str]

    @property
    def focus(self) -> list[Story]:
        scope = "asset" if self.asset is not None else "market"
        return [s for s in self.stories if s.scope == scope]


@dataclass
class Inputs:
    asset: str | None
    states: dict[AgentName, InputState]
    chart: ChartVision | None = None
    vision_failed_images: int = 0
    technical: TechnicalAnalysis | None = None
    technical_extra: Mapping[str, Any] | None = None
    snapshot: MarketSnapshot | None = None
    market_unavailable: list[str] | None = None
    news: NewsInput | None = None
    other_failures: list[AgentResult] | None = None
    dex: SolanaDexSnapshot | None = None
    dex_candidates: list[PoolCandidate] | None = None  # pools found when none was usable


def collect_inputs(prior: Mapping[AgentName, AgentResult], asset: str | None) -> Inputs:
    """Typed view of what each input agent reported. Anything that doesn't parse is marked
    unreadable rather than guessed at."""
    inputs = Inputs(asset=asset, states={})
    for name in INPUT_AGENTS:
        result = prior.get(name)
        if result is None:
            inputs.states[name] = InputState(agent=name, status="not_run")
            continue
        if result.mock:
            inputs.states[name] = InputState(agent=name, status="not_run", detail="mock output")
            continue
        if result.status == "error":
            inputs.states[name] = InputState(agent=name, status="failed", detail=result.error)
            continue
        try:
            found = _PARSERS[name](inputs, result.findings)
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            inputs.states[name] = InputState(
                agent=name, status="unreadable", detail=type(exc).__name__
            )
            continue
        inputs.states[name] = InputState(
            agent=name,
            status="ok" if found else "no_data",
            detail=None if found else result.summary,
        )
    if "dex_market" in prior:
        _collect_dex(inputs, prior["dex_market"])
    return inputs


def _collect_dex(inputs: Inputs, result: AgentResult) -> None:
    """DEX data is only reviewed when the DEX agent ran, so other assets' reviews are
    unchanged."""
    if result.mock or result.status == "error":
        status: InputStatus = "not_run" if result.mock else "failed"
        inputs.states["dex_market"] = InputState(
            agent="dex_market", status=status, detail=result.error
        )
        return
    try:
        raw = result.findings.get("snapshot")
        snapshot = SolanaDexSnapshot.model_validate(raw) if raw is not None else None
        candidates = [
            PoolCandidate.model_validate(c) for c in result.findings.get("candidates", [])
        ]
    except (ValidationError, TypeError, ValueError) as exc:
        inputs.states["dex_market"] = InputState(
            agent="dex_market", status="unreadable", detail=type(exc).__name__
        )
        return
    mint = result.findings.get("mint")
    symbol = snapshot.symbol if snapshot is not None else None
    if not _dex_belongs(inputs.asset, mint if isinstance(mint, str) else None, symbol):
        inputs.states["dex_market"] = InputState(
            agent="dex_market",
            status="no_data",
            detail=f"DEX data is for {mint}, not {inputs.asset}",
        )
        return
    inputs.dex = snapshot
    inputs.dex_candidates = candidates if snapshot is None else None
    inputs.states["dex_market"] = InputState(
        agent="dex_market",
        status="ok" if snapshot is not None else "no_data",
        detail=None if snapshot is not None else result.summary,
    )


def _dex_belongs(asset: str | None, mint: str | None, symbol: str | None) -> bool:
    """Whether DEX data fetched for `mint` is about `asset` (the mint itself, the symbol the
    DEX reported for it, or a registered asset whose mint it is)."""
    if mint is None:
        return False
    if asset is None or asset in (mint, symbol):
        return True
    entry = DEFAULT_REGISTRY.by_symbol(asset)
    return entry is not None and entry.chain == "solana" and entry.address == mint


def _parse_vision(inputs: Inputs, findings: Mapping[str, Any]) -> bool:
    charts = [ChartVision.model_validate(c) for c in findings.get("charts", [])]
    inputs.chart = next((c for c in charts if c.reading.is_price_chart), None)
    inputs.vision_failed_images = len(findings.get("failed", []))
    return inputs.chart is not None


def _parse_technical(inputs: Inputs, findings: Mapping[str, Any]) -> bool:
    if "candle_count" not in findings:
        return False  # the agent ran without an asset
    analysis = TechnicalAnalysis.model_validate(findings)
    if inputs.asset is not None and analysis.symbol != inputs.asset:
        return False
    inputs.technical = analysis
    inputs.technical_extra = findings
    return True


def _parse_market(inputs: Inputs, findings: Mapping[str, Any]) -> bool:
    snapshots = [MarketSnapshot.model_validate(s) for s in findings.get("snapshots", [])]
    inputs.snapshot = next(
        (s for s in snapshots if inputs.asset is None or s.symbol == inputs.asset), None
    )
    inputs.market_unavailable = [
        f"{u['symbol']} ({u['reason']})" for u in findings.get("unavailable", [])
    ]
    return inputs.snapshot is not None


def _parse_news(inputs: Inputs, findings: Mapping[str, Any]) -> bool:
    reports = findings.get("reports", [])
    report = next((r for r in reports if r.get("asset") == inputs.asset), None)
    if report is None and reports:
        report = reports[0]
    if report is None:
        return False
    inputs.news = NewsInput(
        asset=report.get("asset"),
        overall=str(report["overall_news_sentiment"]),
        stories=[Story.model_validate(s) for s in report.get("stories", [])],
        asset_story_count=int(report.get("asset_story_count", 0)),
        conflicts=[str(c) for c in report.get("conflicting_evidence", [])],
        failed_sources=[f"{f['source']} ({f['reason']})" for f in report.get("failed_sources", [])],
    )
    return any(s.sentiment is not None for s in inputs.news.focus)


_PARSERS = {
    "vision": _parse_vision,
    "technical_analysis": _parse_technical,
    "market": _parse_market,
    "news_sentiment": _parse_news,
}


# --- Asset profile ---------------------------------------------------------------------------

# Evidence the rules below already assess (and report as missing) themselves; the profile
# adds whatever else its category requires.
BUILTIN_EVIDENCE = frozenset({"technical_structure", "market_snapshot", "news"})
_TICKER_KEYED_AGENTS: tuple[AgentName, ...] = ("technical_analysis", "market", "news_sentiment")


def observations(inputs: Inputs) -> Observations:
    return Observations(
        snapshot=inputs.snapshot,
        technical=inputs.technical,
        dex=inputs.dex,
        states={name: (s.status, s.detail) for name, s in inputs.states.items()},
    )


def profile_from_results(
    prior: Mapping[AgentName, AgentResult],
    asset: str | None,
    identity: AssetIdentity | None = None,
) -> CryptoAssetProfile | None:
    """The asset's profile, using what this turn's agents reported about it.

    An explicit `identity` (e.g. chain + mint) wins over the bare ticker when it names the
    same asset.
    """
    target: AssetIdentity | str | None = asset
    if identity is not None and (asset is None or identity.symbol in (None, asset)):
        target = identity
    return build_profile(target, observations(collect_inputs(prior, asset)))


def apply_profile(inputs: Inputs, profile: CryptoAssetProfile | None) -> Inputs:
    """Drop evidence the profile says doesn't belong to this asset or can't be trusted.

    * Ticker-keyed data (Kraken, CoinGecko, news) for a contract token whose ticker isn't
      proven to be its own: it may describe another token with the same ticker.
    * Technical analysis on too little history for the asset's category.
    * DEX data fetched for a different mint than the asset's.
    """
    if profile is None:
        return inputs
    if inputs.dex is not None and not (
        profile.chain == "solana" and profile.address == inputs.dex.mint
    ):
        inputs.states["dex_market"] = InputState(
            agent="dex_market",
            status="no_data",
            detail=f"DEX data is for {inputs.dex.canonical_id}, not {profile.canonical_id}",
        )
        inputs.dex = None
    if not profile.ticker_data_attributable:
        detail = (
            f"ticker-matched data can't be tied to {profile.canonical_id}; another token may "
            f"share the ticker {profile.symbol}"
        )
        for name in _TICKER_KEYED_AGENTS:
            if inputs.states[name].status == "ok":
                inputs.states[name] = InputState(agent=name, status="no_data", detail=detail)
        inputs.technical = inputs.technical_extra = inputs.snapshot = inputs.news = None
    elif not profile.technical_usable and inputs.technical is not None:
        status = profile.evidence_status("technical_structure")
        inputs.states["technical_analysis"] = InputState(
            agent="technical_analysis",
            status="no_data",
            detail=status.reason if status else "not enough candle history",
        )
        inputs.technical = inputs.technical_extra = None
    return inputs


def risk_config_for(
    profile: CryptoAssetProfile | None, base: RiskConfig | None = None
) -> RiskConfig:
    """Risk thresholds for this asset: the base config plus its category's overrides."""
    cfg = base or RiskConfig()
    if profile is None or not profile.risk_overrides:
        return cfg
    return dataclasses.replace(cfg, **profile.risk_overrides)


def profile_factors(profile: CryptoAssetProfile) -> tuple[list[RiskFactor], list[str]]:
    """Required evidence for this kind of asset that UpScale can't provide, as an
    uncertainty factor, plus the matching missing-evidence lines."""
    gaps = [
        e
        for e in profile.evidence
        if e.requirement == "required" and e.evidence not in BUILTIN_EVIDENCE and not usable(e)
    ]
    if not gaps:
        return [], []
    labels = [EVIDENCE_LABELS[e.evidence] for e in gaps]
    critical = any(e.decision_critical for e in gaps)
    factor = RiskFactor(
        id="profile_evidence_unavailable",
        category="data_quality",
        affects="uncertainty",
        severity="high" if critical else "medium",
        headline=f"evidence essential for a {profile.category_label} is unavailable",
        explanation=(
            f"For a {profile.category_label}, {', '.join(labels)} "
            f"{'is' if len(labels) == 1 else 'are'} required, but UpScale can't provide "
            f"{'it' if len(labels) == 1 else 'them'} for {profile.symbol} yet."
        ),
        source="risk",
        evidence=[f"{EVIDENCE_LABELS[e.evidence]}: {e.reason}" for e in gaps],
        weakens_if="Unassessed liquidity, holder or token-safety risks can dominate price action.",
    )
    missing = [f"{upper_first(EVIDENCE_LABELS[e.evidence])} ({e.reason})" for e in gaps]
    return [factor], missing


def profile_notes(profile: CryptoAssetProfile) -> list[str]:
    if profile.risk_thresholds_calibrated:
        return []
    return [
        f"{profile.symbol} is profiled as a {profile.category_label}. Move and range "
        "thresholds were designed for major crypto and haven't been adapted to this kind "
        "of asset yet, so market-move severities here are provisional."
    ]


# --- Rules ---------------------------------------------------------------------------------


def _max_severity(*levels: Level | None) -> Level | None:
    present: list[Level] = [lv for lv in levels if lv is not None]
    if not present:
        return None
    return max(present, key=lambda lv: _SEVERITY_RANK[lv])


def _threshold(value: float, medium: float, high: float) -> Level | None:
    return "high" if value >= high else "medium" if value >= medium else None


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def market_factors(s: MarketSnapshot, cfg: RiskConfig) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    move = abs(s.change_24h_pct) if s.change_24h_pct is not None else None
    span = (
        100 * (s.high_24h_usd - s.low_24h_usd) / s.low_24h_usd
        if s.high_24h_usd is not None and s.low_24h_usd is not None and s.low_24h_usd > 0
        else None
    )
    move_level = _threshold(move, cfg.move_medium_pct, cfg.move_high_pct) if move else None
    span_level = _threshold(span, cfg.range_medium_pct, cfg.range_high_pct) if span else None
    severity = _max_severity(move_level, span_level)
    if severity is not None:
        parts = []
        if s.change_24h_pct is not None:
            parts.append(f"moved {s.change_24h_pct:+.1f}% in 24h")
        if span is not None:
            parts.append(f"its 24h range spans {_pct(span)}")
        factors.append(
            RiskFactor(
                id="large_recent_move",
                category="market",
                affects="risk",
                severity=severity,
                headline=f"{s.symbol} {' and '.join(parts)}",
                explanation=(
                    f"{s.symbol} {' and '.join(parts)} ({s.provider}). Large recent swings mean "
                    "levels and signals can be overrun quickly."
                ),
                source="market",
                evidence=[f"Price {usd(s.price_usd)}, 24h change {s.change_24h_pct}%"],
            )
        )
    if s.high_24h_usd is not None and s.low_24h_usd is not None:
        edge = None
        if 100 * (s.high_24h_usd - s.price_usd) / s.price_usd <= cfg.range_edge_pct:
            edge = ("high", s.high_24h_usd)
        elif 100 * (s.price_usd - s.low_24h_usd) / s.price_usd <= cfg.range_edge_pct:
            edge = ("low", s.low_24h_usd)
        if edge:
            factors.append(
                RiskFactor(
                    id=f"at_24h_{edge[0]}",
                    category="market",
                    affects="risk",
                    severity="low",
                    headline=f"{s.symbol} is trading at its 24h {edge[0]}",
                    explanation=(
                        f"{s.symbol} ({usd(s.price_usd)}) is within {_pct(cfg.range_edge_pct)} of "
                        f"its 24h {edge[0]} ({usd(edge[1])}); conditions at the edge of the "
                        "recent range can change quickly."
                    ),
                    source="market",
                )
            )
    return factors


def market_quality_factors(
    s: MarketSnapshot | None, unavailable: Sequence[str], cfg: RiskConfig
) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    if s is not None and s.last_updated is not None:
        lag = (s.fetched_at - s.last_updated).total_seconds() / 60
        if lag > cfg.market_stale_minutes:
            factors.append(
                RiskFactor(
                    id="stale_market_data",
                    category="data_quality",
                    affects="uncertainty",
                    severity="medium",
                    headline="live market data was stale when fetched",
                    explanation=(
                        f"{s.provider} last updated {s.symbol} {round(lag)} minutes before it "
                        "was fetched, so the price may not be current."
                    ),
                    source="market",
                )
            )
    if s is not None and s.volume_24h_usd is None:
        factors.append(
            RiskFactor(
                id="market_volume_missing",
                category="data_quality",
                affects="uncertainty",
                severity="low",
                headline="24h volume is unavailable",
                explanation=f"{s.provider} did not report 24h volume for {s.symbol}.",
                source="market",
            )
        )
    if unavailable:
        factors.append(
            RiskFactor(
                id="market_partially_unavailable",
                category="data_quality",
                affects="uncertainty",
                severity="low",
                headline="some market data was unavailable",
                explanation=f"Live market data was unavailable for {', '.join(unavailable)}.",
                source="market",
            )
        )
    return factors


def live_consistency_factors(
    s: MarketSnapshot | None, t: TechnicalAnalysis | None, cfg: RiskConfig
) -> list[RiskFactor]:
    if s is None or t is None or s.symbol != t.symbol:
        return []
    diff = 100 * abs(t.last_close - s.price_usd) / s.price_usd
    if diff < cfg.live_sources_diff_pct:
        return []
    return [
        RiskFactor(
            id="live_sources_disagree",
            category="data_quality",
            affects="uncertainty",
            severity="medium",
            headline="live price sources disagree",
            explanation=(
                f"The last {t.timeframe} close ({usd(t.last_close)}, {t.provider}) differs from "
                f"the live snapshot price ({usd(s.price_usd)}, {s.provider}) by {_pct(diff)}, "
                "so the candle data may be outdated or refer to a different market."
            ),
            source="technical_analysis",
            weakens_if="Levels and indicators computed from the candles may not match the live price.",
        )
    ]


def _nearest(levels: Sequence[PriceLevel]) -> PriceLevel | None:
    return levels[0] if levels else None


def _break_condition(kind: str, level: PriceLevel, timeframe: str) -> str:
    """The close that breaks a zone: above its upper bound (resistance), below its lower (support)."""
    side, bound = ("above", level.upper) if kind == "resistance" else ("below", level.lower)
    return (
        f"{kind.capitalize()} zone {usd_zone(level.lower, level.upper)}; a {timeframe} close "
        f"{side} ~{usd(bound)} would break it."
    )


def technical_factors(
    a: TechnicalAnalysis, extra: Mapping[str, Any], cfg: RiskConfig
) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    label = f"{a.symbol} {a.timeframe}"
    macd = next((i for i in a.indicators if i.kind == "macd" and i.available), None)
    rsi = next((i for i in a.indicators if i.kind == "rsi" and i.available), None)

    if a.trend.available and a.trend.label == "mixed":
        factors.append(
            RiskFactor(
                id="mixed_trend",
                category="technical",
                affects="risk",
                severity="medium",
                headline="the technical trend is mixed",
                explanation=(
                    f"The rule-based {label} trend is mixed ({' '.join(a.trend.reasons)}), so "
                    "there is no clear directional structure."
                ),
                source="technical_analysis",
            )
        )
    if (
        a.trend.label in ("uptrend", "downtrend")
        and macd is not None
        and macd.value is not None
        and macd.signal is not None
    ):
        against = (a.trend.label == "uptrend" and macd.value < macd.signal) or (
            a.trend.label == "downtrend" and macd.value > macd.signal
        )
        if against:
            side = "below" if macd.value < macd.signal else "above"
            factors.append(
                RiskFactor(
                    id="indicator_conflict",
                    category="technical",
                    affects="risk",
                    severity="medium",
                    headline=f"MACD conflicts with the {a.trend.label}",
                    explanation=(
                        f"{label} is classified as a {a.trend.label}, but the MACD line is {side} "
                        "its signal line, so momentum and trend disagree."
                    ),
                    source="technical_analysis",
                )
            )
    if rsi is not None and rsi.value is not None:
        hi, ext = cfg.rsi_stretched, cfg.rsi_extreme
        if rsi.value >= hi or rsi.value <= 100 - hi:
            zone = "overbought" if rsi.value >= hi else "oversold"
            extreme = rsi.value >= ext or rsi.value <= 100 - ext
            factors.append(
                RiskFactor(
                    id="rsi_stretched",
                    category="technical",
                    affects="risk",
                    severity="medium" if extreme else "low",
                    headline=f"RSI is stretched ({rsi.value:.0f})",
                    explanation=(
                        f"{rsi.name} on {label} is {rsi.value:.1f}, a zone commonly described as "
                        f"{zone}; stretched momentum readings can reverse."
                    ),
                    source="technical_analysis",
                )
            )

    if a.levels.available:
        for kind, level in (
            ("resistance", _nearest(a.levels.resistance)),
            ("support", _nearest(a.levels.support)),
        ):
            if level is None:
                continue
            distance = abs(level.distance_pct)
            if distance <= cfg.level_close_pct:
                factors.append(
                    RiskFactor(
                        id=f"near_{kind}",
                        category="technical",
                        affects="risk",
                        severity="medium" if distance <= cfg.level_near_pct else "low",
                        headline=(
                            f"{a.symbol} is near {kind} (~{usd(level.price)}, "
                            f"{_pct(distance)} away)"
                        ),
                        explanation=(
                            f"The last {a.timeframe} close ({usd(a.last_close)}) is {_pct(distance)} "
                            f"from the approximate {kind} zone {usd_zone(level.lower, level.upper)} "
                            f"(mean ~{usd(level.price)}); price often reacts around such zones in "
                            "either direction."
                        ),
                        source="technical_analysis",
                        weakens_if=_break_condition(kind, level, a.timeframe),
                    )
                )
        inside = a.levels.containing
        if inside is not None:
            zone = usd_zone(inside.lower, inside.upper)
            factors.append(
                RiskFactor(
                    id="inside_level_zone",
                    category="technical",
                    affects="risk",
                    severity="medium",
                    headline=f"{a.symbol} is inside a swing zone ({zone})",
                    explanation=(
                        f"The last {a.timeframe} close ({usd(a.last_close)}) sits inside the "
                        f"approximate swing zone {zone} ({inside.touches} swing point(s)), so that "
                        "zone is neither clear support nor clear resistance; price often reacts "
                        "around such zones in either direction."
                    ),
                    source="technical_analysis",
                    weakens_if=(
                        f"A {a.timeframe} close above ~{usd(inside.upper)} or below "
                        f"~{usd(inside.lower)} would take price out of that zone."
                    ),
                )
            )

    unavailable = [i.name for i in a.indicators if not i.available]
    if not a.trend.available:
        unavailable.append("trend")
    if not a.levels.available:
        unavailable.append("support/resistance")
    if unavailable:
        structural = not a.trend.available or not a.levels.available
        factors.append(
            RiskFactor(
                id="insufficient_history",
                category="technical",
                affects="uncertainty",
                severity="medium" if structural else "low",
                headline=f"only {a.candle_count} candles of history",
                explanation=(
                    f"Only {a.candle_count} {a.timeframe} candles were available, so these could "
                    f"not be calculated: {', '.join(unavailable)}."
                ),
                source="technical_analysis",
            )
        )
    if not a.volume_available:
        factors.append(
            RiskFactor(
                id="volume_unavailable",
                category="technical",
                affects="uncertainty",
                severity="low",
                headline="per-candle volume is unavailable",
                explanation=(
                    f"{a.provider} does not supply per-candle volume, so moves could not be "
                    "checked against volume."
                ),
                source="technical_analysis",
            )
        )
    note = extra.get("timeframe_note")
    if note and extra.get("requested_timeframe_source") != "screenshot":
        factors.append(
            RiskFactor(
                id="unsupported_timeframe",
                category="data_quality",
                affects="uncertainty",
                severity="medium",
                headline=(
                    f"the requested {extra.get('requested_timeframe')} timeframe could only be "
                    f"analyzed using {a.timeframe} candles"
                ),
                explanation=str(note),
                source="technical_analysis",
            )
        )
    return factors


def _fresh(stories: Sequence[Story]) -> list[Story]:
    return [s for s in stories if not s.stale]


def _quote(s: Story) -> str:
    return f"'{s.title}' ({s.source})"


def news_factors(n: NewsInput, technical: TechnicalAnalysis | None) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    subject = n.asset or "the crypto market"
    focus = n.focus
    fresh = _fresh(n.stories)
    fresh_focus = _fresh(focus)

    high_bull = [s for s in fresh_focus if s.sentiment == "bullish" and s.impact == "high"]
    high_bear = [s for s in fresh_focus if s.sentiment == "bearish" and s.impact == "high"]
    if high_bull and high_bear:
        factors.append(
            RiskFactor(
                id="strong_news_conflict",
                category="news",
                affects="risk",
                severity="high",
                headline="high-impact news points in opposite directions",
                explanation=(
                    f"Recent high-impact coverage of {subject} is both bullish "
                    f"({_quote(high_bull[0])}) and bearish ({_quote(high_bear[0])}), so the "
                    "news backdrop is strongly contested."
                ),
                source="news_sentiment",
            )
        )
    elif n.conflicts:
        factors.append(
            RiskFactor(
                id="news_conflict",
                category="news",
                affects="risk",
                severity="medium",
                headline="recent news is conflicting",
                explanation=f"Recent coverage of {subject} conflicts: {n.conflicts[0]}",
                source="news_sentiment",
            )
        )

    high_impact = [s for s in fresh if s.impact == "high"]
    if high_impact and not (high_bull and high_bear):
        titles = "; ".join(_quote(s) for s in high_impact[:2])
        factors.append(
            RiskFactor(
                id="high_impact_news",
                category="news",
                affects="risk",
                severity="medium",
                headline=f"{len(high_impact)} recent high-impact news item(s)",
                explanation=(
                    f"Recent coverage rated high potential impact: {titles}. Developments in "
                    "these stories can change the backdrop quickly."
                ),
                source="news_sentiment",
            )
        )

    events = [s for s in fresh if s.impact in ("medium", "high") and _EVENT_WORDS.search(s.title)]
    if events:
        factors.append(
            RiskFactor(
                id="macro_regulatory_event",
                category="news",
                affects="risk",
                severity="medium",
                headline="regulatory or macro news is in play",
                explanation=(
                    "Recent regulatory/macro coverage may affect the backdrop: "
                    + "; ".join(_quote(s) for s in events[:2])
                    + "."
                ),
                source="news_sentiment",
            )
        )

    if technical is not None and technical.trend.label in ("uptrend", "downtrend"):
        opposed = (n.overall == "bearish" and technical.trend.label == "uptrend") or (
            n.overall == "bullish" and technical.trend.label == "downtrend"
        )
        if opposed:
            factors.append(
                RiskFactor(
                    id="news_vs_trend",
                    category="news",
                    affects="risk",
                    severity="medium",
                    headline=(
                        f"{n.overall} news runs against the {technical.timeframe} "
                        f"{technical.trend.label}"
                    ),
                    explanation=(
                        f"News sentiment for {subject} is {n.overall} while the rule-based "
                        f"{technical.timeframe} trend is a {technical.trend.label}; the evidence "
                        "does not line up."
                    ),
                    source="news_sentiment",
                )
            )

    if focus and not fresh_focus:
        factors.append(
            RiskFactor(
                id="stale_news",
                category="news",
                affects="uncertainty",
                severity="medium",
                headline="all relevant news is stale",
                explanation=f"Every relevant story about {subject} is stale and may be outdated.",
                source="news_sentiment",
            )
        )
    elif any(s.stale for s in n.stories):
        factors.append(
            RiskFactor(
                id="some_stale_news",
                category="news",
                affects="uncertainty",
                severity="low",
                headline="some news is stale",
                explanation="Some news items are stale and were given less weight.",
                source="news_sentiment",
            )
        )
    if not n.stories or (n.asset is not None and n.asset_story_count == 0):
        what = (
            "no recent relevant news" if not n.stories else f"no news specifically about {subject}"
        )
        factors.append(
            RiskFactor(
                id="insufficient_news",
                category="news",
                affects="uncertainty",
                severity="low",
                headline=what,
                explanation=f"There was {what}, so news coverage is insufficient.",
                source="news_sentiment",
            )
        )
    if any(s.sentiment is None for s in focus):
        factors.append(
            RiskFactor(
                id="unclassified_news",
                category="data_quality",
                affects="uncertainty",
                severity="low",
                headline="some articles are unclassified",
                explanation="Some articles could not be classified, so news sentiment is partial.",
                source="news_sentiment",
            )
        )
    if n.failed_sources:
        factors.append(
            RiskFactor(
                id="news_sources_failed",
                category="data_quality",
                affects="uncertainty",
                severity="low",
                headline="some news sources could not be read",
                explanation=f"Some news sources could not be read: {', '.join(n.failed_sources)}.",
                source="news_sentiment",
            )
        )
    return factors


def screenshot_factors(inputs: Inputs, cfg: RiskConfig) -> list[RiskFactor]:
    chart = inputs.chart
    factors: list[RiskFactor] = []
    if chart is None:
        return factors
    r = chart.reading
    shown = r.asset.symbol
    analyzed = inputs.asset
    same_asset = shown is None or analyzed is None or shown == analyzed
    if not same_asset:
        factors.append(
            RiskFactor(
                id="screenshot_asset_mismatch",
                category="screenshot",
                affects="uncertainty",
                severity="medium",
                headline=f"the screenshot shows {shown}, not {analyzed}",
                explanation=(
                    f"The screenshot shows {shown}, but the live analysis is of {analyzed}; the "
                    "screenshot's values were not combined with it."
                ),
                source="vision",
            )
        )

    live, live_source = _live_price(inputs)
    shot = r.displayed_price.value
    if same_asset and shot is not None and live is not None:
        diff = 100 * abs(shot - live) / live
        severity = _threshold(diff, cfg.screenshot_diff_medium_pct, cfg.screenshot_diff_high_pct)
        if severity is not None:
            factors.append(
                RiskFactor(
                    id="screenshot_price_discrepancy",
                    category="screenshot",
                    affects="uncertainty",
                    severity=severity,
                    headline=f"the screenshot price differs from live data by {_pct(diff)}",
                    explanation=(
                        f"The screenshot shows ~{usd(shot)} but the live price is {usd(live)} "
                        f"({live_source}), {_pct(diff)} apart. The screenshot may be stale: live "
                        "data takes precedence, and the screenshot's levels and indicator values "
                        "may no longer apply."
                    ),
                    source="vision",
                    weakens_if="The screenshot is older than the live data.",
                )
            )

    shot_tf = chart.normalized_timeframe or r.timeframe.label
    t = inputs.technical
    if t is not None and shot_tf and shot_tf != t.timeframe and same_asset:
        fallback = (inputs.technical_extra or {}).get("requested_timeframe_source") == "screenshot"
        why = (
            f"{shot_tf} candles aren't available from the market data provider"
            if fallback
            else f"{t.timeframe} was the timeframe requested"
        )
        factors.append(
            RiskFactor(
                id="timeframe_mismatch",
                category="screenshot",
                affects="uncertainty",
                severity="medium",
                headline=(
                    f"the {shot_tf} screenshot could only be compared with {t.timeframe} candles"
                    if fallback
                    else f"the screenshot is {shot_tf} but live analysis used {t.timeframe}"
                ),
                explanation=(
                    f"The screenshot is a {shot_tf} chart, but live technical analysis used "
                    f"{t.timeframe} candles because {why}; the two describe different timeframes."
                ),
                source="technical_analysis",
            )
        )

    inferred = [
        name
        for name, basis in (
            ("asset", r.asset.basis),
            ("timeframe", r.timeframe.basis),
            ("price", r.displayed_price.basis),
        )
        if basis == "inferred"
    ]
    inferred_levels = sum(
        lv.basis == "inferred" for lv in [*r.support_levels, *r.resistance_levels]
    )
    if inferred_levels:
        inferred.append(f"{inferred_levels} level(s)")
    if inferred:
        factors.append(
            RiskFactor(
                id="screenshot_inferred",
                category="screenshot",
                affects="uncertainty",
                severity="low",
                headline="some screenshot readings are inferred",
                explanation=(
                    f"Some screenshot readings were inferred rather than clearly visible: "
                    f"{', '.join(inferred)}."
                ),
                source="vision",
            )
        )
    unreadable = [u.rstrip(". ") for u in [*r.uncertainties, *chart.discarded]]
    if unreadable:
        factors.append(
            RiskFactor(
                id="screenshot_unreadable",
                category="screenshot",
                affects="uncertainty",
                severity="low",
                headline="parts of the screenshot could not be read",
                explanation="Parts of the screenshot could not be read: "
                + "; ".join(unreadable[:3]),
                source="vision",
            )
        )
    return factors


def _live_price(inputs: Inputs) -> tuple[float | None, str | None]:
    if inputs.snapshot is not None:
        return inputs.snapshot.price_usd, f"{inputs.snapshot.provider} snapshot"
    if inputs.technical is not None:
        t = inputs.technical
        return t.last_close, f"last {t.timeframe} close from {t.provider}"
    return None, None


def input_factors(inputs: Inputs) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    for name, state in inputs.states.items():
        label = AGENT_LABELS[name]
        if state.status in ("failed", "unreadable"):
            what = "failed" if state.status == "failed" else "returned output that couldn't be read"
            factors.append(
                RiskFactor(
                    id=f"{name}_{state.status}",
                    category="data_quality",
                    affects="uncertainty",
                    severity="medium",
                    headline=f"{label.lower()} {what}",
                    explanation=f"{label} {what}"
                    + (f" ({state.detail})." if state.detail else "."),
                    source=name,
                )
            )
    if inputs.states["vision"].status in ("ok", "no_data"):
        if inputs.chart is None:
            factors.append(
                RiskFactor(
                    id="screenshot_not_chart",
                    category="screenshot",
                    affects="uncertainty",
                    severity="medium",
                    headline="the screenshot could not be read as a price chart",
                    explanation="The screenshot was not recognized as a price chart.",
                    source="vision",
                )
            )
        elif inputs.vision_failed_images:
            factors.append(
                RiskFactor(
                    id="screenshot_partially_read",
                    category="screenshot",
                    affects="uncertainty",
                    severity="low",
                    headline="some screenshots could not be read",
                    explanation=f"{inputs.vision_failed_images} screenshot(s) could not be read.",
                    source="vision",
                )
            )
    return factors


# --- Solana DEX market ----------------------------------------------------------------------


@dataclass(frozen=True)
class DexRiskConfig:
    """Thresholds for DEX-traded tokens. Separate from `RiskConfig`: moves and depth that
    are extreme for BTC are ordinary for a small DEX token, and vice versa."""

    liquidity_high_usd: float = 25_000.0  # primary pool liquidity below this: high
    liquidity_medium_usd: float = 100_000.0  # ... below this: medium
    pool_age_high_hours: float = 24.0  # primary pool younger than this: high
    pool_age_medium_hours: float = 72.0  # ... younger than this: medium
    move_m5_medium_pct: float = 15.0  # |5-minute price change| at or above this: medium
    move_m5_high_pct: float = 30.0
    move_h1_medium_pct: float = 25.0  # |1-hour price change|
    move_h1_high_pct: float = 50.0
    # Buy (or sell) share of trades at or above this: imbalanced. Needs enough trades.
    imbalance_medium_share: float = 0.80
    imbalance_high_share: float = 0.90
    imbalance_min_txns: int = 20
    imbalance_windows: tuple[Window, ...] = ("h1", "h24")  # first with enough trades is used

    def __post_init__(self) -> None:
        if not 0 < self.liquidity_high_usd <= self.liquidity_medium_usd:
            raise ValueError("liquidity thresholds must satisfy 0 < high <= medium")
        if not 0 < self.pool_age_high_hours <= self.pool_age_medium_hours:
            raise ValueError("pool age thresholds must satisfy 0 < high <= medium")
        if not (
            0 < self.move_m5_medium_pct <= self.move_m5_high_pct
            and 0 < self.move_h1_medium_pct <= self.move_h1_high_pct
        ):
            raise ValueError("move thresholds must satisfy 0 < medium <= high")
        if not 0.5 < self.imbalance_medium_share <= self.imbalance_high_share <= 1:
            raise ValueError("imbalance shares must satisfy 0.5 < medium <= high <= 1")
        if self.imbalance_min_txns < 1:
            raise ValueError("imbalance_min_txns must be positive")


NO_SAFETY_CHECK = (
    "Token authorities and holder concentration aren't checked yet, so this is not "
    "rug-pull detection."
)


def _below(value: float, high: float, medium: float) -> Level | None:
    return "high" if value < high else "medium" if value < medium else None


def dex_factors(d: SolanaDexSnapshot, cfg: DexRiskConfig) -> list[RiskFactor]:
    """Risk from the token's primary DEX pool, relative to DEX-token thresholds."""
    factors: list[RiskFactor] = []
    label = d.symbol or d.canonical_id
    pool = f"{d.dex} pool {d.pair_address}"
    sev = _below(d.liquidity_usd, cfg.liquidity_high_usd, cfg.liquidity_medium_usd)
    if sev is not None:
        factors.append(
            RiskFactor(
                id="dex_low_liquidity",
                category="dex",
                affects="risk",
                severity=sev,
                headline=f"{label}'s primary pool holds only {usd(d.liquidity_usd)} of liquidity",
                explanation=(
                    f"The primary {pool} holds {usd(d.liquidity_usd)} of liquidity (below "
                    f"{usd(cfg.liquidity_high_usd if sev == 'high' else cfg.liquidity_medium_usd)}"
                    "): a modest sell can move the price sharply and exits may be costly. "
                    + NO_SAFETY_CHECK
                ),
                source="dex_market",
                evidence=[f"liquidity {usd(d.liquidity_usd)} ({d.provider})"],
            )
        )
    if d.pool_age_hours is not None:
        sev = _below(d.pool_age_hours, cfg.pool_age_high_hours, cfg.pool_age_medium_hours)
        if sev is not None:
            factors.append(
                RiskFactor(
                    id="dex_new_pool",
                    category="dex",
                    affects="risk",
                    severity=sev,
                    headline=f"the primary pool is only {d.pool_age_hours:.0f} hours old",
                    explanation=(
                        f"The primary {pool} was created {d.pool_age_hours:.0f} hours ago, "
                        "so there is almost no trading history to judge it by."
                    ),
                    source="dex_market",
                )
            )
    moves: list[tuple[Level, str]] = []
    move_rules: tuple[tuple[Window, float, float], ...] = (
        ("m5", cfg.move_m5_medium_pct, cfg.move_m5_high_pct),
        ("h1", cfg.move_h1_medium_pct, cfg.move_h1_high_pct),
    )
    for window, medium, high in move_rules:
        w = d.window(window)
        if w is not None and w.price_change_pct is not None:
            level = _threshold(abs(w.price_change_pct), medium, high)
            if level is not None:
                span = "5 minutes" if window == "m5" else "1 hour"
                moves.append((level, f"{w.price_change_pct:+.1f}% in {span}"))
    if moves:
        sev = _max_severity(*(lv for lv, _ in moves)) or "medium"
        what = " and ".join(text for _, text in moves)
        factors.append(
            RiskFactor(
                id="dex_extreme_move",
                category="dex",
                affects="risk",
                severity=sev,
                headline=f"{label} moved {what}",
                explanation=(
                    f"{label} moved {what} in its primary pool, extreme even for a DEX token."
                ),
                source="dex_market",
            )
        )
    for window in cfg.imbalance_windows:
        w = d.window(window)
        if w is None or w.buys is None or w.sells is None or (w.txns or 0) < cfg.imbalance_min_txns:
            continue
        share = w.buys / (w.buys + w.sells)
        side, top = ("buys", share) if share >= 0.5 else ("sells", 1 - share)
        level = (
            "high"
            if top >= cfg.imbalance_high_share
            else "medium"
            if top >= cfg.imbalance_medium_share
            else None
        )
        if level is not None:
            span = {"m5": "5m", "h1": "1h", "h6": "6h", "h24": "24h"}[window]
            factors.append(
                RiskFactor(
                    id="dex_flow_imbalance",
                    category="dex",
                    affects="risk",
                    severity=level,
                    headline=f"{100 * top:.0f}% of {span} trades were {side}",
                    explanation=(
                        f"{w.buys:,} buys vs {w.sells:,} sells over {span} in the primary pool: "
                        f"{100 * top:.0f}% {side}. One-sided flow this strong is unusual and "
                        "can reverse abruptly."
                    ),
                    source="dex_market",
                )
            )
        break  # only the first window with enough trades
    if not d.primary_clear:
        factors.append(
            RiskFactor(
                id="dex_competing_pools",
                category="dex",
                affects="uncertainty",
                severity="medium",
                headline="several pools compete, so the primary market is unclear",
                explanation=(
                    "The primary pool isn't clearly the token's main market: "
                    + "; ".join(d.ambiguity)
                    + "."
                ),
                source="dex_market",
            )
        )
    return factors


def dex_candidate_factors(
    candidates: Sequence[PoolCandidate], cfg: DexRiskConfig
) -> list[RiskFactor]:
    """Pools exist but none could be used as the market: say why, as evidence."""
    if not candidates:
        return []
    reported = [c.liquidity_usd for c in candidates if c.liquidity_usd is not None]
    if not reported:
        return [
            RiskFactor(
                id="dex_liquidity_missing",
                category="dex",
                affects="uncertainty",
                severity="high",
                headline="no pool for this token reports its liquidity",
                explanation=(
                    f"{len(candidates)} pool(s) exist for this mint but none reports USD "
                    "liquidity, so how much can be traded (or exited) can't be assessed."
                ),
                source="dex_market",
            )
        ]
    deepest = max(reported)
    sev = _below(deepest, cfg.liquidity_high_usd, cfg.liquidity_medium_usd) or "medium"
    return [
        RiskFactor(
            id="dex_low_liquidity",
            category="dex",
            affects="risk",
            severity=sev,
            headline=f"no usable pool: the deepest holds {usd(deepest)} of liquidity",
            explanation=(
                f"{len(candidates)} pool(s) exist for this mint, but none is liquid, active "
                f"and priced enough to count as its market (deepest: {usd(deepest)}). "
                + NO_SAFETY_CHECK
            ),
            source="dex_market",
        )
    ]


# --- Invalidation conditions ---------------------------------------------------------------


def invalidations(inputs: Inputs, cfg: RiskConfig) -> list[InvalidationCondition]:
    out: list[InvalidationCondition] = []
    t = inputs.technical
    if t is not None:
        tf = t.timeframe
        sma = next(
            (i for i in t.indicators if i.kind == "sma" and i.available and i.value is not None),
            None,
        )
        if t.trend.label in ("uptrend", "downtrend") and sma is not None and sma.value:
            side = "below" if t.trend.label == "uptrend" else "above"
            out.append(
                InvalidationCondition(
                    condition=(
                        f"A {tf} close {side} {sma.name} (~{usd(sma.value)}) would end the current "
                        f"rule-based {t.trend.label} classification."
                    ),
                    source="technical_analysis",
                    basis="live_data",
                )
            )
        support = _nearest(t.levels.support)
        resistance = _nearest(t.levels.resistance)
        for kind, zone in (("support", support), ("resistance", resistance)):
            if zone is not None:
                out.append(
                    InvalidationCondition(
                        condition=_break_condition(kind, zone, tf),
                        source="technical_analysis",
                        basis="live_data",
                    )
                )
        inside = t.levels.containing
        if inside is not None:
            out.append(
                InvalidationCondition(
                    condition=(
                        f"Price is inside the swing zone {usd_zone(inside.lower, inside.upper)}; a "
                        f"{tf} close above ~{usd(inside.upper)} or below ~{usd(inside.lower)} "
                        "would take it out of that zone."
                    ),
                    source="technical_analysis",
                    basis="live_data",
                )
            )
    s = inputs.snapshot
    if s is not None and s.high_24h_usd is not None and s.low_24h_usd is not None:
        out.append(
            InvalidationCondition(
                condition=(
                    f"A move outside the 24h range ({usd(s.low_24h_usd)} – {usd(s.high_24h_usd)}) "
                    "would mean conditions have changed from those analyzed."
                ),
                source="market",
                basis="live_data",
            )
        )
    chart = inputs.chart
    if chart is not None and (
        inputs.asset is None or chart.reading.asset.symbol in (None, inputs.asset)
    ):
        shot = chart.reading.displayed_price.value
        if shot is not None:
            out.append(
                InvalidationCondition(
                    condition=(
                        f"If the live price diverges more than {cfg.screenshot_diff_medium_pct:g}% "
                        f"from the screenshot's ~{usd(shot)}, the screenshot's levels and readings "
                        "should be treated as stale."
                    ),
                    source="vision",
                    basis="screenshot",
                )
            )
        if t is None or not t.levels.available:
            for kind, levels, side in (
                ("support", chart.reading.support_levels, "below"),
                ("resistance", chart.reading.resistance_levels, "above"),
            ):
                level = next((lv for lv in levels if lv.basis == "visible" and lv.price), None)
                if level is not None and level.price is not None:
                    out.append(
                        InvalidationCondition(
                            condition=(
                                f"A close {side} the {kind} marked on the screenshot near "
                                f"{usd(level.price)} would break that level (read from the "
                                "screenshot, not live data)."
                            ),
                            source="vision",
                            basis="screenshot",
                        )
                    )
    n = inputs.news
    if n is not None:
        key = next((st for st in _fresh(n.stories) if st.impact == "high"), None)
        if key is not None:
            out.append(
                InvalidationCondition(
                    condition=(
                        f"If the story {_quote(key)} develops further (for example official "
                        "action or follow-up reporting), the news backdrop described here could "
                        "change."
                    ),
                    source="news_sentiment",
                    basis="news",
                )
            )
        if n.overall in ("bullish", "bearish"):
            opposite = "bearish" if n.overall == "bullish" else "bullish"
            out.append(
                InvalidationCondition(
                    condition=(
                        f"New high-impact {opposite} coverage would weaken the current "
                        f"{n.overall} news read."
                    ),
                    source="news_sentiment",
                    basis="news",
                )
            )
    return out


# --- Aggregation ---------------------------------------------------------------------------


def overall_risk(factors: Sequence[RiskFactor], has_evidence: bool) -> OverallRisk:
    if not has_evidence:
        return "unknown"
    risk = [f for f in factors if f.affects == "risk"]
    high = [f for f in risk if f.severity == "high"]
    if len(high) >= 2 or any(
        f.severity in ("medium", "high") and f.category != h.category for h in high for f in risk
    ):
        return "high"
    if any(f.severity in ("medium", "high") for f in risk):
        return "medium"
    return "low"


def uncertainty_level(
    factors: Sequence[RiskFactor], missing_types: int, overall: OverallRisk
) -> Level:
    unc = [f for f in factors if f.affects == "uncertainty"]
    medium = sum(f.severity == "medium" for f in unc)
    if overall == "unknown" or any(f.severity == "high" for f in unc) or medium >= 2:
        return "high"
    if missing_types >= 2:
        return "high"
    if medium or missing_types:
        return "medium"
    return "low"


def _agent_failure_weakens(f: RiskFactor) -> str | None:
    if f.id.endswith(("_failed", "_unreadable")):
        return f"Evidence from {AGENT_LABELS[f.source].lower()} is missing from this review."
    return None


def _missing(inputs: Inputs) -> tuple[list[str], int]:
    """Human-readable missing evidence, plus how many of the three evidence types
    (price, technical structure, news) are missing."""

    def why(name: AgentName) -> str:
        state = inputs.states[name]
        return {
            "not_run": "not part of this request",
            "failed": "the agent failed",
            "unreadable": "its output couldn't be read",
            "no_data": "no data for this asset",
            "ok": "",
        }[state.status]

    missing: list[str] = []
    types = 0
    if inputs.snapshot is None and inputs.technical is None and inputs.dex is None:
        types += 1
        missing.append(f"Live price data ({why('market')})")
    elif inputs.snapshot is None and inputs.dex is None:
        missing.append(f"Live market snapshot: 24h change and range ({why('market')})")
    if inputs.technical is None:
        types += 1
        missing.append(f"Technical structure: trend and levels ({why('technical_analysis')})")
    elif not inputs.technical.volume_available:
        missing.append("Per-candle volume (not supplied by the candle provider)")
    if inputs.news is None or not any(s.sentiment for s in inputs.news.focus):
        types += 1
        missing.append(f"Classified recent news ({why('news_sentiment')})")
    if inputs.chart is not None and inputs.snapshot is None and inputs.technical is None:
        missing.append("A live price to check the screenshot against")
    return missing, types


def assess(
    prior: Mapping[AgentName, AgentResult],
    asset: str | None,
    config: RiskConfig | None = None,
    profile: CryptoAssetProfile | None = None,
    dex_config: DexRiskConfig | None = None,
) -> RiskAssessment:
    cfg = risk_config_for(profile, config)
    dex_cfg = dex_config or DexRiskConfig()
    inputs = apply_profile(collect_inputs(prior, asset), profile)
    factors = input_factors(inputs)
    if inputs.snapshot is not None:
        factors += market_factors(inputs.snapshot, cfg)
    factors += market_quality_factors(inputs.snapshot, inputs.market_unavailable or [], cfg)
    factors += live_consistency_factors(inputs.snapshot, inputs.technical, cfg)
    if inputs.technical is not None:
        factors += technical_factors(inputs.technical, inputs.technical_extra or {}, cfg)
    if inputs.news is not None:
        factors += news_factors(inputs.news, inputs.technical)
    factors += screenshot_factors(inputs, cfg)
    if inputs.dex is not None:
        factors += dex_factors(inputs.dex, dex_cfg)
    elif inputs.dex_candidates:
        factors += dex_candidate_factors(inputs.dex_candidates, dex_cfg)
    extra_factors, extra_missing = profile_factors(profile) if profile else ([], [])
    factors += extra_factors
    for f in factors:
        f.weakens_if = f.weakens_if or WEAKENS.get(f.id) or _agent_failure_weakens(f)
    factors.sort(key=lambda f: (f.affects != "risk", -_SEVERITY_RANK[f.severity]))

    has_evidence = (
        inputs.snapshot is not None
        or inputs.technical is not None
        or inputs.dex is not None
        or bool(inputs.dex_candidates)
        or (inputs.news is not None and any(s.sentiment for s in inputs.news.focus))
    )
    if not has_evidence:
        factors.append(
            RiskFactor(
                id="insufficient_evidence",
                category="data_quality",
                affects="uncertainty",
                severity="medium",
                headline="there is not enough live evidence to assess risk",
                explanation=(
                    "No live market data, technical analysis or classified news was available "
                    "for this request, so the overall risk can't be assessed."
                ),
                source="risk",
                weakens_if="Any conclusion drawn now rests on missing evidence.",
            )
        )
    overall = overall_risk(factors, has_evidence)
    missing, missing_types = _missing(inputs)
    missing += extra_missing
    uncertainty = uncertainty_level(factors, missing_types, overall)

    risk_factors = [f for f in factors if f.affects == "risk"]
    if overall == "unknown":
        overall_reasons = ["There is no live market, technical or news evidence to assess."]
    elif not any(f.severity != "low" for f in risk_factors):
        overall_reasons = ["No medium or high risk factors were found in the available evidence."]
    else:
        overall_reasons = [f.headline for f in risk_factors if f.severity != "low"]
    uncertainty_reasons = [
        f.headline for f in factors if f.affects == "uncertainty" and f.severity != "low"
    ]
    if missing_types:
        verb = "is" if missing_types == 1 else "are"
        uncertainty_reasons.append(
            f"{missing_types} of 3 evidence types (price, technical structure, news) {verb} missing"
        )

    conflicts = [
        f.explanation
        for f in factors
        if f.id
        in {
            "indicator_conflict",
            "strong_news_conflict",
            "news_conflict",
            "news_vs_trend",
            "screenshot_price_discrepancy",
            "screenshot_asset_mismatch",
            "live_sources_disagree",
        }
    ]
    return RiskAssessment(
        asset=asset,
        overall_risk=overall,
        overall_reasons=overall_reasons,
        uncertainty_level=uncertainty,
        uncertainty_reasons=uncertainty_reasons,
        missing_evidence=missing,
        factors=factors,
        invalidation_conditions=invalidations(inputs, cfg),
        conflicting_evidence=conflicts,
        data_quality=[f.explanation for f in factors if f.category == "data_quality"],
        inputs=list(inputs.states.values()),
        rules=RULES,
        asset_profile=profile,
        profile_notes=profile_notes(profile) if profile else [],
    )


def category_label(category: RiskCategory) -> str:
    return _CATEGORY_LABELS[category]
