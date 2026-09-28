"""Deterministic outcome aggregates by cohort, with small-sample protection.

A cohort's statistics (median / mean / percentiles / distribution / rates) are reported
only when at least `min_sample` of its outcomes were measured; otherwise the cohort says
INSUFFICIENT_SAMPLE with its sample size, and only raw counts are shown. Every cohort
reports how many observations it has and how many horizons are complete, partial,
unavailable or still pending: missing outcomes are never silently dropped.

There is no win rate, success probability or expected return here: only what the market
did (returns, excursions, liquidity changes, stage transitions, collapse counts).
"""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field

from upscale.services.outcomes.config import AnalyticsConfig
from upscale.services.outcomes.metrics import distribution, mean, median, percentile
from upscale.services.outcomes.models import (
    NOT_A_PERFORMANCE_CLAIM,
    DecisionObservation,
    HorizonOutcome,
    ScoutObservation,
)

ScoutDimension = Literal[
    "all",
    "stage",
    "score_band",
    "chain",
    "age_band",
    "liquidity_band",
    "market_cap_band",
    "social_state",
    "social_support",
    "safety_status",
    "risk_flag",
    "rank_band",
    "discovery_status",
    "ranking_mode",
    "anchor_reason",
]
SCOUT_DIMENSIONS: tuple[str, ...] = get_args(ScoutDimension)
DecisionDimension = Literal["all", "action", "confidence", "timeframe", "source", "risk_level"]
DECISION_DIMENSIONS: tuple[str, ...] = get_args(DecisionDimension)
SampleStatus = Literal["SUFFICIENT", "INSUFFICIENT_SAMPLE", "NO_DATA"]
SOCIAL_SUPPORTED = {"SOCIAL_EMERGING", "SOCIAL_ACCELERATING", "SOCIAL_STRONG"}


class Stats(BaseModel):
    n: int
    median: float | None
    mean: float | None
    p10: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None


class Cohort(BaseModel):
    group: str
    horizon: str
    observations: int  # observations in the cohort (every one, whatever its status)
    complete: int
    partial: int
    unavailable: int
    pending: int
    measured: int  # horizons with a measured return (the sample size)
    sample_status: SampleStatus
    note: str | None = None
    # Only when sample_status is SUFFICIENT:
    return_pct: Stats | None = None
    mfe_pct: Stats | None = None
    mae_pct: Stats | None = None
    max_drawdown_pct: Stats | None = None
    liquidity_change_pct: Stats | None = None
    return_distribution: dict[str, int] | None = None
    liquidity_collapse_rate: float | None = None
    # Raw counts (always shown, with their sample size):
    market_status_counts: dict[str, int] = Field(default_factory=dict)
    future_stage_counts: dict[str, int] = Field(default_factory=dict)  # original -> future
    liquidity_collapses: int = 0
    pools_gone: int = 0
    price_collapses: int = 0
    missing_reasons: dict[str, int] = Field(default_factory=dict)
    trigger_reached_counts: dict[str, int] = Field(default_factory=dict)  # decisions only


class OutcomeSummary(BaseModel):
    kind: Literal["scout", "decision"]
    computed_at: datetime
    dimension: str
    horizon: str
    min_sample: int
    total_observations: int
    cohorts: list[Cohort]
    disclaimer: str = NOT_A_PERFORMANCE_CLAIM


def _band(value: float | None, edges: Sequence[float], unit: Callable[[float], str]) -> str:
    if value is None:
        return "unknown"
    ordered = sorted(edges)
    if value < ordered[0]:
        return f"<{unit(ordered[0])}"
    for lo, hi in zip(ordered, ordered[1:], strict=False):
        if lo <= value < hi:
            return f"{unit(lo)}-{unit(hi)}"
    return f">={unit(ordered[-1])}"


def _usd(x: float) -> str:
    for suffix, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= div:
            return f"${x / div:g}{suffix}"
    return f"${x:g}"


def scout_groups(o: ScoutObservation, dimension: str, cfg: AnalyticsConfig) -> list[str]:
    """The cohort(s) an observation belongs to (several only for risk flags)."""
    m = o.market
    if dimension == "all":
        return ["all"]
    if dimension == "stage":
        return [o.stage]
    if dimension == "score_band":
        return [_band(o.score, cfg.score_bands, lambda x: f"{x:g}")]
    if dimension == "chain":
        return [o.chain]
    if dimension == "age_band":
        return [_band(m.pool_age_hours, cfg.age_bands_hours, lambda x: f"{x:g}h")]
    if dimension == "liquidity_band":
        return [_band(m.liquidity_usd, cfg.liquidity_bands_usd, _usd)]
    if dimension == "market_cap_band":
        if m.market_cap_usd is None:
            return ["untrusted_or_unknown"]
        return [_band(m.market_cap_usd, cfg.market_cap_bands_usd, _usd)]
    if dimension == "social_state":
        return [o.social.status]
    if dimension == "social_support":
        if o.social.status == "SOCIAL_UNAVAILABLE":
            return ["social_unavailable"]
        return ["social_supported" if o.social.status in SOCIAL_SUPPORTED else "market_only"]
    if dimension == "safety_status":
        return [o.safety.status]
    if dimension == "risk_flag":
        flags = sorted({f.code for f in o.risk_flags if f.severity in ("high", "critical")})
        return flags or ["no_major_flag"]
    if dimension == "rank_band":
        low = 1
        for edge in sorted(cfg.rank_bands):
            if o.rank <= edge:
                return [f"{low}-{edge}"]
            low = edge + 1
        return [f">{low - 1}"]
    if dimension == "discovery_status":
        return [o.discovery_status]
    if dimension == "ranking_mode":
        return [o.ranking_mode]
    if dimension == "anchor_reason":
        return [o.anchor_reason]
    raise ValueError(f"unknown dimension {dimension!r}")


def decision_groups(d: DecisionObservation, dimension: str) -> list[str]:
    if dimension == "all":
        return ["all"]
    if dimension == "action":
        return [d.action.upper()]
    if dimension == "confidence":
        return [d.confidence]
    if dimension == "timeframe":
        return [d.timeframe or "unknown"]
    if dimension == "source":
        return [d.source]
    if dimension == "risk_level":
        return [d.risk.level]
    raise ValueError(f"unknown dimension {dimension!r}")


def _stats(values: list[float], cfg: AnalyticsConfig) -> Stats:
    outer = len(values) >= cfg.min_sample_outer_percentiles
    return Stats(
        n=len(values),
        median=median(values),
        mean=mean(values),
        p10=percentile(values, 10) if outer else None,
        p25=percentile(values, 25),
        p75=percentile(values, 75),
        p90=percentile(values, 90) if outer else None,
    )


def cohort(
    group: str,
    horizon: str,
    rows: Sequence[tuple[str | None, HorizonOutcome | None]],
    cfg: AnalyticsConfig,
) -> Cohort:
    """`rows`: (original stage or None, that observation's horizon row or None)."""
    status = Counter(h.status for _, h in rows if h is not None)
    measured = [
        h
        for _, h in rows
        if h is not None and h.price is not None and h.price.return_pct is not None
    ]
    enough = len(measured) >= cfg.min_sample
    c = Cohort(
        group=group,
        horizon=horizon,
        observations=len(rows),
        complete=status["COMPLETE"],
        partial=status["PARTIAL"],
        unavailable=status["UNAVAILABLE"],
        pending=status["PENDING"] + sum(1 for _, h in rows if h is None),
        measured=len(measured),
        sample_status="SUFFICIENT" if enough else "NO_DATA" if not measured else "INSUFFICIENT_SAMPLE",
    )  # fmt: skip
    if not enough:
        c.note = (
            f"n = {len(measured)} measured: insufficient sample for aggregate evaluation "
            f"(minimum {cfg.min_sample}); inspect the raw outcomes instead"
        )
    finals = [h for _, h in rows if h is not None and h.status != "PENDING"]
    c.market_status_counts = dict(Counter(h.market_status or "UNKNOWN" for h in finals))
    c.future_stage_counts = dict(
        Counter(
            f"{stage or '?'} -> {h.future_stage or 'unknown'}"
            for stage, h in rows
            if h is not None and h.status != "PENDING" and stage is not None
        )
    )
    c.liquidity_collapses = sum(1 for h in finals if h.market_status == "LIQUIDITY_COLLAPSE")
    c.pools_gone = sum(1 for h in finals if h.market_status == "POOL_GONE")
    c.price_collapses = sum(1 for h in finals if h.price is not None and h.price.price_collapsed)
    c.missing_reasons = dict(
        Counter(_reason(m) for h in finals if h.status != "COMPLETE" for m in h.missing)
    )
    c.trigger_reached_counts = dict(
        Counter(name for h in finals for name, t in h.triggers.items() if t.reached)
    )
    if enough:
        returns = [
            h.price.return_pct for h in measured if h.price and h.price.return_pct is not None
        ]
        c.return_pct = _stats(returns, cfg)
        c.return_distribution = distribution(returns, cfg.return_buckets_pct)
        for name in ("mfe_pct", "mae_pct", "max_drawdown_pct"):
            values = [
                getattr(h.price, name) for h in measured if getattr(h.price, name) is not None
            ]
            setattr(c, name, _stats(values, cfg) if values else None)
        liquidity = [
            h.market.liquidity_change_pct
            for h in finals
            if h.market is not None and h.market.liquidity_change_pct is not None
        ]
        c.liquidity_change_pct = _stats(liquidity, cfg) if liquidity else None
        with_market = [
            h for h in finals if h.market_status not in (None, "UNKNOWN", "PROVIDER_UNAVAILABLE")
        ]
        if with_market:
            c.liquidity_collapse_rate = (c.liquidity_collapses + c.pools_gone) / len(with_market)
    return c


def _reason(text: str) -> str:
    """Group reasons by their stable prefix (before any provider-specific detail)."""
    return text.split(":")[0].strip()


def summarize_scout(
    records: Iterable[tuple[ScoutObservation, list[HorizonOutcome]]],
    dimension: str,
    horizon: str,
    cfg: AnalyticsConfig,
    now: datetime | None = None,
) -> OutcomeSummary:
    groups: dict[str, list[tuple[str | None, HorizonOutcome | None]]] = {}
    total = 0
    for o, horizons in records:
        total += 1
        h = next((x for x in horizons if x.horizon == horizon), None)
        for g in scout_groups(o, dimension, cfg):
            groups.setdefault(g, []).append((o.stage, h))
    return OutcomeSummary(
        kind="scout",
        computed_at=now or datetime.now(UTC),
        dimension=dimension,
        horizon=horizon,
        min_sample=cfg.min_sample,
        total_observations=total,
        cohorts=[cohort(g, horizon, rows, cfg) for g, rows in sorted(groups.items())],
    )


def summarize_decisions(
    records: Iterable[tuple[DecisionObservation, list[HorizonOutcome]]],
    dimension: str,
    horizon: str,
    cfg: AnalyticsConfig,
    now: datetime | None = None,
) -> OutcomeSummary:
    groups: dict[str, list[tuple[str | None, HorizonOutcome | None]]] = {}
    total = 0
    for d, horizons in records:
        total += 1
        h = next((x for x in horizons if x.horizon == horizon), None)
        for g in decision_groups(d, dimension):
            groups.setdefault(g, []).append((None, h))
    return OutcomeSummary(
        kind="decision",
        computed_at=now or datetime.now(UTC),
        dimension=dimension,
        horizon=horizon,
        min_sample=cfg.min_sample,
        total_observations=total,
        cohorts=[cohort(g, horizon, rows, cfg) for g, rows in sorted(groups.items())],
    )
