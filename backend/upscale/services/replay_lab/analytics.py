"""Grouped replay summaries. Descriptive statistics of market outcomes, never a
profitability claim.

A cohort gets statistics only with at least `min_sample` measured outcomes from at least
`min_assets` distinct assets (samples of one token are correlated: counting them as
independent would overstate the evidence); p10 / p90 need `min_sample_outer_percentiles`.
Otherwise it reports INSUFFICIENT_SAMPLE with its counts. HOLDOUT is excluded unless a
final evaluation asks for it explicitly (and the access is logged).
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from upscale.services.outcomes.config import AnalyticsConfig
from upscale.services.outcomes.metrics import mean, median, percentile
from upscale.services.replay_lab.config import Split
from upscale.services.replay_lab.models import NOT_A_BACKTEST
from upscale.services.replay_lab.store import AnalysisRow, ReplayStore

MIN_ASSETS = 5
DEFAULT_ANALYTICS = AnalyticsConfig()
PRIOR_MOVE_EDGES = (-20.0, 0.0, 20.0, 50.0, 100.0)
ACCELERATION_EDGES = (0.8, 1.25, 2.0)
ACTIVITY_EDGES = (0.5, 1.0, 2.0)
VOLATILITY_EDGES = (2.0, 5.0, 10.0)


class CohortStats(BaseModel):
    group: str
    status: Literal["OK", "INSUFFICIENT_SAMPLE"]
    samples: int  # decisions in the cohort
    measured: int  # with a measured price outcome at this horizon
    assets: int  # distinct assets among the measured
    median_return_pct: float | None = None
    mean_return_pct: float | None = None
    p25_return_pct: float | None = None
    p75_return_pct: float | None = None
    p10_return_pct: float | None = None
    p90_return_pct: float | None = None
    median_mfe_pct: float | None = None
    median_mae_pct: float | None = None
    median_max_drawdown_pct: float | None = None
    median_time_to_mfe_minutes: float | None = None
    median_liquidity_change_pct: float | None = None
    invalidation_touched: int | None = None
    trigger_touched: int | None = None


class ReplaySummary(BaseModel):
    label: str = NOT_A_BACKTEST
    horizon: str
    group_by: str
    splits: list[str]
    jobs: list[str] | None
    decisions: int
    measured: int
    min_sample: int
    min_assets: int
    cohorts: list[CohortStats]
    warnings: list[str] = Field(default_factory=list)


def _band(value: float | None, edges: Sequence[float], unit: str = "") -> str | None:
    if value is None:
        return None
    edges = sorted(edges)

    def fmt(x: float) -> str:
        if abs(x) >= 1e6:
            return f"{unit}{x / 1e6:g}M"
        if abs(x) >= 1e3:
            return f"{unit}{x / 1e3:g}k"
        return f"{unit}{x:g}"

    if value < edges[0]:
        return f"<{fmt(edges[0])}"
    for lo, hi in zip(edges, edges[1:], strict=False):
        if lo <= value < hi:
            return f"{fmt(lo)}..{fmt(hi)}"
    return f">={fmt(edges[-1])}"


def _f(row: AnalysisRow, key: str) -> float | None:
    value = row.decision.features.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def groupers(cfg: AnalyticsConfig) -> dict[str, Callable[[AnalysisRow], list[str]]]:
    def one(fn: Callable[[AnalysisRow], str | None]) -> Callable[[AnalysisRow], list[str]]:
        return lambda r: [fn(r) or "unavailable"]

    return {
        "stage": one(lambda r: r.decision.scout.stage or r.decision.scout.status),
        "score": one(lambda r: _band(r.decision.scout.score, cfg.score_bands)),
        "liquidity": one(lambda r: _band(_f(r, "liquidity_usd"), cfg.liquidity_bands_usd, "$")),
        "market_cap": one(lambda r: _band(_f(r, "market_cap_usd"), cfg.market_cap_bands_usd, "$")),
        "fdv": one(lambda r: _band(_f(r, "fdv_usd"), cfg.market_cap_bands_usd, "$")),
        "prior_move": one(lambda r: _band(_f(r, "price_change_h24_pct"), PRIOR_MOVE_EDGES)),
        "volume_acceleration": one(
            lambda r: _band(_f(r, "volume_acceleration"), ACCELERATION_EDGES)
        ),
        "txn_acceleration": one(lambda r: _band(_f(r, "txn_acceleration"), ACCELERATION_EDGES)),
        "activity": one(lambda r: _band(_f(r, "activity_h1_vs_h24"), ACTIVITY_EDGES)),
        "volatility": one(lambda r: _band(_f(r, "volatility_4h_pct"), VOLATILITY_EDGES)),
        "risk_flag": lambda r: sorted({f["code"] for f in r.decision.scout.risk_flags}) or ["none"],
        "action": one(lambda r: r.decision.action),
        "confidence": one(lambda r: r.decision.confidence),
        "social": one(
            lambda r: str(r.decision.social.get("state") or r.decision.social.get("status"))
        ),
        "chain": one(lambda r: r.decision.chain),
        "evidence": one(lambda r: r.decision.evidence),
        "split": one(lambda r: r.sample.split),
    }


GROUP_BY = tuple(groupers(AnalyticsConfig()))


def measured(row: AnalysisRow) -> bool:
    o = row.outcome
    return o is not None and o.price is not None and o.price.return_pct is not None


def cohort(
    group: str,
    rows: Sequence[AnalysisRow],
    cfg: AnalyticsConfig,
    min_assets: int = MIN_ASSETS,
) -> CohortStats:
    done = [r for r in rows if measured(r)]
    assets = len({r.sample.asset_id for r in done})
    stats = CohortStats(
        group=group,
        status="INSUFFICIENT_SAMPLE",
        samples=len(rows),
        measured=len(done),
        assets=assets,
    )
    if len(done) < cfg.min_sample or assets < min_assets:
        return stats
    prices = [r.outcome.price for r in done if r.outcome and r.outcome.price]
    returns = [p.return_pct for p in prices if p.return_pct is not None]

    def values(attr: str) -> list[float]:
        return [v for p in prices if (v := getattr(p, attr)) is not None]

    liquidity = [
        r.outcome.market.liquidity_change_pct
        for r in done
        if r.outcome and r.outcome.market and r.outcome.market.liquidity_change_pct is not None
    ]
    stats.status = "OK"
    stats.median_return_pct = _r(median(returns))
    stats.mean_return_pct = _r(mean(returns))
    stats.p25_return_pct = _r(percentile(returns, 25))
    stats.p75_return_pct = _r(percentile(returns, 75))
    if len(returns) >= cfg.min_sample_outer_percentiles:
        stats.p10_return_pct = _r(percentile(returns, 10))
        stats.p90_return_pct = _r(percentile(returns, 90))
    stats.median_mfe_pct = _r(median(values("mfe_pct")))
    stats.median_mae_pct = _r(median(values("mae_pct")))
    stats.median_max_drawdown_pct = _r(median(values("max_drawdown_pct")))
    stats.median_time_to_mfe_minutes = _r(median(values("time_to_mfe_minutes")))
    stats.median_liquidity_change_pct = _r(median(liquidity)) if liquidity else None
    with_levels = [r for r in done if r.outcome and r.outcome.triggers]
    if with_levels:
        stats.invalidation_touched = sum(
            1
            for r in with_levels
            if r.outcome and (t := r.outcome.triggers.get("invalidation")) and t.reached
        )
        stats.trigger_touched = sum(
            1
            for r in with_levels
            if r.outcome
            and any(t.reached for k, t in r.outcome.triggers.items() if k != "invalidation")
        )
    return stats


def _r(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def load_rows(
    store: ReplayStore,
    horizon: str,
    splits: Sequence[Split],
    job_ids: Sequence[str] | None = None,
    include_purged: bool = True,
    final_evaluation: bool = False,
) -> list[AnalysisRow]:
    if "HOLDOUT" in splits and not final_evaluation:
        raise PermissionError(
            "HOLDOUT is reserved for the final evaluation: pass final_evaluation=True "
            "(CLI: --include-holdout --final-evaluation); the access is logged"
        )
    if "HOLDOUT" in splits:
        store.log_holdout_access(
            f"final evaluation summary ({horizon})", ",".join(job_ids or []) or None
        )
    return store.analysis_rows(horizon, splits, job_ids, include_purged)


def summarize(
    store: ReplayStore,
    horizon: str = "1h",
    group_by: str = "stage",
    splits: Sequence[Split] = ("CALIBRATION", "VALIDATION"),
    job_ids: Sequence[str] | None = None,
    cfg: AnalyticsConfig = DEFAULT_ANALYTICS,
    min_assets: int = MIN_ASSETS,
    final_evaluation: bool = False,
) -> ReplaySummary:
    grouping = groupers(cfg)
    if group_by not in grouping:
        raise ValueError(f"group_by must be one of: {', '.join(grouping)}")
    rows = load_rows(store, horizon, splits, job_ids, final_evaluation=final_evaluation)
    buckets: dict[str, list[AnalysisRow]] = {}
    for r in rows:
        for label in grouping[group_by](r):
            buckets.setdefault(label, []).append(r)
    cohorts = [cohort(k, v, cfg, min_assets) for k, v in sorted(buckets.items())]
    warnings = []
    if any(r.sample.universe_basis.startswith("USER_SELECTED") for r in rows):
        warnings.append("includes USER_SELECTED assets: possible selection bias")
    if rows and all(c.status != "OK" for c in cohorts):
        warnings.append(
            f"no cohort reaches {cfg.min_sample} measured outcomes from {min_assets}+ assets: "
            "descriptive counts only, no statistics"
        )
    return ReplaySummary(
        horizon=horizon,
        group_by=group_by,
        splits=list(splits),
        jobs=list(job_ids) if job_ids else None,
        decisions=len(rows),
        measured=sum(1 for r in rows if measured(r)),
        min_sample=cfg.min_sample,
        min_assets=min_assets,
        cohorts=cohorts,
        warnings=warnings,
    )


def as_table(summary: ReplaySummary) -> str:
    head = (
        f"{'group':<28} {'status':<20} {'n':>5} {'meas':>5} {'assets':>6} {'median':>9} "
        f"{'mean':>9} {'p25':>9} {'p75':>9} {'MFE':>8} {'MAE':>8} {'maxDD':>8}"
    )
    lines = [head, "-" * len(head)]

    def f(x: float | None) -> str:
        return "" if x is None else f"{x:+.2f}"

    for c in summary.cohorts:
        lines.append(
            f"{c.group[:28]:<28} {c.status:<20} {c.samples:>5} {c.measured:>5} {c.assets:>6} "
            f"{f(c.median_return_pct):>9} {f(c.mean_return_pct):>9} {f(c.p25_return_pct):>9} "
            f"{f(c.p75_return_pct):>9} {f(c.median_mfe_pct):>8} {f(c.median_mae_pct):>8} "
            f"{f(c.median_max_drawdown_pct):>8}"
        )
    return "\n".join(lines)


def rows_as_dicts(rows: Sequence[AnalysisRow]) -> list[dict[str, Any]]:
    return [
        {
            "sample_key": r.sample.sample_key,
            "asset_id": r.sample.asset_id,
            "decision_at": r.sample.decision_at.isoformat(),
            "split": r.sample.split,
            "stage": r.decision.scout.stage,
            "score": r.decision.scout.score,
            "action": r.decision.action,
            "return_pct": r.outcome.price.return_pct if r.outcome and r.outcome.price else None,
        }
        for r in rows
    ]
