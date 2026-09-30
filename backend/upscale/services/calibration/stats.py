"""Cohort statistics and evidence-strength rules. Descriptive only: associations, never
causes, never profit.

Strength (transparent, stored with every finding):

* INSUFFICIENT_SAMPLE: fewer than the descriptive minimum of measured outcomes or assets;
* WEAK_EVIDENCE: descriptive minimum met on CALIBRATION;
* MODERATE_EVIDENCE: candidate minimum met, the direction holds on at least two horizons,
  and VALIDATION shows the same direction with at least the descriptive minimum;
* STRONGER_EVIDENCE: strong minimum met, all of the above, missing-data rate under 50%,
  and, when both origins have enough data, LIVE_FORWARD and HISTORICAL_REPLAY agree.
"""

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from upscale.services.calibration.config import CalibrationConfig, SampleRules, Strength
from upscale.services.calibration.dataset import Observation, OutcomeView
from upscale.services.outcomes.metrics import mean, median, percentile


class Cohort(BaseModel):
    label: str
    status: str  # OK / INSUFFICIENT_SAMPLE
    n: int  # observations in the cohort
    measured: int  # with a measured return at the horizon
    assets: int
    days: int
    runs: int
    max_per_asset: int
    effective_n: float  # Kish effective size with 1 / (observations of the asset) weights
    origins: dict[str, int] = Field(default_factory=dict)
    regimes: dict[str, int] = Field(default_factory=dict)
    missing_rate: float | None = None
    median_return: float | None = None
    mean_return: float | None = None
    trimmed_mean_return: float | None = None  # 10% trimmed each side (robust to outliers)
    extreme_returns: int = 0  # |return| > EXTREME_RETURN_PCT: check the data before trusting means
    median_ci: tuple[float, float] | None = None  # bootstrap 90% interval of the median
    p25: float | None = None
    p75: float | None = None
    p10: float | None = None
    p90: float | None = None
    positive_share: float | None = None  # share of positive returns (direction, not a win rate)
    median_mfe: float | None = None
    median_mae: float | None = None
    median_drawdown: float | None = None
    median_time_to_mfe: float | None = None
    median_liquidity_change: float | None = None
    collapse_share: float | None = None  # LIQUIDITY_COLLAPSE / POOL_GONE / TOKEN_INACTIVE
    invalidation_touch_share: float | None = None


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def outcome(o: Observation, horizon: str) -> OutcomeView | None:
    v = o.outcomes.get(horizon)
    return v if v is not None and v.measured else None


def cohort(
    label: str,
    observations: Sequence[Observation],
    horizon: str,
    cfg: CalibrationConfig,
    missing_key: str | None = None,
) -> Cohort:
    rules = cfg.samples
    done = [(o, v) for o in observations if (v := outcome(o, horizon)) is not None]
    per_asset = Counter(o.asset_id for o, _ in done)
    weights = [1 / per_asset[o.asset_id] for o, _ in done]
    eff = (sum(weights) ** 2 / sum(w * w for w in weights)) if weights else 0.0
    c = Cohort(
        label=label,
        status="INSUFFICIENT_SAMPLE",
        n=len(observations),
        measured=len(done),
        assets=len(per_asset),
        days=len({o.at.date() for o, _ in done}),
        runs=len({o.run_id for o, _ in done if o.run_id}),
        max_per_asset=max(per_asset.values(), default=0),
        effective_n=round(eff, 2),
        origins=dict(Counter(o.origin for o in observations)),
        regimes=dict(Counter(o.regime.get("volatility", "UNKNOWN") for o in observations)),
        missing_rate=(
            round(sum(1 for o in observations if missing_key in o.missing) / len(observations), 4)
            if missing_key and observations
            else None
        ),
    )
    if len(done) < rules.descriptive_n or c.assets < rules.descriptive_assets:
        return c
    returns = [v.return_pct for _, v in done if v.return_pct is not None]

    def values(attr: str) -> list[float]:
        return [x for _, v in done if (x := getattr(v, attr)) is not None]

    c.status = "OK"
    c.median_return, c.mean_return = _r(median(returns)), _r(mean(returns))
    c.trimmed_mean_return = _r(trimmed_mean(returns, 0.1))
    c.extreme_returns = sum(1 for r in returns if abs(r) > EXTREME_RETURN_PCT)
    c.p25, c.p75 = _r(percentile(returns, 25)), _r(percentile(returns, 75))
    if len(returns) >= rules.outer_percentiles_n:
        c.p10, c.p90 = _r(percentile(returns, 10)), _r(percentile(returns, 90))
    c.median_ci = bootstrap_median_ci(returns, cfg.bootstrap_resamples, cfg.seed)
    c.positive_share = _r(sum(1 for r in returns if r > 0) / len(returns))
    c.median_mfe = _r(median(values("mfe_pct")))
    c.median_mae = _r(median(values("mae_pct")))
    c.median_drawdown = _r(median(values("max_drawdown_pct")))
    c.median_time_to_mfe = _r(median(values("time_to_mfe_minutes")))
    liq = values("liquidity_change_pct")
    c.median_liquidity_change = _r(median(liq)) if liq else None
    terminal = ("LIQUIDITY_COLLAPSE", "POOL_GONE", "TOKEN_INACTIVE")
    c.collapse_share = _r(sum(1 for _, v in done if v.market_status in terminal) / len(done))
    touched = [v.invalidation_touched for _, v in done if v.invalidation_touched is not None]
    c.invalidation_touch_share = _r(sum(touched) / len(touched)) if touched else None
    return c


EXTREME_RETURN_PCT = 1000.0


def trimmed_mean(values: Sequence[float], share: float) -> float | None:
    v = sorted(values)
    k = int(len(v) * share)
    kept = v[k : len(v) - k] if len(v) - 2 * k > 0 else v
    return mean(kept)


def bootstrap_median_ci(
    values: Sequence[float], resamples: int, seed: int
) -> tuple[float, float] | None:
    """A deterministic 90% bootstrap interval of the median (seeded)."""
    if resamples <= 0 or len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    meds = sorted(
        m
        for _ in range(resamples)
        if (m := median([values[rng.randrange(n)] for _ in range(n)])) is not None
    )
    lo, hi = percentile(meds, 5), percentile(meds, 95)
    return (round(lo, 4), round(hi, 4)) if lo is not None and hi is not None else None


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return round(cov / (vx * vy), 4) if vx and vy else None


def meets(c: Cohort, rules: SampleRules, level: str) -> bool:
    n, a = {
        "descriptive": (rules.descriptive_n, rules.descriptive_assets),
        "candidate": (rules.candidate_n, rules.candidate_assets),
        "strong": (rules.strong_n, rules.strong_assets),
    }[level]
    return c.measured >= n and c.assets >= a


def strength(
    cal: Cohort,
    rules: SampleRules,
    horizons_agree: int,
    validation_agrees: bool | None,
    origins_agree: bool | None,
) -> tuple[Strength, str]:
    """The label and the rule that produced it."""
    if not meets(cal, rules, "descriptive"):
        return "INSUFFICIENT_SAMPLE", (
            f"{cal.measured} measured from {cal.assets} assets (needs {rules.descriptive_n} / "
            f"{rules.descriptive_assets})"
        )
    missing_ok = cal.missing_rate is None or cal.missing_rate < 0.5
    if (
        meets(cal, rules, "strong")
        and horizons_agree >= 2
        and validation_agrees is True
        and missing_ok
        and origins_agree is not False
    ):
        return "STRONGER_EVIDENCE", "strong sample, several horizons, validation and origins agree"
    if meets(cal, rules, "candidate") and horizons_agree >= 2 and validation_agrees is True:
        return "MODERATE_EVIDENCE", "candidate-size sample, several horizons, validation agrees"
    why = []
    if not meets(cal, rules, "candidate"):
        why.append("below the candidate sample size")
    if horizons_agree < 2:
        why.append("direction not consistent across horizons")
    if validation_agrees is None:
        why.append("VALIDATION sample too small")
    elif validation_agrees is False:
        why.append("not reproduced on VALIDATION")
    return "WEAK_EVIDENCE", "; ".join(why) or "descriptive only"


def by(
    observations: Sequence[Observation], key: Callable[[Observation], Any]
) -> dict[str, list[Observation]]:
    out: dict[str, list[Observation]] = {}
    for o in observations:
        k = key(o)
        for label in k if isinstance(k, list) else [k]:
            out.setdefault(str(label), []).append(o)
    return out
