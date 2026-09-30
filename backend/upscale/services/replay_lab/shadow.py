"""Foundations for comparing strategies and baselines on the same replay samples.

**Baselines** are simple selection rules over the frozen decision records (what each would
have picked at T), compared by the outcome distribution of the samples they pick. They
answer "does UpScale's selection differ from trivial rules?", never "how much money":
no entry, fill, fee, slippage, size or latency is modeled. `all_samples` is the
buy-and-hold-for-the-horizon reference; `random_half` a deterministic random draw.

**Shadow strategies** (future) evaluate hypothetical entry / exit behavior (enter now, wait
for a pullback, require confirmation, different invalidations...) against a stored
sample's decision and its post-T candle path, from the replay cache, without re-running
ingestion. v1 defines the interface and storage (``shadow_strategies``,
``shadow_evaluations``); no strategy that assumes execution is implemented. Position
context (entry price, exposure, prior recommendation) and goal constraints (daily loss,
drawdown and exposure limits) plug in through `ShadowContext` once production models them.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from upscale.services.market_data import Candle
from upscale.services.outcomes.config import AnalyticsConfig
from upscale.services.replay_lab.analytics import (
    DEFAULT_ANALYTICS,
    MIN_ASSETS,
    CohortStats,
    cohort,
    load_rows,
)
from upscale.services.replay_lab.config import Split
from upscale.services.replay_lab.models import ReplayDecisionRecord
from upscale.services.replay_lab.store import AnalysisRow, ReplayStore


@dataclass(frozen=True)
class Baseline:
    id: str
    description: str
    selects: Callable[[ReplayDecisionRecord], bool]


def _num(r: ReplayDecisionRecord, key: str) -> float | None:
    v = r.features.get(key)
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def _random_half(r: ReplayDecisionRecord) -> bool:
    return int(hashlib.sha256(r.sample_key.encode()).hexdigest(), 16) % 2 == 0


BASELINES: tuple[Baseline, ...] = (
    Baseline("all_samples", "every sample: buy-and-hold for the horizon from T", lambda r: True),
    Baseline("random_half", "a deterministic pseudo-random half of the samples", _random_half),
    Baseline(
        "volume_only",
        "volume acceleration >= 1.5x",
        lambda r: (v := _num(r, "volume_acceleration")) is not None and v >= 1.5,
    ),
    Baseline(
        "liquidity_and_volume",
        "liquidity >= $50k and volume acceleration >= 1.5x",
        lambda r: (
            (liq := _num(r, "liquidity_usd")) is not None
            and liq >= 50_000
            and (v := _num(r, "volume_acceleration")) is not None
            and v >= 1.5
        ),
    ),
    Baseline(
        "simple_momentum",
        "positive 1h and 24h price change",
        lambda r: (
            (a := _num(r, "price_change_h1_pct")) is not None
            and a > 0
            and (b := _num(r, "price_change_h24_pct")) is not None
            and b > 0
        ),
    ),
    Baseline(
        "production_scout",
        "production Growth Scout: eligible, stage ACCELERATING or EARLY",
        lambda r: bool(r.scout.eligible) and r.scout.stage in ("ACCELERATING", "EARLY"),
    ),
    Baseline(
        "production_buy",
        "production Opportunity action BUY",
        lambda r: r.action == "buy",
    ),
)


class BaselineReport(BaseModel):
    horizon: str
    splits: list[str]
    label: str = (
        "Outcome distribution of the samples each rule selects: not simulated trades or "
        "profit. A difference is only meaningful on unseen (HOLDOUT) data."
    )
    baselines: list[dict[str, Any]] = Field(default_factory=list)


def evaluate_baselines(
    store: ReplayStore,
    horizon: str,
    splits: Sequence[Split] = ("CALIBRATION", "VALIDATION"),
    job_ids: Sequence[str] | None = None,
    cfg: AnalyticsConfig = DEFAULT_ANALYTICS,
    final_evaluation: bool = False,
) -> BaselineReport:
    rows = load_rows(store, horizon, splits, job_ids, final_evaluation=final_evaluation)
    report = BaselineReport(horizon=horizon, splits=list(splits))
    for b in BASELINES:
        picked: list[AnalysisRow] = [r for r in rows if b.selects(r.decision)]
        stats: CohortStats = cohort(b.id, picked, cfg, MIN_ASSETS)
        report.baselines.append({"id": b.id, "description": b.description, **stats.model_dump()})
    return report


# --- shadow strategies (interface only in v1) ---------------------------------------------------


@dataclass(frozen=True)
class ShadowContext:
    """What a future shadow / goal-constrained policy may know besides the decision.
    Empty in v1: no position, portfolio or goal model exists in production yet."""

    position: dict[str, Any] | None = None  # entry price, size, age, unrealized P/L...
    portfolio: dict[str, Any] | None = None  # exposure, concurrent positions, day's P/L...
    constraints: dict[str, Any] = field(default_factory=dict)  # max daily loss, drawdown...


class ShadowResult(BaseModel):
    """A hypothetical strategy's behavior on one sample. `execution_model` must describe
    how fills / fees / slippage were simulated; results without one are not P/L."""

    strategy_id: str
    version: str
    horizon: str
    entered: bool
    notes: list[str] = Field(default_factory=list)
    execution_model: str | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)


class ShadowStrategy(Protocol):
    id: str
    version: str
    description: str

    def evaluate(
        self,
        decision: ReplayDecisionRecord,
        path: Sequence[Candle],  # post-T candles of the sample's pool (reveal phase only)
        context: ShadowContext,
    ) -> ShadowResult: ...
