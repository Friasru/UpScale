"""Every outcome-tracking threshold, validated. Change these values, not the logic.

Outcome tracking is measurement only: nothing here feeds back into Growth Scout, Risk,
Opportunity or Technical Analysis.
"""

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from upscale.services.market_data import Timeframe


class HorizonSpec(BaseModel):
    """One fixed horizon after an observation. `candles`: the candle timeframe its price
    path is measured on (finer is better; one request returns up to 1,000 candles)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1, max_length=8)
    minutes: int = Field(gt=0)
    candles: Timeframe
    # After the horizon is due, how long a missing price path may still be collected
    # (candles are history: they can be fetched late) before the horizon is finalized
    # as PARTIAL / UNAVAILABLE with its reasons.
    retry_minutes: int = Field(gt=0)


DEFAULT_HORIZONS = (
    HorizonSpec(label="5m", minutes=5, candles="1m", retry_minutes=120),
    HorizonSpec(label="15m", minutes=15, candles="1m", retry_minutes=180),
    HorizonSpec(label="1h", minutes=60, candles="1m", retry_minutes=360),
    HorizonSpec(label="4h", minutes=240, candles="5m", retry_minutes=720),
    HorizonSpec(label="24h", minutes=1440, candles="15m", retry_minutes=1440),
)


class ObservationPolicy(BaseModel):
    """When a ranked Scout candidate becomes a new evaluation anchor (see `observe`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Scout Momentum points away from the token's latest anchor that count as material.
    min_score_change: float = Field(default=10.0, gt=0)
    # A stage different from the latest anchor's is always material.
    stage_change: bool = True
    # Not ranked for at least this long, then ranked again: a re-entry.
    reentry_gap_minutes: float = Field(default=30.0, gt=0)
    # A new anchor anyway once the latest one is this old (and the evidence is newer).
    reanchor_after_minutes: float = Field(default=360.0, gt=0)
    # Only candidates ranked at or above this position are anchored (None: all ranked).
    max_rank: int | None = Field(default=None, gt=0)


class DecisionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # The same decision (asset, action, timeframe, trigger levels) again within this many
    # seconds is one decision, not two.
    duplicate_seconds: float = Field(default=60.0, ge=0)
    # An Analyze this soon after a Scout anchor of the same token links to that anchor.
    scout_link_minutes: float = Field(default=240.0, gt=0)


class CollectorConfig(BaseModel):
    """The background collector: lowest priority for every shared provider quota."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Wait this long after a horizon ends before measuring it (the last candle must close,
    # and Scout's next snapshot may arrive).
    settle_seconds: float = Field(default=75.0, ge=0)
    # Longest sleep between cycles when nothing is due (a heartbeat, not polling).
    max_sleep_seconds: float = Field(default=900.0, gt=0)
    # At least this long between cycles, so horizons ending seconds apart share one cycle.
    min_sleep_seconds: float = Field(default=15.0, gt=0)
    # No provider requests for this long after a Scout scan or an Analyze finished: an
    # Analyze usually follows a scan (a user opening a card), and shouldn't meet a provider
    # still counting the scan's burst plus outcome work.
    quiet_after_seconds: float = Field(default=60.0, ge=0)
    # A horizon still missing evidence (e.g. deferred for quota) is retried after this.
    retry_seconds: float = Field(default=120.0, gt=0)
    # Provider requests per cycle, per provider (background work waits for the next cycle);
    # `request_budgets` overrides it per provider (DEX Screener's pair lookups are cheap:
    # 300 / min, 30 tokens each; GeckoTerminal's quota is tight and shared with Analyze).
    max_requests_per_cycle: int = Field(default=3, ge=0)
    request_budgets: dict[str, int] = Field(default_factory=lambda: {"DEX Screener": 10})
    # On a provider quota that holds no reservation for Analyze (exchange candles, DEX
    # Screener), leave at least this many requests of the current window free for it. The
    # shared GeckoTerminal quota protects Analyze with its "interactive" reservation, and
    # due outcome work outranks Scout there, so nothing more is kept back on it.
    min_free_calls: int = Field(default=1, ge=0)
    # After a provider really answered "rate limited" (HTTP 429), outcome work leaves it
    # alone this long (UpScale's own quota deferrals are not provider pushback).
    rate_limit_cooldown_seconds: float = Field(default=120.0, ge=0)
    # Short horizons wait for the longest horizon on the same candle timeframe (when it
    # ends within their retry window): one candle request then measures all of them. The
    # time-sensitive horizon-end market state is still captured when each horizon ends.
    coalesce_candles: bool = True
    # Horizon-end market state (liquidity, volume...) must be observed within this much of
    # the horizon end (fraction of the horizon, at least `min_market_tolerance_seconds`).
    market_tolerance: float = Field(default=0.1, gt=0, le=0.5)
    min_market_tolerance_seconds: float = Field(default=180.0, ge=0)
    max_due_per_cycle: int = Field(default=500, gt=0)


class CollapseConfig(BaseModel):
    """Factual thresholds for terminal market states (never labeled "rug")."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Liquidity down at least this much from the observation, or below the floor.
    liquidity_drop_pct: float = Field(default=80.0, gt=0, lt=100)
    liquidity_floor_usd: float = Field(default=1_000.0, ge=0)
    # Price down at least this much from the observation.
    price_drop_pct: float = Field(default=90.0, gt=0, lt=100)


class AnalyticsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Below this many measured outcomes a cohort reports INSUFFICIENT_SAMPLE, not stats.
    min_sample: int = Field(default=20, gt=0)
    # Outer percentiles (p10 / p90) need more observations than the median does.
    min_sample_outer_percentiles: int = Field(default=50, gt=0)
    score_bands: tuple[float, ...] = (40.0, 60.0, 80.0)
    liquidity_bands_usd: tuple[float, ...] = (10_000.0, 50_000.0, 250_000.0, 1_000_000.0)
    market_cap_bands_usd: tuple[float, ...] = (100_000.0, 1_000_000.0, 10_000_000.0, 1e8)
    age_bands_hours: tuple[float, ...] = (1.0, 6.0, 24.0, 168.0)
    rank_bands: tuple[int, ...] = (10, 20, 50)
    # Return distribution bucket edges (percent).
    return_buckets_pct: tuple[float, ...] = (-50.0, -20.0, -5.0, 5.0, 20.0, 50.0, 100.0)


class OutcomeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    horizons: tuple[HorizonSpec, ...] = DEFAULT_HORIZONS
    observation: ObservationPolicy = ObservationPolicy()
    decisions: DecisionPolicy = DecisionPolicy()
    collector: CollectorConfig = CollectorConfig()
    collapse: CollapseConfig = CollapseConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()

    @model_validator(mode="after")
    def _unique_horizons(self) -> "OutcomeConfig":
        labels = [h.label for h in self.horizons]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("horizon labels must be unique and there must be at least one")
        return self

    def horizon(self, label: str) -> HorizonSpec | None:
        return next((h for h in self.horizons if h.label == label), None)


class OutcomeConfigError(ValueError):
    pass


def load_outcome_config(raw: str | None) -> OutcomeConfig:
    """Defaults, overridden by a JSON object (e.g. `UPSCALE_OUTCOME_CONFIG`), validated."""
    if not raw:
        return OutcomeConfig()
    try:
        return OutcomeConfig.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise OutcomeConfigError(f"invalid outcome config: {exc}") from exc
