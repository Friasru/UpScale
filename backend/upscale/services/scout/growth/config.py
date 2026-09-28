"""Every Growth Scout weight, scale and threshold, validated. Change these values, not the
logic.

Defaults are conservative, explainable starting points: they were **not** fitted to the
live sample (far too small to fit anything). Scales say where a measurement saturates:
e.g. `ratio_full = 3` means a 3x acceleration scores as strongly as anything larger, so
one extreme figure can't swamp the rest.
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from upscale.services.solana_dex import Window

_FROZEN = ConfigDict(frozen=True, extra="forbid")

DiscoveryMode = Literal["NEW_AND_EARLY", "ALL_TRENDING"]


def _check_weights(weights: dict[str, float], what: str, max_share: float = 1.0) -> None:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"{what} weights must not all be zero")
    for name, w in weights.items():
        if w / total > max_share + 1e-9:
            raise ValueError(
                f"{what} weight {name} is {w / total:.0%} of the total; at most "
                f"{max_share:.0%} so no single family can dominate"
            )


class FamilyWeights(BaseModel):
    """ScoutMomentumScore = weighted mean of the family scores (each 0..1), x 100, minus
    the risk penalty. Weights are normalized by their sum."""

    model_config = _FROZEN

    market_activity: float = Field(default=0.35, ge=0)
    liquidity_quality: float = Field(default=0.15, ge=0)
    social_momentum: float = Field(default=0.12, ge=0)
    earliness: float = Field(default=0.18, ge=0)
    cross_confirmation: float = Field(default=0.20, ge=0)
    # No family may exceed this share of the total weight.
    max_family_share: float = Field(default=0.40, gt=0, le=1)
    # Social is supporting evidence: capped lower still, so hype can't carry a candidate.
    max_social_share: float = Field(default=0.20, ge=0, le=1)

    def as_dict(self) -> dict[str, float]:
        return {
            "market_activity": self.market_activity,
            "liquidity_quality": self.liquidity_quality,
            "social_momentum": self.social_momentum,
            "earliness": self.earliness,
            "cross_confirmation": self.cross_confirmation,
        }

    @model_validator(mode="after")
    def _balanced(self) -> "FamilyWeights":
        weights = self.as_dict()
        _check_weights(weights, "family", self.max_family_share)
        if self.social_momentum / sum(weights.values()) > self.max_social_share + 1e-9:
            raise ValueError(f"social_momentum may be at most {self.max_social_share:.0%}")
        return self


class MarketActivityConfig(BaseModel):
    model_config = _FROZEN

    # Sub-signal weights inside the family (normalized over the ones available).
    volume_weight: float = Field(default=0.30, ge=0)
    txn_weight: float = Field(default=0.25, ge=0)
    buy_pressure_weight: float = Field(default=0.20, ge=0)
    price_weight: float = Field(default=0.15, ge=0)
    technical_weight: float = Field(default=0.10, ge=0)
    # Rate ratios are log-scaled: 1 -> 0.5 (flat), ratio_full -> 1, 1/ratio_full -> 0.
    ratio_full: float = Field(default=3.0, gt=1)
    # Short windows (fast) and medium windows (slow) used for acceleration, first found.
    short_pairs: tuple[tuple[Window, Window], ...] = (("m15", "h1"), ("m5", "h1"))
    medium_pairs: tuple[tuple[Window, Window], ...] = (("h1", "h6"), ("h6", "h24"))
    # History lookbacks (minutes) compared against, preferred first.
    history_lookbacks: tuple[int, ...] = (60, 30, 360, 15)
    # A window pair is only trusted when its short window had at least this many trades
    # (three trades in 5 minutes can't show acceleration).
    min_short_window_txns: int = Field(default=10, ge=1)
    # Buy share (buys / trades): level mapped from low -> 0 to high -> 1.
    buy_share_low: float = Field(default=0.35, ge=0, lt=1)
    buy_share_high: float = Field(default=0.65, gt=0, le=1)
    # Change in buy share at which the change sub-score saturates.
    buy_share_change_full: float = Field(default=0.15, gt=0, le=1)
    # 1h price change (%) that saturates the price sub-score.
    price_h1_full_pct: float = Field(default=30.0, gt=0)
    # Ratio above which a sub-signal counts as "rising" / below which "falling".
    rising_ratio: float = Field(default=1.3, gt=1)
    # Every trusted ratio at or below this: the market is flat (for social-only hype).
    flat_ratio: float = Field(default=1.1, gt=0)
    falling_ratio: float = Field(default=0.75, gt=0, lt=1)
    buyers_strengthening_change: float = Field(default=0.03, ge=0)
    buyers_weakening_change: float = Field(default=-0.05, le=0)

    @model_validator(mode="after")
    def _ordered(self) -> "MarketActivityConfig":
        if self.buy_share_low >= self.buy_share_high:
            raise ValueError("buy_share_low must be below buy_share_high")
        _check_weights(
            {
                "volume": self.volume_weight,
                "txns": self.txn_weight,
                "buy_pressure": self.buy_pressure_weight,
                "price": self.price_weight,
                "technical": self.technical_weight,
            },
            "market activity",
        )
        return self


class TechnicalConfig(BaseModel):
    """Lightweight technical context from stored snapshots (no candles, no Technical Agent)."""

    model_config = _FROZEN

    lookback_hours: float = Field(default=6.0, gt=0)
    min_snapshots: int = Field(default=4, ge=3)
    # Price must clear the prior high by this much to count as a breakout.
    breakout_margin_pct: float = Field(default=2.0, ge=0)
    # Total change (%) over the lookback below which the trend is "flat".
    flat_trend_pct: float = Field(default=3.0, ge=0)


class LiquidityConfig(BaseModel):
    model_config = _FROZEN

    depth_weight: float = Field(default=0.35, ge=0)
    growth_weight: float = Field(default=0.25, ge=0)
    stability_weight: float = Field(default=0.20, ge=0)
    ratio_weight: float = Field(default=0.20, ge=0)
    # Depth is absolute (a tiny market cap never makes thin liquidity good), log-scaled
    # between these; above `depth_full_usd` more liquidity adds nothing.
    depth_floor_usd: float = Field(default=5_000.0, gt=0)
    depth_full_usd: float = Field(default=250_000.0, gt=0)
    # Liquidity change (%) that saturates the growth sub-score, both ways.
    growth_full_pct: float = Field(default=50.0, gt=0)
    # A drop this large (%) at any lookback scores stability 0.
    stability_drop_full_pct: float = Field(default=50.0, gt=0, le=100)
    # Liquidity / market cap (or FDV) mapped from low -> 0 to high -> 1 (log).
    ratio_low: float = Field(default=0.02, gt=0)
    ratio_high: float = Field(default=0.15, gt=0)
    growth_lookbacks: tuple[int, ...] = (60, 30, 360, 15)
    healthy_min_usd: float = Field(default=25_000.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "LiquidityConfig":
        if self.depth_floor_usd >= self.depth_full_usd:
            raise ValueError("depth_floor_usd must be below depth_full_usd")
        if self.ratio_low >= self.ratio_high:
            raise ValueError("ratio_low must be below ratio_high")
        _check_weights(
            {
                "depth": self.depth_weight,
                "growth": self.growth_weight,
                "stability": self.stability_weight,
                "ratio": self.ratio_weight,
            },
            "liquidity",
        )
        return self


class SocialScoringConfig(BaseModel):
    """Social is supporting evidence. Missing or quiet social is `neutral`, never zero."""

    model_config = _FROZEN

    neutral: float = Field(default=0.4, ge=0, le=1)
    emerging: float = Field(default=0.7, ge=0, le=1)
    accelerating: float = Field(default=0.8, ge=0, le=1)
    strong: float = Field(default=0.9, ge=0, le=1)
    steady: float = Field(default=0.45, ge=0, le=1)
    saturated: float = Field(default=0.4, ge=0, le=1)
    fading: float = Field(default=0.25, ge=0, le=1)
    # Blend of the state score and unique-author growth (log-scaled like market ratios).
    author_growth_weight: float = Field(default=0.35, ge=0, le=1)
    author_ratio_full: float = Field(default=3.0, gt=1)
    # Positive evidence is scaled by attribution quality: 1.0 when every counted mention
    # is EXACT / STRONG, `probable_only_factor` when all are PROBABLE (ticker + context).
    probable_only_factor: float = Field(default=0.4, ge=0, le=1)
    medium_spam_factor: float = Field(default=0.5, ge=0, le=1)  # positive part x this
    high_spam_cap: float = Field(default=0.2, ge=0, le=1)  # score capped at this
    corroborated_bonus: float = Field(default=0.08, ge=0, le=0.5)
    # The social layer's PROBABLE mention weight (`attribution.probable_weight`), used to
    # recover how many counted mentions were EXACT / STRONG. Keep the two equal.
    probable_mention_weight: float = Field(default=0.5, ge=0, lt=1)
    organic_high_bonus: float = Field(default=0.05, ge=0, le=0.5)


class EarlinessConfig(BaseModel):
    """Earliness: how early we appear to be in a real, developing move. Not how small the
    asset is: market size is minor context, and only for a healthy market."""

    model_config = _FROZEN

    move_weight: float = Field(default=0.40, ge=0)
    age_weight: float = Field(default=0.25, ge=0)
    activity_weight: float = Field(default=0.20, ge=0)
    maturity_weight: float = Field(default=0.15, ge=0)
    # How much of the move already happened (largest of the 24h change and the change since
    # Scout first saw it): at most `early_move_pct` scores 1; `late_move_multiple` x or
    # more scores 0 (log-interpolated). Being new is never rewarded on its own.
    early_move_pct: float = Field(default=50.0, ge=0)
    late_move_multiple: float = Field(default=10.0, gt=1)
    # A price that fell made no up-move, but no move is developing either: neutral. (A
    # collapsed market gets no earliness at all.)
    decline_move_score: float = Field(default=0.5, ge=0, le=1)
    # Pool age: below `young_hours` there is little evidence (partial score), up to
    # `prime_hours` full score, decaying to 0 at `old_days`.
    young_hours: float = Field(default=1.0, ge=0)
    young_score: float = Field(default=0.6, ge=0, le=1)
    prime_hours: float = Field(default=72.0, gt=0)
    old_days: float = Field(default=60.0, gt=0)
    # Activity regime: the 6h trade rate over the 24h rate (only for pools a day old or
    # more). Activity that only recently picked up is early; long-steady activity is
    # mature. Log-scaled: 1 -> 0.5, `activity_ratio_full` -> 1.
    activity_ratio_full: float = Field(default=3.0, gt=1)
    # Market size context from market cap (FDV when no market cap): 1 up to
    # `small_cap_usd`, 0 from `large_cap_usd` (log). A small cap only earns more than
    # `unhealthy_maturity_cap` when its liquidity is at least `liquidity.healthy_min_usd`.
    small_cap_usd: float = Field(default=1_000_000.0, gt=0)
    large_cap_usd: float = Field(default=500_000_000.0, gt=0)
    unhealthy_maturity_cap: float = Field(default=0.5, ge=0, le=1)
    # History depth: with few stored snapshots, credit above neutral (0.5) is less
    # certain: it is scaled by `min_confidence` with no history, rising to 1 at
    # `technical.min_snapshots` snapshots. Never applied below neutral.
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    # How far back stored history is read to measure the move since first seen.
    first_seen_lookback_hours: float = Field(default=72.0, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "EarlinessConfig":
        if self.small_cap_usd >= self.large_cap_usd:
            raise ValueError("small_cap_usd must be below large_cap_usd")
        if self.young_hours >= self.prime_hours or self.prime_hours >= self.old_days * 24:
            raise ValueError("expected young_hours < prime_hours < old_days")
        _check_weights(
            {
                "move": self.move_weight,
                "age": self.age_weight,
                "activity": self.activity_weight,
                "maturity": self.maturity_weight,
            },
            "earliness",
        )
        return self


class FlowConfig(BaseModel):
    """Buy / sell flow is read cautiously: providers report trade *counts* (and sometimes
    distinct wallets), never buy / sell dollar volume, and bots or very different trade
    sizes distort counts. Nothing here estimates dollar flow."""

    model_config = _FROZEN

    # Divergence: fewer buys than this share of trades while volume, trades or price
    # rise. The move isn't carried by visible buyers (distribution, or counts distorted).
    weak_buy_share: float = Field(default=0.20, ge=0, lt=1)
    # "Buyers strengthening" also needs this buy share level, not only a rising share (a
    # share rising from 6% to 9% is not buyer strength).
    min_strength_buy_share: float = Field(default=0.40, ge=0, lt=1)
    # Many buys from few wallets: buy share at least `bot_buy_share` while distinct buyers
    # are at most `bot_max_buyer_share` of distinct traders.
    bot_buy_share: float = Field(default=0.70, gt=0, le=1)
    bot_max_buyer_share: float = Field(default=0.45, ge=0, lt=1)
    # Buys without price response: buy share at least this with a flat / falling 1h price.
    unanswered_buy_share: float = Field(default=0.75, gt=0, le=1)
    # Flow evidence in doubt: the buy-pressure sub-score is capped at this.
    doubtful_pressure_cap: float = Field(default=0.3, ge=0, le=1)


class CollapseConfig(BaseModel):
    """MARKET_COLLAPSE: evidence the market has effectively died. A collapsed token gets
    no earliness credit, is FADING whatever else is rising, and is penalized. Discovery
    ranking only: never a SELL."""

    model_config = _FROZEN

    # The price fell at least this much (%) over 1h, 6h or 24h.
    price_drop_pct: float = Field(default=90.0, gt=0, le=100)
    # Liquidity fell at least this much (%) at a stored lookback.
    liquidity_drop_pct: float = Field(default=80.0, gt=0, le=100)
    # Activity dying: the 1h trade rate is at most this share of the 24h rate (pools at
    # least `activity_min_age_hours` old, with enough 24h trades to judge).
    dead_activity_ratio: float = Field(default=0.05, ge=0, lt=1)
    activity_min_age_hours: float = Field(default=2.0, gt=0)
    # Liquidity below this after a stored peak at least `peak_multiple` x higher.
    usable_liquidity_usd: float = Field(default=5_000.0, ge=0)
    peak_multiple: float = Field(default=4.0, gt=1)


class CrossConfirmationConfig(BaseModel):
    model_config = _FROZEN

    # Independent market confirmations (volume, trades, buyers, liquidity, price) needed
    # for the full market part of the score.
    full_market_confirmations: int = Field(default=4, ge=1, le=5)
    # Share of the family earned by market confirmations; the rest by social agreeing.
    # Market-only evidence can therefore still reach `market_share` (0.85 by default).
    market_share: float = Field(default=0.85, gt=0, le=1)
    # Subtracted when evidence contradicts itself (volume up while buyers fade).
    contradiction_penalty: float = Field(default=0.25, ge=0, le=1)


class RiskPenaltyConfig(BaseModel):
    """Points subtracted from the 0..100 score. Flags without a penalty are still shown."""

    model_config = _FROZEN

    max_total: float = Field(default=70.0, ge=0, le=100)
    freeze_authority_active: float = Field(default=25.0, ge=0)
    mint_authority_active: float = Field(default=15.0, ge=0)
    top1_holder_pct: float = Field(default=20.0, gt=0, le=100)
    top1_holder: float = Field(default=10.0, ge=0)
    top10_holders_pct: float = Field(default=50.0, gt=0, le=100)
    top10_holders: float = Field(default=15.0, ge=0)
    thin_market_pump: float = Field(default=30.0, ge=0)
    thin_pump_price_h24_pct: float = Field(default=100.0, gt=0)
    thin_pump_liquidity_usd: float = Field(default=20_000.0, gt=0)
    thin_pump_txns_h24: int = Field(default=150, ge=1)
    social_only_hype: float = Field(default=15.0, ge=0)
    distribution: float = Field(default=12.0, ge=0)
    spam_high: float = Field(default=8.0, ge=0)
    liquidity_draining: float = Field(default=12.0, ge=0)
    liquidity_drain_pct: float = Field(default=30.0, gt=0, le=100)
    unrecognized_quote: float = Field(default=8.0, ge=0)
    high_volume_to_liquidity: float = Field(default=6.0, ge=0)
    primary_pool_unclear: float = Field(default=4.0, ge=0)
    fdv_far_above_market_cap: float = Field(default=4.0, ge=0)
    very_new_pool: float = Field(default=3.0, ge=0)
    insufficient_safety_data: float = Field(default=5.0, ge=0)
    partial_safety_data: float = Field(default=2.0, ge=0)
    market_collapse: float = Field(default=30.0, ge=0)
    stale_market_data: float = Field(default=10.0, ge=0)  # carried on a last good snapshot
    flow_divergence: float = Field(default=6.0, ge=0)


class StageConfig(BaseModel):
    model_config = _FROZEN

    # Fewer 24h trades than this: not enough trustworthy evidence for any stage.
    min_txns_h24: int = Field(default=20, ge=1)
    # A pool this young with no usable acceleration evidence yet is NEW.
    new_pool_hours: float = Field(default=2.0, gt=0)
    # ACCELERATING: rising market indicators (of volume, trades, buyers) on at least this
    # many independent time scales (short windows, medium windows, stored history).
    accelerating_min_indicators: int = Field(default=2, ge=1, le=3)
    accelerating_min_timescales: int = Field(default=2, ge=1, le=3)
    # CROWDED: the move already made (%, largest of 24h / since first seen) ...
    crowded_move_pct: float = Field(default=300.0, gt=0)
    # ... alone, once it is this large, or with `crowded_min_signs` signs of maturity.
    extreme_move_pct: float = Field(default=1000.0, gt=0)
    crowded_min_signs: int = Field(default=2, ge=1)
    # "Vertical": 1h price change at least this, with liquidity not keeping pace.
    vertical_h1_pct: float = Field(default=80.0, gt=0)
    # FADING: at least this many deteriorating indicators, volume or trades among them.
    fading_min_signs: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "StageConfig":
        if self.crowded_move_pct >= self.extreme_move_pct:
            raise ValueError("crowded_move_pct must be below extreme_move_pct")
        return self


class StageAdjustmentConfig(BaseModel):
    """Points added to the base score for the stage (before the risk penalty), so a FADING
    token doesn't outrank a similarly strong ACCELERATING / EARLY one on stale earliness /
    liquidity points. Every stage stays visible and ranked."""

    model_config = _FROZEN

    accelerating: float = Field(default=8.0, ge=-50, le=50)
    early: float = Field(default=4.0, ge=-50, le=50)
    # NEW with promising evidence: buyers or price rising, liquidity confirming, no
    # thin-pump / collapse / divergence flag.
    new_promising: float = Field(default=2.0, ge=-50, le=50)
    new: float = Field(default=0.0, ge=-50, le=50)
    steady: float = Field(default=0.0, ge=-50, le=50)
    crowded: float = Field(default=-8.0, ge=-50, le=50)
    fading: float = Field(default=-12.0, ge=-50, le=50)
    insufficient_data: float = Field(default=-15.0, ge=-50, le=50)

    @model_validator(mode="after")
    def _preference_order(self) -> "StageAdjustmentConfig":
        order = [
            self.accelerating,
            self.early,
            self.new_promising,
            self.steady,
            self.crowded,
            self.fading,
            self.insufficient_data,
        ]
        if order != sorted(order, reverse=True) or self.new > self.new_promising:
            raise ValueError(
                "stage adjustments must follow ACCELERATING >= EARLY >= promising NEW >= "
                "STEADY >= CROWDED >= FADING >= INSUFFICIENT_DATA (and NEW <= promising NEW)"
            )
        return self


class StabilityConfig(BaseModel):
    """Hysteresis against one noisy window. Current evidence stays primary; a reversal
    straight from a rising stage (EARLY / ACCELERATING) to FADING, or from FADING to
    ACCELERATING, within `memory_minutes` of the stored stage needs stronger evidence,
    otherwise the token shows the intermediate stage (STEADY / EARLY) until confirmed.
    A market collapse is never held back."""

    model_config = _FROZEN

    enabled: bool = True
    memory_minutes: float = Field(default=45.0, gt=0)
    # A confirmed reversal to FADING: falling on at least this many time scales.
    fading_min_timescales: int = Field(default=2, ge=1, le=3)
    # A confirmed reversal to ACCELERATING: rising on this many more time scales than
    # ACCELERATING normally needs.
    accelerating_extra_timescales: int = Field(default=1, ge=0, le=2)


class MaturityConfig(BaseModel):
    """NEW_AND_EARLY excludes mature markets unless they show a genuinely new acceleration
    regime (stage ACCELERATING). Maturity is a weighted mean of pool age, market size and
    liquidity depth (each 0 young / small .. 1 established, log-scaled)."""

    model_config = _FROZEN

    age_weight: float = Field(default=0.5, ge=0)
    size_weight: float = Field(default=0.35, ge=0)
    depth_weight: float = Field(default=0.15, ge=0)
    # Liquidity depth: 0 at `liquidity.depth_full_usd`, 1 at this.
    mature_liquidity_usd: float = Field(default=20_000_000.0, gt=0)
    # At or above this maturity, NEW_AND_EARLY needs a new acceleration regime.
    threshold: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def _weights(self) -> "MaturityConfig":
        _check_weights(
            {"age": self.age_weight, "size": self.size_weight, "depth": self.depth_weight},
            "maturity",
        )
        return self


class EligibilityConfig(BaseModel):
    """Applied before ranking. Ineligible tokens are still reported (with the reason)."""

    model_config = _FROZEN

    # NEW_AND_EARLY: established markets are out of scope (Analyze them directly). These
    # are hard backstops; `maturity` handles everything in between.
    established_cap_usd: float = Field(default=500_000_000.0, gt=0)
    established_pool_age_days: float = Field(default=180.0, gt=0)
    maturity: MaturityConfig = MaturityConfig()
    rank_insufficient_data: bool = False


class TrackingConfig(BaseModel):
    """The ranking universe: this run's discoveries plus tokens a discovery listing
    surfaced within `horizon_hours`, refreshed by exact address in the providers'
    reserved refresh capacity. A token not rediscovered for longer expires (dead / stale
    tokens aren't tracked forever).

    When more tokens are tracked than one run can refresh, refresh priority, in minutes:

        minutes since last observed + top-N bonus + rising-stage bonus + never-refreshed bonus

    Bonuses are bounded and staleness grows, so every tracked token is revisited in turn
    (the report estimates how often). Ties rotate by a persisted run counter."""

    model_config = _FROZEN

    horizon_hours: float = Field(default=6.0, gt=0, le=72)
    # Upper bound on tracked tokens refreshed per scan (provider capacity usually binds).
    max_refresh: int = Field(default=300, ge=0, le=2000)
    top_n: int = Field(default=20, ge=0)  # "in or near the top" at the last ranking
    top_bonus_minutes: float = Field(default=240.0, ge=0)
    rising_stage_bonus_minutes: float = Field(default=120.0, ge=0)  # ACCELERATING / EARLY
    never_refreshed_bonus_minutes: float = Field(default=30.0, ge=0)
    # Used to estimate the revisit interval until two runs are stored.
    expected_interval_minutes: float = Field(default=10.0, gt=0)
    # A tracked token a provider failure kept from being refreshed is carried on its last
    # good observation, labeled stale and penalized, while that observation is at most
    # this old; never beyond. A token confirmed gone / unusable is never carried.
    grace_minutes: float = Field(default=15.0, ge=0, le=120)


class SafetyLookupConfig(BaseModel):
    model_config = _FROZEN

    # On-chain safety (Solana) looked up for at most this many top candidates per ranking
    # (0: only use snapshots handed in). Failures leave safety "insufficient", never safe.
    top_k: int = Field(default=10, ge=0, le=50)


class GrowthConfig(BaseModel):
    model_config = _FROZEN

    mode: DiscoveryMode = "NEW_AND_EARLY"
    weights: FamilyWeights = FamilyWeights()
    market: MarketActivityConfig = MarketActivityConfig()
    technical: TechnicalConfig = TechnicalConfig()
    liquidity: LiquidityConfig = LiquidityConfig()
    social: SocialScoringConfig = SocialScoringConfig()
    earliness: EarlinessConfig = EarlinessConfig()
    cross: CrossConfirmationConfig = CrossConfirmationConfig()
    risk: RiskPenaltyConfig = RiskPenaltyConfig()
    stage: StageConfig = StageConfig()
    stage_adjustment: StageAdjustmentConfig = StageAdjustmentConfig()
    stability: StabilityConfig = StabilityConfig()
    flow: FlowConfig = FlowConfig()
    collapse: CollapseConfig = CollapseConfig()
    eligibility: EligibilityConfig = EligibilityConfig()
    tracking: TrackingConfig = TrackingConfig()
    safety: SafetyLookupConfig = SafetyLookupConfig()
    # Stored social momentum older than this isn't used (the token is "unavailable").
    social_max_age_minutes: float = Field(default=90.0, gt=0)
    default_limit: int = Field(default=10, ge=1, le=500)


class GrowthConfigError(ValueError):
    pass


def load_growth_config(raw: str | None) -> GrowthConfig:
    """Defaults, overridden by a JSON object (`UPSCALE_GROWTH_CONFIG`), validated."""
    if not raw:
        return GrowthConfig()
    try:
        return GrowthConfig.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise GrowthConfigError(f"invalid Growth Scout config: {exc}") from exc
