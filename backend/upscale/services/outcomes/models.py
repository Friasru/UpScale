"""Outcome tracking's typed records: what UpScale knew at a moment, and what happened after.

* `ScoutObservation`: a ranked Growth Scout candidate at one evaluation anchor. Immutable.
* `DecisionObservation`: an Analyze BUY / SELL / WAIT decision. Immutable.
* `HorizonOutcome`: the measured market behavior over one fixed horizon after either.

Outcomes are factual metrics (returns, excursions, liquidity changes, stage changes,
trigger touches). Nothing here is a win / loss label, a win rate, an expected profit or a
simulated trade: no entry, fill, size, fee or slippage is ever assumed.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

OUTCOME_SCHEMA_VERSION = 1

AnchorReason = Literal["FIRST_RANKED", "STAGE_CHANGE", "SCORE_CHANGE", "REENTRY", "TIME_ELAPSED"]
DiscoveryStatus = Literal["NEW", "DISCOVERED", "REFRESHED", "STALE_CARRIED"]
HorizonStatus = Literal["PENDING", "COMPLETE", "PARTIAL", "UNAVAILABLE"]
FINAL_STATUSES: tuple[HorizonStatus, ...] = ("COMPLETE", "PARTIAL", "UNAVAILABLE")
MarketStatus = Literal[
    "ACTIVE",
    "LIQUIDITY_COLLAPSE",
    "POOL_GONE",
    "MARKET_UNAVAILABLE",
    "PROVIDER_UNAVAILABLE",
    "TOKEN_INACTIVE",
    "UNKNOWN",
]
PriceSource = Literal["candles", "snapshots"]
MarketSource = Literal["scout_snapshot", "pool_lookup"]

NOT_A_PERFORMANCE_CLAIM = (
    "Outcome tracking records what the market did after UpScale surfaced or analyzed an "
    "asset. It is not a win rate, an expected return or a profitability claim, and no "
    "trade (entry, fill, size, fees, slippage) is assumed."
)


# --- Shared ---------------------------------------------------------------------------------


class PriceRef(BaseModel):
    """The exact market whose prices measure an outcome. Never a ticker match: a DEX pool
    of the exact token (chain + pool, token checked), or the exact exchange market."""

    kind: Literal["dex_pool", "exchange"]
    chain: str | None = None
    pool_address: str | None = None
    token_address: str | None = None
    provider: str | None = None  # exchange: the candle provider (e.g. Kraken)
    pair: str | None = None  # exchange: its pair (e.g. XBT/USD)
    symbol: str | None = None  # exchange: the symbol UpScale asked the provider for


class PricePath(BaseModel):
    """Price behavior over the horizon window [start, end] from the reference price.

    MFE = max(0, highest / reference - 1); MAE = min(0, lowest / reference - 1): the path
    starts at the reference price. With `source="snapshots"` extremes are only the prices
    Scout happened to observe (a lower bound on the true excursions)."""

    source: PriceSource
    provider: str | None = None
    timeframe: str | None = None  # candles only
    points: int  # candles (or snapshots) inside the window
    window_start: datetime
    window_end: datetime
    reference_price: float
    end_price: float | None = None
    end_price_at: datetime | None = None
    return_pct: float | None = None
    highest_price: float | None = None
    highest_at: datetime | None = None
    lowest_price: float | None = None
    lowest_at: datetime | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    time_to_mfe_minutes: float | None = None
    time_to_mae_minutes: float | None = None
    # Largest peak-to-trough decline inside the window (%, <= 0).
    max_drawdown_pct: float | None = None
    price_collapsed: bool | None = None
    notes: list[str] = Field(default_factory=list)


class MarketAtHorizon(BaseModel):
    """The exact pool's market state near the horizon end, vs the observation."""

    source: MarketSource
    provider: str
    observed_at: datetime
    pool_found: bool
    price_usd: float | None = None
    liquidity_usd: float | None = None
    liquidity_change_pct: float | None = None
    market_cap_usd: float | None = None  # only when both ends are trustworthy
    market_cap_change_pct: float | None = None
    fdv_usd: float | None = None
    fdv_change_pct: float | None = None
    volume_h1_usd: float | None = None
    volume_h1_change_pct: float | None = None
    volume_h24_usd: float | None = None
    volume_h24_change_pct: float | None = None
    txns_h1: int | None = None
    txns_h1_change_pct: float | None = None
    buy_share_h1: float | None = None
    buy_share_change: float | None = None  # -1..1
    liquidity_collapsed: bool | None = None


class TriggerOutcome(BaseModel):
    """Whether actual prices reached a decision level (intrabar high / low) and when.
    A touch is price evidence only: it is not a fill and not a close-confirmed trigger."""

    level: float
    direction: Literal["above", "below"]  # reached when price is at or beyond the level
    reached: bool | None  # None: no price evidence
    first_reached_at: datetime | None = None  # the open time of the first candle reaching it
    already_beyond_at_start: bool | None = None


class HorizonOutcome(BaseModel):
    observation_id: int
    horizon: str
    horizon_minutes: int
    due_at: datetime
    status: HorizonStatus = "PENDING"
    market_status: MarketStatus | None = None
    price: PricePath | None = None
    market: MarketAtHorizon | None = None
    future_stage: str | None = None
    future_stage_at: datetime | None = None
    future_rank: int | None = None
    future_score: float | None = None
    # Decision horizons only.
    triggers: dict[str, TriggerOutcome] = Field(default_factory=dict)
    first_trigger_event: str | None = None  # buy_trigger / sell_trigger / invalidation / ...
    price_position: str | None = None  # the end price vs the decision's levels
    # Why anything is missing (never silent).
    missing: list[str] = Field(default_factory=list)
    attempts: int = 0
    last_attempt_at: datetime | None = None
    finalized_at: datetime | None = None


# --- Scout observations --------------------------------------------------------------------


class ScoreComponents(BaseModel):
    market_activity: float | None = None
    liquidity_quality: float | None = None
    social_momentum: float | None = None
    earliness: float | None = None
    cross_confirmation: float | None = None
    base: float
    stage_adjustment: float
    risk_penalty: float


class ObservedMarket(BaseModel):
    price_usd: float | None
    market_cap_usd: float | None  # only when Growth Scout judged it trustworthy
    market_cap_note: str | None = None
    fdv_usd: float | None
    liquidity_usd: float | None
    volume_h1_usd: float | None = None
    volume_h24_usd: float | None = None
    txns_h1: int | None = None
    txns_h24: int | None = None
    buy_share_h1: float | None = None
    buyer_share_h1: float | None = None
    volume_acceleration: float | None = None
    txn_acceleration: float | None = None
    buy_pressure_change: float | None = None
    price_change_h1_pct: float | None = None
    price_change_h24_pct: float | None = None
    pool_age_hours: float | None = None
    tracked_hours: float | None = None


class ObservedSocial(BaseModel):
    status: str
    state: str | None = None
    x_state: str | None = None
    farcaster_state: str | None = None
    spam_risk: str
    attribution: str
    cross_platform: bool | None = None


class ObservedSafety(BaseModel):
    status: str
    holder_top1_pct: float | None = None
    holder_top10_pct: float | None = None
    holder_data_lower_bound: bool | None = None
    mint_authority_active: bool | None = None
    freeze_authority_active: bool | None = None
    liquidity_quality: str
    market_status: str


class ObservedFlag(BaseModel):
    code: str
    severity: str
    detail: str


class ScoutObservation(BaseModel):
    id: int | None = None
    schema_version: int = OUTCOME_SCHEMA_VERSION
    # Identity (symbol / name: display only)
    canonical_id: str
    chain: str
    address: str
    symbol: str | None
    name: str | None
    pool_address: str
    pool_dex: str
    quote_symbol: str | None
    market_provider: str
    # Timing
    observed_at: datetime  # when this market evidence was observed (the outcome start)
    anchored_at: datetime  # the Scout ranking run that produced the anchor
    run_id: str
    anchor_reason: AnchorReason
    previous_observation_id: int | None = None
    # Discovery
    rank: int
    ranking_mode: str
    in_top10: bool
    in_top20: bool
    discovery_status: DiscoveryStatus
    snapshot_age_minutes: float
    # Scout state
    stage: str
    unconfirmed_stage: str | None = None
    stage_reasons: list[str] = Field(default_factory=list)
    score: float
    components: ScoreComponents
    market: ObservedMarket
    social: ObservedSocial
    safety: ObservedSafety
    risk_flags: list[ObservedFlag] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @property
    def price_ref(self) -> PriceRef:
        return PriceRef(
            kind="dex_pool",
            chain=self.chain,
            pool_address=self.pool_address,
            token_address=self.address,
        )


# --- Decision observations -----------------------------------------------------------------


class DecisionLevel(BaseModel):
    price: float
    condition: str
    basis: str
    confirmed: bool | None = None


class DecisionInvalidation(BaseModel):
    price: float
    condition: str
    basis: str
    direction: Literal["above", "below"]


class DecisionRisk(BaseModel):
    level: str
    uncertainty_level: str
    summary: str | None = None


class DecisionObservation(BaseModel):
    id: int | None = None
    schema_version: int = OUTCOME_SCHEMA_VERSION
    analyzed_at: datetime
    source: Literal["scout", "chat"]
    scout_observation_id: int | None = None
    # Identity: chain:address for tokens, exchange:<provider>:<pair> for exchange markets.
    asset_id: str
    symbol: str | None
    chain: str | None = None
    address: str | None = None
    price_ref: PriceRef
    # The decision, as shown
    action: Literal["buy", "sell", "wait"]
    setup_action: str | None = None
    setup: str | None = None
    confidence: str
    position: str
    intent: str
    action_meaning: str | None = None
    reason: str
    timeframe: str | None
    buy_trigger: DecisionLevel | None = None
    sell_trigger: DecisionLevel | None = None
    invalidation: DecisionInvalidation | None = None
    other_invalidations: list[DecisionInvalidation] = Field(default_factory=list)
    entry_zone: tuple[float, float] | None = None
    reference_price: float | None
    reference_basis: Literal["live_price", "last_close"] | None
    last_close: float | None = None
    liquidity_usd: float | None = None  # DEX pool liquidity at decision time
    bullish_score: int | None = None
    bearish_score: int | None = None
    risk: DecisionRisk


# --- Reports -------------------------------------------------------------------------------


class ScoutOutcomeRecord(BaseModel):
    observation: ScoutObservation
    horizons: list[HorizonOutcome]


class DecisionOutcomeRecord(BaseModel):
    observation: DecisionObservation
    horizons: list[HorizonOutcome]


class SurfacingHistory(BaseModel):
    """How often Growth Scout ranked a token before (for the UI's "See more")."""

    first_ranked_at: datetime
    times_ranked: int
