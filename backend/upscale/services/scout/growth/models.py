"""Growth Scout's typed output: ranked discovery candidates, never a trade decision.

ScoutMomentumScore ranks discovery candidates only. It is **not** a probability of profit,
an expected return, a BUY confidence or an Opportunity confidence. A candidate is handed
to the existing Analyze pipeline, which alone decides BUY / SELL / WAIT.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from upscale.services.scout.growth.config import DiscoveryMode
from upscale.services.scout.models import ScoutFeedReport, ScoutPool
from upscale.services.scout.social.models import Level, ProviderCheck

GrowthStage = Literal[
    "NEW",  # recently launched; not enough evidence to describe acceleration yet
    "EARLY",  # activity beginning to strengthen, not yet broad
    "ACCELERATING",  # several market indicators strengthening, on several time scales
    "CROWDED",  # the move looks mature or overheated
    "FADING",  # momentum deteriorating
    "STEADY",  # enough evidence, but nothing strengthening or deteriorating
    "INSUFFICIENT_DATA",  # not enough trustworthy evidence
]
SocialStatus = Literal[
    "SOCIAL_UNAVAILABLE",  # nobody could look (not configured, failed, over budget, stale)
    "SOCIAL_QUIET",  # searched: (almost) no attributable discussion
    "SOCIAL_EMERGING",
    "SOCIAL_ACCELERATING",
    "SOCIAL_STRONG",
    "SOCIAL_STEADY",  # discussion present, roughly steady
    "SOCIAL_SATURATED",  # high attention no longer growing
    "SOCIAL_FADING",
]
VerificationStatus = Literal[
    "VERIFIED_IDENTITY",  # exact chain + contract / mint, well-formed for the chain
    "MARKET_CONFIRMED",  # a clear, priced pool in a recognized quote with real trades
    "SAFETY_CHECKS_COMPLETE",
    "SAFETY_CHECKS_PARTIAL",
    "INSUFFICIENT_SAFETY_DATA",
]
SafetyStatus = Literal[
    "SAFETY_CHECKS_COMPLETE", "SAFETY_CHECKS_PARTIAL", "INSUFFICIENT_SAFETY_DATA"
]
SocialAttribution = Literal["exact", "strong", "probable", "none"]
FamilyName = Literal[
    "market_activity", "liquidity_quality", "social_momentum", "earliness", "cross_confirmation"
]
Severity = Literal["info", "caution", "high", "critical"]
Trend = Literal["up", "down", "flat"]

NOT_A_TRADE_SIGNAL = (
    "Growth Scout ranks discovery candidates by observable momentum. ScoutMomentumScore is "
    "not a probability of profit, an expected return or a BUY confidence. Analyze a "
    "candidate for UpScale's BUY / SELL / WAIT decision."
)


class SubSignal(BaseModel):
    """One normalized measurement inside a family: 0 (strongly negative) .. 0.5 (flat /
    neutral) .. 1 (strongly positive). `raw` is the measured value it came from."""

    name: str
    score: float
    weight: float
    raw: float | None = None
    detail: str


class FamilyScore(BaseModel):
    family: FamilyName
    score: float  # 0..1; the value used in the formula
    weight: float  # normalized share of the total
    contribution: float  # score x weight x 100 (points)
    available: bool  # False: no evidence; `score` is the documented stand-in
    signals: list[SubSignal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RiskFlag(BaseModel):
    code: str
    severity: Severity
    detail: str
    penalty: float = 0.0  # points subtracted from the score


class ScoutMomentumScore(BaseModel):
    """score = clamp(base + stage adjustment - risk penalty, 0, 100), where
    base = 100 x Σ(weight x family score)."""

    score: float
    base: float  # the family contributions, before the stage adjustment and risk penalty
    stage_adjustment: float = 0.0
    risk_penalty: float
    families: list[FamilyScore]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def components(self) -> dict[str, float]:
        """Every term of the score, in points, for explaining a move up or down."""
        parts: dict[str, float] = {f.family: f.contribution for f in self.families}
        return {
            "base_score": self.base,
            **parts,
            "stage_adjustment": self.stage_adjustment,
            "risk_penalty": -self.risk_penalty,
            "final_score": self.score,
        }


class GrowthMarket(BaseModel):
    price_usd: float | None
    # Provider-reported market cap only when judged trustworthy (see `market_cap_note`).
    market_cap_usd: float | None
    market_cap_note: str | None = None
    fdv_usd: float | None
    liquidity_usd: float | None
    pool_age_hours: float | None
    oldest_pool_age_hours: float | None
    first_seen_at: datetime | None
    tracked_hours: float | None  # how long Scout has observed the token
    selected_pool: ScoutPool
    market_provider: str
    pool_count: int


class SocialProviderState(BaseModel):
    """One social provider's outcome for one token (never post text)."""

    provider: str
    status: str  # PROVIDER_OK / PROVIDER_CHECKED_ZERO_MATCHES / PROVIDER_UNAVAILABLE / ...
    detail: str | None = None  # why unavailable (e.g. credits exhausted, deferred)


class GrowthMomentum(BaseModel):
    """The measurements behind the market and social families (None: not measurable)."""

    volume_acceleration: float | None = None  # best trusted rate ratio (1 = flat)
    volume_acceleration_basis: str | None = None
    txn_acceleration: float | None = None
    txn_acceleration_basis: str | None = None
    buy_share: float | None = None  # h1 buys / trades
    buy_pressure_change: float | None = None  # change in buy share (-1..1)
    buyer_share: float | None = None  # h1 distinct buyers / (buyers + sellers)
    price_change_h1_pct: float | None = None
    price_change_h24_pct: float | None = None
    price_velocity_change_pct_per_hour: float | None = None
    liquidity_change_pct: float | None = None
    liquidity_change_basis: str | None = None
    market_cap_change_pct: float | None = None
    move_since_first_seen_pct: float | None = None
    timescales_rising: list[str] = Field(default_factory=list)
    technical: "TechnicalContext | None" = None
    social_status: SocialStatus = "SOCIAL_UNAVAILABLE"
    social_state: str | None = None  # the social layer's own momentum state
    social_reason: str | None = None
    mention_acceleration: float | None = None
    unique_author_acceleration: float | None = None
    engagement_acceleration: float | None = None
    cross_platform_corroborated: bool | None = None
    platforms_active: int | None = None
    # Per provider: searched (with or without matches), unavailable (and why), or not
    # configured. Unavailable is never zero attention.
    social_providers: list[SocialProviderState] = Field(default_factory=list)


class TechnicalContext(BaseModel):
    """From stored Scout snapshots only (no candles, no Technical Agent)."""

    snapshots: int
    span_hours: float
    trend: Trend
    change_pct: float
    breakout: bool  # above every earlier stored price by the configured margin
    volume_confirmed: bool | None  # breakout / trend with rising trading activity
    higher_lows: bool


class GrowthQuality(BaseModel):
    verification: list[VerificationStatus]
    identity_status: Literal["VERIFIED_IDENTITY", "UNVERIFIED"]
    market_confirmed: bool
    safety_status: SafetyStatus
    safety_missing: list[str] = Field(default_factory=list)  # what isn't known
    social_attribution: SocialAttribution = "none"
    exact_mention_share: float | None = None
    spam_risk: Level = "unknown"
    organic_signal: Level = "unknown"
    holder_top1_pct: float | None = None
    holder_top10_pct: float | None = None
    holder_data_lower_bound: bool | None = None
    mint_authority_active: bool | None = None
    freeze_authority_active: bool | None = None
    liquidity_quality: Literal["healthy", "thin", "draining", "unknown"] = "unknown"
    # MARKET_COLLAPSE: the market has effectively died (see `collapse_evidence`).
    market_status: Literal["OK", "MARKET_COLLAPSE"] = "OK"
    collapse_evidence: list[str] = Field(default_factory=list)
    # How far trade-count flow can be trusted: consistent with price / wallets, divergent
    # (see `flow_notes`), count-only (no distinct-wallet data), or unknown (no counts).
    flow_quality: Literal["consistent", "divergent", "count_only", "unknown"] = "unknown"
    flow_notes: list[str] = Field(default_factory=list)
    # 0 (young / small / shallow) .. 1 (established); NEW_AND_EARLY excludes >= threshold
    # unless the token is in a new acceleration regime.
    maturity: float | None = None


class GrowthCandidate(BaseModel):
    rank: int | None = None  # 1-based among eligible candidates; None when not ranked
    # Identity
    canonical_id: str
    symbol: str | None
    name: str | None
    chain: str
    address: str  # contract (EVM) or mint (Solana)
    observed_at: datetime

    # CURRENT: observed this run. STALE_CARRIED: a provider failure kept it from being
    # refreshed; ranked on its last good observation (`snapshot_age_minutes` old),
    # penalized, for a short grace period only. Never presented as current.
    data_status: Literal["CURRENT", "STALE_CARRIED"] = "CURRENT"
    snapshot_age_minutes: float | None = None
    stage: GrowthStage
    stage_reasons: list[str]
    # What the current evidence alone says, when stage stability held the stage back
    # (a one-run reversal awaiting confirmation); None when they agree.
    unconfirmed_stage: GrowthStage | None = None
    market: GrowthMarket
    momentum: GrowthMomentum
    quality: GrowthQuality
    scout_momentum: ScoutMomentumScore
    reasons_surfaced: list[str]
    risk_flags: list[RiskFlag]
    eligible: bool = True
    ineligible_reasons: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        return self.scout_momentum.score


class UniverseReport(BaseModel):
    """Where this ranking's candidates came from, and how far tracking really reaches."""

    discovered: int  # found by this run's discovery listings (including retried ones)
    discovery_retried: int = 0  # deferred / failed feeds sent on capacity refresh left
    # Per discovery provider: feeds offered, run, deferred (their turn comes), requests.
    feeds: list[ScoutFeedReport] = Field(default_factory=list)
    refreshed: int  # tracked tokens re-observed by exact address this run
    carried_stale: int = 0  # unresolved, ranked on the last good observation (labeled)
    expired: int  # tracked recently, but not rediscovered within the horizon: dropped
    unusable: int  # every lookup answered: gone or no longer usable: dropped
    unresolved: int = 0  # a lookup failed: not refreshed this run (carried if in grace)
    deferred: int = 0  # tracked, but beyond this run's refresh capacity (rotated in later)
    # Tracking reach: tokens within the horizon, what one run can refresh, and how long
    # the slowest tracked token waits between refreshes at that rate.
    horizon_hours: float
    tracked: int = 0
    refresh_capacity_requests: dict[str, int] = Field(default_factory=dict)
    refresh_capacity_tokens: int = 0
    estimated_max_revisit_minutes: float | None = None
    horizon_covered: bool = True  # every tracked token revisited within the horizon


class GrowthScoutResult(BaseModel):
    """Ranked discovery candidates. Never BUY / SELL: Analyze decides."""

    computed_at: datetime
    mode: DiscoveryMode
    evaluated: int  # candidates considered (unique canonical ids)
    eligible: int  # ranked
    limit: int | None
    candidates: list[GrowthCandidate]  # the top `limit` ranked, best first
    # Evaluated but not ranked (out of scope for the mode, insufficient data), with why.
    unranked: list[GrowthCandidate] = Field(default_factory=list)
    universe: UniverseReport | None = None  # set by `scan`
    # Set by `scan`: each social provider's run-level outcome (e.g. X credits exhausted).
    social_checks: list[ProviderCheck] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = NOT_A_TRADE_SIGNAL

    def top(self, n: int) -> list[GrowthCandidate]:
        return self.candidates[:n]


GrowthMomentum.model_rebuild()
