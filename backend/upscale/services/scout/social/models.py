"""Typed model of social attention evidence for Scout.

Social activity is **evidence about attention**, never a trading signal: nothing here (and
nothing derived from it) produces BUY / SELL / WAIT. Every figure is attached to an exact
token (`<chain>:<address>`); a mention that can't be attributed to one token is kept as
AMBIGUOUS and never counted for any token.

Privacy: post text is only held in memory while a post is attributed; it is never stored.
Authors are stored only as salted, opaque keys (enough to count distinct authors and spot
one author dominating), never as handles or profiles.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from upscale.services.solana_dex import Window

AttributionLevel = Literal["EXACT", "STRONG", "PROBABLE", "AMBIGUOUS", "REJECTED"]
ProviderStatus = Literal[
    "PROVIDER_OK",  # searched successfully; at least one attributable mention
    "PROVIDER_CHECKED_ZERO_MATCHES",  # searched successfully; nothing attributable
    "PROVIDER_UNAVAILABLE",  # the search failed (outage, rate limit, timeout, bad data)
    "PROVIDER_NOT_CONFIGURED",  # no credentials / access; never searched
]
MomentumState = Literal[
    "UNAVAILABLE",
    "INSUFFICIENT_DATA",
    "QUIET",
    "EMERGING",
    "ACCELERATING",
    "STRONG",
    "STABLE",
    "SATURATED",
    "FADING",
]
Level = Literal["low", "medium", "high", "unknown"]
MarketCrossState = Literal[
    "CORROBORATED",  # attention and market activity rising together
    "UNCONFIRMED_SOCIAL_SPIKE",  # attention rising, market flat
    "CAUTION_LIQUIDITY_FALLING",  # attention rising while liquidity drains
    "MARKET_WITHOUT_SOCIAL",  # market rising, attention not
    "NO_SIGNAL",  # neither rising
    "INSUFFICIENT_DATA",
]

NOT_A_TRADING_SIGNAL = (
    "Social momentum describes public attention only. It is evidence, never a BUY / SELL "
    "signal: STRONG or ACCELERATING attention does not mean buy."
)


# --- Identity -------------------------------------------------------------------------------


class TokenIdentity(BaseModel):
    """What UpScale knows about one exact token, used to attribute mentions."""

    canonical_id: str
    chain: str
    address: str
    symbol: str | None = None
    name: str | None = None
    # Accounts / domains known to belong to the project, e.g. from the token's listed
    # profile links: `{"x": {"projecthandle"}}`, `{"project.xyz"}`. Lowercase.
    official_accounts: dict[str, set[str]] = Field(default_factory=dict)
    official_domains: set[str] = Field(default_factory=set)


# --- Provider output ------------------------------------------------------------------------


class SocialPost(BaseModel):
    """One public post as a provider returned it. Held in memory only; `text` and
    `author_handle` are never stored."""

    provider: str
    platform: str
    post_id: str
    created_at: datetime
    text: str
    author_key: str  # salted opaque key (see `fingerprint.author_key`)
    author_handle: str | None = None  # only to recognize official project accounts
    urls: list[str] = Field(default_factory=list)
    likes: int | None = None
    replies: int | None = None
    reposts: int | None = None
    quotes: int | None = None
    views: int | None = None
    promoted: bool | None = None  # None: the provider doesn't say
    source_url: str | None = None  # only when the provider's terms allow keeping it
    # The provider's own 0..1 account-quality score for the author (e.g. Neynar's user
    # score), when it gives one. Supporting evidence only: never trusted as truth.
    author_quality: float | None = None

    @property
    def engagement(self) -> int | None:
        parts = [v for v in (self.likes, self.replies, self.reposts, self.quotes) if v is not None]
        return sum(parts) if parts else None


class SocialSearchResult(BaseModel):
    provider: str
    platform: str
    checked_at: datetime
    posts: list[SocialPost]
    # Results are complete from this time to `checked_at`. When a search hit its page
    # limit this is the oldest post returned, so older windows are never assumed empty.
    complete_since: datetime


# --- Attribution and stored events ----------------------------------------------------------


class Attribution(BaseModel):
    level: AttributionLevel
    canonical_id: str | None  # None for AMBIGUOUS
    reason: str
    token_reference: str  # what in the post referred to the token (address, $TICKER, ...)
    candidates: list[str] = Field(default_factory=list)  # competing ids when AMBIGUOUS


class SocialEvent(BaseModel):
    """One attributed (or explicitly ambiguous / rejected) mention. Immutable once stored."""

    canonical_id: str | None
    provider: str
    platform: str
    posted_at: datetime
    fetched_at: datetime
    content_id: str
    author_key: str
    token_reference: str
    likes: int | None = None
    replies: int | None = None
    reposts: int | None = None
    quotes: int | None = None
    views: int | None = None
    engagement: int | None = None  # likes + replies + reposts + quotes, as first observed
    attribution_level: AttributionLevel
    attribution_reason: str
    candidates: list[str] = Field(default_factory=list)
    fingerprint: str  # exact-duplicate fingerprint of the normalized text
    simhash: str  # 64-bit near-duplicate fingerprint (hex)
    has_contract: bool
    promoted: bool | None = None
    source_url: str | None = None
    author_quality: float | None = None  # provider's author score (see SocialPost)


# --- Windows, snapshots, momentum -----------------------------------------------------------


class SocialWindowStats(BaseModel):
    """Attention in one rolling window ending at the observation time. Only windows fully
    covered by successful searches are produced; a missing window is absent, not zero."""

    window: Window
    mentions: float  # weighted attributable mentions (promoted excluded)
    unique_authors: int
    engagement: int
    exact_mentions: int
    attributable_posts: int  # EXACT + STRONG + PROBABLE posts, promoted excluded
    promoted_posts: int
    ambiguous_posts: int  # mentions that could not be attached (never counted above)
    sources: int  # platforms contributing mentions


class MetricTrend(BaseModel):
    """recent period vs the previous equivalent period vs a longer baseline, per hour."""

    recent: float
    previous: float
    baseline: float | None = None  # average per period before `previous`; None if uncovered
    velocity_per_hour: float
    previous_velocity_per_hour: float
    acceleration_per_hour: float  # velocity change between the two periods
    acceleration_ratio: float  # (recent + s) / (previous + s), s = smoothing
    baseline_ratio: float | None = None  # (previous + s) / (baseline + s)


class WindowTrend(BaseModel):
    window: Window
    mentions: MetricTrend
    unique_authors: MetricTrend
    engagement: MetricTrend


class SocialQuality(BaseModel):
    spam_risk: Level
    organic_signal_strength: Level
    reasons: list[str] = Field(default_factory=list)
    posts: int = 0
    unique_authors: int = 0
    top_author_share: float | None = None
    authors_per_mention: float | None = None
    duplicate_share: float | None = None
    repeated_contract_share: float | None = None
    promoted_share: float | None = None
    conflicting_contracts: int = 0
    # Provider-supplied author scores (e.g. Neynar), reported beside UpScale's own
    # heuristics above and never used by them: supporting evidence, not truth.
    provider_scored_posts: int = 0
    provider_median_author_quality: float | None = None
    provider_low_quality_share: float | None = None


class ProviderCheck(BaseModel):
    provider: str
    platform: str
    status: ProviderStatus
    checked_at: datetime
    error: str | None = None
    requirement: str | None = None  # what access is missing, for NOT_CONFIGURED


class SocialSourceSnapshot(BaseModel):
    """One provider's view of one token at one time. Append-only in storage."""

    canonical_id: str
    provider: str
    platform: str
    observed_at: datetime
    status: ProviderStatus
    windows: list[SocialWindowStats] = Field(default_factory=list)
    trends: list[WindowTrend] = Field(default_factory=list)
    accelerating: bool = False
    active: bool = False
    error: str | None = None


class CrossPlatformConfirmation(BaseModel):
    providers_configured: int
    providers_checked: int  # searched successfully this run (OK or zero matches)
    platforms_with_activity: int
    platforms_accelerating: int
    source_diversity: float | None  # active / checked; None when nothing was checked
    corroborated: bool  # acceleration on at least `min_platforms` independent platforms
    single_source_available: bool  # only one platform could be checked: not a penalty
    only_one_active_of_several: bool  # several checked, activity on just one
    notes: list[str] = Field(default_factory=list)


class MarketCrossCheck(BaseModel):
    state: MarketCrossState
    reasons: list[str] = Field(default_factory=list)
    market_window: str | None = None  # e.g. "m15/h1"
    volume_rate_ratio: float | None = None
    txn_rate_ratio: float | None = None
    liquidity_change_pct: float | None = None


class SocialMomentum(BaseModel):
    """Attention dynamics for one token. Describes attention only; never a trade signal."""

    canonical_id: str
    computed_at: datetime
    state: MomentumState
    reasons: list[str] = Field(default_factory=list)
    window: Window  # the window the state is judged on
    trend: WindowTrend | None = None  # combined across platforms, for `window`
    windows: list[SocialWindowStats] = Field(default_factory=list)  # combined
    trends: list[WindowTrend] = Field(default_factory=list)  # combined, every covered window
    quality: SocialQuality
    cross_platform: CrossPlatformConfirmation
    market: MarketCrossCheck
    sources: list[SocialSourceSnapshot] = Field(default_factory=list)
    disclaimer: str = NOT_A_TRADING_SIGNAL


class SocialRun(BaseModel):
    started_at: datetime
    momentum: list[SocialMomentum]
    providers: list[ProviderCheck]
    ambiguous_mentions: int = 0
    rejected_mentions: int = 0
    disclaimer: str = NOT_A_TRADING_SIGNAL
