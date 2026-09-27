"""Every social threshold and weight, validated. Change these values, not the logic."""

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from upscale.services.scout.config import ScoutProviderLimits
from upscale.services.scout.models import WINDOW_MINUTES
from upscale.services.solana_dex import Window

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class AttributionConfig(BaseModel):
    model_config = _FROZEN

    # How much a PROBABLE mention counts (EXACT / STRONG count 1; AMBIGUOUS / REJECTED 0).
    probable_weight: float = Field(default=0.5, ge=0, le=1)
    # A name shorter than this is too generic to be matched as a phrase.
    min_name_length: int = Field(default=4, ge=2)


class MomentumConfig(BaseModel):
    model_config = _FROZEN

    windows: tuple[Window, ...] = ("m5", "m15", "m30", "h1", "h6", "h24")
    state_window: Window = "h1"  # the window the momentum state is judged on
    # Periods of the same length before the previous period, averaged into the baseline.
    baseline_periods: int = Field(default=4, ge=1, le=24)
    # Added to both sides of a ratio so small counts can't produce huge ratios.
    smoothing: float = Field(default=1.0, gt=0)
    # Weighted mentions in the recent period below which attention is QUIET.
    min_mentions: float = Field(default=3.0, gt=0)
    accelerating_ratio: float = Field(default=1.5, gt=1)  # recent / previous
    author_accelerating_ratio: float = Field(default=1.3, gt=1)  # unique authors
    fading_ratio: float = Field(default=0.75, gt=0, lt=1)  # recent / previous
    # EMERGING: attention appears from (almost) nothing: previous below this.
    emerging_prior_max: float = Field(default=1.0, ge=0)
    # STRONG: accelerating, sustained over the baseline, and broad.
    strong_baseline_ratio: float = Field(default=1.25, gt=1)
    strong_min_authors: int = Field(default=10, ge=1)
    # SATURATED: very high attention that is no longer growing.
    saturated_min_mentions: float = Field(default=200.0, gt=0)

    @model_validator(mode="after")
    def _state_window_listed(self) -> "MomentumConfig":
        if self.state_window not in self.windows:
            raise ValueError("state_window must be one of windows")
        return self


class QualityConfig(BaseModel):
    model_config = _FROZEN

    min_posts: int = Field(default=5, ge=1)  # fewer posts: quality is "unknown"
    top_author_share_medium: float = Field(default=0.3, gt=0, le=1)
    top_author_share_high: float = Field(default=0.5, gt=0, le=1)
    duplicate_share_medium: float = Field(default=0.25, gt=0, le=1)
    duplicate_share_high: float = Field(default=0.5, gt=0, le=1)
    near_duplicate_bits: int = Field(default=3, ge=0, le=16)  # simhash Hamming distance
    # An author posting the contract this many times counts as repeated promotion.
    contract_repeats_per_author: int = Field(default=3, ge=2)
    repeated_contract_share_medium: float = Field(default=0.3, gt=0, le=1)
    promoted_share_medium: float = Field(default=0.2, gt=0, le=1)
    promoted_share_high: float = Field(default=0.5, gt=0, le=1)
    # A burst (accelerating) with fewer distinct authors per mention than this.
    low_diversity_authors_per_mention: float = Field(default=0.3, gt=0, le=1)
    organic_medium_authors: int = Field(default=5, ge=1)
    organic_high_authors: int = Field(default=20, ge=1)
    organic_high_authors_per_mention: float = Field(default=0.6, gt=0, le=1)
    # Provider author scores below this count as "low quality" in the separate provider
    # evidence (0.55 is Neynar's suggested starting point). Never affects spam_risk.
    provider_low_quality_below: float = Field(default=0.55, ge=0, le=1)


class CrossPlatformConfig(BaseModel):
    model_config = _FROZEN

    min_platforms: int = Field(default=2, ge=2)  # accelerating platforms for "corroborated"


class MarketCheckConfig(BaseModel):
    model_config = _FROZEN

    # Scout market window pairs, first available used (see Scout's window_acceleration).
    window_pairs: tuple[tuple[Window, Window], ...] = (("m15", "h1"), ("m5", "h1"), ("h1", "h6"))
    rising_ratio: float = Field(default=1.3, gt=1)  # market volume / trade rate ratio
    flat_ratio: float = Field(default=1.1, gt=0)
    liquidity_drop_pct: float = Field(default=25.0, gt=0, le=100)
    liquidity_lookback_minutes: int = Field(default=60, gt=0)


class SocialProviderConfig(BaseModel):
    model_config = _FROZEN

    limits: ScoutProviderLimits
    max_results_per_page: int = Field(default=100, gt=0, le=100)
    max_pages: int = Field(default=1, ge=1, le=10)
    # Characters allowed in one search query (terms are batched up to this).
    max_query_chars: int = Field(default=500, ge=50)
    max_terms_per_query: int = Field(default=10, ge=1)
    # Outgoing requests one provider may make in one run (None: only the rate limit).
    # Requests past it fail fast and the affected tokens are reported unavailable.
    max_requests_per_run: int | None = Field(default=None, ge=1)


class SocialConfig(BaseModel):
    model_config = _FROZEN

    attribution: AttributionConfig = AttributionConfig()
    momentum: MomentumConfig = MomentumConfig()
    quality: QualityConfig = QualityConfig()
    cross_platform: CrossPlatformConfig = CrossPlatformConfig()
    market: MarketCheckConfig = MarketCheckConfig()
    # How far back a first search reaches; later searches only fetch what's new.
    search_span_hours: float = Field(default=48.0, gt=0, le=168)
    overlap_minutes: float = Field(default=5.0, ge=0)  # re-read this much on each search
    # Stored mentions older than this are deleted (no long-term author histories).
    event_retention_hours: float = Field(default=72.0, gt=0)
    reddit: SocialProviderConfig = SocialProviderConfig(
        limits=ScoutProviderLimits(calls_per_minute=60, cache_ttl_seconds=60)
    )
    # Neynar: one term per query (see NeynarFarcasterProvider); contract searches can take
    # several seconds; each request consumes Neynar credits, so runs are capped.
    farcaster: SocialProviderConfig = SocialProviderConfig(
        limits=ScoutProviderLimits(calls_per_minute=30, cache_ttl_seconds=60, timeout_seconds=15),
        max_terms_per_query=1,
        max_requests_per_run=60,
    )
    x: SocialProviderConfig = SocialProviderConfig(
        limits=ScoutProviderLimits(calls_per_minute=10, cache_ttl_seconds=120),
        max_query_chars=512,
    )
    discourse: SocialProviderConfig = SocialProviderConfig(
        limits=ScoutProviderLimits(calls_per_minute=20, max_concurrency=2, cache_ttl_seconds=120),
        max_results_per_page=50,
        max_terms_per_query=1,
    )
    # X is paid per post read: never searched unless explicitly allowed, and capped.
    x_allow_paid: bool = False
    x_max_reads_per_run: int = Field(default=500, ge=0)

    @model_validator(mode="after")
    def _span_covers_windows(self) -> "SocialConfig":
        need = WINDOW_MINUTES[self.momentum.state_window] * 2 / 60
        if self.search_span_hours < need:
            raise ValueError(
                f"search_span_hours must cover two {self.momentum.state_window} periods"
            )
        return self


class SocialConfigError(ValueError):
    pass


def load_social_config(raw: str | None) -> SocialConfig:
    """Defaults, overridden by a JSON object (`UPSCALE_SOCIAL_CONFIG`), validated."""
    if not raw:
        return SocialConfig()
    try:
        return SocialConfig.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise SocialConfigError(f"invalid social config: {exc}") from exc
