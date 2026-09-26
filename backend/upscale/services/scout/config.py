"""Every Scout threshold, validated. Change these values, not the logic.

Defaults are deliberately permissive: filters only remove what is clearly unusable (no
verifiable identity, effectively no liquidity, no trades, malformed data). They are not
trading thresholds.
"""

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from upscale.services.chains import DEX_CHAINS, GECKOTERMINAL_NETWORKS
from upscale.services.scout.models import WINDOW_MINUTES
from upscale.services.solana_dex import Window


class ScoutFilterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Below this USD liquidity a pool is effectively empty (same floor as pool selection).
    min_liquidity_usd: float = Field(default=1_000.0, ge=0)
    # A token with fewer trades than this over 24h has no real trading activity.
    min_txns_h24: int = Field(default=1, ge=0)
    # Chains whose address format UpScale can verify. A token on another chain has no
    # verifiable canonical identity and is rejected unless this is switched off.
    require_verifiable_address: bool = True


class ScoutRiskFlagConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    very_new_pool_hours: float = Field(default=24.0, gt=0)
    fdv_to_market_cap_ratio: float = Field(default=3.0, gt=1)
    high_volume_to_liquidity_h24: float = Field(default=10.0, gt=0)


class ScoutFeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Minutes back to compare against stored snapshots.
    lookback_minutes: tuple[int, ...] = (5, 15, 30, 60, 360, 1440)
    # How far an earlier snapshot may be from the exact lookback, as a fraction of it (at
    # least `min_tolerance_seconds`). Beyond that the lookback is reported missing.
    lookback_tolerance: float = Field(default=0.25, gt=0, le=0.5)
    min_tolerance_seconds: float = Field(default=60.0, ge=0)
    # Short vs long windows compared inside one observation.
    window_pairs: tuple[tuple[Window, Window], ...] = (
        ("m5", "h1"),
        ("m15", "h1"),
        ("h1", "h6"),
        ("h1", "h24"),
        ("h6", "h24"),
    )
    # Rolling window compared between snapshots (the first one both snapshots report).
    history_windows: tuple[Window, ...] = ("h1", "m30", "m15", "h6", "h24")

    @field_validator("lookback_minutes")
    @classmethod
    def _positive_lookbacks(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(m <= 0 for m in value):
            raise ValueError("lookbacks must be positive minutes")
        return tuple(sorted(set(value)))

    @field_validator("window_pairs")
    @classmethod
    def _short_before_long(
        cls, value: tuple[tuple[Window, Window], ...]
    ) -> tuple[tuple[Window, Window], ...]:
        for short, long in value:
            if WINDOW_MINUTES[short] >= WINDOW_MINUTES[long]:
                raise ValueError(f"window pair {short}/{long}: the first must be shorter")
        return value


class ScoutProviderLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    calls_per_minute: int = Field(gt=0)
    max_concurrency: int = Field(default=4, gt=0)
    timeout_seconds: float = Field(default=10.0, gt=0)
    cache_ttl_seconds: float = Field(default=30.0, ge=0)


class ScoutConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chains: tuple[str, ...] = ("solana", "ethereum", "base", "bsc")
    kinds: tuple[str, ...] = ("new", "active", "trending")
    # Listing entries taken per provider, kind and chain.
    max_per_listing: int = Field(default=40, gt=0, le=200)
    # Batch size for exact-token lookups (both providers accept 30 addresses per request).
    lookup_batch_size: int = Field(default=30, gt=0, le=30)
    # Minimum seconds between stored snapshots of the same token (repeat fetches of cached
    # data are never stored twice either way).
    min_snapshot_interval_seconds: float = Field(default=30.0, ge=0)
    filters: ScoutFilterConfig = ScoutFilterConfig()
    flags: ScoutRiskFlagConfig = ScoutRiskFlagConfig()
    features: ScoutFeatureConfig = ScoutFeatureConfig()
    # Stay below each provider's published limits (GeckoTerminal ~30/min, DEX Screener
    # 60/min for listings and 300/min for pair lookups).
    geckoterminal: ScoutProviderLimits = ScoutProviderLimits(calls_per_minute=20)
    dexscreener: ScoutProviderLimits = ScoutProviderLimits(calls_per_minute=50)

    @model_validator(mode="after")
    def _known_values(self) -> "ScoutConfig":
        unknown = [c for c in self.chains if c not in DEX_CHAINS | set(GECKOTERMINAL_NETWORKS)]
        if unknown:
            raise ValueError(f"no DEX provider serves chain(s) {', '.join(unknown)}")
        bad = [k for k in self.kinds if k not in ("new", "active", "trending")]
        if bad:
            raise ValueError(f"unknown discovery kind(s) {', '.join(bad)}")
        return self


class ScoutConfigError(ValueError):
    pass


def load_scout_config(raw: str | None) -> ScoutConfig:
    """Defaults, overridden by a JSON object (e.g. `UPSCALE_SCOUT_CONFIG`), validated."""
    if not raw:
        return ScoutConfig()
    try:
        return ScoutConfig.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise ScoutConfigError(f"invalid Scout config: {exc}") from exc
