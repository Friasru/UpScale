"""Scout's typed domain model: discovered tokens, their market evidence, and snapshots.

Identity is always `<chain>:<token address>` (the same canonical id the rest of UpScale
uses), never a ticker: two tokens called XYZ stay two candidates. Every market figure is
exactly what a provider reported; a figure it didn't report stays None. Market cap and FDV
are kept apart: market cap is only ever the provider's reported market cap, never FDV.

Appearing in Scout is not a recommendation. A candidate is something worth analyzing; the
existing UpScale pipeline decides BUY / SELL / WAIT.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from upscale.services.chains import QuoteKind
from upscale.services.solana_dex import Window

DiscoveryKind = Literal["new", "active", "trending", "lookup"]
# Windows Scout tracks, shortest first. Providers report a subset; missing ones are absent.
SCOUT_WINDOWS: tuple[Window, ...] = ("m5", "m15", "m30", "h1", "h6", "h24")
WINDOW_MINUTES: dict[Window, int] = {
    "m5": 5,
    "m15": 15,
    "m30": 30,
    "h1": 60,
    "h6": 360,
    "h24": 1440,
}

NOT_A_RECOMMENDATION = (
    "Scout surfaces tokens with observable market activity; it is not a recommendation. "
    "Analyze a candidate to get UpScale's BUY / SELL / WAIT decision."
)


class ScoutWindow(BaseModel):
    """Trading over one rolling window of the selected pool, as reported."""

    window: Window
    volume_usd: float | None = None
    buys: int | None = None
    sells: int | None = None
    # Distinct wallets, when the provider reports them (GeckoTerminal does).
    buyers: int | None = None
    sellers: int | None = None
    price_change_pct: float | None = None

    @property
    def txns(self) -> int | None:
        if self.buys is None or self.sells is None:
            return None
        return self.buys + self.sells

    @property
    def buy_share(self) -> float | None:
        """Buys / (buys + sells), or None without trades."""
        txns = self.txns
        return self.buys / txns if txns and self.buys is not None else None


class ScoutMarketMetrics(BaseModel):
    """Market state of the token's selected pool at one moment. Nothing is estimated."""

    price_usd: float | None = None
    # The provider's reported (circulating) market cap. Never derived from FDV.
    market_cap_usd: float | None = None
    fdv_usd: float | None = None
    liquidity_usd: float | None = None
    windows: list[ScoutWindow] = Field(default_factory=list)  # only windows reported

    def window(self, name: Window) -> ScoutWindow | None:
        return next((w for w in self.windows if w.window == name), None)


class ScoutPool(BaseModel):
    """The pool chosen as the token's market (by UpScale's standard pool selection)."""

    address: str
    dex: str
    url: str | None = None
    quote_address: str
    quote_symbol: str | None = None
    quote_kind: QuoteKind
    created_at: datetime | None = None
    age_hours: float | None = None  # at observation time


class ScoutSourceEvidence(BaseModel):
    """Where and how a candidate was seen. A listing is evidence of existence, not merit."""

    provider: str
    kind: DiscoveryKind
    listing: str  # e.g. "new_pools", "trending_pools?duration=1h", "token-profiles/latest"
    fetched_at: datetime
    position: int | None = None  # 0-based position in the provider's listing, as returned
    note: str | None = None


class ScoutRiskFlags(BaseModel):
    """Plain facts about the evidence that deserve caution. Flags don't decide anything."""

    primary_pool_unclear: bool = False
    unrecognized_quote: bool = False  # USD liquidity may be inflated by an unknown quote
    market_cap_missing: bool = False
    fdv_only: bool = False  # FDV reported but no market cap
    fdv_far_above_market_cap: bool = False  # large locked / unissued supply
    very_new_pool: bool = False
    pool_age_unknown: bool = False
    high_volume_to_liquidity: bool = False  # thin pool relative to trading (easy to move)
    notes: list[str] = Field(default_factory=list)


class ScoutCandidate(BaseModel):
    # Identity
    canonical_id: str  # "<chain>:<normalized token address>"
    chain: str
    address: str  # contract (EVM) or mint (Solana), normalized for the chain
    symbol: str | None = None
    name: str | None = None

    # Discovery
    observed_at: datetime  # when the market evidence below was fetched
    first_seen_at: datetime | None = None  # first time Scout ever saw it (from the store)
    # The token's own creation time, only if a provider reports it (none does today).
    token_created_at: datetime | None = None
    # Oldest pool observed for the token: a lower bound on its age (pools can migrate).
    oldest_pool_created_at: datetime | None = None
    sources: list[ScoutSourceEvidence] = Field(default_factory=list)

    # Market
    market_provider: str  # provider whose data `pool` and `metrics` come from
    pool: ScoutPool
    metrics: ScoutMarketMetrics
    pool_count: int  # pools observed for this token with it as base token
    # Sum of USD liquidity over observed pools quoted in a recognized asset.
    recognized_quote_liquidity_usd: float | None = None
    primary_clear: bool = True
    ambiguity: list[str] = Field(default_factory=list)

    risk_flags: ScoutRiskFlags = Field(default_factory=ScoutRiskFlags)
    features: "ScoutGrowthFeatures | None" = None  # attached after comparing with history

    @property
    def token_age_hours(self) -> float | None:
        if self.token_created_at is None:
            return None
        return max(0.0, (self.observed_at - self.token_created_at).total_seconds() / 3600)


class ScoutSnapshot(BaseModel):
    """One stored observation of a token's selected pool."""

    canonical_id: str
    observed_at: datetime
    provider: str
    pool_address: str
    dex: str
    metrics: ScoutMarketMetrics
    # Reserved for later milestones; stored as NULL until a real source provides them.
    holder_count: int | None = None
    social: dict[str, Any] | None = None


# --- Derived features -----------------------------------------------------------------------


class WindowAcceleration(BaseModel):
    """Recent vs longer rolling window of the same observation, per minute. A ratio above 1
    means the recent window is running hotter than the longer window's average."""

    short: Window
    long: Window
    volume_rate_ratio: float | None = None
    txn_rate_ratio: float | None = None
    buy_share_change: float | None = None  # short buy share - long buy share (-1 .. 1)
    # Price change per hour in the short window minus that in the long window (% per hour).
    price_velocity_change_pct_per_hour: float | None = None


class HistoryComparison(BaseModel):
    """Now vs a stored snapshot taken about `lookback_minutes` earlier."""

    lookback_minutes: int
    compared_at: datetime  # the earlier snapshot's time
    elapsed_minutes: float  # actual gap (snapshots are never interpolated)
    same_pool: bool
    price_change_pct: float | None = None
    liquidity_change_pct: float | None = None
    # Only when both snapshots carry a provider-reported market cap (never FDV).
    market_cap_change_pct: float | None = None
    fdv_change_pct: float | None = None
    # Rolling-window activity now vs then (e.g. 1h volume now / 1h volume then).
    volume_window: Window | None = None
    volume_ratio: float | None = None
    txn_ratio: float | None = None
    buy_share_change: float | None = None
    notes: list[str] = Field(default_factory=list)


class ScoutGrowthFeatures(BaseModel):
    """Measurements only: nothing here ranks or recommends."""

    computed_at: datetime
    window_acceleration: list[WindowAcceleration] = Field(default_factory=list)
    history: list[HistoryComparison] = Field(default_factory=list)
    # Lookbacks with no stored snapshot close enough in time (never filled in).
    missing_lookbacks: list[int] = Field(default_factory=list)
    volume_to_liquidity_h24: float | None = None


class ScoutRejection(BaseModel):
    canonical_id: str | None
    chain: str | None
    address: str | None
    symbol: str | None
    reasons: list[str]
    provider: str | None = None


class ScoutSourceError(BaseModel):
    provider: str
    kind: DiscoveryKind
    chain: str | None
    error: str


class ScoutRun(BaseModel):
    """Result of one discovery pass."""

    started_at: datetime
    candidates: list[ScoutCandidate]
    rejected: list[ScoutRejection] = Field(default_factory=list)
    errors: list[ScoutSourceError] = Field(default_factory=list)
    disclaimer: str = NOT_A_RECOMMENDATION


ScoutCandidate.model_rebuild()
