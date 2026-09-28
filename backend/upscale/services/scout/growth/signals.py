"""Evidence extraction and normalization for Growth Scout. Pure functions, no I/O.

Everything is read from what Scout and the social layer already measured (market windows,
window acceleration, stored-history comparisons, stored snapshots, SocialMomentum, cached
on-chain safety). Nothing is fetched and nothing is estimated: a measurement that isn't
available stays None and is reported as missing.

**Normalization.** Raw values are never combined directly. Every sub-signal is mapped to
0..1 where 0.5 means flat / neutral:

* rate ratios (acceleration) are log-scaled: ``0.5 + 0.5 x ln(r) / ln(full)``, clamped, so
  3x and 30x both saturate at 1 and 1/3x and 1/30x at 0;
* percentage changes are converted to log returns first (+100% and -50% are symmetric);
* absolute sizes (liquidity) are log-scaled between a floor and a cap, so $100M of
  liquidity doesn't beat $1M by 100x.

**Trusted windows.** A rolling-window pair is only used when the pool is at least as old as
the long window (a 40-minute-old pool's "6h" window is really 40 minutes, which would
fake a large acceleration) and the short window had enough trades to mean something.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.growth.models import (
    SocialAttribution,
    SocialStatus,
    TechnicalContext,
    Trend,
)
from upscale.services.scout.models import (
    WINDOW_MINUTES,
    HistoryComparison,
    ScoutCandidate,
    ScoutSnapshot,
    WindowAcceleration,
)
from upscale.services.scout.social.models import (
    Level,
    SocialMomentum,
    SocialWindowStats,
    WindowTrend,
)
from upscale.services.solana_dex import Window

# --- Normalization --------------------------------------------------------------------------


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ratio_score(ratio: float | None, full: float) -> float | None:
    """1 -> 0.5; `full` -> 1; 1/`full` -> 0 (log scale)."""
    if ratio is None or ratio <= 0 or not math.isfinite(ratio):
        return None
    return clamp(0.5 + 0.5 * math.log(ratio) / math.log(full))


def pct_score(pct: float | None, full_pct: float) -> float | None:
    """A % change on the log-return scale: 0% -> 0.5; +`full_pct` -> 1; the symmetric
    loss (e.g. -23% for +30%) -> 0. A total loss (-100%: providers round a collapse to
    it) is the worst score, never a missing one; below -100% is invalid data."""
    if pct is None or pct < -100 or not math.isfinite(pct):
        return None
    if pct == -100:
        return 0.0
    return ratio_score(1 + pct / 100, 1 + full_pct / 100)


def linear_score(x: float | None, low: float, high: float) -> float | None:
    if x is None:
        return None
    return clamp((x - low) / (high - low))


def log_band_score(x: float | None, floor: float, full: float) -> float | None:
    """0 at or below `floor`, 1 at or above `full`, log-interpolated between."""
    if x is None:
        return None
    if x <= floor:
        return 0.0
    return clamp(math.log(x / floor) / math.log(full / floor))


def mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


# --- Market evidence ------------------------------------------------------------------------


@dataclass
class Timescale:
    """Acceleration measured at one time scale: short windows, medium windows, or now vs a
    stored snapshot."""

    name: str  # "short" | "medium" | "history"
    basis: str  # e.g. "m15 vs h1 rate", "h1 now vs 60m ago"
    volume_ratio: float | None
    txn_ratio: float | None
    buy_share_change: float | None
    sells_ratio: float | None = None  # sell-count rate ratio (short windows only)


@dataclass
class MarketEvidence:
    pool_age_hours: float | None
    oldest_pool_age_hours: float | None
    txns_h24: int | None
    timescales: list[Timescale] = field(default_factory=list)
    buy_share: float | None = None
    buyer_share: float | None = None
    price_h1: float | None = None
    price_h6: float | None = None
    price_h24: float | None = None
    price_velocity_change: float | None = None
    liquidity_change_pct: float | None = None
    liquidity_change_basis: str | None = None
    liquidity_min_change_pct: float | None = None  # worst change over every lookback
    price_change_same_lookback: float | None = None  # price change over the liquidity basis
    market_cap_change_pct: float | None = None
    move_since_first_seen_pct: float | None = None
    price_change_1440: float | None = None
    technical: TechnicalContext | None = None
    stored_snapshots: int = 0  # earlier stored observations (same provider): history depth
    liquidity_peak_usd: float | None = None  # highest stored liquidity of the same pool
    # Trade-rate ratios within this observation (None when a window is missing / partial):
    activity_h1_vs_h24: float | None = None  # for "activity dying"
    activity_h6_vs_h24: float | None = None  # activity regime (pools a day old or more)
    ignored: list[str] = field(default_factory=list)  # evidence dropped, and why

    @property
    def move_extent_pct(self) -> float | None:
        """How much the price already moved: the largest of the 24h change, the change
        since Scout first saw the token, and the change vs a day-old snapshot."""
        moves = [
            m
            for m in (self.price_h24, self.move_since_first_seen_pct, self.price_change_1440)
            if m is not None
        ]
        return max(moves) if moves else None

    def ratios(self, metric: str) -> list[tuple[str, float]]:
        out = []
        for t in self.timescales:
            value = t.volume_ratio if metric == "volume" else t.txn_ratio
            if value is not None:
                out.append((t.basis, value))
        return out

    @property
    def buy_share_change(self) -> float | None:
        return next(
            (t.buy_share_change for t in self.timescales if t.buy_share_change is not None), None
        )


def _age_hours(created: datetime | None, at: datetime) -> float | None:
    if created is None:
        return None
    return max(0.0, (at - created).total_seconds() / 3600)


def _trusted_pair(
    c: ScoutCandidate,
    a: WindowAcceleration,
    age_hours: float | None,
    cfg: GrowthConfig,
    ignored: list[str],
) -> bool:
    if age_hours is not None and age_hours * 60 < WINDOW_MINUTES[a.long]:
        ignored.append(
            f"{a.short}/{a.long}: the pool is only {age_hours:.1f}h old, so its {a.long} "
            "window is partial"
        )
        return False
    short = c.metrics.window(a.short)
    txns = short.txns if short else None
    if txns is None or txns < cfg.market.min_short_window_txns:
        ignored.append(f"{a.short}/{a.long}: only {txns or 0} trades in {a.short}")
        return False
    return a.volume_rate_ratio is not None or a.txn_rate_ratio is not None


def _pair(
    c: ScoutCandidate,
    pairs: Sequence[tuple[Window, Window]],
    age_hours: float | None,
    cfg: GrowthConfig,
    ignored: list[str],
) -> WindowAcceleration | None:
    feats = c.features
    if feats is None:
        return None
    for short, long in pairs:
        a = next(
            (x for x in feats.window_acceleration if x.short == short and x.long == long), None
        )
        if a is not None and _trusted_pair(c, a, age_hours, cfg, ignored):
            return a
    return None


def _sells_ratio(c: ScoutCandidate, short: Window, long: Window) -> float | None:
    s, lw = c.metrics.window(short), c.metrics.window(long)
    if s is None or lw is None or s.sells is None or not lw.sells:
        return None
    return (s.sells / WINDOW_MINUTES[short]) / (lw.sells / WINDOW_MINUTES[long])


def _history(
    c: ScoutCandidate, lookbacks: Sequence[int], age_hours: float | None, ignored: list[str]
) -> HistoryComparison | None:
    feats = c.features
    if feats is None:
        return None
    by_lookback = {h.lookback_minutes: h for h in feats.history}
    for minutes in lookbacks:
        h = by_lookback.get(minutes)
        if h and h.same_pool and (h.volume_ratio is not None or h.txn_ratio is not None):
            # A window the pool was younger than at the earlier snapshot only held the
            # minutes since launch: the ratio would measure the pool filling up, not
            # acceleration.
            then = age_hours * 60 - h.elapsed_minutes if age_hours is not None else None
            if then is not None and h.volume_window and then < WINDOW_MINUTES[h.volume_window]:
                ignored.append(
                    f"{h.volume_window} now vs {h.elapsed_minutes:.0f}m ago: the pool was only "
                    f"{max(0.0, then):.0f} minutes old then, so its {h.volume_window} window "
                    "was partial"
                )
                continue
            return h
    return None


def market_evidence(
    c: ScoutCandidate, history: Sequence[ScoutSnapshot], cfg: GrowthConfig
) -> MarketEvidence:
    m = c.metrics
    pool_age = c.pool.age_hours
    if pool_age is None:
        pool_age = _age_hours(c.pool.created_at, c.observed_at)
    ev = MarketEvidence(
        pool_age_hours=pool_age,
        oldest_pool_age_hours=_age_hours(c.oldest_pool_created_at, c.observed_at),
        txns_h24=(day.txns if (day := m.window("h24")) else None),
    )
    for name, pairs in (("short", cfg.market.short_pairs), ("medium", cfg.market.medium_pairs)):
        a = _pair(c, pairs, pool_age, cfg, ev.ignored)
        if a is not None:
            ev.timescales.append(
                Timescale(
                    name=name,
                    basis=f"{a.short} vs {a.long} rate",
                    volume_ratio=a.volume_rate_ratio,
                    txn_ratio=a.txn_rate_ratio,
                    buy_share_change=a.buy_share_change,
                    sells_ratio=_sells_ratio(c, a.short, a.long),
                )
            )
            if name == "short":
                ev.price_velocity_change = a.price_velocity_change_pct_per_hour
    h = _history(c, cfg.market.history_lookbacks, pool_age, ev.ignored)
    if h is not None:
        ev.timescales.append(
            Timescale(
                name="history",
                basis=f"{h.volume_window} now vs {h.elapsed_minutes:.0f}m ago",
                volume_ratio=h.volume_ratio,
                txn_ratio=h.txn_ratio,
                buy_share_change=h.buy_share_change,
            )
        )
    h1 = m.window("h1")
    if h1 is not None:
        ev.buy_share = h1.buy_share
        if h1.buyers is not None and h1.sellers is not None and h1.buyers + h1.sellers > 0:
            ev.buyer_share = h1.buyers / (h1.buyers + h1.sellers)
        ev.price_h1 = h1.price_change_pct
    ev.price_h6 = w.price_change_pct if (w := m.window("h6")) else None
    ev.price_h24 = w.price_change_pct if (w := m.window("h24")) else None

    feats = c.features
    if feats is not None:
        by_lookback = {x.lookback_minutes: x for x in feats.history}
        for minutes in cfg.liquidity.growth_lookbacks:
            x = by_lookback.get(minutes)
            if x is not None and x.liquidity_change_pct is not None:
                ev.liquidity_change_pct = x.liquidity_change_pct
                ev.liquidity_change_basis = f"over {x.elapsed_minutes:.0f}m"
                ev.price_change_same_lookback = x.price_change_pct
                ev.market_cap_change_pct = x.market_cap_change_pct
                break
        changes = [
            x.liquidity_change_pct for x in feats.history if x.liquidity_change_pct is not None
        ]
        ev.liquidity_min_change_pct = min(changes) if changes else None
        day_ago = by_lookback.get(1440)
        ev.price_change_1440 = day_ago.price_change_pct if day_ago else None

    ev.move_since_first_seen_pct = _move_since_first_seen(c, history)
    ev.technical = technical_context(c, history, cfg)
    earlier = [
        s for s in history if s.provider == c.market_provider and s.observed_at < c.observed_at
    ]
    ev.stored_snapshots = len(earlier)
    peaks = [
        s.metrics.liquidity_usd
        for s in earlier
        if s.pool_address == c.pool.address and s.metrics.liquidity_usd is not None
    ]
    ev.liquidity_peak_usd = max(peaks) if peaks else None
    age_minutes = pool_age * 60 if pool_age is not None else None
    ev.activity_h1_vs_h24 = _txn_rate_ratio(c, "h1", "h24", age_minutes)
    ev.activity_h6_vs_h24 = _txn_rate_ratio(c, "h6", "h24", age_minutes)
    return ev


def _txn_rate_ratio(
    c: ScoutCandidate, short: Window, long: Window, age_minutes: float | None
) -> float | None:
    """Short-window trade rate over the long window's, or None when the pool is younger
    than the long window (it would be partial) or a count is missing."""
    if age_minutes is None or age_minutes < WINDOW_MINUTES[long]:
        return None
    s, lw = c.metrics.window(short), c.metrics.window(long)
    if s is None or lw is None or s.txns is None or not lw.txns:
        return None
    return (s.txns / WINDOW_MINUTES[short]) / (lw.txns / WINDOW_MINUTES[long])


def _move_since_first_seen(c: ScoutCandidate, history: Sequence[ScoutSnapshot]) -> float | None:
    earliest = next(
        (
            s
            for s in history
            if s.provider == c.market_provider
            and s.observed_at < c.observed_at
            and s.metrics.price_usd
        ),
        None,
    )
    now = c.metrics.price_usd
    if earliest is None or now is None or not earliest.metrics.price_usd:
        return None
    return 100.0 * (now - earliest.metrics.price_usd) / earliest.metrics.price_usd


def technical_context(
    c: ScoutCandidate, history: Sequence[ScoutSnapshot], cfg: GrowthConfig
) -> TechnicalContext | None:
    """Trend, breakout and structure from stored snapshots of the same pool. None when
    there are too few to say anything."""
    t = cfg.technical
    since = c.observed_at - timedelta(hours=t.lookback_hours)
    points: dict[datetime, tuple[float, float | None]] = {}
    for s in history:
        if (
            s.provider == c.market_provider
            and s.pool_address == c.pool.address
            and since <= s.observed_at < c.observed_at
            and s.metrics.price_usd
        ):
            h1 = s.metrics.window("h1")
            points[s.observed_at] = (s.metrics.price_usd, h1.volume_usd if h1 else None)
    if c.metrics.price_usd is None:
        return None
    h1_now = c.metrics.window("h1")
    points[c.observed_at] = (c.metrics.price_usd, h1_now.volume_usd if h1_now else None)
    series = [points[k] for k in sorted(points)]
    if len(series) < t.min_snapshots:
        return None
    prices = [p for p, _ in series]
    first, last = prices[0], prices[-1]
    change = 100.0 * (last - first) / first
    trend: Trend = "flat" if abs(change) < t.flat_trend_pct else "up" if change > 0 else "down"
    breakout = last > max(prices[:-1]) * (1 + t.breakout_margin_pct / 100)
    third = max(1, len(prices) // 3)
    higher_lows = min(prices[-third:]) > min(prices[:third])
    v_first, v_last = series[0][1], series[-1][1]
    volume_confirmed = v_last > v_first if v_first and v_last is not None else None
    span = (sorted(points)[-1] - sorted(points)[0]).total_seconds() / 3600
    return TechnicalContext(
        snapshots=len(series),
        span_hours=round(span, 2),
        trend=trend,
        change_pct=round(change, 2),
        breakout=breakout,
        volume_confirmed=volume_confirmed,
        higher_lows=higher_lows,
    )


# --- Social evidence ------------------------------------------------------------------------

_STATUS: dict[str, SocialStatus] = {
    "UNAVAILABLE": "SOCIAL_UNAVAILABLE",
    "INSUFFICIENT_DATA": "SOCIAL_UNAVAILABLE",
    "QUIET": "SOCIAL_QUIET",
    "EMERGING": "SOCIAL_EMERGING",
    "ACCELERATING": "SOCIAL_ACCELERATING",
    "STRONG": "SOCIAL_STRONG",
    "STABLE": "SOCIAL_STEADY",
    "SATURATED": "SOCIAL_SATURATED",
    "FADING": "SOCIAL_FADING",
}
SOCIAL_RISING: frozenset[SocialStatus] = frozenset(
    {"SOCIAL_EMERGING", "SOCIAL_ACCELERATING", "SOCIAL_STRONG"}
)


@dataclass
class SocialEvidence:
    status: SocialStatus
    state: str | None = None
    reason: str | None = None
    trend: WindowTrend | None = None
    window: SocialWindowStats | None = None
    spam_risk: Level = "unknown"
    organic: Level = "unknown"
    corroborated: bool | None = None
    platforms_active: int | None = None
    exact_share: float | None = None
    strong_share: float | None = None  # EXACT + STRONG share of counted mentions
    unavailable_reason: str | None = None
    age_minutes: float | None = None  # how old the measurement is
    providers: list[tuple[str, str, str | None]] = field(default_factory=list)

    @property
    def rising(self) -> bool:
        return self.status in SOCIAL_RISING

    @property
    def attribution(self) -> SocialAttribution:
        if self.exact_share is None or self.strong_share is None:
            return "none"
        if self.exact_share >= 0.5:
            return "exact"
        return "strong" if self.strong_share >= 0.5 else "probable"


def social_evidence(
    momentum: SocialMomentum | None, at: datetime, cfg: GrowthConfig
) -> SocialEvidence:
    """Maps the social layer's momentum. No momentum, or momentum too old to describe the
    present, is SOCIAL_UNAVAILABLE: missing, never zero."""
    if momentum is None:
        return SocialEvidence(status="SOCIAL_UNAVAILABLE", unavailable_reason="not measured")
    age = (at - momentum.computed_at).total_seconds() / 60
    if age > cfg.social_max_age_minutes:
        return SocialEvidence(
            status="SOCIAL_UNAVAILABLE",
            state=momentum.state,
            unavailable_reason=f"last social measurement is {age:.0f} minutes old",
            providers=[(x.provider, "PROVIDER_UNAVAILABLE", "measurement too old")
                       for x in momentum.sources],
        )  # fmt: skip
    status = _STATUS.get(momentum.state, "SOCIAL_UNAVAILABLE")
    window = next((w for w in momentum.windows if w.window == momentum.window), None)
    ev = SocialEvidence(
        status=status,
        state=momentum.state,
        reason=momentum.reasons[0] if momentum.reasons else None,
        trend=momentum.trend,
        window=window,
        spam_risk=momentum.quality.spam_risk,
        organic=momentum.quality.organic_signal_strength,
        corroborated=momentum.cross_platform.corroborated,
        platforms_active=momentum.cross_platform.platforms_with_activity,
        age_minutes=age,
        providers=[(x.provider, x.status, x.error) for x in momentum.sources],
    )
    if status == "SOCIAL_UNAVAILABLE":
        ev.unavailable_reason = momentum.reasons[0] if momentum.reasons else momentum.state
    if window is not None and window.attributable_posts > 0:
        n = window.attributable_posts
        # mentions = EXACT + STRONG + w x PROBABLE; attributable = EXACT + STRONG + PROBABLE
        w = cfg.social.probable_mention_weight
        probable = min(n, max(0.0, (n - window.mentions) / (1 - w)))
        ev.exact_share = window.exact_mentions / n
        ev.strong_share = (n - probable) / n
    return ev
