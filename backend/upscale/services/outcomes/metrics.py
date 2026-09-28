"""Pure outcome measurements: price paths, excursions, trigger touches, market changes and
robust statistics. No I/O; every function is deterministic.

Price path methodology (`path_from_candles`):

* The window is [start, end]: start is when the observed price was measured, end is
  start + horizon. Only candles lying entirely inside it count (open >= start and open +
  interval <= end), so no price from before the observation or after the horizon leaks
  in. Candles only exist for intervals with trades: a gap is a gap, never filled.
* The path starts at the reference price, so MFE = max(0, highest / reference - 1) and
  MAE = min(0, lowest / reference - 1). Time to MFE / MAE is measured to the open of the
  candle that reached the extreme (resolution: one candle).
* Max drawdown compares each candle's low with the highest price reached *before* that
  candle (reference included), because the order of high and low inside one candle is
  unknown.
* The ending price is the last in-window candle's close.
"""

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from upscale.services.market_data import Candle
from upscale.services.outcomes.models import (
    MarketAtHorizon,
    MarketSource,
    MarketStatus,
    ObservedMarket,
    PricePath,
    TriggerOutcome,
)
from upscale.services.scout.models import ScoutMarketMetrics


def pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1) * 100


def in_window(
    candles: Sequence[Candle], interval: timedelta, start: datetime, end: datetime
) -> list[Candle]:
    return sorted(
        (c for c in candles if c.timestamp >= start and c.timestamp + interval <= end),
        key=lambda c: c.timestamp,
    )


def path_from_candles(
    candles: Sequence[Candle],
    interval: timedelta,
    reference: float,
    start: datetime,
    end: datetime,
    *,
    provider: str,
    timeframe: str,
    price_drop_pct: float,
) -> PricePath:
    inside = in_window(candles, interval, start, end)
    path = PricePath(
        source="candles",
        provider=provider,
        timeframe=timeframe,
        points=len(inside),
        window_start=start,
        window_end=end,
        reference_price=reference,
    )
    if not inside:
        path.notes.append("no trades in the window (no candles): price path not measurable")
        return path
    high = max(inside, key=lambda c: (c.high, -c.timestamp.timestamp()))  # earliest max
    low = min(inside, key=lambda c: (c.low, c.timestamp.timestamp()))  # earliest min
    last = inside[-1]
    peak, drawdown = reference, 0.0
    for c in inside:
        drawdown = min(drawdown, (c.low / peak - 1) * 100)
        peak = max(peak, c.high)
    path.end_price = last.close
    path.end_price_at = last.timestamp + interval
    path.return_pct = pct(last.close, reference)
    path.highest_price, path.highest_at = high.high, high.timestamp
    path.lowest_price, path.lowest_at = low.low, low.timestamp
    path.mfe_pct = max(0.0, (high.high / reference - 1) * 100)
    path.mae_pct = min(0.0, (low.low / reference - 1) * 100)
    path.time_to_mfe_minutes = _minutes(high.timestamp - start) if path.mfe_pct > 0 else None
    path.time_to_mae_minutes = _minutes(low.timestamp - start) if path.mae_pct < 0 else None
    path.max_drawdown_pct = drawdown
    path.price_collapsed = low.low <= reference * (1 - price_drop_pct / 100)
    covered = len(inside) * interval / (end - start)
    if covered < 0.5:
        path.notes.append(
            f"candles cover {covered:.0%} of the window (intervals without trades have no candle)"
        )
    return path


def path_from_points(
    points: Sequence[tuple[datetime, float]],
    reference: float,
    start: datetime,
    end: datetime,
    end_tolerance: timedelta,
    *,
    provider: str | None,
    price_drop_pct: float,
) -> PricePath:
    """From stored Scout snapshots of the exact pool: the ending price is the snapshot
    nearest the horizon end (within `end_tolerance`); extremes are only the observed
    points inside the window, so MFE / MAE are lower bounds."""
    inside = sorted((t, p) for t, p in points if start < t <= end)
    near_end = [
        (t, p) for t, p in points if abs((t - end).total_seconds()) <= end_tolerance.total_seconds()
    ]
    path = PricePath(
        source="snapshots",
        provider=provider,
        points=len(inside),
        window_start=start,
        window_end=end,
        reference_price=reference,
        notes=["from stored Scout snapshots only: extremes between snapshots are unknown, so "
               "MFE / MAE are lower bounds"],
    )  # fmt: skip
    if near_end:
        t, p = min(near_end, key=lambda x: (abs((x[0] - end).total_seconds()), x[0]))
        path.end_price, path.end_price_at = p, t
        path.return_pct = pct(p, reference)
    if inside:
        high = max(inside, key=lambda x: (x[1], -x[0].timestamp()))
        low = min(inside, key=lambda x: (x[1], x[0].timestamp()))
        path.highest_price, path.highest_at = high[1], high[0]
        path.lowest_price, path.lowest_at = low[1], low[0]
        path.mfe_pct = max(0.0, (high[1] / reference - 1) * 100)
        path.mae_pct = min(0.0, (low[1] / reference - 1) * 100)
        path.time_to_mfe_minutes = _minutes(high[0] - start) if path.mfe_pct > 0 else None
        path.time_to_mae_minutes = _minutes(low[0] - start) if path.mae_pct < 0 else None
        path.price_collapsed = low[1] <= reference * (1 - price_drop_pct / 100)
    return path


def trigger_outcome(
    candles: Sequence[Candle],
    level: float,
    direction: str,
    reference: float | None,
) -> TriggerOutcome:
    """First in-window candle whose high (direction "above") or low ("below") reaches
    `level`. `candles` must already be the in-window candles, oldest first."""
    above = direction == "above"
    first = next((c for c in candles if (c.high >= level if above else c.low <= level)), None)
    return TriggerOutcome(
        level=level,
        direction="above" if above else "below",
        reached=first is not None if candles else None,
        first_reached_at=first.timestamp if first else None,
        already_beyond_at_start=(
            (reference >= level if above else reference <= level) if reference else None
        ),
    )


def first_event(triggers: dict[str, TriggerOutcome]) -> str | None:
    """Which level was reached first: its name, "none", or "same_candle:a+b" when two
    were first reached within the same candle (their order is unknown)."""
    if not triggers or all(t.reached is None for t in triggers.values()):
        return None
    reached = [(t.first_reached_at, name) for name, t in triggers.items() if t.first_reached_at]
    if not reached:
        return "none"
    earliest = min(at for at, _ in reached)
    names = sorted(name for at, name in reached if at == earliest)
    return names[0] if len(names) == 1 else "same_candle:" + "+".join(names)


def market_at_horizon(
    metrics: ScoutMarketMetrics | None,
    reference: ObservedMarket | None,
    *,
    source: MarketSource,
    provider: str,
    observed_at: datetime,
    pool_found: bool,
    liquidity_drop_pct: float,
    liquidity_floor_usd: float,
    reference_liquidity: float | None = None,
) -> MarketAtHorizon:
    out = MarketAtHorizon(
        source=source, provider=provider, observed_at=observed_at, pool_found=pool_found
    )
    if metrics is None or not pool_found:
        return out
    ref = reference
    liq0 = ref.liquidity_usd if ref is not None else reference_liquidity
    h1, h24 = metrics.window("h1"), metrics.window("h24")
    out.price_usd = metrics.price_usd
    out.liquidity_usd = metrics.liquidity_usd
    out.liquidity_change_pct = pct(metrics.liquidity_usd, liq0)
    if ref is not None and ref.market_cap_usd is not None:
        out.market_cap_usd = metrics.market_cap_usd
        out.market_cap_change_pct = pct(metrics.market_cap_usd, ref.market_cap_usd)
    out.fdv_usd = metrics.fdv_usd
    out.fdv_change_pct = pct(metrics.fdv_usd, ref.fdv_usd if ref else None)
    out.volume_h1_usd = h1.volume_usd if h1 else None
    out.volume_h1_change_pct = pct(out.volume_h1_usd, ref.volume_h1_usd if ref else None)
    out.volume_h24_usd = h24.volume_usd if h24 else None
    out.volume_h24_change_pct = pct(out.volume_h24_usd, ref.volume_h24_usd if ref else None)
    out.txns_h1 = h1.txns if h1 else None
    out.txns_h1_change_pct = pct(
        float(out.txns_h1) if out.txns_h1 is not None else None,
        float(ref.txns_h1) if ref is not None and ref.txns_h1 is not None else None,
    )
    out.buy_share_h1 = h1.buy_share if h1 else None
    if out.buy_share_h1 is not None and ref is not None and ref.buy_share_h1 is not None:
        out.buy_share_change = out.buy_share_h1 - ref.buy_share_h1
    if out.liquidity_usd is not None:
        dropped = (
            out.liquidity_change_pct is not None and out.liquidity_change_pct <= -liquidity_drop_pct
        )
        out.liquidity_collapsed = dropped or out.liquidity_usd < liquidity_floor_usd
    return out


def market_status(
    market: MarketAtHorizon | None, price: PricePath | None, provider_failed: bool
) -> MarketStatus:
    """A factual terminal / activity status (never "rug")."""
    if market is not None and not market.pool_found:
        return "POOL_GONE"
    if market is not None and market.liquidity_collapsed:
        return "LIQUIDITY_COLLAPSE"
    if market is not None and market.price_usd is None:
        return "MARKET_UNAVAILABLE"  # listed, but without a usable price
    inactive = (market is not None and market.txns_h1 == 0) or (
        price is not None and price.source == "candles" and price.points == 0
    )
    if inactive:
        return "TOKEN_INACTIVE"
    if market is not None or (price is not None and price.points > 0):
        return "ACTIVE"
    return "PROVIDER_UNAVAILABLE" if provider_failed else "UNKNOWN"


# --- Robust statistics ---------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (the common "type 7" definition)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def median(values: Sequence[float]) -> float | None:
    return percentile(values, 50)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def distribution(values: Sequence[float], edges: Sequence[float]) -> dict[str, int]:
    """Counts per bucket: "<e0", "e0..e1", ..., ">=en" (percent edges)."""
    edges = sorted(edges)
    labels = [f"<{edges[0]:g}"] + [f"{a:g}..{b:g}" for a, b in zip(edges, edges[1:], strict=False)]
    labels.append(f">={edges[-1]:g}")
    out = dict.fromkeys(labels, 0)
    for v in values:
        i = sum(v >= e for e in edges)
        out[labels[i]] += 1
    return out


def _minutes(delta: timedelta) -> float:
    return round(delta.total_seconds() / 60, 2)
