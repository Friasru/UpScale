"""Growth measurements for a candidate. Measurements only: nothing here ranks or decides.

Two independent sources of acceleration, both from real observations:

* **Within one observation** (`window_acceleration`): a provider's rolling windows are
  nested (the 5m window is inside the 1h window), so the per-minute rate of a short window
  over the per-minute rate of a longer one says whether activity is speeding up (> 1) or
  slowing down (< 1) right now. Works on the very first sighting.
* **Against stored history** (`compare_with`): now vs the stored snapshot closest to each
  lookback (5m, 15m, 30m, 1h, ...). The actual gap is reported; a lookback with no
  snapshot close enough is listed as missing, never interpolated. Snapshots from another
  provider are not compared (their figures aren't measured the same way), and pool-level
  figures are only compared when the selected pool is the same one.

A ratio or change is None whenever either side is missing or its base is zero.
"""

import asyncio
from datetime import datetime, timedelta

from upscale.services.scout.config import ScoutFeatureConfig
from upscale.services.scout.models import (
    WINDOW_MINUTES,
    HistoryComparison,
    ScoutCandidate,
    ScoutGrowthFeatures,
    ScoutMarketMetrics,
    ScoutSnapshot,
    WindowAcceleration,
)
from upscale.services.scout.store import ScoutSnapshotStore


def pct_change(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then <= 0:
        return None
    return 100.0 * (now - then) / then


def ratio(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then <= 0:
        return None
    return now / then


def window_acceleration(
    metrics: ScoutMarketMetrics, cfg: ScoutFeatureConfig
) -> list[WindowAcceleration]:
    out = []
    for short_name, long_name in cfg.window_pairs:
        short, long = metrics.window(short_name), metrics.window(long_name)
        if short is None or long is None:
            continue
        ms, ml = WINDOW_MINUTES[short_name], WINDOW_MINUTES[long_name]
        s_txns, l_txns = short.txns, long.txns
        s_share, l_share = short.buy_share, long.buy_share
        out.append(
            WindowAcceleration(
                short=short_name,
                long=long_name,
                volume_rate_ratio=_rate_ratio(short.volume_usd, ms, long.volume_usd, ml),
                txn_rate_ratio=_rate_ratio(s_txns, ms, l_txns, ml),
                buy_share_change=(
                    s_share - l_share if s_share is not None and l_share is not None else None
                ),
                price_velocity_change_pct_per_hour=(
                    short.price_change_pct * 60 / ms - long.price_change_pct * 60 / ml
                    if short.price_change_pct is not None and long.price_change_pct is not None
                    else None
                ),
            )
        )
    return out


def _rate_ratio(
    short: float | None, short_minutes: int, long: float | None, long_minutes: int
) -> float | None:
    if short is None or long is None or long <= 0:
        return None
    return (short / short_minutes) / (long / long_minutes)


def compare_with(
    candidate: ScoutCandidate,
    then: ScoutSnapshot,
    lookback_minutes: int,
    cfg: ScoutFeatureConfig,
) -> HistoryComparison:
    now, past = candidate.metrics, then.metrics
    same_pool = then.pool_address == candidate.pool.address
    comparison = HistoryComparison(
        lookback_minutes=lookback_minutes,
        compared_at=then.observed_at,
        elapsed_minutes=(candidate.observed_at - then.observed_at).total_seconds() / 60,
        same_pool=same_pool,
        price_change_pct=pct_change(now.price_usd, past.price_usd),
        market_cap_change_pct=pct_change(now.market_cap_usd, past.market_cap_usd),
        fdv_change_pct=pct_change(now.fdv_usd, past.fdv_usd),
    )
    if not same_pool:
        comparison.notes.append(
            f"the selected pool changed from {then.pool_address} to {candidate.pool.address}; "
            "pool liquidity and activity are not compared"
        )
        return comparison
    comparison.liquidity_change_pct = pct_change(now.liquidity_usd, past.liquidity_usd)
    for name in cfg.history_windows:
        w_now, w_then = now.window(name), past.window(name)
        if w_now is None or w_then is None:
            continue
        if w_now.volume_usd is None and w_now.txns is None:
            continue
        comparison.volume_window = name
        comparison.volume_ratio = ratio(w_now.volume_usd, w_then.volume_usd)
        comparison.txn_ratio = ratio(
            float(w_now.txns) if w_now.txns is not None else None,
            float(w_then.txns) if w_then.txns is not None else None,
        )
        s_now, s_then = w_now.buy_share, w_then.buy_share
        if s_now is not None and s_then is not None:
            comparison.buy_share_change = s_now - s_then
        break
    return comparison


async def compute_features(
    candidate: ScoutCandidate,
    store: ScoutSnapshotStore,
    cfg: ScoutFeatureConfig,
    computed_at: datetime | None = None,
) -> ScoutGrowthFeatures:
    async def earlier(minutes: int) -> ScoutSnapshot | None:
        tolerance = max(cfg.min_tolerance_seconds, minutes * 60 * cfg.lookback_tolerance)
        return await store.nearest_snapshot(
            candidate.canonical_id,
            candidate.observed_at - timedelta(minutes=minutes),
            timedelta(seconds=tolerance),
            before=candidate.observed_at,
            provider=candidate.market_provider,
        )

    found = await asyncio.gather(*(earlier(m) for m in cfg.lookback_minutes))
    history, missing = [], []
    for minutes, snapshot in zip(cfg.lookback_minutes, found, strict=True):
        if snapshot is None:
            missing.append(minutes)
        else:
            history.append(compare_with(candidate, snapshot, minutes, cfg))
    day = candidate.metrics.window("h24")
    return ScoutGrowthFeatures(
        computed_at=computed_at or candidate.observed_at,
        window_acceleration=window_acceleration(candidate.metrics, cfg),
        history=history,
        missing_lookbacks=missing,
        volume_to_liquidity_h24=ratio(
            day.volume_usd if day else None, candidate.metrics.liquidity_usd
        ),
    )
