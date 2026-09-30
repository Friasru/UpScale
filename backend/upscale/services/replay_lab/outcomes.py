"""After the decision is stored: what the market did, with the live outcome definitions.

Runs only in the REVEALED phase (the clock holds the receipt of the committed decision)
and reads the decision back from the replay store, never from the objects that produced
it, so nothing measured here can reach or change the decision.

Per horizon (the live horizons: 5m / 15m / 1h on 1m candles, 4h on 5m, 24h on 15m):

* price path: `path_from_candles` from the reference price at T over [T, T + h]: return,
  MFE, MAE, max drawdown, time to MFE / MAE, collapse;
* decision levels: the live collector's trigger / invalidation touches, first event and
  end position (touches are price evidence, not fills);
* market at the horizon end: the archive's snapshot of the same pool nearest the end
  (within the live tolerance), via `market_at_horizon` (liquidity change, volume...);
* future stage: the live system's recorded Growth Scout stage near the horizon end;
* `market_status` and COMPLETE / PARTIAL / UNAVAILABLE as the live collector decides them.
"""

from datetime import timedelta

from upscale.services.market_data import Timeframe
from upscale.services.outcomes.collector import _decision_triggers, _price_position
from upscale.services.outcomes.config import CollapseConfig, CollectorConfig, HorizonSpec
from upscale.services.outcomes.metrics import (
    first_event,
    in_window,
    market_at_horizon,
    market_status,
    path_from_candles,
)
from upscale.services.outcomes.models import HorizonStatus, MarketAtHorizon, PricePath
from upscale.services.replay_lab.archive import ScoutArchive
from upscale.services.replay_lab.candles import PROVIDER, PointInTimeCandles, interval_of
from upscale.services.replay_lab.clock import HistoricalClock, LookaheadError
from upscale.services.replay_lab.models import (
    ReplayDecisionRecord,
    ReplayFidelity,
    ReplayHorizonOutcome,
)
from upscale.services.replay_lab.reconstruct import observed_market

DEFAULT_COLLAPSE = CollapseConfig()
DEFAULT_COLLECTOR = CollectorConfig()
FIDELITY_WINDOW = timedelta(minutes=10)  # a live ranking run using the snapshot at T


def measure(
    clock: HistoricalClock,
    record: ReplayDecisionRecord,
    candles: PointInTimeCandles,
    archive: ScoutArchive | None,
    horizons: tuple[HorizonSpec, ...],
    collapse: CollapseConfig = DEFAULT_COLLAPSE,
    collector: CollectorConfig = DEFAULT_COLLECTOR,
) -> list[ReplayHorizonOutcome]:
    if clock.phase != "REVEALED" or clock.receipt is None:
        raise LookaheadError("outcomes can only be measured after the decision is stored")
    if clock.receipt.decision_at != record.decision_at:
        raise LookaheadError("the receipt belongs to another decision")
    t = record.decision_at
    reference = record.reference_price
    ref_snapshot = None
    if archive is not None and record.evidence == "RECORDED":
        ref_snapshot = archive.snapshot_at(record.asset_id, record.pool_address, t)
    reference_market = observed_market(ref_snapshot) if ref_snapshot is not None else None
    out: list[ReplayHorizonOutcome] = []
    for spec in horizons:
        end = t + timedelta(minutes=spec.minutes)
        clock.check_request(end, f"{spec.label} outcome window")
        missing: list[str] = []
        price: PricePath | None = None
        triggers = {}
        first = position = None
        timeframe: Timeframe = spec.candles
        if reference is None:
            missing.append("no reference price at T: price outcome not measurable")
        elif not candles.loaded(timeframe):
            missing.append(f"{timeframe} candles unavailable for the window")
        else:
            window = candles.window(timeframe, t, end)
            interval = interval_of(timeframe)
            price = path_from_candles(
                window, interval, reference, t, end, provider=PROVIDER, timeframe=timeframe,
                price_drop_pct=collapse.price_drop_pct,
            )  # fmt: skip
            inside = in_window(window, interval, t, end)
            if price.points == 0:
                missing.append("no trades in the window: price path not measurable")
            if record.decision is not None and inside:
                triggers = _decision_triggers(record.decision, inside, reference)
                first = first_event(triggers)
                position = _price_position(record.decision, price.end_price)
        tolerance = timedelta(
            seconds=max(
                collector.min_market_tolerance_seconds,
                collector.market_tolerance * spec.minutes * 60,
            )
        )
        market: MarketAtHorizon | None = None
        future_stage = future_at = future_score = None
        if archive is not None and record.evidence == "RECORDED":
            later = archive.snapshots_between(
                record.asset_id, record.pool_address, t, end + tolerance
            )
            near = [
                s
                for s in later
                if abs((s.observed_at - end).total_seconds()) <= tolerance.total_seconds()
            ]
            if near:
                s = min(
                    near, key=lambda x: (abs((x.observed_at - end).total_seconds()), x.observed_at)
                )
                clock.check_time(s.observed_at, "horizon-end snapshot")
                market = market_at_horizon(
                    s.metrics, reference_market, source="scout_snapshot", provider=s.provider,
                    observed_at=s.observed_at, pool_found=True,
                    liquidity_drop_pct=collapse.liquidity_drop_pct,
                    liquidity_floor_usd=collapse.liquidity_floor_usd,
                )  # fmt: skip
            else:
                missing.append(
                    "horizon-end market state not observed (no recorded snapshot of this pool "
                    f"within {tolerance.total_seconds() / 60:.0f} min of the horizon end)"
                )
            stage = archive.stage_near(record.asset_id, end, tolerance.total_seconds(), t)
            if stage is not None:
                clock.check_time(stage[0], "future stage")
                future_at, future_stage, _, future_score = stage
            else:
                missing.append(
                    "no live Growth Scout run near the horizon end: future stage unknown"
                )
        else:
            missing.append(
                "horizon-end market state (liquidity, volume, market cap) has no historical "
                "source for candle-only samples"
            )
        priced = price is not None and price.points > 0
        status: HorizonStatus
        if priced and market is not None:
            status = "COMPLETE"
        elif priced or market is not None:
            status = "PARTIAL"
        else:
            status = "UNAVAILABLE"
        out.append(
            ReplayHorizonOutcome(
                horizon=spec.label,
                horizon_minutes=spec.minutes,
                window_start=t,
                window_end=end,
                status=status,
                market_status=market_status(market, price, provider_failed=False),
                price=price,
                market=market,
                future_stage=future_stage,
                future_stage_at=future_at,
                future_score=future_score,
                triggers=triggers,
                first_trigger_event=first,
                price_position=position,
                missing=missing,
            )
        )
    return out


def fidelity(
    clock: HistoricalClock, record: ReplayDecisionRecord, archive: ScoutArchive | None
) -> ReplayFidelity:
    """The live system's own stage / score for the ranking run that used the snapshot at T
    (read only after the decision is stored)."""
    if clock.phase != "REVEALED":
        raise LookaheadError("fidelity is checked after the decision is stored")
    if archive is None or record.evidence != "RECORDED":
        return ReplayFidelity(notes=["no live ranking to compare (candle-only sample)"])
    t = record.decision_at
    live = archive.stage_near(
        record.asset_id, t + FIDELITY_WINDOW / 2, FIDELITY_WINDOW.total_seconds() / 2, t
    )
    if live is None:
        return ReplayFidelity(notes=["no live Growth Scout run within 10 minutes after T"])
    at, stage, rank, score = live
    replay_stage, replay_score = record.scout.stage, record.scout.score
    notes = []
    if record.scout.status != "RECONSTRUCTED":
        notes.append("replay could not reconstruct Scout for this sample")
    notes.append(
        "live runs may differ legitimately: live social / on-chain safety evidence and other "
        "pools of the token are not available to replay"
    )
    return ReplayFidelity(
        live_stage=stage,
        live_score=score,
        live_rank=rank,
        live_run_at=at,
        stage_matches=(stage == replay_stage) if replay_stage is not None else None,
        score_delta=round(replay_score - score, 2)
        if replay_score is not None and score is not None
        else None,
        notes=notes,
    )
