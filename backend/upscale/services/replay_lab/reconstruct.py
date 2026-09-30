"""Point-in-time market state, and production Growth Scout evaluated on it.

One recorded Scout snapshot at T becomes one `DexPool` (the provider-normalized pool
model): price, liquidity, market cap, FDV and rolling-window volume / trades / buyers
exactly as recorded, plus the pool's immutable metadata (quote token, DEX, creation time).
That pool then goes through the SAME production code a live scan uses:

    build_candidates (pool selection, Scout risk flags)
      -> compute_features (window acceleration, history comparisons)
      -> GrowthScoutService.rank (stage, stabilization, ScoutMomentumScore, eligibility)

with a throwaway in-memory Scout store holding only what existed at T (snapshots at or
before T, stages stored before T) and `now` = T. Nothing is fetched here.

What can't be reconstructed is reported, never invented: other pools of the token (their
state at T wasn't recorded: pool count 1, primary treated as clear, oldest pool = this
pool), discovery listings (none), and on-chain safety (holders / authorities at T are
unknown: production then flags INSUFFICIENT_SAFETY_DATA, as it does live without it).
"""

from datetime import datetime, timedelta

from upscale.services.outcomes.models import ObservedMarket
from upscale.services.replay_lab.archive import PoolMetadata, ScoutArchive
from upscale.services.replay_lab.clock import HistoricalClock, LookaheadError
from upscale.services.replay_lab.models import ScoutReplay
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.features import compute_features
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.growth.models import GrowthCandidate
from upscale.services.scout.growth.service import GrowthScoutService
from upscale.services.scout.models import ScoutSnapshot
from upscale.services.scout.normalize import Listing, build_candidates
from upscale.services.scout.social.models import SocialMomentum
from upscale.services.scout.store import ScoutSnapshotStore
from upscale.services.solana_dex import DexPool, TokenRef, WindowStats

REPLAY_LISTING = "historical_replay"
# History loaded into the point-in-time store: covers every lookback the live code uses.
HISTORY_HOURS = 48.0


def pool_at(
    snapshot: ScoutSnapshot,
    meta: PoolMetadata,
    chain: str,
    token: str,
    symbol: str | None,
    name: str | None,
) -> DexPool:
    m = snapshot.metrics
    return DexPool(
        chain=chain,
        dex=meta.dex or snapshot.dex,
        pair_address=snapshot.pool_address,
        url=meta.url,
        base=TokenRef(address=token, symbol=symbol, name=name),
        quote=TokenRef(address=meta.quote_address, symbol=meta.quote_symbol),
        price_usd=m.price_usd,
        liquidity_usd=m.liquidity_usd,
        market_cap_usd=m.market_cap_usd,
        fdv_usd=m.fdv_usd,
        pair_created_at=meta.created_at,
        windows=[
            WindowStats(
                window=w.window,
                buys=w.buys,
                sells=w.sells,
                volume_usd=w.volume_usd,
                price_change_pct=w.price_change_pct,
                buyers=w.buyers,
                sellers=w.sellers,
            )
            for w in m.windows
        ],
    )


def observed_market(snapshot: ScoutSnapshot) -> ObservedMarket:
    """The market at T in the outcome module's reference form (for horizon changes)."""
    m = snapshot.metrics
    h1, h24 = m.window("h1"), m.window("h24")
    return ObservedMarket(
        price_usd=m.price_usd,
        market_cap_usd=m.market_cap_usd,
        fdv_usd=m.fdv_usd,
        liquidity_usd=m.liquidity_usd,
        volume_h1_usd=h1.volume_usd if h1 else None,
        volume_h24_usd=h24.volume_usd if h24 else None,
        txns_h1=h1.txns if h1 else None,
        txns_h24=h24.txns if h24 else None,
        buy_share_h1=h1.buy_share if h1 else None,
    )


async def evaluate_scout(
    clock: HistoricalClock,
    archive: ScoutArchive,
    pool: DexPool,
    snapshot: ScoutSnapshot,
    first_seen_at: datetime | None,
    social: SocialMomentum | None,
    scout_config: ScoutConfig,
    growth_config: GrowthConfig,
) -> tuple[ScoutReplay, GrowthCandidate | None, list[str]]:
    """Production Growth Scout for this pool at T. Returns the replay record, the full
    candidate, and notes on what was unavailable."""
    clock.check_decision_phase("Scout evaluation")
    t = clock.decision_at
    notes: list[str] = []
    cid = snapshot.canonical_id
    listing = Listing(provider=snapshot.provider, kind="lookup", name=REPLAY_LISTING, fetched_at=t)
    built = build_candidates([pool], listing, scout_config, restrict_to={cid})
    if not built.candidates:
        reasons = sorted({r for x in built.rejected for r in x.reasons})
        return (
            ScoutReplay(
                status="NOT_RECONSTRUCTIBLE",
                reason="production pool selection rejected the pool at T: " + "; ".join(reasons),
            ),
            None,
            notes,
        )
    candidate = built.candidates[0]
    store = ScoutSnapshotStore(":memory:")
    try:
        await store.record_seen(candidate)
        history = archive.snapshots(cid, t - timedelta(hours=HISTORY_HOURS), t)
        for s in history:
            clock.check_time(s.observed_at, "Scout history snapshot")
            await store.save_snapshot(s, 0.0)
        stages = archive.stages_before(cid, t - timedelta(hours=HISTORY_HOURS), t)
        for at, stage, rank, score in stages:
            if at >= t:
                raise LookaheadError("a Scout stage from the decision time or later")
            ranks = {cid: (rank, score)} if score is not None else None
            await store.record_stages(at, {cid: stage}, ranks)
        if first_seen_at is not None and first_seen_at > t:
            raise LookaheadError("first-seen time after the decision time")
        features = await compute_features(candidate, store, scout_config.features)
        candidate = candidate.model_copy(
            update={"first_seen_at": first_seen_at, "features": features}
        )
        if social is not None:
            clock.check_time(social.computed_at, "social momentum")
        growth = GrowthScoutService(
            store, growth_config, social_store=None, safety=None, now=lambda: t
        )
        result = await growth.rank(
            [candidate], social={cid: social} if social is not None else None, limit=1
        )
    finally:
        store.close()
    g = next(iter([*result.candidates, *result.unranked]), None)
    if g is None:
        return (
            ScoutReplay(status="NOT_RECONSTRUCTIBLE", reason="Growth Scout returned no evaluation"),
            None,
            notes,
        )
    if first_seen_at is None:
        notes.append("first-seen time unknown before T")
    sm = g.scout_momentum
    return (
        ScoutReplay(
            status="RECONSTRUCTED",
            stage=g.stage,
            unconfirmed_stage=g.unconfirmed_stage,
            stage_reasons=list(g.stage_reasons),
            score=sm.score,
            base=sm.base,
            stage_adjustment=sm.stage_adjustment,
            risk_penalty=sm.risk_penalty,
            families={f.family: f.contribution for f in sm.families},
            eligible=g.eligible,
            ineligible_reasons=list(g.ineligible_reasons),
            risk_flags=[
                {"code": f.code, "severity": f.severity, "penalty": f.penalty, "detail": f.detail}
                for f in g.risk_flags
            ],
            reasons=list(g.reasons_surfaced),
            social_status=g.momentum.social_status,
            safety_status=g.quality.safety_status,
            candidate=g.model_dump(mode="json"),
        ),
        g,
        notes,
    )
