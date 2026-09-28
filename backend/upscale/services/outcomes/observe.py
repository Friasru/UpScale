"""Turning Scout rankings and Analyze decisions into immutable outcome observations.

Scout observation policy (thresholds in `ObservationPolicy`), per ranked candidate, in
this order (deterministic):

1. Evidence no newer than the token's latest anchor (e.g. an identical rerun, or a stale
   carried observation already anchored): no new anchor.
2. No earlier anchor: FIRST_RANKED.
3. Not ranked in at least one ranking run since it was last ranked, and gone for at
   least `reentry_gap_minutes`: REENTRY.
4. Stage differs from the latest anchor's: STAGE_CHANGE.
5. Scout Momentum moved at least `min_score_change` points: SCORE_CHANGE.
6. The latest anchor is at least `reanchor_after_minutes` old: TIME_ELAPSED.
7. Otherwise: no new anchor (Scout's own snapshot history keeps every poll).

Decisions: every Analyze whose Opportunity agent produced a real decision on a
measurable market (an exact DEX pool of the exact token, or the exact exchange market
the candles came from) is stored as analyzed, once.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import ValidationError

from upscale.schemas import AgentResult, ChatRequest, ChatResponse
from upscale.services.opportunity import Invalidation, OpportunityAssessment, Trigger
from upscale.services.outcomes.config import OutcomeConfig
from upscale.services.outcomes.models import (
    AnchorReason,
    DecisionInvalidation,
    DecisionLevel,
    DecisionObservation,
    DecisionRisk,
    DiscoveryStatus,
    ObservedFlag,
    ObservedMarket,
    ObservedSafety,
    ObservedSocial,
    PriceRef,
    ScoreComponents,
    ScoutObservation,
)
from upscale.services.outcomes.store import Anchor, OutcomeStore, RankedState
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.solana_dex import SolanaDexSnapshot

logger = logging.getLogger("upscale.outcomes")
X_PROVIDER = "X"
FARCASTER_PROVIDER = "Farcaster (Neynar)"


# --- Scout ----------------------------------------------------------------------------------


def anchor_reason(
    g: GrowthCandidate,
    anchor: Anchor | None,
    ranked: RankedState | None,
    runs_since_ranked: int,
    now: datetime,
    config: OutcomeConfig,
) -> AnchorReason | None:
    p = config.observation
    if anchor is not None and g.observed_at <= anchor.observed_at:
        return None
    if anchor is None:
        return "FIRST_RANKED"
    if (
        ranked is not None
        and runs_since_ranked > 0
        and now - ranked.last_ranked_at >= timedelta(minutes=p.reentry_gap_minutes)
    ):
        return "REENTRY"
    if p.stage_change and g.stage != anchor.stage:
        return "STAGE_CHANGE"
    if abs(g.score - anchor.score) >= p.min_score_change:
        return "SCORE_CHANGE"
    if now - anchor.anchored_at >= timedelta(minutes=p.reanchor_after_minutes):
        return "TIME_ELAPSED"
    return None


def discovery_status(g: GrowthCandidate) -> DiscoveryStatus:
    if g.data_status == "STALE_CARRIED":
        return "STALE_CARRIED"
    first = g.market.first_seen_at
    if first is not None and abs((g.observed_at - first).total_seconds()) < 1:
        return "NEW"
    listed = any(k != "lookup" for k in g.source_kinds)
    return "DISCOVERED" if listed or not g.source_kinds else "REFRESHED"


def scout_observation(
    g: GrowthCandidate,
    result: GrowthScoutResult,
    reason: AnchorReason,
    previous: Anchor | None,
) -> ScoutObservation:
    m, mo, q, sm = g.market, g.momentum, g.quality, g.scout_momentum
    families = {f.family: f.contribution for f in sm.families}
    h1, h24 = (next((w for w in m.windows if w.window == n), None) for n in ("h1", "h24"))
    providers = {p.provider: p.status for p in mo.social_providers}
    rank = g.rank or 0
    age = g.snapshot_age_minutes
    if age is None:
        age = max(0.0, (result.computed_at - g.observed_at).total_seconds() / 60)
    return ScoutObservation(
        canonical_id=g.canonical_id,
        chain=g.chain,
        address=g.address,
        symbol=g.symbol,
        name=g.name,
        pool_address=m.selected_pool.address,
        pool_dex=m.selected_pool.dex,
        quote_symbol=m.selected_pool.quote_symbol,
        market_provider=m.market_provider,
        observed_at=g.observed_at,
        anchored_at=result.computed_at,
        run_id=result.computed_at.isoformat(),
        anchor_reason=reason,
        previous_observation_id=previous.id if previous else None,
        rank=rank,
        ranking_mode=result.mode,
        in_top10=rank <= 10,
        in_top20=rank <= 20,
        discovery_status=discovery_status(g),
        snapshot_age_minutes=round(age, 2),
        stage=g.stage,
        unconfirmed_stage=g.unconfirmed_stage,
        stage_reasons=list(g.stage_reasons),
        score=sm.score,
        components=ScoreComponents(
            market_activity=families.get("market_activity"),
            liquidity_quality=families.get("liquidity_quality"),
            social_momentum=families.get("social_momentum"),
            earliness=families.get("earliness"),
            cross_confirmation=families.get("cross_confirmation"),
            base=sm.base,
            stage_adjustment=sm.stage_adjustment,
            risk_penalty=sm.risk_penalty,
        ),
        market=ObservedMarket(
            price_usd=m.price_usd,
            market_cap_usd=m.market_cap_usd,
            market_cap_note=m.market_cap_note,
            fdv_usd=m.fdv_usd,
            liquidity_usd=m.liquidity_usd,
            volume_h1_usd=h1.volume_usd if h1 else None,
            volume_h24_usd=h24.volume_usd if h24 else None,
            txns_h1=h1.txns if h1 else None,
            txns_h24=h24.txns if h24 else None,
            buy_share_h1=mo.buy_share,
            buyer_share_h1=mo.buyer_share,
            volume_acceleration=mo.volume_acceleration,
            txn_acceleration=mo.txn_acceleration,
            buy_pressure_change=mo.buy_pressure_change,
            price_change_h1_pct=mo.price_change_h1_pct,
            price_change_h24_pct=mo.price_change_h24_pct,
            pool_age_hours=m.oldest_pool_age_hours or m.pool_age_hours,
            tracked_hours=m.tracked_hours,
        ),
        social=ObservedSocial(
            status=mo.social_status,
            state=mo.social_state,
            x_state=providers.get(X_PROVIDER),
            farcaster_state=providers.get(FARCASTER_PROVIDER),
            spam_risk=q.spam_risk,
            attribution=q.social_attribution,
            cross_platform=mo.cross_platform_corroborated,
        ),
        safety=ObservedSafety(
            status=q.safety_status,
            holder_top1_pct=q.holder_top1_pct,
            holder_top10_pct=q.holder_top10_pct,
            holder_data_lower_bound=q.holder_data_lower_bound,
            mint_authority_active=q.mint_authority_active,
            freeze_authority_active=q.freeze_authority_active,
            liquidity_quality=q.liquidity_quality,
            market_status=q.market_status,
        ),
        risk_flags=[
            ObservedFlag(code=f.code, severity=f.severity, detail=f.detail) for f in g.risk_flags
        ],
        reasons=list(g.reasons_surfaced),
    )


async def record_scout_run(
    store: OutcomeStore, result: GrowthScoutResult, config: OutcomeConfig
) -> list[ScoutObservation]:
    """Anchor the ranked candidates the policy selects; returns the new anchors. The
    ranking itself is only read, never changed."""
    now = result.computed_at
    ranked = [g for g in result.candidates if g.rank is not None]
    ids = [g.canonical_id for g in ranked]
    anchors = await store.latest_anchors(ids)
    state = await store.ranked_state(ids)
    oldest = min((s.last_ranked_at for s in state.values()), default=now)
    runs = await store.run_times(oldest)
    limit = config.observation.max_rank
    new: list[ScoutObservation] = []
    for g in ranked:
        if limit is not None and (g.rank or 0) > limit:
            continue
        s = state.get(g.canonical_id)
        missed = sum(1 for t in runs if s is not None and s.last_ranked_at < t < now)
        reason = anchor_reason(g, anchors.get(g.canonical_id), s, missed, now, config)
        if reason is not None:
            new.append(scout_observation(g, result, reason, anchors.get(g.canonical_id)))
    stored = await store.add_scout_observations(new, config.horizons)
    await store.record_run(now, ids)
    return stored


# --- Decisions ------------------------------------------------------------------------------


def _agent(response: ChatResponse, name: str) -> AgentResult | None:
    analysis = response.analysis
    if analysis is None:
        return None
    return next(
        (r for r in analysis.agent_results if r.agent == name and r.status == "ok" and not r.mock),
        None,
    )


def _level(t: Trigger | None) -> DecisionLevel | None:
    if t is None:
        return None
    return DecisionLevel(price=t.price, condition=t.condition, basis=t.basis, confirmed=t.confirmed)


def _invalidation(inv: Invalidation | None, action: str) -> DecisionInvalidation | None:
    if inv is None:
        return None
    # A BUY is invalidated by a close below its level; a SELL (exit long) by one above.
    direction: Literal["above", "below"] = "above" if action == "sell" else "below"
    return DecisionInvalidation(
        price=inv.price, condition=inv.condition, basis=inv.basis, direction=direction
    )


def _dex_snapshot(response: ChatResponse) -> SolanaDexSnapshot | None:
    dex = _agent(response, "dex_market")
    raw = dex.findings.get("snapshot") if dex else None
    if not isinstance(raw, dict):
        return None
    try:
        return SolanaDexSnapshot.model_validate(raw)
    except ValidationError:
        return None


def decision_from_response(
    request: ChatRequest, response: ChatResponse, now: datetime
) -> DecisionObservation | None:
    """The decision this reply showed, on the exact market it was made from; None when no
    real decision was made or its market can't be measured exactly."""
    opportunity = _agent(response, "opportunity")
    if opportunity is None:
        return None
    try:
        a = OpportunityAssessment.model_validate(opportunity.findings)
    except ValidationError:
        return None
    technical = _agent(response, "technical_analysis")
    tf: dict[str, Any] = technical.findings if technical else {}
    dex = _dex_snapshot(response)
    chain = request.asset.chain if request.asset else dex.chain if dex else None
    address = request.asset.address.strip() if request.asset else dex.mint if dex else None
    if dex is not None and chain is not None and (dex.chain != chain or dex.mint != address):
        dex = None  # another token's market: never used
    pools = tf.get("pools") if isinstance(tf.get("pools"), dict) else None
    technical_pool = (pools or {}).get("technical_pool") or {}
    ref: PriceRef | None = None
    if chain and address and isinstance(technical_pool.get("address"), str):
        ref = PriceRef(
            kind="dex_pool", chain=chain, pool_address=technical_pool["address"],
            token_address=address,
        )  # fmt: skip
    elif chain and address and dex is not None:
        ref = PriceRef(
            kind="dex_pool", chain=chain, pool_address=dex.pair_address, token_address=address
        )
    elif not address and isinstance(tf.get("provider"), str) and isinstance(tf.get("symbol"), str):
        ref = PriceRef(
            kind="exchange", provider=tf["provider"], pair=tf.get("pair"), symbol=tf["symbol"]
        )
    if ref is None:
        return None
    risk_result = _agent(response, "risk")
    reference = a.live_price if a.live_price is not None else a.last_close
    asset_id = (
        f"{chain}:{address}"
        if chain and address
        else f"exchange:{ref.provider}:{ref.pair or ref.symbol}"
    )
    return DecisionObservation(
        analyzed_at=now,
        source="scout" if request.asset is not None else "chat",
        asset_id=asset_id,
        symbol=a.asset or (request.asset.symbol if request.asset else None),
        chain=chain,
        address=address,
        price_ref=ref,
        action=a.action,
        setup_action=a.setup_action,
        setup=a.setup,
        confidence=a.confidence,
        position=a.position,
        intent=a.intent,
        action_meaning=a.action_meaning or None,
        reason=a.summary,
        timeframe=a.timeframe,
        buy_trigger=_level(a.bullish_trigger),
        sell_trigger=_level(a.bearish_trigger),
        invalidation=_invalidation(a.invalidation, a.action),
        other_invalidations=[
            i for i in (_invalidation(x, a.action) for x in a.other_invalidations) if i
        ],
        entry_zone=(a.entry_zone.lower, a.entry_zone.upper) if a.entry_zone else None,
        reference_price=reference,
        reference_basis=(
            "live_price" if a.live_price is not None else "last_close" if a.last_close else None
        ),
        last_close=a.last_close,
        liquidity_usd=dex.liquidity_usd if dex else None,
        bullish_score=a.bullish_score,
        bearish_score=a.bearish_score,
        risk=DecisionRisk(
            level=a.risk_level,
            uncertainty_level=a.uncertainty_level,
            summary=risk_result.summary if risk_result else None,
        ),
    )


def decision_signature(d: DecisionObservation) -> str:
    """Asset, decision, timeframe and levels: two decisions with the same signature a few
    seconds apart are one decision (a double click), not two analyses."""
    parts = [
        d.asset_id,
        d.action,
        d.timeframe or "",
        repr(d.buy_trigger.price if d.buy_trigger else None),
        repr(d.sell_trigger.price if d.sell_trigger else None),
        repr(d.invalidation.price if d.invalidation else None),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


async def record_decision(
    store: OutcomeStore,
    request: ChatRequest,
    response: ChatResponse,
    now: datetime,
    config: OutcomeConfig,
) -> DecisionObservation | None:
    policy = config.decisions
    if not policy.enabled:
        return None
    d = decision_from_response(request, response, now)
    if d is None:
        return None
    signature = decision_signature(d)
    recent = await store.recent_decisions(
        d.asset_id, now - timedelta(seconds=policy.duplicate_seconds)
    )
    if any(decision_signature(e) == signature for e in recent):
        return None
    if d.chain and d.address:
        since = now - timedelta(minutes=policy.scout_link_minutes)
        anchors = await store.scout_observations(canonical_id=d.asset_id, since=since, limit=1)
        if anchors and anchors[0].observed_at <= now:
            d = d.model_copy(update={"scout_observation_id": anchors[0].id})
    key = f"{signature}:{now.timestamp():.6f}"
    return await store.add_decision(d, key, config.horizons)
