"""Developer-oriented outcome inspection API (read-only).

* `GET /outcomes/scout`: Scout observations (newest first) with their horizons.
* `GET /outcomes/scout/{id}`: one observation.
* `GET /outcomes/decisions`, `GET /outcomes/decisions/{id}`: Analyze decisions.
* `GET /outcomes/summary`: cohort aggregates for one horizon (small cohorts report
  INSUFFICIENT_SAMPLE, never statistics).
* `GET /outcomes/status`: stored counts and the collector's last cycle.
* `GET /outcomes/replay`: historical replay from stored Scout snapshots (not a backtest).

Nothing here is a win rate, expected return or profitability claim.
"""

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query

import upscale.services as services
from upscale.services.outcomes.analytics import (
    DECISION_DIMENSIONS,
    SCOUT_DIMENSIONS,
    OutcomeSummary,
    summarize_decisions,
    summarize_scout,
)
from upscale.services.outcomes.models import (
    NOT_A_PERFORMANCE_CLAIM,
    DecisionOutcomeRecord,
    ScoutOutcomeRecord,
)
from upscale.services.outcomes.replay import ReplayReport, replay

router = APIRouter(prefix="/outcomes", tags=["outcomes"])
Limit = Annotated[int, Query(ge=1, le=500)]


@router.get("/scout")
async def scout_outcomes(
    canonical_id: str | None = None,
    stage: str | None = None,
    since: datetime | None = None,
    limit: Limit = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ScoutOutcomeRecord]:
    store = services.outcome_store
    observations = await store.scout_observations(canonical_id, stage, since, limit, offset)
    horizons = await store.horizons("scout", [o.id for o in observations if o.id is not None])
    return [
        ScoutOutcomeRecord(observation=o, horizons=horizons.get(o.id or -1, []))
        for o in observations
    ]


@router.get("/scout/{observation_id}")
async def scout_outcome(observation_id: int) -> ScoutOutcomeRecord:
    store = services.outcome_store
    o = await store.scout_observation(observation_id)
    if o is None:
        raise HTTPException(status_code=404, detail="no such Scout observation")
    horizons = await store.horizons("scout", [observation_id])
    return ScoutOutcomeRecord(observation=o, horizons=horizons[observation_id])


@router.get("/decisions")
async def decision_outcomes(
    asset_id: str | None = None,
    action: Literal["buy", "sell", "wait"] | None = None,
    limit: Limit = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DecisionOutcomeRecord]:
    store = services.outcome_store
    decisions = await store.decisions(asset_id, action, limit, offset)
    horizons = await store.horizons("decision", [d.id for d in decisions if d.id is not None])
    return [
        DecisionOutcomeRecord(observation=d, horizons=horizons.get(d.id or -1, []))
        for d in decisions
    ]


@router.get("/decisions/{decision_id}")
async def decision_outcome(decision_id: int) -> DecisionOutcomeRecord:
    store = services.outcome_store
    d = await store.decision(decision_id)
    if d is None:
        raise HTTPException(status_code=404, detail="no such decision observation")
    horizons = await store.horizons("decision", [decision_id])
    return DecisionOutcomeRecord(observation=d, horizons=horizons[decision_id])


@router.get("/summary")
async def outcome_summary(
    kind: Literal["scout", "decision"] = "scout",
    group_by: str = "stage",
    horizon: str = "1h",
) -> OutcomeSummary:
    config = services.outcome_config
    if config.horizon(horizon) is None:
        labels = [h.label for h in config.horizons]
        raise HTTPException(status_code=422, detail=f"horizon must be one of {labels}")
    dimensions = SCOUT_DIMENSIONS if kind == "scout" else DECISION_DIMENSIONS
    if group_by not in dimensions:
        raise HTTPException(status_code=422, detail=f"group_by must be one of {list(dimensions)}")
    store = services.outcome_store
    if kind == "scout":
        return summarize_scout(await store.all_scout(), group_by, horizon, config.analytics)
    return summarize_decisions(await store.all_decisions(), group_by, horizon, config.analytics)


@router.get("/status")
async def outcome_status() -> dict[str, Any]:
    store, collector = services.outcome_store, services.outcome_collector
    cc = services.outcome_config.collector
    cycle = collector.last_cycle
    return {
        "counts": await store.counts(),
        "next_wake": await store.next_wake(cc.settle_seconds, cc.retry_seconds),
        "horizons": [h.label for h in services.outcome_config.horizons],
        "last_cycle": cycle.__dict__ if cycle else None,
        "disclaimer": NOT_A_PERFORMANCE_CLAIM,
    }


@router.get("/replay")
async def outcome_replay(limit: Limit = 200, canonical_id: str | None = None) -> ReplayReport:
    """Read-only: what Scout's own later snapshots recorded after stored rankings."""
    return await asyncio.to_thread(
        replay, services.outcome_store.path, services.outcome_config.horizons, limit=limit,
        canonical_id=canonical_id,
    )  # fmt: skip
