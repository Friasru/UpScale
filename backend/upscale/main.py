import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import upscale.services as services
from upscale.config import CORS_ORIGINS, OUTCOMES_COLLECTOR
from upscale.orchestrator import Orchestrator
from upscale.outcomes_api import router as outcomes_router
from upscale.schemas import ChatRequest, ChatResponse
from upscale.scout_api import (
    SCOUT_LIMITS,
    ChainFilter,
    ScoutFeed,
    ScoutFilters,
    ScoutView,
    StageFilter,
)
from upscale.services.outcomes import record_decision, record_scout_run
from upscale.services.outcomes.models import SurfacingHistory
from upscale.services.scout.growth.models import GrowthScoutResult

logger = logging.getLogger("upscale.outcomes")
# Analyze requests in flight, and when a scan or Analyze last finished: background outcome
# collection makes no provider requests while a user is waiting or just after (on top of
# the quota reserved for interactive work).
_interactive = 0
_last_activity = -1e9


def _busy() -> bool:
    quiet = services.outcome_config.collector.quiet_after_seconds
    recent = time.monotonic() - _last_activity < quiet
    return scout_feed.refreshing or _interactive > 0 or recent


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    stop = asyncio.Event()
    task = None
    if OUTCOMES_COLLECTOR and services.outcome_config.collector.enabled:
        collector = services.outcome_collector
        collector.busy = _busy
        task = asyncio.create_task(collector.run_forever(stop))
    try:
        yield
    finally:
        stop.set()
        if task is not None:
            await task


app = FastAPI(title="UpScale API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(outcomes_router)

orchestrator = Orchestrator()


async def _scan() -> GrowthScoutResult:
    # Every ranked candidate is kept (the view filters and takes the top N); nothing here
    # runs the Analyze agents.
    global _last_activity
    try:
        result = await services.growth_scout_service.scan(
            services.scout_service, services.social_scout_service, limit=500
        )
    finally:
        _last_activity = time.monotonic()
    try:  # measurement only: a failure here never affects the ranking
        if await record_scout_run(services.outcome_store, result, services.outcome_config):
            services.outcome_collector.wake()
    except Exception:
        logger.exception("could not record Scout outcome observations")
    return result


async def _surfacing(canonical_ids: list[str]) -> dict[str, SurfacingHistory]:
    try:
        return await services.outcome_store.surfacing(canonical_ids)
    except Exception:
        logger.exception("could not read Scout surfacing history")
        return {}


scout_feed = ScoutFeed(_scan, history=_surfacing)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    global _interactive, _last_activity
    _interactive += 1
    try:
        response = await orchestrator.respond(request)
    finally:
        _interactive -= 1
        _last_activity = time.monotonic()
    try:  # measurement only: the reply is never delayed by more than a local write
        await record_decision(
            services.outcome_store, request, response, datetime.now(UTC), services.outcome_config
        )
    except Exception:
        logger.exception("could not record the decision observation")
    return response


def _filters(
    limit: int, chain: ChainFilter | None, stage: StageFilter | None, min_liquidity: float | None
) -> ScoutFilters:
    if limit not in SCOUT_LIMITS:
        raise HTTPException(status_code=422, detail=f"limit must be one of {SCOUT_LIMITS}")
    return ScoutFilters(chain=chain, stage=stage, min_liquidity_usd=min_liquidity)


@app.get("/scout")
async def scout(
    limit: int = 10,
    chain: ChainFilter | None = None,
    stage: StageFilter | None = None,
    min_liquidity: Annotated[float | None, Query(ge=0)] = None,
) -> ScoutView:
    """The current Growth Scout ranking (runs one scan if there is none yet)."""
    filters = _filters(limit, chain, stage, min_liquidity)
    return await scout_feed.view(limit, filters)


@app.post("/scout/refresh")
async def scout_refresh(
    limit: int = 10,
    chain: ChainFilter | None = None,
    stage: StageFilter | None = None,
    min_liquidity: Annotated[float | None, Query(ge=0)] = None,
) -> ScoutView:
    """Re-run Scout (shared with a scan already running; skipped if one just finished)."""
    filters = _filters(limit, chain, stage, min_liquidity)
    await scout_feed.refresh()
    return await scout_feed.view(limit, filters)
