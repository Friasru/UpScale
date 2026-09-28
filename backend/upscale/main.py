from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import upscale.services as services
from upscale.config import CORS_ORIGINS
from upscale.orchestrator import Orchestrator
from upscale.schemas import ChatRequest, ChatResponse
from upscale.scout_api import (
    SCOUT_LIMITS,
    ChainFilter,
    ScoutFeed,
    ScoutFilters,
    ScoutView,
    StageFilter,
)
from upscale.services.scout.growth.models import GrowthScoutResult

app = FastAPI(title="UpScale API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

orchestrator = Orchestrator()


async def _scan() -> GrowthScoutResult:
    # Every ranked candidate is kept (the view filters and takes the top N); nothing here
    # runs the Analyze agents.
    return await services.growth_scout_service.scan(
        services.scout_service, services.social_scout_service, limit=500
    )


scout_feed = ScoutFeed(_scan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    return await orchestrator.respond(request)


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
