import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import upscale.services as services
from upscale.background_scout import (
    BackgroundScout,
    BackgroundScoutStatus,
    ScanDeferred,
    ScanFailed,
    ScanSummary,
    defer_reason,
    load_settings,
    outcome_backlog,
    provider_pressure,
)
from upscale.config import (
    BACKGROUND_SCOUT,
    BACKGROUND_SCOUT_INTERVAL_MINUTES,
    CORS_ORIGINS,
    OUTCOMES_COLLECTOR,
)
from upscale.evidence_api import router as evidence_router
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
from upscale.services.evidence_archive import hooks as evidence
from upscale.services.outcomes import record_decision, record_scout_run
from upscale.services.outcomes.models import SurfacingHistory
from upscale.services.scout.growth.models import GrowthScoutResult

logger = logging.getLogger("upscale.outcomes")
# Analyze requests in flight, and when a scan or Analyze last finished: background outcome
# collection makes no provider requests while a user is waiting or just after (on top of
# the quota reserved for interactive work).
_interactive = 0
_last_activity = -1e9
_last_analyze = -1e9  # when an Analyze last finished (background Scout stays away just after)
# New outcome anchors recorded by the latest scan (None: not recorded / unknown).
_scan_anchors: int | None = None


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
    background_scout.start()
    try:
        yield
    finally:
        await background_scout.stop()
        if _enrichment_task is not None and not _enrichment_task.done():
            _enrichment_task.cancel()
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
app.include_router(evidence_router)

orchestrator = Orchestrator()


async def _scan() -> GrowthScoutResult:
    # Every ranked candidate is kept (the view filters and takes the top N); nothing here
    # runs the Analyze agents.
    global _last_activity, _scan_anchors
    _scan_anchors = None
    # Evidence emitted during the scan (market, social, on-chain safety) is collected, so the
    # archived evaluation links exactly what it used.
    with evidence.evaluation() as used:
        try:
            result = await services.growth_scout_service.scan(
                services.scout_service, services.social_scout_service, limit=500
            )
        finally:
            _last_activity = time.monotonic()
    # Archive the ranking as final now (after every lookup it used); no request.
    evidence.emit("scout", result, decision_at=datetime.now(UTC), scope=used)
    _start_enrichment(result)
    try:  # measurement only: a failure here never affects the ranking
        anchored = await record_scout_run(services.outcome_store, result, services.outcome_config)
        _scan_anchors = len(anchored)
        if anchored:
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
_enrichment_task: asyncio.Task[object] | None = None


def _enrichment_defer_reason() -> str | None:
    """Safety enrichment is below every other production workload: Analyze, due outcome
    work, manual and background Scout."""
    cc = services.outcome_config.collector
    quiet = cc.quiet_after_seconds
    if _interactive > 0 or time.monotonic() - _last_analyze < quiet:
        return "Analyze is active"
    backlog = outcome_backlog(
        services.outcome_collector.last_cycle,
        datetime.now(UTC),
        timedelta(seconds=2 * cc.max_sleep_seconds),
    )
    if backlog is not None:
        return backlog
    if scout_feed.refreshing or background_scout.running:
        return "a Scout scan is running"
    return None


def _start_enrichment(result: GrowthScoutResult) -> None:
    """Optional on-chain safety enrichment after a scan (lowest priority, bounded)."""
    global _enrichment_task
    enrichment = services.safety_enrichment
    if not enrichment.settings.enabled or enrichment.running:
        return
    if _enrichment_task is not None and not _enrichment_task.done():
        return
    enrichment.busy = _enrichment_defer_reason
    _enrichment_task = asyncio.create_task(enrichment.after_scan(result))


def _background_defer_reason() -> str | None:
    """Why background Scout should wait: Analyze, then due outcome work, then a manual
    Scout refresh, then provider pressure (see upscale.background_scout)."""
    cc = services.outcome_config.collector
    quiet = cc.quiet_after_seconds
    # The collector's latest cycle only counts while it is recent (it runs at least every
    # `max_sleep_seconds`, sooner while work is pending).
    max_age = timedelta(seconds=2 * cc.max_sleep_seconds)
    cooldown = background_scout.settings.rate_limit_cooldown_minutes * 60
    return defer_reason(
        analyze_active=_interactive > 0 or time.monotonic() - _last_analyze < quiet,
        outcome_backlog=outcome_backlog(
            services.outcome_collector.last_cycle, datetime.now(UTC), max_age
        ),
        scout_running=scout_feed.refreshing,
        provider_pressure=provider_pressure(services.scout_service.gates(), cooldown),
    )


async def _background_scan() -> ScanSummary:
    """Exactly the scan a manual refresh runs (shared single-flight feed, same persistence
    and outcome-anchor policy); the view's last good results stay if it fails."""
    with evidence.component("scout_background"):
        ran = await scout_feed.refresh()
    if not ran:
        raise ScanDeferred("a Scout scan finished moments ago")
    result = scout_feed.result
    if scout_feed.error is not None or result is None:
        raise ScanFailed(scout_feed.error or "Scout scan produced no result")
    return ScanSummary(
        candidates_discovered=result.universe.discovered if result.universe else None,
        candidates_ranked=result.eligible,
        new_outcome_anchors=_scan_anchors,
    )


background_scout = BackgroundScout(
    load_settings(BACKGROUND_SCOUT, BACKGROUND_SCOUT_INTERVAL_MINUTES),
    _background_scan,
    _background_defer_reason,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    global _interactive, _last_activity, _last_analyze
    _interactive += 1
    try:
        with evidence.component("analyze"), evidence.evaluation() as used:
            response = await orchestrator.respond(request)
    finally:
        _interactive -= 1
        _last_activity = _last_analyze = time.monotonic()
    now = datetime.now(UTC)
    observation = None
    try:  # measurement only: the reply is never delayed by more than a local write
        observation = await record_decision(
            services.outcome_store, request, response, now, services.outcome_config
        )
    except Exception:
        logger.exception("could not record the decision observation")
    # Archive why this decision was made (queued; never delays or changes the reply).
    with evidence.component("analyze"):
        evidence.emit(
            "decision", response, request=request, observation=observation, at=now, scope=used
        )
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
    with evidence.component("scout_manual"):
        await scout_feed.refresh()
    return await scout_feed.view(limit, filters)


@app.get("/scout/background/status")
async def scout_background_status() -> BackgroundScoutStatus:
    """The background Scout scheduler's actual state and its last attempt."""
    return background_scout.status()
