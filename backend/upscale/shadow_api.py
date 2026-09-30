"""Shadow / Paper Strategy Engine, read-only API (developer-oriented). Simulated results
only: nothing here places, signs or suggests a real trade. A missing or failing shadow
database is reported, never raised into the app."""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastapi import APIRouter, Query

from upscale.services.shadow.config import NOT_REAL_PROFIT, default_shadow_db
from upscale.services.shadow.engine import ShadowEngine, ShadowError, resolve_run
from upscale.services.shadow.store import ShadowStore, ShadowStoreError, readable

router = APIRouter(prefix="/shadow", tags=["shadow"])
logger = logging.getLogger("upscale.shadow")
T = TypeVar("T")


def _read(fn: Callable[[ShadowEngine], T], empty: T) -> T | dict[str, Any]:
    path = Path(default_shadow_db()).expanduser()
    if not path.exists():
        return empty
    store = ShadowStore(path, read_only=True)
    try:
        return fn(ShadowEngine(store, None))
    except (ShadowError, ShadowStoreError) as exc:
        return {"label": NOT_REAL_PROFIT, "error": str(exc)}
    except Exception as exc:  # a damaged shadow database never breaks the API
        logger.exception("shadow API read failed")
        return {"label": NOT_REAL_PROFIT, "error": type(exc).__name__}
    finally:
        store.close()


def _run(engine: ShadowEngine, run: str | None) -> str:
    return resolve_run(engine.store, run)


def _time(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    value = datetime.fromisoformat(raw)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@router.get("/status")
async def shadow_status() -> dict[str, Any]:
    empty: dict[str, Any] = {"label": NOT_REAL_PROFIT, "shadow_db": None, "runs": []}
    result = await asyncio.to_thread(_read, lambda e: e.status(), empty)
    assert isinstance(result, dict)
    return result


@router.get("/strategies")
async def shadow_strategies() -> list[dict[str, Any]] | dict[str, Any]:
    return await asyncio.to_thread(
        _read,
        lambda e: [
            {"key": s.key, "config_hash": s.config_hash, **s.model_dump(mode="json")}
            for s in e.store.strategies()
        ],  # fmt: skip
        [],
    )


@router.get("/positions")
async def shadow_positions(
    run: str | None = None,
    strategy: str | None = None,
    asset: str | None = None,
    status: Literal["OPEN", "CLOSED"] | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    return await asyncio.to_thread(
        _read,
        lambda e: readable(e.store.positions(_run(e, run), strategy, asset, status)),
        [],
    )


@router.get("/trades")
async def shadow_trades(
    run: str | None = None,
    strategy: str | None = None,
    asset: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    return await asyncio.to_thread(
        _read,
        lambda e: readable(
            e.store.trades(_run(e, run), strategy, asset, _time(since), _time(until))
        ),
        [],
    )


@router.get("/decisions")
async def shadow_decisions(
    run: str | None = None,
    strategy: str | None = None,
    asset: str | None = None,
    action: Literal["ENTER", "HOLD", "EXIT", "NO_ACTION"] | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 200,
) -> list[dict[str, Any]] | dict[str, Any]:
    return await asyncio.to_thread(
        _read,
        lambda e: readable(
            e.store.decisions(_run(e, run), strategy, asset, action=action, limit=limit)
        ),
        [],
    )


@router.get("/metrics")
async def shadow_metrics(
    run: str | None = None,
    strategy: str | None = None,
    asset: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    empty: dict[str, Any] = {"label": NOT_REAL_PROFIT, "strategies": []}
    result = await asyncio.to_thread(
        _read,
        lambda e: e.metrics(_run(e, run), strategy, asset, _time(since), _time(until)),
        empty,
    )
    assert isinstance(result, dict)
    return result


@router.get("/diagnostics")
async def shadow_diagnostics(
    run: str | None = None,
    strategy: Annotated[list[str] | None, Query()] = None,
    asset: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """Why each strategy entered or rejected Scout candidates (stored rows only)."""
    empty: dict[str, Any] = {"label": NOT_REAL_PROFIT, "strategies": []}
    result = await asyncio.to_thread(
        _read,
        lambda e: e.diagnostics(_run(e, run), strategy, asset, _time(since), _time(until)),
        empty,
    )
    assert isinstance(result, dict)
    return result


@router.get("/rejections")
async def shadow_rejections(
    run: str | None = None,
    strategy: str | None = None,
    asset: str | None = None,
    reason: str | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 200,
) -> list[dict[str, Any]] | dict[str, Any]:
    return await asyncio.to_thread(
        _read,
        lambda e: readable(
            e.store.rejections(_run(e, run), strategy, asset, reason=reason, limit=limit)
        ),
        [],
    )


@router.get("/funnel")
async def shadow_funnel(
    run: str | None = None,
    strategy: Annotated[list[str] | None, Query()] = None,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """Sequential entry funnel and conditional counts (exact aggregate counters)."""
    empty: dict[str, Any] = {"label": NOT_REAL_PROFIT, "strategies": []}
    result = await asyncio.to_thread(
        _read,
        lambda e: e.funnel(_run(e, run), strategy, _time(since), _time(until)),
        empty,
    )
    assert isinstance(result, dict)
    return result


@router.get("/storage")
async def shadow_storage() -> dict[str, Any]:
    """Shadow database size, rows per table and diagnostics storage settings."""
    empty: dict[str, Any] = {"label": NOT_REAL_PROFIT, "shadow_db": None}
    result = await asyncio.to_thread(_read, lambda e: e.storage(), empty)
    assert isinstance(result, dict)
    return result
