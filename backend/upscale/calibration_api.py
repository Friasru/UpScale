"""Calibration Engine, read-only API (developer-oriented). Nothing here changes production
configuration or candidate state, and HOLDOUT is never read (counts only)."""

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from upscale.services.calibration.config import (
    default_calibration_db,
    default_live_db,
    default_replay_db,
)
from upscale.services.calibration.engine import NOT_A_PROFIT_CLAIM, CalibrationEngine
from upscale.services.calibration.store import CalibrationStore

router = APIRouter(prefix="/calibration", tags=["calibration"])


def _engine() -> CalibrationEngine | None:
    path = Path(default_calibration_db()).expanduser()
    store = CalibrationStore(path, read_only=True) if path.exists() else None
    if store is None:
        return None
    return CalibrationEngine(store, default_live_db(), default_replay_db())


def _readiness() -> dict[str, Any]:
    engine = _engine() or CalibrationEngine(
        CalibrationStore(":memory:"), default_live_db(), default_replay_db()
    )
    try:
        return engine.readiness()
    finally:
        engine.store.close()


@router.get("/status")
async def calibration_status() -> dict[str, Any]:
    engine = _engine()
    if engine is None:
        return {
            "label": NOT_A_PROFIT_CLAIM,
            "calibration_db": None,
            "runs": [],
            "candidates_by_status": {},
        }
    try:
        return await asyncio.to_thread(engine.status)
    finally:
        engine.store.close()


@router.get("/readiness")
async def calibration_readiness() -> dict[str, Any]:
    return await asyncio.to_thread(_readiness)


@router.get("/findings")
async def calibration_findings() -> list[dict[str, Any]]:
    engine = _engine()
    if engine is None:
        return []
    try:
        return await asyncio.to_thread(engine.store.findings)
    finally:
        engine.store.close()


@router.get("/candidates")
async def calibration_candidates() -> list[dict[str, Any]]:
    engine = _engine()
    if engine is None:
        return []
    try:
        return await asyncio.to_thread(engine.store.candidates)
    finally:
        engine.store.close()
