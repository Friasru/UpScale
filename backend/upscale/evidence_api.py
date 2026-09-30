"""Point-in-Time Evidence Archive status (read-only, developer-oriented).

* `GET /evidence/status`: totals, archive writer health, coverage over Growth Scout
  observations (market / on-chain safety / social / decision-grade) and replay readiness
  counts. Evidence coverage only: not a profitability or trading-readiness claim.
"""

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query

import upscale.services as services
from upscale.services.evidence_archive.status import status

router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get("/status")
async def evidence_status(days: Annotated[float, Query(gt=0, le=365)] = 7.0) -> dict[str, Any]:
    store = services.evidence_store
    enrichment = services.safety_enrichment
    base: dict[str, Any] = {
        "enabled": store is not None,
        "enrichment": {
            "enabled": enrichment.settings.enabled,
            "max_per_refresh": enrichment.settings.max_per_refresh,
            "last": enrichment.last_report.as_dict() if enrichment.last_report else None,
        },
    }
    if store is None:
        return base | {"reason": "UPSCALE_EVIDENCE_ARCHIVE is off"}
    recorder = services.evidence_recorder
    stats = recorder.stats.as_dict() if recorder is not None else None
    report = await asyncio.to_thread(status, store, datetime.now(UTC), days, stats)
    return base | report
