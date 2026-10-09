"""The Safety port the bridge depends on (B2: protocol + offline implementation only).

A future B3 adapter will implement it with Safety V2's own service; B2 never imports Safety.
Results separate *observed token evidence* (may lead to a Safety snapshot and an Opportunity
decision) from *infrastructure failures* (never an Opportunity decision: the job is
DEFERRED, RETRY_WAIT or FAILED). A preflight is conservative, never a guarantee: Safety's
own RequestGuard stays the final authority.

`ReadOnlySafetyPort` is what the offline CLI uses: it can only *reuse* existing Safety
snapshots (read-only, B1's reuse preview); any needed collection is
PROVIDER_NOT_CONFIGURED (live Safety collection is not implemented).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from upscale.services.opportunity_orchestrator.dry_run import classify_safety
from upscale.services.opportunity_orchestrator.models import SafetyReuse
from upscale.services.opportunity_orchestrator.policy import PreflightLabel
from upscale.services.opportunity_orchestrator.readers import SafetyDb

TokenOutcome = Literal[
    "COMPLETE", "HOLDERS_PARTIAL", "NOT_A_MINT", "ACCOUNT_MISSING", "NO_POOLS", "MARKET_CLOSED"
]
InfraCategory = Literal[
    "PROVIDER_UNAVAILABLE",
    "PROVIDER_TIMEOUT",
    "SAFETY_BUDGET_EXHAUSTED",
    "SAFETY_COOLDOWN",
    "PROVIDER_NOT_CONFIGURED",
    "DATABASE_LOCK",
    "COLLECTION_EXCEPTION",
]
# Infrastructure that defers without consuming a retry attempt.
DEFERRING: tuple[InfraCategory, ...] = (
    "SAFETY_BUDGET_EXHAUSTED", "SAFETY_COOLDOWN", "PROVIDER_NOT_CONFIGURED",
)  # fmt: skip
# Transient infrastructure: one failed attempt each.
TRANSIENT: tuple[InfraCategory, ...] = (
    "PROVIDER_UNAVAILABLE", "PROVIDER_TIMEOUT", "DATABASE_LOCK", "COLLECTION_EXCEPTION",
)  # fmt: skip


@dataclass(frozen=True)
class SafetyInspection:
    reuse: SafetyReuse
    snapshot_id: int | None = None
    as_of: datetime | None = None
    detail: str = ""


@dataclass(frozen=True)
class Preflight:
    label: PreflightLabel
    blocked: InfraCategory | None = None  # a DEFERRING category, or None to go ahead
    retry_at: datetime | None = None
    detail: str = ""


@dataclass(frozen=True)
class CollectionResult:
    kind: Literal["TOKEN_EVIDENCE", "INFRASTRUCTURE"]
    outcome: TokenOutcome | InfraCategory
    retry_at: datetime | None = None
    detail: str = ""


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: int
    as_of: datetime


class SafetyPort(Protocol):
    def inspect(self, canonical_id: str, now: datetime) -> SafetyInspection: ...

    def preflight(self, canonical_id: str, now: datetime) -> Preflight: ...

    def collect(self, canonical_id: str, now: datetime) -> CollectionResult: ...

    def snapshot(self, canonical_id: str, now: datetime) -> SnapshotRef: ...


class ReadOnlySafetyPort:
    """Reads the Safety database (``mode=ro``); never collects or writes."""

    def __init__(self, safety_db: str | None):
        self.safety_db = safety_db

    def inspect(self, canonical_id: str, now: datetime) -> SafetyInspection:
        db = SafetyDb(self.safety_db)
        try:
            p = classify_safety(db.status, db.latest_snapshot(canonical_id, now), now)
        finally:
            db.close()
        return SafetyInspection(p.reuse, p.snapshot_id, p.as_of, p.detail)

    def preflight(self, canonical_id: str, now: datetime) -> Preflight:
        return Preflight("COST_BOUND_UNKNOWN", "PROVIDER_NOT_CONFIGURED",
                         detail="LIVE SAFETY COLLECTION: NOT IMPLEMENTED")  # fmt: skip

    def collect(self, canonical_id: str, now: datetime) -> CollectionResult:
        return CollectionResult("INFRASTRUCTURE", "PROVIDER_NOT_CONFIGURED",
                                detail="LIVE SAFETY COLLECTION: NOT IMPLEMENTED")  # fmt: skip

    def snapshot(self, canonical_id: str, now: datetime) -> SnapshotRef:
        raise NotImplementedError("the read-only port never builds Safety snapshots")
