"""Typed results of the bridge's read-only admission, priority and dry-run.

Admission means only "worth considering for Safety budget": never safe, ENTER, BUY or a
prediction of profit.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

AdmissionReason = Literal[
    "ADMITTED",
    "NOT_A_SCOUT_RECORD",
    "UNSUPPORTED_CHAIN",
    "INVALID_IDENTITY",
    "INVALID_SCOUT_TIMING",
    "CAUSAL_INVALID",
    "SCOUT_INELIGIBLE",
    "UNRANKED",
    "RANK_TOO_LOW",
    "STALE_CARRIED",
    "IDENTITY_NOT_VERIFIED",
    "STAGE_NOT_ADMISSIBLE",
    "STAGE_UNCONFIRMED",
    "SCOUT_MARKET_COLLAPSE",
    "DIVERGENT_FLOW",
    "SCOUT_TOO_OLD",
    "RUN_NOT_SETTLED",
]
SafetyReuse = Literal[
    "FRESH_REUSABLE", "REFRESH_RECOMMENDED", "NOT_READY", "MISSING", "INCOMPATIBLE"
]
DbStatus = Literal["READABLE", "MISSING", "INCOMPATIBLE"]
ProjectedAction = Literal[
    "REJECT",
    "REUSE_SAFETY_AND_WOULD_DECIDE",
    "WOULD_REQUIRE_SAFETY_COLLECTION",
    "DEFER_BUDGET",
    "WAITING_FOR_DATA",
]
# (stage order, rank, -market time, canonical_id): sorts ascending, best first.
PriorityKey = tuple[int, int, float, str]


@dataclass(frozen=True)
class AdmissionResult:
    canonical_id: str
    scout_record_id: str
    scout_run_time: datetime
    market_observed_at: datetime | None
    admitted: bool
    reasons: tuple[AdmissionReason, ...]
    stage: str | None
    rank: int | None
    age_seconds: float | None
    priority: PriorityKey | None  # only when admitted


@dataclass(frozen=True)
class ScoutRun:
    """All archived Scout records sharing one decision time."""

    decision_time: datetime
    record_count: int
    last_archived_at: datetime
    settled: bool


@dataclass(frozen=True)
class SafetyPreview:
    db_status: DbStatus
    reuse: SafetyReuse
    collection_required: Literal["YES", "NO", "UNKNOWN"]
    snapshot_id: int | None = None
    as_of: datetime | None = None
    age_seconds: float | None = None
    domains: tuple[tuple[str, str], ...] = ()  # (domain, stored status)
    detail: str = ""


@dataclass(frozen=True)
class OpportunityPreview:
    db_status: DbStatus
    decision_id: int | None = None
    decision_at: datetime | None = None
    decision: str | None = None
    quality: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class BudgetPreview:
    """What Safety's request ledger says, read-only. The daily limit is known only when
    configured explicitly; the collection cost bound isn't knowable without Safety."""

    db_status: DbStatus
    day: str
    requests_used_today: int | None
    daily_limit: int | None
    remaining: int | None
    limit_source: Literal["ENV", "NOT_KNOWN"]
    cooldown_until: datetime | None
    cost_bound: Literal["COST_BOUND_UNKNOWN"] = "COST_BOUND_UNKNOWN"
    cost_bound_note: str = (
        "a collection's request count depends on Safety's holder page cap, retry count and "
        "provider scan capability, which B1 can't read without importing Safety"
    )


@dataclass(frozen=True)
class CandidatePlan:
    admission: AdmissionResult
    safety: SafetyPreview | None  # None: not inspected (rejected)
    opportunity: OpportunityPreview | None
    action: ProjectedAction
    order: int | None  # 1-based processing order among admitted candidates


@dataclass(frozen=True)
class DryRunReport:
    now: datetime
    archive_status: DbStatus
    selected_run: ScoutRun | None
    newer_unsettled_runs: tuple[ScoutRun, ...]
    truncated: bool
    candidates: tuple[CandidatePlan, ...]
    budget: BudgetPreview
    safety_db_status: DbStatus
    opportunity_db_status: DbStatus
    note: str = (
        "DRY RUN: nothing was collected, written or decided. Admission means only 'worth "
        "considering for Safety budget' - never safe, ENTER or BUY. LIVE PROCESSING: NOT "
        "IMPLEMENTED."
    )
