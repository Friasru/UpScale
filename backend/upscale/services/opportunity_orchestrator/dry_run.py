"""The B1 dry-run: what the future bridge *would* do with the newest settled Scout run.
Reads only; collects, writes and decides nothing.

Safety reuse preview (answers only "would the bridge probably need a new collection?"):
``MISSING`` (no database / no snapshot at or before now), ``INCOMPATIBLE`` (not rules
version 4 / snapshot schema v2, or a body failing its hash), ``NOT_READY`` (a required
domain not assessed in the stored body: token mint VERIFIED with both authorities
AVAILABLE; ``holders.status`` AVAILABLE or PARTIAL; ``market.status`` AVAILABLE; Safety
coverage COMPLETE never substitutes), ``REFRESH_RECOMMENDED`` (ready but older than 45
min), else ``FRESH_REUSABLE``. Nothing here is a safety judgement; Opportunity's own
60-minute rule and readiness check remain authoritative.

Projected actions: ``REJECT`` (not admitted), ``WAITING_FOR_DATA`` (only the run isn't
settled, or the Safety database can't be read), ``REUSE_SAFETY_AND_WOULD_DECIDE``,
``DEFER_BUDGET`` (a collection would be needed and Safety is cooling down or its known
budget is spent), else ``WOULD_REQUIRE_SAFETY_COLLECTION``.
"""

import hashlib
import json
import os
import zlib
from datetime import UTC, datetime
from typing import Any, Literal

from upscale.services.opportunity_model.repository import OpportunityStorageError
from upscale.services.opportunity_orchestrator.admission import (
    evaluate_candidate,
    group_runs,
    prioritized,
    select_run,
)
from upscale.services.opportunity_orchestrator.config import (
    POLICY,
    SAFETY_BUDGET_ENV,
    SAFETY_RULES_VERSION,
    SAFETY_SNAPSHOT_SCHEMA,
    OrchestratorPolicy,
)
from upscale.services.opportunity_orchestrator.models import (
    AdmissionResult,
    BudgetPreview,
    CandidatePlan,
    DbStatus,
    DryRunReport,
    OpportunityPreview,
    ProjectedAction,
    SafetyPreview,
    SafetyReuse,
)
from upscale.services.opportunity_orchestrator.readers import (
    OpportunityDb,
    SafetyDb,
    SafetySnapshotRow,
    scan_archive,
)


def _get(d: Any, *path: str) -> Any:
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def classify_safety(
    db_status: DbStatus, row: SafetySnapshotRow | None, now: datetime,
    policy: OrchestratorPolicy = POLICY,
) -> SafetyPreview:  # fmt: skip
    """Pure: the reuse preview of the latest Safety snapshot at or before ``now``."""
    if db_status == "MISSING":
        return SafetyPreview("MISSING", "MISSING", "YES", detail="no Safety V2 database")
    if db_status == "INCOMPATIBLE":
        return SafetyPreview("INCOMPATIBLE", "INCOMPATIBLE", "UNKNOWN",
                             detail="the Safety V2 database can't be read")  # fmt: skip
    if row is None:
        return SafetyPreview(db_status, "MISSING", "YES", detail="no snapshot of this token")
    age = (now - row.as_of).total_seconds()

    def preview(
        reuse: SafetyReuse,
        required: Literal["YES", "NO"],
        detail: str,
        domains: tuple[tuple[str, str], ...] = (),
    ) -> SafetyPreview:
        assert row is not None
        return SafetyPreview(db_status, reuse, required, row.id, row.as_of, age, domains, detail)

    try:
        text = zlib.decompress(row.body_zlib).decode()
        valid = hashlib.sha256(text.encode()).hexdigest() == row.body_hash
        body = json.loads(text) if valid else {}
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError):
        valid, body = False, {}
    if (
        not valid or row.rules_version != SAFETY_RULES_VERSION
        or body.get("rules_version") != SAFETY_RULES_VERSION
        or body.get("schema_version") != SAFETY_SNAPSHOT_SCHEMA
    ):  # fmt: skip
        return preview(
            "INCOMPATIBLE", "YES", f"snapshot rules {row.rules_version!r} / body unusable"
        )
    domains = (
        ("token_mint", str(_get(body, "identity", "token_mint", "status"))),
        ("mint_authority", str(_get(body, "authority", "mint_authority", "status"))),
        ("freeze_authority", str(_get(body, "authority", "freeze_authority", "status"))),
        ("holders", str(_get(body, "holders", "status"))),
        ("market", str(_get(body, "market", "status"))),
    )
    status = dict(domains)
    ready = (
        status["token_mint"] == "VERIFIED"
        and status["mint_authority"] == status["freeze_authority"] == "AVAILABLE"
        and status["holders"] in ("AVAILABLE", "PARTIAL")
        and status["market"] == "AVAILABLE"
    )
    if not ready:
        return preview("NOT_READY", "YES", "a required domain wasn't assessed", domains)
    if age > policy.safety_reuse_max_age_s:
        return preview("REFRESH_RECOMMENDED", "YES",
                       f"{age:.0f}s old (> {policy.safety_reuse_max_age_s:.0f}s)", domains)  # fmt: skip
    return preview("FRESH_REUSABLE", "NO",
                   "probably reusable; Opportunity decides readiness itself", domains)  # fmt: skip


def budget_preview(db: SafetyDb, now: datetime, env: dict[str, str] | None = None) -> BudgetPreview:
    source = os.environ if env is None else env
    day = now.astimezone(UTC).strftime("%Y-%m-%d")
    used = db.requests_on(day)
    raw = (source.get(SAFETY_BUDGET_ENV) or "").strip()
    limit = int(raw) if raw.isdigit() else None
    return BudgetPreview(
        db_status=db.status, day=day, requests_used_today=used, daily_limit=limit,
        remaining=max(0, limit - used) if limit is not None and used is not None else None,
        limit_source="ENV" if limit is not None else "NOT_KNOWN",
        cooldown_until=db.cooldown_until(now),
    )  # fmt: skip


def _opportunity(db: OpportunityDb, canonical_id: str) -> OpportunityPreview:
    if db.status != "READABLE":
        return OpportunityPreview(db.status, detail=db.detail)
    try:
        s = db.latest(canonical_id)
    except OpportunityStorageError as exc:
        return OpportunityPreview("INCOMPATIBLE", detail=str(exc))
    if s is None:
        return OpportunityPreview("READABLE", detail="no stored decision")
    return OpportunityPreview("READABLE", s.id, s.decision_at, s.decision.decision,
                              s.decision.quality)  # fmt: skip


def project(
    a: AdmissionResult, safety: SafetyPreview | None, budget: BudgetPreview
) -> ProjectedAction:
    """Pure: the action the future bridge would take. Old Opportunity decisions never
    change admission or the projection."""
    if not a.admitted:
        return "WAITING_FOR_DATA" if a.reasons == ("RUN_NOT_SETTLED",) else "REJECT"
    assert safety is not None
    if safety.reuse == "FRESH_REUSABLE":
        return "REUSE_SAFETY_AND_WOULD_DECIDE"
    if safety.db_status == "INCOMPATIBLE":
        return "WAITING_FOR_DATA"
    if budget.cooldown_until is not None or budget.remaining == 0:
        return "DEFER_BUDGET"
    return "WOULD_REQUIRE_SAFETY_COLLECTION"


def dry_run(
    evidence_db: str,
    safety_db: str | None,
    opportunity_db: str,
    now: datetime,
    policy: OrchestratorPolicy = POLICY,
    env: dict[str, str] | None = None,
) -> DryRunReport:
    scan = scan_archive(evidence_db, now, policy.run_lookback_s, policy.archive_scan_limit)
    runs = group_runs(scan.records, now, policy) if not scan.truncated else []
    selected, newer = select_run(runs)
    if selected is None and runs:  # nothing settled yet: show the newest, waiting
        selected = runs[0]
        newer = ()
    sdb, odb = SafetyDb(safety_db), OpportunityDb(opportunity_db)
    try:
        budget = budget_preview(sdb, now, env)
        plans: list[CandidatePlan] = []
        if selected is not None:
            run, records = selected
            results = [evaluate_candidate(r, now=now, policy=policy, run_settled=run.settled)
                       for r in records]  # fmt: skip
            order = {r.scout_record_id: i for i, r in enumerate(prioritized(results), start=1)}
            for a in sorted(results, key=lambda r: (order.get(r.scout_record_id, 10**9),
                                                     r.canonical_id, r.scout_record_id)):  # fmt: skip
                safety = (classify_safety(sdb.status, sdb.latest_snapshot(a.canonical_id, now),
                                          now, policy) if a.admitted else None)  # fmt: skip
                plans.append(CandidatePlan(
                    a, safety, _opportunity(odb, a.canonical_id), project(a, safety, budget),
                    order.get(a.scout_record_id),
                ))  # fmt: skip
        return DryRunReport(
            now=now, archive_status=scan.status,
            selected_run=selected[0] if selected else None, newer_unsettled_runs=newer,
            truncated=scan.truncated, candidates=tuple(plans), budget=budget,
            safety_db_status=sdb.status, opportunity_db_status=odb.status,
        )  # fmt: skip
    finally:
        sdb.close()
        odb.close()
