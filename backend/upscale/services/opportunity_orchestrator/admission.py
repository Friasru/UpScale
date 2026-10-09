"""Pure admission, priority and Scout-run selection. No I/O and no clock: ``now`` is passed.

Admission (all must hold): an archived ``scout`` record with timing version >= 2 and
``causal_valid``; a Solana candidate whose id is ``solana:<valid mint>`` (exact, case
sensitive) and matches the record; Scout ``eligible`` with a ``rank`` <= ``max_rank``;
``data_status`` CURRENT; ``quality.identity_status`` VERIFIED_IDENTITY; ``stage`` EARLY or
ACCELERATING with no ``unconfirmed_stage``; no positive MARKET_COLLAPSE evidence;
``quality.flow_quality`` not divergent; Scout's market evidence at most
``admission_market_age_s`` old (exactly at the limit is admitted); and the Scout run settled.
Social evidence is never read. Every failed condition is reported (several may apply).

Priority (admitted only, ascending sort = best first): ACCELERATING before EARLY, lower Scout
rank, newer market evidence, canonical id. Scout's rank orders *budget*; it never feeds an
Opportunity decision.

Run selection: a run is every Scout record sharing one decision time. It is settled when
``now - max(archived_at) >= settle_s`` (a quiet-period heuristic: a delayed record could
still arrive later). The newest settled run is used; newer unsettled runs are reported.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from upscale.services.chains import is_solana_address
from upscale.services.evidence_archive.store import EvidenceRecord
from upscale.services.opportunity_orchestrator.config import (
    ADMISSIBLE_STAGES,
    CURRENT_DATA,
    DIVERGENT_FLOW,
    MARKET_COLLAPSE,
    MIN_SCOUT_TIMING_VERSION,
    POLICY,
    STAGE_PRIORITY,
    VERIFIED_IDENTITY,
    OrchestratorPolicy,
)
from upscale.services.opportunity_orchestrator.models import (
    AdmissionReason,
    AdmissionResult,
    PriorityKey,
    ScoutRun,
)


def _time(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _identity(r: EvidenceRecord, c: dict[str, Any]) -> list[AdmissionReason]:
    chain, address, cid = c.get("chain"), c.get("address"), c.get("canonical_id")
    if chain != "solana" or r.chain not in (None, "solana"):
        return ["UNSUPPORTED_CHAIN"]
    if (
        not isinstance(address, str)
        or not is_solana_address(address)
        or cid != f"solana:{address}"
        or r.asset_id != cid
        or r.address not in (None, address)
    ):
        return ["INVALID_IDENTITY"]
    return []


def evaluate_candidate(
    r: EvidenceRecord,
    *,
    now: datetime,
    policy: OrchestratorPolicy = POLICY,
    run_settled: bool = True,
) -> AdmissionResult:
    """Is this archived Scout candidate worth considering for Safety budget?"""
    reasons: list[AdmissionReason] = []
    payload = r.payload if isinstance(r.payload, dict) else {}
    c = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    assert isinstance(c, dict)
    timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
    assert isinstance(timing, dict)
    quality = c.get("quality") if isinstance(c.get("quality"), dict) else {}
    assert isinstance(quality, dict)
    if r.kind != "scout":
        reasons.append("NOT_A_SCOUT_RECORD")
    reasons += _identity(r, c)
    version = timing.get("version")
    market_at = _time(timing.get("market_observed_at"))
    if (
        not isinstance(version, int) or isinstance(version, bool)
        or version < MIN_SCOUT_TIMING_VERSION or market_at is None
        or market_at > r.observed_at or market_at > now
    ):  # fmt: skip
        reasons.append("INVALID_SCOUT_TIMING")
    if r.links.get("causal_valid") is not True:
        reasons.append("CAUSAL_INVALID")
    if c.get("eligible") is not True:
        reasons.append("SCOUT_INELIGIBLE")
    rank = c.get("rank")
    rank = rank if isinstance(rank, int) and not isinstance(rank, bool) else None
    if rank is None:
        reasons.append("UNRANKED")
    elif rank > policy.max_rank or rank < 1:
        reasons.append("RANK_TOO_LOW")
    if c.get("data_status") != CURRENT_DATA:
        reasons.append("STALE_CARRIED")
    if quality.get("identity_status") != VERIFIED_IDENTITY:
        reasons.append("IDENTITY_NOT_VERIFIED")
    stage = c.get("stage") if isinstance(c.get("stage"), str) else None
    if stage not in ADMISSIBLE_STAGES:
        reasons.append("STAGE_NOT_ADMISSIBLE")
    if c.get("unconfirmed_stage") is not None:
        reasons.append("STAGE_UNCONFIRMED")
    if quality.get("market_status") == MARKET_COLLAPSE:
        reasons.append("SCOUT_MARKET_COLLAPSE")
    if quality.get("flow_quality") == DIVERGENT_FLOW:
        reasons.append("DIVERGENT_FLOW")
    age = (now - market_at).total_seconds() if market_at is not None else None
    if age is not None and age > policy.admission_market_age_s:
        reasons.append("SCOUT_TOO_OLD")
    if not run_settled:
        reasons.append("RUN_NOT_SETTLED")
    admitted = not reasons
    priority: PriorityKey | None = None
    if admitted:
        assert stage is not None and rank is not None and market_at is not None
        priority = (STAGE_PRIORITY[stage], rank, -market_at.timestamp(), r.asset_id)
    return AdmissionResult(
        canonical_id=str(c.get("canonical_id") or r.asset_id),
        scout_record_id=r.record_id,
        scout_run_time=r.observed_at,
        market_observed_at=market_at,
        admitted=admitted,
        reasons=("ADMITTED",) if admitted else tuple(dict.fromkeys(reasons)),
        stage=stage,
        rank=rank,
        age_seconds=age,
        priority=priority,
    )


def prioritized(results: Iterable[AdmissionResult]) -> list[AdmissionResult]:
    """Admitted results, best first (deterministic, independent of input order)."""
    admitted = [r for r in results if r.admitted and r.priority is not None]
    return sorted(admitted, key=lambda r: (r.priority, r.scout_record_id))


def group_runs(
    records: Sequence[EvidenceRecord], now: datetime, policy: OrchestratorPolicy = POLICY
) -> list[tuple[ScoutRun, list[EvidenceRecord]]]:
    """Scout runs (records sharing a decision time), newest first, with settle state."""
    by_time: dict[datetime, list[EvidenceRecord]] = {}
    for r in records:
        if r.kind == "scout" and r.observed_at <= now:
            by_time.setdefault(r.observed_at, []).append(r)
    out = []
    for t in sorted(by_time, reverse=True):
        rs = sorted(by_time[t], key=lambda r: r.id)
        last = max(r.archived_at for r in rs)
        settled = (now - last).total_seconds() >= policy.settle_s
        out.append((ScoutRun(t, len(rs), last, settled), rs))
    return out


def select_run(
    runs: Sequence[tuple[ScoutRun, list[EvidenceRecord]]],
) -> tuple[tuple[ScoutRun, list[EvidenceRecord]] | None, tuple[ScoutRun, ...]]:
    """(the newest settled run, the newer unsettled runs it skipped)."""
    newer: list[ScoutRun] = []
    for run, records in runs:
        if run.settled:
            return (run, records), tuple(newer)
        newer.append(run)
    return None, tuple(newer)
