"""Orchestration: build -> decide -> persist one decision under a run, and (separately)
check whether the current upstream stores still reproduce a stored input.

The only clock read is here (``clock``): a LIVE_FORWARD decision's ``decision_at`` is the
moment it reads; ``decided_at`` / run times are audit timestamps outside every hash. Nothing
here collects evidence: the Evidence Archive and Safety V2 are read through the O1
read-only loaders, and a missing source stays missing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from upscale.log_safety import redact
from upscale.services.opportunity_model.decision import decide
from upscale.services.opportunity_model.decision_models import OpportunityDecision
from upscale.services.opportunity_model.fingerprints import differing, input_fingerprints
from upscale.services.opportunity_model.loaders import (
    ArchiveReader,
    ArchiveSource,
    SafetyReader,
    SafetySource,
)
from upscale.services.opportunity_model.models import (
    OpportunityCausalityError,
    OpportunityError,
    OpportunityIdentityError,
    OpportunityOrigin,
)
from upscale.services.opportunity_model.repository import OpportunityRepository
from upscale.services.opportunity_model.service import build_input

Clock = Callable[[], datetime]
SourceVerifyStatus = Literal[
    "MATCH", "SOURCE_MISSING", "SOURCE_CHANGED", "INCOMPATIBLE", "CAUSALITY_ERROR"
]
_READ = ("AVAILABLE", "STALE")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RecordResult:
    run_id: int
    decision_id: int
    created: bool
    decision: OpportunityDecision


def record_decision(
    repo: OpportunityRepository,
    canonical_id: str,
    origin: OpportunityOrigin,
    archive: ArchiveSource,
    safety: SafetySource,
    as_of: datetime | None = None,
    clock: Clock = utc_now,
) -> RecordResult:
    """HISTORICAL_REPLAY needs an explicit ``as_of``; LIVE_FORWARD takes the clock (and
    refuses an ``as_of``). Any exception, KeyboardInterrupt or SystemExit after the run is
    created finishes it ABORTED with a redacted reason and is re-raised; a stored decision is
    all-or-nothing."""
    if origin == "HISTORICAL_REPLAY":
        if as_of is None:
            raise ValueError("a HISTORICAL_REPLAY decision needs an explicit as_of")
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        decision_at = as_of.astimezone(UTC)
    else:
        if as_of is not None:
            raise ValueError("a LIVE_FORWARD decision is made at the current time (no as_of)")
        decision_at = clock()
    run_id = repo.start_run(
        canonical_id, origin, clock(), decision_at if origin == "HISTORICAL_REPLAY" else None
    )
    try:
        inp = build_input(canonical_id, decision_at, origin, archive, safety)
        decision = decide(inp)
        stored = repo.store(run_id, inp, decision, decided_at=clock(), finished_at=clock())
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        # The decision transaction already rolled back (`repository._tx`). Finish the run
        # ABORTED with a redacted reason, then re-raise: Ctrl-C / exit are never swallowed.
        # Only uncatchable termination (SIGKILL, power loss, a hard crash) between the
        # committed RUNNING row and this point can leave a run RUNNING, with no decision.
        repo.abort_run(run_id, redact(f"{type(exc).__name__}: {exc}"), clock())
        raise
    return RecordResult(run_id, stored.decision_id, stored.created, decision)


def record_from_paths(
    repo: OpportunityRepository,
    canonical_id: str,
    origin: OpportunityOrigin,
    evidence_db: str,
    safety_db: str | None,
    as_of: datetime | None = None,
    clock: Clock = utc_now,
) -> RecordResult:
    archive, safety = ArchiveReader(evidence_db), SafetyReader(safety_db)
    try:
        return record_decision(repo, canonical_id, origin, archive, safety, as_of, clock)
    finally:
        archive.close()
        safety.close()


@dataclass(frozen=True)
class SourceVerifyResult:
    decision_id: int
    status: SourceVerifyStatus
    detail: str
    sources: tuple[str, ...] = ()


def verify_sources(
    repo: OpportunityRepository, decision_id: int, archive: ArchiveSource, safety: SafetySource
) -> SourceVerifyResult:
    """Do today's upstream stores (read-only) still rebuild the stored input exactly? This
    is *not* decision verification: a stored decision is verified from its stored input
    alone (`OpportunityRepository.verify_decision`)."""
    stored = repo.get(decision_id)
    diff = differing(stored.input_fingerprints, input_fingerprints())
    if diff:
        return SourceVerifyResult(decision_id, "INCOMPATIBLE",
                                  "input code changed since storage: " + ", ".join(diff))  # fmt: skip
    inp = stored.input
    try:
        rebuilt = build_input(inp.canonical_id, inp.decision_at, inp.origin, archive, safety)
    except OpportunityCausalityError as exc:
        return SourceVerifyResult(decision_id, "CAUSALITY_ERROR", str(exc))
    except OpportunityIdentityError as exc:
        return SourceVerifyResult(decision_id, "SOURCE_CHANGED", str(exc))
    except OpportunityError as exc:
        return SourceVerifyResult(decision_id, "INCOMPATIBLE", str(exc))
    if rebuilt.input_hash() == stored.input_hash:
        return SourceVerifyResult(decision_id, "MATCH", "the sources rebuild the stored input")
    names = ("scout", "technical", "social", "news", "safety")
    old, new = inp.sources, rebuilt.sources
    missing = tuple(
        n for n in names
        if getattr(old, n).ref.status in _READ and getattr(new, n).ref.status not in _READ
    )  # fmt: skip
    if missing:
        return SourceVerifyResult(decision_id, "SOURCE_MISSING",
                                  "previously read sources are no longer readable", missing)  # fmt: skip
    changed = tuple(n for n in names if getattr(old, n) != getattr(new, n))
    return SourceVerifyResult(decision_id, "SOURCE_CHANGED",
                              "the sources now give a different input", changed)  # fmt: skip
