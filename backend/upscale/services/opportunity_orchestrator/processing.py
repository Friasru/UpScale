"""Offline job processing: enqueue admitted Scout candidates, drive each job through Safety
(via a `SafetyPort`) to a real Opportunity decision, retry transient failures, recover
stale work. Single worker; at-least-once with idempotent effects.

Flow: QUEUED -> (fresh reusable Safety) SNAPSHOTTED, or -> per-token throttle -> preflight
-> COLLECTING -> collect -> snapshot -> SNAPSHOTTED; then SNAPSHOTTED -> DECIDING -> the real
Opportunity recorder (LIVE_FORWARD, ``decision_at`` = the orchestration clock at decision
time, never the Scout time) -> DECIDED.

* Observed token evidence (complete / partial holders, NOT_A_MINT, ACCOUNT_MISSING,
  NO_POOLS, a closed market) continues to Opportunity.
* Infrastructure never creates an Opportunity decision. Budget, cooldown and "not
  configured" defer (no attempt counted); provider unavailable / timeout / database lock /
  a collection exception count one failed attempt: RETRY_WAIT after 10 min, then 30 min,
  and the third failed attempt is FAILED. ``attempt_count`` counts collection attempts that
  started and failed transiently or were interrupted.
* Per token: no new collection within 60 min of the last *successful* one, and at most 6
  collection starts (successful or failed) per UTC day (a bridge throttle; Safety's
  RequestGuard stays the final authority).
* Decision key: entering DECIDING stores ``opportunity_decision_at`` and
  ``opportunity_rules_version`` once (with the lease and the event). The Opportunity
  decision is made under exactly that key, from exactly the job's Scout record and Safety
  snapshot (pinned read-only readers handed to Opportunity's own recorder: O1
  normalization, the engine and O3 storage are unchanged). The exact input is built and
  provenance-checked before anything is written, and the stored decision is checked again
  before it is linked.
* Recovery: a stale COLLECTING job (lease older than 15 min) becomes RETRY_WAIT / FAILED,
  counting the interrupted attempt once; a stale SNAPSHOTTED job resumes to a decision; a
  stale DECIDING job looks up the exact *stored* Opportunity key (token, LIVE_FORWARD,
  stored decision_at, stored rules version) and adopts it only when the decision's stored
  input names this job's Scout record and Safety snapshot; another occupant fails the job;
  no decision yet -> it is recorded again under the same key, unless the running rules
  version differs (OPPORTUNITY_RULES_VERSION_MISMATCH). Nearby decisions are never
  looked at.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from upscale.log_safety import redact
from upscale.services.opportunity_model import config as opportunity_config
from upscale.services.opportunity_model.models import OpportunityError, OpportunityInput
from upscale.services.opportunity_model.recorder import record_decision
from upscale.services.opportunity_model.repository import (
    OpportunityConflictError,
    OpportunityRepository,
)
from upscale.services.opportunity_model.service import build_input
from upscale.services.opportunity_orchestrator.admission import (
    evaluate_candidate,
    group_runs,
    select_run,
)
from upscale.services.opportunity_orchestrator.config import (
    POLICY,
    PROCESSING,
    OrchestratorPolicy,
    ProcessingPolicy,
)
from upscale.services.opportunity_orchestrator.models import ScoutRun
from upscale.services.opportunity_orchestrator.readers import (
    PinnedArchive,
    PinnedSafety,
    opportunity_decision_by_key,
    scan_archive,
)
from upscale.services.opportunity_orchestrator.repository import (
    Job,
    OrchestratorRepository,
)
from upscale.services.opportunity_orchestrator.safety_port import (
    DEFERRING,
    CollectionResult,
    InfraCategory,
    SafetyPort,
)

Clock = Callable[[], datetime]


def _same_time(a: datetime | None, b: datetime | None) -> bool:
    """Equal as stored (UTC epoch seconds, the representation both databases keep)."""
    return a is not None and b is not None and a.timestamp() == b.timestamp()


def _next_utc_day(now: datetime) -> datetime:
    day = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return day + timedelta(days=1)


# --- enqueue --------------------------------------------------------------------------------


@dataclass
class EnqueueResult:
    run: ScoutRun | None
    newer_unsettled: tuple[ScoutRun, ...] = ()
    created: list[Job] = field(default_factory=list)
    existing: list[Job] = field(default_factory=list)
    superseded: list[Job] = field(default_factory=list)
    rejected: int = 0


def enqueue(
    repo: OrchestratorRepository, evidence_db: str, now: datetime,
    policy: OrchestratorPolicy = POLICY,
) -> EnqueueResult:  # fmt: skip
    """Jobs for the admitted candidates of the newest settled Scout run (B1 admission)."""
    scan = scan_archive(evidence_db, now, policy.run_lookback_s, policy.archive_scan_limit)
    selected, newer = (
        select_run(group_runs(scan.records, now, policy)) if not scan.truncated else (None, ())
    )
    out = EnqueueResult(selected[0] if selected else None, newer)
    if selected is None:
        return out
    for r in selected[1]:
        a = evaluate_candidate(r, now=now, policy=policy, run_settled=True)
        if not a.admitted:
            out.rejected += 1
            continue
        job, created = repo.enqueue(a, now)
        (out.created if created else out.existing).append(job)
        if created:
            out.superseded += supersede_older(repo, job, now)
    return out


def supersede_older(repo: OrchestratorRepository, job: Job, now: datetime) -> list[Job]:
    """Older waiting jobs of the same token become SUPERSEDED; in-flight and terminal jobs
    are left alone."""
    out = []
    for old in repo.jobs(("QUEUED", "DEFERRED", "RETRY_WAIT"), job.canonical_id):
        if old.id != job.id and old.scout_run_time < job.scout_run_time:
            out.append(repo.transition(old, "SUPERSEDED", now, "SUPERSEDED",
                                       f"by job {job.id} (newer Scout run)"))  # fmt: skip
    return out


# --- throttle / retry -------------------------------------------------------------------------


def throttle_until(
    repo: OrchestratorRepository, canonical_id: str, now: datetime, proc: ProcessingPolicy
) -> datetime | None:
    """When the next collection for this token may start, or None if now. The minimum
    interval runs from the last *successful* collection (so retry backoff still applies to
    failures); the daily cap counts *every* collection start, failed or not."""
    starts = repo.collection_starts(canonical_id)
    successes = repo.collection_successes(canonical_id)
    waits: list[datetime] = []
    if successes:
        allowed = max(successes) + timedelta(seconds=proc.min_collection_interval_s)
        if allowed > now:
            waits.append(allowed)
    day = now.astimezone(UTC).date()
    if sum(s.astimezone(UTC).date() == day for s in starts) >= proc.max_collections_per_day:
        waits.append(_next_utc_day(now))
    return max(waits) if waits else None


def fail_attempt(
    repo: OrchestratorRepository, job: Job, category: str, note: str, now: datetime,
    proc: ProcessingPolicy,
) -> Job:  # fmt: skip
    """One failed / interrupted collection attempt: RETRY_WAIT with backoff, or FAILED."""
    n = job.attempt_count + 1
    if n >= proc.max_attempts:
        return repo.transition(job, "FAILED", now, category, note, attempt_count=n)
    wait = timedelta(seconds=proc.retry_backoff_s[n - 1])
    return repo.transition(job, "RETRY_WAIT", now, category, note, attempt_count=n,
                           next_attempt_at=now + wait)  # fmt: skip


def _defer_until(category: InfraCategory, retry_at: datetime | None, now: datetime,
                 proc: ProcessingPolicy) -> datetime:  # fmt: skip
    if retry_at is not None and retry_at > now:
        return retry_at
    if category == "SAFETY_BUDGET_EXHAUSTED":
        return _next_utc_day(now)
    return now + timedelta(seconds=proc.not_configured_wait_s)


# --- processing ------------------------------------------------------------------------------


@dataclass
class Processor:
    repo: OrchestratorRepository
    port: SafetyPort
    opportunity: OpportunityRepository
    evidence_db: str
    safety_db: str | None
    opportunity_db: str
    clock: Clock
    proc: ProcessingPolicy = PROCESSING

    def release_due(self) -> list[Job]:
        now = self.clock()
        return [self.repo.transition(j, "QUEUED", now, "RELEASED", "wait over")
                for j in self.repo.jobs(("DEFERRED", "RETRY_WAIT"))
                if j.next_attempt_at is not None and j.next_attempt_at <= now]  # fmt: skip

    def runnable(self) -> list[Job]:
        busy = {j.canonical_id for j in self.repo.jobs(("COLLECTING", "DECIDING"))}
        ready = [j for j in self.repo.jobs(("QUEUED", "SNAPSHOTTED")) if j.canonical_id not in busy]
        return sorted(ready, key=lambda j: j.priority)

    def process(self, limit: int | None = None) -> list[Job]:
        """Recover stale work, release due waits, then run the best jobs in order."""
        self.recover()
        self.release_due()
        done = []
        for job in self.runnable()[:limit]:
            done.append(self.run(self.repo.job(job.id)))
        return done

    def run(self, job: Job) -> Job:
        if job.state == "QUEUED":
            job = self._safety(job)
        if job.state == "SNAPSHOTTED":
            job = self._decide(job)
        return job

    def _safety(self, job: Job) -> Job:
        now = self.clock()
        cid = job.canonical_id
        seen = self.port.inspect(cid, now)
        if seen.reuse == "FRESH_REUSABLE" and seen.as_of is not None:
            return self.repo.transition(job, "SNAPSHOTTED", now, "SAFETY_REUSED", seen.detail,
                                        safety_snapshot_id=seen.snapshot_id,
                                        safety_as_of=seen.as_of)  # fmt: skip
        wait = throttle_until(self.repo, cid, now, self.proc)
        if wait is not None:
            return self.repo.transition(job, "DEFERRED", now, "THROTTLED",
                                        "per-token collection limit", next_attempt_at=wait)  # fmt: skip
        pre = self.port.preflight(cid, now)
        if pre.blocked is not None:
            return self.repo.transition(
                job, "DEFERRED", now, pre.blocked, f"{pre.label}: {pre.detail}",
                next_attempt_at=_defer_until(pre.blocked, pre.retry_at, now, self.proc),
            )  # fmt: skip
        job = self.repo.transition(job, "COLLECTING", now, "COLLECTION_STARTED", pre.label,
                                   lease_started_at=now)  # fmt: skip
        try:
            res = self.port.collect(cid, now)
        except Exception as exc:
            res = CollectionResult("INFRASTRUCTURE", "COLLECTION_EXCEPTION",
                                   detail=redact(f"{type(exc).__name__}: {exc}"))  # fmt: skip
        after = self.clock()
        if res.kind == "TOKEN_EVIDENCE":
            try:
                snap = self.port.snapshot(cid, after)
            except Exception as exc:
                return fail_attempt(self.repo, job, "COLLECTION_EXCEPTION",
                                    redact(f"snapshot: {type(exc).__name__}: {exc}"), after,
                                    self.proc)  # fmt: skip
            return self.repo.transition(job, "SNAPSHOTTED", after, f"TOKEN_EVIDENCE:{res.outcome}",
                                        res.detail, safety_snapshot_id=snap.snapshot_id,
                                        safety_as_of=snap.as_of)  # fmt: skip
        category = res.outcome
        if category in DEFERRING:
            return self.repo.transition(
                job, "DEFERRED", after, category, res.detail,
                next_attempt_at=_defer_until(category, res.retry_at, after, self.proc),
            )  # fmt: skip
        return fail_attempt(self.repo, job, category, res.detail, after, self.proc)

    def _decide(self, job: Job) -> Job:
        """SNAPSHOTTED -> DECIDING, storing the decision key once (exact decision time and
        Opportunity rules version, with the lease and the event, atomically), then the
        Opportunity decision under exactly that key."""
        now = self.clock()
        job = self.repo.transition(
            job, "DECIDING", now, "DECISION_STARTED", None, lease_started_at=now,
            opportunity_decision_at=now, opportunity_rules_version=current_rules_version(),
        )  # fmt: skip
        return self._record(job, "DECIDED")

    def _exact_clock(self, decision_at: datetime) -> Clock:
        """The Opportunity recorder reads its clock first for a LIVE_FORWARD decision_at:
        that read returns exactly the job's stored time. Later reads are its audit times
        (run start / finish, decided_at): the real clock, never earlier than decision_at."""
        first = [decision_at]

        def clock() -> datetime:
            return first.pop() if first else max(self.clock(), decision_at)

        return clock

    def _fail(self, job: Job, category: str, note: str) -> Job:
        return self.repo.transition(job, "FAILED", self.clock(), category, redact(note))

    def _record(self, job: Job, category: str) -> Job:
        """A real Opportunity decision for this job: LIVE_FORWARD at exactly the stored
        decision time, under the stored rules version, from exactly the job's Scout record
        and Safety snapshot (pinned readers). The exact input is built and checked *before*
        anything is persisted, and the stored decision is checked again before linking."""
        decision_at, version = job.opportunity_decision_at, job.opportunity_rules_version
        assert decision_at is not None and version is not None
        if version != current_rules_version():
            return self._fail(job, "OPPORTUNITY_RULES_VERSION_MISMATCH",
                              f"the job's decision key is rules version {version}; the running "
                              f"Opportunity is {current_rules_version()}")  # fmt: skip
        archive = PinnedArchive(self.evidence_db, job.canonical_id, job.scout_record_id,
                                job.scout_run_time)  # fmt: skip
        safety = PinnedSafety(self.safety_db, job.safety_snapshot_id)
        try:
            planned = build_input(job.canonical_id, decision_at, "LIVE_FORWARD", archive, safety)
            wrong = input_provenance_issues(job, planned)
            if wrong:  # nothing has been written
                return self._fail(job, "OPPORTUNITY_PROVENANCE_CONFLICT",
                                  "the input for this job would use other evidence: "
                                  + "; ".join(wrong))  # fmt: skip
            res = record_decision(self.opportunity, job.canonical_id, "LIVE_FORWARD", archive,
                                  safety, clock=self._exact_clock(decision_at))  # fmt: skip
        except OpportunityConflictError as exc:
            return self._fail(job, "OPPORTUNITY_CONFLICT", str(exc))
        except (OpportunityError, ValueError) as exc:
            return self._fail(job, "OPPORTUNITY_ERROR", f"{type(exc).__name__}: {exc}")
        finally:
            archive.close()
        wrong = self.provenance_mismatch(job, res.decision_id)
        if wrong:  # should be impossible with pinned readers: never link it
            return self._fail(job, "OPPORTUNITY_PROVENANCE_CONFLICT",
                              f"decision {res.decision_id} doesn't match this job: "
                              + "; ".join(wrong))  # fmt: skip
        return self.repo.transition(job, "DECIDED", self.clock(),
                                    f"{category}:{res.decision.decision}",
                                    f"decision {res.decision_id}",
                                    opportunity_decision_id=res.decision_id)  # fmt: skip

    def provenance_mismatch(self, job: Job, decision_id: int) -> list[str]:
        """Why a stored Opportunity decision isn't this job's (empty: it is). Reads only the
        decision's stored canonical input."""
        stored = self.opportunity.get(decision_id)
        out = input_provenance_issues(job, stored.input)
        if stored.decision.rules_version != job.opportunity_rules_version:
            out.append(f"rules version {stored.decision.rules_version}")
        claimed = [x.id for x in self.repo.jobs(("DECIDED",)) if x.opportunity_decision_id
                   == decision_id and x.id != job.id]  # fmt: skip
        if claimed:
            out.append(f"already linked to job {claimed[0]}")
        return out

    # --- recovery ----------------------------------------------------------------------------

    def recover(self) -> list[Job]:
        """Idempotent: stale COLLECTING -> RETRY_WAIT / FAILED (one interrupted attempt);
        stale SNAPSHOTTED -> a decision; stale DECIDING -> the exact *stored* Opportunity key
        (canonical_id, LIVE_FORWARD, stored decision_at, stored rules version): adopted only
        if its stored input is this job's Scout record and Safety snapshot (even if the
        running rules version changed since); another occupant fails the job; no decision
        yet -> recorded again under the same key, unless the running rules version differs
        (OPPORTUNITY_RULES_VERSION_MISMATCH: a new version never finishes an old attempt)."""
        now = self.clock()
        lease = timedelta(seconds=self.proc.lease_s)
        out = []
        for j in self.repo.jobs(("COLLECTING",)):
            if j.lease_started_at is not None and j.lease_started_at + lease <= now:
                out.append(fail_attempt(self.repo, j, "INTERRUPTED",
                                        "stale COLLECTING lease", now, self.proc))  # fmt: skip
        for j in self.repo.jobs(("SNAPSHOTTED",)):
            if j.updated_at + lease <= now:
                out.append(self._decide(j))
        for j in self.repo.jobs(("DECIDING",)):
            if j.lease_started_at is None or j.lease_started_at + lease > now:
                continue
            assert j.opportunity_decision_at is not None and j.opportunity_rules_version
            found = opportunity_decision_by_key(self.opportunity_db, j.canonical_id,
                                                j.opportunity_decision_at,
                                                j.opportunity_rules_version)  # fmt: skip
            if found is None:
                out.append(self._record(j, "REDECIDED_AFTER_RECOVERY"))
                continue
            wrong = self.provenance_mismatch(j, found)
            if wrong:
                out.append(self.repo.transition(
                    j, "FAILED", now, "OPPORTUNITY_PROVENANCE_CONFLICT",
                    f"decision {found} holds this job's Opportunity key with other evidence: "
                    + "; ".join(wrong),
                ))  # fmt: skip
            else:
                out.append(self.repo.transition(
                    j, "DECIDED", now, "ADOPTED", f"decision {found} (exact key, provenance "
                    "verified)", opportunity_decision_id=found,
                ))  # fmt: skip
        return out


def current_rules_version() -> str:
    """The Opportunity rules version of the running code (read at call time)."""
    return opportunity_config.OPPORTUNITY_RULES_VERSION


def input_provenance_issues(job: Job, inp: OpportunityInput) -> list[str]:
    """Why an Opportunity input isn't built from exactly this job's evidence (empty: it
    is): token, LIVE_FORWARD, the stored decision time, the job's Scout record and Safety
    snapshot (and its as_of)."""
    safety = inp.sources.safety.ref
    out = []
    if inp.canonical_id != job.canonical_id:
        out.append(f"token {inp.canonical_id}")
    if inp.origin != "LIVE_FORWARD":
        out.append(f"origin {inp.origin}")
    if not _same_time(inp.decision_at, job.opportunity_decision_at):
        out.append(f"decision_at {inp.decision_at.isoformat()}")
    if inp.sources.scout.ref.record_id != job.scout_record_id:
        out.append(f"Scout record {inp.sources.scout.ref.record_id} "
                   f"({inp.sources.scout.ref.status})")  # fmt: skip
    if safety.snapshot_id != job.safety_snapshot_id:
        out.append(f"Safety snapshot {safety.snapshot_id} ({safety.status})")
    if None not in (safety.as_of, job.safety_as_of) and not _same_time(
        safety.as_of, job.safety_as_of
    ):
        out.append(f"Safety as_of {safety.as_of}")
    return out
