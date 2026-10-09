"""Opportunity orchestration bridge, B2: persistent jobs, the state machine, a fake Safety
port and offline processing into real Opportunity decisions. Offline: fixture stores only;
the fake port writes Safety-format snapshots into a *test* Safety file, never real Safety."""

import io
import json
import socket
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.opportunity_model_fakes import Archive, rule, scout_record
from tests.test_opportunity_model_decision import _clean_safety
from tests.test_opportunity_orchestrator_b1 import MINTS, NOW, T, cand, safety_store
from upscale.services.opportunity_model.recorder import record_from_paths
from upscale.services.opportunity_model.repository import (
    OpportunityConflictError,
    OpportunityRepository,
)
from upscale.services.opportunity_orchestrator import processing
from upscale.services.opportunity_orchestrator.cli import LIVE_SAFETY_NOT_IMPLEMENTED, main
from upscale.services.opportunity_orchestrator.config import PROCESSING
from upscale.services.opportunity_orchestrator.processing import (
    Processor,
    enqueue,
    throttle_until,
)
from upscale.services.opportunity_orchestrator.repository import (
    STATES,
    TERMINAL,
    TRANSITIONS,
    Job,
    OrchestratorRepository,
    OrchestratorStorageError,
)
from upscale.services.opportunity_orchestrator.safety_port import (
    CollectionResult,
    Preflight,
    ReadOnlySafetyPort,
    SafetyInspection,
    SnapshotRef,
)
from upscale.services.safety_v2.repository import encode_body

CID = {t: f"solana:{m}" for t, m in MINTS.items()}


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw: float) -> datetime:
        self.t += timedelta(**kw)
        return self.t


def _body(outcome: str, cid: str, as_of: datetime) -> dict[str, Any]:
    body = json.loads(json.dumps(_clean_safety(as_of)))
    body["identity"] |= {"canonical_id": cid, "mint": cid.split(":")[1]}
    flags = {f["id"]: f for f in body["flags"]}
    if outcome == "HOLDERS_PARTIAL":
        body["holders"]["status"] = "PARTIAL"
        flags["FEW_HOLDERS"] |= {"outcome": "UNDETERMINED"}
        body["undetermined"] = [{"id": "FEW_HOLDERS", "needs": "a complete scan", "reason": "x"}]
        body["coverage"]["coverage"] = "PARTIAL"
    elif outcome in ("NOT_A_MINT", "ACCOUNT_MISSING"):
        flags["NOT_A_TOKEN_MINT"] |= rule("NOT_A_TOKEN_MINT", "TRIGGERED", "critical")
    elif outcome == "NO_POOLS":
        flags["NO_ELIGIBLE_MARKET"] |= rule("NO_ELIGIBLE_MARKET", "TRIGGERED", "high")
    elif outcome == "MARKET_CLOSED":
        flags["MARKET_CLOSED_ON_CHAIN"] |= rule("MARKET_CLOSED_ON_CHAIN", "TRIGGERED", "high")
    body["flags"] = list(flags.values())
    return body


class FakeSafety:
    """A scripted Safety port. ``snapshot`` writes a Safety-format snapshot into the *test*
    Safety file so the real Opportunity loaders can read it."""

    def __init__(self, safety_db: str, script: dict[str, list[Any]] | None = None,
                 preflight: Preflight | None = None, reuse: str | None = None):  # fmt: skip
        self.path, self.script = safety_db, script or {}
        self.pre = preflight or Preflight("CONSERVATIVE_PREFLIGHT_OK")
        self.reuse = reuse
        self.calls: list[tuple[str, str]] = []
        self.last: dict[str, str] = {}

    def inspect(self, cid: str, now: datetime) -> SafetyInspection:
        seen = ReadOnlySafetyPort(self.path).inspect(cid, now)
        return SafetyInspection(self.reuse) if self.reuse else seen  # type: ignore[arg-type]

    def preflight(self, cid: str, now: datetime) -> Preflight:
        return self.pre

    def collect(self, cid: str, now: datetime) -> CollectionResult:
        self.calls.append(("collect", cid))
        item = self.script[cid].pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, CollectionResult):
            self.last[cid] = item.outcome
            return item
        self.last[cid] = item
        return CollectionResult("TOKEN_EVIDENCE", item)

    def snapshot(self, cid: str, now: datetime) -> SnapshotRef:
        self.calls.append(("snapshot", cid))
        _, blob, digest = encode_body(_body(self.last[cid], cid, now))
        c = sqlite3.connect(self.path)
        cur = c.execute(
            "INSERT INTO safety_snapshots (canonical_id, as_of, schema_version, rules_version, "
            "fingerprints_json, coverage, band, body_zlib, body_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (cid, now.timestamp(), "4", "4", "{}", "COMPLETE", "-", blob, digest),
        )
        c.commit()
        c.close()
        assert cur.lastrowid is not None
        return SnapshotRef(cur.lastrowid, now)


def infra(category: str, retry_at: datetime | None = None) -> CollectionResult:
    return CollectionResult("INFRASTRUCTURE", category, retry_at)  # type: ignore[arg-type]


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    arch = Archive(tmp_path / "evidence.sqlite3")
    for tag, over in (("TokA", {"rank": 1}), ("TokB", {"rank": 2}), ("TokC", {"rank": 3}),
                      ("TokD", {"stage": "FADING"})):  # fmt: skip
        arch.add(scout_record(T, cand=cand(tag, **over), asset_id=CID[tag]))
    arch.close()
    safety = safety_store(tmp_path / "safety.sqlite3", [])
    return {"tmp": tmp_path, "evidence": str(arch.path), "safety": str(safety),
            "opportunity": str(tmp_path / "opportunity.sqlite3"),
            "orch": str(tmp_path / "orchestrator.sqlite3")}  # fmt: skip


@pytest.fixture
def env(world: dict[str, Any]) -> Any:
    repo = OrchestratorRepository(world["orch"])
    opp = OpportunityRepository(world["opportunity"])
    clock = Clock(NOW)
    yield world, repo, opp, clock
    repo.close()
    opp.close()


def processor(env: Any, port: Any) -> Processor:
    world, repo, opp, clock = env
    return Processor(repo, port, opp, world["evidence"], world["safety"], world["opportunity"],
                     clock)  # fmt: skip


def decisions(world: dict[str, Any]) -> int:
    c = sqlite3.connect(world["opportunity"])
    try:
        return int(c.execute("SELECT COUNT(*) FROM opportunity_decisions").fetchone()[0])
    finally:
        c.close()


def job_for(repo: OrchestratorRepository, tag: str) -> Job:
    (j,) = [j for j in repo.jobs() if j.canonical_id == CID[tag] and j.state not in TERMINAL] or [
        sorted((j for j in repo.jobs() if j.canonical_id == CID[tag]), key=lambda j: j.id)[-1]]  # fmt: skip
    return j


# --- schema ---------------------------------------------------------------------------------------


def test_schema_and_foreign_database_refusal(tmp_path: Path) -> None:
    path = tmp_path / "o.sqlite3"
    r = OrchestratorRepository(path)
    assert dict(r._conn.execute("SELECT key, value FROM orch_meta").fetchall()) == {
        "component": "opportunity_orchestrator", "schema_version": "1"}  # fmt: skip
    r.close()
    c = sqlite3.connect(path)
    c.execute("UPDATE orch_meta SET value = '2' WHERE key = 'schema_version'")
    c.commit()
    c.close()
    safety = safety_store(tmp_path / "safety.sqlite3", [])
    for p in (path, safety):
        before = (p.read_bytes(), p.stat().st_mtime_ns)
        with pytest.raises(OrchestratorStorageError):
            OrchestratorRepository(p)
        assert (p.read_bytes(), p.stat().st_mtime_ns) == before
    with pytest.raises(OrchestratorStorageError):
        OrchestratorRepository(tmp_path / "none.sqlite3", read_only=True)
    assert not (tmp_path / "none.sqlite3").exists()


# --- enqueue / supersession ---------------------------------------------------------------------


def test_enqueue_creates_one_job_per_admitted_record_and_is_idempotent(env: Any) -> None:
    world, repo, _, clock = env
    first = enqueue(repo, world["evidence"], clock())
    assert len(first.created) == 3 and first.rejected == 1 and not first.existing
    again = enqueue(repo, world["evidence"], clock())
    assert not again.created and len(again.existing) == 3
    assert len(repo.jobs()) == 3
    for j in repo.jobs():
        assert j.state == "QUEUED" and [e["to"] for e in repo.events(j.id)] == ["QUEUED"]


def test_a_newer_run_supersedes_waiting_jobs_only(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    a, b = job_for(repo, "TokA"), job_for(repo, "TokB")
    repo.transition(b, "COLLECTING", clock(), lease_started_at=clock())  # in flight
    t2 = T + timedelta(minutes=10)
    arch = Archive(Path(world["evidence"]))
    for tag in ("TokA", "TokB"):
        arch.add(scout_record(t2, cand=cand(tag, t2), asset_id=CID[tag]))
    arch.close()
    res = enqueue(repo, world["evidence"], clock.advance(minutes=12))
    assert {j.canonical_id for j in res.created} == {CID["TokA"], CID["TokB"]}
    assert [j.id for j in res.superseded] == [a.id]
    assert repo.job(a.id).state == "SUPERSEDED" and repo.job(b.id).state == "COLLECTING"


# --- state machine -----------------------------------------------------------------------------


def _fields(to: str, now: datetime) -> dict[str, Any]:
    return {"COLLECTING": {"lease_started_at": now},
            "DECIDING": {"lease_started_at": now, "opportunity_decision_at": now,
                         "opportunity_rules_version": "1"},
            "SNAPSHOTTED": {"safety_as_of": now}, "DEFERRED": {"next_attempt_at": now},
            "RETRY_WAIT": {"next_attempt_at": now},
            "DECIDED": {"opportunity_decision_id": 1}}.get(to, {})  # fmt: skip


def _job_in(repo: OrchestratorRepository, state: str, now: datetime) -> Job:
    """A fresh job driven into `state` through legal transitions only."""
    from tests.test_opportunity_orchestrator_b1 import record
    from upscale.services.opportunity_orchestrator.admission import evaluate_candidate

    n = len(repo.jobs())  # a distinct Scout record per job
    job, _ = repo.enqueue(
        evaluate_candidate(record("TokA", observed=T - timedelta(seconds=n)), now=NOW), now
    )
    path = {"QUEUED": [], "COLLECTING": ["COLLECTING"], "SNAPSHOTTED": ["SNAPSHOTTED"],
            "DECIDING": ["SNAPSHOTTED", "DECIDING"], "DEFERRED": ["DEFERRED"],
            "RETRY_WAIT": ["COLLECTING", "RETRY_WAIT"]}[state]  # fmt: skip
    for step in path:
        job = repo.transition(job, step, now, **_fields(step, now))  # type: ignore[arg-type]
    return job


def test_every_legal_transition_works_and_every_illegal_one_is_refused(env: Any) -> None:
    _, repo, _, clock = env
    now = clock()
    for src in ("QUEUED", "COLLECTING", "SNAPSHOTTED", "DECIDING", "DEFERRED", "RETRY_WAIT"):
        for dst in STATES:
            job = _job_in(repo, src, now)
            events = len(repo.events(job.id))
            if (src, dst) in TRANSITIONS:
                moved = repo.transition(job, dst, now, "T", **_fields(dst, now))  # type: ignore[arg-type]
                assert moved.state == dst and len(repo.events(job.id)) == events + 1
            else:
                with pytest.raises(sqlite3.DatabaseError):
                    repo.transition(job, dst, now, "T", **_fields(dst, now))  # type: ignore[arg-type]
                assert repo.job(job.id).state == src and len(repo.events(job.id)) == events


def test_terminal_jobs_immutable_fields_and_events_are_protected(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    j = repo.transition(job_for(repo, "TokA"), "SUPERSEDED", clock(), "T")
    db = repo._conn
    for sql in (f"UPDATE orch_jobs SET note = 'x' WHERE id = {j.id}",
                f"UPDATE orch_jobs SET rank = 9 WHERE id = {job_for(repo, 'TokB').id}",
                f"UPDATE orch_jobs SET stage = 'EARLY' WHERE id = {job_for(repo, 'TokB').id}",
                "DELETE FROM orch_jobs", "UPDATE orch_events SET note = 'x'",
                "DELETE FROM orch_events"):  # fmt: skip
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(sql)
    # A state change made any way writes its event in the same statement.
    b = job_for(repo, "TokB")
    db.execute("UPDATE orch_jobs SET state = 'DEFERRED', next_attempt_at = 1.0, "
               f"updated_at = 2.0, category = 'RAW' WHERE id = {b.id}")  # fmt: skip
    assert repo.events(b.id)[-1]["to"] == "DEFERRED" and repo.events(b.id)[-1]["category"] == "RAW"
    with pytest.raises(OrchestratorStorageError):  # stale read: the job moved meanwhile
        repo.transition(b, "COLLECTING", clock(), lease_started_at=clock())


# --- processing: token evidence --------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["COMPLETE", "HOLDERS_PARTIAL", "NOT_A_MINT",
                                     "ACCOUNT_MISSING", "NO_POOLS", "MARKET_CLOSED"])  # fmt: skip
def test_token_evidence_reaches_a_real_live_opportunity_decision(env: Any, outcome: str) -> None:
    world, repo, opp, clock = env
    enqueue(repo, world["evidence"], clock())
    port = FakeSafety(world["safety"], {CID["TokA"]: [outcome]})
    proc = processor(env, port)
    clock.advance(seconds=30)
    job = proc.run(job_for(repo, "TokA"))
    assert job.state == "DECIDED" and job.opportunity_decision_id is not None
    assert [e["to"] for e in repo.events(job.id)] == [
        "QUEUED", "COLLECTING", "SNAPSHOTTED", "DECIDING", "DECIDED"]  # fmt: skip
    stored = opp.get(job.opportunity_decision_id)
    assert stored.origin == "LIVE_FORWARD" and stored.decision_at == clock()
    assert stored.decision_at > job.scout_run_time  # never backdated to the Scout time
    assert job.safety_as_of is not None and job.safety_as_of <= stored.decision_at
    if outcome in ("NOT_A_MINT", "ACCOUNT_MISSING", "NO_POOLS", "MARKET_CLOSED"):
        assert stored.decision.decision == "SKIP" and stored.decision.vetoes
    if outcome == "HOLDERS_PARTIAL":
        assert stored.decision.safety_entry_readiness["HOLDERS"] == "READY_PARTIAL"


def test_fresh_reusable_safety_skips_collection(env: Any) -> None:
    world, repo, opp, clock = env
    safety_store(world["tmp"] / "s2.sqlite3", [])
    c = sqlite3.connect(world["safety"])
    _, blob, digest = encode_body(_body("COMPLETE", CID["TokA"], T))
    c.execute("INSERT INTO safety_snapshots (canonical_id, as_of, schema_version, rules_version, "
              "fingerprints_json, coverage, band, body_zlib, body_hash) VALUES (?,?,?,?,?,?,?,?,?)",
              (CID["TokA"], T.timestamp(), "4", "4", "{}", "COMPLETE", "-", blob, digest))  # fmt: skip
    c.commit()
    c.close()
    enqueue(repo, world["evidence"], clock())
    port = FakeSafety(world["safety"])
    job = processor(env, port).run(job_for(repo, "TokA"))
    assert job.state == "DECIDED" and port.calls == []
    assert [e["category"] for e in repo.events(job.id)][1] == "SAFETY_REUSED"


# --- processing: infrastructure -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "state", "attempts", "wait"),
    [
        (infra("PROVIDER_UNAVAILABLE"), "RETRY_WAIT", 1, timedelta(minutes=10)),
        (infra("PROVIDER_TIMEOUT"), "RETRY_WAIT", 1, timedelta(minutes=10)),
        (infra("DATABASE_LOCK"), "RETRY_WAIT", 1, timedelta(minutes=10)),
        (RuntimeError("boom"), "RETRY_WAIT", 1, timedelta(minutes=10)),
        # Reported by a *started* collection: one attempt each (pre-collection blocks are
        # the preflight's, tested below). Budget: the next UTC day; cooldown 7 min < 10 min.
        (infra("SAFETY_BUDGET_EXHAUSTED"), "RETRY_WAIT", 1, None),
        (
            infra("SAFETY_COOLDOWN", NOW + timedelta(minutes=7)),
            "RETRY_WAIT",
            1,
            timedelta(minutes=10),
        ),
        (
            infra("PROVIDER_RATE_LIMITED", NOW + timedelta(minutes=45)),
            "RETRY_WAIT",
            1,
            timedelta(minutes=45),
        ),
        (infra("PROVIDER_NOT_CONFIGURED"), "RETRY_WAIT", 1, timedelta(minutes=10)),
    ],
)
def test_infrastructure_never_creates_an_opportunity_decision(
    env: Any, result: Any, state: str, attempts: int, wait: timedelta | None
) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    job = processor(env, FakeSafety(world["safety"], {CID["TokA"]: [result]})).run(
        job_for(repo, "TokA")
    )
    assert (job.state, job.attempt_count) == (state, attempts)
    expected = NOW + wait if wait else datetime(2026, 10, 9, tzinfo=NOW.tzinfo)  # next UTC day
    assert job.next_attempt_at == expected
    assert decisions(world) == 0 and job.opportunity_decision_id is None


@pytest.mark.parametrize("blocked", ["SAFETY_BUDGET_EXHAUSTED", "SAFETY_COOLDOWN",
                                     "PROVIDER_NOT_CONFIGURED"])  # fmt: skip
def test_a_blocked_preflight_defers_without_starting_a_collection(env: Any, blocked: str) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    pre = Preflight("CONSERVATIVE_PREFLIGHT_INSUFFICIENT", blocked)  # type: ignore[arg-type]
    port = FakeSafety(world["safety"], preflight=pre)
    job = processor(env, port).run(job_for(repo, "TokA"))
    assert job.state == "DEFERRED" and job.attempt_count == 0 and port.calls == []
    assert "COLLECTING" not in [e["to"] for e in repo.events(job.id)]
    assert decisions(world) == 0


def test_retries_back_off_exactly_then_fail(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    script = {CID["TokA"]: [infra("PROVIDER_UNAVAILABLE")] * 3}
    proc = processor(env, FakeSafety(world["safety"], script))
    j = proc.run(job_for(repo, "TokA"))
    assert (j.state, j.attempt_count, j.next_attempt_at) == (
        "RETRY_WAIT",
        1,
        NOW + timedelta(minutes=10),
    )
    clock.advance(minutes=10, seconds=-1)
    assert proc.release_due() == []
    clock.advance(seconds=1)
    proc.process()
    j = repo.job(j.id)
    assert (j.state, j.attempt_count) == ("RETRY_WAIT", 2)
    assert j.next_attempt_at == clock() + timedelta(minutes=30)
    clock.advance(minutes=30)
    proc.process()
    j = repo.job(j.id)
    assert (j.state, j.attempt_count, j.category) == ("FAILED", 3, "PROVIDER_UNAVAILABLE")
    assert decisions(world) == 0
    assert PROCESSING.retry_backoff_s == (600, 1800)  # no unreachable third wait


# --- per-token throttle -------------------------------------------------------------------------


def test_per_token_interval_after_success_and_daily_cap(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]}))
    done = proc.run(job_for(repo, "TokA"))
    success_at = repo.events(done.id)[2]["at"]
    assert throttle_until(repo, CID["TokA"], success_at + timedelta(minutes=59), PROCESSING) == (
        success_at + timedelta(minutes=60))  # fmt: skip
    assert throttle_until(repo, CID["TokA"], success_at + timedelta(minutes=60), PROCESSING) is None
    # Failed starts don't block a retry, but six starts in a UTC day do.
    for _ in range(5):
        j = _job_in(repo, "COLLECTING", clock())
        repo.transition(j, "RETRY_WAIT", clock(), next_attempt_at=clock(), attempt_count=1)
    later = success_at + timedelta(hours=2)
    assert throttle_until(repo, CID["TokA"], later, PROCESSING) == datetime(
        2026, 10, 9, tzinfo=NOW.tzinfo)  # fmt: skip


def test_a_throttled_job_is_deferred_without_a_collection(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]})).run(
        job_for(repo, "TokA")
    )
    t2 = T + timedelta(minutes=20)
    arch = Archive(Path(world["evidence"]))
    arch.add(scout_record(t2, cand=cand("TokA", t2), asset_id=CID["TokA"]))
    arch.close()
    clock.advance(minutes=21)
    enqueue(repo, world["evidence"], clock())
    port = FakeSafety(world["safety"], reuse="NOT_READY")  # pretend reuse isn't possible
    j = processor(env, port).run(job_for(repo, "TokA"))
    assert (j.state, j.category, port.calls) == ("DEFERRED", "THROTTLED", [])


# --- Opportunity failures, recovery ------------------------------------------------------------


def test_an_opportunity_conflict_fails_the_job(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())

    def conflict(*a: Any, **k: Any) -> Any:
        raise OpportunityConflictError("decision 1 already exists with a different input")

    monkeypatch.setattr(processing, "record_decision", conflict)
    j = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]})).run(
        job_for(repo, "TokA")
    )
    assert (j.state, j.category) == ("FAILED", "OPPORTUNITY_CONFLICT") and decisions(world) == 0


def test_a_stale_collection_counts_one_interrupted_attempt_idempotently(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: [KeyboardInterrupt()]}))
    with pytest.raises(KeyboardInterrupt):
        proc.run(job_for(repo, "TokA"))
    j = job_for(repo, "TokA")
    assert j.state == "COLLECTING"
    clock.advance(minutes=15, seconds=-1)
    assert proc.recover() == [] and repo.job(j.id).state == "COLLECTING"
    clock.advance(seconds=1)
    (r,) = proc.recover()
    assert (r.state, r.attempt_count, r.category) == ("RETRY_WAIT", 1, "INTERRUPTED")
    assert proc.recover() == [] and repo.job(j.id).attempt_count == 1
    assert decisions(world) == 0


def test_a_stale_snapshotted_job_resumes_to_a_decision(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    port = FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]})
    proc = processor(env, port)
    j = proc._safety(job_for(repo, "TokA"))  # crash right after the snapshot
    assert j.state == "SNAPSHOTTED"
    clock.advance(minutes=15)
    (r,) = proc.recover()
    assert r.state == "DECIDED" and decisions(world) == 1
    assert proc.recover() == []


def _crash_in_decision(monkeypatch: pytest.MonkeyPatch, after_recording: bool) -> None:
    """Crash inside DECIDING: after (or before) the Opportunity commit, before the job links
    the decision id."""
    real = processing.record_decision

    def crash(*a: Any, **k: Any) -> Any:
        if after_recording:
            real(*a, **k)
        raise KeyboardInterrupt

    monkeypatch.setattr(processing, "record_decision", crash)


def _crashed(
    env: Any, monkeypatch: pytest.MonkeyPatch, after_recording: bool
) -> tuple[Processor, Job]:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]}))
    clock.advance(seconds=30)
    _crash_in_decision(monkeypatch, after_recording)
    with pytest.raises(KeyboardInterrupt):
        proc.run(job_for(repo, "TokA"))
    monkeypatch.undo()
    job = job_for(repo, "TokA")
    assert job.state == "DECIDING" and job.opportunity_decision_id is None
    return proc, job


def _manual_decision(env: Any, at: datetime) -> int:
    world, _, opp, _ = env
    return record_from_paths(opp, CID["TokA"], "LIVE_FORWARD", world["evidence"],
                             world["safety"], clock=lambda: at).decision_id  # fmt: skip


def test_a_b_the_decision_time_is_stored_once_and_used_exactly(env: Any) -> None:
    world, repo, opp, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]}))
    job = proc._safety(job_for(repo, "TokA"))
    clock.advance(seconds=7)
    deciding_at = clock()
    done = proc._decide(job)
    assert done.opportunity_decision_at == deciding_at
    deciding = [e for e in repo.events(done.id) if e["to"] == "DECIDING"]
    assert len(deciding) == 1 and deciding[0]["at"] == deciding_at
    stored = opp.get(done.opportunity_decision_id)
    assert stored.decision_at == deciding_at and stored.decision.decision_at == deciding_at
    assert stored.decided_at >= stored.decision_at  # audit time from the real clock


def test_c_g_a_crash_after_the_commit_adopts_the_exact_verified_decision(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, _, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=True)
    assert decisions(world) == 1
    clock.advance(minutes=15)
    (r,) = proc.recover()
    assert (r.state, r.category, r.opportunity_decision_id) == ("DECIDED", "ADOPTED", 1)
    assert r.opportunity_decision_at == job.opportunity_decision_at
    assert decisions(world) == 1
    assert proc.recover() == [] and repo.job(job.id).state == "DECIDED"  # I: idempotent


def test_d_h_a_nearby_decision_is_ignored_and_the_same_time_is_reused(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, opp, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=False)
    assert job.opportunity_decision_at is not None
    nearby = _manual_decision(env, job.opportunity_decision_at + timedelta(minutes=1))
    clock.advance(minutes=15)
    (r,) = proc.recover()
    assert r.state == "DECIDED" and r.category.startswith("REDECIDED_AFTER_RECOVERY")
    assert r.opportunity_decision_id not in (None, nearby)
    assert opp.get(r.opportunity_decision_id).decision_at == job.opportunity_decision_at
    assert proc.recover() == []  # I: idempotent


def test_e_an_exact_time_decision_from_another_scout_record_is_not_adopted(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, _, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=False)
    at = job.opportunity_decision_at
    assert at is not None
    newer = T + timedelta(minutes=1)  # a newer Scout record, readable before `at`
    arch = Archive(Path(world["evidence"]))
    arch.add(scout_record(newer, cand=cand("TokA", newer), asset_id=CID["TokA"]), archived=newer)
    arch.close()
    occupant = _manual_decision(env, at)
    clock.advance(minutes=15)
    (r,) = proc.recover()
    assert (r.state, r.category) == ("FAILED", "OPPORTUNITY_PROVENANCE_CONFLICT")
    assert "Scout record" in r.note and r.opportunity_decision_id is None
    assert decisions(world) == 1 and occupant == 1  # nothing overwritten or added


def test_f_an_exact_time_decision_from_another_safety_snapshot_is_not_adopted(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, _, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=False)
    at = job.opportunity_decision_at
    assert at is not None
    other = FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]})
    other.collect(CID["TokA"], at)
    other.snapshot(CID["TokA"], at)  # a newer snapshot row, readable at `at`
    _manual_decision(env, at)
    clock.advance(minutes=15)
    (r,) = proc.recover()
    assert (r.state, r.category) == ("FAILED", "OPPORTUNITY_PROVENANCE_CONFLICT")
    assert "Safety snapshot" in r.note and decisions(world) == 1


def test_j_the_decision_time_can_never_change(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]}))
    done = proc.run(job_for(repo, "TokA"))
    queued = job_for(repo, "TokB")
    for sql in (
        f"UPDATE orch_jobs SET opportunity_decision_at = 1.0 WHERE id = {done.id}",
        f"UPDATE orch_jobs SET state = 'DEFERRED', next_attempt_at = 1.0, "
        f"opportunity_decision_at = 1.0 WHERE id = {queued.id}",  # only entering DECIDING
    ):
        with pytest.raises(sqlite3.DatabaseError):
            repo._conn.execute(sql)
    deciding = _job_in(repo, "DECIDING", clock())
    with pytest.raises(sqlite3.DatabaseError):
        repo.transition(deciding, "FAILED", clock(), opportunity_decision_at=clock() + timedelta(1))


def test_k_a_conflict_on_the_exact_key_fails_the_job(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, _, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=False)

    def conflict(*a: Any, **k: Any) -> Any:
        raise OpportunityConflictError("decision exists for this key with a different input")

    monkeypatch.setattr(processing, "record_decision", conflict)
    clock.advance(minutes=15)
    (r,) = proc.recover()
    assert (r.state, r.category) == ("FAILED", "OPPORTUNITY_CONFLICT") and decisions(world) == 0


def test_recovery_reads_only_the_exact_key(env: Any) -> None:
    src = (Path(processing.__file__).read_text()
           + Path(processing.__file__).with_name("readers.py").read_text())  # fmt: skip
    assert "opportunity_live_decisions" not in src
    assert "decision_at = ?" in src  # an exact match, no window


# --- CLI / isolation ------------------------------------------------------------------------------


def cli(world: dict[str, Any], *argv: str, at: datetime = NOW) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["--evidence-db", world["evidence"], "--safety-db", world["safety"],
                 "--opportunity-db", world["opportunity"], "--orchestrator-db", world["orch"],
                 *argv], out=out, clock=lambda: at)  # fmt: skip
    return code, out.getvalue()


def test_cli_offline_commands(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSCALE_HELIUS_API_KEY", "TESTSECRET-b2")
    outs = []
    for argv in (["scan", "--dry-run"], ["status"], ["jobs"]):
        code, out = cli(world, *argv)
        assert code == 0
        outs.append(out)
    assert not Path(world["orch"]).exists() and not Path(world["opportunity"]).exists()
    code, out = cli(world, "enqueue")
    assert code == 0 and "created 3" in out
    code, out = cli(world, "process", at=NOW + timedelta(seconds=10))
    assert code == 0 and LIVE_SAFETY_NOT_IMPLEMENTED in out and "3 job(s) processed" in out
    assert out.count("DEFERRED [PROVIDER_NOT_CONFIGURED]") == 3
    assert decisions(world) == 0  # nothing reusable: no decision
    code, jobs = cli(world, "jobs", "--state", "DEFERRED")
    assert code == 0 and jobs.count("DEFERRED") == 3
    code, rec = cli(world, "recover")
    assert code == 0 and "0 job(s) recovered" in rec
    code, status = cli(world, "status")
    assert '"DEFERRED": 3' in status
    outs += [out, jobs, rec, status]
    assert (
        "TESTSECRET" not in "".join(outs) and b"TESTSECRET" not in Path(world["orch"]).read_bytes()
    )
    assert cli(world, "scan")[0] == 2
    refused = io.StringIO()  # B3: --live exists but needs an explicit spending ceiling
    assert main(["--orchestrator-db", world["orch"], "process", "--live"], out=refused) == 2
    assert "--max-requests" in refused.getvalue()


def test_offline_process_decides_from_existing_fresh_safety(world: dict[str, Any]) -> None:
    c = sqlite3.connect(world["safety"])
    _, blob, digest = encode_body(_body("COMPLETE", CID["TokB"], T))
    c.execute("INSERT INTO safety_snapshots (canonical_id, as_of, schema_version, rules_version, "
              "fingerprints_json, coverage, band, body_zlib, body_hash) VALUES (?,?,?,?,?,?,?,?,?)",
              (CID["TokB"], T.timestamp(), "4", "4", "{}", "COMPLETE", "-", blob, digest))  # fmt: skip
    c.commit()
    c.close()
    safety_before = Path(world["safety"]).read_bytes()
    cli(world, "enqueue")
    code, out = cli(world, "process", at=NOW + timedelta(seconds=10))
    assert code == 0 and out.count(" DECIDED [") == 1 and decisions(world) == 1
    assert Path(world["safety"]).read_bytes() == safety_before  # Safety is only read


def test_processing_makes_no_network_calls(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    script = {CID[t]: ["COMPLETE"] for t in ("TokA", "TokB", "TokC")}
    done = processor(env, FakeSafety(world["safety"], script)).process()
    assert [j.state for j in done] == ["DECIDED"] * 3
    assert [j.canonical_id for j in done] == [CID["TokA"], CID["TokB"], CID["TokC"]]  # priority


# --- pinned decision key and provenance (pre-commit hardening) --------------------------------


def _stored_input(env: Any, decision_id: int) -> Any:
    return env[2].get(decision_id).input


def _pinned(job: Job, inp: Any) -> None:
    assert inp.sources.scout.ref.record_id == job.scout_record_id
    assert inp.sources.safety.ref.snapshot_id == job.safety_snapshot_id
    assert inp.sources.safety.ref.as_of == job.safety_as_of
    assert inp.decision_at == job.opportunity_decision_at and inp.origin == "LIVE_FORWARD"


def test_rules_version_is_stored_once_with_the_decision_time(env: Any) -> None:
    world, repo, opp, clock = env
    enqueue(repo, world["evidence"], clock())
    done = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]})).run(
        job_for(repo, "TokA"))  # fmt: skip
    assert done.opportunity_rules_version == "1"
    assert opp.get(done.opportunity_decision_id).decision.rules_version == "1"
    assert [e["to"] for e in repo.events(done.id)].count("DECIDING") == 1
    _pinned(done, _stored_input(env, done.opportunity_decision_id))  # I: pinned input
    for sql in (f"UPDATE orch_jobs SET opportunity_rules_version = '2' WHERE id = {done.id}",
                f"UPDATE orch_jobs SET state = 'DEFERRED', next_attempt_at = 1.0, "
                f"opportunity_rules_version = '1' WHERE id = {job_for(repo, 'TokB').id}"):  # fmt: skip
        with pytest.raises(sqlite3.DatabaseError):  # B: immutable, set only entering DECIDING
            repo._conn.execute(sql)
    deciding = _job_in(repo, "DECIDING", clock())
    assert deciding.opportunity_rules_version == "1"
    with pytest.raises(sqlite3.DatabaseError):  # never changed once set
        repo.transition(deciding, "FAILED", clock(), opportunity_rules_version="2")


def test_recovery_adopts_under_the_stored_version_after_a_runtime_change(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, _, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=True)
    monkeypatch.setattr(processing, "current_rules_version", lambda: "2")
    clock.advance(minutes=15)
    (r,) = proc.recover()  # C, D: exact lookup under the stored "1"
    assert (r.state, r.category, r.opportunity_decision_id) == ("DECIDED", "ADOPTED", 1)
    assert r.opportunity_rules_version == "1" and decisions(world) == 1


def test_no_decision_and_a_new_runtime_version_fails_explicitly(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, _, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=False)
    monkeypatch.setattr(processing, "current_rules_version", lambda: "2")
    clock.advance(minutes=15)
    (r,) = proc.recover()  # E
    assert (r.state, r.category) == ("FAILED", "OPPORTUNITY_RULES_VERSION_MISMATCH")
    assert decisions(world) == 0 and proc.recover() == []


def _late_safety(world: dict[str, Any], at: datetime) -> int:
    """Snapshot B: newer than the job's A and readable at `at`."""
    other = FakeSafety(world["safety"], {CID["TokA"]: ["NO_POOLS"]})
    other.collect(CID["TokA"], at)
    return other.snapshot(CID["TokA"], at).snapshot_id


def test_a_late_safety_snapshot_never_replaces_the_jobs_snapshot(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]}))
    job = proc._safety(job_for(repo, "TokA"))  # snapshot A
    clock.advance(seconds=20)
    b = _late_safety(world, clock() - timedelta(seconds=5))  # B: latest at decision time
    done = proc._decide(job)
    assert done.state == "DECIDED" and b != job.safety_snapshot_id
    inp = _stored_input(env, done.opportunity_decision_id)
    _pinned(done, inp)  # F: A, never B (B would have vetoed: NO_POOLS)
    assert inp.sources.safety.ref.snapshot_id != b


def test_a_late_safety_snapshot_before_recovery_is_never_used(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, repo, _, clock = env
    proc, job = _crashed(env, monkeypatch, after_recording=False)
    assert job.opportunity_decision_at is not None
    b = _late_safety(world, job.opportunity_decision_at)
    clock.advance(minutes=15)
    (r,) = proc.recover()
    assert r.state == "DECIDED"
    inp = _stored_input(env, r.opportunity_decision_id)
    _pinned(r, inp)
    assert inp.sources.safety.ref.snapshot_id != b


def test_a_competing_scout_record_never_replaces_the_jobs_record(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    job = job_for(repo, "TokA")
    newer = T + timedelta(minutes=1)  # readable before the decision (archived in time)
    arch = Archive(Path(world["evidence"]))
    arch.add(scout_record(newer, cand=cand("TokA", newer, stage="FADING"), asset_id=CID["TokA"]),
             archived=newer)  # fmt: skip
    arch.close()
    done = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]})).run(job)
    assert done.state == "DECIDED"
    inp = _stored_input(env, done.opportunity_decision_id)
    _pinned(done, inp)  # G: the job's record, not the newer FADING one
    stage = next(f for f in inp.sources.scout.context if f.aspect == "scout.stage")
    assert stage.value["stage"] == "ACCELERATING"


def test_a_provenance_mismatch_before_recording_writes_nothing(env: Any) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]}))
    job = proc._safety(job_for(repo, "TokA"))
    c = sqlite3.connect(world["safety"])  # the job's snapshot vanishes (test fixture only)
    c.execute("DELETE FROM safety_snapshots WHERE id = ?", (job.safety_snapshot_id,))
    c.commit()
    c.close()
    r = proc._decide(job)  # H
    assert (r.state, r.category) == ("FAILED", "OPPORTUNITY_PROVENANCE_CONFLICT")
    assert "Safety snapshot None" in r.note and decisions(world) == 0
    assert r.opportunity_decision_id is None


def test_a_post_record_mismatch_is_never_linked(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    world, repo, _, clock = env
    enqueue(repo, world["evidence"], clock())
    proc = processor(env, FakeSafety(world["safety"], {CID["TokA"]: ["COMPLETE"]}))
    monkeypatch.setattr(Processor, "provenance_mismatch", lambda self, job, did: ["forced"])
    r = proc.run(job_for(repo, "TokA"))  # J: the check runs before linking
    assert (r.state, r.category) == ("FAILED", "OPPORTUNITY_PROVENANCE_CONFLICT")
    assert r.opportunity_decision_id is None and decisions(world) == 1
