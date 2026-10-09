"""Opportunity orchestration bridge: a waiting job is re-admitted from exactly its own Scout
record before it can reach Safety. A stale / revoked / sourceless job is SUPERSEDED (schema 1,
no new state) and never spends Safety budget or creates an Opportunity decision. Offline:
fixture stores and a scripted Safety port only."""

import dataclasses
import io
import socket
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.opportunity_model_fakes import Archive, scout_record
from tests.test_opportunity_orchestrator_b1 import NOW, T, cand, safety_store
from tests.test_opportunity_orchestrator_b2 import CID, Clock, FakeSafety, decisions, infra
from upscale.services.evidence_archive.store import EvidenceStore
from upscale.services.opportunity_model.repository import OpportunityRepository
from upscale.services.opportunity_orchestrator.admission import evaluate_candidate
from upscale.services.opportunity_orchestrator.cli import main
from upscale.services.opportunity_orchestrator.config import (
    ORCHESTRATOR_DB_SCHEMA_VERSION,
    POLICY,
    OrchestratorPolicy,
)
from upscale.services.opportunity_orchestrator.processing import (
    Processor,
    enqueue,
    revalidate,
    supersede_inadmissible,
)
from upscale.services.opportunity_orchestrator.repository import (
    TERMINAL,
    Job,
    OrchestratorRepository,
)
from upscale.services.opportunity_orchestrator.safety_port import Preflight, SafetyInspection

LIMIT = timedelta(seconds=POLICY.admission_market_age_s)  # 15 min, B1's own policy value
EPS = timedelta(microseconds=1)


class CountingSafety(FakeSafety):
    """FakeSafety that also counts `inspect` and `preflight` (every port touch)."""

    def inspect(self, cid: str, now: datetime) -> SafetyInspection:
        self.calls.append(("inspect", cid))
        return super().inspect(cid, now)

    def preflight(self, cid: str, now: datetime) -> Preflight:
        self.calls.append(("preflight", cid))
        return super().preflight(cid, now)


class World:
    """Fixture stores: an archive whose records are added per test, a test Safety file, an
    Opportunity and an orchestrator database, a clock at NOW (2 min after T)."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.evidence = str(tmp / "evidence.sqlite3")
        self.safety = str(safety_store(tmp / "safety.sqlite3", []))
        self.opportunity = str(tmp / "opportunity.sqlite3")
        self.orch = str(tmp / "orchestrator.sqlite3")
        self.clock = Clock(NOW)
        self.repo = OrchestratorRepository(self.orch)
        self.opp = OpportunityRepository(self.opportunity)

    def add(self, tag: str, observed: datetime = T, **over: Any) -> None:
        arch = Archive(Path(self.evidence))
        arch.add(scout_record(observed, cand=cand(tag, market_at=observed, **over),
                              asset_id=CID[tag]))  # fmt: skip
        arch.close()

    def admit(self, tag: str, observed: datetime = T, at: datetime | None = None) -> Job:
        """A job straight from the exact archived record (no run selection)."""
        store = EvidenceStore(self.evidence, read_only=True)
        rows = store.records(kind="scout", asset_id=CID[tag], since=observed, until=observed)
        store.close()
        (r,) = rows
        a = evaluate_candidate(r, now=at or observed + timedelta(minutes=2))
        assert a.admitted, a.reasons
        job, created = self.repo.enqueue(a, at or observed + timedelta(minutes=2))
        assert created
        return job

    def proc(self, port: Any, policy: OrchestratorPolicy = POLICY) -> Processor:
        return Processor(self.repo, port, self.opp, self.evidence, self.safety, self.opportunity,
                         self.clock, policy=policy)  # fmt: skip

    def events(self, job: Job) -> list[tuple[str | None, str, str | None]]:
        return [(e["from"], e["to"], e["category"]) for e in self.repo.events(job.id)]

    def total_events(self) -> int:
        return int(self.repo._conn.execute("SELECT COUNT(*) FROM orch_events").fetchone()[0])

    def close(self) -> None:
        self.repo.close()
        self.opp.close()


@pytest.fixture
def w(tmp_path: Path) -> Any:
    world = World(tmp_path)
    yield world
    world.close()


def snapshots(w: World) -> int:
    c = sqlite3.connect(w.safety)
    try:
        return int(c.execute("SELECT COUNT(*) FROM safety_snapshots").fetchone()[0])
    finally:
        c.close()


# --- A / B: the exact boundary ------------------------------------------------------------------


def test_a_exactly_at_the_limit_stays_admissible_and_runs(w: World) -> None:
    w.add("TokA")
    job = w.admit("TokA")
    w.clock.t = T + LIMIT  # market evidence exactly admission_market_age_s old
    port = CountingSafety(w.safety, {CID["TokA"]: ["COMPLETE"]})
    proc = w.proc(port)
    assert proc.revalidate_waiting() == []
    assert [j.id for j in proc.runnable()] == [job.id]
    done = proc.run(w.repo.job(job.id))
    assert done.state == "DECIDED" and ("collect", CID["TokA"]) in port.calls


def test_b_one_microsecond_past_the_limit_expires(w: World) -> None:
    w.add("TokA")
    job = w.admit("TokA")
    w.clock.t = T + LIMIT + EPS
    proc = w.proc(CountingSafety(w.safety))
    (gone,) = proc.revalidate_waiting()
    assert (gone.id, gone.state, gone.category) == (job.id, "SUPERSEDED", "ADMISSION_EXPIRED")
    assert gone.note == (f"market_observed_at {T.isoformat()} is 900.000001s old "
                         "(max 900.000000s)")  # fmt: skip
    assert w.events(gone)[-1] == ("QUEUED", "SUPERSEDED", "ADMISSION_EXPIRED")
    assert proc.runnable() == []


# --- C: a stale job never touches Safety or Opportunity ----------------------------------------


def test_c_stale_queued_spends_nothing_by_any_path(w: World) -> None:
    for tag in ("TokA", "TokB"):
        w.add(tag)
    a, b = w.admit("TokA"), w.admit("TokB")
    w.clock.t = T + LIMIT + timedelta(seconds=1)
    port = CountingSafety(w.safety, {CID["TokA"]: ["COMPLETE"], CID["TokB"]: ["COMPLETE"]})
    proc = w.proc(port)
    # defence in depth: a direct caller that skipped the sweep and `runnable`
    direct = proc.run(w.repo.job(a.id))
    assert (direct.state, direct.category) == ("SUPERSEDED", "ADMISSION_EXPIRED")
    # the normal processing path
    assert proc.process() == []
    assert (w.repo.job(b.id).state, w.repo.job(b.id).category) == (
        "SUPERSEDED", "ADMISSION_EXPIRED")  # fmt: skip
    assert port.calls == [] and snapshots(w) == 0 and decisions(vars(w)) == 0


# --- D / E / F / G: deferred and retried jobs use their actual age -----------------------------


def test_d_stale_deferred_is_superseded(w: World) -> None:
    w.add("TokA")
    job = w.admit("TokA")
    blocked = Preflight("CONSERVATIVE_PREFLIGHT_INSUFFICIENT", "SAFETY_BUDGET_EXHAUSTED")
    port = CountingSafety(w.safety, preflight=blocked)
    proc = w.proc(port)
    assert proc.run(w.repo.job(job.id)).state == "DEFERRED"
    w.clock.t = T + LIMIT + EPS
    (gone,) = proc.revalidate_waiting()
    assert w.events(gone)[-1] == ("DEFERRED", "SUPERSEDED", "ADMISSION_EXPIRED")
    assert [c for c in port.calls if c[0] == "collect"] == []


def test_e_stale_retry_wait_is_superseded_not_collected(w: World) -> None:
    w.add("TokA")
    job = w.admit("TokA")
    port = CountingSafety(w.safety, {CID["TokA"]: [infra("PROVIDER_UNAVAILABLE"), "COMPLETE"]})
    proc = w.proc(port)
    assert proc.run(w.repo.job(job.id)).state == "RETRY_WAIT"  # failed at T + 2 min
    w.clock.t = w.repo.job(job.id).next_attempt_at + timedelta(minutes=30)  # type: ignore[operator]
    assert proc.process() == []
    j = w.repo.job(job.id)
    assert (j.state, j.category, j.attempt_count) == ("SUPERSEDED", "ADMISSION_EXPIRED", 1)
    assert w.events(j)[-1] == ("RETRY_WAIT", "SUPERSEDED", "ADMISSION_EXPIRED")
    assert port.calls.count(("collect", CID["TokA"])) == 1 and decisions(vars(w)) == 0


def test_f_first_retry_inside_the_window_still_runs(w: World) -> None:
    w.add("TokA")
    job = w.admit("TokA")  # admitted at T + 2 min
    port = CountingSafety(w.safety, {CID["TokA"]: [infra("PROVIDER_UNAVAILABLE"), "COMPLETE"]})
    proc = w.proc(port)
    first = proc.run(w.repo.job(job.id))  # Scout age 1 failure at 2 min
    assert first.next_attempt_at == NOW + timedelta(minutes=10)  # retry schedule unchanged
    w.clock.t = first.next_attempt_at  # Scout age 12 min <= 15 min
    (done,) = proc.process()
    assert (done.state, done.attempt_count) == ("DECIDED", 1)
    assert port.calls.count(("collect", CID["TokA"])) == 2 and decisions(vars(w)) == 1


def test_g_retry_past_the_window_expires_instead_of_collecting(w: World) -> None:
    w.add("TokA")
    job = w.admit("TokA")
    w.clock.t = T + timedelta(minutes=8)  # first attempt at Scout age 8 min
    port = CountingSafety(w.safety, {CID["TokA"]: [infra("PROVIDER_UNAVAILABLE"), "COMPLETE"]})
    proc = w.proc(port)
    first = proc.run(w.repo.job(job.id))
    assert first.next_attempt_at == T + timedelta(minutes=18)
    w.clock.t = first.next_attempt_at  # Scout age 18 min > 15 min
    assert proc.process() == []
    j = w.repo.job(job.id)
    assert (j.state, j.category) == ("SUPERSEDED", "ADMISSION_EXPIRED")
    assert port.calls.count(("collect", CID["TokA"])) == 1


# --- H / I: a stale job is never runnable, so it can't beat a fresh one ------------------------

FRESH_AT = T + timedelta(minutes=20)


@pytest.mark.parametrize(("stale", "fresh"), [
    ({"stage": "EARLY", "rank": 2}, {"stage": "EARLY", "rank": 3}),  # H
    ({"stage": "ACCELERATING", "rank": 1}, {"stage": "EARLY", "rank": 10}),  # I
])  # fmt: skip
def test_h_i_stale_higher_priority_never_beats_a_fresh_job(
    w: World, stale: dict[str, Any], fresh: dict[str, Any]
) -> None:
    w.add("TokA", T, **stale)
    w.add("TokB", FRESH_AT, **fresh)
    old = w.admit("TokA", T)
    new = w.admit("TokB", FRESH_AT)
    assert old.priority < new.priority  # by B1 priority alone the stale job would win
    w.clock.t = FRESH_AT + timedelta(minutes=2)
    port = CountingSafety(w.safety, {CID["TokB"]: ["COMPLETE"]})
    proc = w.proc(port)
    assert [j.id for j in proc.runnable()] == [new.id]  # even without the sweep
    (done,) = proc.process(limit=1)
    assert (done.id, done.state) == (new.id, "DECIDED")
    assert w.repo.job(old.id).category == "ADMISSION_EXPIRED"
    assert {c[1] for c in port.calls} == {CID["TokB"]}


def test_valid_jobs_keep_b1_priority(w: World) -> None:
    for tag, over in (("TokA", {"stage": "EARLY", "rank": 3}),
                      ("TokB", {"stage": "EARLY", "rank": 1}),
                      ("TokC", {"stage": "ACCELERATING", "rank": 7})):  # fmt: skip
        w.add(tag, **over)
    jobs = {tag: w.admit(tag) for tag in ("TokA", "TokB", "TokC")}
    proc = w.proc(CountingSafety(w.safety))
    assert [j.id for j in proc.runnable()] == [jobs[t].id for t in ("TokC", "TokB", "TokA")]


# --- J / K / L: exactly the original record ------------------------------------------------------


def test_j_the_original_record_is_revalidated_never_a_newer_one(w: World) -> None:
    w.add("TokA", T)
    job = w.admit("TokA", T)
    w.add("TokA", FRESH_AT)  # a newer, admissible record of the same token
    later = FRESH_AT + timedelta(minutes=1)
    check = revalidate(w.evidence, [w.repo.job(job.id)], later)[job.id]
    assert check.verdict == "ADMISSION_EXPIRED" and T.isoformat() in check.note
    w.clock.t = later
    assert w.proc(CountingSafety(w.safety)).runnable() == []
    # a different record id at the job's own run time is not substituted either
    forged = dataclasses.replace(w.repo.job(job.id), scout_record_id="0" * 32)
    assert revalidate(w.evidence, [forged], NOW)[job.id].verdict == "ADMISSION_SOURCE_MISSING"


def test_k_missing_source_record_supersedes_without_safety(w: World) -> None:
    w.add("TokA")
    store = EvidenceStore(w.evidence, read_only=True)
    (r,) = store.records(kind="scout", asset_id=CID["TokA"])
    store.close()
    a = dataclasses.replace(evaluate_candidate(r, now=NOW), scout_record_id="f" * 32)
    job, _ = w.repo.enqueue(a, NOW)  # its record isn't in the archive (e.g. retention)
    port = CountingSafety(w.safety, {CID["TokA"]: ["COMPLETE"]})
    proc = w.proc(port)
    assert proc.runnable() == []
    assert proc.process() == []
    j = w.repo.job(job.id)
    assert (j.state, j.category) == ("SUPERSEDED", "ADMISSION_SOURCE_MISSING")
    assert "f" * 32 in (j.note or "") and port.calls == [] and decisions(vars(w)) == 0


def test_k_unreadable_archive_concludes_nothing_and_spends_nothing(w: World) -> None:
    w.add("TokA")
    job = w.admit("TokA")
    port = CountingSafety(w.safety, {CID["TokA"]: ["COMPLETE"]})
    proc = w.proc(port)
    proc.evidence_db = str(w.tmp / "gone.sqlite3")
    assert proc.revalidate_waiting() == [] and proc.runnable() == []
    assert proc.run(w.repo.job(job.id)).state == "QUEUED"
    assert port.calls == [] and not (w.tmp / "gone.sqlite3").exists()


def test_l_current_policy_rejection_revokes(w: World) -> None:
    w.add("TokA", rank=2)
    job = w.admit("TokA")
    port = CountingSafety(w.safety, {CID["TokA"]: ["COMPLETE"]})
    proc = w.proc(port, OrchestratorPolicy(max_rank=1))
    (gone,) = proc.revalidate_waiting()
    assert (gone.id, gone.category) == (job.id, "ADMISSION_REVOKED")
    assert gone.note == f"Scout record {job.scout_record_id} no longer admitted: RANK_TOO_LOW"
    assert port.calls == []


def test_l_revocation_lists_every_failed_rule_including_age(w: World) -> None:
    w.add("TokA", rank=2)
    job = w.admit("TokA")
    late = T + LIMIT + EPS
    check = revalidate(w.evidence, [job], late, OrchestratorPolicy(max_rank=1))[job.id]
    assert check.verdict == "ADMISSION_REVOKED"
    assert check.note.endswith("RANK_TOO_LOW,SCOUT_TOO_OLD")


# --- M / N / O / P: jobs past the collection boundary, and terminal jobs ------------------------


def test_m_n_o_in_flight_jobs_are_never_revalidated(w: World) -> None:
    for tag in ("TokA", "TokB", "TokC"):
        w.add(tag)
    a, b, c = (w.admit(t) for t in ("TokA", "TokB", "TokC"))
    collecting = w.repo.transition(a, "COLLECTING", NOW, "COLLECTION_STARTED", None,
                                   lease_started_at=NOW)  # fmt: skip
    snap = {"safety_snapshot_id": 1, "safety_as_of": NOW}
    snapped = w.repo.transition(b, "SNAPSHOTTED", NOW, "SAFETY_REUSED", None, **snap)
    snapped_c = w.repo.transition(c, "SNAPSHOTTED", NOW, "SAFETY_REUSED", None, **snap)
    deciding = w.repo.transition(snapped_c, "DECIDING", NOW, "DECISION_STARTED", None,
                                 lease_started_at=NOW, opportunity_decision_at=NOW,
                                 opportunity_rules_version="1")  # fmt: skip
    w.clock.t = T + timedelta(hours=2)
    before = w.total_events()
    proc = w.proc(CountingSafety(w.safety))
    assert proc.revalidate_waiting() == []
    assert supersede_inadmissible(w.repo, w.evidence, w.clock()) == []
    assert [w.repo.job(j.id).state for j in (collecting, snapped, deciding)] == [
        "COLLECTING", "SNAPSHOTTED", "DECIDING"]  # fmt: skip
    assert snapped.id in [j.id for j in proc.runnable()]  # SNAPSHOTTED still goes on
    assert w.total_events() == before


def test_p_terminal_jobs_are_untouched(w: World) -> None:
    for tag in ("TokA", "TokB", "TokC"):
        w.add(tag)
    a, b, c = (w.admit(t) for t in ("TokA", "TokB", "TokC"))
    port = CountingSafety(w.safety, {CID["TokA"]: ["COMPLETE"]})
    assert w.proc(port).run(w.repo.job(a.id)).state == "DECIDED"
    started = w.repo.transition(b, "COLLECTING", NOW, "COLLECTION_STARTED", None,
                                lease_started_at=NOW)  # fmt: skip
    w.repo.transition(started, "FAILED", NOW, "PROVIDER_UNAVAILABLE", None, attempt_count=3)
    w.repo.transition(c, "SUPERSEDED", NOW, "SUPERSEDED", "by hand")
    rows = {j.id: j for j in w.repo.jobs()}
    w.clock.t = T + timedelta(days=1)
    before = w.total_events()
    assert w.proc(port).revalidate_waiting() == [] and w.proc(port).process() == []
    assert {j.id: j for j in w.repo.jobs()} == rows and w.total_events() == before
    assert {j.state for j in rows.values()} <= set(TERMINAL)


# --- Q: idempotent ----------------------------------------------------------------------------


def test_q_the_sweep_is_idempotent(w: World) -> None:
    for tag in ("TokA", "TokB"):
        w.add(tag)
    w.admit("TokA"), w.admit("TokB")
    w.clock.t = T + LIMIT + EPS
    proc = w.proc(CountingSafety(w.safety))
    assert len(proc.revalidate_waiting()) == 2
    after_first = w.total_events()
    assert proc.revalidate_waiting() == [] and proc.process() == []
    assert supersede_inadmissible(w.repo, w.evidence, w.clock()) == []
    assert w.total_events() == after_first


# --- enqueue hygiene ---------------------------------------------------------------------------


def test_enqueue_supersedes_stale_waiting_jobs_of_other_tokens(w: World) -> None:
    w.add("TokA", T)
    old = w.admit("TokA", T)
    w.add("TokB", FRESH_AT)
    res = enqueue(w.repo, w.evidence, FRESH_AT + timedelta(minutes=2))
    assert [j.canonical_id for j in res.created] == [CID["TokB"]]
    assert [(j.id, j.category) for j in res.inadmissible] == [(old.id, "ADMISSION_EXPIRED")]
    assert res.superseded == []  # B2 supersession is per token; this was another token


def test_enqueue_never_supersedes_a_job_it_just_admitted(w: World) -> None:
    w.add("TokA", T)
    res = enqueue(w.repo, w.evidence, T + LIMIT)  # admitted exactly at the limit
    assert len(res.created) == 1 and res.inadmissible == []


# --- R / S / T: read-only commands, schema 1 -------------------------------------------------


def _orch_args(w: World) -> list[str]:
    return ["--evidence-db", w.evidence, "--safety-db", w.safety, "--opportunity-db",
            w.opportunity, "--orchestrator-db", w.orch]  # fmt: skip


def test_r_s_scan_status_and_jobs_never_write(w: World) -> None:
    w.add("TokA")
    w.admit("TokA")
    w.clock.t = T + timedelta(hours=1)  # the job is stale: nothing read-only may supersede it
    w.repo.close()
    paths = [Path(p) for p in (w.orch, w.evidence, w.safety)]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    for cmd in (["scan", "--dry-run"], ["status"], ["jobs"]):
        assert main(_orch_args(w) + cmd, out=io.StringIO(), clock=w.clock) == 0
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == before
    assert not Path(w.opportunity).exists() or Path(w.opportunity).stat().st_size >= 0
    w.repo = OrchestratorRepository(w.orch, read_only=True)
    assert [j.state for j in w.repo.jobs()] == ["QUEUED"]


def test_t_schema_stays_version_1(w: World) -> None:
    assert ORCHESTRATOR_DB_SCHEMA_VERSION == 1
    meta = dict(w.repo._conn.execute("SELECT key, value FROM orch_meta").fetchall())
    assert meta == {"component": "opportunity_orchestrator", "schema_version": "1"}


# --- production schema-1 compatibility ---------------------------------------------------------


def _ddl(path: str) -> list[tuple[str, str, str]]:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return c.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    finally:
        c.close()


def test_production_like_schema_1_database_needs_no_migration(w: World) -> None:
    """Like production after the first live validation: job 1 DECIDED, the rest QUEUED from
    the same (now stale) Scout run. Supersession is a legal schema-1 transition."""
    for tag in ("TokA", "TokB", "TokC"):
        w.add(tag)
    jobs = [w.admit(t) for t in ("TokA", "TokB", "TokC")]
    port = CountingSafety(w.safety, {CID["TokA"]: ["COMPLETE"]})
    assert w.proc(port).run(w.repo.job(jobs[0].id)).state == "DECIDED"
    w.repo.close()
    w.opp.close()
    ddl, meta_before = _ddl(w.orch), sqlite3.connect(w.orch).execute(
        "SELECT key, value FROM orch_meta ORDER BY key").fetchall()  # fmt: skip
    decided = OrchestratorRepository(w.orch).job(jobs[0].id)
    events_before = {j.id: len(OrchestratorRepository(w.orch).events(j.id)) for j in jobs}
    # the fixed code opens the existing file as is
    w.repo, w.opp = OrchestratorRepository(w.orch), OpportunityRepository(w.opportunity)
    w.clock.t = T + timedelta(hours=1)
    port.calls.clear()
    proc = w.proc(port)
    assert proc.process(limit=1) == []  # nothing runnable: nothing selected, nothing spent
    assert port.calls == []
    assert w.repo.job(jobs[0].id) == decided
    for j in jobs[1:]:
        now_j = w.repo.job(j.id)
        assert (now_j.state, now_j.category) == ("SUPERSEDED", "ADMISSION_EXPIRED")
        assert len(w.repo.events(j.id)) == events_before[j.id] + 1
        assert w.events(now_j)[-1] == ("QUEUED", "SUPERSEDED", "ADMISSION_EXPIRED")
    assert len(w.repo.events(jobs[0].id)) == events_before[jobs[0].id]
    assert _ddl(w.orch) == ddl  # no table rebuild, no trigger change
    meta_after = w.repo._conn.execute("SELECT key, value FROM orch_meta ORDER BY key").fetchall()
    assert meta_after == meta_before


# --- network ---------------------------------------------------------------------------------


def test_revalidation_makes_no_network_calls(w: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    w.add("TokA")
    w.admit("TokA")
    w.clock.t = T + LIMIT + EPS
    port = CountingSafety(w.safety)
    assert len(w.proc(port).revalidate_waiting()) == 1 and port.calls == []
