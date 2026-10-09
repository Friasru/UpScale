"""Orchestration bridge command line (offline; no provider is ever called).

Read-only (never create a database):

    python -m upscale.services.opportunity_orchestrator scan --dry-run [--limit N]
    python -m upscale.services.opportunity_orchestrator status
    python -m upscale.services.opportunity_orchestrator jobs [--state STATE]

Offline writes to the orchestrator database (and, for process / recover, real Opportunity
decisions from Safety snapshots that already exist):

    python -m upscale.services.opportunity_orchestrator enqueue
    python -m upscale.services.opportunity_orchestrator process [--limit N]
    python -m upscale.services.opportunity_orchestrator recover

Live Safety collection is not implemented: a job that needs a collection is DEFERRED
(PROVIDER_NOT_CONFIGURED). ``scan`` without ``--dry-run`` refuses.
"""

import argparse
import fcntl
import json
import os
import sqlite3
import sys
import zlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from upscale.log_safety import redact
from upscale.services.opportunity_model.config import (
    evidence_db_path,
    opportunity_db_path,
    safety_db_path,
)
from upscale.services.opportunity_model.models import OpportunityError
from upscale.services.opportunity_model.repository import OpportunityRepository
from upscale.services.opportunity_orchestrator.config import LIVE_MAX_JOBS, orchestrator_db_path
from upscale.services.opportunity_orchestrator.dry_run import dry_run
from upscale.services.opportunity_orchestrator.models import CandidatePlan, DryRunReport
from upscale.services.opportunity_orchestrator.processing import Processor, enqueue
from upscale.services.opportunity_orchestrator.repository import (
    Job,
    OrchestratorRepository,
    OrchestratorStorageError,
)
from upscale.services.opportunity_orchestrator.safety_port import (
    PortInvariantError,
    ReadOnlySafetyPort,
)

LIVE_NOT_IMPLEMENTED = "LIVE PROCESSING: NOT IMPLEMENTED (no live Safety collection)"
LIVE_SAFETY_NOT_IMPLEMENTED = "LIVE SAFETY COLLECTION: NOT IMPLEMENTED"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.opportunity_orchestrator",
                                description="Scout -> Safety -> Opportunity bridge (offline)")  # fmt: skip
    p.add_argument("--evidence-db", default=None, help="default: UPSCALE_EVIDENCE_DB")
    p.add_argument("--safety-db", default=None, help="default: UPSCALE_SAFETY_V2_DB")
    p.add_argument("--opportunity-db", default=None, help="default: UPSCALE_OPPORTUNITY_DB")
    p.add_argument("--orchestrator-db", default=None, help="default: UPSCALE_ORCHESTRATOR_DB")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="what the bridge would do with the newest settled Scout run")
    s.add_argument("--dry-run", action="store_true", help="required (zero writes)")
    s.add_argument("--limit", type=int, default=None, help="show at most N admitted candidates")
    sub.add_parser("status", help="read-only bridge status")
    sub.add_parser("enqueue", help="jobs for the admitted candidates of the newest settled run")
    j = sub.add_parser("jobs", help="list jobs (read-only)")
    j.add_argument("--state", default=None)
    pr = sub.add_parser("process", help="run jobs (offline unless --live)")
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument(
        "--live",
        action="store_true",
        help="REAL Safety collection for exactly one job (provider requests)",
    )
    pr.add_argument("--max-requests", type=int, default=None,
                    help="required with --live: an operator admission ceiling; the collection "
                         "starts only if Safety's own worst-case attempt bound is <= N (Safety "
                         "exceeding its bound is a terminal invariant breach)")  # fmt: skip
    sub.add_parser("recover", help="recover stale jobs (offline)")
    return p


def _t(t: datetime | None) -> str:
    return t.isoformat() if t is not None else "-"


def _age(s: float | None) -> str:
    return f"{s / 60:.1f}m" if s is not None else "-"


def _header(out: TextIO, r: DryRunReport) -> None:
    run = r.selected_run
    out.write(f"now: {r.now.isoformat()}\n")
    out.write(f"evidence archive: {r.archive_status}" + ("  (scan TRUNCATED: no run selected)"
                                                         if r.truncated else "") + "\n")  # fmt: skip
    if run is None:
        out.write("scout run: none in the lookback window\n")
    else:
        out.write(f"scout run: {run.decision_time.isoformat()} ({run.record_count} records, "
                  f"{'settled' if run.settled else 'NOT SETTLED: waiting for the archive'})\n")  # fmt: skip
    for n in r.newer_unsettled_runs:
        out.write(f"newer unsettled run (not used): {n.decision_time.isoformat()} "
                  f"(last archived {n.last_archived_at.isoformat()})\n")  # fmt: skip
    b = r.budget
    out.write(f"safety db: {r.safety_db_status}   opportunity db: {r.opportunity_db_status}\n")
    out.write(f"safety budget {b.day}: used {b.requests_used_today if b.requests_used_today is not None else '-'}"
              f", limit {b.daily_limit if b.daily_limit is not None else 'NOT_KNOWN'} "
              f"({b.limit_source}), remaining "
              f"{b.remaining if b.remaining is not None else 'NOT_KNOWN'}, cooldown until "
              f"{_t(b.cooldown_until)}\n")  # fmt: skip
    out.write(f"collection cost: {b.cost_bound} ({b.cost_bound_note})\n")


def _plan(out: TextIO, p: CandidatePlan) -> None:
    a = p.admission
    out.write(f"{(str(p.order) if p.order else '-'):>3} {a.canonical_id} stage={a.stage} "
              f"rank={a.rank} age={_age(a.age_seconds)} -> {p.action}\n")  # fmt: skip
    out.write(f"      admission: {', '.join(a.reasons)}  record {a.scout_record_id}\n")
    if p.safety is not None:
        s = p.safety
        out.write(f"      safety: {s.reuse} (collection required: {s.collection_required}) "
                  f"snapshot {s.snapshot_id or '-'} as_of {_t(s.as_of)} age "
                  f"{_age(s.age_seconds)}; {s.detail}\n")  # fmt: skip
        if s.domains:
            out.write("      domains: " + ", ".join(f"{k}={v}" for k, v in s.domains) + "\n")
    o = p.opportunity
    if o is not None:
        last = (f"decision {o.decision_id} {o.decision} ({o.quality}) at {_t(o.decision_at)}"
                if o.decision_id else o.detail or "none")  # fmt: skip
        out.write(f"      opportunity (informational): {o.db_status} {last}\n")


def _job_line(j: Job) -> str:
    extra = []
    if j.next_attempt_at is not None and j.state in ("DEFERRED", "RETRY_WAIT"):
        extra.append(f"next {j.next_attempt_at.isoformat()}")
    if j.opportunity_decision_id is not None:
        extra.append(f"decision {j.opportunity_decision_id}")
    if j.safety_snapshot_id is not None:
        extra.append(f"safety snapshot {j.safety_snapshot_id}")
    return (f"job {j.id} {j.canonical_id} {j.state} [{j.category or '-'}] stage={j.stage} "
            f"rank={j.rank} attempts={j.attempt_count} " + " ".join(extra)).rstrip() + "\n"  # fmt: skip


def main(
    argv: Sequence[str] | None = None,
    out: TextIO = sys.stdout,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    live_port: Callable[[str], Any] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    paths = (args.evidence_db or evidence_db_path(), args.safety_db or safety_db_path(),
             args.opportunity_db or opportunity_db_path())  # fmt: skip
    orch = args.orchestrator_db or orchestrator_db_path()
    if args.command == "scan" and not args.dry_run:
        out.write(f"error: scan needs --dry-run. {LIVE_NOT_IMPLEMENTED}\n")
        return 2
    if args.command == "process" and args.live:
        return _live(args, paths, orch, out, clock, live_port)
    try:
        if args.command in ("scan", "status"):
            return _report(args, out, dry_run(*paths, clock()), orch)
        if args.command == "jobs":
            if not Path(orch).is_file():
                out.write("orchestrator db: missing (no jobs)\n")
                return 0
            repo = OrchestratorRepository(orch, read_only=True)
            try:
                states = (args.state,) if args.state else None
                for j in repo.jobs(states):
                    out.write(_job_line(j))
            finally:
                repo.close()
            return 0
        repo = OrchestratorRepository(orch)
        try:
            if args.command == "enqueue":
                res = enqueue(repo, paths[0], clock())
                run = res.run.decision_time.isoformat() if res.run else "none settled"
                out.write(f"scout run: {run}; created {len(res.created)}, existing "
                          f"{len(res.existing)}, superseded {len(res.superseded)}, rejected "
                          f"{res.rejected}\n")  # fmt: skip
                return 0
            opportunity = OpportunityRepository(paths[2])
            try:
                proc = Processor(repo, ReadOnlySafetyPort(paths[1]), opportunity, paths[0],
                                 paths[1], paths[2], clock)  # fmt: skip
                jobs = proc.recover() if args.command == "recover" else proc.process(args.limit)
            finally:
                opportunity.close()
            for j in jobs:
                out.write(_job_line(j))
            out.write(
                f"{len(jobs)} job(s) {'recovered' if args.command == 'recover' else 'processed'}\n"
            )
            out.write(LIVE_SAFETY_NOT_IMPLEMENTED + "\n")
            return 0
        finally:
            repo.close()
    except (OrchestratorStorageError, OpportunityError, ValueError) as exc:
        out.write(f"error: {exc}\n")
        return 2


def _report(args: argparse.Namespace, out: TextIO, report: DryRunReport, orch: str) -> int:
    _header(out, report)
    admitted = [p for p in report.candidates if p.admission.admitted]
    if args.command == "status":
        out.write(f"candidates: {len(report.candidates)}   admitted: {len(admitted)}   "
                  f"rejected: {len(report.candidates) - len(admitted)}\n")  # fmt: skip
        if Path(orch).is_file():
            repo = OrchestratorRepository(orch, read_only=True)
            try:
                out.write("orchestrator db: " + json.dumps(repo.counts(), sort_keys=True) + "\n")
            finally:
                repo.close()
        else:
            out.write("orchestrator db: missing\n")
    else:
        shown = admitted[: args.limit] if args.limit is not None else admitted
        out.write(f"admitted (priority order): {len(admitted)}\n")
        for p in shown:
            _plan(out, p)
        out.write(f"rejected / waiting: {len(report.candidates) - len(admitted)}\n")
        for p in report.candidates:
            if not p.admission.admitted:
                _plan(out, p)
    out.write(report.note + "\n")
    out.write(LIVE_NOT_IMPLEMENTED + "\n")
    return 0


# --- B3: live processing (exactly one job) ---------------------------------------------------

NO_ORDER = "ENTER means entry conditions satisfied; no order was placed."
CONCURRENCY_NOTE = (
    "Do not run manual Safety collect / snapshot commands while this runs: the live lock only "
    "serializes orchestrator live runs; Safety's database ledger stays authoritative, but "
    "check-then-spend races with another process remain possible."
)


class LiveLockHeld(Exception):
    pass


@contextmanager
def live_lock(orchestrator_db: str) -> Iterator[Path]:
    """An exclusive, non-blocking advisory lock next to the orchestrator database, held for
    the whole live run and released on any exit (including Ctrl-C / SystemExit)."""
    path = Path(f"{orchestrator_db}.live.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise LiveLockHeld(f"another live orchestrator run holds {path}") from None
    try:
        yield path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _domains(safety_db: str, snapshot_id: int) -> str:
    """Stored statuses of the domains Opportunity requires, from the snapshot body."""
    conn = sqlite3.connect(f"file:{Path(safety_db).resolve()}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT body_zlib FROM safety_snapshots WHERE id = ?",
                           (snapshot_id,)).fetchone()  # fmt: skip
    finally:
        conn.close()
    body = json.loads(zlib.decompress(row[0])) if row else {}

    def get(*keys: str) -> Any:
        node: Any = body
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        return node

    return (f"token_mint={get('identity', 'token_mint', 'status')} "
            f"mint_authority={get('authority', 'mint_authority', 'status')} "
            f"freeze_authority={get('authority', 'freeze_authority', 'status')} "
            f"holders={get('holders', 'status')} market={get('market', 'status')}")  # fmt: skip


def _live(
    args: argparse.Namespace, paths: tuple[str, str | None, str], orch: str, out: TextIO,
    clock: Callable[[], datetime], live_port: Callable[[str], Any] | None,
) -> int:  # fmt: skip
    """``process --live``: no implicit enqueue or recover; the single highest-priority
    runnable job only; then stop."""
    evidence, safety, opportunity_db = paths
    if args.max_requests is None or args.max_requests < 1:
        out.write("error: --live requires --max-requests N (a positive spending ceiling)\n")
        return 2
    if args.limit not in (None, LIVE_MAX_JOBS):
        out.write(
            f"error: --live processes exactly {LIVE_MAX_JOBS} job (--limit {LIVE_MAX_JOBS})\n"
        )
        return 2
    out.write("databases:\n")
    for name, path in (("evidence archive", evidence), ("safety", safety),
                       ("opportunity", opportunity_db), ("orchestrator", orch)):  # fmt: skip
        state = "missing" if path is None else "present" if Path(path).is_file() else "missing"
        out.write(f"  {name:<16} {path or 'NOT CONFIGURED'} ({state})\n")
    if safety is None:
        out.write("error: no Safety V2 database configured (UPSCALE_SAFETY_V2_DB / --safety-db)\n")
        return 2
    if not Path(evidence).is_file():
        out.write("error: the live Evidence Archive is missing: refusing\n")
        return 2
    if not Path(orch).is_file():
        out.write("error: no orchestrator database: run enqueue first\n")
        return 2
    for name, path in (("safety", safety), ("opportunity", opportunity_db)):
        if not Path(path).is_file():
            out.write(f"warning: the {name} database will be created at {path}\n")
    out.write(CONCURRENCY_NOTE + "\n")
    try:
        with live_lock(orch):
            return _live_one(args, evidence, safety, opportunity_db, orch, out, clock, live_port)
    except LiveLockHeld as exc:
        out.write(f"error: {exc}\n")
        return 2
    except PortInvariantError as exc:
        out.write(f"INVARIANT ERROR: {exc}\nprocessing stopped\n")
        return 3
    except (OrchestratorStorageError, OpportunityError, ValueError) as exc:
        out.write(f"error: {exc}\n")
        return 2
    except Exception as exc:  # e.g. Safety refusing its database: report, never a traceback
        out.write(f"error: {type(exc).__name__}: {redact(str(exc))}\n")
        return 2


def _live_one(
    args: argparse.Namespace, evidence: str, safety: str, opportunity_db: str, orch: str,
    out: TextIO, clock: Callable[[], datetime], live_port: Callable[[str], Any] | None,
) -> int:  # fmt: skip
    if live_port is None:  # imported only here: offline commands never load Safety
        from upscale.services.opportunity_orchestrator.safety_adapter import RealSafetyPort

        live_port = RealSafetyPort
    port = live_port(safety)
    repo = OrchestratorRepository(orch)
    opportunity = OpportunityRepository(opportunity_db)
    try:
        proc = Processor(repo, port, opportunity, evidence, safety, opportunity_db, clock)
        proc.release_due()  # due waits become QUEUED (no provider call); never recover
        runnable = proc.runnable()
        if not runnable:
            out.write("no runnable job (run scan --dry-run / enqueue)\n")
            return 0
        job = runnable[0]
        caps, budget = port.capabilities(), port.budget()
        out.write(f"job {job.id} {job.canonical_id} state={job.state} stage={job.stage} "
                  f"rank={job.rank}\n")  # fmt: skip
        out.write(
            f"safety provider: rpc={caps.rpc} dex={'configured' if caps.dex_configured else 'NOT configured'}\n"
        )
        out.write(f"safety budget {budget.day}: used {budget.used_today}, remaining "
                  f"{budget.remaining}, cooldown until "
                  f"{budget.cooldown_until.isoformat() if budget.cooldown_until else '-'}\n")  # fmt: skip
        out.write(f"collection bound (Safety's own worst case): logical {budget.bound.logical_max}, "
                  f"attempts {budget.bound.attempt_max}; reserve {budget.reserve}; admission "
                  f"ceiling --max-requests {args.max_requests}\n")  # fmt: skip
        if budget.bound.attempt_max > args.max_requests:
            out.write(f"refused: Safety's attempt bound {budget.bound.attempt_max} exceeds "
                      f"--max-requests {args.max_requests}; nothing was collected\n")  # fmt: skip
            return 2
        if (
            job.state == "QUEUED"
            and port.inspect(job.canonical_id, clock()).reuse != "FRESH_REUSABLE"
        ):
            pre = port.preflight(job.canonical_id, clock())
            out.write(f"preflight: {pre.label}" + (f" ({pre.blocked})" if pre.blocked else "")
                      + f" - {pre.detail}\n")  # fmt: skip
        done = proc.run(repo.job(job.id))
        for report in port.reports:
            out.write(f"collection: {report.result.kind} {report.result.outcome}; Safety "
                      f"requests used {report.requests} (bound {report.bound.attempt_max}); "
                      + ", ".join(f"{k}={v}" for k, v in report.outcomes) + "\n")  # fmt: skip
        out.write(f"safety budget used today after: {port.budget().used_today}\n")
        out.write(_job_line(done))
        if done.safety_snapshot_id is not None:
            as_of = done.safety_as_of.isoformat() if done.safety_as_of else "-"
            out.write(f"safety snapshot {done.safety_snapshot_id} as_of {as_of}: "
                      f"{_domains(safety, done.safety_snapshot_id)}\n")  # fmt: skip
        if done.opportunity_decision_id is not None:
            stored = opportunity.get(done.opportunity_decision_id)
            d = stored.decision
            out.write(f"opportunity decision {stored.id}: {d.decision} (quality {d.quality}) "
                      f"at {stored.decision_at.isoformat()}\n")  # fmt: skip
        out.write("1 job processed; stopping (live mode never processes a second job)\n")
        out.write(NO_ORDER + "\n")
        return 0
    finally:
        opportunity.close()
        repo.close()
