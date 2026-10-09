"""Orchestration bridge command line, B1: read-only.

    python -m upscale.services.opportunity_orchestrator scan --dry-run [--limit N]
    python -m upscale.services.opportunity_orchestrator status

Nothing is collected, written or decided; no provider is called. Live processing is not
implemented: ``scan`` without ``--dry-run`` refuses.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TextIO

from upscale.services.opportunity_model.config import (
    evidence_db_path,
    opportunity_db_path,
    safety_db_path,
)
from upscale.services.opportunity_orchestrator.dry_run import dry_run
from upscale.services.opportunity_orchestrator.models import CandidatePlan, DryRunReport

LIVE_NOT_IMPLEMENTED = "LIVE PROCESSING: NOT IMPLEMENTED (B1 is read-only)"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.opportunity_orchestrator",
                                description="Scout -> Safety -> Opportunity bridge (B1, read-only)")  # fmt: skip
    p.add_argument("--evidence-db", default=None, help="default: UPSCALE_EVIDENCE_DB")
    p.add_argument("--safety-db", default=None, help="default: UPSCALE_SAFETY_V2_DB")
    p.add_argument("--opportunity-db", default=None, help="default: UPSCALE_OPPORTUNITY_DB")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="what the bridge would do with the newest settled Scout run")
    s.add_argument("--dry-run", action="store_true", help="required in B1")
    s.add_argument("--limit", type=int, default=None, help="show at most N admitted candidates")
    sub.add_parser("status", help="read-only bridge status")
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


def main(
    argv: Sequence[str] | None = None,
    out: TextIO = sys.stdout,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "scan" and not args.dry_run:
        out.write(f"error: scan needs --dry-run. {LIVE_NOT_IMPLEMENTED}\n")
        return 2
    report = dry_run(
        args.evidence_db or evidence_db_path(), args.safety_db or safety_db_path(),
        args.opportunity_db or opportunity_db_path(), clock(),
    )  # fmt: skip
    _header(out, report)
    admitted = [p for p in report.candidates if p.admission.admitted]
    if args.command == "status":
        out.write(f"candidates: {len(report.candidates)}   admitted: {len(admitted)}   "
                  f"rejected: {len(report.candidates) - len(admitted)}\n")  # fmt: skip
        out.write("orchestrator db: none (B1)\n")
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
