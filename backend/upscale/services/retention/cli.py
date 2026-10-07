"""Retention command line.

    python -m upscale.services.retention status [--json]
    python -m upscale.services.retention cleanup --dry-run [--json]
    python -m upscale.services.retention cleanup [--max-seconds 600] [--json]

`status` and `cleanup --dry-run` open every database read-only. `cleanup` deletes (in
small transactions, never VACUUM) whatever the policy allows; it ignores
``UPSCALE_RETENTION_ENABLED`` (that only controls the background schedule). Databases
default to ``UPSCALE_SCOUT_DB`` / ``UPSCALE_EVIDENCE_DB`` / ``UPSCALE_SHADOW_DB``; the
Shadow database is only ever read.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from upscale.config import EVIDENCE_DB_PATH, SCOUT_DB_PATH
from upscale.services.retention import engine
from upscale.services.retention.config import load_settings
from upscale.services.shadow.config import default_shadow_db


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.retention")
    p.add_argument("--scout-db", default=None, help="default: UPSCALE_SCOUT_DB")
    p.add_argument("--evidence-db", default=None, help="default: UPSCALE_EVIDENCE_DB")
    p.add_argument("--shadow-db", default=None, help="read only (default: UPSCALE_SHADOW_DB)")
    for name in ("evidence", "snapshot", "social", "outcome"):
        p.add_argument(f"--{name}-days", default=None, help="override the env (floors apply)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("status", help="sizes, freelist, row ages and cutoffs (read-only)")
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("cleanup", help="delete expired rows (or report them with --dry-run)")
    c.add_argument("--dry-run", action="store_true", help="read-only: report, change nothing")
    c.add_argument("--max-seconds", type=float, default=600.0, help="per database")
    c.add_argument("--json", action="store_true")
    return p


def _table_line(t: dict[str, Any], dry_run: bool) -> str:
    if t.get("skipped"):
        return f"  {t['table']:<28} skipped: {t['skipped']}"
    if t.get("error"):
        return f"  {t['table']:<28} ERROR: {t['error']}"
    n = t["eligible"] if dry_run else t["deleted"]
    verb = "eligible" if dry_run else "deleted"
    span = (
        f" [{t['eligible_oldest']} .. {t['eligible_newest']}]" if t.get("eligible_oldest") else ""
    )
    line = (
        f"  {t['table']:<28} cutoff {t['cutoff']}: {n} {verb} of {t['older_than_cutoff']} "
        f"older rows ({t['total_rows']} total), ~{t['approx_mb']} MB{span}"
    )
    for reason, k in t["protected"].items():
        line += f"\n      protected {k:>8}  {reason}"
    if t.get("children_deleted"):
        line += f"\n      with {t['children_deleted']}"
    if t.get("incomplete"):
        line += "\n      (time budget reached: the rest is left for the next pass)"
    return line


def _storage_line(label: str, s: dict[str, Any] | None) -> str:
    if not s:
        return f"  {label}: n/a"
    return (
        f"  {label}: file {s['file_mb']} MB, wal {s['wal_mb']} MB, page_size {s['page_size']}, "
        f"page_count {s['page_count']}, freelist_count {s['freelist_count']} "
        f"({s['freelist_mb']} MB reusable)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings(
        evidence_days=args.evidence_days,
        snapshot_days=args.snapshot_days,
        social_days=args.social_days,
        outcome_days=args.outcome_days,
    )
    scout = args.scout_db or SCOUT_DB_PATH
    evidence = args.evidence_db or EVIDENCE_DB_PATH
    shadow = args.shadow_db or default_shadow_db()
    if args.command == "status":
        report = engine.status(settings, scout, evidence, shadow)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
            return 0
        print(f"retention status at {report['now']}; policy {json.dumps(report['policy'])}")
        print(f"shadow guard: {json.dumps(report['shadow_guard'], default=str)}")
        if "filesystem" in report:
            print(f"filesystem: {report['filesystem']}")
        for name, d in report["databases"].items():
            print(f"{name}: {d['path']}")
            if d.get("error"):
                print(f"  ERROR: {d['error']}")
                continue
            print(_storage_line("storage", d.get("storage")))
            for t in d.get("tables", []):
                print(f"  {t['table']:<28} {t.get('rows', 0)} rows {t.get('oldest')} .. "
                      f"{t.get('newest')}; cutoff {t['cutoff']}: {t.get('older_than_cutoff', 0)} "
                      f"older{'; skipped: ' + t['skipped'] if t.get('skipped') else ''}")  # fmt: skip
        return 0
    report = engine.run(
        settings, scout, evidence, shadow, dry_run=args.dry_run, budget_seconds=args.max_seconds
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"retention {report['mode']} at {report['now']}"
              f"{' (NO CHANGES MADE)' if args.dry_run else ''}")  # fmt: skip
        print(f"policy: {json.dumps(report['policy'])}")
        print(f"shadow guard: {json.dumps(report['shadow_guard'], default=str)}")
        for name, d in report["databases"].items():
            print(f"{name}: {d['path']}")
            if d.get("error"):
                print(f"  ERROR: {d['error']}")
            print(_storage_line("before", d.get("before")))
            for t in d.get("tables", []):
                print(_table_line(t, args.dry_run))
            if not args.dry_run:
                print(_storage_line("after ", d.get("after")))
                print(f"  foreign key violations: {d.get('foreign_key_violations')}")
        tot = report["totals"]
        print(f"total: {tot['eligible_rows']} eligible, {tot['deleted_rows']} deleted, "
              f"~{tot['approx_reusable_mb']} MB reusable (freelist pages; the files do not "
              "shrink without VACUUM)")  # fmt: skip
        for e in tot["errors"]:
            print(f"error: {e}", file=sys.stderr)
    return 1 if report["totals"]["errors"] else 0
