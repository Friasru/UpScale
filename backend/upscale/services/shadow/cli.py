"""Shadow / Paper Strategy command line (simulation only: no orders, keys or signing).

python -m upscale.services.shadow init
python -m upscale.services.shadow strategies
python -m upscale.services.shadow run [--run production] [--since 2026-09-30T05:50:00Z] [--until ...]
python -m upscale.services.shadow status
python -m upscale.services.shadow positions [--open | --closed]
python -m upscale.services.shadow trades
python -m upscale.services.shadow decisions [--action ENTER]
python -m upscale.services.shadow metrics
python -m upscale.services.shadow diagnostics [--strategy scout_technical --strategy ...] [--json]
python -m upscale.services.shadow rejections [--reason TECHNICAL_NOT_AVAILABLE]

Filters: --run, --strategy, --asset, --since, --until.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from upscale.services.evidence_archive import hooks as evidence_hooks
from upscale.services.evidence_archive.store import EvidenceStore, EvidenceStoreError
from upscale.services.shadow.config import (
    CLEAN_DATA_CUTOFF,
    DEFAULT_RUN_ID,
    default_evidence_db,
    default_shadow_db,
    parse_timestamp,
)
from upscale.services.shadow.engine import ShadowEngine, ShadowError, resolve_run
from upscale.services.shadow.store import ShadowStore, ShadowStoreError, readable


def _timestamp(raw: str) -> datetime:
    try:
        return parse_timestamp(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _filters(p: argparse.ArgumentParser, run_default: str | None = None) -> None:
    p.add_argument("--run", default=run_default, help=f"run id (default: {DEFAULT_RUN_ID})")
    p.add_argument("--strategy", default=None, help="strategy id")
    p.add_argument("--asset", default=None, help="canonical asset id, e.g. solana:<mint>")
    p.add_argument("--since", type=_timestamp, default=None, help="ISO-8601, inclusive")
    p.add_argument("--until", type=_timestamp, default=None, help="ISO-8601, exclusive")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.shadow")
    p.add_argument("--db", default=None, help="shadow database (UPSCALE_SHADOW_DB)")
    p.add_argument("--evidence-db", default=None, help="Evidence Archive (read-only)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="register the v1 baseline strategies (idempotent)")
    sub.add_parser("strategies")
    r = sub.add_parser("run", help="create (first time) or continue a paper run")
    r.add_argument("--run", default=DEFAULT_RUN_ID)
    r.add_argument("--since", type=_timestamp, default=CLEAN_DATA_CUTOFF,
                   help=f"first decision time (default: the clean-data cutoff {CLEAN_DATA_CUTOFF.isoformat()})")  # fmt: skip
    r.add_argument("--until", type=_timestamp, default=None, help="last decision time")
    r.add_argument("--strategy", action="append", default=None,
                   help="strategy id (repeatable; default: every registered strategy)")  # fmt: skip
    r.add_argument("--allow-contaminated", action="store_true",
                   help="allow --since before the clean-data cutoff (labeled research run)")  # fmt: skip
    sub.add_parser("status")
    pos = sub.add_parser("positions")
    _filters(pos)
    group = pos.add_mutually_exclusive_group()
    group.add_argument("--open", action="store_true")
    group.add_argument("--closed", action="store_true")
    _filters(sub.add_parser("trades"))
    d = sub.add_parser("decisions")
    _filters(d)
    d.add_argument("--action", choices=("ENTER", "HOLD", "EXIT", "NO_ACTION"), default=None)
    d.add_argument("--limit", type=int, default=200)
    _filters(sub.add_parser("metrics"))
    dg = sub.add_parser("diagnostics", help="why each strategy entered or rejected candidates")
    dg.add_argument("--run", default=None, help=f"run id (default: {DEFAULT_RUN_ID})")
    dg.add_argument("--strategy", action="append", default=None, help="repeatable")
    dg.add_argument("--asset", default=None)
    dg.add_argument("--since", type=_timestamp, default=None)
    dg.add_argument("--until", type=_timestamp, default=None)
    dg.add_argument("--json", action="store_true", help="print JSON")
    rj = sub.add_parser("rejections", help="individual rejection diagnostics")
    _filters(rj)
    rj.add_argument("--reason", default=None, help="only rows with this reason code")
    rj.add_argument("--limit", type=int, default=200)
    return p


def _diagnostics_text(d: dict[str, Any]) -> str:
    lines = [f"run {d['run_id']}: diagnostics available from "
             f"{d['diagnostics_available_from'] or 'NOT AVAILABLE'}", d["note"]]  # fmt: skip
    for s in d["strategies"]:
        lines += [
            f"{s['strategy']}:",
            f"  evaluated: {s['evaluated']}",
            f"  entered: {s['entered']}",
            f"  blocked (qualified, stopped by position/risk control): {s['blocked']}",
            f"  rejected: {s['rejected']}",
            f"  held (position already open): {s['held']}",
            "  reasons:" if s["reasons"] else "  reasons: none",
        ]
        lines += [f"    {code}: {v['count']} ({v['pct_of_rejected']}%)"
                  for code, v in s["reasons"].items()]  # fmt: skip
        if s["before_diagnostics_first_reason_only"]:
            lines.append("  before diagnostics (first failed rule only, v1 codes):")
            lines += [f"    {k}: {v}" for k, v in s["before_diagnostics_first_reason_only"].items()]
    return "\n".join(lines)


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    evidence_hooks.install(None)  # a simulation process never archives production evidence
    store = ShadowStore(args.db or default_shadow_db())
    evidence_path = args.evidence_db or default_evidence_db()
    evidence = EvidenceStore(evidence_path, read_only=True)
    engine = ShadowEngine(store, evidence)
    try:
        if args.command == "init":
            for key, state in engine.init_baselines().items():
                print(f"{key}: {state}")
            print("Baseline strategies are comparison rules, not recommendations.")
        elif args.command == "strategies":
            _print([{"key": s.key, "config_hash": s.config_hash, **s.model_dump(mode="json")}
                    for s in store.strategies()])  # fmt: skip
        elif args.command == "run":
            engine.ensure_run(
                args.run, args.since, args.until, args.strategy, args.allow_contaminated,
                args={"cli": list(argv or sys.argv[1:])},
            )  # fmt: skip
            _print(engine.run(args.run))
        elif args.command == "status":
            _print(engine.status())
        else:
            run_id = resolve_run(store, args.run)
            common = {"run_id": run_id, "strategy_id": args.strategy, "asset_id": args.asset,
                      "since": args.since, "until": args.until}  # fmt: skip
            if args.command == "positions":
                status = "OPEN" if args.open else "CLOSED" if args.closed else None
                _print(readable(store.positions(status=status, **common)))
            elif args.command == "trades":
                _print(readable(store.trades(**common)))
            elif args.command == "decisions":
                _print(readable(store.decisions(action=args.action, limit=args.limit, **common)))
            elif args.command == "diagnostics":
                d = engine.diagnostics(run_id, args.strategy, args.asset, args.since, args.until)
                print(json.dumps(d, indent=2, default=str) if args.json else _diagnostics_text(d))
            elif args.command == "rejections":
                _print(readable(store.rejections(reason=args.reason, limit=args.limit, **common)))
            elif args.command == "metrics":
                _print(engine.metrics(run_id, args.strategy, args.asset, args.since, args.until))
        return 0
    except (ShadowError, ShadowStoreError, EvidenceStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
        evidence.close()
