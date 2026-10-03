"""Shadow / Paper Strategy command line (simulation only: no orders, keys or signing).

python -m upscale.services.shadow init
python -m upscale.services.shadow strategies
python -m upscale.services.shadow run [--run production] [--since 2026-09-30T05:50:00Z] [--until ...]
    [--availability-policy EVIDENCE_AWARE_V2 | LEGACY_V1]  (a new run only; default V2)
    [--execution-model IDEALIZED_NO_FEES | REALISTIC_V1]  (a new run only; default IDEALIZED)
    [--entry-fee-bps 30] [--exit-fee-bps 30] [--slippage-bps 50] [--latency-seconds 60]
    [--price-impact [--max-price-impact-bps 1000]] [--entry-max-wait-minutes 0 (indefinite) | N]
python -m upscale.services.shadow status
python -m upscale.services.shadow positions [--open | --closed]
python -m upscale.services.shadow trades
python -m upscale.services.shadow decisions [--action ENTER]
python -m upscale.services.shadow metrics
python -m upscale.services.shadow diagnostics [--strategy scout_technical --strategy ...] [--json]
python -m upscale.services.shadow rejections [--reason TECHNICAL_NOT_AVAILABLE]
python -m upscale.services.shadow funnel [--run ...] [--strategy scout_technical ...] [--json]
python -m upscale.services.shadow storage [--json]
python -m upscale.services.shadow audit-unavailable [--run continuous-v1] [--strategy ...] [--json]
python -m upscale.services.shadow report [--run continuous-v2] [--strategy ...] [--since ...] [--until ...] [--json]
python -m upscale.services.shadow compare --base continuous-v2 --other continuous-v2-realistic
    [--strategy ...] [--since ...] [--until ...] [--json]

Filters: --run, --strategy, --asset, --since, --until.
"""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from upscale.services.evidence_archive import hooks as evidence_hooks
from upscale.services.evidence_archive.store import EvidenceStore, EvidenceStoreError
from upscale.services.shadow.audit import AuditSettings, audit_unavailable
from upscale.services.shadow.audit import text as audit_text
from upscale.services.shadow.config import (
    AVAILABILITY_POLICIES,
    CLEAN_DATA_CUTOFF,
    DEFAULT_RUN_ID,
    DETAIL_MODES,
    EXECUTION_MODELS,
    ExecutionConfig,
    default_evidence_db,
    default_shadow_db,
    load_diagnostics_settings,
    parse_timestamp,
    run_execution,
)
from upscale.services.shadow.engine import ShadowEngine, ShadowError, resolve_run
from upscale.services.shadow.report import compare_runs, shadow_report
from upscale.services.shadow.report import compare_text as compare_text
from upscale.services.shadow.report import text as report_text
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
    r.add_argument("--availability-policy", choices=AVAILABILITY_POLICIES, default=None,
                   help="market availability policy of a NEW run (default EVIDENCE_AWARE_V2: "
                        "missing evidence never closes a position; LEGACY_V1: closes it as "
                        "MARKET_UNAVAILABLE after the timeouts); an existing run keeps its own")  # fmt: skip
    r.add_argument("--execution-model", choices=EXECUTION_MODELS, default=None,
                   help="execution model of a NEW run (default IDEALIZED_NO_FEES; REALISTIC_V1: "
                        "latency, adverse slippage, fees, optional price impact; requires "
                        "EVIDENCE_AWARE_V2); an existing run keeps its own")  # fmt: skip
    defaults = ExecutionConfig()
    for flag, default, what in (
        ("--entry-fee-bps", defaults.entry_fee_bps, "entry fee"),
        ("--exit-fee-bps", defaults.exit_fee_bps, "exit fee"),
        ("--slippage-bps", defaults.slippage_bps, "adverse slippage per side"),
        ("--latency-seconds", defaults.latency_seconds, "fixed execution latency"),
        ("--max-price-impact-bps", defaults.max_price_impact_bps, "price impact cap"),
        ("--entry-max-wait-minutes", defaults.entry_max_wait_minutes,
         "cancel an unfilled entry intent after this long (0, the default: wait indefinitely)"),
    ):  # fmt: skip
        r.add_argument(flag, type=float, default=None,
                       help=f"REALISTIC_V1 {what} (default {default or 0:g})")  # fmt: skip
    r.add_argument("--price-impact", action="store_true",
                   help="REALISTIC_V1: constant-product price impact from observed liquidity")  # fmt: skip
    r.add_argument("--allow-contaminated", action="store_true",
                   help="allow --since before the clean-data cutoff (labeled research run)")  # fmt: skip
    r.add_argument("--diagnostics-detail", choices=DETAIL_MODES, default=None,
                   help="rejection rows to store for this step: full (one-off research), "
                        "sampled (default, UPSCALE_SHADOW_DIAGNOSTICS_DETAIL) or aggregate "
                        "(counters only); counts are exact in every mode")  # fmt: skip
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
    fn = sub.add_parser("funnel", help="sequential entry funnel and conditional counts")
    fn.add_argument("--run", default=None, help=f"run id (default: {DEFAULT_RUN_ID})")
    fn.add_argument("--strategy", action="append", default=None, help="repeatable")
    fn.add_argument("--since", type=_timestamp, default=None,
                    help="ISO-8601, inclusive (whole UTC hours)")  # fmt: skip
    fn.add_argument("--until", type=_timestamp, default=None,
                    help="ISO-8601, exclusive (whole UTC hours)")  # fmt: skip
    fn.add_argument("--json", action="store_true", help="print JSON")
    sg = sub.add_parser("storage", help="shadow database size and diagnostics storage")
    sg.add_argument("--json", action="store_true", help="print JSON")
    au = sub.add_parser("audit-unavailable",
                        help="classify every MARKET_UNAVAILABLE exit (read-only, no requests)")  # fmt: skip
    au.add_argument("--run", default=None, help=f"run id (default: {DEFAULT_RUN_ID})")
    au.add_argument("--strategy", action="append", default=None, help="repeatable")
    au.add_argument("--since", type=_timestamp, default=None, help="exit time, ISO-8601, inclusive")
    au.add_argument("--until", type=_timestamp, default=None, help="exit time, ISO-8601, exclusive")
    au.add_argument("--json", action="store_true", help="print JSON")
    au.add_argument("--collection-gap-minutes", type=float, default=AuditSettings.collection_gap_minutes,
                    help="largest gap between Scout runs counted as collection running")  # fmt: skip
    au.add_argument("--recovery-hours", type=float, default=AuditSettings.recovery_hours,
                    help="how long after an exit to look for the exact pool (retrospective)")  # fmt: skip
    rp = sub.add_parser("report",
                        help="strategy validation report (read-only, no requests)")  # fmt: skip
    rp.add_argument("--run", default=None, help=f"run id (default: {DEFAULT_RUN_ID})")
    rp.add_argument("--strategy", action="append", default=None, help="repeatable")
    rp.add_argument("--since", type=_timestamp, default=None, help="ISO-8601, inclusive")
    rp.add_argument("--until", type=_timestamp, default=None, help="ISO-8601, exclusive")
    rp.add_argument("--json", action="store_true", help="print JSON")
    cp = sub.add_parser("compare", help="factual deltas between two runs (read-only)")
    cp.add_argument("--base", required=True, help="e.g. continuous-v2")
    cp.add_argument("--other", required=True, help="e.g. continuous-v2-realistic")
    cp.add_argument("--strategy", action="append", default=None, help="repeatable")
    cp.add_argument("--since", type=_timestamp, default=None, help="ISO-8601, inclusive")
    cp.add_argument("--until", type=_timestamp, default=None, help="ISO-8601, exclusive")
    cp.add_argument("--json", action="store_true", help="print JSON")
    return p


_EXECUTION_FLAGS = ("entry_fee_bps", "exit_fee_bps", "slippage_bps", "latency_seconds",
                    "max_price_impact_bps", "entry_max_wait_minutes")  # fmt: skip


def _execution(args: argparse.Namespace, store: ShadowStore) -> ExecutionConfig | None:
    """The REALISTIC_V1 settings a `run` asks for (None: IDEALIZED_NO_FEES or continue)."""
    given = {k: getattr(args, k) for k in _EXECUTION_FLAGS if getattr(args, k) is not None}
    if args.price_impact:
        given["price_impact"] = True
    if args.execution_model != "REALISTIC_V1":
        if given:
            raise ShadowError("execution settings need --execution-model REALISTIC_V1")
        existing = store.run(args.run)
        if args.execution_model == "IDEALIZED_NO_FEES" and existing and run_execution(existing):
            raise ShadowError(f"run {args.run} is REALISTIC_V1: a run is never redefined")
        return None
    if given.get("entry_max_wait_minutes") == 0:
        given["entry_max_wait_minutes"] = None
    try:
        return ExecutionConfig(**given)
    except ValueError as exc:
        raise ShadowError(f"invalid execution settings: {exc}") from exc


def _audit(args: argparse.Namespace) -> int:
    """Both databases opened read-only (mode=ro): this command can never write either."""
    store = ShadowStore(args.db or default_shadow_db(), read_only=True)
    evidence = EvidenceStore(args.evidence_db or default_evidence_db(), read_only=True)
    try:
        settings = AuditSettings(collection_gap_minutes=args.collection_gap_minutes,
                                 recovery_hours=args.recovery_hours)  # fmt: skip
        d = audit_unavailable(store, evidence, resolve_run(store, args.run), args.strategy,
                              args.since, args.until, settings)  # fmt: skip
        print(json.dumps(d, indent=2, default=str) if args.json else audit_text(d))
        return 0
    except (ShadowError, ShadowStoreError, EvidenceStoreError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
        evidence.close()


def _report(args: argparse.Namespace) -> int:
    """Both databases opened read-only (mode=ro). Without an Evidence Archive file the
    report still runs; only the held-position availability section is unavailable."""
    store = ShadowStore(args.db or default_shadow_db(), read_only=True)
    path = args.evidence_db or default_evidence_db()
    evidence = EvidenceStore(path, read_only=True) if Path(path).expanduser().exists() else None
    try:
        d = shadow_report(store, evidence, resolve_run(store, args.run), args.strategy,
                          args.since, args.until)  # fmt: skip
        print(json.dumps(d, indent=2, default=str) if args.json else report_text(d))
        return 0
    except (ShadowError, ShadowStoreError, EvidenceStoreError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
        if evidence is not None:
            evidence.close()


def _compare(args: argparse.Namespace) -> int:
    store = ShadowStore(args.db or default_shadow_db(), read_only=True)
    path = args.evidence_db or default_evidence_db()
    evidence = EvidenceStore(path, read_only=True) if Path(path).expanduser().exists() else None
    try:
        d = compare_runs(store, evidence, args.base, args.other, args.strategy, args.since,
                         args.until)  # fmt: skip
        print(json.dumps(d, indent=2, default=str) if args.json else compare_text(d))
        return 0
    except (ShadowError, ShadowStoreError, EvidenceStoreError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
        if evidence is not None:
            evidence.close()


def _num(v: float | None, suffix: str = "%", digits: int = 1) -> str:
    return "-" if v is None else f"{v:.{digits}f}{suffix}"


def _funnel_text(d: dict[str, Any]) -> str:
    w = d["window"]
    lines = [
        f"run {d['run_id']}: diagnostics from {d['diagnostics_available_from'] or 'NOT AVAILABLE'}"
        f", aggregate counters from {d['aggregates_available_from'] or 'not yet'}",
        f"window: {w['since'] or 'start'} .. {w['until'] or 'now'}",
        d["note"],
    ]
    for s in d["strategies"]:
        lines += ["", f"{s['strategy']}  ({s['name']})",
                  f"  {'gate':<34}{'input':>8}{'passed':>8}{'failed':>8}"
                  f"{'% prev':>9}{'% total':>9}"]  # fmt: skip
        for row in s["funnel"]:
            lines.append(
                f"  {row['gate']:<34}{row['input']:>8}{row['passed']:>8}{row['failed']:>8}"
                f"{_num(row['pct_of_previous']):>9}{_num(row['pct_of_evaluated']):>9}"
            )
            for code, n in (row.get("failed_by_reason") or {}).items():
                lines.append(f"    - {code}: {n}")
        lines.append(f"  held (asset already open, not evaluated): {s['held']}")
        for c in s["conditional"]:
            if c["among"] == "all evaluated":
                continue
            lines.append(f"  among candidates {c['among']} ({c['base']}):")
            for ch in c["checks"]:
                given = f" (given {ch['given']})" if "given" in ch else ""
                lines.append(f"    {ch['gate']}: {ch['passed']}/{ch['of']} "
                             f"({_num(ch['pct'])}){given}")  # fmt: skip
            a = c["all_group_rules"]
            lines.append(f"    all {c['group']} rules: {a['passed']}/{a['of']} ({_num(a['pct'])})")
    return "\n".join(lines)


def _storage_text(d: dict[str, Any]) -> str:
    g, st = d["estimated_growth"], d["settings"]
    vol = d["volume"]
    lines = [
        f"shadow db: {d['shadow_db']}",
        f"size: {d['size_mb']:.2f} MB (reusable free pages {d['free_pages_mb']:.2f} MB)",
        f"volume: {vol['used_mb']:.0f} / {vol['total_mb']:.0f} MB used, "
        f"{vol['free_mb']:.0f} MB free" if vol else "volume: unknown",
        "rows by table:",
        *[f"  {t}: {n}" for t, n in d["rows"].items()],
        f"rejection detail rows: {d['rejection_detail_rows']}",
        f"aggregate counter rows: {d['aggregate_counter_rows']}",
        f"estimated growth ({g['basis']}):",
        f"  rejection detail: {_num(g['rejection_detail_mb_per_day'], ' MB/day', 3)}",
        f"  aggregate counters: {_num(g['aggregate_counters_mb_per_day'], ' MB/day', 3)}",
        f"  all tables: {_num(g['total_mb_per_day'], ' MB/day', 3)}"
        + (f", volume full in ~{g['days_until_volume_full']:.0f} days"
           if g["days_until_volume_full"] is not None else ""),
        f"diagnostics detail: {st['diagnostics_detail']} (sample {st['sample_per_reason_per_day']}"
        f"/strategy/primary reason/UTC day, {st['sample_per_reason_per_hour']}/hour)",
        "retention: " + (f"enabled, {st['retention_days']:g} days (sampled rows only)"
                         if st["retention_enabled"] else "disabled"),
        "runs:",
    ]  # fmt: skip
    for r in d["runs"]:
        lines.append(
            f"  {r['run_id']}: {r['rejection_rows']} rejection rows "
            f"({r['rejection_rows_protected']} protected, pre-counter), counters from "
            f"{r['aggregates_available_from'] or 'not yet'}, modes "
            f"{','.join(r['diagnostics_detail_modes']) or '-'}"
        )
    return "\n".join(lines)


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
    if args.command == "audit-unavailable":
        return _audit(args)
    if args.command == "report":
        return _report(args)
    if args.command == "compare":
        return _compare(args)
    store = ShadowStore(args.db or default_shadow_db())
    evidence_path = args.evidence_db or default_evidence_db()
    evidence = EvidenceStore(evidence_path, read_only=True)
    detail = getattr(args, "diagnostics_detail", None)
    engine = ShadowEngine(store, evidence, diagnostics=load_diagnostics_settings(detail))
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
                availability_policy=args.availability_policy,
                execution=_execution(args, store),
            )  # fmt: skip
            _print(engine.run(args.run))
        elif args.command == "status":
            _print(engine.status())
        elif args.command == "storage":
            d = engine.storage()
            print(json.dumps(d, indent=2, default=str) if args.json else _storage_text(d))
        else:
            run_id = resolve_run(store, args.run)
            common = {"run_id": run_id, "strategy_id": args.strategy, "asset_id": getattr(args, "asset", None),
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
            elif args.command == "funnel":
                d = engine.funnel(run_id, args.strategy, args.since, args.until)
                print(json.dumps(d, indent=2, default=str) if args.json else _funnel_text(d))
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
