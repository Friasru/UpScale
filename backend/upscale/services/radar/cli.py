"""Radar V1 command line (Solana; descriptive wallet / holder intelligence; CLI-only).

Local only (no provider request):

    python -m upscale.services.radar status [--json]
    python -m upscale.services.radar targets list [--status ACTIVE] [--json]
    python -m upscale.services.radar targets add --token <mint> --pool <pool> [--dex D]
        [--pool-created-at ISO]
    python -m upscale.services.radar targets sync-scout [--scout-db PATH] [--since ISO]
        [--limit 10]                      (Scout database opened read-only)
    python -m upscale.services.radar targets deactivate --token <mint>
    python -m upscale.services.radar inspect --token <mint> [--json]
    python -m upscale.services.radar wallet --address <wallet> [--scout-db PATH] [--json]
    python -m upscale.services.radar retention-status [--json]
    python -m upscale.services.radar cleanup --dry-run [--json]

Makes Solana provider requests (Radar's own budget; the count is printed):

    python -m upscale.services.radar snapshot --token <mint> [--json]
    python -m upscale.services.radar snapshot --token <mint> --no-fetch   (local preview)
    python -m upscale.services.radar collect [--limit 5] [--json]

Deletes Radar rows (Radar's database only):

    python -m upscale.services.radar cleanup [--json]
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from upscale.services.market_data import InvalidRequestError
from upscale.services.radar import retention
from upscale.services.radar.config import SNAPSHOT_SCHEMA, RadarSettings, load_settings
from upscale.services.radar.models import iso, parse_canonical, solana_identity, ts
from upscale.services.radar.readonly import default_scout_db, parse_timestamp
from upscale.services.radar.service import RadarService, SnapshotResult
from upscale.services.radar.wallet_history import describe_wallet

PROVIDER_NOTE = "MAKES SOLANA PROVIDER REQUESTS (Radar's own daily budget)"


def _timestamp(raw: str) -> datetime:
    try:
        return parse_timestamp(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.radar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--db", default=None, help="Radar database (default: UPSCALE_RADAR_DB)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("status", help="settings, budget, cooldown, row counts (local only)")
    s.add_argument("--json", action="store_true")

    t = sub.add_parser("targets", help="manage Radar targets (local only)")
    tsub = t.add_subparsers(dest="action", required=True)
    tl = tsub.add_parser("list")
    tl.add_argument("--status", default=None)
    tl.add_argument("--json", action="store_true")
    ta = tsub.add_parser("add")
    ta.add_argument("--token", required=True)
    ta.add_argument("--pool", required=True)
    ta.add_argument("--dex", default=None)
    ta.add_argument("--pool-created-at", type=_timestamp, default=None)
    tss = tsub.add_parser("sync-scout", help="add Solana tokens Scout anchored (read-only)")
    tss.add_argument("--scout-db", default=None, help="default: UPSCALE_SCOUT_DB (read-only)")
    tss.add_argument("--since", type=_timestamp, default=None)
    tss.add_argument("--limit", type=int, default=10)
    td = tsub.add_parser("deactivate")
    td.add_argument("--token", required=True)

    i = sub.add_parser("inspect", help="stored target state and latest snapshot (local only)")
    i.add_argument("--token", required=True)
    i.add_argument("--json", action="store_true")

    sn = sub.add_parser("snapshot", help=f"collect + snapshot one target. {PROVIDER_NOTE}")
    sn.add_argument("--token", required=True)
    sn.add_argument("--no-fetch", action="store_true",
                    help="local preview from stored data: no request, nothing saved")  # fmt: skip
    sn.add_argument("--json", action="store_true")
    co = sub.add_parser("collect", help=f"snapshot active targets. {PROVIDER_NOTE}")
    co.add_argument("--limit", type=int, default=5)
    co.add_argument("--json", action="store_true")

    w = sub.add_parser("wallet", help="descriptive wallet history (local only)")
    w.add_argument("--address", required=True)
    w.add_argument("--scout-db", default=None, help="for outcome labels (read-only)")
    w.add_argument("--json", action="store_true")

    r = sub.add_parser("retention-status", help="Radar retention plan (read-only)")
    r.add_argument("--json", action="store_true")
    c = sub.add_parser("cleanup", help="delete expired Radar rows (or --dry-run)")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--json", action="store_true")
    return p


def _out(data: Any, as_json: bool, text: str) -> None:
    print(json.dumps(data, sort_keys=True, indent=2, default=str) if as_json else text)


def _cid(raw: str) -> str:
    _, mint = parse_canonical(raw)
    return solana_identity(mint)[0]


def _status(settings: RadarSettings, svc: RadarService) -> dict[str, Any]:
    until = svc.guard.cooldown_until()
    today = svc.guard.day()
    week = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
    return {
        "db": svc.repo.path,
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "background_enabled": settings.enabled,
        "background_note": "V1 is CLI-only: no background loop exists",
        "provider": svc.provider.name if svc.provider else None,
        "budget": {
            "daily": settings.daily_request_budget,
            "used_today": svc.guard.used_today(),
            "remaining_today": svc.guard.remaining_today(),
            "day": today,
            "max_rps": settings.max_rps,
            "concurrency": settings.concurrency,
        },  # fmt: skip
        "cooldown_until": until.isoformat() if until else None,
        "optional_enrichment": {
            "deep_backfill": settings.deep_backfill,
            "wallet_age": settings.wallet_age,
            "first_funder": settings.first_funder,
            "verify_deployer": settings.verify_deployer,
        },  # fmt: skip
        "requests_last_7_days": [
            {
                "day": d,
                "method": m,
                "calls": c,
                "ok": ok,
                "rate_limited": rl,
                "timeouts": to,
                "failures": f,
            }
            for d, m, c, ok, rl, to, f in svc.repo.requests_by_day(week)
        ],  # fmt: skip
        "rows": svc.repo.row_counts(),
    }


def _status_text(s: dict[str, Any]) -> str:
    b = s["budget"]
    lines = [
        "=== RADAR STATUS (local only) ===",
        f"db: {s['db']}  schema: {s['snapshot_schema']}",
        f"background: {'on' if s['background_enabled'] else 'off'} ({s['background_note']})",
        f"provider: {s['provider'] or 'NOT CONFIGURED'}",
        f"budget {b['day']}: {b['used_today']}/{b['daily']} used, {b['remaining_today']} left "
        f"(max {b['max_rps']} rps, concurrency {b['concurrency']})",
        f"cooldown until: {s['cooldown_until'] or '-'}",
        "optional: " + ", ".join(f"{k}={'on' if v else 'off'}" for k, v in s["optional_enrichment"].items()),
    ]  # fmt: skip
    lines += [f"  {t}: {n}" for t, n in s["rows"].items()]
    return "\n".join(lines)


def _metric(m: dict[str, Any]) -> str:
    if m["value"] is None:
        return f"{m['status']}" + (f" ({m['reason']})" if m.get("reason") else "")
    bound = "≥" if m.get("lower_bound") else ""
    return f"{bound}{m['value']} [{m['status']}]"


def snapshot_text(body: dict[str, Any]) -> str:
    h, a, e = body["holders"], body["activity"], body["early_activity"]
    lg, rp, cl, cr = body["large_holders"], body["repeated"], body["clusters"], body["creators"]
    lines = [
        f"=== RADAR SNAPSHOT {body['schema_version']} ===",
        f"{body['identity']['canonical_id']}  pool {body['identity']['pool_address']} "
        f"({body['identity']['dex'] or '?'})",
        f"observed_at: {body['observed_at']}  headline coverage: {body['coverage']['overall']}  "
        f"holder source: {h['source'] or '-'}",
        f"holder_count: {_metric(h['holder_count'])}",
        f"top1_pct: {_metric(h['top1_pct'])}   top10_pct: {_metric(h['top10_pct'])}",
        f"top10_change_pp: {_metric(h['top10_change_pp'])}   holder_count_change: "
        f"{_metric(h['holder_count_change'])}",
        f"large holders: accumulation {_metric(lg['large_holder_accumulation_count'])}, "
        f"reduction {_metric(lg['large_holder_reduction_count'])}",
        f"large wallets: accumulation {_metric(lg['large_wallet_accumulation_count'])}, "
        f"reduction {_metric(lg['large_wallet_reduction_count'])}, exit "
        f"{_metric(lg['large_wallet_exit_count'])}",
        f"activity since {a['window_since'] or 'tracking start'}: {a['transactions_parsed']} tx "
        f"parsed, {a['transactions_skipped']} skipped",
        f"  interacting_wallets: {_metric(a['interacting_wallets'])}",
        f"  TOKEN_INFLOW wallets: {_metric(a['token_inflow_wallets'])}   TOKEN_OUTFLOW wallets: "
        f"{_metric(a['token_outflow_wallets'])}",
        f"  net_inflow_wallets: {_metric(a['net_inflow_wallets'])}",
        f"early (window end {e['window_end'] or '?'}): wallets {_metric(e['early_wallet_count'])}, "
        f"repeated early {_metric(e['repeated_early_wallet_count'])}",
        f"repeated wallets: {_metric(rp['repeated_wallet_count'])}, REPEATED_WALLET_GROUP: "
        f"{_metric(rp['repeated_wallet_group_count'])}",
        f"COORDINATED_TIMING_PATTERN: {_metric(cl['coordinated_timing']['count'])}   "
        f"FUNDING_CLUSTER: {_metric(cl['funding']['count'])}",
    ]  # fmt: skip
    for key in ("pool_creator_candidate", "token_deployer"):
        c = cr[key]
        lines.append(f"{key.upper()}: {c['status']} {c.get('identity') or ''}"
                     f" ({c.get('method') or c.get('reason') or ''})")  # fmt: skip
    return "\n".join(lines)


def _result_dict(r: SnapshotResult) -> dict[str, Any]:
    return {"canonical_id": r.canonical_id, "observed_at": r.observed_at.isoformat(),
            "provider_requests": r.requests, "saved": r.saved, "snapshot_id": r.snapshot_id,
            "body_hash": r.body_hash, "steps": {k: v.as_dict() for k, v in r.steps.items()},
            "snapshot": r.body}  # fmt: skip


def _result_text(r: SnapshotResult) -> str:
    steps = ", ".join(f"{k}={v.status}({v.requests})" for k, v in r.steps.items())
    head = (f"provider requests: {r.requests}  collection steps: {steps or 'none (local preview)'}  "
            f"saved: {r.saved}")  # fmt: skip
    return head + "\n" + snapshot_text(r.body)


def _service(settings: RadarSettings) -> RadarService:
    from upscale import config  # loads .env and installs secret-safe logging

    return RadarService(
        settings, helius_api_key=config.HELIUS_API_KEY, rpc_url=config.SOLANA_RPC_URL
    )


def _retention(settings: RadarSettings, dry_run: bool) -> dict[str, Any]:
    path = Path(settings.db_path).expanduser()
    if not path.is_file():
        return {"db": str(path), "found": False, "tables": []}
    now = datetime.now(UTC)
    if dry_run:
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        try:
            rows = retention.plan(conn, settings, now)
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            rows = retention.cleanup(conn, settings, now)
        finally:
            conn.close()
    return {"db": str(path), "found": True, "dry_run": dry_run, "size_bytes": path.stat().st_size,
            "tables": rows}  # fmt: skip


def _retention_text(r: dict[str, Any]) -> str:
    if not r["found"]:
        return f"Radar database not found: {r['db']}"
    lines = [f"=== RADAR RETENTION ({'dry run' if r['dry_run'] else 'cleanup'}) {r['db']} "
             f"({r['size_bytes']:,} bytes) ==="]  # fmt: skip
    for t in r["tables"]:
        if t.get("skipped"):
            lines.append(f"  {t['table']:<24} skipped: {t['skipped']}")
            continue
        n = t["eligible"] if r["dry_run"] else t.get("deleted", 0)
        verb = "eligible" if r["dry_run"] else "deleted"
        lines.append(f"  {t['table']:<24} {t['retention_days']}d cutoff {t['cutoff']}: {n} {verb} "
                     f"of {t['older_than_cutoff']} older ({t['total_rows']} total)")  # fmt: skip
        lines += [f"      protected {k:>6}  {reason}" for reason, k in t["protected"].items()]
    lines.append("kept forever: radar_targets, radar_wallet_entries, radar_wallets, radar_creators")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    a = parser().parse_args(argv)
    settings = load_settings()
    if a.db:
        settings = settings.model_copy(update={"db_path": a.db})
    try:
        return _run(a, settings)
    except InvalidRequestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(a: argparse.Namespace, settings: RadarSettings) -> int:
    if a.command == "retention-status" or (a.command == "cleanup" and a.dry_run):
        r = _retention(settings, dry_run=True)
        _out(r, a.json, _retention_text(r))
        return 0
    if a.command == "cleanup":
        r = _retention(settings, dry_run=False)
        _out(r, a.json, _retention_text(r))
        return 0
    svc = _service(settings)
    try:
        if a.command == "status":
            st = _status(settings, svc)
            _out(st, a.json, _status_text(st))
        elif a.command == "targets":
            return _targets(a, svc)
        elif a.command == "inspect":
            cid = _cid(a.token)
            target = svc.repo.get_target(cid)
            if target is None:
                print(f"{cid} is not a Radar target", file=sys.stderr)
                return 1
            latest = svc.repo.snapshot_as_of(cid, ts(datetime.now(UTC)))
            info: dict[str, Any] = {"target": target.__dict__, "snapshot_times": [iso(t) for t in svc.repo.snapshot_times(cid)],
                    "latest_snapshot": latest[1] if latest else None}  # fmt: skip
            _out(info, a.json, f"target: {target}\nsnapshots: {len(info['snapshot_times'])}\n"
                 + (snapshot_text(latest[1]) if latest else "no snapshot yet"))  # fmt: skip
        elif a.command == "snapshot":
            if not a.no_fetch:
                print(f"note: {PROVIDER_NOTE}", file=sys.stderr)
            res = asyncio.run(svc.snapshot(_cid(a.token), fetch=not a.no_fetch))
            _out(_result_dict(res), a.json, _result_text(res))
        elif a.command == "collect":
            print(f"note: {PROVIDER_NOTE}", file=sys.stderr)
            results = asyncio.run(svc.snapshot_active(a.limit))
            rows = [_result_dict(r) if isinstance(r, SnapshotResult) else {"canonical_id": r[0], "error": r[1]}
                    for r in results]  # fmt: skip
            text = "\n\n".join(_result_text(r) if isinstance(r, SnapshotResult) else f"{r[0]}: {r[1]}"
                               for r in results) or "no active targets"  # fmt: skip
            _out(rows, a.json, text + f"\nrequests used this run: {svc.guard.used_this_run}")
        elif a.command == "wallet":
            w = describe_wallet(
                svc.repo, settings, a.address.strip(), datetime.now(UTC), a.scout_db
            )
            _out(w, a.json, json.dumps(w, indent=2, default=str))
    finally:
        svc.repo.close()
    return 0


def _targets(a: argparse.Namespace, svc: RadarService) -> int:
    if a.action == "list":
        rows = svc.repo.targets(a.status)
        _out([t.__dict__ for t in rows], a.json,
             "\n".join(f"{t.canonical_id} pool={t.pool_address} dex={t.dex} source={t.source} "
                       f"status={t.status} snapshots={t.snapshots_taken} early={t.early_status}"
                       for t in rows) or "no targets")  # fmt: skip
    elif a.action == "add":
        cid, added = svc.add_target(a.token, a.pool, a.dex, a.pool_created_at)
        print(f"{cid}: {'added' if added else 'already a target (unchanged)'}")
    elif a.action == "sync-scout":
        r = svc.sync_from_scout(a.scout_db or default_scout_db(), a.since, a.limit)
        print(json.dumps(r, indent=2))
    elif a.action == "deactivate":
        ok = svc.repo.set_target_status(_cid(a.token), "INACTIVE")
        print("deactivated" if ok else "not a target")
        return 0 if ok else 1
    return 0
