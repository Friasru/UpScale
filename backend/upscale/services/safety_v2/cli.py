"""Safety V2 command line (Solana token-safety evidence: mint account and holders).

Local only (no provider request):

    python -m upscale.services.safety_v2 status [--json]
    python -m upscale.services.safety_v2 targets add --token <mint|solana:mint>
    python -m upscale.services.safety_v2 targets pin-pool --token <mint> --pool <address>
        [--dex raydium-amm-v4]
    python -m upscale.services.safety_v2 targets list [--json]
    python -m upscale.services.safety_v2 snapshot --token <mint> --no-fetch [--no-save]
    python -m upscale.services.safety_v2 show (--id N | --token <mint>) [--json]
    python -m upscale.services.safety_v2 rebuild --id N [--json]

Makes Solana provider requests (Safety V2's own daily budget; the count is printed):

    python -m upscale.services.safety_v2 collect --token <mint> [--no-holders] [--no-market]
        [--json]
    python -m upscale.services.safety_v2 snapshot --token <mint> [--no-holders] [--no-market]
        [--no-save] [--json]

A collection reads the mint account (1 request), unless ``--no-holders`` the holders (at
most 4 + UPSCALE_SAFETY_V2_HOLDER_MAX_PAGES requests) and, unless ``--no-market``, the
market (1 DEX request, only when UPSCALE_SAFETY_V2_DEX_URL is set, plus at most one
getAccountInfo(pool) when a previously reported pool went missing). Wallet proof is read
only from UPSCALE_SAFETY_V2_RADAR_DB, read-only.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from upscale.services.market_data import InvalidRequestError
from upscale.services.safety_v2.config import SafetySettings, load_settings
from upscale.services.safety_v2.models import SafetyError, solana_identity
from upscale.services.safety_v2.service import SafetyService, SnapshotResult, service_from_env

PROVIDER_NOTE = "MAKES SOLANA RPC AND DEX PROVIDER REQUESTS (Safety V2's own daily budget)"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.safety_v2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--db", default=None, help="database (default: UPSCALE_SAFETY_V2_DB)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("status", help="budget, cooldown, row counts (local only)")
    s.add_argument("--json", action="store_true")
    t = sub.add_parser("targets", help="manage targets (local only)")
    tsub = t.add_subparsers(dest="action", required=True)
    ta = tsub.add_parser("add")
    ta.add_argument("--token", required=True)
    tp = tsub.add_parser("pin-pool", help="pin an exact pool (its owner is a POOL_OR_VAULT)")
    tp.add_argument("--token", required=True)
    tp.add_argument("--pool", required=True)
    tp.add_argument("--dex", default=None, help="e.g. raydium-amm-v4 (corroborates its "
                    "shared vault authority)")  # fmt: skip
    tl = tsub.add_parser("list")
    tl.add_argument("--json", action="store_true")
    c = sub.add_parser("collect", help=f"one mint + holder + market observation. {PROVIDER_NOTE}")
    c.add_argument("--token", required=True)
    c.add_argument("--no-holders", action="store_true", help="skip the holder component")
    c.add_argument("--no-market", action="store_true", help="skip the market component")
    c.add_argument("--json", action="store_true")
    sn = sub.add_parser("snapshot", help=f"collect + snapshot. {PROVIDER_NOTE}")
    sn.add_argument("--token", required=True)
    sn.add_argument("--no-fetch", action="store_true", help="from stored data only: no request")
    sn.add_argument("--no-holders", action="store_true", help="skip the holder component")
    sn.add_argument("--no-market", action="store_true", help="skip the market component")
    sn.add_argument("--no-save", action="store_true", help="print the body without storing it")
    sn.add_argument("--json", action="store_true")
    sh = sub.add_parser("show", help="a stored snapshot (local only)")
    g = sh.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int)
    g.add_argument("--token")
    sh.add_argument("--json", action="store_true")
    r = sub.add_parser("rebuild", help="re-derive a stored snapshot (local only)")
    r.add_argument("--id", type=int, required=True)
    r.add_argument("--json", action="store_true")
    return p


def _out(data: Any, as_json: bool, text: str) -> None:
    print(json.dumps(data, sort_keys=True, indent=2, default=str) if as_json else text)


def _cid(raw: str) -> str:
    return solana_identity(raw)[0]


def _service(settings: SafetySettings) -> SafetyService:
    return service_from_env(settings)


def body_text(body: dict[str, Any]) -> str:
    a = body["assessment"]
    lines = [
        f"{body['identity']['canonical_id']} as of {body['as_of']}",
        f"identity: {body['identity']['token_mint']['status']} ({body['identity']['token_mint']['reason']})",
        f"band: {a['band']}  coverage: {a['coverage']}",
    ]
    for k, v in body["authority"].items():
        lines.append(
            f"  {k}: {v['status']} {json.dumps(v['value']) if v['value'] is not None else v['reason']}"
        )
    h = body.get("holders")
    if h:
        lines.append(f"holders: {h['status']} source={h['source']} fetched_at={h['fetched_at']}")
        for k in ("top1_pct", "top10_pct", "top10_wallet_pct", "holder_count"):
            v = h[k]
            bound = ">=" if v["lower_bound"] else ""
            lines.append(f"  {k}: {v['status']} "
                         f"{bound + str(v['value']) if v['value'] is not None else v['reason']}")  # fmt: skip
    m = body.get("market")
    if m:
        pool = m["primary_pool"]["value"]["address"] if m["primary_pool"]["value"] else None
        liq = m["primary_liquidity_usd"]
        lines.append(f"market: {m['status']} pool={pool} presence={m['pool_presence']['state']} "
                     f"liquidity={liq['value'] if liq['value'] is not None else liq['status']} "
                     f"({m['basis']})")  # fmt: skip
    for f in body["flags"]:
        lines.append(f"  [{f['outcome']}] {f['id']} ({f['severity']}): {f['reason']}")
    return "\n".join(lines)


def _result_dict(res: SnapshotResult) -> dict[str, Any]:
    return {"canonical_id": res.canonical_id, "saved": res.saved, "snapshot_id": res.snapshot_id,
            "body_hash": res.body_hash, "body": res.body,
            "collected": res.collected.__dict__ if res.collected else None}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    a = parser().parse_args(argv)
    settings = load_settings()
    if a.db:
        settings = settings.model_copy(update={"db_path": a.db})
    try:
        return _run(a, settings)
    except (InvalidRequestError, SafetyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(a: argparse.Namespace, settings: SafetySettings) -> int:
    svc = _service(settings)
    try:
        if a.command == "status":
            until = svc.guard.cooldown_until()
            st = {"db_path": settings.db_path, "provider": svc.provider.name if svc.provider else None,
                  "daily_request_budget": settings.daily_request_budget,
                  "holder_max_pages": settings.holder_max_pages,
                  "radar_db_configured": settings.radar_db_path is not None,
                  "market_provider": svc.dex.name if svc.dex else None,
                  "remaining_today": svc.guard.remaining_today(),
                  "cooldown_until": until.isoformat() if until else None,
                  "requests_today": svc.repo.requests_by_method(svc.guard.day()),
                  "rows": svc.repo.counts()}  # fmt: skip
            _out(st, a.json, json.dumps(st, indent=2))
        elif a.command == "targets":
            if a.action == "add":
                cid, added = svc.add_target(a.token)
                print(f"{cid}: {'added' if added else 'already a target'}")
            elif a.action == "pin-pool":
                pinned = svc.pin_pool(a.token, a.pool, a.dex)
                print(f"{a.pool}: {'pinned' if pinned else 'already pinned'}")
            else:
                rows: list[dict[str, Any]] = [
                    {"canonical_id": c, "source": s, "added_at": t}
                    for c, s, t in svc.repo.targets()
                ]
                _out(rows, a.json, "\n".join(r["canonical_id"] for r in rows) or "no targets")
        elif a.command == "collect":
            print(f"note: {PROVIDER_NOTE}", file=sys.stderr)
            got = asyncio.run(svc.collect(_cid(a.token), holders=not a.no_holders,
                                          market=not a.no_market))  # fmt: skip
            _out(got.__dict__, a.json, f"{got.outcome} (collection {got.collection_id}, "
                 f"observation {got.observation_id}, holders {got.holder_outcome}, "
                 f"market {got.market_outcome}, "
                 f"requests {got.requests})")  # fmt: skip
        elif a.command == "snapshot":
            if not a.no_fetch:
                print(f"note: {PROVIDER_NOTE}", file=sys.stderr)
            res = asyncio.run(svc.snapshot(_cid(a.token), fetch=not a.no_fetch, save=not a.no_save,
                                           holders=not a.no_holders,
                                           market=not a.no_market))  # fmt: skip
            _out(_result_dict(res), a.json, body_text(res.body) + f"\nhash: {res.body_hash}")
        elif a.command == "show":
            sid = a.id if a.id is not None else svc.repo.latest_snapshot_id(_cid(a.token))
            if sid is None:
                print("no snapshot yet", file=sys.stderr)
                return 1
            row = svc.repo.snapshot(sid)
            _out(row.body, a.json, body_text(row.body) + f"\nhash: {row.body_hash}")
        elif a.command == "rebuild":
            rb = svc.rebuild(a.id)
            _out(rb.__dict__, a.json, f"snapshot {rb.snapshot_id}: {rb.status}"
                 + "".join(f"\n  {k}: stored {s} != current {c}" for k, (s, c) in rb.mismatched.items()))  # fmt: skip
            return 0 if rb.status == "REPRODUCED" else 1
    finally:
        svc.repo.close()
    return 0
