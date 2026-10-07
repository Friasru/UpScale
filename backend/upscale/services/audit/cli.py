"""Audit / Learning V1 command line (read-only: no provider request, no write anywhere).

python -m upscale.services.audit status [--json]
python -m upscale.services.audit report [--since ISO] [--until ISO] [--population scout|shadow]
    [--strategy scout_technical] [--run continuous-v2] [--horizon 4h] [--all-horizons] [--json]
python -m upscale.services.audit feature-audit --decision-id scout:<anchor id> | shadow:<decision id> [--json]

Databases default to ``UPSCALE_SCOUT_DB`` / ``UPSCALE_EVIDENCE_DB`` / ``UPSCALE_SHADOW_DB``
and are opened ``mode=ro``; a missing one is reported, never created. The Calibration
database is never opened.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from upscale.services.audit import sources
from upscale.services.audit.config import (
    HORIZONS,
    AuditConfig,
    default_evidence_db,
    default_scout_db,
    default_shadow_db,
)
from upscale.services.audit.dataset import Paths, load
from upscale.services.audit.report import build, feature_audit, feature_text, text, to_json
from upscale.services.shadow.config import parse_timestamp


def _timestamp(raw: str) -> datetime:
    try:
        return parse_timestamp(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.audit")
    p.add_argument("--scout-db", default=None, help="default: UPSCALE_SCOUT_DB (read-only)")
    p.add_argument("--evidence-db", default=None, help="default: UPSCALE_EVIDENCE_DB (read-only)")
    p.add_argument("--shadow-db", default=None, help="default: UPSCALE_SHADOW_DB (read-only)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("status", help="what Audit can read (row counts, time ranges)")
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("report", help="descriptive findings and hypotheses for Calibration")
    r.add_argument("--since", type=_timestamp, default=None, help="decision time, inclusive")
    r.add_argument("--until", type=_timestamp, default=None, help="decision time, exclusive")
    r.add_argument("--population", choices=("scout", "shadow"), default=None)
    r.add_argument("--strategy", default=None, help="Shadow strategy id (leaves out Scout anchors)")
    r.add_argument("--run", default=None, help="Shadow run id (leaves out Scout anchors)")
    r.add_argument("--horizon", choices=HORIZONS, default=None, help="primary horizon (default 4h)")
    r.add_argument("--all-horizons", action="store_true",
                   help="groupings at every horizon (default: the primary horizon only)")  # fmt: skip
    r.add_argument("--json", action="store_true")
    f = sub.add_parser("feature-audit", help="exactly which point-in-time fields one decision used")
    f.add_argument("--decision-id", required=True,
                   help="scout:<anchor id> or shadow:<Shadow decision id> (a bare id: Shadow)")  # fmt: skip
    f.add_argument("--json", action="store_true")
    return p


def _paths(a: argparse.Namespace) -> Paths:
    return Paths(
        scout=a.scout_db or default_scout_db(),
        evidence=a.evidence_db or default_evidence_db(),
        shadow=a.shadow_db or default_shadow_db(),
    )


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None


def status(paths: Paths) -> dict[str, Any]:
    out: dict[str, Any] = {"read_only": True, "databases": {}}
    spec = (
        ("scout", paths.scout, sources.SCOUT_TABLES,
         "SELECT MIN(anchored_at), MAX(anchored_at) FROM scout_outcome_observations"),
        ("evidence", paths.evidence, sources.EVIDENCE_TABLES,
         "SELECT MIN(observed_at), MAX(observed_at) FROM evidence_records"),
        ("shadow", paths.shadow, sources.SHADOW_TABLES,
         "SELECT MIN(decision_at), MAX(decision_at) FROM shadow_decisions"),
    )  # fmt: skip
    for name, path, names, span in spec:
        conn = sources.connect(path)
        if conn is None:
            out["databases"][name] = {"path": path, "found": False}
            continue
        try:
            counts = sources.row_counts(conn, names)
            first = last = None
            if names[0] in counts:
                first, last = conn.execute(span).fetchone()
            entry: dict[str, Any] = {"path": path, "found": True, "rows": counts,
                                     "from": _iso(first), "to": _iso(last)}  # fmt: skip
            if name == "evidence" and counts:
                entry["by_kind"] = dict(conn.execute(
                    "SELECT kind, COUNT(*) FROM evidence_records GROUP BY kind ORDER BY kind").fetchall())  # fmt: skip
            if name == "shadow":
                entry["runs"] = [
                    {"run_id": r["run_id"], "execution_model": r["execution_model"],
                     "clean_data": bool(r["clean_data"]), "since": _iso(r["since"])}
                    for r in sources.shadow_runs(conn)
                ]  # fmt: skip
                if "shadow_decisions" in counts:
                    entry["enter_decisions"] = dict(conn.execute(
                        "SELECT run_id || ' / ' || strategy_id, COUNT(*) FROM shadow_decisions "
                        "WHERE action = 'ENTER' GROUP BY run_id, strategy_id ORDER BY 1").fetchall())  # fmt: skip
            out["databases"][name] = entry
        finally:
            conn.close()
    return out


def status_text(s: dict[str, Any]) -> str:
    out = ["=== AUDIT STATUS (read-only) ==="]
    for name, d in s["databases"].items():
        if not d["found"]:
            out.append(f"{name}: NOT FOUND ({d['path']})")
            continue
        out.append(f"{name}: {d['path']}  ({d['from'] or '-'} .. {d['to'] or '-'})")
        out += [f"  {t}: {n}" for t, n in d["rows"].items()]
        for k, v in (d.get("by_kind") or {}).items():
            out.append(f"  evidence kind {k}: {v}")
        for r in d.get("runs", []):
            out.append(f"  run {r['run_id']} [{r['execution_model']}] since {r['since']}")
        for k, v in (d.get("enter_decisions") or {}).items():
            out.append(f"  ENTER {k}: {v}")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    a = parser().parse_args(argv)
    paths = _paths(a)
    if a.command == "status":
        s = status(paths)
        print(
            json.dumps(s, sort_keys=True, indent=2) if a.json else status_text(s),
            end="" if not a.json else "\n",
        )
        return 0
    if a.command == "report":
        cfg = AuditConfig(primary_horizon=a.horizon) if a.horizon else AuditConfig()
        ds = load(paths, cfg, a.since, a.until, a.population, a.strategy, a.run)
        filters = {
            "since": a.since.isoformat() if a.since else None,
            "until": a.until.isoformat() if a.until else None,
            "population": a.population, "strategy": a.strategy, "run": a.run,
        }  # fmt: skip
        r = build(ds, cfg, filters, all_horizons=a.all_horizons)
        print(to_json(r) if a.json else text(r), end="\n" if a.json else "")
        return 0
    raw = a.decision_id
    key = raw if raw.startswith(("scout:", "shadow:")) else f"shadow:{raw}"
    ds = load(paths, AuditConfig(text_features=True), only=key)
    if not ds.observations:
        print(f"no Scout anchor or Shadow ENTER decision {key!r} found", file=sys.stderr)
        return 1
    audit = feature_audit(ds.observations[0])
    if a.json:
        print(json.dumps(audit, sort_keys=True, indent=2, default=str))
    else:
        print(feature_text(audit), end="")
    return 0
