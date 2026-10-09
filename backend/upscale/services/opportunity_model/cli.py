"""Opportunity Model V1 command line (offline). ``decide`` reads the Evidence Archive and
Safety V2 read-only and stores one decision; ``show`` / ``explain`` / ``verify`` / ``status``
read only the Opportunity database (``verify --sources`` also re-reads the upstream stores,
read-only). Nothing here collects evidence, calls a provider or places an order.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any, TextIO

from upscale.services.opportunity_model.config import (
    evidence_db_path,
    opportunity_db_path,
    safety_db_path,
)
from upscale.services.opportunity_model.decision_models import OpportunityDecision, Reason
from upscale.services.opportunity_model.loaders import ArchiveReader, SafetyReader
from upscale.services.opportunity_model.models import OpportunityError
from upscale.services.opportunity_model.recorder import (
    Clock,
    record_from_paths,
    utc_now,
    verify_sources,
)
from upscale.services.opportunity_model.repository import OpportunityRepository, StoredDecision

NO_ORDER = "ENTER means entry conditions satisfied; no order was placed."


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.opportunity_model",
                                description="Opportunity Model V1 (offline decisions)")  # fmt: skip
    p.add_argument(
        "--db", default=None, help="Opportunity database (default: UPSCALE_OPPORTUNITY_DB)"
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decide", help="build, decide and store one decision (no collection)")
    d.add_argument("--asset", required=True, help="solana:<mint>")
    d.add_argument("--as-of", default=None, help="ISO time with zone (HISTORICAL_REPLAY)")
    d.add_argument("--historical", action="store_true", help="HISTORICAL_REPLAY (needs --as-of)")
    d.add_argument("--evidence-db", default=None, help="default: UPSCALE_EVIDENCE_DB")
    d.add_argument("--safety-db", default=None, help="default: UPSCALE_SAFETY_V2_DB")
    s = sub.add_parser("show", help="a stored decision (local only)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int)
    g.add_argument("--asset")
    e = sub.add_parser("explain", help="the stored structured reasons (local only)")
    e.add_argument("--id", type=int, required=True)
    v = sub.add_parser("verify", help="re-derive a stored decision from its stored input")
    v.add_argument("--id", type=int, required=True)
    v.add_argument("--sources", action="store_true",
                   help="also check the upstream stores still rebuild the input (read-only)")  # fmt: skip
    v.add_argument("--evidence-db", default=None)
    v.add_argument("--safety-db", default=None)
    sub.add_parser("status", help="database counts (local only)")
    return p


def _time(raw: str) -> datetime:
    t = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"--as-of {raw!r} needs a timezone")
    return t


def _short(h: str) -> str:
    return h[:16]


def _summary(out: TextIO, did: int, d: OpportunityDecision, input_hash: str) -> None:
    lines = [
        f"decision id:  {did}",
        f"asset:        {d.canonical_id}",
        f"decision_at:  {d.decision_at.isoformat()} ({d.origin})",
        f"decision:     {d.decision}" + (f"  [{', '.join(d.skip_basis)}]" if d.skip_basis else ""),
        f"quality:      {d.quality}",
        f"setup:        {d.setup_band}",
        f"technical:    {d.technical_band}",
        f"social:       {d.social_band}",
        f"risk:         {d.risk_tier} {list(d.risk_groups)}",
        f"vetoes:       {len(d.vetoes)}   blockers: {len(d.blockers)}   "
        f"ineligible: {len(d.ineligible)}",
        f"input hash:   {_short(input_hash)}",
        f"decision hash:{_short(d.decision_hash())}",
        NO_ORDER,
    ]  # fmt: skip
    out.write("\n".join(lines) + "\n")


def _show(out: TextIO, repo: OpportunityRepository, s: StoredDecision) -> None:
    d = s.decision
    _summary(out, s.id, d, s.input_hash)
    out.write(f"decided_at:   {s.decided_at.isoformat()}  run: {s.run_id}\n")
    out.write(f"pool:         {s.pool_address or '-'}\n")
    out.write(f"positive:     {list(d.positive_groups)}\n")
    out.write("safety ready: " + json.dumps(d.safety_entry_readiness, sort_keys=True) + "\n")
    out.write("sources:\n")
    for ref in repo.source_refs(s.id):
        extra = (f" snapshot {ref['safety_snapshot_id']} as_of {ref['safety_as_of']} rules "
                 f"{ref['safety_rules_version']}" if ref["source_name"] == "safety"
                 and ref["safety_snapshot_id"] else "")  # fmt: skip
        out.write(f"  {ref['source_name']:<18} {ref['status']:<13} "
                  f"{ref['record_id'] or '-'}{extra}\n")  # fmt: skip
    for title, reasons in (("veto", d.vetoes), ("ineligible", d.ineligible),
                           ("blocker", d.blockers)):  # fmt: skip
        for reason in reasons:
            out.write(f"  {title}: {reason.code}\n")
    out.write(f"full hashes:  input {s.input_hash}\n              decision {s.decision_hash}\n")


def _reason_line(r: Reason) -> str:
    where = f" [{r.source}{':' + r.source_record_id if r.source_record_id else ''}]"
    paths = f" ({', '.join(r.evidence_paths)})" if r.evidence_paths else ""
    return f"  - {r.code} {r.status}: {r.message}{where}{paths}"


def _explain(out: TextIO, s: StoredDecision) -> None:
    d = s.decision
    out.write(f"Decision\n  {d.decision} for {d.canonical_id} at {d.decision_at.isoformat()} "
              f"({d.origin})" + (f", skip basis {', '.join(d.skip_basis)}" if d.skip_basis
                                 else "") + "\n")  # fmt: skip
    out.write(f"Evidence quality\n  {d.quality}; Safety readiness "
              f"{json.dumps(d.safety_entry_readiness, sort_keys=True)}; setup {d.setup_band}, "
              f"technical {d.technical_band}, social {d.social_band}, risk {d.risk_tier}\n")  # fmt: skip
    for title, reasons in (
        ("Positive reasons", d.positive_reasons), ("Risks", d.risks), ("Vetoes", d.vetoes),
        ("Blockers", d.blockers), ("Missing evidence", d.missing_evidence),
        ("Ineligible reasons", d.ineligible),
    ):  # fmt: skip
        out.write(f"{title}\n")
        out.write("".join(_reason_line(r) + "\n" for r in reasons) or "  (none)\n")
    out.write("Upgrade path\n")
    out.write("".join(f"  - {u.code} [{u.gate}]: {u.message}\n" for u in d.upgrade_path)
              or "  (none)\n")  # fmt: skip
    if d.upgrade_path:
        out.write("  Every listed gate must pass; clearing one alone doesn't make ENTER.\n")
    out.write(NO_ORDER + "\n")


def main(
    argv: Sequence[str] | None = None, out: TextIO = sys.stdout, clock: Clock = utc_now
) -> int:
    args = _parser().parse_args(argv)
    db = args.db or opportunity_db_path()
    try:
        if args.command == "decide":
            if args.historical and not args.as_of:
                out.write("error: --historical needs --as-of\n")
                return 2
            if args.as_of and not args.historical:
                out.write("error: a live decision is made now; use --historical with --as-of\n")
                return 2
            repo = OpportunityRepository(db)
            try:
                res = record_from_paths(
                    repo, args.asset, "HISTORICAL_REPLAY" if args.historical else "LIVE_FORWARD",
                    args.evidence_db or evidence_db_path(), args.safety_db or safety_db_path(),
                    _time(args.as_of) if args.as_of else None, clock,
                )  # fmt: skip
            finally:
                repo.close()
            if not res.created:
                out.write(f"identical decision already stored (run {res.run_id})\n")
            _summary(out, res.decision_id, res.decision, res.decision.input_hash)
            return 0
        repo = OpportunityRepository(db, read_only=True)
        try:
            return _read_command(args, repo, out)
        finally:
            repo.close()
    except (OpportunityError, ValueError) as exc:
        out.write(f"error: {exc}\n")
        return 2


def _read_command(args: argparse.Namespace, repo: OpportunityRepository, out: TextIO) -> int:
    if args.command == "show":
        stored = repo.get(args.id) if args.id is not None else repo.latest_for(args.asset)
        if stored is None:
            out.write(f"no decision for {args.asset}\n")
            return 1
        _show(out, repo, stored)
        return 0
    if args.command == "explain":
        _explain(out, repo.get(args.id))
        return 0
    if args.command == "verify":
        res = repo.verify_decision(args.id)
        out.write(f"decision {res.decision_id}: {res.status} - {res.detail}\n")
        for key in res.differing_fingerprints:
            out.write(f"  changed: {key}\n")
        code = 0 if res.status == "REPRODUCED" else 1
        if args.sources:
            archive = ArchiveReader(args.evidence_db or evidence_db_path())
            safety = SafetyReader(args.safety_db or safety_db_path())
            try:
                src = verify_sources(repo, args.id, archive, safety)
            finally:
                archive.close()
                safety.close()
            out.write(f"sources: {src.status} - {src.detail}"
                      + (f" ({', '.join(src.sources)})" if src.sources else "") + "\n")  # fmt: skip
        return code
    st: dict[str, Any] = repo.status()
    for key, value in st.items():
        out.write(
            f"{key}: {json.dumps(value, sort_keys=True) if isinstance(value, dict) else value}\n"
        )
    return 0
