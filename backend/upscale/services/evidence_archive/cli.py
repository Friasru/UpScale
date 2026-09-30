"""Point-in-Time Evidence Archive command line (read-only).

    python -m upscale.services.evidence_archive status [--days 7] [--json]
    python -m upscale.services.evidence_archive readiness [--replay-db PATH]
    python -m upscale.services.evidence_archive show --asset solana:<mint> --kind safety \\
        [--at 2026-10-01T12:00]

`show` returns the latest record observed at or before `--at` (default: now): exactly what
Replay Lab would be allowed to see at that time.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from upscale.config import EVIDENCE_DB_PATH
from upscale.services.evidence_archive.status import status
from upscale.services.evidence_archive.store import KINDS, EvidenceStore, EvidenceStoreError, Kind
from upscale.services.replay_lab.config import default_replay_db


def _at(value: str) -> datetime:
    at = datetime.fromisoformat(value)
    return at.replace(tzinfo=UTC) if at.tzinfo is None else at


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.evidence_archive")
    p.add_argument("--db", default=None, help="evidence archive (default: UPSCALE_EVIDENCE_DB)")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("status", "readiness"):
        s = sub.add_parser(name)
        s.add_argument("--days", type=float, default=7.0)
        s.add_argument("--replay-db", default=None)
        s.add_argument("--json", action="store_true")
    show = sub.add_parser("show", help="latest record at or before a time")
    show.add_argument("--asset", required=True, help="canonical id, e.g. solana:<mint>")
    show.add_argument("--kind", required=True, choices=KINDS)
    show.add_argument("--at", type=_at, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path = Path(args.db or EVIDENCE_DB_PATH).expanduser()
    if not path.exists():
        print(f"no evidence archive at {path} yet (it is created on the first archived record)")
        return 1
    store = EvidenceStore(path, read_only=True)
    try:
        now = datetime.now(UTC)
        if args.command == "show":
            at = args.at or now
            rec = store.latest(cast(Kind, args.kind), args.asset, until=at)
            if rec is None:
                print(
                    f"no {args.kind} evidence for {args.asset} observed at or before {at.isoformat()}"
                )
                return 1
            print(json.dumps({
                "kind": rec.kind, "asset_id": rec.asset_id, "pool": rec.pool_address,
                "provider": rec.provider, "component": rec.component,
                "observed_at": rec.observed_at.isoformat(), "archived_at": rec.archived_at.isoformat(),
                "availability": rec.availability, "reason": rec.reason, "versions": rec.versions,
                "links": rec.links, "payload": rec.payload,
            }, indent=2, default=str))  # fmt: skip
            return 0
        report = status(store, now, args.days, replay_db=args.replay_db or default_replay_db())
        if args.command == "readiness":
            report = {"label": report["label"], "readiness": report["readiness"]}
        if args.json or args.command == "readiness":
            print(json.dumps(report, indent=2, default=str))
            return 0
        cov = report["coverage"]
        print(report["label"])
        print(f"archive {report['database']}: {report['total_snapshots']} snapshots, "
              f"{report['distinct_assets']} assets, {report['snapshots_per_hour_last_24h']}/h "
              f"(last 24h); {report['oldest_snapshot']} .. {report['newest_snapshot']}; "
              f"last write {report['last_archive_write']}")  # fmt: skip
        print(f"by kind: {json.dumps(report['snapshots_by_kind'])}")
        print(f"coverage over {cov['scout_observations']} Scout observations (last {args.days:g} days): "
              f"market {cov['market_pct']}%, safety {cov['safety_pct']}%, social "
              f"{cov['social_pct']}%, decision-grade {cov['decision_grade_pct']}%")  # fmt: skip
        print(f"readiness: {json.dumps(report['readiness'], default=str)}")
        if report["missing_by_reason"]:
            print("missing evidence by reason:")
            for reason, n in report["missing_by_reason"].items():
                print(f"  {n:>6}  {reason}")
        if report["provider_failures"]:
            print(f"provider failures: {json.dumps(report['provider_failures'])}")
        return 0
    except EvidenceStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
