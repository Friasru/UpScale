"""Calibration Engine command line (research output only; never changes production).

python -m upscale.services.calibration status
python -m upscale.services.calibration readiness
python -m upscale.services.calibration analyze --horizon 1h
python -m upscale.services.calibration findings
python -m upscale.services.calibration create-candidates --horizon 1h
python -m upscale.services.calibration candidates
python -m upscale.services.calibration validate <candidate_id>
python -m upscale.services.calibration compare <candidate_id>
python -m upscale.services.calibration final-evaluate <candidate_id> --confirm-final-evaluation
python -m upscale.services.calibration origins --horizon 1h   (LIVE / REPLAY / SHADOW, separate)
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from upscale.services.calibration.config import (
    HORIZONS,
    ORIGINS,
    TimeFilter,
    default_calibration_db,
    default_live_db,
    default_replay_db,
    default_shadow_db,
    parse_timestamp,
)
from upscale.services.calibration.dataset import HoldoutSealedError
from upscale.services.calibration.engine import CalibrationEngine, CalibrationError
from upscale.services.calibration.store import CalibrationStore, CalibrationStoreError
from upscale.services.evidence_archive import hooks as evidence_hooks


def _timestamp(raw: str) -> datetime:
    try:
        return parse_timestamp(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _cohort(p: argparse.ArgumentParser) -> None:
    p.add_argument("--since", type=_timestamp, default=None,
                   help="only observations at or after this ISO-8601 time (LIVE_FORWARD by default)")  # fmt: skip
    p.add_argument(
        "--until", type=_timestamp, default=None, help="only observations before this time"
    )
    p.add_argument("--time-filter-all-origins", action="store_true",
                   help="apply --since/--until to replay samples too")  # fmt: skip
    p.add_argument("--origin", choices=("LIVE_FORWARD", "HISTORICAL_REPLAY"), default=None,
                   help="analyze only this origin")  # fmt: skip


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.calibration")
    p.add_argument("--db", default=None, help="calibration database (UPSCALE_CALIBRATION_DB)")
    p.add_argument("--live-db", default=None, help="live Scout / outcome database, read-only")
    p.add_argument("--replay-db", default=None, help="Replay Lab database, read-only")
    p.add_argument("--shadow-db", default=None, help="Shadow / Paper database, read-only")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    _cohort(sub.add_parser("readiness"))
    a = sub.add_parser("analyze")
    _cohort(a)
    a.add_argument("--horizon", default="1h", choices=HORIZONS)
    a.add_argument("--full", action="store_true", help="print the full results as JSON")
    f = sub.add_parser("findings")
    f.add_argument("--run", default=None)
    c = sub.add_parser("create-candidates")
    c.add_argument("--horizon", default="1h", choices=HORIZONS)
    _cohort(c)
    sub.add_parser("candidates")
    og = sub.add_parser("origins", help="LIVE_FORWARD / HISTORICAL_REPLAY / SHADOW, never merged")
    og.add_argument("--horizon", default="1h", choices=HORIZONS)
    _cohort(og)
    for name in ("validate", "compare"):
        v = sub.add_parser(name)
        v.add_argument("candidate_id")
        _cohort(v)
    fe = sub.add_parser(
        "final-evaluate", help="reads HOLDOUT: only with --confirm-final-evaluation"
    )
    fe.add_argument("candidate_id")
    fe.add_argument("--confirm-final-evaluation", action="store_true")
    mp = sub.add_parser("record-manual-promotion", help="records a human's manual promotion only")
    mp.add_argument("candidate_id")
    mp.add_argument("--confirm-manual-promotion", action="store_true")
    return p


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    evidence_hooks.install(None)  # a research process never archives production evidence
    store = CalibrationStore(args.db or default_calibration_db())
    try:
        time_filter = TimeFilter(
            since=getattr(args, "since", None),
            until=getattr(args, "until", None),
            origins=ORIGINS
            if getattr(args, "time_filter_all_origins", False)
            else ("LIVE_FORWARD",),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    engine = CalibrationEngine(
        store, args.live_db or default_live_db(), args.replay_db or default_replay_db(),
        time_filter=time_filter, origin=getattr(args, "origin", None),
        shadow_db=args.shadow_db or default_shadow_db(),
    )  # fmt: skip
    try:
        if args.command == "status":
            _print(engine.status())
        elif args.command == "readiness":
            _print(engine.readiness())
        elif args.command == "analyze":
            run_id, results = engine.analyze(args.horizon, {"cli": list(argv or sys.argv[1:])})
            if args.full:
                _print(results)
            else:
                print(results["label"])
                print(
                    f"run {run_id} (horizon {args.horizon}): dataset {json.dumps(results['dataset'], default=str)}"
                )
                print(f"by origin: {json.dumps(results['by_origin'], default=str)}")
                print(f"stages: {json.dumps(results['stages']['questions'])}")
                print(f"confidence: {results['confidence']['scout_score_verdict']} / "
                      f"{results['confidence']['opportunity_confidence_verdict']}")  # fmt: skip
                print(results["multiple_comparisons"]["warning"])
                for fnd in results["findings"]:
                    print(f"- [{fnd['strength']}] {fnd['statement']} ({fnd['rule']})")
                if not results["findings"]:
                    print("no finding meets the minimum sample and material-difference rules yet")
        elif args.command == "findings":
            _print(store.findings(args.run))
        elif args.command == "create-candidates":
            run_id, ids = engine.create_candidates(args.horizon)
            print(f"run {run_id}: {len(ids)} experimental candidate(s): {', '.join(ids) or 'none'}")
            print("Candidates are research only: production configuration is unchanged.")
        elif args.command == "candidates":
            _print(store.candidates())
        elif args.command == "origins":
            _print(engine.compare_origins(args.horizon))
        elif args.command == "validate":
            result = engine.validate(args.candidate_id)
            print(f"{args.candidate_id}: {result['outcome']}")
            _print(result["judgement"])
        elif args.command == "compare":
            _print(engine.compare(args.candidate_id))
        elif args.command == "final-evaluate":
            _print(engine.final_evaluate(args.candidate_id, args.confirm_final_evaluation))
        elif args.command == "record-manual-promotion":
            engine.record_manual_promotion(args.candidate_id, args.confirm_manual_promotion)
            print(
                f"{args.candidate_id}: recorded as PROMOTED_MANUALLY (no configuration was changed)"
            )
        return 0
    except (CalibrationError, HoldoutSealedError, CalibrationStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
