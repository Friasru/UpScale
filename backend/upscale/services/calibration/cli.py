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
"""

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from upscale.services.calibration.config import (
    HORIZONS,
    default_calibration_db,
    default_live_db,
    default_replay_db,
)
from upscale.services.calibration.dataset import HoldoutSealedError
from upscale.services.calibration.engine import CalibrationEngine, CalibrationError
from upscale.services.calibration.store import CalibrationStore, CalibrationStoreError
from upscale.services.evidence_archive import hooks as evidence_hooks


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upscale.services.calibration")
    p.add_argument("--db", default=None, help="calibration database (UPSCALE_CALIBRATION_DB)")
    p.add_argument("--live-db", default=None, help="live Scout / outcome database, read-only")
    p.add_argument("--replay-db", default=None, help="Replay Lab database, read-only")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("readiness")
    a = sub.add_parser("analyze")
    a.add_argument("--horizon", default="1h", choices=HORIZONS)
    a.add_argument("--full", action="store_true", help="print the full results as JSON")
    f = sub.add_parser("findings")
    f.add_argument("--run", default=None)
    c = sub.add_parser("create-candidates")
    c.add_argument("--horizon", default="1h", choices=HORIZONS)
    sub.add_parser("candidates")
    for name in ("validate", "compare"):
        sub.add_parser(name).add_argument("candidate_id")
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
    engine = CalibrationEngine(
        store, args.live_db or default_live_db(), args.replay_db or default_replay_db()
    )
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
