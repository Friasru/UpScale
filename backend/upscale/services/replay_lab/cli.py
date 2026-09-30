"""Replay Lab command line.

    python -m upscale.services.replay_lab run --start 2026-09-28 --end 2026-09-29 \\
        --chains solana --max-samples 20 --min-spacing-minutes 60 --mode MARKET_ONLY
    python -m upscale.services.replay_lab status [JOB_ID]
    python -m upscale.services.replay_lab summary [--horizon 1h] [--group-by stage]
    python -m upscale.services.replay_lab resume JOB_ID

Replay data goes to ``UPSCALE_REPLAY_DB`` (default ``~/.upscale/replay.sqlite3``); the Scout
database is only read (``--archive``, default ``UPSCALE_SCOUT_DB``). Replay pauses while a
local UpScale backend is running (``--backend-url``): it has the lower priority.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from upscale.config import GROWTH_CONFIG, OUTCOME_CONFIG, SCOUT_CONFIG
from upscale.services.evidence_archive import hooks as evidence_hooks
from upscale.services.evidence_archive.store import EvidenceStore
from upscale.services.outcomes.config import load_outcome_config
from upscale.services.replay_lab.analytics import GROUP_BY, as_table, summarize
from upscale.services.replay_lab.archive import ArchiveUnavailableError, ScoutArchive
from upscale.services.replay_lab.candles import HistoricalCandleFetcher
from upscale.services.replay_lab.clock import LookaheadError
from upscale.services.replay_lab.config import (
    SPLITS,
    AssetSpec,
    Mode,
    ReplayJobConfig,
    Split,
    SplitConfig,
    default_archive_db,
    default_evidence_db,
    default_replay_db,
)
from upscale.services.replay_lab.engine import JobLockedError, ReplayRunner
from upscale.services.replay_lab.findings import generate
from upscale.services.replay_lab.models import NOT_A_BACKTEST
from upscale.services.replay_lab.quota import (
    DEFAULT_BACKEND_URL,
    LocalBackendProbe,
    ReplayGate,
    production_limiter,
)
from upscale.services.replay_lab.shadow import evaluate_baselines
from upscale.services.replay_lab.store import JobRow, ReplayStore
from upscale.services.scout.config import load_scout_config
from upscale.services.scout.growth.config import load_growth_config


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m upscale.services.replay_lab", description=NOT_A_BACKTEST
    )
    p.add_argument(
        "--db",
        default=None,
        help="replay database (default: UPSCALE_REPLAY_DB or ~/.upscale/replay.sqlite3)",
    )
    p.add_argument("--archive", default=None, help="Scout database to read (read-only)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan a new replay job and run it")
    run.add_argument("--start", type=_date, required=True)
    run.add_argument("--end", type=_date, required=True)
    run.add_argument("--chains", "--chain", default="solana", help="comma-separated chains")
    run.add_argument("--max-samples", type=int, default=100)
    run.add_argument("--min-spacing-minutes", type=float, default=60.0)
    run.add_argument("--max-per-asset", type=int, default=10)
    run.add_argument("--mode", choices=("MARKET_ONLY", "MARKET_PLUS_SOCIAL"), default="MARKET_ONLY")
    run.add_argument("--evidence", choices=("RECORDED", "CANDLES"), default="RECORDED",
                     help="RECORDED: Scout's stored snapshots (full evidence); CANDLES: pool candles only")  # fmt: skip
    run.add_argument(
        "--asset", action="append", default=[], help="chain:token or chain:token:pool (repeatable)"
    )
    run.add_argument("--provider", choices=("GeckoTerminal",), default="GeckoTerminal")
    run.add_argument("--split-strategy", choices=("time",), default="time")
    run.add_argument("--calibration-pct", type=float, default=70.0)
    run.add_argument("--validation-pct", type=float, default=15.0)
    run.add_argument("--holdout-pct", type=float, default=15.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--plan-only", action="store_true", help="plan and store the job; don't run it"
    )
    _runner_flags(run)

    resume = sub.add_parser("resume", help="continue a paused / interrupted job")
    resume.add_argument("job_id")
    _runner_flags(resume)

    status = sub.add_parser("status", help="jobs and their progress")
    status.add_argument("job_id", nargs="?")

    summary = sub.add_parser("summary", help="grouped outcome statistics (excludes HOLDOUT)")
    summary.add_argument("--horizon", default="1h")
    summary.add_argument("--group-by", default="stage", choices=GROUP_BY)
    summary.add_argument("--split", default="CALIBRATION,VALIDATION", help="comma-separated splits")
    summary.add_argument("--job", action="append", default=[])
    summary.add_argument("--baselines", action="store_true", help="compare simple baseline rules")
    summary.add_argument("--findings", action="store_true",
                         help="derive EXPERIMENTAL calibration findings (CALIBRATION only) and store them")  # fmt: skip
    summary.add_argument("--include-holdout", action="store_true")
    summary.add_argument("--final-evaluation", action="store_true",
                         help="required with --include-holdout: HOLDOUT is for the final evaluation only")  # fmt: skip
    summary.add_argument("--json", action="store_true")
    return p


def _runner_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend-url", default=DEFAULT_BACKEND_URL,
                   help="replay pauses while an UpScale backend answers here (it has priority)")  # fmt: skip
    p.add_argument("--exit-on-pause", action="store_true",
                   help="stop (job stays PAUSED) instead of sleeping when production needs the provider")  # fmt: skip
    p.add_argument("--evidence-db", default=None,
                   help="Point-in-Time Evidence Archive to read (default: UPSCALE_EVIDENCE_DB or "
                        "evidence.sqlite3 next to the Scout database)")  # fmt: skip
    p.add_argument("--safety-max-age-minutes", type=float, default=60.0,
                   help="archived on-chain safety older than this before T is not used")  # fmt: skip


def _store(args: argparse.Namespace) -> ReplayStore:
    archive = args.archive or default_archive_db()
    return ReplayStore(args.db or default_replay_db(), forbidden=[archive, default_archive_db()])


def _archive(args: argparse.Namespace) -> ScoutArchive | None:
    path = Path(args.archive or default_archive_db()).expanduser()
    return ScoutArchive(path) if path.exists() else None


def _evidence(args: argparse.Namespace) -> EvidenceStore | None:
    path = Path(args.evidence_db or default_evidence_db()).expanduser()
    return EvidenceStore(path, read_only=True) if path.exists() else None


def _runner(args: argparse.Namespace, store: ReplayStore) -> ReplayRunner:
    scout_config = load_scout_config(SCOUT_CONFIG)
    outcome_config = load_outcome_config(OUTCOME_CONFIG)
    gate = ReplayGate(
        production_limiter(scout_config.geckoterminal), LocalBackendProbe(args.backend_url)
    )
    return ReplayRunner(
        store,
        _archive(args),
        HistoricalCandleFetcher(store, gate),
        scout_config,
        load_growth_config(GROWTH_CONFIG),
        collapse=outcome_config.collapse,
        collector=outcome_config.collector,
        log=lambda line: print(line, flush=True),
        evidence=_evidence(args),
        safety_max_age=timedelta(minutes=args.safety_max_age_minutes),
    )


def _print_job(job: JobRow) -> None:
    c = job.counts
    done = c.get("COMPLETE", 0) + c.get("SKIPPED", 0) + c.get("FAILED", 0)
    print(f"{job.job_id}  {job.status}{f' ({job.status_reason})' if job.status_reason else ''}")
    print(
        f"  samples: {job.planned} planned, {done} attempted, {c.get('COMPLETE', 0)} complete, "
        f"{c.get('SKIPPED', 0)} skipped, {c.get('FAILED', 0)} failed, "
        f"{c.get('DECIDED', 0)} decided awaiting outcomes, {c.get('PLANNED', 0)} pending"
    )
    print(f"  splits: {json.dumps(job.splits)}  window: {job.config.start:%Y-%m-%d %H:%M} .. "
          f"{job.config.end:%Y-%m-%d %H:%M} UTC  mode: {job.config.mode}  evidence: {job.config.evidence}")  # fmt: skip
    if job.provider_usage:
        print(f"  provider usage: {json.dumps(job.provider_usage)}")
    planning = {k: v for k, v in job.planning.items() if k != "versions"}
    print(f"  planning: {json.dumps(planning)}")


async def _run(runner: ReplayRunner, store: ReplayStore, job_id: str, exit_on_pause: bool) -> int:
    try:
        result = await runner.run(job_id, exit_on_pause=exit_on_pause)
    except JobLockedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except LookaheadError as exc:
        print(f"LOOKAHEAD DETECTED, job failed: {exc}", file=sys.stderr)
        return 2
    _print_job(result.job)
    if result.paused:
        print(f"paused: {result.pause_reason}. Continue with: resume {job_id}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # A replay process never archives "evidence": it only reads the past.
    evidence_hooks.install(None)
    store = _store(args)
    try:
        if args.command == "run":
            config = ReplayJobConfig(
                start=args.start,
                end=args.end,
                chains=tuple(c.strip() for c in args.chains.split(",") if c.strip()),
                mode=cast(Mode, args.mode),
                evidence=args.evidence,
                max_samples=args.max_samples,
                min_spacing_minutes=args.min_spacing_minutes,
                max_per_asset=args.max_per_asset,
                assets=tuple(AssetSpec.parse(a) for a in args.asset),
                split=SplitConfig(
                    strategy=args.split_strategy,
                    calibration_pct=args.calibration_pct,
                    validation_pct=args.validation_pct,
                    holdout_pct=args.holdout_pct,
                ),
                provider=args.provider,
                horizons=load_outcome_config(OUTCOME_CONFIG).horizons,
                seed=args.seed,
            )
            runner = _runner(args, store)
            job_id = runner.create_job(config)
            job = store.job(job_id)
            assert job is not None
            print(f"planned {job_id}")
            _print_job(job)
            if args.plan_only or job.planned == 0:
                if job.planned == 0:
                    store.set_job_status(job_id, "COMPLETE", "no samples matched")
                return 0
            return asyncio.run(_run(runner, store, job_id, args.exit_on_pause))
        if args.command == "resume":
            runner = _runner(args, store)
            return asyncio.run(_run(runner, store, args.job_id, args.exit_on_pause))
        if args.command == "status":
            ids = [args.job_id] if args.job_id else store.job_ids()
            if not ids:
                print("no replay jobs yet")
            for job_id in ids:
                job = store.job(job_id)
                if job is None:
                    print(f"unknown job {job_id}", file=sys.stderr)
                    return 1
                _print_job(job)
            return 0
        if args.command == "summary":
            splits = [s.strip().upper() for s in args.split.split(",") if s.strip()]
            if args.include_holdout:
                if not args.final_evaluation:
                    print("error: --include-holdout needs --final-evaluation", file=sys.stderr)
                    return 1
                splits.append("HOLDOUT")
            bad = [s for s in splits if s not in SPLITS]
            if bad:
                print(f"error: unknown split(s) {bad}", file=sys.stderr)
                return 1
            chosen = cast(list[Split], sorted(set(splits), key=SPLITS.index))
            try:
                result = summarize(
                    store, args.horizon, args.group_by, chosen, args.job or None,
                    final_evaluation=args.final_evaluation,
                )  # fmt: skip
            except PermissionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if args.json:
                print(result.model_dump_json(indent=2))
            else:
                print(result.label)
                print(f"horizon {result.horizon}, grouped by {result.group_by}, splits "
                      f"{', '.join(result.splits)}: {result.decisions} decisions, "
                      f"{result.measured} measured (stats need {result.min_sample}+ measured "
                      f"from {result.min_assets}+ assets)")  # fmt: skip
                print(as_table(result))
                for w in result.warnings:
                    print(f"warning: {w}")
            if args.baselines:
                report = evaluate_baselines(
                    store, args.horizon, chosen, args.job or None,
                    final_evaluation=args.final_evaluation,
                )  # fmt: skip
                print(report.model_dump_json(indent=2))
            if args.findings:
                found = generate(store, args.horizon, args.job or None)
                store.add_findings(found)
                print(f"{len(found)} EXPERIMENTAL finding(s) stored in the replay database "
                      "(production configuration is unchanged):")  # fmt: skip
                for f in found:
                    print(f"- [{f['validation']}] {f['statement']}")
            return 0
    except ArchiveUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted: the job is PAUSED; continue with resume <job_id>", file=sys.stderr)
        return 130
    finally:
        store.close()
    return 1
