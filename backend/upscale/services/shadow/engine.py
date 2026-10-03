"""Shadow engine: replays archived production evidence in time order through each frozen
strategy's paper book, and stores what they would have done. No order, key, signature or
provider request exists anywhere in this path.

Anti-lookahead, by construction:

* evidence is consumed as one ordered stream (``observed_at``, archive id); the engine's
  clock only moves forward and every book sees an event only once the clock reached it;
* a Scout decision carries only what production concluded at its decision time; an
  Analyze lookup is bounded to that time (``observed_at <= T``);
* exits are triggered by later price observations as they arrive, never by looking
  ahead; a decision row is written with the event that caused it and never changes;
* processing stops ``settle`` seconds before now, so records of the same moment still in
  the archive writer's queue are not skipped past.

A run is resumable: the cursor and every book's state are checkpointed in the same
transaction as the rows they produced, so a restart continues exactly where it stopped.

Diagnostics storage (`DiagnosticsSettings`): every Scout evaluation's outcome is counted in
exact per-hour aggregate counters (written with the checkpoint); the detailed rejection
rows are kept in full, as a deterministic bounded sample (default) or not at all, and
sampled rows may expire when retention is explicitly enabled. None of it feeds back into a
strategy: decisions, trades, positions and equity are identical in every mode.
"""

import math
import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from upscale.services.evidence_archive.store import EvidenceStore
from upscale.services.shadow.book import Book, Output, Position
from upscale.services.shadow.config import (
    CLEAN_DATA_CUTOFF,
    DEFAULT_POLICY,
    DEFAULT_RUN_ID,
    NOT_REAL_PROFIT,
    SETTLE_SECONDS,
    AvailabilityPolicy,
    DiagnosticsSettings,
    ExecutionConfig,
    StrategyConfig,
    load_diagnostics_settings,
    run_execution,
    run_policy,
)
from upscale.services.shadow.evidence import PRICE_KINDS, AnalyzeView, EvidenceTimeline
from upscale.services.shadow.funnel import (
    GATES,
    Combo,
    funnel,
    parse_reasons_key,
    primary_reason,
    reasons_key,
)
from upscale.services.shadow.metrics import strategy_metrics
from upscale.services.shadow.store import BUCKET_SECONDS, ShadowStore, ShadowStoreError
from upscale.services.shadow.strategies import BASELINES


class ShadowError(Exception):
    pass


# Runs being advanced in this process (by any engine): one step per run at a time. Across
# processes the checkpoint's cursor guards it (`ShadowStore.commit(expected_cursor=...)`).
_ADVANCING: set[tuple[str, str]] = set()
_ADVANCING_LOCK = threading.Lock()


class LookaheadError(AssertionError):
    """Evidence later than the simulation clock reached a strategy."""


def _dt(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, UTC) if ts is not None else None


def _bucket(t: datetime) -> float:
    return math.floor(t.timestamp() / BUCKET_SECONDS) * float(BUCKET_SECONDS)


def _hour_floor(t: datetime) -> datetime:
    return datetime.fromtimestamp(_bucket(t), UTC)


def _hour_ceil(t: datetime) -> datetime:
    ts = math.ceil(t.timestamp() / BUCKET_SECONDS) * float(BUCKET_SECONDS)
    return datetime.fromtimestamp(ts, UTC)


class ShadowEngine:
    def __init__(
        self,
        store: ShadowStore,
        evidence: EvidenceStore | None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        settle_seconds: float = SETTLE_SECONDS,
        commit_every: int = 5000,
        diagnostics: DiagnosticsSettings | None = None,
    ):
        self.store = store
        self.evidence = evidence
        self.now = now
        self.settle = timedelta(seconds=settle_seconds)
        self.commit_every = commit_every
        # None: from the environment (default: sampled detail, retention disabled).
        self.diagnostics_settings = diagnostics or load_diagnostics_settings()

    # --- strategies and runs -------------------------------------------------------------------

    def init_baselines(self) -> dict[str, str]:
        return {s.key: self.store.register(s) for s in BASELINES}

    def ensure_run(
        self,
        run_id: str,
        since: datetime,
        until: datetime | None = None,
        strategy_ids: Sequence[str] | None = None,
        allow_contaminated: bool = False,
        args: dict[str, Any] | None = None,
        availability_policy: AvailabilityPolicy | None = None,
        execution: ExecutionConfig | None = None,
    ) -> dict[str, Any]:
        """The run `run_id`, created on first use with the latest registered version of
        each strategy, the market availability policy (default: EVIDENCE_AWARE_V2) and the
        execution model (default IDEALIZED_NO_FEES; an `ExecutionConfig`: REALISTIC_V1),
        frozen from then on. An existing run is never redefined (runs created before the
        policy existed are LEGACY_V1, before execution models IDEALIZED_NO_FEES)."""
        existing = self.store.run(run_id)
        if existing is not None:
            if execution is not None and execution != run_execution(existing):
                raise ShadowError(
                    f"run {run_id} exists with execution model {existing['execution_model']}"
                    " and its own settings: a run is never redefined, use a new --run id"
                )
            if availability_policy is not None and availability_policy != run_policy(existing):
                raise ShadowError(
                    f"run {run_id} exists with availability policy {run_policy(existing)}: a "
                    "run is never redefined, use a new --run id"
                )
            if abs(existing["since_ts"] - since.timestamp()) > 1e-6 or (
                existing["until_ts"] != (until.timestamp() if until else None)
            ):
                raise ShadowError(
                    f"run {run_id} exists with since={existing['since']} until="
                    f"{existing['until']}: a run is never redefined, use a new --run id"
                )
            if strategy_ids and sorted(strategy_ids) != sorted(
                s["strategy_id"] for s in existing["strategies"]
            ):
                raise ShadowError(f"run {run_id} exists with other strategies: use a new --run id")
            return existing
        if since < CLEAN_DATA_CUTOFF and not allow_contaminated:
            raise ShadowError(
                f"--since {since.isoformat()} is before the clean-data cutoff "
                f"{CLEAN_DATA_CUTOFF.isoformat()}: pre-fix observations are contaminated "
                "(use --allow-contaminated for a labeled research run only)"
            )
        if until is not None and until <= since:
            raise ShadowError("--until must be after --since")
        latest = {s.strategy_id: s for s in self.store.latest_strategies()}
        wanted = list(strategy_ids) if strategy_ids else sorted(latest)
        missing = [w for w in wanted if w not in latest]
        if missing or not wanted:
            raise ShadowError(
                f"unregistered strategies {missing or '(none registered)'}: run "
                "`python -m upscale.services.shadow init` first"
            )
        policy = availability_policy or DEFAULT_POLICY
        extra: dict[str, Any] = {}
        if execution is not None:
            if policy != "EVIDENCE_AWARE_V2":
                raise ShadowError("REALISTIC_V1 execution requires --availability-policy "
                                  "EVIDENCE_AWARE_V2")  # fmt: skip
            extra = {"execution": execution.model_dump(mode="json"),
                     "execution_hash": execution.config_hash}  # fmt: skip
        self.store.create_run(
            run_id, since, until, [latest[w] for w in wanted],
            clean_data=since >= CLEAN_DATA_CUTOFF,
            args={**(args or {}), "availability_policy": policy, **extra},
            execution_model=execution.name if execution is not None else None,
        )  # fmt: skip
        run = self.store.run(run_id)
        assert run is not None
        return run

    def _strategies(self, run: dict[str, Any]) -> list[StrategyConfig]:
        out = []
        for frozen in run["strategies"]:
            cfg = self.store.strategy(frozen["strategy_id"], frozen["version"])
            if cfg.config_hash != frozen["config_hash"]:
                raise ShadowError(
                    f"{cfg.key} no longer matches the hash frozen by run {run['run_id']}"
                )
            out.append(cfg)
        return out

    # --- processing ---------------------------------------------------------------------------

    def run(self, run_id: str, max_events: int | None = None) -> dict[str, Any]:
        """Process every archived event after the run's cursor up to min(until, now -
        settle); returns what this step did. Refused while another step advances the same
        run (one run is never advanced twice concurrently)."""
        key = (str(Path(self.store.path).expanduser().resolve()), run_id)
        with _ADVANCING_LOCK:
            if key in _ADVANCING:
                raise ShadowError(f"run {run_id} is already being advanced")
            _ADVANCING.add(key)
        try:
            return self._run(run_id, max_events)
        finally:
            with _ADVANCING_LOCK:
                _ADVANCING.discard(key)

    def _run(self, run_id: str, max_events: int | None) -> dict[str, Any]:
        if self.evidence is None:
            raise ShadowError("the Evidence Archive is not available (UPSCALE_EVIDENCE_ARCHIVE)")
        run = self.store.run(run_id)
        if run is None:
            raise ShadowError(f"unknown run {run_id}")
        strategies = self._strategies(run)
        cp = self.store.checkpoint(run_id)
        policy = run_policy(run)
        execution = run_execution(run)
        if execution is not None and run["args"].get("execution_hash") != execution.config_hash:
            raise ShadowError(f"run {run_id}: execution settings fail their hash check")
        books = {
            s.key: Book(run_id, s, cp["books"].get(s.key, {}).get("state"), policy, execution)
            for s in strategies
        }
        self._check_consistency(run_id, books)
        started = self.now()
        limit = started - self.settle
        if run["until_ts"] is not None:
            limit = min(limit, datetime.fromtimestamp(run["until_ts"], UTC))
        cursor: tuple[float, int] = (float(cp["cursor"][0]), int(cp["cursor"][1]))
        stats: dict[str, Any] = dict(cp["stats"])
        # Evidence archived after the previous step scanned (on the engine's own clock) but
        # observed before its cursor arrived too late to be replayed in order: counted only.
        scanned = stats.get("scanned_at")
        late = (
            self.evidence.archived_late(PRICE_KINDS, run["since_ts"], cursor[0], scanned)
            if isinstance(scanned, int | float)
            else 0
        )
        report: dict[str, Any] = {
            "label": NOT_REAL_PROFIT, "run_id": run_id, "clean_data": run["clean_data"],
            "availability_policy": policy, "execution_model": run["execution_model"],
            "from": _dt(cursor[0]), "until": limit, "events": 0, "late_evidence_ignored": late,
            "actions": Counter(), "fills": 0,
        }  # fmt: skip
        if limit.timestamp() <= cursor[0]:
            report["note"] = "nothing new to process yet"
            return self._finish(report, books)
        # Rejection diagnostics and aggregate counters exist from here on for this run
        # (never reconstructed for evaluations an older version already processed).
        start = max(_dt(cursor[0]) or limit, _dt(run["since_ts"]) or limit)
        self.store.mark_diagnostics(run_id, start)
        self.store.mark_aggregates(run_id, start, legacy=cursor[0] >= run["since_ts"])
        detail = self.diagnostics_settings.detail
        modes = stats.setdefault("diagnostics_detail_modes", [])
        if detail not in modes:
            modes.append(detail)
        stats["diagnostics_detail"] = detail
        report["diagnostics_detail"] = detail
        timeline = EvidenceTimeline(self.evidence)
        clock = _dt(cursor[0]) or limit
        group_at: datetime | None = None
        group_scout = False
        pending = {k: Output() for k in books}
        processed = unflushed = 0

        def lookup(asset: str, at: datetime, max_age: timedelta) -> AnalyzeView | None:
            if at > clock:
                raise LookaheadError(f"Analyze lookup at {at.isoformat()} after the clock")
            view = timeline.analyze(asset, at, max_age)
            if view is not None and view.observed_at > clock:
                raise LookaheadError("an Analyze decision later than the clock was returned")
            return view

        def snapshot(at: datetime) -> None:
            for key, book in books.items():
                pending[key].equity.append(book.snapshot(at))

        committed = cursor  # what the stored checkpoint must still say when we commit

        def flush(cursor: tuple[float, int], through: float) -> None:
            nonlocal unflushed, committed
            stats["events"] = stats.get("events", 0) + unflushed
            unflushed = 0
            self._commit(run_id, books, pending, cursor, through, stats, timeline, report,
                         committed)  # fmt: skip
            committed = cursor
            for key in pending:
                pending[key] = Output()

        for event in timeline.events(cursor, limit):
            if event.at < clock:
                raise LookaheadError("the evidence stream went back in time")
            if event.at > limit:
                raise LookaheadError("an event after the processing limit was returned")
            if group_at is not None and event.at > group_at and group_scout:
                snapshot(group_at)
                group_scout = False
            clock = group_at = event.at
            group_scout = group_scout or event.scout is not None
            for key, book in books.items():
                pending[key].extend(book.on_event(event, lookup))
            cursor = (event.at.timestamp(), event.seq)
            processed += 1
            unflushed += 1
            if processed % self.commit_every == 0:
                flush(cursor, event.at.timestamp())
            if max_events is not None and processed >= max_events:
                limit = event.at
                break
        # Everything up to `limit` is known: time-based closes due by then are final.
        clock = max(clock, limit)
        for key, book in books.items():
            pending[key].extend(book.sweep(limit))
        if group_at is not None and group_scout:
            snapshot(group_at)
        report["events"] = processed
        stats["scanned_at"] = started.timestamp()
        stats["late_evidence_ignored"] = stats.get("late_evidence_ignored", 0) + late
        flush((limit.timestamp(), cursor[1]) if limit.timestamp() > cursor[0] else cursor,
              limit.timestamp())  # fmt: skip
        self.store.record_metrics(
            run_id, [self._metrics(run_id, b.cfg) for b in books.values()], limit.timestamp()
        )
        d = self.diagnostics_settings
        if d.retention_days is not None and d.detail != "full":
            marker = self.store.aggregates_from(run_id)
            assert marker is not None
            cutoff = limit - timedelta(days=d.retention_days)
            report["rejection_rows_expired"] = self.store.prune_rejections(
                run_id, cutoff, float(marker["recorded_from"])
            )
        return self._finish(report, books)

    def _sample(
        self, book: Book, rejections: list[dict[str, Any]], stats: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """The detailed rejection rows to store. ``sampled``: per strategy, primary reason
        (the code of the first funnel gate the evaluation fails) and UTC day at most N rows,
        at most ceil(N / 24) of them per UTC hour, first come first kept in the replay
        order. Deterministic (the order is) and resumable (the quotas are checkpointed)."""
        d = self.diagnostics_settings
        if d.detail == "full":
            return rejections
        if d.detail == "aggregate":
            return []
        q = stats.setdefault("rejection_sampling", {}).setdefault(book.cfg.key, {})
        kept = []
        for x in rejections:
            at = x["decision_at"].astimezone(UTC)
            day, hour = at.strftime("%Y-%m-%d"), at.strftime("%Y-%m-%dT%H")
            if q.get("day") != day:
                q["day"], q["day_counts"] = day, {}
            if q.get("hour") != hour:
                q["hour"], q["hour_counts"] = hour, {}
            code = primary_reason(x["reasons"])
            if (q["day_counts"].get(code, 0) < d.sample_per_reason
                    and q["hour_counts"].get(code, 0) < d.hourly_quota):  # fmt: skip
                q["day_counts"][code] = q["day_counts"].get(code, 0) + 1
                q["hour_counts"][code] = q["hour_counts"].get(code, 0) + 1
                kept.append(x)
        return kept

    def _commit(
        self,
        run_id: str,
        books: dict[str, Book],
        pending: dict[str, Output],
        cursor: tuple[float, int],
        through: float,
        stats: dict[str, Any],
        timeline: EvidenceTimeline,
        report: dict[str, Any],
        expected: tuple[float, int] | None = None,
    ) -> None:
        decisions, trades, equity, rejections, executions = [], [], [], [], []
        opened, marked, closed = [], [], []
        counters: Counter[tuple[str, int, float, str, str]] = Counter()
        for key, book in books.items():
            out = pending[key]
            decisions += out.decisions
            trades += out.trades
            executions += out.executions
            equity += out.equity
            kept = self._sample(book, out.rejections, stats)
            rejections += kept
            report["rejection_rows_stored"] = report.get("rejection_rows_stored", 0) + len(kept)
            for at, outcome, reasons in out.evaluations:
                counters[(book.cfg.strategy_id, book.cfg.version, _bucket(at), outcome,
                          reasons_key(reasons))] += 1  # fmt: skip
            opened += [(book.cfg.strategy_id, book.cfg.version, p) for p in out.opened]
            closed += out.closed
            marked += [book.positions[pid] for pid in sorted(book.touched) if pid in book.positions]
            book.touched.clear()
            report["actions"].update(d["action"] for d in out.decisions)
            report["fills"] += len(out.trades)
            report["rejections"] = report.get("rejections", 0) + len(out.rejections)
        skipped = stats.setdefault("skipped", {})
        for why, n in timeline.skipped.items():
            skipped[why] = skipped.get(why, 0) + n
        timeline.skipped.clear()
        stats["last_commit_at"] = time.time()
        self.store.commit(
            run_id, decisions=decisions, opened=opened, marked=marked, closed=closed,
            trades=trades, equity=equity, rejections=rejections, executions=executions,
            counters=[(*k, n) for k, n in sorted(counters.items())],
            cursor=cursor, processed_until=through,
            books={k: {"strategy_id": b.cfg.strategy_id, "strategy_version": b.cfg.version,
                       "state": b.state()} for k, b in books.items()},
            stats=stats, expected_cursor=expected,
        )  # fmt: skip

    def _finish(self, report: dict[str, Any], books: dict[str, Book]) -> dict[str, Any]:
        report["actions"] = dict(report["actions"])
        report["books"] = {
            key: {
                "equity": round(b.equity, 2), "cash": round(b.cash, 2),
                "open_positions": len(b.positions), "closed_positions": b.closed_positions,
                "realized_pnl": round(b.realized_pnl, 2),
                "unresolved_cost": round(b.unresolved_cost, 2),
                **({"pending_entries": len(b.pending_entries)} if b.execution else {}),
            }
            for key, b in books.items()
        }  # fmt: skip
        return report

    def _check_consistency(self, run_id: str, books: dict[str, Book]) -> None:
        """The checkpointed open positions are exactly the stored OPEN rows."""
        stored = {p["position_id"] for p in self.store.positions(run_id=run_id, status="OPEN")}
        held = {pid for b in books.values() for pid in b.positions}
        if stored != held:
            raise ShadowStoreError(
                f"run {run_id}: checkpoint and stored open positions disagree "
                f"({len(held)} vs {len(stored)}): refusing to continue"
            )

    # --- reports ------------------------------------------------------------------------------

    def _metrics(
        self,
        run_id: str,
        cfg: StrategyConfig,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any]:
        sid = cfg.strategy_id
        trades = [t for t in self.store.trades(run_id, sid, asset_id, since, until)
                  if t["strategy_version"] == cfg.version]  # fmt: skip
        positions = [p for p in self.store.positions(run_id, sid, asset_id, None, since, until)
                     if p["strategy_version"] == cfg.version]  # fmt: skip
        equity = [e for e in self.store.equity(run_id, sid) if e["strategy_version"] == cfg.version
                  and (since is None or e["at"] >= since.timestamp())
                  and (until is None or e["at"] < until.timestamp())]  # fmt: skip
        m = strategy_metrics(sid, cfg.version, trades, positions, equity,
                             cfg.risk.initial_capital_usd,
                             self.store.unresolved_marks(run_id, sid, cfg.version))  # fmt: skip
        if asset_id is not None:
            m["note"] = "filtered to one asset: equity, drawdown and exposure are portfolio-wide"
        return m | {"run_id": run_id, "name": cfg.name}

    def metrics(
        self,
        run_id: str,
        strategy_id: str | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any]:
        run = self.store.run(run_id)
        if run is None:
            raise ShadowError(f"unknown run {run_id}")
        strategies = [s for s in self._strategies(run) if strategy_id in (None, s.strategy_id)]
        return {
            "label": NOT_REAL_PROFIT,
            "run_id": run_id,
            "clean_data": run["clean_data"],
            "availability_policy": run_policy(run),
            "since": run["since"],
            "filters": {
                "strategy": strategy_id,
                "asset": asset_id,
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
            },  # fmt: skip
            "strategies": [self._metrics(run_id, s, asset_id, since, until) for s in strategies],
        }

    # --- evaluation outcomes (aggregate counters + legacy rows) ---------------------------

    def _outcomes(
        self,
        run: dict[str, Any],
        strategies: Sequence[StrategyConfig],
        since: datetime | None,
        until: datetime | None,
        asset_id: str | None = None,
    ) -> tuple[dict[tuple[str, int], list[Combo]], dict[str, Any]]:
        """Per (strategy, version): exact evaluation outcomes in the window, widened to whole
        UTC hours (the counters' resolution) and starting no earlier than diagnostics.

        Sources: the aggregate counters for everything processed since they began; the
        stored rows for what a run processed before that (a diagnostics-era run stored
        every rejection then). With an asset filter only rows can answer (counters have no
        asset): exact only if every rejection of the window was stored."""
        run_id = run["run_id"]
        marker = self.store.diagnostics_from(run_id)
        agg = self.store.aggregates_from(run_id)
        start = datetime.fromisoformat(marker["at"]) if marker else None
        lo = _hour_floor(since) if since else None
        hi = _hour_ceil(until) if until else None
        lo = max(lo, start) if lo and start else lo or start
        combos: dict[tuple[str, int], list[Combo]] = {}

        def add(sid: str, version: int, outcome: str, reasons: tuple[str, ...], n: int) -> None:
            combos.setdefault((sid, version), []).append((outcome, reasons, n))

        exact = True
        if asset_id is not None:
            for row in self.store.row_outcomes(run_id, lo, hi, asset_id):
                add(*row)
            modes = self.store.checkpoint(run_id)["stats"].get("diagnostics_detail_modes", [])
            exact = agg is None or set(modes) <= {"full"}
        else:
            if agg is not None:
                bucket_lo = _hour_floor(lo) if lo else None
                for sid, version, outcome, key, n in self.store.funnel_counts(
                    run_id, bucket_lo, hi
                ):
                    add(sid, version, outcome, parse_reasons_key(key), n)
            if agg is None or agg.get("legacy_rows"):
                # Pre-counter rows: recorded before the counters began AND decided no
                # later than the cursor then (both bounds, so no row is counted twice).
                before = float(agg["recorded_from"]) if agg else None
                upto = datetime.fromisoformat(agg["at"]).timestamp() if agg else None
                for row in self.store.row_outcomes(run_id, lo, hi, None, before, upto):
                    add(*row)
        keys = {(s.strategy_id, s.version) for s in strategies}
        meta = {
            "diagnostics_available_from": marker["at"] if marker else None,
            "aggregates_available_from": agg["at"] if agg else None,
            "window": {
                "since": lo.isoformat() if lo else None,
                "until": hi.isoformat() if hi else None,
                "requested_since": since.isoformat() if since else None,
                "requested_until": until.isoformat() if until else None,
                "asset": asset_id,
            },
            "exact": exact,
        }
        return {k: v for k, v in combos.items() if k in keys}, meta

    def _select(
        self, run_id: str, strategy_ids: Sequence[str] | None
    ) -> tuple[dict[str, Any], list[StrategyConfig]]:
        run = self.store.run(run_id)
        if run is None:
            raise ShadowError(f"unknown run {run_id}")
        strategies = self._strategies(run)
        if strategy_ids:
            unknown = sorted(set(strategy_ids) - {s.strategy_id for s in strategies})
            if unknown:
                raise ShadowError(f"run {run_id} has no strategy {', '.join(unknown)}")
            strategies = [s for s in strategies if s.strategy_id in strategy_ids]
        return run, strategies

    def funnel(
        self,
        run_id: str,
        strategy_ids: Sequence[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any]:
        """Where each strategy's evaluations drop out, gate by gate in the documented
        diagnostic order (`funnel.GATES`), plus conditional counts per rule group. Exact:
        counted from the aggregate counters (and a pre-counter run's full rows), never from
        sampled rows. No evidence re-read, no provider request."""
        run, strategies = self._select(run_id, strategy_ids)
        combos, meta = self._outcomes(run, strategies, since, until)
        if meta["diagnostics_available_from"] is None:
            combos = {}  # a pre-diagnostics run: rejections were never recorded
        out = [{"strategy": cfg.key, "name": cfg.name,
                **funnel(cfg, combos.get((cfg.strategy_id, cfg.version), []))}
               for cfg in strategies]  # fmt: skip
        widened = (since is not None and since != _hour_floor(since)) or (
            until is not None and until != _hour_ceil(until)
        )
        return {
            "label": NOT_REAL_PROFIT,
            "run_id": run_id,
            **meta,
            "gate_order": [g.name for g in GATES],
            "note": (
                "Sequential funnel in a fixed diagnostic order: each rejected evaluation "
                "counts as failing only the FIRST gate one of its reasons belongs to (the "
                "strategy itself checks every rule). pct_of_previous = passed / input; "
                "pct_of_evaluated = passed / TOTAL_EVALUATED. Conditional checks count each "
                "gate on its own among evaluations passing every earlier rule group."
                + (
                    " The window was widened to whole UTC hours (counter resolution)."
                    if widened
                    else ""
                )
                + (
                    ""
                    if meta["diagnostics_available_from"]
                    else " No diagnostics yet for this run (earlier evaluations are not "
                    "reconstructed)."
                )
            ),  # fmt: skip
            "strategies": out,
        }

    def diagnostics(
        self,
        run_id: str,
        strategy_ids: Sequence[str] | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any]:
        """Why each strategy entered or not (exact counts, as `funnel`). Counts start when
        diagnostics began for the run: evaluated = entered + blocked (qualified, stopped by
        position / risk control) + rejected (entry rules failed). One rejection can carry
        several reasons."""
        run, strategies = self._select(run_id, strategy_ids)
        combos, meta = self._outcomes(run, strategies, since, until, asset_id)
        books = self.store.checkpoint(run_id)["books"]
        out = []
        for cfg in strategies:
            f = funnel(cfg, combos.get((cfg.strategy_id, cfg.version), []))
            n_rej = f["rejected"]
            counts = (books.get(cfg.key) or {}).get("state", {}).get("counts", {})
            legacy = {k.removeprefix("no_entry:"): v for k, v in sorted(counts.items())
                      if k.startswith("no_entry:")}  # fmt: skip
            out.append({
                "strategy": cfg.key, "name": cfg.name,
                "evaluated": f["evaluated"], "entered": f["entered"],
                "blocked": f["blocked"], "rejected": n_rej, "held": f["held"],
                "reasons": {code: {"count": n, "pct_of_rejected": round(100 * n / n_rej, 1)}
                            for code, n in f["reasons"].items()},
                # Counted by the version before diagnostics: the FIRST failed rule only, with
                # v1 codes, over the whole run (not filtered). Never reconstructed.
                "before_diagnostics_first_reason_only": legacy or None,
            })  # fmt: skip
        marker = meta["diagnostics_available_from"]
        note = (
            "evaluated = entered + blocked + rejected, counted from when diagnostics "
            "began; one rejection may carry several reasons, so reason counts can exceed "
            "rejected. Earlier evaluations are not reconstructed."
            if marker
            else "no rejection diagnostics yet: this run has not been processed by a "
            "diagnostics-enabled version (earlier evaluations are not reconstructed)"
        )
        if not meta["exact"]:
            note += (
                " Asset filter: counted from stored rows, and this run stored only a "
                "sample of its rejections since aggregate counters began: rejected counts "
                "are a lower bound."
            )
        return {
            "label": NOT_REAL_PROFIT,
            "run_id": run_id,
            "diagnostics_version": (self.store.diagnostics_from(run_id) or {}).get(
                "diagnostics_version"
            ),
            **meta,
            "note": note,
            "strategies": out,
        }

    def storage(self) -> dict[str, Any]:
        """Shadow database size, rows per table, detailed vs aggregate diagnostics, the
        latest day's growth and the diagnostics storage settings (read-only)."""
        path = Path(self.store.path).expanduser()
        files = {
            name: p.stat().st_size
            for name, p in (("db", path), ("wal", Path(f"{path}-wal")),
                            ("shm", Path(f"{path}-shm")))
            if p.exists()
        }  # fmt: skip
        st = self.store.storage()
        tables = st["tables"]
        growth: dict[str, Any] = {}
        total_day = 0.0
        known = True
        for table, info in tables.items():
            span = info.get("last_day_span_hours")
            if not span or span < 1.0:
                continue
            rows_day = info["rows_last_day"] * 24.0 / span
            size = info.get("bytes") or info.get("bytes_estimated")
            per_row = size / info["rows"] if size and info["rows"] else None
            est = rows_day * per_row if per_row is not None else None
            growth[table] = {"rows_per_day": round(rows_day),
                             "bytes_per_row": round(per_row) if per_row else None,
                             "mb_per_day": round(est / 1e6, 3) if est is not None else None}  # fmt: skip
            if est is None:
                known = False
            else:
                total_day += est
        try:
            disk = shutil.disk_usage(path.parent)
            volume: dict[str, Any] | None = {
                "path": str(path.parent), "total_mb": round(disk.total / 1e6, 1),
                "used_mb": round(disk.used / 1e6, 1), "free_mb": round(disk.free / 1e6, 1),
            }  # fmt: skip
        except OSError:
            volume, disk = None, None
        runs = []
        for r in self.store.runs():
            agg = self.store.aggregates_from(r["run_id"])
            stats = self.store.checkpoint(r["run_id"])["stats"]
            before = float(agg["recorded_from"]) if agg else None
            runs.append({
                "run_id": r["run_id"],
                "rejection_rows": self.store.rejection_rows(r["run_id"]),
                # Rows stored before the run's counters began: the only record of those
                # evaluations, never expired by retention.
                "rejection_rows_protected": self.store.rejection_rows(r["run_id"], before)
                if agg is None or agg.get("legacy_rows") else 0,
                "aggregates_available_from": agg["at"] if agg else None,
                "diagnostics_detail_modes": stats.get("diagnostics_detail_modes", []),
            })  # fmt: skip
        d = self.diagnostics_settings
        total_mb = sum(files.values()) / 1e6
        return {
            "label": NOT_REAL_PROFIT,
            "shadow_db": str(path),
            "files_bytes": files,
            "size_mb": round(total_mb, 3),
            "free_pages_mb": round(st["free_pages"] * st["page_size"] / 1e6, 3),
            "volume": volume,
            "rows": {t: i["rows"] for t, i in tables.items()},
            "rejection_detail_rows": tables.get("shadow_rejections", {}).get("rows", 0),
            "aggregate_counter_rows": tables.get("shadow_funnel_counts", {}).get("rows", 0),
            "tables": tables,
            "estimated_growth": {
                "basis": "rows of each table's latest processed day (by data time), scaled "
                "to 24h, times its average bytes per row",
                "tables": growth,
                "rejection_detail_mb_per_day": (growth.get("shadow_rejections") or {}).get(
                    "mb_per_day"
                ),
                "aggregate_counters_mb_per_day": (growth.get("shadow_funnel_counts") or {}).get(
                    "mb_per_day"
                ),
                "total_mb_per_day": round(total_day / 1e6, 3) if known and growth else None,
                "days_until_volume_full": round(disk.free / total_day, 1)
                if disk is not None and known and total_day > 0
                else None,
            },  # fmt: skip
            "settings": {
                "diagnostics_detail": d.detail,
                "sample_per_reason_per_day": d.sample_per_reason,
                "sample_per_reason_per_hour": d.hourly_quota,
                "retention_enabled": d.retention_days is not None,
                "retention_days": d.retention_days,
            },
            "runs": runs,
        }

    def status(self) -> dict[str, Any]:
        runs = []
        for r in self.store.runs():
            cp = self.store.checkpoint(r["run_id"])
            policy = run_policy(r)
            at = _dt(cp["processed_until"]) or self.now()
            books = {
                k: {
                    "equity": round(b["state"]["cash"] + sum(
                        p["remaining_quantity"] * p["last_price"]
                        for p in b["state"]["positions"].values()) + sum(
                        e["budget_usd"]
                        for e in (b["state"].get("pending_entries") or {}).values()), 2),
                    "open_positions": len(b["state"]["positions"]),
                    "closed_positions": b["state"].get("closed_positions", 0),
                    **({"open_market_states": _states(b, at, self.store)}
                       if policy == "EVIDENCE_AWARE_V2" else {}),
                }
                for k, b in cp["books"].items()
                if b.get("state")
            }  # fmt: skip
            runs.append({
                "run_id": r["run_id"], "availability_policy": policy,
                "since": r["since"], "until": r["until"],
                "clean_data": r["clean_data"], "execution_model": r["execution_model"],
                "strategies": [f"{s['strategy_id']}@v{s['version']}" for s in r["strategies"]],
                "processed_until": _dt(cp["processed_until"]),
                "last_update": _dt(cp["updated_at"]), "books": books,
                "stats": cp["stats"],
            })  # fmt: skip
        return {
            "label": NOT_REAL_PROFIT,
            "shadow_db": self.store.path,
            "evidence_db": self.evidence.path if self.evidence is not None else None,
            "clean_data_cutoff": CLEAN_DATA_CUTOFF.isoformat(),
            "strategies": [s.key for s in self.store.strategies()],
            "runs": runs,
            "rows": self.store.counts(),
            "diagnostics_storage": {
                "detail": self.diagnostics_settings.detail,
                "sample_per_reason": self.diagnostics_settings.sample_per_reason,
                "retention_days": self.diagnostics_settings.retention_days,
            },
        }


def _states(book: dict[str, Any], at: datetime, store: ShadowStore) -> dict[str, int]:
    """Open positions by market state at the processed time (EVIDENCE_AWARE_V2 reporting)."""
    cfg = store.strategy(book["strategy_id"], book["strategy_version"])
    out: Counter[str] = Counter()
    for raw in book["state"]["positions"].values():
        p = Position.from_json(raw)
        out[p.market_state(at, cfg.exit.market_unavailable_after_minutes)] += 1
    return dict(sorted(out.items()))


def resolve_run(store: ShadowStore, requested: str | None) -> str:
    if requested:
        return requested
    runs: list[str] = [str(r["run_id"]) for r in store.runs()]
    if DEFAULT_RUN_ID in runs or not runs:
        return DEFAULT_RUN_ID
    if len(runs) == 1:
        return runs[0]
    raise ShadowError(f"several runs exist ({', '.join(runs)}): choose one with --run")
