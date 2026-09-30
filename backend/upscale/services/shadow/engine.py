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
"""

import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from upscale.services.evidence_archive.store import EvidenceStore
from upscale.services.shadow.book import Book, Output
from upscale.services.shadow.config import (
    CLEAN_DATA_CUTOFF,
    DEFAULT_RUN_ID,
    NOT_REAL_PROFIT,
    SETTLE_SECONDS,
    StrategyConfig,
)
from upscale.services.shadow.evidence import PRICE_KINDS, AnalyzeView, EvidenceTimeline
from upscale.services.shadow.metrics import strategy_metrics
from upscale.services.shadow.store import ShadowStore, ShadowStoreError
from upscale.services.shadow.strategies import BASELINES


class ShadowError(Exception):
    pass


class LookaheadError(AssertionError):
    """Evidence later than the simulation clock reached a strategy."""


def _dt(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, UTC) if ts is not None else None


class ShadowEngine:
    def __init__(
        self,
        store: ShadowStore,
        evidence: EvidenceStore | None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        settle_seconds: float = SETTLE_SECONDS,
        commit_every: int = 5000,
    ):
        self.store = store
        self.evidence = evidence
        self.now = now
        self.settle = timedelta(seconds=settle_seconds)
        self.commit_every = commit_every

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
    ) -> dict[str, Any]:
        """The run `run_id`, created on first use with the latest registered version of
        each strategy (frozen from then on). An existing run is never redefined."""
        existing = self.store.run(run_id)
        if existing is not None:
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
        self.store.create_run(
            run_id, since, until, [latest[w] for w in wanted],
            clean_data=since >= CLEAN_DATA_CUTOFF, args=args or {},
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
        settle); returns what this step did."""
        if self.evidence is None:
            raise ShadowError("the Evidence Archive is not available (UPSCALE_EVIDENCE_ARCHIVE)")
        run = self.store.run(run_id)
        if run is None:
            raise ShadowError(f"unknown run {run_id}")
        strategies = self._strategies(run)
        cp = self.store.checkpoint(run_id)
        books = {
            s.key: Book(run_id, s, cp["books"].get(s.key, {}).get("state")) for s in strategies
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
            "from": _dt(cursor[0]), "until": limit, "events": 0, "late_evidence_ignored": late,
            "actions": Counter(), "fills": 0,
        }  # fmt: skip
        if limit.timestamp() <= cursor[0]:
            report["note"] = "nothing new to process yet"
            return self._finish(report, books)
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

        def flush(cursor: tuple[float, int], through: float) -> None:
            nonlocal unflushed
            stats["events"] = stats.get("events", 0) + unflushed
            unflushed = 0
            self._commit(run_id, books, pending, cursor, through, stats, timeline, report)
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
        return self._finish(report, books)

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
    ) -> None:
        decisions, trades, equity = [], [], []
        opened, marked, closed = [], [], []
        for key, book in books.items():
            out = pending[key]
            decisions += out.decisions
            trades += out.trades
            equity += out.equity
            opened += [(book.cfg.strategy_id, book.cfg.version, p) for p in out.opened]
            closed += out.closed
            marked += [book.positions[pid] for pid in sorted(book.touched) if pid in book.positions]
            book.touched.clear()
            report["actions"].update(d["action"] for d in out.decisions)
            report["fills"] += len(out.trades)
        skipped = stats.setdefault("skipped", {})
        for why, n in timeline.skipped.items():
            skipped[why] = skipped.get(why, 0) + n
        timeline.skipped.clear()
        stats["last_commit_at"] = time.time()
        self.store.commit(
            run_id, decisions=decisions, opened=opened, marked=marked, closed=closed,
            trades=trades, equity=equity, cursor=cursor, processed_until=through,
            books={k: {"strategy_id": b.cfg.strategy_id, "strategy_version": b.cfg.version,
                       "state": b.state()} for k, b in books.items()},
            stats=stats,
        )  # fmt: skip

    def _finish(self, report: dict[str, Any], books: dict[str, Book]) -> dict[str, Any]:
        report["actions"] = dict(report["actions"])
        report["books"] = {
            key: {
                "equity": round(b.equity, 2), "cash": round(b.cash, 2),
                "open_positions": len(b.positions), "closed_positions": b.closed_positions,
                "realized_pnl": round(b.realized_pnl, 2),
                "unresolved_cost": round(b.unresolved_cost, 2),
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
                             cfg.risk.initial_capital_usd)  # fmt: skip
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
            "since": run["since"],
            "filters": {
                "strategy": strategy_id,
                "asset": asset_id,
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
            },  # fmt: skip
            "strategies": [self._metrics(run_id, s, asset_id, since, until) for s in strategies],
        }

    def status(self) -> dict[str, Any]:
        runs = []
        for r in self.store.runs():
            cp = self.store.checkpoint(r["run_id"])
            books = {
                k: {
                    "equity": round(b["state"]["cash"] + sum(
                        p["remaining_quantity"] * p["last_price"]
                        for p in b["state"]["positions"].values()), 2),
                    "open_positions": len(b["state"]["positions"]),
                    "closed_positions": b["state"].get("closed_positions", 0),
                }
                for k, b in cp["books"].items()
                if b.get("state")
            }  # fmt: skip
            runs.append({
                "run_id": r["run_id"], "since": r["since"], "until": r["until"],
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
        }


def resolve_run(store: ShadowStore, requested: str | None) -> str:
    if requested:
        return requested
    runs: list[str] = [str(r["run_id"]) for r in store.runs()]
    if DEFAULT_RUN_ID in runs or not runs:
        return DEFAULT_RUN_ID
    if len(runs) == 1:
        return runs[0]
    raise ShadowError(f"several runs exist ({', '.join(runs)}): choose one with --run")
