"""Durable outcome storage: the same SQLite file as Scout (separate tables), append-only.

Schema (version 1, recorded in ``outcome_meta``; Scout's own ``PRAGMA user_version`` and
``scout_meta`` are never touched):

* ``scout_outcome_observations``: one immutable row per Scout evaluation anchor. Key
  columns for querying, plus the full `ScoutObservation` as JSON. ``(canonical_id,
  observed_at)`` is unique: the same market evidence is never anchored twice.
* ``decision_observations``: one immutable row per recorded Analyze decision.
* ``scout_outcome_horizons`` / ``decision_outcome_horizons``: one row per observation and
  fixed horizon, created PENDING with its due time. Each part of a measurement (price
  path, horizon-end market state, future stage) is written once (``COALESCE``), and a
  row leaves PENDING exactly once: a finalized row can never change again.
* ``outcome_ranked`` / ``outcome_runs``: bookkeeping for the observation policy (when a
  token was last ranked, which ranking runs happened), never outcome data.

Immutability is enforced by the database itself (triggers abort any UPDATE / DELETE of an
observation and any change to a finalized horizon), not only by the code. All times are
UTC epoch seconds. No credentials are stored.
"""

import asyncio
import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from upscale.services.outcomes.config import HorizonSpec
from upscale.services.outcomes.models import (
    OUTCOME_SCHEMA_VERSION,
    DecisionObservation,
    HorizonOutcome,
    HorizonStatus,
    MarketAtHorizon,
    MarketStatus,
    PricePath,
    ScoutObservation,
    SurfacingHistory,
    TriggerOutcome,
)

Kind = Literal["scout", "decision"]
_TABLES: dict[Kind, tuple[str, str]] = {
    "scout": ("scout_outcome_observations", "scout_outcome_horizons"),
    "decision": ("decision_observations", "decision_outcome_horizons"),
}
_HORIZON_COLUMNS = """
    observation_id INTEGER NOT NULL REFERENCES {obs}(id),
    horizon TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    due_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    market_status TEXT,
    return_pct REAL,
    mfe_pct REAL,
    mae_pct REAL,
    liquidity_change_pct REAL,
    future_stage TEXT,
    price_json TEXT,
    market_json TEXT,
    stage_json TEXT,
    triggers_json TEXT,
    missing_json TEXT NOT NULL DEFAULT '[]',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at REAL,
    finalized_at REAL,
    PRIMARY KEY (observation_id, horizon)
"""
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS outcome_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scout_outcome_observations (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    chain TEXT NOT NULL,
    address TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    observed_at REAL NOT NULL,
    anchored_at REAL NOT NULL,
    run_id TEXT NOT NULL,
    anchor_reason TEXT NOT NULL,
    rank INTEGER NOT NULL,
    stage TEXT NOT NULL,
    score REAL NOT NULL,
    discovery_status TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    body_json TEXT NOT NULL,
    UNIQUE (canonical_id, observed_at)
);
CREATE INDEX IF NOT EXISTS scout_outcome_observations_by_asset
    ON scout_outcome_observations (canonical_id, observed_at);
CREATE INDEX IF NOT EXISTS scout_outcome_observations_by_time
    ON scout_outcome_observations (anchored_at);
CREATE TABLE IF NOT EXISTS decision_observations (
    id INTEGER PRIMARY KEY,
    asset_id TEXT NOT NULL,
    analyzed_at REAL NOT NULL,
    action TEXT NOT NULL,
    source TEXT NOT NULL,
    scout_observation_id INTEGER REFERENCES scout_outcome_observations(id),
    decision_key TEXT NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL,
    body_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_observations_by_asset
    ON decision_observations (asset_id, analyzed_at);
CREATE TABLE IF NOT EXISTS scout_outcome_horizons (
    {_HORIZON_COLUMNS.format(obs="scout_outcome_observations")}
);
CREATE INDEX IF NOT EXISTS scout_outcome_horizons_due
    ON scout_outcome_horizons (status, due_at);
CREATE TABLE IF NOT EXISTS decision_outcome_horizons (
    {_HORIZON_COLUMNS.format(obs="decision_observations")}
);
CREATE INDEX IF NOT EXISTS decision_outcome_horizons_due
    ON decision_outcome_horizons (status, due_at);
CREATE TABLE IF NOT EXISTS outcome_ranked (
    canonical_id TEXT PRIMARY KEY,
    first_ranked_at REAL NOT NULL,
    last_ranked_at REAL NOT NULL,
    times_ranked INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS outcome_runs (run_at REAL PRIMARY KEY, ranked INTEGER NOT NULL);
"""
_IMMUTABLE = (
    [
        f"""
CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END;
"""
        for table in ("scout_outcome_observations", "decision_observations")
    ]
    + [
        f"""
CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} rows are never deleted'); END;
"""
        for table in (
            "scout_outcome_observations",
            "decision_observations",
            "scout_outcome_horizons",
            "decision_outcome_horizons",
        )
    ]
    + [
        f"""
CREATE TRIGGER IF NOT EXISTS {table}_final BEFORE UPDATE ON {table}
WHEN OLD.status != 'PENDING'
BEGIN SELECT RAISE(ABORT, 'finalized {table} rows are immutable'); END;
"""
        for table in ("scout_outcome_horizons", "decision_outcome_horizons")
    ]
)
_VERSION_KEY = "outcome_schema_version"


class OutcomeStoreError(Exception):
    pass


@dataclass(frozen=True)
class Anchor:
    """A token's latest Scout anchor (for the observation policy)."""

    id: int
    observed_at: datetime
    anchored_at: datetime
    stage: str
    score: float


@dataclass(frozen=True)
class RankedState:
    first_ranked_at: datetime
    last_ranked_at: datetime
    times_ranked: int


@dataclass
class HorizonUpdate:
    """One collection attempt. Parts already stored are kept (first write wins)."""

    price: PricePath | None = None
    market: MarketAtHorizon | None = None
    future_stage: str | None = None
    future_stage_at: datetime | None = None
    future_rank: int | None = None
    future_score: float | None = None
    triggers: dict[str, TriggerOutcome] | None = None
    first_trigger_event: str | None = None
    price_position: str | None = None
    missing: list[str] = field(default_factory=list)
    # Set to leave PENDING (final). None: stays PENDING (e.g. deferred for quota).
    finalize: HorizonStatus | None = None
    market_status: MarketStatus | None = None


@dataclass(frozen=True)
class DueHorizon:
    kind: Kind
    observation: ScoutObservation | DecisionObservation
    horizon: HorizonOutcome


class OutcomeStore:
    """Thread-safe; every public coroutine runs its SQL in a worker thread."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --- public API (async) ---------------------------------------------------------------

    async def add_scout_observations(
        self, observations: Sequence[ScoutObservation], horizons: Sequence[HorizonSpec]
    ) -> list[ScoutObservation]:
        """Store new anchors with their PENDING horizons. An anchor of evidence already
        anchored (same token, same observation time) is skipped. Returns those stored."""
        return await asyncio.to_thread(self._add_scout, list(observations), list(horizons))

    async def add_decision(
        self, decision: DecisionObservation, key: str, horizons: Sequence[HorizonSpec]
    ) -> DecisionObservation | None:
        """Store a decision (None if `key` was already stored)."""
        return await asyncio.to_thread(self._add_decision, decision, key, list(horizons))

    async def latest_anchors(self, canonical_ids: Sequence[str]) -> dict[str, Anchor]:
        return await asyncio.to_thread(self._latest_anchors, list(canonical_ids))

    async def ranked_state(self, canonical_ids: Sequence[str]) -> dict[str, RankedState]:
        return await asyncio.to_thread(self._ranked_state, list(canonical_ids))

    async def run_times(self, since: datetime) -> list[datetime]:
        """Recorded ranking runs since `since`, oldest first."""
        return await asyncio.to_thread(self._run_times, since)

    async def record_run(self, at: datetime, ranked: Sequence[str]) -> None:
        """One ranking run and the tokens it ranked (idempotent per run time)."""
        await asyncio.to_thread(self._record_run, at, list(ranked))

    async def surfacing(self, canonical_ids: Sequence[str]) -> dict[str, SurfacingHistory]:
        state = await self.ranked_state(canonical_ids)
        return {
            cid: SurfacingHistory(first_ranked_at=s.first_ranked_at, times_ranked=s.times_ranked)
            for cid, s in state.items()
        }

    async def ensure_horizons(self, horizons: Sequence[HorizonSpec]) -> int:
        """Add PENDING rows for horizons configured after an observation was stored (e.g.
        a new 7d horizon). Existing rows are never touched. Returns rows added."""
        return await asyncio.to_thread(self._ensure_horizons, list(horizons))

    async def due(
        self, now: datetime, settle_seconds: float, retry_seconds: float, limit: int
    ) -> list[DueHorizon]:
        """PENDING horizons whose window ended at least `settle_seconds` ago and that were
        not attempted in the last `retry_seconds`, oldest first (decisions first at equal
        due time)."""
        return await asyncio.to_thread(self._due, now, settle_seconds, retry_seconds, limit)

    async def companions(
        self, due: Sequence[DueHorizon], ended_before: datetime
    ) -> list[DueHorizon]:
        """Other PENDING horizons of the observations in `due` whose window ended before
        `ended_before` (whatever their retry spacing)."""
        return await asyncio.to_thread(self._companions, list(due), ended_before)

    async def next_wake(self, settle_seconds: float, retry_seconds: float) -> datetime | None:
        """When the next PENDING horizon becomes collectable (never attempted: due + settle;
        attempted: last attempt + retry). None: nothing pending."""
        return await asyncio.to_thread(self._next_wake, settle_seconds, retry_seconds)

    async def update_horizon(
        self, kind: Kind, observation_id: int, horizon: str, update: HorizonUpdate, at: datetime
    ) -> bool:
        """Apply one attempt to a PENDING horizon. False if it was already final (a rerun
        of the collector is a no-op)."""
        return await asyncio.to_thread(self._update, kind, observation_id, horizon, update, at)

    async def scout_observations(
        self,
        canonical_id: str | None = None,
        stage: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ScoutObservation]:
        return await asyncio.to_thread(self._scout_list, canonical_id, stage, since, limit, offset)

    async def scout_observation(self, observation_id: int) -> ScoutObservation | None:
        return await asyncio.to_thread(self._scout_one, observation_id)

    async def decisions(
        self,
        asset_id: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DecisionObservation]:
        return await asyncio.to_thread(self._decision_list, asset_id, action, limit, offset)

    async def decision(self, decision_id: int) -> DecisionObservation | None:
        return await asyncio.to_thread(self._decision_one, decision_id)

    async def recent_decisions(self, asset_id: str, since: datetime) -> list[DecisionObservation]:
        return await asyncio.to_thread(self._recent_decisions, asset_id, since)

    async def horizons(
        self, kind: Kind, observation_ids: Sequence[int]
    ) -> dict[int, list[HorizonOutcome]]:
        return await asyncio.to_thread(self._horizons, kind, list(observation_ids))

    async def all_scout(self) -> list[tuple[ScoutObservation, list[HorizonOutcome]]]:
        """Every Scout observation with its horizons (for aggregate analytics)."""
        return await asyncio.to_thread(self._all_scout)

    async def all_decisions(self) -> list[tuple[DecisionObservation, list[HorizonOutcome]]]:
        decisions = await asyncio.to_thread(self._decision_list, None, None, 1_000_000, 0)
        ids = [d.id for d in decisions if d.id is not None]
        horizons = await self.horizons("decision", ids)
        return [(d, horizons.get(d.id or -1, [])) for d in decisions]

    async def counts(self) -> dict[str, dict[str, int]]:
        """Horizon rows per status, per kind."""
        return await asyncio.to_thread(self._counts)

    # --- SQL ------------------------------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS outcome_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = conn.execute(
                "SELECT value FROM outcome_meta WHERE key = ?", (_VERSION_KEY,)
            ).fetchone()
            version = int(row[0]) if row else 0
            if version > OUTCOME_SCHEMA_VERSION:
                conn.close()
                raise OutcomeStoreError(
                    f"outcome store {self.path} has schema v{version}; this UpScale knows "
                    f"v{OUTCOME_SCHEMA_VERSION}"
                )
            conn.executescript(_SCHEMA)
            for trigger in _IMMUTABLE:
                conn.executescript(trigger)
            conn.execute(
                "INSERT INTO outcome_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (_VERSION_KEY, str(OUTCOME_SCHEMA_VERSION)),
            )
            conn.commit()
            self._conn = conn
        return self._conn

    def _insert_horizons(
        self,
        db: sqlite3.Connection,
        kind: Kind,
        observation_id: int,
        start: datetime,
        horizons: Iterable[HorizonSpec],
    ) -> None:
        table = _TABLES[kind][1]
        db.executemany(
            f"""
            INSERT OR IGNORE INTO {table} (observation_id, horizon, horizon_minutes, due_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (observation_id, h.label, h.minutes, start.timestamp() + h.minutes * 60)
                for h in horizons
            ],
        )

    def _add_scout(
        self, observations: list[ScoutObservation], horizons: list[HorizonSpec]
    ) -> list[ScoutObservation]:
        stored: list[ScoutObservation] = []
        with self._lock:
            db = self._db()
            with db:
                for o in observations:
                    cur = db.execute(
                        """
                        INSERT OR IGNORE INTO scout_outcome_observations (canonical_id, chain,
                            address, pool_address, observed_at, anchored_at, run_id,
                            anchor_reason, rank, stage, score, discovery_status,
                            schema_version, body_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            o.canonical_id,
                            o.chain,
                            o.address,
                            o.pool_address,
                            o.observed_at.timestamp(),
                            o.anchored_at.timestamp(),
                            o.run_id,
                            o.anchor_reason,
                            o.rank,
                            o.stage,
                            o.score,
                            o.discovery_status,
                            o.schema_version,
                            o.model_dump_json(exclude={"id"}),
                        ),
                    )
                    if cur.rowcount != 1 or cur.lastrowid is None:
                        continue  # this evidence is already anchored
                    self._insert_horizons(db, "scout", cur.lastrowid, o.observed_at, horizons)
                    stored.append(o.model_copy(update={"id": cur.lastrowid}))
        return stored

    def _add_decision(
        self, d: DecisionObservation, key: str, horizons: list[HorizonSpec]
    ) -> DecisionObservation | None:
        with self._lock:
            db = self._db()
            with db:
                cur = db.execute(
                    """
                    INSERT OR IGNORE INTO decision_observations (asset_id, analyzed_at, action,
                        source, scout_observation_id, decision_key, schema_version, body_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        d.asset_id,
                        d.analyzed_at.timestamp(),
                        d.action,
                        d.source,
                        d.scout_observation_id,
                        key,
                        d.schema_version,
                        d.model_dump_json(exclude={"id"}),
                    ),
                )
                if cur.rowcount != 1 or cur.lastrowid is None:
                    return None
                self._insert_horizons(db, "decision", cur.lastrowid, d.analyzed_at, horizons)
        return d.model_copy(update={"id": cur.lastrowid})

    def _latest_anchors(self, ids: list[str]) -> dict[str, Anchor]:
        out: dict[str, Anchor] = {}
        with self._lock:
            db = self._db()
            for chunk in _chunks(ids):
                rows = db.execute(
                    f"""
                    SELECT canonical_id, id, observed_at, anchored_at, stage, score
                    FROM scout_outcome_observations o
                    WHERE canonical_id IN ({",".join("?" * len(chunk))}) AND observed_at = (
                        SELECT MAX(observed_at) FROM scout_outcome_observations
                        WHERE canonical_id = o.canonical_id
                    )
                    """,
                    chunk,
                ).fetchall()
                for cid, oid, observed, anchored, stage, score in rows:
                    out[cid] = Anchor(oid, _dt(observed), _dt(anchored), stage, score)
        return out

    def _ranked_state(self, ids: list[str]) -> dict[str, RankedState]:
        out: dict[str, RankedState] = {}
        with self._lock:
            db = self._db()
            for chunk in _chunks(ids):
                rows = db.execute(
                    f"""
                    SELECT canonical_id, first_ranked_at, last_ranked_at, times_ranked
                    FROM outcome_ranked WHERE canonical_id IN ({",".join("?" * len(chunk))})
                    """,
                    chunk,
                ).fetchall()
                for cid, first, last, times in rows:
                    out[cid] = RankedState(_dt(first), _dt(last), times)
        return out

    def _run_times(self, since: datetime) -> list[datetime]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT run_at FROM outcome_runs WHERE run_at >= ? ORDER BY run_at",
                    (since.timestamp(),),
                )
                .fetchall()
            )
        return [_dt(r[0]) for r in rows]

    def _record_run(self, at: datetime, ranked: list[str]) -> None:
        t = at.timestamp()
        with self._lock:
            db = self._db()
            with db:
                cur = db.execute(
                    "INSERT OR IGNORE INTO outcome_runs (run_at, ranked) VALUES (?, ?)",
                    (t, len(ranked)),
                )
                if cur.rowcount != 1:
                    return  # this run was already recorded
                db.executemany(
                    """
                    INSERT INTO outcome_ranked (canonical_id, first_ranked_at, last_ranked_at,
                        times_ranked) VALUES (?, ?, ?, 1)
                    ON CONFLICT (canonical_id) DO UPDATE SET
                        first_ranked_at = MIN(first_ranked_at, excluded.first_ranked_at),
                        last_ranked_at = MAX(last_ranked_at, excluded.last_ranked_at),
                        times_ranked = times_ranked + 1
                    """,
                    [(cid, t, t) for cid in dict.fromkeys(ranked)],
                )

    def _ensure_horizons(self, horizons: list[HorizonSpec]) -> int:
        added = 0
        with self._lock:
            db = self._db()
            with db:
                for kind, (obs, table) in _TABLES.items():
                    start = "observed_at" if kind == "scout" else "analyzed_at"
                    for h in horizons:
                        cur = db.execute(
                            f"""
                            INSERT OR IGNORE INTO {table}
                                (observation_id, horizon, horizon_minutes, due_at)
                            SELECT id, ?, ?, {start} + ? FROM {obs}
                            """,
                            (h.label, h.minutes, h.minutes * 60),
                        )
                        added += cur.rowcount
        return added

    def _due(self, now: datetime, settle: float, retry: float, limit: int) -> list[DueHorizon]:
        cutoff = now.timestamp() - settle
        out: list[DueHorizon] = []
        with self._lock:
            db = self._db()
            kinds: tuple[Kind, ...] = ("decision", "scout")
            for kind in kinds:
                obs, table = _TABLES[kind]
                rows = db.execute(
                    f"""
                    SELECT o.body_json, o.id, {_H_COLUMNS} FROM {table} h
                    JOIN {obs} o ON o.id = h.observation_id
                    WHERE h.status = 'PENDING' AND h.due_at <= ?
                        AND (h.last_attempt_at IS NULL OR h.last_attempt_at <= ?)
                    ORDER BY h.due_at, h.observation_id LIMIT ?
                    """,
                    (cutoff, now.timestamp() - retry, limit),
                ).fetchall()
                for row in rows:
                    observation = _observation(kind, row[0], row[1])
                    out.append(DueHorizon(kind, observation, _horizon(row[2:])))
        out.sort(key=lambda d: (d.horizon.due_at, d.kind != "decision"))
        return out[:limit]

    def _companions(self, due: list[DueHorizon], ended_before: datetime) -> list[DueHorizon]:
        seen = {(d.kind, d.horizon.observation_id, d.horizon.horizon) for d in due}
        observations = {(d.kind, d.horizon.observation_id): d.observation for d in due}
        out: list[DueHorizon] = []
        with self._lock:
            db = self._db()
            kinds: tuple[Kind, ...] = ("decision", "scout")
            for kind in kinds:
                ids = [oid for k, oid in observations if k == kind]
                table = _TABLES[kind][1]
                for chunk in _chunks(ids):
                    rows = db.execute(
                        f"""
                        SELECT {_H_COLUMNS} FROM {table} h
                        WHERE status = 'PENDING' AND due_at <= ?
                            AND observation_id IN ({",".join("?" * len(chunk))})
                        """,
                        [ended_before.timestamp(), *chunk],
                    ).fetchall()
                    for r in rows:
                        h = _horizon(r)
                        if (kind, h.observation_id, h.horizon) not in seen:
                            observation = observations[(kind, h.observation_id)]
                            out.append(DueHorizon(kind, observation, h))
        return out

    def _next_wake(self, settle: float, retry: float) -> datetime | None:
        times: list[float] = []
        with self._lock:
            db = self._db()
            for _, table in _TABLES.values():
                row = db.execute(
                    f"""
                    SELECT MIN(CASE WHEN last_attempt_at IS NULL THEN due_at + ?
                                    ELSE MAX(due_at + ?, last_attempt_at + ?) END)
                    FROM {table} WHERE status = 'PENDING'
                    """,
                    (settle, settle, retry),
                ).fetchone()
                if row[0] is not None:
                    times.append(row[0])
        return _dt(min(times)) if times else None

    def _update(self, kind: Kind, oid: int, horizon: str, u: HorizonUpdate, at: datetime) -> bool:
        table = _TABLES[kind][1]
        stage = (
            json.dumps(
                {
                    "stage": u.future_stage,
                    "at": u.future_stage_at.timestamp() if u.future_stage_at else None,
                    "rank": u.future_rank,
                    "score": u.future_score,
                }
            )
            if u.future_stage is not None
            else None
        )
        extra = (
            json.dumps(
                {
                    "triggers": {k: v.model_dump(mode="json") for k, v in u.triggers.items()},
                    "first_event": u.first_trigger_event,
                    "price_position": u.price_position,
                }
            )
            if u.triggers is not None
            else None
        )
        with self._lock:
            db = self._db()
            with db:
                row = db.execute(
                    f"SELECT missing_json FROM {table} WHERE observation_id = ? AND horizon = ? "
                    "AND status = 'PENDING'",
                    (oid, horizon),
                ).fetchone()
                if row is None:
                    return False
                # The latest attempt's reasons replace earlier ones (they describe what is
                # still missing now); a finalized row keeps its last reasons forever.
                missing = json.dumps(list(dict.fromkeys(u.missing)))
                p, m = u.price, u.market
                db.execute(
                    f"""
                    UPDATE {table} SET
                        price_json = COALESCE(price_json, ?),
                        return_pct = COALESCE(return_pct, ?),
                        mfe_pct = COALESCE(mfe_pct, ?),
                        mae_pct = COALESCE(mae_pct, ?),
                        market_json = COALESCE(market_json, ?),
                        liquidity_change_pct = COALESCE(liquidity_change_pct, ?),
                        stage_json = COALESCE(stage_json, ?),
                        future_stage = COALESCE(future_stage, ?),
                        triggers_json = COALESCE(triggers_json, ?),
                        market_status = COALESCE(?, market_status),
                        missing_json = ?,
                        attempts = attempts + 1,
                        last_attempt_at = ?,
                        status = COALESCE(?, status),
                        finalized_at = CASE WHEN ? IS NULL THEN NULL ELSE ? END
                    WHERE observation_id = ? AND horizon = ? AND status = 'PENDING'
                    """,
                    (
                        p.model_dump_json() if p else None,
                        p.return_pct if p else None,
                        p.mfe_pct if p else None,
                        p.mae_pct if p else None,
                        m.model_dump_json() if m else None,
                        m.liquidity_change_pct if m else None,
                        stage,
                        u.future_stage,
                        extra,
                        u.market_status,
                        missing,
                        at.timestamp(),
                        u.finalize,
                        u.finalize,
                        at.timestamp(),
                        oid,
                        horizon,
                    ),
                )
        return True

    def _scout_list(
        self,
        canonical_id: str | None,
        stage: str | None,
        since: datetime | None,
        limit: int,
        offset: int,
    ) -> list[ScoutObservation]:
        where: list[str] = ["1 = 1"]
        params: list[Any] = []
        if canonical_id:
            where.append("canonical_id = ?")
            params.append(canonical_id)
        if stage:
            where.append("stage = ?")
            params.append(stage)
        if since:
            where.append("observed_at >= ?")
            params.append(since.timestamp())
        with self._lock:
            rows = (
                self._db()
                .execute(
                    f"""
                    SELECT body_json, id FROM scout_outcome_observations
                    WHERE {" AND ".join(where)} ORDER BY observed_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*params, limit, offset],
                )
                .fetchall()
            )
        return [_scout(r[0], r[1]) for r in rows]

    def _scout_one(self, oid: int) -> ScoutObservation | None:
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT body_json, id FROM scout_outcome_observations WHERE id = ?", (oid,)
                )
                .fetchone()
            )
        return _scout(row[0], row[1]) if row else None

    def _decision_list(
        self, asset_id: str | None, action: str | None, limit: int, offset: int
    ) -> list[DecisionObservation]:
        where: list[str] = ["1 = 1"]
        params: list[Any] = []
        if asset_id:
            where.append("asset_id = ?")
            params.append(asset_id)
        if action:
            where.append("action = ?")
            params.append(action)
        with self._lock:
            rows = (
                self._db()
                .execute(
                    f"""
                    SELECT body_json, id FROM decision_observations
                    WHERE {" AND ".join(where)} ORDER BY analyzed_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*params, limit, offset],
                )
                .fetchall()
            )
        return [_decision(r[0], r[1]) for r in rows]

    def _decision_one(self, did: int) -> DecisionObservation | None:
        with self._lock:
            row = (
                self._db()
                .execute("SELECT body_json, id FROM decision_observations WHERE id = ?", (did,))
                .fetchone()
            )
        return _decision(row[0], row[1]) if row else None

    def _recent_decisions(self, asset_id: str, since: datetime) -> list[DecisionObservation]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT body_json, id FROM decision_observations "
                    "WHERE asset_id = ? AND analyzed_at >= ? ORDER BY analyzed_at",
                    (asset_id, since.timestamp()),
                )
                .fetchall()
            )
        return [_decision(r[0], r[1]) for r in rows]

    def _horizons(self, kind: Kind, ids: list[int]) -> dict[int, list[HorizonOutcome]]:
        table = _TABLES[kind][1]
        out: dict[int, list[HorizonOutcome]] = {i: [] for i in ids}
        with self._lock:
            db = self._db()
            for chunk in _chunks(ids):
                rows = db.execute(
                    f"""
                    SELECT {_H_COLUMNS} FROM {table} h
                    WHERE observation_id IN ({",".join("?" * len(chunk))})
                    ORDER BY observation_id, horizon_minutes
                    """,
                    chunk,
                ).fetchall()
                for r in rows:
                    h = _horizon(r)
                    out[h.observation_id].append(h)
        return out

    def _all_scout(self) -> list[tuple[ScoutObservation, list[HorizonOutcome]]]:
        with self._lock:
            db = self._db()
            rows = db.execute(
                "SELECT body_json, id FROM scout_outcome_observations ORDER BY id"
            ).fetchall()
            hrows = db.execute(
                f"SELECT {_H_COLUMNS} FROM scout_outcome_horizons h "
                "ORDER BY observation_id, horizon_minutes"
            ).fetchall()
        by_obs: dict[int, list[HorizonOutcome]] = {}
        for r in hrows:
            h = _horizon(r)
            by_obs.setdefault(h.observation_id, []).append(h)
        return [(_scout(r[0], r[1]), by_obs.get(r[1], [])) for r in rows]

    def _counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._lock:
            db = self._db()
            for kind, (obs, table) in _TABLES.items():
                observations = db.execute(f"SELECT COUNT(*) FROM {obs}").fetchone()[0]
                rows = db.execute(f"SELECT status, COUNT(*) FROM {table} GROUP BY status")
                out[kind] = {"observations": int(observations)} | {
                    status: int(n) for status, n in rows
                }
        return out


_H_COLUMNS = (
    "h.observation_id, h.horizon, h.horizon_minutes, h.due_at, h.status, h.market_status, "
    "h.price_json, h.market_json, h.stage_json, h.triggers_json, h.missing_json, h.attempts, "
    "h.last_attempt_at, h.finalized_at"
)


def _chunks(ids: Sequence[Any], size: int = 500) -> list[list[Any]]:
    return [list(ids[i : i + size]) for i in range(0, len(ids), size)]


def _dt(value: float) -> datetime:
    return datetime.fromtimestamp(value, UTC)


def _scout(body: str, oid: int) -> ScoutObservation:
    return ScoutObservation.model_validate_json(body).model_copy(update={"id": oid})


def _decision(body: str, did: int) -> DecisionObservation:
    return DecisionObservation.model_validate_json(body).model_copy(update={"id": did})


def _observation(kind: Kind, body: str, oid: int) -> ScoutObservation | DecisionObservation:
    return _scout(body, oid) if kind == "scout" else _decision(body, oid)


def _horizon(r: Sequence[Any]) -> HorizonOutcome:
    stage = json.loads(r[8]) if r[8] else None
    extra = json.loads(r[9]) if r[9] else None
    return HorizonOutcome(
        observation_id=r[0],
        horizon=r[1],
        horizon_minutes=r[2],
        due_at=_dt(r[3]),
        status=r[4],
        market_status=r[5],
        price=PricePath.model_validate_json(r[6]) if r[6] else None,
        market=MarketAtHorizon.model_validate_json(r[7]) if r[7] else None,
        future_stage=stage["stage"] if stage else None,
        future_stage_at=_dt(stage["at"]) if stage and stage["at"] is not None else None,
        future_rank=stage["rank"] if stage else None,
        future_score=stage["score"] if stage else None,
        triggers={
            k: TriggerOutcome.model_validate(v)
            for k, v in (extra or {}).get("triggers", {}).items()
        },
        first_trigger_event=(extra or {}).get("first_event"),
        price_position=(extra or {}).get("price_position"),
        missing=json.loads(r[10]),
        attempts=r[11],
        last_attempt_at=_dt(r[12]) if r[12] is not None else None,
        finalized_at=_dt(r[13]) if r[13] is not None else None,
    )
