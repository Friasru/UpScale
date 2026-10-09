"""Orchestration state (SQLite, its own file, schema version 1): jobs and their events.

* ``orch_meta``: component and schema version.
* ``orch_jobs``: one job per admitted Scout record, unique per (canonical_id,
  scout_record_id). The admission facts (token, record, run time, market time, stage, rank,
  created_at) are immutable; lifecycle fields change only through a legal state transition.
  Terminal jobs (DECIDED, FAILED, SUPERSEDED) never change.
  The job's Opportunity decision key, ``opportunity_decision_at`` (the exact LIVE_FORWARD
  ``decision_at``) and ``opportunity_rules_version``, is NULL until SNAPSHOTTED ->
  DECIDING, set once then (in the same statement as the transition and its event) and
  never changed afterwards (trigger).
* ``orch_events``: append-only. Written *by triggers* in the same statement as the job
  insert / state change, so a transition and its event always commit together.

Legal transitions are enforced in SQL (`TRANSITIONS`). A database of another component or
schema version, or Orchestrator tables without metadata, is refused before any write.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from upscale.services.opportunity_orchestrator.config import (
    ORCHESTRATOR_COMPONENT,
    ORCHESTRATOR_DB_SCHEMA_VERSION,
    PROCESSING,
)
from upscale.services.opportunity_orchestrator.models import AdmissionResult

JobState = Literal[
    "QUEUED", "COLLECTING", "SNAPSHOTTED", "DECIDING", "DECIDED", "DEFERRED", "RETRY_WAIT",
    "FAILED", "SUPERSEDED",
]  # fmt: skip
STATES: tuple[JobState, ...] = (
    "QUEUED", "COLLECTING", "SNAPSHOTTED", "DECIDING", "DECIDED", "DEFERRED", "RETRY_WAIT",
    "FAILED", "SUPERSEDED",
)  # fmt: skip
TERMINAL: tuple[JobState, ...] = ("DECIDED", "FAILED", "SUPERSEDED")
IN_FLIGHT: tuple[JobState, ...] = ("COLLECTING", "SNAPSHOTTED", "DECIDING")
TRANSITIONS: frozenset[tuple[JobState, JobState]] = frozenset(
    {
        ("QUEUED", "COLLECTING"),  # a collection is needed and allowed
        ("QUEUED", "SNAPSHOTTED"),  # fresh reusable Safety evidence
        ("QUEUED", "DEFERRED"),  # throttle, budget, cooldown, nothing configured
        ("QUEUED", "SUPERSEDED"),
        ("COLLECTING", "SNAPSHOTTED"),  # token evidence observed and snapshotted
        ("COLLECTING", "DEFERRED"),  # budget / cooldown hit during the collection
        ("COLLECTING", "RETRY_WAIT"),  # transient failure / interrupted
        ("COLLECTING", "FAILED"),  # final transient failure
        ("SNAPSHOTTED", "DECIDING"),
        ("DECIDING", "DECIDED"),
        ("DECIDING", "FAILED"),  # conflict / causality / incompatible input
        ("DEFERRED", "QUEUED"),
        ("DEFERRED", "SUPERSEDED"),
        ("RETRY_WAIT", "QUEUED"),
        ("RETRY_WAIT", "SUPERSEDED"),
    }
)
_IMMUTABLE = ("id", "canonical_id", "scout_record_id", "scout_run_time", "market_observed_at",
              "stage", "rank", "created_at")  # fmt: skip


class OrchestratorStorageError(Exception):
    """The orchestrator database refused the operation."""


def _list(values: Any) -> str:
    return ",".join(f"'{v}'" for v in values)


_PAIRS = ",".join(f"('{a}','{b}')" for a, b in sorted(TRANSITIONS))
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS orch_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS orch_jobs (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    scout_record_id TEXT NOT NULL,
    scout_run_time REAL NOT NULL,
    market_observed_at REAL NOT NULL,
    stage TEXT NOT NULL,
    rank INTEGER NOT NULL,
    created_at REAL NOT NULL,
    state TEXT NOT NULL CHECK (state IN ({_list(STATES)})),
    attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_count BETWEEN 0 AND {PROCESSING.max_attempts}),
    next_attempt_at REAL,
    lease_started_at REAL,
    safety_snapshot_id INTEGER,
    safety_as_of REAL,
    opportunity_decision_id INTEGER,
    opportunity_decision_at REAL,
    opportunity_rules_version TEXT,
    category TEXT,
    note TEXT,
    updated_at REAL NOT NULL,
    UNIQUE (canonical_id, scout_record_id),
    CHECK (state NOT IN ('DEFERRED', 'RETRY_WAIT') OR next_attempt_at IS NOT NULL),
    CHECK (state NOT IN ('COLLECTING', 'DECIDING') OR lease_started_at IS NOT NULL),
    CHECK (state NOT IN ('SNAPSHOTTED', 'DECIDING', 'DECIDED') OR safety_as_of IS NOT NULL),
    CHECK (state != 'DECIDED' OR opportunity_decision_id IS NOT NULL),
    CHECK (state NOT IN ('DECIDING', 'DECIDED') OR (opportunity_decision_at IS NOT NULL
           AND opportunity_rules_version IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS orch_jobs_by_token ON orch_jobs (canonical_id, state);
CREATE TABLE IF NOT EXISTS orch_events (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES orch_jobs(id),
    at REAL NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    category TEXT,
    note TEXT
);
CREATE INDEX IF NOT EXISTS orch_events_by_job ON orch_events (job_id, id);
CREATE TRIGGER IF NOT EXISTS orch_jobs_insert_queued BEFORE INSERT ON orch_jobs
WHEN NEW.state != 'QUEUED' OR NEW.attempt_count != 0
BEGIN SELECT RAISE(ABORT, 'a job starts QUEUED with no attempts'); END;
CREATE TRIGGER IF NOT EXISTS orch_jobs_created AFTER INSERT ON orch_jobs
BEGIN INSERT INTO orch_events (job_id, at, from_state, to_state, category, note)
      VALUES (NEW.id, NEW.created_at, NULL, NEW.state, NEW.category, NEW.note); END;
CREATE TRIGGER IF NOT EXISTS orch_jobs_terminal BEFORE UPDATE ON orch_jobs
WHEN OLD.state IN ({_list(TERMINAL)})
BEGIN SELECT RAISE(ABORT, 'a terminal job never changes'); END;
CREATE TRIGGER IF NOT EXISTS orch_jobs_immutable BEFORE UPDATE ON orch_jobs
WHEN {" OR ".join(f"NEW.{c} IS NOT OLD.{c}" for c in _IMMUTABLE)}
BEGIN SELECT RAISE(ABORT, 'admission facts of a job are immutable'); END;
CREATE TRIGGER IF NOT EXISTS orch_jobs_decision_key_once BEFORE UPDATE ON orch_jobs
WHEN (OLD.opportunity_decision_at IS NOT NULL
      AND NEW.opportunity_decision_at IS NOT OLD.opportunity_decision_at)
  OR (OLD.opportunity_rules_version IS NOT NULL
      AND NEW.opportunity_rules_version IS NOT OLD.opportunity_rules_version)
  OR ((OLD.opportunity_decision_at IS NULL AND NEW.opportunity_decision_at IS NOT NULL
       OR OLD.opportunity_rules_version IS NULL AND NEW.opportunity_rules_version IS NOT NULL)
      AND NOT (OLD.state = 'SNAPSHOTTED' AND NEW.state = 'DECIDING'))
BEGIN SELECT RAISE(ABORT, 'the Opportunity decision key is set once, entering DECIDING'); END;
CREATE TRIGGER IF NOT EXISTS orch_jobs_legal BEFORE UPDATE ON orch_jobs
WHEN (OLD.state, NEW.state) NOT IN (VALUES {_PAIRS})
BEGIN SELECT RAISE(ABORT, 'illegal job state transition'); END;
CREATE TRIGGER IF NOT EXISTS orch_jobs_event AFTER UPDATE ON orch_jobs
BEGIN INSERT INTO orch_events (job_id, at, from_state, to_state, category, note)
      VALUES (NEW.id, NEW.updated_at, OLD.state, NEW.state, NEW.category, NEW.note); END;
CREATE TRIGGER IF NOT EXISTS orch_jobs_no_delete BEFORE DELETE ON orch_jobs
BEGIN SELECT RAISE(ABORT, 'jobs are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS orch_events_no_update BEFORE UPDATE ON orch_events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS orch_events_no_delete BEFORE DELETE ON orch_events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""
TABLES = frozenset({"orch_meta", "orch_jobs", "orch_events"})


def _ts(t: datetime | None) -> float | None:
    return t.astimezone(UTC).timestamp() if t is not None else None


def _dt(x: float | None) -> datetime | None:
    return datetime.fromtimestamp(x, UTC) if x is not None else None


@dataclass(frozen=True)
class Job:
    id: int
    canonical_id: str
    scout_record_id: str
    scout_run_time: datetime
    market_observed_at: datetime
    stage: str
    rank: int
    created_at: datetime
    state: JobState
    attempt_count: int
    next_attempt_at: datetime | None
    lease_started_at: datetime | None
    safety_snapshot_id: int | None
    safety_as_of: datetime | None
    opportunity_decision_id: int | None
    opportunity_decision_at: datetime | None
    opportunity_rules_version: str | None
    category: str | None
    note: str | None
    updated_at: datetime

    @property
    def priority(self) -> tuple[int, int, float, str, int]:
        """B1's priority, then the job id."""
        stage = {"ACCELERATING": 0, "EARLY": 1}.get(self.stage, 9)
        return (stage, self.rank, -self.market_observed_at.timestamp(), self.canonical_id, self.id)


_COLS = (
    "id, canonical_id, scout_record_id, scout_run_time, market_observed_at, stage, rank, "
    "created_at, state, attempt_count, next_attempt_at, lease_started_at, safety_snapshot_id, "
    "safety_as_of, opportunity_decision_id, opportunity_decision_at, opportunity_rules_version, "
    "category, note, updated_at"
)


def _job(r: tuple[Any, ...]) -> Job:
    return Job(
        r[0], r[1], r[2], _dt(r[3]), _dt(r[4]), r[5], r[6], _dt(r[7]), r[8], r[9], _dt(r[10]),  # type: ignore[arg-type]
        _dt(r[11]), r[12], _dt(r[13]), r[14], _dt(r[15]), r[16], r[17], r[18], _dt(r[19]),  # type: ignore[arg-type]
    )  # fmt: skip


class OrchestratorRepository:
    """`read_only=True` opens ``mode=ro`` + ``query_only`` and never creates anything."""

    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = Path(path).expanduser()
        self.read_only = read_only
        exists = self.path.exists()
        if exists:
            self._inspect()
        if read_only:
            if not exists:
                raise OrchestratorStorageError(f"no orchestrator database at {self.path}")
            self._conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
            self._conn.execute("PRAGMA query_only = ON")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.executescript(
                "BEGIN IMMEDIATE;" + _SCHEMA
                + "INSERT OR IGNORE INTO orch_meta (key, value) VALUES "
                f"('component', '{ORCHESTRATOR_COMPONENT}'), "
                f"('schema_version', '{ORCHESTRATOR_DB_SCHEMA_VERSION}');COMMIT;"
            )  # fmt: skip
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()
            raise
        self._conn = conn

    def _inspect(self) -> None:
        try:
            ro = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise OrchestratorStorageError(f"can't open {self.path}: {exc}") from exc
        try:
            tables = {r[0] for r in ro.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")}  # fmt: skip
            if not tables:
                return
            foreign = sorted(tables - TABLES)
            if foreign:
                raise OrchestratorStorageError(
                    f"{self.path} holds other components' tables ({', '.join(foreign[:5])})"
                )
            if "orch_meta" not in tables:
                raise OrchestratorStorageError(
                    f"{self.path} has orchestrator tables but no metadata"
                )
            meta = dict(ro.execute("SELECT key, value FROM orch_meta").fetchall())
        except sqlite3.DatabaseError as exc:
            raise OrchestratorStorageError(f"{self.path} isn't a readable database: {exc}") from exc
        finally:
            ro.close()
        if meta.get("component") != ORCHESTRATOR_COMPONENT:
            raise OrchestratorStorageError(f"{self.path} metadata isn't the orchestrator's")
        if meta.get("schema_version") != str(ORCHESTRATOR_DB_SCHEMA_VERSION):
            raise OrchestratorStorageError(
                f"{self.path} has orchestrator schema version {meta.get('schema_version')!r}; "
                f"this UpScale knows {ORCHESTRATOR_DB_SCHEMA_VERSION} (never migrated)"
            )

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise OrchestratorStorageError("this repository is read-only")
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # --- jobs ----------------------------------------------------------------------------------

    def enqueue(self, a: AdmissionResult, at: datetime) -> tuple[Job, bool]:
        """One job per admitted record; the exact same record returns the existing job."""
        if not a.admitted or a.stage is None or a.rank is None or a.market_observed_at is None:
            raise OrchestratorStorageError("only an admitted candidate becomes a job")
        with self._tx() as db:
            row = db.execute(f"SELECT {_COLS} FROM orch_jobs WHERE canonical_id = ? AND "
                             "scout_record_id = ?", (a.canonical_id, a.scout_record_id)).fetchone()  # fmt: skip
            if row is not None:
                return _job(row), False
            cur = db.execute(
                "INSERT INTO orch_jobs (canonical_id, scout_record_id, scout_run_time, "
                "market_observed_at, stage, rank, created_at, state, updated_at, category, note) "
                "VALUES (?,?,?,?,?,?,?, 'QUEUED', ?, 'ADMITTED', ?)",
                (a.canonical_id, a.scout_record_id, _ts(a.scout_run_time),
                 _ts(a.market_observed_at), a.stage, a.rank, _ts(at), _ts(at),
                 ",".join(a.reasons)),
            )  # fmt: skip
            row = db.execute(f"SELECT {_COLS} FROM orch_jobs WHERE id = ?", (cur.lastrowid,))
            return _job(row.fetchone()), True

    def transition(
        self, job: Job, to: JobState, at: datetime, category: str | None = None,
        note: str | None = None, **fields: Any,
    ) -> Job:  # fmt: skip
        """Move `job` (still in its read state) to `to`; the event is written atomically by
        trigger. Raises if the job changed meanwhile or the transition is illegal."""
        allowed = {"attempt_count", "next_attempt_at", "lease_started_at", "safety_snapshot_id",
                   "safety_as_of", "opportunity_decision_id", "opportunity_decision_at",
                   "opportunity_rules_version"}  # fmt: skip
        if set(fields) - allowed:
            raise OrchestratorStorageError(f"not lifecycle fields: {sorted(set(fields) - allowed)}")
        values = {k: (_ts(v) if isinstance(v, datetime) else v) for k, v in fields.items()}
        values |= {"state": to, "updated_at": _ts(at), "category": category, "note": note}
        sets = ", ".join(f"{k} = ?" for k in values)
        with self._tx() as db:
            cur = db.execute(f"UPDATE orch_jobs SET {sets} WHERE id = ? AND state = ?",
                             [*values.values(), job.id, job.state])  # fmt: skip
            if cur.rowcount != 1:
                raise OrchestratorStorageError(f"job {job.id} is no longer {job.state}")
            row = db.execute(f"SELECT {_COLS} FROM orch_jobs WHERE id = ?", (job.id,)).fetchone()
        return _job(row)

    def job(self, job_id: int) -> Job:
        row = self._conn.execute(
            f"SELECT {_COLS} FROM orch_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise OrchestratorStorageError(f"no job {job_id}")
        return _job(row)

    def jobs(
        self, states: tuple[JobState, ...] | None = None, canonical_id: str | None = None
    ) -> list[Job]:
        where: list[str] = ["1 = 1"]
        args: list[str] = []
        if states:
            where.append(f"state IN ({','.join('?' for _ in states)})")
            args += list(states)
        if canonical_id:
            where.append("canonical_id = ?")
            args.append(canonical_id)
        rows = self._conn.execute(f"SELECT {_COLS} FROM orch_jobs WHERE {' AND '.join(where)} "
                                  "ORDER BY id", args).fetchall()  # fmt: skip
        return [_job(r) for r in rows]

    def events(self, job_id: int) -> list[dict[str, Any]]:
        cur = self._conn.execute("SELECT at, from_state, to_state, category, note FROM orch_events "
                                 "WHERE job_id = ? ORDER BY id", (job_id,))  # fmt: skip
        return [{"at": _dt(r[0]), "from": r[1], "to": r[2], "category": r[3], "note": r[4]}
                for r in cur.fetchall()]  # fmt: skip

    def collection_starts(self, canonical_id: str) -> list[datetime]:
        """When collections for this token started (every QUEUED -> COLLECTING event)."""
        rows = self._conn.execute(
            "SELECT e.at FROM orch_events e JOIN orch_jobs j ON j.id = e.job_id "
            "WHERE j.canonical_id = ? AND e.to_state = 'COLLECTING' ORDER BY e.at",
            (canonical_id,),
        ).fetchall()
        return [_dt(r[0]) for r in rows]  # type: ignore[misc]

    def collection_successes(self, canonical_id: str) -> list[datetime]:
        """When collections for this token produced a snapshot (COLLECTING -> SNAPSHOTTED)."""
        rows = self._conn.execute(
            "SELECT e.at FROM orch_events e JOIN orch_jobs j ON j.id = e.job_id "
            "WHERE j.canonical_id = ? AND e.from_state = 'COLLECTING' "
            "AND e.to_state = 'SNAPSHOTTED' ORDER BY e.at",
            (canonical_id,),
        ).fetchall()
        return [_dt(r[0]) for r in rows]  # type: ignore[misc]

    def counts(self) -> dict[str, int]:
        return dict(self._conn.execute("SELECT state, COUNT(*) FROM orch_jobs GROUP BY state "
                                       "ORDER BY state").fetchall())  # fmt: skip
