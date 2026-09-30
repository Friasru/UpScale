"""The replay database (SQLite, standard library): HISTORICAL_REPLAY data only.

Isolation: it refuses to open the live Scout / outcome database (by path, and by content:
a file holding live Scout or outcome tables is never used), and every job, sample,
decision and outcome row carries ``origin = 'HISTORICAL_REPLAY'`` (enforced by CHECK).

Schema (version 1):

* ``replay_jobs``: config, status (PENDING / RUNNING / PAUSED / COMPLETE / FAILED),
  timestamps, counters, provider usage, split configuration, runner lease.
* ``replay_samples``: the planned decision times, one row per (job, sample key), each in
  exactly one split. Identity, decision time and split can't change (triggers).
* ``replay_decisions``: the frozen decision record per sample (JSON + SHA-256). Immutable:
  updates and deletes are refused by triggers.
* ``replay_outcomes`` / ``replay_reveals``: horizon outcomes and the post-decision
  fidelity check. Insertable only once the sample's decision exists; immutable.
* ``candle_cache``: immutable historical candles by provider, chain, token, pool,
  timeframe and [start, end).
* ``holdout_windows`` / ``holdout_access_log``: HOLDOUT time ranges (sticky across jobs)
  and every explicit final-evaluation read of HOLDOUT data.
* ``calibration_findings``: EXPERIMENTAL findings (never HOLDOUT-derived); nothing here
  is production configuration.
* ``shadow_strategies`` / ``shadow_evaluations``: hypothetical strategies evaluated later
  against stored samples without re-ingesting data (empty in v1).
"""

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from upscale.services.market_data import Candle
from upscale.services.replay_lab.clock import DecisionReceipt, utc
from upscale.services.replay_lab.config import (
    HISTORICAL_REPLAY,
    JobStatus,
    ReplayJobConfig,
    SampleStatus,
    Split,
)
from upscale.services.replay_lab.models import (
    PlannedSample,
    ReplayDecisionRecord,
    ReplayFidelity,
    ReplayHorizonOutcome,
)

SCHEMA_VERSION = 1
# Tables that only ever exist in the live Scout / outcome database.
LIVE_TABLES = frozenset(
    {"scout_snapshots", "scout_tokens", "scout_outcome_observations", "decision_observations"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS replay_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS replay_jobs (
    job_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL CHECK (origin = 'HISTORICAL_REPLAY'),
    config_json TEXT NOT NULL,
    split_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','PAUSED','COMPLETE','FAILED')),
    status_reason TEXT,
    last_sample_id INTEGER,
    planned INTEGER NOT NULL DEFAULT 0,
    planning_json TEXT,
    provider_usage_json TEXT,
    heartbeat_at REAL,
    runner_pid INTEGER
);
CREATE TABLE IF NOT EXISTS replay_samples (
    id INTEGER PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES replay_jobs(job_id),
    origin TEXT NOT NULL CHECK (origin = 'HISTORICAL_REPLAY'),
    sample_key TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    chain TEXT NOT NULL,
    token_address TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    symbol TEXT,
    decision_at REAL NOT NULL,
    evidence TEXT NOT NULL,
    universe_basis TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('CALIBRATION','VALIDATION','HOLDOUT')),
    purged INTEGER NOT NULL,
    cohort_at REAL NOT NULL,
    plan_order INTEGER NOT NULL,
    snapshot_provider TEXT,
    market_observed_at REAL,
    status TEXT NOT NULL CHECK (status IN ('PLANNED','DECIDED','COMPLETE','SKIPPED','FAILED')),
    skip_reason TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    UNIQUE (job_id, sample_key)
);
CREATE INDEX IF NOT EXISTS replay_samples_by_job ON replay_samples (job_id, plan_order);
CREATE TABLE IF NOT EXISTS replay_decisions (
    sample_id INTEGER PRIMARY KEY REFERENCES replay_samples(id),
    origin TEXT NOT NULL CHECK (origin = 'HISTORICAL_REPLAY'),
    decision_at REAL NOT NULL,
    decided_at REAL NOT NULL,
    record_json TEXT NOT NULL,
    record_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS replay_outcomes (
    sample_id INTEGER NOT NULL REFERENCES replay_decisions(sample_id),
    horizon TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    origin TEXT NOT NULL CHECK (origin = 'HISTORICAL_REPLAY'),
    status TEXT NOT NULL,
    record_json TEXT NOT NULL,
    revealed_at REAL NOT NULL,
    PRIMARY KEY (sample_id, horizon)
);
CREATE TABLE IF NOT EXISTS replay_reveals (
    sample_id INTEGER PRIMARY KEY REFERENCES replay_decisions(sample_id),
    revealed_at REAL NOT NULL,
    fidelity_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candle_cache (
    provider TEXT NOT NULL,
    chain TEXT NOT NULL,
    token_address TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    fetched_at REAL NOT NULL,
    complete INTEGER NOT NULL,
    candles_json TEXT NOT NULL,
    PRIMARY KEY (provider, chain, token_address, pool_address, timeframe, start_ts, end_ts)
);
CREATE TABLE IF NOT EXISTS holdout_windows (
    job_id TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS holdout_access_log (
    at REAL NOT NULL,
    purpose TEXT NOT NULL,
    job_id TEXT
);
CREATE TABLE IF NOT EXISTS calibration_findings (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    status TEXT NOT NULL CHECK (status = 'EXPERIMENTAL'),
    split_used TEXT NOT NULL CHECK (split_used = 'CALIBRATION'),
    job_ids TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    horizon TEXT NOT NULL,
    statement TEXT NOT NULL,
    validation TEXT NOT NULL,
    evidence_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_strategies (
    strategy_id TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT NOT NULL,
    params_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (strategy_id, version)
);
CREATE TABLE IF NOT EXISTS shadow_evaluations (
    sample_id INTEGER NOT NULL REFERENCES replay_decisions(sample_id),
    strategy_id TEXT NOT NULL,
    version TEXT NOT NULL,
    horizon TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (sample_id, strategy_id, version, horizon)
);

CREATE TRIGGER IF NOT EXISTS replay_decisions_no_update BEFORE UPDATE ON replay_decisions
BEGIN SELECT RAISE(ABORT, 'replay decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS replay_decisions_no_delete BEFORE DELETE ON replay_decisions
BEGIN SELECT RAISE(ABORT, 'replay decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS replay_outcomes_no_update BEFORE UPDATE ON replay_outcomes
BEGIN SELECT RAISE(ABORT, 'replay outcomes are immutable'); END;
CREATE TRIGGER IF NOT EXISTS replay_outcomes_no_delete BEFORE DELETE ON replay_outcomes
BEGIN SELECT RAISE(ABORT, 'replay outcomes are immutable'); END;
CREATE TRIGGER IF NOT EXISTS replay_reveals_no_update BEFORE UPDATE ON replay_reveals
BEGIN SELECT RAISE(ABORT, 'replay reveals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS replay_outcomes_after_decision BEFORE INSERT ON replay_outcomes
WHEN NOT EXISTS (SELECT 1 FROM replay_decisions d WHERE d.sample_id = NEW.sample_id)
    OR NEW.revealed_at < (SELECT d.decided_at FROM replay_decisions d
                          WHERE d.sample_id = NEW.sample_id)
BEGIN SELECT RAISE(ABORT, 'outcomes can only be revealed after the decision is stored'); END;
CREATE TRIGGER IF NOT EXISTS replay_samples_fixed BEFORE UPDATE ON replay_samples
WHEN NEW.split != OLD.split OR NEW.decision_at != OLD.decision_at
    OR NEW.sample_key != OLD.sample_key OR NEW.pool_address != OLD.pool_address
    OR NEW.asset_id != OLD.asset_id
BEGIN SELECT RAISE(ABORT, 'a sample''s identity, time and split are fixed'); END;
CREATE TRIGGER IF NOT EXISTS replay_samples_no_delete BEFORE DELETE ON replay_samples
BEGIN SELECT RAISE(ABORT, 'replay samples are never deleted'); END;
"""


class ReplayStoreError(Exception):
    pass


class ReplayIsolationError(ReplayStoreError):
    """The replay store was pointed at live (LIVE_FORWARD) data."""


class DecisionIntegrityError(ReplayStoreError):
    """A stored decision no longer matches its hash."""


@dataclass(frozen=True)
class StoredSample:
    id: int
    status: SampleStatus
    attempts: int
    plan: PlannedSample
    skip_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class JobRow:
    job_id: str
    config: ReplayJobConfig
    status: JobStatus
    status_reason: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    planned: int
    counts: dict[str, int]
    splits: dict[str, int]
    provider_usage: dict[str, Any]
    planning: dict[str, Any]
    heartbeat_at: datetime | None
    runner_pid: int | None
    last_sample_id: int | None


@dataclass(frozen=True)
class AnalysisRow:
    """One sample with its frozen decision and one horizon's outcome (None: not revealed)."""

    sample: PlannedSample
    decision: ReplayDecisionRecord
    outcome: ReplayHorizonOutcome | None
    job_id: str


def check_isolation(path: str, forbidden: Iterable[str]) -> None:
    """Refuse the live database: by path, and by content (live Scout / outcome tables)."""
    if path == ":memory:":
        return
    target = Path(path).expanduser().resolve()
    for other in forbidden:
        if other and other != ":memory:" and target == Path(other).expanduser().resolve():
            raise ReplayIsolationError(
                f"{target} is the live Scout / outcome database: replay data must stay separate"
            )
    if target.exists() and target.stat().st_size > 0:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        try:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        except sqlite3.DatabaseError as exc:
            raise ReplayIsolationError(f"{target} is not a replay database: {exc}") from exc
        finally:
            conn.close()
        live = names & LIVE_TABLES
        if live:
            raise ReplayIsolationError(
                f"{target} holds live tables ({', '.join(sorted(live))}): replay data must "
                "stay separate from LIVE_FORWARD data"
            )


def record_hash(record_json: str) -> str:
    return hashlib.sha256(record_json.encode()).hexdigest()


class ReplayStore:
    """Synchronous: Replay Lab is a single-process CLI workload."""

    def __init__(
        self,
        path: str | Path,
        forbidden: Sequence[str] = (),
        clock: Any = time.time,
    ):
        self.path = str(path)
        check_isolation(self.path, forbidden)
        self._clock = clock
        self._conn: sqlite3.Connection | None = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path != ":memory:":
                Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(Path(self.path).expanduser()) if self.path != ":memory:" else self.path
            )
            conn.execute("PRAGMA foreign_keys = ON")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                conn.close()
                raise ReplayStoreError(
                    f"replay store {self.path} has schema v{version}; this UpScale knows "
                    f"v{SCHEMA_VERSION}"
                )
            conn.executescript(_SCHEMA)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(replay_samples)")}
            if "market_observed_at" not in columns:  # stores created before decision timing
                conn.execute("ALTER TABLE replay_samples ADD COLUMN market_observed_at REAL")
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO replay_meta (key, value) VALUES ('origin', ?)",
                    (HISTORICAL_REPLAY,),
                )
            origin = conn.execute("SELECT value FROM replay_meta WHERE key = 'origin'").fetchone()
            if origin is None or origin[0] != HISTORICAL_REPLAY:
                conn.close()
                raise ReplayIsolationError(f"{self.path} is not a HISTORICAL_REPLAY store")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            self._conn = conn
        return self._conn

    # --- jobs ---------------------------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        config: ReplayJobConfig,
        samples: Sequence[PlannedSample],
        planning: dict[str, Any],
    ) -> None:
        db = self._db()
        now = self._clock()
        with db:
            db.execute(
                """
                INSERT INTO replay_jobs (job_id, origin, config_json, split_json, created_at,
                    status, planned, planning_json, provider_usage_json)
                VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?, '{}')
                """,
                (
                    job_id,
                    HISTORICAL_REPLAY,
                    config.model_dump_json(),
                    config.split.model_dump_json(),
                    now,
                    len(samples),
                    json.dumps(planning, default=str),
                ),
            )
            db.executemany(
                """
                INSERT INTO replay_samples (job_id, origin, sample_key, asset_id, chain,
                    token_address, pool_address, symbol, decision_at, evidence, universe_basis,
                    split, purged, cohort_at, plan_order, snapshot_provider, market_observed_at,
                    status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PLANNED', ?)
                """,
                [
                    (
                        job_id,
                        HISTORICAL_REPLAY,
                        s.sample_key,
                        s.asset_id,
                        s.chain,
                        s.token_address,
                        s.pool_address,
                        s.symbol,
                        s.decision_at.timestamp(),
                        s.evidence,
                        s.universe_basis,
                        s.split,
                        int(s.purged),
                        s.cohort_at.timestamp(),
                        s.plan_order,
                        s.snapshot_provider,
                        s.market_observed_at.timestamp() if s.market_observed_at else None,
                        now,
                    )
                    for s in samples
                ],
            )
            holdout = [s.decision_at.timestamp() for s in samples if s.split == "HOLDOUT"]
            if holdout:
                db.execute(
                    "INSERT INTO holdout_windows (job_id, start_ts, end_ts) VALUES (?, ?, ?)",
                    (job_id, min(holdout), max(holdout)),
                )

    def job_ids(self) -> list[str]:
        rows = self._db().execute("SELECT job_id FROM replay_jobs ORDER BY created_at").fetchall()
        return [r[0] for r in rows]

    def job(self, job_id: str) -> JobRow | None:
        db = self._db()
        row = db.execute(
            """
            SELECT job_id, config_json, status, status_reason, created_at, started_at,
                finished_at, planned, provider_usage_json, planning_json, heartbeat_at,
                runner_pid, last_sample_id
            FROM replay_jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        counts = dict(
            db.execute(
                "SELECT status, COUNT(*) FROM replay_samples WHERE job_id = ? GROUP BY status",
                (job_id,),
            ).fetchall()
        )
        splits = dict(
            db.execute(
                "SELECT split, COUNT(*) FROM replay_samples WHERE job_id = ? GROUP BY split",
                (job_id,),
            ).fetchall()
        )
        return JobRow(
            job_id=row[0],
            config=ReplayJobConfig.model_validate_json(row[1]),
            status=row[2],
            status_reason=row[3],
            created_at=utc(row[4]),
            started_at=utc(row[5]) if row[5] is not None else None,
            finished_at=utc(row[6]) if row[6] is not None else None,
            planned=row[7],
            counts={k: int(v) for k, v in counts.items()},
            splits={k: int(v) for k, v in splits.items()},
            provider_usage=json.loads(row[8] or "{}"),
            planning=json.loads(row[9] or "{}"),
            heartbeat_at=utc(row[10]) if row[10] is not None else None,
            runner_pid=row[11],
            last_sample_id=row[12],
        )

    def set_job_status(self, job_id: str, status: JobStatus, reason: str | None = None) -> None:
        now = self._clock()
        db = self._db()
        with db:
            db.execute(
                """
                UPDATE replay_jobs SET status = ?, status_reason = ?,
                    started_at = CASE WHEN ? = 'RUNNING' AND started_at IS NULL THEN ?
                                      ELSE started_at END,
                    finished_at = CASE WHEN ? IN ('COMPLETE', 'FAILED') THEN ?
                                       ELSE finished_at END
                WHERE job_id = ?
                """,
                (status, reason, status, now, status, now, job_id),
            )

    def claim_job(self, job_id: str, pid: int, stale_seconds: float, force: bool = False) -> bool:
        """Take the runner lease: False while another live runner holds it (`force`: the
        holder is known to be gone, e.g. its process no longer exists)."""
        now = self._clock()
        db = self._db()
        with db:
            cur = db.execute(
                """
                UPDATE replay_jobs SET runner_pid = ?, heartbeat_at = ?
                WHERE job_id = ? AND (? OR runner_pid IS NULL OR heartbeat_at IS NULL
                    OR heartbeat_at < ? OR runner_pid = ?)
                """,
                (pid, now, job_id, int(force), now - stale_seconds, pid),
            )
        return cur.rowcount == 1

    def heartbeat(self, job_id: str, last_sample_id: int | None = None) -> None:
        db = self._db()
        with db:
            db.execute(
                """
                UPDATE replay_jobs SET heartbeat_at = ?,
                    last_sample_id = COALESCE(?, last_sample_id) WHERE job_id = ?
                """,
                (self._clock(), last_sample_id, job_id),
            )

    def release_job(self, job_id: str) -> None:
        db = self._db()
        with db:
            db.execute(
                "UPDATE replay_jobs SET runner_pid = NULL, heartbeat_at = NULL WHERE job_id = ?",
                (job_id,),
            )

    def save_usage(self, job_id: str, usage: dict[str, Any]) -> None:
        db = self._db()
        with db:
            db.execute(
                "UPDATE replay_jobs SET provider_usage_json = ? WHERE job_id = ?",
                (json.dumps(usage), job_id),
            )

    # --- samples ------------------------------------------------------------------------------

    def samples(self, job_id: str, statuses: Sequence[str] | None = None) -> list[StoredSample]:
        sql = f"""
            SELECT id, status, attempts, skip_reason, error, sample_key, asset_id, chain,
                token_address, pool_address, symbol, decision_at, evidence, universe_basis,
                split, purged, cohort_at, plan_order, snapshot_provider, market_observed_at
            FROM replay_samples WHERE job_id = ?
            {"AND status IN (" + ",".join("?" * len(statuses)) + ")" if statuses else ""}
            ORDER BY plan_order
        """
        rows = self._db().execute(sql, [job_id, *(statuses or [])]).fetchall()
        return [_stored(r) for r in rows]

    def mark_sample(
        self,
        sample_id: int,
        status: SampleStatus,
        *,
        skip_reason: str | None = None,
        error: str | None = None,
        attempt: bool = False,
    ) -> None:
        db = self._db()
        with db:
            cur = db.execute(
                """
                UPDATE replay_samples SET status = ?, skip_reason = ?, error = ?,
                    attempts = attempts + ?, updated_at = ?
                WHERE id = ? AND status NOT IN ('COMPLETE')
                """,
                (status, skip_reason, error, int(attempt), self._clock(), sample_id),
            )
        if cur.rowcount != 1:
            raise ReplayStoreError(f"sample {sample_id} is complete (or unknown): not changed")

    def note_attempt(self, sample_id: int, error: str | None) -> int:
        db = self._db()
        with db:
            db.execute(
                "UPDATE replay_samples SET attempts = attempts + 1, error = ?, updated_at = ? "
                "WHERE id = ?",
                (error, self._clock(), sample_id),
            )
        row = db.execute(
            "SELECT attempts FROM replay_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        return int(row[0])

    def holdout_windows(self) -> list[tuple[datetime, datetime]]:
        rows = self._db().execute("SELECT start_ts, end_ts FROM holdout_windows").fetchall()
        return [(utc(a), utc(b)) for a, b in rows]

    # --- decisions and outcomes ---------------------------------------------------------------

    def add_decision(self, sample_id: int, record: ReplayDecisionRecord) -> DecisionReceipt:
        """Freeze the decision (and mark the sample DECIDED) in one transaction."""
        body = record.model_dump_json()
        digest = record_hash(body)
        db = self._db()
        row = db.execute(
            "SELECT decision_at, status FROM replay_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        if row is None:
            raise ReplayStoreError(f"unknown sample {sample_id}")
        if abs(row[0] - record.decision_at.timestamp()) > 1e-6:
            raise ReplayStoreError("the decision's time differs from the planned sample time")
        try:
            with db:
                db.execute(
                    """
                    INSERT INTO replay_decisions (sample_id, origin, decision_at, decided_at,
                        record_json, record_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        HISTORICAL_REPLAY,
                        record.decision_at.timestamp(),
                        self._clock(),
                        body,
                        digest,
                    ),
                )
                db.execute(
                    "UPDATE replay_samples SET status = 'DECIDED', error = NULL, updated_at = ? "
                    "WHERE id = ?",
                    (self._clock(), sample_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ReplayStoreError(f"sample {sample_id} already has a decision") from exc
        return DecisionReceipt(sample_id, record.decision_at, digest)

    def decision(self, sample_id: int) -> tuple[ReplayDecisionRecord, DecisionReceipt] | None:
        row = (
            self._db()
            .execute(
                "SELECT record_json, record_hash FROM replay_decisions WHERE sample_id = ?",
                (sample_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        body, digest = row
        if record_hash(body) != digest:
            raise DecisionIntegrityError(f"decision of sample {sample_id} fails its hash check")
        record = ReplayDecisionRecord.model_validate_json(body)
        return record, DecisionReceipt(sample_id, record.decision_at, digest)

    def add_outcomes(
        self,
        receipt: DecisionReceipt,
        outcomes: Sequence[ReplayHorizonOutcome],
        fidelity: ReplayFidelity,
    ) -> None:
        """Store what happened after T and complete the sample, in one transaction. The
        decision must exist and still match its receipt."""
        stored = self.decision(receipt.sample_id)
        if stored is None or stored[1].record_hash != receipt.record_hash:
            raise ReplayStoreError("outcomes need the exact stored decision they measure")
        db = self._db()
        now = self._clock()
        with db:
            db.executemany(
                """
                INSERT INTO replay_outcomes (sample_id, horizon, horizon_minutes, origin, status,
                    record_json, revealed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        receipt.sample_id,
                        o.horizon,
                        o.horizon_minutes,
                        HISTORICAL_REPLAY,
                        o.status,
                        o.model_dump_json(),
                        now,
                    )
                    for o in outcomes
                ],
            )
            db.execute(
                "INSERT INTO replay_reveals (sample_id, revealed_at, fidelity_json) "
                "VALUES (?, ?, ?)",
                (receipt.sample_id, now, fidelity.model_dump_json()),
            )
            db.execute(
                "UPDATE replay_samples SET status = 'COMPLETE', error = NULL, updated_at = ? "
                "WHERE id = ?",
                (now, receipt.sample_id),
            )

    def outcomes(self, sample_id: int) -> list[ReplayHorizonOutcome]:
        rows = (
            self._db()
            .execute(
                "SELECT record_json FROM replay_outcomes WHERE sample_id = ? "
                "ORDER BY horizon_minutes",
                (sample_id,),
            )
            .fetchall()
        )
        return [ReplayHorizonOutcome.model_validate_json(r[0]) for r in rows]

    def fidelity(self, sample_id: int) -> ReplayFidelity | None:
        row = (
            self._db()
            .execute("SELECT fidelity_json FROM replay_reveals WHERE sample_id = ?", (sample_id,))
            .fetchone()
        )
        return ReplayFidelity.model_validate_json(row[0]) if row else None

    # --- analysis reads ------------------------------------------------------------------------

    def analysis_rows(
        self,
        horizon: str,
        splits: Sequence[Split],
        job_ids: Sequence[str] | None = None,
        include_purged: bool = True,
    ) -> list[AnalysisRow]:
        where = [f"s.split IN ({','.join('?' * len(splits))})"]
        params: list[Any] = [*splits]
        if job_ids:
            where.append(f"s.job_id IN ({','.join('?' * len(job_ids))})")
            params += list(job_ids)
        if not include_purged:
            where.append("s.purged = 0")
        rows = (
            self._db()
            .execute(
                f"""
                SELECT s.id, s.status, s.attempts, s.skip_reason, s.error, s.sample_key,
                    s.asset_id, s.chain, s.token_address, s.pool_address, s.symbol,
                    s.decision_at, s.evidence, s.universe_basis, s.split, s.purged,
                    s.cohort_at, s.plan_order, s.snapshot_provider, s.market_observed_at,
                    d.record_json, d.record_hash, o.record_json, s.job_id
                FROM replay_samples s
                JOIN replay_decisions d ON d.sample_id = s.id
                LEFT JOIN replay_outcomes o ON o.sample_id = s.id AND o.horizon = ?
                WHERE {" AND ".join(where)}
                ORDER BY s.decision_at, s.id
                """,
                [horizon, *params],
            )
            .fetchall()
        )
        out = []
        for r in rows:
            if record_hash(r[20]) != r[21]:
                raise DecisionIntegrityError(f"decision of sample {r[0]} fails its hash check")
            out.append(
                AnalysisRow(
                    sample=_stored(r[:20]).plan,
                    decision=ReplayDecisionRecord.model_validate_json(r[20]),
                    outcome=ReplayHorizonOutcome.model_validate_json(r[22]) if r[22] else None,
                    job_id=r[23],
                )
            )
        return out

    def log_holdout_access(self, purpose: str, job_id: str | None) -> None:
        db = self._db()
        with db:
            db.execute(
                "INSERT INTO holdout_access_log (at, purpose, job_id) VALUES (?, ?, ?)",
                (self._clock(), purpose, job_id),
            )

    def holdout_accesses(self) -> int:
        return int(self._db().execute("SELECT COUNT(*) FROM holdout_access_log").fetchone()[0])

    # --- findings -----------------------------------------------------------------------------

    def add_findings(self, findings: Sequence[dict[str, Any]]) -> int:
        db = self._db()
        now = self._clock()
        with db:
            db.executemany(
                """
                INSERT INTO calibration_findings (created_at, status, split_used, job_ids, kind,
                    subject, horizon, statement, validation, evidence_json)
                VALUES (?, 'EXPERIMENTAL', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        now,
                        f["split_used"],
                        json.dumps(f["job_ids"]),
                        f["kind"],
                        f["subject"],
                        f["horizon"],
                        f["statement"],
                        f["validation"],
                        json.dumps(f["evidence"], default=str),
                    )
                    for f in findings
                ],
            )
        return len(findings)

    def findings(self) -> list[dict[str, Any]]:
        rows = (
            self._db()
            .execute(
                "SELECT created_at, status, split_used, kind, subject, horizon, statement, "
                "validation FROM calibration_findings ORDER BY id"
            )
            .fetchall()
        )
        keys = ("created_at", "status", "split_used", "kind", "subject", "horizon", "statement",
                "validation")  # fmt: skip
        return [dict(zip(keys, r, strict=True)) for r in rows]

    # --- candle cache -------------------------------------------------------------------------

    def cached_candles(
        self,
        provider: str,
        chain: str,
        token: str,
        pool: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> tuple[list[Candle], float, bool] | None:
        """(candles, fetched_at, complete) for exactly this cached range, or None. A
        corrupt entry is dropped (a cache failure never affects stored observations)."""
        db = self._db()
        key = (provider, chain, token, pool, timeframe, start.timestamp(), end.timestamp())
        row = db.execute(
            """
            SELECT candles_json, fetched_at, complete FROM candle_cache
            WHERE provider = ? AND chain = ? AND token_address = ? AND pool_address = ?
                AND timeframe = ? AND start_ts = ? AND end_ts = ?
            """,
            key,
        ).fetchone()
        if row is None:
            return None
        try:
            candles = [Candle.model_validate(c) for c in json.loads(row[0])]
        except (ValueError, TypeError):
            with db:
                db.execute(
                    """
                    DELETE FROM candle_cache WHERE provider = ? AND chain = ? AND token_address = ?
                        AND pool_address = ? AND timeframe = ? AND start_ts = ? AND end_ts = ?
                    """,
                    key,
                )
            return None
        return candles, float(row[1]), bool(row[2])

    def cache_candles(
        self,
        provider: str,
        chain: str,
        token: str,
        pool: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        candles: Sequence[Candle],
        fetched_at: float,
        complete: bool,
    ) -> None:
        db = self._db()
        with db:
            db.execute(
                """
                INSERT OR REPLACE INTO candle_cache (provider, chain, token_address, pool_address,
                    timeframe, start_ts, end_ts, fetched_at, complete, candles_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    chain,
                    token,
                    pool,
                    timeframe,
                    start.timestamp(),
                    end.timestamp(),
                    fetched_at,
                    int(complete),
                    json.dumps([c.model_dump(mode="json") for c in candles]),
                ),
            )

    def cache_entries(self) -> int:
        return int(self._db().execute("SELECT COUNT(*) FROM candle_cache").fetchone()[0])


def _stored(r: Sequence[Any]) -> StoredSample:
    return StoredSample(
        id=r[0],
        status=r[1],
        attempts=r[2],
        skip_reason=r[3],
        error=r[4],
        plan=PlannedSample(
            sample_key=r[5],
            asset_id=r[6],
            chain=r[7],
            token_address=r[8],
            pool_address=r[9],
            symbol=r[10],
            decision_at=utc(r[11]),
            evidence=r[12],
            universe_basis=r[13],
            split=r[14],
            purged=bool(r[15]),
            cohort_at=utc(r[16]),
            plan_order=r[17],
            snapshot_provider=r[18],
            market_observed_at=utc(r[19]) if len(r) > 19 and r[19] is not None else None,
        ),
    )


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
