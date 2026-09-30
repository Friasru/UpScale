"""Calibration storage (``UPSCALE_CALIBRATION_DB``, default ``calibration.sqlite3`` next to
the Scout database). Research output only: nothing here is production configuration.

Tables: ``calibration_runs``, ``calibration_findings``, ``feature_statistics``,
``calibration_candidates``, ``candidate_parameters``, ``candidate_metrics``,
``candidate_lineage``, ``validation_runs``, ``final_evaluations``, ``holdout_access_log``.

Append-only (triggers), except a candidate's status, which may only move forward:

    DRAFT -> CALIBRATED -> VALIDATED | REJECTED
    VALIDATED -> FROZEN_FOR_FINAL_TEST -> PROMOTED_MANUALLY

A candidate's parameters, reason and lineage never change: an altered candidate is a new
candidate id. PROMOTED_MANUALLY is only recorded on an explicit human command and changes
no configuration.
"""

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATUSES = (
    "DRAFT",
    "CALIBRATED",
    "VALIDATED",
    "REJECTED",
    "FROZEN_FOR_FINAL_TEST",
    "PROMOTED_MANUALLY",
)
APPEND_ONLY = (
    "calibration_runs", "calibration_findings", "feature_statistics", "candidate_parameters",
    "candidate_metrics", "candidate_lineage", "validation_runs", "final_evaluations",
    "holdout_access_log",
)  # fmt: skip

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS calibration_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS calibration_runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    args_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    versions_json TEXT NOT NULL,
    dataset_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    comparisons INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS calibration_findings (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES calibration_runs(run_id),
    finding_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    horizon TEXT NOT NULL,
    strength TEXT NOT NULL,
    statement TEXT NOT NULL,
    body_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feature_statistics (
    run_id TEXT NOT NULL REFERENCES calibration_runs(run_id),
    feature TEXT NOT NULL,
    bucket TEXT NOT NULL,
    horizon TEXT NOT NULL,
    origin TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('CALIBRATION', 'VALIDATION')),
    body_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration_candidates (
    candidate_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    parent_version_json TEXT NOT NULL,
    parent_candidate_id TEXT,
    source_run_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    source_findings_json TEXT NOT NULL,
    changes_json TEXT NOT NULL,
    complexity_json TEXT NOT NULL,
    origin_mix_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({",".join(repr(s) for s in STATUSES)})),
    status_updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_parameters (
    candidate_id TEXT NOT NULL REFERENCES calibration_candidates(candidate_id),
    parameter TEXT NOT NULL,
    from_value TEXT,
    to_value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_metrics (
    candidate_id TEXT NOT NULL REFERENCES calibration_candidates(candidate_id),
    phase TEXT NOT NULL CHECK (phase IN ('CALIBRATION', 'VALIDATION', 'HOLDOUT')),
    created_at REAL NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_lineage (
    candidate_id TEXT NOT NULL REFERENCES calibration_candidates(candidate_id),
    parent_candidate_id TEXT,
    source_run_id TEXT NOT NULL,
    note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES calibration_candidates(candidate_id),
    created_at REAL NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    outcome TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS final_evaluations (
    id INTEGER PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES calibration_candidates(candidate_id),
    created_at REAL NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holdout_access_log (
    id INTEGER PRIMARY KEY,
    at REAL NOT NULL,
    candidate_id TEXT,
    purpose TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    holdout_window_json TEXT NOT NULL,
    data_version TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS candidates_no_delete BEFORE DELETE ON calibration_candidates
BEGIN SELECT RAISE(ABORT, 'candidates are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS candidates_fixed BEFORE UPDATE ON calibration_candidates
WHEN NEW.candidate_id != OLD.candidate_id OR NEW.changes_json != OLD.changes_json
    OR NEW.reason != OLD.reason OR NEW.parent_version_json != OLD.parent_version_json
    OR COALESCE(NEW.parent_candidate_id, '') != COALESCE(OLD.parent_candidate_id, '')
    OR NEW.source_run_id != OLD.source_run_id OR NEW.created_at != OLD.created_at
    OR NEW.complexity_json != OLD.complexity_json OR NEW.origin_mix_json != OLD.origin_mix_json
    OR NEW.source_findings_json != OLD.source_findings_json
BEGIN SELECT RAISE(ABORT, 'a candidate is immutable: create a new candidate id'); END;
CREATE TRIGGER IF NOT EXISTS candidates_forward BEFORE UPDATE OF status ON calibration_candidates
WHEN NOT (
    (OLD.status = 'DRAFT' AND NEW.status = 'CALIBRATED')
    OR (OLD.status = 'CALIBRATED' AND NEW.status IN ('VALIDATED', 'REJECTED'))
    OR (OLD.status = 'VALIDATED' AND NEW.status = 'FROZEN_FOR_FINAL_TEST')
    OR (OLD.status = 'FROZEN_FOR_FINAL_TEST' AND NEW.status = 'PROMOTED_MANUALLY')
)
BEGIN SELECT RAISE(ABORT, 'candidate status can only move forward'); END;
""" + "".join(
    f"""
CREATE TRIGGER IF NOT EXISTS {t}_no_update BEFORE UPDATE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
CREATE TRIGGER IF NOT EXISTS {t}_no_delete BEFORE DELETE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
"""
    for t in APPEND_ONLY
)


class CalibrationStoreError(Exception):
    pass


def _dt(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None


class CalibrationStore:
    def __init__(
        self, path: str | Path, clock: Callable[[], float] = time.time, read_only: bool = False
    ):
        self.path = str(path)
        self._clock = clock
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            target = Path(self.path).expanduser()
            if self.read_only:
                if not target.exists():
                    raise CalibrationStoreError(f"no calibration database at {target}")
                self._conn = sqlite3.connect(
                    f"file:{target.resolve()}?mode=ro", uri=True, check_same_thread=False
                )
                return self._conn
            if self.path != ":memory:":
                target.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(target) if self.path != ":memory:" else self.path, check_same_thread=False
            )
            conn.execute("PRAGMA foreign_keys = ON")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                conn.close()
                raise CalibrationStoreError(f"calibration store has schema v{version}")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            self._conn = conn
        return self._conn

    def _write(self, sql: str, params: Any = (), many: bool = False) -> None:
        if self.read_only:
            raise CalibrationStoreError("calibration store opened read-only")
        with self._lock:
            db = self._db()
            with db:
                if many:
                    db.executemany(sql, params)
                else:
                    db.execute(sql, params)

    def _read(self, sql: str, params: Any = ()) -> list[Any]:
        with self._lock:
            return list(self._db().execute(sql, params).fetchall())

    # --- runs and findings --------------------------------------------------------------------

    def add_run(
        self, run_id: str, kind: str, args: dict[str, Any], config: dict[str, Any],
        versions: dict[str, str], dataset: dict[str, Any], results: dict[str, Any], comparisons: int,
    ) -> None:  # fmt: skip
        self._write(
            "INSERT INTO calibration_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                kind,
                self._clock(),
                json.dumps(args, default=str),
                json.dumps(config, default=str),
                json.dumps(versions),
                json.dumps(dataset, default=str),
                json.dumps(results, default=str),
                comparisons,
            ),  # fmt: skip
        )

    def runs(self, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT run_id, kind, created_at, dataset_json, comparisons FROM calibration_runs "
            + ("WHERE kind = ? " if kind else "")
            + "ORDER BY created_at",
            (kind,) if kind else (),
        )
        return [{"run_id": r[0], "kind": r[1], "created_at": _dt(r[2]), "dataset": json.loads(r[3]),
                 "comparisons": r[4]} for r in rows]  # fmt: skip

    def run_results(self, run_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT results_json FROM calibration_runs WHERE run_id = ?", (run_id,))
        return json.loads(rows[0][0]) if rows else None

    def latest_run(self, kind: str) -> str | None:
        rows = self._read(
            "SELECT run_id FROM calibration_runs WHERE kind = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (kind,),
        )
        return rows[0][0] if rows else None

    def add_findings(self, run_id: str, findings: list[dict[str, Any]]) -> None:
        self._write(
            "INSERT INTO calibration_findings (run_id, finding_id, kind, subject, horizon, strength, "
            "statement, body_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    f["finding_id"],
                    f["kind"],
                    f["subject"],
                    f["horizon"],
                    f["strength"],
                    f["statement"],
                    json.dumps(f, default=str),
                )
                for f in findings
            ],  # fmt: skip
            many=True,
        )

    def findings(self, run_id: str | None = None) -> list[dict[str, Any]]:
        run_id = run_id or self.latest_run("analyze")
        if run_id is None:
            return []
        rows = self._read(
            "SELECT body_json FROM calibration_findings WHERE run_id = ? ORDER BY id", (run_id,)
        )
        return [json.loads(r[0]) for r in rows]

    def add_feature_rows(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self._write(
            "INSERT INTO feature_statistics VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    r["feature"],
                    r["bucket"],
                    r["horizon"],
                    r["origin"],
                    r["split"],
                    json.dumps(r["cohort"], default=str),
                )
                for r in rows
            ],  # fmt: skip
            many=True,
        )

    # --- candidates -----------------------------------------------------------------------------

    def add_candidate(self, c: dict[str, Any]) -> bool:
        """Store a new candidate (DRAFT); False when the same id already exists."""
        if self.candidate(c["candidate_id"]) is not None:
            return False
        now = self._clock()
        self._write(
            "INSERT INTO calibration_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DRAFT', ?)",
            (
                c["candidate_id"],
                now,
                json.dumps(c["parent_version"], sort_keys=True),
                c.get("parent_candidate_id"),
                c["source_run_id"],
                c["reason"],
                json.dumps(c["source_findings"]),
                json.dumps(c["changes"], sort_keys=True),
                json.dumps(c["complexity"], sort_keys=True),
                json.dumps(c["origin_mix"], sort_keys=True),
                now,
            ),  # fmt: skip
        )
        self._write(
            "INSERT INTO candidate_parameters VALUES (?, ?, ?, ?)",
            [
                (
                    c["candidate_id"],
                    ch["parameter"],
                    json.dumps(ch.get("from")),
                    json.dumps(ch["to"]),
                )
                for ch in c["changes"]
            ],  # fmt: skip
            many=True,
        )
        self._write(
            "INSERT INTO candidate_lineage VALUES (?, ?, ?, ?)",
            (c["candidate_id"], c.get("parent_candidate_id"), c["source_run_id"], c["reason"]),
        )
        return True

    def set_status(self, candidate_id: str, status: str) -> None:
        self._write(
            "UPDATE calibration_candidates SET status = ?, status_updated_at = ? WHERE candidate_id = ?",
            (status, self._clock(), candidate_id),
        )

    def candidate(self, candidate_id: str) -> dict[str, Any] | None:
        rows = self._read(
            "SELECT candidate_id, created_at, parent_version_json, parent_candidate_id, source_run_id, "
            "reason, source_findings_json, changes_json, complexity_json, origin_mix_json, status, "
            "status_updated_at FROM calibration_candidates WHERE candidate_id = ?",
            (candidate_id,),
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "candidate_id": r[0], "created_at": _dt(r[1]), "parent_version": json.loads(r[2]),
            "parent_candidate_id": r[3], "source_run_id": r[4], "reason": r[5],
            "source_findings": json.loads(r[6]), "changes": json.loads(r[7]),
            "complexity": json.loads(r[8]), "origin_mix": json.loads(r[9]), "status": r[10],
            "status_updated_at": _dt(r[11]),
        }  # fmt: skip

    def candidates(self) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT candidate_id FROM calibration_candidates ORDER BY created_at, candidate_id"
        )
        return [c for (cid,) in rows if (c := self.candidate(cid)) is not None]

    def add_metrics(
        self, candidate_id: str, phase: str, fingerprint: str, metrics: dict[str, Any]
    ) -> None:
        self._write(
            "INSERT INTO candidate_metrics VALUES (?, ?, ?, ?, ?)",
            (candidate_id, phase, self._clock(), fingerprint, json.dumps(metrics, default=str)),
        )

    def metrics(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT phase, created_at, dataset_fingerprint, metrics_json FROM candidate_metrics "
            "WHERE candidate_id = ? ORDER BY rowid", (candidate_id,),
        )  # fmt: skip
        return [
            {"phase": r[0], "created_at": _dt(r[1]), "dataset": r[2], "metrics": json.loads(r[3])}
            for r in rows
        ]

    def add_validation(
        self, candidate_id: str, fingerprint: str, outcome: str, result: dict[str, Any]
    ) -> None:
        self._write(
            "INSERT INTO validation_runs (candidate_id, created_at, dataset_fingerprint, outcome, result_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (candidate_id, self._clock(), fingerprint, outcome, json.dumps(result, default=str)),
        )

    def validations(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT created_at, dataset_fingerprint, outcome, result_json FROM validation_runs "
            "WHERE candidate_id = ? ORDER BY id", (candidate_id,),
        )  # fmt: skip
        return [
            {"created_at": _dt(r[0]), "dataset": r[1], "outcome": r[2], "result": json.loads(r[3])}
            for r in rows
        ]

    def add_final_evaluation(
        self, candidate_id: str, fingerprint: str, result: dict[str, Any]
    ) -> None:
        self._write(
            "INSERT INTO final_evaluations (candidate_id, created_at, dataset_fingerprint, result_json) "
            "VALUES (?, ?, ?, ?)",
            (candidate_id, self._clock(), fingerprint, json.dumps(result, default=str)),
        )

    def final_evaluations(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT created_at, dataset_fingerprint, result_json FROM final_evaluations "
            "WHERE candidate_id = ? ORDER BY id", (candidate_id,),
        )  # fmt: skip
        return [
            {"created_at": _dt(r[0]), "dataset": r[1], "result": json.loads(r[2])} for r in rows
        ]

    # --- HOLDOUT --------------------------------------------------------------------------------

    def log_holdout_access(
        self,
        candidate_id: str | None,
        purpose: str,
        triggered_by: str,
        window: dict[str, Any],
        data_version: str,
    ) -> None:
        self._write(
            "INSERT INTO holdout_access_log (at, candidate_id, purpose, triggered_by, holdout_window_json, "
            "data_version) VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._clock(),
                candidate_id,
                purpose,
                triggered_by,
                json.dumps(window, default=str),
                data_version,
            ),
        )

    def holdout_accesses(self) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT at, candidate_id, purpose, triggered_by, holdout_window_json, data_version "
            "FROM holdout_access_log ORDER BY id"
        )
        return [{"at": _dt(r[0]), "candidate_id": r[1], "purpose": r[2], "triggered_by": r[3],
                 "window": json.loads(r[4]), "data_version": r[5]} for r in rows]  # fmt: skip
