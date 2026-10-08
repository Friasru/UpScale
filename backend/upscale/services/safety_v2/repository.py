"""Safety V2 SQLite storage (its own file; never another component's database).

Phase 1 tables: ``safety_meta``, ``safety_targets``, ``safety_requests``,
``safety_collections``, ``safety_mint_observations``, ``safety_snapshots``.

* Observations and snapshots are append-only (UPDATE / DELETE abort in triggers).
* A collection is RUNNING until it is finished DONE or ABORTED, and a finished collection
  is final. An observation can only be added to a RUNNING collection of the same target,
  never before that collection started.
* A non-MINT observation can't carry authorities, decimals or supply, so a failed or
  non-mint read can never be read back as a revoked authority.
* A database created by another (or an unknown) schema version is refused, and so is a
  database that already holds other components' tables (Safety V2 never writes into, e.g.,
  a Radar or Scout database by mistake).
"""

import hashlib
import json
import sqlite3
import threading
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upscale.services.safety_v2.config import DB_SCHEMA_VERSION, RULES_VERSION, SNAPSHOT_SCHEMA
from upscale.services.safety_v2.models import MINT_OUTCOMES, SafetySchemaError, SafetyStateError
from upscale.services.safety_v2.provider import MintObservation, RequestOutcome
from upscale.services.solana_chain import TOKEN_2022_PROGRAM, TOKEN_PROGRAM

_OUTCOMES = ",".join(repr(o) for o in MINT_OUTCOMES)
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS safety_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS safety_targets (
    canonical_id TEXT PRIMARY KEY,
    chain TEXT NOT NULL CHECK (chain = 'solana'),
    mint TEXT NOT NULL,
    source TEXT NOT NULL,
    added_at REAL NOT NULL,
    CHECK (canonical_id = chain || ':' || mint)
);
CREATE TABLE IF NOT EXISTS safety_requests (
    day TEXT NOT NULL,
    method TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 0,
    rate_limited INTEGER NOT NULL DEFAULT 0,
    timeouts INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, method)
);
CREATE TABLE IF NOT EXISTS safety_collections (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'DONE', 'ABORTED')),
    requests INTEGER NOT NULL DEFAULT 0 CHECK (requests >= 0),
    reasons_json TEXT NOT NULL DEFAULT '[]',
    CHECK ((status = 'RUNNING') = (finished_at IS NULL)),
    CHECK (finished_at IS NULL OR started_at <= finished_at)
);
CREATE TRIGGER IF NOT EXISTS safety_collections_final BEFORE UPDATE ON safety_collections
WHEN OLD.status != 'RUNNING' OR NEW.canonical_id != OLD.canonical_id
    OR NEW.started_at != OLD.started_at
BEGIN SELECT RAISE(ABORT, 'safety_collections: a finished collection is final'); END;
CREATE TRIGGER IF NOT EXISTS safety_collections_no_delete BEFORE DELETE ON safety_collections
BEGIN SELECT RAISE(ABORT, 'safety_collections rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_mint_observations (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    collection_id INTEGER NOT NULL REFERENCES safety_collections(id),
    fetched_at REAL NOT NULL,
    provider TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ({_OUTCOMES})),
    reason TEXT,
    raw_hash TEXT,
    context_slot INTEGER,
    program_owner TEXT,
    token_program TEXT CHECK (token_program IN ('spl_token', 'token_2022')),
    decimals INTEGER CHECK (decimals BETWEEN 0 AND 255),
    supply_raw TEXT,
    mint_authority TEXT,
    freeze_authority TEXT,
    extensions_json TEXT,
    CHECK ((outcome = 'MINT') = (token_program IS NOT NULL)),
    CHECK ((outcome = 'MINT') = (decimals IS NOT NULL)),
    CHECK ((outcome = 'MINT') = (supply_raw IS NOT NULL)),
    CHECK ((outcome = 'MINT') = (extensions_json IS NOT NULL)),
    CHECK (outcome = 'MINT' OR (mint_authority IS NULL AND freeze_authority IS NULL)),
    CHECK ((outcome = 'MINT') = (reason IS NULL)),
    CHECK ((outcome IN ('PROVIDER_FAILED', 'NOT_COLLECTED')) = (raw_hash IS NULL)),
    CHECK (supply_raw IS NULL OR (supply_raw != '' AND supply_raw NOT GLOB '*[^0-9]*')),
    -- A MINT is owned by exactly the token program it names; a missing account has no owner.
    CHECK (outcome != 'MINT' OR (program_owner IS NOT NULL AND (
        (program_owner = '{TOKEN_PROGRAM}' AND token_program = 'spl_token')
        OR (program_owner = '{TOKEN_2022_PROGRAM}' AND token_program = 'token_2022')))),
    CHECK (outcome != 'ACCOUNT_MISSING' OR program_owner IS NULL)
);
CREATE INDEX IF NOT EXISTS safety_mint_by_target
    ON safety_mint_observations (canonical_id, fetched_at);
CREATE TRIGGER IF NOT EXISTS safety_mint_in_collection BEFORE INSERT ON safety_mint_observations
WHEN NOT EXISTS (SELECT 1 FROM safety_collections c WHERE c.id = NEW.collection_id
    AND c.status = 'RUNNING' AND c.canonical_id = NEW.canonical_id
    AND c.started_at <= NEW.fetched_at)
BEGIN SELECT RAISE(ABORT, 'safety_mint_observations: needs a RUNNING collection of the same target started at or before fetched_at'); END;
CREATE TRIGGER IF NOT EXISTS safety_mint_no_update BEFORE UPDATE ON safety_mint_observations
BEGIN SELECT RAISE(ABORT, 'safety_mint_observations rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_mint_no_delete BEFORE DELETE ON safety_mint_observations
BEGIN SELECT RAISE(ABORT, 'safety_mint_observations rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_snapshots (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    as_of REAL NOT NULL,
    schema_version TEXT NOT NULL,
    rules_version TEXT NOT NULL,
    fingerprints_json TEXT NOT NULL,
    coverage TEXT NOT NULL CHECK (coverage IN ('COMPLETE', 'PARTIAL', 'INSUFFICIENT')),
    band TEXT NOT NULL
        CHECK (band IN ('CRITICAL_EVIDENCE', 'ELEVATED_EVIDENCE', 'NO_TRIGGERED_FLAGS')),
    body_zlib BLOB NOT NULL,
    body_hash TEXT NOT NULL,
    UNIQUE (canonical_id, as_of)
);
CREATE TRIGGER IF NOT EXISTS safety_snapshots_no_update BEFORE UPDATE ON safety_snapshots
BEGIN SELECT RAISE(ABORT, 'safety_snapshots rows are immutable'); END;
CREATE TRIGGER IF NOT EXISTS safety_snapshots_no_delete BEFORE DELETE ON safety_snapshots
BEGIN SELECT RAISE(ABORT, 'safety_snapshots rows are immutable'); END;
"""
TABLES = (
    "safety_meta", "safety_targets", "safety_requests", "safety_collections",
    "safety_mint_observations", "safety_snapshots",
)  # fmt: skip
_OUTCOME_COLUMN = {"ok": "ok", "rate_limited": "rate_limited", "timeout": "timeouts",
                   "failed": "failures"}  # fmt: skip


def encode_body(body: Mapping[str, Any]) -> tuple[str, bytes, str]:
    """Canonical JSON (sorted keys, compact, no NaN), its zlib blob and SHA-256."""
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text, zlib.compress(text.encode(), 9), hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class MintRow:
    id: int
    canonical_id: str
    collection_id: int
    fetched_at: float
    provider: str
    outcome: str
    reason: str | None
    raw_hash: str | None
    context_slot: int | None
    program_owner: str | None
    token_program: str | None
    decimals: int | None
    supply_raw: str | None
    mint_authority: str | None
    freeze_authority: str | None
    extensions: list[dict[str, Any]] | None


@dataclass(frozen=True)
class SnapshotRow:
    id: int
    canonical_id: str
    as_of: float
    schema_version: str
    rules_version: str
    fingerprints: dict[str, str]
    coverage: str
    band: str
    body: dict[str, Any]
    body_hash: str
    stored_text: str


_MINT_COLUMNS = (
    "id, canonical_id, collection_id, fetched_at, provider, outcome, reason, raw_hash, "
    "context_slot, program_owner, token_program, decimals, supply_raw, mint_authority, "
    "freeze_authority, extensions_json"
)


def _mint_row(r: tuple[Any, ...]) -> MintRow:
    return MintRow(
        id=r[0], canonical_id=r[1], collection_id=r[2], fetched_at=r[3], provider=r[4],
        outcome=r[5], reason=r[6], raw_hash=r[7], context_slot=r[8], program_owner=r[9],
        token_program=r[10], decimals=r[11], supply_raw=r[12], mint_authority=r[13],
        freeze_authority=r[14], extensions=json.loads(r[15]) if r[15] is not None else None,
    )  # fmt: skip


def _require_compatible(conn: sqlite3.Connection, path: str) -> None:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    names.discard("sqlite_sequence")
    foreign = sorted(n for n in names if not n.startswith("safety_"))
    if foreign:
        raise SafetySchemaError(
            f"{path} already holds other tables ({', '.join(foreign[:5])}): Safety V2 only uses "
            "its own database (UPSCALE_SAFETY_V2_DB)"
        )
    if not names:
        return  # a new database
    found = None
    if "safety_meta" in names:
        row = conn.execute("SELECT value FROM safety_meta WHERE key = 'schema_version'").fetchone()
        found = row[0] if row else None
    if found != str(DB_SCHEMA_VERSION):
        raise SafetySchemaError(
            f"{path} is a Safety V2 database with schema version {found or 'unknown'}; this "
            f"Safety V2 needs version {DB_SCHEMA_VERSION}. It isn't migrated: start a fresh "
            "database and keep the old file for reference."
        )


class SafetyRepository:
    """Synchronous and thread-safe (one connection behind a lock)."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser())
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def db(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.path, check_same_thread=False)
                try:
                    _require_compatible(conn, self.path)
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.executescript(_SCHEMA)
                    with conn:
                        for key, value in (
                            ("schema_version", str(DB_SCHEMA_VERSION)),
                            ("snapshot_schema", SNAPSHOT_SCHEMA),
                        ):
                            conn.execute(
                                "INSERT OR IGNORE INTO safety_meta (key, value) VALUES (?, ?)",
                                (key, value),
                            )
                except BaseException:
                    conn.close()
                    raise
                self._conn = conn
            return self._conn

    # --- meta + request ledger (RequestLedger protocol) -----------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = (
                self.db().execute("SELECT value FROM safety_meta WHERE key = ?", (key,)).fetchone()
            )
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        if key in ("schema_version", "snapshot_schema"):
            raise SafetyStateError(f"{key} is fixed when the database is created")
        with self._lock, self.db() as conn:
            conn.execute(
                "INSERT INTO safety_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def requests_on(self, day: str) -> int:
        with self._lock:
            row = (
                self.db()
                .execute(
                    "SELECT COALESCE(SUM(calls), 0) FROM safety_requests WHERE day = ?", (day,)
                )
                .fetchone()
            )
        return int(row[0])

    def record_request(self, day: str, method: str, outcome: RequestOutcome) -> None:
        col = _OUTCOME_COLUMN[outcome]
        with self._lock, self.db() as conn:
            conn.execute(
                f"INSERT INTO safety_requests (day, method, calls, {col}) VALUES (?, ?, 1, 1) "
                f"ON CONFLICT (day, method) DO UPDATE SET calls = calls + 1, {col} = {col} + 1",
                (day, method),
            )

    def requests_by_method(self, day: str) -> dict[str, int]:
        with self._lock:
            rows = (
                self.db()
                .execute(
                    "SELECT method, calls FROM safety_requests WHERE day = ? ORDER BY method",
                    (day,),
                )
                .fetchall()
            )
        return {m: int(c) for m, c in rows}

    # --- targets -------------------------------------------------------------------------

    def add_target(self, canonical_id: str, mint: str, source: str, added_at: float) -> bool:
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO safety_targets (canonical_id, chain, mint, source, added_at) "
                "VALUES (?, 'solana', ?, ?, ?)",
                (canonical_id, mint, source, added_at),
            )
        return cur.rowcount == 1

    def target_mint(self, canonical_id: str) -> str | None:
        with self._lock:
            row = (
                self.db()
                .execute("SELECT mint FROM safety_targets WHERE canonical_id = ?", (canonical_id,))
                .fetchone()
            )
        return row[0] if row else None

    def targets(self) -> list[tuple[str, str, float]]:
        with self._lock:
            rows = (
                self.db()
                .execute(
                    "SELECT canonical_id, source, added_at FROM safety_targets ORDER BY canonical_id"
                )
                .fetchall()
            )
        return [(c, s, a) for c, s, a in rows]

    # --- collections ----------------------------------------------------------------------

    def start_collection(self, canonical_id: str, started_at: float) -> int:
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO safety_collections (canonical_id, started_at, status) "
                "VALUES (?, ?, 'RUNNING')",
                (canonical_id, started_at),
            )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def finish_collection(
        self, collection_id: int, status: str, finished_at: float, requests: int, reasons: list[str]
    ) -> None:
        if status not in ("DONE", "ABORTED"):
            raise SafetyStateError(f"a collection finishes DONE or ABORTED, not {status}")
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "UPDATE safety_collections SET status = ?, finished_at = ?, requests = ?, "
                "reasons_json = ? WHERE id = ? AND status = 'RUNNING'",
                (status, finished_at, requests, json.dumps(reasons), collection_id),
            )
        if cur.rowcount != 1:
            raise SafetyStateError(f"collection {collection_id} isn't RUNNING")

    def collection(self, collection_id: int) -> tuple[str, float, float | None, int, list[str]]:
        """(status, started_at, finished_at, requests, reasons)."""
        with self._lock:
            row = (
                self.db()
                .execute(
                    "SELECT status, started_at, finished_at, requests, reasons_json "
                    "FROM safety_collections WHERE id = ?",
                    (collection_id,),
                )
                .fetchone()
            )
        if row is None:
            raise SafetyStateError(f"no collection {collection_id}")
        return row[0], row[1], row[2], row[3], json.loads(row[4])

    # --- mint observations ----------------------------------------------------------------

    def record_mint_observation(
        self,
        canonical_id: str,
        collection_id: int,
        fetched_at: float,
        provider: str,
        obs: MintObservation,
    ) -> int:
        extensions = (
            json.dumps(
                [{"extension": n, "state": s} for n, s in obs.extensions],
                sort_keys=True, separators=(",", ":"),
            )
            if obs.extensions is not None
            else None
        )  # fmt: skip
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO safety_mint_observations (canonical_id, collection_id, fetched_at, "
                "provider, outcome, reason, raw_hash, context_slot, program_owner, token_program, "
                "decimals, supply_raw, mint_authority, freeze_authority, extensions_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (canonical_id, collection_id, fetched_at, provider, obs.outcome, obs.reason,
                 obs.raw_hash, obs.context_slot, obs.program_owner, obs.token_program,
                 obs.decimals, obs.supply_raw, obs.mint_authority, obs.freeze_authority,
                 extensions),
            )  # fmt: skip
        assert cur.lastrowid is not None
        return cur.lastrowid

    def latest_mint_observation(self, canonical_id: str, as_of: float) -> MintRow | None:
        """The newest observation of this exact identity fetched at or before `as_of`."""
        with self._lock:
            row = (
                self.db()
                .execute(
                    f"SELECT {_MINT_COLUMNS} FROM safety_mint_observations WHERE canonical_id = ? "
                    "AND fetched_at <= ? ORDER BY fetched_at DESC, id DESC LIMIT 1",
                    (canonical_id, as_of),
                )
                .fetchone()
            )
        return _mint_row(row) if row else None

    def mint_observation(self, observation_id: int) -> MintRow:
        with self._lock:
            row = (
                self.db()
                .execute(
                    f"SELECT {_MINT_COLUMNS} FROM safety_mint_observations WHERE id = ?",
                    (observation_id,),
                )
                .fetchone()
            )
        if row is None:
            raise SafetyStateError(f"no mint observation {observation_id}")
        return _mint_row(row)

    # --- snapshots ------------------------------------------------------------------------

    def save_snapshot(
        self,
        canonical_id: str,
        as_of: float,
        coverage: str,
        band: str,
        fingerprints: Mapping[str, str],
        body: Mapping[str, Any],
    ) -> tuple[int, str]:
        _, blob, digest = encode_body(body)
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO safety_snapshots (canonical_id, as_of, schema_version, rules_version, "
                "fingerprints_json, coverage, band, body_zlib, body_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (canonical_id, as_of, SNAPSHOT_SCHEMA, RULES_VERSION,
                 json.dumps(dict(fingerprints), sort_keys=True), coverage, band, blob, digest),
            )  # fmt: skip
        assert cur.lastrowid is not None
        return cur.lastrowid, digest

    def snapshot(self, snapshot_id: int) -> SnapshotRow:
        with self._lock:
            row = (
                self.db()
                .execute(
                    "SELECT id, canonical_id, as_of, schema_version, rules_version, "
                    "fingerprints_json, coverage, band, body_zlib, body_hash "
                    "FROM safety_snapshots WHERE id = ?",
                    (snapshot_id,),
                )
                .fetchone()
            )
        if row is None:
            raise SafetyStateError(f"no snapshot {snapshot_id}")
        text = zlib.decompress(row[8]).decode()
        return SnapshotRow(row[0], row[1], row[2], row[3], row[4], json.loads(row[5]), row[6],
                           row[7], json.loads(text), row[9], text)  # fmt: skip

    def latest_snapshot_id(self, canonical_id: str) -> int | None:
        with self._lock:
            row = (
                self.db()
                .execute(
                    "SELECT id FROM safety_snapshots WHERE canonical_id = ? "
                    "ORDER BY as_of DESC, id DESC LIMIT 1",
                    (canonical_id,),
                )
                .fetchone()
            )
        return row[0] if row else None

    def counts(self) -> dict[str, int]:
        with self._lock:
            conn = self.db()
            return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in TABLES}
