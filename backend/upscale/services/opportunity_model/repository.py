"""Opportunity Model decision storage (SQLite, its own file, schema version 1).

Every decision is stored with the exact canonical `OpportunityInput` it was decided from
and the exact canonical `OpportunityDecision` body, both zlib-compressed, their SHA-256
hashes over the *uncompressed* canonical JSON, and the input / decision code fingerprints.
A stored decision can therefore be re-verified (`verify_decision`) from this database alone:
no Safety, Evidence Archive, Radar or provider is ever read for that.

* ``opportunity_meta``: component and schema version.
* ``opportunity_runs``: one row per decide attempt (RUNNING -> DONE / ABORTED, once).
* ``opportunity_decisions``: the decision, its bodies, hashes and fingerprints. Unique per
  (canonical_id, decision_at, origin, rules_version).
* ``opportunity_source_refs`` / ``opportunity_reasons``: an audit / query index *derived by
  the repository* from the stored canonical bodies (never supplied by a caller).

Decisions, refs and reasons are append-only (UPDATE / DELETE abort in triggers); rows can
only be added inside a RUNNING run. A database of another component, another (or unknown)
schema version, or with Opportunity tables but missing / corrupt metadata is refused before
anything is written.
"""

import hashlib
import json
import sqlite3
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from upscale.services.opportunity_model.config import (
    DECISION_SCHEMA,
    INPUT_SCHEMA,
    OPPORTUNITY_COMPONENT,
    OPPORTUNITY_DB_SCHEMA_VERSION,
)
from upscale.services.opportunity_model.decision import decide
from upscale.services.opportunity_model.decision_models import OpportunityDecision, Reason
from upscale.services.opportunity_model.fingerprints import (
    decision_fingerprints,
    differing,
    input_fingerprints,
)
from upscale.services.opportunity_model.models import (
    OpportunityError,
    OpportunityInput,
    OpportunityOrigin,
    SafetySourceRef,
    SourceRef,
)

VerifyStatus = Literal["REPRODUCED", "FINGERPRINT_MISMATCH", "DIVERGED", "CORRUPT"]
ReasonKind = Literal["POSITIVE", "RISK", "MISSING", "VETO", "BLOCKER", "INELIGIBLE", "UPGRADE"]
SOURCE_NAMES = ("scout", "technical", "technical_analyze", "social", "news", "safety")


class OpportunityStorageError(OpportunityError):
    """The database refused the operation, or a stored row can't be trusted."""


class OpportunityConflictError(OpportunityStorageError):
    """A decision with the same identity already exists with different content."""


def _list(values: tuple[str, ...]) -> str:
    return ",".join(f"'{v}'" for v in values)


_RUN_STATUSES = ("RUNNING", "DONE", "ABORTED")
_ORIGINS = ("LIVE_FORWARD", "HISTORICAL_REPLAY")
_KINDS = ("POSITIVE", "RISK", "MISSING", "VETO", "BLOCKER", "INELIGIBLE", "UPGRADE")


def _immutable(table: str) -> str:
    return f"""
CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END;
CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END;"""


def _child_of_running_decision(table: str) -> str:
    return f"""
CREATE TRIGGER IF NOT EXISTS {table}_only_while_running BEFORE INSERT ON {table}
WHEN NOT EXISTS (SELECT 1 FROM opportunity_decisions d JOIN opportunity_runs r
                 ON r.id = d.run_id WHERE d.id = NEW.decision_id AND r.status = 'RUNNING')
BEGIN SELECT RAISE(ABORT, '{table} rows are only added with their decision'); END;"""


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS opportunity_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS opportunity_runs (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ({_list(_ORIGINS)})),
    requested_as_of REAL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL CHECK (status IN ({_list(_RUN_STATUSES)})),
    reason TEXT,
    CHECK ((status = 'RUNNING') = (finished_at IS NULL)),
    CHECK (status != 'ABORTED' OR reason IS NOT NULL),
    CHECK (origin != 'HISTORICAL_REPLAY' OR requested_as_of IS NOT NULL)
);
CREATE TRIGGER IF NOT EXISTS opportunity_runs_finish_once BEFORE UPDATE ON opportunity_runs
WHEN OLD.status != 'RUNNING' OR NEW.status = 'RUNNING' OR NEW.id != OLD.id
     OR NEW.canonical_id != OLD.canonical_id OR NEW.origin != OLD.origin
     OR NEW.requested_as_of IS NOT OLD.requested_as_of OR NEW.started_at != OLD.started_at
BEGIN SELECT RAISE(ABORT, 'a run only finishes once, RUNNING -> DONE / ABORTED'); END;
CREATE TRIGGER IF NOT EXISTS opportunity_runs_no_delete BEFORE DELETE ON opportunity_runs
BEGIN SELECT RAISE(ABORT, 'opportunity_runs rows are immutable'); END;
CREATE TABLE IF NOT EXISTS opportunity_decisions (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL UNIQUE REFERENCES opportunity_runs(id),
    canonical_id TEXT NOT NULL,
    chain TEXT NOT NULL,
    pool_address TEXT,
    decision_at REAL NOT NULL,
    decided_at REAL NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ({_list(_ORIGINS)})),
    decision TEXT NOT NULL CHECK (decision IN ('SKIP', 'WATCH', 'ENTER')),
    quality TEXT NOT NULL CHECK (quality IN ('COMPLETE', 'PARTIAL', 'INSUFFICIENT')),
    setup_band TEXT NOT NULL CHECK (setup_band IN ('NONE', 'WEAK', 'MODERATE', 'STRONG')),
    technical_band TEXT NOT NULL
        CHECK (technical_band IN ('CONFIRMING', 'NEUTRAL', 'CONTRADICTING', 'UNAVAILABLE')),
    social_band TEXT NOT NULL
        CHECK (social_band IN ('SUPPORTING', 'NEUTRAL', 'CONTRADICTING', 'UNAVAILABLE')),
    risk_tier TEXT NOT NULL CHECK (risk_tier IN ('CLEAN', 'ELEVATED_1', 'ELEVATED_2_PLUS')),
    rules_version TEXT NOT NULL,
    input_schema TEXT NOT NULL,
    decision_schema TEXT NOT NULL,
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64),
    decision_hash TEXT NOT NULL CHECK (length(decision_hash) = 64),
    input_body_zlib BLOB NOT NULL,
    decision_body_zlib BLOB NOT NULL,
    input_fingerprints_json TEXT NOT NULL,
    decision_fingerprints_json TEXT NOT NULL,
    veto_count INTEGER NOT NULL CHECK (veto_count >= 0),
    blocker_count INTEGER NOT NULL CHECK (blocker_count >= 0),
    ineligible_count INTEGER NOT NULL CHECK (ineligible_count >= 0),
    UNIQUE (canonical_id, decision_at, origin, rules_version),
    CHECK (decision != 'ENTER' OR (veto_count = 0 AND blocker_count = 0
           AND ineligible_count = 0 AND quality != 'INSUFFICIENT')),
    CHECK (decision != 'SKIP' OR veto_count > 0 OR ineligible_count > 0 OR setup_band = 'NONE'),
    CHECK (decided_at >= decision_at OR origin = 'HISTORICAL_REPLAY')
);
CREATE INDEX IF NOT EXISTS opportunity_decisions_by_asset
    ON opportunity_decisions (canonical_id, decision_at, id);
CREATE TRIGGER IF NOT EXISTS opportunity_decisions_in_running_run
BEFORE INSERT ON opportunity_decisions
WHEN NOT EXISTS (SELECT 1 FROM opportunity_runs WHERE id = NEW.run_id AND status = 'RUNNING'
                 AND canonical_id = NEW.canonical_id AND origin = NEW.origin)
BEGIN SELECT RAISE(ABORT, 'a decision is only added to a RUNNING run of the same target'); END;
{_immutable("opportunity_decisions")}
CREATE TABLE IF NOT EXISTS opportunity_source_refs (
    id INTEGER PRIMARY KEY,
    decision_id INTEGER NOT NULL REFERENCES opportunity_decisions(id),
    source_name TEXT NOT NULL CHECK (source_name IN ({_list(SOURCE_NAMES)})),
    status TEXT NOT NULL,
    store TEXT,
    record_id TEXT,
    fingerprint TEXT,
    observed_at TEXT,
    age_seconds REAL,
    reason TEXT,
    safety_snapshot_id INTEGER,
    safety_as_of TEXT,
    safety_rules_version TEXT,
    safety_body_hash TEXT,
    UNIQUE (decision_id, source_name),
    CHECK (source_name = 'safety' OR (safety_snapshot_id IS NULL AND safety_as_of IS NULL
           AND safety_rules_version IS NULL AND safety_body_hash IS NULL))
);
{_child_of_running_decision("opportunity_source_refs")}
{_immutable("opportunity_source_refs")}
CREATE TABLE IF NOT EXISTS opportunity_reasons (
    id INTEGER PRIMARY KEY,
    decision_id INTEGER NOT NULL REFERENCES opportunity_decisions(id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    kind TEXT NOT NULL CHECK (kind IN ({_list(_KINDS)})),
    code TEXT NOT NULL,
    group_name TEXT,
    source TEXT,
    source_record_id TEXT,
    status TEXT,
    gate TEXT,
    message TEXT NOT NULL,
    evidence_paths_json TEXT NOT NULL,
    UNIQUE (decision_id, ordinal)
);
{_child_of_running_decision("opportunity_reasons")}
{_immutable("opportunity_reasons")}
"""
TABLES = frozenset(
    {"opportunity_meta", "opportunity_runs", "opportunity_decisions", "opportunity_source_refs",
     "opportunity_reasons"}
)  # fmt: skip


# --- canonical bodies -------------------------------------------------------------------------


def _ts(t: datetime) -> float:
    return t.astimezone(UTC).timestamp()


def _iso(t: datetime | None) -> str | None:
    return t.astimezone(UTC).isoformat() if t is not None else None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _decode(blob: bytes, digest: str, what: str) -> str:
    try:
        text = zlib.decompress(blob).decode()
    except (zlib.error, UnicodeDecodeError) as exc:
        raise OpportunityStorageError(f"{what} body can't be decompressed: {exc}") from exc
    if _sha(text) != digest:
        raise OpportunityStorageError(f"{what} body fails its hash check")
    return text


def load_input(text: str) -> OpportunityInput:
    try:
        inp = OpportunityInput.model_validate_json(text)
    except (ValidationError, OpportunityError) as exc:
        raise OpportunityStorageError(
            f"stored input isn't a valid OpportunityInput: {exc}"
        ) from exc
    if inp.canonical_json() != text:
        raise OpportunityStorageError("stored input isn't in canonical form")
    return inp


def load_decision(text: str) -> OpportunityDecision:
    try:
        dec = OpportunityDecision.model_validate_json(text)
    except ValidationError as exc:
        raise OpportunityStorageError(f"stored decision isn't valid: {exc}") from exc
    if dec.canonical_json() != text:
        raise OpportunityStorageError("stored decision isn't in canonical form")
    return dec


def source_rows(inp: OpportunityInput) -> list[dict[str, Any]]:
    """The normalized source references, derived from the canonical input only."""
    s = inp.sources
    refs: list[tuple[str, SourceRef]] = [
        ("scout", s.scout.ref), ("technical", s.technical.ref),
        ("technical_analyze", s.technical.analyze_ref), ("social", s.social.ref),
        ("news", s.news.ref), ("safety", s.safety.ref),
    ]  # fmt: skip
    out = []
    for name, ref in refs:
        row = {
            "source_name": name, "status": ref.status, "store": ref.store,
            "record_id": ref.record_id, "fingerprint": ref.fingerprint,
            "observed_at": _iso(ref.observed_at), "age_seconds": ref.age_seconds,
            "reason": ref.reason, "safety_snapshot_id": None, "safety_as_of": None,
            "safety_rules_version": None, "safety_body_hash": None,
        }  # fmt: skip
        if isinstance(ref, SafetySourceRef):
            row |= {"safety_snapshot_id": ref.snapshot_id, "safety_as_of": _iso(ref.as_of),
                    "safety_rules_version": ref.rules_version, "safety_body_hash": ref.body_hash}  # fmt: skip
        out.append(row)
    return out


def reason_rows(dec: OpportunityDecision) -> list[dict[str, Any]]:
    """The normalized reasons, derived from the canonical decision only, in body order."""
    groups: list[tuple[ReasonKind, tuple[Reason, ...]]] = [
        ("VETO", dec.vetoes), ("INELIGIBLE", dec.ineligible), ("BLOCKER", dec.blockers),
        ("POSITIVE", dec.positive_reasons), ("RISK", dec.risks), ("MISSING", dec.missing_evidence),
    ]  # fmt: skip
    out: list[dict[str, Any]] = []
    for kind, reasons in groups:
        for r in reasons:
            out.append({
                "kind": kind, "code": r.code, "group_name": r.group, "source": r.source,
                "source_record_id": r.source_record_id, "status": r.status, "gate": None,
                "message": r.message, "evidence_paths_json": json.dumps(list(r.evidence_paths)),
            })  # fmt: skip
    for u in dec.upgrade_path:
        out.append({"kind": "UPGRADE", "code": u.code, "group_name": None, "source": "engine",
                    "source_record_id": None, "status": None, "gate": u.gate,
                    "message": u.message, "evidence_paths_json": "[]"})  # fmt: skip
    for i, row in enumerate(out):
        row["ordinal"] = i
    return out


def pool_address(inp: OpportunityInput) -> str | None:
    """The exact pool Safety V2 selected (only when its market read succeeded)."""
    market = inp.sources.safety.market
    if not isinstance(market, dict) or market.get("status") != "AVAILABLE":
        return None
    pool = market.get("primary_pool")
    value = pool.get("value") if isinstance(pool, dict) else None
    address = value.get("address") if isinstance(value, dict) else None
    return address if isinstance(address, str) else None


# --- results -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredDecision:
    id: int
    run_id: int
    canonical_id: str
    decision_at: datetime
    decided_at: datetime
    origin: str
    pool_address: str | None
    input_hash: str
    decision_hash: str
    input_fingerprints: dict[str, str]
    decision_fingerprints: dict[str, str]
    input: OpportunityInput
    decision: OpportunityDecision


@dataclass(frozen=True)
class StoreResult:
    decision_id: int
    created: bool  # False: an identical decision already existed (idempotent)


@dataclass(frozen=True)
class VerifyResult:
    decision_id: int
    status: VerifyStatus
    detail: str
    differing_fingerprints: tuple[str, ...] = ()
    stored_hash: str | None = None
    recomputed_hash: str | None = None


# --- the repository ---------------------------------------------------------------------------


class OpportunityRepository:
    """`read_only=True` opens ``mode=ro`` + ``query_only`` and never creates anything."""

    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = Path(path).expanduser()
        self.read_only = read_only
        self._conn = self._open()

    # opening ------------------------------------------------------------------------------

    def _inspect(self) -> None:
        """Refuse a file that isn't an Opportunity database of this schema, before writing."""
        try:
            ro = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise OpportunityStorageError(f"can't open {self.path}: {exc}") from exc
        try:
            tables = {r[0] for r in ro.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'")}  # fmt: skip
            if not tables:
                return
            foreign = sorted(tables - TABLES)
            if foreign:
                raise OpportunityStorageError(
                    f"{self.path} holds other components' tables ({', '.join(foreign[:5])}); "
                    "Opportunity never writes into another database"
                )
            if "opportunity_meta" not in tables:
                raise OpportunityStorageError(f"{self.path} has Opportunity tables but no metadata")
            meta = dict(ro.execute("SELECT key, value FROM opportunity_meta").fetchall())
        except sqlite3.DatabaseError as exc:
            raise OpportunityStorageError(f"{self.path} isn't a readable database: {exc}") from exc
        finally:
            ro.close()
        if meta.get("component") != OPPORTUNITY_COMPONENT:
            raise OpportunityStorageError(f"{self.path} metadata is missing or not Opportunity's")
        version = meta.get("schema_version")
        if version != str(OPPORTUNITY_DB_SCHEMA_VERSION):
            raise OpportunityStorageError(
                f"{self.path} has Opportunity schema version {version!r}; this UpScale knows "
                f"{OPPORTUNITY_DB_SCHEMA_VERSION} (never migrated)"
            )

    def _open(self) -> sqlite3.Connection:
        exists = self.path.exists()
        if exists:
            self._inspect()
        if self.read_only:
            if not exists:
                raise OpportunityStorageError(f"no Opportunity database at {self.path}")
            conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = ON")
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'opportunity_meta'"
            ).fetchone():
                conn.close()
                raise OpportunityStorageError(f"{self.path} is an empty database")
            return conn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        # One transaction for tables + metadata (executescript commits anything pending
        # first): a crash can't leave Opportunity tables without their metadata.
        try:
            conn.executescript(
                "BEGIN IMMEDIATE;" + _SCHEMA
                + "INSERT OR IGNORE INTO opportunity_meta (key, value) VALUES "
                f"('component', '{OPPORTUNITY_COMPONENT}'), "
                f"('schema_version', '{OPPORTUNITY_DB_SCHEMA_VERSION}');COMMIT;"
            )  # fmt: skip
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()
            raise
        return conn

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise OpportunityStorageError("this repository is read-only")
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # runs ----------------------------------------------------------------------------------

    def start_run(
        self,
        canonical_id: str,
        origin: OpportunityOrigin,
        started_at: datetime,
        requested_as_of: datetime | None,
    ) -> int:
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO opportunity_runs (canonical_id, origin, requested_as_of, "
                "started_at, status) VALUES (?, ?, ?, ?, 'RUNNING')",
                (canonical_id, origin, _ts(requested_as_of) if requested_as_of else None,
                 _ts(started_at)),
            )  # fmt: skip
            assert cur.lastrowid is not None
            return cur.lastrowid

    def abort_run(self, run_id: int, reason: str, at: datetime) -> None:
        with self._tx() as db:
            db.execute(
                "UPDATE opportunity_runs SET status = 'ABORTED', finished_at = ?, reason = ? "
                "WHERE id = ? AND status = 'RUNNING'",
                (_ts(at), reason[:2000], run_id),
            )

    def run(self, run_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, canonical_id, origin, requested_as_of, started_at, finished_at, status, "
            "reason FROM opportunity_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        keys = ("id", "canonical_id", "origin", "requested_as_of", "started_at", "finished_at",
                "status", "reason")  # fmt: skip
        return dict(zip(keys, row, strict=True)) if row else None

    # storing -------------------------------------------------------------------------------

    def store(
        self,
        run_id: int,
        inp: OpportunityInput,
        decision: OpportunityDecision,
        decided_at: datetime,
        finished_at: datetime,
    ) -> StoreResult:
        """Persist one decision atomically and finish its run DONE. The decision must be
        exactly what the pure engine makes of `inp`; the index rows are derived here from
        the canonical bodies. An identical existing decision is returned (idempotent); the
        same key with other content raises `OpportunityConflictError` (never overwritten).
        """
        input_text = inp.canonical_json()
        decision_text = decision.canonical_json()
        input_hash, decision_hash = _sha(input_text), _sha(decision_text)
        if input_hash != inp.input_hash() or decision.input_hash != input_hash:
            raise OpportunityStorageError("the decision wasn't made from this input")
        if decide(inp).canonical_json() != decision_text:
            raise OpportunityStorageError(
                "the decision isn't what the engine makes of this input (refused)"
            )
        stored_input = load_input(input_text)  # the exact bodies that will be read back
        stored_decision = load_decision(decision_text)
        key = (inp.canonical_id, _ts(inp.decision_at), inp.origin, decision.rules_version)
        with self._tx() as db:
            existing = db.execute(
                "SELECT id, input_hash, decision_hash FROM opportunity_decisions WHERE "
                "canonical_id = ? AND decision_at = ? AND origin = ? AND rules_version = ?",
                key,
            ).fetchone()
            if existing is not None:
                if (existing[1], existing[2]) != (input_hash, decision_hash):
                    raise OpportunityConflictError(
                        f"decision {existing[0]} already exists for {inp.canonical_id} at "
                        f"{inp.decision_at.isoformat()} ({inp.origin}, rules "
                        f"{decision.rules_version}) with a different "
                        + ("input" if existing[1] != input_hash else "decision")
                        + "; it is never overwritten"
                    )
                self._finish(db, run_id, finished_at, f"identical to decision {existing[0]}")
                return StoreResult(existing[0], created=False)
            cur = db.execute(
                """INSERT INTO opportunity_decisions (run_id, canonical_id, chain, pool_address,
                    decision_at, decided_at, origin, decision, quality, setup_band,
                    technical_band, social_band, risk_tier, rules_version, input_schema,
                    decision_schema, input_hash, decision_hash, input_body_zlib,
                    decision_body_zlib, input_fingerprints_json, decision_fingerprints_json,
                    veto_count, blocker_count, ineligible_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, stored_input.canonical_id, stored_input.chain,
                    pool_address(stored_input), _ts(stored_input.decision_at), _ts(decided_at),
                    stored_input.origin, stored_decision.decision, stored_decision.quality,
                    stored_decision.setup_band, stored_decision.technical_band,
                    stored_decision.social_band, stored_decision.risk_tier,
                    stored_decision.rules_version, stored_input.schema_version,
                    stored_decision.schema_version, input_hash, decision_hash,
                    zlib.compress(input_text.encode(), 9), zlib.compress(decision_text.encode(), 9),
                    json.dumps(input_fingerprints(), sort_keys=True),
                    json.dumps(decision_fingerprints(), sort_keys=True),
                    len(stored_decision.vetoes), len(stored_decision.blockers),
                    len(stored_decision.ineligible),
                ),
            )  # fmt: skip
            decision_id = cur.lastrowid
            assert decision_id is not None
            _insert_source_refs(db, decision_id, source_rows(stored_input))
            _insert_reasons(db, decision_id, reason_rows(stored_decision))
            self._finish(db, run_id, finished_at, None)
        return StoreResult(decision_id, created=True)

    @staticmethod
    def _finish(db: sqlite3.Connection, run_id: int, at: datetime, reason: str | None) -> None:
        cur = db.execute(
            "UPDATE opportunity_runs SET status = 'DONE', finished_at = ?, reason = ? "
            "WHERE id = ? AND status = 'RUNNING'",
            (_ts(at), reason, run_id),
        )
        if cur.rowcount != 1:
            raise OpportunityStorageError(f"run {run_id} isn't RUNNING")

    # reading --------------------------------------------------------------------------------

    _COLUMNS = (
        "id, run_id, canonical_id, decision_at, decided_at, origin, pool_address, input_hash, "
        "decision_hash, input_fingerprints_json, decision_fingerprints_json, input_body_zlib, "
        "decision_body_zlib, decision, quality, rules_version, input_schema, decision_schema"
    )

    def get(self, decision_id: int) -> StoredDecision:
        row = self._conn.execute(
            f"SELECT {self._COLUMNS} FROM opportunity_decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            raise OpportunityStorageError(f"no decision {decision_id}")
        return self._decode_row(row)

    def latest_for(self, canonical_id: str) -> StoredDecision | None:
        row = self._conn.execute(
            f"SELECT {self._COLUMNS} FROM opportunity_decisions WHERE canonical_id = ? "
            "ORDER BY decision_at DESC, id DESC LIMIT 1",
            (canonical_id,),
        ).fetchone()
        return self._decode_row(row) if row else None

    def _decode_row(self, row: tuple[Any, ...]) -> StoredDecision:
        (did, run_id, cid, decision_at, decided_at, origin, pool, ih, dh, ifp, dfp, ib, db_,
         decision, quality, rules, ischema, dschema) = row  # fmt: skip
        inp = load_input(_decode(ib, ih, f"decision {did} input"))
        dec = load_decision(_decode(db_, dh, f"decision {did}"))
        checks = {
            "canonical_id": (cid, inp.canonical_id, dec.canonical_id),
            "decision_at": (decision_at, _ts(inp.decision_at), _ts(dec.decision_at)),
            "origin": (origin, inp.origin, dec.origin),
            "input_hash": (ih, dec.input_hash, ih),
            "decision": (decision, dec.decision, decision),
            "quality": (quality, dec.quality, quality),
            "rules_version": (rules, dec.rules_version, rules),
            "schemas": ((ischema, dschema), (inp.schema_version, dec.schema_version),
                        (INPUT_SCHEMA, DECISION_SCHEMA)),
        }  # fmt: skip
        for name, values in checks.items():
            if len({json.dumps(v) for v in values}) != 1:
                raise OpportunityStorageError(f"decision {did}: {name} disagrees with its bodies")
        try:
            fingerprints = (json.loads(ifp), json.loads(dfp))
        except json.JSONDecodeError as exc:
            raise OpportunityStorageError(f"decision {did}: fingerprints unreadable") from exc
        return StoredDecision(
            id=did, run_id=run_id, canonical_id=cid,
            decision_at=datetime.fromtimestamp(decision_at, UTC),
            decided_at=datetime.fromtimestamp(decided_at, UTC), origin=origin,
            pool_address=pool, input_hash=ih, decision_hash=dh,
            input_fingerprints=fingerprints[0], decision_fingerprints=fingerprints[1],
            input=inp, decision=dec,
        )  # fmt: skip

    def source_refs(self, decision_id: int) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT source_name, status, store, record_id, fingerprint, observed_at, "
            "age_seconds, reason, safety_snapshot_id, safety_as_of, safety_rules_version, "
            "safety_body_hash FROM opportunity_source_refs WHERE decision_id = ? ORDER BY id",
            (decision_id,),
        )
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    def reasons(self, decision_id: int) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT ordinal, kind, code, group_name, source, source_record_id, status, gate, "
            "message, evidence_paths_json FROM opportunity_reasons WHERE decision_id = ? "
            "ORDER BY ordinal",
            (decision_id,),
        )
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    def status(self) -> dict[str, Any]:
        db = self._conn
        meta = dict(db.execute("SELECT key, value FROM opportunity_meta").fetchall())

        def counts(column: str, table: str = "opportunity_decisions") -> dict[str, int]:
            return dict(db.execute(f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} "
                                   f"ORDER BY {column}").fetchall())  # fmt: skip

        latest = db.execute("SELECT MAX(decision_at) FROM opportunity_decisions").fetchone()[0]
        return {
            "schema_version": meta.get("schema_version"),
            "decisions": db.execute("SELECT COUNT(*) FROM opportunity_decisions").fetchone()[0],
            "runs": db.execute("SELECT COUNT(*) FROM opportunity_runs").fetchone()[0],
            "runs_by_status": counts("status", "opportunity_runs"),
            "by_decision": counts("decision"),
            "by_quality": counts("quality"),
            "by_origin": counts("origin"),
            "latest_decision_at": _iso(datetime.fromtimestamp(latest, UTC)) if latest else None,
        }

    # verification ------------------------------------------------------------------------------

    def verify_decision(
        self,
        decision_id: int,
        current: Callable[[], dict[str, str]] = decision_fingerprints,
        engine: Callable[[OpportunityInput], OpportunityDecision] = decide,
    ) -> VerifyResult:
        """Re-derive a stored decision from its stored canonical input with the pure engine.
        Reads only this database (no upstream store), writes nothing."""
        try:
            stored = self.get(decision_id)
        except OpportunityStorageError as exc:
            if str(exc).startswith("no decision"):
                raise
            return VerifyResult(decision_id, "CORRUPT", str(exc))
        diff = differing(stored.decision_fingerprints, current())
        if diff:
            return VerifyResult(
                decision_id, "FINGERPRINT_MISMATCH",
                "decision code / versions changed since this decision was stored: "
                + ", ".join(diff),
                tuple(diff), stored.decision_hash,
            )  # fmt: skip
        try:
            recomputed = engine(stored.input)
        except OpportunityError as exc:
            return VerifyResult(decision_id, "DIVERGED", f"the engine now refuses the input: {exc}",
                                stored_hash=stored.decision_hash)  # fmt: skip
        digest = recomputed.decision_hash()
        if digest == stored.decision_hash:
            return VerifyResult(decision_id, "REPRODUCED", "identical decision body",
                                stored_hash=stored.decision_hash, recomputed_hash=digest)  # fmt: skip
        return VerifyResult(decision_id, "DIVERGED",
                            "same fingerprints, different decision body",
                            stored_hash=stored.decision_hash, recomputed_hash=digest)  # fmt: skip


def _insert_source_refs(
    db: sqlite3.Connection, decision_id: int, rows: list[dict[str, Any]]
) -> None:
    for r in rows:
        db.execute(
            "INSERT INTO opportunity_source_refs (decision_id, source_name, status, store, "
            "record_id, fingerprint, observed_at, age_seconds, reason, safety_snapshot_id, "
            "safety_as_of, safety_rules_version, safety_body_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (decision_id, r["source_name"], r["status"], r["store"], r["record_id"],
             r["fingerprint"], r["observed_at"], r["age_seconds"], r["reason"],
             r["safety_snapshot_id"], r["safety_as_of"], r["safety_rules_version"],
             r["safety_body_hash"]),
        )  # fmt: skip


def _insert_reasons(db: sqlite3.Connection, decision_id: int, rows: list[dict[str, Any]]) -> None:
    for r in rows:
        db.execute(
            "INSERT INTO opportunity_reasons (decision_id, ordinal, kind, code, group_name, "
            "source, source_record_id, status, gate, message, evidence_paths_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (decision_id, r["ordinal"], r["kind"], r["code"], r["group_name"], r["source"],
             r["source_record_id"], r["status"], r["gate"], r["message"],
             r["evidence_paths_json"]),
        )  # fmt: skip
