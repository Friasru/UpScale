"""The Point-in-Time Evidence Archive database (SQLite, append-only).

A dedicated file (``UPSCALE_EVIDENCE_DB``; by default ``evidence.sqlite3`` next to the Scout
database, so on Railway it lives on the same persistent ``/data`` volume). It never touches
the Scout, outcome or replay tables.

Schema (version 1): ``evidence_records``, one row per point-in-time evidence snapshot:

* identity: kind, canonical asset id (``<chain>:<address>``: EVM lowercased, Solana
  case-sensitive), chain, address, pool, DEX, provider, the production component;
* time: ``observed_at`` (authoritative: when the evidence was true), ``provider_at`` (a
  provider timestamp, when one exists) and ``archived_at`` (when it was written; never
  evidence of anything);
* ``availability`` (AVAILABLE / NOT_AVAILABLE / NOT_SUPPORTED / PROVIDER_FAILED /
  RATE_LIMITED / NOT_COLLECTED) and the reason;
* the payload (redacted JSON, zlib-compressed), its SHA-256, a fingerprint (the payload
  without volatile timestamps, for de-duplication) and the production version fingerprints.

Rows can't be updated or deleted (triggers): a correction is a new row. ``observed_at`` can
never be later than ``archived_at`` (plus clock skew).
"""

import hashlib
import json
import sqlite3
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from upscale import log_safety

SCHEMA_VERSION = 1
Kind = Literal["market", "dex_market", "safety", "social", "scout", "decision"]
KINDS: tuple[Kind, ...] = ("market", "dex_market", "safety", "social", "scout", "decision")
Availability = Literal[
    "AVAILABLE",
    "NOT_AVAILABLE",
    "NOT_SUPPORTED",
    "PROVIDER_FAILED",
    "RATE_LIMITED",
    "NOT_COLLECTED",
]
AVAILABILITY: tuple[Availability, ...] = (
    "AVAILABLE", "NOT_AVAILABLE", "NOT_SUPPORTED", "PROVIDER_FAILED", "RATE_LIMITED", "NOT_COLLECTED",
)  # fmt: skip
MAX_CLOCK_SKEW_SECONDS = 300
# Timestamps that change without the evidence changing: left out of the fingerprint.
VOLATILE_KEYS = frozenset(
    {"fetched_at", "observed_at", "computed_at", "as_of", "compared_at", "elapsed_minutes",
     "snapshot_age_minutes", "tracked_hours", "age_minutes", "timings"}
)  # fmt: skip

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS evidence_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence_records (
    id INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ({",".join(repr(k) for k in KINDS)})),
    asset_id TEXT NOT NULL,
    chain TEXT,
    address TEXT,
    pool_address TEXT,
    dex TEXT,
    provider TEXT,
    component TEXT NOT NULL,
    observed_at REAL NOT NULL,
    provider_at REAL,
    archived_at REAL NOT NULL,
    availability TEXT NOT NULL CHECK (availability IN ({",".join(repr(a) for a in AVAILABILITY)})),
    reason TEXT,
    fingerprint TEXT NOT NULL,
    payload BLOB NOT NULL,
    payload_hash TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    versions_json TEXT NOT NULL,
    links_json TEXT NOT NULL DEFAULT '{{}}',
    CHECK (observed_at <= archived_at + {MAX_CLOCK_SKEW_SECONDS})
);
CREATE INDEX IF NOT EXISTS evidence_by_asset ON evidence_records (kind, asset_id, observed_at);
CREATE INDEX IF NOT EXISTS evidence_by_time ON evidence_records (observed_at);
CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence_records
BEGIN SELECT RAISE(ABORT, 'evidence records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS evidence_no_delete BEFORE DELETE ON evidence_records
BEGIN SELECT RAISE(ABORT, 'evidence records are append-only'); END;
"""


class EvidenceStoreError(Exception):
    pass


class FutureEvidenceError(RuntimeError):
    """Evidence observed after the requested time came back from a point-in-time query."""


@dataclass
class PendingRecord:
    """A record as production handed it over (not yet written)."""

    kind: Kind
    asset_id: str
    observed_at: datetime
    payload: dict[str, Any]
    chain: str | None = None
    address: str | None = None
    pool_address: str | None = None
    dex: str | None = None
    provider: str | None = None
    component: str = "unknown"
    provider_at: datetime | None = None
    availability: Availability = "AVAILABLE"
    reason: str | None = None
    links: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    # Set by `prepare` (redacted payload text and its identities).
    text: str | None = None
    payload_hash: str | None = None
    fingerprint: str | None = None
    record_id: str | None = None


def prepare(r: PendingRecord) -> PendingRecord:
    """Redact the payload and compute the record's identities (once)."""
    if r.text is None:
        r.text = log_safety.redact(json.dumps(r.payload, sort_keys=True, default=str))
        r.payload_hash = hashlib.sha256(r.text.encode()).hexdigest()
        r.fingerprint = fingerprint(r.kind, json.loads(r.text), r.availability)
        observed = r.observed_at.timestamp()
        r.record_id = hashlib.sha256(
            f"{r.kind}|{r.asset_id}|{r.pool_address}|{r.provider}|{observed:.6f}|{r.payload_hash}".encode()
        ).hexdigest()[:32]
    return r


def link_of(r: PendingRecord) -> dict[str, Any]:
    """How a decision record refers to one piece of evidence it used."""
    prepare(r)
    return {
        "kind": r.kind,
        "asset_id": r.asset_id,
        "pool_address": r.pool_address,
        "provider": r.provider,
        "observed_at": r.observed_at.isoformat(),
        "availability": r.availability,
        "fingerprint": r.fingerprint,
        "record_id": r.record_id,
    }


@dataclass(frozen=True)
class EvidenceRecord:
    id: int
    record_id: str
    kind: Kind
    asset_id: str
    chain: str | None
    address: str | None
    pool_address: str | None
    dex: str | None
    provider: str | None
    component: str
    observed_at: datetime
    provider_at: datetime | None
    archived_at: datetime
    availability: Availability
    reason: str | None
    fingerprint: str
    payload: dict[str, Any]
    payload_hash: str
    versions: dict[str, str]
    links: dict[str, Any]


def _utc(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, UTC) if ts is not None else None


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def fingerprint(kind: str, payload: dict[str, Any], availability: str) -> str:
    body = json.dumps([kind, availability, _strip(payload)], sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:32]


_COLUMNS = (
    "id, record_id, kind, asset_id, chain, address, pool_address, dex, provider, component, "
    "observed_at, provider_at, archived_at, availability, reason, fingerprint, payload, "
    "payload_hash, versions_json, links_json"
)


class EvidenceStore:
    """Thread-safe (one connection behind a lock): the writer thread appends, status and
    replay reads share it."""

    def __init__(
        self, path: str | Path, clock: Callable[[], float] = time.time, read_only: bool = False
    ):
        self.path = str(path)
        self._clock = clock
        self.read_only = read_only  # readers (Replay Lab, status CLI) never write
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None and self.read_only:
            target = Path(self.path).expanduser().resolve()
            if not target.exists():
                raise EvidenceStoreError(f"no evidence archive at {target}")
            self._conn = sqlite3.connect(
                f"file:{target}?mode=ro", uri=True, check_same_thread=False
            )
            return self._conn
        if self._conn is None:
            if self.path != ":memory:":
                Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(Path(self.path).expanduser()) if self.path != ":memory:" else self.path,
                check_same_thread=False,
            )
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                conn.close()
                raise EvidenceStoreError(
                    f"evidence archive {self.path} has schema v{version}; this UpScale knows "
                    f"v{SCHEMA_VERSION}"
                )
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            self._conn = conn
        return self._conn

    # --- writing ------------------------------------------------------------------------------

    def append(self, r: PendingRecord, dedupe_seconds: float = 0.0) -> bool:
        """Store the record unless it duplicates the latest one for the same key (same
        fingerprint within `dedupe_seconds`, or the identical observation). True if stored."""
        if self.read_only:
            raise EvidenceStoreError("this evidence store is opened read-only")
        prepare(r)
        text, payload_hash, fp, record_id = r.text, r.payload_hash, r.fingerprint, r.record_id
        assert text is not None and payload_hash and fp and record_id
        observed = r.observed_at.timestamp()
        with self._lock:
            db = self._db()
            last = db.execute(
                """
                SELECT fingerprint, observed_at FROM evidence_records
                WHERE kind = ? AND asset_id = ? AND COALESCE(pool_address, '') = ?
                    AND COALESCE(provider, '') = ?
                ORDER BY observed_at DESC, id DESC LIMIT 1
                """,
                (r.kind, r.asset_id, r.pool_address or "", r.provider or ""),
            ).fetchone()
            if last is not None and last[0] == fp and abs(observed - last[1]) <= dedupe_seconds:
                return False
            now = self._clock()
            with db:
                cur = db.execute(
                    """
                    INSERT OR IGNORE INTO evidence_records (record_id, kind, asset_id, chain,
                        address, pool_address, dex, provider, component, observed_at,
                        provider_at, archived_at, availability, reason, fingerprint, payload,
                        payload_hash, schema_version, versions_json, links_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id, r.kind, r.asset_id, r.chain, r.address, r.pool_address, r.dex,
                        r.provider, r.component, observed,
                        r.provider_at.timestamp() if r.provider_at else None, now, r.availability,
                        r.reason, fp, zlib.compress(text.encode()), payload_hash, SCHEMA_VERSION,
                        json.dumps(r.versions, sort_keys=True), json.dumps(r.links, default=str),
                    ),
                )  # fmt: skip
        return cur.rowcount == 1

    # --- point-in-time reads --------------------------------------------------------------------

    def latest(
        self,
        kind: Kind,
        asset_id: str,
        until: datetime,
        since: datetime | None = None,
        pool: str | None = None,
    ) -> EvidenceRecord | None:
        """The latest record with observed_at <= `until` (and >= `since`), never later."""
        params: list[Any] = [kind, asset_id, until.timestamp(), since.timestamp() if since else 0.0]
        if pool:
            params.append(pool)
        with self._lock:
            row = (
                self._db()
                .execute(
                    f"""
                    SELECT {_COLUMNS} FROM evidence_records
                    WHERE kind = ? AND asset_id = ? AND observed_at <= ? AND observed_at >= ?
                    {"AND pool_address = ?" if pool else ""}
                    ORDER BY observed_at DESC, id DESC LIMIT 1
                    """,
                    params,
                )
                .fetchone()
            )
        record = _record(row) if row else None
        if record is not None and record.observed_at > until:
            raise FutureEvidenceError(f"{kind} evidence observed after {until.isoformat()}")
        return record

    def resolve(self, link: dict[str, Any]) -> EvidenceRecord | None:
        """The archived record a decision linked: the exact row, or, when that repeat was
        de-duplicated, the identical earlier state (same fingerprint, observed no later)."""
        observed = datetime.fromisoformat(link["observed_at"]).timestamp()
        with self._lock:
            db = self._db()
            row = db.execute(
                f"SELECT {_COLUMNS} FROM evidence_records WHERE record_id = ?",
                (link.get("record_id"),),
            ).fetchone()
            if row is None:
                row = db.execute(
                    f"""
                    SELECT {_COLUMNS} FROM evidence_records
                    WHERE kind = ? AND asset_id = ? AND fingerprint = ? AND observed_at <= ?
                    ORDER BY observed_at DESC, id DESC LIMIT 1
                    """,
                    (link["kind"], link["asset_id"], link.get("fingerprint"), observed),
                ).fetchone()
        record = _record(row) if row else None
        if record is not None and record.observed_at.timestamp() > observed + 1e-6:
            raise FutureEvidenceError("a linked record resolved to later evidence")
        return record

    def decisions(
        self, kind: Kind, since: datetime | None = None, until: datetime | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """(asset_id, observed_at, links) of Scout / Analyze decision records (no payloads)."""
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT asset_id, observed_at, links_json FROM evidence_records "
                    "WHERE kind = ? AND observed_at >= ? AND observed_at <= ? ORDER BY observed_at",
                    (
                        kind,
                        since.timestamp() if since else 0.0,
                        until.timestamp() if until else 1e12,
                    ),
                )
                .fetchall()
            )
        return [(r[0], r[1], json.loads(r[2])) for r in rows]

    def latest_any(self, kind: Kind, until: datetime) -> EvidenceRecord | None:
        """The latest record of `kind` for any asset with observed_at <= `until`."""
        with self._lock:
            row = (
                self._db()
                .execute(
                    f"SELECT {_COLUMNS} FROM evidence_records WHERE kind = ? AND observed_at <= ? "
                    "ORDER BY observed_at DESC, id DESC LIMIT 1",
                    (kind, until.timestamp()),
                )
                .fetchone()
            )
        record = _record(row) if row else None
        if record is not None and record.observed_at > until:
            raise FutureEvidenceError(f"{kind} evidence observed after {until.isoformat()}")
        return record

    def records(
        self,
        kind: Kind | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 10_000,
    ) -> list[EvidenceRecord]:
        where: list[str] = ["1 = 1"]
        params: list[Any] = []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if asset_id:
            where.append("asset_id = ?")
            params.append(asset_id)
        if since:
            where.append("observed_at >= ?")
            params.append(since.timestamp())
        if until:
            where.append("observed_at <= ?")
            params.append(until.timestamp())
        with self._lock:
            rows = (
                self._db()
                .execute(
                    f"SELECT {_COLUMNS} FROM evidence_records WHERE {' AND '.join(where)} "
                    "ORDER BY observed_at, id LIMIT ?",
                    [*params, limit],
                )
                .fetchall()
            )
        return [_record(r) for r in rows]

    def after(
        self,
        kinds: Sequence[Kind],
        cursor: tuple[float, int],
        until: datetime,
        limit: int = 2000,
    ) -> list[EvidenceRecord]:
        """Records of `kinds` strictly after `cursor` = (observed_at, id) in that order and
        observed at or before `until`, oldest first (a resumable, ordered scan)."""
        at, rid = cursor
        marks = ",".join("?" for _ in kinds)
        with self._lock:
            rows = (
                self._db()
                .execute(
                    f"SELECT {_COLUMNS} FROM evidence_records WHERE kind IN ({marks}) "
                    "AND (observed_at > ? OR (observed_at = ? AND id > ?)) AND observed_at <= ? "
                    "ORDER BY observed_at, id LIMIT ?",
                    [*kinds, at, at, rid, until.timestamp(), limit],
                )
                .fetchall()
            )
        out = [_record(r) for r in rows]
        if any(r.observed_at > until for r in out):
            raise FutureEvidenceError(f"evidence observed after {until.isoformat()}")
        return out

    def archived_late(
        self, kinds: Sequence[Kind], since: float, observed_by: float, archived_after: float
    ) -> int:
        """Records of `kinds` observed in [since, observed_by] but archived after
        `archived_after` (they arrived after an ordered scan had passed them)."""
        marks = ",".join("?" for _ in kinds)
        with self._lock:
            row = (
                self._db()
                .execute(
                    f"SELECT COUNT(*) FROM evidence_records WHERE kind IN ({marks}) "
                    "AND observed_at >= ? AND observed_at <= ? AND archived_at > ?",
                    [*kinds, since, observed_by, archived_after],
                )
                .fetchone()
            )
        return int(row[0])

    def index(
        self, since: datetime | None = None
    ) -> list[tuple[str, str, float, str, str | None, str, dict[str, Any]]]:
        """(kind, asset_id, observed_at, availability, provider, reason, links) for coverage
        metrics (no payloads)."""
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT kind, asset_id, observed_at, availability, provider, "
                    "COALESCE(reason, ''), links_json FROM evidence_records "
                    "WHERE observed_at >= ? ORDER BY observed_at",
                    (since.timestamp() if since else 0.0,),
                )
                .fetchall()
            )
        return [(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6])) for r in rows]

    def totals(self) -> dict[str, Any]:
        with self._lock:
            db = self._db()
            row = db.execute(
                "SELECT COUNT(*), COUNT(DISTINCT asset_id), MIN(observed_at), MAX(observed_at), "
                "MAX(archived_at) FROM evidence_records"
            ).fetchone()
            kinds = dict(
                db.execute("SELECT kind, COUNT(*) FROM evidence_records GROUP BY kind").fetchall()
            )
        return {
            "total": row[0],
            "assets": row[1],
            "oldest": _utc(row[2]),
            "newest": _utc(row[3]),
            "last_write": _utc(row[4]),
            "by_kind": {k: int(v) for k, v in kinds.items()},
        }


def _record(r: Sequence[Any]) -> EvidenceRecord:
    text = zlib.decompress(r[16]).decode()
    if hashlib.sha256(text.encode()).hexdigest() != r[17]:
        raise EvidenceStoreError(f"evidence record {r[1]} fails its hash check")
    observed = _utc(r[10])
    archived = _utc(r[12])
    assert observed is not None and archived is not None
    return EvidenceRecord(
        id=r[0], record_id=r[1], kind=r[2], asset_id=r[3], chain=r[4], address=r[5],
        pool_address=r[6], dex=r[7], provider=r[8], component=r[9], observed_at=observed,
        provider_at=_utc(r[11]), archived_at=archived, availability=r[13], reason=r[14],
        fingerprint=r[15], payload=json.loads(text), payload_hash=r[17],
        versions=json.loads(r[18]), links=json.loads(r[19]),
    )  # fmt: skip
