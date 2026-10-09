"""Read-only access for the dry-run: the Evidence Archive (Scout records), the Safety V2
database (snapshots, request ledger, cooldown) and the Opportunity database. Every store is
opened ``mode=ro`` (Safety additionally ``query_only``); a missing file is reported, never
created. Safety V2 code is never imported: its tables are read as stored.
"""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from upscale.services.evidence_archive.store import EvidenceRecord, EvidenceStore, Kind
from upscale.services.opportunity_model.loaders import ArchiveReader, SafetyRow
from upscale.services.opportunity_model.repository import (
    OpportunityRepository,
    OpportunityStorageError,
    StoredDecision,
)
from upscale.services.opportunity_orchestrator.config import SAFETY_COOLDOWN_META_KEY
from upscale.services.opportunity_orchestrator.models import DbStatus


@dataclass(frozen=True)
class ArchiveScan:
    status: DbStatus
    records: tuple[EvidenceRecord, ...]
    truncated: bool
    detail: str = ""


def scan_archive(path: str, now: datetime, lookback_s: float, limit: int) -> ArchiveScan:
    """Scout records observed in ``[now - lookback_s, now]``, oldest first."""
    p = Path(path).expanduser()
    if not p.is_file():
        return ArchiveScan("MISSING", (), False, f"no evidence archive at {p}")
    store = EvidenceStore(p, read_only=True)
    try:
        rows = store.records(kind="scout", since=now - timedelta(seconds=lookback_s), until=now,
                             limit=limit)  # fmt: skip
    except Exception as exc:  # an unreadable / foreign file
        return ArchiveScan("INCOMPATIBLE", (), False, f"{type(exc).__name__}: {exc}")
    finally:
        store.close()
    return ArchiveScan("READABLE", tuple(rows), len(rows) >= limit)


@dataclass(frozen=True)
class SafetySnapshotRow:
    id: int
    as_of: datetime
    rules_version: str
    body_zlib: bytes
    body_hash: str


class SafetyDb:
    """The Safety V2 database, read-only, read as stored (no Safety code)."""

    def __init__(self, path: str | None):
        self.path = Path(path).expanduser() if path else None
        self.status: DbStatus = "MISSING"
        self.detail = "no Safety V2 database configured" if path is None else ""
        self._conn: sqlite3.Connection | None = None
        if self.path is None:
            return
        if not self.path.is_file():
            self.detail = f"no Safety V2 database at {self.path}"
            return
        try:
            conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = ON")
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        except sqlite3.DatabaseError as exc:
            self.status, self.detail = "INCOMPATIBLE", f"unreadable: {exc}"
            return
        if not {"safety_snapshots", "safety_requests", "safety_meta"} <= tables:
            conn.close()
            self.status, self.detail = "INCOMPATIBLE", "not a Safety V2 database"
            return
        self._conn, self.status = conn, "READABLE"

    def latest_snapshot(self, canonical_id: str, until: datetime) -> SafetySnapshotRow | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT id, as_of, rules_version, body_zlib, body_hash FROM safety_snapshots "
            "WHERE canonical_id = ? AND as_of <= ? ORDER BY as_of DESC, id DESC LIMIT 1",
            (canonical_id, until.timestamp()),
        ).fetchone()
        if row is None:
            return None
        return SafetySnapshotRow(
            row[0], datetime.fromtimestamp(row[1], UTC), row[2], row[3], row[4]
        )

    def requests_on(self, day: str) -> int | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT COALESCE(SUM(calls), 0) FROM safety_requests WHERE day = ?", (day,)
        ).fetchone()
        return int(row[0])

    def cooldown_until(self, now: datetime) -> datetime | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT value FROM safety_meta WHERE key = ?", (SAFETY_COOLDOWN_META_KEY,)
        ).fetchone()
        try:
            until = datetime.fromtimestamp(float(row[0]), UTC) if row else None
        except (TypeError, ValueError):
            return None
        return until if until is not None and until > now else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class OpportunityDb:
    """The Opportunity database through its own read-only repository."""

    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.repo: OpportunityRepository | None = None
        self.status: DbStatus = "MISSING"
        self.detail = ""
        if not self.path.is_file():
            self.detail = f"no Opportunity database at {self.path}"
            return
        try:
            self.repo = OpportunityRepository(self.path, read_only=True)
            self.status = "READABLE"
        except OpportunityStorageError as exc:
            self.status, self.detail = "INCOMPATIBLE", str(exc)

    def latest(self, canonical_id: str) -> StoredDecision | None:
        return self.repo.latest_for(canonical_id) if self.repo is not None else None

    def close(self) -> None:
        if self.repo is not None:
            self.repo.close()
            self.repo = None


def opportunity_decision_by_key(
    path: str, canonical_id: str, decision_at: datetime, rules_version: str
) -> int | None:
    """The Opportunity decision stored under the exact key (canonical_id, LIVE_FORWARD,
    decision_at, rules_version), read-only; None when there is none."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    conn = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        row = conn.execute(
            "SELECT id FROM opportunity_decisions WHERE canonical_id = ? AND origin = "
            "'LIVE_FORWARD' AND decision_at = ? AND rules_version = ?",
            (canonical_id, decision_at.timestamp(), rules_version),
        ).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else None


def original_scout_records(
    path: str, keys: dict[int, tuple[str, str, datetime]]
) -> dict[int, EvidenceRecord | None] | None:
    """The exact archived Scout record of each key ``(canonical_id, scout_record_id,
    scout_run_time)``, by the same correlation `PinnedArchive` uses (that token, that run
    time, that record id; never a newer record), or None for a key whose record is gone.
    None overall when the archive itself can't be read (missing file, foreign / locked
    database): nothing can be concluded about any record then."""
    p = Path(path).expanduser()
    if not keys:
        return {}
    if not p.is_file():
        return None
    store = EvidenceStore(p, read_only=True)
    out: dict[int, EvidenceRecord | None] = {}
    try:
        for key, (canonical_id, record_id, run_time) in keys.items():
            rows = store.records(kind="scout", asset_id=canonical_id, since=run_time,
                                 until=run_time)  # fmt: skip
            out[key] = next((r for r in rows if r.record_id == record_id), None)
    except Exception:  # an unreadable / foreign / locked file
        return None
    finally:
        store.close()
    return out


class PinnedArchive:
    """An O1 `ArchiveSource` (read-only) whose ``scout`` evidence is exactly one archived
    record: the job's own Scout record, never a newer one. Other kinds (social, Analyze)
    come from the archive as usual. O1 still applies every causality / identity check to
    whatever is returned."""

    def __init__(self, path: str, canonical_id: str, scout_record_id: str, run_time: datetime):
        self.reader = ArchiveReader(path)
        self.path = Path(path).expanduser()
        self.canonical_id, self.record_id, self.run_time = canonical_id, scout_record_id, run_time

    def latest(self, kind: Kind, asset_id: str, until: datetime) -> EvidenceRecord | None:
        if kind != "scout":
            return self.reader.latest(kind, asset_id, until)
        if asset_id != self.canonical_id or not self.path.is_file():
            return None
        store = EvidenceStore(self.path, read_only=True)
        try:
            rows = store.records(kind="scout", asset_id=asset_id, since=self.run_time,
                                 until=self.run_time)  # fmt: skip
        finally:
            store.close()
        return next((r for r in rows if r.record_id == self.record_id), None)

    def close(self) -> None:
        self.reader.close()


class PinnedSafety:
    """An O1 `SafetySource` (read-only) that returns exactly the job's Safety snapshot, by
    id, never "the latest at or before" a time. O1 still checks its ``as_of``, identity,
    rules version and body hash."""

    def __init__(self, path: str | None, snapshot_id: int | None):
        self.path = Path(path).expanduser() if path else None
        self.snapshot_id = snapshot_id

    def latest_snapshot(self, canonical_id: str, until: datetime) -> SafetyRow | None:
        if self.path is None or self.snapshot_id is None or not self.path.is_file():
            return None
        conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA query_only = ON")
            row = conn.execute(
                "SELECT id, canonical_id, as_of, schema_version, rules_version, coverage, band, "
                "body_zlib, body_hash FROM safety_snapshots WHERE id = ? AND canonical_id = ?",
                (self.snapshot_id, canonical_id),
            ).fetchone()
        except sqlite3.DatabaseError:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        return SafetyRow(id=row[0], canonical_id=row[1], as_of=datetime.fromtimestamp(row[2], UTC),
                         schema_version=row[3], rules_version=row[4], coverage=row[5],
                         band=row[6], body_zlib=row[7], body_hash=row[8])  # fmt: skip
