"""Read-only access to the stores Opportunity V1 reads: the Evidence Archive and the Safety V2
database. Selection only (the latest row at or before ``until``, ties broken by row id);
interpretation is in `normalize`. Nothing here writes, and Radar is never opened.

* Evidence Archive: `EvidenceStore` in read-only mode (``mode=ro``).
* Safety V2: its own SQLite file opened ``mode=ro`` with ``PRAGMA query_only``; Safety
  V2's repository is never used (opening it may create tables).
"""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from upscale.services.evidence_archive.store import (
    EvidenceRecord,
    EvidenceStore,
    EvidenceStoreError,
    FutureEvidenceError,
    Kind,
)
from upscale.services.opportunity_model.models import OpportunityCausalityError


class StoreUnavailable(Exception):
    """A store couldn't be read (missing file, not the expected database)."""


@dataclass(frozen=True)
class SafetyRow:
    """One ``safety_snapshots`` row, as stored."""

    id: int
    canonical_id: str
    as_of: datetime
    schema_version: str
    rules_version: str
    coverage: str
    band: str
    body_zlib: bytes
    body_hash: str


class ArchiveSource(Protocol):
    def latest(self, kind: Kind, asset_id: str, until: datetime) -> EvidenceRecord | None: ...


class SafetySource(Protocol):
    def latest_snapshot(self, canonical_id: str, until: datetime) -> SafetyRow | None: ...


class ArchiveReader:
    """The Evidence Archive, read-only."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._store: EvidenceStore | None = None

    def latest(self, kind: Kind, asset_id: str, until: datetime) -> EvidenceRecord | None:
        if self._store is None:
            if not self.path.is_file():
                raise StoreUnavailable(f"no evidence archive at {self.path}")
            self._store = EvidenceStore(self.path, read_only=True)
        try:
            record = self._store.latest(kind, asset_id, until)
        except FutureEvidenceError as exc:
            raise OpportunityCausalityError(str(exc)) from exc
        except (EvidenceStoreError, sqlite3.Error) as exc:
            raise StoreUnavailable(f"evidence archive unreadable: {exc}") from exc
        if record is not None and record.observed_at > until:
            raise OpportunityCausalityError(
                f"{kind} record {record.record_id} observed after {until.isoformat()}"
            )
        return record

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


_SNAPSHOT_COLUMNS = (
    "id, canonical_id, as_of, schema_version, rules_version, coverage, band, body_zlib, body_hash"
)


class SafetyReader:
    """The Safety V2 database, read-only (``mode=ro`` + ``query_only``)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path is not None else None
        self._conn: sqlite3.Connection | None = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path is None:
                raise StoreUnavailable("no Safety V2 database configured (UPSCALE_SAFETY_V2_DB)")
            if not self.path.is_file():
                raise StoreUnavailable(f"no Safety V2 database at {self.path}")
            conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = ON")
            found = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'safety_snapshots'"
            ).fetchone()
            if found is None:
                conn.close()
                raise StoreUnavailable(f"{self.path} is not a Safety V2 database")
            self._conn = conn
        return self._conn

    def latest_snapshot(self, canonical_id: str, until: datetime) -> SafetyRow | None:
        try:
            row = (
                self._db()
                .execute(
                    f"SELECT {_SNAPSHOT_COLUMNS} FROM safety_snapshots "
                    "WHERE canonical_id = ? AND as_of <= ? ORDER BY as_of DESC, id DESC LIMIT 1",
                    (canonical_id, until.timestamp()),
                )
                .fetchone()
            )
        except sqlite3.Error as exc:
            raise StoreUnavailable(f"Safety V2 database unreadable: {exc}") from exc
        if row is None:
            return None
        snap = SafetyRow(
            id=row[0], canonical_id=row[1], as_of=datetime.fromtimestamp(row[2], UTC),
            schema_version=row[3], rules_version=row[4], coverage=row[5], band=row[6],
            body_zlib=row[7], body_hash=row[8],
        )  # fmt: skip
        if snap.as_of > until:
            raise OpportunityCausalityError(
                f"Safety snapshot {snap.id} as_of {snap.as_of.isoformat()} is after "
                f"{until.isoformat()}"
            )
        return snap

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
