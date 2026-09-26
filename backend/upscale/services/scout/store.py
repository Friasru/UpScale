"""Durable local storage for Scout: when each token was first seen, and its snapshots.

SQLite (standard library), one file, opened lazily on first use. Only observed values are
stored: there is no interpolation and no backfill, so "15 minutes ago" is answered by the
stored snapshot closest to that moment within a tolerance, or not at all.

Schema (version 1):

* ``scout_tokens``: one row per canonical id (``<chain>:<address>``) with the first time
  and source Scout saw it, and the last time it was seen.
* ``scout_snapshots``: one row per observation of a token's selected pool: time, provider,
  pool, price, market cap / FDV (each only if reported), liquidity, the reported rolling
  windows as JSON, and nullable holder / social columns reserved for later sources.
  ``(canonical_id, provider, pool_address, observed_at)`` is unique, so storing the same
  (cached) observation twice is a no-op; ``(canonical_id, observed_at)`` is indexed for
  time-window lookups.
"""

import asyncio
import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from upscale.services.scout.models import (
    ScoutCandidate,
    ScoutMarketMetrics,
    ScoutSnapshot,
    ScoutWindow,
)

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scout_tokens (
    canonical_id TEXT PRIMARY KEY,
    chain TEXT NOT NULL,
    address TEXT NOT NULL,
    symbol TEXT,
    name TEXT,
    first_seen_at REAL NOT NULL,
    first_seen_provider TEXT NOT NULL,
    first_seen_kind TEXT NOT NULL,
    last_seen_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scout_snapshots (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES scout_tokens(canonical_id),
    observed_at REAL NOT NULL,
    provider TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    dex TEXT NOT NULL,
    price_usd REAL,
    market_cap_usd REAL,
    fdv_usd REAL,
    liquidity_usd REAL,
    windows_json TEXT NOT NULL,
    holder_count INTEGER,
    social_json TEXT,
    UNIQUE (canonical_id, provider, pool_address, observed_at)
);
CREATE INDEX IF NOT EXISTS scout_snapshots_by_time
    ON scout_snapshots (canonical_id, observed_at);
CREATE INDEX IF NOT EXISTS scout_tokens_by_last_seen ON scout_tokens (last_seen_at);
"""


class ScoutStoreError(Exception):
    pass


class ScoutSnapshotStore:
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

    async def record_seen(self, candidate: ScoutCandidate) -> datetime:
        """Remember the token (first sighting is kept) and return its first-seen time."""
        return await asyncio.to_thread(self._record_seen, candidate)

    async def save_snapshot(self, snapshot: ScoutSnapshot, min_interval_seconds: float) -> bool:
        """Store a snapshot unless one from the same provider is less than
        `min_interval_seconds` older (or identical). True if stored."""
        return await asyncio.to_thread(self._save_snapshot, snapshot, min_interval_seconds)

    async def nearest_snapshot(
        self,
        canonical_id: str,
        target: datetime,
        tolerance: timedelta,
        before: datetime,
        provider: str | None = None,
    ) -> ScoutSnapshot | None:
        """The stored snapshot closest to `target` (within `tolerance`), strictly earlier
        than `before`, or None: missing history is never filled in."""
        return await asyncio.to_thread(
            self._nearest, canonical_id, target, tolerance, before, provider
        )

    async def history(
        self, canonical_id: str, since: datetime | None = None, limit: int = 500
    ) -> list[ScoutSnapshot]:
        """Snapshots oldest first."""
        return await asyncio.to_thread(self._history, canonical_id, since, limit)

    async def first_seen(self, canonical_id: str) -> datetime | None:
        return await asyncio.to_thread(self._first_seen, canonical_id)

    async def tracked_tokens(self, seen_since: datetime) -> list[tuple[str, str]]:
        """(chain, address) of tokens seen since `seen_since`, most recent first."""
        return await asyncio.to_thread(self._tracked, seen_since)

    async def prune(self, older_than: datetime) -> int:
        """Delete snapshots older than `older_than`; returns how many were deleted."""
        return await asyncio.to_thread(self._prune, older_than)

    # --- SQL ------------------------------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                conn.close()
                raise ScoutStoreError(
                    f"Scout store {self.path} has schema v{version}; this UpScale knows "
                    f"v{SCHEMA_VERSION}"
                )
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            self._conn = conn
        return self._conn

    def _record_seen(self, c: ScoutCandidate) -> datetime:
        seen = c.observed_at.timestamp()
        source = c.sources[0] if c.sources else None
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    """
                    INSERT INTO scout_tokens (canonical_id, chain, address, symbol, name,
                        first_seen_at, first_seen_provider, first_seen_kind, last_seen_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (canonical_id) DO UPDATE SET
                        symbol = COALESCE(excluded.symbol, symbol),
                        name = COALESCE(excluded.name, name),
                        first_seen_at = MIN(first_seen_at, excluded.first_seen_at),
                        last_seen_at = MAX(last_seen_at, excluded.last_seen_at)
                    """,
                    (
                        c.canonical_id,
                        c.chain,
                        c.address,
                        c.symbol,
                        c.name,
                        seen,
                        source.provider if source else "unknown",
                        source.kind if source else "lookup",
                        seen,
                    ),
                )
            row = db.execute(
                "SELECT first_seen_at FROM scout_tokens WHERE canonical_id = ?",
                (c.canonical_id,),
            ).fetchone()
        return _dt(row[0])

    def _save_snapshot(self, s: ScoutSnapshot, min_interval: float) -> bool:
        t = s.observed_at.timestamp()
        with self._lock:
            db = self._db()
            recent = db.execute(
                """
                SELECT 1 FROM scout_snapshots
                WHERE canonical_id = ? AND provider = ? AND observed_at <= ?
                    AND observed_at > ?
                LIMIT 1
                """,
                (s.canonical_id, s.provider, t, t - min_interval),
            ).fetchone()
            duplicate = db.execute(
                """
                SELECT 1 FROM scout_snapshots WHERE canonical_id = ? AND provider = ?
                    AND pool_address = ? AND observed_at = ?
                """,
                (s.canonical_id, s.provider, s.pool_address, t),
            ).fetchone()
            if recent or duplicate:
                return False
            m = s.metrics
            with db:
                db.execute(
                    """
                    INSERT INTO scout_snapshots (canonical_id, observed_at, provider,
                        pool_address, dex, price_usd, market_cap_usd, fdv_usd, liquidity_usd,
                        windows_json, holder_count, social_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.canonical_id,
                        t,
                        s.provider,
                        s.pool_address,
                        s.dex,
                        m.price_usd,
                        m.market_cap_usd,
                        m.fdv_usd,
                        m.liquidity_usd,
                        json.dumps([w.model_dump() for w in m.windows]),
                        s.holder_count,
                        json.dumps(s.social) if s.social is not None else None,
                    ),
                )
        return True

    def _nearest(
        self,
        canonical_id: str,
        target: datetime,
        tolerance: timedelta,
        before: datetime,
        provider: str | None,
    ) -> ScoutSnapshot | None:
        t, tol = target.timestamp(), tolerance.total_seconds()
        sql = f"""
            SELECT {_COLUMNS} FROM scout_snapshots
            WHERE canonical_id = ? AND observed_at < ? AND observed_at BETWEEN ? AND ?
            {"AND provider = ?" if provider else ""}
            ORDER BY ABS(observed_at - ?), observed_at DESC LIMIT 1
        """
        params: list[Any] = [canonical_id, before.timestamp(), t - tol, t + tol]
        if provider:
            params.append(provider)
        params.append(t)
        with self._lock:
            row = self._db().execute(sql, params).fetchone()
        return _snapshot(row) if row else None

    def _history(
        self, canonical_id: str, since: datetime | None, limit: int
    ) -> list[ScoutSnapshot]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    f"""
                    SELECT {_COLUMNS} FROM scout_snapshots
                    WHERE canonical_id = ? AND observed_at >= ?
                    ORDER BY observed_at DESC LIMIT ?
                    """,
                    (canonical_id, since.timestamp() if since else 0.0, limit),
                )
                .fetchall()
            )
        return [_snapshot(r) for r in reversed(rows)]

    def _first_seen(self, canonical_id: str) -> datetime | None:
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT first_seen_at FROM scout_tokens WHERE canonical_id = ?",
                    (canonical_id,),
                )
                .fetchone()
            )
        return _dt(row[0]) if row else None

    def _tracked(self, seen_since: datetime) -> list[tuple[str, str]]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    """
                    SELECT chain, address FROM scout_tokens WHERE last_seen_at >= ?
                    ORDER BY last_seen_at DESC
                    """,
                    (seen_since.timestamp(),),
                )
                .fetchall()
            )
        return [(r[0], r[1]) for r in rows]

    def _prune(self, older_than: datetime) -> int:
        with self._lock:
            db = self._db()
            with db:
                cur = db.execute(
                    "DELETE FROM scout_snapshots WHERE observed_at < ?", (older_than.timestamp(),)
                )
        return cur.rowcount


_COLUMNS = (
    "canonical_id, observed_at, provider, pool_address, dex, price_usd, market_cap_usd, "
    "fdv_usd, liquidity_usd, windows_json, holder_count, social_json"
)


def _dt(value: float) -> datetime:
    return datetime.fromtimestamp(value, UTC)


def _snapshot(row: Sequence[Any]) -> ScoutSnapshot:
    windows = [ScoutWindow.model_validate(w) for w in json.loads(row[9])]
    return ScoutSnapshot(
        canonical_id=row[0],
        observed_at=_dt(row[1]),
        provider=row[2],
        pool_address=row[3],
        dex=row[4],
        metrics=ScoutMarketMetrics(
            price_usd=row[5],
            market_cap_usd=row[6],
            fdv_usd=row[7],
            liquidity_usd=row[8],
            windows=windows,
        ),
        holder_count=row[10],
        social=json.loads(row[11]) if row[11] is not None else None,
    )


def snapshot_of(candidate: ScoutCandidate) -> ScoutSnapshot:
    return ScoutSnapshot(
        canonical_id=candidate.canonical_id,
        observed_at=candidate.observed_at,
        provider=candidate.market_provider,
        pool_address=candidate.pool.address,
        dex=candidate.pool.dex,
        metrics=candidate.metrics,
    )
