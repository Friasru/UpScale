"""Durable local storage for Scout: when each token was first seen, and its snapshots.

SQLite (standard library), one file, opened lazily on first use. Only observed values are
stored: there is no interpolation and no backfill, so "15 minutes ago" is answered by the
stored snapshot closest to that moment within a tolerance, or not at all.

Schema (version 2):

* ``scout_tokens``: one row per canonical id (``<chain>:<address>``) with the first time
  and source Scout saw it, the last time it was observed (any way), and the last time a
  discovery listing surfaced it (exact-address refreshes don't count: they keep a token's
  history going, never its place among tracked candidates).
* ``scout_snapshots``: one row per observation of a token's selected pool: time, provider,
  pool, price, market cap / FDV (each only if reported), liquidity, the reported rolling
  windows as JSON, and nullable holder / social columns reserved for later sources.
  ``(canonical_id, provider, pool_address, observed_at)`` is unique, so storing the same
  (cached) observation twice is a no-op; ``(canonical_id, observed_at)`` is indexed for
  time-window lookups.
* ``scout_growth_stages``: Growth Scout's stage, rank and score per token and ranking run,
  so a one-run reversal can be told from a sustained change and tracked-token refresh can
  favor recent leaders (append-only).
* ``scout_feed_schedule``: per provider and discovery feed (``<kind>:<chain>``), its
  stride-scheduling pass value: which feeds run next when a provider's capacity can't
  run them all (persisted so the rotation continues across runs).
* ``scout_latest``: each token's latest full observation (one row per token, replaced),
  so a tracked token a provider outage kept from being refreshed can be carried for a
  short, labeled grace period instead of vanishing.

Version 1 stores are upgraded in place (the discovery time starts at the last-seen time).
"""

import asyncio
import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from upscale.services.scout.models import (
    ScoutCandidate,
    ScoutMarketMetrics,
    ScoutSnapshot,
    ScoutWindow,
)

SCHEMA_VERSION = 2
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
    last_seen_at REAL NOT NULL,
    last_discovered_at REAL
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
CREATE TABLE IF NOT EXISTS scout_growth_stages (
    canonical_id TEXT NOT NULL,
    computed_at REAL NOT NULL,
    stage TEXT NOT NULL,
    rank INTEGER,
    score REAL,
    PRIMARY KEY (canonical_id, computed_at)
);
CREATE TABLE IF NOT EXISTS scout_feed_schedule (
    provider TEXT NOT NULL,
    feed TEXT NOT NULL,
    pass REAL NOT NULL,
    last_run_at REAL,
    PRIMARY KEY (provider, feed)
);
CREATE TABLE IF NOT EXISTS scout_latest (
    canonical_id TEXT PRIMARY KEY,
    observed_at REAL NOT NULL,
    body_json TEXT NOT NULL
);
"""
_MIGRATE_V1 = """
ALTER TABLE scout_tokens ADD COLUMN last_discovered_at REAL;
UPDATE scout_tokens SET last_discovered_at = last_seen_at;
"""


class ScoutStoreError(Exception):
    pass


@dataclass(frozen=True)
class TrackedToken:
    canonical_id: str
    chain: str
    address: str
    last_seen_at: datetime  # last observed (discovery or refresh)
    last_discovered_at: datetime  # last surfaced by a discovery listing
    market_provider: str | None  # the provider that last priced it (None: unknown)


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

    async def discovered_tokens(self, since: datetime) -> list[tuple[str, str, str]]:
        """(canonical id, chain, address) of tokens a discovery listing surfaced since
        `since` (exact-address refreshes don't count), most recent first."""
        return await asyncio.to_thread(self._discovered, since)

    async def expired_count(self, discovered_before: datetime, seen_since: datetime) -> int:
        """Tokens still observed since `seen_since` but not surfaced by discovery since
        `discovered_before`: recently dropped from the tracked universe."""
        return await asyncio.to_thread(self._expired, discovered_before, seen_since)

    async def record_stages(
        self,
        at: datetime,
        stages: dict[str, str],
        ranks: dict[str, tuple[int | None, float]] | None = None,
    ) -> None:
        """Growth Scout's stage (and rank, score) per token for one ranking run."""
        await asyncio.to_thread(self._record_stages, at, stages, ranks or {})

    async def latest_growth(
        self, canonical_ids: Sequence[str]
    ) -> dict[str, tuple[datetime, str, int | None]]:
        """Each token's most recent Growth Scout (time, stage, rank)."""
        return await asyncio.to_thread(self._latest_growth, list(canonical_ids))

    async def ranking_run_times(self, limit: int = 6) -> list[datetime]:
        """The most recent Growth Scout ranking runs, newest first."""
        return await asyncio.to_thread(self._run_times, limit)

    async def ranking_runs(self) -> int:
        """How many Growth Scout ranking runs are stored (a persisted rotation counter)."""
        return await asyncio.to_thread(self._ranking_runs)

    async def tracked_state(self, discovered_since: datetime) -> list[TrackedToken]:
        """Tokens a discovery listing surfaced since `discovered_since`, with when they
        were last observed, last discovered, and which provider last priced them."""
        return await asyncio.to_thread(self._tracked_state, discovered_since)

    async def feed_passes(self, provider: str) -> dict[str, float]:
        """Each discovery feed's stored pass value for `provider`."""
        return await asyncio.to_thread(self._feed_passes, provider)

    async def save_feed_passes(
        self, provider: str, passes: dict[str, float], ran: Sequence[str], at: datetime
    ) -> None:
        """Store pass values; `ran`: the feeds that ran at `at`."""
        await asyncio.to_thread(self._save_feed_passes, provider, passes, list(ran), at)

    async def untrack(self, canonical_ids: Sequence[str]) -> None:
        """Stop tracking tokens confirmed gone / unusable (until a listing surfaces them
        again). Their history is kept."""
        await asyncio.to_thread(self._untrack, list(canonical_ids))

    async def save_latest(self, candidate: ScoutCandidate) -> None:
        """Replace the token's latest full observation (kept only if newer)."""
        await asyncio.to_thread(self._save_latest, candidate)

    async def latest_candidates(self, canonical_ids: Sequence[str]) -> dict[str, ScoutCandidate]:
        return await asyncio.to_thread(self._latest_candidates, list(canonical_ids))

    async def recent_stages(
        self, canonical_ids: Sequence[str], since: datetime
    ) -> dict[str, list[tuple[datetime, str]]]:
        """Stored stages since `since` per token, oldest first."""
        return await asyncio.to_thread(self._recent_stages, list(canonical_ids), since)

    async def growth_near(
        self, canonical_id: str, target: datetime, tolerance: timedelta
    ) -> tuple[datetime, str, int | None, float | None] | None:
        """The stored Growth Scout (time, stage, rank, score) of the run closest to `target`
        within `tolerance`, or None (never interpolated)."""
        return await asyncio.to_thread(self._growth_near, canonical_id, target, tolerance)

    async def tokens(self) -> list[tuple[str, str, str, str | None, str | None]]:
        """(canonical id, chain, address, symbol, name) of every token ever seen."""
        return await asyncio.to_thread(self._tokens)

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
            if version == 1:
                conn.executescript(_MIGRATE_V1)
            conn.executescript(_SCHEMA)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(scout_growth_stages)")}
            for column, kind in (("rank", "INTEGER"), ("score", "REAL")):
                if column not in columns:  # stores written before these columns existed
                    conn.execute(f"ALTER TABLE scout_growth_stages ADD COLUMN {column} {kind}")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            self._conn = conn
        return self._conn

    def _record_seen(self, c: ScoutCandidate) -> datetime:
        seen = c.observed_at.timestamp()
        source = c.sources[0] if c.sources else None
        # Surfaced by a listing (new / active / trending), not only an exact lookup.
        discovered = seen if any(s.kind != "lookup" for s in c.sources) else None
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    """
                    INSERT INTO scout_tokens (canonical_id, chain, address, symbol, name,
                        first_seen_at, first_seen_provider, first_seen_kind, last_seen_at,
                        last_discovered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (canonical_id) DO UPDATE SET
                        symbol = COALESCE(excluded.symbol, symbol),
                        name = COALESCE(excluded.name, name),
                        first_seen_at = MIN(first_seen_at, excluded.first_seen_at),
                        last_seen_at = MAX(last_seen_at, excluded.last_seen_at),
                        last_discovered_at = MAX(
                            COALESCE(last_discovered_at, excluded.last_discovered_at),
                            COALESCE(excluded.last_discovered_at, last_discovered_at)
                        )
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
                        discovered,
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

    def _discovered(self, since: datetime) -> list[tuple[str, str, str]]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    """
                    SELECT canonical_id, chain, address FROM scout_tokens
                    WHERE last_discovered_at >= ? ORDER BY last_discovered_at DESC
                    """,
                    (since.timestamp(),),
                )
                .fetchall()
            )
        return [(r[0], r[1], r[2]) for r in rows]

    def _expired(self, discovered_before: datetime, seen_since: datetime) -> int:
        with self._lock:
            row = (
                self._db()
                .execute(
                    """
                    SELECT COUNT(*) FROM scout_tokens
                    WHERE last_seen_at >= ? AND last_discovered_at < ?
                    """,
                    (seen_since.timestamp(), discovered_before.timestamp()),
                )
                .fetchone()
            )
        return int(row[0])

    def _record_stages(
        self, at: datetime, stages: dict[str, str], ranks: dict[str, tuple[int | None, float]]
    ) -> None:
        with self._lock:
            db = self._db()
            with db:
                db.executemany(
                    """
                    INSERT OR IGNORE INTO scout_growth_stages
                        (canonical_id, computed_at, stage, rank, score) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (cid, at.timestamp(), stage, *ranks.get(cid, (None, None)))
                        for cid, stage in stages.items()
                    ],
                )

    def _latest_growth(self, ids: list[str]) -> dict[str, tuple[datetime, str, int | None]]:
        out: dict[str, tuple[datetime, str, int | None]] = {}
        with self._lock:
            db = self._db()
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                rows = db.execute(
                    f"""
                    SELECT canonical_id, computed_at, stage, rank FROM scout_growth_stages s
                    WHERE canonical_id IN ({",".join("?" * len(chunk))}) AND computed_at = (
                        SELECT MAX(computed_at) FROM scout_growth_stages
                        WHERE canonical_id = s.canonical_id
                    )
                    """,
                    chunk,
                ).fetchall()
                for cid, at, stage, rank in rows:
                    out[cid] = (_dt(at), stage, rank)
        return out

    def _run_times(self, limit: int) -> list[datetime]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT DISTINCT computed_at FROM scout_growth_stages "
                    "ORDER BY computed_at DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        return [_dt(r[0]) for r in rows]

    def _ranking_runs(self) -> int:
        with self._lock:
            row = (
                self._db()
                .execute("SELECT COUNT(DISTINCT computed_at) FROM scout_growth_stages")
                .fetchone()
            )
        return int(row[0])

    def _tracked_state(self, since: datetime) -> list["TrackedToken"]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    """
                    SELECT t.canonical_id, t.chain, t.address, t.last_seen_at,
                        t.last_discovered_at, 1,
                        COALESCE(
                            json_extract(l.body_json, '$.market_provider'),
                            (SELECT s.provider FROM scout_snapshots s
                             WHERE s.canonical_id = t.canonical_id
                             ORDER BY s.observed_at DESC LIMIT 1)
                        )
                    FROM scout_tokens t LEFT JOIN scout_latest l USING (canonical_id)
                    WHERE t.last_discovered_at >= ?
                    """,
                    (since.timestamp(),),
                )
                .fetchall()
            )
        return [
            TrackedToken(
                canonical_id=r[0],
                chain=r[1],
                address=r[2],
                last_seen_at=_dt(r[3]),
                last_discovered_at=_dt(r[4]),
                market_provider=r[6],
            )
            for r in rows
        ]

    def _feed_passes(self, provider: str) -> dict[str, float]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT feed, pass FROM scout_feed_schedule WHERE provider = ?", (provider,)
                )
                .fetchall()
            )
        return {r[0]: float(r[1]) for r in rows}

    def _save_feed_passes(
        self, provider: str, passes: dict[str, float], ran: list[str], at: datetime
    ) -> None:
        with self._lock:
            db = self._db()
            with db:
                db.executemany(
                    """
                    INSERT INTO scout_feed_schedule (provider, feed, pass, last_run_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (provider, feed) DO UPDATE SET
                        pass = excluded.pass,
                        last_run_at = COALESCE(excluded.last_run_at, last_run_at)
                    """,
                    [
                        (provider, feed, value, at.timestamp() if feed in ran else None)
                        for feed, value in passes.items()
                    ],
                )

    def _untrack(self, ids: list[str]) -> None:
        with self._lock:
            db = self._db()
            with db:
                db.executemany(
                    "UPDATE scout_tokens SET last_discovered_at = NULL WHERE canonical_id = ?",
                    [(cid,) for cid in ids],
                )

    def _save_latest(self, c: ScoutCandidate) -> None:
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    """
                    INSERT INTO scout_latest (canonical_id, observed_at, body_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT (canonical_id) DO UPDATE SET
                        observed_at = excluded.observed_at, body_json = excluded.body_json
                    WHERE excluded.observed_at >= scout_latest.observed_at
                    """,
                    (c.canonical_id, c.observed_at.timestamp(), c.model_dump_json()),
                )

    def _latest_candidates(self, ids: list[str]) -> dict[str, ScoutCandidate]:
        out: dict[str, ScoutCandidate] = {}
        with self._lock:
            db = self._db()
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                rows = db.execute(
                    f"""
                    SELECT canonical_id, body_json FROM scout_latest
                    WHERE canonical_id IN ({",".join("?" * len(chunk))})
                    """,
                    chunk,
                ).fetchall()
                for cid, body in rows:
                    out[cid] = ScoutCandidate.model_validate_json(body)
        return out

    def _recent_stages(
        self, ids: list[str], since: datetime
    ) -> dict[str, list[tuple[datetime, str]]]:
        out: dict[str, list[tuple[datetime, str]]] = {}
        with self._lock:
            db = self._db()
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                rows = db.execute(
                    f"""
                    SELECT canonical_id, computed_at, stage FROM scout_growth_stages
                    WHERE computed_at >= ? AND canonical_id IN ({",".join("?" * len(chunk))})
                    ORDER BY computed_at
                    """,
                    [since.timestamp(), *chunk],
                ).fetchall()
                for cid, at, stage in rows:
                    out.setdefault(cid, []).append((_dt(at), stage))
        return out

    def _growth_near(
        self, canonical_id: str, target: datetime, tolerance: timedelta
    ) -> tuple[datetime, str, int | None, float | None] | None:
        t, tol = target.timestamp(), tolerance.total_seconds()
        with self._lock:
            row = (
                self._db()
                .execute(
                    """
                    SELECT computed_at, stage, rank, score FROM scout_growth_stages
                    WHERE canonical_id = ? AND computed_at BETWEEN ? AND ?
                    ORDER BY ABS(computed_at - ?), computed_at LIMIT 1
                    """,
                    (canonical_id, t - tol, t + tol, t),
                )
                .fetchone()
            )
        return (_dt(row[0]), row[1], row[2], row[3]) if row else None

    def _tokens(self) -> list[tuple[str, str, str, str | None, str | None]]:
        with self._lock:
            rows = (
                self._db()
                .execute("SELECT canonical_id, chain, address, symbol, name FROM scout_tokens")
                .fetchall()
            )
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

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
