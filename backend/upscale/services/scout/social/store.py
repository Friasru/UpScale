"""Durable social storage for Scout (same SQLite file as Scout's market snapshots, separate
tables). Everything is append-only: history is never rewritten.

* ``scout_social_checks``: every search outcome per token and provider (status, the time
  span the search fully covered, error). Coverage is how a window of zero mentions is told
  apart from a window nobody looked at: PROVIDER_UNAVAILABLE and
  PROVIDER_CHECKED_ZERO_MATCHES are stored as such.
* ``scout_social_events``: one row per (provider, post, token) mention, including
  AMBIGUOUS mentions (with an empty token id and their candidates) and REJECTED ones. A
  post seen again is ignored (``INSERT OR IGNORE``): first-observed values are kept.
  No post text and no author handle is stored; authors are salted opaque keys. A
  provider's own author score (e.g. Neynar's), when given, is kept per mention as
  supporting evidence.
* ``scout_social_snapshots``: per token, provider and run, the windows and trends computed
  at that moment.
* ``scout_social_momentum``: per token and run, the momentum state and its evidence.
* ``scout_social_usage``: per provider and run, requests sent, results consumed, cache
  hits and estimated cost (what enforces a provider's daily result budget).
* ``scout_meta``: the local salt for author keys (never leaves this file).

Events older than the retention period are deleted by `prune` (no long-term author
histories); snapshots and momentum hold only aggregates.
"""

import asyncio
import json
import secrets
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from upscale.services.scout.social.models import (
    ProviderStatus,
    SocialEvent,
    SocialMomentum,
    SocialSourceSnapshot,
)

SOCIAL_SCHEMA_VERSION = "2"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scout_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scout_social_checks (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    platform TEXT NOT NULL,
    checked_at REAL NOT NULL,
    status TEXT NOT NULL,
    covered_from REAL,
    covered_to REAL,
    matches INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS scout_social_checks_by_token
    ON scout_social_checks (canonical_id, provider, checked_at);
CREATE TABLE IF NOT EXISTS scout_social_events (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    platform TEXT NOT NULL,
    content_id TEXT NOT NULL,
    posted_at REAL NOT NULL,
    fetched_at REAL NOT NULL,
    author_key TEXT NOT NULL,
    token_reference TEXT NOT NULL,
    likes INTEGER, replies INTEGER, reposts INTEGER, quotes INTEGER, views INTEGER,
    engagement INTEGER,
    attribution_level TEXT NOT NULL,
    attribution_reason TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    simhash TEXT NOT NULL,
    has_contract INTEGER NOT NULL,
    promoted INTEGER,
    source_url TEXT,
    author_quality REAL,
    UNIQUE (provider, content_id, canonical_id, token_reference)
);
CREATE INDEX IF NOT EXISTS scout_social_events_by_token
    ON scout_social_events (canonical_id, posted_at);
CREATE TABLE IF NOT EXISTS scout_social_snapshots (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    observed_at REAL NOT NULL,
    status TEXT NOT NULL,
    body_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scout_social_snapshots_by_token
    ON scout_social_snapshots (canonical_id, observed_at);
CREATE TABLE IF NOT EXISTS scout_social_momentum (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    computed_at REAL NOT NULL,
    state TEXT NOT NULL,
    body_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scout_social_momentum_by_token
    ON scout_social_momentum (canonical_id, computed_at);
CREATE TABLE IF NOT EXISTS scout_social_usage (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    run_at REAL NOT NULL,
    day TEXT NOT NULL,
    requests INTEGER NOT NULL,
    results INTEGER NOT NULL,
    cache_hits INTEGER NOT NULL,
    estimated_cost_usd REAL
);
CREATE INDEX IF NOT EXISTS scout_social_usage_by_day ON scout_social_usage (provider, day);
"""


class CoverageCheck:
    """One stored search outcome (read side)."""

    def __init__(
        self,
        provider: str,
        platform: str,
        checked_at: datetime,
        status: ProviderStatus,
        covered_from: datetime | None,
        covered_to: datetime | None,
    ):
        self.provider = provider
        self.platform = platform
        self.checked_at = checked_at
        self.status = status
        self.covered_from = covered_from
        self.covered_to = covered_to


class SocialStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --- async API ------------------------------------------------------------------------

    async def author_salt(self) -> bytes:
        return await asyncio.to_thread(self._salt)

    async def record_check(
        self,
        canonical_id: str,
        provider: str,
        platform: str,
        checked_at: datetime,
        status: ProviderStatus,
        covered_from: datetime | None,
        matches: int | None,
        error: str | None,
    ) -> None:
        await asyncio.to_thread(
            self._record_check,
            canonical_id,
            provider,
            platform,
            checked_at,
            status,
            covered_from,
            matches,
            error,
        )

    async def add_events(self, events: Sequence[SocialEvent]) -> int:
        """Store new mentions; ones already stored are left untouched. Returns how many
        were new."""
        return await asyncio.to_thread(self._add_events, events)

    async def events(
        self, canonical_id: str, since: datetime, until: datetime
    ) -> list[SocialEvent]:
        return await asyncio.to_thread(self._events, canonical_id, since, until)

    async def ambiguous_events(self, since: datetime) -> list[SocialEvent]:
        return await asyncio.to_thread(self._events, "", since, None)

    async def checks(self, canonical_id: str, since: datetime) -> list[CoverageCheck]:
        return await asyncio.to_thread(self._checks, canonical_id, since)

    async def last_covered_to(self, canonical_id: str, provider: str) -> datetime | None:
        return await asyncio.to_thread(self._last_covered_to, canonical_id, provider)

    async def add_snapshot(self, snapshot: SocialSourceSnapshot) -> None:
        await asyncio.to_thread(self._add_snapshot, snapshot)

    async def snapshots(self, canonical_id: str) -> list[SocialSourceSnapshot]:
        return await asyncio.to_thread(self._snapshots, canonical_id)

    async def add_momentum(self, momentum: SocialMomentum) -> None:
        await asyncio.to_thread(self._add_momentum, momentum)

    async def momentum_history(self, canonical_id: str) -> list[SocialMomentum]:
        return await asyncio.to_thread(self._momentum_history, canonical_id)

    async def prune(self, older_than: datetime) -> int:
        return await asyncio.to_thread(self._prune, older_than)

    async def record_usage(self, provider: str, run_at: datetime, usage: dict[str, Any]) -> None:
        await asyncio.to_thread(self._record_usage, provider, run_at, usage)

    async def results_today(self, provider: str, at: datetime) -> int:
        """Results consumed by `provider` on `at`'s UTC day."""
        return await asyncio.to_thread(self._results_today, provider, at)

    # --- SQL ------------------------------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False)
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT value FROM scout_meta WHERE key = 'social_schema_version'"
            ).fetchone()
            if row and row[0] == "1":
                with conn:  # v1 -> v2: provider author score per mention
                    conn.execute("ALTER TABLE scout_social_events ADD COLUMN author_quality REAL")
                    conn.execute(
                        "UPDATE scout_meta SET value = ? WHERE key = 'social_schema_version'",
                        (SOCIAL_SCHEMA_VERSION,),
                    )
            elif row and row[0] != SOCIAL_SCHEMA_VERSION:
                conn.close()
                raise RuntimeError(f"unsupported social schema version {row[0]}")
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO scout_meta VALUES ('social_schema_version', ?)",
                    (SOCIAL_SCHEMA_VERSION,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO scout_meta VALUES ('author_salt', ?)",
                    (secrets.token_hex(32),),
                )
            self._conn = conn
        return self._conn

    def _salt(self) -> bytes:
        with self._lock:
            row = (
                self._db()
                .execute("SELECT value FROM scout_meta WHERE key = 'author_salt'")
                .fetchone()
            )
        return bytes.fromhex(row[0])

    def _record_check(
        self,
        canonical_id: str,
        provider: str,
        platform: str,
        checked_at: datetime,
        status: str,
        covered_from: datetime | None,
        matches: int | None,
        error: str | None,
    ) -> None:
        ok = status in ("PROVIDER_OK", "PROVIDER_CHECKED_ZERO_MATCHES")
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    """
                    INSERT INTO scout_social_checks (canonical_id, provider, platform,
                        checked_at, status, covered_from, covered_to, matches, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical_id,
                        provider,
                        platform,
                        checked_at.timestamp(),
                        status,
                        covered_from.timestamp() if ok and covered_from else None,
                        checked_at.timestamp() if ok else None,
                        matches,
                        error,
                    ),
                )

    def _add_events(self, events: Sequence[SocialEvent]) -> int:
        rows = [
            (
                e.canonical_id or "",
                e.provider,
                e.platform,
                e.content_id,
                e.posted_at.timestamp(),
                e.fetched_at.timestamp(),
                e.author_key,
                e.token_reference,
                e.likes,
                e.replies,
                e.reposts,
                e.quotes,
                e.views,
                e.engagement,
                e.attribution_level,
                e.attribution_reason,
                json.dumps(e.candidates),
                e.fingerprint,
                e.simhash,
                int(e.has_contract),
                None if e.promoted is None else int(e.promoted),
                e.source_url,
                e.author_quality,
            )
            for e in events
        ]
        with self._lock:
            db = self._db()
            before = db.total_changes
            with db:
                db.executemany(
                    """
                    INSERT OR IGNORE INTO scout_social_events (canonical_id, provider,
                        platform, content_id, posted_at, fetched_at, author_key,
                        token_reference, likes, replies, reposts, quotes, views, engagement,
                        attribution_level, attribution_reason, candidates_json, fingerprint,
                        simhash, has_contract, promoted, source_url, author_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            return db.total_changes - before

    def _events(
        self, canonical_id: str, since: datetime, until: datetime | None
    ) -> list[SocialEvent]:
        sql = f"SELECT {_EVENT_COLUMNS} FROM scout_social_events WHERE canonical_id = ? AND posted_at >= ?"
        params: list[Any] = [canonical_id, since.timestamp()]
        if until is not None:
            sql += " AND posted_at <= ?"
            params.append(until.timestamp())
        with self._lock:
            rows = self._db().execute(sql + " ORDER BY posted_at, id", params).fetchall()
        return [_event(r) for r in rows]

    def _checks(self, canonical_id: str, since: datetime) -> list[CoverageCheck]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    """
                SELECT provider, platform, checked_at, status, covered_from, covered_to
                FROM scout_social_checks
                WHERE canonical_id = ? AND checked_at >= ? ORDER BY checked_at, id
                """,
                    (canonical_id, since.timestamp()),
                )
                .fetchall()
            )
        return [
            CoverageCheck(r[0], r[1], _dt(r[2]), r[3], _opt_dt(r[4]), _opt_dt(r[5])) for r in rows
        ]

    def _last_covered_to(self, canonical_id: str, provider: str) -> datetime | None:
        with self._lock:
            row = (
                self._db()
                .execute(
                    """
                SELECT MAX(covered_to) FROM scout_social_checks
                WHERE canonical_id = ? AND provider = ? AND covered_to IS NOT NULL
                """,
                    (canonical_id, provider),
                )
                .fetchone()
            )
        return _opt_dt(row[0]) if row else None

    def _add_snapshot(self, s: SocialSourceSnapshot) -> None:
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    """
                    INSERT INTO scout_social_snapshots (canonical_id, provider, observed_at,
                        status, body_json) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        s.canonical_id,
                        s.provider,
                        s.observed_at.timestamp(),
                        s.status,
                        s.model_dump_json(),
                    ),
                )

    def _snapshots(self, canonical_id: str) -> list[SocialSourceSnapshot]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    """
                SELECT body_json FROM scout_social_snapshots WHERE canonical_id = ?
                ORDER BY observed_at, id
                """,
                    (canonical_id,),
                )
                .fetchall()
            )
        return [SocialSourceSnapshot.model_validate_json(r[0]) for r in rows]

    def _add_momentum(self, m: SocialMomentum) -> None:
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    """
                    INSERT INTO scout_social_momentum (canonical_id, computed_at, state, body_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (m.canonical_id, m.computed_at.timestamp(), m.state, m.model_dump_json()),
                )

    def _momentum_history(self, canonical_id: str) -> list[SocialMomentum]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    """
                SELECT body_json FROM scout_social_momentum WHERE canonical_id = ?
                ORDER BY computed_at, id
                """,
                    (canonical_id,),
                )
                .fetchall()
            )
        return [SocialMomentum.model_validate_json(r[0]) for r in rows]

    def _record_usage(self, provider: str, run_at: datetime, usage: dict[str, Any]) -> None:
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    """
                    INSERT INTO scout_social_usage (provider, run_at, day, requests, results,
                        cache_hits, estimated_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        provider,
                        run_at.timestamp(),
                        _utc_day(run_at),
                        usage.get("requests") or 0,
                        usage.get("results") or 0,
                        usage.get("cache_hits") or 0,
                        usage.get("estimated_cost_usd"),
                    ),
                )

    def _results_today(self, provider: str, at: datetime) -> int:
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT SUM(results) FROM scout_social_usage WHERE provider = ? AND day = ?",
                    (provider, _utc_day(at)),
                )
                .fetchone()
            )
        return int(row[0] or 0)

    def _prune(self, older_than: datetime) -> int:
        with self._lock:
            db = self._db()
            with db:
                cur = db.execute(
                    "DELETE FROM scout_social_events WHERE posted_at < ?",
                    (older_than.timestamp(),),
                )
        return cur.rowcount


_EVENT_COLUMNS = (
    "canonical_id, provider, platform, content_id, posted_at, fetched_at, author_key, "
    "token_reference, likes, replies, reposts, quotes, views, engagement, attribution_level, "
    "attribution_reason, candidates_json, fingerprint, simhash, has_contract, promoted, source_url, "
    "author_quality"
)


def _utc_day(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%d")


def _dt(value: float) -> datetime:
    return datetime.fromtimestamp(value, UTC)


def _opt_dt(value: float | None) -> datetime | None:
    return _dt(value) if value is not None else None


def _event(r: Sequence[Any]) -> SocialEvent:
    return SocialEvent(
        canonical_id=r[0] or None,
        provider=r[1],
        platform=r[2],
        content_id=r[3],
        posted_at=_dt(r[4]),
        fetched_at=_dt(r[5]),
        author_key=r[6],
        token_reference=r[7],
        likes=r[8],
        replies=r[9],
        reposts=r[10],
        quotes=r[11],
        views=r[12],
        engagement=r[13],
        attribution_level=r[14],
        attribution_reason=r[15],
        candidates=json.loads(r[16]),
        fingerprint=r[17],
        simhash=r[18],
        has_contract=bool(r[19]),
        promoted=None if r[20] is None else bool(r[20]),
        source_url=r[21],
        author_quality=r[22],
    )
