"""The Scout archive: what the live Scout actually recorded, opened read-only.

Scout's stored snapshots are genuine point-in-time evidence (price, liquidity, market cap,
FDV, rolling-window volume / trades / buyers exactly as a provider reported them at that
moment), as are its stored stage history and social momentum. Replay reads them through
explicit time bounds: every query takes the latest instant it may return, and results are
re-checked against it (`LookaheadError` if anything later comes back).

Immutable pool metadata (quote token, DEX, creation time) is taken from the token's latest
stored observation only for the same pool, and a creation time after T is refused: those
facts are identical at any time after the pool was created.

The connection uses SQLite's ``mode=ro``: the archive can't be written through it.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from upscale.services.replay_lab.clock import LookaheadError, utc
from upscale.services.scout.models import ScoutMarketMetrics, ScoutSnapshot, ScoutWindow
from upscale.services.scout.social.models import SocialMomentum


class ArchiveUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchivedToken:
    canonical_id: str
    chain: str
    address: str
    symbol: str | None
    name: str | None
    first_seen_at: datetime


@dataclass(frozen=True)
class PoolMetadata:
    """Facts fixed when the pool was created (valid at any later time)."""

    pool_address: str
    dex: str
    quote_address: str
    quote_symbol: str | None
    created_at: datetime | None
    url: str | None


@dataclass(frozen=True)
class SnapshotTime:
    canonical_id: str
    chain: str
    address: str
    symbol: str | None
    pool_address: str
    provider: str
    observed_at: datetime


_COLUMNS = (
    "canonical_id, observed_at, provider, pool_address, dex, price_usd, market_cap_usd, "
    "fdv_usd, liquidity_usd, windows_json"
)


class ScoutArchive:
    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser())
        self._conn: sqlite3.Connection | None = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            target = Path(self.path)
            if not target.exists():
                raise ArchiveUnavailableError(f"no Scout archive at {target}")
            conn = sqlite3.connect(f"file:{target.resolve()}?mode=ro", uri=True)
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            if "scout_snapshots" not in names:
                conn.close()
                raise ArchiveUnavailableError(f"{target} has no Scout snapshots")
            self._conn = conn
        return self._conn

    def has_table(self, name: str) -> bool:
        row = (
            self._db()
            .execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,))
            .fetchone()
        )
        return row is not None

    # --- planning -----------------------------------------------------------------------------

    def snapshot_times(
        self, start: datetime, end: datetime, chains: tuple[str, ...]
    ) -> list[SnapshotTime]:
        rows = (
            self._db()
            .execute(
                f"""
                SELECT s.canonical_id, t.chain, t.address, t.symbol, s.pool_address, s.provider,
                    s.observed_at
                FROM scout_snapshots s JOIN scout_tokens t USING (canonical_id)
                WHERE s.observed_at >= ? AND s.observed_at <= ? AND s.price_usd IS NOT NULL
                    AND t.chain IN ({",".join("?" * len(chains))})
                ORDER BY s.observed_at, s.canonical_id
                """,
                [start.timestamp(), end.timestamp(), *chains],
            )
            .fetchall()
        )
        return [SnapshotTime(r[0], r[1], r[2], r[3], r[4], r[5], utc(r[6])) for r in rows]

    def tokens(self, chains: tuple[str, ...]) -> list[ArchivedToken]:
        rows = (
            self._db()
            .execute(
                f"""
                SELECT canonical_id, chain, address, symbol, name, first_seen_at FROM scout_tokens
                WHERE chain IN ({",".join("?" * len(chains))}) ORDER BY canonical_id
                """,
                list(chains),
            )
            .fetchall()
        )
        return [ArchivedToken(r[0], r[1], r[2], r[3], r[4], utc(r[5])) for r in rows]

    def token(self, canonical_id: str) -> ArchivedToken | None:
        r = (
            self._db()
            .execute(
                "SELECT canonical_id, chain, address, symbol, name, first_seen_at "
                "FROM scout_tokens WHERE canonical_id = ?",
                (canonical_id,),
            )
            .fetchone()
        )
        return ArchivedToken(r[0], r[1], r[2], r[3], r[4], utc(r[5])) if r else None

    def pool_at(self, canonical_id: str, until: datetime) -> str | None:
        """The pool Scout last priced the token on, at or before `until`."""
        r = (
            self._db()
            .execute(
                "SELECT pool_address FROM scout_snapshots WHERE canonical_id = ? AND "
                "observed_at <= ? ORDER BY observed_at DESC LIMIT 1",
                (canonical_id, until.timestamp()),
            )
            .fetchone()
        )
        return r[0] if r else None

    # --- point-in-time evidence (explicit upper bound) ----------------------------------------

    def snapshots(
        self,
        canonical_id: str,
        since: datetime,
        until: datetime,
        pool: str | None = None,
        limit: int = 500,
    ) -> list[ScoutSnapshot]:
        """Stored snapshots with since <= observed_at <= until, oldest first (the newest
        `limit`, like the live store's history)."""
        params: list[Any] = [canonical_id, since.timestamp(), until.timestamp()]
        if pool:
            params.append(pool)
        rows = (
            self._db()
            .execute(
                f"""
                SELECT {_COLUMNS} FROM scout_snapshots
                WHERE canonical_id = ? AND observed_at >= ? AND observed_at <= ?
                {"AND pool_address = ?" if pool else ""}
                ORDER BY observed_at DESC LIMIT ?
                """,
                [*params, limit],
            )
            .fetchall()
        )
        out = [_snapshot(r) for r in reversed(rows)]
        for s in out:
            if s.observed_at > until:
                raise LookaheadError(f"archive returned a snapshot after {until.isoformat()}")
        return out

    def snapshot_at(self, canonical_id: str, pool: str, at: datetime) -> ScoutSnapshot | None:
        """The snapshot of this pool stored exactly at `at` (a RECORDED sample's evidence)."""
        found = self.snapshots(canonical_id, at, at, pool=pool, limit=1)
        return found[0] if found else None

    def stages_before(
        self, canonical_id: str, since: datetime, before: datetime
    ) -> list[tuple[datetime, str, int | None, float | None]]:
        """Growth Scout stages the live system stored strictly before `before`."""
        rows = (
            self._db()
            .execute(
                """
                SELECT computed_at, stage, rank, score FROM scout_growth_stages
                WHERE canonical_id = ? AND computed_at >= ? AND computed_at < ?
                ORDER BY computed_at
                """,
                (canonical_id, since.timestamp(), before.timestamp()),
            )
            .fetchall()
        )
        out = [(utc(r[0]), r[1], r[2], r[3]) for r in rows]
        if any(at >= before for at, *_ in out):
            raise LookaheadError(f"archive returned a stage at or after {before.isoformat()}")
        return out

    def social_momentum(self, canonical_id: str, until: datetime) -> SocialMomentum | None:
        """The latest stored social momentum computed at or before `until` (proof that
        this attention evidence existed then). None when none was stored."""
        if not self.has_table("scout_social_momentum"):
            return None
        rows = (
            self._db()
            .execute(
                """
                SELECT body_json, computed_at FROM scout_social_momentum
                WHERE canonical_id = ? AND computed_at <= ? ORDER BY computed_at DESC, id DESC
                LIMIT 5
                """,
                (canonical_id, until.timestamp()),
            )
            .fetchall()
        )
        found = [SocialMomentum.model_validate_json(r[0]) for r in rows]
        measured = [m for m in found if m.state != "UNAVAILABLE"]
        best = (measured or found)[0] if found else None
        if best is not None and best.computed_at > until:
            raise LookaheadError(f"social momentum computed after {until.isoformat()}")
        return best

    def pool_metadata(self, canonical_id: str, pool: str, at: datetime) -> PoolMetadata | None:
        """Immutable facts of `pool` from the token's latest stored observation of that same
        pool. None when not recorded; a pool created after `at` is refused."""
        r = (
            self._db()
            .execute("SELECT body_json FROM scout_latest WHERE canonical_id = ?", (canonical_id,))
            .fetchone()
            if self.has_table("scout_latest")
            else None
        )
        if r is None:
            return None
        body = json.loads(r[0])
        p = body.get("pool") or {}
        if p.get("address") != pool or not p.get("quote_address"):
            return None
        created = datetime.fromisoformat(p["created_at"]) if p.get("created_at") else None
        if created is not None and created > at:
            raise LookaheadError(
                f"pool {pool} was created at {created.isoformat()}, after the decision time"
            )
        return PoolMetadata(
            pool_address=pool,
            dex=p.get("dex") or "unknown",
            quote_address=p["quote_address"],
            quote_symbol=p.get("quote_symbol"),
            created_at=created,
            url=p.get("url"),
        )

    # --- after the decision (reveal phase only; callers hold a DecisionReceipt) --------------

    def snapshots_between(
        self, canonical_id: str, pool: str, start: datetime, end: datetime
    ) -> list[ScoutSnapshot]:
        rows = (
            self._db()
            .execute(
                f"""
                SELECT {_COLUMNS} FROM scout_snapshots
                WHERE canonical_id = ? AND pool_address = ? AND observed_at > ? AND observed_at <= ?
                ORDER BY observed_at
                """,
                (canonical_id, pool, start.timestamp(), end.timestamp()),
            )
            .fetchall()
        )
        return [_snapshot(r) for r in rows]

    def stage_near(
        self, canonical_id: str, target: datetime, tolerance_seconds: float, not_before: datetime
    ) -> tuple[datetime, str, int | None, float | None] | None:
        r = (
            self._db()
            .execute(
                """
                SELECT computed_at, stage, rank, score FROM scout_growth_stages
                WHERE canonical_id = ? AND computed_at BETWEEN ? AND ? AND computed_at >= ?
                ORDER BY ABS(computed_at - ?), computed_at LIMIT 1
                """,
                (
                    canonical_id,
                    target.timestamp() - tolerance_seconds,
                    target.timestamp() + tolerance_seconds,
                    not_before.timestamp(),
                    target.timestamp(),
                ),
            )
            .fetchone()
        )
        return (utc(r[0]), r[1], r[2], r[3]) if r else None


def _snapshot(r: tuple[Any, ...]) -> ScoutSnapshot:
    return ScoutSnapshot(
        canonical_id=r[0],
        observed_at=utc(r[1]),
        provider=r[2],
        pool_address=r[3],
        dex=r[4],
        metrics=ScoutMarketMetrics(
            price_usd=r[5],
            market_cap_usd=r[6],
            fdv_usd=r[7],
            liquidity_usd=r[8],
            windows=[ScoutWindow.model_validate(w) for w in json.loads(r[9])],
        ),
    )
