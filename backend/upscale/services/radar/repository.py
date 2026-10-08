"""Radar's own SQLite database (``UPSCALE_RADAR_DB``; never the Scout / Evidence / Shadow
databases). Only normalized rows and provenance are stored: raw transaction JSON never is.

Time columns (UTC epoch seconds):

* ``block_time``: when something happened on chain (from the provider), stored raw.
* ``fetched_at``: when Radar learned it (local clock). Chain and local clocks differ, so
  the CHECK is ``block_time - fetched_at <= CHAIN_CLOCK_TOLERANCE_S`` (2 s); a block time
  ahead of ``fetched_at`` is stored with ``chain_clock_ahead_s`` (NULL otherwise), and
  features use ``min(block_time, fetched_at)`` (`effective_time`) as the event time.
* ``observed_at``: the moment a holder snapshot or Radar snapshot describes. Every input
  of a snapshot satisfies ``fetched_at <= observed_at`` (`load_inputs` only reads such
  rows and re-checks them).
* ``determined_at`` / ``profiled_at``: when a creator or wallet profile was established.

Tables:

* ``radar_targets``: tracked tokens (canonical id, mint, pool) plus the activity cursor.
* ``radar_scans``: one row per bounded scan (activity / early), with its coverage. A scan
  is ``RUNNING`` only while it runs (snapshots never read one); an exception inside it
  finishes it ``ABORTED`` with the reason.
* ``radar_tx``: normalized transactions (signature, slot, block time, fee payer).
* ``radar_wallet_flows``: per wallet, per transaction ``TOKEN_INFLOW`` / ``TOKEN_OUTFLOW``.
* ``radar_wallet_entries``: long-lived roll-up per (participant, token) for
  ``NORMAL_WALLET`` / ``UNKNOWN`` participants. The first-learned block time and direction
  never change; a later, deeper scan that finds earlier history is stored as a separate
  revision with ``revised_at`` (when Radar learned it), so an as-of read before
  ``revised_at`` returns exactly what Radar knew then. ``early_fetched_at`` /
  ``wallet_evidence_at`` record when the entry became known-early / proven a wallet.
* ``radar_holder_snapshots`` / ``radar_holder_balances``: holder metrics and large
  owners' balances (only owners above the large-holder threshold, plus prior large owners).
* ``radar_creators``: ``POOL_CREATOR_CANDIDATE`` / ``TOKEN_DEPLOYER`` with method, status
  and provenance (never inferred one from the other).
* ``radar_wallets``: optional wallet-age / first-funder profiles.
* ``radar_snapshots``: ``radar.snapshot.v1`` bodies (zlib JSON + sha256), immutable.
* ``radar_requests``: Radar's request ledger per UTC day and method (the budget).
"""

import hashlib
import json
import sqlite3
import threading
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from upscale.services.radar.config import DB_SCHEMA_VERSION, SNAPSHOT_SCHEMA
from upscale.services.radar.models import (
    CHAIN_CLOCK_TOLERANCE_S,
    ParsedTx,
    RadarCausalityError,
    RadarSchemaError,
    Target,
    effective_time,
)
from upscale.services.radar.provider import Outcome

_SCHEMA = """
CREATE TABLE IF NOT EXISTS radar_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS radar_targets (
    canonical_id TEXT PRIMARY KEY,
    chain TEXT NOT NULL,
    mint TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    dex TEXT,
    pool_created_at REAL,
    pool_created_source TEXT,
    source TEXT NOT NULL,
    selected_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    last_signature TEXT,
    last_scan_at REAL,
    early_status TEXT,
    early_scanned_at REAL,
    snapshots_taken INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS radar_scans (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('activity', 'early')),
    started_at REAL NOT NULL,
    fetched_at REAL NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    signatures_listed INTEGER NOT NULL,
    txs_parsed INTEGER NOT NULL,
    txs_skipped INTEGER NOT NULL,
    reached_oldest INTEGER NOT NULL DEFAULT 0,
    window_from REAL,
    window_to REAL,
    reasons_json TEXT NOT NULL,
    CHECK (started_at <= fetched_at)
);
CREATE INDEX IF NOT EXISTS radar_scans_by_target ON radar_scans (canonical_id, fetched_at);
CREATE TABLE IF NOT EXISTS radar_tx (
    canonical_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    slot INTEGER,
    block_time REAL,
    fetched_at REAL NOT NULL,
    fee_payer TEXT,
    failed INTEGER NOT NULL,
    scan_id INTEGER,
    provider TEXT NOT NULL,
    chain_clock_ahead_s REAL,
    PRIMARY KEY (canonical_id, signature),
    {CLOCK_CHECKS}
);
CREATE INDEX IF NOT EXISTS radar_tx_by_fetch ON radar_tx (fetched_at);
CREATE TABLE IF NOT EXISTS radar_wallet_flows (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    wallet TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('TOKEN_INFLOW', 'TOKEN_OUTFLOW')),
    amount_raw TEXT NOT NULL,
    counterparty TEXT NOT NULL,
    participant TEXT NOT NULL,
    signer INTEGER NOT NULL,
    block_time REAL,
    fetched_at REAL NOT NULL,
    is_early INTEGER,
    scan_id INTEGER,
    provider TEXT NOT NULL,
    chain_clock_ahead_s REAL,
    UNIQUE (canonical_id, signature, wallet),
    {CLOCK_CHECKS}
);
CREATE INDEX IF NOT EXISTS radar_flows_by_target ON radar_wallet_flows (canonical_id, fetched_at);
CREATE INDEX IF NOT EXISTS radar_flows_by_wallet ON radar_wallet_flows (wallet, block_time);
CREATE INDEX IF NOT EXISTS radar_flows_by_fetch ON radar_wallet_flows (fetched_at);
CREATE TABLE IF NOT EXISTS radar_wallet_entries (
    wallet TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    first_block_time REAL,
    first_direction TEXT NOT NULL,
    first_fetched_at REAL NOT NULL,
    revised_block_time REAL,
    revised_direction TEXT,
    revised_at REAL,
    early_fetched_at REAL,
    wallet_evidence_at REAL,
    PRIMARY KEY (wallet, canonical_id),
    CHECK (revised_at IS NULL OR revised_at >= first_fetched_at),
    CHECK (early_fetched_at IS NULL OR early_fetched_at >= first_fetched_at),
    CHECK (wallet_evidence_at IS NULL OR wallet_evidence_at >= first_fetched_at)
);
CREATE INDEX IF NOT EXISTS radar_entries_by_target ON radar_wallet_entries (canonical_id);
CREATE TABLE IF NOT EXISTS radar_holder_snapshots (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    observed_at REAL NOT NULL,
    provider TEXT NOT NULL,
    source TEXT NOT NULL,
    holder_count INTEGER,
    holder_count_complete INTEGER NOT NULL,
    top1_pct REAL,
    top10_pct REAL,
    reliable INTEGER NOT NULL,
    lower_bound INTEGER NOT NULL,
    supply_raw TEXT,
    decimals INTEGER,
    reasons_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS radar_holder_snapshots_by_target
    ON radar_holder_snapshots (canonical_id, observed_at);
CREATE TABLE IF NOT EXISTS radar_holder_balances (
    snapshot_id INTEGER NOT NULL REFERENCES radar_holder_snapshots(id),
    owner TEXT NOT NULL,
    amount_raw TEXT,
    pct REAL,
    reason TEXT NOT NULL,
    owner_type TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, owner)
);
CREATE TABLE IF NOT EXISTS radar_creators (
    canonical_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('POOL_CREATOR_CANDIDATE', 'TOKEN_DEPLOYER')),
    status TEXT NOT NULL,
    identity TEXT,
    method TEXT NOT NULL,
    signature TEXT,
    block_time REAL,
    determined_at REAL NOT NULL,
    provider TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    PRIMARY KEY (canonical_id, role),
    CHECK (status != 'VERIFIED' OR role = 'TOKEN_DEPLOYER'),
    CHECK (identity IS NULL OR status != 'UNAVAILABLE')
);
CREATE TABLE IF NOT EXISTS radar_wallets (
    wallet TEXT PRIMARY KEY,
    profiled_at REAL NOT NULL,
    provider TEXT NOT NULL,
    age_status TEXT NOT NULL,
    oldest_block_time REAL,
    oldest_signature TEXT,
    funder TEXT,
    funder_status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS radar_snapshots (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    observed_at REAL NOT NULL,
    schema_version TEXT NOT NULL,
    coverage TEXT NOT NULL,
    body_zlib BLOB NOT NULL,
    body_hash TEXT NOT NULL,
    UNIQUE (canonical_id, observed_at)
);
CREATE TRIGGER IF NOT EXISTS radar_snapshots_no_update BEFORE UPDATE ON radar_snapshots
BEGIN SELECT RAISE(ABORT, 'radar_snapshots rows are immutable'); END;
CREATE TABLE IF NOT EXISTS radar_requests (
    day TEXT NOT NULL,
    method TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 0,
    rate_limited INTEGER NOT NULL DEFAULT 0,
    timeouts INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, method)
);
"""
# Chain time may lead the local clock by at most the tolerance, and exactly then
# chain_clock_ahead_s holds the lead (the same subtraction `record_tx` checks).
_CLOCK_CHECKS = f"""CHECK (block_time IS NULL OR block_time - fetched_at <= {CHAIN_CLOCK_TOLERANCE_S!r}),
    CHECK ((block_time IS NOT NULL AND block_time > fetched_at) = (chain_clock_ahead_s IS NOT NULL)),
    CHECK (chain_clock_ahead_s IS NULL OR chain_clock_ahead_s = block_time - fetched_at)"""
_SCHEMA = _SCHEMA.replace("{CLOCK_CHECKS}", _CLOCK_CHECKS)
TABLES = (
    "radar_meta", "radar_targets", "radar_scans", "radar_tx", "radar_wallet_flows",
    "radar_wallet_entries", "radar_holder_snapshots", "radar_holder_balances",
    "radar_creators", "radar_wallets", "radar_snapshots", "radar_requests",
)  # fmt: skip
# Participants that get a long-lived wallet-entry roll-up (programs, pools and routers
# never do: they can't become wallet history).
ENTRY_PARTICIPANTS = frozenset({"NORMAL_WALLET", "UNKNOWN"})
_OUTCOME_COLUMN = {"ok": "ok", "rate_limited": "rate_limited", "timeout": "timeouts",
                   "failed": "failures"}  # fmt: skip


@dataclass(frozen=True)
class HolderRow:
    id: int
    observed_at: float
    provider: str
    source: str
    holder_count: int | None
    holder_count_complete: bool
    top1_pct: float | None
    top10_pct: float | None
    reliable: bool
    lower_bound: bool
    reasons: list[str]
    # owner -> (raw, pct, reason, owner_type)
    balances: dict[str, tuple[int | None, float | None, str, str]]


@dataclass(frozen=True)
class FlowRow:
    signature: str
    wallet: str
    direction: str
    amount_raw: int
    counterparty: str
    block_time: float | None
    fetched_at: float
    is_early: bool | None
    participant: str = "NORMAL_WALLET"
    signer: bool = True
    chain_clock_ahead_s: float | None = None  # raw block time's lead over fetched_at

    @property
    def event_time(self) -> float | None:
        """The block time features use (never after `fetched_at`; see `effective_time`)."""
        return effective_time(self.block_time, self.fetched_at)


@dataclass(frozen=True)
class ScanRow:
    id: int
    kind: str
    started_at: float
    fetched_at: float
    status: str
    signatures_listed: int
    txs_parsed: int
    txs_skipped: int
    reached_oldest: bool
    window_from: float | None
    window_to: float | None
    reasons: list[str]


@dataclass(frozen=True)
class CreatorRow:
    role: str
    status: str
    identity: str | None
    method: str
    signature: str | None
    block_time: float | None
    determined_at: float
    provider: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class WalletProfile:
    wallet: str
    profiled_at: float
    age_status: str
    oldest_block_time: float | None
    funder: str | None
    funder_status: str


@dataclass(frozen=True)
class EntryRow:
    """A (participant, token) roll-up as known at the read time ``as_of``."""

    wallet: str
    canonical_id: str
    first_block_time: float | None  # earliest flow block time known at as_of
    block_time_known_at: float  # when Radar learned that block time (<= as_of)
    first_fetched_at: float  # when Radar first saw the participant in this token
    first_direction: str
    early: bool  # known early as of the read time
    proven_wallet: bool  # proven NORMAL_WALLET (signed a transaction) as of the read time


@dataclass
class Inputs:
    """Everything a snapshot at `as_of` may use (all fetched at or before `as_of`)."""

    as_of: float
    holders: list[HolderRow] = field(default_factory=list)  # newest first (max 2)
    flows: list[FlowRow] = field(default_factory=list)
    scans: list[ScanRow] = field(default_factory=list)
    creators: dict[str, CreatorRow] = field(default_factory=dict)
    previous_snapshot_at: float | None = None
    entries: list[EntryRow] = field(default_factory=list)  # this token
    other_entries: list[EntryRow] = field(default_factory=list)  # other tokens, same wallets
    profiles: dict[str, WalletProfile] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredTx:
    inserted: bool  # False for a transaction already stored (nothing written)
    flows: int  # new flows
    chain_clock_ahead_s: float | None  # set when the tolerance was used for this row


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _chunks(items: Sequence[str], size: int = 500) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def encode_body(body: dict[str, Any]) -> tuple[str, bytes, str]:
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text, zlib.compress(text.encode(), 9), hashlib.sha256(text.encode()).hexdigest()


def _require_compatible(conn: sqlite3.Connection, path: str) -> None:
    """Refuse a Radar database another schema version created (no migration: Radar isn't
    deployed, and an old CHECK would otherwise fail later as an IntegrityError mid-scan)."""
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not names & set(TABLES):
        return  # a new database
    found = None
    if "radar_meta" in names:
        row = conn.execute("SELECT value FROM radar_meta WHERE key = 'schema_version'").fetchone()
        found = row[0] if row else None
    if found != str(DB_SCHEMA_VERSION):
        raise RadarSchemaError(
            f"{path} is a Radar database with schema version {found or 'unknown'}; this Radar "
            f"needs version {DB_SCHEMA_VERSION} (chain-clock tolerance columns and checks). "
            "It isn't migrated: start a fresh Radar database (UPSCALE_RADAR_DB) and keep the "
            "old file for reference."
        )


class RadarRepository:
    """Synchronous and thread-safe (one connection behind a lock)."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser())
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def db(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False)
            try:
                _require_compatible(conn, self.path)
            except RadarSchemaError:
                conn.close()
                raise
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA)
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO radar_meta (key, value) VALUES ('schema_version', ?)",
                    (str(DB_SCHEMA_VERSION),),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO radar_meta (key, value) VALUES ('snapshot_schema', ?)",
                    (SNAPSHOT_SCHEMA,),
                )
            self._conn = conn
        return self._conn

    # --- request ledger (RequestLedger protocol) ------------------------------------------

    def requests_on(self, day: str) -> int:
        with self._lock:
            row = (
                self.db()
                .execute("SELECT COALESCE(SUM(calls), 0) FROM radar_requests WHERE day = ?", (day,))
                .fetchone()
            )
        return int(row[0])

    def record_request(self, day: str, method: str, outcome: Outcome) -> None:
        col = _OUTCOME_COLUMN[outcome]
        with self._lock, self.db() as conn:
            conn.execute(
                f"INSERT INTO radar_requests (day, method, calls, {col}) VALUES (?, ?, 1, 1) "
                f"ON CONFLICT (day, method) DO UPDATE SET calls = calls + 1, {col} = {col} + 1",
                (day, method),
            )

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.db().execute("SELECT value FROM radar_meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self.db() as conn:
            conn.execute(
                "INSERT INTO radar_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def requests_by_day(self, since_day: str) -> list[tuple[str, str, int, int, int, int, int]]:
        with self._lock:
            rows = (
                self.db()
                .execute(
                    "SELECT day, method, calls, ok, rate_limited, timeouts, failures FROM "
                    "radar_requests WHERE day >= ? ORDER BY day, method",
                    (since_day,),
                )
                .fetchall()
            )
        return [tuple(r) for r in rows]

    # --- targets --------------------------------------------------------------------------

    def upsert_target(
        self,
        canonical_id: str,
        mint: str,
        pool_address: str,
        dex: str | None,
        pool_created_at: float | None,
        pool_created_source: str | None,
        source: str,
        selected_at: float,
    ) -> bool:
        """Add a target (True) or keep the existing one unchanged (False): a target's pool
        never silently changes under its stored history."""
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO radar_targets (canonical_id, chain, mint, pool_address, "
                "dex, pool_created_at, pool_created_source, source, selected_at) "
                "VALUES (?, 'solana', ?, ?, ?, ?, ?, ?, ?)",
                (
                    canonical_id,
                    mint,
                    pool_address,
                    dex,
                    pool_created_at,
                    pool_created_source,
                    source,
                    selected_at,
                ),  # fmt: skip
            )
            return cur.rowcount == 1

    def set_target_status(self, canonical_id: str, status: str) -> bool:
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "UPDATE radar_targets SET status = ? WHERE canonical_id = ?", (status, canonical_id)
            )
            return cur.rowcount == 1

    _TARGET_COLS = (
        "canonical_id, chain, mint, pool_address, dex, pool_created_at, pool_created_source, "
        "source, selected_at, status, last_signature, early_status, snapshots_taken, last_scan_at"
    )

    @staticmethod
    def _target(r: Sequence[Any]) -> Target:
        return Target(*r)

    def get_target(self, canonical_id: str) -> Target | None:
        with self._lock:
            row = (
                self.db()
                .execute(
                    f"SELECT {self._TARGET_COLS} FROM radar_targets WHERE canonical_id = ?",
                    (canonical_id,),
                )
                .fetchone()
            )
        return self._target(row) if row else None

    def targets(self, status: str | None = None) -> list[Target]:
        sql = f"SELECT {self._TARGET_COLS} FROM radar_targets"
        args: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            args = (status,)
        with self._lock:
            rows = self.db().execute(sql + " ORDER BY selected_at, canonical_id", args).fetchall()
        return [self._target(r) for r in rows]

    def advance_cursor(self, canonical_id: str, signature: str | None, at: float) -> None:
        with self._lock, self.db() as conn:
            conn.execute(
                "UPDATE radar_targets SET last_signature = COALESCE(?, last_signature), "
                "last_scan_at = ? WHERE canonical_id = ?",
                (signature, at, canonical_id),
            )

    def set_early_status(self, canonical_id: str, status: str, at: float) -> None:
        with self._lock, self.db() as conn:
            conn.execute(
                "UPDATE radar_targets SET early_status = ?, early_scanned_at = ? "
                "WHERE canonical_id = ?",
                (status, at, canonical_id),
            )

    # --- transactions and flows -----------------------------------------------------------

    def known_signatures(self, canonical_id: str, signatures: Sequence[str]) -> set[str]:
        out: set[str] = set()
        with self._lock:
            for chunk in _chunks(list(signatures)):
                rows = (
                    self.db()
                    .execute(
                        f"SELECT signature FROM radar_tx WHERE canonical_id = ? AND signature IN "
                        f"({_marks(len(chunk))})",
                        (canonical_id, *chunk),
                    )
                    .fetchall()
                )
                out.update(r[0] for r in rows)
        return out

    def record_scan(
        self,
        canonical_id: str,
        kind: str,
        started_at: float,
        fetched_at: float,
        provider: str,
        status: str,
        signatures_listed: int,
        txs_parsed: int,
        txs_skipped: int,
        reached_oldest: bool,
        window: tuple[float | None, float | None],
        reasons: Sequence[str],
    ) -> int:
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO radar_scans (canonical_id, kind, started_at, fetched_at, provider, "
                "status, signatures_listed, txs_parsed, txs_skipped, reached_oldest, "
                "window_from, window_to, reasons_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    canonical_id,
                    kind,
                    started_at,
                    fetched_at,
                    provider,
                    status,
                    signatures_listed,
                    txs_parsed,
                    txs_skipped,
                    int(reached_oldest),
                    window[0],
                    window[1],
                    json.dumps(list(reasons)),
                ),  # fmt: skip
            )
            return int(cur.lastrowid or 0)

    def finish_scan(
        self,
        scan_id: int,
        status: str,
        parsed: int,
        skipped: int,
        reached_oldest: bool,
        window: tuple[float | None, float | None],
        reasons: Sequence[str],
        fetched_at: float,
    ) -> None:
        with self._lock, self.db() as conn:
            conn.execute(
                "UPDATE radar_scans SET status = ?, txs_parsed = ?, txs_skipped = ?, "
                "reached_oldest = ?, window_from = ?, window_to = ?, reasons_json = ?, "
                "fetched_at = ? WHERE id = ?",
                (
                    status,
                    parsed,
                    skipped,
                    int(reached_oldest),
                    window[0],
                    window[1],
                    json.dumps(list(reasons)),
                    fetched_at,
                    scan_id,
                ),  # fmt: skip
            )

    @staticmethod
    def _roll_up(
        conn: sqlite3.Connection,
        canonical_id: str,
        wallet: str,
        direction: str,
        participant: str,
        block_time: float | None,
        fetched_at: float,
        early: bool,
    ) -> None:
        """Insert the entry, or record what this fetch adds. The first-learned values never
        change; an earlier block time found later becomes a revision stamped `fetched_at`."""
        evidence = fetched_at if participant == "NORMAL_WALLET" else None
        early_at = fetched_at if early else None
        row = conn.execute(
            "SELECT COALESCE(revised_block_time, first_block_time) FROM radar_wallet_entries "
            "WHERE wallet = ? AND canonical_id = ?",
            (wallet, canonical_id),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO radar_wallet_entries (wallet, canonical_id, first_block_time, "
                "first_direction, first_fetched_at, early_fetched_at, wallet_evidence_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (wallet, canonical_id, block_time, direction, fetched_at, early_at, evidence),
            )
            return
        known = row[0]
        if block_time is not None and (known is None or block_time < known):
            conn.execute(
                "UPDATE radar_wallet_entries SET revised_block_time = ?, revised_direction = ?, "
                "revised_at = ? WHERE wallet = ? AND canonical_id = ?",
                (block_time, direction, fetched_at, wallet, canonical_id),
            )
        conn.execute(
            "UPDATE radar_wallet_entries SET "
            "early_fetched_at = MIN(COALESCE(early_fetched_at, ?), COALESCE(?, early_fetched_at)), "
            "wallet_evidence_at = MIN(COALESCE(wallet_evidence_at, ?), "
            "  COALESCE(?, wallet_evidence_at)) WHERE wallet = ? AND canonical_id = ?",
            (early_at, early_at, evidence, evidence, wallet, canonical_id),
        )

    def mark_early_entries(self, canonical_id: str, cutoff: float, at: float) -> None:
        """Entries whose first flow falls in the early window, learned at `at` (when the
        pool's creation time became known): only `early_fetched_at` is set, so snapshots
        before `at` never see it."""
        with self._lock, self.db() as conn:
            conn.execute(
                "UPDATE radar_wallet_entries SET early_fetched_at = ? WHERE canonical_id = ? "
                "AND early_fetched_at IS NULL AND CASE WHEN revised_at IS NOT NULL "
                "THEN MIN(revised_block_time, revised_at) "
                "ELSE MIN(first_block_time, first_fetched_at) END <= ?",
                (at, canonical_id, cutoff),
            )

    def record_tx(
        self,
        canonical_id: str,
        tx: ParsedTx,
        fetched_at: float,
        provider: str,
        scan_id: int | None,
        early_cutoff: float | None,
    ) -> StoredTx:
        """Store one normalized transaction, its flows and the wallet-entry roll-up in one
        transaction (nothing for a repeat). A block time ahead of `fetched_at` by at most
        ``CHAIN_CLOCK_TOLERANCE_S`` is stored raw with ``chain_clock_ahead_s``; further ahead
        raises `RadarCausalityError`."""
        ahead: float | None = None
        if tx.block_time is not None and tx.block_time > fetched_at:
            ahead = tx.block_time - fetched_at
            if ahead > CHAIN_CLOCK_TOLERANCE_S:
                raise RadarCausalityError(
                    f"{tx.signature}: block time {tx.block_time} is {ahead:.3f}s after fetch "
                    f"time {fetched_at} (chain-clock tolerance {CHAIN_CLOCK_TOLERANCE_S}s)"
                )
        event_time = effective_time(tx.block_time, fetched_at)
        is_early: int | None = None
        if early_cutoff is not None and event_time is not None:
            is_early = int(event_time <= early_cutoff)
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO radar_tx (canonical_id, signature, slot, block_time, "
                "fetched_at, fee_payer, failed, scan_id, provider, chain_clock_ahead_s) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    canonical_id,
                    tx.signature,
                    tx.slot,
                    tx.block_time,
                    fetched_at,
                    tx.fee_payer,
                    int(tx.failed),
                    scan_id,
                    provider,
                    ahead,
                ),  # fmt: skip
            )
            if cur.rowcount == 0:
                return StoredTx(False, 0, None)
            for f in tx.flows:
                conn.execute(
                    "INSERT OR IGNORE INTO radar_wallet_flows (canonical_id, signature, wallet, "
                    "direction, amount_raw, counterparty, participant, signer, block_time, "
                    "fetched_at, is_early, scan_id, provider, chain_clock_ahead_s) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (canonical_id, tx.signature, f.wallet, f.direction, str(f.amount_raw),
                     f.counterparty, f.participant, int(f.signer), tx.block_time, fetched_at,
                     is_early, scan_id, provider, ahead),
                )  # fmt: skip
                if f.participant in ENTRY_PARTICIPANTS:
                    self._roll_up(conn, canonical_id, f.wallet, f.direction, f.participant,
                                  tx.block_time, fetched_at, is_early == 1)  # fmt: skip
            return StoredTx(True, len(tx.flows), ahead)

    # --- holders --------------------------------------------------------------------------

    def record_holder_snapshot(
        self,
        canonical_id: str,
        observed_at: float,
        provider: str,
        source: str,
        holder_count: int | None,
        holder_count_complete: bool,
        top1_pct: float | None,
        top10_pct: float | None,
        reliable: bool,
        lower_bound: bool,
        supply_raw: int | None,
        decimals: int | None,
        reasons: Sequence[str],
        balances: Sequence[tuple[str, int | None, float | None, str, str]],
    ) -> int:
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO radar_holder_snapshots (canonical_id, observed_at, provider, source, "
                "holder_count, holder_count_complete, top1_pct, top10_pct, reliable, lower_bound, "
                "supply_raw, decimals, reasons_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    canonical_id,
                    observed_at,
                    provider,
                    source,
                    holder_count,
                    int(holder_count_complete),
                    top1_pct,
                    top10_pct,
                    int(reliable),
                    int(lower_bound),
                    str(supply_raw) if supply_raw is not None else None,
                    decimals,
                    json.dumps(list(reasons)),
                ),  # fmt: skip
            )
            sid = int(cur.lastrowid or 0)
            conn.executemany(
                "INSERT INTO radar_holder_balances (snapshot_id, owner, amount_raw, pct, reason, "
                "owner_type) VALUES (?,?,?,?,?,?)",
                [
                    (sid, o, str(a) if a is not None else None, p, r, t)
                    for o, a, p, r, t in balances
                ],
            )
            return sid

    def _holder_rows(self, canonical_id: str, as_of: float, limit: int) -> list[HolderRow]:
        conn = self.db()
        rows = conn.execute(
            "SELECT id, observed_at, provider, source, holder_count, holder_count_complete, "
            "top1_pct, top10_pct, reliable, lower_bound, reasons_json FROM radar_holder_snapshots "
            "WHERE canonical_id = ? AND observed_at <= ? ORDER BY observed_at DESC, id DESC "
            "LIMIT ?",
            (canonical_id, as_of, limit),
        ).fetchall()
        out = []
        for r in rows:
            bal = {
                b[0]: (int(b[1]) if b[1] is not None else None, b[2], b[3], b[4])
                for b in conn.execute(
                    "SELECT owner, amount_raw, pct, reason, owner_type FROM radar_holder_balances "
                    "WHERE snapshot_id = ? ORDER BY owner",
                    (r[0],),
                )
            }
            out.append(
                HolderRow(
                    id=r[0],
                    observed_at=r[1],
                    provider=r[2],
                    source=r[3],
                    holder_count=r[4],
                    holder_count_complete=bool(r[5]),
                    top1_pct=r[6],
                    top10_pct=r[7],
                    reliable=bool(r[8]),
                    lower_bound=bool(r[9]),
                    reasons=json.loads(r[10]),
                    balances=bal,
                )  # fmt: skip
            )
        return out

    def latest_large_owners(self, canonical_id: str) -> set[str]:
        """Owners that were large in the most recent holder snapshot (to re-measure)."""
        with self._lock:
            rows = self._holder_rows(canonical_id, 1e13, 1)
        return {o for o, b in rows[0].balances.items() if b[2] == "LARGE"} if rows else set()

    # --- creators and wallet profiles -----------------------------------------------------

    def record_creator(
        self,
        canonical_id: str,
        role: str,
        status: str,
        identity: str | None,
        method: str,
        signature: str | None,
        block_time: float | None,
        determined_at: float,
        provider: str,
        provenance: dict[str, Any],
    ) -> None:
        with self._lock, self.db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO radar_creators (canonical_id, role, status, identity, "
                "method, signature, block_time, determined_at, provider, provenance_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    canonical_id,
                    role,
                    status,
                    identity,
                    method,
                    signature,
                    block_time,
                    determined_at,
                    provider,
                    json.dumps(provenance, sort_keys=True),
                ),  # fmt: skip
            )

    def creator(self, canonical_id: str, role: str) -> CreatorRow | None:
        with self._lock:
            got = self._creators(canonical_id, 1e13)
        return got.get(role)

    def _creators(self, canonical_id: str, as_of: float) -> dict[str, CreatorRow]:
        rows = (
            self.db()
            .execute(
                "SELECT role, status, identity, method, signature, block_time, determined_at, "
                "provider, provenance_json FROM radar_creators WHERE canonical_id = ? AND "
                "determined_at <= ?",
                (canonical_id, as_of),
            )
            .fetchall()
        )
        return {
            r[0]: CreatorRow(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], json.loads(r[8]))
            for r in rows
        }

    def record_wallet_profile(
        self,
        wallet: str,
        profiled_at: float,
        provider: str,
        age_status: str,
        oldest_block_time: float | None,
        oldest_signature: str | None,
        funder: str | None,
        funder_status: str,
    ) -> None:
        with self._lock, self.db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO radar_wallets (wallet, profiled_at, provider, age_status, "
                "oldest_block_time, oldest_signature, funder, funder_status) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    wallet,
                    profiled_at,
                    provider,
                    age_status,
                    oldest_block_time,
                    oldest_signature,
                    funder,
                    funder_status,
                ),  # fmt: skip
            )

    def profiled_wallets(self, wallets: Sequence[str]) -> set[str]:
        out: set[str] = set()
        with self._lock:
            for chunk in _chunks(list(wallets)):
                out.update(
                    r[0]
                    for r in self.db().execute(
                        f"SELECT wallet FROM radar_wallets WHERE wallet IN ({_marks(len(chunk))})",
                        chunk,
                    )
                )
        return out

    # --- causal reads for snapshots -------------------------------------------------------

    @staticmethod
    def _resolve_entry(r: Sequence[Any], as_of: float) -> tuple[float | None, str, float]:
        """(effective block time, direction, when Radar learned it) as known at `as_of`: the
        revision only once ``revised_at <= as_of``, otherwise the first-learned values. The
        block time is capped at when Radar learned it (`effective_time`)."""
        first_bt, first_dir, first_at, rev_bt, rev_dir, rev_at = r[2], r[3], r[4], r[5], r[6], r[7]
        if rev_at is not None and rev_at <= as_of:
            return effective_time(rev_bt, rev_at), rev_dir, rev_at
        return effective_time(first_bt, first_at), first_dir, first_at

    def _entries(self, sql_where: str, args: Sequence[Any], as_of: float) -> list[EntryRow]:
        rows = (
            self.db()
            .execute(
                "SELECT wallet, canonical_id, first_block_time, first_direction, first_fetched_at, "
                "revised_block_time, revised_direction, revised_at, early_fetched_at, "
                "wallet_evidence_at FROM radar_wallet_entries WHERE first_fetched_at <= ? AND "
                f"{sql_where} ORDER BY canonical_id, wallet",
                (as_of, *args),
            )
            .fetchall()
        )
        out = []
        for r in rows:
            block_time, direction, known_at = self._resolve_entry(r, as_of)
            if known_at > as_of:  # never return what was learned after as_of
                raise RadarCausalityError(
                    f"entry {r[0]} / {r[1]}: block time learned at {known_at} > as_of {as_of}"
                )
            out.append(EntryRow(
                wallet=r[0], canonical_id=r[1], first_block_time=block_time,
                block_time_known_at=known_at, first_fetched_at=r[4], first_direction=direction,
                early=r[8] is not None and r[8] <= as_of,
                proven_wallet=r[9] is not None and r[9] <= as_of,
            ))  # fmt: skip
        return out

    def load_inputs(self, canonical_id: str, as_of: float) -> Inputs:
        """Every stored input known at `as_of`. Raises `RadarCausalityError` if a row
        fetched later slips through (a guard against query mistakes)."""
        with self._lock:
            conn = self.db()
            inp = Inputs(as_of=as_of)
            inp.holders = self._holder_rows(canonical_id, as_of, 2)
            inp.flows = [
                FlowRow(
                    r[0],
                    r[1],
                    r[2],
                    int(r[3]),
                    r[4],
                    r[5],
                    r[6],
                    None if r[7] is None else bool(r[7]),
                    r[8],
                    bool(r[9]),
                    r[10],
                )  # fmt: skip
                for r in conn.execute(
                    "SELECT signature, wallet, direction, amount_raw, counterparty, block_time, "
                    "fetched_at, is_early, participant, signer, chain_clock_ahead_s FROM "
                    "radar_wallet_flows WHERE "
                    "canonical_id = ? AND fetched_at <= ? ORDER BY block_time, signature, wallet",
                    (canonical_id, as_of),
                )
            ]
            inp.scans = [
                ScanRow(
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                    r[7],
                    bool(r[8]),
                    r[9],
                    r[10],
                    json.loads(r[11]),
                )  # fmt: skip
                for r in conn.execute(
                    "SELECT id, kind, started_at, fetched_at, status, signatures_listed, "
                    "txs_parsed, txs_skipped, reached_oldest, window_from, window_to, "
                    "reasons_json FROM radar_scans WHERE canonical_id = ? AND fetched_at <= ? "
                    "AND status != 'RUNNING' ORDER BY fetched_at, id",  # unfinished: never read
                    (canonical_id, as_of),
                )
            ]
            inp.creators = self._creators(canonical_id, as_of)
            prev = conn.execute(
                "SELECT MAX(observed_at) FROM radar_snapshots WHERE canonical_id = ? AND "
                "observed_at < ?",
                (canonical_id, as_of),
            ).fetchone()
            inp.previous_snapshot_at = prev[0] if prev else None
            inp.entries = self._entries("canonical_id = ?", (canonical_id,), as_of)
            wallets = sorted({e.wallet for e in inp.entries})
            for chunk in _chunks(wallets):
                inp.other_entries += self._entries(
                    f"canonical_id != ? AND wallet IN ({_marks(len(chunk))})",
                    (canonical_id, *chunk),
                    as_of,
                )
            inp.other_entries.sort(key=lambda e: (e.canonical_id, e.wallet))
            for chunk in _chunks(wallets):
                for r in conn.execute(
                    "SELECT wallet, profiled_at, age_status, oldest_block_time, funder, "
                    f"funder_status FROM radar_wallets WHERE profiled_at <= ? AND wallet IN "
                    f"({_marks(len(chunk))})",
                    (as_of, *chunk),
                ):
                    inp.profiles[r[0]] = WalletProfile(*r)
        late = (
            [f.fetched_at for f in inp.flows]
            + [h.observed_at for h in inp.holders]
            + [s.fetched_at for s in inp.scans]
            + [c.determined_at for c in inp.creators.values()]
            + [e.first_fetched_at for e in inp.entries + inp.other_entries]
            + [e.block_time_known_at for e in inp.entries + inp.other_entries]
            + [p.profiled_at for p in inp.profiles.values()]
        )
        if any(t > as_of for t in late):
            raise RadarCausalityError("a snapshot input was fetched after the snapshot time")
        return inp

    # --- snapshots ------------------------------------------------------------------------

    def save_snapshot(
        self, canonical_id: str, observed_at: float, coverage: str, body: dict[str, Any]
    ) -> tuple[int, str]:
        _, blob, digest = encode_body(body)
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO radar_snapshots (canonical_id, observed_at, schema_version, "
                "coverage, body_zlib, body_hash) VALUES (?,?,?,?,?,?)",
                (canonical_id, observed_at, SNAPSHOT_SCHEMA, coverage, blob, digest),
            )
            conn.execute(
                "UPDATE radar_targets SET snapshots_taken = snapshots_taken + 1 "
                "WHERE canonical_id = ?",
                (canonical_id,),
            )
            return int(cur.lastrowid or 0), digest

    def snapshot_as_of(
        self, canonical_id: str, decision_at: float
    ) -> tuple[float, dict[str, Any]] | None:
        """The latest snapshot with ``observed_at <= decision_at`` (for Audit: a Radar
        feature is only usable at a decision made after it was observed)."""
        with self._lock:
            row = (
                self.db()
                .execute(
                    "SELECT observed_at, body_zlib FROM radar_snapshots WHERE canonical_id = ? AND "
                    "observed_at <= ? ORDER BY observed_at DESC LIMIT 1",
                    (canonical_id, decision_at),
                )
                .fetchone()
            )
        if row is None:
            return None
        body: dict[str, Any] = json.loads(zlib.decompress(row[1]))
        return float(row[0]), body

    def snapshot_times(self, canonical_id: str) -> list[float]:
        with self._lock:
            rows = (
                self.db()
                .execute(
                    "SELECT observed_at FROM radar_snapshots WHERE canonical_id = ? "
                    "ORDER BY observed_at",
                    (canonical_id,),
                )
                .fetchall()
            )
        return [r[0] for r in rows]

    # --- wallets (CLI) --------------------------------------------------------------------

    def wallet_entries(self, wallet: str, as_of: float) -> list[EntryRow]:
        with self._lock:
            return self._entries("wallet = ?", (wallet,), as_of)

    def wallet_profile(self, wallet: str) -> WalletProfile | None:
        with self._lock:
            r = (
                self.db()
                .execute(
                    "SELECT wallet, profiled_at, age_status, oldest_block_time, funder, funder_status "
                    "FROM radar_wallets WHERE wallet = ?",
                    (wallet,),
                )
                .fetchone()
            )
        return WalletProfile(*r) if r else None

    def wallet_flow_count(self, wallet: str, as_of: float) -> int:
        with self._lock:
            row = (
                self.db()
                .execute(
                    "SELECT COUNT(*) FROM radar_wallet_flows WHERE wallet = ? AND fetched_at <= ?",
                    (wallet, as_of),
                )
                .fetchone()
            )
        return int(row[0])

    def row_counts(self) -> dict[str, int]:
        with self._lock:
            conn = self.db()
            return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in TABLES}
