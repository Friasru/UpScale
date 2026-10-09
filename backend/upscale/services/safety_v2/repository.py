"""Safety V2 SQLite storage (its own file; never another component's database).

Phase 1 tables: ``safety_meta``, ``safety_targets``, ``safety_requests``,
``safety_collections``, ``safety_mint_observations``, ``safety_snapshots``.

Phase 2 tables (schema version 2; a Phase 1 version-1 database is refused before any
mutation, never migrated and never given the new tables):

* ``safety_target_pools``: exact pool addresses pinned on a target (``pinned_at`` is when
  Safety V2 learned it; a pin applies only to snapshots with ``as_of >= pinned_at``).
* ``safety_holder_observations``: one bounded holder collection (or why it failed / wasn't
  attempted), independent of the mint observation of the same collection.
* ``safety_holder_balances``: that observation's owner-aggregated non-zero balances, with
  what the owner's own account looked like. Classification is never stored: it is derived
  per snapshot ``as_of``.

Phase 4A tables (schema version 4; a version-3 database is refused the same way): durable,
append-only copies of the Radar evidence Safety used, so snapshots and rebuilds never read
Radar (which overwrites creator rows and deletes flows / scans by retention).

* ``safety_radar_captures``: one row per capture attempt in a collection (CAPTURED /
  NOT_CONFIGURED / UNAVAILABLE / INCOMPATIBLE), with what was new and what was a repeat.
* ``safety_radar_wallet_proofs``, ``safety_radar_creator_evidence``,
  ``safety_radar_flow_evidence``, ``safety_radar_activity_coverage``: captured facts. Each
  keeps Radar's stable ``source_key`` (unique per target: re-ingesting a fact is a no-op),
  Radar's knowledge time (``source_time``), Safety's ``captured_at`` and the capture id.
  A fact is usable as of ``as_of`` only when ``source_time <= as_of AND captured_at <=
  as_of``.

Phase 3 tables (schema version 3):

* ``safety_market_observations``: one DEX-provider market read (POOLS / NO_POOLS /
  PROVIDER_FAILED / NOT_COLLECTED), with the collection-time primary (audit only: the
  snapshot re-derives selection per ``as_of``).
* ``safety_market_pools``: every pool that read returned (parsed), with its identity
  relative to the exact mint, PROVIDER_REPORTED figures, eligibility and rejection reasons.
* ``safety_pool_account_observations``: one ``getAccountInfo(pool)``: exact on-chain
  presence evidence for one pool address.

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
from upscale.services.safety_v2.market import AccountCheck, MarketPool, ObservedMarket
from upscale.services.safety_v2.models import (
    HOLDER_OUTCOMES,
    HOLDER_SOURCES,
    MARKET_OUTCOMES,
    MINT_OUTCOMES,
    OWNER_LOOKUPS,
    POOL_ACCOUNT_OUTCOMES,
    POOL_IDENTITIES,
    SafetySchemaError,
    SafetyStateError,
)
from upscale.services.safety_v2.provider import (
    HolderObservation,
    MarketObservation,
    MintObservation,
    OwnerFact,
    PoolAccountObservation,
    RequestOutcome,
)
from upscale.services.safety_v2.sources import (
    RadarCapture,
    WalletProof,
)
from upscale.services.solana_chain import TOKEN_2022_PROGRAM, TOKEN_PROGRAM

_OUTCOMES = ",".join(repr(o) for o in MINT_OUTCOMES)
_HOLDER_OUTCOMES = ",".join(repr(o) for o in HOLDER_OUTCOMES)
_HOLDER_SOURCES = ",".join(repr(o) for o in HOLDER_SOURCES)
_LOOKUPS = ",".join(repr(o) for o in OWNER_LOOKUPS)
HOLDER_ORIGIN = "safety_v2.collect_holders"
MARKET_ORIGIN = "safety_v2.collect_market"
_MARKET_OUTCOMES = ",".join(repr(o) for o in MARKET_OUTCOMES)
_IDENTITIES = ",".join(repr(o) for o in POOL_IDENTITIES)
_ACCOUNT_OUTCOMES = ",".join(repr(o) for o in POOL_ACCOUNT_OUTCOMES)


def _radar_fact_table(name: str, columns: str) -> str:
    """A captured-Radar-fact table: append-only, unique per (target, source key), written
    only by a CAPTURED capture of a RUNNING collection of the same target."""
    return f"""CREATE TABLE IF NOT EXISTS {name} (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    capture_id INTEGER NOT NULL REFERENCES safety_radar_captures(id),
    collection_id INTEGER NOT NULL REFERENCES safety_collections(id),
    captured_at REAL NOT NULL,
    source_system TEXT NOT NULL CHECK (source_system = 'radar'),
    radar_schema_version TEXT NOT NULL,
    source_key TEXT NOT NULL,
    source_time REAL NOT NULL,{columns}
    UNIQUE (canonical_id, source_key)
);
CREATE TRIGGER IF NOT EXISTS {name}_from_capture BEFORE INSERT ON {name}
WHEN NOT EXISTS (SELECT 1 FROM safety_radar_captures r JOIN safety_collections c
    ON c.id = r.collection_id WHERE r.id = NEW.capture_id AND r.status = 'CAPTURED'
    AND r.canonical_id = NEW.canonical_id AND r.collection_id = NEW.collection_id
    AND r.captured_at = NEW.captured_at AND c.status = 'RUNNING')
BEGIN SELECT RAISE(ABORT, '{name}: needs a CAPTURED capture of a RUNNING collection'); END;
CREATE TRIGGER IF NOT EXISTS {name}_no_update BEFORE UPDATE ON {name}
BEGIN SELECT RAISE(ABORT, '{name} rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS {name}_no_delete BEFORE DELETE ON {name}
BEGIN SELECT RAISE(ABORT, '{name} rows are append-only'); END;"""


_RUNNING = """WHEN NOT EXISTS (SELECT 1 FROM safety_collections c WHERE c.id = NEW.collection_id
    AND c.status = 'RUNNING' AND c.canonical_id = NEW.canonical_id
    AND c.started_at <= NEW.fetched_at)"""
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
CREATE TABLE IF NOT EXISTS safety_target_pools (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    pool_address TEXT NOT NULL,
    dex TEXT,
    pinned_at REAL NOT NULL,
    source TEXT NOT NULL,
    UNIQUE (canonical_id, pool_address),
    CHECK (canonical_id != 'solana:' || pool_address)
);
CREATE TRIGGER IF NOT EXISTS safety_pools_no_update BEFORE UPDATE ON safety_target_pools
BEGIN SELECT RAISE(ABORT, 'safety_target_pools rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_pools_no_delete BEFORE DELETE ON safety_target_pools
BEGIN SELECT RAISE(ABORT, 'safety_target_pools rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_holder_observations (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    collection_id INTEGER NOT NULL REFERENCES safety_collections(id),
    fetched_at REAL NOT NULL,
    provider TEXT NOT NULL,
    origin TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ({_HOLDER_OUTCOMES})),
    reason TEXT,
    source TEXT CHECK (source IN ({_HOLDER_SOURCES})),
    supply_raw TEXT,
    decimals INTEGER CHECK (decimals BETWEEN 0 AND 255),
    pages_read INTEGER NOT NULL CHECK (pages_read >= 0),
    max_pages INTEGER NOT NULL CHECK (max_pages >= pages_read),
    token_accounts_seen INTEGER NOT NULL CHECK (token_accounts_seen >= 0),
    largest_accounts_seen INTEGER NOT NULL CHECK (largest_accounts_seen BETWEEN 0 AND 20),
    skipped_rows INTEGER NOT NULL CHECK (skipped_rows >= 0),
    holder_count INTEGER CHECK (holder_count >= 0),
    holder_count_complete INTEGER NOT NULL CHECK (holder_count_complete IN (0, 1)),
    reliable INTEGER NOT NULL CHECK (reliable IN (0, 1)),
    reasons_json TEXT NOT NULL,
    owners_hash TEXT,
    CHECK ((outcome = 'COLLECTED') = (source IS NOT NULL)),
    CHECK ((outcome = 'COLLECTED') = (supply_raw IS NOT NULL)),
    CHECK ((outcome = 'COLLECTED') = (decimals IS NOT NULL)),
    CHECK ((outcome = 'COLLECTED') = (owners_hash IS NOT NULL)),
    CHECK ((outcome = 'COLLECTED') = (reason IS NULL)),
    CHECK (outcome = 'COLLECTED' OR (holder_count IS NULL AND holder_count_complete = 0
        AND reliable = 0 AND token_accounts_seen = 0 AND pages_read = 0)),
    CHECK (holder_count_complete = 0 OR (source = 'full_scan' AND holder_count IS NOT NULL)),
    CHECK (source IS NOT 'largest_accounts' OR (pages_read = 0 AND holder_count IS NULL)),
    CHECK (source NOT IN ('full_scan', 'partial_scan') OR pages_read >= 1),
    CHECK (supply_raw IS NULL OR (supply_raw != '' AND supply_raw NOT GLOB '*[^0-9]*'))
);
CREATE INDEX IF NOT EXISTS safety_holders_by_target
    ON safety_holder_observations (canonical_id, fetched_at);
CREATE TRIGGER IF NOT EXISTS safety_holders_in_collection
BEFORE INSERT ON safety_holder_observations
WHEN NOT EXISTS (SELECT 1 FROM safety_collections c WHERE c.id = NEW.collection_id
    AND c.status = 'RUNNING' AND c.canonical_id = NEW.canonical_id
    AND c.started_at <= NEW.fetched_at)
BEGIN SELECT RAISE(ABORT, 'safety_holder_observations: needs a RUNNING collection of the same target started at or before fetched_at'); END;
CREATE TRIGGER IF NOT EXISTS safety_holders_no_update BEFORE UPDATE ON safety_holder_observations
BEGIN SELECT RAISE(ABORT, 'safety_holder_observations rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_holders_no_delete BEFORE DELETE ON safety_holder_observations
BEGIN SELECT RAISE(ABORT, 'safety_holder_observations rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_holder_balances (
    observation_id INTEGER NOT NULL REFERENCES safety_holder_observations(id),
    rank INTEGER NOT NULL CHECK (rank >= 1),
    owner_key TEXT NOT NULL,
    owner TEXT,
    accounts_json TEXT NOT NULL,
    amount_raw TEXT NOT NULL
        CHECK (amount_raw != '' AND amount_raw NOT GLOB '*[^0-9]*' AND amount_raw NOT GLOB '0*'),
    lookup TEXT NOT NULL CHECK (lookup IN ({_LOOKUPS})),
    owner_program TEXT,
    PRIMARY KEY (observation_id, owner_key),
    UNIQUE (observation_id, rank),
    CHECK ((owner IS NULL) = (owner_key GLOB 'unresolved:*')),
    CHECK (owner IS NULL OR owner_key = owner),
    CHECK ((lookup = 'FOUND') = (owner_program IS NOT NULL)),
    CHECK (owner IS NOT NULL OR lookup = 'NOT_LOOKED_UP')
);
CREATE TRIGGER IF NOT EXISTS safety_balances_in_collection
BEFORE INSERT ON safety_holder_balances
WHEN NOT EXISTS (SELECT 1 FROM safety_holder_observations o JOIN safety_collections c
    ON c.id = o.collection_id WHERE o.id = NEW.observation_id AND o.outcome = 'COLLECTED'
    AND c.status = 'RUNNING')
BEGIN SELECT RAISE(ABORT, 'safety_holder_balances: needs a COLLECTED observation of a RUNNING collection'); END;
CREATE TRIGGER IF NOT EXISTS safety_balances_no_update BEFORE UPDATE ON safety_holder_balances
BEGIN SELECT RAISE(ABORT, 'safety_holder_balances rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_balances_no_delete BEFORE DELETE ON safety_holder_balances
BEGIN SELECT RAISE(ABORT, 'safety_holder_balances rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_market_observations (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    collection_id INTEGER NOT NULL REFERENCES safety_collections(id),
    fetched_at REAL NOT NULL,
    provider TEXT NOT NULL,
    origin TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ({_MARKET_OUTCOMES})),
    reason TEXT,
    raw_hash TEXT,
    rows_returned INTEGER NOT NULL CHECK (rows_returned >= 0),
    malformed_rows INTEGER NOT NULL CHECK (malformed_rows BETWEEN 0 AND rows_returned),
    primary_pool TEXT,
    primary_clear INTEGER CHECK (primary_clear IN (0, 1)),
    reasons_json TEXT NOT NULL,
    CHECK ((outcome IN ('POOLS', 'NO_POOLS')) = (raw_hash IS NOT NULL)),
    CHECK ((outcome IN ('POOLS', 'NO_POOLS')) = (reason IS NULL)),
    CHECK (outcome IN ('POOLS', 'NO_POOLS') OR (rows_returned = 0 AND malformed_rows = 0)),
    CHECK (primary_pool IS NULL OR outcome = 'POOLS'),
    CHECK ((primary_pool IS NULL) = (primary_clear IS NULL))
);
CREATE INDEX IF NOT EXISTS safety_market_by_target
    ON safety_market_observations (canonical_id, fetched_at);
CREATE TRIGGER IF NOT EXISTS safety_market_in_collection
BEFORE INSERT ON safety_market_observations {_RUNNING}
BEGIN SELECT RAISE(ABORT, 'safety_market_observations: needs a RUNNING collection of the same target started at or before fetched_at'); END;
CREATE TRIGGER IF NOT EXISTS safety_market_no_update BEFORE UPDATE ON safety_market_observations
BEGIN SELECT RAISE(ABORT, 'safety_market_observations rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_market_no_delete BEFORE DELETE ON safety_market_observations
BEGIN SELECT RAISE(ABORT, 'safety_market_observations rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_market_pools (
    observation_id INTEGER NOT NULL REFERENCES safety_market_observations(id),
    rank INTEGER NOT NULL CHECK (rank >= 1),
    pair_address TEXT NOT NULL,
    identity TEXT NOT NULL CHECK (identity IN ({_IDENTITIES})),
    chain TEXT NOT NULL,
    dex TEXT NOT NULL,
    base_mint TEXT NOT NULL,
    base_symbol TEXT,
    quote_address TEXT NOT NULL,
    quote_symbol TEXT,
    quote_kind TEXT NOT NULL,
    liquidity_usd REAL CHECK (liquidity_usd >= 0),
    volume_24h REAL CHECK (volume_24h >= 0),
    txns_24h INTEGER CHECK (txns_24h >= 0),
    price_usd REAL CHECK (price_usd > 0),
    pair_created_at REAL,
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    rejections_json TEXT NOT NULL,
    PRIMARY KEY (observation_id, pair_address),
    UNIQUE (observation_id, rank),
    CHECK (eligible = 0 OR (identity = 'EXACT_BASE' AND rejections_json = '[]'
        AND liquidity_usd IS NOT NULL AND price_usd IS NOT NULL AND txns_24h IS NOT NULL)),
    CHECK (identity != 'EXACT_BASE' OR chain = 'solana'),
    CHECK (identity = 'EXACT_BASE' OR rejections_json != '[]')
);
CREATE TRIGGER IF NOT EXISTS safety_market_pools_exact_mint
BEFORE INSERT ON safety_market_pools
WHEN NEW.identity = 'EXACT_BASE' AND NOT EXISTS (SELECT 1 FROM safety_market_observations o
    WHERE o.id = NEW.observation_id AND o.canonical_id = 'solana:' || NEW.base_mint)
BEGIN SELECT RAISE(ABORT, 'safety_market_pools: an EXACT_BASE pool must have the target mint as its base'); END;
CREATE TRIGGER IF NOT EXISTS safety_market_pools_in_collection
BEFORE INSERT ON safety_market_pools
WHEN NOT EXISTS (SELECT 1 FROM safety_market_observations o JOIN safety_collections c
    ON c.id = o.collection_id WHERE o.id = NEW.observation_id
    AND o.outcome IN ('POOLS', 'NO_POOLS') AND c.status = 'RUNNING')
BEGIN SELECT RAISE(ABORT, 'safety_market_pools: needs a successful observation of a RUNNING collection'); END;
CREATE TRIGGER IF NOT EXISTS safety_market_pools_no_update BEFORE UPDATE ON safety_market_pools
BEGIN SELECT RAISE(ABORT, 'safety_market_pools rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_market_pools_no_delete BEFORE DELETE ON safety_market_pools
BEGIN SELECT RAISE(ABORT, 'safety_market_pools rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_pool_account_observations (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    collection_id INTEGER NOT NULL REFERENCES safety_collections(id),
    pool_address TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    provider TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ({_ACCOUNT_OUTCOMES})),
    program_owner TEXT,
    lamports INTEGER CHECK (lamports >= 0),
    raw_hash TEXT,
    reason TEXT,
    CHECK (canonical_id != 'solana:' || pool_address),
    CHECK ((outcome = 'EXISTS') = (program_owner IS NOT NULL)),
    CHECK (outcome = 'EXISTS' OR lamports IS NULL),
    CHECK ((outcome IN ('EXISTS', 'ACCOUNT_MISSING')) = (raw_hash IS NOT NULL)),
    CHECK ((outcome IN ('EXISTS', 'ACCOUNT_MISSING')) = (reason IS NULL))
);
CREATE TRIGGER IF NOT EXISTS safety_pool_accounts_in_collection
BEFORE INSERT ON safety_pool_account_observations {_RUNNING}
BEGIN SELECT RAISE(ABORT, 'safety_pool_account_observations: needs a RUNNING collection of the same target started at or before fetched_at'); END;
CREATE TRIGGER IF NOT EXISTS safety_pool_accounts_no_update
BEFORE UPDATE ON safety_pool_account_observations
BEGIN SELECT RAISE(ABORT, 'safety_pool_account_observations rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_pool_accounts_no_delete
BEFORE DELETE ON safety_pool_account_observations
BEGIN SELECT RAISE(ABORT, 'safety_pool_account_observations rows are append-only'); END;
CREATE TABLE IF NOT EXISTS safety_radar_captures (
    id INTEGER PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES safety_targets(canonical_id),
    collection_id INTEGER NOT NULL REFERENCES safety_collections(id),
    captured_at REAL NOT NULL,
    fetched_at REAL NOT NULL,
    source_system TEXT NOT NULL CHECK (source_system = 'radar'),
    status TEXT NOT NULL
        CHECK (status IN ('CAPTURED', 'NOT_CONFIGURED', 'UNAVAILABLE', 'INCOMPATIBLE')),
    reason TEXT,
    radar_schema_version TEXT,
    new_facts INTEGER NOT NULL DEFAULT 0 CHECK (new_facts >= 0),
    repeated_facts INTEGER NOT NULL DEFAULT 0 CHECK (repeated_facts >= 0),
    CHECK (fetched_at = captured_at),
    CHECK ((status = 'CAPTURED') = (reason IS NULL)),
    CHECK ((status = 'CAPTURED') = (radar_schema_version IS NOT NULL)),
    CHECK (status = 'CAPTURED' OR (new_facts = 0 AND repeated_facts = 0))
);
CREATE TRIGGER IF NOT EXISTS safety_radar_captures_in_collection
BEFORE INSERT ON safety_radar_captures {_RUNNING}
BEGIN SELECT RAISE(ABORT, 'safety_radar_captures: needs a RUNNING collection of the same target started at or before fetched_at'); END;
CREATE TRIGGER IF NOT EXISTS safety_radar_captures_no_update BEFORE UPDATE ON safety_radar_captures
BEGIN SELECT RAISE(ABORT, 'safety_radar_captures rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS safety_radar_captures_no_delete BEFORE DELETE ON safety_radar_captures
BEGIN SELECT RAISE(ABORT, 'safety_radar_captures rows are append-only'); END;
{
    _radar_fact_table(
        "safety_radar_wallet_proofs",
        '''
    wallet TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('SIGNER_FLOW', 'WALLET_ENTRY')),
    radar_canonical_id TEXT NOT NULL,
    signature TEXT,
    CHECK ((kind = 'SIGNER_FLOW') = (signature IS NOT NULL)),''',
    )
}
{
    _radar_fact_table(
        "safety_radar_creator_evidence",
        '''
    role TEXT NOT NULL CHECK (role IN ('POOL_CREATOR_CANDIDATE', 'TOKEN_DEPLOYER')),
    status TEXT NOT NULL CHECK (status IN ('CANDIDATE', 'VERIFIED', 'UNAVAILABLE')),
    address TEXT,
    method TEXT NOT NULL,
    signature TEXT,
    block_time REAL,
    radar_provider TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    CHECK (status != 'VERIFIED' OR role = 'TOKEN_DEPLOYER'),
    CHECK (status != 'CANDIDATE' OR role = 'POOL_CREATOR_CANDIDATE'),
    CHECK ((status = 'UNAVAILABLE') = (address IS NULL)),''',
    )
}
{
    _radar_fact_table(
        "safety_radar_flow_evidence",
        '''
    wallet TEXT NOT NULL,
    signature TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction = 'TOKEN_OUTFLOW'),
    amount_raw TEXT NOT NULL CHECK (amount_raw != '' AND amount_raw NOT GLOB '*[^0-9]*'),
    block_time REAL,
    signer INTEGER NOT NULL CHECK (signer IN (0, 1)),
    participant TEXT NOT NULL,
    counterparty TEXT NOT NULL,
    radar_scan_id INTEGER,''',
    )
}
{
    _radar_fact_table(
        "safety_radar_activity_coverage",
        '''
    kind TEXT NOT NULL CHECK (kind IN ('SCAN', 'GAP')),
    radar_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status != 'RUNNING'),
    facts_json TEXT NOT NULL,''',
    )
}
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
    "safety_mint_observations", "safety_snapshots", "safety_target_pools",
    "safety_holder_observations", "safety_holder_balances", "safety_market_observations",
    "safety_market_pools", "safety_pool_account_observations", "safety_radar_captures",
    "safety_radar_wallet_proofs", "safety_radar_creator_evidence", "safety_radar_flow_evidence",
    "safety_radar_activity_coverage",
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
class HolderRow:
    id: int
    canonical_id: str
    collection_id: int
    fetched_at: float
    provider: str
    origin: str
    outcome: str
    reason: str | None
    source: str | None
    supply_raw: str | None
    decimals: int | None
    pages_read: int
    max_pages: int
    token_accounts_seen: int
    largest_accounts_seen: int
    skipped_rows: int
    holder_count: int | None
    holder_count_complete: bool
    reliable: bool
    reasons: list[str]
    owners_hash: str | None


@dataclass(frozen=True)
class PoolPin:
    id: int
    canonical_id: str
    pool_address: str
    dex: str | None
    pinned_at: float
    source: str


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


_HOLDER_COLUMNS = (
    "id, canonical_id, collection_id, fetched_at, provider, origin, outcome, reason, source, "
    "supply_raw, decimals, pages_read, max_pages, token_accounts_seen, largest_accounts_seen, "
    "skipped_rows, holder_count, holder_count_complete, reliable, reasons_json, owners_hash"
)


def _holder_row(r: tuple[Any, ...]) -> HolderRow:
    return HolderRow(
        id=r[0], canonical_id=r[1], collection_id=r[2], fetched_at=r[3], provider=r[4],
        origin=r[5], outcome=r[6], reason=r[7], source=r[8], supply_raw=r[9], decimals=r[10],
        pages_read=r[11], max_pages=r[12], token_accounts_seen=r[13],
        largest_accounts_seen=r[14], skipped_rows=r[15], holder_count=r[16],
        holder_count_complete=bool(r[17]), reliable=bool(r[18]), reasons=json.loads(r[19]),
        owners_hash=r[20],
    )  # fmt: skip


def owners_hash(owners: tuple[OwnerFact, ...] | list[OwnerFact]) -> str:
    """SHA-256 of the canonical owner facts (provenance of one holder observation)."""
    rows = [[o.key, o.owner, list(o.accounts), str(o.amount_raw), o.lookup, o.owner_program]
            for o in owners]  # fmt: skip
    text = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


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

    # --- pool pins (Phase 2) ---------------------------------------------------------------

    def pin_pool(
        self, canonical_id: str, pool_address: str, dex: str | None, source: str, at: float
    ) -> bool:
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO safety_target_pools (canonical_id, pool_address, dex, "
                "pinned_at, source) VALUES (?, ?, ?, ?, ?)",
                (canonical_id, pool_address, dex, at, source),
            )
        return cur.rowcount == 1

    def pool_pins(self, canonical_id: str, as_of: float) -> list[PoolPin]:
        """Pins of this target learned at or before `as_of`."""
        with self._lock:
            rows = (
                self.db()
                .execute(
                    "SELECT id, canonical_id, pool_address, dex, pinned_at, source FROM "
                    "safety_target_pools WHERE canonical_id = ? AND pinned_at <= ? "
                    "ORDER BY pool_address",
                    (canonical_id, as_of),
                )
                .fetchall()
            )
        return [PoolPin(*r) for r in rows]

    # --- holder observations (Phase 2) -----------------------------------------------------

    def record_holder_observation(
        self,
        canonical_id: str,
        collection_id: int,
        fetched_at: float,
        provider: str,
        obs: HolderObservation,
    ) -> int:
        """The observation and its balances, in one transaction."""
        collected = obs.outcome == "COLLECTED"
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO safety_holder_observations (canonical_id, collection_id, "
                "fetched_at, provider, origin, outcome, reason, source, supply_raw, decimals, "
                "pages_read, max_pages, token_accounts_seen, largest_accounts_seen, "
                "skipped_rows, holder_count, holder_count_complete, reliable, reasons_json, "
                "owners_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (canonical_id, collection_id, fetched_at, provider, HOLDER_ORIGIN, obs.outcome,
                 obs.reason, obs.source,
                 str(obs.supply_raw) if obs.supply_raw is not None else None, obs.decimals,
                 obs.pages_read, obs.max_pages, obs.token_accounts_seen,
                 obs.largest_accounts_seen, obs.skipped_rows, obs.holder_count,
                 int(obs.holder_count_complete), int(obs.reliable), json.dumps(list(obs.reasons)),
                 owners_hash(obs.owners) if collected else None),
            )  # fmt: skip
            oid = cur.lastrowid
            assert oid is not None
            conn.executemany(
                "INSERT INTO safety_holder_balances (observation_id, rank, owner_key, owner, "
                "accounts_json, amount_raw, lookup, owner_program) VALUES (?,?,?,?,?,?,?,?)",
                [(oid, rank, o.key, o.owner, json.dumps(list(o.accounts)), str(o.amount_raw),
                  o.lookup, o.owner_program)
                 for rank, o in enumerate(obs.owners, 1)],
            )  # fmt: skip
        return oid

    def latest_holder_observation(self, canonical_id: str, as_of: float) -> HolderRow | None:
        """The newest holder observation of this exact identity fetched at or before
        `as_of`."""
        with self._lock:
            row = (
                self.db()
                .execute(
                    f"SELECT {_HOLDER_COLUMNS} FROM safety_holder_observations "
                    "WHERE canonical_id = ? AND fetched_at <= ? "
                    "ORDER BY fetched_at DESC, id DESC LIMIT 1",
                    (canonical_id, as_of),
                )
                .fetchone()
            )
        return _holder_row(row) if row else None

    def holder_observations(self, canonical_id: str, as_of: float) -> list[HolderRow]:
        """Every holder observation of this identity fetched at or before `as_of`, in
        (fetched_at, id) order: the stable order Phase 4B pairs are chosen in."""
        with self._lock:
            rows = (
                self.db()
                .execute(
                    f"SELECT {_HOLDER_COLUMNS} FROM safety_holder_observations "
                    "WHERE canonical_id = ? AND fetched_at <= ? ORDER BY fetched_at, id",
                    (canonical_id, as_of),
                )
                .fetchall()
            )
        return [_holder_row(r) for r in rows]

    def holder_observation(self, observation_id: int) -> HolderRow:
        with self._lock:
            row = (
                self.db()
                .execute(
                    f"SELECT {_HOLDER_COLUMNS} FROM safety_holder_observations WHERE id = ?",
                    (observation_id,),
                )
                .fetchone()
            )
        if row is None:
            raise SafetyStateError(f"no holder observation {observation_id}")
        return _holder_row(row)

    def holder_balances(self, observation_id: int) -> list[OwnerFact]:
        """The observation's balances, largest first (as collected)."""
        with self._lock:
            rows = (
                self.db()
                .execute(
                    "SELECT owner_key, owner, accounts_json, amount_raw, lookup, owner_program "
                    "FROM safety_holder_balances WHERE observation_id = ? ORDER BY rank",
                    (observation_id,),
                )
                .fetchall()
            )
        return [OwnerFact(k, o, tuple(json.loads(a)), int(amt), lk, p)
                for k, o, a, amt, lk, p in rows]  # fmt: skip

    # --- market observations (Phase 3) -----------------------------------------------------

    def record_market_observation(
        self,
        canonical_id: str,
        collection_id: int,
        fetched_at: float,
        provider: str,
        obs: MarketObservation,
    ) -> int:
        """The observation and its pools, in one transaction."""
        clear = None if obs.primary_clear is None else int(obs.primary_clear)
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO safety_market_observations (canonical_id, collection_id, "
                "fetched_at, provider, origin, outcome, reason, raw_hash, rows_returned, "
                "malformed_rows, primary_pool, primary_clear, reasons_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (canonical_id, collection_id, fetched_at, provider, MARKET_ORIGIN, obs.outcome,
                 obs.reason, obs.raw_hash, obs.rows_returned, obs.malformed_rows,
                 obs.primary_pool, clear, json.dumps(list(obs.reasons))),
            )  # fmt: skip
            oid = cur.lastrowid
            assert oid is not None
            conn.executemany(
                "INSERT INTO safety_market_pools (observation_id, rank, pair_address, identity, "
                "chain, dex, base_mint, base_symbol, quote_address, quote_symbol, quote_kind, "
                "liquidity_usd, volume_24h, txns_24h, price_usd, pair_created_at, eligible, "
                "rejections_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(oid, rank, p.pair_address, p.identity, p.chain, p.dex, p.base_mint,
                  p.base_symbol, p.quote_address, p.quote_symbol, p.quote_kind, p.liquidity_usd,
                  p.volume_24h, p.txns_24h, p.price_usd, p.pair_created_at, int(p.eligible),
                  json.dumps(list(p.rejections)))
                 for rank, p in enumerate(obs.pools, 1)],
            )  # fmt: skip
        return oid

    def market_observations(self, canonical_id: str, as_of: float) -> list[ObservedMarket]:
        """Every market observation of this identity fetched at or before `as_of`, oldest
        first, with its pools (in stored rank order)."""
        with self._lock:
            conn = self.db()
            rows = conn.execute(
                "SELECT id, collection_id, fetched_at, outcome, reason, provider FROM "
                "safety_market_observations WHERE canonical_id = ? AND fetched_at <= ? "
                "ORDER BY fetched_at, id",
                (canonical_id, as_of),
            ).fetchall()
            pools: dict[int, list[MarketPool]] = {}
            if rows:
                for r in conn.execute(
                    "SELECT p.observation_id, p.pair_address, p.identity, p.chain, p.dex, "
                    "p.base_mint, p.base_symbol, p.quote_address, p.quote_symbol, p.quote_kind, "
                    "p.liquidity_usd, p.volume_24h, p.txns_24h, p.price_usd, p.pair_created_at, "
                    "p.eligible, p.rejections_json FROM safety_market_pools p "
                    "JOIN safety_market_observations o ON o.id = p.observation_id "
                    "WHERE o.canonical_id = ? AND o.fetched_at <= ? "
                    "ORDER BY p.observation_id, p.rank",
                    (canonical_id, as_of),
                ):
                    pools.setdefault(r[0], []).append(
                        MarketPool(
                            pair_address=r[1],
                            identity=r[2],
                            chain=r[3],
                            dex=r[4],
                            base_mint=r[5],
                            base_symbol=r[6],
                            quote_address=r[7],
                            quote_symbol=r[8],
                            quote_kind=r[9],
                            liquidity_usd=r[10],
                            volume_24h=r[11],
                            txns_24h=r[12],
                            price_usd=r[13],
                            pair_created_at=r[14],
                            eligible=bool(r[15]),
                            rejections=tuple(json.loads(r[16])),
                        )  # fmt: skip
                    )
        return [ObservedMarket(i, c, f, o, why, tuple(pools.get(i, ())), prov)
                for i, c, f, o, why, prov in rows]  # fmt: skip

    def record_pool_account(
        self,
        canonical_id: str,
        collection_id: int,
        pool_address: str,
        fetched_at: float,
        provider: str,
        obs: PoolAccountObservation,
    ) -> int:
        with self._lock, self.db() as conn:
            cur = conn.execute(
                "INSERT INTO safety_pool_account_observations (canonical_id, collection_id, "
                "pool_address, fetched_at, provider, outcome, program_owner, lamports, raw_hash, "
                "reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (canonical_id, collection_id, pool_address, fetched_at, provider, obs.outcome,
                 obs.program_owner, obs.lamports, obs.raw_hash, obs.reason),
            )  # fmt: skip
        assert cur.lastrowid is not None
        return cur.lastrowid

    def pool_accounts(self, canonical_id: str, as_of: float) -> list[AccountCheck]:
        with self._lock:
            rows = (
                self.db()
                .execute(
                    "SELECT id, collection_id, pool_address, fetched_at, outcome, program_owner, "
                    "reason FROM safety_pool_account_observations WHERE canonical_id = ? AND "
                    "fetched_at <= ? ORDER BY fetched_at, id",
                    (canonical_id, as_of),
                )
                .fetchall()
            )
        return [AccountCheck(*r) for r in rows]

    # --- captured Radar evidence (Phase 4A) ------------------------------------------------

    def record_radar_capture(
        self, canonical_id: str, collection_id: int, captured_at: float, cap: RadarCapture
    ) -> tuple[int, int, int]:
        """The capture run and every fact, in **one** transaction (a failure leaves
        nothing). Facts already captured for this target (same source key) are skipped.
        (capture id, new facts, repeated facts)."""
        base = "canonical_id, capture_id, collection_id, captured_at, source_system, " \
            "radar_schema_version, source_key, source_time"  # fmt: skip
        tables: list[tuple[str, str, list[tuple[str, float, tuple[Any, ...]]]]] = [
            ("safety_radar_wallet_proofs", "wallet, kind, radar_canonical_id, signature",
             [(p.source_key, p.source_time, (p.wallet, p.kind, p.radar_canonical_id,
                                             p.signature)) for p in cap.proofs]),
            ("safety_radar_creator_evidence",
             "role, status, address, method, signature, block_time, radar_provider, "
             "provenance_json",
             [(c.source_key, c.source_time,
               (c.role, c.status, c.address, c.method, c.signature, c.block_time, c.provider,
                json.dumps(c.provenance, sort_keys=True, default=str))) for c in cap.creators]),
            ("safety_radar_flow_evidence",
             "wallet, signature, direction, amount_raw, block_time, signer, participant, "
             "counterparty, radar_scan_id",
             [(f.source_key, f.source_time,
               (f.wallet, f.signature, f.direction, f.amount_raw, f.block_time, int(f.signer),
                f.participant, f.counterparty, f.radar_scan_id)) for f in cap.flows]),
            ("safety_radar_activity_coverage", "kind, radar_id, status, facts_json",
             [(v.source_key, v.source_time,
               (v.kind, v.radar_id, v.status, json.dumps(v.facts, sort_keys=True, default=str)))
              for v in cap.coverage]),
        ]  # fmt: skip
        with self._lock, self.db() as conn:
            # Count first, so the capture row is written once and never updated.
            fresh: list[tuple[str, str, list[tuple[str, float, tuple[Any, ...]]]]] = []
            new = repeated = 0
            for table, columns, rows in tables:
                seen = {r[0] for r in conn.execute(
                    f"SELECT source_key FROM {table} WHERE canonical_id = ?", (canonical_id,)
                )}  # fmt: skip
                keep = []
                for row in rows:
                    if row[0] in seen:
                        repeated += 1
                    else:
                        seen.add(row[0])
                        keep.append(row)
                new += len(keep)
                fresh.append((table, columns, keep))
            cur = conn.execute(
                "INSERT INTO safety_radar_captures (canonical_id, collection_id, captured_at, "
                "fetched_at, source_system, status, reason, radar_schema_version, new_facts, "
                "repeated_facts) VALUES (?, ?, ?, ?, 'radar', ?, ?, ?, ?, ?)",
                (canonical_id, collection_id, captured_at, captured_at, cap.status, cap.reason,
                 cap.radar_schema_version, new, repeated),
            )  # fmt: skip
            cid = cur.lastrowid
            assert cid is not None
            common = (canonical_id, cid, collection_id, captured_at, "radar",
                      cap.radar_schema_version)  # fmt: skip
            for table, columns, rows in fresh:
                n = len(columns.split(","))
                conn.executemany(
                    f"INSERT INTO {table} ({base}, {columns}) VALUES ({','.join('?' * (8 + n))})",
                    [(*common, key, at, *values) for key, at, values in rows],
                )
        return cid, new, repeated

    def latest_radar_capture(
        self, canonical_id: str, as_of: float
    ) -> tuple[int, str, str | None, float] | None:
        """(id, status, reason, captured_at) of the newest capture attempt at or before
        `as_of`."""
        with self._lock:
            row = (
                self.db()
                .execute(
                    "SELECT id, status, reason, captured_at FROM safety_radar_captures "
                    "WHERE canonical_id = ? "
                    "AND captured_at <= ? ORDER BY captured_at DESC, id DESC LIMIT 1",
                    (canonical_id, as_of),
                )
                .fetchone()
            )
        return (row[0], row[1], row[2], row[3]) if row else None

    def captured_wallet_proofs(
        self, canonical_id: str, wallets: list[str], as_of: float
    ) -> dict[str, WalletProof]:
        """Per wallet, the earliest captured positive proof usable at `as_of`
        (``source_time <= as_of`` and ``captured_at <= as_of``)."""
        out: dict[str, WalletProof] = {}
        with self._lock:
            conn = self.db()
            for i in range(0, len(wallets), 500):
                chunk = wallets[i : i + 500]
                marks = ",".join("?" * len(chunk))
                for w, at, sig, rcid, cap_at, rid, key in conn.execute(
                    "SELECT wallet, source_time, signature, radar_canonical_id, captured_at, id, "
                    "source_key FROM safety_radar_wallet_proofs WHERE canonical_id = ? AND "
                    f"source_time <= ? AND captured_at <= ? AND wallet IN ({marks}) "
                    "ORDER BY wallet, source_time, id",
                    (canonical_id, as_of, as_of, *chunk),
                ):
                    if w not in out:
                        out[w] = WalletProof(w, at, sig, rcid, cap_at, rid, key)
        return out

    def captured_facts(self, table: str, canonical_id: str, as_of: float) -> list[dict[str, Any]]:
        """Captured rows of a Radar-fact table usable at `as_of`, in capture order."""
        if table not in ("safety_radar_creator_evidence", "safety_radar_flow_evidence",
                         "safety_radar_activity_coverage", "safety_radar_wallet_proofs"):  # fmt: skip
            raise ValueError(table)
        with self._lock:
            cur = self.db().execute(
                f"SELECT * FROM {table} WHERE canonical_id = ? AND source_time <= ? AND "
                "captured_at <= ? ORDER BY source_time, id",
                (canonical_id, as_of, as_of),
            )
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

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
