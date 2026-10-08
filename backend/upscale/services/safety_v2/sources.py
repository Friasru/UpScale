"""Read-only Radar evidence for Safety V2, captured durably into Safety's own database.

Radar's code is never imported: Safety V2 depends only on Radar's *stored* schema, gated on
``radar_meta.schema_version = '3'`` and the expected columns. The file is opened
``mode=ro`` with ``PRAGMA query_only`` (never created, never written; it stays
byte-identical).

Radar is read **only during a Safety collection** (`read_radar_capture`), and what is read
is appended to Safety's own tables. Snapshots and rebuilds never read Radar: Radar
overwrites creator determinations (``INSERT OR REPLACE``) and its retention deletes flows
and scans after days, so live Radar rows can't reproduce a past snapshot.

Facts captured for one target (each with a stable Radar source key and Radar's knowledge
time; Safety adds ``captured_at``):

* **Wallet proof** (positive only), for owners in the target's latest holder observation:
  the earliest ``radar_wallet_flows`` row with ``signer = 1`` and ``participant =
  'NORMAL_WALLET'`` (a signer is a keypair: programs and PDAs can't sign), and the
  ``radar_wallet_entries.wallet_evidence_at`` roll-up (kept forever by Radar).
* **Creator / deployer**: every ``radar_creators`` row of the exact target, roles kept
  apart (``POOL_CREATOR_CANDIDATE`` / ``TOKEN_DEPLOYER``); never upgraded.
* **Flows**: ``TOKEN_OUTFLOW`` rows of the exact target token whose wallet is one of the
  captured creator-role addresses. A TOKEN_OUTFLOW is a balance decrease, never a sale.
* **Activity coverage**: the target's finished ``radar_scans`` rows and every observed
  state of its ``radar_activity_gaps`` rows (listing continuity, open gaps, parse caps).

Not configured, missing, unreadable or incompatible: no fact is captured and the status
says why; nothing fails.
"""

import json
import math
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RADAR_SCHEMA_VERSION = "3"
_FLOW_COLUMNS = {"wallet", "signer", "participant", "fetched_at", "signature", "canonical_id",
                 "direction", "amount_raw", "block_time", "counterparty", "scan_id"}  # fmt: skip
_COLUMNS = {
    "radar_wallet_flows": _FLOW_COLUMNS,
    "radar_wallet_entries": {"wallet", "canonical_id", "wallet_evidence_at"},
    "radar_creators": {"canonical_id", "role", "status", "identity", "method", "signature",
                       "block_time", "determined_at", "provider", "provenance_json"},
    "radar_scans": {"id", "canonical_id", "kind", "started_at", "fetched_at", "status"},
    "radar_activity_gaps": {"id", "canonical_id", "status", "opened_at", "updated_at"},
}  # fmt: skip
_CHUNK = 500  # SQLite host parameters per query

ProofStatus = Literal["AVAILABLE", "NOT_CONFIGURED", "NOT_CONSULTED", "UNAVAILABLE",
                      "INCOMPATIBLE"]  # fmt: skip
CaptureStatus = Literal["CAPTURED", "NOT_CONFIGURED", "UNAVAILABLE", "INCOMPATIBLE"]


@dataclass(frozen=True)
class WalletProof:
    wallet: str
    fetched_at: float  # when Radar learned it (source knowledge time)
    signature: str | None  # None for a wallet-entry roll-up proof
    canonical_id: str  # the Radar target whose evidence carried the proof
    captured_at: float | None = None  # when Safety captured it (None: read live, tests)
    capture_row_id: int | None = None
    source_key: str | None = None


@dataclass(frozen=True)
class WalletProofs:
    status: ProofStatus
    reason: str | None = None
    proofs: dict[str, WalletProof] = field(default_factory=dict)
    capture_id: int | None = None  # the Safety capture run the status comes from


NOT_CONSULTED = WalletProofs("NOT_CONSULTED", "no owner needed wallet proof")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _open(path: str | None) -> tuple[sqlite3.Connection | None, CaptureStatus, str | None]:
    """A compatible read-only Radar connection, or why there is none."""
    if path is None:
        return None, "NOT_CONFIGURED", "no Radar database configured (UPSCALE_SAFETY_V2_RADAR_DB)"
    p = Path(path).expanduser()
    if not p.is_file():
        return None, "UNAVAILABLE", "the configured Radar database doesn't exist"
    try:
        conn = _connect(p)
    except sqlite3.Error as exc:
        return None, "UNAVAILABLE", f"the Radar database can't be opened ({exc})"
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"radar_meta", "radar_wallet_flows"} <= tables:
            conn.close()
            return None, "INCOMPATIBLE", "not a Radar database (radar tables missing)"
        row = conn.execute("SELECT value FROM radar_meta WHERE key = 'schema_version'").fetchone()
        found = row[0] if row else None
        if found != RADAR_SCHEMA_VERSION:
            conn.close()
            return (
                None,
                "INCOMPATIBLE",
                (f"Radar schema version {found or 'unknown'}, not {RADAR_SCHEMA_VERSION}"),
            )
        for table, needed in _COLUMNS.items():
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if not needed <= columns:
                conn.close()
                return None, "INCOMPATIBLE", f"{table} lacks expected columns"
    except sqlite3.Error as exc:
        conn.close()
        return None, "UNAVAILABLE", f"the Radar database can't be read ({exc})"
    return conn, "CAPTURED", None


def _signer_proofs(
    conn: sqlite3.Connection, wallets: list[str], as_of: float
) -> dict[str, WalletProof]:
    proofs: dict[str, WalletProof] = {}
    for i in range(0, len(wallets), _CHUNK):
        chunk = wallets[i : i + _CHUNK]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            "SELECT wallet, fetched_at, signature, canonical_id FROM radar_wallet_flows "
            f"WHERE signer = 1 AND participant = 'NORMAL_WALLET' AND fetched_at <= ? "
            f"AND wallet IN ({marks}) ORDER BY wallet, fetched_at, signature, canonical_id",
            (as_of, *chunk),
        ).fetchall()
        for wallet, fetched_at, signature, cid in rows:
            if wallet not in proofs:
                proofs[wallet] = WalletProof(wallet, float(fetched_at), signature, cid)
    return proofs


def radar_wallet_proofs(path: str | None, wallets: Iterable[str], as_of: float) -> WalletProofs:
    """The earliest signer-flow proof per wallet that Radar knew at or before `as_of`, read
    live (used by capture; snapshots read Safety's captured copies instead)."""
    wanted = sorted(set(wallets))
    if not wanted:
        return NOT_CONSULTED
    conn, status, reason = _open(path)
    if conn is None:
        return WalletProofs("UNAVAILABLE" if status == "CAPTURED" else status, reason)
    try:
        return WalletProofs("AVAILABLE", None, _signer_proofs(conn, wanted, as_of))
    except sqlite3.Error as exc:
        return WalletProofs("UNAVAILABLE", f"the Radar database can't be read ({exc})")
    finally:
        conn.close()


# --- capture ---------------------------------------------------------------------------------


def _key(*parts: Any) -> str:
    """A stable source key (canonical JSON of the identifying fields)."""
    return json.dumps(list(parts), separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CapturedProof:
    wallet: str
    kind: Literal["SIGNER_FLOW", "WALLET_ENTRY"]
    source_key: str
    source_time: float
    radar_canonical_id: str
    signature: str | None


@dataclass(frozen=True)
class CapturedCreator:
    role: str
    status: str
    address: str | None
    method: str
    signature: str | None
    block_time: float | None
    source_key: str
    source_time: float  # determined_at
    provider: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CapturedFlow:
    wallet: str
    signature: str
    direction: str
    amount_raw: str
    block_time: float | None
    source_key: str
    source_time: float  # Radar fetched_at
    signer: bool
    participant: str
    counterparty: str
    radar_scan_id: int | None


@dataclass(frozen=True)
class CapturedCoverage:
    kind: Literal["SCAN", "GAP"]
    radar_id: int
    status: str
    source_key: str
    source_time: float  # scan fetched_at / gap updated_at
    facts: dict[str, Any]  # the whole source row, to reproduce coverage later


@dataclass(frozen=True)
class RadarCapture:
    status: CaptureStatus
    reason: str | None
    radar_schema_version: str | None = None
    proofs: tuple[CapturedProof, ...] = ()
    creators: tuple[CapturedCreator, ...] = ()
    flows: tuple[CapturedFlow, ...] = ()
    coverage: tuple[CapturedCoverage, ...] = ()


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _rows(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
    cur = conn.execute(sql, args)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


def read_radar_capture(path: str | None, canonical_id: str, wallets: Iterable[str]) -> RadarCapture:
    """Everything Safety captures for `canonical_id`, read in one read-only pass."""
    conn, status, reason = _open(path)
    if conn is None:
        return RadarCapture(status, reason)
    wanted = sorted(set(wallets))
    try:
        proofs: list[CapturedProof] = []
        for p in _signer_proofs(conn, wanted, math.inf).values():
            proofs.append(CapturedProof(p.wallet, "SIGNER_FLOW",
                                        _key("flow", p.canonical_id, p.signature, p.wallet),
                                        p.fetched_at, p.canonical_id, p.signature))  # fmt: skip
        for i in range(0, len(wanted), _CHUNK):
            chunk = wanted[i : i + _CHUNK]
            marks = ",".join("?" * len(chunk))
            for wallet, cid, at in conn.execute(
                "SELECT wallet, canonical_id, wallet_evidence_at FROM radar_wallet_entries "
                f"WHERE wallet_evidence_at IS NOT NULL AND wallet IN ({marks}) "
                "ORDER BY wallet, wallet_evidence_at, canonical_id",
                chunk,
            ):
                if (t := _finite(at)) is not None:
                    proofs.append(CapturedProof(wallet, "WALLET_ENTRY",
                                                _key("entry", cid, wallet, t), t, cid, None))  # fmt: skip

        creators: list[CapturedCreator] = []
        for r in _rows(conn, "SELECT role, status, identity, method, signature, block_time, "
                       "determined_at, provider, provenance_json FROM radar_creators "
                       "WHERE canonical_id = ? ORDER BY role", (canonical_id,)):  # fmt: skip
            at = _finite(r["determined_at"])
            if at is None or r["role"] not in ("POOL_CREATOR_CANDIDATE", "TOKEN_DEPLOYER"):
                continue
            try:
                provenance = json.loads(r["provenance_json"])
            except (TypeError, ValueError):
                provenance = {"unparsed": str(r["provenance_json"])}
            creators.append(CapturedCreator(
                role=r["role"], status=r["status"], address=r["identity"], method=r["method"],
                signature=r["signature"], block_time=_finite(r["block_time"]),
                source_key=_key("creator", canonical_id, r["role"], r["status"], r["identity"],
                                r["signature"], at),
                source_time=at, provider=r["provider"],
                provenance=provenance if isinstance(provenance, dict) else {"value": provenance},
            ))  # fmt: skip

        flows: list[CapturedFlow] = []
        identities = sorted({c.address for c in creators if c.address})
        if identities:
            marks = ",".join("?" * len(identities))
            for r in _rows(conn, "SELECT wallet, signature, direction, amount_raw, block_time, "
                           "fetched_at, signer, participant, counterparty, scan_id FROM "
                           "radar_wallet_flows WHERE canonical_id = ? AND direction = "
                           f"'TOKEN_OUTFLOW' AND wallet IN ({marks}) ORDER BY fetched_at, "
                           "signature, wallet", (canonical_id, *identities)):  # fmt: skip
                at = _finite(r["fetched_at"])
                if at is None:
                    continue
                flows.append(CapturedFlow(
                    wallet=r["wallet"], signature=r["signature"], direction=r["direction"],
                    amount_raw=str(r["amount_raw"]), block_time=_finite(r["block_time"]),
                    source_key=_key("flow", canonical_id, r["signature"], r["wallet"]),
                    source_time=at, signer=bool(r["signer"]), participant=r["participant"],
                    counterparty=r["counterparty"], radar_scan_id=r["scan_id"],
                ))  # fmt: skip

        coverage: list[CapturedCoverage] = []
        for r in _rows(conn, "SELECT * FROM radar_scans WHERE canonical_id = ? AND status != "
                       "'RUNNING' ORDER BY id", (canonical_id,)):  # fmt: skip
            at = _finite(r["fetched_at"])
            if at is not None:
                coverage.append(CapturedCoverage(
                    "SCAN", r["id"], r["status"],
                    _key("scan", canonical_id, r["id"], r["started_at"], at, r["status"]), at, r,
                ))  # fmt: skip
        for r in _rows(conn, "SELECT * FROM radar_activity_gaps WHERE canonical_id = ? "
                       "ORDER BY id", (canonical_id,)):  # fmt: skip
            at = _finite(r["updated_at"])
            if at is not None:
                coverage.append(CapturedCoverage(
                    "GAP", r["id"], r["status"],
                    _key("gap", canonical_id, r["id"], r["opened_at"], at, r["status"],
                         r.get("before_signature")), at, r,
                ))  # fmt: skip
    except sqlite3.Error as exc:
        return RadarCapture("UNAVAILABLE", f"the Radar database can't be read ({exc})")
    finally:
        conn.close()
    return RadarCapture("CAPTURED", None, RADAR_SCHEMA_VERSION, tuple(proofs), tuple(creators),
                        tuple(flows), tuple(coverage))  # fmt: skip
