"""Read-only external evidence for Safety V2: positive wallet proof from a Radar database.

Radar's code is never imported: Safety V2 depends only on Radar's *stored* schema, gated on
``radar_meta.schema_version = '3'``. The file is opened ``mode=ro`` with ``PRAGMA
query_only`` (never created, never written; the file stays byte-identical).

The only thing read is signer evidence: a ``radar_wallet_flows`` row with ``signer = 1``
and ``participant = 'NORMAL_WALLET'`` means the owner signed a transaction, so it is a
keypair (programs and PDAs can't sign). Every row is filtered by Radar's ``fetched_at <=
as_of`` (when Radar learned it): a proof learned after ``as_of`` doesn't exist for that
snapshot. Repeated-wallet interpretation, creators / deployers and flows are not used.

A Radar database that isn't configured, is missing, unreadable or of another schema gives
no proof (``status`` says why): owners stay UNKNOWN, and nothing fails.
"""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RADAR_SCHEMA_VERSION = "3"
_FLOW_COLUMNS = {"wallet", "signer", "participant", "fetched_at", "signature", "canonical_id"}
_CHUNK = 500  # SQLite host parameters per query

ProofStatus = Literal["AVAILABLE", "NOT_CONFIGURED", "NOT_CONSULTED", "UNAVAILABLE",
                      "INCOMPATIBLE"]  # fmt: skip


@dataclass(frozen=True)
class WalletProof:
    wallet: str
    fetched_at: float  # when Radar learned it (knowledge time)
    signature: str
    canonical_id: str  # the Radar target whose transaction carried the signature


@dataclass(frozen=True)
class WalletProofs:
    status: ProofStatus
    reason: str | None = None
    proofs: dict[str, WalletProof] = field(default_factory=dict)


NOT_CONSULTED = WalletProofs("NOT_CONSULTED", "no owner needed wallet proof")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def radar_wallet_proofs(path: str | None, wallets: Iterable[str], as_of: float) -> WalletProofs:
    """The earliest signer proof per wallet that Radar knew at or before `as_of`."""
    wanted = sorted(set(wallets))
    if not wanted:
        return NOT_CONSULTED
    if path is None:
        return WalletProofs("NOT_CONFIGURED", "no Radar database configured "
                            "(UPSCALE_SAFETY_V2_RADAR_DB)")  # fmt: skip
    p = Path(path).expanduser()
    if not p.is_file():
        return WalletProofs("UNAVAILABLE", "the configured Radar database doesn't exist")
    try:
        conn = _connect(p)
    except sqlite3.Error as exc:
        return WalletProofs("UNAVAILABLE", f"the Radar database can't be opened ({exc})")
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"radar_meta", "radar_wallet_flows"} <= tables:
            return WalletProofs("INCOMPATIBLE", "not a Radar database (radar tables missing)")
        row = conn.execute("SELECT value FROM radar_meta WHERE key = 'schema_version'").fetchone()
        found = row[0] if row else None
        if found != RADAR_SCHEMA_VERSION:
            return WalletProofs(
                "INCOMPATIBLE",
                f"Radar schema version {found or 'unknown'}, not {RADAR_SCHEMA_VERSION}",
            )
        columns = {r[1] for r in conn.execute("PRAGMA table_info(radar_wallet_flows)")}
        if not _FLOW_COLUMNS <= columns:
            return WalletProofs("INCOMPATIBLE", "radar_wallet_flows lacks expected columns")
        proofs: dict[str, WalletProof] = {}
        for i in range(0, len(wanted), _CHUNK):
            chunk = wanted[i : i + _CHUNK]
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
        return WalletProofs("AVAILABLE", None, proofs)
    except sqlite3.Error as exc:
        return WalletProofs("UNAVAILABLE", f"the Radar database can't be read ({exc})")
    finally:
        conn.close()
