"""Read-only, batched access to the existing databases (Scout, Evidence Archive, Shadow).

Every connection is opened ``mode=ro`` with ``PRAGMA query_only``: Audit can't write,
create a table or change a checkpoint even by mistake. A missing database is reported as
missing (None), never created. No provider, network or Calibration access exists here.

Queries are batched (``IN (...)`` chunks on indexed columns) so the cost grows with the
number of decisions analyzed, never one query per decision.
"""

import json
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upscale.services.evidence_archive.store import _COLUMNS as EVIDENCE_COLUMNS
from upscale.services.evidence_archive.store import EvidenceRecord
from upscale.services.evidence_archive.store import _record as evidence_record
from upscale.services.outcomes.audit import latest_audits

# Every table Audit may read (reported by `status`, asserted by the tests).
SCOUT_TABLES = ("scout_outcome_observations", "scout_outcome_horizons", "outcome_integrity_audits")
EVIDENCE_TABLES = ("evidence_records",)
SHADOW_TABLES = (
    "shadow_runs", "shadow_decisions", "shadow_positions", "shadow_trades", "shadow_executions",
)  # fmt: skip


class AuditSourceError(Exception):
    pass


def connect(path: str | Path | None) -> sqlite3.Connection | None:
    """A read-only connection, or None when the file doesn't exist (never created)."""
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    conn = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def chunks(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _marks(n: int) -> str:
    return ",".join("?" * n)


# --- Scout database: anchors and their horizons -----------------------------------------------


@dataclass(frozen=True)
class AnchorRow:
    id: int
    canonical_id: str
    chain: str
    address: str
    pool_address: str
    observed_at: float  # the market observation (the outcome window's start)
    anchored_at: float  # the ranking run's start
    run_id: str
    anchor_reason: str
    rank: int
    stage: str
    score: float
    discovery_status: str


@dataclass(frozen=True)
class HorizonRow:
    observation_id: int
    horizon: str
    status: str
    market_status: str | None
    return_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None
    liquidity_change_pct: float | None
    future_stage: str | None
    finalized_at: float | None


def _anchor(r: Sequence[Any]) -> AnchorRow:
    return AnchorRow(
        id=r[0], canonical_id=r[1], chain=r[2], address=r[3], pool_address=r[4],
        observed_at=r[5], anchored_at=r[6], run_id=r[7], anchor_reason=r[8], rank=r[9],
        stage=r[10], score=r[11], discovery_status=r[12],
    )  # fmt: skip


def anchors(conn: sqlite3.Connection, lo: float | None, hi: float | None) -> list[AnchorRow]:
    """Scout anchors whose ranking run started in [lo, hi) (either bound optional)."""
    if "scout_outcome_observations" not in tables(conn):
        return []
    rows = conn.execute(
        "SELECT id, canonical_id, chain, address, pool_address, observed_at, anchored_at, run_id, "
        "anchor_reason, rank, stage, score, discovery_status "
        "FROM scout_outcome_observations WHERE anchored_at >= ? AND anchored_at < ? ORDER BY id",
        (lo if lo is not None else -1.0, hi if hi is not None else 1e13),
    ).fetchall()
    return [_anchor(r) for r in rows]


def anchor_bodies(
    conn: sqlite3.Connection, ids: Sequence[int], batch: int
) -> dict[int, dict[str, Any]]:
    """The stored `ScoutObservation` bodies of `ids` (read only when needed: they're large)."""
    out: dict[int, dict[str, Any]] = {}
    for chunk in chunks(sorted(set(ids)), batch):
        for oid, body in conn.execute(
            "SELECT id, body_json FROM scout_outcome_observations "
            f"WHERE id IN ({_marks(len(chunk))})",
            chunk,
        ):
            out[oid] = json.loads(body)
    return out


def horizons(conn: sqlite3.Connection, ids: Sequence[int], batch: int) -> list[HorizonRow]:
    if not ids or "scout_outcome_horizons" not in tables(conn):
        return []
    out: list[HorizonRow] = []
    for chunk in chunks(sorted(ids), batch):
        out += [
            HorizonRow(*r)
            for r in conn.execute(
                "SELECT observation_id, horizon, status, market_status, return_pct, mfe_pct, "
                "mae_pct, liquidity_change_pct, future_stage, finalized_at "
                f"FROM scout_outcome_horizons WHERE observation_id IN ({_marks(len(chunk))})",
                chunk,
            )
        ]
    return out


def integrity(conn: sqlite3.Connection) -> dict[tuple[str, int, str], tuple[str, str]]:
    """(kind, observation id, horizon) -> the latest integrity audit (status, reason)."""
    return latest_audits(conn)


def anchor_by_id(conn: sqlite3.Connection, oid: int) -> AnchorRow | None:
    if "scout_outcome_observations" not in tables(conn):
        return None
    r = conn.execute(
        "SELECT id, canonical_id, chain, address, pool_address, observed_at, anchored_at, run_id, "
        "anchor_reason, rank, stage, score, discovery_status "
        "FROM scout_outcome_observations WHERE id = ?",
        (oid,),
    ).fetchone()
    return _anchor(r) if r else None


def anchors_for(
    conn: sqlite3.Connection, keys: Sequence[tuple[str, float]], batch: int
) -> dict[tuple[str, float], AnchorRow]:
    """Anchors of exact (canonical id, market observation time) pairs."""
    if not keys or "scout_outcome_observations" not in tables(conn):
        return {}
    wanted = {(a, round(t, 3)) for a, t in keys}
    out: dict[tuple[str, float], AnchorRow] = {}
    assets = sorted({a for a, _ in keys})
    lo, hi = min(t for _, t in keys) - 1, max(t for _, t in keys) + 1
    for chunk in chunks(assets, batch):
        for r in conn.execute(
            "SELECT id, canonical_id, chain, address, pool_address, observed_at, anchored_at, "
            "run_id, anchor_reason, rank, stage, score, discovery_status "
            f"FROM scout_outcome_observations WHERE canonical_id IN ({_marks(len(chunk))}) "
            "AND observed_at >= ? AND observed_at <= ?",
            [*chunk, lo, hi],
        ):
            key = (r[1], round(r[5], 3))
            if key in wanted:
                out[key] = _anchor(r)
    return out


# --- Evidence Archive ----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceIndex:
    """An evidence row without its payload (cheap to scan)."""

    id: int
    asset_id: str
    observed_at: float
    provider_at: float | None
    started_at: str | None  # a Scout record's ranking run start (its links), else None


def evidence_index(
    conn: sqlite3.Connection,
    kind: str,
    assets: Sequence[str],
    lo: float,
    hi: float,
    batch: int,
) -> list[EvidenceIndex]:
    """Index rows of `kind` for `assets` observed in [lo, hi] (the (kind, asset_id,
    observed_at) index), without payloads."""
    if not assets or "evidence_records" not in tables(conn):
        return []
    out: list[EvidenceIndex] = []
    for chunk in chunks(sorted(set(assets)), batch):
        for r in conn.execute(
            "SELECT id, asset_id, observed_at, provider_at, "
            "json_extract(links_json, '$.evaluation_started_at') FROM evidence_records "
            f"WHERE kind = ? AND asset_id IN ({_marks(len(chunk))}) "
            "AND observed_at >= ? AND observed_at <= ?",
            [kind, *chunk, lo, hi],
        ):
            out.append(EvidenceIndex(*r))
    return out


def evidence_by_ids(
    conn: sqlite3.Connection, ids: Sequence[int], batch: int
) -> dict[int, EvidenceRecord]:
    """Full records (payload decompressed and hash-checked) by row id."""
    out: dict[int, EvidenceRecord] = {}
    if not ids:
        return out
    for chunk in chunks(sorted(set(ids)), batch):
        for r in conn.execute(
            f"SELECT {EVIDENCE_COLUMNS} FROM evidence_records WHERE id IN ({_marks(len(chunk))})",
            chunk,
        ):
            rec = evidence_record(r)
            out[rec.id] = rec
    return out


def evidence_by_record_ids(
    conn: sqlite3.Connection, record_ids: Sequence[str], batch: int
) -> dict[str, EvidenceRecord]:
    out: dict[str, EvidenceRecord] = {}
    if not record_ids or "evidence_records" not in tables(conn):
        return out
    for chunk in chunks(sorted(set(record_ids)), batch):
        for r in conn.execute(
            f"SELECT {EVIDENCE_COLUMNS} FROM evidence_records "
            f"WHERE record_id IN ({_marks(len(chunk))})",
            chunk,
        ):
            rec = evidence_record(r)
            out[rec.record_id] = rec
    return out


# --- Shadow database -----------------------------------------------------------------------------


def _dicts(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


def shadow_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if "shadow_runs" not in tables(conn):
        return []
    return _dicts(
        conn.execute(
            "SELECT run_id, created_at, since, until, clean_data, execution_model, "
            "strategies_json, args_json FROM shadow_runs ORDER BY run_id"
        )
    )


def shadow_entries(
    conn: sqlite3.Connection,
    run_ids: Sequence[str],
    strategy: str | None,
    lo: float | None,
    hi: float | None,
) -> list[dict[str, Any]]:
    """ENTER decisions of `run_ids` (the (run, strategy, decision_at) index)."""
    if not run_ids or "shadow_decisions" not in tables(conn):
        return []
    out: list[dict[str, Any]] = []
    for run_id in sorted(run_ids):
        where = ["run_id = ?", "action = 'ENTER'", "decision_at >= ?", "decision_at < ?"]
        params: list[Any] = [run_id, lo if lo is not None else -1.0, hi if hi is not None else 1e13]
        if strategy is not None:
            where.append("strategy_id = ?")
            params.append(strategy)
        out += _dicts(
            conn.execute(
                "SELECT decision_id, run_id, strategy_id, strategy_version, asset_id, chain, "
                "address, pool, decision_at, reference_price, reference_price_at, reason, "
                "evidence_json, fingerprints_json, position_id FROM shadow_decisions "
                f"WHERE {' AND '.join(where)} ORDER BY decision_at, id",
                params,
            )
        )
    return out


def shadow_rows(
    conn: sqlite3.Connection, table: str, run_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Every row of a per-run Shadow table for `run_ids` (positions, trades, executions)."""
    assert table in ("shadow_positions", "shadow_trades", "shadow_executions")
    if not run_ids or table not in tables(conn):
        return []
    out: list[dict[str, Any]] = []
    for run_id in sorted(run_ids):
        out += _dicts(
            conn.execute(f"SELECT * FROM {table} WHERE run_id = ? ORDER BY rowid", (run_id,))
        )
    return out


def shadow_decision(conn: sqlite3.Connection, decision_id: str) -> dict[str, Any] | None:
    if "shadow_decisions" not in tables(conn):
        return None
    rows = _dicts(
        conn.execute(
            "SELECT decision_id, run_id, strategy_id, strategy_version, action, asset_id, chain, "
            "address, pool, decision_at, reference_price, reference_price_at, reason, "
            "evidence_json, fingerprints_json, position_id FROM shadow_decisions "
            "WHERE decision_id = ?",
            (decision_id,),
        )
    )
    return rows[0] if rows else None


def row_counts(conn: sqlite3.Connection, names: Sequence[str]) -> dict[str, int]:
    present = tables(conn)
    return {
        t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        for t in names
        if t in present
    }
