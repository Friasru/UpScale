"""Radar-local retention (Radar's own database only; the production retention engine is
neither used nor changed).

| Table                                         | Kept         |
|-----------------------------------------------|--------------|
| radar_tx, radar_wallet_flows, radar_scans     | 7 days       |
| radar_holder_balances                         | 7 days       |
| radar_snapshots, radar_holder_snapshots       | 30 days      |
| radar_requests                                | 90 days      |
| radar_targets, radar_wallet_entries, radar_wallets, radar_creators, radar_meta | forever |

Protections (never deleted, whatever their age):

* wallet / unknown-participant flows whose wallet-entry roll-up doesn't exist (the
  roll-up must survive them; programs, pools and routers have no roll-up);
* the latest holder snapshot (and its balances) and the latest Radar snapshot of every
  ACTIVE target: the baselines the next snapshot compares with;
* the activity-cursor transaction and the early-history scan of every ACTIVE target.

`plan` is read-only (the dry run); `cleanup` deletes in one transaction per table.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from upscale.services.radar.config import RadarSettings
from upscale.services.radar.models import iso, ts

_LATEST_HOLDER = (
    "SELECT MAX(h.id) FROM radar_holder_snapshots h JOIN radar_targets t "
    "ON t.canonical_id = h.canonical_id AND t.status = 'ACTIVE' GROUP BY h.canonical_id"
)
_LATEST_SNAPSHOT = (
    "SELECT MAX(s.id) FROM radar_snapshots s JOIN radar_targets t "
    "ON t.canonical_id = s.canonical_id AND t.status = 'ACTIVE' GROUP BY s.canonical_id"
)


@dataclass(frozen=True)
class Policy:
    table: str
    days: int
    time_col: str
    protect: dict[str, str]  # reason -> SQL condition on the table's rows (alias r)


def policies(s: RadarSettings) -> list[Policy]:
    return [
        Policy("radar_wallet_flows", s.tx_retention_days, "r.fetched_at", {
            "wallet-entry roll-up missing": "r.participant IN ('NORMAL_WALLET', 'UNKNOWN') AND "
            "NOT EXISTS (SELECT 1 FROM radar_wallet_entries e WHERE e.wallet = r.wallet "
            "AND e.canonical_id = r.canonical_id)",
        }),
        Policy("radar_tx", s.tx_retention_days, "r.fetched_at", {
            "activity cursor of an active target": "EXISTS (SELECT 1 FROM radar_targets t "
            "WHERE t.status = 'ACTIVE' AND t.canonical_id = r.canonical_id "
            "AND t.last_signature = r.signature)",
        }),
        Policy("radar_scans", s.tx_retention_days, "r.fetched_at", {
            "early-history scan of an active target": "r.kind = 'early' AND EXISTS (SELECT 1 "
            "FROM radar_targets t WHERE t.status = 'ACTIVE' AND t.canonical_id = r.canonical_id)",
        }),
        Policy("radar_holder_balances", s.holder_balance_retention_days,
               "(SELECT h.observed_at FROM radar_holder_snapshots h WHERE h.id = r.snapshot_id)",
               {"baseline of an active target": f"r.snapshot_id IN ({_LATEST_HOLDER})"}),
        Policy("radar_holder_snapshots", s.snapshot_retention_days, "r.observed_at",
               {"baseline of an active target": f"r.id IN ({_LATEST_HOLDER})"}),
        Policy("radar_snapshots", s.snapshot_retention_days, "r.observed_at",
               {"latest snapshot of an active target": f"r.id IN ({_LATEST_SNAPSHOT})"}),
        Policy("radar_requests", s.request_retention_days, "r.day", {}),
    ]  # fmt: skip


def _cutoff(p: Policy, now: datetime) -> tuple[Any, str]:
    when = now - timedelta(days=p.days)
    if p.table == "radar_requests":
        day = when.strftime("%Y-%m-%d")
        return day, day
    return ts(when), iso(ts(when)) or ""


def _where(p: Policy) -> str:
    keep = " OR ".join(f"({c})" for c in p.protect.values())
    return f"{p.time_col} < ?" + (f" AND NOT ({keep})" if keep else "")


def plan(conn: sqlite3.Connection, settings: RadarSettings, now: datetime) -> list[dict[str, Any]]:
    """Read-only: per table, rows older than the cutoff, how many are eligible, and how
    many each protection keeps."""
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    out: list[dict[str, Any]] = []
    for p in policies(settings):
        if p.table not in existing:
            out.append({"table": p.table, "skipped": "table not found"})
            continue
        cut, label = _cutoff(p, now)
        total = conn.execute(f"SELECT COUNT(*) FROM {p.table}").fetchone()[0]
        older = conn.execute(
            f"SELECT COUNT(*) FROM {p.table} r WHERE {p.time_col} < ?", (cut,)
        ).fetchone()[0]
        eligible = conn.execute(
            f"SELECT COUNT(*) FROM {p.table} r WHERE {_where(p)}", (cut,)
        ).fetchone()[0]
        protected = {
            reason: conn.execute(
                f"SELECT COUNT(*) FROM {p.table} r WHERE {p.time_col} < ? AND ({cond})", (cut,)
            ).fetchone()[0]
            for reason, cond in p.protect.items()
        }
        out.append({"table": p.table, "retention_days": p.days, "cutoff": label,
                    "total_rows": total, "older_than_cutoff": older, "eligible": eligible,
                    "protected": protected})  # fmt: skip
    return out


def cleanup(
    conn: sqlite3.Connection, settings: RadarSettings, now: datetime
) -> list[dict[str, Any]]:
    """Delete eligible rows. Holder balances go before their holder snapshots."""
    out = plan(conn, settings, now)
    for p, row in zip(policies(settings), out, strict=True):
        if row.get("skipped"):
            continue
        cut, _ = _cutoff(p, now)
        with conn:
            if p.table == "radar_holder_snapshots":
                conn.execute(
                    f"DELETE FROM radar_holder_balances WHERE snapshot_id IN "
                    f"(SELECT r.id FROM radar_holder_snapshots r WHERE {_where(p)})",
                    (cut,),
                )
            cur = conn.execute(
                f"DELETE FROM {p.table} WHERE rowid IN "
                f"(SELECT r.rowid FROM {p.table} r WHERE {_where(p)})",
                (cut,),
            )
        row["deleted"] = cur.rowcount
    return out
