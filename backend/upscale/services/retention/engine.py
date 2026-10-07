"""Bounded retention for the Scout and Evidence Archive databases (never Shadow).

What may be deleted (each table on its own cutoff, oldest rows first, in small
transactions; everything else is never touched):

* Scout database: ``scout_snapshots``, ``scout_social_snapshots``,
  ``scout_social_momentum``, ``scout_social_checks`` (raw, high-frequency history) and
  finalized outcome history (``scout_outcome_observations`` with their
  ``scout_outcome_horizons``, always together);
* Evidence Archive: ``evidence_records``.

Each table keeps, whatever its age, the rows a production reader still needs:

* ``scout_snapshots``: each token's latest snapshot (Scout's tracked-provider fallback) and
  every snapshot of a token with a PENDING outcome horizon (the collector reads its price
  history from the observation onward); the cutoff is never inside Growth Scout's history
  lookbacks or the longest outcome horizon plus its retry window.
* ``scout_social_momentum``: each token's latest row and latest measured (not
  UNAVAILABLE) row, exactly what Growth Scout reads.
* ``scout_social_checks``: per token and provider, the latest check, the check with the
  latest ``covered_to`` and the first PROVIDER_UNAVAILABLE check (the social scheduler's
  inputs); the cutoff is never inside the momentum span social analysis reads.
* ``scout_social_snapshots``: each token's latest row per provider.
* Outcome history: an observation goes only with all of its horizons, only when every
  horizon is final (COMPLETE / PARTIAL / UNAVAILABLE: a final horizon can never change
  again) and was finalized before the cutoff, and never when an Analyze decision links it.
* ``evidence_records``: the latest record per (kind, asset); everything at or after the
  earliest unfinished Shadow run's cursor (minus a lookback); every record of an asset,
  token address or pool with an open Shadow position, a pending entry intent or a
  pending exit intent (from the positions table and every checkpoint's book state); and
  every record around each closed Shadow position's holding period (reports read it).
  Matching is case-insensitive on asset id, address and pool: an uncertain match keeps.
  Without a readable Shadow database no evidence is deleted.

Mechanics: candidates are scanned in rowid order below the highest rowid older than the
cutoff, protections are re-checked inside each delete transaction (``BEGIN IMMEDIATE``),
and the append-only delete triggers are dropped and recreated inside that same
transaction (other connections never see them missing; a failure rolls both back). No
VACUUM: freed pages stay in the freelist and are reused by later writes. A WAL checkpoint
(PASSIVE: it never waits for readers) runs after a cleanup.
"""

import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from upscale.services.evidence_archive.store import SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION
from upscale.services.outcomes.config import load_outcome_config
from upscale.services.outcomes.models import FINAL_STATUSES, OUTCOME_SCHEMA_VERSION
from upscale.services.retention.config import (
    OUTCOME_FLOOR_DAYS,
    RAW_FLOOR_DAYS,
    RetentionSettings,
)
from upscale.services.scout.growth.config import load_growth_config
from upscale.services.scout.models import WINDOW_MINUTES
from upscale.services.scout.social.config import load_social_config
from upscale.services.scout.social.store import SOCIAL_SCHEMA_VERSION
from upscale.services.scout.store import SCHEMA_VERSION as SCOUT_SCHEMA_VERSION
from upscale.services.shadow.store import SCHEMA_VERSION as SHADOW_SCHEMA_VERSION

logger = logging.getLogger("upscale.retention")

DAY = 86_400.0
# Evidence this long before an unfinished Shadow run's cursor is kept (Analyze lookups
# reach back `max_age` before a decision; the default is 60 minutes).
SHADOW_LOOKBACK_SECONDS = 2 * DAY
# Evidence this long before a position's entry and after its close is kept (reports).
POSITION_MARGIN_SECONDS = DAY
# Added to every production read-back window (snapshots, social checks).
DEPENDENCY_MARGIN_SECONDS = DAY
# A table whose newest row isn't at least this much newer than the cutoff is left alone:
# its writer may have stopped, and pruning would empty it.
MIN_RECENT_SECONDS = DAY

# Never pruned by retention (listed in reports).
PROTECTED_TABLES: dict[str, tuple[str, ...]] = {
    "scout": (
        "scout_latest", "scout_tokens", "scout_growth_stages", "scout_feed_schedule",
        "scout_meta", "scout_social_events (own 72h event retention)", "scout_social_usage",
        "decision_observations", "decision_outcome_horizons", "outcome_ranked",
        "outcome_runs", "outcome_meta",
    ),
    "evidence": ("evidence_meta",),
    "shadow": ("every table: shadow.sqlite3 is only read, never written",),
}  # fmt: skip
_FINAL = ", ".join(f"'{s}'" for s in FINAL_STATUSES)


class RetentionError(Exception):
    pass


# --- plans -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Protection:
    """Rows for which `sql` (a predicate on alias ``t``) is true are kept."""

    reason: str
    sql: str


@dataclass(frozen=True)
class Child:
    """Rows deleted together with their parent (``fk`` references the parent's rowid)."""

    table: str
    fk: str
    trigger: str | None


@dataclass
class Plan:
    database: str
    table: str
    time_column: str
    cutoff: float
    basis: dict[str, Any]
    protections: tuple[Protection, ...] = ()
    size_sql: str = "0"  # approximate bytes of one row (and its children)
    trigger: str | None = None
    children: tuple[Child, ...] = ()
    guard: Callable[[Sequence[Any]], str | None] | None = None  # on `guard_columns`
    guard_columns: tuple[str, ...] = ()
    skip: str | None = None
    floor_days: float = RAW_FLOOR_DAYS


@dataclass
class TableResult:
    database: str
    table: str
    cutoff: float | None
    basis: dict[str, Any]
    total_rows: int = 0
    older_than_cutoff: int = 0
    eligible: int = 0
    deleted: int = 0
    children_deleted: dict[str, int] = field(default_factory=dict)
    protected: dict[str, int] = field(default_factory=dict)
    eligible_oldest: float | None = None
    eligible_newest: float | None = None
    eligible_bytes: int = 0
    skipped: str | None = None
    incomplete: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "table": self.table,
            "cutoff": _iso(self.cutoff),
            "basis": self.basis,
            "total_rows": self.total_rows,
            "older_than_cutoff": self.older_than_cutoff,
            "eligible": self.eligible,
            "deleted": self.deleted,
            "children_deleted": self.children_deleted,
            "protected": dict(sorted(self.protected.items())),
            "eligible_oldest": _iso(self.eligible_oldest),
            "eligible_newest": _iso(self.eligible_newest),
            "approx_mb": round(self.eligible_bytes / 1e6, 2),
            "skipped": self.skipped,
            "incomplete": self.incomplete,
            "error": self.error,
        }


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _require(conn: sqlite3.Connection, table: str, columns: Sequence[str]) -> str | None:
    """None if `table` exists with every column, else why the table is skipped."""
    if not _has_table(conn, table):
        return f"table {table} does not exist"
    missing = sorted(set(columns) - _columns(conn, table))
    return f"table {table} lacks columns {missing}: unknown schema" if missing else None


def _meta(conn: sqlite3.Connection, table: str, key: str) -> str | None:
    if not _has_table(conn, table):
        return None
    row = conn.execute(f"SELECT value FROM {table} WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else None


# --- Shadow protection -------------------------------------------------------------------------


def _keys(*values: Any) -> set[str]:
    """Case-insensitive identities of an asset: canonical id, bare address, pool."""
    out: set[str] = set()
    for v in values:
        if isinstance(v, str) and v.strip():
            v = v.strip().lower()
            out.add(v)
            if ":" in v:
                out.add(v.split(":", 1)[1])
    return out


@dataclass
class ShadowGuard:
    """What evidence active and recorded Shadow state needs (read from shadow.sqlite3)."""

    floor: float | None = None  # evidence observed at / after this is kept
    floor_run: str | None = None
    always: set[str] = field(default_factory=set)  # open positions / pending intents
    windows: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def reason(self, row: Sequence[Any]) -> str | None:
        """`row` = (asset_id, address, pool_address, observed_at)."""
        asset_id, address, pool, at = row
        keys = _keys(asset_id, address, pool)
        if keys & self.always:
            return "Shadow open position / pending intent asset"
        for k in keys:
            for lo, hi in self.windows.get(k, ()):
                if lo <= at <= hi:
                    return "Shadow position holding window"
        return None


def load_shadow_guard(path: str | Path, ignored_runs: Sequence[str] = ()) -> ShadowGuard:
    """Read-only. Raises RetentionError when the state can't be read with certainty."""
    target = Path(path).expanduser()
    if not target.exists():
        raise RetentionError(f"no shadow database at {target}: evidence protection unverifiable")
    conn = sqlite3.connect(f"file:{target.resolve()}?mode=ro", uri=True)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SHADOW_SCHEMA_VERSION:
            raise RetentionError(f"shadow database schema v{version} is newer than this code")
        for table in ("shadow_runs", "shadow_checkpoints", "shadow_positions"):
            if not _has_table(conn, table):
                raise RetentionError(f"shadow database lacks {table}")
        g = ShadowGuard()
        runs = conn.execute(
            "SELECT r.run_id, r.until, c.cursor_at, c.processed_until, c.books_json "
            "FROM shadow_runs r LEFT JOIN shadow_checkpoints c USING (run_id)"
        ).fetchall()
        floors: list[tuple[float, str]] = []
        unfinished: list[str] = []
        books_open = books_pending_entries = 0
        for run_id, until, cursor_at, processed_until, books_json in runs:
            if cursor_at is None:
                raise RetentionError(f"shadow run {run_id} has no checkpoint")
            finished = until is not None and processed_until >= until
            if not finished:
                unfinished.append(run_id)
                if run_id not in ignored_runs:
                    floors.append((float(cursor_at) - SHADOW_LOOKBACK_SECONDS, run_id))
            try:
                books = json.loads(books_json or "{}")
                for book in books.values():
                    state = book.get("state") or {}
                    for p in (state.get("positions") or {}).values():
                        g.always |= _keys(p.get("asset_id"), p.get("address"), p.get("pool"))
                        books_open += 1
                    for e in (state.get("pending_entries") or {}).values():
                        g.always |= _keys(e.get("asset_id"), e.get("address"), e.get("pool"))
                        books_pending_entries += 1
            except (ValueError, AttributeError, TypeError) as exc:
                raise RetentionError(f"shadow run {run_id}: unreadable book state ({exc})") from exc
        if floors:
            g.floor, g.floor_run = min(floors)
        open_rows = closed_rows = 0
        for asset_id, address, pool, entry_at, status, pending_exit, closed_at in conn.execute(
            "SELECT asset_id, address, pool, entry_at, status, pending_exit, closed_at "
            "FROM shadow_positions"
        ):
            keys = _keys(asset_id, address, pool)
            if status != "CLOSED" or pending_exit is not None or closed_at is None:
                g.always |= keys
                open_rows += 1
                continue
            closed_rows += 1
            window = (entry_at - POSITION_MARGIN_SECONDS, closed_at + POSITION_MARGIN_SECONDS)
            for k in keys:
                g.windows.setdefault(k, []).append(window)
        g.summary = {
            "database": str(target),
            "runs": len(runs),
            "unfinished_runs": unfinished,
            "ignored_runs": [r for r in ignored_runs if r in unfinished],
            "cursor_floor": _iso(g.floor),
            "cursor_floor_run": g.floor_run,
            "open_positions_in_books": books_open,
            "pending_entry_intents": books_pending_entries,
            "open_or_pending_position_rows": open_rows,
            "closed_positions": closed_rows,
            "protected_identities": len(g.always),
        }
        return g
    finally:
        conn.close()


# --- building the plans ------------------------------------------------------------------------


def _latest(table: str, time_column: str, *keys: str, extra: str = "") -> str:
    """Predicate: `t` is the latest row of its key group (ties are all kept)."""
    match = " AND ".join(f"x.{k} = t.{k}" for k in keys)
    return f"t.{time_column} >= (SELECT MAX(x.{time_column}) FROM {table} x WHERE {match}{extra})"


def scout_plans(conn: sqlite3.Connection, s: RetentionSettings, now: float) -> list[Plan]:
    from upscale.config import GROWTH_CONFIG, OUTCOME_CONFIG, SOCIAL_CONFIG

    growth = load_growth_config(GROWTH_CONFIG)
    outcomes = load_outcome_config(OUTCOME_CONFIG)
    social = load_social_config(SOCIAL_CONFIG)
    growth_s = 3600 * max(
        growth.technical.lookback_hours, growth.earliness.first_seen_lookback_hours
    )
    horizon_s = 60 * max(h.minutes + h.retry_minutes for h in outcomes.horizons)
    longest = max(WINDOW_MINUTES[w] for w in social.momentum.windows)
    momentum_s = 60 * longest * (2 + social.momentum.baseline_periods)

    version = conn.execute("PRAGMA user_version").fetchone()[0]
    scout_ok = None if version == SCOUT_SCHEMA_VERSION else f"Scout schema v{version} unknown"
    social_version = _meta(conn, "scout_meta", "social_schema_version")
    social_ok = (
        None
        if social_version == SOCIAL_SCHEMA_VERSION
        else f"social schema version {social_version!r} unknown"
    )
    outcome_version = _meta(conn, "outcome_meta", "outcome_schema_version")
    has_outcomes = _has_table(conn, "scout_outcome_horizons")
    outcome_ok = (
        None
        if outcome_version == str(OUTCOME_SCHEMA_VERSION)
        else f"outcome schema version {outcome_version!r} unknown"
    )

    plans: list[Plan] = []

    # scout_snapshots
    keep = max(s.snapshot_days * DAY, growth_s + DEPENDENCY_MARGIN_SECONDS,
               horizon_s + DEPENDENCY_MARGIN_SECONDS)  # fmt: skip
    pending = (
        "EXISTS (SELECT 1 FROM scout_outcome_observations o JOIN scout_outcome_horizons h "
        "ON h.observation_id = o.id WHERE o.canonical_id = t.canonical_id "
        f"AND h.status NOT IN ({_FINAL}))"
    )
    snap = Plan(
        "scout", "scout_snapshots", "observed_at", now - keep,
        {"retention_days": s.snapshot_days, "effective_days": keep / DAY,
         "growth_lookback_hours": growth_s / 3600, "outcome_horizon_hours": horizon_s / 3600},
        protections=(
            Protection("latest snapshot of the token", _latest("scout_snapshots", "observed_at",
                                                               "canonical_id")),
            *((Protection("token has a pending outcome horizon", pending),) if has_outcomes else ()),
        ),
        size_sql="LENGTH(t.windows_json) + COALESCE(LENGTH(t.social_json), 0) + "
                 "LENGTH(t.canonical_id) + LENGTH(t.pool_address) + 96",
        skip=scout_ok or (outcome_ok if has_outcomes else None)
        or _require(conn, "scout_snapshots", ("canonical_id", "observed_at", "windows_json")),
    )  # fmt: skip
    plans.append(snap)

    # social history
    social_days = s.social_days * DAY
    plans.append(Plan(
        "scout", "scout_social_snapshots", "observed_at", now - social_days,
        {"retention_days": s.social_days},
        protections=(Protection("latest social snapshot of the token and provider",
                                _latest("scout_social_snapshots", "observed_at", "canonical_id",
                                        "provider")),),
        size_sql="LENGTH(t.body_json) + LENGTH(t.canonical_id) + 64",
        skip=social_ok or _require(conn, "scout_social_snapshots",
                                   ("canonical_id", "provider", "observed_at", "body_json")),
    ))  # fmt: skip
    plans.append(Plan(
        "scout", "scout_social_momentum", "computed_at", now - social_days,
        {"retention_days": s.social_days},
        protections=(
            Protection("latest momentum of the token",
                       _latest("scout_social_momentum", "computed_at", "canonical_id")),
            Protection("latest measured momentum of the token (Growth Scout reads it)",
                       "t.state != 'UNAVAILABLE' AND " + _latest(
                           "scout_social_momentum", "computed_at", "canonical_id",
                           extra=" AND x.state != 'UNAVAILABLE'")),
        ),
        size_sql="LENGTH(t.body_json) + LENGTH(t.canonical_id) + 48",
        skip=social_ok or _require(conn, "scout_social_momentum",
                                   ("canonical_id", "computed_at", "state", "body_json")),
    ))  # fmt: skip
    keep = max(social_days, momentum_s + DEPENDENCY_MARGIN_SECONDS)
    pair = "x.canonical_id = t.canonical_id AND x.provider = t.provider"
    plans.append(Plan(
        "scout", "scout_social_checks", "checked_at", now - keep,
        {"retention_days": s.social_days, "effective_days": keep / DAY,
         "momentum_span_hours": momentum_s / 3600},
        protections=(
            Protection("latest check of the token and provider",
                       _latest("scout_social_checks", "checked_at", "canonical_id", "provider")),
            Protection("latest covered_to of the token and provider (scheduler reads it)",
                       "t.covered_to IS NOT NULL AND t.covered_to >= (SELECT MAX(x.covered_to) "
                       f"FROM scout_social_checks x WHERE {pair})"),
            Protection("first PROVIDER_UNAVAILABLE check (scheduler reads it)",
                       "t.status = 'PROVIDER_UNAVAILABLE' AND t.checked_at <= (SELECT "
                       f"MIN(x.checked_at) FROM scout_social_checks x WHERE {pair} "
                       "AND x.status = 'PROVIDER_UNAVAILABLE')"),
        ),
        size_sql="LENGTH(t.canonical_id) + COALESCE(LENGTH(t.error), 0) + 80",
        skip=social_ok or _require(conn, "scout_social_checks",
                                   ("canonical_id", "provider", "checked_at", "status",
                                    "covered_to")),
    ))  # fmt: skip

    # finalized outcome history (observation + every horizon, together)
    cutoff = now - s.outcome_days * DAY
    h_exists = "EXISTS (SELECT 1 FROM scout_outcome_horizons h WHERE h.observation_id = t.id"
    decisions = _has_table(conn, "decision_observations")
    plans.append(Plan(
        "scout", "scout_outcome_observations", "anchored_at", cutoff,
        {"retention_days": s.outcome_days},
        protections=(
            Protection("pending / non-final horizon", f"{h_exists} AND h.status NOT IN ({_FINAL}))"),
            Protection("horizon finalized within the retention window",
                       f"{h_exists} AND (h.finalized_at IS NULL OR h.finalized_at >= {cutoff!r}))"),
            Protection("observation without horizons",
                       "NOT EXISTS (SELECT 1 FROM scout_outcome_horizons h "
                       "WHERE h.observation_id = t.id)"),
            *((Protection("linked by an Analyze decision observation",
                          "EXISTS (SELECT 1 FROM decision_observations d "
                          "WHERE d.scout_observation_id = t.id)"),) if decisions else ()),
        ),
        size_sql="LENGTH(t.body_json) + 160 + (SELECT COALESCE(SUM(COALESCE(LENGTH(h.price_json), 0)"
                 " + COALESCE(LENGTH(h.market_json), 0) + COALESCE(LENGTH(h.stage_json), 0) + "
                 "COALESCE(LENGTH(h.triggers_json), 0) + 120), 0) FROM scout_outcome_horizons h "
                 "WHERE h.observation_id = t.id)",
        trigger="scout_outcome_observations_no_delete",
        children=(Child("scout_outcome_horizons", "observation_id",
                        "scout_outcome_horizons_no_delete"),),
        skip=(None if has_outcomes else "no outcome tables") or outcome_ok
        or _require(conn, "scout_outcome_observations", ("id", "anchored_at", "body_json"))
        or _require(conn, "scout_outcome_horizons",
                    ("observation_id", "status", "finalized_at")),
        floor_days=OUTCOME_FLOOR_DAYS,
    ))  # fmt: skip
    return plans


def evidence_plans(
    conn: sqlite3.Connection, s: RetentionSettings, now: float, guard: ShadowGuard | str
) -> list[Plan]:
    """`guard`: the Shadow protection, or why it couldn't be loaded (nothing is deleted)."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    cutoff = now - s.evidence_days * DAY
    basis: dict[str, Any] = {"retention_days": s.evidence_days}
    skip = None if version == EVIDENCE_SCHEMA_VERSION else f"evidence schema v{version} unknown"
    if isinstance(guard, str):
        skip = skip or f"Shadow protection unavailable: {guard}"
    elif guard.floor is not None and guard.floor < cutoff:
        cutoff = guard.floor
        basis["held_back_by_shadow_run"] = guard.floor_run
    return [
        Plan(
            "evidence", "evidence_records", "observed_at", cutoff, basis,
            protections=(Protection("latest record of the kind and asset", _latest(
                "evidence_records", "observed_at", "kind", "asset_id")),),
            size_sql="LENGTH(t.payload) + LENGTH(t.versions_json) + LENGTH(t.links_json) + "
                     "LENGTH(t.asset_id) + 260",
            trigger="evidence_no_delete",
            guard=None if isinstance(guard, str) else guard.reason,
            guard_columns=("asset_id", "address", "pool_address", "observed_at"),
            skip=skip or _require(conn, "evidence_records",
                                  ("kind", "asset_id", "address", "pool_address", "observed_at",
                                   "payload")),
        )
    ]  # fmt: skip


# --- executing a plan --------------------------------------------------------------------------


def _scan_sql(plan: Plan) -> str:
    case = (
        "CASE " + " ".join(f"WHEN {p.sql} THEN ?" for p in plan.protections) + " ELSE NULL END"
        if plan.protections
        else "NULL"
    )
    extra = "".join(f", t.{c}" for c in plan.guard_columns)
    return (
        f"SELECT t.rowid, {case}, t.{plan.time_column}, {plan.size_sql}{extra} "
        f"FROM {plan.table} t WHERE t.rowid > ? AND t.rowid <= ? AND t.{plan.time_column} < ? "
        "ORDER BY t.rowid LIMIT ?"
    )


def _keep_sql(plan: Plan) -> str:
    return " OR ".join(f"({p.sql})" for p in plan.protections) or "0"


def _suspend(conn: sqlite3.Connection, names: Sequence[str]) -> list[tuple[str, str]]:
    saved = []
    for name in names:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
        ).fetchone()
        if row is None or not row[0]:
            raise RetentionError(f"expected trigger {name} is missing: unknown schema")
        saved.append((name, str(row[0])))
        conn.execute(f"DROP TRIGGER {name}")
    return saved


def _restore(conn: sqlite3.Connection, saved: Sequence[tuple[str, str]]) -> None:
    for name, sql in saved:
        conn.execute(sql)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
        ).fetchone()
        if row is None or row[0] != sql:
            raise RetentionError(f"trigger {name} could not be restored")


def _delete(conn: sqlite3.Connection, plan: Plan, ids: Sequence[int]) -> tuple[int, dict[str, int]]:
    """One transaction: re-check, suspend triggers, delete children and rows, restore."""
    marks = ",".join("?" for _ in ids)
    conn.execute("BEGIN IMMEDIATE")
    try:
        still = [
            r[0]
            for r in conn.execute(
                f"SELECT t.rowid FROM {plan.table} t WHERE t.rowid IN ({marks}) "
                f"AND t.{plan.time_column} < ? AND NOT ({_keep_sql(plan)})",
                [*ids, plan.cutoff],
            )
        ]
        children: dict[str, int] = {}
        n = 0
        if still:
            triggers = [c.trigger for c in plan.children if c.trigger] + (
                [plan.trigger] if plan.trigger else []
            )
            saved = _suspend(conn, triggers)
            m = ",".join("?" for _ in still)
            for c in plan.children:
                children[c.table] = conn.execute(
                    f"DELETE FROM {c.table} WHERE {c.fk} IN ({m})", still
                ).rowcount
            n = conn.execute(f"DELETE FROM {plan.table} WHERE rowid IN ({m})", still).rowcount
            _restore(conn, saved)
        conn.execute("COMMIT")
        return n, children
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _precheck(conn: sqlite3.Connection, plan: Plan, now: float, r: TableResult) -> str | None:
    """Why this plan must not run (None: safe)."""
    if plan.skip:
        return plan.skip
    if plan.cutoff > now - plan.floor_days * DAY:
        return f"cutoff {_iso(plan.cutoff)} is inside the {plan.floor_days:g}-day floor"
    row = conn.execute(f"SELECT COUNT(*), MAX({plan.time_column}) FROM {plan.table}").fetchone()
    r.total_rows = int(row[0])
    newest = row[1]
    if newest is None:
        return "table is empty"
    if newest > now + DAY:
        return f"newest row {_iso(newest)} is in the future: clock or data problem"
    if newest < plan.cutoff + MIN_RECENT_SECONDS:
        return (
            f"newest row {_iso(newest)} is not a day newer than the cutoff: the writer may "
            "have stopped; refusing to prune"
        )
    return None


def run_plan(
    conn: sqlite3.Connection,
    plan: Plan,
    now: float,
    *,
    dry_run: bool,
    batch_size: int,
    pause: float = 0.0,
    deadline: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> TableResult:
    r = TableResult(plan.database, plan.table, plan.cutoff, plan.basis)
    try:
        r.skipped = _precheck(conn, plan, now, r)
        if r.skipped:
            return r
        upper = conn.execute(
            f"SELECT MAX(rowid) FROM {plan.table} WHERE {plan.time_column} < ?", (plan.cutoff,)
        ).fetchone()[0]
        if upper is None:
            return r
        sql = _scan_sql(plan)
        reasons = [p.reason for p in plan.protections]
        last = -(2**63)
        g = len(plan.guard_columns)
        while True:
            if deadline is not None and monotonic() > deadline:
                r.incomplete = True
                break
            rows = conn.execute(sql, [*reasons, last, upper, plan.cutoff, batch_size]).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            ids: list[int] = []
            for row in rows:
                r.older_than_cutoff += 1
                reason = row[1]
                if reason is None and plan.guard is not None:
                    reason = plan.guard(row[4 : 4 + g])
                if reason is not None:
                    r.protected[reason] = r.protected.get(reason, 0) + 1
                    continue
                ids.append(row[0])
                at = float(row[2])
                r.eligible += 1
                r.eligible_bytes += int(row[3] or 0)
                r.eligible_oldest = at if r.eligible_oldest is None else min(r.eligible_oldest, at)
                r.eligible_newest = at if r.eligible_newest is None else max(r.eligible_newest, at)
            if ids and not dry_run:
                n, children = _delete(conn, plan, ids)
                r.deleted += n
                for k, v in children.items():
                    r.children_deleted[k] = r.children_deleted.get(k, 0) + v
                if pause > 0:
                    sleep(pause)
    except Exception as exc:
        logger.exception("retention %s.%s failed", plan.database, plan.table)
        r.error = f"{type(exc).__name__}: {exc}"
    return r


# --- databases -----------------------------------------------------------------------------------


def storage(conn: sqlite3.Connection, path: str | Path) -> dict[str, Any]:
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    p = Path(path).expanduser()
    wal = p.with_name(p.name + "-wal")
    return {
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist,
        "freelist_mb": round(freelist * page_size / 1e6, 2),
        "file_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None,
        "wal_mb": round(wal.stat().st_size / 1e6, 2) if wal.exists() else 0.0,
    }


def _connect(path: str | Path, read_only: bool) -> sqlite3.Connection:
    target = Path(path).expanduser()
    if not target.exists():
        raise RetentionError(f"no database at {target}")
    if read_only:
        return sqlite3.connect(f"file:{target.resolve()}?mode=ro", uri=True)
    conn = sqlite3.connect(str(target), isolation_level=None, timeout=10)
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _process(
    name: str,
    path: str | Path,
    build: Callable[[sqlite3.Connection], list[Plan]],
    s: RetentionSettings,
    now: float,
    dry_run: bool,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    budget: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(Path(path).expanduser()), "tables": []}
    try:
        conn = _connect(path, read_only=dry_run)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        out["before"] = storage(conn, path)
        plans = build(conn)
        deadline = None if budget is None else monotonic() + budget
        results = [
            run_plan(conn, p, now, dry_run=dry_run, batch_size=s.batch_size,
                     pause=s.batch_pause_seconds, deadline=deadline, sleep=sleep,
                     monotonic=monotonic)
            for p in plans
        ]  # fmt: skip
        out["tables"] = [r.as_dict() for r in results]
        if not dry_run:
            out["wal_checkpoint"] = list(conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone())
            touched = [p.table for p in plans] + [c.table for p in plans for c in p.children]
            out["foreign_key_violations"] = sum(
                len(conn.execute(f"PRAGMA foreign_key_check({t})").fetchall())
                for t in touched
                if _has_table(conn, t)
            )
            out["after"] = storage(conn, path)
        for r in results:
            if r.skipped or r.error or not dry_run:
                logger.info(
                    "retention %s %s.%s: cutoff %s, %s %d of %d older rows (protected %s)%s%s",
                    "dry-run" if dry_run else "cleanup", name, r.table, _iso(r.cutoff),
                    "eligible" if dry_run else "deleted", r.eligible if dry_run else r.deleted,
                    r.older_than_cutoff, r.protected or "{}",
                    f"; skipped: {r.skipped}" if r.skipped else "",
                    f"; error: {r.error}" if r.error else "",
                )  # fmt: skip
    except Exception as exc:  # isolated: one database never stops the other
        logger.exception("retention %s failed", name)
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    return out


def run(
    settings: RetentionSettings,
    scout_db: str | Path,
    evidence_db: str | Path,
    shadow_db: str | Path,
    *,
    dry_run: bool = True,
    now: datetime | None = None,
    budget_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """One retention pass over both databases (never raises). `dry_run`: read-only."""
    at = (now or datetime.now(UTC)).timestamp()
    report: dict[str, Any] = {
        "mode": "dry-run" if dry_run else "cleanup",
        "now": _iso(at),
        "policy": settings.policy(),
        "never_pruned": PROTECTED_TABLES,
    }
    guard: ShadowGuard | str
    try:
        guard = load_shadow_guard(shadow_db, settings.ignored_shadow_runs)
        report["shadow_guard"] = guard.summary
    except Exception as exc:
        guard = f"{type(exc).__name__}: {exc}"
        report["shadow_guard"] = {"error": guard}
        logger.warning("retention: evidence is kept (Shadow protection unavailable: %s)", guard)
    report["databases"] = {
        "scout": _process("scout", scout_db, lambda c: scout_plans(c, settings, at), settings,
                          at, dry_run, sleep, monotonic, budget_seconds),
        "evidence": _process("evidence", evidence_db,
                             lambda c: evidence_plans(c, settings, at, guard), settings, at,
                             dry_run, sleep, monotonic, budget_seconds),
    }  # fmt: skip
    tables = [t for d in report["databases"].values() for t in d.get("tables", [])]
    report["totals"] = {
        "eligible_rows": sum(t["eligible"] for t in tables),
        "deleted_rows": sum(t["deleted"] for t in tables),
        "approx_reusable_mb": round(sum(t["approx_mb"] for t in tables), 2),
        "incomplete": any(t["incomplete"] for t in tables),
        "errors": [f"{t['database']}.{t['table']}: {t['error']}" for t in tables if t["error"]]
        + [f"{k}: {d['error']}" for k, d in report["databases"].items() if d.get("error")],
    }
    return report


def status(
    settings: RetentionSettings,
    scout_db: str | Path,
    evidence_db: str | Path,
    shadow_db: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read-only: sizes, freelist, per-table rows / age range and current cutoffs."""
    at = (now or datetime.now(UTC)).timestamp()
    out: dict[str, Any] = {"now": _iso(at), "policy": settings.policy(),
                           "never_pruned": PROTECTED_TABLES, "databases": {}}  # fmt: skip
    guard: ShadowGuard | str
    try:
        guard = load_shadow_guard(shadow_db, settings.ignored_shadow_runs)
        out["shadow_guard"] = guard.summary
    except Exception as exc:
        guard = f"{type(exc).__name__}: {exc}"
        out["shadow_guard"] = {"error": guard}
    builders: list[tuple[str, str | Path, Callable[[sqlite3.Connection], list[Plan]]]] = [
        ("scout", scout_db, lambda c: scout_plans(c, settings, at)),
        ("evidence", evidence_db, lambda c: evidence_plans(c, settings, at, guard)),
    ]
    for name, path, build in builders:
        d: dict[str, Any] = {"path": str(Path(path).expanduser())}
        try:
            conn = _connect(path, read_only=True)
        except Exception as exc:
            d["error"] = f"{type(exc).__name__}: {exc}"
            out["databases"][name] = d
            continue
        try:
            d["storage"] = storage(conn, path)
            tables = []
            for p in build(conn):
                t: dict[str, Any] = {"table": p.table, "cutoff": _iso(p.cutoff),
                                     "basis": p.basis, "skipped": p.skip}  # fmt: skip
                if _has_table(conn, p.table):
                    row = conn.execute(
                        f"SELECT COUNT(*), MIN({p.time_column}), MAX({p.time_column}), "
                        f"SUM({p.time_column} < ?) FROM {p.table}",
                        (p.cutoff,),
                    ).fetchone()
                    t |= {"rows": row[0], "oldest": _iso(row[1]), "newest": _iso(row[2]),
                          "older_than_cutoff": row[3] or 0}  # fmt: skip
                tables.append(t)
            d["tables"] = tables
        except Exception as exc:
            d["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        out["databases"][name] = d
    try:
        free = os.statvfs(Path(scout_db).expanduser().parent)
        out["filesystem"] = {
            "total_mb": round(free.f_blocks * free.f_frsize / 1e6, 1),
            "available_mb": round(free.f_bavail * free.f_frsize / 1e6, 1),
        }
    except OSError:
        pass
    return out
