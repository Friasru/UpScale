"""Shadow database (``UPSCALE_SHADOW_DB``; default ``shadow.sqlite3`` next to the Scout
database, on Railway ``/data/shadow.sqlite3``). Separate from Scout, outcome, evidence,
replay and calibration databases: nothing here is read by production decisions.

Tables (schema version 2; a version 1 file gains ``shadow_rejections`` in place):

* ``shadow_strategies``: every registered (strategy_id, version), its rules and hash.
  Append-only: changed rules need a new version.
* ``shadow_runs``: a paper book definition (start / end time, the frozen strategy
  versions and hashes, execution model). Append-only.
* ``shadow_decisions``: ENTER / HOLD / EXIT / NO_ACTION decisions with the evidence
  summary and fingerprints they used. Append-only.
* ``shadow_positions``: one row per simulated position. Its entry columns never change; the
  mark columns (last price, peak / trough, remaining quantity, pending exit) change while
  OPEN; once CLOSED the row is frozen.
* ``shadow_trades``: one row per simulated exit fill (partial or final). Append-only.
* ``shadow_equity``: the paper portfolio after each Scout decision time. Append-only.
* ``shadow_metrics``: metric snapshots per run and strategy. Append-only.
* ``shadow_checkpoints``: per run, the resume cursor and each book's state, replaced in
  the same transaction as the rows it produced (derived state, not history).
* ``shadow_rejections`` (v2): one row per Scout evaluation a strategy did not enter on
  because its entry rules failed: every failed rule as a stable reason code, the observed
  values the rules read, and the evidence fingerprints. Diagnostics only (never a
  decision). Append-only. Qualified entries stopped by position / risk control stay
  NO_ACTION decisions, as in v1.
* ``shadow_meta``: schema metadata, and per run the time from which rejection
  diagnostics exist (``diagnostics_from:<run_id>``, written once).

Immutability is enforced by triggers, not only by the code.
"""

import json
import sqlite3
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from upscale.services.shadow.book import Position
from upscale.services.shadow.config import StrategyConfig

SCHEMA_VERSION = 2
DIAGNOSTICS_VERSION = 1
APPEND_ONLY = (
    "shadow_strategies", "shadow_runs", "shadow_decisions", "shadow_trades", "shadow_equity",
    "shadow_metrics", "shadow_rejections",
)  # fmt: skip
_POSITION_ENTRY = (
    "position_id", "run_id", "strategy_id", "strategy_version", "asset_id", "chain", "address",
    "symbol", "pool", "dex", "entry_decision_id", "entry_at", "entry_price", "entry_price_at",
    "quantity", "cost_usd",
)  # fmt: skip

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shadow_strategies (
    strategy_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    registered_at REAL NOT NULL,
    PRIMARY KEY (strategy_id, version)
);
CREATE TABLE IF NOT EXISTS shadow_runs (
    run_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    since REAL NOT NULL,
    until REAL,
    clean_data BOOLEAN NOT NULL,
    execution_model TEXT NOT NULL,
    strategies_json TEXT NOT NULL,
    args_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_decisions (
    id INTEGER PRIMARY KEY,
    decision_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES shadow_runs(run_id),
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('ENTER', 'HOLD', 'EXIT', 'NO_ACTION')),
    asset_id TEXT NOT NULL,
    chain TEXT,
    address TEXT,
    pool TEXT,
    decision_at REAL NOT NULL,
    reference_price REAL,
    reference_price_at REAL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    fingerprints_json TEXT NOT NULL,
    position_id TEXT,
    recorded_at REAL NOT NULL,
    CHECK (reference_price_at IS NULL OR reference_price_at <= decision_at)
);
CREATE INDEX IF NOT EXISTS shadow_decisions_by_run
    ON shadow_decisions (run_id, strategy_id, decision_at);
CREATE TABLE IF NOT EXISTS shadow_positions (
    position_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_runs(run_id),
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    asset_id TEXT NOT NULL,
    chain TEXT,
    address TEXT,
    symbol TEXT,
    pool TEXT NOT NULL,
    dex TEXT,
    entry_decision_id TEXT NOT NULL,
    entry_at REAL NOT NULL,
    entry_price REAL NOT NULL CHECK (entry_price > 0),
    entry_price_at REAL NOT NULL,
    quantity REAL NOT NULL CHECK (quantity > 0),
    cost_usd REAL NOT NULL CHECK (cost_usd > 0),
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED')),
    remaining_quantity REAL NOT NULL,
    last_price REAL NOT NULL,
    last_price_at REAL NOT NULL,
    peak_price REAL NOT NULL,
    trough_price REAL NOT NULL,
    tp_hit INTEGER NOT NULL,
    pending_exit TEXT,
    closed_at REAL,
    exit_reason TEXT,
    CHECK (entry_price_at <= entry_at)
);
CREATE INDEX IF NOT EXISTS shadow_positions_by_run ON shadow_positions (run_id, strategy_id, status);
CREATE TABLE IF NOT EXISTS shadow_trades (
    id INTEGER PRIMARY KEY,
    trade_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES shadow_runs(run_id),
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    position_id TEXT NOT NULL REFERENCES shadow_positions(position_id),
    fill_no INTEGER NOT NULL,
    final BOOLEAN NOT NULL,
    asset_id TEXT NOT NULL,
    chain TEXT,
    address TEXT,
    symbol TEXT,
    pool TEXT NOT NULL,
    entry_decision_id TEXT NOT NULL,
    exit_decision_id TEXT NOT NULL,
    entry_at REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_at REAL NOT NULL,
    exit_price REAL,
    exit_price_at REAL,
    exit_price_record TEXT,
    quantity REAL NOT NULL,
    fraction REAL,
    cost_usd REAL NOT NULL,
    proceeds_usd REAL,
    pnl_usd REAL,
    return_pct REAL,
    mfe_pct REAL,
    mae_pct REAL,
    holding_minutes REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    trigger_level REAL,
    price_basis TEXT NOT NULL,
    execution_model TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    CHECK (exit_at >= entry_at),
    CHECK (exit_price_at IS NULL OR exit_price_at <= exit_at),
    CHECK ((exit_price IS NULL) = (exit_reason = 'MARKET_UNAVAILABLE'))
);
CREATE INDEX IF NOT EXISTS shadow_trades_by_run ON shadow_trades (run_id, strategy_id, exit_at);
CREATE TABLE IF NOT EXISTS shadow_equity (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_runs(run_id),
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    at REAL NOT NULL,
    cash REAL NOT NULL,
    open_value REAL NOT NULL,
    equity REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    unresolved_cost REAL NOT NULL,
    exposure_pct REAL NOT NULL,
    open_positions INTEGER NOT NULL,
    drawdown_pct REAL NOT NULL,
    max_drawdown_pct REAL NOT NULL,
    UNIQUE (run_id, strategy_id, strategy_version, at)
);
CREATE TABLE IF NOT EXISTS shadow_metrics (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_runs(run_id),
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    computed_at REAL NOT NULL,
    through REAL,
    metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_rejections (
    id INTEGER PRIMARY KEY,
    rejection_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES shadow_runs(run_id),
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    asset_id TEXT NOT NULL,
    pool TEXT,
    decision_at REAL NOT NULL,
    scout_record_id TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    observed_json TEXT NOT NULL,
    fingerprints_json TEXT NOT NULL,
    diagnostics_version INTEGER NOT NULL,
    recorded_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS shadow_rejections_by_run
    ON shadow_rejections (run_id, strategy_id, decision_at);
CREATE TABLE IF NOT EXISTS shadow_checkpoints (
    run_id TEXT PRIMARY KEY REFERENCES shadow_runs(run_id),
    cursor_at REAL NOT NULL,
    cursor_id INTEGER NOT NULL,
    processed_until REAL NOT NULL,
    books_json TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _triggers() -> str:
    out = []
    for table in APPEND_ONLY:
        out.append(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;\n"
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;"
        )
    changed = " OR ".join(f"NEW.{c} IS NOT OLD.{c}" for c in _POSITION_ENTRY)
    out.append(
        "CREATE TRIGGER IF NOT EXISTS shadow_positions_entry_frozen BEFORE UPDATE ON "
        f"shadow_positions WHEN {changed} "
        "BEGIN SELECT RAISE(ABORT, 'a shadow position entry never changes'); END;\n"
        "CREATE TRIGGER IF NOT EXISTS shadow_positions_closed_frozen BEFORE UPDATE ON "
        "shadow_positions WHEN OLD.status = 'CLOSED' "
        "BEGIN SELECT RAISE(ABORT, 'a closed shadow position never changes'); END;\n"
        "CREATE TRIGGER IF NOT EXISTS shadow_positions_no_delete BEFORE DELETE ON "
        "shadow_positions BEGIN SELECT RAISE(ABORT, 'shadow positions are kept'); END;"
    )
    return "\n".join(out)


class ShadowStoreError(Exception):
    pass


def _ts(t: datetime | None) -> float | None:
    return t.timestamp() if t is not None else None


def _utc(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None


class ShadowStore:
    """Thread-safe (one connection behind a lock)."""

    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = str(path)
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        if self.read_only:
            target = Path(self.path).expanduser().resolve()
            if not target.exists():
                raise ShadowStoreError(f"no shadow database at {target}")
            self._conn = sqlite3.connect(
                f"file:{target}?mode=ro", uri=True, check_same_thread=False
            )
            return self._conn
        memory = self.path == ":memory:"
        if not memory:
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.path if memory else str(Path(self.path).expanduser()), check_same_thread=False
        )
        conn.execute("PRAGMA foreign_keys = ON")
        if not memory:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            conn.close()
            raise ShadowStoreError(
                f"shadow database {self.path} has schema v{version}; this UpScale knows "
                f"v{SCHEMA_VERSION}"
            )
        conn.executescript(_SCHEMA + _triggers())
        conn.execute(
            "INSERT INTO shadow_meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        self._conn = conn
        return conn

    def _write_guard(self) -> None:
        if self.read_only:
            raise ShadowStoreError("this shadow store is opened read-only")

    # --- strategies ---------------------------------------------------------------------------

    def register(self, cfg: StrategyConfig) -> str:
        """ "created", or "exists" (identical rules). Different rules under an existing
        (id, version) are refused: a changed strategy is a new version."""
        self._write_guard()
        with self._lock:
            db = self._db()
            row = db.execute(
                "SELECT config_hash FROM shadow_strategies WHERE strategy_id = ? AND version = ?",
                (cfg.strategy_id, cfg.version),
            ).fetchone()
            if row is not None:
                if row[0] != cfg.config_hash:
                    raise ShadowStoreError(
                        f"{cfg.key} is already registered with different rules "
                        f"({row[0]} != {cfg.config_hash}): register a new version instead"
                    )
                return "exists"
            with db:
                db.execute(
                    "INSERT INTO shadow_strategies VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (cfg.strategy_id, cfg.version, cfg.name, cfg.config_hash,
                     cfg.model_dump_json(), cfg.created_at.timestamp(), time.time()),
                )  # fmt: skip
            return "created"

    def strategies(self, strategy_id: str | None = None) -> list[StrategyConfig]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT config_json, config_hash FROM shadow_strategies "
                    "WHERE ? IS NULL OR strategy_id = ? ORDER BY strategy_id, version",
                    (strategy_id, strategy_id),
                )
                .fetchall()
            )
        out = []
        for body, digest in rows:
            cfg = StrategyConfig.model_validate_json(body)
            if cfg.config_hash != digest:
                raise ShadowStoreError(f"{cfg.key} fails its configuration hash check")
            out.append(cfg)
        return out

    def strategy(self, strategy_id: str, version: int) -> StrategyConfig:
        found = [s for s in self.strategies(strategy_id) if s.version == version]
        if not found:
            raise ShadowStoreError(f"unknown strategy {strategy_id}@v{version}")
        return found[0]

    def latest_strategies(self) -> list[StrategyConfig]:
        latest: dict[str, StrategyConfig] = {}
        for s in self.strategies():
            if s.strategy_id not in latest or s.version > latest[s.strategy_id].version:
                latest[s.strategy_id] = s
        return sorted(latest.values(), key=lambda s: s.strategy_id)

    # --- runs -----------------------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        since: datetime,
        until: datetime | None,
        strategies: Sequence[StrategyConfig],
        clean_data: bool,
        args: dict[str, Any],
    ) -> None:
        self._write_guard()
        frozen = [
            {"strategy_id": s.strategy_id, "version": s.version, "config_hash": s.config_hash}
            for s in strategies
        ]
        models = {s.execution_model for s in strategies}
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    "INSERT INTO shadow_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, time.time(), since.timestamp(), _ts(until), clean_data,
                     ",".join(sorted(models)), json.dumps(frozen), json.dumps(args, default=str)),
                )  # fmt: skip
                db.execute(
                    "INSERT INTO shadow_checkpoints VALUES (?, ?, 0, ?, '{}', '{}', ?)",
                    (run_id, since.timestamp() - 1e-6, since.timestamp(), time.time()),
                )

    def run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT run_id, created_at, since, until, clean_data, execution_model, "
                    "strategies_json, args_json FROM shadow_runs WHERE run_id = ?",
                    (run_id,),
                )
                .fetchone()
            )
        if row is None:
            return None
        return {
            "run_id": row[0], "created_at": _utc(row[1]), "since": _utc(row[2]),
            "until": _utc(row[3]), "clean_data": bool(row[4]), "execution_model": row[5],
            "strategies": json.loads(row[6]), "args": json.loads(row[7]),
            "since_ts": row[2], "until_ts": row[3],
        }  # fmt: skip

    def runs(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = [r[0] for r in self._db().execute(
                "SELECT run_id FROM shadow_runs ORDER BY created_at")]  # fmt: skip
        return [r for i in ids if (r := self.run(i)) is not None]

    def checkpoint(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT cursor_at, cursor_id, processed_until, books_json, stats_json, updated_at "
                    "FROM shadow_checkpoints WHERE run_id = ?",
                    (run_id,),
                )
                .fetchone()
            )
        if row is None:
            raise ShadowStoreError(f"run {run_id} has no checkpoint")
        return {
            "cursor": (row[0], row[1]), "processed_until": row[2],
            "books": json.loads(row[3]), "stats": json.loads(row[4]), "updated_at": row[5],
        }  # fmt: skip

    # --- the batch write (one transaction) -----------------------------------------------------

    def commit(
        self,
        run_id: str,
        *,
        decisions: Sequence[dict[str, Any]],
        opened: Sequence[tuple[str, int, Position]],
        marked: Sequence[Position],
        closed: Sequence[Position],
        trades: Sequence[dict[str, Any]],
        equity: Sequence[dict[str, Any]],
        rejections: Sequence[dict[str, Any]] = (),
        cursor: tuple[float, int],
        processed_until: float,
        books: dict[str, Any],
        stats: dict[str, Any],
    ) -> None:
        """Every row one processing step produced, plus the checkpoint that resumes after
        it, atomically: a crash never leaves rows without the matching checkpoint."""
        self._write_guard()
        now = time.time()
        with self._lock:
            db = self._db()
            with db:
                db.executemany(
                    "INSERT OR IGNORE INTO shadow_decisions (decision_id, run_id, strategy_id, "
                    "strategy_version, action, asset_id, chain, address, pool, decision_at, "
                    "reference_price, reference_price_at, reason, evidence_json, "
                    "fingerprints_json, position_id, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(d["decision_id"], d["run_id"], d["strategy_id"], d["strategy_version"],
                      d["action"], d["asset_id"], d["chain"], d["address"], d["pool"],
                      d["decision_at"].timestamp(), d["reference_price"],
                      _ts(d["reference_price_at"]), d["reason"],
                      json.dumps(d["evidence"], default=str, sort_keys=True),
                      json.dumps(d["fingerprints"], default=str), d["position_id"], now)
                     for d in decisions],
                )  # fmt: skip
                for sid, version, p in opened:
                    db.execute(
                        "INSERT INTO shadow_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                        "?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                        (p.position_id, run_id, sid, version,
                         p.asset_id, p.chain, p.address, p.symbol, p.pool, p.dex,
                         p.entry_decision_id, p.entry_at.timestamp(), p.entry_price,
                         p.entry_price_at.timestamp(), p.quantity, p.cost_usd,
                         p.quantity, p.entry_price, p.entry_price_at.timestamp(), p.entry_price,
                         p.entry_price, 0, None),
                    )  # fmt: skip
                db.executemany(
                    "INSERT INTO shadow_trades (trade_id, run_id, strategy_id, strategy_version, "
                    "position_id, fill_no, final, asset_id, chain, address, symbol, pool, "
                    "entry_decision_id, exit_decision_id, entry_at, entry_price, exit_at, "
                    "exit_price, exit_price_at, exit_price_record, quantity, fraction, cost_usd, "
                    "proceeds_usd, pnl_usd, return_pct, mfe_pct, mae_pct, holding_minutes, "
                    "exit_reason, trigger_level, price_basis, execution_model, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(t["trade_id"], t["run_id"], t["strategy_id"], t["strategy_version"],
                      t["position_id"], t["fill_no"], t["final"], t["asset_id"], t["chain"],
                      t["address"], t["symbol"], t["pool"], t["entry_decision_id"],
                      t["exit_decision_id"], t["entry_at"].timestamp(), t["entry_price"],
                      t["exit_at"].timestamp(), t["exit_price"], _ts(t["exit_price_at"]),
                      t["exit_price_record"], t["quantity"], t["fraction"], t["cost_usd"],
                      t["proceeds_usd"], t["pnl_usd"], t["return_pct"], t["mfe_pct"],
                      t["mae_pct"], t["holding_minutes"], t["exit_reason"], t["trigger_level"],
                      t["price_basis"], t["execution_model"], now) for t in trades],
                )  # fmt: skip
                for p in marked:
                    db.execute(
                        "UPDATE shadow_positions SET remaining_quantity = ?, last_price = ?, "
                        "last_price_at = ?, peak_price = ?, trough_price = ?, tp_hit = ?, "
                        "pending_exit = ? WHERE position_id = ? AND status = 'OPEN'",
                        (p.remaining_quantity, p.last_price, p.last_price_at.timestamp(),
                         p.peak_price, p.trough_price, p.tp_hit, p.pending_exit, p.position_id),
                    )  # fmt: skip
                for p in closed:
                    assert p.closed_at is not None
                    db.execute(
                        "UPDATE shadow_positions SET status = 'CLOSED', remaining_quantity = 0, "
                        "last_price = ?, last_price_at = ?, peak_price = ?, trough_price = ?, "
                        "tp_hit = ?, pending_exit = NULL, closed_at = ?, exit_reason = ? "
                        "WHERE position_id = ?",
                        (p.last_price, p.last_price_at.timestamp(), p.peak_price, p.trough_price,
                         p.tp_hit, p.closed_at.timestamp(), p.exit_reason, p.position_id),
                    )  # fmt: skip
                db.executemany(
                    "INSERT OR IGNORE INTO shadow_equity (run_id, strategy_id, strategy_version, "
                    "at, cash, open_value, equity, realized_pnl, unrealized_pnl, unresolved_cost, "
                    "exposure_pct, open_positions, drawdown_pct, max_drawdown_pct) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(e["run_id"], e["strategy_id"], e["strategy_version"], e["at"].timestamp(),
                      e["cash"], e["open_value"], e["equity"], e["realized_pnl"],
                      e["unrealized_pnl"], e["unresolved_cost"], e["exposure_pct"],
                      e["open_positions"], e["drawdown_pct"], e["max_drawdown_pct"])
                     for e in equity],
                )  # fmt: skip
                db.executemany(
                    "INSERT OR IGNORE INTO shadow_rejections (rejection_id, run_id, strategy_id, "
                    "strategy_version, asset_id, pool, decision_at, scout_record_id, "
                    "reasons_json, observed_json, fingerprints_json, diagnostics_version, "
                    "recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(x["rejection_id"], x["run_id"], x["strategy_id"], x["strategy_version"],
                      x["asset_id"], x["pool"], x["decision_at"].timestamp(),
                      x["scout_record_id"], json.dumps(x["reasons"]),
                      json.dumps(x["observed"], default=str, sort_keys=True),
                      json.dumps(x["fingerprints"], default=str), DIAGNOSTICS_VERSION, now)
                     for x in rejections],
                )  # fmt: skip
                cur = db.execute(
                    "UPDATE shadow_checkpoints SET cursor_at = ?, cursor_id = ?, "
                    "processed_until = ?, books_json = ?, stats_json = ?, updated_at = ? "
                    "WHERE run_id = ?",
                    (
                        cursor[0],
                        cursor[1],
                        processed_until,
                        json.dumps(books),
                        json.dumps(stats, default=str),
                        now,
                        run_id,
                    ),
                )
                if cur.rowcount != 1:
                    raise ShadowStoreError(f"run {run_id} has no checkpoint")

    def record_metrics(
        self, run_id: str, metrics: Sequence[dict[str, Any]], through: float | None
    ) -> None:
        self._write_guard()
        now = time.time()
        with self._lock:
            db = self._db()
            with db:
                db.executemany(
                    "INSERT INTO shadow_metrics (run_id, strategy_id, strategy_version, "
                    "computed_at, through, metrics_json) VALUES (?, ?, ?, ?, ?, ?)",
                    [(run_id, m["strategy_id"], m["strategy_version"], now, through,
                      json.dumps(m, default=str)) for m in metrics],
                )  # fmt: skip

    # --- reads ------------------------------------------------------------------------------------

    def _rows(self, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._db().execute(sql, list(params))
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(names, r, strict=True)) for r in rows]

    @staticmethod
    def _where(
        run_id: str | None,
        strategy_id: str | None,
        asset_id: str | None,
        time_col: str,
        since: datetime | None,
        until: datetime | None,
    ) -> tuple[str, list[Any]]:
        where: list[str] = ["1 = 1"]
        params: list[Any] = []
        for col, value in (
            ("run_id", run_id),
            ("strategy_id", strategy_id),
            ("asset_id", asset_id),
        ):
            if value is not None:
                where.append(f"{col} = ?")
                params.append(value)
        if since is not None:
            where.append(f"{time_col} >= ?")
            params.append(since.timestamp())
        if until is not None:
            where.append(f"{time_col} < ?")
            params.append(until.timestamp())
        return " AND ".join(where), params

    def positions(
        self,
        run_id: str | None = None,
        strategy_id: str | None = None,
        asset_id: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        where, params = self._where(run_id, strategy_id, asset_id, "entry_at", since, until)
        if status is not None:
            where += " AND status = ?"
            params.append(status)
        return self._rows(f"SELECT * FROM shadow_positions WHERE {where} ORDER BY entry_at", params)

    def trades(
        self,
        run_id: str | None = None,
        strategy_id: str | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        where, params = self._where(run_id, strategy_id, asset_id, "entry_at", since, until)
        return self._rows(f"SELECT * FROM shadow_trades WHERE {where} ORDER BY exit_at, id", params)

    def decisions(
        self,
        run_id: str | None = None,
        strategy_id: str | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        action: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        where, params = self._where(run_id, strategy_id, asset_id, "decision_at", since, until)
        if action is not None:
            where += " AND action = ?"
            params.append(action)
        return self._rows(
            f"SELECT * FROM shadow_decisions WHERE {where} ORDER BY decision_at, id LIMIT ?",
            [*params, limit],
        )

    def equity(self, run_id: str, strategy_id: str | None = None) -> list[dict[str, Any]]:
        where, params = self._where(run_id, strategy_id, None, "at", None, None)
        return self._rows(f"SELECT * FROM shadow_equity WHERE {where} ORDER BY at, id", params)

    # --- rejection diagnostics -----------------------------------------------------------------

    def mark_diagnostics(self, run_id: str, at: datetime) -> None:
        """Record, once, that rejection diagnostics exist for `run_id` from `at` on (earlier
        evaluations of an older run are never reconstructed)."""
        self._write_guard()
        body = json.dumps({"at": at.isoformat(), "diagnostics_version": DIAGNOSTICS_VERSION})
        with self._lock:
            db = self._db()
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO shadow_meta (key, value) VALUES (?, ?)",
                    (f"diagnostics_from:{run_id}", body),
                )

    def diagnostics_from(self, run_id: str) -> dict[str, Any] | None:
        if not self._has("shadow_meta"):
            return None
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT value FROM shadow_meta WHERE key = ?", (f"diagnostics_from:{run_id}",)
                )
                .fetchone()
            )
        return json.loads(row[0]) if row else None

    def _has(self, table: str) -> bool:
        """A read-only v1 file has no v2 tables until a writer opens it once."""
        with self._lock:
            return (
                self._db()
                .execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,))
                .fetchone()
                is not None
            )

    def rejections(
        self,
        run_id: str | None = None,
        strategy_id: str | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        reason: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if not self._has("shadow_rejections"):
            return []
        where, params = self._where(run_id, strategy_id, asset_id, "decision_at", since, until)
        if reason is not None:
            where += " AND EXISTS (SELECT 1 FROM json_each(reasons_json) j WHERE j.value = ?)"
            params.append(reason)
        rows = self._rows(
            f"SELECT * FROM shadow_rejections WHERE {where} ORDER BY decision_at, id LIMIT ?",
            [*params, limit],
        )
        for r in rows:
            r["reasons"] = json.loads(r.pop("reasons_json"))
            r["observed"] = json.loads(r.pop("observed_json"))
        return rows

    def rejection_summary(
        self,
        run_id: str,
        strategy_id: str | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[tuple[str, int], dict[str, Any]]:
        """Per (strategy, version): rejected evaluations and how often each reason occurs."""
        if not self._has("shadow_rejections"):
            return {}
        where, params = self._where(run_id, strategy_id, asset_id, "decision_at", since, until)
        out: dict[tuple[str, int], dict[str, Any]] = {}
        with self._lock:
            db = self._db()
            for sid, version, n in db.execute(
                f"SELECT strategy_id, strategy_version, COUNT(*) FROM shadow_rejections "
                f"WHERE {where} GROUP BY strategy_id, strategy_version",
                params,
            ):
                out[(sid, version)] = {"rejected": n, "reasons": {}}
            for sid, version, code, n in db.execute(
                f"SELECT strategy_id, strategy_version, j.value, COUNT(*) FROM shadow_rejections, "
                f"json_each(reasons_json) j WHERE {where} "
                "GROUP BY strategy_id, strategy_version, j.value",
                params,
            ):
                out[(sid, version)]["reasons"][code] = n
        return out

    def decision_counts(
        self,
        run_id: str,
        strategy_id: str | None = None,
        asset_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[tuple[str, int], dict[str, int]]:
        where, params = self._where(run_id, strategy_id, asset_id, "decision_at", since, until)
        out: dict[tuple[str, int], dict[str, int]] = {}
        with self._lock:
            for sid, version, action, n in self._db().execute(
                f"SELECT strategy_id, strategy_version, action, COUNT(*) FROM shadow_decisions "
                f"WHERE {where} GROUP BY strategy_id, strategy_version, action",
                params,
            ):
                out.setdefault((sid, version), {})[action] = n
        return out

    def counts(self) -> dict[str, int]:
        with self._lock:
            db = self._db()
            return {
                t: int(db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                if db.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (t,)).fetchone()
                else 0
                for t in ("shadow_strategies", "shadow_runs", "shadow_decisions",
                          "shadow_positions", "shadow_trades", "shadow_equity", "shadow_metrics",
                          "shadow_rejections")
            }  # fmt: skip


TIME_COLUMNS = frozenset(
    {"entry_at", "entry_price_at", "last_price_at", "closed_at", "exit_at", "exit_price_at",
     "decision_at", "reference_price_at", "recorded_at", "at"}
)  # fmt: skip


def readable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        for k in TIME_COLUMNS & r.keys():
            if isinstance(r[k], int | float):
                r[k] = datetime.fromtimestamp(r[k], UTC).isoformat()
        for k in ("evidence_json", "fingerprints_json"):
            if isinstance(r.get(k), str):
                r[k.removesuffix("_json")] = json.loads(r.pop(k))
        out.append(r)
    return out
