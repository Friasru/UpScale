"""Outcome integrity audit (read-only by default).

    python -m upscale.services.outcomes integrity-audit [--db PATH] [--threshold 1000 ...]
        [--observation ID] [--kind scout|decision] [--horizon 1h] [--all]
        [--verify] [--max-requests 40] [--request-spacing 10] [--write] [--json]

* Default: reads the live outcome database read-only (``mode=ro``) and classifies every
  numeric outcome above the smallest threshold with the deterministic checks of
  `integrity` (no network).
* ``--verify``: provider evidence, with as few requests as possible (throttled, capped):
  the pool's orientation per distinct pool (1 request each) and, for rows above 1,000%,
  an independent reconstruction from candles priced for the exact token (one request per
  pool and timeframe when the windows fit in one response).
* ``--write``: appends the classifications to ``outcome_integrity_audits`` (created if
  missing, append-only). Outcome rows are never modified.
"""

import argparse
import asyncio
import json
import math
import sqlite3
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from upscale.services.geckoterminal import GeckoTerminalProvider
from upscale.services.market_data import (
    TIMEFRAME_SECONDS,
    MarketDataError,
    ProviderRateLimitedError,
    Timeframe,
)
from upscale.services.outcomes.integrity import (
    INVALID_STATUSES,
    VALID_STATUSES,
    Verification,
    classify,
    discontinuity,
)
from upscale.services.outcomes.metrics import path_from_candles
from upscale.services.outcomes.models import PricePath
from upscale.services.versions import code_revision

AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcome_integrity_audits (
    audit_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('scout', 'decision')),
    observation_id INTEGER NOT NULL,
    horizon TEXT NOT NULL,
    integrity_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    audited_at REAL NOT NULL,
    reference_price_checked REAL,
    horizon_price_checked REAL,
    expected_return REAL,
    stored_return REAL,
    provider_evidence_json TEXT NOT NULL,
    code_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outcome_integrity_audits_by_row
    ON outcome_integrity_audits (kind, observation_id, horizon, audit_id);
CREATE TRIGGER IF NOT EXISTS outcome_integrity_audits_no_update
BEFORE UPDATE ON outcome_integrity_audits
BEGIN SELECT RAISE(ABORT, 'integrity audits are append-only'); END;
CREATE TRIGGER IF NOT EXISTS outcome_integrity_audits_no_delete
BEFORE DELETE ON outcome_integrity_audits
BEGIN SELECT RAISE(ABORT, 'integrity audits are append-only'); END;
"""
_TABLES = {"scout": ("scout_outcome_observations", "scout_outcome_horizons"),
           "decision": ("decision_observations", "decision_outcome_horizons")}  # fmt: skip


@dataclass
class AuditRow:
    kind: str
    observation_id: int
    horizon: str
    horizon_minutes: int
    status: str  # the horizon's outcome status
    return_pct: float
    mfe_pct: float | None
    mae_pct: float | None
    market_status: str | None
    attempts: int
    missing: list[str]
    path: PricePath | None
    body: dict[str, Any]
    identity: dict[str, Any] = field(default_factory=dict)
    integrity: str = "UNAUDITED"
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def token(self) -> str | None:
        return self.identity.get("token")

    def report(self) -> dict[str, Any]:
        p = self.path
        return {
            **self.identity,
            "kind": self.kind, "observation_id": self.observation_id, "horizon": self.horizon,
            "reference_price": p.reference_price if p else None,
            "horizon_at": p.end_price_at.isoformat() if p and p.end_price_at else None,
            "horizon_price": p.end_price if p else None, "return_pct": self.return_pct,
            "mfe_pct": self.mfe_pct, "mae_pct": self.mae_pct,
            "max_drawdown_pct": p.max_drawdown_pct if p else None,
            "path": {"source": p.source, "provider": p.provider, "timeframe": p.timeframe, "points": p.points,
                     "lowest": p.lowest_price, "highest": p.highest_price} if p else None,
            "discontinuity_factor": discontinuity(p) if p else None,
            "market_status": self.market_status, "outcome_status": self.status,
            "attempts": self.attempts, "missing": self.missing,
            "integrity": self.integrity, "reason": self.reason, "evidence": self.evidence,
        }  # fmt: skip


def _identity(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    if kind == "scout":
        return {
            "canonical_id": body["canonical_id"], "chain": body["chain"], "token": body["address"],
            "symbol": body.get("symbol"), "pool": body["pool_address"], "dex": body.get("pool_dex"),
            "quote_symbol": body.get("quote_symbol"), "market_provider": body.get("market_provider"),
            "reference_at": body["observed_at"], "production_version": f"outcome schema v{body.get('schema_version')}",
        }  # fmt: skip
    ref = body.get("price_ref") or {}
    return {
        "canonical_id": body.get("asset_id"), "chain": body.get("chain"), "token": ref.get("token_address"),
        "symbol": body.get("symbol"), "pool": ref.get("pool_address"), "dex": None,
        "scout_observation_id": body.get("scout_observation_id"), "reference_at": body.get("analyzed_at"),
        "production_version": f"outcome schema v{body.get('schema_version')}",
    }  # fmt: skip


def load_rows(
    db: str | Path,
    min_abs_return: float | None,
    observation: int | None = None,
    kind: str | None = None,
    horizon: str | None = None,
) -> list[AuditRow]:
    conn = sqlite3.connect(f"file:{Path(db).expanduser().resolve()}?mode=ro", uri=True)
    try:
        out: list[AuditRow] = []
        for k, (obs, table) in _TABLES.items():
            if kind and k != kind:
                continue
            where = ["h.return_pct IS NOT NULL"]
            params: list[Any] = []
            if min_abs_return is not None:
                where.append("ABS(h.return_pct) > ?")
                params.append(min_abs_return)
            if observation is not None:
                where.append("h.observation_id = ?")
                params.append(observation)
            if horizon:
                where.append("h.horizon = ?")
                params.append(horizon)
            rows = conn.execute(
                f"""
                SELECT h.observation_id, h.horizon, h.horizon_minutes, h.status, h.return_pct, h.mfe_pct,
                    h.mae_pct, h.market_status, h.attempts, h.missing_json, h.price_json, o.body_json
                FROM {table} h JOIN {obs} o ON o.id = h.observation_id
                WHERE {" AND ".join(where)} ORDER BY h.observation_id, h.horizon_minutes
                """,
                params,
            ).fetchall()
            for r in rows:
                body = json.loads(r[11])
                out.append(AuditRow(
                    kind=k, observation_id=r[0], horizon=r[1], horizon_minutes=r[2], status=r[3],
                    return_pct=r[4], mfe_pct=r[5], mae_pct=r[6], market_status=r[7], attempts=r[8],
                    missing=json.loads(r[9] or "[]"),
                    path=PricePath.model_validate_json(r[10]) if r[10] else None,
                    body=body, identity=_identity(k, body),
                ))  # fmt: skip
        return out
    finally:
        conn.close()


class Verifier:
    """Throttled, capped provider evidence (GeckoTerminal)."""

    def __init__(
        self,
        provider: GeckoTerminalProvider | None = None,
        max_requests: int = 40,
        spacing_seconds: float = 15.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.provider = provider or GeckoTerminalProvider()
        self.max_requests = max_requests
        self.spacing = spacing_seconds
        self.sleep = sleep
        self.now = now
        self.requests = 0
        self.skipped: list[str] = []

    async def _call(self, what: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        if self.requests >= self.max_requests:
            self.skipped.append(f"{what}: request cap reached")
            return None
        if self.requests:
            await self.sleep(self.spacing)
        self.requests += 1
        try:
            return await fn()
        except ProviderRateLimitedError:
            if self.requests >= self.max_requests:
                self.skipped.append(f"{what}: rate limited, request cap reached")
                return None
            await self.sleep(self.spacing * 4)  # one back-off retry
            self.requests += 1
            try:
                return await fn()
            except MarketDataError as exc:
                self.skipped.append(f"{what}: {exc}")
                return None
        except MarketDataError as exc:
            self.skipped.append(f"{what}: {exc}")
            return None

    def _pool(self, chain: str, pool: str) -> Callable[[], Awaitable[Any]]:
        async def fetch() -> Any:
            return await self.provider.pool_tokens(chain, pool)

        return fetch

    def _candles(
        self, chain: str, pool: str, token: str, timeframe: Timeframe, limit: int, before: datetime
    ) -> Callable[[], Awaitable[Any]]:
        async def fetch() -> Any:
            return await self.provider.fetch_pool_candles(
                chain, pool, timeframe, limit, symbol=token, canonical_id=f"{chain}:{token}",
                now=self.now(), before=before, contiguous=False, token=token,
            )  # fmt: skip

        return fetch

    async def verify(
        self, rows: Sequence[AuditRow], reconstruct_above: float
    ) -> dict[tuple[str, int, str], Verification]:
        out: dict[tuple[str, int, str], Verification] = {}
        dex = [r for r in rows if r.path is not None and r.path.source == "candles" and r.identity.get("pool")
               and r.identity.get("chain") and r.token]  # fmt: skip
        pools: dict[tuple[str, str], tuple[str | None, str | None]] = {}
        for chain, pool in sorted({(r.identity["chain"], r.identity["pool"]) for r in dex}):
            got = await self._call(f"pool {pool}", self._pool(chain, pool))
            if got is not None:
                pools[(chain, pool)] = got
        groups: dict[tuple[str, str, str, str], list[AuditRow]] = {}
        for r in dex:
            if abs(r.return_pct) > reconstruct_above and r.path and r.path.timeframe:
                groups.setdefault(
                    (r.identity["chain"], r.identity["pool"], r.token or "", r.path.timeframe), []
                ).append(r)
        recon: dict[tuple[str, int, str], PricePath] = {}
        for (chain, pool, token, tf), members in sorted(groups.items()):
            timeframe: Timeframe = tf  # type: ignore[assignment]
            interval = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            start = min(m.path.window_start for m in members if m.path)
            end = max(m.path.window_end for m in members if m.path)
            need = math.ceil((end - start) / interval) + 2
            if need > 1000:
                self.skipped.append(f"{pool} {tf}: windows span more than one response")
                continue
            series = await self._call(
                f"candles {pool} {tf}",
                self._candles(chain, pool, token, timeframe, need, end + interval),
            )
            if series is None:
                continue
            for m in members:
                assert m.path is not None
                recon[(m.kind, m.observation_id, m.horizon)] = path_from_candles(
                    series.candles, interval, m.path.reference_price, m.path.window_start, m.path.window_end,
                    provider=series.provider, timeframe=tf, price_drop_pct=90.0,
                )  # fmt: skip
        for r in dex:
            base, quote = pools.get((r.identity["chain"], r.identity["pool"]), (None, None))
            out[(r.kind, r.observation_id, r.horizon)] = Verification(
                pool_base=base, pool_quote=quote, reconstructed=recon.get((r.kind, r.observation_id, r.horizon)),
            )  # fmt: skip
        return out


def audit(
    rows: Sequence[AuditRow], verifications: dict[tuple[str, int, str], Verification] | None = None
) -> None:
    for r in rows:
        v = (verifications or {}).get((r.kind, r.observation_id, r.horizon))
        r.integrity, r.reason, r.evidence = classify(r.path, r.return_pct, r.token, v)


def summary(rows: Sequence[AuditRow], thresholds: Sequence[float]) -> dict[str, Any]:
    status = Counter(r.integrity for r in rows)
    return {
        "rows_checked": len(rows),
        "above_threshold": {
            f">{t:g}%": sum(1 for r in rows if abs(r.return_pct) > t) for t in sorted(thresholds)
        },
        "valid": sum(n for s, n in status.items() if s in VALID_STATUSES),
        "invalid": sum(n for s, n in status.items() if s in INVALID_STATUSES),
        "unknown": status.get("UNKNOWN_INTEGRITY", 0),
        "by_status": dict(status),
        "by_provider": dict(
            Counter(f"{(r.path.provider if r.path else None)}: {r.integrity}" for r in rows)
        ),
        "by_chain": dict(Counter(f"{r.identity.get('chain')}: {r.integrity}" for r in rows)),
        "by_pool": dict(
            Counter(
                f"{r.identity.get('symbol')} {r.identity.get('pool')}: {r.integrity}" for r in rows
            )
        ),
        "invalid_observations": sorted(
            {(r.kind, r.observation_id) for r in rows if r.integrity in INVALID_STATUSES}
        ),
    }


def write_audits(db: str | Path, rows: Sequence[AuditRow], at: float | None = None) -> int:
    """Append the classifications (creating the sidecar table if needed). Never touches
    the outcome tables."""
    conn = sqlite3.connect(Path(db).expanduser())
    try:
        conn.executescript(AUDIT_SCHEMA)
        now = at if at is not None else time.time()
        version = code_revision()
        with conn:
            conn.executemany(
                """
                INSERT INTO outcome_integrity_audits (kind, observation_id, horizon, integrity_status, reason,
                    audited_at, reference_price_checked, horizon_price_checked, expected_return,
                    stored_return, provider_evidence_json, code_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.kind,
                        r.observation_id,
                        r.horizon,
                        r.integrity,
                        r.reason,
                        now,
                        r.path.reference_price if r.path else None,
                        r.evidence.get("reconstructed_end_price"),
                        r.evidence.get("reconstructed_return_pct"),
                        r.return_pct,
                        json.dumps(r.evidence, default=str),
                        version,
                    )
                    for r in rows
                ],  # fmt: skip
            )
        return len(rows)
    finally:
        conn.close()


def latest_audits(conn: sqlite3.Connection) -> dict[tuple[str, int, str], tuple[str, str]]:
    """(kind, observation id, horizon) -> the latest (status, reason); {} without the table."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "outcome_integrity_audits" not in tables:
        return {}
    out: dict[tuple[str, int, str], tuple[str, str]] = {}
    for kind, oid, horizon, status, reason in conn.execute(
        "SELECT kind, observation_id, horizon, integrity_status, reason FROM outcome_integrity_audits "
        "ORDER BY audit_id"
    ):
        out[(kind, oid, horizon)] = (status, reason)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from upscale.config import SCOUT_DB_PATH

    p = argparse.ArgumentParser(prog="python -m upscale.services.outcomes integrity-audit")
    p.add_argument("--db", default=SCOUT_DB_PATH)
    p.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=None,
        help="repeatable; default 100, 500, 1000",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="audit every numeric outcome, not only above the thresholds",
    )
    p.add_argument("--observation", type=int, default=None)
    p.add_argument("--kind", choices=("scout", "decision"), default=None)
    p.add_argument("--horizon", default=None)
    p.add_argument(
        "--verify", action="store_true", help="provider evidence (a few throttled requests)"
    )
    p.add_argument("--max-requests", type=int, default=40)
    p.add_argument("--request-spacing", type=float, default=15.0)
    p.add_argument(
        "--write",
        action="store_true",
        help="append classifications (outcome rows are never changed)",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    thresholds = args.threshold or [100.0, 500.0, 1000.0]
    db = Path(args.db).expanduser()
    if not db.exists():
        print(f"no outcome database at {db}")
        return 1
    rows = load_rows(
        db, None if args.all else min(thresholds), args.observation, args.kind, args.horizon
    )
    verifier = None
    verifications = None
    if args.verify:
        verifier = Verifier(max_requests=args.max_requests, spacing_seconds=args.request_spacing)
        verifications = asyncio.run(verifier.verify(rows, reconstruct_above=1000.0))
    audit(rows, verifications)
    report: dict[str, Any] = {
        "mode": "WRITE" if args.write else "READ_ONLY",
        "database": str(db),
        "summary": summary(rows, thresholds),
        "provider_requests": verifier.requests if verifier else 0,
        "verification_gaps": verifier.skipped if verifier else [],
        "rows_above_1000": [r.report() for r in rows if abs(r.return_pct) > 1000],
    }
    if args.write:
        report["written"] = write_audits(db, rows)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        s: dict[str, Any] = report["summary"]
        print(
            f"{report['mode']} audit of {db}: {s['rows_checked']} rows; {json.dumps(s['above_threshold'])}"
        )
        print(
            f"valid {s['valid']}, invalid {s['invalid']}, unknown {s['unknown']}; by status {json.dumps(s['by_status'])}"
        )
        print(f"by pool: {json.dumps(s['by_pool'], ensure_ascii=False)}")
        print(
            f"provider requests: {report['provider_requests']}; gaps: {report['verification_gaps']}"
        )
        for r in report["rows_above_1000"]:
            print(f"- {r['kind']} {r['observation_id']} {r['horizon']} {r['symbol']} {r['token']} pool {r['pool']} "
                  f"({r['dex']}, quote {r.get('quote_symbol')}): ref {r['reference_price']} @ {r['reference_at']} -> "
                  f"{r['horizon_price']} @ {r['horizon_at']}: {r['return_pct']:+,.0f}% "
                  f"[{r['integrity']}] {r['reason']}")  # fmt: skip
        if args.write:
            print(f"appended {report['written']} classification(s) to outcome_integrity_audits")
    return 0
