"""Historical replay from available stored snapshots. NOT a backtest.

Given a Growth Scout ranking stored at time T (``scout_growth_stages``), what did Scout's
own later snapshots of the same token *and the same pool* record afterward? The reference
is the snapshot at (or just before) T; later snapshots are only those Scout happened to
take, so extremes between them are unknown and excursions are lower bounds. Nothing is
fetched, estimated or written: the database is opened read-only.

Run: ``python -m upscale.services.outcomes.replay [--db PATH] [--limit N]``
"""

import argparse
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from upscale.services.outcomes.config import HorizonSpec, OutcomeConfig
from upscale.services.outcomes.metrics import pct

REPLAY_LABEL = (
    "historical replay from available stored snapshots: not a backtest (only the prices "
    "Scout happened to record; no candles, no fills, no trades assumed)"
)


class ReplayHorizon(BaseModel):
    horizon: str
    status: str  # MEASURED / NO_SNAPSHOT_NEAR_END / NOT_ELAPSED
    snapshots: int  # later snapshots of the same pool inside the window
    end_price: float | None = None
    end_price_at: datetime | None = None
    return_pct: float | None = None
    mfe_lower_bound_pct: float | None = None
    mae_lower_bound_pct: float | None = None
    future_stage: str | None = None


class ReplayObservation(BaseModel):
    canonical_id: str
    symbol: str | None
    run_at: datetime
    stage: str
    rank: int | None
    score: float | None
    pool_address: str
    reference_price: float
    reference_at: datetime
    horizons: list[ReplayHorizon]


class ReplayReport(BaseModel):
    label: str = REPLAY_LABEL
    database: str
    runs: int
    observations: list[ReplayObservation]
    skipped: dict[str, int] = Field(default_factory=dict)


def replay(
    path: str | Path,
    horizons: tuple[HorizonSpec, ...] = OutcomeConfig().horizons,
    reference_tolerance: timedelta = timedelta(minutes=10),
    end_tolerance_fraction: float = 0.1,
    min_end_tolerance: timedelta = timedelta(minutes=3),
    limit: int = 200,
    canonical_id: str | None = None,
    ranked_only: bool = True,
) -> ReplayReport:
    uri = f"file:{Path(path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return _replay(conn, str(path), horizons, reference_tolerance, end_tolerance_fraction,
                       min_end_tolerance, limit, canonical_id, ranked_only)  # fmt: skip
    finally:
        conn.close()


def _replay(
    conn: sqlite3.Connection,
    path: str,
    horizons: tuple[HorizonSpec, ...],
    ref_tol: timedelta,
    end_frac: float,
    min_end_tol: timedelta,
    limit: int,
    canonical_id: str | None,
    ranked_only: bool,
) -> ReplayReport:
    where = ["1 = 1"]
    params: list[object] = []
    if ranked_only:
        where.append("g.rank IS NOT NULL")
    if canonical_id:
        where.append("g.canonical_id = ?")
        params.append(canonical_id)
    rows = conn.execute(
        f"""
        SELECT g.canonical_id, g.computed_at, g.stage, g.rank, g.score, t.symbol
        FROM scout_growth_stages g LEFT JOIN scout_tokens t USING (canonical_id)
        WHERE {" AND ".join(where)}
        ORDER BY g.computed_at, g.rank LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    runs = conn.execute("SELECT COUNT(DISTINCT computed_at) FROM scout_growth_stages").fetchone()
    latest = conn.execute("SELECT MAX(observed_at) FROM scout_snapshots").fetchone()[0] or 0.0
    skipped: dict[str, int] = {}
    out: list[ReplayObservation] = []
    for cid, run_at, stage, rank, score, symbol in rows:
        ref = conn.execute(
            """
            SELECT observed_at, pool_address, price_usd FROM scout_snapshots
            WHERE canonical_id = ? AND observed_at <= ? AND observed_at >= ?
                AND price_usd IS NOT NULL
            ORDER BY observed_at DESC LIMIT 1
            """,
            (cid, run_at, run_at - ref_tol.total_seconds()),
        ).fetchone()
        if ref is None:
            skipped["no reference snapshot at the ranking time"] = (
                skipped.get("no reference snapshot at the ranking time", 0) + 1
            )
            continue
        t0, pool, p0 = ref
        replayed = []
        for h in horizons:
            end = t0 + h.minutes * 60
            tol = max(min_end_tol.total_seconds(), end_frac * h.minutes * 60)
            if end > latest:
                replayed.append(ReplayHorizon(horizon=h.label, status="NOT_ELAPSED", snapshots=0))
                continue
            later = conn.execute(
                """
                SELECT observed_at, price_usd FROM scout_snapshots
                WHERE canonical_id = ? AND pool_address = ? AND observed_at > ?
                    AND observed_at <= ? AND price_usd IS NOT NULL
                ORDER BY observed_at
                """,
                (cid, pool, t0, end + tol),
            ).fetchall()
            inside = [(t, p) for t, p in later if t <= end]
            near = [(t, p) for t, p in later if abs(t - end) <= tol]
            stage_row = conn.execute(
                """
                SELECT stage FROM scout_growth_stages WHERE canonical_id = ?
                    AND computed_at BETWEEN ? AND ? ORDER BY ABS(computed_at - ?) LIMIT 1
                """,
                (cid, end - tol, end + tol, end),
            ).fetchone()
            r = ReplayHorizon(
                horizon=h.label,
                status="MEASURED" if near else "NO_SNAPSHOT_NEAR_END",
                snapshots=len(inside),
                future_stage=stage_row[0] if stage_row else None,
            )
            if near:
                t, p = min(near, key=lambda x: abs(x[0] - end))
                r.end_price, r.end_price_at, r.return_pct = p, _dt(t), pct(p, p0)
            if inside:
                r.mfe_lower_bound_pct = max(0.0, (max(p for _, p in inside) / p0 - 1) * 100)
                r.mae_lower_bound_pct = min(0.0, (min(p for _, p in inside) / p0 - 1) * 100)
            replayed.append(r)
        out.append(
            ReplayObservation(
                canonical_id=cid, symbol=symbol, run_at=_dt(run_at), stage=stage, rank=rank,
                score=score, pool_address=pool, reference_price=p0, reference_at=_dt(t0),
                horizons=replayed,
            )
        )  # fmt: skip
    return ReplayReport(database=path, runs=int(runs[0]), observations=out, skipped=skipped)


def _dt(value: float) -> datetime:
    return datetime.fromtimestamp(value, UTC)


def main() -> None:  # pragma: no cover - a developer utility
    from upscale.config import SCOUT_DB_PATH

    parser = argparse.ArgumentParser(description=REPLAY_LABEL)
    parser.add_argument("--db", default=SCOUT_DB_PATH)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--token", default=None, help="canonical id, e.g. solana:<mint>")
    args = parser.parse_args()
    print(replay(args.db, limit=args.limit, canonical_id=args.token).model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
