"""Descriptive wallet history over Radar targets (no wallet-quality score in V1).

For one wallet as of time T: the Radar tokens it had flows in (known by T), and, when a
Scout database is given (read-only), the outcome of the first Scout anchor of each token
observed at or after the wallet's first flow, counting only horizons finalized by T. A
label is never used before it exists, and nothing here feeds a decision.

``meets_minimum_sample`` needs at least ``history_min_unique_tokens`` distinct tokens and
``history_min_observations`` outcome observations: one or two wins never qualify.
"""

import statistics
from datetime import datetime
from typing import Any

from upscale.services.radar.config import RadarSettings
from upscale.services.radar.models import iso, ts
from upscale.services.radar.readonly import connect as connect_read_only
from upscale.services.radar.readonly import tables
from upscale.services.radar.repository import RadarRepository


def _outcome(
    conn: Any, canonical_id: str, entered_at: float, horizon: str, as_of: float
) -> float | None:
    row = conn.execute(
        "SELECT h.return_pct FROM scout_outcome_observations o JOIN scout_outcome_horizons h "
        "ON h.observation_id = o.id WHERE o.canonical_id = ? AND o.observed_at >= ? "
        "AND h.horizon = ? AND h.finalized_at IS NOT NULL AND h.finalized_at <= ? "
        "AND h.return_pct IS NOT NULL ORDER BY o.observed_at LIMIT 1",
        (canonical_id, entered_at, horizon, as_of),
    ).fetchone()
    return float(row[0]) if row else None


def describe_wallet(
    repo: RadarRepository,
    settings: RadarSettings,
    wallet: str,
    as_of: datetime,
    scout_db: str | None = None,
) -> dict[str, Any]:
    t = ts(as_of)
    entries = repo.wallet_entries(wallet, t)
    profile = repo.wallet_profile(wallet)
    out: dict[str, Any] = {
        "wallet": wallet,
        "as_of": as_of.isoformat(),
        "unique_tokens": len({e.canonical_id for e in entries}),
        "early_entries": sum(e.early for e in entries),
        "flows_in_retention": repo.wallet_flow_count(wallet, t),
        "tokens": [
            {
                "canonical_id": e.canonical_id,
                "first_block_time": iso(e.first_block_time),
                "block_time_known_at": iso(e.block_time_known_at),
                "first_known_at": iso(e.first_fetched_at),
                "first_direction": e.first_direction,
                "early": e.early,
                "proven_wallet": e.proven_wallet,
            }
            for e in entries
        ],  # fmt: skip
        "profile": None
        if profile is None or profile.profiled_at > t
        else {
            "age_status": profile.age_status,
            "oldest_block_time": iso(profile.oldest_block_time),
            "funder": profile.funder,
            "funder_status": profile.funder_status,
        },  # fmt: skip
        "outcomes": {"status": "NOT_COLLECTED", "reason": "no Scout database given"},
    }
    conn = connect_read_only(scout_db) if scout_db else None
    returns: list[float] = []
    if conn is not None:
        try:
            if {"scout_outcome_observations", "scout_outcome_horizons"} <= tables(conn):
                for e in entries:
                    if e.first_block_time is None:
                        continue
                    r = _outcome(conn, e.canonical_id, e.first_block_time,
                                 settings.history_horizon, t)  # fmt: skip
                    if r is not None:
                        returns.append(r)
        finally:
            conn.close()
        n = len(returns)
        out["outcomes"] = {
            "status": "AVAILABLE" if n else "UNAVAILABLE",
            "horizon": settings.history_horizon,
            "observations": n,
            "median_return_pct": statistics.median(returns) if n else None,
            "hit_rate": sum(r > 0 for r in returns) / n if n else None,
            "catastrophic_count": sum(r <= settings.catastrophic_return_pct for r in returns),
            "note": "first Scout anchor at/after the wallet's first flow; finalized by as_of",
        }
    out["meets_minimum_sample"] = (
        out["unique_tokens"] >= settings.history_min_unique_tokens
        and len(returns) >= settings.history_min_observations
    )
    out["minimums"] = {
        "unique_tokens": settings.history_min_unique_tokens,
        "observations": settings.history_min_observations,
    }
    return out
