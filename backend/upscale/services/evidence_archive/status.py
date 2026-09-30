"""Archive status, coverage and replay readiness. Factual counts only: nothing here says a
strategy works or that UpScale is ready to trade.

Coverage is measured over Growth Scout observations (one per token per ranking run): the
moments Replay Lab can sample. For each, is there archived evidence at or before it?

* market: a market / DEX record of the token within `market_window` before it;
* safety: an AVAILABLE on-chain safety record within `safety_max_age` before it;
* social: an AVAILABLE social record within `social_max_age` before it;
* decision-grade: market + safety (what production Opportunity needs for a DEX token);
* replay-usable: decision-grade and old enough for every outcome horizon to have elapsed.
"""

import bisect
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from upscale.services.evidence_archive.store import EvidenceStore

NOT_A_READINESS_CLAIM = (
    "Evidence coverage only: counts of archived point-in-time evidence. Not a measure of "
    "profitability or of readiness to trade."
)
MARKET_WINDOW = timedelta(minutes=10)
SAFETY_MAX_AGE = timedelta(minutes=60)
SOCIAL_MAX_AGE = timedelta(minutes=60)
OUTCOME_SETTLE = timedelta(hours=24, minutes=5)
SPLIT_PCT = {"CALIBRATION": 70, "VALIDATION": 15, "HOLDOUT": 15}


def _within(times: list[float], at: float, window: timedelta) -> bool:
    i = bisect.bisect_right(times, at)
    return i > 0 and at - times[i - 1] <= window.total_seconds()


def status(
    store: EvidenceStore,
    now: datetime,
    days: float = 7.0,
    recorder_stats: dict[str, Any] | None = None,
    replay_db: str | None = None,
) -> dict[str, Any]:
    since = now - timedelta(days=days)
    rows = store.index(since)
    totals = store.totals()
    by_kind: dict[str, dict[str, list[float]]] = {}
    missing: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for kind, asset, at, availability, provider, reason in rows:
        if availability == "AVAILABLE":
            k = "market" if kind == "dex_market" else kind
            by_kind.setdefault(k, {}).setdefault(asset, []).append(at)
        else:
            missing[f"{kind}: {availability}: {reason[:120]}"] += 1
            if availability in ("PROVIDER_FAILED", "RATE_LIMITED"):
                failures[f"{provider or 'unknown'} ({availability})"] += 1
    for assets in by_kind.values():
        for times in assets.values():
            times.sort()

    def has(kind: str, asset: str, at: float, window: timedelta) -> bool:
        return _within(by_kind.get(kind, {}).get(asset, []), at, window)

    observations = sorted(
        {(asset, at) for asset, times in by_kind.get("scout", {}).items() for at in times},
        key=lambda x: x[1],
    )
    counts = Counter[str]()
    settled = (now - OUTCOME_SETTLE).timestamp()
    for asset, at in observations:
        market = has("market", asset, at, MARKET_WINDOW)
        safety = has("safety", asset, at, SAFETY_MAX_AGE)
        social = has("social", asset, at, SOCIAL_MAX_AGE)
        counts["market"] += market
        counts["safety"] += safety
        counts["social"] += social
        grade = market and safety
        counts["decision_grade"] += grade
        counts["outcomes_elapsed"] += at <= settled
        counts["replay_usable"] += grade and at <= settled
    n = len(observations)

    def pct(k: str) -> float | None:
        return round(100 * counts[k] / n, 1) if n else None

    recent = [r for r in rows if r[2] >= (now - timedelta(hours=24)).timestamp()]
    usable = counts["replay_usable"]
    return {
        "label": NOT_A_READINESS_CLAIM,
        "database": store.path,
        "window_days": days,
        "total_snapshots": totals["total"],
        "snapshots_by_kind": totals["by_kind"],
        "distinct_assets": totals["assets"],
        "snapshots_per_hour_last_24h": round(len(recent) / 24, 2),
        "oldest_snapshot": totals["oldest"].isoformat() if totals["oldest"] else None,
        "newest_snapshot": totals["newest"].isoformat() if totals["newest"] else None,
        "last_archive_write": totals["last_write"].isoformat() if totals["last_write"] else None,
        "coverage": {
            "scout_observations": n,
            "market_pct": pct("market"),
            "safety_pct": pct("safety"),
            "social_pct": pct("social"),
            "decision_grade_pct": pct("decision_grade"),
            "definitions": {
                "market": f"market/DEX evidence within {MARKET_WINDOW} before",
                "safety": f"AVAILABLE on-chain safety within {SAFETY_MAX_AGE} before",
                "social": f"AVAILABLE social within {SOCIAL_MAX_AGE} before",
            },
        },
        "readiness": {
            "timestamps": n,
            "with_complete_market_evidence": counts["market"],
            "with_onchain_safety": counts["safety"],
            "with_social_evidence": counts["social"],
            "full_opportunity_evaluation_possible": counts["decision_grade"],
            "outcome_horizons_elapsed": counts["outcomes_elapsed"],
            "replay_usable": usable,
            "projected_split_if_all_replayed": {k: usable * v // 100 for k, v in SPLIT_PCT.items()},
            "replay_lab": replay_counts(replay_db),
        },
        "missing_by_reason": dict(missing.most_common(25)),
        "provider_failures": dict(failures),
        "recorder": recorder_stats,
    }


def replay_counts(path: str | None) -> dict[str, Any] | None:
    """What Replay Lab has already measured (read-only; None without a replay database)."""
    if not path or not Path(path).expanduser().exists():
        return None
    conn = sqlite3.connect(f"file:{Path(path).expanduser().resolve()}?mode=ro", uri=True)
    try:
        splits = dict(
            conn.execute(
                "SELECT split, COUNT(*) FROM replay_samples WHERE status = 'COMPLETE' GROUP BY split"
            ).fetchall()
        )
        complete = conn.execute(
            "SELECT COUNT(*) FROM (SELECT sample_id FROM replay_outcomes "
            "WHERE status IN ('COMPLETE', 'PARTIAL') GROUP BY sample_id HAVING COUNT(*) >= 5)"
        ).fetchone()[0]
        with_safety = conn.execute(
            "SELECT COUNT(*) FROM replay_decisions WHERE record_json LIKE '%\"onchain_safety\": {%' "
            "OR record_json LIKE '%\"onchain_safety\":{%'"
        ).fetchone()[0]
    except sqlite3.DatabaseError as exc:
        return {"error": str(exc)}
    finally:
        conn.close()
    return {
        "completed_samples_by_split": {k: int(v) for k, v in splits.items()},
        "samples_with_all_5_horizons_measured": int(complete),
        "decisions_with_archived_onchain_safety": int(with_safety),
    }
