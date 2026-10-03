"""Shadow strategy validation report (`report`): a daily / checkpoint view of one run, per
strategy and side by side. Read-only: the shadow database and the Evidence Archive are
opened ``mode=ro`` and nothing else is read; no provider or network request, no write, no
effect on any book. Decisions, trades, positions and equity are reported as stored.

Window (``--since`` inclusive, ``--until`` exclusive): decisions by decision time,
positions opened by entry time, trades closed by their final fill's exit time. The account
(equity, cash, P/L, drawdown) is run-to-date *as of* the window end: the latest checkpoint
when the window reaches the processed time, otherwise the last equity snapshot before
``--until`` (the stored marks of open positions are current, so a past ``--until`` never
reads them).

Execution (`execution` section): IDEALIZED_NO_FEES fills have no friction (gross = net);
REALISTIC_V1 fills are read from ``shadow_executions`` (fees, slippage, impact, latency
drift, execution delay, gross vs net P/L), and pending entry / exit intents from the
checkpoint. `compare_runs` puts two runs side by side (e.g. continuous-v2 against
continuous-v2-realistic) as factual deltas, never a ranking.

Anti-lookahead: every figure comes from rows the book wrote at the time, or from archived
evidence observed no later than the report's as-of time. Market availability over a held
position's life is rebuilt with the book's own rules (`Position.market_state`): at each
instant only observations up to that instant count. Nothing here feeds a decision.
"""

import json
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, get_args

from upscale.services.evidence_archive.store import EvidenceRecord, EvidenceStore
from upscale.services.shadow.book import (
    LIQUIDITY_COLLAPSE_USD,
    MIN_TIMING_VERSION,
    PRICE_STALE_AFTER_MINUTES,
    Position,
    latency_drift_usd,
)
from upscale.services.shadow.config import (
    CLEAN_DATA_CUTOFF,
    EXECUTION_NOTE,
    NOT_REAL_PROFIT,
    REALISTIC_NOTE,
    AvailabilityPolicy,
    ExitReason,
    StrategyConfig,
    run_execution,
    run_policy,
)
from upscale.services.shadow.engine import ShadowError
from upscale.services.shadow.evidence import is_watch, market_price, scout_view, watch_status
from upscale.services.shadow.metrics import (
    MIN_ASSETS,
    MIN_TRADES,
    closed_positions,
    effective_sample_size,
    last_mark_series,
    trimmed_mean,
)
from upscale.services.shadow.store import ShadowStore

COMPARED = ("scout_threshold", "scout_technical", "scout_safety_technical", "random_eligible")
EXIT_REASONS: tuple[str, ...] = get_args(ExitReason)
ACTIONS = ("ENTER", "HOLD", "EXIT", "NO_ACTION")
STATES = (
    "MARKET_AVAILABLE", "LIQUIDITY_COLLAPSE", "PRICE_STALE", "PROVIDER_UNAVAILABLE",
    "EVIDENCE_GAP", "MARKET_NOT_FOUND",
)  # fmt: skip
FRESH = ("MARKET_AVAILABLE", "LIQUIDITY_COLLAPSE")  # a price within PRICE_STALE_AFTER_MINUTES
GAP = ("PROVIDER_UNAVAILABLE", "EVIDENCE_GAP", "MARKET_NOT_FOUND")
WATCH_STATUSES = (
    "PRICED", "LIQUIDITY_COLLAPSE", "NOT_FOUND", "NOT_LISTED", "PROVIDER_FAILED", "RATE_LIMITED",
)  # fmt: skip
IDEALIZED_WARNING = (
    "IDEALIZED_NO_FEES is NOT live-realistic performance: fills at the observed exact-pool "
    "price with no fees, slippage, latency or price impact. Real trading would do worse."
)
REALISTIC_WARNING = (
    "REALISTIC_V1 is still a simulation: fixed latency, fixed adverse slippage and fees, "
    "optional constant-product impact; no gas, MEV, failed transactions or partial fills. "
    "Not real profit."
)
NOT_RANKED = (
    "Side by side in a fixed order, NOT ranked: no winner is selected. Strategies flagged "
    "INSUFFICIENT_SAMPLE cannot be compared meaningfully yet."
)
EXCURSION_NOTE = (
    "MFE / MAE: the best / worst observed exact-pool price while held, relative to entry, "
    "as the book recorded it at each fill (causal: observations up to the fill only). "
    "Discrete observations: the true intrabar extremes are unknown."
)


def _dt(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, UTC) if ts is not None else None


def _iso(t: datetime | None) -> str | None:
    return t.isoformat() if t is not None else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    """The q-th percentile (0-100), linear between the closest ranks."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


DELAY_BUCKETS_MINUTES = (5, 15, 30, 60)


def delay_stats(delays: Sequence[float]) -> dict[str, Any]:
    """Observed execution delays (seconds): median, p90, max and the share filled within
    each bucket (intent to fill observation, inclusive)."""
    return {
        "median_execution_delay_seconds": _median(delays),
        "p90_execution_delay_seconds": _percentile(delays, 90),
        "max_execution_delay_seconds": max(delays) if delays else None,
        "filled_within_pct": {
            f"{b}m": sum(d <= b * 60 for d in delays) / len(delays) * 100 if delays else None
            for b in DELAY_BUCKETS_MINUTES
        },
    }


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


@dataclass(frozen=True)
class Window:
    since: datetime | None
    until: datetime | None
    as_of: datetime  # min(until, processed time)
    current: bool  # the window reaches the processed time: the checkpoint is the account

    def before_end(self, ts: float) -> bool:
        if self.until is not None:
            return ts < self.until.timestamp()
        return ts <= self.as_of.timestamp()

    def contains(self, ts: float) -> bool:
        return self.before_end(ts) and (self.since is None or ts >= self.since.timestamp())


# --- market availability (rebuilt with the book's own rules) -------------------------------------


def _position(row: dict[str, Any]) -> Position:
    """A stored position at its entry: only the facts `market_state` reads change below."""
    at = _dt(row["entry_at"])
    price_at = _dt(row["entry_price_at"])
    assert at is not None and price_at is not None
    return Position(
        position_id=row["position_id"], asset_id=row["asset_id"], chain=row["chain"],
        address=row["address"], symbol=row["symbol"], pool=row["pool"], dex=row["dex"],
        entry_decision_id=row["entry_decision_id"], entry_at=at, entry_price=row["entry_price"],
        entry_price_at=price_at, quantity=row["quantity"], cost_usd=row["cost_usd"],
        remaining_quantity=row["quantity"], last_price=row["entry_price"],
        last_price_at=price_at, peak_price=row["entry_price"], trough_price=row["entry_price"],
    )  # fmt: skip


def _apply(p: Position, r: EvidenceRecord, policy: AvailabilityPolicy) -> str | None:
    """Update `p`'s availability facts as `Book.on_price` / `Book.on_status` would for
    this record; returns the watch status of a held-position watch record of the pool."""
    v2 = policy == "EVIDENCE_AWARE_V2"
    watch: str | None = None
    if (
        r.kind == "market"
        and is_watch(r)
        and (r.payload["watch"].get("pool") or r.pool_address) == p.pool
    ):
        watch = r.payload["watch"].get("status")
    if r.kind == "scout":
        s = scout_view(r)
        if (
            s.causal_valid is False
            or s.timing_version is None
            or s.timing_version < MIN_TIMING_VERSION
        ):
            return watch
        obs = s.price
    else:
        obs = market_price(r)
        if obs is None:
            st = watch_status(r)
            if st is None or not v2 or st.pool != p.pool or st.at <= p.last_price_at:
                return watch
            p.watch_status, p.watch_status_at = st.status, st.at
            if st.status == "NOT_FOUND" and st.authoritative:
                p.not_found_count += 1
                p.not_found_since = p.not_found_since or st.at
            return watch
        if obs.source == "watch" and not v2:
            return watch  # LEGACY_V1 books ignore held-position watch evidence
    if obs is None or obs.asset_id != p.asset_id or obs.pool != p.pool:
        return watch
    if obs.at <= p.last_price_at:
        return watch
    p.last_price_at = obs.at
    if v2:
        p.last_liquidity_usd = obs.liquidity_usd
        p.watch_status = p.watch_status_at = p.not_found_since = None
        p.not_found_count = 0
    return watch


@dataclass
class Held:
    """One position's market availability over its holding time (to the window end)."""

    seconds: dict[str, float]  # inside the window, by state
    seen: set[str]  # every state of its whole holding time (to its exit or the as-of time)
    final_state: str | None  # at the as-of time, if still open then
    watch: dict[str, str]  # watch record id -> status, observed inside the window


def availability(
    row: dict[str, Any],
    records: Sequence[EvidenceRecord],
    policy: AvailabilityPolicy,
    unavailable_after_minutes: float,
    w: Window,
) -> Held:
    p = _position(row)
    start = p.entry_at
    closed = row["closed_at"] is not None and w.before_end(row["closed_at"])
    end = _dt(row["closed_at"]) if closed else w.as_of
    assert end is not None
    lo = max(start, w.since) if w.since is not None else start
    stale = timedelta(minutes=PRICE_STALE_AFTER_MINUTES)
    gap = timedelta(minutes=unavailable_after_minutes)
    out = Held({}, set(), None, {})

    def span(a: datetime, b: datetime) -> None:
        if b <= a:
            return
        cuts = sorted({a, b} | {x for x in (p.last_price_at + stale, p.last_price_at + gap)
                                if a < x < b})  # fmt: skip
        for c0, c1 in zip(cuts, cuts[1:], strict=False):
            state = p.market_state(c0 + (c1 - c0) / 2, unavailable_after_minutes)
            out.seen.add(state)
            a2, b2 = max(c0, lo), c1
            if b2 > a2:
                out.seconds[state] = out.seconds.get(state, 0.0) + (b2 - a2).total_seconds()

    cur = start
    for r in records:
        t = r.observed_at
        if t <= start:
            continue
        if closed and t > end:
            break
        if not closed and not w.before_end(t.timestamp()):
            break
        span(cur, t)
        status = _apply(p, r, policy)
        if status is not None and t >= lo:
            out.watch[r.record_id] = status
        cur = t
    span(cur, end)
    if not closed:
        out.final_state = p.market_state(w.as_of, unavailable_after_minutes)
        out.seen.add(out.final_state)
    return out


def _records(
    evidence: EvidenceStore, asset_id: str, since: datetime, until: datetime
) -> list[EvidenceRecord]:
    rows = [r for kind in ("market", "scout")
            for r in evidence.records(kind=kind, asset_id=asset_id, since=since, until=until,
                                      limit=10_000_000)]  # fmt: skip
    return sorted(rows, key=lambda r: (r.observed_at, r.id))


# --- one strategy --------------------------------------------------------------------------------


def _drawdown(points: Sequence[float], start: float) -> tuple[float, float, float]:
    """(max drawdown %, peak, minimum) of a value series starting at `start`."""
    peak = low = start
    dd = 0.0
    for v in points:
        peak = max(peak, v)
        low = min(low, v)
        dd = min(dd, (v / peak - 1) * 100 if peak > 0 else 0.0)
    return dd, peak, low


def _max_simultaneous(positions: Sequence[dict[str, Any]], w: Window) -> int:
    """Open positions over time from entry / close times (a close at the same time as an
    entry counts first: the book exits before it enters on one event)."""
    events = []
    for p in positions:
        if not w.before_end(p["entry_at"]):
            continue
        events.append((p["entry_at"], 1))
        if p["closed_at"] is not None and w.before_end(p["closed_at"]):
            events.append((p["closed_at"], -1))
    n = best = 0
    for _, d in sorted(events):
        n += d
        best = max(best, n)
    return best


def _trade(c: dict[str, Any], symbols: dict[str, str | None]) -> dict[str, Any]:
    return {k: c[k] for k in ("position_id", "asset_id", "return_pct", "pnl_usd", "exit_reason")} | {
        "symbol": symbols.get(c["position_id"]), "exit_at": _iso(_dt(c["exit_at"]))}  # fmt: skip


def strategy_report(
    shadow: ShadowStore,
    evidence: EvidenceStore | None,
    run: dict[str, Any],
    cfg: StrategyConfig,
    book: dict[str, Any] | None,
    decisions: dict[str, Any],
    w: Window,
    records: dict[str, list[EvidenceRecord]],
) -> dict[str, Any]:
    run_id, sid, version = run["run_id"], cfg.strategy_id, cfg.version
    policy = run_policy(run)
    capital = cfg.risk.initial_capital_usd
    positions = [p for p in shadow.positions(run_id, sid) if p["strategy_version"] == version]
    fills = [t for t in shadow.trades(run_id, sid) if t["strategy_version"] == version
             and w.before_end(t["exit_at"])]  # fmt: skip
    equity = [e for e in shadow.equity(run_id, sid) if e["strategy_version"] == version]
    symbols = {p["position_id"]: p["symbol"] for p in positions}

    # --- sample / activity
    opened = [p for p in positions if w.contains(p["entry_at"])]
    closed_all = closed_positions(fills)  # positions whose final fill is before the window end
    closed = [c for c in closed_all if w.contains(c["exit_at"])]
    closed_ids = {c["position_id"] for c in closed_all}
    open_end = [p for p in positions if w.before_end(p["entry_at"])
                and p["position_id"] not in closed_ids]  # fmt: skip
    resolved = [c for c in closed if c["resolved"]]
    closed_assets = [c["asset_id"] for c in closed]
    d = decisions.get("by_action", {})
    reasons = []
    if len(resolved) < MIN_TRADES:
        reasons.append(f"{len(resolved)} resolved closed trades < {MIN_TRADES}")
    if len(set(closed_assets)) < MIN_ASSETS:
        reasons.append(f"{len(set(closed_assets))} distinct closed assets < {MIN_ASSETS}")
    sample = {
        "decisions": sum(d.values()),
        "decisions_by_action": {a: d.get(a, 0) for a in ACTIONS},
        "positions_opened": len(opened),
        "positions_closed": len(closed),
        "resolved_closed": len(resolved),
        "unresolved_closed": len(closed) - len(resolved),
        "currently_open": len(open_end),
        "unique_assets_entered": len({p["asset_id"] for p in opened}),
        "unique_assets_closed": len(set(closed_assets)),
        "effective_sample_size": round(effective_sample_size(closed_assets), 2),
        "effective_sample_size_note": "closed trades with each asset's trades treated as fully "
        "correlated: n^2 / sum_a n_a^2",
        "status": "INSUFFICIENT_SAMPLE" if reasons else "DESCRIPTIVE",
        "insufficient_reasons": reasons,
        "thresholds": {"min_resolved_trades": MIN_TRADES, "min_distinct_assets": MIN_ASSETS},
        "first_decision_at": decisions.get("first"),
        "last_decision_at": decisions.get("last"),
    }

    # --- account as of the window end
    rows = [e for e in equity if w.before_end(e["at"])]
    marks_all = last_mark_series(equity, shadow.unresolved_marks(run_id, sid, version))
    marks = marks_all[: len(rows)]
    state = (book or {}).get("state")
    if w.current and state:
        held = state["positions"].values()
        open_value = sum(p["remaining_quantity"] * p["last_price"] for p in held)
        open_cost = sum(p["remaining_quantity"] * p["entry_price"] for p in held)
        cash, realized = state["cash"], state["realized_pnl"]
        unresolved_cost = state["unresolved_cost"]
        unresolved_mark = sum(m for _, m in shadow.unresolved_marks(run_id, sid, version))
        account = {"source": "checkpoint", "at": _iso(w.as_of)}
        # Cash reserved by pending REALISTIC_V1 entry intents counts at cost.
        reserved = sum(e["budget_usd"] for e in (state.get("pending_entries") or {}).values())
        equity_now, unrealized = cash + open_value + reserved, open_value - open_cost
    elif rows:
        last = rows[-1]
        cash, realized, unrealized = last["cash"], last["realized_pnl"], last["unrealized_pnl"]
        unresolved_cost, unresolved_mark = last["unresolved_cost"], marks[-1]
        equity_now = last["equity"]
        account = {"source": "equity snapshot", "at": _iso(_dt(last["at"]))}
    else:
        cash, realized, unrealized, unresolved_cost, unresolved_mark = capital, 0.0, 0.0, 0.0, 0.0
        equity_now = capital
        account = {"source": "initial capital (no activity yet)", "at": None}
    series = [e["equity"] for e in rows]
    lm_series = [e["equity"] + m for e, m in zip(rows, marks, strict=True)]
    if account["source"] == "checkpoint":
        series.append(equity_now)
        lm_series.append(equity_now + unresolved_mark)
    dd, peak, low = _drawdown(series, capital)
    lm_dd, _, _ = _drawdown(lm_series, capital)
    window_return = window_dd = None
    if w.since is not None:
        before = [e["equity"] for e in rows if e["at"] < w.since.timestamp()]
        start = before[-1] if before else capital
        window_return = (equity_now / start - 1) * 100 if start > 0 else None
        inside = [e["equity"] for e in rows if e["at"] >= w.since.timestamp()]
        if account["source"] == "checkpoint":
            inside.append(equity_now)
        window_dd = _drawdown(inside, start)[0]

    # --- performance (closed trades in the window, resolved only)
    returns = [c["return_pct"] for c in resolved]
    pnls = [c["pnl_usd"] for c in resolved]
    winners = [c for c in resolved if c["return_pct"] > 0]
    losers = [c for c in resolved if c["return_pct"] < 0]
    gains = sum(c["pnl_usd"] for c in winners)
    losses = -sum(c["pnl_usd"] for c in losers)
    best = max(resolved, key=lambda c: (c["return_pct"], c["position_id"]), default=None)
    worst = min(resolved, key=lambda c: (c["return_pct"], c["position_id"]), default=None)
    performance = {
        "starting_capital_usd": capital,
        "account": account,
        "current_equity_usd": equity_now,
        "cash_usd": cash,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": unrealized,
        "total_return_pct": (equity_now / capital - 1) * 100,
        "last_mark_equity_usd": equity_now + unresolved_mark,
        "last_mark_total_return_pct": ((equity_now + unresolved_mark) / capital - 1) * 100,
        "window_return_pct": window_return,
        "closed_trade_pnl_usd": sum(pnls),
        "median_return_pct": _median(returns),
        "mean_return_pct": _mean(returns),
        "trimmed_mean_return_pct": trimmed_mean(returns),
        "win_rate": len(winners) / len(resolved) if resolved else None,
        "profit_factor": gains / losses if losses > 0 else None,
        "profit_factor_note": None if losses > 0 else "undefined: no losing resolved trade",
        "average_winner_pct": _mean([c["return_pct"] for c in winners]),
        "median_winner_pct": _median([c["return_pct"] for c in winners]),
        "average_winner_usd": _mean([c["pnl_usd"] for c in winners]),
        "average_loser_pct": _mean([c["return_pct"] for c in losers]),
        "median_loser_pct": _median([c["return_pct"] for c in losers]),
        "average_loser_usd": _mean([c["pnl_usd"] for c in losers]),
        "best_trade": _trade(best, symbols) if best else None,
        "worst_trade": _trade(worst, symbols) if worst else None,
        "note": "return statistics: resolved closed trades only (a MARKET_UNAVAILABLE trade "
        "has no return and is never a win, a loss or zero); equity, cash and P/L: "
        "run-to-date as of the window end",
    }  # fmt: skip
    risk = {
        "max_drawdown_pct": dd,
        "last_mark_max_drawdown_pct": lm_dd,
        "window_max_drawdown_pct": window_dd,
        "peak_equity_usd": peak,
        "minimum_equity_usd": low,
        "max_simultaneous_positions": _max_simultaneous(positions, w),
        "unresolved_cost_usd": unresolved_cost,
        "unresolved_last_mark_value_usd": unresolved_mark,
        "unresolved_capital_pct": unresolved_cost / capital * 100,
        "note": "drawdown / peak / minimum: written-off equity at each equity snapshot (after "
        "each Scout decision time) and the as-of account; last_mark_*: unresolved positions "
        "at their last observed (stale) mark, reporting only",
    }

    # --- trade behaviour
    final = {t["position_id"]: t for t in fills if t["final"]}
    exit_decisions = shadow.decisions_by_id([t["exit_decision_id"] for t in fills])
    exits = Counter(c["exit_reason"] for c in closed)
    collapse = sum(
        1 for c in closed if c["exit_reason"] == "SIGNAL_EXIT"
        and "MARKET_COLLAPSE" in (exit_decisions.get(final[c["position_id"]]["exit_decision_id"])
                                  or {}).get("reason", "")
    )  # fmt: skip
    mu_codes: Counter[str] = Counter()
    for c in closed:
        if c["exit_reason"] == "MARKET_UNAVAILABLE":
            dec = exit_decisions.get(final[c["position_id"]]["exit_decision_id"]) or {}
            code = json.loads(dec.get("evidence_json") or "{}").get("reason_code")
            mu_codes[code or "unknown"] += 1
    in_window = [t for t in fills if w.contains(t["exit_at"])]
    delays = []
    for t in in_window:
        decided = exit_decisions.get(t["exit_decision_id"])
        if decided is not None and t["exit_at"] > decided["decision_at"]:
            delays.append((t["exit_at"] - decided["decision_at"]) / 60)
    overshoot = [
        (c["exit_at"] - c["entry_at"]) / 60 - cfg.exit.max_hold_minutes
        for c in closed if c["exit_reason"] == "MAX_HOLD_TIME"
    ]  # fmt: skip
    holding = [c["holding_minutes"] for c in closed]
    behaviour = {
        "exit_reasons": {r: exits.get(r, 0) for r in EXIT_REASONS},
        "take_profit": exits.get("TAKE_PROFIT", 0),
        "stop_loss": exits.get("STOP_LOSS", 0),
        "trailing_stop": exits.get("TRAILING_STOP", 0),
        "signal_exit": exits.get("SIGNAL_EXIT", 0),
        "max_hold": exits.get("MAX_HOLD_TIME", 0),
        "market_unavailable": exits.get("MARKET_UNAVAILABLE", 0),
        "market_unavailable_by_reason_code": dict(sorted(mu_codes.items())),
        "liquidity_collapse_exits": collapse,
        "liquidity_collapse_note": "SIGNAL_EXIT on Scout's MARKET_COLLAPSE (the strategies "
        "have no liquidity-collapse exit reason of their own)",
        "fills_by_reason": dict(sorted(Counter(t["exit_reason"] for t in in_window).items())),
        "average_holding_minutes": _mean(holding),
        "median_holding_minutes": _median(holding),
        "delayed_exit_fills": len(delays),
        "max_exit_fill_delay_minutes": max(delays) if delays else None,
        "delayed_exit_note": "fills after their exit decision (an exit that waited for the "
        "next observed exact-pool price)",
        "max_hold_overshoot_median_minutes": _median(overshoot),
        "max_hold_overshoot_max_minutes": max(overshoot) if overshoot else None,
    }

    # --- excursions
    mfe = [c["mfe_pct"] for c in closed if c["mfe_pct"] is not None]
    mae = [c["mae_pct"] for c in closed if c["mae_pct"] is not None]
    excursion: dict[str, Any] = {
        "closed_trades": len(mfe),
        "mfe_median_pct": _median(mfe), "mfe_mean_pct": _mean(mfe),
        "mae_median_pct": _median(mae), "mae_mean_pct": _mean(mae),
        "note": EXCURSION_NOTE,
    }  # fmt: skip
    if w.current:
        omfe = [max(0.0, p["peak_price"] / p["entry_price"] - 1) * 100 for p in open_end]
        omae = [min(0.0, p["trough_price"] / p["entry_price"] - 1) * 100 for p in open_end]
        excursion["open_to_date"] = {"positions": len(open_end),
                                     "mfe_median_pct": _median(omfe), "mae_median_pct": _median(omae)}  # fmt: skip
    else:
        excursion["open_to_date"] = None
        excursion["open_to_date_note"] = (
            "unavailable for an --until before the processed time: open positions' stored "
            "peak / trough marks are current, not point in time"
        )

    # --- availability / infrastructure
    avail: dict[str, Any] = {"policy": policy}
    if evidence is None:
        avail["available"] = False
        avail["note"] = "no Evidence Archive: held-position availability cannot be rebuilt"
        affected = None
    else:
        held_rows = [p for p in positions if w.before_end(p["entry_at"])
                     and (p["closed_at"] is None or p["position_id"] not in closed_ids
                          or w.since is None or p["closed_at"] >= w.since.timestamp())]  # fmt: skip
        seconds: dict[str, float] = dict.fromkeys(STATES, 0.0)
        ever: Counter[str] = Counter()
        now: Counter[str] = Counter()
        watch: dict[str, str] = {}
        gap_exits = stale_exits = 0
        for p in held_rows:
            h = availability(p, records.get(p["asset_id"], []), policy,
                             cfg.exit.market_unavailable_after_minutes, w)  # fmt: skip
            for label, n in h.seconds.items():
                seconds[label] += n
            ever.update(h.seen)
            watch |= h.watch
            if h.final_state is not None:
                now[h.final_state] += 1
            elif w.contains(p["closed_at"]):
                gap_exits += bool(h.seen & set(GAP))
                stale_exits += "PRICE_STALE" in h.seen
        total = sum(seconds.values())
        fresh = sum(seconds[s] for s in FRESH)
        affected = {
            "exits_after_evidence_gap": gap_exits,
            "exits_after_price_stale": stale_exits,
            "delayed_exit_fills": len(delays),
            "market_unavailable_exits": exits.get("MARKET_UNAVAILABLE", 0),
        }
        avail |= {
            "available": True,
            "open_positions_by_state": {s: now.get(s, 0) for s in STATES},
            "positions_ever_in_state": {s: ever.get(s, 0) for s in STATES},
            "held_hours_by_state": {s: seconds.get(s, 0.0) / 3600 for s in STATES},
            "held_position_hours": total / 3600,
            "fresh_price_time_pct": fresh / total * 100 if total > 0 else None,
            "market_available_time_pct": seconds["MARKET_AVAILABLE"] / total * 100
            if total > 0 else None,
            "watch_records": {s: sum(1 for v in watch.values() if v == s) for s in WATCH_STATUSES},
            "watch_record_ids": sorted(watch),
            "evidence_gap_exits": affected,
            "note": f"states over each held position's life in the window, rebuilt from archived "
            f"exact-pool observations with the book's rules ({policy}): fresh = a valid price "
            f"within {PRICE_STALE_AFTER_MINUTES:g} min (MARKET_AVAILABLE, or LIQUIDITY_COLLAPSE "
            f"below ${LIQUIDITY_COLLAPSE_USD:,.0f} of pool liquidity); EVIDENCE_GAP after "
            f"{cfg.exit.market_unavailable_after_minutes:g} min without one"
            + ("; LEGACY_V1 books ignore held-position watch evidence (counted, not applied)"
               if policy == "LEGACY_V1" else ""),
        }  # fmt: skip
    avail["market_unavailable_exits"] = exits.get("MARKET_UNAVAILABLE", 0)
    avail["liquidity_collapse_exits"] = collapse

    return {
        "strategy": cfg.key,
        "strategy_id": sid,
        "strategy_version": version,
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "sample": sample,
        "performance": performance,
        "risk": risk,
        "trade_behavior": behaviour,
        "excursion": excursion,
        "availability": avail,
        "execution": execution_section(shadow, run, cfg, state, in_window, w),
        "_affected": affected,
    }


def execution_section(
    shadow: ShadowStore,
    run: dict[str, Any],
    cfg: StrategyConfig,
    state: dict[str, Any] | None,
    fills: Sequence[dict[str, Any]],
    w: Window,
) -> dict[str, Any]:
    """Trading friction of the fills in the window and the intents still pending."""
    execution = run_execution(run)
    run_id, sid, version = run["run_id"], cfg.strategy_id, cfg.version
    net = sum(t["pnl_usd"] for t in fills if t["pnl_usd"] is not None)
    out: dict[str, Any] = {
        "execution_model": run["execution_model"],
        "settings": execution.model_dump(mode="json") if execution else None,
        "net_realized_pnl_usd": net,
    }
    pending: dict[str, Any] | None = None
    if w.current and state:
        entries = list((state.get("pending_entries") or {}).values())
        exits = Counter(p["pending_exit"] for p in state["positions"].values()
                        if p.get("pending_exit"))  # fmt: skip
        pending = {
            "entry_intents": len(entries),
            "entry_reserved_usd": sum(e["budget_usd"] for e in entries),
            "exit_intents": sum(exits.values()),
            "exit_intents_by_reason": dict(sorted(exits.items())),
        }
    out["pending"] = pending
    if pending is None:
        out["pending_note"] = "pending intents live in the checkpoint: shown only when the "
        "window reaches the processed time"
    if execution is None:
        return out | {
            "gross_realized_pnl_usd": net, "total_friction_usd": 0.0, "fees_usd": 0.0,
            "slippage_cost_usd": 0.0, "price_impact_cost_usd": 0.0, "latency_cost_usd": None,
            "average_slippage_bps": 0.0, "average_execution_delay_seconds": None,
            "cancelled_entry_intents": 0,
            "note": "IDEALIZED_NO_FEES: fills at the triggering observation, no friction "
            "(gross = net); latency is not modeled",
        }  # fmt: skip
    rows = [x for x in shadow.executions(run_id, sid)
            if x["strategy_version"] == version and w.contains(x["filled_at"])]  # fmt: skip
    sells = [x for x in rows if x["side"] == "SELL"]
    delays = [x["delay_seconds"] for x in rows]
    traded = sum(x["observed_value_usd"] for x in rows)
    friction = sum(x["friction_usd"] for x in rows)
    cancels = [
        d for d in shadow.decisions(run_id, sid, None, w.since, w.until, "NO_ACTION", 10**9)
        if d["strategy_version"] == version and d["reason"].startswith("entry order cancelled")
        and w.before_end(d["decision_at"])
    ]  # fmt: skip
    released = sum(json.loads(d["evidence_json"]).get("released_usd") or 0.0 for d in cancels)
    return out | {
        "fills": {"BUY": len(rows) - len(sells), "SELL": len(sells)},
        "gross_realized_pnl_usd": sum(x["gross_pnl_usd"] or 0.0 for x in sells),
        "realized_trade_friction_usd": sum(x["trade_friction_usd"] for x in sells),
        "total_friction_usd": friction,
        "fees_usd": sum(x["fee_usd"] for x in rows),
        "slippage_cost_usd": sum(x["slippage_cost_usd"] for x in rows),
        "price_impact_cost_usd": sum(x["impact_cost_usd"] for x in rows),
        "price_impact_status": dict(sorted(Counter(x["impact_status"] for x in rows).items())),
        "latency_cost_usd": sum(
            latency_drift_usd(x["side"], x["reference_price"], x["observed_price"], x["quantity"])
            for x in rows
        ),
        "stored_latency_cost_usd": sum(x["latency_cost_usd"] for x in rows),
        "latency_cost_note": "derived from each fill's stored reference price, observed "
        "price and quantity (latency_drift_usd); stored_latency_cost_usd sums the rows' own "
        "column, which for fills recorded before this fix scaled a BUY's drift by "
        "reference / observed price (unbounded when the price collapsed before the fill)",
        "average_slippage_bps": _mean([x["slippage_bps"] for x in rows]),
        "average_price_impact_bps": _mean([x["impact_bps"] for x in rows]),
        "effective_friction_bps": friction / traded * 1e4 if traded > 0 else None,
        "configured_min_latency_seconds": execution.latency_seconds,
        "average_execution_delay_seconds": _mean(delays),
        **delay_stats(delays),
        "cancelled_entry_intents": len(cancels),
        "cancelled_entry_released_usd": released,
        "cancelled_note": "a cancelled entry order is a NO_ACTION decision (ENTRY_NOT_FILLED): "
        "not a trade, no P/L, its reserved cash released",
        "execution_delay_note": "observed: intent time to the fill observation (the first "
        "archived exact-pool price at or after the minimum latency); not market-data "
        "resolution. The configured latency is only the earliest eligible time: the "
        "actual delay is set by when the next exact-pool observation is archived (Scout / "
        "held-position watch cadence)",
        "note": REALISTIC_NOTE + " Gross P/L: observed-price P/L of the quantity held; "
        "realized trade friction: exit costs plus the entry costs of the quantity sold "
        "(gross - friction = net, exactly). Latency drift: the USD effect of the price moving "
        "between an intent's reference observation and its fill observation (adverse "
        "positive, may be negative); an analytic of fill timing, not a term of gross or net. "
        "Friction and fees cover BUY and SELL fills in the window, open positions included.",
    }  # fmt: skip


# --- the run -------------------------------------------------------------------------------------


def _strategies(
    shadow: ShadowStore, run: dict[str, Any], wanted: Sequence[str] | None
) -> list[StrategyConfig]:
    out = []
    for frozen in run["strategies"]:
        cfg = shadow.strategy(frozen["strategy_id"], frozen["version"])
        if cfg.config_hash != frozen["config_hash"]:
            raise ShadowError(f"{cfg.key} no longer matches the hash frozen by run {run['run_id']}")
        out.append(cfg)
    if wanted:
        unknown = sorted(set(wanted) - {s.strategy_id for s in out})
        if unknown:
            raise ShadowError(f"run {run['run_id']} has no strategy {', '.join(unknown)}")
        out = [s for s in out if s.strategy_id in wanted]
    order = {sid: i for i, sid in enumerate(COMPARED)}
    return sorted(
        out, key=lambda s: (order.get(s.strategy_id, len(order)), s.strategy_id, s.version)
    )


def shadow_report(
    shadow: ShadowStore,
    evidence: EvidenceStore | None,
    run_id: str,
    strategies: Sequence[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    run = shadow.run(run_id)
    if run is None:
        raise ShadowError(f"unknown run {run_id}")
    if since is not None and until is not None and until <= since:
        raise ShadowError("--until must be after --since")
    cfgs = _strategies(shadow, run, strategies)
    cp = shadow.checkpoint(run_id)
    processed = _dt(cp["processed_until"])
    assert processed is not None
    as_of = min(until, processed) if until is not None else processed
    w = Window(since, until, as_of, until is None or until >= processed)
    counts: dict[tuple[str, int], dict[str, Any]] = {}
    for sid, version, action, n, first, last in shadow.decision_counts(run_id, since, until):
        c = counts.setdefault((sid, version), {"by_action": {}, "first": None, "last": None})
        c["by_action"][action] = n
        c["first"] = min(filter(None, (c["first"], _iso(_dt(first)))))
        c["last"] = max(filter(None, (c["last"], _iso(_dt(last)))))
    # Evidence of every held asset, once, up to the as-of time (never later).
    records: dict[str, list[EvidenceRecord]] = {}
    if evidence is not None:
        spans: dict[str, tuple[datetime, datetime]] = {}
        for cfg in cfgs:
            for p in shadow.positions(run_id, cfg.strategy_id):
                if not w.before_end(p["entry_at"]):
                    continue
                a = _dt(p["entry_at"])
                closed_at = p["closed_at"]
                b = _dt(closed_at) if closed_at is not None and w.before_end(closed_at) else as_of
                assert a is not None and b is not None
                lo, hi = spans.get(p["asset_id"], (a, b))
                spans[p["asset_id"]] = (min(lo, a), max(hi, b))
        for asset, (lo, hi) in sorted(spans.items()):
            records[asset] = [r for r in _records(evidence, asset, lo, hi)
                              if r.observed_at <= as_of]  # fmt: skip
    reports = [
        strategy_report(shadow, evidence, run, cfg, cp["books"].get(cfg.key),
                        counts.get((cfg.strategy_id, cfg.version), {}), w, records)
        for cfg in cfgs
    ]  # fmt: skip
    affected = [r.pop("_affected") for r in reports]
    watch_ids = {i for r in reports for i in r["availability"].get("watch_record_ids", [])}
    for r in reports:
        r["availability"].pop("watch_record_ids", None)
    stats = cp["stats"]
    mu = sum(r["trade_behavior"]["market_unavailable"] for r in reports)
    known = [a for a in affected if a is not None]
    # Only exits that followed missing evidence: a fill that merely waited for the next
    # observation (minimum latency, watch cadence) is a delayed fill, not an evidence gap.
    gap_affected = (
        any(a["exits_after_evidence_gap"] > 0 or a["market_unavailable_exits"] > 0 for a in known)
        if known else None
    )  # fmt: skip
    since_run = _dt(run["since_ts"])
    assert since_run is not None
    integrity = {
        "run_id": run_id,
        "availability_policy": run_policy(run),
        "execution_model": run["execution_model"],
        "execution_note": REALISTIC_NOTE if run_execution(run) else EXECUTION_NOTE,
        "execution_settings": (e.model_dump(mode="json") if (e := run_execution(run)) else None),
        "execution_warning": REALISTIC_WARNING if run_execution(run) else IDEALIZED_WARNING,
        "idealized_warning": IDEALIZED_WARNING,
        "run_since": run["since"],
        "run_until": run["until"],
        "run_created_at": run["created_at"],
        "processed_until": _iso(processed),
        "last_checkpoint_at": _iso(_dt(cp["updated_at"])),
        "report_start": _iso(max(since, since_run) if since else since_run),
        "report_end": _iso(as_of),
        "clean_data": run["clean_data"],
        "clean_data_cutoff": CLEAN_DATA_CUTOFF.isoformat(),
        "starts_after_clean_cutoff": since_run >= CLEAN_DATA_CUTOFF,
        "late_evidence_ignored": stats.get("late_evidence_ignored", 0),
        "skipped_evidence": stats.get("skipped", {}),
        "market_unavailable_exits_exist": mu > 0,
        "market_unavailable_exits": mu,
        "evidence_gaps_affected_exits": gap_affected,
        "evidence_gaps_affected_exits_rule": "true when a reported strategy has an exit after "
        "an evidence gap (PROVIDER_UNAVAILABLE, EVIDENCE_GAP or MARKET_NOT_FOUND while held) "
        "or a MARKET_UNAVAILABLE exit; delayed exit fills alone never count",
        "evidence_gap_exits_by_strategy": {r["strategy"]: a for r, a in zip(reports, affected, strict=True)},
        "evidence_archive": evidence.path if evidence is not None else None,
        "anti_lookahead": "every figure is a stored row written by the book at the time, or "
        "archived evidence observed no later than the report end; availability states use "
        "only observations up to each instant. Nothing here is retrospective and nothing "
        "changes a decision.",
        "availability_reconstruction_note": "rebuilt by observation time: exact for the "
        "evidence the book replayed when late_evidence_ignored is 0 (late records would be "
        "counted here but were never applied by the book)",
    }  # fmt: skip
    return {
        "label": NOT_REAL_PROFIT,
        "report": "shadow strategy validation",
        "read_only": True,
        "provider_requests": 0,
        "run_id": run_id,
        "window": {
            "since": _iso(since),
            "until": _iso(until),
            "as_of": _iso(as_of),
            "account_source": "checkpoint" if w.current else "equity snapshots",
        },  # fmt: skip
        "filters": {"strategies": list(strategies) if strategies else None},
        "strategies": reports,
        "comparison": comparison(reports),
        "run_watch_records": len(watch_ids),
        "data_integrity": integrity,
    }


def comparison(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    present = {r["strategy_id"] for r in reports}
    for r in reports:
        p, a, x = r["performance"], r["availability"], r["excursion"]
        failures = a.get("market_unavailable_exits", 0)
        rows.append({
            "strategy": r["strategy"],
            "sample": r["sample"]["status"],
            "insufficient_sample": r["sample"]["status"] == "INSUFFICIENT_SAMPLE",
            "total_return_pct": p["total_return_pct"],
            "closed_trades": r["sample"]["positions_closed"],
            "win_rate": p["win_rate"],
            "profit_factor": p["profit_factor"],
            "median_return_pct": p["median_return_pct"],
            "max_drawdown_pct": r["risk"]["max_drawdown_pct"],
            "mfe_median_pct": x["mfe_median_pct"],
            "mae_median_pct": x["mae_median_pct"],
            "market_unavailable_exits": failures,
            "exits_after_evidence_gap": (a.get("evidence_gap_exits") or {}).get("exits_after_evidence_gap"),
            "fresh_price_time_pct": a.get("fresh_price_time_pct"),
        })  # fmt: skip
    return {
        "note": NOT_RANKED,
        "order": "fixed: " + ", ".join(COMPARED) + ", then any other strategy by id",
        "missing_baselines": [s for s in COMPARED if s not in present],
        "rows": rows,
    }


# --- text ---------------------------------------------------------------------------------------


def _f(v: Any, fmt: str = ".2f", suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:{fmt}}{suffix}"


def _usd(v: Any) -> str:
    return "-" if v is None else f"${v:,.2f}"


def text(d: dict[str, Any]) -> str:
    g = d["data_integrity"]
    win = d["window"]
    lines = [
        f"SHADOW VALIDATION REPORT  run {d['run_id']}  (read-only, no provider requests)",
        d["label"],
        f"execution model: {g['execution_model']}",
        g["execution_warning"],
        f"window: {win['since'] or 'run start'} .. {win['until'] or 'now'}; as of {win['as_of']} "
        f"(account from {win['account_source']})",
    ]
    for r in d["strategies"]:
        s, p, k, b, x, a = (r["sample"], r["performance"], r["risk"], r["trade_behavior"],
                            r["excursion"], r["availability"])  # fmt: skip
        flag = (
            "  ** INSUFFICIENT SAMPLE: " + "; ".join(s["insufficient_reasons"])
            if s["insufficient_reasons"]
            else ""
        )
        best, worst = p["best_trade"], p["worst_trade"]
        lines += [
            "",
            f"=== {r['strategy']}  ({r['name']}) ===",
            f"[sample] {s['status']}{flag}",
            f"  decisions {s['decisions']} ({', '.join(f'{k2} {v}' for k2, v in s['decisions_by_action'].items())})",
            f"  positions opened {s['positions_opened']}, closed {s['positions_closed']} "
            f"(resolved {s['resolved_closed']}, unresolved {s['unresolved_closed']}), open {s['currently_open']}",
            f"  unique assets entered {s['unique_assets_entered']}, closed {s['unique_assets_closed']}; "
            f"effective sample size {s['effective_sample_size']:.2f}",
            f"  decisions from {s['first_decision_at'] or '-'} to {s['last_decision_at'] or '-'}",
            "[performance]",
            f"  capital {_usd(p['starting_capital_usd'])}  equity {_usd(p['current_equity_usd'])}  "
            f"cash {_usd(p['cash_usd'])}  ({p['account']['source']} {p['account']['at'] or ''})",
            f"  realized {_usd(p['realized_pnl_usd'])}  unrealized {_usd(p['unrealized_pnl_usd'])}  "
            f"total return {_f(p['total_return_pct'], '.2f', '%')}  last-mark {_f(p['last_mark_total_return_pct'], '.2f', '%')}"
            + (f"  window {_f(p['window_return_pct'], '.2f', '%')}" if p["window_return_pct"] is not None else ""),
            f"  closed-trade return median {_f(p['median_return_pct'], '.2f', '%')}  mean "
            f"{_f(p['mean_return_pct'], '.2f', '%')}  trimmed {_f(p['trimmed_mean_return_pct'], '.2f', '%')}",
            f"  win rate {_f(p['win_rate'] * 100 if p['win_rate'] is not None else None, '.1f', '%')}  "
            f"profit factor {_f(p['profit_factor'])}" + (f" ({p['profit_factor_note']})" if p["profit_factor_note"] else ""),
            f"  winners avg {_f(p['average_winner_pct'], '.2f', '%')} median {_f(p['median_winner_pct'], '.2f', '%')}; "
            f"losers avg {_f(p['average_loser_pct'], '.2f', '%')} median {_f(p['median_loser_pct'], '.2f', '%')}",
            f"  best {best['asset_id']} {_f(best['return_pct'], '.2f', '%')}" if best else "  best -",
            f"  worst {worst['asset_id']} {_f(worst['return_pct'], '.2f', '%')}" if worst else "  worst -",
            "[risk]",
            f"  max drawdown {_f(k['max_drawdown_pct'], '.2f', '%')}  last-mark {_f(k['last_mark_max_drawdown_pct'], '.2f', '%')}"
            + (f"  window {_f(k['window_max_drawdown_pct'], '.2f', '%')}" if k["window_max_drawdown_pct"] is not None else ""),
            f"  peak equity {_usd(k['peak_equity_usd'])}  minimum equity {_usd(k['minimum_equity_usd'])}  "
            f"max simultaneous positions {k['max_simultaneous_positions']}",
            f"  unresolved cost {_usd(k['unresolved_cost_usd'])}  last-mark value "
            f"{_usd(k['unresolved_last_mark_value_usd'])}  ({_f(k['unresolved_capital_pct'], '.2f', '%')} of capital)",
            "[trade behavior]",
            "  exits: " + ", ".join(f"{k2} {v}" for k2, v in b["exit_reasons"].items()),
            f"  liquidity-collapse exits {b['liquidity_collapse_exits']}  MARKET_UNAVAILABLE by code "
            f"{b['market_unavailable_by_reason_code'] or '{}'}",
            f"  holding avg {_f(b['average_holding_minutes'], '.0f', ' min')} median "
            f"{_f(b['median_holding_minutes'], '.0f', ' min')}; delayed exit fills {b['delayed_exit_fills']} "
            f"(max {_f(b['max_exit_fill_delay_minutes'], '.0f', ' min')})",
            "[excursion]",
            f"  closed MFE median {_f(x['mfe_median_pct'], '.2f', '%')} mean {_f(x['mfe_mean_pct'], '.2f', '%')}; "
            f"MAE median {_f(x['mae_median_pct'], '.2f', '%')} mean {_f(x['mae_mean_pct'], '.2f', '%')}",
            (f"  open to date ({x['open_to_date']['positions']}): MFE median "
             f"{_f(x['open_to_date']['mfe_median_pct'], '.2f', '%')} MAE median "
             f"{_f(x['open_to_date']['mae_median_pct'], '.2f', '%')}")
            if x["open_to_date"] else f"  open to date: {x.get('open_to_date_note', '-')}",
        ]  # fmt: skip
        lines += _execution_text(r["execution"])
        lines.append("[availability]")
        if a.get("available"):
            lines += [
                "  open now by state: " + ", ".join(f"{k2} {v}" for k2, v in a["open_positions_by_state"].items()),
                "  positions ever in state: " + ", ".join(f"{k2} {v}" for k2, v in a["positions_ever_in_state"].items()),
                "  held hours by state: " + ", ".join(f"{k2} {v:.1f}" for k2, v in a["held_hours_by_state"].items()),
                f"  fresh pricing {_f(a['fresh_price_time_pct'], '.1f', '%')} of "
                f"{a['held_position_hours']:.1f} held-position hours",
                "  watch records: " + ", ".join(f"{k2} {v}" for k2, v in a["watch_records"].items()),
                f"  exits after an evidence gap {a['evidence_gap_exits']['exits_after_evidence_gap']}, "
                f"after PRICE_STALE {a['evidence_gap_exits']['exits_after_price_stale']}",
            ]  # fmt: skip
        else:
            lines.append(f"  {a.get('note')}")
    c = d["comparison"]
    head = (f"  {'strategy':<28}{'sample':>9}{'return':>9}{'closed':>7}{'win':>7}{'PF':>6}"
            f"{'median':>8}{'maxDD':>8}{'MFE':>8}{'MAE':>8}{'MU':>4}{'gap':>5}")  # fmt: skip
    lines += ["", "=== comparison (NOT ranked) ===", c["note"], head]
    for row in c["rows"]:
        win_rate = row["win_rate"] * 100 if row["win_rate"] is not None else None
        lines.append(
            f"  {row['strategy']:<28}{'LOW' if row['insufficient_sample'] else 'ok':>9}"
            f"{_f(row['total_return_pct'], '.2f', '%'):>9}{row['closed_trades']:>7}"
            f"{_f(win_rate, '.0f', '%'):>7}{_f(row['profit_factor']):>6}"
            f"{_f(row['median_return_pct'], '.1f', '%'):>8}{_f(row['max_drawdown_pct'], '.1f', '%'):>8}"
            f"{_f(row['mfe_median_pct'], '.1f', '%'):>8}{_f(row['mae_median_pct'], '.1f', '%'):>8}"
            f"{row['market_unavailable_exits']:>4}{_f(row['exits_after_evidence_gap'], 'd'):>5}"
        )
    if c["missing_baselines"]:
        lines.append(f"  not in this report: {', '.join(c['missing_baselines'])}")
    lines += [
        "  LOW = INSUFFICIENT_SAMPLE; MU = MARKET_UNAVAILABLE exits; gap = exits after an "
        "evidence gap",
        "",
        "=== data integrity ===",
        f"  run {g['run_id']}  policy {g['availability_policy']}  execution {g['execution_model']}",
        f"  run since {g['run_since']} until {g['run_until'] or 'open'}; processed until "
        f"{g['processed_until']}; report {g['report_start']} .. {g['report_end']}",
        f"  clean data: {g['clean_data']} (cutoff {g['clean_data_cutoff']}, starts after it: "
        f"{g['starts_after_clean_cutoff']})",
        f"  late evidence ignored: {g['late_evidence_ignored']}; skipped evidence: {g['skipped_evidence'] or '{}'}",
        f"  MARKET_UNAVAILABLE exits exist: {'yes' if g['market_unavailable_exits_exist'] else 'no'} "
        f"({g['market_unavailable_exits']})",
        "  evidence gaps affected exits: " + {True: "yes", False: "no", None: "unknown (no evidence archive)"}[g["evidence_gaps_affected_exits"]],
        f"  {g['execution_warning']}",
        f"  {g['idealized_warning']}",
        f"  {g['anti_lookahead']}",
    ]  # fmt: skip
    return "\n".join(lines)


def _execution_text(e: dict[str, Any]) -> list[str]:
    p = e["pending"]
    lines = [
        f"[execution] {e['execution_model']}",
        f"  gross realized {_usd(e['gross_realized_pnl_usd'])}  net realized "
        f"{_usd(e['net_realized_pnl_usd'])}  total friction {_usd(e['total_friction_usd'])}",
        f"  fees {_usd(e['fees_usd'])}  slippage {_usd(e['slippage_cost_usd'])}  price impact "
        f"{_usd(e['price_impact_cost_usd'])}  latency drift {_usd(e['latency_cost_usd'])}",
        f"  average slippage {_f(e['average_slippage_bps'], '.1f', ' bps')}  observed execution "
        f"delay avg {_f(e['average_execution_delay_seconds'], '.0f', ' s')} median "
        f"{_f(e.get('median_execution_delay_seconds'), '.0f', ' s')} p90 "
        f"{_f(e.get('p90_execution_delay_seconds'), '.0f', ' s')} max "
        f"{_f(e.get('max_execution_delay_seconds'), '.0f', ' s')} (minimum latency "
        f"{_f(e.get('configured_min_latency_seconds'), '.0f', ' s')})",
        "  filled within "
        + "  ".join(f"{k} {_f(v, '.0f', '%')}" for k, v in (e.get("filled_within_pct") or {}).items()),
        f"  cancelled entry orders {e['cancelled_entry_intents']} (not trades, no P/L; "
        f"{_usd(e.get('cancelled_entry_released_usd', 0.0))} released)",
        f"  pending: {p['entry_intents']} entry intents ({_usd(p['entry_reserved_usd'])} "
        f"reserved), {p['exit_intents']} exit intents {p['exit_intents_by_reason'] or ''}"
        if p else f"  pending: {e.get('pending_note')}",
    ]  # fmt: skip
    return lines


# --- two runs side by side ----------------------------------------------------------------------

COMPARED_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("sample", "decisions", "decisions"),
    ("sample", "positions_opened", "positions_opened"),
    ("sample", "positions_closed", "closed_trades"),
    ("sample", "resolved_closed", "resolved_closed"),
    ("sample", "currently_open", "currently_open"),
    ("performance", "total_return_pct", "total_return_pct"),
    ("performance", "current_equity_usd", "equity_usd"),
    ("performance", "realized_pnl_usd", "realized_pnl_usd"),
    ("performance", "unrealized_pnl_usd", "unrealized_pnl_usd"),
    ("performance", "win_rate", "win_rate"),
    ("performance", "profit_factor", "profit_factor"),
    ("performance", "median_return_pct", "median_return_pct"),
    ("performance", "mean_return_pct", "mean_return_pct"),
    ("risk", "max_drawdown_pct", "max_drawdown_pct"),
    ("risk", "unresolved_cost_usd", "unresolved_cost_usd"),
    ("excursion", "mfe_median_pct", "mfe_median_pct"),
    ("excursion", "mae_median_pct", "mae_median_pct"),
    ("trade_behavior", "market_unavailable", "market_unavailable_exits"),
    ("execution", "gross_realized_pnl_usd", "gross_realized_pnl_usd"),
    ("execution", "net_realized_pnl_usd", "net_realized_pnl_usd"),
    ("execution", "total_friction_usd", "total_friction_usd"),
    ("execution", "fees_usd", "fees_usd"),
    ("execution", "slippage_cost_usd", "slippage_cost_usd"),
    ("execution", "price_impact_cost_usd", "price_impact_cost_usd"),
    ("execution", "latency_cost_usd", "latency_cost_usd"),
    ("execution", "average_execution_delay_seconds", "average_execution_delay_seconds"),
    ("execution", "cancelled_entry_intents", "cancelled_entry_intents"),
)


def _number(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


@dataclass(frozen=True)
class Side:
    """One run's view of one strategy for the comparison."""

    enters: dict[tuple[str, float], str]  # (asset, ENTER decision time) -> decision id
    positions: dict[str, dict[str, Any]]  # entry decision id -> position row
    closed: dict[str, dict[str, Any]]  # position id -> closed trade (before the window end)
    gross: dict[str, float]  # position id -> gross P/L of its fills
    friction: dict[str, float]  # position id -> trading friction of its fills


def _side(shadow: ShadowStore, run: dict[str, Any], cfg: StrategyConfig, w: Window) -> Side:
    run_id, sid, version = run["run_id"], cfg.strategy_id, cfg.version
    enters = {
        (d["asset_id"], d["decision_at"]): d["decision_id"]
        for d in shadow.decisions(run_id, sid, None, w.since, w.until, "ENTER", 10**9)
        if d["strategy_version"] == version and w.before_end(d["decision_at"])
    }  # fmt: skip
    positions = {p["entry_decision_id"]: p for p in shadow.positions(run_id, sid)
                 if p["strategy_version"] == version and w.before_end(p["entry_at"])}  # fmt: skip
    fills = [t for t in shadow.trades(run_id, sid)
             if t["strategy_version"] == version and w.before_end(t["exit_at"])]  # fmt: skip
    closed = {c["position_id"]: c for c in closed_positions(fills)}
    gross: dict[str, float] = {}
    friction: dict[str, float] = {}
    if run_execution(run) is None:  # IDEALIZED_NO_FEES: no friction, gross = net
        for c in closed.values():
            if c["pnl_usd"] is not None:
                gross[c["position_id"]], friction[c["position_id"]] = c["pnl_usd"], 0.0
    else:
        for x in shadow.executions(run_id, sid):
            if x["side"] != "SELL" or x["strategy_version"] != version:
                continue
            if not w.before_end(x["filled_at"]):
                continue
            pid = x["position_id"]
            gross[pid] = gross.get(pid, 0.0) + (x["gross_pnl_usd"] or 0.0)
            friction[pid] = friction.get(pid, 0.0) + x["trade_friction_usd"]
    return Side(enters, positions, closed, gross, friction)


def _sums(pairs: Sequence[tuple[float, float]]) -> dict[str, float]:
    base = sum(a for a, _ in pairs)
    other = sum(b for _, b in pairs)
    return {"base": base, "other": other, "delta": other - base}


def _first(at: float | None, **detail: Any) -> dict[str, Any] | None:
    return None if at is None else {"at": _iso(_dt(at))} | detail


def divergence(
    shadow: ShadowStore,
    a: tuple[str, str, int],
    b: tuple[str, str, int],
    since: datetime | None,
    until: datetime | None,
) -> dict[str, Any]:
    """The first ENTER decision, and the first decision of any kind, made in one run and
    not the other (same asset, action and decision time)."""
    out: dict[str, Any] = {}
    for label, actions in (("first_entry_divergence", ("ENTER",)),
                           ("first_decision_divergence", ACTIONS)):  # fmt: skip
        found = []
        for x, y, only in ((a, b, "base"), (b, a, "other")):
            row = shadow.first_decision_difference(x, y, since, until, actions)
            if row is not None:
                found.append((row, only))
        if not found:
            out[label] = None
            continue
        (at, action, asset), only = min(found)
        out[label] = _first(at, action=action, asset_id=asset, only_in=only)
    return out


def compare_runs(
    shadow: ShadowStore,
    evidence: EvidenceStore | None,
    base: str,
    other: str,
    strategies: Sequence[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """`other` against `base`, per strategy present in both, over the same window ending
    at the earlier of the two runs' processed times. Factual only, never a ranking.

    The runs make the same strategy decisions only until their portfolios diverge (cash
    after fees, pending fills, open slots, cooldowns). So it separates:

    A. matched entry opportunities (an ENTER decision of the same asset at the same decision
       time in both) and matched completed trades (both filled, closed and resolved);
    B. the execution delta on matched completed trades: gross P/L, net P/L and friction;
    C / D. ENTER decisions only in base / only in other;
    E. portfolio divergence: the first ENTER decision and the first decision of any kind
       made in one run only, and the portfolio totals, which mix execution friction with
       everything that diverged and are never attributed to friction alone."""
    runs = {}
    for run_id in (base, other):
        run = shadow.run(run_id)
        if run is None:
            raise ShadowError(f"unknown run {run_id}")
        runs[run_id] = run
    processed = []
    for run_id in (base, other):
        t = _dt(shadow.checkpoint(run_id)["processed_until"])
        assert t is not None
        processed.append(t)
    end = min([*processed, *([until] if until else [])])
    if since is not None and end <= since:
        raise ShadowError("the window is empty: --since is after the runs' common end")
    ids = [set(s["strategy_id"] for s in runs[r]["strategies"]) for r in (base, other)]
    common = sorted(ids[0] & ids[1])
    if strategies:
        missing = sorted(set(strategies) - set(common))
        if missing:
            raise ShadowError(f"not in both runs: {', '.join(missing)}")
        common = [s for s in common if s in strategies]
    if not common:
        raise ShadowError(f"runs {base} and {other} share no strategy")
    reports = {r: shadow_report(shadow, evidence, r, common, since, end)
               for r in (base, other)}  # fmt: skip
    windows = [Window(since, end, min(end, t), end >= t) for t in processed]
    order = {sid: i for i, sid in enumerate(COMPARED)}
    out: list[dict[str, Any]] = []
    for sid in sorted(common, key=lambda x: (order.get(x, len(order)), x)):
        a = next(r for r in reports[base]["strategies"] if r["strategy_id"] == sid)
        b = next(r for r in reports[other]["strategies"] if r["strategy_id"] == sid)
        totals = {}
        for section, key, label in COMPARED_FIELDS:
            va, vb = a[section].get(key), b[section].get(key)
            na, nb = _number(va), _number(vb)
            totals[label] = {"base": va, "other": vb,
                             "delta": nb - na if na is not None and nb is not None else None}  # fmt: skip
        cfg_a = shadow.strategy(sid, a["strategy_version"])
        cfg_b = shadow.strategy(sid, b["strategy_version"])
        sa = _side(shadow, runs[base], cfg_a, windows[0])
        sb = _side(shadow, runs[other], cfg_b, windows[1])
        matched = sorted(set(sa.enters) & set(sb.enters))
        only_a = sorted(set(sa.enters) - set(sb.enters))
        only_b = sorted(set(sb.enters) - set(sa.enters))
        filled = [(sa.positions[sa.enters[k]], sb.positions[sb.enters[k]]) for k in matched
                  if sa.enters[k] in sa.positions and sb.enters[k] in sb.positions]  # fmt: skip
        done = [(sa.closed[pa["position_id"]], sb.closed[pb["position_id"]]) for pa, pb in filled
                if pa["position_id"] in sa.closed and pb["position_id"] in sb.closed
                and sa.closed[pa["position_id"]]["resolved"]
                and sb.closed[pb["position_id"]]["resolved"]]  # fmt: skip
        ret = [cb["return_pct"] - ca["return_pct"] for ca, cb in done]

        def unmatched(side: Side, keys: Sequence[tuple[str, float]]) -> dict[str, Any]:
            pos = [side.positions[side.enters[k]] for k in keys if side.enters[k] in side.positions]
            closed = [side.closed[p["position_id"]] for p in pos if p["position_id"] in side.closed]
            return {
                "enter_decisions": len(keys), "filled": len(pos), "closed": len(closed),
                "net_pnl_usd": sum(c["pnl_usd"] for c in closed if c["pnl_usd"] is not None),
                "first": [{"asset_id": k[0], "decision_at": _iso(_dt(k[1]))} for k in keys[:5]],
            }  # fmt: skip

        out.append({
            "strategy_id": sid,
            "base_strategy": a["strategy"], "other_strategy": b["strategy"],
            "same_rules": cfg_a.config_hash == cfg_b.config_hash,
            "base_sample": a["sample"]["status"], "other_sample": b["sample"]["status"],
            "insufficient_sample": "INSUFFICIENT_SAMPLE" in (a["sample"]["status"],
                                                             b["sample"]["status"]),
            "matched": {  # A and B
                "enter_decisions": len(matched),
                "filled_in_both": len(filled),
                "completed_in_both": len(done),
                "gross_pnl_usd": _sums([(sa.gross.get(ca["position_id"], 0.0),
                                         sb.gross.get(cb["position_id"], 0.0)) for ca, cb in done]),
                "net_pnl_usd": _sums([(ca["pnl_usd"], cb["pnl_usd"]) for ca, cb in done]),
                "friction_usd": _sums([(sa.friction.get(ca["position_id"], 0.0),
                                        sb.friction.get(cb["position_id"], 0.0)) for ca, cb in done]),
                "return_delta_pct_points_median": _median(ret),
                "return_delta_pct_points_mean": _mean(ret),
                "note": "same asset and ENTER decision time in both runs, both closed and "
                "resolved. Net delta = gross delta (fill prices and timing: latency, a later "
                "exit observation, the smaller quantity bought after fees) minus the friction "
                "delta (fees, slippage, impact)",
            },
            "only_in_base": unmatched(sa, only_a),  # C
            "only_in_other": unmatched(sb, only_b),  # D
            "divergence": divergence(  # E
                shadow, (base, sid, cfg_a.version), (other, sid, cfg_b.version), since, end
            ) | {"note": "after the first divergence the portfolios differ: portfolio totals "
                 "mix execution friction with different decisions and are NOT caused by "
                 "fees / slippage / latency alone"},
            "portfolio_totals": totals,
        })  # fmt: skip
    info = {
        r: {
            "execution_model": runs[r]["execution_model"],
            "availability_policy": run_policy(runs[r]),
            "since": runs[r]["since"],
            "processed_until": _iso(t),
            "execution_settings": (e.model_dump(mode="json") if (e := run_execution(runs[r]))
                                   else None),
        }
        for r, t in zip((base, other), processed, strict=True)
    }  # fmt: skip
    notes = [
        "Factual comparison (other against base), NOT a ranking: no run or strategy is selected.",
        "Only matched completed trades isolate execution: everything else includes "
        "portfolio divergence.",
    ]
    if runs[base]["since"] != runs[other]["since"]:
        notes.append("the runs start at different times: use --since at or after the later "
                     "start for like-for-like figures")  # fmt: skip
    return {
        "label": NOT_REAL_PROFIT,
        "report": "shadow run comparison",
        "read_only": True,
        "provider_requests": 0,
        "base": base,
        "other": other,
        "runs": info,
        "window": {
            "since": _iso(since),
            "until": _iso(end),
            "note": "ends at the earlier processed time of the two runs (or --until)",
        },  # fmt: skip
        "notes": notes,
        "strategies_only_in_base": sorted(ids[0] - ids[1]),
        "strategies_only_in_other": sorted(ids[1] - ids[0]),
        "strategies": out,
    }


def compare_text(d: dict[str, Any]) -> str:
    lines = [
        f"SHADOW RUN COMPARISON  {d['other']} against {d['base']}  (read-only, no provider requests)",
        d["label"],
        *d["notes"],
        f"window: {d['window']['since'] or 'run start'} .. {d['window']['until']}",
    ]  # fmt: skip
    for r, i in d["runs"].items():
        lines.append(f"  {r}: {i['execution_model']}, {i['availability_policy']}, since "
                     f"{i['since']}, processed until {i['processed_until']}")  # fmt: skip
    for s in d["strategies"]:
        flag = "  ** INSUFFICIENT SAMPLE" if s["insufficient_sample"] else ""
        mt, ob, oo, dv = s["matched"], s["only_in_base"], s["only_in_other"], s["divergence"]
        lines += [
            "",
            f"=== {s['strategy_id']} ({s['base_strategy']} vs {s['other_strategy']})"
            f"{'' if s['same_rules'] else '  ** DIFFERENT RULES'}{flag}",
            f"  A matched ENTER decisions {mt['enter_decisions']}, filled in both "
            f"{mt['filled_in_both']}, completed in both {mt['completed_in_both']}",
            f"  B matched completed trades: gross {_usd(mt['gross_pnl_usd']['base'])} -> "
            f"{_usd(mt['gross_pnl_usd']['other'])}; net {_usd(mt['net_pnl_usd']['base'])} -> "
            f"{_usd(mt['net_pnl_usd']['other'])} (delta {_usd(mt['net_pnl_usd']['delta'])}); "
            f"friction {_usd(mt['friction_usd']['base'])} -> {_usd(mt['friction_usd']['other'])}; "
            f"return delta median {_f(mt['return_delta_pct_points_median'], '.2f', ' pp')}",
            f"  C only in base: {ob['enter_decisions']} ENTER decisions ({ob['filled']} filled, "
            f"{ob['closed']} closed, net {_usd(ob['net_pnl_usd'])})",
            f"  D only in other: {oo['enter_decisions']} ENTER decisions ({oo['filled']} filled, "
            f"{oo['closed']} closed, net {_usd(oo['net_pnl_usd'])})",
            "  E first ENTER divergence: " + (
                f"{dv['first_entry_divergence']['at']} {dv['first_entry_divergence']['asset_id']} "
                f"(only in {dv['first_entry_divergence']['only_in']})"
                if dv["first_entry_divergence"] else "none"),
            "    first decision divergence: " + (
                f"{dv['first_decision_divergence']['at']} {dv['first_decision_divergence']['action']} "
                f"{dv['first_decision_divergence']['asset_id']} (only in "
                f"{dv['first_decision_divergence']['only_in']})"
                if dv["first_decision_divergence"] else "none"),
            f"    {dv['note']}",
            f"  portfolio totals (include divergence)  {'base':>14}{'other':>14}{'delta':>14}",
        ]  # fmt: skip
        for label, v in s["portfolio_totals"].items():
            lines.append(f"  {label:<38}{_f(_number(v['base']), ',.2f'):>14}"
                         f"{_f(_number(v['other']), ',.2f'):>14}{_f(v['delta'], ',.2f'):>14}")  # fmt: skip
    if d["strategies_only_in_base"] or d["strategies_only_in_other"]:
        lines.append(f"\nstrategies only in base: {d['strategies_only_in_base']}; only in other: "
                     f"{d['strategies_only_in_other']}")  # fmt: skip
    return "\n".join(lines)
