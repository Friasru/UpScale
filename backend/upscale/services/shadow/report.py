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
)
from upscale.services.shadow.config import (
    CLEAN_DATA_CUTOFF,
    EXECUTION_NOTE,
    NOT_REAL_PROFIT,
    AvailabilityPolicy,
    ExitReason,
    StrategyConfig,
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
        equity_now, unrealized = cash + open_value, open_value - open_cost
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
            for state, n in h.seconds.items():
                seconds[state] += n
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
        "_affected": affected,
    }


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
    gap_affected = (
        any(a["exits_after_evidence_gap"] or a["delayed_exit_fills"] or a["market_unavailable_exits"]
            for a in known)
        if known else None
    )  # fmt: skip
    since_run = _dt(run["since_ts"])
    assert since_run is not None
    integrity = {
        "run_id": run_id,
        "availability_policy": run_policy(run),
        "execution_model": run["execution_model"],
        "execution_note": EXECUTION_NOTE,
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
        g["idealized_warning"],
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
            "[availability]",
        ]  # fmt: skip
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
        f"  {g['idealized_warning']}",
        f"  {g['anti_lookahead']}",
    ]  # fmt: skip
    return "\n".join(lines)
