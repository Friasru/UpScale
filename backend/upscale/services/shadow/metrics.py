"""Per-strategy paper metrics, from the immutable shadow rows only.

A *trade* is one closed position (its fills aggregated). Its return is
``(sum of proceeds - cost) / cost``; a position with any MARKET_UNAVAILABLE fill has no
return (``unresolved``) and is left out of every return statistic, never counted as a
win, a loss or zero.

Equity is reported two ways (reporting only; decisions and sizing only ever use the
written-off book equity):

* written off (``equity_usd`` / ``written_off_equity_usd``, ``max_drawdown_pct`` /
  ``written_off_max_drawdown_pct``): the book's own conservative accounting, a
  MARKET_UNAVAILABLE position valued at zero;
* last mark (``last_mark_equity_usd``, ``last_mark_max_drawdown_pct``): the same, with each
  unresolved position carried at its last observed exact-pool mark instead. STALE: that
  mark is the last price seen before the evidence stopped, not a price it could have been
  sold at. Rebuilt from the immutable rows (equity rows, MARKET_UNAVAILABLE fills, the
  positions' last marks), so it covers a run's whole history.

Repeated scans of one asset are not independent evidence: `effective_sample_size` treats
the trades of one asset as fully correlated (``n^2 / sum_a n_a^2``, i.e. n when every asset
traded once, 1 when one asset holds every trade).
"""

import statistics
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from upscale.services.shadow.config import EXECUTION_NOTE, NOT_REAL_PROFIT

MIN_TRADES = 20
MIN_ASSETS = 5
TRIM = 0.10  # trimmed mean: 10% cut from each end


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def trimmed_mean(values: Sequence[float], cut: float = TRIM) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = int(len(v) * cut)
    kept = v[k : len(v) - k] if len(v) - 2 * k > 0 else v
    return statistics.fmean(kept)


def closed_positions(trades: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fills grouped into closed positions (only positions whose final fill exists)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        by.setdefault(t["position_id"], []).append(t)
    out = []
    for pid, fills in by.items():
        fills.sort(key=lambda t: t["fill_no"])
        if not any(t["final"] for t in fills):
            continue  # partially exited, still open
        cost = sum(t["cost_usd"] for t in fills)
        resolved = all(t["proceeds_usd"] is not None for t in fills)
        proceeds = sum(t["proceeds_usd"] for t in fills) if resolved else None
        last = fills[-1]
        out.append({
            "position_id": pid, "asset_id": last["asset_id"], "entry_at": last["entry_at"],
            "exit_at": last["exit_at"], "cost_usd": cost, "resolved": resolved,
            "pnl_usd": proceeds - cost if proceeds is not None else None,
            "return_pct": (proceeds / cost - 1) * 100 if proceeds is not None and cost else None,
            "mfe_pct": last["mfe_pct"], "mae_pct": last["mae_pct"],
            "holding_minutes": last["holding_minutes"], "exit_reason": last["exit_reason"],
            "fills": len(fills),
        })  # fmt: skip
    return sorted(out, key=lambda p: (p["exit_at"], p["position_id"]))


LAST_MARK_NOTE = (
    "last_mark_*: REPORTING ONLY. Unresolved (MARKET_UNAVAILABLE) positions carried at their "
    "last observed exact-pool mark (stale, retrospective): never used for decisions or sizing; "
    "written_off_* is the book's own accounting."
)


def last_mark_series(
    equity: Sequence[dict[str, Any]], unresolved: Sequence[tuple[float, float]]
) -> list[float]:
    """Each equity row's unresolved last-mark value: the marks of the first k unresolved
    fills, k such that their cost adds up to the row's own `unresolved_cost` (the book adds
    them in this order, so rows and fills align exactly)."""
    out: list[float] = []
    k, cost, mark = 0, 0.0, 0.0
    for e in equity:
        target = e["unresolved_cost"]
        while k < len(unresolved) and cost + unresolved[k][0] <= target + 1e-6 * max(1.0, target):
            cost += unresolved[k][0]
            mark += unresolved[k][1]
            k += 1
        out.append(mark)
    return out


def effective_sample_size(assets: Sequence[str]) -> float:
    if not assets:
        return 0.0
    counts = Counter(assets).values()
    return len(assets) ** 2 / sum(n * n for n in counts)


def strategy_metrics(
    strategy_id: str,
    version: int,
    trades: Sequence[dict[str, Any]],
    positions: Sequence[dict[str, Any]],
    equity: Sequence[dict[str, Any]],
    capital: float,
    unresolved: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    closed = closed_positions(trades)
    resolved = [p for p in closed if p["resolved"]]
    returns = [p["return_pct"] for p in resolved]
    pnls = [p["pnl_usd"] for p in resolved]
    assets = [p["asset_id"] for p in closed]
    gains = sum(x for x in pnls if x > 0)
    losses = -sum(x for x in pnls if x < 0)
    open_rows = [p for p in positions if p["status"] == "OPEN"]
    days = {datetime.fromtimestamp(p["entry_at"], UTC).date() for p in positions}
    last = equity[-1] if equity else None
    per_asset = Counter(assets)
    marks = last_mark_series(equity, unresolved)
    peak, last_mark_dd = capital, 0.0
    for e, m in zip(equity, marks, strict=True):
        value = e["equity"] + m
        peak = max(peak, value)
        last_mark_dd = min(last_mark_dd, (value / peak - 1) * 100 if peak > 0 else 0.0)
    written_off = last["equity"] if last else capital
    unresolved_cost = last["unresolved_cost"] if last else 0.0
    unresolved_mark = marks[-1] if marks else 0.0
    sample = (
        "INSUFFICIENT_SAMPLE"
        if len(resolved) < MIN_TRADES or len(set(assets)) < MIN_ASSETS
        else "DESCRIPTIVE"
    )
    return {
        "strategy_id": strategy_id,
        "strategy_version": version,
        "label": NOT_REAL_PROFIT,
        "execution_model": EXECUTION_NOTE,
        "sample": sample,
        "trades": len(closed),
        "resolved_trades": len(resolved),
        "unresolved_trades": len(closed) - len(resolved),
        "open_positions": len(open_rows),
        "closed_positions": len(closed),
        "positions_opened": len(positions),
        "distinct_assets": len(set(assets)),
        "effective_sample_size": round(effective_sample_size(assets), 2),
        "trades_per_asset": {
            "mean": round(len(assets) / len(per_asset), 2) if per_asset else None,
            "max": max(per_asset.values()) if per_asset else None,
        },
        "days_represented": len(days),
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else None,
        "median_return_pct": _median(returns),
        "mean_return_pct": _mean(returns),
        "trimmed_mean_return_pct": trimmed_mean(returns),
        "median_mfe_pct": _median([p["mfe_pct"] for p in closed if p["mfe_pct"] is not None]),
        "median_mae_pct": _median([p["mae_pct"] for p in closed if p["mae_pct"] is not None]),
        # Undefined without a losing trade (never reported as infinite).
        "profit_factor": gains / losses if losses > 0 else None,
        "profit_factor_note": None if losses > 0 else "undefined: no losing resolved trade",
        "average_holding_minutes": _mean([p["holding_minutes"] for p in closed]),
        "exit_reasons": dict(Counter(p["exit_reason"] for p in closed)),
        "max_drawdown_pct": min((e["max_drawdown_pct"] for e in equity), default=0.0),
        "written_off_max_drawdown_pct": min((e["max_drawdown_pct"] for e in equity), default=0.0),
        "last_mark_max_drawdown_pct": last_mark_dd,
        "average_exposure_pct": _mean([e["exposure_pct"] for e in equity]),
        "initial_capital_usd": capital,
        "equity_usd": last["equity"] if last else capital,
        "cash_usd": last["cash"] if last else capital,
        "realized_pnl_usd": last["realized_pnl"] if last else 0.0,
        "unrealized_pnl_usd": last["unrealized_pnl"] if last else 0.0,
        "unresolved_cost_usd": unresolved_cost,
        "written_off_equity_usd": written_off,
        "last_mark_equity_usd": written_off + unresolved_mark,
        "unresolved_last_mark_value_usd": unresolved_mark,
        "unresolved_capital_pct": unresolved_cost / capital * 100 if capital else None,
        "accounting_note": LAST_MARK_NOTE,
        "exposure_pct": last["exposure_pct"] if last else 0.0,
        "equity_at": datetime.fromtimestamp(last["at"], UTC).isoformat() if last else None,
    }
