"""MARKET_UNAVAILABLE audit: why each priceless Shadow exit happened (read-only).

Reads the shadow database and the Evidence Archive, both opened read-only (``mode=ro``),
and nothing else: no provider or network request, no write, no effect on any book. The
strategies' decisions are not replayed or changed; this only explains rows already stored.

For each MARKET_UNAVAILABLE exit (a position closed without a price), the window
W = (last valid exact-pool price observed, exit time] is inspected:

* ``EVIDENCE_COLLECTION_GAP``: a valid price of the exact (token, pool) observed inside W
  is in the archive but was not used (archived after Shadow had passed it, or a
  ``dex_market`` record, a kind Shadow does not read), or Scout runs stopped (the largest
  gap between Scout runs inside W exceeds ``collection_gap_minutes``);
* ``POOL_CHANGED``: the token was priced inside W, but on another pool;
* ``PROVIDER_GAP``: Scout carried the token with ``STALE_CARRIED`` (a provider failure
  kept it from being refreshed) or a price record of the token is ``RATE_LIMITED`` /
  ``PROVIDER_FAILED`` inside W;
* ``TRUE_MARKET_DISAPPEARANCE``: a price record of the token is ``NOT_AVAILABLE`` (not
  found) inside W and the exact pool is never priced again in the stored evidence;
* ``TEMPORARY_UNAVAILABLE``: none of the above, and the exact pool is priced again later;
* ``LIQUIDITY_COLLAPSE``: none of the above; the last observations before W report
  MARKET_COLLAPSE, or liquidity fell to ``collapse_ratio`` of the entry liquidity or less;
* ``UNKNOWN``: none of the above (typically: the token left Scout's discovery universe
  while collection kept running, and is never observed again).

The first matching rule wins; every flag is reported for every case regardless.

RETROSPECTIVE: whether the exact pool was observed again after the exit (recovery) uses
evidence later than the exit. It is used for this classification only, is reported under
``retrospective``, and never feeds a strategy decision.
"""

import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from upscale.services.evidence_archive.store import EvidenceRecord, EvidenceStore
from upscale.services.shadow.config import NOT_REAL_PROFIT
from upscale.services.shadow.evidence import valid_price
from upscale.services.shadow.store import ShadowStore

CLASSES = (
    "TRUE_MARKET_DISAPPEARANCE", "LIQUIDITY_COLLAPSE", "PROVIDER_GAP",
    "EVIDENCE_COLLECTION_GAP", "TEMPORARY_UNAVAILABLE", "POOL_CHANGED", "UNKNOWN",
)  # fmt: skip
PRICE_KINDS = ("market", "scout", "dex_market")
FAILED = ("RATE_LIMITED", "PROVIDER_FAILED")
STALE_TIMEOUT = "MARKET_UNAVAILABLE_TIMEOUT"  # no price of the exact pool for N minutes
EXIT_PRICE_TIMEOUT = "EXIT_PRICE_TIMEOUT"  # pending exit / max hold found no price in time
RETROSPECTIVE_NOTE = (
    "RETROSPECTIVE: recovery fields use evidence observed after the exit, for this "
    "classification only; no strategy decision ever used it"
)


@dataclass(frozen=True)
class AuditSettings:
    collection_gap_minutes: float = 90.0  # 3x the default background Scout interval
    collapse_ratio: float = 0.2
    recovery_hours: float = 72.0  # how far after the exit to look for a recovery


@dataclass(frozen=True)
class Obs:
    """What one archived record says about the token's price, pool and liquidity."""

    record_id: str
    kind: str
    observed_at: datetime  # the record's time (a Scout decision time for scout records)
    price_at: datetime  # when its price was observed
    archived_at: datetime
    availability: str
    pool: str | None
    price: float | None
    liquidity_usd: float | None
    market_status: str | None
    data_status: str | None


def _dig(d: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def _time(raw: Any, default: datetime) -> datetime:
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return default


def _dt(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, UTC) if ts is not None else None


def _iso(t: datetime | None) -> str | None:
    return t.isoformat() if t is not None else None


def observation(r: EvidenceRecord) -> Obs:
    p = r.payload
    c = p.get("candidate") or {}
    price_at, status, data_status = r.observed_at, None, None
    if r.kind == "market":
        price, pool = _dig(c, "metrics", "price_usd"), _dig(c, "pool", "address")
        liquidity = _dig(c, "metrics", "liquidity_usd")
    elif r.kind == "scout":
        price = _dig(c, "market", "price_usd")
        pool = _dig(c, "market", "selected_pool", "address")
        liquidity = _dig(c, "market", "liquidity_usd")
        status, data_status = _dig(c, "quality", "market_status"), c.get("data_status")
        raw = _dig(p, "timing", "market_observed_at") or c.get("observed_at")
        price_at = _time(raw, r.observed_at)
    elif r.kind == "dex_market":
        s = p.get("snapshot") or {}
        price, pool, liquidity = s.get("price_usd"), s.get("pair_address"), s.get("liquidity_usd")
    else:
        price = pool = liquidity = None
    return Obs(
        record_id=r.record_id, kind=r.kind, observed_at=r.observed_at, price_at=price_at,
        archived_at=r.archived_at, availability=r.availability,
        pool=pool if isinstance(pool, str) and pool else r.pool_address,
        price=valid_price(price), liquidity_usd=_num(liquidity), market_status=status,
        data_status=data_status,
    )  # fmt: skip


def _priced(o: Obs, pool: str) -> bool:
    """A usable price of the exact pool (what Shadow itself would accept, any kind)."""
    return (
        o.kind in PRICE_KINDS
        and o.price is not None
        and o.pool == pool
        and o.availability == "AVAILABLE"
        and o.data_status in (None, "CURRENT")
    )


def _mechanism(reason: str) -> tuple[str, str]:
    if "triggered" in reason:
        return EXIT_PRICE_TIMEOUT, "PENDING_EXIT"
    if "maximum hold" in reason:
        return EXIT_PRICE_TIMEOUT, "MAX_HOLD"
    return STALE_TIMEOUT, "STALE_PRICE"


def _runs(evidence: EvidenceStore, start: datetime, end: datetime) -> list[datetime]:
    """Scout run decision times in (start, end] (one per ranking run; no payloads read)."""
    times = {at for _, at, _ in evidence.decisions("scout", start, end) if at > start.timestamp()}
    return [datetime.fromtimestamp(t, UTC) for t in sorted(times)]


def classify_case(
    shadow: ShadowStore,
    evidence: EvidenceStore,
    trade: dict[str, Any],
    position: dict[str, Any],
    decision: dict[str, Any] | None,
    settings: AuditSettings,
    timeouts: dict[str, float],
) -> dict[str, Any]:
    asset, pool = position["asset_id"], position["pool"]
    entry_at = _dt(position["entry_at"])
    last_at = _dt(position["last_price_at"])
    exit_at = _dt(trade["exit_at"])
    assert entry_at is not None and last_at is not None and exit_at is not None
    horizon = exit_at + timedelta(hours=settings.recovery_hours)
    records = evidence.records(asset_id=asset, since=entry_at, until=horizon, limit=1_000_000)
    rows = [observation(r) for r in records]
    # Before: what the book had seen (by price time: a Scout record's decision time can be
    # just after the price it carries). Inside / after: by the record's own time.
    before = [o for o in rows if o.price_at <= last_at]
    inside = [o for o in rows if last_at < o.observed_at <= exit_at]
    after = [o for o in rows if o.observed_at > exit_at]  # RETROSPECTIVE only

    # A valid price of the exact pool newer than the last one the book used, inside W.
    unused = [o for o in inside if _priced(o, pool) and o.price_at > last_at]
    other_pools = sorted({o.pool for o in inside if o.kind in PRICE_KINDS
                          and o.price is not None and o.pool not in (None, pool)})  # fmt: skip
    stale_carried = [o for o in inside if o.kind == "scout" and o.data_status == "STALE_CARRIED"]
    failures = [o for o in inside if o.availability in FAILED]
    price_failures = [o for o in failures if o.kind in PRICE_KINDS]
    not_found = [o for o in inside if o.kind in PRICE_KINDS and o.availability == "NOT_AVAILABLE"]
    recovered = [o for o in after if _priced(o, pool)]

    runs = _runs(evidence, last_at, exit_at)
    edges = [last_at, *runs, exit_at]
    gap = max((b - a).total_seconds() for a, b in zip(edges, edges[1:], strict=False)) / 60
    collection_gap = gap > settings.collection_gap_minutes
    # The first Scout run after the last valid price that produced no current price of the
    # exact pool (when the token first went missing); none: no Scout run inside W.
    priced_runs = {o.observed_at for o in inside if o.kind == "scout" and _priced(o, pool)}
    first_missing = next((t for t in runs if t not in priced_runs), None)

    liquidity = [o.liquidity_usd for o in rows if o.liquidity_usd is not None]
    entry_liq = liquidity[0] if liquidity else None
    last_liq = next((o.liquidity_usd for o in reversed(before) if o.liquidity_usd is not None),
                    None)  # fmt: skip
    collapsed = any(o.market_status == "MARKET_COLLAPSE" for o in before[-3:]) or bool(
        entry_liq and last_liq is not None and last_liq <= entry_liq * settings.collapse_ratio
    )
    last_obs = next((o for o in reversed(before) if _priced(o, pool)), None)

    if unused or collection_gap:
        cls = "EVIDENCE_COLLECTION_GAP"
    elif other_pools:
        cls = "POOL_CHANGED"
    elif stale_carried or price_failures:
        cls = "PROVIDER_GAP"
    elif not_found and not recovered:
        cls = "TRUE_MARKET_DISAPPEARANCE"
    elif recovered:
        cls = "TEMPORARY_UNAVAILABLE"
    elif collapsed:
        cls = "LIQUIDITY_COLLAPSE"
    else:
        cls = "UNKNOWN"
    reason = (decision or {}).get("reason") or ""
    mechanism, trigger = _mechanism(reason)
    first = recovered[0] if recovered else None
    return {
        "classification": cls,
        "strategy_id": trade["strategy_id"],
        "strategy_version": trade["strategy_version"],
        "position_id": position["position_id"],
        "chain": position["chain"],
        "asset_id": asset,
        "address": position["address"],
        "symbol": position["symbol"],
        "pool": pool,
        "entry_at": _iso(entry_at),
        "entry_price": position["entry_price"],
        "cost_usd": trade["cost_usd"],
        "last_valid_observation": {
            "kind": last_obs.kind,
            "record_id": last_obs.record_id,
            "observed_at": _iso(last_obs.observed_at),
            "price_observed_at": _iso(last_obs.price_at),
        }
        if last_obs
        else None,  # fmt: skip
        "last_valid_price": position["last_price"],
        "last_valid_price_at": _iso(last_at),
        "first_unavailable_at": _iso(first_missing or last_at),
        "first_unavailable_basis": "first Scout run without a current price of the exact pool"
        if first_missing
        else "no Scout run after the last price: the last price time",
        "exit_at": _iso(exit_at),
        "unavailable_minutes": (exit_at - last_at).total_seconds() / 60,
        "exit_mechanism": mechanism,
        "exit_trigger": trigger,
        "timeout_minutes": timeouts["market_unavailable_after_minutes"]
        if mechanism == STALE_TIMEOUT
        else timeouts["max_exit_delay_minutes"],
        "decision_reason": reason,
        "other_pool_appeared": bool(other_pools),
        "other_pools": other_pools,
        "collection_gap": collection_gap,
        "scout_runs_in_window": len(runs),
        "max_scout_run_gap_minutes": round(gap, 1),
        "unused_exact_pool_prices": [
            {
                "kind": o.kind,
                "record_id": o.record_id,
                "price_observed_at": _iso(o.price_at),
                "archived_at": _iso(o.archived_at),
                "price": o.price,
            }
            for o in unused[:5]
        ],  # fmt: skip
        "provider_failure_records": len(failures),
        "provider_failures_by_kind": dict(Counter(f"{o.kind}:{o.availability}" for o in failures)),
        "stale_carried_scout_records": len(stale_carried),
        "not_found_records": len(not_found),
        "liquidity_collapsed": collapsed,
        "entry_liquidity_usd": entry_liq,
        "last_liquidity_usd": last_liq,
        "retrospective": {
            "note": RETROSPECTIVE_NOTE,
            "searched_until": _iso(horizon),
            "later_exact_pool_observations": len(recovered),
            "later_recovered": bool(recovered),
            "first_recovery_at": _iso(first.price_at) if first else None,
            "first_recovery_price": first.price if first else None,
            "first_recovery_kind": first.kind if first else None,
        },
    }


def audit_unavailable(
    shadow: ShadowStore,
    evidence: EvidenceStore,
    run_id: str,
    strategies: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    settings: AuditSettings | None = None,
) -> dict[str, Any]:
    """Every MARKET_UNAVAILABLE exit of `run_id` (exit time in [since, until)), classified."""
    s = settings or AuditSettings()
    trades = [
        t for t in shadow.trades(run_id=run_id)
        if t["exit_reason"] == "MARKET_UNAVAILABLE" and t["final"]
        and (not strategies or t["strategy_id"] in strategies)
        and (since is None or t["exit_at"] >= since.timestamp())
        and (until is None or t["exit_at"] < until.timestamp())
    ]  # fmt: skip
    positions = {p["position_id"]: p for p in shadow.positions(run_id=run_id)}
    decisions = shadow.decisions_by_id([t["exit_decision_id"] for t in trades])
    timeouts: dict[tuple[str, int], dict[str, float]] = {}
    cases = []
    for t in trades:
        key = (t["strategy_id"], t["strategy_version"])
        if key not in timeouts:
            x = shadow.strategy(*key).exit
            timeouts[key] = {"market_unavailable_after_minutes": x.market_unavailable_after_minutes,
                             "max_exit_delay_minutes": x.max_exit_delay_minutes}  # fmt: skip
        cases.append(classify_case(
            shadow, evidence, t, positions[t["position_id"]],
            decisions.get(t["exit_decision_id"]), s, timeouts[key],
        ))  # fmt: skip
    return {
        "label": NOT_REAL_PROFIT,
        "run_id": run_id,
        "read_only": True,
        "window": {"since": _iso(since), "until": _iso(until), "on": "exit time"},
        "settings": {
            "collection_gap_minutes": s.collection_gap_minutes,
            "collapse_ratio": s.collapse_ratio,
            "recovery_hours": s.recovery_hours,
        },  # fmt: skip
        "rules": "first match wins: EVIDENCE_COLLECTION_GAP, POOL_CHANGED, PROVIDER_GAP, "
        "TRUE_MARKET_DISAPPEARANCE, TEMPORARY_UNAVAILABLE, LIQUIDITY_COLLAPSE, UNKNOWN",
        "retrospective_note": RETROSPECTIVE_NOTE,
        "summary": summary(cases),
        "cases": cases,
    }


def summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [c["unavailable_minutes"] for c in cases]
    recovered = sum(1 for c in cases if c["retrospective"]["later_recovered"])
    return {
        "total_market_unavailable": len(cases),
        "by_classification": {
            k: sum(1 for c in cases if c["classification"] == k) for k in CLASSES
        },  # fmt: skip
        "by_strategy": dict(sorted(Counter(c["strategy_id"] for c in cases).items())),
        "by_chain": dict(sorted(Counter(c["chain"] or "unknown" for c in cases).items())),
        "by_exit_mechanism": dict(sorted(Counter(c["exit_mechanism"] for c in cases).items())),
        # RETROSPECTIVE: later evidence of the exact pool (within the recovery horizon).
        "later_recovered": recovered,
        "permanent_disappearance": len(cases) - recovered,
        "permanent_disappearance_note": "no later observation of the exact pool in the "
        "stored evidence within the recovery horizon (not proof the market is gone)",
        "evidence_or_provider_gap": sum(
            1 for c in cases if c["classification"] in ("EVIDENCE_COLLECTION_GAP", "PROVIDER_GAP")
        ),
        "median_unavailable_minutes": statistics.median(durations) if durations else None,
        "unresolved_cost_usd": sum(c["cost_usd"] for c in cases),
    }


def text(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"run {report['run_id']}: MARKET_UNAVAILABLE audit (read-only; {report['label']})",
        report["rules"],
        report["retrospective_note"],
        "",
    ]
    for c in report["cases"]:
        r = c["retrospective"]
        lines += [
            f"{c['classification']}  {c['strategy_id']}  {c['chain']}:{c['address']} "
            f"({c['symbol'] or '?'})  pool {c['pool']}",
            f"  entry {c['entry_at']} @ {c['entry_price']:.10g}; last valid price "
            f"{c['last_valid_price']:.10g} at {c['last_valid_price_at']}",
            f"  first unavailable {c['first_unavailable_at']}; exit {c['exit_at']} after "
            f"{c['unavailable_minutes']:.0f} min ({c['exit_mechanism']}/{c['exit_trigger']}, "
            f"{c['timeout_minutes']:g} min)",
            f"  other pool: {', '.join(c['other_pools']) or 'no'}; collection gap: "
            f"{'yes' if c['collection_gap'] else 'no'} (max {c['max_scout_run_gap_minutes']} "
            f"min between {c['scout_runs_in_window']} Scout runs); provider failures: "
            f"{c['provider_failure_records']}, STALE_CARRIED: {c['stale_carried_scout_records']}"
            f", unused exact-pool prices: {len(c['unused_exact_pool_prices'])}",
            f"  RETROSPECTIVE later exact-pool observations: {r['later_exact_pool_observations']}"
            + (f", first recovery {r['first_recovery_at']} @ {r['first_recovery_price']:.10g}"
               if r["later_recovered"] else ""),
        ]  # fmt: skip
    med = s["median_unavailable_minutes"]
    lines += [
        "",
        f"total MARKET_UNAVAILABLE: {s['total_market_unavailable']}",
        "by classification:", *[f"  {k}: {v}" for k, v in s["by_classification"].items()],
        "by strategy:", *[f"  {k}: {v}" for k, v in s["by_strategy"].items()],
        "by chain:", *[f"  {k}: {v}" for k, v in s["by_chain"].items()],
        "by exit mechanism:", *[f"  {k}: {v}" for k, v in s["by_exit_mechanism"].items()],
        f"later recovered (retrospective): {s['later_recovered']}",
        f"permanent disappearance (no later exact-pool observation): "
        f"{s['permanent_disappearance']}",
        f"evidence/provider gap: {s['evidence_or_provider_gap']}",
        f"median unavailable duration: {'-' if med is None else f'{med:.0f} min'}",
        f"unresolved cost of these exits: ${s['unresolved_cost_usd']:,.2f}",
    ]  # fmt: skip
    return "\n".join(lines)
