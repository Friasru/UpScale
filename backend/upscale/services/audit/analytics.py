"""Descriptive analytics over the Audit dataset: groups, asset-aware samples, hypotheses.

Repeated observations of one token are not independent. Every group reports its raw
observations, its unique assets and an effective sample size that treats each asset's
observations as fully correlated (``n^2 / sum_a n_a^2``, the Shadow report's measure);
asset-weighted figures give each asset the same total weight. Comparisons behind a
hypothesis use asset-clustered (ratio-estimator) standard errors, so a token seen 50
times counts as one cluster, not 50 independent draws.

Nothing here is causal, and nothing here changes production. Hypotheses are phrased as
things for Calibration to test.
"""

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from upscale.services.audit.config import AuditConfig, Buckets
from upscale.services.audit.dataset import AuditObservation

BASELINE_STRATEGY = "random_eligible"
UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class Item:
    """One observation's label for one target (a horizon or the Shadow trade)."""

    asset_id: str
    return_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None
    pnl_usd: float | None = None  # Shadow trades only (profit factor in USD)


def horizon_item(o: AuditObservation, horizon: str) -> Item:
    h = o.horizons.get(horizon)
    if h is None:
        return Item(o.asset_id, None, None, None)
    return Item(o.asset_id, h.return_pct, h.mfe_pct, h.mae_pct)


def trade_item(o: AuditObservation) -> Item:
    t = o.trade
    if t is None or not t.resolved:
        return Item(o.asset_id, None, None, None)
    return Item(o.asset_id, t.return_pct, t.mfe_pct, t.mae_pct, t.pnl_usd)


# --- statistics ---------------------------------------------------------------------------------


def effective_sample_size(assets: Sequence[str]) -> float:
    if not assets:
        return 0.0
    return len(assets) ** 2 / sum(n * n for n in Counter(assets).values())


def trimmed_mean(values: Sequence[float], cut: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = int(len(v) * cut)
    kept = v[k : len(v) - k] if len(v) - 2 * k > 0 else v
    return sum(kept) / len(kept)


def _r(x: float | None, nd: int = 4) -> float | None:
    """Rounded for a stable, readable report (never used in a comparison)."""
    return None if x is None else round(x, nd)


def _median(v: Sequence[float]) -> float | None:
    """The median of an already sorted list."""
    if not v:
        return None
    k = len(v) // 2
    return v[k] if len(v) % 2 else (v[k - 1] + v[k]) / 2


def summarize(items: Sequence[Item], cfg: AuditConfig) -> dict[str, Any]:
    """Every descriptive figure of one group, in one pass over its items."""
    lab, mins = cfg.labels, cfg.minimums
    returns: list[float] = []
    mfe: list[float] = []
    mae: list[float] = []
    per_asset: dict[str, list[float]] = {}  # asset -> [n, wins, sum of returns]
    wins = cat = large = hi_mfe = hi_mae = pnl_n = 0
    gains = losses = pnl_gains = pnl_losses = 0.0
    for i in items:
        r = i.return_pct
        if r is None:
            continue
        returns.append(r)
        a = per_asset.get(i.asset_id)
        if a is None:
            a = per_asset[i.asset_id] = [0.0, 0.0, 0.0]
        a[0] += 1
        a[2] += r
        if r > 0:
            wins += 1
            a[1] += 1
            gains += r
        elif r < 0:
            losses -= r
        cat += r <= lab.catastrophic_pct
        large += r >= lab.large_winner_pct
        if i.mfe_pct is not None:
            mfe.append(i.mfe_pct)
            hi_mfe += i.mfe_pct >= lab.high_mfe_pct
        if i.mae_pct is not None:
            mae.append(i.mae_pct)
            hi_mae += i.mae_pct <= lab.high_mae_pct
        if i.pnl_usd is not None:
            pnl_n += 1
            if i.pnl_usd > 0:
                pnl_gains += i.pnl_usd
            else:
                pnl_losses -= i.pnl_usd
    n = len(returns)
    returns.sort()
    mfe.sort()
    mae.sort()
    k = len(per_asset)
    ess = n * n / sum(a[0] * a[0] for a in per_asset.values()) if n else 0.0
    reasons = []
    if n < mins.measured:
        reasons.append(f"{n} measured < {mins.measured}")
    if k < mins.unique_assets:
        reasons.append(f"{k} unique assets < {mins.unique_assets}")
    if ess < mins.effective_sample:
        reasons.append(f"effective sample {ess:.1f} < {mins.effective_sample:g}")
    if n and pnl_n == n:
        gains, losses, pf_basis = pnl_gains, pnl_losses, "USD P/L"
    else:
        pf_basis = "equal-weight returns"
    trim = int(n * cfg.trim)
    kept = returns[trim : n - trim] if n - 2 * trim > 0 else returns
    return {
        "observations": len(items),
        "measured": n,
        "unique_assets": k,
        "effective_sample_size": _r(ess, 2),
        "win_rate": _r(wins / n) if n else None,
        "asset_weighted_win_rate": _r(sum(a[1] / a[0] for a in per_asset.values()) / k) if k else None,
        "median_return_pct": _r(_median(returns)),
        "mean_return_pct": _r(sum(returns) / n) if n else None,
        "trimmed_mean_return_pct": _r(sum(kept) / len(kept)) if kept else None,
        "asset_weighted_mean_return_pct": _r(sum(a[2] / a[0] for a in per_asset.values()) / k) if k else None,
        "median_mfe_pct": _r(_median(mfe)),
        "median_mae_pct": _r(_median(mae)),
        "catastrophic_rate": _r(cat / n) if n else None,
        "large_winner_rate": _r(large / n) if n else None,
        "high_mfe_rate": _r(hi_mfe / len(mfe)) if mfe else None,
        "high_mae_rate": _r(hi_mae / len(mae)) if mae else None,
        "profit_factor": _r(gains / losses) if losses > 0 else None,
        "profit_factor_basis": pf_basis if losses > 0 else "undefined: no loss",
        "status": "INSUFFICIENT_SAMPLE" if reasons else "DESCRIPTIVE",
        "insufficient_reasons": reasons,
    }  # fmt: skip


# --- dimensions ---------------------------------------------------------------------------------------


def _fmt(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def bucket(value: float | None, edges: Sequence[float], unit: str = "") -> str:
    """Lower bound inclusive; below the first edge is its own bucket; None: UNAVAILABLE."""
    if value is None:
        return UNAVAILABLE
    for i, e in enumerate(edges):
        if value < e:
            lo = edges[i - 1] if i else None
            return f"<{_fmt(e)}{unit}" if lo is None else f"{_fmt(lo)}-{_fmt(e)}{unit}"
    return f">={_fmt(edges[-1])}{unit}"


def bucket_order(edges: Sequence[float], unit: str = "") -> list[str]:
    return (
        [f"<{_fmt(edges[0])}{unit}"]
        + [f"{_fmt(a)}-{_fmt(b)}{unit}" for a, b in zip(edges, edges[1:], strict=False)]
        + [f">={_fmt(edges[-1])}{unit}", UNAVAILABLE]
    )


def _risk(v: float | None, edges: Sequence[float]) -> str:
    if v is None:
        return UNAVAILABLE
    if v <= 0:
        return "0"
    return bucket(v, edges[1:]).replace("<5", "0-5")


def _delay(o: AuditObservation, b: Buckets) -> str:
    if o.execution is None:
        return "NOT_FILLED"
    v = o.execution.values.get("execution_delay_seconds")
    if not isinstance(v, int | float):
        return UNAVAILABLE
    edges = b.execution_delay_seconds
    labels = ["<=1m", "1-5m", "5-15m", "15-60m", ">60m"]
    for e, label in zip(edges, labels, strict=False):
        if v < e:
            return label
    return labels[len(edges)]


def _tech_confirmation(o: AuditObservation) -> str:
    g = o.snapshot.groups.get("technical")
    if g is None or not g.available:
        return f"{UNAVAILABLE} ({g.availability if g else 'NOT_AVAILABLE'})"
    return "CONFIRMED" if g.values.get("technical_confirmed") else "NOT_CONFIRMED"


def _authority(o: AuditObservation) -> str:
    s = o.snapshot
    vals = (s.value("mint_authority_active"), s.value("freeze_authority_active"))
    if any(v is True for v in vals):
        return "AUTHORITY_ACTIVE"
    if all(v is False for v in vals):
        return "NO_ACTIVE_AUTHORITY"
    return UNAVAILABLE


Dimension = tuple[str, str, Callable[[AuditObservation], str]]


def dimensions(cfg: AuditConfig, population: str) -> list[Dimension]:
    """(key, description, bucket function). Execution delay and strategy: Shadow only."""
    b = cfg.buckets

    def text(name: str) -> Callable[[AuditObservation], str]:
        return lambda o: o.snapshot.text(name) or UNAVAILABLE

    dims: list[Dimension] = [
        ("score_bucket", "Scout score", lambda o: bucket(o.snapshot.num("score"), b.score)),
        ("stage", "Scout stage", text("stage")),
        ("technical_confirmation", "Technical confirmation (Scout snapshot up-trend, >= 3 snapshots)",
         _tech_confirmation),
        ("technical_trend", "Technical trend (Scout snapshot context)", text("technical_trend")),
        ("liquidity_bucket", "Liquidity (USD)",
         lambda o: bucket(o.snapshot.num("liquidity_usd"), b.liquidity_usd)),
        ("market_cap_bucket", "Market cap (USD)",
         lambda o: bucket(o.snapshot.num("market_cap_usd"), b.market_cap_usd)),
        ("social_status", "Social momentum status", text("social_status")),
        ("risk_penalty_bucket", "Risk penalty (points)",
         lambda o: _risk(o.snapshot.num("risk_penalty"), b.risk_penalty)),
        ("safety_status", "Safety availability", text("safety_status")),
        ("authority", "Mint / freeze authority", _authority),
        ("blocking_risk_flags", "High / critical risk flags",
         lambda o: "PRESENT" if (o.snapshot.num("blocking_risk_flags") or 0) > 0 else "NONE"),
        ("stage_x_technical", "Stage x technical confirmation",
         lambda o: f"{o.snapshot.text('stage') or UNAVAILABLE} / {_tech_confirmation(o)}"),
        ("social_x_technical_trend", "Social status x technical trend",
         lambda o: f"{o.snapshot.text('social_status') or UNAVAILABLE} / "
                   f"{o.snapshot.text('technical_trend') or UNAVAILABLE}"),
    ]  # fmt: skip
    if population == "shadow":
        dims += [
            ("execution_delay_bucket", "Entry execution delay (intent to fill observation)",
             lambda o: _delay(o, b)),
            ("strategy", "Strategy", lambda o: o.strategy or UNAVAILABLE),
        ]  # fmt: skip
    return dims


def _sort_key(dim: str, cfg: AuditConfig) -> Callable[[str], tuple[int, str]]:
    b = cfg.buckets
    orders = {
        "score_bucket": bucket_order(b.score),
        "liquidity_bucket": bucket_order(b.liquidity_usd),
        "market_cap_bucket": bucket_order(b.market_cap_usd),
        "risk_penalty_bucket": ["0", "0-5", "5-10", "10-20", ">=20", UNAVAILABLE],
        "execution_delay_bucket": ["<=1m", "1-5m", "5-15m", "15-60m", ">60m", UNAVAILABLE,
                                   "NOT_FILLED"],
        "stage": ["NEW", "EARLY", "ACCELERATING", "CROWDED", "FADING", "STEADY",
                  "INSUFFICIENT_DATA", UNAVAILABLE],
    }  # fmt: skip
    order = orders.get(dim, [])
    return lambda k: (order.index(k) if k in order else len(order), k)


Labeled = list[tuple[str, str, list[str]]]  # (dimension, description, bucket per observation)


def label(observations: Sequence[AuditObservation], cfg: AuditConfig, population: str) -> Labeled:
    """Every observation's bucket in every dimension (computed once, reused per target)."""
    return [
        (key, desc, [f(o) for o in observations]) for key, desc, f in dimensions(cfg, population)
    ]


def grouped(items: Sequence[Item], labeled: Labeled, cfg: AuditConfig) -> dict[str, dict[str, Any]]:
    """`items[i]` is observation i's label for one target; `labeled` its buckets."""
    out: dict[str, dict[str, Any]] = {}
    for key, desc, buckets in labeled:
        groups: dict[str, list[Item]] = {}
        for b, item in zip(buckets, items, strict=True):
            groups.setdefault(b, []).append(item)
        ordered = sorted(groups, key=_sort_key(key, cfg))
        out[key] = {"description": desc,
                    "buckets": {k: summarize(groups[k], cfg) for k in ordered}}  # fmt: skip
    return out


# --- comparisons and hypotheses ------------------------------------------------------------------------


@dataclass(frozen=True)
class Ratio:
    """A ratio estimator over asset clusters: sum(y) / sum(n) with its cluster-robust SE."""

    value: float
    se: float
    n: int
    assets: int
    ess: float


Agg = dict[str, tuple[int, float]]  # asset -> (observations, sum of the metric)


def ratio_of(agg: Agg) -> Ratio | None:
    """sum(y) / sum(n) over asset clusters. SE: sqrt(k / (k - 1) * sum_a (y_a - R n_a)^2) /
    sum(n), the cluster-robust (linearized) variance of a ratio estimator; infinite with a
    single asset."""
    cells = [(n, y) for n, y in agg.values() if n > 0]
    total = sum(n for n, _ in cells)
    if total == 0:
        return None
    r = sum(y for _, y in cells) / total
    k = len(cells)
    se = (
        math.inf
        if k < 2
        else math.sqrt(k / (k - 1) * sum((y - r * n) ** 2 for n, y in cells)) / total
    )
    return Ratio(r, se, total, k, total * total / sum(n * n for n, _ in cells))


def ratio(pairs: Sequence[tuple[str, float]]) -> Ratio | None:
    agg: dict[str, tuple[int, float]] = {}
    for a, y in pairs:
        n, s = agg.get(a, (0, 0.0))
        agg[a] = (n + 1, s + y)
    return ratio_of(agg)


METRICS: tuple[tuple[str, str, str], ...] = (
    # (key, label, kind): rate in percentage points, or mean in percent
    ("catastrophic", "catastrophic-loss rate", "rate"),
    ("large_winner", "large-winner rate", "rate"),
    ("win", "win rate", "rate"),
    ("mae", "mean MAE", "mean"),
    ("return", "clipped mean return", "mean"),
)


def aggregates(items: Sequence[Item], cfg: AuditConfig) -> dict[str, Agg]:
    """Per metric, per asset: (observations, sum). Rates are 0 / 100 per observation."""
    lab, clip = cfg.labels, cfg.hypotheses.return_clip_pct
    raw: dict[str, dict[str, list[float]]] = {m: {} for m, _, _ in METRICS}

    def add(metric: str, asset: str, y: float) -> None:
        cell = raw[metric].get(asset)
        if cell is None:
            raw[metric][asset] = [1.0, y]
        else:
            cell[0] += 1
            cell[1] += y

    for i in items:
        r = i.return_pct
        if r is None:
            continue
        add("catastrophic", i.asset_id, 100.0 * (r <= lab.catastrophic_pct))
        add("large_winner", i.asset_id, 100.0 * (r >= lab.large_winner_pct))
        add("win", i.asset_id, 100.0 * (r > 0))
        add("return", i.asset_id, max(-clip, min(clip, r)))
        if i.mae_pct is not None:
            add("mae", i.asset_id, i.mae_pct)
    return {m: {a: (int(c[0]), c[1]) for a, c in cells.items()} for m, cells in raw.items()}


def subtract(total: dict[str, Agg], part: dict[str, Agg]) -> dict[str, Agg]:
    """`total` without `part` (the rest of a dimension), per metric and asset."""
    out: dict[str, Agg] = {}
    for m, cells in total.items():
        mine = part.get(m, {})
        rest: Agg = {}
        for a, (n, y) in cells.items():
            pn, py = mine.get(a, (0, 0.0))
            if n - pn > 0:
                rest[a] = (n - pn, y - py)
        out[m] = rest
    return out


def _enough(r: Ratio, cfg: AuditConfig) -> bool:
    m = cfg.minimums
    return r.n >= m.measured and r.assets >= m.unique_assets and r.ess >= m.effective_sample


def compare_aggs(a: dict[str, Agg], b: dict[str, Agg], cfg: AuditConfig) -> list[dict[str, Any]]:
    """Every metric of group `a` against group `b` (descriptive; see `HypothesisRules`)."""
    rules = cfg.hypotheses
    out = []
    for metric, name, kind in METRICS:
        ra, rb = ratio_of(a.get(metric, {})), ratio_of(b.get(metric, {}))
        if ra is None or rb is None:
            continue
        diff = ra.value - rb.value
        se = math.sqrt(ra.se**2 + rb.se**2)
        effect = rules.min_rate_effect if kind == "rate" else rules.min_return_effect
        sufficient = _enough(ra, cfg) and _enough(rb, cfg)
        # A zero SE (no spread on either side) is decided by the effect size alone.
        strong = abs(diff) >= effect and (
            se == 0 or (math.isfinite(se) and abs(diff) >= rules.z * se)
        )
        out.append({
            "metric": metric, "label": name, "kind": kind,
            "group": round(ra.value, 3), "other": round(rb.value, 3), "difference": round(diff, 3),
            "standard_error": round(se, 3) if math.isfinite(se) else None,
            "z": round(diff / se, 2) if 0 < se < math.inf else None,
            "group_n": ra.n, "other_n": rb.n, "group_assets": ra.assets, "other_assets": rb.assets,
            "group_ess": round(ra.ess, 1), "other_ess": round(rb.ess, 1),
            "sufficient_sample": sufficient, "surfaced": sufficient and strong,
            "notable_but_insufficient": abs(diff) >= effect and not sufficient,
        })  # fmt: skip
    return out


def compare(a: Sequence[Item], b: Sequence[Item], cfg: AuditConfig) -> list[dict[str, Any]]:
    return compare_aggs(aggregates(a, cfg), aggregates(b, cfg), cfg)


_PHRASES = {
    "catastrophic": ("a higher catastrophic-loss frequency", "a lower catastrophic-loss frequency"),
    "large_winner": ("a higher large-winner frequency", "a lower large-winner frequency"),
    "win": ("a higher win rate", "a lower win rate"),
    "mae": ("a shallower mean MAE (smaller adverse excursions)", "a deeper mean MAE (larger adverse excursions)"),
    "return": ("a higher clipped mean return", "a lower clipped mean return"),
}  # fmt: skip


def _phrase(c: dict[str, Any]) -> str:
    up, down = _PHRASES[c["metric"]]
    return up if c["difference"] > 0 else down


def _join(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def hypothesis(
    population: str, dimension: str, bucket: str, subject: str, other: str, target: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:  # fmt: skip
    """One hypothesis per group: every metric that met the rules, strongest first."""
    evidence = sorted(
        evidence, key=lambda c: (-abs(c["z"] if c["z"] is not None else 999.0), c["metric"])
    )
    phrase = _join([_phrase(c) for c in evidence])
    return {
        "population": population, "dimension": dimension, "bucket": bucket, "target": target,
        "observation": f"{subject} appear to show {phrase} than {other} ({target}).",
        "hypothesis": f"Candidate hypothesis for Calibration: test whether {subject} are "
        f"associated with {phrase} than {other} ({target}), out of sample.",
        # No spread at all (z undefined): strongest.
        "strength_z": max(abs(c["z"]) if c["z"] is not None else 999.0 for c in evidence),
        "evidence": evidence,
    }  # fmt: skip


def scout_hypotheses(
    observations: Sequence[AuditObservation], cfg: AuditConfig, labeled: Labeled | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Each bucket of each dimension against the rest of that dimension, at the primary
    horizon (per-asset aggregates: the rest is the total minus the bucket)."""
    h = cfg.primary_horizon
    items = [horizon_item(o, h) for o in observations]
    total = aggregates(items, cfg)
    found: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for key, desc, buckets in labeled if labeled is not None else label(observations, cfg, "scout"):
        groups: dict[str, list[Item]] = {}
        for b, item in zip(buckets, items, strict=True):
            groups.setdefault(b, []).append(item)
        if len(groups) < 2:
            continue
        # Two buckets: one comparison (the other is its mirror image).
        for name in sorted(groups)[:1] if len(groups) == 2 else sorted(groups):
            mine = aggregates(groups[name], cfg)
            met = []
            for c in compare_aggs(mine, subtract(total, mine), cfg):
                counts["tested"] += 1
                counts["notable_but_insufficient"] += c["notable_but_insufficient"]
                if c["surfaced"]:
                    met.append(c)
            if met:
                found.append(hypothesis(
                    "scout", key, name, f"Scout anchors with {desc} = {name}",
                    "the other buckets of that dimension", f"{h} horizon", met,
                ))  # fmt: skip
    return found, dict(counts)


def shadow_hypotheses(
    observations: Sequence[AuditObservation], cfg: AuditConfig
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per run, each strategy's resolved trades against the random baseline's."""
    found: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    runs = sorted({o.run_id for o in observations if o.run_id})
    for run in runs:
        mine = [o for o in observations if o.run_id == run]
        base = [trade_item(o) for o in mine if o.strategy_id == BASELINE_STRATEGY]
        if not base:
            continue
        for sid in sorted({o.strategy_id for o in mine if o.strategy_id} - {BASELINE_STRATEGY}):
            items = [trade_item(o) for o in mine if o.strategy_id == sid]
            met = []
            for c in compare(items, base, cfg):
                counts["tested"] += 1
                counts["notable_but_insufficient"] += c["notable_but_insufficient"]
                if c["surfaced"]:
                    met.append(c)
            if met:
                found.append(hypothesis(
                    "shadow", "strategy", sid, f"{sid} trades", f"{BASELINE_STRATEGY} trades",
                    f"trade outcome, run {run}", met,
                ))  # fmt: skip
    return found, dict(counts)


def rank_hypotheses(found: list[dict[str, Any]], cfg: AuditConfig) -> list[dict[str, Any]]:
    """Strongest first (largest |z| of any metric), deterministic ties, at most `max_listed`."""
    ordered = sorted(
        found,
        key=lambda h: (-h["strength_z"], h["population"], h["dimension"], h["bucket"], h["target"]),
    )
    return ordered[: cfg.hypotheses.max_listed]
