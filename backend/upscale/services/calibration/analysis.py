"""Calibration analyses. Everything here reads observations and reports associations; none
of it changes production scoring, stages, thresholds, confidence or decisions.

Scout signal analyses use observations that carry a Scout evaluation (live Scout anchors,
replay samples); Opportunity analyses use observations that carry a decision (live Analyze
decisions, replay samples). Every table is reported per origin (LIVE_FORWARD,
HISTORICAL_REPLAY) and combined, the combined row showing its origin composition.
"""

import hashlib
import itertools
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from upscale.services.calibration.config import HORIZONS, CalibrationConfig
from upscale.services.calibration.dataset import FAMILIES, Observation
from upscale.services.calibration.stats import (
    Cohort,
    by,
    cohort,
    meets,
    outcome,
    spearman,
    strength,
)
from upscale.services.scout.growth.config import GrowthConfig

SCOUT_KINDS = frozenset({"scout", "replay"})
DECISION_KINDS = frozenset({"decision", "replay"})
STAGES = ("NEW", "EARLY", "ACCELERATING", "STEADY", "CROWDED", "FADING", "INSUFFICIENT_DATA")
LANGUAGE = (
    "Associations in this dataset only: not causes, not predictions and not realized trading "
    "profit (no entry, fill, fee, slippage or size is assumed)."
)


def band(value: float | None, edges: Sequence[float]) -> str | None:
    if value is None:
        return None
    if value < edges[0]:
        return f"<{edges[0]:g}"
    for lo, hi in itertools.pairwise(edges):
        if lo <= value < hi:
            return f"{lo:g}..{hi:g}"
    return f">={edges[-1]:g}"


def category(value: Any) -> str:
    return "unknown" if value is None else str(value)


def family_scores(
    o: Observation, weights: dict[str, float] | None = None
) -> dict[str, float] | None:
    """Each family's 0..1 score, from its stored contribution and the production weights."""
    w = weights or production_weights()
    out = {}
    for f in FAMILIES:
        c = o.f(f"family_{f}")
        if c is None or w[f] <= 0:
            return None
        out[f] = c / (100 * w[f])
    return out


def production_weights() -> dict[str, float]:
    raw = GrowthConfig().weights.as_dict()
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# Feature -> (which observations, bucket function, missing-evidence label)
Bucketer = Callable[[Observation], Any]
FEATURES: dict[str, tuple[frozenset[str], Bucketer, str | None]] = {
    "stage": (SCOUT_KINDS, lambda o: o.s("stage"), None),
    "score": (SCOUT_KINDS, lambda o: band(o.f("score"), (20, 40, 50, 60, 70, 80)), None),
    **{
        f"family_{f}": (
            SCOUT_KINDS,
            (lambda name: lambda o: band((family_scores(o) or {}).get(name), (0.4, 0.6)))(f),
            None,
        )
        for f in FAMILIES
    },
    "stage_adjustment": (SCOUT_KINDS, lambda o: band(o.f("stage_adjustment"), (-5, 0, 5)), None),
    "risk_penalty": (SCOUT_KINDS, lambda o: band(o.f("risk_penalty"), (5, 15, 30)), None),
    "liquidity_usd": (
        SCOUT_KINDS | DECISION_KINDS,
        lambda o: band(o.f("liquidity_usd"), (10e3, 25e3, 50e3, 100e3, 250e3, 1e6)),
        "LIQUIDITY_NOT_AVAILABLE",
    ),  # fmt: skip
    "market_cap_usd": (
        SCOUT_KINDS,
        lambda o: band(o.f("market_cap_usd"), (1e5, 1e6, 1e7, 1e8)),
        "MARKET_CAP_NOT_AVAILABLE",
    ),  # fmt: skip
    "fdv_usd": (SCOUT_KINDS, lambda o: band(o.f("fdv_usd"), (1e5, 1e6, 1e7, 1e8)), None),
    "liquidity_to_cap": (
        SCOUT_KINDS,
        lambda o: band(o.f("liquidity_to_cap"), (0.02, 0.05, 0.1, 0.2)),
        None,
    ),
    "volume_acceleration": (
        SCOUT_KINDS,
        lambda o: band(o.f("volume_acceleration"), (0.8, 1.25, 2, 4)),
        None,
    ),
    "txn_acceleration": (
        SCOUT_KINDS,
        lambda o: band(o.f("txn_acceleration"), (0.8, 1.25, 2, 4)),
        None,
    ),
    "buy_share_h1": (SCOUT_KINDS, lambda o: band(o.f("buy_share_h1"), (0.4, 0.5, 0.6)), None),
    "buy_pressure_change": (
        SCOUT_KINDS,
        lambda o: band(o.f("buy_pressure_change"), (-0.05, 0, 0.05)),
        None,
    ),
    "prior_move_h24_pct": (
        SCOUT_KINDS,
        lambda o: band(o.f("price_change_h24_pct"), (-20, 0, 20, 50, 100)),
        None,
    ),
    "price_velocity_h1_pct": (
        SCOUT_KINDS,
        lambda o: band(o.f("price_change_h1_pct"), (-10, 0, 10, 30)),
        None,
    ),
    "volatility_4h_pct": (SCOUT_KINDS, lambda o: band(o.f("volatility_4h_pct"), (2, 5, 10)), None),
    "pool_age_hours": (SCOUT_KINDS, lambda o: band(o.f("pool_age_hours"), (1, 6, 24, 168)), None),
    "liquidity_change_pct": (
        SCOUT_KINDS,
        lambda o: band(o.f("liquidity_change_pct"), (-20, 0, 20)),
        None,
    ),
    "technical_trend": (
        DECISION_KINDS,
        lambda o: category(o.s("technical_trend")),
        "TECHNICAL_NOT_AVAILABLE",
    ),
    "rsi_14": (
        DECISION_KINDS,
        lambda o: band(o.f("rsi_14"), (30, 50, 70)),
        "TECHNICAL_NOT_AVAILABLE",
    ),
    "safety_status": (
        SCOUT_KINDS,
        lambda o: category(o.s("safety_status")),
        "SAFETY_NOT_AVAILABLE",
    ),
    "mint_authority_active": (
        SCOUT_KINDS,
        lambda o: category(o.features.get("mint_authority_active")),
        "SAFETY_NOT_AVAILABLE",
    ),  # fmt: skip
    "freeze_authority_active": (
        SCOUT_KINDS,
        lambda o: category(o.features.get("freeze_authority_active")),
        "SAFETY_NOT_AVAILABLE",
    ),  # fmt: skip
    "holder_top1_pct": (
        SCOUT_KINDS,
        lambda o: band(o.f("holder_top1_pct"), (5, 10, 20)),
        "SAFETY_NOT_AVAILABLE",
    ),
    "holder_top10_pct": (
        SCOUT_KINDS,
        lambda o: band(o.f("holder_top10_pct"), (30, 40, 50)),
        "SAFETY_NOT_AVAILABLE",
    ),
    "holder_data_incomplete": (
        SCOUT_KINDS,
        lambda o: category(o.features.get("holder_data_lower_bound")),
        "SAFETY_NOT_AVAILABLE",
    ),  # fmt: skip
    "social_status": (
        SCOUT_KINDS,
        lambda o: category(o.s("social_status")),
        "SOCIAL_NOT_AVAILABLE",
    ),
    "social_cross_platform": (
        SCOUT_KINDS,
        lambda o: category(o.features.get("social_cross_platform")),
        "SOCIAL_NOT_AVAILABLE",
    ),  # fmt: skip
    "social_spam_risk": (
        SCOUT_KINDS,
        lambda o: category(o.s("social_spam_risk")),
        "SOCIAL_NOT_AVAILABLE",
    ),
    "social_attribution": (
        SCOUT_KINDS,
        lambda o: category(o.s("social_attribution")),
        "SOCIAL_NOT_AVAILABLE",
    ),
    "risk_flag": (SCOUT_KINDS, lambda o: sorted(o.flags) or ["none"], None),
    "risk_level": (DECISION_KINDS, lambda o: category(o.s("risk_level")), None),
    "uncertainty_level": (DECISION_KINDS, lambda o: category(o.s("uncertainty_level")), None),
    "action": (DECISION_KINDS, lambda o: category(o.s("action")), None),
    "confidence": (DECISION_KINDS, lambda o: category(o.s("confidence")), None),
    "has_trigger": (DECISION_KINDS, lambda o: category(o.features.get("has_trigger")), None),
    "has_invalidation": (
        DECISION_KINDS,
        lambda o: category(o.features.get("has_invalidation")),
        None,
    ),
    "missing_evidence": (
        SCOUT_KINDS | DECISION_KINDS,
        lambda o: sorted(o.missing) or ["NONE"],
        None,
    ),
    "regime_volatility": (SCOUT_KINDS, lambda o: o.regime.get("volatility", "UNKNOWN"), None),
    "regime_direction": (SCOUT_KINDS, lambda o: o.regime.get("direction", "UNKNOWN"), None),
    "regime_meme_activity": (SCOUT_KINDS, lambda o: o.regime.get("meme_activity", "UNKNOWN"), None),
    "regime_liquidity": (SCOUT_KINDS, lambda o: o.regime.get("liquidity", "UNKNOWN"), None),
}


class FeatureRow(BaseModel):
    feature: str
    bucket: str
    horizon: str
    origin: str  # LIVE_FORWARD / HISTORICAL_REPLAY / COMBINED
    split: str
    cohort: Cohort


class Finding(BaseModel):
    finding_id: str
    kind: str
    subject: str
    horizon: str
    statement: str
    strength: str
    rule: str
    calibration: dict[str, Any]
    validation: dict[str, Any] | None = None
    origins: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def _for(observations: Sequence[Observation], kinds: frozenset[str]) -> list[Observation]:
    return [o for o in observations if o.kind in kinds]


ORIGIN_VIEWS = ("LIVE_FORWARD", "HISTORICAL_REPLAY", "COMBINED")


def _origin(observations: Sequence[Observation], view: str) -> list[Observation]:
    if view == "COMBINED":
        return [o for o in observations if o.origin != "SHADOW"]
    return [o for o in observations if o.origin == view]


def feature_rows(
    observations: Sequence[Observation], horizon: str, cfg: CalibrationConfig, split: str
) -> list[FeatureRow]:
    rows = []
    for name, (kinds, key, missing) in FEATURES.items():
        scoped = _for(observations, kinds)
        for view in ORIGIN_VIEWS:
            for bucket, members in sorted(by(_origin(scoped, view), key).items()):
                if bucket == "None":
                    bucket = "unavailable"
                rows.append(
                    FeatureRow(
                        feature=name,
                        bucket=bucket,
                        horizon=horizon,
                        origin=view,
                        split=split,
                        cohort=cohort(f"{name}={bucket}", members, horizon, cfg, missing),
                    )  # fmt: skip
                )
    return rows


def _groups(observations: Sequence[Observation], name: str) -> dict[str, list[Observation]]:
    kinds, key, _ = FEATURES[name]
    return {
        ("unavailable" if k == "None" else k): v
        for k, v in by(_for(observations, kinds), key).items()
    }


def _fid(*parts: str) -> str:
    return "f-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


def _summary(c: Cohort) -> dict[str, Any]:
    return c.model_dump(include={"status", "measured", "assets", "effective_n", "origins", "median_return",
                                 "mean_return", "trimmed_mean_return", "extreme_returns", "median_ci", "median_mfe", "median_mae",
                                 "median_drawdown", "positive_share", "missing_rate"})  # fmt: skip


def _diff(a: Cohort, b: Cohort, metric: str = "median_return") -> float | None:
    x, y = getattr(a, metric), getattr(b, metric)
    return None if x is None or y is None else x - y


def compare_buckets(
    cal: Sequence[Observation],
    val: Sequence[Observation],
    feature: str,
    horizon: str,
    cfg: CalibrationConfig,
) -> Finding | None:
    """The two supported buckets whose median returns differ most, checked across horizons,
    on VALIDATION and per origin. None without two supported buckets or a material gap."""
    missing = FEATURES[feature][2]
    groups = _groups(cal, feature)
    cohorts = {
        k: cohort(k, v, horizon, cfg, missing) for k, v in groups.items() if k != "unavailable"
    }
    ok = {k: c for k, c in cohorts.items() if c.status == "OK"}
    if len(ok) < 2:
        return None
    hi = max(ok, key=lambda k: (ok[k].median_return or 0, k))
    lo = min(ok, key=lambda k: (ok[k].median_return or 0, k))
    gap = _diff(ok[hi], ok[lo])
    if gap is None or gap < cfg.material_return_pp:
        return None

    def sign_at(obs: Sequence[Observation], h: str) -> bool | None:
        g = _groups(obs, feature)
        a = cohort(hi, g.get(hi, []), h, cfg)
        b = cohort(lo, g.get(lo, []), h, cfg)
        d = _diff(a, b) if a.status == b.status == "OK" else None
        return None if d is None else d > 0

    horizons_agree = sum(1 for h in HORIZONS if sign_at(cal, h) is True)
    validation = sign_at(val, horizon)
    live = sign_at(_origin(cal, "LIVE_FORWARD"), horizon)
    replay = sign_at(_origin(cal, "HISTORICAL_REPLAY"), horizon)
    origins_agree = None if live is None or replay is None else live == replay
    label, rule = strength(ok[hi], cfg.samples, horizons_agree, validation, origins_agree)
    notes = []
    if live is None and replay is True:
        notes.append("replay-only pattern: LIVE_FORWARD sample too small to confirm")
    if replay is None and live is True:
        notes.append("live-only pattern: no supporting HISTORICAL_REPLAY sample")
    if origins_agree is False:
        notes.append("LIVE_FORWARD and HISTORICAL_REPLAY disagree")
    counts = {o: ok[hi].origins.get(o, 0) for o in ("LIVE_FORWARD", "HISTORICAL_REPLAY")}
    if counts["HISTORICAL_REPLAY"] > 5 * max(1, counts["LIVE_FORWARD"]):
        notes.append(
            f"origin imbalance: replay {counts['HISTORICAL_REPLAY']} vs live {counts['LIVE_FORWARD']}"
        )
    vg = _groups(val, feature)
    return Finding(
        finding_id=_fid("bucket", feature, hi, lo, horizon),
        kind="feature_effect",
        subject=f"{feature}: {hi} vs {lo}",
        horizon=horizon,
        statement=(
            f"{feature} = {hi} is associated with a higher median {horizon} return than {lo} in this "
            f"dataset ({ok[hi].median_return:+.2f}% vs {ok[lo].median_return:+.2f}%; median MAE "
            f"{ok[hi].median_mae}% vs {ok[lo].median_mae}%)."
        ),
        strength=label,
        rule=rule,
        calibration={"high": _summary(ok[hi]), "low": _summary(ok[lo]), "gap_pp": round(gap, 4)},
        validation={
            "agrees": validation,
            "high": _summary(cohort(hi, vg.get(hi, []), horizon, cfg)),
            "low": _summary(cohort(lo, vg.get(lo, []), horizon, cfg)),
        },
        origins={"live_agrees": live, "replay_agrees": replay, "horizons_agreeing": horizons_agree},
        notes=notes,
    )


def stage_calibration(
    cal: Sequence[Observation], val: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> dict[str, Any]:
    groups = _groups(cal, "stage")
    cohorts = {s: cohort(s, groups.get(s, []), horizon, cfg) for s in STAGES}
    ok = {s: c for s, c in cohorts.items() if c.status == "OK"}
    questions: dict[str, Any] = {}

    def ask(name: str, a: str, b: str, metric: str, higher: bool) -> None:
        if a in ok and b in ok:
            d = _diff(ok[a], ok[b], metric)
            if d is None:
                questions[name] = "INSUFFICIENT_SAMPLE"
            elif abs(d) < (cfg.material_mae_pp if "mae" in metric else cfg.material_return_pp):
                questions[name] = f"no material difference ({metric} {d:+.2f} pp)"
            else:
                yes = d > 0 if higher else d < 0
                questions[name] = f"{'yes' if yes else 'no'} ({metric} {d:+.2f} pp)"
        else:
            questions[name] = "INSUFFICIENT_SAMPLE"

    ask(
        "ACCELERATING stronger continuation than EARLY",
        "ACCELERATING",
        "EARLY",
        "median_return",
        True,
    )
    ask(
        "CROWDED worse forward risk than STEADY (deeper MAE)",
        "CROWDED",
        "STEADY",
        "median_mae",
        False,
    )
    ask("FADING lower continuation than STEADY", "FADING", "STEADY", "median_return", False)
    ask("ACCELERATING deeper MAE than EARLY", "ACCELERATING", "EARLY", "median_mae", False)
    indistinct = [
        f"{a} ~ {b}"
        for a, b in itertools.combinations(sorted(ok), 2)
        if (d := _diff(ok[a], ok[b])) is not None and abs(d) < cfg.material_return_pp
        and _overlap(ok[a].median_ci, ok[b].median_ci)
    ]  # fmt: skip
    transitions = _transitions(cal, horizon, cfg)
    return {
        "stages": {s: _summary(c) for s, c in cohorts.items()},
        "questions": questions,
        "indistinguishable": indistinct,
        "transitions": transitions,
        "validation": {
            s: _summary(cohort(s, _groups(val, "stage").get(s, []), horizon, cfg)) for s in STAGES
        },
    }


def _overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    return a is not None and b is not None and a[0] <= b[1] and b[0] <= a[1]


def _transitions(
    obs: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> dict[str, Any]:
    """Stage changes between an asset's consecutive observations (same origin)."""
    by_asset: dict[tuple[str, str], list[Observation]] = {}
    for o in _for(obs, SCOUT_KINDS):
        if o.s("stage"):
            by_asset.setdefault((o.origin, o.asset_id), []).append(o)
    groups: dict[str, list[Observation]] = {}
    for items in by_asset.values():
        items.sort(key=lambda o: o.at)
        for a, b in itertools.pairwise(items):
            if a.s("stage") != b.s("stage"):
                groups.setdefault(f"{a.s('stage')}->{b.s('stage')}", []).append(b)
    return {k: _summary(cohort(k, v, horizon, cfg)) for k, v in sorted(groups.items())}


def confidence_calibration(
    cal: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> dict[str, Any]:
    def verdict(cohorts: dict[str, Cohort], order: Sequence[str]) -> str:
        ok = [(k, cohorts[k]) for k in order if k in cohorts and cohorts[k].status == "OK"]
        if len(ok) < 2:
            return "INSUFFICIENT_SAMPLE"
        meds = [c.median_return or 0.0 for _, c in ok]
        spread = meds[-1] - meds[0]
        if abs(spread) < cfg.material_return_pp:
            return "LITTLE_SEPARATION: confidence adds little in this dataset"
        rho = spearman(list(range(len(meds))), meds)
        if spread < 0:
            return "INVERTED: higher confidence behaved worse (possibly overconfident)"
        if rho is not None and rho < 0.8:
            return "PARTIAL_SEPARATION: better at the top, not monotonic"
        return "SEPARATES: higher confidence associated with stronger outcomes"

    score_edges = ("0-20", "20-40", "40-60", "60-80", "80-100")
    scout = [o for o in _for(cal, SCOUT_KINDS) if o.f("score") is not None]

    def score_bucket(o: Observation) -> str:
        s = o.f("score") or 0.0
        return score_edges[min(4, int(s // 20))]

    score_cohorts = {k: cohort(k, v, horizon, cfg) for k, v in by(scout, score_bucket).items()}
    pairs = [(o.f("score"), outcome(o, horizon)) for o in scout]
    measured = [
        (s, v.return_pct)
        for s, v in pairs
        if s is not None and v is not None and v.return_pct is not None
    ]
    decisions = _for(cal, DECISION_KINDS)
    conf = {
        k: cohort(k, v, horizon, cfg)
        for k, v in by(decisions, lambda o: category(o.s("confidence"))).items()
    }
    return {
        "scout_score_buckets": {
            k: _summary(score_cohorts[k]) for k in score_edges if k in score_cohorts
        },
        "scout_score_verdict": verdict(score_cohorts, score_edges),
        "scout_score_rank_correlation": spearman([s for s, _ in measured], [r for _, r in measured])
        if len(measured) >= cfg.samples.descriptive_n
        else None,
        "opportunity_confidence": {k: _summary(c) for k, c in conf.items()},
        "opportunity_confidence_verdict": verdict(conf, ("low", "medium", "high")),
    }


# --- interactions -------------------------------------------------------------------------------------

CONDITIONS: dict[str, Callable[[Observation], bool]] = {
    "ACCELERATING": lambda o: o.s("stage") == "ACCELERATING",
    "EARLY": lambda o: o.s("stage") == "EARLY",
    "liquidity>=100k": lambda o: (o.f("liquidity_usd") or 0) >= 100_000,
    "liquidity<25k": lambda o: (
        o.f("liquidity_usd") is not None and (o.f("liquidity_usd") or 0) < 25_000
    ),
    "liquidity rising": lambda o: (o.f("liquidity_change_pct") or 0) > 0,
    "moderate prior move (0-50%)": lambda o: 0 <= (o.f("price_change_h24_pct") or -1) < 50,
    "large prior pump (>=100%)": lambda o: (o.f("price_change_h24_pct") or 0) >= 100,
    "volume acceleration>=2": lambda o: (o.f("volume_acceleration") or 0) >= 2,
    "very high volume acceleration>=4": lambda o: (o.f("volume_acceleration") or 0) >= 4,
    "complete safety": lambda o: o.s("safety_status") == "SAFETY_CHECKS_COMPLETE",
    "top10<30%": lambda o: (
        o.f("holder_top10_pct") is not None and (o.f("holder_top10_pct") or 0) < 30
    ),
    "top10>50%": lambda o: (o.f("holder_top10_pct") or 0) > 50,
    "social rising": lambda o: (
        o.s("social_status") in ("SOCIAL_EMERGING", "SOCIAL_ACCELERATING", "SOCIAL_STRONG")
    ),
    "BUY": lambda o: o.s("action") == "buy",
    "technical uptrend": lambda o: o.s("technical_trend") == "uptrend",
}


def interactions(
    cal: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> tuple[list[dict[str, Any]], int]:
    """Combinations of 2..max_interaction_size conditions with enough support (counted
    before any statistics), at most `max_patterns`. Returns (patterns, tested)."""
    names = sorted(CONDITIONS)
    matches = {n: {o.key for o in cal if CONDITIONS[n](o)} for n in names}
    by_key = {o.key: o for o in cal}
    base = cohort("all", list(cal), horizon, cfg)
    out: list[dict[str, Any]] = []
    tested = 0
    for size in range(2, cfg.max_interaction_size + 1):
        for combo in itertools.combinations(names, size):
            keys = set.intersection(*(matches[n] for n in combo))
            if len(keys) < cfg.samples.descriptive_n:
                continue  # never evaluated: no statistics on unsupported patterns
            if tested >= cfg.max_patterns:
                break
            tested += 1
            c = cohort(" + ".join(combo), [by_key[k] for k in sorted(keys)], horizon, cfg)
            if c.status != "OK":
                continue
            out.append({
                "pattern": list(combo), "cohort": _summary(c),
                "vs_all_return_pp": _diff(c, base), "vs_all_mae_pp": _diff(c, base, "median_mae"),
            })  # fmt: skip
    out.sort(key=lambda x: -(x["vs_all_return_pp"] or 0))
    return out, tested


# --- sensitivity ------------------------------------------------------------------------------------------

FILTERS: dict[str, tuple[str, Callable[[Observation, float], bool]]] = {
    "min_liquidity_usd": ("liquidity_floors_usd", lambda o, x: (o.f("liquidity_usd") or 0) >= x),
    "min_score": ("score_floors", lambda o, x: (o.f("score") or 0) >= x),
    "max_top10_pct": (
        "top10_ceilings_pct",
        lambda o, x: o.f("holder_top10_pct") is None or (o.f("holder_top10_pct") or 0) <= x,
    ),  # fmt: skip
    "max_prior_move_pct": (
        "prior_move_ceilings_pct",
        lambda o, x: (o.f("price_change_h24_pct") or 0) <= x,
    ),  # fmt: skip
}


def threshold_sensitivity(
    cal: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> dict[str, list[dict[str, Any]]]:
    scout = _for(cal, SCOUT_KINDS)
    base = cohort("base", scout, horizon, cfg)
    base_median = base.median_return
    out: dict[str, list[dict[str, Any]]] = {}
    for name, (values_attr, keep) in FILTERS.items():
        rows = []
        for x in getattr(cfg, values_attr):
            kept = [o for o in scout if keep(o, x)]
            dropped = [o for o in scout if not keep(o, x)]
            c = cohort(f"{name}={x:g}", kept, horizon, cfg)
            dropped_returns = [
                v.return_pct
                for o in dropped
                if (v := outcome(o, horizon)) and v.return_pct is not None
            ]
            rows.append({
                "value": x,
                "retained": c.measured, "retention": round(c.measured / base.measured, 4) if base.measured else None,
                "assets": c.assets, "diversity_loss": round(1 - c.assets / base.assets, 4) if base.assets else None,
                "cohort": _summary(c),
                "return_vs_base_pp": _diff(c, base), "mae_vs_base_pp": _diff(c, base, "median_mae"),
                "drawdown_vs_base_pp": _diff(c, base, "median_drawdown"),
                "false_filter_risk": round(sum(1 for r in dropped_returns if base_median is not None and r > base_median)
                                           / len(dropped_returns), 4) if dropped_returns else None,
            })  # fmt: skip
        out[name] = rows
    return out


def rescore(o: Observation, weights: dict[str, float]) -> float | None:
    """The Scout score with other (normalized) family weights, stage adjustment and risk
    penalty unchanged. Offline only."""
    fs = family_scores(o)
    if fs is None:
        return None
    total = sum(weights.values())
    base = sum(100 * fs[f] * weights[f] / total for f in FAMILIES)
    return max(0.0, min(100.0, base + (o.f("stage_adjustment") or 0) - (o.f("risk_penalty") or 0)))


def weight_variants(cfg: CalibrationConfig) -> list[tuple[str, dict[str, float]]]:
    """Production weights and bounded one-family perturbations, renormalized to sum 1."""
    base = production_weights()
    out = [("production", base)]
    for f in FAMILIES:
        for step in cfg.weight_steps:
            w = dict(base)
            w[f] = base[f] * (1 + step)
            total = sum(w.values())
            out.append((f"{f}{step:+.0%}", {k: round(v / total, 6) for k, v in w.items()}))
    return out


def score_quality(
    obs: Sequence[Observation], weights: dict[str, float], horizon: str, cfg: CalibrationConfig
) -> dict[str, Any]:
    pairs = [(s, o) for o in obs if (s := rescore(o, weights)) is not None and outcome(o, horizon)]
    if len(pairs) < cfg.samples.descriptive_n:
        return {"status": "INSUFFICIENT_SAMPLE", "measured": len(pairs)}
    rho = spearman([s for s, _ in pairs], [outcome(o, horizon).return_pct or 0.0 for _, o in pairs])  # type: ignore[union-attr]
    ranked = sorted(pairs, key=lambda x: (-x[0], x[1].key))
    top = [o for _, o in ranked[: max(1, len(ranked) // 5)]]
    c = cohort("top quintile", top, horizon, cfg)
    return {
        "status": "OK",
        "measured": len(pairs),
        "rank_correlation": rho,
        "top_quintile": _summary(c),
    }


def weight_sensitivity(
    cal: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> list[dict[str, Any]]:
    scout = _for(cal, SCOUT_KINDS)
    out = []
    base_rho = None
    for name, weights in weight_variants(cfg):
        q = score_quality(scout, weights, horizon, cfg)
        if name == "production":
            base_rho = q.get("rank_correlation")
        delta = (
            (q["rank_correlation"] - base_rho)
            if q.get("rank_correlation") is not None and base_rho is not None
            else None
        )
        out.append(
            {"variant": name, "weights": weights, **q, "rank_correlation_vs_production": delta}
        )
    return out


def profiles(cal: Sequence[Observation], horizon: str, cfg: CalibrationConfig) -> dict[str, Any]:
    """Calibrated, factual tendencies per stage for future agent logic (never guarantees)."""
    out = {}
    for stage, members in sorted(_groups(cal, "stage").items()):
        c = cohort(stage, members, horizon, cfg)
        maes = sorted(
            v.mae_pct for o in members if (v := outcome(o, horizon)) and v.mae_pct is not None
        )
        completeness = [1 - len(o.missing) / 5 for o in members]
        out[stage] = {
            "sample": c.measured, "assets": c.assets,
            "reliability": "INSUFFICIENT_SAMPLE" if c.status != "OK"
            else "STRONGER" if meets(c, cfg.samples, "strong")
            else "MODERATE" if meets(c, cfg.samples, "candidate") else "WEAK",
            "continuation_tendency": c.positive_share, "median_return": c.median_return,
            "expected_drawdown_range": [maes[len(maes) // 4], maes[len(maes) // 2]] if c.status == "OK" and maes else None,
            "evidence_completeness": round(sum(completeness) / len(completeness), 3) if completeness else None,
        }  # fmt: skip
    return out


def missing_evidence(
    cal: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> dict[str, Any]:
    labels = sorted({m for o in cal for m in o.missing})
    out = {}
    for label in labels:
        with_ = cohort(f"{label}", [o for o in cal if label in o.missing], horizon, cfg)
        without = cohort(f"not {label}", [o for o in cal if label not in o.missing], horizon, cfg)
        out[label] = {"missing": _summary(with_), "present": _summary(without),
                      "return_gap_pp": _diff(with_, without), "mae_gap_pp": _diff(with_, without, "median_mae")}  # fmt: skip
    return out
