"""Experimental candidate configurations: evaluated offline, never applied to production.

A candidate is a small, explicit change to production behavior, evaluated as a selection
and ranking policy over observations:

* ``filter.min_liquidity_usd`` / ``filter.min_score`` / ``filter.max_top10_pct`` /
  ``filter.max_prior_move_pct``: a threshold (from the bounded sensitivity grid);
* ``weights``: one bounded family-weight perturbation (renormalized to sum 1).

v1 only generates single-change candidates (lowest complexity): the simplest change
that shows a material improvement is preferred, and complex rule sets are not produced.

A candidate "improves" on the production baseline when at least one of median return,
median MAE or score rank correlation improves by the material margin and none of them is
materially worse. Outcomes are market measurements: never realized profit.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from upscale.services.calibration.analysis import (
    DECISION_KINDS,
    FILTERS,
    SCOUT_KINDS,
    _for,
    _summary,
    production_weights,
    rescore,
    weight_variants,
)
from upscale.services.calibration.config import CalibrationConfig
from upscale.services.calibration.dataset import Observation
from upscale.services.calibration.stats import cohort, meets, outcome, spearman

PARAMETERS = {f"filter.{name}" for name in FILTERS} | {"weights"}


def candidate_id(parent: dict[str, str], changes: list[dict[str, Any]]) -> str:
    body = json.dumps([parent, changes], sort_keys=True)
    return "cand-" + hashlib.sha256(body.encode()).hexdigest()[:10]


def complexity(changes: list[dict[str, Any]]) -> dict[str, Any]:
    params = sum(len(c["to"]) if isinstance(c["to"], dict) else 1 for c in changes)
    level = "low" if len(changes) <= 1 else "medium" if len(changes) <= 3 else "high"
    return {"changes": len(changes), "conditions": sum(1 for c in changes if c["parameter"].startswith("filter.")),
            "parameters_altered": params, "level": level}  # fmt: skip


def _threshold(
    keep: Callable[[Observation, float], bool], x: float
) -> Callable[[Observation], bool]:
    return lambda o: keep(o, x)


def policy(changes: list[dict[str, Any]]) -> tuple[Callable[[Observation], bool], dict[str, float]]:
    filters: list[Callable[[Observation], bool]] = []
    weights = production_weights()
    for ch in changes:
        p = ch["parameter"]
        if p not in PARAMETERS:
            raise ValueError(f"unknown candidate parameter {p}")
        if p == "weights":
            weights = {k: float(v) for k, v in ch["to"].items()}
        else:
            filters.append(_threshold(FILTERS[p.removeprefix("filter.")][1], float(ch["to"])))
    return (lambda o: all(f(o) for f in filters)), weights


def evaluate(
    changes: list[dict[str, Any]],
    observations: Sequence[Observation],
    horizon: str,
    cfg: CalibrationConfig,
) -> dict[str, Any]:
    keep, weights = policy(changes)
    universe = _for(observations, SCOUT_KINDS)
    base = cohort("universe", universe, horizon, cfg)
    selected = [o for o in universe if keep(o)]
    c = cohort("selected", selected, horizon, cfg)
    scored = [
        (s, o) for o in selected if (s := rescore(o, weights)) is not None and outcome(o, horizon)
    ]
    rho = (
        spearman([s for s, _ in scored], [outcome(o, horizon).return_pct or 0.0 for _, o in scored])  # type: ignore[union-attr]
        if len(scored) >= cfg.samples.descriptive_n else None
    )  # fmt: skip
    ranked = [o for _, o in sorted(scored, key=lambda x: (-x[0], x[1].key))]
    top = cohort("top quintile", ranked[: max(1, len(ranked) // 5)], horizon, cfg)
    return {
        "selected": _summary(c),
        "retention": round(c.measured / base.measured, 4) if base.measured else None,
        "rank_correlation": rho,
        "top_quintile": _summary(top),
        "origins": c.origins,
        "meets_candidate_sample": meets(c, cfg.samples, "candidate"),
        "meets_descriptive_sample": meets(c, cfg.samples, "descriptive"),
    }


def judge(cand: dict[str, Any], base: dict[str, Any], cfg: CalibrationConfig) -> dict[str, Any]:
    a, b = cand["selected"], base["selected"]

    def d(key: str) -> float | None:
        return None if a.get(key) is None or b.get(key) is None else round(a[key] - b[key], 4)

    ret, mae, dd = d("median_return"), d("median_mae"), d("median_drawdown")
    rho = (
        round(cand["rank_correlation"] - base["rank_correlation"], 4)
        if cand.get("rank_correlation") is not None and base.get("rank_correlation") is not None else None
    )  # fmt: skip
    better = (
        (ret is not None and ret >= cfg.material_return_pp)
        or (mae is not None and mae >= cfg.material_mae_pp)  # MAE <= 0: higher is shallower
        or (rho is not None and rho >= cfg.material_rank_correlation)
    )
    worse = (
        (ret is not None and ret <= -cfg.material_return_pp)
        or (mae is not None and mae <= -cfg.material_mae_pp)
        or (rho is not None and rho <= -cfg.material_rank_correlation)
    )
    return {"return_pp": ret, "mae_pp": mae, "drawdown_pp": dd, "rank_correlation": rho,
            "improves": better and not worse, "materially_worse": worse}  # fmt: skip


def _hash_half(o: Observation) -> bool:
    return int(hashlib.sha256(o.key.encode()).hexdigest(), 16) % 2 == 0


BASELINES: dict[str, tuple[frozenset[str], Callable[[Observation], bool]]] = {
    "production_scout": (SCOUT_KINDS, lambda o: True),
    "production_opportunity_buy": (DECISION_KINDS, lambda o: o.s("action") == "buy"),
    "simple_momentum": (
        SCOUT_KINDS,
        lambda o: (o.f("price_change_h1_pct") or 0) > 0 and (o.f("price_change_h24_pct") or 0) > 0,
    ),  # fmt: skip
    "liquidity_and_volume": (
        SCOUT_KINDS,
        lambda o: (
            (o.f("liquidity_usd") or 0) >= 50_000 and (o.f("volume_acceleration") or 0) >= 1.5
        ),
    ),  # fmt: skip
    "buy_and_hold_horizon": (SCOUT_KINDS | DECISION_KINDS, lambda o: True),
    "random_eligible_half": (SCOUT_KINDS, _hash_half),
}


def baselines(
    observations: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> dict[str, Any]:
    out = {}
    for name, (kinds, keep) in BASELINES.items():
        picked = [o for o in _for(observations, kinds) if keep(o)]
        out[name] = _summary(cohort(name, picked, horizon, cfg))
    return out


def generate(
    cal: Sequence[Observation], horizon: str, cfg: CalibrationConfig
) -> tuple[list[dict[str, Any]], int]:
    """Single-change candidates that improve on production on CALIBRATION, with enough
    support and retention: the best per threshold dimension and the best weight variant.
    Returns (candidates, variants tested) for the multiple-comparison record."""
    base = evaluate([], cal, horizon, cfg)
    out: list[dict[str, Any]] = []
    tested = 0
    for name, (values_attr, _) in FILTERS.items():
        best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        for x in getattr(cfg, values_attr):
            tested += 1
            changes = [{"parameter": f"filter.{name}", "from": None, "to": x}]
            m = evaluate(changes, cal, horizon, cfg)
            j = judge(m, base, cfg)
            if not (
                j["improves"]
                and m["meets_candidate_sample"]
                and (m["retention"] or 0) >= cfg.min_retention
            ):
                continue
            gain = (j["return_pp"] or 0) + (j["mae_pp"] or 0)
            if best is None or gain > best[0]:
                best = (gain, {"changes": changes, "metrics": m, "judgement": j}, m)
        if best is not None:
            out.append(
                best[1]
                | {"reason": f"threshold sensitivity: {name} = {best[1]['changes'][0]['to']:g}"}
            )
    prod = production_weights()
    best_w: tuple[float, dict[str, Any]] | None = None
    for variant, weights in weight_variants(cfg)[1:]:
        tested += 1
        changes = [{"parameter": "weights", "from": prod, "to": weights, "variant": variant}]
        m = evaluate(changes, cal, horizon, cfg)
        j = judge(m, base, cfg)
        if (
            j["improves"]
            and m["meets_candidate_sample"]
            and (j["rank_correlation"] or 0) > (best_w[0] if best_w else 0)
        ):
            best_w = (j["rank_correlation"] or 0, {"changes": changes, "metrics": m, "judgement": j,
                                                    "reason": f"bounded weight perturbation {variant}"})  # fmt: skip
    if best_w is not None:
        out.append(best_w[1])
    return out, tested
