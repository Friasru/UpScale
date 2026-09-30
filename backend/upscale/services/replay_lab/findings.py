"""EXPERIMENTAL calibration findings from historical replay.

Computed on CALIBRATION samples only (purged samples excluded), then checked for the same
direction on VALIDATION. HOLDOUT is never read here. A finding is a hypothesis for a human
to review: it is stored in the replay database (``calibration_findings``), never applied to
production weights, thresholds, stages or decision logic, and it is not evidence that any
setup is profitable.
"""

from collections.abc import Callable, Sequence
from typing import Any

from upscale.services.outcomes.config import AnalyticsConfig
from upscale.services.outcomes.metrics import median
from upscale.services.replay_lab.analytics import DEFAULT_ANALYTICS, MIN_ASSETS, groupers, measured
from upscale.services.replay_lab.store import AnalysisRow, ReplayStore

SUFFIX = " (experimental, historical replay only; production logic is unchanged)"
WEAK_CORRELATION = 0.1
MATERIAL_MAE_GAP = 5.0  # percentage points of median MAE
MATERIAL_RETURN_GAP = 2.0  # percentage points of median return


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def _ret(r: AnalysisRow) -> float:
    assert r.outcome is not None and r.outcome.price is not None
    assert r.outcome.price.return_pct is not None
    return r.outcome.price.return_pct


def _mae(r: AnalysisRow) -> float | None:
    return r.outcome.price.mae_pct if r.outcome and r.outcome.price else None


def _enough(rows: Sequence[AnalysisRow], cfg: AnalyticsConfig) -> bool:
    return len(rows) >= cfg.min_sample and len({r.sample.asset_id for r in rows}) >= MIN_ASSETS


def _validate(
    val: Sequence[AnalysisRow],
    cfg: AnalyticsConfig,
    holds: Callable[[Sequence[AnalysisRow]], bool | None],
) -> str:
    if not _enough(val, cfg):
        return "INSUFFICIENT_VALIDATION_SAMPLE"
    result = holds(val)
    if result is None:
        return "INSUFFICIENT_VALIDATION_SAMPLE"
    return "CONSISTENT_ON_VALIDATION" if result else "NOT_REPRODUCED_ON_VALIDATION"


def _cmp(pair: tuple[float, float] | None, pred: Callable[[float, float], bool]) -> bool | None:
    return None if pair is None else pred(*pair)


def generate(
    store: ReplayStore,
    horizon: str = "4h",
    job_ids: Sequence[str] | None = None,
    cfg: AnalyticsConfig = DEFAULT_ANALYTICS,
) -> list[dict[str, Any]]:
    cal = [r for r in store.analysis_rows(horizon, ("CALIBRATION",), job_ids, False) if measured(r)]
    val = [r for r in store.analysis_rows(horizon, ("VALIDATION",), job_ids, False) if measured(r)]
    if any(r.sample.split == "HOLDOUT" for r in (*cal, *val)):
        raise RuntimeError("HOLDOUT data reached calibration findings")
    jobs = sorted({r.job_id for r in cal})
    out: list[dict[str, Any]] = []

    def add(
        kind: str, subject: str, statement: str, validation: str, evidence: dict[str, Any]
    ) -> None:
        out.append(
            {
                "kind": kind, "subject": subject, "horizon": horizon,
                "statement": "EXPERIMENTAL: " + statement + SUFFIX, "validation": validation,
                "evidence": evidence, "split_used": "CALIBRATION", "job_ids": jobs,
            }
        )  # fmt: skip

    # 1. Scout score vs forward return.
    scored = [r for r in cal if r.decision.scout.score is not None]
    if _enough(scored, cfg):
        rho = spearman([r.decision.scout.score or 0.0 for r in scored], [_ret(r) for r in scored])
        if rho is not None and rho < WEAK_CORRELATION:

            def same(rows: Sequence[AnalysisRow]) -> bool | None:
                s = [r for r in rows if r.decision.scout.score is not None]
                v = spearman([r.decision.scout.score or 0.0 for r in s], [_ret(r) for r in s])
                return None if v is None else (v < WEAK_CORRELATION)

            add(
                "signal_weight", "scout_score",
                f"ScoutMomentumScore shows {'an inverse' if rho < -WEAK_CORRELATION else 'a weak'} "
                f"rank relationship with {horizon} returns (Spearman {rho:+.2f}); some score "
                "families may be over-weighted",
                _validate(val, cfg, same), {"spearman": rho, "n": len(scored)},
            )  # fmt: skip

    # 2. Stage ordering: ACCELERATING should not trail STEADY.
    def stage_gap(rows: Sequence[AnalysisRow]) -> tuple[float, float] | None:
        acc = [_ret(r) for r in rows if r.decision.scout.stage == "ACCELERATING"]
        steady = [_ret(r) for r in rows if r.decision.scout.stage == "STEADY"]
        if len(acc) < cfg.min_sample or len(steady) < cfg.min_sample:
            return None
        a, s = median(acc), median(steady)
        return (a, s) if a is not None and s is not None else None

    gap = stage_gap(cal)
    if gap is not None and gap[0] < gap[1] - MATERIAL_RETURN_GAP:
        add(
            "stage_transition", "ACCELERATING",
            f"ACCELERATING shows a lower median {horizon} return ({gap[0]:+.2f}%) than STEADY "
            f"({gap[1]:+.2f}%): the ACCELERATING transition may be too aggressive",
            _validate(val, cfg, lambda rows: _cmp(stage_gap(rows), lambda a, b: a < b)),
            {"accelerating_median": gap[0], "steady_median": gap[1]},
        )  # fmt: skip

    # 3. Risk flags with deeper adverse excursions.
    flags = sorted({f["code"] for r in cal for f in r.decision.scout.risk_flags})
    for code in flags:

        def mae_gap(rows: Sequence[AnalysisRow], code: str = code) -> tuple[float, float] | None:
            with_ = [
                m
                for r in rows
                if any(f["code"] == code for f in r.decision.scout.risk_flags)
                and (m := _mae(r)) is not None
            ]
            without = [
                m
                for r in rows
                if not any(f["code"] == code for f in r.decision.scout.risk_flags)
                and (m := _mae(r)) is not None
            ]
            if len(with_) < cfg.min_sample or len(without) < cfg.min_sample:
                return None
            a, b = median(with_), median(without)
            return (a, b) if a is not None and b is not None else None

        g2 = mae_gap(cal)
        if g2 is not None and g2[0] < g2[1] - MATERIAL_MAE_GAP:
            add(
                "high_mae_family", code,
                f"samples flagged {code} show a deeper median {horizon} MAE ({g2[0]:+.2f}%) than "
                f"unflagged ones ({g2[1]:+.2f}%): its penalty may be too small",
                _validate(
                    val, cfg,
                    lambda rows, c=code: _cmp(mae_gap(rows, c), lambda a, b: a < b - MATERIAL_MAE_GAP),  # type: ignore[misc]
                ),
                {"flagged_median_mae": g2[0], "unflagged_median_mae": g2[1]},
            )  # fmt: skip

    # 4. Prior move: large 24h movers vs the rest.
    def prior_gap(rows: Sequence[AnalysisRow]) -> tuple[float, float] | None:
        big = [
            _ret(r)
            for r in rows
            if (m := r.decision.features.get("price_change_h24_pct")) is not None and m >= 50
        ]
        rest = [
            _ret(r)
            for r in rows
            if (m := r.decision.features.get("price_change_h24_pct")) is not None and m < 20
        ]
        if len(big) < cfg.min_sample or len(rest) < cfg.min_sample:
            return None
        a, b = median(big), median(rest)
        return (a, b) if a is not None and b is not None else None

    g3 = prior_gap(cal)
    if g3 is not None and abs(g3[0] - g3[1]) >= MATERIAL_RETURN_GAP:
        weaker = g3[0] < g3[1]
        add(
            "prior_move_penalty", "price_change_h24_pct>=50",
            (f"after a 24h move of 50%+, median {horizon} return is {g3[0]:+.2f}% vs {g3[1]:+.2f}% "
             "for moves under 20%: " + ("weaker continuation; the prior-move penalty may need to be stronger"
                                        if weaker else "stronger continuation; the prior-move penalty may be too strict")),
            _validate(val, cfg, lambda rows: _cmp(prior_gap(rows), lambda a, b: (a < b) == weaker)),
            {"large_move_median": g3[0], "small_move_median": g3[1]},
        )  # fmt: skip

    # 5. Thin liquidity: deeper adverse excursions in the lowest band.
    liquidity = groupers(cfg)["liquidity"]
    lowest = f"<${cfg.liquidity_bands_usd[0] / 1e3:g}k"

    def liq_gap(rows: Sequence[AnalysisRow]) -> tuple[float, float] | None:
        thin = [m for r in rows if liquidity(r) == [lowest] and (m := _mae(r)) is not None]
        rest = [
            m
            for r in rows
            if liquidity(r) not in ([lowest], ["unavailable"]) and (m := _mae(r)) is not None
        ]
        if len(thin) < cfg.min_sample or len(rest) < cfg.min_sample:
            return None
        a, b = median(thin), median(rest)
        return (a, b) if a is not None and b is not None else None

    g4 = liq_gap(cal)
    if g4 is not None and g4[0] < g4[1] - MATERIAL_MAE_GAP:
        add(
            "liquidity_threshold", lowest,
            f"pools under {lowest[1:]} liquidity show a deeper median {horizon} MAE ({g4[0]:+.2f}%) "
            f"than deeper pools ({g4[1]:+.2f}%): the liquidity threshold may be inadequate",
            _validate(val, cfg, lambda rows: _cmp(liq_gap(rows), lambda a, b: a < b - MATERIAL_MAE_GAP)),
            {"thin_median_mae": g4[0], "deeper_median_mae": g4[1]},
        )  # fmt: skip
    return out
