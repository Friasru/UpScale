"""The Audit / Learning V1 report (JSON and text) and the per-decision feature audit.

Deterministic: the same databases and arguments give byte-identical output (sorted
keys, rounded figures, no wall-clock time; ``data_through`` is the latest decision
analyzed). Read-only: built from `dataset.load`, which only opens databases ``mode=ro``.
"""

import json
import statistics
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from upscale.services.audit.analytics import (
    Labeled,
    effective_sample_size,
    grouped,
    horizon_item,
    label,
    rank_hypotheses,
    scout_hypotheses,
    shadow_hypotheses,
    summarize,
    trade_item,
)
from upscale.services.audit.config import HORIZONS, AuditConfig
from upscale.services.audit.dataset import AuditObservation, Dataset
from upscale.services.audit.features import FeatureGroup

REPORT_VERSION = 1
GROUPS = ("identity", "scout", "market", "technical", "analyze_technical", "social", "safety")
DISCLAIMER = (
    "Descriptive, read-only audit of what UpScale knew at each decision and what happened "
    "afterwards. Associations only: nothing here is causal, a forecast or a production "
    "recommendation. Shadow results are simulated, not real profit. Hypotheses are for "
    "Calibration to test out of sample; Audit never changes production."
)
EXTREME_RETURN_PCT = 1000.0


def _iso(t: datetime | None) -> str | None:
    return t.isoformat() if t is not None else None


def _pct(part: int, whole: int) -> float | None:
    return round(part / whole * 100, 2) if whole else None


def _percentile(v: Sequence[float], q: float) -> float | None:
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


# --- sections ------------------------------------------------------------------------------------


def _availability(obs: Sequence[AuditObservation]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = {g: Counter() for g in GROUPS}
    for o in obs:
        for g in GROUPS:
            grp = o.snapshot.groups.get(g)
            out[g][grp.availability if grp else "NOT_AVAILABLE"] += 1
    return {g: dict(sorted(c.items())) for g, c in out.items()}


def _horizon_coverage(obs: Sequence[AuditObservation]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for h in HORIZONS:
        c: Counter[str] = Counter()
        for o in obs:
            hl = o.horizons.get(h)
            if hl is None:
                c["no_label"] += 1
            else:
                c[hl.status] += 1
                c["measured"] += int(hl.return_pct is not None)
        out[h] = dict(sorted(c.items()))
    return out


def _scout_section(
    obs: Sequence[AuditObservation], cfg: AuditConfig, labeled: Labeled, all_horizons: bool
) -> dict[str, Any]:
    assets = [o.asset_id for o in obs]
    return {
        "observations": len(obs),
        "unique_assets": len(set(assets)),
        "effective_sample_size": round(effective_sample_size(assets), 2),
        "decision_basis": dict(sorted(Counter(o.decision_basis for o in obs).items())),
        "feature_availability": _availability(obs),
        "horizon_coverage": _horizon_coverage(obs),
        "by_horizon": {h: summarize([horizon_item(o, h) for o in obs], cfg) for h in HORIZONS},
        "groupings": {
            h: grouped([horizon_item(o, h) for o in obs], labeled, cfg)
            for h in (HORIZONS if all_horizons else (cfg.primary_horizon,))
        },
    }


def _exit_flags(obs: Sequence[AuditObservation]) -> dict[str, int]:
    reasons = Counter(o.trade.exit_reason for o in obs if o.trade and o.trade.exit_reason)
    return dict(sorted(reasons.items()))


def _execution(obs: Sequence[AuditObservation], model: str) -> dict[str, Any]:
    fills = [o for o in obs if o.execution is not None]
    xs = [o.execution.values for o in fills if o.execution is not None]
    trades = [o.trade for o in obs if o.trade is not None and o.trade.resolved]

    def vals(name: str) -> list[float]:
        return [float(x) for v in xs if isinstance(x := v.get(name), int | float)]

    delays = vals("execution_delay_seconds")
    drift = vals("latency_drift_usd")
    stored = vals("stored_latency_cost_usd")
    gross = [t.gross_pnl_usd for t in trades if t.gross_pnl_usd is not None]
    net = [t.pnl_usd for t in trades if t.pnl_usd is not None]
    friction = [t.friction_usd for t in trades if t.friction_usd is not None]
    return {
        "execution_model": model,
        "entries_filled": len(fills),
        "median_entry_delay_seconds": _r(statistics.median(delays)) if delays else None,
        "p90_entry_delay_seconds": _r(_percentile(delays, 90)),
        "max_entry_delay_seconds": _r(max(delays)) if delays else None,
        "mean_entry_slippage_bps": _r(statistics.fmean(vals("slippage_bps"))) if xs else None,
        "mean_entry_price_impact_bps": _r(statistics.fmean(vals("price_impact_bps")))
        if xs
        else None,
        "entry_fees_usd": _r(sum(vals("fee_usd"))),
        "entry_latency_drift_usd": _r(sum(drift)),
        "entry_stored_latency_cost_usd": _r(sum(stored)),
        "latency_drift_note": "latency_drift_usd: recomputed from each BUY fill's reference "
        "price, observed price and quantity (the corrected formula, shadow.book."
        "latency_drift_usd); stored: the row's own column (pre-fix rows may differ)",
        "closed_resolved_trades": len(trades),
        "gross_pnl_usd": _r(sum(gross)) if gross else None,
        "net_pnl_usd": _r(sum(net)) if net else None,
        "trade_friction_usd": _r(sum(friction)) if friction else None,
    }


def _shadow_section(
    obs: Sequence[AuditObservation], ds: Dataset, cfg: AuditConfig
) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for run in sorted({o.run_id for o in obs if o.run_id}):
        mine = [o for o in obs if o.run_id == run]
        meta = ds.runs.get(run, {})
        strategies: dict[str, Any] = {}
        for s in sorted({o.strategy for o in mine if o.strategy}):
            rows = [o for o in mine if o.strategy == s]
            status = Counter(o.trade.status for o in rows if o.trade)
            strategies[s] = {
                "entries": len(rows),
                "trade_status": dict(sorted(status.items())),
                "unique_assets_entered": len({o.asset_id for o in rows}),
                "trade_outcome": summarize([trade_item(o) for o in rows], cfg),
                "exit_reasons": _exit_flags(rows),
                "median_holding_minutes": _r(statistics.median(held)) if (held := [
                    o.trade.holding_minutes for o in rows
                    if o.trade and o.trade.holding_minutes is not None]) else None,
                "execution": _execution(rows, meta.get("execution_model", "")),
                "groupings": grouped([trade_item(o) for o in rows], label(rows, cfg, "shadow"), cfg),
            }  # fmt: skip
        runs[run] = {
            "execution_model": meta.get("execution_model"),
            "clean_data": meta.get("clean_data"),
            "since": meta.get("since"),
            "entries": len(mine),
            "feature_availability": _availability(mine),
            "strategies": strategies,
            "note": "each run is analyzed on its own: a REALISTIC_V1 run re-executes the same "
            "Scout decisions as its IDEALIZED twin and is never pooled with it",
        }
    return {"runs": runs}


def _quality(ds: Dataset, obs: Sequence[AuditObservation], cfg: AuditConfig) -> dict[str, Any]:
    lags = ds.decision_lag_minutes
    findings: list[str] = []
    inconsistent = extreme = sign = 0
    for o in obs:
        for h in o.horizons.values():
            r, up, down = h.return_pct, h.mfe_pct, h.mae_pct
            if r is not None and abs(r) >= EXTREME_RETURN_PCT:
                extreme += 1
            if (up is not None and up < -1e-9) or (down is not None and down > 1e-9):
                sign += 1
            if (
                r is not None
                and up is not None
                and down is not None
                and not (down - 1e-6 <= r <= up + 1e-6)
            ):
                inconsistent += 1
    drift_mismatch = 0
    for o in obs:
        x = o.execution.values if o.execution is not None else {}
        stored, drift = x.get("stored_latency_cost_usd"), x.get("latency_drift_usd")
        if isinstance(stored, int | float) and isinstance(drift, int | float):
            drift_mismatch += abs(stored - drift) > 0.01
    late = sum(1 for x in lags if x > cfg.decision_lag_note_minutes)
    q = ds.quality
    unlinked = q.get("scout_unlinked_evidence", 0)
    if unlinked:
        findings.append(
            f"{unlinked} Scout anchors have no linked Scout evidence record (pruned by retention "
            "or never archived): their Technical context is NOT_LINKED and their decision time "
            "is the ranking run's start"
        )
    if q.get("scout_evidence_causal_violation"):
        findings.append(
            f"{q['scout_evidence_causal_violation']} linked Scout evidence records are marked "
            "causal_valid=false by the archive (evidence observed after the ranking): the later "
            "groups were excluded"
        )
    excluded = {
        k.removeprefix("excluded_future_"): v
        for k, v in q.items()
        if k.startswith("excluded_future_")
    }
    if excluded:
        findings.append(
            f"feature groups excluded as observed after their decision: {dict(sorted(excluded.items()))}"
        )
    if late:
        findings.append(
            f"{late} Scout decisions were final more than {cfg.decision_lag_note_minutes:g} min "
            "after their market observation (the outcome window starts at the market observation)"
        )
    if inconsistent:
        findings.append(
            f"{inconsistent} horizon labels have a return outside their own [MAE, MFE] range"
        )
    if sign:
        findings.append(f"{sign} horizon labels have a negative MFE or a positive MAE")
    if extreme:
        findings.append(
            f"{extreme} horizon labels move by {EXTREME_RETURN_PCT:g}% or more (kept: size alone excludes nothing)"
        )
    if drift_mismatch:
        findings.append(
            f"{drift_mismatch} REALISTIC_V1 entry fills store a latency cost that differs from "
            "the corrected latency drift (pre-fix rows; Audit reports the corrected value)"
        )
    not_filled = sum(1 for o in obs if o.trade is not None and o.trade.status == "NOT_FILLED")
    if not_filled:
        findings.append(
            f"{not_filled} Shadow ENTER decisions have no position (entry pending or cancelled)"
        )
    integrity = {k.removeprefix("horizon_integrity_excluded_"): v for k, v in q.items()
                 if k.startswith("horizon_integrity_excluded_")}  # fmt: skip
    if integrity:
        findings.append(
            f"horizon labels excluded by an integrity audit (invalid): {dict(sorted(integrity.items()))}"
        )
    return {
        "counters": dict(sorted(q.items())),
        "decision_lag_minutes": {
            "median": _r(statistics.median(lags)) if lags else None,
            "p90": _r(_percentile(lags, 90)),
            "max": _r(max(lags)) if lags else None,
            "over_note_threshold": late,
        },  # fmt: skip
        "horizon_labels_outside_mae_mfe": inconsistent,
        "horizon_labels_bad_excursion_sign": sign,
        "horizon_labels_extreme": extreme,
        "latency_drift_mismatches": drift_mismatch,
        "findings": findings,
    }


def build(
    ds: Dataset, cfg: AuditConfig, filters: dict[str, Any], all_horizons: bool = False
) -> dict[str, Any]:
    """The report. Groupings: the primary horizon only, or every horizon (`all_horizons`)."""
    scout = [o for o in ds.observations if o.population == "scout"]
    shadow = [o for o in ds.observations if o.population == "shadow"]
    labeled = label(scout, cfg, "scout")
    s_found, s_counts = scout_hypotheses(scout, cfg, labeled)
    h_found, h_counts = shadow_hypotheses(shadow, cfg)
    found = s_found + h_found
    surfaced = rank_hypotheses(list(found), cfg)
    latest = max((o.decision_at for o in ds.observations), default=None)
    earliest = min((o.decision_at for o in ds.observations), default=None)
    return {
        "report": "audit_learning_v1",
        "report_version": REPORT_VERSION,
        "read_only": True,
        "disclaimer": DISCLAIMER,
        "filters": filters,
        "databases": ds.databases,
        "data_from": _iso(earliest),
        "data_through": _iso(latest),
        "labels": cfg.labels.model_dump(),
        "minimums": cfg.minimums.model_dump(),
        "primary_horizon": cfg.primary_horizon,
        "notes": ds.notes,
        "scout": _scout_section(scout, cfg, labeled, all_horizons) if scout else None,
        "shadow": _shadow_section(shadow, ds, cfg) if shadow else None,
        "data_quality": _quality(ds, ds.observations, cfg),
        "hypotheses": surfaced,
        "hypothesis_stats": {
            "comparisons_tested": s_counts.get("tested", 0) + h_counts.get("tested", 0),
            "groups_meeting_rules": len(found),
            "listed": len(surfaced),
            "notable_but_insufficient_sample": s_counts.get("notable_but_insufficient", 0)
            + h_counts.get("notable_but_insufficient", 0),
            "rules": cfg.hypotheses.model_dump(),
        },
    }


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2, default=str)


# --- text ------------------------------------------------------------------------------------------


def _f(x: Any, suffix: str = "", nd: int = 1) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and abs(x) >= 1e5:
        return f"{x:.1e}{suffix}"  # an outlier-driven mean stays one column wide
    if isinstance(x, float):
        return f"{x:.{nd}f}{suffix}"
    return f"{x}{suffix}"


def _rate(x: Any) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


HEAD = (f"{'bucket':<34} {'n':>6} {'assets':>6} {'ESS':>6} {'win':>6} {'aw-win':>6} {'med':>7} "
        f"{'mean':>7} {'trim':>7} {'MFE':>6} {'MAE':>7} {'cat':>6} {'large':>6} {'PF':>5}  status")  # fmt: skip


def _row(name: str, s: dict[str, Any]) -> str:
    flag = "" if s["status"] == "DESCRIPTIVE" else "INSUFFICIENT"
    return (f"{name[:34]:<34} {s['measured']:>6} {s['unique_assets']:>6} "
            f"{_f(s['effective_sample_size']):>6} {_rate(s['win_rate']):>6} "
            f"{_rate(s['asset_weighted_win_rate']):>6} {_f(s['median_return_pct'], '%'):>7} "
            f"{_f(s['mean_return_pct'], '%'):>7} {_f(s['trimmed_mean_return_pct'], '%'):>7} "
            f"{_f(s['median_mfe_pct'], '%'):>6} {_f(s['median_mae_pct'], '%'):>7} "
            f"{_rate(s['catastrophic_rate']):>6} {_rate(s['large_winner_rate']):>6} "
            f"{_f(s['profit_factor'], nd=2):>5}  {flag}").rstrip()  # fmt: skip


def text(r: dict[str, Any]) -> str:
    lab, f = r["labels"], r["filters"]
    out = [
        "=== UPSCALE AUDIT / LEARNING V1 (read-only, descriptive) ===",
        r["disclaimer"],
        "",
        f"databases: scout={r['databases']['scout'] or 'MISSING'}  "
        f"evidence={r['databases']['evidence'] or 'MISSING'}  shadow={r['databases']['shadow'] or 'MISSING'}",
        f"filters: since={f.get('since') or '-'} until={f.get('until') or '-'} "
        f"population={f.get('population') or 'all'} strategy={f.get('strategy') or 'all'} "
        f"run={f.get('run') or 'all'}",
        f"decisions analyzed: {r['data_from'] or '-'} .. {r['data_through'] or '-'}",
        f"labels: large winner >= +{lab['large_winner_pct']:g}%, catastrophic <= "
        f"{lab['catastrophic_pct']:g}%, high MFE >= +{lab['high_mfe_pct']:g}%, high MAE <= "
        f"{lab['high_mae_pct']:g}% (labels are future outcomes, never features)",
        f"minimums: {r['minimums']['measured']} measured, {r['minimums']['unique_assets']} unique "
        f"assets, effective sample {r['minimums']['effective_sample']:g} (below: INSUFFICIENT, never "
        "a hypothesis)",
        "columns: n = measured observations; ESS = effective sample size (each asset's "
        "observations fully correlated); aw-win = asset-weighted win rate; med / mean / trim = "
        "median / mean / trimmed-mean return; MFE / MAE = medians; cat / large = catastrophic / "
        "large-winner rates; PF = profit factor",
    ]
    for n in r["notes"]:
        out.append(f"note: {n}")
    s = r["scout"]
    if s is not None:
        out += ["", "=== SCOUT ANCHORS: COVERAGE ===",
                f"observations {s['observations']}, unique assets {s['unique_assets']}, effective "
                f"sample {s['effective_sample_size']:g}",
                "decision time basis: " + ", ".join(f"{k} {v}" for k, v in s["decision_basis"].items()),
                "feature availability (point in time):"]  # fmt: skip
        for g, c in s["feature_availability"].items():
            out.append(f"  {g:<18} " + ", ".join(f"{k} {v}" for k, v in c.items()))
        out.append("horizon labels:")
        for h, c in s["horizon_coverage"].items():
            out.append(f"  {h:<5} " + ", ".join(f"{k} {v}" for k, v in c.items()))
        out += ["", "=== SCOUT ANCHORS: OUTCOMES BY HORIZON ===", HEAD]
        for h, summary in s["by_horizon"].items():
            out.append(_row(h, summary))
        ph = r["primary_horizon"]
        out += ["", f"=== SCOUT ANCHORS: GROUPINGS @ {ph} (descriptive, not causal) ==="]
        for key, d in s["groupings"][ph].items():
            out += ["", f"[{key}] {d['description']}", HEAD]
            out += [_row(name, b) for name, b in d["buckets"].items()]
    sh = r["shadow"]
    if sh is not None:
        out += ["", "=== SHADOW ENTRIES: STRATEGY COMPARISON (simulated, not real profit) ===",
                "side by side in a fixed order, NOT ranked; trade outcome = resolved closed trades"]  # fmt: skip
        for run, d in sh["runs"].items():
            out += ["", f"run {run} [{d['execution_model']}] since {d['since']}, {d['entries']} ENTER decisions",
                    HEAD]  # fmt: skip
            for sid, st in d["strategies"].items():
                out.append(_row(sid, st["trade_outcome"]))
            for sid, st in d["strategies"].items():
                ts = ", ".join(f"{k} {v}" for k, v in st["trade_status"].items())
                ex = ", ".join(f"{k} {v}" for k, v in st["exit_reasons"].items()) or "-"
                out.append(f"  {sid}: entries {st['entries']} ({ts}); exits: {ex}")
                x = st["execution"]
                if d["execution_model"] == "REALISTIC_V1":
                    out.append(
                        f"    execution: filled {x['entries_filled']}, entry delay median "
                        f"{_f(x['median_entry_delay_seconds'], 's')} p90 {_f(x['p90_entry_delay_seconds'], 's')}"
                        f", slippage {_f(x['mean_entry_slippage_bps'], ' bps')}, entry fees "
                        f"${_f(x['entry_fees_usd'], nd=2)}, latency drift ${_f(x['entry_latency_drift_usd'], nd=2)}"
                        f", gross ${_f(x['gross_pnl_usd'], nd=2)} / net ${_f(x['net_pnl_usd'], nd=2)}"
                        f" (friction ${_f(x['trade_friction_usd'], nd=2)})"
                    )  # fmt: skip
            if d["execution_model"] == "REALISTIC_V1":
                for sid, st in d["strategies"].items():
                    g = st["groupings"]["execution_delay_bucket"]
                    out += [f"  [{sid}] entry execution delay vs trade outcome", "  " + HEAD]
                    out += ["  " + _row(name, b) for name, b in g["buckets"].items()]
    q = r["data_quality"]
    out += ["", "=== DATA QUALITY ==="]
    lag = q["decision_lag_minutes"]
    out.append(f"Scout decision lag after market observation (min): median {_f(lag['median'])}, "
               f"p90 {_f(lag['p90'])}, max {_f(lag['max'])}")  # fmt: skip
    out += [f"- {x}" for x in q["findings"]] or ["- nothing unusual found"]
    hs = r["hypothesis_stats"]
    out += ["", "=== HYPOTHESES FOR CALIBRATION ===",
            f"{hs['comparisons_tested']} comparisons tested; {hs['groups_meeting_rules']} groups met the rules "
            f"(both sides above the minimums, |difference| >= {hs['rules']['min_rate_effect']:g} pp "
            f"for rates / {hs['rules']['min_return_effect']:g} pp for means, and >= "
            f"{hs['rules']['z']:g} asset-clustered standard errors); {hs['listed']} listed; "
            f"{hs['notable_but_insufficient_sample']} large-looking differences NOT surfaced "
            "(insufficient sample)."]  # fmt: skip
    if not r["hypotheses"]:
        out.append("No difference met the rules: nothing to hand to Calibration yet.")
    for i, h in enumerate(r["hypotheses"], 1):
        out += ["", f"H{i}. {h['observation']}"]
        for c in h["evidence"]:
            unit = "pp" if c["kind"] == "rate" else "% pts"
            out.append(
                f"    {c['label']}: {c['group']:.1f} vs {c['other']:.1f} ({c['difference']:+.1f} "
                f"{unit}, SE {_f(c['standard_error'])}, z {_f(c['z'])}); n {c['group_n']} vs "
                f"{c['other_n']}, assets {c['group_assets']} vs {c['other_assets']}, ESS "
                f"{c['group_ess']:g} vs {c['other_ess']:g}"
            )
        out.append(f"    {h['hypothesis']}")
    out += ["", "These are associations in past data, not causes. Many comparisons were made, so "
            "some differences will be chance: each needs an out-of-sample Calibration test before "
            "anything changes."]  # fmt: skip
    return "\n".join(out) + "\n"


# --- feature audit ---------------------------------------------------------------------------------


def _group(g: FeatureGroup, decision_at: datetime) -> dict[str, Any]:
    return {
        "availability": g.availability,
        "observed_at": _iso(g.observed_at),
        "observed_before_decision": None if g.observed_at is None else g.observed_at <= decision_at,
        "seconds_before_decision": None
        if g.observed_at is None
        else round((decision_at - g.observed_at).total_seconds(), 3),
        "source": g.source,
        "values": dict(sorted(g.values.items())),
    }


def feature_audit(o: AuditObservation) -> dict[str, Any]:
    d = o.decision_at
    out: dict[str, Any] = {
        "key": o.key,
        "population": o.population,
        "asset_id": o.asset_id,
        "decision_at": _iso(d),
        "decision_time_basis": o.decision_basis,
        "sources": o.sources,
        "decision_time_features": {n: _group(g, d) for n, g in sorted(o.snapshot.groups.items())},
        "excluded_as_future": list(o.snapshot.excluded_future),
        "future_labels": {
            "note": "outcomes AFTER the decision: labels only, never features",
            "horizons": {h: vars(v) for h, v in sorted(o.horizons.items())},
        },
    }
    if o.population == "shadow":
        out |= {"run_id": o.run_id, "strategy": o.strategy, "execution_model": o.execution_model}
        out["post_decision_execution"] = (
            _group(o.execution, o.execution.observed_at or d) | {
                "note": "entry fill facts: known only after the decision (at the fill), "
                "compared only with the trade outcome that starts at the fill"}
            if o.execution is not None else None
        )  # fmt: skip
        t = o.trade
        out["future_labels"]["trade"] = (
            {k: (_iso(v) if isinstance(v, datetime) else v) for k, v in vars(t).items()}
            if t is not None
            else None
        )
    return out


def feature_text(a: dict[str, Any]) -> str:
    out = [
        f"=== FEATURE AUDIT {a['key']} ===",
        f"asset {a['asset_id']}  decision_at {a['decision_at']}  ({a['decision_time_basis']})",
        f"sources: {json.dumps(a['sources'], sort_keys=True)}",
    ]
    if a.get("strategy"):
        out.append(f"run {a['run_id']}  strategy {a['strategy']}  execution {a['execution_model']}")
    out += ["", "DECISION-TIME FEATURES (observed_at <= decision_at):"]
    for name, g in a["decision_time_features"].items():
        out.append(f"[{name}] {g['availability']}  observed {g['observed_at'] or '-'}  "
                   f"({_f(g['seconds_before_decision'], 's before', 0)})  source {g['source']}")  # fmt: skip
        out += [f"    {k} = {v}" for k, v in g["values"].items()]
    if a["excluded_as_future"]:
        out.append(f"EXCLUDED (observed after the decision): {', '.join(a['excluded_as_future'])}")
    x = a.get("post_decision_execution")
    if x is not None:
        out += [
            "",
            f"POST-DECISION EXECUTION (entry fill at {x['observed_at']}; not a decision-time feature):",
        ]
        out += [f"    {k} = {v}" for k, v in x["values"].items()]
    out += ["", "FUTURE LABELS (outcomes after the decision; never features):"]
    for h, v in a["future_labels"]["horizons"].items():
        out.append(f"  {h:<5} {json.dumps(v, sort_keys=True, default=str)}")
    if a["future_labels"].get("trade") is not None:
        out.append(
            f"  trade {json.dumps(a['future_labels']['trade'], sort_keys=True, default=str)}"
        )
    return "\n".join(out) + "\n"
