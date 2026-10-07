"""The Audit dataset: decisions, their causal feature snapshots, and their future labels.

Two populations, analyzed separately and never pooled:

* ``scout``: Growth Scout anchors (``scout_outcome_observations``), each a ranked
  candidate at one evaluation, with the fixed-horizon outcomes (5m .. 24h). The decision
  time is when the ranking was final: the linked Evidence Archive ``scout`` record's
  observation time (same token, same market observation). Without that record (pruned
  or never archived), the ranking run's start (``anchored_at``) and the anchor's own
  stored fields; Technical context is then NOT_LINKED.
* ``shadow``: Shadow ENTER decisions, per run and strategy, with the simulated trade that
  followed (gross / net P/L, exit, excursions, REALISTIC_V1 friction) and, when the same
  Scout evaluation was also anchored, its fixed-horizon outcomes.

Features are rebuilt only from evidence observed at or before the decision
(`features.build_snapshot`); outcomes are attached only as labels.
"""

import json
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from upscale.services.audit import sources
from upscale.services.audit.config import AuditConfig
from upscale.services.audit.features import (
    TECH_MIN_SNAPSHOTS,
    FeatureGroup,
    Snapshot,
    analyze_group,
    anchor_groups,
    assert_snapshot,
    build_snapshot,
    candidate_groups,
    execution_group,
    missing_group,
    num,
    when,
)
from upscale.services.evidence_archive.store import EvidenceRecord
from upscale.services.outcomes.integrity import INVALID_STATUSES
from upscale.services.shadow.metrics import closed_positions

DECISION_SLACK_SECONDS = 6 * 3600  # a ranking finishes at most this long after it started


@dataclass(frozen=True)
class HorizonLabel:
    status: str  # COMPLETE / PARTIAL / UNAVAILABLE (PENDING and invalid rows are never labels)
    return_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None
    market_status: str | None
    liquidity_change_pct: float | None
    future_stage: str | None


@dataclass(frozen=True)
class TradeLabel:
    status: str  # CLOSED / CLOSED_UNRESOLVED / OPEN / NOT_FILLED
    return_pct: float | None = None  # net of REALISTIC_V1 friction (as the book booked it)
    gross_return_pct: float | None = None
    pnl_usd: float | None = None
    gross_pnl_usd: float | None = None
    friction_usd: float | None = None
    cost_usd: float | None = None
    exit_reason: str | None = None
    exit_at: datetime | None = None
    holding_minutes: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "CLOSED" and self.return_pct is not None


@dataclass(frozen=True)
class AuditObservation:
    population: str  # scout / shadow
    key: str  # scout:<anchor id> / shadow:<decision id>
    asset_id: str
    chain: str
    decision_at: datetime
    decision_basis: str
    snapshot: Snapshot
    horizons: dict[str, HorizonLabel]
    anchor_id: int | None = None
    run_id: str | None = None
    strategy_id: str | None = None
    strategy_version: int | None = None
    execution_model: str | None = None
    execution: FeatureGroup | None = None  # post-decision: the entry fill (shadow only)
    trade: TradeLabel | None = None
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def strategy(self) -> str | None:
        return f"{self.strategy_id}@v{self.strategy_version}" if self.strategy_id else None


@dataclass
class Dataset:
    observations: list[AuditObservation]
    quality: Counter[str]
    notes: list[str]
    databases: dict[str, str | None]
    runs: dict[str, dict[str, Any]]
    decision_lag_minutes: list[float]


@dataclass(frozen=True)
class Paths:
    scout: str | None
    evidence: str | None
    shadow: str | None


def _ts(t: datetime | None) -> float | None:
    return t.timestamp() if t is not None else None


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


# --- shared lookups --------------------------------------------------------------------------------


def _analyze(
    ev: sqlite3.Connection | None,
    wanted: Sequence[tuple[str, datetime]],
    cfg: AuditConfig,
    quality: Counter[str],
) -> dict[tuple[str, datetime], FeatureGroup]:
    """Per (asset, decision time): the latest archived Analyze decision observed in
    [decision - max age, decision], as an analyze_technical group (batched)."""
    if ev is None or not wanted:
        return {}
    age = cfg.analyze_max_age_minutes * 60
    lo = min(t.timestamp() for _, t in wanted) - age
    hi = max(t.timestamp() for _, t in wanted)
    by_asset: dict[str, list[sources.EvidenceIndex]] = {}
    for r in sources.evidence_index(ev, "decision", [a for a, _ in wanted], lo, hi, cfg.batch):
        by_asset.setdefault(r.asset_id, []).append(r)
    for rows in by_asset.values():
        rows.sort(key=lambda r: (r.observed_at, r.id))
    chosen: dict[tuple[str, datetime], int] = {}
    for asset, at in wanted:
        best = None
        for r in by_asset.get(asset, []):
            if r.observed_at > at.timestamp():
                break
            if r.observed_at >= at.timestamp() - age:
                best = r
        if best is not None:
            chosen[(asset, at)] = best.id
    records = sources.evidence_by_ids(ev, list(chosen.values()), cfg.batch)
    out: dict[tuple[str, datetime], FeatureGroup] = {}
    for key, rid in chosen.items():
        rec = records[rid]
        if rec.observed_at > key[1]:  # the query bounds it; never trust a single layer
            quality["analyze_future_rejected"] += 1
            continue
        out[key] = analyze_group(rec.payload, rec.observed_at, f"evidence:decision:{rec.record_id}")
    return out


def _horizon_labels(
    rows: Sequence[sources.HorizonRow],
    audits: dict[tuple[str, int, str], tuple[str, str]],
    quality: Counter[str],
) -> dict[int, dict[str, HorizonLabel]]:
    out: dict[int, dict[str, HorizonLabel]] = {}
    for h in rows:
        if h.status == "PENDING":
            quality[f"horizon_pending_{h.horizon}"] += 1
            continue
        verdict = audits.get(("scout", h.observation_id, h.horizon))
        if verdict is not None and verdict[0] in INVALID_STATUSES:
            quality[f"horizon_integrity_excluded_{h.horizon}"] += 1
            continue
        out.setdefault(h.observation_id, {})[h.horizon] = HorizonLabel(
            h.status, h.return_pct, h.mfe_pct, h.mae_pct, h.market_status,
            h.liquidity_change_pct, h.future_stage,
        )  # fmt: skip
    return out


def _candidate(rec: EvidenceRecord) -> tuple[dict[str, Any], datetime | None, list[dict[str, Any]]]:
    c = rec.payload.get("candidate") or {}
    timing = rec.payload.get("timing") or {}
    market_at = when(timing.get("market_observed_at")) or when(c.get("observed_at"))
    links = [x for x in rec.links.get("evidence") or [] if isinstance(x, dict)]
    return c, market_at, links


# --- Scout anchors ---------------------------------------------------------------------------------


def _scout(
    scout: sqlite3.Connection,
    ev: sqlite3.Connection | None,
    cfg: AuditConfig,
    since: datetime | None,
    until: datetime | None,
    quality: Counter[str],
    lags: list[float],
    anchored: tuple[float, float] | None = None,
) -> list[AuditObservation]:
    lo = since.timestamp() - DECISION_SLACK_SECONDS if since else None
    anchors = sources.anchors(scout, *(anchored or (lo, _ts(until))))
    if not anchors:
        return []
    match: dict[int, sources.EvidenceIndex] = {}  # anchor id -> its Scout evidence row
    if ev is not None:
        idx = sources.evidence_index(
            ev, "scout", [a.canonical_id for a in anchors],
            min(a.anchored_at for a in anchors) - 1,
            max(a.anchored_at for a in anchors) + DECISION_SLACK_SECONDS, cfg.batch,
        )  # fmt: skip
        by_key: dict[tuple[str, float], list[sources.EvidenceIndex]] = {}
        for r in idx:
            if r.provider_at is not None:
                by_key.setdefault((r.asset_id, round(r.provider_at, 3)), []).append(r)
        for a in anchors:
            cands = [r for r in by_key.get((a.canonical_id, round(a.observed_at, 3)), [])
                     if r.observed_at >= a.anchored_at - 1e-3]  # fmt: skip
            if not cands:
                continue
            same_run = [r for r in cands if (t := when(r.started_at)) is not None
                        and abs(t.timestamp() - a.anchored_at) < 1e-3]  # fmt: skip
            match[a.id] = min(same_run or cands, key=lambda r: (r.observed_at, r.id))
    audits = sources.integrity(scout)
    labels = _horizon_labels(
        sources.horizons(scout, [a.id for a in anchors], cfg.batch), audits, quality
    )
    # The decision time is known from the index row; payloads are read chunk by chunk below.
    pre: list[tuple[sources.AnchorRow, datetime, int | None]] = []
    for a in anchors:
        row = match.get(a.id)
        decision_at = _dt(row.observed_at) if row is not None else _dt(a.anchored_at)
        if since is not None and decision_at < since or until is not None and decision_at >= until:
            continue
        pre.append((a, decision_at, row.id if row is not None else None))
    analyze = _analyze(ev, [(a.canonical_id, at) for a, at, _ in pre], cfg, quality)
    out: list[AuditObservation] = []
    for chunk in sources.chunks(pre, cfg.batch):
        records = (sources.evidence_by_ids(ev, [r for _, _, r in chunk if r is not None], cfg.batch)
                   if ev is not None else {})  # fmt: skip
        bodies = sources.anchor_bodies(scout, [a.id for a, _, r in chunk if r is None], cfg.batch)
        for a, decision_at, rid in chunk:
            rec = records[rid] if rid is not None else None
            out.append(_anchor_observation(a, decision_at, rec, bodies.get(a.id, {}), analyze,
                                           quality, lags, labels, cfg))  # fmt: skip
    return out


def _anchor_observation(
    a: sources.AnchorRow,
    decision_at: datetime,
    rec: EvidenceRecord | None,
    body: dict[str, Any],
    analyze: dict[tuple[str, datetime], FeatureGroup],
    quality: Counter[str],
    lags: list[float],
    labels: dict[int, dict[str, HorizonLabel]],
    cfg: AuditConfig,
) -> AuditObservation:
    src = f"scout_outcome_observations:{a.id}"
    extra = {"anchor_reason": a.anchor_reason, "discovery_status": a.discovery_status}
    if rec is not None:
        if rec.observed_at != decision_at:
            raise AssertionError(f"anchor {a.id}: evidence row changed while reading")
        c, market_at, links = _candidate(rec)
        groups = candidate_groups(
            c, rec.observed_at, market_at, f"evidence:scout:{rec.record_id}", links
        )
        basis = "SCOUT_EVIDENCE_DECISION_TIME"
        quality["scout_linked_evidence"] += 1
        if rec.links.get("causal_valid") is False:
            quality["scout_evidence_causal_violation"] += 1
        if c.get("canonical_id") != a.canonical_id:
            raise AssertionError(f"anchor {a.id} linked to evidence of another asset")
    else:
        groups = anchor_groups(body, decision_at, src)
        groups.append(missing_group("technical", "NOT_LINKED", src))
        basis = "RANKING_RUN_START (no linked Scout evidence)"
        quality["scout_unlinked_evidence"] += 1
    groups = [_with(g, extra) if g.name == "scout" else g for g in groups]
    groups.append(analyze.get((a.canonical_id, decision_at))
                  or missing_group("analyze_technical", "NOT_AVAILABLE", "evidence:decision"))  # fmt: skip
    snap = build_snapshot(decision_at, groups, cfg.text_features)
    for name in snap.excluded_future:
        quality[f"excluded_future_{name}"] += 1
    lags.append((decision_at.timestamp() - a.observed_at) / 60)
    return AuditObservation(
        population="scout", key=f"scout:{a.id}", asset_id=a.canonical_id, chain=a.chain,
        decision_at=decision_at, decision_basis=basis, snapshot=snap,
        horizons=labels.get(a.id, {}), anchor_id=a.id,
        sources={"anchor": src, **({"scout_evidence": rec.record_id} if rec else {})},
    )  # fmt: skip


def _with(g: FeatureGroup, extra: dict[str, Any]) -> FeatureGroup:
    return FeatureGroup(g.name, g.availability, g.observed_at, g.source, {**g.values, **extra})


# --- Shadow entries ----------------------------------------------------------------------------------


def summary_groups(
    evidence: dict[str, Any], decision_at: datetime, source: str
) -> list[FeatureGroup]:
    """The evidence summary a Shadow ENTER decision stored at its decision time (used when
    the Scout evidence record it names can't be read)."""
    s, r, t = evidence.get("scout") or {}, evidence.get("risk") or {}, evidence.get("technical")
    missing = set(evidence.get("missing") or [])
    groups = [
        FeatureGroup("scout", "AVAILABLE", decision_at, source, {
            "score": num(s.get("score")), "stage": s.get("stage"), "rank": s.get("rank"),
            "eligible": s.get("eligible"), "data_status": s.get("data_status"),
            "risk_penalty": num(r.get("risk_penalty")),
            "risk_flags": ",".join(sorted({str(f.get("code")) for f in r.get("flags") or []})),
            "blocking_risk_flags": sum(1 for f in r.get("flags") or []
                                       if f.get("severity") in ("high", "critical")),
        }),
        FeatureGroup("market", "AVAILABLE", when(s.get("market_observed_at")) or decision_at, source, {
            "price_usd": num(s.get("price_usd")), "liquidity_usd": num(s.get("liquidity_usd")),
            "market_cap_usd": num(s.get("market_cap_usd")),
        }),
    ]  # fmt: skip
    if isinstance(t, dict):
        snaps = t.get("snapshots") if isinstance(t.get("snapshots"), int) else None
        groups.append(FeatureGroup("technical", "AVAILABLE", decision_at, source, {
            "technical_trend": t.get("trend"), "technical_snapshots": snaps,
            "technical_breakout": t.get("breakout"),
            "technical_confirmed": t.get("trend") == "up" and (snaps or 0) >= TECH_MIN_SNAPSHOTS,
        }))  # fmt: skip
    else:
        groups.append(missing_group("technical", "NOT_AVAILABLE", source))
    social = s.get("social_status")
    groups.append(FeatureGroup("social", "NOT_AVAILABLE" if "SOCIAL_NOT_AVAILABLE" in missing
                               else "AVAILABLE", decision_at, source,
                               {"social_status": social or "SOCIAL_UNAVAILABLE"}))  # fmt: skip
    groups.append(FeatureGroup("safety", "NOT_AVAILABLE" if "SAFETY_NOT_AVAILABLE" in missing
                               else "AVAILABLE", decision_at, source, {
        "safety_status": r.get("safety_status") or "INSUFFICIENT_SAFETY_DATA",
        "mint_authority_active": r.get("mint_authority_active"),
        "freeze_authority_active": r.get("freeze_authority_active"),
        "holder_top10_pct": num(r.get("holder_top10_pct")), "market_status": r.get("market_status"),
    }))  # fmt: skip
    return groups


def _trade(
    position: dict[str, Any] | None,
    fills: list[dict[str, Any]],
    sells: list[dict[str, Any]],
    realistic: bool,
) -> TradeLabel:
    if position is None:
        return TradeLabel(status="NOT_FILLED")
    closed = closed_positions(fills)
    if not closed:
        return TradeLabel(status="OPEN", cost_usd=position["cost_usd"])
    c = closed[0]
    cost = c["cost_usd"]
    if realistic:
        gross = sum(x["gross_pnl_usd"] for x in sells if x["gross_pnl_usd"] is not None)
        friction = sum(x["trade_friction_usd"] for x in sells)
        gross_ok = c["resolved"] and all(x["gross_pnl_usd"] is not None for x in sells) and sells
    else:
        gross, friction, gross_ok = c["pnl_usd"], 0.0, c["resolved"]
    return TradeLabel(
        status="CLOSED" if c["resolved"] else "CLOSED_UNRESOLVED",
        return_pct=c["return_pct"],
        gross_return_pct=gross / cost * 100 if gross_ok and gross is not None and cost else None,
        pnl_usd=c["pnl_usd"],
        gross_pnl_usd=gross if gross_ok else None,
        friction_usd=friction if c["resolved"] else None,
        cost_usd=cost,
        exit_reason=c["exit_reason"],
        exit_at=_dt(c["exit_at"]),
        holding_minutes=c["holding_minutes"],
        mfe_pct=c["mfe_pct"],
        mae_pct=c["mae_pct"],
    )


def _shadow(
    sh: sqlite3.Connection,
    scout: sqlite3.Connection | None,
    ev: sqlite3.Connection | None,
    cfg: AuditConfig,
    since: datetime | None,
    until: datetime | None,
    run: str | None,
    strategy: str | None,
    quality: Counter[str],
    runs_out: dict[str, dict[str, Any]],
) -> list[AuditObservation]:
    runs = {r["run_id"]: r for r in sources.shadow_runs(sh) if run is None or r["run_id"] == run}
    for r in runs.values():
        runs_out[r["run_id"]] = {"execution_model": r["execution_model"],
                                 "clean_data": bool(r["clean_data"]),
                                 "since": _dt(r["since"]).isoformat()}  # fmt: skip
    entries = sources.shadow_entries(sh, list(runs), strategy, _ts(since), _ts(until))
    if not entries:
        return []
    record_ids: dict[str, str] = {}
    for e in entries:
        own = next(
            (f for f in json.loads(e["fingerprints_json"]) if f.get("kind") == "scout"), None
        )
        if own and own.get("record_id"):
            record_ids[e["decision_id"]] = own["record_id"]
    records = (sources.evidence_by_record_ids(ev, list(record_ids.values()), cfg.batch)
               if ev is not None else {})  # fmt: skip
    used = sorted({e["run_id"] for e in entries})
    positions = {
        p["entry_decision_id"]: p for p in sources.shadow_rows(sh, "shadow_positions", used)
    }
    fills: dict[str, list[dict[str, Any]]] = {}
    for t in sources.shadow_rows(sh, "shadow_trades", used):
        fills.setdefault(t["position_id"], []).append(t)
    buys: dict[str, dict[str, Any]] = {}
    sells: dict[str, list[dict[str, Any]]] = {}
    for x in sources.shadow_rows(sh, "shadow_executions", used):
        if x["side"] == "BUY":
            buys[x["position_id"]] = x
        else:
            sells.setdefault(x["position_id"], []).append(x)
    # The same Scout evaluation's anchor (same token, same market observation): its horizons.
    keys: dict[str, tuple[str, float]] = {}
    for e in entries:
        rec = records.get(record_ids.get(e["decision_id"], ""))
        if rec is not None:
            _, market_at, _ = _candidate(rec)
            if market_at is not None:
                keys[e["decision_id"]] = (e["asset_id"], market_at.timestamp())
    anchors = sources.anchors_for(scout, list(keys.values()), cfg.batch) if scout else {}
    labels: dict[int, dict[str, HorizonLabel]] = {}
    if scout is not None and anchors:
        labels = _horizon_labels(
            sources.horizons(scout, [a.id for a in anchors.values()], cfg.batch),
            sources.integrity(scout), Counter(),
        )  # fmt: skip
    analyze = _analyze(ev, [(e["asset_id"], _dt(e["decision_at"])) for e in entries], cfg, quality)
    out: list[AuditObservation] = []
    for e in entries:
        decision_at = _dt(e["decision_at"])
        src = f"shadow_decisions:{e['decision_id']}"
        rec = records.get(record_ids.get(e["decision_id"], ""))
        if rec is not None:
            c, market_at, links = _candidate(rec)
            groups = candidate_groups(
                c, rec.observed_at, market_at, f"evidence:scout:{rec.record_id}", links
            )
            basis = "SHADOW_DECISION_TIME (Scout evidence record)"
            quality["shadow_linked_evidence"] += 1
        else:
            groups = summary_groups(json.loads(e["evidence_json"]), decision_at, src)
            basis = "SHADOW_DECISION_TIME (stored evidence summary)"
            quality["shadow_summary_only"] += 1
        groups.append(analyze.get((e["asset_id"], decision_at))
                      or missing_group("analyze_technical", "NOT_AVAILABLE", "evidence:decision"))  # fmt: skip
        snap = build_snapshot(decision_at, groups, cfg.text_features)
        for name in snap.excluded_future:
            quality[f"excluded_future_{name}"] += 1
        model = runs[e["run_id"]]["execution_model"]
        position = positions.get(e["decision_id"])
        execution = None
        if position is not None:
            execution = execution_group(position, buys.get(position["position_id"]), model)
            if execution.observed_at is not None and execution.observed_at < decision_at:
                raise AssertionError(f"entry fill of {e['decision_id']} before its decision")
        pid = position["position_id"] if position else ""
        trade = _trade(position, fills.get(pid, []), sells.get(pid, []), model == "REALISTIC_V1")
        key = keys.get(e["decision_id"])
        anchor = anchors.get((key[0], round(key[1], 3))) if key else None
        out.append(AuditObservation(
            population="shadow", key=f"shadow:{e['decision_id']}", asset_id=e["asset_id"],
            chain=e["chain"] or e["asset_id"].partition(":")[0], decision_at=decision_at,
            decision_basis=basis, snapshot=snap,
            horizons=labels.get(anchor.id, {}) if anchor else {},
            anchor_id=anchor.id if anchor else None, run_id=e["run_id"],
            strategy_id=e["strategy_id"], strategy_version=e["strategy_version"],
            execution_model=model, execution=execution, trade=trade,
            sources={"decision": src, **({"scout_evidence": rec.record_id} if rec else {}),
                     **({"position": pid} if pid else {})},
        ))  # fmt: skip
    return out


# --- entry point -------------------------------------------------------------------------------------


def _one(
    key: str,
    scout: sqlite3.Connection | None,
    ev: sqlite3.Connection | None,
    sh: sqlite3.Connection | None,
    cfg: AuditConfig,
    quality: Counter[str],
    lags: list[float],
    runs: dict[str, dict[str, Any]],
) -> list[AuditObservation]:
    kind, _, ident = key.partition(":")
    found: list[AuditObservation] = []
    if kind == "scout" and scout is not None and ident.isdigit():
        a = sources.anchor_by_id(scout, int(ident))
        if a is not None:
            window = (a.anchored_at, a.anchored_at + 1e-3)
            found = _scout(scout, ev, cfg, None, None, quality, lags, anchored=window)
    elif kind == "shadow" and sh is not None:
        d = sources.shadow_decision(sh, ident)
        if d is not None and d["action"] == "ENTER":
            at = d["decision_at"]
            found = _shadow(sh, scout, ev, cfg, _dt(at - 1e-3), _dt(at + 1e-3), d["run_id"],
                            d["strategy_id"], quality, runs)  # fmt: skip
    return [o for o in found if o.key == key]


def assert_causal(observations: Sequence[AuditObservation]) -> None:
    """The dataset-wide anti-lookahead assertion, run before any analysis: every
    decision-time feature group was observed at or before its decision, and every entry
    fill (execution facts) precedes the trade outcome it is compared with."""
    for o in observations:
        assert_snapshot(o.snapshot)
        if o.snapshot.decision_at != o.decision_at:
            raise AssertionError(f"{o.key}: snapshot time differs from the decision time")
        x, t = o.execution, o.trade
        if x is not None and x.observed_at is not None:
            if x.observed_at < o.decision_at:
                raise AssertionError(f"{o.key}: entry filled before the decision")
            if t is not None and t.exit_at is not None and x.observed_at > t.exit_at:
                raise AssertionError(f"{o.key}: entry filled after the exit")


def load(
    paths: Paths,
    cfg: AuditConfig,
    since: datetime | None = None,
    until: datetime | None = None,
    population: str | None = None,
    strategy: str | None = None,
    run: str | None = None,
    only: str | None = None,
) -> Dataset:
    """Every observation in [since, until) by decision time. `strategy` / `run` restrict
    the Shadow population and leave out Scout anchors (they belong to no strategy).
    `only`: one observation key (``scout:<anchor id>`` / ``shadow:<decision id>``), built
    by exactly the same code path as the report (for `feature-audit`)."""
    quality: Counter[str] = Counter()
    notes: list[str] = []
    lags: list[float] = []
    runs: dict[str, dict[str, Any]] = {}
    scout = sources.connect(paths.scout)
    ev = sources.connect(paths.evidence)
    sh = sources.connect(paths.shadow)
    databases = {
        "scout": str(paths.scout) if scout else None,
        "evidence": str(paths.evidence) if ev else None,
        "shadow": str(paths.shadow) if sh else None,
    }
    for name, conn, path in (("Scout", scout, paths.scout), ("Evidence Archive", ev, paths.evidence),
                             ("Shadow", sh, paths.shadow)):  # fmt: skip
        if conn is None:
            notes.append(f"{name} database not found ({path}): its data is left out")
    try:
        observations: list[AuditObservation] = []
        if only is not None:
            observations = _one(only, scout, ev, sh, cfg, quality, lags, runs)
            population = "none"
        want_scout = population in (None, "scout") and strategy is None and run is None
        if want_scout and scout is not None:
            observations += _scout(scout, ev, cfg, since, until, quality, lags)
        if population in (None, "shadow") and sh is not None:
            observations += _shadow(sh, scout, ev, cfg, since, until, run, strategy, quality, runs)
        if strategy is not None or run is not None:
            notes.append("--strategy / --run: Scout anchors left out (they belong to no strategy)")
    finally:
        for conn in (scout, ev, sh):
            if conn is not None:
                conn.close()
    observations.sort(key=lambda o: (o.population, o.decision_at, o.key))
    assert_causal(observations)
    return Dataset(observations, quality, notes, databases, runs, lags)
