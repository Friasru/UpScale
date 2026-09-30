"""The calibration dataset: live and replay observations, normalized, never merged.

* LIVE_FORWARD: the live outcome tables (Scout anchors, Analyze decisions and their
  completed horizons), read-only. Split by the sticky calendar policy
  (`LiveSplitPolicy`).
* HISTORICAL_REPLAY: Replay Lab samples, decisions and outcomes, read-only, keeping the
  split each sample was assigned when planned (including sticky HOLDOUT windows).
* SHADOW: shadow-strategy evaluations (none exist yet); strategy calibration, kept apart
  from Scout signal calibration.

Every observation keeps its origin, split, decision time, asset and version fingerprints.
Features are the values the system had at decision time; missing evidence is a label
(`SAFETY_NOT_AVAILABLE`...), never a zero. HOLDOUT observations are only returned when a
loader is called with `include_holdout=True`, which only the final evaluation does.
"""

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from upscale.services.calibration.config import (
    CalibrationConfig,
    CorrelationRules,
    LiveSplitPolicy,
    Origin,
    Split,
)
from upscale.services.outcomes.audit import latest_audits
from upscale.services.outcomes.integrity import INVALID_STATUSES
from upscale.services.outcomes.models import DecisionObservation, HorizonOutcome, ScoutObservation
from upscale.services.outcomes.store import _H_COLUMNS, _decision, _horizon, _scout
from upscale.services.replay_lab.models import ReplayDecisionRecord, ReplayHorizonOutcome
from upscale.services.replay_lab.store import record_hash

Kind = Literal["scout", "decision", "replay", "shadow"]
Value = float | str | bool | None
FAMILIES = (
    "market_activity",
    "liquidity_quality",
    "social_momentum",
    "earliness",
    "cross_confirmation",
)


class HoldoutSealedError(PermissionError):
    """HOLDOUT data was requested outside an explicit final evaluation."""


@dataclass(frozen=True)
class OutcomeView:
    status: str
    return_pct: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    max_drawdown_pct: float | None = None
    time_to_mfe_minutes: float | None = None
    time_to_mae_minutes: float | None = None
    liquidity_change_pct: float | None = None
    market_status: str | None = None
    trigger_touched: bool | None = None
    invalidation_touched: bool | None = None

    @property
    def measured(self) -> bool:
        return self.return_pct is not None


@dataclass(frozen=True)
class Observation:
    origin: Origin
    kind: Kind
    key: str
    asset_id: str
    at: datetime  # decision time: only evidence at or before it was used
    split: Split
    purged: bool
    features: dict[str, Value]
    flags: frozenset[str]  # Scout risk flag codes
    missing: frozenset[str]  # SAFETY_NOT_AVAILABLE, SOCIAL_NOT_AVAILABLE...
    outcomes: dict[str, OutcomeView]
    run_id: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    regime: dict[str, str] = field(default_factory=dict)
    # Horizons left out because an integrity audit confirmed them invalid (horizon -> status).
    integrity_excluded: dict[str, str] = field(default_factory=dict)

    def f(self, name: str) -> float | None:
        v = self.features.get(name)
        return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None

    def s(self, name: str) -> str | None:
        v = self.features.get(name)
        return v if isinstance(v, str) else None


# --- splits ------------------------------------------------------------------------------------


def live_split(at: datetime, policy: LiveSplitPolicy) -> Split:
    day = (at.astimezone(UTC).date() - date.fromisoformat(policy.anchor)).days % policy.cycle_days
    if day < policy.calibration_days:
        return "CALIBRATION"
    if day < policy.calibration_days + policy.validation_days:
        return "VALIDATION"
    return "HOLDOUT"


def live_purged(
    at: datetime, policy: LiveSplitPolicy, horizon: timedelta = timedelta(hours=24)
) -> bool:
    """Its longest outcome window ends in a day of another split (leakage across splits)."""
    return live_split(at, policy) != live_split(at + horizon, policy)


# --- features -------------------------------------------------------------------------------------


def _missing(features: dict[str, Value]) -> frozenset[str]:
    out = set()
    if features.get("safety_status") in (None, "INSUFFICIENT_SAFETY_DATA"):
        out.add("SAFETY_NOT_AVAILABLE")
    if features.get("social_status") in (None, "SOCIAL_UNAVAILABLE"):
        out.add("SOCIAL_NOT_AVAILABLE")
    if features.get("market_cap_usd") is None:
        out.add("MARKET_CAP_NOT_AVAILABLE")
    if features.get("liquidity_usd") is None:
        out.add("LIQUIDITY_NOT_AVAILABLE")
    if features.get("technical_trend") is None and features.get("rsi_14") is None:
        out.add("TECHNICAL_NOT_AVAILABLE")
    return frozenset(out)


def _ratio(a: Value, b: Value) -> float | None:
    if isinstance(a, int | float) and isinstance(b, int | float) and b:
        return float(a) / float(b)
    return None


def scout_features(o: ScoutObservation) -> dict[str, Value]:
    m, c, s, q = o.market, o.components, o.social, o.safety
    return {
        "stage": o.stage,
        "score": o.score,
        **{f"family_{k}": getattr(c, k) for k in FAMILIES},
        "stage_adjustment": c.stage_adjustment,
        "risk_penalty": c.risk_penalty,
        "liquidity_usd": m.liquidity_usd,
        "market_cap_usd": m.market_cap_usd,
        "fdv_usd": m.fdv_usd,
        "liquidity_to_cap": _ratio(m.liquidity_usd, m.market_cap_usd or m.fdv_usd),
        "volume_acceleration": m.volume_acceleration,
        "txn_acceleration": m.txn_acceleration,
        "buy_share_h1": m.buy_share_h1,
        "buy_pressure_change": m.buy_pressure_change,
        "price_change_h1_pct": m.price_change_h1_pct,
        "price_change_h24_pct": m.price_change_h24_pct,
        "pool_age_hours": m.pool_age_hours,
        "social_status": s.status,
        "social_state": s.state,
        "social_cross_platform": s.cross_platform,
        "social_spam_risk": s.spam_risk,
        "social_attribution": s.attribution,
        "safety_status": q.status,
        "holder_top1_pct": q.holder_top1_pct,
        "holder_top10_pct": q.holder_top10_pct,
        "holder_data_lower_bound": q.holder_data_lower_bound,
        "mint_authority_active": q.mint_authority_active,
        "freeze_authority_active": q.freeze_authority_active,
        "liquidity_quality": q.liquidity_quality,
        "rank": o.rank,
    }


def decision_features(d: DecisionObservation) -> dict[str, Value]:
    return {
        "action": d.action,
        "confidence": d.confidence,
        "setup": d.setup,
        "risk_level": d.risk.level,
        "uncertainty_level": d.risk.uncertainty_level,
        "bullish_score": d.bullish_score,
        "bearish_score": d.bearish_score,
        "has_trigger": d.buy_trigger is not None or d.sell_trigger is not None,
        "has_invalidation": d.invalidation is not None,
        "liquidity_usd": d.liquidity_usd,
        "timeframe": d.timeframe,
    }


def replay_features(r: ReplayDecisionRecord) -> dict[str, Value]:
    f = r.features
    cand = r.scout.candidate or {}
    quality = cand.get("quality") or {}
    momentum = cand.get("momentum") or {}
    out: dict[str, Value] = {
        "stage": r.scout.stage,
        "score": r.scout.score,
        **{f"family_{k}": r.scout.families.get(k) for k in FAMILIES},
        "stage_adjustment": r.scout.stage_adjustment,
        "risk_penalty": r.scout.risk_penalty,
        "social_status": r.scout.social_status or r.social.get("status"),
        "social_state": r.social.get("state"),
        "social_cross_platform": momentum.get("cross_platform_corroborated"),
        "safety_status": r.scout.safety_status,
        "holder_top1_pct": quality.get("holder_top1_pct"),
        "holder_top10_pct": quality.get("holder_top10_pct"),
        "holder_data_lower_bound": quality.get("holder_data_lower_bound"),
        "mint_authority_active": quality.get("mint_authority_active"),
        "freeze_authority_active": quality.get("freeze_authority_active"),
        "action": r.action,
        "confidence": r.confidence,
        "risk_level": r.risk_level,
        "uncertainty_level": r.uncertainty_level,
    }
    for key in ("liquidity_usd", "market_cap_usd", "fdv_usd", "volume_acceleration",
                "txn_acceleration", "buy_share_h1", "buy_pressure_change", "price_change_h1_pct",
                "price_change_h24_pct", "pool_age_hours", "volatility_4h_pct", "rsi_14",
                "technical_trend", "liquidity_change_pct", "liquidity_quality"):  # fmt: skip
        v = f.get(key)
        out[key] = v if isinstance(v, int | float | str | bool) or v is None else None
    out["liquidity_to_cap"] = _ratio(
        out.get("liquidity_usd"), out.get("market_cap_usd") or out.get("fdv_usd")
    )
    if r.decision is not None:
        out["has_trigger"] = (
            r.decision.buy_trigger is not None or r.decision.sell_trigger is not None
        )
        out["has_invalidation"] = r.decision.invalidation is not None
        out["bullish_score"] = r.decision.bullish_score
        out["bearish_score"] = r.decision.bearish_score
    return out


def _live_outcome(h: HorizonOutcome) -> OutcomeView:
    p = h.price
    return OutcomeView(
        status=h.status,
        return_pct=p.return_pct if p else None,
        mfe_pct=p.mfe_pct if p else None,
        mae_pct=p.mae_pct if p else None,
        max_drawdown_pct=p.max_drawdown_pct if p else None,
        time_to_mfe_minutes=p.time_to_mfe_minutes if p else None,
        time_to_mae_minutes=p.time_to_mae_minutes if p else None,
        liquidity_change_pct=h.market.liquidity_change_pct if h.market else None,
        market_status=h.market_status,
        trigger_touched=_touched(h.triggers, invalidation=False),
        invalidation_touched=_touched(h.triggers, invalidation=True),
    )


def _replay_outcome(h: ReplayHorizonOutcome) -> OutcomeView:
    p = h.price
    return OutcomeView(
        status=h.status,
        return_pct=p.return_pct if p else None,
        mfe_pct=p.mfe_pct if p else None,
        mae_pct=p.mae_pct if p else None,
        max_drawdown_pct=p.max_drawdown_pct if p else None,
        time_to_mfe_minutes=p.time_to_mfe_minutes if p else None,
        time_to_mae_minutes=p.time_to_mae_minutes if p else None,
        liquidity_change_pct=h.market.liquidity_change_pct if h.market else None,
        market_status=h.market_status,
        trigger_touched=_touched(h.triggers, invalidation=False),
        invalidation_touched=_touched(h.triggers, invalidation=True),
    )


def _touched(triggers: dict[str, Any], invalidation: bool) -> bool | None:
    items = [t for k, t in triggers.items() if (k == "invalidation") == invalidation]
    reached = [t.reached for t in items if t.reached is not None]
    return any(reached) if reached else None


# --- loaders ----------------------------------------------------------------------------------------


def _ro(path: str | Path) -> sqlite3.Connection | None:
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def load_live(
    path: str | Path,
    cfg: CalibrationConfig,
    include_holdout: bool = False,
    exclude_invalid: bool = True,
) -> list[Observation]:
    """`exclude_invalid`: outcomes an integrity audit confirmed invalid (the latest audit of
    that horizon, `outcomes.integrity.INVALID_STATUSES`) are left out and reported. Valid
    extreme moves and unaudited outcomes are always kept: size alone excludes nothing."""
    conn = _ro(path)
    if conn is None:
        return []
    try:
        if "scout_outcome_observations" not in _tables(conn):
            return []
        policy = cfg.live_split
        out: list[Observation] = []
        horizons: dict[tuple[str, int], dict[str, OutcomeView]] = {}
        audits = latest_audits(conn)
        excluded: dict[tuple[str, int], dict[str, str]] = {}
        for table, kind in (
            ("scout_outcome_horizons", "scout"),
            ("decision_outcome_horizons", "decision"),
        ):
            for r in conn.execute(f"SELECT {_H_COLUMNS} FROM {table} h ORDER BY observation_id"):
                h = _horizon(r)
                verdict = audits.get((kind, h.observation_id, h.horizon))
                if exclude_invalid and verdict is not None and verdict[0] in INVALID_STATUSES:
                    excluded.setdefault((kind, h.observation_id), {})[h.horizon] = verdict[0]
                    continue
                horizons.setdefault((kind, h.observation_id), {})[h.horizon] = _live_outcome(h)
        scouts: dict[int, ScoutObservation] = {}
        for body, oid in conn.execute(
            "SELECT body_json, id FROM scout_outcome_observations ORDER BY id"
        ):
            o = _scout(body, oid)
            scouts[oid] = o
            split = live_split(o.observed_at, policy)
            if split == "HOLDOUT" and not include_holdout:
                continue
            feats = scout_features(o)
            out.append(Observation(
                origin="LIVE_FORWARD", kind="scout", key=f"live:scout:{oid}", asset_id=o.canonical_id,
                at=o.observed_at, split=split, purged=live_purged(o.observed_at, policy),
                features=feats, flags=frozenset(f.code for f in o.risk_flags), missing=_missing(feats),
                outcomes=horizons.get(("scout", oid), {}), run_id=o.run_id,
                integrity_excluded=excluded.get(("scout", oid), {}),
            ))  # fmt: skip
        for body, did in conn.execute(
            "SELECT body_json, id FROM decision_observations ORDER BY id"
        ):
            d = _decision(body, did)
            split = live_split(d.analyzed_at, policy)
            if split == "HOLDOUT" and not include_holdout:
                continue
            feats = decision_features(d)
            linked = scouts.get(d.scout_observation_id or -1)
            flags: frozenset[str] = frozenset()
            if linked is not None and linked.observed_at <= d.analyzed_at:
                feats = scout_features(linked) | feats  # Scout context known at the decision
                flags = frozenset(f.code for f in linked.risk_flags)
            out.append(Observation(
                origin="LIVE_FORWARD", kind="decision", key=f"live:decision:{did}", asset_id=d.asset_id,
                at=d.analyzed_at, split=split, purged=live_purged(d.analyzed_at, policy),
                features=feats, flags=flags, missing=_missing(feats),
                outcomes=horizons.get(("decision", did), {}),
                integrity_excluded=excluded.get(("decision", did), {}),
            ))  # fmt: skip
        return out
    finally:
        conn.close()


def load_replay(path: str | Path, include_holdout: bool = False) -> list[Observation]:
    conn = _ro(path)
    if conn is None:
        return []
    try:
        if "replay_decisions" not in _tables(conn):
            return []
        where = "" if include_holdout else "WHERE s.split != 'HOLDOUT'"
        rows = conn.execute(
            f"""
            SELECT s.id, s.job_id, s.sample_key, s.asset_id, s.decision_at, s.split, s.purged,
                d.record_json, d.record_hash
            FROM replay_samples s JOIN replay_decisions d ON d.sample_id = s.id {where}
            ORDER BY s.decision_at, s.id
            """
        ).fetchall()
        outs: dict[int, dict[str, OutcomeView]] = {}
        for sid, body in conn.execute("SELECT sample_id, record_json FROM replay_outcomes"):
            h = ReplayHorizonOutcome.model_validate_json(body)
            outs.setdefault(sid, {})[h.horizon] = _replay_outcome(h)
        out: list[Observation] = []
        seen: set[str] = set()
        for sid, job, key, asset, at, split, purged, body, digest in rows:
            if record_hash(body) != digest:
                raise RuntimeError(f"replay decision of sample {sid} fails its hash check")
            if key in seen:  # the same planned sample in two jobs: one observation
                continue
            seen.add(key)
            r = ReplayDecisionRecord.model_validate_json(body)
            feats = replay_features(r)
            out.append(Observation(
                origin="HISTORICAL_REPLAY", kind="replay", key=f"replay:{key}", asset_id=asset,
                at=datetime.fromtimestamp(at, UTC), split=split, purged=bool(purged), features=feats,
                flags=frozenset(f["code"] for f in r.scout.risk_flags), missing=_missing(feats),
                outcomes=outs.get(sid, {}), run_id=job, versions=r.versions,
            ))  # fmt: skip
        return out
    finally:
        conn.close()


def load_shadow(path: str | Path) -> list[Observation]:
    """Shadow-strategy evaluations (strategy calibration only). None exist in v1."""
    conn = _ro(path)
    if conn is None:
        return []
    try:
        if "shadow_evaluations" not in _tables(conn):
            return []
        n = conn.execute("SELECT COUNT(*) FROM shadow_evaluations").fetchone()[0]
        return [] if not n else []  # the evaluation format is defined when strategies exist
    finally:
        conn.close()


# --- correlation controls and regimes ---------------------------------------------------------------


def diversify(observations: Sequence[Observation], rules: CorrelationRules) -> list[Observation]:
    """Per origin, split and asset: at least `min_spacing` apart (earliest kept) and at most
    `max_per_asset`, spread over the asset's history. Deterministic."""
    groups: dict[tuple[str, str, str, str], list[Observation]] = {}
    for o in sorted(observations, key=lambda o: (o.at, o.key)):
        groups.setdefault((o.origin, o.kind, o.split, o.asset_id), []).append(o)
    kept: list[Observation] = []
    spacing = timedelta(minutes=rules.min_spacing_minutes)
    for items in groups.values():
        spaced: list[Observation] = []
        for o in items:
            if not spaced or o.at - spaced[-1].at >= spacing:
                spaced.append(o)
        if len(spaced) > rules.max_per_asset:
            step = len(spaced) / rules.max_per_asset
            spaced = [spaced[int(i * step)] for i in range(rules.max_per_asset)]
        kept += spaced
    return sorted(kept, key=lambda o: (o.at, o.key))


def with_regimes(observations: Sequence[Observation]) -> list[Observation]:
    """Simple deterministic regimes from the observed universe of the PREVIOUS UTC day
    (causal: a day's label uses only earlier data). UNKNOWN without a previous day."""
    by_day: dict[date, list[Observation]] = {}
    for o in observations:
        by_day.setdefault(o.at.astimezone(UTC).date(), []).append(o)

    def med(values: Iterable[float | None]) -> float | None:
        v = sorted(x for x in values if x is not None)
        return v[len(v) // 2] if v else None

    daily = {
        d: {
            "move": med(o.f("price_change_h24_pct") for o in obs),
            "vol": med(
                abs(x) if (x := o.f("price_change_h1_pct")) is not None else None for o in obs
            ),
            "assets": float(len({o.asset_id for o in obs})),
            "liq": med(o.f("liquidity_change_pct") for o in obs),
        }
        for d, obs in by_day.items()
    }
    vol_mid = med(v["vol"] for v in daily.values())
    assets_mid = med(v["assets"] for v in daily.values())
    out = []
    for o in observations:
        prev = daily.get(o.at.astimezone(UTC).date() - timedelta(days=1))
        regime: dict[str, str] = {}
        if prev is None:
            regime = {"volatility": "UNKNOWN", "direction": "UNKNOWN", "meme_activity": "UNKNOWN",
                      "liquidity": "UNKNOWN"}  # fmt: skip
        else:
            regime["volatility"] = (
                "UNKNOWN" if prev["vol"] is None or vol_mid is None
                else "HIGH_VOLATILITY" if prev["vol"] > vol_mid else "LOW_VOLATILITY"
            )  # fmt: skip
            regime["direction"] = (
                "UNKNOWN" if prev["move"] is None else "RISK_ON" if prev["move"] > 0 else "RISK_OFF"
            )
            regime["meme_activity"] = (
                "UNKNOWN" if assets_mid is None
                else "HIGH_MEME_ACTIVITY" if (prev["assets"] or 0) > assets_mid else "LOW_MEME_ACTIVITY"
            )  # fmt: skip
            regime["liquidity"] = (
                "UNKNOWN" if prev["liq"] is None
                else "LIQUIDITY_EXPANSION" if prev["liq"] > 0 else "LIQUIDITY_CONTRACTION"
            )  # fmt: skip
        out.append(Observation(**{**o.__dict__, "regime": regime}))
    return out


def fingerprint(observations: Sequence[Observation]) -> str:
    """The data snapshot: which observations, with which measured outcomes."""
    h = hashlib.sha256()
    for o in sorted(observations, key=lambda o: o.key):
        h.update(o.key.encode())
        h.update(
            json.dumps(
                {k: v.return_pct for k, v in sorted(o.outcomes.items())}, default=str
            ).encode()
        )
    return h.hexdigest()[:16]
