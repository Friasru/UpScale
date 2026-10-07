"""Point-in-time feature snapshots: what UpScale knew at a decision, group by group.

A snapshot is a set of `FeatureGroup` s (identity, scout, market, technical, analyze
technical, social, safety). Each group carries the time its evidence was observed and
where it came from. The invariant, enforced when a snapshot is built and asserted again
over the whole dataset before any analysis (`assert_causal`):

    group.observed_at <= decision_at

A group whose evidence is later than the decision is never used: it is replaced by an
EXCLUDED_FUTURE group with no values, and counted. Missing evidence is a NOT_AVAILABLE /
NOT_LINKED group with no values: never zero, never "safe".

Execution facts of a Shadow entry (fill price, delay, fees, slippage, impact, latency
drift) exist only after the decision. They are kept apart (`execution`), stamped with the
entry fill time, and only ever compared against the trade outcome that starts at that
fill, never presented as decision-time knowledge.
"""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from upscale.services.shadow.book import latency_drift_usd

Value = float | int | str | bool | None
WINDOWS = ("m5", "h1", "h6", "h24")
TECH_MIN_SNAPSHOTS = 3  # scout_technical's own confirmation: an up-trend over >= 3 snapshots
BLOCKING_SEVERITIES = ("high", "critical")


class LookaheadError(AssertionError):
    """A decision-time feature was observed after its decision."""


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    availability: str  # AVAILABLE / NOT_AVAILABLE / NOT_LINKED / EXCLUDED_FUTURE
    observed_at: datetime | None  # when its evidence was observed (None: no evidence)
    source: str  # the exact rows it came from
    values: dict[str, Value] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.availability == "AVAILABLE"


def missing_group(name: str, availability: str, source: str) -> FeatureGroup:
    return FeatureGroup(name, availability, None, source, {})


@dataclass(frozen=True)
class Snapshot:
    decision_at: datetime
    groups: dict[str, FeatureGroup]
    excluded_future: tuple[str, ...] = ()  # groups dropped because they were later

    def value(self, name: str) -> Value:
        for g in self.groups.values():  # feature names are unique across groups
            if name in g.values:
                return g.values[name]
        return None

    def num(self, name: str) -> float | None:
        v = self.value(name)
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            return None
        return float(v)

    def text(self, name: str) -> str | None:
        v = self.value(name)
        return v if isinstance(v, str) else None


# Free-text fields shown by `feature-audit` only (never bucketed; kept out of a full report's
# memory).
TEXT_FEATURES = frozenset({"surfaced_reasons", "stage_reasons"})


def build_snapshot(
    decision_at: datetime, groups: list[FeatureGroup], keep_text: bool = True
) -> Snapshot:
    """The causal snapshot: every later group is excluded (never silently used)."""
    kept: dict[str, FeatureGroup] = {}
    dropped: list[str] = []
    for g in groups:
        if g.observed_at is not None and g.observed_at > decision_at:
            dropped.append(g.name)
            kept[g.name] = missing_group(g.name, "EXCLUDED_FUTURE", g.source)
        elif not keep_text and TEXT_FEATURES & g.values.keys():
            values = {k: v for k, v in g.values.items() if k not in TEXT_FEATURES}
            kept[g.name] = FeatureGroup(g.name, g.availability, g.observed_at, g.source, values)
        else:
            kept[g.name] = g
    snap = Snapshot(decision_at, kept, tuple(dropped))
    assert_snapshot(snap)
    return snap


def assert_snapshot(s: Snapshot) -> None:
    for g in s.groups.values():
        if g.values and g.observed_at is None:
            raise LookaheadError(f"feature group {g.name} has values but no observation time")
        if g.observed_at is not None and g.observed_at > s.decision_at:
            raise LookaheadError(
                f"feature group {g.name} observed {g.observed_at.isoformat()} after the "
                f"decision at {s.decision_at.isoformat()}"
            )


# --- value helpers --------------------------------------------------------------------------------


def num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def dig(d: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def when(raw: Any) -> datetime | None:
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return datetime.fromtimestamp(raw, UTC)
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


def canonical(chain: str, address: str) -> str:
    """The canonical asset id: EVM addresses lowercased, Solana (and every other chain)
    case-sensitive, exactly as Scout and the Evidence Archive key them."""
    from upscale.services.scout.normalize import canonical_id

    return canonical_id(chain, address)


def _bool(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _flags(flags: Any) -> tuple[str, int]:
    items = [f for f in flags or [] if isinstance(f, dict)]
    codes = sorted({str(f.get("code")) for f in items})
    blocking = sum(1 for f in items if f.get("severity") in BLOCKING_SEVERITIES)
    return ",".join(codes), blocking


# --- groups from a Growth Scout candidate (Evidence Archive ``scout`` record) ------------------------


def candidate_groups(
    c: dict[str, Any],
    decision_at: datetime,
    market_at: datetime | None,
    source: str,
    links: list[dict[str, Any]],
) -> list[FeatureGroup]:
    """Every group a Growth Scout candidate payload holds. `market_at`: when its market
    evidence was observed; `links`: the social / safety evidence the ranking linked."""
    m, mo, q, sm = (c.get(k) or {} for k in ("market", "momentum", "quality", "scout_momentum"))
    pool = m.get("selected_pool") or {}
    codes, blocking = _flags(c.get("risk_flags"))
    families = {f.get("family"): num(f.get("contribution")) for f in sm.get("families") or []
                if isinstance(f, dict)}  # fmt: skip
    rank = c.get("rank")
    identity = FeatureGroup("identity", "AVAILABLE", decision_at, source, {
        "chain": c.get("chain"), "asset_id": c.get("canonical_id"), "address": c.get("address"),
        "pool_address": pool.get("address"), "dex": pool.get("dex"),
        "market_provider": m.get("market_provider"), "symbol": c.get("symbol"),
    })  # fmt: skip
    scout = FeatureGroup("scout", "AVAILABLE", decision_at, source, {
        "score": num(sm.get("score")), "rank": rank if isinstance(rank, int) else None,
        "stage": c.get("stage"), "unconfirmed_stage": c.get("unconfirmed_stage"),
        "eligible": _bool(c.get("eligible")), "data_status": c.get("data_status"),
        "base_score": num(sm.get("base")), "stage_adjustment": num(sm.get("stage_adjustment")),
        "risk_penalty": num(sm.get("risk_penalty")),
        **{f"family_{k}": families.get(k) for k in ("market_activity", "liquidity_quality",
            "social_momentum", "earliness", "cross_confirmation")},
        "risk_flags": codes, "blocking_risk_flags": blocking,
        "surfaced_reasons": ";".join(str(x) for x in c.get("reasons_surfaced") or []),
        "stage_reasons": ";".join(str(x) for x in c.get("stage_reasons") or []),
    })  # fmt: skip
    windows = {w.get("window"): w for w in m.get("windows") or [] if isinstance(w, dict)}
    market_values: dict[str, Value] = {
        "price_usd": num(m.get("price_usd")), "liquidity_usd": num(m.get("liquidity_usd")),
        "market_cap_usd": num(m.get("market_cap_usd")), "fdv_usd": num(m.get("fdv_usd")),
        "pool_age_hours": num(m.get("oldest_pool_age_hours")) or num(m.get("pool_age_hours")),
        "tracked_hours": num(m.get("tracked_hours")),
        "volume_acceleration": num(mo.get("volume_acceleration")),
        "txn_acceleration": num(mo.get("txn_acceleration")),
        "buy_share_h1": num(mo.get("buy_share")),
        "buy_pressure_change": num(mo.get("buy_pressure_change")),
    }  # fmt: skip
    for w in WINDOWS:
        win = windows.get(w) or {}
        market_values[f"volume_{w}_usd"] = num(win.get("volume_usd"))
        market_values[f"price_change_{w}_pct"] = num(win.get("price_change_pct"))
    market_values["price_change_h1_pct"] = (
        num(mo.get("price_change_h1_pct")) or market_values["price_change_h1_pct"]
    )
    market_values["price_change_h24_pct"] = (
        num(mo.get("price_change_h24_pct")) or market_values["price_change_h24_pct"]
    )
    market = FeatureGroup("market", "AVAILABLE", market_at or decision_at, source, market_values)
    t = mo.get("technical")
    if isinstance(t, dict):
        snaps = t.get("snapshots") if isinstance(t.get("snapshots"), int) else None
        technical = FeatureGroup("technical", "AVAILABLE", market_at or decision_at, source, {
            "technical_trend": t.get("trend"), "technical_snapshots": snaps,
            "technical_span_hours": num(t.get("span_hours")),
            "technical_change_pct": num(t.get("change_pct")),
            "technical_breakout": _bool(t.get("breakout")),
            "technical_volume_confirmed": _bool(t.get("volume_confirmed")),
            "technical_higher_lows": _bool(t.get("higher_lows")),
            "technical_confirmed": t.get("trend") == "up" and (snaps or 0) >= TECH_MIN_SNAPSHOTS,
        })  # fmt: skip
    else:
        technical = missing_group("technical", "NOT_AVAILABLE", source)
    social_link = _latest_link(links, "social")
    providers = [p for p in mo.get("social_providers") or [] if isinstance(p, dict)]
    social_status = mo.get("social_status")
    social_values: dict[str, Value] = {
        "social_status": social_status, "social_state": mo.get("social_state"),
        "social_attribution": q.get("social_attribution"), "social_spam_risk": q.get("spam_risk"),
        "social_cross_platform": _bool(mo.get("cross_platform_corroborated")),
        "social_platforms_active": mo.get("platforms_active")
        if isinstance(mo.get("platforms_active"), int) else None,
        "mention_acceleration": num(mo.get("mention_acceleration")),
        "unique_author_acceleration": num(mo.get("unique_author_acceleration")),
        "engagement_acceleration": num(mo.get("engagement_acceleration")),
        "social_providers": ",".join(sorted(f"{p.get('provider')}:{p.get('status')}"
                                            for p in providers)),
        "social_providers_ok": sum(1 for p in providers if str(p.get("status", "")).startswith(
            "PROVIDER_OK") or p.get("status") == "PROVIDER_CHECKED_ZERO_MATCHES"),
    }  # fmt: skip
    if social_status in (None, "SOCIAL_UNAVAILABLE"):
        # The unavailable status itself is decision-time knowledge (Scout knew it had none).
        social = FeatureGroup("social", "NOT_AVAILABLE", decision_at, source, {
            "social_status": "SOCIAL_UNAVAILABLE",
            "social_providers": social_values["social_providers"]})  # fmt: skip
    else:
        at = when(social_link.get("observed_at")) if social_link else None
        social = FeatureGroup("social", "AVAILABLE", at or decision_at,
                              _src(source, social_link), social_values)  # fmt: skip
    safety_status = q.get("safety_status")
    safety_link = _latest_link(links, "safety")
    safety_values: dict[str, Value] = {
        "safety_status": safety_status,
        "mint_authority_active": _bool(q.get("mint_authority_active")),
        "freeze_authority_active": _bool(q.get("freeze_authority_active")),
        "holder_top1_pct": num(q.get("holder_top1_pct")),
        "holder_top10_pct": num(q.get("holder_top10_pct")),
        "holder_data_lower_bound": _bool(q.get("holder_data_lower_bound")),
        "liquidity_quality": q.get("liquidity_quality"), "market_status": q.get("market_status"),
        "safety_missing": ",".join(sorted(str(x) for x in q.get("safety_missing") or [])),
    }  # fmt: skip
    if safety_status in (None, "INSUFFICIENT_SAFETY_DATA"):
        safety = FeatureGroup("safety", "NOT_AVAILABLE", decision_at, source, {
            "safety_status": "INSUFFICIENT_SAFETY_DATA",
            "liquidity_quality": safety_values["liquidity_quality"],
            "market_status": safety_values["market_status"]})  # fmt: skip
    else:
        at = when(safety_link.get("observed_at")) if safety_link else None
        safety = FeatureGroup("safety", "AVAILABLE", at or decision_at,
                              _src(source, safety_link), safety_values)  # fmt: skip
    return [identity, scout, market, technical, social, safety]


def _latest_link(links: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    mine = [x for x in links if isinstance(x, dict) and x.get("kind") == kind]
    return max(mine, key=lambda x: str(x.get("observed_at"))) if mine else {}


def _src(source: str, link: dict[str, Any]) -> str:
    return f"{source} + evidence:{link['kind']}:{link.get('record_id')}" if link else source


# --- groups from a Scout anchor body (outcome store) ------------------------------------------------


def anchor_groups(body: dict[str, Any], decision_at: datetime, source: str) -> list[FeatureGroup]:
    """The groups a stored `ScoutObservation` holds (no Technical context: anchors don't
    store it; it comes from the linked Scout evidence record when one exists)."""
    m, c, s, q = (body.get(k) or {} for k in ("market", "components", "social", "safety"))
    market_at = when(body.get("observed_at")) or decision_at
    codes = ",".join(sorted({str(f.get("code")) for f in body.get("risk_flags") or []
                             if isinstance(f, dict)}))  # fmt: skip
    blocking = sum(1 for f in body.get("risk_flags") or []
                   if isinstance(f, dict) and f.get("severity") in BLOCKING_SEVERITIES)  # fmt: skip
    identity = FeatureGroup("identity", "AVAILABLE", decision_at, source, {
        "chain": body.get("chain"), "asset_id": body.get("canonical_id"),
        "address": body.get("address"), "pool_address": body.get("pool_address"),
        "dex": body.get("pool_dex"), "market_provider": body.get("market_provider"),
        "symbol": body.get("symbol"),
    })  # fmt: skip
    scout = FeatureGroup("scout", "AVAILABLE", decision_at, source, {
        "score": num(body.get("score")), "rank": body.get("rank"), "stage": body.get("stage"),
        "unconfirmed_stage": body.get("unconfirmed_stage"), "eligible": True,
        "data_status": "STALE_CARRIED" if body.get("discovery_status") == "STALE_CARRIED"
        else "CURRENT",
        "base_score": num(c.get("base")), "stage_adjustment": num(c.get("stage_adjustment")),
        "risk_penalty": num(c.get("risk_penalty")),
        **{f"family_{k}": num(c.get(k)) for k in ("market_activity", "liquidity_quality",
            "social_momentum", "earliness", "cross_confirmation")},
        "risk_flags": codes, "blocking_risk_flags": blocking,
        "surfaced_reasons": ";".join(str(x) for x in body.get("reasons") or []),
        "stage_reasons": ";".join(str(x) for x in body.get("stage_reasons") or []),
    })  # fmt: skip
    market = FeatureGroup("market", "AVAILABLE", market_at, source, {
        "price_usd": num(m.get("price_usd")), "liquidity_usd": num(m.get("liquidity_usd")),
        "market_cap_usd": num(m.get("market_cap_usd")), "fdv_usd": num(m.get("fdv_usd")),
        "pool_age_hours": num(m.get("pool_age_hours")), "tracked_hours": num(m.get("tracked_hours")),
        "volume_acceleration": num(m.get("volume_acceleration")),
        "txn_acceleration": num(m.get("txn_acceleration")),
        "buy_share_h1": num(m.get("buy_share_h1")),
        "buy_pressure_change": num(m.get("buy_pressure_change")),
        "volume_h1_usd": num(m.get("volume_h1_usd")), "volume_h24_usd": num(m.get("volume_h24_usd")),
        "price_change_h1_pct": num(m.get("price_change_h1_pct")),
        "price_change_h24_pct": num(m.get("price_change_h24_pct")),
    })  # fmt: skip
    status = s.get("status")
    if status in (None, "SOCIAL_UNAVAILABLE"):
        social = FeatureGroup("social", "NOT_AVAILABLE", decision_at, source,
                              {"social_status": "SOCIAL_UNAVAILABLE"})  # fmt: skip
    else:
        social = FeatureGroup("social", "AVAILABLE", decision_at, source, {
            "social_status": status, "social_state": s.get("state"),
            "social_attribution": s.get("attribution"), "social_spam_risk": s.get("spam_risk"),
            "social_cross_platform": _bool(s.get("cross_platform")),
            "social_providers": ",".join(f"{k}:{v}" for k, v in (("farcaster", s.get("farcaster_state")),
                                                                ("x", s.get("x_state"))) if v),
        })  # fmt: skip
    safety_status = q.get("status")
    if safety_status in (None, "INSUFFICIENT_SAFETY_DATA"):
        safety = FeatureGroup("safety", "NOT_AVAILABLE", decision_at, source, {
            "safety_status": "INSUFFICIENT_SAFETY_DATA",
            "liquidity_quality": q.get("liquidity_quality"), "market_status": q.get("market_status")})  # fmt: skip
    else:
        safety = FeatureGroup("safety", "AVAILABLE", decision_at, source, {
            "safety_status": safety_status,
            "mint_authority_active": _bool(q.get("mint_authority_active")),
            "freeze_authority_active": _bool(q.get("freeze_authority_active")),
            "holder_top1_pct": num(q.get("holder_top1_pct")),
            "holder_top10_pct": num(q.get("holder_top10_pct")),
            "holder_data_lower_bound": _bool(q.get("holder_data_lower_bound")),
            "liquidity_quality": q.get("liquidity_quality"), "market_status": q.get("market_status"),
        })  # fmt: skip
    return [identity, scout, market, social, safety]


# --- Analyze Technical (RSI / MACD / support-resistance) ---------------------------------------------


def analyze_group(payload: dict[str, Any], observed_at: datetime, source: str) -> FeatureGroup:
    """The Technical Agent's findings of an archived Analyze decision. The group's time is
    the later of the decision record and its last candle (a candle can't be newer than
    the analysis, but if it ever were, the group would be excluded as future)."""
    f = dig(payload, "agents", "technical", "findings")
    decision = payload.get("decision") or {}
    if not isinstance(f, dict) or not f:
        return missing_group("analyze_technical", "NOT_AVAILABLE", source)
    indicators = {i.get("name"): i for i in f.get("indicators") or [] if isinstance(i, dict)}
    rsi = next(
        (i for i in indicators.values() if i.get("kind") == "rsi" and i.get("available")), None
    )
    macd = next(
        (i for i in indicators.values() if i.get("kind") == "macd" and i.get("available")), None
    )
    hist = num(macd.get("histogram")) if macd else None
    levels = f.get("levels") or {}
    support = (levels.get("support") or [None])[0]
    resistance = (levels.get("resistance") or [None])[0]
    last_candle = when(f.get("last_candle_at"))
    stamp = max(observed_at, last_candle) if last_candle is not None else observed_at
    return FeatureGroup("analyze_technical", "AVAILABLE", stamp, source, {
        "analyze_trend": dig(f, "trend", "label"), "analyze_timeframe": f.get("timeframe"),
        "analyze_rsi_14": num(rsi.get("value")) if rsi else None,
        "analyze_macd_state": None if hist is None else "bullish" if hist > 0
        else "bearish" if hist < 0 else "flat",
        "analyze_support_distance_pct": num(support.get("distance_pct"))
        if isinstance(support, dict) else None,
        "analyze_resistance_distance_pct": num(resistance.get("distance_pct"))
        if isinstance(resistance, dict) else None,
        "analyze_action": decision.get("action"), "analyze_confidence": decision.get("confidence"),
        "analyze_candles": f.get("candle_count") if isinstance(f.get("candle_count"), int) else None,
    })  # fmt: skip


# --- execution (post-decision: the entry fill) -----------------------------------------------------


def execution_group(
    position: dict[str, Any], buy: dict[str, Any] | None, model: str
) -> FeatureGroup:
    """Entry execution facts, stamped with the entry fill time. REALISTIC_V1 reads the BUY
    execution row; IDEALIZED_NO_FEES fills at the observed price with no friction."""
    filled = when(position["entry_at"])
    src = f"shadow_positions:{position['position_id']}"
    if buy is None:
        return FeatureGroup("execution", "AVAILABLE", filled, src, {
            "execution_model": model, "intent_price": position["entry_price"],
            "observed_fill_price": position["entry_price"],
            "execution_price": position["entry_price"], "execution_delay_seconds": 0.0,
            "fee_usd": 0.0, "fee_bps": 0.0, "slippage_bps": 0.0, "slippage_cost_usd": 0.0,
            "price_impact_bps": 0.0, "impact_status": "NOT_MODELED",
            "latency_drift_usd": 0.0, "stored_latency_cost_usd": 0.0,
            "entry_friction_usd": 0.0,
        })  # fmt: skip
    drift = latency_drift_usd("BUY", buy["reference_price"], buy["observed_price"], buy["quantity"])
    intent = when(buy["intent_at"])
    return FeatureGroup("execution", "AVAILABLE", when(buy["filled_at"]),
                        f"{src} + shadow_executions:{buy['execution_id']}", {
        "execution_model": model, "intent_price": buy["reference_price"],
        "observed_fill_price": buy["observed_price"], "execution_price": buy["execution_price"],
        "intent_at": intent.isoformat() if intent else None,
        "execution_delay_seconds": buy["delay_seconds"], "configured_latency_seconds": buy["latency_seconds"],
        "fee_usd": buy["fee_usd"], "fee_bps": buy["fee_bps"], "slippage_bps": buy["slippage_bps"],
        "slippage_cost_usd": buy["slippage_cost_usd"], "price_impact_bps": buy["impact_bps"],
        "price_impact_cost_usd": buy["impact_cost_usd"], "impact_status": buy["impact_status"],
        "fill_liquidity_usd": buy["liquidity_usd"], "latency_drift_usd": drift,
        "stored_latency_cost_usd": buy["latency_cost_usd"], "entry_friction_usd": buy["friction_usd"],
    })  # fmt: skip
