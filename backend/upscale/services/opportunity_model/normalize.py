"""Pure, deterministic interpretation of stored evidence into Opportunity V1 source facts.

No I/O and no clock: every function reads only its arguments. A record a loader selected
is checked again here: observed after ``decision_at`` raises `OpportunityCausalityError`,
another chain / token raises `OpportunityIdentityError` (never "missing"). For
LIVE_FORWARD, an archive record written after ``decision_at`` was not physically readable
at that time and raises too; HISTORICAL_REPLAY may use it (its evidence was observed in
time).

Scout: Opportunity reads Scout's per-sub-signal evidence only. Its composite score, family
scores, legacy on-chain safety, social copies, technical sub-signal and Safety-owned
market / liquidity facts are diagnostics. ``cross_confirmation`` is dropped entirely: its
archived family score mixes market agreement with a social term weighted by a config share
that isn't archived (``cross.market_share``), minus unarchived contradiction penalties, then
clamped, so the social part can't be separated exactly; and its market confirmations repeat
FLOW / PRICE / LIQUIDITY facts already kept elsewhere.
"""

import hashlib
import json
import zlib
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from upscale.services.chains import is_solana_address
from upscale.services.evidence_archive.store import EvidenceRecord
from upscale.services.opportunity_model.config import (
    ARCHIVE_STORE,
    MIN_SCOUT_TIMING_VERSION,
    NEWS_AGENT,
    SAFETY_RULES_VERSION,
    SAFETY_SNAPSHOT_SCHEMA,
    SAFETY_STORE,
    SUPPORTED_CHAINS,
    TECHNICAL_AGENT,
    Freshness,
)
from upscale.services.opportunity_model.loaders import SafetyRow
from upscale.services.opportunity_model.models import (
    Fact,
    Layer,
    NewsFacts,
    OpportunityCausalityError,
    OpportunityIdentityError,
    OpportunityOrigin,
    SafetyFacts,
    SafetyRule,
    SafetyRuleState,
    SafetySourceRef,
    ScoutFacts,
    ScoutMarketView,
    SocialFacts,
    SourceRef,
    SourceStatus,
    TechnicalFacts,
)
from upscale.services.opportunity_model.ownership import (
    OWNERSHIP,
    SAFETY_RULES,
    SCOUT_FLAGS,
    SCOUT_RETAINED_SIGNALS,
    SCOUT_SCORING_FLAGS,
    SCOUT_SIGNALS,
    fact,
)

CROSS_CONFIRMATION_DROPPED = (
    "dropped from Opportunity V1: its archived score mixes market agreement with a social "
    "term (weighted by an unarchived config share, minus unarchived penalties, clamped), so "
    "the social part can't be separated exactly; its market confirmations repeat facts kept "
    "elsewhere"
)
# Scout archives only the flags it raised, and each flag's check treats a missing input as
# "condition not met" (no per-flag evaluated / not-evaluated state is recorded), so a
# missing flag can't be told apart from a check that couldn't run.
SCOUT_FLAG_ABSENT = (
    "Scout raised no {code} flag; Scout records raised flags only and its check treats "
    "missing inputs as not met, so absence isn't proof the risk is absent"
)
SCOUT_NO_COLLAPSE = (
    "Scout found no MARKET_COLLAPSE evidence; each collapse test is skipped when its inputs "
    "are missing, so market_status OK isn't proof the market is healthy"
)
FLOW_NOTES_PARTIAL = (
    "flow notes list the doubts Scout found; a doubt check whose inputs are missing (e.g. "
    "the 1h price, distinct-wallet counts) is skipped, so the list is a lower bound"
)

# --- identity and time ------------------------------------------------------------------------


def parse_canonical(canonical_id: str) -> tuple[str, str]:
    """(chain, address). Solana mints are case-sensitive base58 and must be exact."""
    chain, sep, address = canonical_id.partition(":")
    if not sep or not chain or not address:
        raise OpportunityIdentityError(f"{canonical_id!r} is not <chain>:<address>")
    if chain == "solana" and not is_solana_address(address):
        raise OpportunityIdentityError(f"{address!r} is not a Solana mint address")
    return chain, address


def supported(chain: str) -> bool:
    return chain in SUPPORTED_CHAINS


def _time(raw: Any, what: str) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise OpportunityCausalityError(f"{what} is not a timestamp: {raw!r}")
    t = datetime.fromisoformat(raw)
    if t.tzinfo is None:
        raise OpportunityCausalityError(f"{what} has no timezone: {raw!r}")
    return t


def _not_after(t: datetime | None, decision_at: datetime, what: str) -> None:
    if t is not None and t > decision_at:
        raise OpportunityCausalityError(
            f"{what} {t.isoformat()} is after decision_at {decision_at.isoformat()}"
        )


def _same_token(cid: str, address: str, chain: Any, mint: Any, own_id: Any, what: str) -> None:
    """Exact identity (Solana is case-sensitive: a case variant is another mint)."""
    if own_id is not None and own_id != cid:
        raise OpportunityIdentityError(f"{what} is for {own_id!r}, not {cid!r}")
    if chain is not None and chain != cid.partition(":")[0]:
        raise OpportunityIdentityError(f"{what} is on chain {chain!r}, not {cid!r}")
    if mint is not None and mint != address:
        raise OpportunityIdentityError(f"{what} names token {mint!r}, not {address!r}")


def _check_record(
    r: EvidenceRecord, cid: str, address: str, decision_at: datetime, origin: OpportunityOrigin
) -> None:
    _same_token(cid, address, r.chain, r.address, r.asset_id, f"{r.kind} record {r.record_id}")
    _not_after(r.observed_at, decision_at, f"{r.kind} record {r.record_id} observed_at")
    if origin == "LIVE_FORWARD" and r.archived_at > decision_at:
        raise OpportunityCausalityError(
            f"{r.kind} record {r.record_id} was archived at {r.archived_at.isoformat()}, "
            f"after the live decision at {decision_at.isoformat()}: it wasn't readable then"
        )


def _fresh(
    observed: datetime, decision_at: datetime, limit_s: float, what: str
) -> tuple[SourceStatus, float, str | None]:
    age = (decision_at - observed).total_seconds()
    if age <= limit_s:
        return "AVAILABLE", age, None
    return "STALE", age, f"{what} is {age:.0f}s old at decision_at (limit {limit_s:.0f}s)"


def missing_ref(status: SourceStatus, reason: str, store: str | None = None) -> SourceRef:
    return SourceRef(status=status, store=store, reason=reason)


def _record_ref(
    r: EvidenceRecord,
    status: SourceStatus,
    age: float,
    reason: str | None,
    observed: datetime | None = None,
) -> SourceRef:
    return SourceRef(
        status=status, store=ARCHIVE_STORE, record_id=r.record_id, fingerprint=r.fingerprint,
        observed_at=observed or r.observed_at, age_seconds=age, reason=reason,
    )  # fmt: skip


def fill_missing(layer: Layer, facts: Iterable[Fact], reason: str) -> tuple[Fact, ...]:
    """The facts, plus a NOT_AVAILABLE fact for every aspect `layer` owns but didn't
    report: missing evidence is explicit, never absent-and-forgotten."""
    out = list(facts)
    have = {f.aspect for f in out if f.role != "DIAGNOSTIC"}
    for aspect, o in OWNERSHIP.items():
        if o.layer == layer and aspect not in have:
            out.append(fact(aspect, layer, "-", None, reason))
    return tuple(sorted(out, key=_order))


def _order(f: Fact) -> tuple[str, str, str, str]:
    return (f.aspect, f.role, f.path, json.dumps(f.value, sort_keys=True))


def _dig(d: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(d, Mapping):
            return None
        d = d.get(key)
    return d


def _pick(d: Any, *keys: str) -> dict[str, Any]:
    return {k: d.get(k) for k in keys} if isinstance(d, Mapping) else {}


# --- Scout ---------------------------------------------------------------------------------------


def scout(
    r: EvidenceRecord | None,
    cid: str,
    address: str,
    decision_at: datetime,
    origin: OpportunityOrigin,
    fresh: Freshness,
) -> tuple[ScoutFacts, dict[str, Any] | None]:
    """Scout's facts, and the archived TechnicalContext (owned by Technical) if any."""
    if r is None:
        why = "no Scout evaluation of this token was archived at or before decision_at"
        return _scout_missing(missing_ref("NOT_COLLECTED", why, ARCHIVE_STORE), why), None
    _check_record(r, cid, address, decision_at, origin)
    c = r.payload.get("candidate")
    timing = r.payload.get("timing")
    if not isinstance(c, dict) or not isinstance(timing, dict):
        why = "archived Scout record has no candidate / timing"
        return _scout_missing(_bad_ref(r, decision_at, why), why), None
    _same_token(cid, address, c.get("chain"), c.get("address"), c.get("canonical_id"),
                f"Scout candidate in {r.record_id}")  # fmt: skip
    market_at = _time(timing.get("market_observed_at") or c.get("observed_at"),
                      "Scout market_observed_at")  # fmt: skip
    decided = _time(timing.get("decision_at"), "Scout decision_at")
    _not_after(market_at, decision_at, "Scout market evidence")
    _not_after(decided, decision_at, "Scout decision_at")
    if market_at is not None and market_at > r.observed_at:
        raise OpportunityCausalityError(
            f"Scout record {r.record_id}: market evidence after its own decision time"
        )
    version = timing.get("version")
    if not isinstance(version, int) or version < MIN_SCOUT_TIMING_VERSION:
        why = f"Scout timing version {version!r} predates causal decision times"
        return _scout_missing(_bad_ref(r, decision_at, why), why), None
    if r.links.get("causal_valid") is not True:
        why = "the archived Scout decision used evidence observed after its own decision time"
        return _scout_missing(_bad_ref(r, decision_at, why), why), None
    if r.availability != "AVAILABLE":
        why = f"Scout record availability {r.availability}: {r.reason or 'no reason'}"
        return _scout_missing(_record_ref(r, "UNAVAILABLE", _age(r, decision_at), why), why), None
    # Age is of the market evidence (older than or equal to the record's decision time).
    evidence_at = market_at or r.observed_at
    status, age, why_stale = _fresh(
        evidence_at, decision_at, fresh.scout_s, "Scout market evidence"
    )
    ref = _record_ref(r, status, age, why_stale)
    market = ScoutMarketView(facts=_scout_market(c))
    context = _scout_context(c)
    diagnostics = _scout_diagnostics(c)
    technical = _dig(c, "momentum", "technical")
    return (
        ScoutFacts(
            ref=ref,
            market_observed_at=market_at,
            market_view=market,
            context=context,
            diagnostics=tuple(sorted(diagnostics, key=_order)),
        ),
        technical if isinstance(technical, dict) else None,
    )


def _age(r: EvidenceRecord, decision_at: datetime) -> float:
    return (decision_at - r.observed_at).total_seconds()


def _bad_ref(r: EvidenceRecord, decision_at: datetime, why: str) -> SourceRef:
    return _record_ref(r, "INCOMPATIBLE", _age(r, decision_at), why)


def _scout_missing(ref: SourceRef, why: str) -> ScoutFacts:
    facts = fill_missing("scout", (), why)
    return ScoutFacts(
        ref=ref,
        market_view=ScoutMarketView(facts=tuple(f for f in facts if f.role == "SCORING")),
        context=tuple(f for f in facts if f.role == "CONTEXT"),
    )


def _families(c: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fams = _dig(c, "scout_momentum", "families")
    return {f["family"]: f for f in fams or [] if isinstance(f, dict) and "family" in f}


_SIGNAL_MEASURES: dict[str, tuple[str, ...]] = {
    "flow.volume_acceleration": ("volume_acceleration", "volume_acceleration_basis"),
    "flow.trade_acceleration": ("txn_acceleration", "txn_acceleration_basis"),
    "flow.buy_pressure": ("buy_share", "buyer_share", "buy_pressure_change"),
    "liquidity.growth": ("liquidity_change_pct", "liquidity_change_basis"),
    "earliness.move_already_made": ("move_since_first_seen_pct",),
}


def _scout_market(c: Mapping[str, Any]) -> tuple[Fact, ...]:
    families = _families(c)
    momentum = c.get("momentum") or {}
    out: list[Fact] = []
    for family, name in SCOUT_RETAINED_SIGNALS:
        aspect = SCOUT_SIGNALS[(family, name)]
        path = f"candidate.scout_momentum.families[{family}].signals[{name}]"
        fam = families.get(family)
        sig = next(
            (s for s in (fam or {}).get("signals") or [] if isinstance(s, dict)
             and s.get("name") == name),
            None,
        )  # fmt: skip
        if sig is None:
            why = (
                f"Scout reported no {family} family"
                if fam is None
                else f"Scout's {family} family had no evidence (its stand-in isn't evidence)"
                if not fam.get("available", False)
                else f"Scout reported no trusted {name} sub-signal"
            )
            out.append(fact(aspect, "scout", path, None, why))
            continue
        value = _pick(sig, "score", "weight", "raw", "detail")
        measures = _pick(momentum, *_SIGNAL_MEASURES.get(aspect, ()))
        if measures:
            value["measures"] = measures
        out.append(fact(aspect, "scout", path, value))
    q = c.get("quality") or {}
    flow = q.get("flow_quality")
    if flow in (None, "unknown"):
        out.append(fact("flow.quality", "scout", "candidate.quality.flow_quality", None,
                        "Scout had no trade counts to judge flow quality"))  # fmt: skip
    else:
        out.append(fact("flow.quality", "scout", "candidate.quality.flow_quality",
                        {"flow_quality": flow, "notes": list(q.get("flow_notes") or []),
                         "notes_complete": False, "note": FLOW_NOTES_PARTIAL}))  # fmt: skip
    flags = [f for f in c.get("risk_flags") or [] if isinstance(f, dict)]
    for code in SCOUT_SCORING_FLAGS:
        aspect = SCOUT_FLAGS[code]
        mine = [f for f in flags if f.get("code") == code]
        path = f"candidate.risk_flags[{code}]"
        if mine:
            value = {"flagged": True, "severity": mine[0].get("severity"),
                     "details": [f.get("detail") for f in mine]}  # fmt: skip
            out.append(fact(aspect, "scout", path, value))
        else:
            out.append(fact(aspect, "scout", path, None, SCOUT_FLAG_ABSENT.format(code=code)))
    return tuple(sorted(out, key=_order))


def _scout_context(c: Mapping[str, Any]) -> tuple[Fact, ...]:
    q = c.get("quality") or {}
    return (
        fact("scout.data_status", "scout", "candidate.data_status",
             _pick(c, "data_status", "snapshot_age_minutes")),
        fact("scout.eligibility", "scout", "candidate.eligible",
             _pick(c, "eligible", "ineligible_reasons", "rank")),
        fact("scout.market_collapse", "scout", "candidate.quality.market_status",
             _pick(q, "market_status", "collapse_evidence")
             if q.get("market_status") == "MARKET_COLLAPSE" else None, SCOUT_NO_COLLAPSE),
        fact("scout.stage", "scout", "candidate.stage",
             _pick(c, "stage", "stage_reasons", "unconfirmed_stage")),
    )  # fmt: skip


def _diag(aspect: str, path: str, value: Any) -> Fact:
    if value is None or value == {}:
        return fact(aspect, "scout", path, None, "not reported by Scout", diagnostic=True)
    return fact(aspect, "scout", path, value, diagnostic=True)


def _scout_diagnostics(c: Mapping[str, Any]) -> list[Fact]:
    sm = c.get("scout_momentum") or {}
    q = c.get("quality") or {}
    m = c.get("momentum") or {}
    mk = c.get("market") or {}
    families = _families(c)
    out = [
        _diag("scout.composite", "candidate.scout_momentum", {
            **_pick(sm, "score", "base", "stage_adjustment", "risk_penalty"),
            "families": [_pick(f, "family", "score", "weight", "contribution", "available")
                         for f in families.values()],
            "note": "Scout's composite and family scores: diagnostics only, never inputs",
        }),
        _diag("authority.mint", "candidate.quality.mint_authority_active",
              _pick(q, "mint_authority_active")),
        _diag("authority.freeze", "candidate.quality.freeze_authority_active",
              _pick(q, "freeze_authority_active")),
        _diag("holders.top1", "candidate.quality.holder_top1_pct",
              _pick(q, "holder_top1_pct", "holder_data_lower_bound")),
        _diag("holders.top10", "candidate.quality.holder_top10_pct",
              _pick(q, "holder_top10_pct", "holder_data_lower_bound")),
        _diag("safety.coverage", "candidate.quality.safety_status",
              _pick(q, "safety_status", "safety_missing", "verification")),
        _diag("identity.token_mint", "candidate.quality.identity_status",
              _pick(q, "identity_status")),
        _diag("liquidity.level", "candidate.quality.liquidity_quality",
              {**_pick(q, "liquidity_quality"), **_pick(mk, "liquidity_usd")}),
        _diag("market.pool_age", "candidate.market.pool_age_hours",
              _pick(mk, "pool_age_hours", "oldest_pool_age_hours")),
        _diag("attention.momentum", "candidate.momentum.social_status",
              _pick(m, "social_status", "social_state", "social_reason", "mention_acceleration",
                    "unique_author_acceleration", "engagement_acceleration")),
        _diag("attention.cross_platform", "candidate.momentum.cross_platform_corroborated",
              _pick(m, "cross_platform_corroborated", "platforms_active")),
        _diag("attention.attribution", "candidate.quality.social_attribution",
              _pick(q, "social_attribution", "exact_mention_share")),
        _diag("attention.spam", "candidate.quality.spam_risk",
              _pick(q, "spam_risk", "organic_signal")),
    ]  # fmt: skip
    for family, fam in sorted(families.items()):
        if family == "cross_confirmation":
            out.append(_diag("scout.cross_confirmation",
                             "candidate.scout_momentum.families[cross_confirmation]",
                             {"family": fam, "dropped": CROSS_CONFIRMATION_DROPPED}))  # fmt: skip
            continue
        for s in fam.get("signals") or []:
            name = s.get("name") if isinstance(s, dict) else None
            if (family, name) in SCOUT_RETAINED_SIGNALS:
                continue
            aspect = SCOUT_SIGNALS.get((family, str(name)), "scout.unrouted")
            path = f"candidate.scout_momentum.families[{family}].signals[{name}]"
            out.append(_diag(aspect, path, _pick(s, "name", "score", "weight", "raw", "detail")))
    for f in c.get("risk_flags") or []:
        code = f.get("code") if isinstance(f, dict) else None
        if code in SCOUT_SCORING_FLAGS:
            continue
        aspect = SCOUT_FLAGS.get(str(code), "scout.unrouted")
        out.append(_diag(aspect, f"candidate.risk_flags[{code}]",
                         _pick(f, "code", "severity", "detail", "penalty")))  # fmt: skip
    return out


# --- Technical -----------------------------------------------------------------------------------


def technical(
    scout_ref: SourceRef,
    context: dict[str, Any] | None,
    analyze: EvidenceRecord | None,
    analyze_missing: SourceRef | None,
    cid: str,
    address: str,
    decision_at: datetime,
    origin: OpportunityOrigin,
    fresh: Freshness,
) -> TechnicalFacts:
    facts: list[Fact] = []
    if scout_ref.status in ("AVAILABLE", "STALE") and context is not None:
        ref = scout_ref
        path = "candidate.momentum.technical"
        facts += [
            fact("price.snapshot_trend", "technical", path,
                 _pick(context, "trend", "snapshots", "span_hours", "change_pct")),
            fact("price.breakout", "technical", f"{path}.breakout", _pick(context, "breakout")),
            fact("price.higher_lows", "technical", f"{path}.higher_lows",
                 _pick(context, "higher_lows")),
        ]  # fmt: skip
        vc = context.get("volume_confirmed")
        facts.append(
            fact(
                "price.volume_confirmed",
                "technical",
                f"{path}.volume_confirmed",
                None if vc is None else {"volume_confirmed": vc},
                "Scout couldn't tell whether trading activity confirmed the move",
            )  # fmt: skip
        )
    elif scout_ref.status in ("AVAILABLE", "STALE"):
        ref = scout_ref.model_copy(
            update={"status": "NOT_COLLECTED",
                    "reason": "Scout had too few stored snapshots for a TechnicalContext"}
        )  # fmt: skip
    else:
        ref = missing_ref(scout_ref.status, f"no Scout record: {scout_ref.reason}", ARCHIVE_STORE)
    analyze_ref, analyze_facts = _analyze_technical(
        analyze, analyze_missing, cid, address, decision_at, origin, fresh
    )
    facts += analyze_facts
    return TechnicalFacts(
        ref=ref,
        analyze_ref=analyze_ref,
        facts=fill_missing("technical", facts, "no technical evidence of this kind was read"),
    )


def _analyze_ref(
    r: EvidenceRecord, decision_at: datetime, fresh: Freshness
) -> tuple[SourceRef, bool]:
    status, age, why = _fresh(r.observed_at, decision_at, fresh.analyze_s, "Analyze decision")
    return _record_ref(r, status, age, why), status in ("AVAILABLE", "STALE")


def _agent(r: EvidenceRecord, name: str) -> tuple[dict[str, Any] | None, str | None]:
    a = _dig(r.payload, "agents", name)
    if not isinstance(a, dict):
        return None, f"the archived Analyze decision has no {name} result"
    if a.get("mock"):
        return None, f"the archived {name} result is a mock"
    if a.get("status") != "ok":
        return None, f"the archived {name} result failed: {a.get('error') or a.get('status')}"
    findings = a.get("findings")
    if not isinstance(findings, dict):
        return None, f"the archived {name} result has no findings"
    return findings, None


def _analyze_technical(
    r: EvidenceRecord | None,
    missing: SourceRef | None,
    cid: str,
    address: str,
    decision_at: datetime,
    origin: OpportunityOrigin,
    fresh: Freshness,
) -> tuple[SourceRef, list[Fact]]:
    path = f"agents.{TECHNICAL_AGENT}.findings.trend"
    if r is None:
        ref = missing or missing_ref(
            "NOT_COLLECTED", "no Analyze decision was archived at or before decision_at",
            ARCHIVE_STORE,
        )  # fmt: skip
        return ref, [fact("price.analyze_trend", "technical", path, None, str(ref.reason))]
    _check_record(r, cid, address, decision_at, origin)
    ref, _ = _analyze_ref(r, decision_at, fresh)
    findings, why = _agent(r, TECHNICAL_AGENT)
    if findings is None:
        ref = ref.model_copy(update={"status": "UNAVAILABLE", "reason": why})
        return ref, [fact("price.analyze_trend", "technical", path, None, str(why))]
    _same_token(cid, address, None, None, findings.get("canonical_id"),
                f"Analyze technical findings in {r.record_id}")  # fmt: skip
    _not_after(_time(findings.get("as_of"), "Analyze technical as_of"), decision_at,
               "Analyze technical as_of")  # fmt: skip
    _not_after(_time(findings.get("last_candle_at"), "Analyze last candle"), decision_at,
               "Analyze last candle")  # fmt: skip
    trend = findings.get("trend") or {}
    if not trend.get("available") or trend.get("label") is None:
        reason = trend.get("unavailable_reason") or "Analyze reported no trend"
        return ref, [fact("price.analyze_trend", "technical", path, None, str(reason))]
    value = {
        **_pick(trend, "label", "method", "reasons"),
        **_pick(findings, "timeframe", "candle_count", "last_candle_at", "provider"),
        "note": "Analyze's final action, confidence and overall risk are not inputs",
    }
    return ref, [fact("price.analyze_trend", "technical", path, value)]


# --- News ------------------------------------------------------------------------------------------


def news(
    r: EvidenceRecord | None,
    missing: SourceRef | None,
    cid: str,
    address: str,
    decision_at: datetime,
    origin: OpportunityOrigin,
    fresh: Freshness,
) -> NewsFacts:
    if r is None:
        ref = missing or missing_ref(
            "NOT_COLLECTED", "no Analyze decision was archived at or before decision_at",
            ARCHIVE_STORE,
        )  # fmt: skip
        return NewsFacts(ref=ref, facts=fill_missing("news", (), str(ref.reason)))
    _check_record(r, cid, address, decision_at, origin)
    ref, _ = _analyze_ref(r, decision_at, fresh)
    findings, why = _agent(r, NEWS_AGENT)
    if findings is None:
        ref = ref.model_copy(update={"status": "UNAVAILABLE", "reason": why})
        return NewsFacts(ref=ref, facts=fill_missing("news", (), str(why)))
    reports = [x for x in findings.get("reports") or [] if isinstance(x, dict)]
    summaries, impacts, models = [], [], set()
    for rep in reports:
        for key in ("retrieved_at", "analyzed_at"):
            _not_after(_time(rep.get(key), f"news {key}"), decision_at, f"news {key}")
        if rep.get("sentiment_model"):
            models.add(str(rep["sentiment_model"]))
        summaries.append(_pick(rep, "asset", "subject", "overall_news_sentiment",
                               "market_news_sentiment", "sentiment_counts", "asset_story_count",
                               "stale_story_count", "sentiment_model", "provider"))  # fmt: skip
        for s in rep.get("stories") or []:
            if not isinstance(s, dict):
                continue
            _not_after(_time(s.get("published_at"), "story published_at"), decision_at,
                       "news story published_at")  # fmt: skip
            if s.get("impact") in ("high", "medium"):
                impacts.append(_pick(s, "source", "published_at", "sentiment", "impact",
                                     "scope", "stale"))  # fmt: skip
    path = f"agents.{NEWS_AGENT}.findings.reports"
    facts = [
        fact("news.sentiment", "news", path, {"reports": summaries} if summaries else None,
             "the news result had no reports"),
        fact("news.impact", "news", f"{path}[].stories",
             {"stories": sorted(impacts, key=lambda s: json.dumps(s, sort_keys=True))}
             if summaries else None, "the news result had no reports"),
    ]  # fmt: skip
    return NewsFacts(
        ref=ref,
        llm_labelled=bool(models) if reports else None,
        sentiment_models=tuple(sorted(models)),
        facts=fill_missing("news", facts, "-"),
    )


# --- Social ------------------------------------------------------------------------------------


def social(
    r: EvidenceRecord | None,
    cid: str,
    address: str,
    decision_at: datetime,
    origin: OpportunityOrigin,
    fresh: Freshness,
) -> SocialFacts:
    if r is None:
        why = "no social evidence for this token was archived at or before decision_at"
        return SocialFacts(ref=missing_ref("NOT_COLLECTED", why, ARCHIVE_STORE),
                           facts=fill_missing("social", (), why))  # fmt: skip
    _check_record(r, cid, address, decision_at, origin)
    m = r.payload.get("momentum")
    if not isinstance(m, dict):
        why = "archived social record has no momentum"
        return SocialFacts(ref=_record_ref(r, "INCOMPATIBLE", _age(r, decision_at), why),
                           facts=fill_missing("social", (), why))  # fmt: skip
    _same_token(cid, address, None, None, m.get("canonical_id"), f"social record {r.record_id}")
    for s in m.get("sources") or []:
        if isinstance(s, dict):
            _same_token(cid, address, None, None, s.get("canonical_id"),
                        f"social source in {r.record_id}")  # fmt: skip
            _not_after(_time(s.get("observed_at"), "social source observed_at"), decision_at,
                       "social source observed_at")  # fmt: skip
    _not_after(_time(m.get("computed_at"), "social computed_at"), decision_at,
               "social computed_at")  # fmt: skip
    providers = fact("attention.providers", "social", "providers", {
        "providers": r.payload.get("providers") or {},
        "sources": sorted((_pick(s, "provider", "platform", "status", "error")
                           for s in m.get("sources") or [] if isinstance(s, dict)),
                          key=lambda s: json.dumps(s, sort_keys=True)),
    })  # fmt: skip
    if r.availability != "AVAILABLE" or m.get("state") == "UNAVAILABLE":
        why = f"social unavailable: {r.reason or '; '.join(m.get('reasons') or []) or 'no reason'}"
        ref = _record_ref(r, "UNAVAILABLE", _age(r, decision_at), why)
        return SocialFacts(ref=ref, facts=fill_missing("social", [providers], why))
    status, age, why_stale = _fresh(r.observed_at, decision_at, fresh.social_s, "social evidence")
    window = m.get("window")
    stats = next((w for w in m.get("windows") or [] if isinstance(w, dict)
                  and w.get("window") == window), None)  # fmt: skip
    trend = m.get("trend") or {}
    facts = [
        providers,
        fact("attention.momentum", "social", "momentum.state", {
            **_pick(m, "state", "window", "reasons"),
            "acceleration_ratio": {k: _dig(trend, k, "acceleration_ratio")
                                   for k in ("mentions", "unique_authors", "engagement")},
        }),
        fact("attention.attribution", "social", f"momentum.windows[{window}]",
             _pick(stats, "exact_mentions", "attributable_posts", "ambiguous_posts",
                   "promoted_posts", "sources") if stats else None,
             f"no fully covered {window} window"),
        fact("attention.spam", "social", "momentum.quality",
             _pick(m.get("quality"), "spam_risk", "organic_signal_strength", "reasons",
                   "top_author_share", "duplicate_share", "promoted_share",
                   "conflicting_contracts") or None, "no social quality reported"),
        fact("attention.market_cross", "social", "momentum.market",
             _pick(m.get("market"), "state", "reasons", "market_window") or None,
             "no market cross-check reported"),
        fact("attention.cross_platform", "social", "momentum.cross_platform",
             _pick(m.get("cross_platform"), "providers_configured", "providers_checked",
                   "platforms_with_activity", "platforms_accelerating", "corroborated",
                   "single_source_available", "only_one_active_of_several") or None,
             "no cross-platform confirmation reported"),
    ]  # fmt: skip
    return SocialFacts(ref=_record_ref(r, status, age, why_stale),
                       facts=fill_missing("social", facts, "-"))  # fmt: skip


# --- Safety V2 -----------------------------------------------------------------------------------

_SECTIONS = ("identity", "authority", "holders", "market", "creator", "changes", "coverage",
             "assessment")  # fmt: skip


def safety_missing(status: SourceStatus, reason: str) -> SafetyFacts:
    ref = SafetySourceRef(status=status, store=SAFETY_STORE, reason=reason)
    return SafetyFacts(ref=ref, facts=fill_missing("safety", (), reason))


def safety(
    row: SafetyRow | None, cid: str, address: str, decision_at: datetime, fresh: Freshness
) -> SafetyFacts:
    """A Safety V2 snapshot with ``as_of <= decision_at`` (HISTORICAL_REPLAY may use one
    materialized later: its inputs were all known by ``as_of``)."""
    if row is None:
        return safety_missing(
            "NOT_COLLECTED", "no Safety V2 snapshot of this token has as_of <= decision_at"
        )
    if row.canonical_id != cid:
        raise OpportunityIdentityError(f"Safety snapshot {row.id} is for {row.canonical_id!r}")
    _not_after(row.as_of, decision_at, f"Safety snapshot {row.id} as_of")
    status, age, why = _fresh(row.as_of, decision_at, fresh.safety_s, "Safety V2 snapshot")

    def ref(s: SourceStatus, reason: str | None) -> SafetySourceRef:
        return SafetySourceRef(
            status=s, store=SAFETY_STORE, record_id=f"safety_snapshots:{row.id}",
            fingerprint=row.body_hash, observed_at=row.as_of, age_seconds=age, reason=reason,
            snapshot_id=row.id, as_of=row.as_of, rules_version=row.rules_version,
            body_hash=row.body_hash,
        )  # fmt: skip

    def refuse(s: SourceStatus, reason: str) -> SafetyFacts:
        return SafetyFacts(ref=ref(s, reason), facts=fill_missing("safety", (), reason))

    if row.rules_version != SAFETY_RULES_VERSION:
        return refuse("INCOMPATIBLE", f"Safety rules version {row.rules_version!r}; V1 "
                      f"requires {SAFETY_RULES_VERSION!r} (older versions aren't partial)")  # fmt: skip
    text = zlib.decompress(row.body_zlib).decode()
    if hashlib.sha256(text.encode()).hexdigest() != row.body_hash:
        return refuse("UNAVAILABLE", f"Safety snapshot {row.id} body fails its hash check")
    body = json.loads(text)
    if (
        body.get("schema_version") != SAFETY_SNAPSHOT_SCHEMA
        or body.get("rules_version") != row.rules_version
    ):
        return refuse("INCOMPATIBLE", f"Safety body schema {body.get('schema_version')!r} / "
                      f"rules {body.get('rules_version')!r} isn't {SAFETY_SNAPSHOT_SCHEMA}")  # fmt: skip
    identity = body.get("identity") or {}
    _same_token(cid, address, identity.get("chain"), identity.get("mint"),
                identity.get("canonical_id"), f"Safety snapshot {row.id}")  # fmt: skip
    as_of = _time(body.get("as_of"), "Safety body as_of")
    if as_of != row.as_of:
        return refuse("INCOMPATIBLE", f"Safety snapshot {row.id} body as_of disagrees with row")
    rules, problems = _safety_rules(body)
    if problems:
        return refuse("INCOMPATIBLE", "contradictory or unknown Safety rule metadata: "
                      + "; ".join(problems))  # fmt: skip
    facts = [
        fact(SAFETY_RULES[r.id], "safety", f"rules[{r.id}]",
             {"rule": r.id, **r.model_dump(mode="json", exclude={"id", "evidence"})})
        for r in rules
    ]  # fmt: skip
    cov = body.get("coverage") or {}
    facts += [
        fact("identity.token_mint", "safety", "identity.token_mint",
             {**_pick(identity.get("token_mint"), "status", "reason"),
              "pool_match": identity.get("pool_match")}),
        fact("safety.coverage", "safety", "coverage",
             {**_pick(cov, "coverage", "reasons", "components"),
              "band": _dig(body, "assessment", "band")}),
    ]  # fmt: skip
    return SafetyFacts(
        ref=ref(status, why),
        snapshot_schema=body.get("schema_version"),
        rules=rules,
        facts=fill_missing("safety", facts, "-"),
        **{k: body.get(k) for k in _SECTIONS},
    )


_OUTCOMES: dict[str, SafetyRuleState] = {
    "TRIGGERED": "TRIGGERED", "NOT_TRIGGERED": "NOT_TRIGGERED", "UNDETERMINED": "UNDETERMINED",
}  # fmt: skip


def _safety_rules(body: Mapping[str, Any]) -> tuple[tuple[SafetyRule, ...], list[str]]:
    """Each rule exactly as the body reports it: evaluated (``flags``), ``out_of_scope`` or
    ``not_supported``. A rule listed more than once, or in two of those, is contradictory
    metadata: reported as a problem (the snapshot is INCOMPATIBLE), never reconciled."""
    rules: list[SafetyRule] = []
    problems: list[str] = []
    for f in body.get("flags") or []:
        if f.get("outcome") not in _OUTCOMES:
            problems.append(f"{f.get('id')} has outcome {f.get('outcome')!r}")
            continue
        rules.append(SafetyRule(
            id=f["id"], state=_OUTCOMES[f["outcome"]], severity=f.get("severity"),
            reason=f.get("reason"),
            needs=next((u.get("needs") for u in body.get("undetermined") or []
                        if u.get("id") == f["id"]), None),
            evidence=tuple(f.get("evidence") or ()),
        ))  # fmt: skip
    cov = body.get("coverage") or {}
    listed: tuple[tuple[SafetyRuleState, str], ...] = (
        ("OUT_OF_SCOPE", "out_of_scope"),
        ("NOT_SUPPORTED", "not_supported"),
    )
    for state, key in listed:
        for x in cov.get(key) or []:
            rules.append(SafetyRule(id=x["id"], state=state, reason=x.get("reason")))
    seen: dict[str, list[str]] = {}
    for r in rules:
        seen.setdefault(r.id, []).append(r.state)
    problems += [f"{i} is reported as {' and '.join(st)}" for i, st in sorted(seen.items())
                 if len(st) > 1]  # fmt: skip
    problems += [f"unknown rule {i}" for i in sorted(seen) if i not in SAFETY_RULES]
    return tuple(sorted(rules, key=lambda r: r.id)), problems
