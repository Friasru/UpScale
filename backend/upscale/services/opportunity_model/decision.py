"""Opportunity Model V1 decision engine (OPPORTUNITY_RULES_VERSION "1"): one canonical
`OpportunityInput` -> one SKIP / WATCH / ENTER `OpportunityDecision`.

Pure: no I/O, no clock, no database, no provider. Every input is already in the
`OpportunityInput`; only facts with a scoring owner are read (diagnostic copies never are),
so one fact can't count twice. Fixed precedence:

1. integrity (an inconsistent input raises `OpportunityDecisionError`: never a decision);
2. hard vetoes (Safety-owned, plus Scout's *positive* MARKET_COLLAPSE evidence) -> SKIP;
3. evidence quality; 4. Scout setup band; 5. blockers and unresolved veto / blocker rules;
6. Technical and Social bands; 7. risk groups; 8. positive groups; 9. the decision table.

ENTER = every ENTER gate passes; WATCH = no veto and a setup, but at least one gate fails
(each failed gate is one `upgrade_path` item); SKIP = a hard veto, no setup, or a required
Safety domain (IDENTITY / authorities, HOLDERS, MARKET) never assessed (`safety_readiness`:
Safety coverage COMPLETE only means every *applicable* rule resolved; a domain Safety never
collected is out of scope there but not entry-ready here). That SKIP means "not
entry-eligible with current evidence", never "proven malicious" (``skip_basis``).

Monotonicity contract. (A) Core degradations never upgrade: Safety NOT_TRIGGERED ->
UNDETERMINED -> TRIGGERED on the same evidence; a required domain becoming PARTIAL /
unassessed or the snapshot becoming stale / unavailable; fresh Scout -> stale /
unavailable; Technical -> unavailable; weaker / missing FLOW evidence; an added or escalated
veto, blocker or risk group. (LOW_LIQUIDITY / VERY_NEW_POOL read exact provider fields:
their UNDETERMINED -> TRIGGERED medium can only happen when the exact value appears, i.e.
new evidence.) (B) Social and News are optional: absent, they add zero positive evidence,
are listed as missing and don't lower quality; an absent observation doesn't disprove a
negative one another observation had, and no monotonic relation between those two
different evidence states is claimed. Missing evidence
never becomes a favorable value: Scout flag / collapse absences are NOT_AVAILABLE and are
never read as clean, and a Safety rule that is UNDETERMINED (or absent from a read
snapshot) and would veto or block when TRIGGERED blocks ENTER.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from upscale.services.chains import is_solana_address
from upscale.services.opportunity_model.config import (
    DECISION_CONFIG,
    DECISION_SCHEMA,
    FRESHNESS,
    INPUT_SCHEMA,
    OPPORTUNITY_RULES_VERSION,
    SAFETY_RULES_VERSION,
    DecisionConfig,
)
from upscale.services.opportunity_model.decision_models import (
    SETUP_ORDER,
    Decision,
    OpportunityDecision,
    OpportunityDecisionError,
    Quality,
    Reason,
    RiskTier,
    SetupBand,
    SkipBasis,
    SocialBand,
    TechnicalBand,
    UpgradeItem,
)
from upscale.services.opportunity_model.models import (
    CorrelationGroup,
    Fact,
    Layer,
    OpportunityInput,
    OwnershipError,
    SourceRef,
)
from upscale.services.opportunity_model.ownership import OWNERSHIP, SAFETY_RULES, check

# Safety rules that veto (SKIP) when TRIGGERED; LOW_LIQUIDITY only at high severity.
VETO_RULES = (
    "IDENTITY_MISMATCH", "NOT_A_TOKEN_MINT", "UNEXPECTED_TOKEN_PROGRAM",
    "MALFORMED_MINT_ACCOUNT", "FREEZE_AUTHORITY_ACTIVE", "MARKET_CLOSED_ON_CHAIN",
    "NO_ELIGIBLE_MARKET", "LIQUIDITY_COLLAPSE",
)  # fmt: skip
HIGH_VETO_RULES = ("LOW_LIQUIDITY",)
# Safety rules that cap at WATCH when TRIGGERED; the HIGH ones only at high severity.
BLOCKER_RULES = ("MINT_AUTHORITY_ACTIVE", "PRIMARY_MARKET_UNCLEAR", "MARKET_NOT_REPORTED")
HIGH_BLOCKER_RULES = (
    "TOP1_CONCENTRATION", "TOP10_CONCENTRATION", "VERIFIED_DEPLOYER_HOLDS_SUPPLY",
    "VERY_NEW_POOL",
)  # fmt: skip
# Severity-gated rules whose Safety evidence can be a lower bound (holder shares, deployer
# holding from a partial scan). LOW_LIQUIDITY and VERY_NEW_POOL read provider-reported
# liquidity / pool age, which Safety only ever reports AVAILABLE (exact) or missing (the rule
# is then UNDETERMINED), so a lower-bound check would never apply to them.
LOWER_BOUND_RULES = ("TOP1_CONCENTRATION", "TOP10_CONCENTRATION", "VERIFIED_DEPLOYER_HOLDS_SUPPLY")
CRITICAL_RULES = (*VETO_RULES, *HIGH_VETO_RULES, *BLOCKER_RULES, *HIGH_BLOCKER_RULES)
_SECTIONS = ("identity", "authority", "holders", "market", "creator", "changes", "coverage")
RULE_STATES = ("TRIGGERED", "NOT_TRIGGERED", "UNDETERMINED", "OUT_OF_SCOPE", "NOT_SUPPORTED")
# Scout-owned facts a decision-bearing setup needs; one missing makes quality PARTIAL.
CORE_FLOW = ("flow.buy_pressure", "flow.quality", "flow.trade_acceleration",
             "flow.volume_acceleration")  # fmt: skip
SCOUT_FLOW_RISKS = {
    "flow.thin_pump": "SCOUT_THIN_MARKET_PUMP",
    "flow.distribution": "SCOUT_DISTRIBUTION",
    "flow.volume_to_liquidity": "SCOUT_HIGH_VOLUME_TO_LIQUIDITY",
}
NO_SETUP_STAGES = ("FADING", "INSUFFICIENT_DATA")
WEAK_STAGES = ("NEW", "EARLY", "ACCELERATING", "STEADY", "CROWDED")
MODERATE_STAGES = ("EARLY", "ACCELERATING")
SUPPORTING_SOCIAL = ("EMERGING", "ACCELERATING", "STRONG")
CONTRADICTING_CROSS = ("UNCONFIRMED_SOCIAL_SPIKE", "CAUTION_LIQUIDITY_FALLING")
_LIMITS = {"scout": FRESHNESS.scout_s, "safety": FRESHNESS.safety_s,
           "social": FRESHNESS.social_s, "analyze": FRESHNESS.analyze_s,
           "news": FRESHNESS.analyze_s}  # fmt: skip


# --- evidence access --------------------------------------------------------------------------


class _Ev:
    """Owned (non-diagnostic) facts by aspect, and source references."""

    def __init__(self, inp: OpportunityInput):
        self.inp = inp
        self.owned: dict[str, Fact] = {f.aspect: f for f in inp.facts() if f.role != "DIAGNOSTIC"}
        s = inp.sources
        self.safety_read = s.safety.ref.status in ("AVAILABLE", "STALE")

    def fact(self, aspect: str) -> Fact | None:
        f = self.owned.get(aspect)
        return f if f is not None and f.status == "AVAILABLE" else None

    def value(self, aspect: str) -> dict[str, Any]:
        f = self.fact(aspect)
        return f.value if f is not None and isinstance(f.value, dict) else {}

    def raw(self, aspect: str) -> float | None:
        v = self.value(aspect).get("raw")
        return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None

    def rule(self, rule_id: str) -> tuple[str, str | None, Fact | None]:
        """(state, severity, fact); MISSING when the snapshot didn't report it."""
        f = self.fact(SAFETY_RULES[rule_id])
        if f is None:
            return "MISSING", None, self.owned.get(SAFETY_RULES[rule_id])
        v = f.value if isinstance(f.value, dict) else {}
        sev = v.get("severity")
        return str(v.get("state")), sev if isinstance(sev, str) else None, f

    def exact(self, rule_id: str) -> bool:
        """Every evidence field the rule read is AVAILABLE (exact): its severity isn't a
        lower bound. A PARTIAL field (or one that can't be found) could hide a higher
        severity."""
        safety = self.inp.sources.safety
        r = next((x for x in safety.rules if x.id == rule_id), None)
        if r is None or not r.evidence:
            return False
        for path in r.evidence:
            section, _, name = path.partition(".")
            data = getattr(safety, section, None) if section in _SECTIONS else None
            field_ = data.get(name) if isinstance(data, dict) else None
            if not isinstance(field_, dict) or field_.get("status") != "AVAILABLE":
                return False
        return True

    def ref(self, f: Fact) -> SourceRef:
        s = self.inp.sources
        if f.layer == "technical":
            return s.technical.analyze_ref if f.path.startswith("agents.") else s.technical.ref
        return {"scout": s.scout.ref, "social": s.social.ref, "news": s.news.ref,
                "safety": s.safety.ref}[f.layer]  # fmt: skip

    def reason(self, code: str, f: Fact, status: str, message: str) -> Reason:
        return Reason(
            code=code, group=f.group, source=f.layer, source_record_id=self.ref(f).record_id,
            evidence_paths=(f.path,), status=status, message=message,
        )  # fmt: skip


def _engine(code: str, group: CorrelationGroup | None, status: str, message: str,
            source: Layer | None = None, record: str | None = None,
            paths: tuple[str, ...] = ()) -> Reason:  # fmt: skip
    return Reason(code=code, group=group, source=source or "engine", source_record_id=record,
                  evidence_paths=paths, status=status, message=message)  # fmt: skip


def _sorted(reasons: Iterable[Reason]) -> tuple[Reason, ...]:
    unique = {r.model_dump_json(): r for r in reasons}
    return tuple(sorted(unique.values(), key=lambda r: (
        r.code, r.source_record_id or "", r.evidence_paths, r.status, r.message)))  # fmt: skip


# --- 1. integrity -------------------------------------------------------------------------------


def check_integrity(inp: OpportunityInput) -> None:
    def fail(why: str) -> None:
        raise OpportunityDecisionError(why)

    if inp.schema_version != INPUT_SCHEMA:
        fail(f"unsupported input schema {inp.schema_version!r}")
    at = inp.decision_at
    if at.tzinfo is None or at.utcoffset() != timedelta(0):
        fail("decision_at must be timezone-aware UTC")
    if inp.origin not in ("LIVE_FORWARD", "HISTORICAL_REPLAY"):
        fail(f"unknown origin {inp.origin!r}")
    if not inp.chain or not inp.address or inp.canonical_id != f"{inp.chain}:{inp.address}":
        fail(f"inconsistent identity {inp.canonical_id!r} / {inp.chain}:{inp.address}")
    if inp.chain == "solana" and not is_solana_address(inp.address):
        fail(f"{inp.address!r} is not a Solana mint")
    try:
        check(inp.facts())
    except OwnershipError as exc:
        fail(f"ownership violation: {exc}")
    for f in inp.facts():
        if f.role == "DIAGNOSTIC":
            continue
        o = OWNERSHIP[f.aspect]
        if (f.group, f.layer, f.role) != (o.group, o.layer, o.role):
            fail(f"{f.aspect}: impossible owner / group {f.layer} / {f.group}")
    s = inp.sources
    refs: list[tuple[str, SourceRef]] = [
        ("scout", s.scout.ref), ("scout", s.technical.ref), ("analyze", s.technical.analyze_ref),
        ("social", s.social.ref), ("news", s.news.ref), ("safety", s.safety.ref),
    ]  # fmt: skip
    for kind, ref in refs:
        if ref.observed_at is not None and ref.observed_at > at:
            fail(f"a {kind} source observed {ref.observed_at.isoformat()} reached O2 after "
                 f"decision_at {at.isoformat()}")  # fmt: skip
        if ref.age_seconds is not None and ref.age_seconds < 0:
            fail(f"a {kind} source has a negative age")
        if ref.status == "AVAILABLE" and (ref.age_seconds or 0) > _LIMITS[kind]:
            fail(f"a {kind} source is AVAILABLE beyond its freshness limit")
    if s.safety.ref.as_of is not None and s.safety.ref.as_of > at:
        fail("the Safety snapshot as_of is after decision_at")
    if s.scout.market_observed_at is not None and s.scout.market_observed_at > at:
        fail("Scout market evidence is after decision_at")
    if s.safety.ref.status in ("AVAILABLE", "STALE") and (
        s.safety.ref.rules_version != SAFETY_RULES_VERSION
    ):
        fail(f"a usable Safety snapshot must be rules version {SAFETY_RULES_VERSION}")
    for rule_id, aspect in SAFETY_RULES.items():
        rf = next((x for x in s.safety.facts if x.aspect == aspect and x.role != "DIAGNOSTIC"),
                  None)  # fmt: skip
        if rf is None or rf.status != "AVAILABLE":
            continue
        v = rf.value if isinstance(rf.value, dict) else {}
        if v.get("rule") != rule_id or v.get("state") not in RULE_STATES:
            fail(f"{aspect} doesn't hold a valid {rule_id} result")


# --- 2. vetoes, 5. blockers -------------------------------------------------------------------------


def _rule_message(rule_id: str, state: str, severity: str | None) -> str:
    return f"Safety V2 {rule_id} is {state}" + (f" ({severity})" if severity else "")


def vetoes(ev: _Ev) -> list[Reason]:
    out = []
    for rule_id in (*VETO_RULES, *HIGH_VETO_RULES):
        state, sev, f = ev.rule(rule_id)
        if (
            state == "TRIGGERED"
            and f is not None
            and (rule_id not in HIGH_VETO_RULES or sev == "high")
        ):
            out.append(
                ev.reason(rule_id, f, f"TRIGGERED/{sev}", _rule_message(rule_id, state, sev))
            )
    collapse = ev.fact("scout.market_collapse")
    if collapse is not None:
        out.append(ev.reason("MARKET_COLLAPSE", collapse, "MARKET_COLLAPSE",
                             "Scout preserved positive MARKET_COLLAPSE evidence"))  # fmt: skip
    return out


def blockers(ev: _Ev) -> list[Reason]:
    out = []
    for rule_id in (*BLOCKER_RULES, *HIGH_BLOCKER_RULES):
        state, sev, f = ev.rule(rule_id)
        if (
            state == "TRIGGERED"
            and f is not None
            and (rule_id not in HIGH_BLOCKER_RULES or sev == "high")
        ):
            out.append(
                ev.reason(rule_id, f, f"TRIGGERED/{sev}", _rule_message(rule_id, state, sev))
            )
    for rule_id in LOWER_BOUND_RULES:
        state, sev, f = ev.rule(rule_id)
        if state == "TRIGGERED" and f is not None and sev != "high" and not ev.exact(rule_id):
            out.append(ev.reason(
                f"UNRESOLVED_{rule_id}", f, f"TRIGGERED/{sev} (lower bound)",
                f"Safety V2 {rule_id} is {sev} on incomplete evidence: the true value could "
                "reach the high (veto / blocking) line",
            ))  # fmt: skip
    if ev.safety_read:
        for rule_id in CRITICAL_RULES:
            state, _, f = ev.rule(rule_id)
            if state in ("UNDETERMINED", "MISSING"):
                aspect = SAFETY_RULES[rule_id]
                g = OWNERSHIP[aspect].group
                out.append(_engine(
                    f"UNRESOLVED_{rule_id}", g, state,
                    f"Safety V2 {rule_id} is {state.lower()}: it would veto or block if "
                    "TRIGGERED, so it can't be treated as clear",
                    "safety", ev.inp.sources.safety.ref.record_id, (f"rules[{rule_id}]",),
                ))  # fmt: skip
    data = ev.fact("scout.data_status")
    if data is not None and ev.value("scout.data_status").get("data_status") == "STALE_CARRIED":
        out.append(ev.reason("STALE_CARRIED_SCOUT", data, "STALE_CARRIED",
                             "Scout carried a stale market observation (not refreshed)"))  # fmt: skip
    stage = ev.fact("scout.stage")
    unconfirmed = ev.value("scout.stage").get("unconfirmed_stage")
    if stage is not None and unconfirmed is not None:
        out.append(ev.reason("STAGE_UNCONFIRMED", stage, str(unconfirmed),
                             f"Scout's stage change to {unconfirmed} isn't confirmed yet"))  # fmt: skip
    return out


# --- 2b. Safety entry readiness -------------------------------------------------------------------

REQUIRED_SAFETY_DOMAINS = ("IDENTITY", "HOLDERS", "MARKET")
NOT_READY_CODES = {"IDENTITY": "IDENTITY_SAFETY_NOT_READY", "HOLDERS": "HOLDER_SAFETY_NOT_READY",
                   "MARKET": "MARKET_SAFETY_NOT_READY"}  # fmt: skip
NOT_MALICIOUS = (
    "not entry-eligible with current evidence; this is not evidence the token is malicious"
)


@dataclass(frozen=True)
class DomainReadiness:
    """Whether one Safety domain Opportunity requires for entry was genuinely assessed.
    Safety coverage COMPLETE means every *applicable* rule is resolved; a domain Safety
    never collected is out of scope there, but never entry-ready here."""

    domain: str
    state: str  # READY / READY_PARTIAL / NOT_READY
    status: str  # the Safety section / field status it was read from
    reason: str


def _status(section: Any, *path: str) -> str:
    node = section
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return str(node) if isinstance(node, str) else "MISSING"


def safety_readiness(inp: OpportunityInput) -> tuple[DomainReadiness, ...]:
    """From the stored Safety V2 v4 body, never from rule applicability:

    * IDENTITY: ``identity.token_mint.status`` VERIFIED and both
      ``authority.mint_authority`` / ``authority.freeze_authority`` AVAILABLE;
    * HOLDERS: ``holders.status`` AVAILABLE (complete scan) -> READY, PARTIAL (lower
      bounds) -> READY_PARTIAL; UNAVAILABLE (no observation), NOT_COLLECTED,
      PROVIDER_UNAVAILABLE or UNKNOWN (untrustworthy) -> NOT_READY;
    * MARKET: ``market.status`` AVAILABLE (the latest market read succeeded: pools, or a
      successful "no pools" handled by NO_ELIGIBLE_MARKET) -> READY; otherwise NOT_READY.
      Individual market rules may be out of scope (e.g. MARKET_CLOSED_ON_CHAIN for a
      healthy reported pool).

    Creator / deployer and holder-change history are not required."""
    sf = inp.sources.safety
    if sf.ref.status not in ("AVAILABLE", "STALE") or sf.identity is None:
        why = f"no usable Safety V2 snapshot ({sf.ref.status}: {sf.ref.reason})"
        return tuple(DomainReadiness(d, "NOT_READY", sf.ref.status, why)
                     for d in REQUIRED_SAFETY_DOMAINS)  # fmt: skip
    mint = _status(sf.identity, "token_mint", "status")
    auth = [_status(sf.authority, k, "status") for k in ("mint_authority", "freeze_authority")]
    identity = (
        DomainReadiness("IDENTITY", "READY", mint, "identity verified, authorities read")
        if mint == "VERIFIED" and auth == ["AVAILABLE", "AVAILABLE"]
        else DomainReadiness("IDENTITY", "NOT_READY", f"{mint}/{'/'.join(auth)}",
                             "token identity or authorities weren't established")
    )  # fmt: skip
    hs = _status(sf.holders, "status")
    holders = DomainReadiness(
        "HOLDERS",
        "READY" if hs == "AVAILABLE" else "READY_PARTIAL" if hs == "PARTIAL" else "NOT_READY",
        hs,
        _status(sf.holders, "reason") if hs not in ("AVAILABLE",) else "complete holder scan",
    )
    ms = _status(sf.market, "status")
    market = DomainReadiness(
        "MARKET", "READY" if ms == "AVAILABLE" else "NOT_READY", ms,
        "a successful market observation" if ms == "AVAILABLE" else _status(sf.market, "reason"),
    )  # fmt: skip
    return identity, holders, market


# --- 3. quality ---------------------------------------------------------------------------------------


def quality(
    ev: _Ev, technical: TechnicalBand, readiness: tuple[DomainReadiness, ...]
) -> tuple[Quality, list[Reason]]:
    s = ev.inp.sources
    insufficient: list[Reason] = []
    partial: list[Reason] = []
    info: list[Reason] = []

    def miss(out: list[Reason], code: str, status: str, msg: str, source: Layer | None,
             ref: SourceRef | None = None) -> None:  # fmt: skip
        out.append(_engine(code, None, status, msg, source, ref.record_id if ref else None))

    if s.scout.ref.status == "NOT_SUPPORTED":
        miss(insufficient, "NOT_SUPPORTED_CHAIN", "NOT_SUPPORTED",
             "Opportunity V1 is Solana-only", None)  # fmt: skip
    else:
        if s.scout.ref.status != "AVAILABLE":
            miss(insufficient, "NEED_FRESH_SCOUT", s.scout.ref.status,
                 f"Scout evidence is {s.scout.ref.status}", "scout", s.scout.ref)  # fmt: skip
        if s.safety.ref.status != "AVAILABLE":
            miss(insufficient, "NEED_FRESH_SAFETY", s.safety.ref.status,
                 f"Safety V2 snapshot is {s.safety.ref.status}", "safety", s.safety.ref)  # fmt: skip
        for r in readiness:
            if r.state == "NOT_READY":
                miss(insufficient, NOT_READY_CODES[r.domain], r.status,
                     f"required Safety {r.domain} evidence not assessed ({r.status}: "
                     f"{r.reason}): {NOT_MALICIOUS}", "safety", s.safety.ref)  # fmt: skip
            elif r.state == "READY_PARTIAL":
                miss(partial, "NEED_COMPLETE_HOLDER_EVIDENCE", r.status,
                     f"Safety {r.domain} evidence is {r.status} (lower bounds)", "safety",
                     s.safety.ref)  # fmt: skip
        if s.safety.ref.status == "AVAILABLE":
            coverage = ev.value("safety.coverage").get("coverage")
            if coverage not in ("COMPLETE", "PARTIAL"):
                miss(insufficient, "SAFETY_COVERAGE_INSUFFICIENT", str(coverage),
                     f"Safety V2 coverage is {coverage}", "safety", s.safety.ref)  # fmt: skip
            elif coverage == "PARTIAL":
                miss(partial, "NEED_COMPLETE_SAFETY", "PARTIAL",
                     "Safety V2 coverage is PARTIAL (undetermined rules)", "safety", s.safety.ref)  # fmt: skip
            for f in s.safety.facts:
                v = f.value if isinstance(f.value, dict) else {}
                if f.role != "DIAGNOSTIC" and v.get("state") == "OUT_OF_SCOPE":
                    info.append(ev.reason(f"OUT_OF_SCOPE_{v.get('rule')}", f, "OUT_OF_SCOPE",
                                          f"Safety V2 {v.get('rule')} is out of scope "
                                          "(doesn't lower quality)"))  # fmt: skip
        if technical == "UNAVAILABLE":
            miss(partial, "NEED_TECHNICAL_EVIDENCE", "UNAVAILABLE",
                 "no fresh, measured technical evidence", "technical", s.technical.ref)  # fmt: skip
        if s.scout.ref.status in ("AVAILABLE", "STALE"):
            for aspect in CORE_FLOW:
                core = ev.owned.get(aspect)
                if core is not None and core.status != "AVAILABLE":
                    partial.append(ev.reason("NEED_COMPLETE_FLOW_EVIDENCE", core, "NOT_AVAILABLE",
                                             f"Scout {aspect} not available: {core.reason}"))  # fmt: skip
    for ref, layer, code in ((s.social.ref, "social", "SOCIAL_NOT_AVAILABLE"),
                             (s.news.ref, "news", "NEWS_NOT_AVAILABLE")):  # fmt: skip
        if ref.status != "AVAILABLE":
            info.append(_engine(code, None, ref.status,
                                f"{layer} evidence is {ref.status} (optional; quality unchanged)",
                                layer, ref.record_id))  # type: ignore[arg-type]  # fmt: skip
    q: Quality = "INSUFFICIENT" if insufficient else "PARTIAL" if partial else "COMPLETE"
    return q, [*insufficient, *partial, *info]


# --- 4. setup -----------------------------------------------------------------------------------------


@dataclass
class _Setup:
    band: SetupBand = "NONE"
    flow_positive: bool = False
    liquidity_positive: bool = False
    reasons: list[Reason] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    why_not: list[str] = field(default_factory=list)


def setup(ev: _Ev, cfg: DecisionConfig) -> _Setup:
    """Scout-owned facts only (sub-signal raws and measures; never Scout's composite or
    family scores). Missing values never satisfy a condition, so making a fact unavailable
    can't raise the band."""
    out = _Setup()
    vol, txn = ev.raw("flow.volume_acceleration"), ev.raw("flow.trade_acceleration")
    measures = ev.value("flow.buy_pressure").get("measures") or {}
    share, change = measures.get("buy_share"), measures.get("buy_pressure_change")
    fq = ev.value("flow.quality").get("flow_quality")
    growth, stability = ev.raw("liquidity.growth"), ev.raw("liquidity.stability")
    move, regime = ev.raw("earliness.move_already_made"), ev.raw("earliness.activity_regime")
    stage = ev.value("scout.stage").get("stage")
    eligible = ev.value("scout.eligibility").get("eligible")

    vol_rising = vol is not None and vol >= cfg.rising_ratio
    txn_rising = txn is not None and txn >= cfg.rising_ratio
    buyers = (
        fq in ("consistent", "count_only")
        and isinstance(share, int | float) and share >= cfg.min_strength_buy_share
        and isinstance(change, int | float) and change >= cfg.buyers_strengthening_change
    )  # fmt: skip
    if fq == "divergent":
        out.contradictions.append("flow quality divergent")
    for aspect, code in SCOUT_FLOW_RISKS.items():
        if aspect != "flow.volume_to_liquidity" and ev.fact(aspect) is not None:
            out.contradictions.append(code.lower())
    if vol is not None and vol <= cfg.falling_ratio:
        out.contradictions.append("volume falling")
    if txn is not None and txn <= cfg.falling_ratio:
        out.contradictions.append("trades falling")
    if stability is not None and stability <= -cfg.liquidity_drain_pct:
        out.contradictions.append("liquidity draining")
    out.flow_positive = (
        (vol_rising or txn_rising)
        and fq in ("consistent", "count_only")
        and (ev.fact("flow.distribution") is None)
    )
    out.liquidity_positive = (
        growth is not None and growth >= cfg.liquidity_growing_pct
        and stability is not None and stability > -cfg.liquidity_drain_pct
    )  # fmt: skip
    for aspect, ok in (("flow.volume_acceleration", vol_rising),
                       ("flow.trade_acceleration", txn_rising),
                       ("flow.buy_pressure", buyers)):  # fmt: skip
        f = ev.fact(aspect)
        if ok and f is not None:
            out.reasons.append(ev.reason(aspect.upper().replace(".", "_"), f, "POSITIVE",
                                         f"Scout {aspect} meets its threshold"))  # fmt: skip
    if out.liquidity_positive:
        f = ev.fact("liquidity.growth")
        assert f is not None
        out.reasons.append(ev.reason("LIQUIDITY_GROWING", f, "POSITIVE",
                                     f"liquidity +{growth:.1f}% and not draining"))  # fmt: skip

    if eligible is not True:
        out.why_not.append("Scout eligibility not established")
        return out
    if stage is None or stage in NO_SETUP_STAGES:
        out.why_not.append(f"Scout stage {stage}")
        return out
    if not (vol_rising or txn_rising or buyers):
        out.why_not.append("no positive FLOW evidence")
        return out
    out.band = "WEAK" if stage in WEAK_STAGES else "NONE"
    trusted = vol is not None and txn is not None and stability is not None and move is not None
    moderate = (
        stage in MODERATE_STAGES and trusted and (vol_rising or txn_rising)
        and fq in ("consistent", "count_only") and not out.contradictions
        and move is not None and move < cfg.late_move_pct
    )  # fmt: skip
    if moderate:
        out.band = "MODERATE"
        strong = (
            stage == "ACCELERATING" and vol_rising and txn_rising and buyers
            and fq == "consistent" and move is not None and move <= cfg.early_move_pct
            and regime is not None and regime >= cfg.min_activity_regime
            and (growth is None or growth >= cfg.liquidity_steady_pct)
        )  # fmt: skip
        if strong:
            out.band = "STRONG"
    if out.band != "STRONG":
        out.why_not += [f"contradiction: {c}" for c in out.contradictions]
    return out


# --- 6. technical and social bands ----------------------------------------------------------------


def technical(ev: _Ev, cfg: DecisionConfig) -> tuple[TechnicalBand, list[Reason], list[Reason]]:
    """Per source: UP (confirming), DOWN, or NEUTRAL. Any DOWN -> CONTRADICTING; CONFIRMING
    only when every usable source confirms; a disagreement is reported, never averaged."""
    s = ev.inp.sources
    readings: list[tuple[str, Fact]] = []
    if s.technical.ref.status == "AVAILABLE" and ev.fact("price.snapshot_trend") is not None:
        snap = ev.value("price.snapshot_trend")
        f = ev.fact("price.snapshot_trend")
        assert f is not None
        confirmed = any(ev.value(a).get(k) is True for a, k in (
            ("price.breakout", "breakout"), ("price.higher_lows", "higher_lows"),
            ("price.volume_confirmed", "volume_confirmed")))  # fmt: skip
        enough = (snap.get("snapshots") or 0) >= cfg.min_technical_snapshots
        trend = snap.get("trend")
        readings.append(("DOWN" if trend == "down" else
                         "UP" if trend == "up" and enough and confirmed else "NEUTRAL", f))  # fmt: skip
    if s.technical.analyze_ref.status == "AVAILABLE" and ev.fact("price.analyze_trend") is not None:
        f = ev.fact("price.analyze_trend")
        assert f is not None
        label = ev.value("price.analyze_trend").get("label")
        readings.append(({"uptrend": "UP", "downtrend": "DOWN"}.get(str(label), "NEUTRAL"), f))
    reasons = [ev.reason(f"TECHNICAL_{r}", f, r, f"{f.aspect} reads {r.lower()}")
               for r, f in readings]  # fmt: skip
    kinds = {r for r, _ in readings}
    conflict = []
    if len(kinds) > 1:
        conflict.append(_engine("TECHNICAL_CONFLICT", None, "/".join(sorted(kinds)),
                                "technical sources disagree; the conservative reading is used",
                                "technical"))  # fmt: skip
    band: TechnicalBand = (
        "UNAVAILABLE" if not readings else "CONTRADICTING" if "DOWN" in kinds
        else "CONFIRMING" if kinds == {"UP"} else "NEUTRAL"
    )  # fmt: skip
    return band, reasons, conflict


def social(ev: _Ev, cfg: DecisionConfig) -> tuple[SocialBand, list[Reason]]:
    s = ev.inp.sources
    m = ev.fact("attention.momentum")
    if s.social.ref.status != "AVAILABLE" or m is None:
        return "UNAVAILABLE", []
    state = ev.value("attention.momentum").get("state")
    spam = ev.value("attention.spam").get("spam_risk")
    cross = ev.value("attention.market_cross").get("state")
    attr = ev.value("attention.attribution")
    contra = []
    if cross in CONTRADICTING_CROSS:
        contra.append(f"market cross-check {cross}")
    if spam == "high":
        contra.append("high spam risk")
    if contra:
        return "CONTRADICTING", [ev.reason("SOCIAL_CONTRADICTING", m, "; ".join(contra),
                                           "social evidence contradicts: " + "; ".join(contra))]  # fmt: skip
    exact, total = attr.get("exact_mentions"), attr.get("attributable_posts")
    attributed = (
        isinstance(exact, int) and isinstance(total, int) and total > 0 and exact >= 1
        and exact >= cfg.min_exact_mention_share * total
    )  # fmt: skip
    cross_ok = ev.fact("attention.market_cross") is None or cross == "CORROBORATED"
    if state in SUPPORTING_SOCIAL and spam in ("low", "medium") and attributed and cross_ok:
        return "SUPPORTING", [ev.reason("SOCIAL_SUPPORTING", m, str(state),
                                        f"attention {state}, attributed, spam {spam}")]  # fmt: skip
    return "NEUTRAL", []


# --- 7. risks --------------------------------------------------------------------------------------


def risks(ev: _Ev, social_band: SocialBand, social_reasons: list[Reason],
          classified: set[str]) -> list[Reason]:  # fmt: skip
    """Non-veto, non-blocker risks. Each correlation group counts once in the tier."""
    out = []
    for rule_id in SAFETY_RULES:
        if rule_id in classified:
            continue
        state, sev, f = ev.rule(rule_id)
        if state == "TRIGGERED" and f is not None:
            out.append(
                ev.reason(rule_id, f, f"TRIGGERED/{sev}", _rule_message(rule_id, state, sev))
            )
    for aspect, code in SCOUT_FLOW_RISKS.items():
        f = ev.fact(aspect)
        if f is not None:
            out.append(ev.reason(code, f, "FLAGGED", f"Scout raised {aspect}"))
    fq = ev.fact("flow.quality")
    if fq is not None and ev.value("flow.quality").get("flow_quality") == "divergent":
        out.append(ev.reason("SCOUT_FLOW_DIVERGENT", fq, "divergent",
                             "Scout flow quality is divergent"))  # fmt: skip
    if social_band == "CONTRADICTING":
        out += [r.model_copy(update={"code": "SOCIAL_CONTRADICTING"}) for r in social_reasons]
    news = ev.fact("news.impact")
    if news is not None and ev.inp.sources.news.ref.status == "AVAILABLE":
        stories = ev.value("news.impact").get("stories") or []
        bad = [x for x in stories if isinstance(x, dict) and x.get("impact") == "high"
               and x.get("sentiment") == "bearish" and x.get("scope") == "asset"
               and x.get("stale") is False]  # fmt: skip
        if bad:
            llm = ev.inp.sources.news.llm_labelled
            models = ", ".join(ev.inp.sources.news.sentiment_models) or "unknown"
            out.append(ev.reason(
                "NEWS_HIGH_IMPACT_BEARISH", news, "medium",
                f"{len(bad)} fresh high-impact bearish asset story(ies); labels "
                + (f"are language-model generated ({models})" if llm else "not model-labelled"),
            ))  # fmt: skip
    return out


# --- 9. the decision ---------------------------------------------------------------------------------


def decide(inp: OpportunityInput, config: DecisionConfig = DECISION_CONFIG) -> OpportunityDecision:
    check_integrity(inp)
    ev = _Ev(inp)
    veto = vetoes(ev)
    tech_band, tech_reasons, tech_conflict = technical(ev, config)
    readiness = safety_readiness(inp)
    not_ready = [r for r in readiness if r.state == "NOT_READY"]
    q, missing = quality(ev, tech_band, readiness)
    st = setup(ev, config)
    block = blockers(ev)
    soc_band, soc_reasons = social(ev, config)
    classified = (
        {r.code for r in veto}
        | {r.code for r in block}
        | {r.code.removeprefix("UNRESOLVED_") for r in block}
    )
    risk = risks(ev, soc_band, soc_reasons, classified)
    risk_groups = sorted({r.group for r in risk if r.group is not None})
    tier: RiskTier = ("CLEAN" if not risk_groups else "ELEVATED_1" if len(risk_groups) == 1
                      else "ELEVATED_2_PLUS")  # fmt: skip
    positive: list[CorrelationGroup] = []
    if st.flow_positive:
        positive.append("FLOW")
    if st.liquidity_positive:
        positive.append("LIQUIDITY")
    if tech_band == "CONFIRMING":
        positive.append("PRICE")
    if soc_band == "SUPPORTING":
        positive.append("ATTENTION")
    positive_reasons = [*st.reasons, *(r for r in tech_reasons if r.status == "UP")]
    if soc_band == "SUPPORTING":
        positive_reasons += soc_reasons

    upgrade: dict[str, UpgradeItem] = {}

    def gate(code: str, name: str, message: str) -> None:
        upgrade.setdefault(code, UpgradeItem(code=code, gate=name, message=message))

    ineligible = [
        _engine(NOT_READY_CODES[r.domain], None, r.status,
                f"required Safety {r.domain} evidence wasn't assessed ({r.status}: {r.reason}): "
                f"{NOT_MALICIOUS}", "safety", inp.sources.safety.ref.record_id,
                (r.domain.lower(),))
        for r in not_ready
    ]  # fmt: skip
    basis: list[SkipBasis] = []
    if veto:
        basis.append("HARD_VETO")
    if not_ready:
        basis.append("REQUIRED_SAFETY_NOT_READY")
    if st.band == "NONE":
        basis.append("NO_SETUP")
    skip_basis = tuple(basis)
    decision: Decision
    if skip_basis:
        decision = "SKIP"
    else:
        for r in missing:
            if (
                r.status not in ("OUT_OF_SCOPE",)
                and r.code not in ("SOCIAL_NOT_AVAILABLE", "NEWS_NOT_AVAILABLE")
                and not r.code.startswith("OUT_OF_SCOPE_")
            ):
                if q == "INSUFFICIENT" and r.code in _INSUFFICIENT_CODES:
                    gate(r.code, "quality", r.message)
                elif q == "PARTIAL" and st.band != "STRONG":
                    gate(r.code, "quality", r.message + " (PARTIAL quality needs a STRONG setup)")
        if SETUP_ORDER[st.band] < SETUP_ORDER["MODERATE"]:
            gate("NEED_STRONGER_SETUP", "setup",
                 f"setup is {st.band}; ENTER needs MODERATE or STRONG")  # fmt: skip
        for r in block:
            gate(r.code, "blockers", r.message)
        if tech_band == "UNAVAILABLE":
            gate("NEED_TECHNICAL_EVIDENCE", "technical", "no fresh, measured technical evidence")
        elif tech_band == "CONTRADICTING":
            gate("TECHNICAL_CONTRADICTING", "technical", "technical evidence contradicts")
        elif tech_band == "NEUTRAL" and st.band != "STRONG":
            gate("NEED_TECHNICAL_CONFIRMATION", "technical",
                 "technical is NEUTRAL; ENTER needs CONFIRMING (or NEUTRAL with STRONG setup)")  # fmt: skip
        if soc_band == "CONTRADICTING":
            gate("SOCIAL_CONTRADICTING", "social", "social evidence contradicts")
        if tier == "ELEVATED_2_PLUS":
            gate("TOO_MANY_RISK_GROUPS", "risk",
                 f"{len(risk_groups)} risk groups: {', '.join(risk_groups)}")  # fmt: skip
        if "FLOW" not in positive:
            gate("NEED_FLOW_CONFIRMATION", "positive_groups", "FLOW is not a positive group")
        if len(positive) < 2:
            gate("NEED_SECOND_POSITIVE_GROUP", "positive_groups",
                 f"{len(positive)} independent positive group(s); ENTER needs 2 incl. FLOW")  # fmt: skip
        decision = "WATCH" if upgrade else "ENTER"
    return OpportunityDecision(
        schema_version=DECISION_SCHEMA,
        rules_version=OPPORTUNITY_RULES_VERSION,
        canonical_id=inp.canonical_id,
        decision_at=inp.decision_at,
        origin=inp.origin,
        input_hash=inp.input_hash(),
        thresholds=config.as_dict(),
        decision=decision,
        skip_basis=skip_basis if decision == "SKIP" else (),
        quality=q,
        safety_entry_readiness={r.domain: r.state for r in readiness},
        setup_band=st.band,
        technical_band=tech_band,
        social_band=soc_band,
        risk_tier=tier,
        positive_groups=tuple(sorted(positive)),
        risk_groups=tuple(risk_groups),
        vetoes=_sorted(veto),
        ineligible=_sorted(ineligible),
        blockers=_sorted(block),
        positive_reasons=_sorted(positive_reasons),
        risks=_sorted([*risk, *tech_conflict]),
        missing_evidence=_sorted(missing),
        upgrade_path=tuple(sorted(upgrade.values(), key=lambda u: (u.gate, u.code)))
        if decision == "WATCH"
        else (),
    )


_INSUFFICIENT_CODES = ("NOT_SUPPORTED_CHAIN", "NEED_FRESH_SCOUT", "NEED_FRESH_SAFETY",
                       "SAFETY_COVERAGE_INSUFFICIENT")  # fmt: skip
