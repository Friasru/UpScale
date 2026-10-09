"""Opportunity Model V1 (O2): the pure SKIP / WATCH / ENTER decision engine.

Inputs are real O1 `OpportunityInput`s built from local fixtures, then varied fact by fact
(re-validated through the O1 model, so every variant is a legal input). Offline."""

import copy
import json
import re
import socket
import sqlite3
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.opportunity_model_fakes import (
    Archive,
    at,
    build,
    candidate,
    decision_record,
    news_findings,
    rule,
    safety_body,
    safety_db,
    scout_record,
    social_record,
)
from upscale.services.opportunity_model import decision as engine
from upscale.services.opportunity_model.decision import (
    BLOCKER_RULES,
    CRITICAL_RULES,
    HIGH_BLOCKER_RULES,
    VETO_RULES,
    decide,
)
from upscale.services.opportunity_model.decision_models import (
    DECISION_ORDER,
    OpportunityDecision,
    OpportunityDecisionError,
)
from upscale.services.opportunity_model.models import OpportunityInput
from upscale.services.opportunity_model.ownership import SAFETY_RULES
from upscale.services.opportunity_model.service import build_input

T = at()
CHANGE_RULES = ("CONCENTRATION_RISING", "LARGE_HOLDER_EXIT", "RAPID_HOLDER_LOSS")
SEVERITY = {"NOT_A_TOKEN_MINT": "critical", "UNEXPECTED_TOKEN_PROGRAM": "critical",
            "MALFORMED_MINT_ACCOUNT": "critical", "IDENTITY_MISMATCH": "critical"}  # fmt: skip
LISTS = (("scout", "market_view", "facts"), ("scout", "context"), ("scout", "diagnostics"),
         ("technical", "facts"), ("social", "facts"), ("news", "facts"), ("safety", "facts"))  # fmt: skip


# The exact evidence field each severity-gated rule reads (as Safety V2's rules do).
EVIDENCE = {
    "TOP1_CONCENTRATION": "holders.top1_pct", "TOP10_CONCENTRATION": "holders.top10_pct",
    "VERIFIED_DEPLOYER_HOLDS_SUPPLY": "creator.deployer_holding_pct",
    "LOW_LIQUIDITY": "market.primary_liquidity_usd", "VERY_NEW_POOL": "market.pool_age_hours",
}  # fmt: skip


def _clean_safety(as_of: Any) -> dict[str, Any]:
    flags = [rule(r, "NOT_TRIGGERED", SEVERITY.get(r, "medium")) for r in SAFETY_RULES
             if r not in (*CHANGE_RULES, "TOKEN_2022_EXTENSION_RISK")]  # fmt: skip
    for f in flags:
        if f["id"] in EVIDENCE:
            f["evidence"] = [EVIDENCE[f["id"]]]
    body = safety_body(as_of, flags=flags)
    body["coverage"] = {**body["coverage"], "coverage": "COMPLETE", "reasons": []}
    body["undetermined"] = []
    # Section statuses as Safety V2 writes them (features.holder_section / market_section).
    body["authority"] = {k: {"status": "AVAILABLE", "value": None, "lower_bound": False,
                             "reason": None} for k in ("mint_authority", "freeze_authority")}  # fmt: skip
    body["holders"] = {"status": "AVAILABLE", "reason": None, "source": "full_scan"}
    body["market"] = {"status": "AVAILABLE", "reason": None,
                      "pool_presence": {"state": "REPORTED", "reason": None}}  # fmt: skip
    body["creator"] = {"source_status": "CAPTURED"}
    for path in EVIDENCE.values():
        section, _, name = path.partition(".")
        body[section][name] = {"status": "AVAILABLE", "value": 1.0, "lower_bound": False,
                               "reason": None}  # fmt: skip
    return body


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> OpportunityInput:
    """A clean ENTER: every gate passes."""
    d = tmp_path_factory.mktemp("o2")
    arch = Archive(d / "evidence.sqlite3")
    arch.add(scout_record(T, cand=candidate(T, risk_flags=[])))
    arch.add(social_record(T))
    arch.add(decision_record(T))
    arch.close()
    safety_db(d / "safety.sqlite3", [(T, "4", _clean_safety(T))])
    return build(arch.path, d / "safety.sqlite3")


# --- variation helpers -------------------------------------------------------------------------


def _dump(inp: OpportunityInput) -> dict[str, Any]:
    return copy.deepcopy(inp.model_dump(mode="json"))


def _fact(d: dict[str, Any], aspect: str) -> dict[str, Any]:
    for path in LISTS:
        node: Any = d["sources"]
        for k in path:
            node = node[k]
        for f in node:
            if f["aspect"] == aspect and f["role"] != "DIAGNOSTIC":
                return f
    raise KeyError(aspect)


def vary(inp: OpportunityInput, edit: Callable[[dict[str, Any]], None]) -> OpportunityInput:
    d = _dump(inp)
    edit(d)
    return OpportunityInput.model_validate(d)


def set_rule(rule_id: str, state: str, severity: str = "medium") -> Callable[[dict], None]:
    def edit(d: dict[str, Any]) -> None:
        f = _fact(d, SAFETY_RULES[rule_id])
        f["value"] = {**f["value"], "state": state, "severity": severity}

    return edit


def unverified() -> Callable[[dict], None]:
    """Safety couldn't verify the token identity (stored section and its fact)."""
    token = {"status": "UNVERIFIED", "reason": "mint unread"}
    return both(section("identity", token_mint=token),
                set_value("identity.token_mint", status="UNVERIFIED"))  # fmt: skip


def degrade_holders_to_partial(d: dict[str, Any]) -> None:
    """A complete holder scan becomes a partial one (never repairs a missing domain)."""
    if d["sources"]["safety"]["holders"].get("status") == "AVAILABLE":
        section("holders", status="PARTIAL", reason="page cap")(d)


def section(name: str, **kv: Any) -> Callable[[dict], None]:
    """Change a stored Safety section (e.g. ``holders.status``)."""

    def edit(d: dict[str, Any]) -> None:
        d["sources"]["safety"][name] = {**d["sources"]["safety"][name], **kv}

    return edit


HOLDER_RULES = ("FEW_HOLDERS", "LARGE_UNCLASSIFIED_PROGRAM_OWNER", "LARGE_UNKNOWN_OWNER",
                "TOP10_CONCENTRATION", "TOP1_CONCENTRATION")  # fmt: skip
MARKET_RULES = ("LIQUIDITY_COLLAPSE", "LOW_LIQUIDITY", "MARKET_CLOSED_ON_CHAIN",
                "MARKET_NOT_REPORTED", "NO_ELIGIBLE_MARKET", "PRIMARY_MARKET_UNCLEAR",
                "VERY_NEW_POOL")  # fmt: skip


def no_holders(status: str = "UNAVAILABLE", rules: str = "OUT_OF_SCOPE") -> Callable[[dict], None]:
    """No usable holder observation, as Safety V2 records it: the section's status, and the
    holder rules out of scope (none) or UNDETERMINED (a failed read)."""
    return both(section("holders", status=status, reason=f"holders {status}"),
                *(set_rule(r, rules) for r in HOLDER_RULES))  # fmt: skip


def no_market(status: str = "UNAVAILABLE", rules: str = "OUT_OF_SCOPE") -> Callable[[dict], None]:
    return both(section("market", status=status, reason=f"market {status}"),
                *(set_rule(r, rules) for r in MARKET_RULES))  # fmt: skip


def partial_evidence(rule_id: str) -> Callable[[dict], None]:
    """The rule's evidence field becomes a lower bound (PARTIAL)."""

    def edit(d: dict[str, Any]) -> None:
        if rule_id in EVIDENCE:
            section, _, name = EVIDENCE[rule_id].partition(".")
            d["sources"]["safety"][section][name].update(status="PARTIAL", lower_bound=True,
                                                         reason="partial scan")  # fmt: skip

    return edit


def undetermine(rule_id: str) -> Callable[[dict], None]:
    """UNDETERMINED as Safety V2 produces it: on incomplete evidence."""
    return both(set_rule(rule_id, "UNDETERMINED"), partial_evidence(rule_id))


def escalate(rule_id: str, severity: str) -> Callable[[dict], None]:
    """TRIGGERED at `severity`, unless the rule is already TRIGGERED at least as severely
    (never a de-escalation)."""

    def edit(d: dict[str, Any]) -> None:
        v = _fact(d, SAFETY_RULES[rule_id])["value"]
        if v["state"] == "TRIGGERED" and (v["severity"] != "medium" or severity == "medium"):
            return
        set_rule(rule_id, "TRIGGERED", severity)(d)

    return edit


def set_value(aspect: str, **kv: Any) -> Callable[[dict], None]:
    def edit(d: dict[str, Any]) -> None:
        f = _fact(d, aspect)
        f.update(status="AVAILABLE", reason=None)
        f["value"] = {**(f["value"] or {}), **kv}

    return edit


def unavailable(aspect: str) -> Callable[[dict], None]:
    def edit(d: dict[str, Any]) -> None:
        _fact(d, aspect).update(status="NOT_AVAILABLE", value=None, reason="removed in test")

    return edit


def flag(aspect: str) -> Callable[[dict], None]:
    return set_value(aspect, flagged=True, severity="high", details=["test"])


def ref(source: str, status: str, key: str = "ref", **kv: Any) -> Callable[[dict], None]:
    def edit(d: dict[str, Any]) -> None:
        d["sources"][source][key].update(status=status, reason=f"{status} in test", **kv)

    return edit


def both(*edits: Callable[[dict], None]) -> Callable[[dict], None]:
    def edit(d: dict[str, Any]) -> None:
        for e in edits:
            e(d)

    return edit


def run(inp: OpportunityInput, *edits: Callable[[dict], None]) -> OpportunityDecision:
    return decide(vary(inp, both(*edits)) if edits else inp)


def codes(reasons: Any) -> set[str]:
    return {r.code for r in reasons}


def raw(aspect: str, value: float) -> Callable[[dict], None]:
    return set_value(aspect, raw=value)


def buy_change(change: float | None) -> Callable[[dict], None]:
    def edit(d: dict[str, Any]) -> None:
        f = _fact(d, "flow.buy_pressure")
        f["value"]["measures"] = {**f["value"]["measures"], "buy_pressure_change": change}

    return edit


TECH_NONE = both(
    ref("technical", "NOT_COLLECTED"), ref("technical", "NOT_COLLECTED", "analyze_ref")
)
MODERATE = set_value("scout.stage", stage="EARLY")


# --- canonical outcomes ----------------------------------------------------------------------


def test_canonical_enter(base: OpportunityInput) -> None:
    d = decide(base)
    assert (d.decision, d.quality, d.setup_band, d.technical_band, d.social_band, d.risk_tier) == (
        "ENTER", "COMPLETE", "STRONG", "CONFIRMING", "SUPPORTING", "CLEAN")  # fmt: skip
    assert d.positive_groups == ("ATTENTION", "FLOW", "LIQUIDITY", "PRICE")
    assert d.vetoes == d.blockers == d.upgrade_path == () and d.risk_groups == ()
    assert d.schema_version == "opportunity.decision.v1" and d.rules_version == "1"
    assert d.input_hash == base.input_hash()


# --- hard vetoes ---------------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", VETO_RULES)
def test_every_hard_veto_skips(base: OpportunityInput, rule_id: str) -> None:
    d = run(base, set_rule(rule_id, "TRIGGERED", "high"))
    assert d.decision == "SKIP" and rule_id in codes(d.vetoes)
    assert d.upgrade_path == ()  # no path suggesting a veto can be overridden


def test_low_liquidity_vetoes_only_at_high_severity(base: OpportunityInput) -> None:
    assert run(base, set_rule("LOW_LIQUIDITY", "TRIGGERED", "high")).decision == "SKIP"
    medium = run(base, set_rule("LOW_LIQUIDITY", "TRIGGERED", "medium"))
    assert medium.decision == "ENTER" and medium.risk_groups == ("LIQUIDITY",)
    assert medium.risk_tier == "ELEVATED_1" and not medium.vetoes


def test_only_positive_scout_collapse_evidence_vetoes(base: OpportunityInput) -> None:
    assert _fact(_dump(base), "scout.market_collapse")["status"] == "NOT_AVAILABLE"
    assert "MARKET_COLLAPSE" not in codes(decide(base).vetoes)  # absence isn't health...
    d = run(base, set_value("scout.market_collapse", market_status="MARKET_COLLAPSE",
                            collapse_evidence=["price collapsed -95% over 24h"]))  # fmt: skip
    assert d.decision == "SKIP" and "MARKET_COLLAPSE" in codes(d.vetoes)


def test_a_veto_beats_every_later_stage(base: OpportunityInput) -> None:
    d = run(base, set_rule("FREEZE_AUTHORITY_ACTIVE", "TRIGGERED", "high"),
            set_rule("TOP10_CONCENTRATION", "TRIGGERED", "high"), TECH_NONE)  # fmt: skip
    assert d.decision == "SKIP" and d.upgrade_path == ()


# --- blockers ---------------------------------------------------------------------------------------

BLOCKERS = [(r, "high") for r in (*BLOCKER_RULES, *HIGH_BLOCKER_RULES)] + [
    ("MINT_AUTHORITY_ACTIVE", "medium"), ("PRIMARY_MARKET_UNCLEAR", "medium"),
    ("MARKET_NOT_REPORTED", "medium"),
]  # fmt: skip


@pytest.mark.parametrize(("rule_id", "severity"), BLOCKERS)
def test_every_blocker_caps_at_watch(base: OpportunityInput, rule_id: str, severity: str) -> None:
    d = run(base, set_rule(rule_id, "TRIGGERED", severity))
    assert d.decision == "WATCH" and rule_id in codes(d.blockers)
    assert rule_id in {u.code for u in d.upgrade_path}


def test_market_not_reported_is_a_blocker_not_a_skip(base: OpportunityInput) -> None:
    d = run(base, set_rule("MARKET_NOT_REPORTED", "TRIGGERED", "medium"))
    assert d.decision == "WATCH" and not d.vetoes


@pytest.mark.parametrize("rule_id", ["TOP1_CONCENTRATION", "TOP10_CONCENTRATION",
                                     "VERIFIED_DEPLOYER_HOLDS_SUPPLY", "VERY_NEW_POOL"])  # fmt: skip
def test_medium_severity_of_high_only_blockers_is_a_risk(
    base: OpportunityInput, rule_id: str
) -> None:
    d = run(base, set_rule(rule_id, "TRIGGERED", "medium"))
    assert d.decision == "ENTER" and not d.blockers and d.risk_tier == "ELEVATED_1"


def test_scout_stale_carried_and_unconfirmed_stage_block(base: OpportunityInput) -> None:
    carried = run(base, set_value("scout.data_status", data_status="STALE_CARRIED"))
    assert carried.decision == "WATCH" and "STALE_CARRIED_SCOUT" in codes(carried.blockers)
    unconfirmed = run(base, set_value("scout.stage", unconfirmed_stage="FADING"))
    assert unconfirmed.decision == "WATCH" and "STAGE_UNCONFIRMED" in codes(unconfirmed.blockers)


# --- unresolved rules ----------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", CRITICAL_RULES)
def test_an_undetermined_veto_or_blocker_rule_prevents_enter(
    base: OpportunityInput, rule_id: str
) -> None:
    d = run(base, set_rule(rule_id, "UNDETERMINED"))
    assert d.decision == "WATCH"
    assert f"UNRESOLVED_{rule_id}" in codes(d.blockers)
    assert rule_id not in codes(d.risks)  # never treated as NOT_TRIGGERED nor TRIGGERED


@pytest.mark.parametrize("rule_id", engine.LOWER_BOUND_RULES)
def test_a_lower_bound_medium_trigger_of_a_high_gated_rule_stays_unresolved(
    base: OpportunityInput, rule_id: str
) -> None:
    exact = run(base, set_rule(rule_id, "TRIGGERED", "medium"))
    assert exact.decision == "ENTER" and not exact.blockers  # exact: a plain risk
    bound = run(base, set_rule(rule_id, "TRIGGERED", "medium"), partial_evidence(rule_id))
    assert bound.decision == "WATCH" and f"UNRESOLVED_{rule_id}" in codes(bound.blockers)
    assert rule_id not in codes(bound.risks)


@pytest.mark.parametrize("rule_id", ["LOW_LIQUIDITY", "VERY_NEW_POOL"])
def test_exact_provider_fields_keep_medium_market_triggers_ordinary_risks(
    base: OpportunityInput, rule_id: str
) -> None:
    """Liquidity and pool age are provider-reported values Safety stores AVAILABLE or not at
    all (the rule is then UNDETERMINED): no lower-bound reading applies to them."""
    d = run(base, set_rule(rule_id, "TRIGGERED", "medium"))
    assert d.decision == "ENTER" and not d.blockers and d.risk_tier == "ELEVATED_1"
    assert rule_id not in engine.LOWER_BOUND_RULES


def test_a_rule_missing_from_a_read_snapshot_is_unresolved(base: OpportunityInput) -> None:
    d = run(base, unavailable(SAFETY_RULES["FREEZE_AUTHORITY_ACTIVE"]))
    assert d.decision == "WATCH" and "UNRESOLVED_FREEZE_AUTHORITY_ACTIVE" in codes(d.blockers)


def test_out_of_scope_rules_neither_block_nor_lower_quality(base: OpportunityInput) -> None:
    d = run(base, set_rule("TOP10_CONCENTRATION", "OUT_OF_SCOPE"))
    assert d.decision == "ENTER" and d.quality == "COMPLETE"
    assert "OUT_OF_SCOPE_TOP10_CONCENTRATION" in codes(d.missing_evidence)


# --- quality ---------------------------------------------------------------------------------------


def test_quality_levels(base: OpportunityInput) -> None:
    partial = set_value("safety.coverage", coverage="PARTIAL")
    assert run(base, partial).quality == "PARTIAL"
    assert run(base, partial).decision == "ENTER"  # PARTIAL + STRONG may enter
    pm = run(base, partial, MODERATE)
    assert (pm.quality, pm.setup_band, pm.decision) == ("PARTIAL", "MODERATE", "WATCH")
    assert "NEED_COMPLETE_SAFETY" in {u.code for u in pm.upgrade_path}
    for edit, code in (
        (set_value("safety.coverage", coverage="INSUFFICIENT"), "SAFETY_COVERAGE_INSUFFICIENT"),
        (ref("scout", "STALE", age_seconds=1201.0), "NEED_FRESH_SCOUT"),
        (ref("safety", "STALE", age_seconds=3700.0), "NEED_FRESH_SAFETY"),
    ):
        d = run(base, edit)
        assert (d.quality, d.decision) == ("INSUFFICIENT", "WATCH"), code
        assert code in {u.code for u in d.upgrade_path}
    # Required Safety never assessed: SKIP (not entry-eligible), never WATCH.
    for edit, code in (
        (ref("safety", "UNAVAILABLE"), "HOLDER_SAFETY_NOT_READY"),
        (unverified(), "IDENTITY_SAFETY_NOT_READY"),
        (
            section("authority", freeze_authority={"status": "PROVIDER_UNAVAILABLE"}),
            "IDENTITY_SAFETY_NOT_READY",
        ),
    ):
        d = run(base, edit)
        assert (d.quality, d.decision) == ("INSUFFICIENT", "SKIP"), code
        assert d.skip_basis == ("REQUIRED_SAFETY_NOT_READY",) and code in codes(d.ineligible)


def test_missing_optional_news_and_social_keep_quality(base: OpportunityInput) -> None:
    d = run(base, ref("news", "NOT_COLLECTED"), ref("social", "NOT_COLLECTED"))
    assert d.quality == "COMPLETE" and d.decision == "ENTER"
    assert {"NEWS_NOT_AVAILABLE", "SOCIAL_NOT_AVAILABLE"} <= codes(d.missing_evidence)


def test_missing_or_stale_technical_caps_quality_and_blocks_enter(base: OpportunityInput) -> None:
    for edit in (TECH_NONE, both(ref("technical", "STALE", age_seconds=1300.0),
                                 ref("technical", "STALE", "analyze_ref", age_seconds=3700.0))):  # fmt: skip
        d = run(base, edit)
        assert (d.technical_band, d.quality, d.decision) == ("UNAVAILABLE", "PARTIAL", "WATCH")
        assert "NEED_TECHNICAL_EVIDENCE" in {u.code for u in d.upgrade_path}


def test_missing_core_flow_fact_makes_quality_partial(base: OpportunityInput) -> None:
    d = run(base, unavailable("flow.quality"))
    assert d.quality == "PARTIAL" and "NEED_COMPLETE_FLOW_EVIDENCE" in codes(d.missing_evidence)


def test_non_solana_is_insufficient_and_skipped() -> None:
    class Boom:
        def latest(self, *a: Any) -> None:
            raise AssertionError

        def latest_snapshot(self, *a: Any) -> None:
            raise AssertionError

    d = decide(build_input("base:0x" + "ab" * 20, T, "HISTORICAL_REPLAY", Boom(), Boom()))
    assert (d.quality, d.setup_band, d.decision) == ("INSUFFICIENT", "NONE", "SKIP")


# --- setup -------------------------------------------------------------------------------------------


def test_setup_bands(base: OpportunityInput) -> None:
    assert decide(base).setup_band == "STRONG"
    assert run(base, MODERATE).setup_band == "MODERATE"
    weak = run(base, set_value("scout.stage", stage="STEADY"))
    assert (weak.setup_band, weak.decision) == ("WEAK", "WATCH")
    assert "NEED_STRONGER_SETUP" in {u.code for u in weak.upgrade_path}
    for edit in (set_value("scout.stage", stage="FADING"),
                 set_value("scout.eligibility", eligible=False),
                 both(raw("flow.volume_acceleration", 1.0), raw("flow.trade_acceleration", 1.0),
                      buy_change(0.0))):  # fmt: skip
        d = run(base, edit)
        assert (d.setup_band, d.decision) == ("NONE", "SKIP")


@pytest.mark.parametrize(
    ("edit", "band"),
    [
        (raw("flow.volume_acceleration", 1.3), "STRONG"),
        (raw("flow.volume_acceleration", 1.2999), "MODERATE"),
        (raw("earliness.move_already_made", 50.0), "STRONG"),
        (raw("earliness.move_already_made", 50.01), "MODERATE"),
        (raw("earliness.move_already_made", 900.0), "WEAK"),
        (raw("earliness.activity_regime", 1.0), "STRONG"),
        (raw("earliness.activity_regime", 0.99), "MODERATE"),
        (buy_change(0.03), "STRONG"),
        (buy_change(0.0299), "MODERATE"),
        (buy_change(None), "MODERATE"),  # no fallback: missing never satisfies
        (raw("liquidity.stability", -29.99), "STRONG"),
        (raw("liquidity.stability", -30.0), "WEAK"),  # draining: a Scout contradiction
        (raw("flow.trade_acceleration", 0.75), "WEAK"),  # falling trades
        (flag("flow.distribution"), "WEAK"),
        (set_value("flow.quality", flow_quality="count_only"), "MODERATE"),
        (set_value("flow.quality", flow_quality="divergent"), "WEAK"),
    ],
)
def test_setup_boundaries(base: OpportunityInput, edit: Callable[[dict], None], band: str) -> None:
    assert run(base, edit).setup_band == band


def test_scout_composite_legacy_social_and_technical_copies_never_change_the_setup(
    base: OpportunityInput,
) -> None:
    def scrub(d: dict[str, Any]) -> None:
        for f in d["sources"]["scout"]["diagnostics"]:
            if f["aspect"] == "scout.composite":
                f["value"] = {**f["value"], "score": 0.0, "risk_penalty": 70.0}
            elif f["status"] == "AVAILABLE":
                f["value"] = {"changed": True}

    before, after = decide(base), run(base, scrub)
    assert before.body() | {"input_hash": ""} == after.body() | {"input_hash": ""}


# --- technical / social ----------------------------------------------------------------------------


def test_technical_bands_and_conservative_conflict(base: OpportunityInput) -> None:
    flat = both(set_value("price.snapshot_trend", trend="flat"),
                set_value("price.analyze_trend", label="mixed"))  # fmt: skip
    neutral = run(base, flat)
    assert (neutral.technical_band, neutral.decision) == ("NEUTRAL", "ENTER")  # with STRONG
    nm = run(base, flat, MODERATE)
    assert nm.decision == "WATCH" and "NEED_TECHNICAL_CONFIRMATION" in {
        u.code for u in nm.upgrade_path
    }
    conflict = run(base, set_value("price.analyze_trend", label="downtrend"))
    assert (conflict.technical_band, conflict.decision) == ("CONTRADICTING", "WATCH")
    assert "TECHNICAL_CONFLICT" in codes(conflict.risks)
    half = run(base, set_value("price.analyze_trend", label="mixed"))
    assert half.technical_band == "NEUTRAL"  # one confirming, one neutral: not CONFIRMING
    few = run(base, ref("technical", "NOT_COLLECTED", "analyze_ref"),
              set_value("price.snapshot_trend", snapshots=3))  # fmt: skip
    assert few.technical_band == "NEUTRAL"
    assert run(base, TECH_NONE).technical_band == "UNAVAILABLE"


def test_social_bands(base: OpportunityInput) -> None:
    assert decide(base).social_band == "SUPPORTING"
    assert run(base, set_value("attention.momentum", state="STABLE")).social_band == "NEUTRAL"
    assert run(base, set_value("attention.attribution", exact_mentions=4)).social_band == "NEUTRAL"
    for edit in (set_value("attention.spam", spam_risk="high"),
                 set_value("attention.market_cross", state="UNCONFIRMED_SOCIAL_SPIKE")):  # fmt: skip
        d = run(base, edit)
        assert (d.social_band, d.decision) == ("CONTRADICTING", "WATCH")
        assert d.risk_groups == ("ATTENTION",)
    gone = run(base, ref("social", "NOT_COLLECTED"))
    assert (gone.social_band, gone.decision) == ("UNAVAILABLE", "ENTER")


def test_attention_alone_never_qualifies(base: OpportunityInput) -> None:
    only_social = run(
        base, raw("flow.volume_acceleration", 1.0), raw("flow.trade_acceleration", 1.0),
        raw("liquidity.growth", 0.0), TECH_NONE,
    )  # fmt: skip
    assert only_social.social_band == "SUPPORTING"
    assert only_social.decision != "ENTER"


# --- risk and positive groups ------------------------------------------------------------------------


def test_risks_count_once_per_group(base: OpportunityInput) -> None:
    holders = run(base, set_rule("TOP1_CONCENTRATION", "TRIGGERED"),
                  set_rule("FEW_HOLDERS", "TRIGGERED"), set_rule("LARGE_HOLDER_EXIT", "TRIGGERED"))  # fmt: skip
    assert holders.risk_groups == ("HOLDERS",) and holders.decision == "ENTER"
    assert {"TOP1_CONCENTRATION", "FEW_HOLDERS", "LARGE_HOLDER_EXIT"} <= codes(holders.risks)
    two = run(base, set_rule("TOP1_CONCENTRATION", "TRIGGERED"),
              set_rule("LOW_LIQUIDITY", "TRIGGERED", "medium"))  # fmt: skip
    assert two.risk_groups == ("HOLDERS", "LIQUIDITY") and two.risk_tier == "ELEVATED_2_PLUS"
    assert two.decision == "WATCH" and "TOO_MANY_RISK_GROUPS" in {u.code for u in two.upgrade_path}


def test_a_blocking_rule_isnt_also_counted_as_a_risk(base: OpportunityInput) -> None:
    d = run(base, set_rule("TOP10_CONCENTRATION", "TRIGGERED", "high"),
            set_rule("TOP1_CONCENTRATION", "TRIGGERED"))  # fmt: skip
    assert "TOP10_CONCENTRATION" in codes(d.blockers)
    assert "TOP10_CONCENTRATION" not in codes(d.risks) and d.risk_groups == ("HOLDERS",)


def test_scout_copies_of_safety_facts_add_no_group(base: OpportunityInput) -> None:
    def legacy(d: dict[str, Any]) -> None:
        for f in d["sources"]["scout"]["diagnostics"]:
            if f["aspect"] in ("holders.top10", "authority.mint", "liquidity.collapse"):
                f.update(status="AVAILABLE", reason=None, value={"code": "legacy", "flagged": True})

    d = run(base, legacy)
    assert d.decision == "ENTER" and d.risk_groups == () and not d.blockers


def test_scout_flow_risks_and_news(base: OpportunityInput) -> None:
    d = run(base, flag("flow.volume_to_liquidity"))
    assert d.risk_groups == ("FLOW",) and d.decision == "ENTER"
    bad = news_findings()
    bad["reports"][0]["stories"][0]["sentiment"] = "bearish"

    def news(dd: dict[str, Any]) -> None:
        f = _fact(dd, "news.impact")
        f["value"] = {"stories": [{**s, "sentiment": "bearish"} for s in f["value"]["stories"]]}

    n = run(base, news)
    assert n.risk_groups == ("NEWS",) and n.decision == "ENTER"  # never a veto or SKIP alone
    risk = next(r for r in n.risks if r.code == "NEWS_HIGH_IMPACT_BEARISH")
    assert "language-model generated" in risk.message
    assert "NEWS" not in n.positive_groups
    assert run(base, news, set_rule("FEW_HOLDERS", "TRIGGERED")).decision == "WATCH"


def test_positive_groups_require_flow_and_a_second_group(base: OpportunityInput) -> None:
    no_flow = run(base, raw("flow.volume_acceleration", 1.0), raw("flow.trade_acceleration", 1.0))
    assert "FLOW" not in no_flow.positive_groups and no_flow.decision == "WATCH"
    assert "NEED_FLOW_CONFIRMATION" in {u.code for u in no_flow.upgrade_path}
    flat = both(set_value("price.snapshot_trend", trend="flat"),
                set_value("price.analyze_trend", label="mixed"))  # fmt: skip
    alone = run(base, raw("liquidity.growth", 5.0), flat, ref("social", "NOT_COLLECTED"))
    assert alone.positive_groups == ("FLOW",) and alone.decision == "WATCH"
    assert "NEED_SECOND_POSITIVE_GROUP" in {u.code for u in alone.upgrade_path}
    # Safety NOT_TRIGGERED rules are never positive evidence.
    assert not {"HOLDERS", "AUTHORITY", "IDENTITY", "MARKET_STRUCTURE"} & set(
        decide(base).positive_groups
    )


# --- monotonicity ---------------------------------------------------------------------------------

RISKS: list[Callable[[dict], None]] = [
    *(escalate(r, "high") for r in SAFETY_RULES if r != "TOKEN_2022_EXTENSION_RISK"),
    *(escalate(r, "medium") for r in SAFETY_RULES if r != "TOKEN_2022_EXTENSION_RISK"),
    flag("flow.distribution"), flag("flow.thin_pump"), flag("flow.volume_to_liquidity"),
    set_value("flow.quality", flow_quality="divergent"),
    set_value("price.analyze_trend", label="downtrend"),
    set_value("attention.spam", spam_risk="high"),
    set_value("scout.data_status", data_status="STALE_CARRIED"),
]  # fmt: skip
REMOVALS: list[Callable[[dict], None]] = [
    *(unavailable(a) for a in ("flow.volume_acceleration", "flow.trade_acceleration",
                               "flow.buy_pressure", "flow.quality", "liquidity.growth",
                               "liquidity.stability", "earliness.move_already_made",
                               "earliness.activity_regime", "price.snapshot_trend",
                               "price.analyze_trend", "attention.momentum",
                               "attention.attribution")),
    TECH_NONE, ref("scout", "STALE", age_seconds=1201.0), ref("scout", "UNAVAILABLE"),
    buy_change(None),
    # Required Safety degradations.
    degrade_holders_to_partial,
    no_holders(), no_holders("NOT_COLLECTED"), no_holders("PROVIDER_UNAVAILABLE", "UNDETERMINED"),
    no_holders("UNKNOWN", "UNDETERMINED"), no_market(), no_market("NOT_COLLECTED"),
    no_market("PROVIDER_UNAVAILABLE", "UNDETERMINED"),
    section("authority", mint_authority={"status": "UNAVAILABLE"}),
    ref("safety", "STALE", age_seconds=3700.0), ref("safety", "UNAVAILABLE"),
    ref("safety", "INCOMPATIBLE"),
]  # fmt: skip


@pytest.fixture(scope="module")
def starts(base: OpportunityInput) -> list[OpportunityInput]:
    """ENTER, several WATCH and SKIP starting points."""
    return [
        base,
        vary(base, MODERATE),
        vary(base, set_rule("TOP1_CONCENTRATION", "TRIGGERED", "high")),
        vary(base, set_rule("FEW_HOLDERS", "UNDETERMINED")),
        vary(base, set_value("scout.stage", stage="STEADY")),
        vary(base, set_rule("FREEZE_AUTHORITY_ACTIVE", "TRIGGERED", "high")),
        vary(base, no_holders()),  # SKIP: required Safety not assessed
        vary(
            base,
            both(
                section("holders", status="PARTIAL", reason="page cap"), undetermine("FEW_HOLDERS")
            ),
        ),  # fmt: skip
    ]


def _rank(inp: OpportunityInput) -> int:
    return DECISION_ORDER[decide(inp).decision]


def test_adding_a_risk_never_upgrades(starts: list[OpportunityInput]) -> None:
    for s in starts:
        for edit in RISKS:
            assert _rank(vary(s, edit)) <= _rank(s)


def test_removing_positive_evidence_never_upgrades(starts: list[OpportunityInput]) -> None:
    for s in starts:
        for edit in REMOVALS:
            assert _rank(vary(s, edit)) <= _rank(s)


@pytest.mark.parametrize("rule_id", [r for r in SAFETY_RULES if r != "TOKEN_2022_EXTENSION_RISK"])
def test_rule_state_escalations_never_upgrade(starts: list[OpportunityInput], rule_id: str) -> None:
    for s in starts:
        if _fact(_dump(s), SAFETY_RULES[rule_id])["value"]["state"] != "NOT_TRIGGERED":
            continue  # the property starts from NOT_TRIGGERED
        undetermined = vary(s, undetermine(rule_id))
        assert _rank(undetermined) <= _rank(s)  # NOT_TRIGGERED -> UNDETERMINED
        # Holder rules can stay on the same (lower-bound) evidence at either severity. The
        # provider-exact market rules can only become TRIGGERED once their exact field is
        # known (new evidence): see test_exact_market_triggers_arrive_with_new_evidence.
        sevs = ("high",) if rule_id in EXACT_FIELD_RULES else ("medium", "high")
        for sev in sevs:
            triggered = vary(undetermined, set_rule(rule_id, "TRIGGERED", sev))
            assert _rank(triggered) <= _rank(undetermined)  # UNDETERMINED -> TRIGGERED


EXACT_FIELD_RULES = ("LOW_LIQUIDITY", "VERY_NEW_POOL")


@pytest.mark.parametrize("rule_id", EXACT_FIELD_RULES)
def test_exact_market_triggers_arrive_with_new_evidence(
    base: OpportunityInput, rule_id: str
) -> None:
    """Safety V2 leaves LOW_LIQUIDITY / VERY_NEW_POOL UNDETERMINED only when the provider
    field is missing; TRIGGERED needs the exact value. UNDETERMINED (WATCH: a possible veto /
    blocker) -> TRIGGERED medium (an exact ordinary risk) is evidence being added, not a
    degradation, and may move WATCH -> ENTER; TRIGGERED high never upgrades."""
    section_, _, name = EVIDENCE[rule_id].partition(".")
    missing = both(set_rule(rule_id, "UNDETERMINED"),
                   section(section_, **{name: {"status": "UNAVAILABLE", "value": None}}))  # fmt: skip
    undetermined = decide(vary(base, missing))
    assert undetermined.decision == "WATCH"
    assert f"UNRESOLVED_{rule_id}" in codes(undetermined.blockers)
    medium = run(base, set_rule(rule_id, "TRIGGERED", "medium"))  # field AVAILABLE
    assert medium.decision == "ENTER" and medium.risk_tier == "ELEVATED_1"
    high = run(base, set_rule(rule_id, "TRIGGERED", "high"))
    assert DECISION_ORDER[high.decision] <= DECISION_ORDER[undetermined.decision]


def test_adding_a_veto_always_skips(starts: list[OpportunityInput]) -> None:
    for s in starts:
        for rule_id in VETO_RULES:
            assert decide(vary(s, set_rule(rule_id, "TRIGGERED", "high"))).decision == "SKIP"


def test_strengthening_a_positive_signal_never_downgrades(starts: list[OpportunityInput]) -> None:
    for s in starts:
        for aspect, lo, hi in (("flow.volume_acceleration", 1.2, 3.0),
                               ("flow.trade_acceleration", 1.2, 3.0),
                               ("liquidity.growth", 5.0, 30.0),
                               ("earliness.activity_regime", 0.9, 2.0)):  # fmt: skip
            weaker = vary(s, raw(aspect, lo))
            assert _rank(vary(weaker, raw(aspect, hi))) >= _rank(weaker)


# --- explanations, determinism, integrity, isolation ----------------------------------------------


def test_every_reason_is_sourced_and_every_watch_explains(starts: list[OpportunityInput]) -> None:
    for s in starts:
        d = decide(s)
        for r in (*d.vetoes, *d.blockers, *d.positive_reasons, *d.risks):
            assert r.source and r.message and r.status
            if r.source != "engine":
                assert r.evidence_paths and r.source_record_id
        if d.decision == "WATCH":
            assert d.upgrade_path
        else:
            assert d.upgrade_path == ()


def test_the_body_holds_no_execution_instruction(base: OpportunityInput) -> None:
    keys: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            keys.update(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(decide(base).body())
    forbidden = {"order", "size", "quantity", "amount", "price_limit", "stop_loss", "take_profit",
                 "transaction", "wallet", "slippage", "position"}  # fmt: skip
    assert not keys & forbidden


def test_identical_inputs_give_identical_bodies_regardless_of_fact_order(
    base: OpportunityInput,
) -> None:
    a, b = decide(base), decide(OpportunityInput.model_validate_json(base.canonical_json()))
    assert a.canonical_json() == b.canonical_json() and a.decision_hash() == b.decision_hash()

    def reverse(d: dict[str, Any]) -> None:
        for path in LISTS:
            node: Any = d["sources"]
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = list(reversed(node[path[-1]]))

    c = run(base, reverse)
    assert c.body() | {"input_hash": ""} == a.body() | {"input_hash": ""}
    json.loads(a.canonical_json())


@pytest.mark.parametrize(
    "update",
    [
        {"canonical_id": "solana:" + "1" * 44},
        {"schema_version": "opportunity.input.v0"},
        {"decision_at": T.replace(tzinfo=None)},
        {"origin": "LIVE"},
    ],
)
def test_integrity_failures_raise_not_decide(base: OpportunityInput, update: dict) -> None:
    with pytest.raises(OpportunityDecisionError):
        decide(base.model_copy(update=update))


def test_future_or_duplicated_facts_raise(base: OpportunityInput) -> None:
    s = base.sources
    future = s.model_copy(update={"social": s.social.model_copy(update={
        "ref": s.social.ref.model_copy(update={"observed_at": T + timedelta(seconds=1)})})})  # fmt: skip
    with pytest.raises(OpportunityDecisionError, match="after"):
        decide(base.model_copy(update={"sources": future}))
    dup = s.model_copy(update={"safety": s.safety.model_copy(update={
        "facts": (*s.safety.facts, s.safety.facts[0])})})  # fmt: skip
    with pytest.raises(OpportunityDecisionError, match="ownership"):
        decide(base.model_copy(update={"sources": dup}))
    fresh_lie = s.model_copy(update={"scout": s.scout.model_copy(update={
        "ref": s.scout.ref.model_copy(update={"age_seconds": 5000.0})})})  # fmt: skip
    with pytest.raises(OpportunityDecisionError, match="freshness"):
        decide(base.model_copy(update={"sources": fresh_lie}))
    bad_rule = vary(base, set_value(SAFETY_RULES["TOP1_CONCENTRATION"], rule="TOP10_CONCENTRATION"))
    with pytest.raises(OpportunityDecisionError, match="valid"):
        decide(bad_rule)


def test_the_engine_is_pure(base: OpportunityInput, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("I/O")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(sqlite3, "connect", refuse)
    monkeypatch.setattr("builtins.open", refuse)
    assert decide(base).decision == "ENTER"
    src = Path(engine.__file__).read_text() + Path(engine.__file__).with_name(
        "decision_models.py").read_text()  # fmt: skip
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    tokens = {t for ln in imports for t in re.split(r"[ .,()]+", ln)}
    for word in ("sqlite3", "socket", "loaders", "service", "normalize", "evidence_archive",
                 "httpx", "anthropic", "radar", "shadow", "scout", "safety_v2", "execution",
                 "opportunity", "time", "random", "uuid", "os"):  # fmt: skip
        assert word not in tokens, word
    assert "datetime.now" not in src and "time.time" not in src


def test_a_real_safety_v2_snapshot_decides_end_to_end(tmp_path: Path) -> None:
    import asyncio

    from tests.opportunity_model_fakes import CID, OTHER
    from tests.safety_v2_fakes import MINT, FakeRpc, make_service, mint_value

    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value(mint_authority=OTHER)}))
    svc.add_target(MINT)
    snap = asyncio.run(svc.snapshot(CID))
    arch = Archive(tmp_path / "evidence.sqlite3")
    when = snap.as_of + timedelta(minutes=1)
    arch.add(scout_record(when, cand=candidate(when, risk_flags=[])))
    arch.close()
    d = decide(build(arch.path, tmp_path / "safety.sqlite3", when))
    # A mint-only snapshot: Safety coverage is COMPLETE (holder / market rules out of scope),
    # but Opportunity's required HOLDERS and MARKET domains were never assessed.
    assert d.safety_entry_readiness == {"IDENTITY": "READY", "HOLDERS": "NOT_READY",
                                        "MARKET": "NOT_READY"}  # fmt: skip
    assert d.decision == "SKIP" and d.skip_basis == ("REQUIRED_SAFETY_NOT_READY",)
    assert {"HOLDER_SAFETY_NOT_READY", "MARKET_SAFETY_NOT_READY"} == codes(d.ineligible)
    assert "MINT_AUTHORITY_ACTIVE" in codes(d.blockers) and not d.vetoes


# --- required Safety domains (entry readiness) ------------------------------------------------


def _skipped_for(d: OpportunityDecision, *missing: str) -> None:
    assert (d.decision, d.quality) == ("SKIP", "INSUFFICIENT")
    assert d.skip_basis == ("REQUIRED_SAFETY_NOT_READY",) and not d.vetoes
    assert codes(d.ineligible) == set(missing)
    for r in d.ineligible:
        assert "not evidence the token is malicious" in r.message
    assert d.upgrade_path == ()


def test_a_no_holder_observation_skips(base: OpportunityInput) -> None:
    d = run(base, no_holders())
    _skipped_for(d, "HOLDER_SAFETY_NOT_READY")
    assert d.setup_band == "STRONG" and d.technical_band == "CONFIRMING"  # all else perfect


def test_b_no_market_observation_skips(base: OpportunityInput) -> None:
    _skipped_for(run(base, no_market()), "MARKET_SAFETY_NOT_READY")


def test_c_neither_domain_skips(base: OpportunityInput) -> None:
    _skipped_for(run(base, no_holders(), no_market()),
                 "HOLDER_SAFETY_NOT_READY", "MARKET_SAFETY_NOT_READY")  # fmt: skip


def test_d_safety_coverage_complete_is_not_entry_readiness(base: OpportunityInput) -> None:
    edited = vary(base, no_holders())
    assert _fact(_dump(edited), "safety.coverage")["value"]["coverage"] == "COMPLETE"
    _skipped_for(decide(edited), "HOLDER_SAFETY_NOT_READY")


def test_e_holder_provider_unavailable_skips(base: OpportunityInput) -> None:
    _skipped_for(run(base, no_holders("PROVIDER_UNAVAILABLE", "UNDETERMINED")),
                 "HOLDER_SAFETY_NOT_READY")  # fmt: skip
    _skipped_for(run(base, no_holders("UNKNOWN", "UNDETERMINED")), "HOLDER_SAFETY_NOT_READY")
    _skipped_for(run(base, no_holders("NOT_COLLECTED")), "HOLDER_SAFETY_NOT_READY")


def test_f_market_provider_unavailable_skips(base: OpportunityInput) -> None:
    _skipped_for(run(base, no_market("PROVIDER_UNAVAILABLE", "UNDETERMINED")),
                 "MARKET_SAFETY_NOT_READY")  # fmt: skip
    _skipped_for(run(base, no_market("NOT_COLLECTED")), "MARKET_SAFETY_NOT_READY")


def test_g_partial_holder_evidence_continues_through_partial_and_unresolved_logic(
    base: OpportunityInput,
) -> None:
    d = run(base, section("holders", status="PARTIAL", reason="page cap"),
            undetermine("FEW_HOLDERS"), set_rule("TOP10_CONCENTRATION", "TRIGGERED", "medium"),
            partial_evidence("TOP10_CONCENTRATION"))  # fmt: skip
    assert d.safety_entry_readiness["HOLDERS"] == "READY_PARTIAL"
    assert d.decision == "WATCH" and d.skip_basis == () and not d.ineligible
    assert d.quality == "PARTIAL"
    assert "UNRESOLVED_TOP10_CONCENTRATION" in codes(d.blockers)
    assert "NEED_COMPLETE_HOLDER_EVIDENCE" in codes(d.missing_evidence)


def test_h_closure_out_of_scope_on_a_healthy_reported_pool_is_ready(base: OpportunityInput) -> None:
    d = run(base, set_rule("MARKET_CLOSED_ON_CHAIN", "OUT_OF_SCOPE"))
    assert d.safety_entry_readiness["MARKET"] == "READY" and d.decision == "ENTER"


def test_i_no_previous_holder_observation_keeps_holders_ready(base: OpportunityInput) -> None:
    for r in CHANGE_RULES:
        assert _fact(_dump(base), SAFETY_RULES[r])["value"]["state"] == "OUT_OF_SCOPE"
    d = decide(base)
    assert d.safety_entry_readiness["HOLDERS"] == "READY" and d.decision == "ENTER"


def test_j_unconfigured_creator_source_is_optional(base: OpportunityInput) -> None:
    d = run(base, section("creator", source_status="NOT_CONFIGURED"),
            set_rule("VERIFIED_DEPLOYER_HOLDS_SUPPLY", "OUT_OF_SCOPE"))  # fmt: skip
    assert d.safety_entry_readiness == {"IDENTITY": "READY", "HOLDERS": "READY", "MARKET": "READY"}
    assert d.decision == "ENTER"


def test_a_successful_no_pool_market_is_assessed_and_vetoed(base: OpportunityInput) -> None:
    d = run(base, section("market", status="AVAILABLE"),
            set_rule("NO_ELIGIBLE_MARKET", "TRIGGERED", "high"))  # fmt: skip
    assert d.safety_entry_readiness["MARKET"] == "READY"
    assert d.decision == "SKIP" and d.skip_basis == ("HARD_VETO",)


def test_removing_more_safety_never_lifts_a_required_safety_skip(base: OpportunityInput) -> None:
    start = vary(base, no_holders())
    for edit in (no_market(), ref("safety", "UNAVAILABLE"), ref("safety", "INCOMPATIBLE"),
                 ref("safety", "STALE", age_seconds=3700.0),
                 unverified()):  # fmt: skip
        assert decide(vary(start, edit)).decision == "SKIP"


# --- optional sources, Scout absences --------------------------------------------------------------


def test_optional_social_and_news_are_different_evidence_states_not_degradations(
    base: OpportunityInput,
) -> None:
    """Contract B: Social / News are optional. Their absence adds zero positive evidence and
    is listed as missing, but it doesn't disprove a negative Social / News state another
    observation had: CONTRADICTING social (WATCH) and no social at all (here ENTER) are
    different evidence states, so no monotonic relation between them is claimed."""
    contra = run(base, set_value("attention.spam", spam_risk="high"))
    absent = run(base, ref("social", "NOT_COLLECTED"))
    assert contra.decision == "WATCH" and absent.decision == "ENTER"
    assert "ATTENTION" not in absent.positive_groups
    assert "SOCIAL_NOT_AVAILABLE" in codes(absent.missing_evidence)
    assert absent.quality == decide(base).quality


def test_absent_scout_risk_flags_create_no_positive_reason(base: OpportunityInput) -> None:
    view = {f["aspect"]: f for f in _dump(base)["sources"]["scout"]["market_view"]["facts"]}
    for aspect in ("flow.thin_pump", "flow.distribution", "flow.volume_to_liquidity"):
        assert view[aspect]["status"] == "NOT_AVAILABLE"
    d = decide(base)
    paths = {p for r in d.positive_reasons for p in r.evidence_paths}
    assert not [p for p in paths if "risk_flags" in p]
    assert not {"SCOUT_THIN_MARKET_PUMP", "SCOUT_DISTRIBUTION"} & codes(d.positive_reasons)
