"""Opportunity Model V1 (O1): the canonical, read-only, causally valid input layer.
Offline: local archive / Safety fixtures only; nothing reaches a network."""

import asyncio
import hashlib
import socket
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.opportunity_model_fakes import (
    CID,
    OTHER,
    SAFETY_FLAGS,
    VARIANT,
    Archive,
    at,
    build,
    candidate,
    decision_record,
    momentum,
    news_findings,
    rule,
    safety_body,
    safety_db,
    scout_record,
    social_record,
    technical_findings,
)
from tests.safety_v2_fakes import MINT, FakeRpc, make_service, mint_value
from upscale.services.evidence_archive.store import EvidenceRecord, Kind
from upscale.services.opportunity_model import normalize
from upscale.services.opportunity_model.config import FRESHNESS
from upscale.services.opportunity_model.loaders import ArchiveReader, SafetyReader, SafetyRow
from upscale.services.opportunity_model.models import (
    Fact,
    OpportunityCausalityError,
    OpportunityIdentityError,
    OpportunityInput,
    OwnershipError,
)
from upscale.services.opportunity_model.ownership import OWNERSHIP
from upscale.services.opportunity_model.service import OpportunityInputService, build_input

T = at()


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Archive, Path, Path]:
    arch = Archive(tmp_path / "evidence.sqlite3")
    return arch, arch.path, tmp_path / "safety.sqlite3"


def _facts(inp: OpportunityInput, layer: str | None = None) -> dict[str, Fact]:
    """Owned (non-diagnostic) facts by aspect."""
    return {f.aspect: f for f in inp.facts() if f.role != "DIAGNOSTIC"
            and (layer is None or f.layer == layer)}  # fmt: skip


def _std(arch: Archive, safety: Path, when: Any = None) -> None:
    when = when or T
    arch.add(scout_record(when))
    arch.add(social_record(when))
    arch.add(decision_record(when))
    safety_db(safety, [(when, "4", safety_body(when))])


# --- causality ----------------------------------------------------------------------------------


def test_sources_exactly_at_decision_time_are_used(paths: Any) -> None:
    arch, a, s = paths
    _std(arch, s)
    inp = build(a, s)
    src = inp.sources
    for ref in (src.scout.ref, src.technical.ref, src.technical.analyze_ref, src.social.ref,
                src.news.ref, src.safety.ref):  # fmt: skip
        assert ref.status == "AVAILABLE", ref
        assert ref.age_seconds == 0.0
    assert src.safety.ref.as_of == T and src.safety.ref.snapshot_id == 1


def test_later_records_are_never_selected_and_dont_leak_backward(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(at(-5)))
    safety_db(s, [(at(-5), "4", safety_body(at(-5)))])
    before = build(a, s)
    later = candidate(at(5), stage="FADING")
    arch.add(scout_record(at(5), cand=later))
    arch.add(social_record(at(1)))
    arch.add(decision_record(at(1)))
    safety_db(s, [(at(1), "4", safety_body(at(1), flags=[rule("FREEZE_AUTHORITY_ACTIVE",
                                                             "TRIGGERED")]))])  # fmt: skip
    after = build(a, s)
    assert after.input_hash() == before.input_hash()
    assert after.sources.scout.ref.observed_at == at(-5)
    assert after.sources.social.ref.status == "NOT_COLLECTED"
    assert after.sources.safety.ref.as_of == at(-5)


class _ForcedArchive:
    def __init__(self, record: EvidenceRecord | None, kind: Kind = "scout"):
        self.record, self.kind = record, kind

    def latest(self, kind: Kind, asset_id: str, until: Any) -> EvidenceRecord | None:
        return self.record if kind == self.kind else None


class _ForcedSafety:
    def __init__(self, row: SafetyRow | None):
        self.row = row

    def latest_snapshot(self, canonical_id: str, until: Any) -> SafetyRow | None:
        return self.row


def _stored(arch: Archive, kind: Kind) -> EvidenceRecord:
    reader = ArchiveReader(arch.path)
    try:
        rec = reader.latest(kind, CID, at(60))
    finally:
        reader.close()
    assert rec is not None
    return rec


@pytest.mark.parametrize("kind", ["scout", "social", "decision"])
def test_a_forced_future_archive_record_raises(paths: Any, kind: Kind) -> None:
    arch, _, _ = paths
    maker = {"scout": scout_record, "social": social_record, "decision": decision_record}[kind]
    arch.add(maker(at(seconds=1)))
    future = _stored(arch, kind)
    with pytest.raises(OpportunityCausalityError):
        build_input(CID, T, "HISTORICAL_REPLAY", _ForcedArchive(future, kind), _ForcedSafety(None))


def test_safety_as_of_after_decision_time_is_not_selected_and_forced_raises(paths: Any) -> None:
    arch, a, s = paths
    safety_db(s, [(at(seconds=1), "4", safety_body(at(seconds=1)))])
    assert build(a, s).sources.safety.ref.status == "NOT_COLLECTED"
    reader = SafetyReader(s)
    row = reader.latest_snapshot(CID, at(10))
    reader.close()
    with pytest.raises(OpportunityCausalityError):
        build_input(CID, T, "HISTORICAL_REPLAY", _ForcedArchive(None), _ForcedSafety(row))
    with pytest.raises(OpportunityCausalityError):
        normalize.safety(row, CID, MINT, T, FRESHNESS)


def test_embedded_scout_market_time_after_its_decision_raises(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(at(-2), market_at=at(-1)))  # market evidence after its decision
    with pytest.raises(OpportunityCausalityError):
        build(a, s)


def test_live_forward_refuses_a_record_archived_after_the_decision(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(at(-1)), archived=at(seconds=30))  # observed in time, written later
    assert build(a, s).sources.scout.ref.status == "AVAILABLE"  # replay: observed in time
    with pytest.raises(OpportunityCausalityError):
        build(a, s, origin="LIVE_FORWARD")


def test_live_service_uses_its_clock_and_replay_the_given_time(paths: Any) -> None:
    arch, a, s = paths
    _std(arch, s, at(-1))
    arch.close()
    svc = OpportunityInputService(str(a), str(s), clock=lambda: T)
    live = svc.build_live(CID)
    replay = svc.build_replay(CID, T)
    assert (live.origin, live.decision_at) == ("LIVE_FORWARD", T)
    assert replay.origin == "HISTORICAL_REPLAY"
    assert live.sources.model_dump() == replay.sources.model_dump()


# --- freshness ----------------------------------------------------------------------------------


def test_scout_freshness_boundary(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(at(-20)))
    fresh = build(a, s).sources
    assert fresh.scout.ref.status == "AVAILABLE" and fresh.scout.ref.age_seconds == 1200
    assert fresh.technical.ref.status == "AVAILABLE"
    stale = build(a, s, at(seconds=1)).sources
    assert stale.scout.ref.status == "STALE" and "limit 1200s" in str(stale.scout.ref.reason)
    assert stale.technical.ref.status == "STALE"  # Scout's TechnicalContext inherits it
    assert stale.scout.market_view.facts == fresh.scout.market_view.facts  # kept, marked


def test_scout_age_is_measured_from_its_market_evidence(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(at(-1), market_at=at(-25)))  # a carried, older market reading
    assert build(a, s).sources.scout.ref.status == "STALE"


@pytest.mark.parametrize("source", ["safety", "social", "analyze"])
def test_sixty_minute_freshness_boundaries(paths: Any, source: str) -> None:
    arch, a, s = paths
    if source == "safety":
        safety_db(s, [(at(-60), "4", safety_body(at(-60)))])
    elif source == "social":
        arch.add(social_record(at(-60)))
    else:
        arch.add(decision_record(at(-60)))

    def refs(inp: OpportunityInput) -> list[Any]:
        src = inp.sources
        return {"safety": [src.safety.ref], "social": [src.social.ref],
                "analyze": [src.technical.analyze_ref, src.news.ref]}[source]  # fmt: skip

    assert {r.status for r in refs(build(a, s))} == {"AVAILABLE"}
    assert {r.status for r in refs(build(a, s, at(seconds=1)))} == {"STALE"}


# --- identity -----------------------------------------------------------------------------------


def test_case_variant_in_a_scout_record_is_an_identity_error(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(T, cand=candidate(T, address=VARIANT)))
    with pytest.raises(OpportunityIdentityError):
        build(a, s)


def test_case_variant_or_other_mint_in_safety_is_an_identity_error(paths: Any) -> None:
    arch, a, s = paths
    safety_db(s, [(T, "4", safety_body(T, mint=VARIANT))])
    with pytest.raises(OpportunityIdentityError):
        build(a, s)


def test_another_mint_in_social_is_an_identity_error(paths: Any) -> None:
    arch, a, s = paths
    arch.add(social_record(T, m=momentum(T, cid=f"solana:{OTHER}")))
    with pytest.raises(OpportunityIdentityError):
        build(a, s)


def test_another_token_in_analyze_technical_is_an_identity_error(paths: Any) -> None:
    arch, a, s = paths
    arch.add(decision_record(T, technical=technical_findings(canonical_id=f"solana:{OTHER}")))
    with pytest.raises(OpportunityIdentityError):
        build(a, s)


def test_a_case_variant_request_is_another_token(paths: Any) -> None:
    arch, a, s = paths
    _std(arch, s)
    inp = build(a, s, cid=f"solana:{VARIANT}")
    assert inp.sources.scout.ref.status == "NOT_COLLECTED"
    assert inp.sources.safety.ref.status == "NOT_COLLECTED"


def test_malformed_solana_identity_is_rejected(paths: Any) -> None:
    _, a, s = paths
    for bad in ("solana:not-a-mint", "solana:", MINT, f"solana: {MINT}"):
        with pytest.raises(OpportunityIdentityError):
            build(a, s, cid=bad)


def test_evm_tokens_are_not_supported_and_nothing_is_read() -> None:
    class Boom:
        def latest(self, *a: Any) -> None:
            raise AssertionError("read")

        def latest_snapshot(self, *a: Any) -> None:
            raise AssertionError("read")

    inp = build_input("base:0x" + "ab" * 20, T, "HISTORICAL_REPLAY", Boom(), Boom())
    src = inp.sources
    refs = [src.scout.ref, src.technical.ref, src.technical.analyze_ref, src.social.ref,
            src.news.ref, src.safety.ref]  # fmt: skip
    assert {r.status for r in refs} == {"NOT_SUPPORTED"}
    assert not inp.scoring_facts() or all(f.status == "NOT_AVAILABLE" for f in inp.facts())


# --- Safety V2 ----------------------------------------------------------------------------------


def _real_safety(tmp_path: Path) -> tuple[Path, Any]:
    svc, clock, _ = make_service(tmp_path, FakeRpc({MINT: mint_value(mint_authority=OTHER)}))
    svc.add_target(MINT)
    res = asyncio.run(svc.snapshot(CID))
    return tmp_path / "safety.sqlite3", res


def test_a_real_safety_v4_snapshot_is_loaded_as_stored(tmp_path: Path) -> None:
    safety, res = _real_safety(tmp_path)
    digest = hashlib.sha256(safety.read_bytes()).hexdigest()
    arch = Archive(tmp_path / "evidence.sqlite3")
    arch.close()
    inp = build(arch.path, safety, res.as_of + timedelta(minutes=5))
    sf = inp.sources.safety
    assert sf.ref.status == "AVAILABLE" and sf.ref.rules_version == "4"
    assert (sf.ref.snapshot_id, sf.ref.as_of, sf.ref.body_hash) == (
        res.snapshot_id,
        res.as_of,
        res.body_hash,
    )
    states = {r.id: r.state for r in sf.rules}
    for f in res.body["flags"]:
        assert states[f["id"]] == f["outcome"]  # tri-state preserved, never a boolean
    for o in res.body["coverage"]["out_of_scope"]:
        if o["id"] not in {f["id"] for f in res.body["flags"]}:
            assert states[o["id"]] == "OUT_OF_SCOPE"
    assert states["MINT_AUTHORITY_ACTIVE"] == "TRIGGERED"
    for key in ("identity", "authority", "holders", "market", "creator", "changes", "coverage"):
        assert getattr(sf, key) == res.body[key]
    assert _facts(inp)["authority.mint"].value["state"] == "TRIGGERED"  # type: ignore[index]
    assert hashlib.sha256(safety.read_bytes()).hexdigest() == digest  # read-only


@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_older_safety_rules_versions_are_incompatible(paths: Any, version: str) -> None:
    _, a, s = paths
    safety_db(s, [(T, version, safety_body(T, rules_version=version))])
    sf = build(a, s).sources.safety
    assert sf.ref.status == "INCOMPATIBLE" and sf.ref.rules_version == version
    assert sf.rules == () and sf.holders is None
    assert all(f.status == "NOT_AVAILABLE" for f in sf.facts)


def test_safety_rule_states_are_preserved(paths: Any) -> None:
    arch, a, s = paths
    safety_db(s, [(T, "4", safety_body(T))])
    sf = build(a, s).sources.safety
    states = {r.id: r.state for r in sf.rules}
    assert states["FEW_HOLDERS"] == "UNDETERMINED"
    assert states["TOP10_CONCENTRATION"] == "TRIGGERED"
    assert states["LARGE_HOLDER_EXIT"] == "OUT_OF_SCOPE"
    assert states["TOKEN_2022_EXTENSION_RISK"] == "NOT_SUPPORTED"
    assert states["VERIFIED_DEPLOYER_HOLDS_SUPPLY"] == "NOT_TRIGGERED"
    few = next(r for r in sf.rules if r.id == "FEW_HOLDERS")
    assert few.needs == "a complete holder scan"
    rising = _facts(build(a, s))["holders.concentration_rising"]
    assert rising.status == "AVAILABLE" and rising.value["state"] == "OUT_OF_SCOPE"  # type: ignore[index]


@pytest.mark.parametrize(
    "where", ["not_supported", "out_of_scope", "flags"], ids=["pre-fix", "scoped", "twice"]
)
def test_contradictory_safety_rule_metadata_is_incompatible_not_reconciled(
    paths: Any, where: str
) -> None:
    """A rule evaluated and also listed elsewhere (e.g. a v4 body built before the Safety
    fix that listed VERIFIED_DEPLOYER_HOLDS_SUPPLY as not_supported) is never resolved by
    picking a side."""
    _, a, s = paths
    body = safety_body(T)
    dup = {"id": "VERIFIED_DEPLOYER_HOLDS_SUPPLY", "reason": "later phase"}
    if where == "flags":
        body["flags"].append(rule("VERIFIED_DEPLOYER_HOLDS_SUPPLY", "TRIGGERED", "medium"))
    else:
        body["coverage"][where].append(dup)
    safety_db(s, [(T, "4", body)])
    sf = build(a, s).sources.safety
    assert sf.ref.status == "INCOMPATIBLE"
    assert "VERIFIED_DEPLOYER_HOLDS_SUPPLY is reported as" in str(sf.ref.reason)
    assert sf.rules == () and all(f.status == "NOT_AVAILABLE" for f in sf.facts)


def test_safety_body_failing_its_hash_or_unknown_rules_is_not_used(paths: Any) -> None:
    _, a, s = paths
    safety_db(s, [(T, "4", safety_body(T, flags=[*SAFETY_FLAGS, rule("NEW_RULE", "TRIGGERED")]))])
    assert build(a, s).sources.safety.ref.status == "INCOMPATIBLE"
    conn = sqlite3.connect(s)
    conn.execute("UPDATE safety_snapshots SET body_hash = 'x'")
    conn.commit()
    conn.close()
    assert build(a, s).sources.safety.ref.status == "UNAVAILABLE"


def test_missing_stores_are_unavailable_not_empty(tmp_path: Path) -> None:
    inp = build(tmp_path / "none.sqlite3", tmp_path / "none2.sqlite3")
    src = inp.sources
    assert {src.scout.ref.status, src.social.ref.status, src.news.ref.status,
            src.safety.ref.status} == {"UNAVAILABLE"}  # fmt: skip
    assert not (tmp_path / "none.sqlite3").exists() and not (tmp_path / "none2.sqlite3").exists()


# --- Scout ownership -----------------------------------------------------------------------------

RETAINED = {
    "flow.volume_acceleration", "flow.trade_acceleration", "flow.buy_pressure", "flow.quality",
    "flow.thin_pump", "flow.distribution", "flow.volume_to_liquidity",
    "earliness.activity_regime", "earliness.move_already_made", "liquidity.growth",
    "liquidity.stability",
}  # fmt: skip


def test_scout_market_view_keeps_only_market_and_discovery_facts(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(T))
    view = build(a, s).sources.scout.market_view.facts
    assert {f.aspect for f in view} == RETAINED
    assert all(f.role == "SCORING" and f.layer == "scout" for f in view)
    by = {f.aspect: f for f in view}
    assert by["flow.volume_acceleration"].value["measures"]["volume_acceleration"] == 2.4  # type: ignore[index]
    assert by["flow.thin_pump"].value["flagged"] is True  # type: ignore[index]
    assert by["flow.distribution"].status == "NOT_AVAILABLE"  # not flagged != proof of none
    assert by["flow.distribution"].value is None


def test_scout_composite_score_is_diagnostic_only(paths: Any) -> None:
    arch, a, s = paths
    low = candidate(T, scout_momentum={**candidate(T)["scout_momentum"], "score": 5.0,
                                       "base": 4.0, "risk_penalty": 60.0})  # fmt: skip
    arch.add(scout_record(T))
    first = build(a, s).sources.scout
    arch.add(scout_record(T, cand=low))  # same time, later row id: chosen
    second = build(a, s).sources.scout
    assert second.ref.record_id != first.ref.record_id
    assert second.market_view == first.market_view
    composite = [f for f in second.diagnostics if f.aspect == "scout.composite"]
    assert composite[0].role == "DIAGNOSTIC" and composite[0].scoring_owner is None
    assert composite[0].value["score"] == 5.0  # type: ignore[index]


def test_scout_legacy_safety_technical_and_social_are_diagnostics(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(T))
    inp = build(a, s)
    diags = [f for f in inp.sources.scout.diagnostics]
    for aspect in ("authority.mint", "holders.top10", "safety.coverage", "price.snapshot_trend",
                   "price.h1_change", "attention.momentum", "attention.market_cross",
                   "liquidity.collapse", "liquidity.level", "market.pool_age"):  # fmt: skip
        copies = [f for f in diags if f.aspect == aspect]
        assert copies, aspect
        assert all(f.role == "DIAGNOSTIC" and f.scoring_owner != "scout" for f in copies)
    scout_scoring = {f.aspect for f in inp.scoring_facts() if f.layer == "scout"}
    assert not scout_scoring & {a for a, o in OWNERSHIP.items() if o.layer != "scout"}
    assert not {
        f for f in scout_scoring if f.startswith(("price.snap", "attention.", "authority."))
    }


def test_missing_scout_sub_signals_stay_missing(paths: Any) -> None:
    arch, a, s = paths
    c = candidate(T)
    fams = c["scout_momentum"]["families"]
    fams[1] = {**fams[1], "signals": [x for x in fams[1]["signals"] if x["name"] != "growth"]}
    fams[3] = {**fams[3], "available": False, "signals": []}
    arch.add(scout_record(T, cand=c))
    view = {f.aspect: f for f in build(a, s).sources.scout.market_view.facts}
    assert view["liquidity.growth"].status == "NOT_AVAILABLE"
    assert view["earliness.activity_regime"].status == "NOT_AVAILABLE"
    assert "stand-in" in str(view["earliness.activity_regime"].reason)


def test_untrustworthy_scout_records_are_incompatible(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(at(-2), causal_valid=False))
    assert build(a, s).sources.scout.ref.status == "INCOMPATIBLE"
    arch.add(scout_record(at(-1), version=1))
    inp = build(a, s)
    assert inp.sources.scout.ref.status == "INCOMPATIBLE"
    assert all(f.status == "NOT_AVAILABLE" for f in inp.sources.scout.market_view.facts)


# --- cross_confirmation ----------------------------------------------------------------------------


def test_cross_confirmation_is_dropped_not_estimated(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(T))
    first = build(a, s).sources.scout
    c = candidate(T)
    cross = c["scout_momentum"]["families"][4]
    c["scout_momentum"]["families"][4] = {**cross, "score": 0.1, "signals": cross["signals"][:1]}
    arch.add(scout_record(T, cand=c))
    second = build(a, s).sources.scout
    assert second.market_view == first.market_view
    dropped = [f for f in second.diagnostics if f.aspect == "scout.cross_confirmation"]
    assert len(dropped) == 1 and dropped[0].scoring_owner is None
    assert "can't be separated exactly" in dropped[0].value["dropped"]  # type: ignore[index]


# --- Technical -----------------------------------------------------------------------------------


def test_scout_technical_context_is_normalized_into_technical(paths: Any) -> None:
    arch, a, s = paths
    c = candidate(T)
    c["momentum"]["technical"]["volume_confirmed"] = None
    arch.add(scout_record(T, cand=c))
    tech = _facts(build(a, s), "technical")
    assert tech["price.snapshot_trend"].value == {
        "trend": "up",
        "snapshots": 6,
        "span_hours": 2.5,
        "change_pct": 18.0,
    }
    assert tech["price.breakout"].value == {"breakout": True}
    assert tech["price.volume_confirmed"].status == "NOT_AVAILABLE"  # unknown, not False
    assert tech["price.analyze_trend"].status == "NOT_AVAILABLE"


def test_analyze_final_action_confidence_and_risk_are_ignored(paths: Any) -> None:
    arch, a, s = paths
    arch.add(decision_record(at(-1), action="buy", confidence="high"))
    first = build(a, s).sources.technical.facts
    arch.add(decision_record(at(-1), action="sell", confidence="low"))
    second = build(a, s)
    assert second.sources.technical.facts == first
    trend = _facts(second)["price.analyze_trend"]
    assert trend.value["label"] == "uptrend"  # type: ignore[index]
    assert '"action"' not in second.canonical_json()
    assert '"confidence"' not in second.canonical_json()


def test_a_failed_analyze_technical_result_is_unavailable(paths: Any) -> None:
    arch, a, s = paths
    arch.add(decision_record(T, technical_status="error"))
    tech = build(a, s).sources.technical
    assert tech.analyze_ref.status == "UNAVAILABLE"
    assert "no candles" in str(tech.analyze_ref.reason)


# --- Social -----------------------------------------------------------------------------------------


def test_unavailable_social_stays_unavailable(paths: Any) -> None:
    arch, a, s = paths
    arch.add(social_record(T, state="UNAVAILABLE"))
    inp = build(a, s)
    assert inp.sources.social.ref.status == "UNAVAILABLE"
    owned = _facts(inp, "social")
    for aspect in ("attention.momentum", "attention.spam", "attention.attribution"):
        assert owned[aspect].status == "NOT_AVAILABLE" and owned[aspect].value is None
    assert owned["attention.providers"].value["providers"] == {"X": "PROVIDER_UNAVAILABLE"}  # type: ignore[index]


def test_social_quality_attribution_and_cross_check_are_preserved(paths: Any) -> None:
    arch, a, s = paths
    arch.add(social_record(T))
    owned = _facts(build(a, s), "social")
    assert owned["attention.spam"].value["spam_risk"] == "medium"  # type: ignore[index]
    assert owned["attention.attribution"].value["exact_mentions"] == 8  # type: ignore[index]
    assert owned["attention.market_cross"].value["state"] == "CORROBORATED"  # type: ignore[index]
    assert owned["attention.momentum"].value["state"] == "ACCELERATING"  # type: ignore[index]


# --- News --------------------------------------------------------------------------------------------


def test_news_is_archived_context_only_and_marked_llm_labelled(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(T))
    inp0 = build(a, s)
    assert (
        inp0.sources.news.ref.status == "NOT_COLLECTED" and inp0.sources.news.llm_labelled is None
    )
    arch.add(decision_record(T))
    inp = build(a, s)
    news = inp.sources.news
    assert news.ref.status == "AVAILABLE" and news.llm_labelled is True
    assert news.sentiment_models == ("claude-x",)
    assert {f.role for f in news.facts} == {"CONTEXT"}
    assert not [f for f in inp.scoring_facts() if f.group == "NEWS"]
    arch.add(decision_record(T, news=news_findings(None)))  # same time, later row
    assert build(a, s).sources.news.llm_labelled is False


def test_a_news_story_published_after_the_decision_raises(paths: Any) -> None:
    arch, a, s = paths
    late = news_findings()
    late["reports"][0]["stories"][0]["published_at"] = at(seconds=1).isoformat()
    arch.add(decision_record(at(-1), news=late))
    with pytest.raises(OpportunityCausalityError):
        build(a, s)


# --- Double counting ---------------------------------------------------------------------------------


def test_one_scoring_owner_per_aspect(paths: Any) -> None:
    arch, a, s = paths
    _std(arch, s)
    inp = build(a, s)
    owned = [f for f in inp.facts() if f.role != "DIAGNOSTIC"]
    aspects = [f.aspect for f in owned]
    assert len(aspects) == len(set(aspects))
    for f in owned:
        assert f.layer == OWNERSHIP[f.aspect].layer and f.group == OWNERSHIP[f.aspect].group
    owned_aspects = {a for a, o in OWNERSHIP.items() if o.layer is not None}
    assert set(aspects) == owned_aspects  # every owned aspect explicit (value or missing)


def test_holder_concentration_and_authority_have_only_the_safety_owner(paths: Any) -> None:
    arch, a, s = paths
    _std(arch, s)
    inp = build(a, s)
    for aspect in ("holders.top10", "authority.mint"):
        copies = [f for f in inp.facts() if f.aspect == aspect]
        assert {f.layer for f in copies} == {"scout", "safety"}
        assert [f.layer for f in copies if f.role == "SCORING"] == ["safety"]


def test_liquidity_collapse_and_draining_never_both_score(paths: Any) -> None:
    arch, a, s = paths
    _std(arch, s)
    inp = build(a, s)
    collapse = [f for f in inp.facts() if f.aspect == "liquidity.collapse"]
    scoring = [f for f in collapse if f.role == "SCORING"]
    assert len(scoring) == 1 and scoring[0].layer == "safety"
    assert scoring[0].value["state"] == "TRIGGERED"  # type: ignore[index]
    draining = [f for f in collapse if f.layer == "scout"]
    assert draining and draining[0].role == "DIAGNOSTIC"
    assert draining[0].value["code"] == "liquidity_draining"  # type: ignore[index]


def test_the_ownership_map_rejects_a_second_scoring_owner(paths: Any) -> None:
    arch, a, s = paths
    _std(arch, s)
    inp = build(a, s)
    data = inp.model_dump()
    facts = list(data["sources"]["safety"]["facts"])
    data["sources"]["safety"]["facts"] = [*facts, dict(facts[0])]
    with pytest.raises(OwnershipError, match="more than one owned fact"):
        OpportunityInput.model_validate(data)
    with pytest.raises(ValueError, match="only its owner"):
        Fact(aspect="holders.top10", group="HOLDERS", layer="scout", role="SCORING",
             status="AVAILABLE", value={"pct": 55}, path="x", scoring_owner="safety")  # fmt: skip
    wrong = dict(data["sources"]["scout"]["market_view"]["facts"][0])
    wrong["scoring_owner"] = "safety"
    data2 = inp.model_dump()
    diags = list(data2["sources"]["scout"]["diagnostics"])
    data2["sources"]["scout"]["diagnostics"] = [*diags, {**wrong, "role": "DIAGNOSTIC"}]
    with pytest.raises(OwnershipError, match="owned by scout"):
        OpportunityInput.model_validate(data2)


# --- read-only / isolation ------------------------------------------------------------------------


def test_stores_are_byte_identical_and_only_opened_read_only(
    paths: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    arch, a, s = paths
    _std(arch, s)
    arch.close()
    before = {p: p.read_bytes() for p in (a, s)}
    opened: list[str] = []
    real = sqlite3.connect

    def spy(database: Any, *args: Any, **kw: Any) -> sqlite3.Connection:
        opened.append(str(database))
        return real(database, *args, **kw)

    monkeypatch.setattr(sqlite3, "connect", spy)
    build(a, s)
    build(a, s, origin="LIVE_FORWARD", decision_at=at(1))
    assert {p: p.read_bytes() for p in (a, s)} == before
    assert opened and all(o.startswith("file:") and o.endswith("?mode=ro") for o in opened)
    assert not [o for o in opened if "radar" in o.lower()]


def test_no_socket_is_ever_opened(paths: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    arch, a, s = paths
    _std(arch, s)

    def refuse(*args: Any, **kw: Any) -> None:
        raise AssertionError("network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert build(a, s).sources.scout.ref.status == "AVAILABLE"


def test_the_package_imports_no_execution_radar_provider_or_llm_code() -> None:
    pkg = Path(__file__).parents[1] / "upscale" / "services" / "opportunity_model"
    forbidden = ("radar", "shadow", "execution", "httpx", "anthropic", "news_sentiment_model",
                 "solana_chain", "solana_dex", "dexscreener", "held_position", "safety_v2.service",
                 "safety_v2.repository", "safety_v2.provider", "services.opportunity ")  # fmt: skip
    for path in pkg.glob("*.py"):
        imports = [ln for ln in path.read_text().splitlines()
                   if ln.startswith(("import ", "from "))]  # fmt: skip
        for ln in imports:
            assert not any(word in ln for word in forbidden), (path.name, ln)


# --- determinism -------------------------------------------------------------------------------------


def test_identical_records_give_an_identical_input_hash(tmp_path: Path) -> None:
    hashes = []
    for name in ("one", "two"):
        d = tmp_path / name
        d.mkdir()
        arch = Archive(d / "evidence.sqlite3")
        _std(arch, d / "safety.sqlite3", at(-3))
        arch.close()
        inp = build(arch.path, d / "safety.sqlite3")
        hashes.append((inp.input_hash(), inp.canonical_json()))
    assert hashes[0] == hashes[1]
    assert OpportunityInput.model_validate_json(hashes[0][1]).input_hash() == hashes[0][0]


def test_same_time_records_tie_break_by_row_id(paths: Any) -> None:
    arch, a, s = paths
    arch.add(scout_record(T, cand=candidate(T, stage="EARLY")))
    arch.add(scout_record(T, cand=candidate(T, stage="CROWDED")))
    stage = {f.aspect: f for f in build(a, s).sources.scout.context}["scout.stage"]
    assert stage.value["stage"] == "CROWDED"  # type: ignore[index]
    assert build(a, s).input_hash() == build(a, s).input_hash()


def test_an_unconfigured_safety_database_is_unavailable(
    paths: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    arch, a, _ = paths
    arch.add(scout_record(T))
    monkeypatch.delenv("UPSCALE_SAFETY_V2_DB", raising=False)
    inp = OpportunityInputService(str(a), clock=lambda: T).build_replay(CID, T)
    assert inp.sources.safety.ref.status == "UNAVAILABLE"
    assert "UPSCALE_SAFETY_V2_DB" in str(inp.sources.safety.ref.reason)


def test_scout_absences_are_never_proof(paths: Any) -> None:
    """Scout archives only raised flags and an OK market status whose tests skip missing
    inputs: neither absence becomes a NOT_TRIGGERED / healthy value."""
    arch, a, s = paths
    arch.add(scout_record(T))
    sc = build(a, s).sources.scout
    view = {f.aspect: f for f in sc.market_view.facts}
    for aspect in ("flow.distribution", "flow.volume_to_liquidity"):
        assert view[aspect].status == "NOT_AVAILABLE" and view[aspect].value is None
        assert "treats missing inputs as not met" in str(view[aspect].reason)
    assert view["flow.quality"].value["notes_complete"] is False  # type: ignore[index]
    collapse = {f.aspect: f for f in sc.context}["scout.market_collapse"]
    assert collapse.status == "NOT_AVAILABLE" and collapse.value is None
    c = candidate(T)
    c["quality"] = {**c["quality"], "market_status": "MARKET_COLLAPSE",
                    "collapse_evidence": ["price collapsed -90% over 24h"]}  # fmt: skip
    arch.add(scout_record(T, cand=c))
    collapsed = {f.aspect: f for f in build(a, s).sources.scout.context}["scout.market_collapse"]
    assert collapsed.value == {"market_status": "MARKET_COLLAPSE",
                               "collapse_evidence": ["price collapsed -90% over 24h"]}  # fmt: skip
