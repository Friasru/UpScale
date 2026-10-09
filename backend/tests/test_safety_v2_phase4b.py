"""Safety V2 Phase 4B: creator / verified-deployer features and temporal holder changes,
built only from Safety-owned evidence (captured Radar facts, Safety holder observations).
Offline; Radar databases are local fixtures, read only during collection."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.safety_v2_fakes import (
    MINT,
    OTHER_PROGRAM,
    Clock,
    FakeRpc,
    HolderChain,
    addr,
    make_service,
    mint_value,
    num_addr,
)
from tests.test_safety_v2_capture import NOW, T2, _exec, creator, flow, gap, radar, scan
from upscale.services.safety_v2 import service as service_module
from upscale.services.safety_v2 import sources
from upscale.services.safety_v2.features import (
    ChangeInputs,
    CreatorInputs,
    HolderInputs,
    build_body,
    code_fingerprints,
)
from upscale.services.safety_v2.models import SafetyCausalityError, SafetyIdentityError, ts
from upscale.services.safety_v2.provider import OwnerFact
from upscale.services.safety_v2.repository import HolderRow, PoolPin, owners_hash
from upscale.services.solana_chain import INCINERATOR

CID = f"solana:{MINT}"
SUPPLY = 1_000_000_000
PCT = SUPPLY // 100
WHALE, DEPLOYER, CREATOR_ADDR = addr("WhaLe"), addr("DepLoyer"), addr("CreatorAA")
POOL = addr("PooLAAA")
EXIT_ACCT, WHALE_ACCT = addr("ExitAcct"), addr("WhaLeAcct")
T, N, U = "TRIGGERED", "NOT_TRIGGERED", "UNDETERMINED"
DAY = 24 * 3600.0


def flags(body: dict[str, Any]) -> dict[str, str]:
    return {f["id"]: f["outcome"] for f in body["flags"]}


def oos(body: dict[str, Any]) -> set[str]:
    return {o["id"] for o in body["coverage"]["out_of_scope"]}


def service(tmp_path: Path, chain: HolderChain, radar_path: Path | None = None,
            **kw: Any) -> tuple[Any, Clock, FakeRpc]:  # fmt: skip
    rpc = FakeRpc({MINT: mint_value()}, holders=chain)
    kw.setdefault("helius", True)
    svc, clock, rpc = make_service(tmp_path, rpc, clock=Clock(T2),
                                   radar_db_path=str(radar_path) if radar_path else None, **kw)  # fmt: skip
    svc.add_target(MINT)
    return svc, clock, rpc


def collect(svc: Any) -> Any:
    return asyncio.run(svc.collect(CID, holders=True))


def snap(svc: Any) -> Any:
    return asyncio.run(svc.snapshot(CID, holders=True))


def base_chain(n: int = 100, amount: int = PCT // 10) -> HolderChain:
    chain = HolderChain()
    for i in range(n):
        chain.add(num_addr("Hdr", i), amount)
    return chain


# --- creator / deployer roles ----------------------------------------------------------------


def _with_roles(tmp_path: Path, roles: list[tuple[str, str, str | None]],
                chain: HolderChain | None = None) -> dict[str, Any]:  # fmt: skip
    r = radar(tmp_path / "radar.sqlite3")
    for role, status, identity in roles:
        creator(r, role, status, identity, NOW - 100)
    svc, _, _ = service(tmp_path, chain or base_chain(), r)
    body: dict[str, Any] = snap(svc).body
    return body


def test_candidate_only_is_never_a_deployer(tmp_path: Path) -> None:
    chain = base_chain()
    chain.add(CREATOR_ADDR, 30 * PCT)
    body = _with_roles(tmp_path, [("POOL_CREATOR_CANDIDATE", "CANDIDATE", CREATOR_ADDR)], chain)
    c = body["creator"]
    assert c["source_status"] == "CAPTURED"
    assert (c["pool_creator_candidate"]["status"], c["pool_creator_candidate"]["address"]) == (
        "CANDIDATE", CREATOR_ADDR)  # fmt: skip
    assert c["token_deployer"]["status"] == "UNVERIFIED"
    assert c["candidate_matches_verified_deployer"] is None
    assert c["deployer_holding_pct"]["status"] == "UNAVAILABLE"
    assert "VERIFIED_DEPLOYER_HOLDS_SUPPLY" in oos(body)
    assert c["pool_creator_candidate"]["capture_row_id"] is not None
    assert c["pool_creator_candidate"]["source_key"]


def test_verified_deployer_only(tmp_path: Path) -> None:
    body = _with_roles(tmp_path, [("TOKEN_DEPLOYER", "VERIFIED", DEPLOYER)])
    c = body["creator"]
    assert c["pool_creator_candidate"]["status"] == "NOT_COLLECTED"
    assert (c["token_deployer"]["status"], c["token_deployer"]["address"]) == ("VERIFIED", DEPLOYER)
    assert c["candidate_matches_verified_deployer"] is None
    assert c["deployer_holding_pct"] == {"status": "AVAILABLE", "value": 0.0,
                                         "lower_bound": False, "reason": None}  # fmt: skip
    assert flags(body)["VERIFIED_DEPLOYER_HOLDS_SUPPLY"] == N  # absent from a complete scan


@pytest.mark.parametrize(("candidate", "expected"), [(DEPLOYER, True), (CREATOR_ADDR, False)])
def test_candidate_matches_verified_deployer(
    tmp_path: Path, candidate: str, expected: bool
) -> None:
    body = _with_roles(tmp_path, [("POOL_CREATOR_CANDIDATE", "CANDIDATE", candidate),
                                  ("TOKEN_DEPLOYER", "VERIFIED", DEPLOYER)])  # fmt: skip
    assert body["creator"]["candidate_matches_verified_deployer"] is expected


def test_an_unverified_deployer_row_is_never_a_match(tmp_path: Path) -> None:
    body = _with_roles(tmp_path, [("POOL_CREATOR_CANDIDATE", "CANDIDATE", DEPLOYER),
                                  ("TOKEN_DEPLOYER", "UNAVAILABLE", None)])  # fmt: skip
    assert body["creator"]["token_deployer"]["status"] == "UNAVAILABLE"
    assert body["creator"]["candidate_matches_verified_deployer"] is None
    assert "VERIFIED_DEPLOYER_HOLDS_SUPPLY" in oos(body)


def test_a_replaced_candidate_never_leaks_backward(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    creator(r, "POOL_CREATOR_CANDIDATE", "UNAVAILABLE", None, NOW - 100)
    svc, clock, _ = service(tmp_path, base_chain(), r)
    old = snap(svc)
    clock.advance(3600)
    creator(r, "POOL_CREATOR_CANDIDATE", "CANDIDATE", CREATOR_ADDR, ts(clock.now()) - 10)
    new = snap(svc)
    assert old.body["creator"]["pool_creator_candidate"]["status"] == "UNAVAILABLE"
    assert new.body["creator"]["pool_creator_candidate"]["address"] == CREATOR_ADDR
    assert svc.rebuild(old.snapshot_id).status == "REPRODUCED"


@pytest.mark.parametrize("kind", ["none", "unconfigured", "incompatible"])
def test_unusable_or_absent_creator_source_stays_honest(tmp_path: Path, kind: str) -> None:
    if kind == "none":  # no capture at all at as_of
        body = build_body(CID, MINT, NOW, None, code_fingerprints())
        assert body["creator"]["source_status"] == "NOT_CAPTURED"
    else:
        path = None
        if kind == "incompatible":
            path = radar(tmp_path / "radar.sqlite3")
            _exec(path, "UPDATE radar_meta SET value = '2' WHERE key = 'schema_version'")
        svc, _, _ = service(tmp_path, base_chain(), path)
        body = snap(svc).body
        assert body["creator"]["source_status"] == kind.upper().replace("UNCONFIGURED",
                                                                        "NOT_CONFIGURED")  # fmt: skip
    c = body["creator"]
    assert c["token_deployer"]["status"] == "NOT_COLLECTED"  # never "absent"
    assert c["candidate_matches_verified_deployer"] is None
    assert "VERIFIED_DEPLOYER_HOLDS_SUPPLY" in oos(body)


# --- deployer holding ------------------------------------------------------------------------


def _holding(tmp_path: Path, amount: int | None, **kw: Any) -> dict[str, Any]:
    chain = kw.pop("chain", None) or base_chain()
    if amount is not None:
        chain.add(DEPLOYER, amount)
    r = radar(tmp_path / "radar.sqlite3")
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 100)
    svc, _, _ = service(tmp_path, chain, r, **kw)
    body: dict[str, Any] = snap(svc).body
    return body


@pytest.mark.parametrize(("amount", "outcome", "severity"), [
    (5 * PCT - 1, N, "medium"), (5 * PCT, T, "medium"), (10 * PCT - 1, T, "medium"),
    (10 * PCT, T, "high"), (12 * PCT, T, "high"),
])  # fmt: skip
def test_complete_scan_holding_boundaries(tmp_path: Path, amount: int, outcome: str,
                                          severity: str) -> None:  # fmt: skip
    body = _holding(tmp_path, amount)
    h = body["creator"]["deployer_holding_pct"]
    assert h["status"] == "AVAILABLE" and h["value"] == round(100 * amount / SUPPLY, 6)
    f = next(x for x in body["flags"] if x["id"] == "VERIFIED_DEPLOYER_HOLDS_SUPPLY")
    assert (f["outcome"], f["severity"]) == (outcome, severity)


def _partial_chain() -> HolderChain:
    return base_chain(1500, 10_000)  # > one 1,000-row page


@pytest.mark.parametrize(("amount", "outcome"), [(7 * PCT, T), (2 * PCT, U)])
def test_partial_scan_observed_deployer_is_a_lower_bound(tmp_path: Path, amount: int,
                                                         outcome: str) -> None:  # fmt: skip
    body = _holding(tmp_path, amount, chain=_partial_chain(), holder_max_pages=1)
    h = body["creator"]["deployer_holding_pct"]
    assert h["status"] == "PARTIAL" and h["lower_bound"] is True
    assert flags(body)["VERIFIED_DEPLOYER_HOLDS_SUPPLY"] == outcome


def test_partial_scan_absent_deployer_is_never_zero(tmp_path: Path) -> None:
    body = _holding(tmp_path, None, chain=_partial_chain(), holder_max_pages=1)
    h = body["creator"]["deployer_holding_pct"]
    assert h["status"] == "UNKNOWN" and h["value"] is None and "not proof of zero" in h["reason"]
    assert flags(body)["VERIFIED_DEPLOYER_HOLDS_SUPPLY"] == U


@pytest.mark.parametrize("amount", [8 * PCT, None])
def test_largest_accounts_only_holding(tmp_path: Path, amount: int | None) -> None:
    body = _holding(tmp_path, amount, helius=False)
    h = body["creator"]["deployer_holding_pct"]
    if amount:
        assert (h["status"], h["value"], h["lower_bound"]) == ("PARTIAL", 8.0, True)
        assert flags(body)["VERIFIED_DEPLOYER_HOLDS_SUPPLY"] == T
    else:
        assert h["status"] == "UNKNOWN" and h["value"] is None
        assert flags(body)["VERIFIED_DEPLOYER_HOLDS_SUPPLY"] == U


# --- deployer outflows -----------------------------------------------------------------------


def _outflow_setup(tmp_path: Path) -> tuple[Any, Clock, Path]:
    r = radar(tmp_path / "radar.sqlite3")
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 500)
    flow(r, DEPLOYER, NOW - 60, sig="out1", direction="TOKEN_OUTFLOW", amount="1000")
    flow(r, DEPLOYER, NOW - 55, sig="out2", direction="TOKEN_OUTFLOW", amount="250")
    flow(r, DEPLOYER, NOW - 50, sig="in1", direction="TOKEN_INFLOW")
    flow(r, addr("QtherWa"), NOW - 40, sig="x", direction="TOKEN_OUTFLOW")
    flow(r, DEPLOYER, NOW - 30, sig="y", direction="TOKEN_OUTFLOW", cid="solana:other")
    scan(r, 1, NOW - 20)
    svc, clock, _ = service(tmp_path, base_chain(), r)
    return svc, clock, r


def test_deployer_outflows_are_evidence_with_partial_coverage(tmp_path: Path) -> None:
    svc, _, _ = _outflow_setup(tmp_path)
    body = snap(svc).body
    o = body["creator"]["deployer_outflows"]
    assert o["kind"] == "TOKEN_OUTFLOW" and o["status"] == "PARTIAL"
    assert o["event_count"] == {"status": "PARTIAL", "value": 2, "lower_bound": True,
                                "reason": o["coverage_reasons"][0]}  # fmt: skip
    assert o["total_amount_raw"] == "1250"
    assert [e["signature"] for e in o["events"]] == ["out1", "out2"]
    assert "touching the tracked pool" in o["coverage_reasons"][0]
    assert "VERIFIED_DEPLOYER_OUTFLOWS" not in flags(body) | {r: "" for r in oos(body)}
    text = json.dumps(body).lower()
    assert not any(w in text for w in ("sale", "sell", "dump", "cash-out"))


def test_zero_observed_outflows_is_never_proof_of_none(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 500)
    scan(r, 1, NOW - 20)
    gap(r, 1, NOW - 15)
    svc, _, _ = service(tmp_path, base_chain(), r)
    o = snap(svc).body["creator"]["deployer_outflows"]
    assert o["event_count"]["status"] == "PARTIAL" and o["event_count"]["lower_bound"]
    assert any("gaps are open" in x for x in o["coverage_reasons"])


def test_no_captured_coverage_leaves_outflows_unavailable(tmp_path: Path) -> None:
    body = _with_roles(tmp_path, [("TOKEN_DEPLOYER", "VERIFIED", DEPLOYER)])
    o = body["creator"]["deployer_outflows"]
    assert o["status"] == "UNAVAILABLE" and o["event_count"]["status"] == "UNAVAILABLE"


def test_outflows_captured_later_or_deleted_from_radar_change_nothing_backward(
    tmp_path: Path,
) -> None:
    svc, clock, r = _outflow_setup(tmp_path)
    first = snap(svc)
    _exec(r, "DELETE FROM radar_wallet_flows")  # Radar retention
    clock.advance(600)
    flow(r, DEPLOYER, ts(clock.now()) - 5, sig="out3", direction="TOKEN_OUTFLOW", amount="7")
    later = snap(svc)
    assert later.body["creator"]["deployer_outflows"]["total_amount_raw"] == "1257"
    assert svc.rebuild(first.snapshot_id).status == "REPRODUCED"
    assert svc.build(CID, NOW)["creator"]["deployer_outflows"]["total_amount_raw"] == "1250"


# --- temporal pair selection / applicability -------------------------------------------------


def test_a_single_scan_keeps_change_rules_out_of_scope(tmp_path: Path) -> None:
    svc, _, _ = service(tmp_path, base_chain())
    body = snap(svc).body
    assert {"CONCENTRATION_RISING", "RAPID_HOLDER_LOSS", "LARGE_HOLDER_EXIT"} <= oos(body)
    assert body["changes"]["status"] == "UNAVAILABLE"
    assert body["assessment"]["coverage"] == "COMPLETE"  # no retroactive degradation


def test_pair_selection_and_same_time_tie_break(tmp_path: Path) -> None:
    svc, clock, _ = service(tmp_path, base_chain())
    a = collect(svc)
    b = collect(svc)  # same fetched_at: the row id breaks the tie
    clock.advance(3600)
    c = collect(svc)
    ch = svc.build(CID, ts(clock.now()))["changes"]
    assert (ch["previous_holder_observation_id"], ch["current_holder_observation_id"]) == (
        b.holder_observation_id, c.holder_observation_id)  # fmt: skip
    assert ch["elapsed_seconds"] == 3600
    at_b = svc.build(CID, NOW)["changes"]
    assert (at_b["previous_holder_observation_id"], at_b["current_holder_observation_id"]) == (
        a.holder_observation_id, b.holder_observation_id)  # fmt: skip
    assert at_b["elapsed_seconds"] == 0


@pytest.mark.parametrize(("whale_after", "outcome"), [
    (31 * PCT, T),  # 20.9 % -> 31.9 %: +11 pp
    (30 * PCT, T),  # exactly +10 pp
    (30 * PCT - 1, N),
    (25 * PCT, N),
])  # fmt: skip
def test_concentration_rising_boundaries(tmp_path: Path, whale_after: int, outcome: str) -> None:
    chain = base_chain()
    chain.add(WHALE, 20 * PCT, WHALE_ACCT)
    svc, clock, _ = service(tmp_path, chain)
    collect(svc)
    clock.advance(3600)
    chain.accounts[WHALE_ACCT] = (WHALE, whale_after)
    body = snap(svc).body
    pp = body["changes"]["top10_change_pp"]["value"]
    assert pp == round(100 * (whale_after - 20 * PCT) / SUPPLY, 6)
    assert flags(body)["CONCENTRATION_RISING"] == outcome
    assert "percentage points" in next(f["reason"] for f in body["flags"]
                                       if f["id"] == "CONCENTRATION_RISING")  # fmt: skip


@pytest.mark.parametrize("order", ["partial_current", "partial_prior"])
def test_partial_pairs_give_no_exact_delta(tmp_path: Path, order: str) -> None:
    small, big = base_chain(), base_chain(1500, 10_000)
    chain = small if order == "partial_current" else big
    svc, clock, rpc = service(tmp_path, chain, holder_max_pages=1)
    collect(svc)
    clock.advance(3600)
    rpc.holders = big if order == "partial_current" else small
    body = snap(svc).body
    ch = body["changes"]
    assert ch["status"] == "UNKNOWN" and ch["top10_change_pp"]["value"] is None
    f = flags(body)
    assert (f["CONCENTRATION_RISING"], f["RAPID_HOLDER_LOSS"], f["LARGE_HOLDER_EXIT"]) == (U, U, U)
    assert body["assessment"]["coverage"] == "PARTIAL"


def test_a_later_scan_never_changes_an_earlier_comparison(tmp_path: Path) -> None:
    chain = base_chain()
    svc, clock, _ = service(tmp_path, chain)
    collect(svc)
    clock.advance(3600)
    first = snap(svc)
    clock.advance(3600)
    chain.add(WHALE, 40 * PCT)
    collect(svc)
    assert svc.rebuild(first.snapshot_id).status == "REPRODUCED"


def test_previous_of_another_token_is_refused() -> None:
    def obs(i: int, cid: str, at: float) -> Any:
        bal = [OwnerFact(WHALE, WHALE, ("a",), PCT, "MISSING")]
        return HolderRow(i, cid, i, at, "p", "o", "COLLECTED", None, "full_scan", str(SUPPLY),
                         6, 1, 2, 1, 1, 0, 1, True, True, [], owners_hash(bal)), tuple(bal)  # fmt: skip

    cur, cb = obs(2, CID, 20.0)
    prev, pb = obs(1, "solana:" + addr("QtherMint"), 10.0)
    with pytest.raises(SafetyIdentityError):
        build_body(CID, MINT, 20.0, None, code_fingerprints(), HolderInputs(cur, cb),
                   changes=ChangeInputs(True, HolderInputs(prev, pb)))  # fmt: skip
    future_pin = PoolPin(1, CID, POOL, None, 15.0, "manual")  # after the previous scan
    prev, pb = obs(1, CID, 10.0)
    with pytest.raises(SafetyCausalityError, match="pool pin"):
        build_body(CID, MINT, 20.0, None, code_fingerprints(), HolderInputs(cur, cb),
                   changes=ChangeInputs(True, HolderInputs(prev, pb, (future_pin,))))  # fmt: skip


# --- rapid holder loss -----------------------------------------------------------------------


@pytest.mark.parametrize(("after", "outcome"), [(120, T), (140, T), (141, N)])
def test_rapid_holder_loss_uses_meaningful_holders(
    tmp_path: Path, after: int, outcome: str
) -> None:
    chain = base_chain(200, 10_000)  # 200 meaningful holders (1e-5 of supply each)
    svc, clock, _ = service(tmp_path, chain)
    collect(svc)
    clock.advance(DAY)
    for i in range(after, 200):
        del chain.accounts[num_addr("Tacc", i)]
    body = snap(svc).body
    ch = body["changes"]
    assert ch["meaningful_holder_count_change_pct"]["value"] == round(100 * (after - 200) / 200, 6)
    assert flags(body)["RAPID_HOLDER_LOSS"] == outcome
    reason = next(f["reason"] for f in body["flags"] if f["id"] == "RAPID_HOLDER_LOSS")
    assert "meaningful holders" in reason


def test_dust_holder_churn_alone_never_triggers_holder_loss(tmp_path: Path) -> None:
    chain = base_chain(100, 10_000)
    for i in range(400):
        chain.add(num_addr("Dst", i), 1)  # dust: below 1e-6 of supply
    svc, clock, _ = service(tmp_path, chain)
    collect(svc)
    clock.advance(DAY)
    for i in range(100, 500):
        del chain.accounts[num_addr("Tacc", i)]  # every dust owner leaves
    body = snap(svc).body
    assert body["changes"]["holder_count_change_pct"]["value"] == -80.0
    assert body["changes"]["meaningful_holder_count_change_pct"]["value"] == 0.0
    assert flags(body)["RAPID_HOLDER_LOSS"] == N


def test_holder_loss_outside_the_window_or_on_a_tiny_baseline_is_out_of_scope(
    tmp_path: Path,
) -> None:
    chain = base_chain(200, 10_000)
    svc, clock, _ = service(tmp_path, chain)
    collect(svc)
    clock.advance(7 * DAY + 1)
    for i in range(100, 200):
        del chain.accounts[num_addr("Tacc", i)]
    body = snap(svc).body
    assert "RAPID_HOLDER_LOSS" in oos(body) and "RAPID_HOLDER_LOSS" not in flags(body)
    assert flags(body)["CONCENTRATION_RISING"] in (T, N)  # still evaluated


def test_zero_previous_meaningful_holders(tmp_path: Path) -> None:
    chain = HolderChain()
    for i in range(50):
        chain.add(num_addr("Dst", i), 1)  # dust only: 0 meaningful holders
    svc, clock, _ = service(tmp_path, chain)
    collect(svc)
    clock.advance(DAY)
    chain.add(WHALE, PCT)
    body = snap(svc).body
    m = body["changes"]["meaningful_holder_count_change_pct"]
    assert m["status"] == "UNAVAILABLE" and "zero" in m["reason"]
    assert "RAPID_HOLDER_LOSS" in oos(body)  # baseline 0 < 30


# --- large holder exit -----------------------------------------------------------------------


def _exit(tmp_path: Path, before: int, after: int | None, *, owner: str = WHALE,
          setup: Any = None, between: Any = None) -> dict[str, Any]:  # fmt: skip
    chain = base_chain()
    acct = EXIT_ACCT
    chain.add(owner, before, acct)
    if setup:
        setup(chain)
    svc, clock, _ = service(tmp_path, chain)
    collect(svc)
    clock.advance(3600)
    if between:
        between(svc)
    clock.advance(1)
    if after is None:
        del chain.accounts[acct]
    else:
        chain.accounts[acct] = (owner, after)
    body: dict[str, Any] = snap(svc).body
    return body


def test_a_large_holder_that_goes_to_zero_exits(tmp_path: Path) -> None:
    body = _exit(tmp_path, 6 * PCT, None)
    exits = body["changes"]["large_holder_exits"]["value"]
    assert [(e["owner"], e["previous_pct"], e["current_amount_raw"],
             e["classification_at_previous"]) for e in exits] == [(WHALE, 6.0, "0", "UNKNOWN")]  # fmt: skip
    assert flags(body)["LARGE_HOLDER_EXIT"] == T


@pytest.mark.parametrize(("before", "after"), [(6 * PCT, 2 * PCT), (4 * PCT, None)])
def test_reductions_and_small_holders_are_not_exits(tmp_path: Path, before: int,
                                                    after: int | None) -> None:  # fmt: skip
    body = _exit(tmp_path, before, after)
    assert body["changes"]["large_holder_exits"]["value"] == []
    assert flags(body)["LARGE_HOLDER_EXIT"] == N


def test_burn_and_pool_owners_never_exit(tmp_path: Path) -> None:
    assert flags(_exit(tmp_path / "a", 10 * PCT, None, owner=INCINERATOR))["LARGE_HOLDER_EXIT"] == N
    (tmp_path / "b").mkdir()
    body = _exit(tmp_path / "b", 10 * PCT, None, owner=POOL)
    assert flags(body)["LARGE_HOLDER_EXIT"] == T  # unpinned: just an unknown owner
    (tmp_path / "c").mkdir()
    rpc_chain = base_chain()
    acct = EXIT_ACCT
    rpc_chain.add(POOL, 10 * PCT, acct)
    svc, clock, _ = service(tmp_path / "c", rpc_chain)
    svc.pin_pool(MINT, POOL)
    clock.advance(1)
    collect(svc)
    clock.advance(3600)
    del rpc_chain.accounts[acct]
    assert flags(snap(svc).body)["LARGE_HOLDER_EXIT"] == N  # pinned before the previous scan


def test_program_owned_holder_can_exit(tmp_path: Path) -> None:
    def program(chain: HolderChain) -> None:
        chain.owner_programs[WHALE] = OTHER_PROGRAM

    body = _exit(tmp_path, 6 * PCT, None, setup=program)
    assert body["changes"]["large_holder_exits"]["value"][0]["classification_at_previous"] == (
        "PROGRAM_OWNED")  # fmt: skip


def test_a_later_pin_never_rewrites_the_previous_classification(tmp_path: Path) -> None:
    body = _exit(tmp_path, 6 * PCT, None, owner=POOL,
                 between=lambda svc: svc.pin_pool(MINT, POOL))  # fmt: skip
    exits = body["changes"]["large_holder_exits"]["value"]
    assert [(e["owner"], e["classification_at_previous"]) for e in exits] == [(POOL, "UNKNOWN")]


def test_partial_current_scan_prevents_an_exit_claim(tmp_path: Path) -> None:
    chain = base_chain()
    chain.add(WHALE, 6 * PCT)
    svc, clock, rpc = service(tmp_path, chain, holder_max_pages=1)
    collect(svc)
    clock.advance(3600)
    rpc.holders = base_chain(1500, 10_000)  # partial, whale not observed
    body = snap(svc).body
    assert body["changes"]["large_holder_exits"]["status"] == "UNKNOWN"
    assert flags(body)["LARGE_HOLDER_EXIT"] == U


# --- causality / determinism / isolation -----------------------------------------------------


def test_future_captured_facts_raise() -> None:
    cap = (1, "CAPTURED", None, 10.0)
    row = {"id": 1, "canonical_id": CID, "role": "TOKEN_DEPLOYER", "status": "VERIFIED",
           "address": DEPLOYER, "source_time": 11.0, "captured_at": 10.0}  # fmt: skip
    with pytest.raises(SafetyCausalityError, match="source_time"):
        build_body(CID, MINT, 10.0, None, code_fingerprints(),
                   creator=CreatorInputs(cap, (row,)))  # fmt: skip
    row |= {"source_time": 9.0, "captured_at": 10.5}
    with pytest.raises(SafetyCausalityError, match="captured_at"):
        build_body(CID, MINT, 10.0, None, code_fingerprints(),
                   creator=CreatorInputs(cap, (row,)))  # fmt: skip
    with pytest.raises(SafetyCausalityError, match="capture"):
        build_body(CID, MINT, 9.0, None, code_fingerprints(), creator=CreatorInputs(cap))


def test_same_evidence_same_hash(tmp_path: Path) -> None:
    hashes = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        r = radar(tmp_path / name / "radar.sqlite3")
        creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 100)
        chain = base_chain()
        chain.add(DEPLOYER, 7 * PCT)
        svc, clock, _ = service(tmp_path / name, chain, r)
        collect(svc)
        clock.advance(3600)
        res = snap(svc)
        assert svc.rebuild(res.snapshot_id).status == "REPRODUCED"
        hashes.append(res.body_hash)
    assert hashes[0] == hashes[1]


def test_build_and_rebuild_never_read_radar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, clock, r = _outflow_setup(tmp_path)
    collect(svc)
    clock.advance(60)
    res = snap(svc)

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("Radar read outside collection")

    monkeypatch.setattr(service_module, "read_radar_capture", forbidden)
    monkeypatch.setattr(sources, "_connect", forbidden)
    r.unlink()
    assert svc.rebuild(res.snapshot_id).status == "REPRODUCED"
    svc.build(CID, ts(clock.now()))
