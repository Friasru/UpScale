"""Safety V2 Phase 3: market + liquidity evidence. Exact pool identity, pinned anchors,
primary selection, provider-reported liquidity, same-pool collapse history, not-reported
timing, on-chain closure proof, pool age, coverage, anti-lookahead, failures and rebuilds.
Offline: every DEX and RPC request is answered by `FakeRpc`."""

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.safety_v2_fakes import (
    MINT,
    OTHER_PROGRAM,
    T0,
    USDC,
    Clock,
    FakeRpc,
    addr,
    make_service,
    mint_value,
    pair_row,
)
from upscale.services.evidence_archive import hooks
from upscale.services.safety_v2 import features
from upscale.services.safety_v2 import service as service_module
from upscale.services.safety_v2.features import MarketInputs, build_body, code_fingerprints
from upscale.services.safety_v2.market import AccountCheck, ObservedMarket
from upscale.services.safety_v2.models import SafetyCausalityError, ts
from upscale.services.safety_v2.provider import classify_market
from upscale.services.safety_v2.repository import PoolPin
from upscale.services.safety_v2.rules import (
    MARKET_NOT_REPORTED_MIN_DURATION_S,
    MARKET_RULE_IDS,
    MarketFacts,
    evaluate_market,
)

CID = f"solana:{MINT}"
P1, P2, P3 = addr("PqqLAAA"), addr("PqqLBBB"), addr("PqqLCCC")
T, N, U = "TRIGGERED", "NOT_TRIGGERED", "UNDETERMINED"
HOUR = 3600.0
EXISTS = {"owner": OTHER_PROGRAM, "lamports": 5_000_000, "data": ["", "base64"]}


def setup(tmp_path: Path, rows: Any, **kw: Any) -> tuple[Any, Clock, FakeRpc]:
    rpc = FakeRpc({MINT: mint_value()}, dex=rows)
    svc, clock, rpc = make_service(tmp_path, rpc, dex=True, **kw)
    svc.add_target(MINT)
    return svc, clock, rpc


def snap(svc: Any, **kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = asyncio.run(svc.snapshot(CID, market=True, **kw)).body
    return body


def collect(svc: Any) -> Any:
    return asyncio.run(svc.collect(CID, market=True))


def flags(body: dict[str, Any]) -> dict[str, str]:
    return {f["id"]: f["outcome"] for f in body["flags"]}


def flag(body: dict[str, Any], rule: str) -> dict[str, Any]:
    return next(f for f in body["flags"] if f["id"] == rule)


def out_of_scope(body: dict[str, Any]) -> set[str]:
    return {o["id"] for o in body["coverage"]["out_of_scope"]}


def m(body: dict[str, Any]) -> dict[str, Any]:
    market: dict[str, Any] = body["market"]
    return market


# --- exact identity --------------------------------------------------------------------------


def test_an_exact_mint_pool_is_the_market(tmp_path: Path) -> None:
    svc, _, rpc = setup(tmp_path, [pair_row(P1)])
    body = snap(svc)
    mk = m(body)
    assert mk["status"] == "AVAILABLE" and mk["basis"] == "PROVIDER_REPORTED"
    assert mk["primary_pool"]["value"] == {"address": P1, "dex": "raydium", "anchor": "SELECTED"}
    assert mk["primary_liquidity_usd"] == {"status": "AVAILABLE", "value": 200_000.0,
                                           "lower_bound": False, "reason": None}  # fmt: skip
    assert mk["pool_presence"]["state"] == "REPORTED"
    assert mk["primary_clear"]["value"] is True
    f = flags(body)
    assert {k: f[k] for k in MARKET_RULE_IDS if k != "MARKET_CLOSED_ON_CHAIN"} == {
        "LOW_LIQUIDITY": N, "LIQUIDITY_COLLAPSE": U, "MARKET_NOT_REPORTED": N,
        "NO_ELIGIBLE_MARKET": N, "PRIMARY_MARKET_UNCLEAR": N, "VERY_NEW_POOL": N,
    }  # fmt: skip
    # DEX presence is no on-chain proof: the closure rule is out of scope, not NOT_TRIGGERED.
    assert "MARKET_CLOSED_ON_CHAIN" not in f
    assert "MARKET_CLOSED_ON_CHAIN" in out_of_scope(body)
    assert m(body)["closure_check"]["applicable"] is False
    assert body["identity"]["pool_match"]["status"] == "UNVERIFIED"  # provider-reported only
    assert body["identity"]["pool_match"]["evidence"]["value"]["pool"] == P1
    assert rpc.calls == ["getAccountInfo", "dex.tokenPairs"]  # no pool check when reported


def test_other_mints_symbols_and_quote_side_pools_are_never_the_market(tmp_path: Path) -> None:
    case_variant = "m" + MINT[1:]  # base58 is case-sensitive: another mint
    rows = [
        pair_row(P1, base=addr("MpostorAA"), base_symbol="TKN"),  # same symbol only
        pair_row(P2, base=case_variant),
        pair_row(P3, base=USDC, quote=MINT),  # the mint is the quote token
        pair_row(addr("PqqLDDD"), base=" " + MINT),  # whitespace variant
        pair_row(addr("PqqLEEE"), chain="ethereum"),
    ]
    svc, _, _ = setup(tmp_path, rows)
    body = snap(svc)
    mk = m(body)
    identities = {p["pair_address"]: p["identity"] for p in mk["other_pools"]}
    assert identities == {P1: "OTHER_MINT", P2: "OTHER_MINT", P3: "QUOTE_SIDE",
                          addr("PqqLDDD"): "OTHER_MINT", addr("PqqLEEE"): "OTHER_CHAIN"}  # fmt: skip
    assert mk["primary_pool"]["status"] == "UNAVAILABLE"
    assert mk["pool_presence"]["state"] == "UNAVAILABLE"
    assert flags(body)["NO_ELIGIBLE_MARKET"] == T
    assert (
        svc.repo.db().execute("SELECT outcome FROM safety_market_observations").fetchone()[0]
        == "NO_POOLS"
    )


def test_malformed_pools_are_rejected() -> None:
    rows: list[Any] = [
        pair_row("not a pool!"),
        {"chainId": "solana", "dexId": "raydium", "pairAddress": P2},  # no tokens
        "garbage",
        pair_row(P1),
    ]
    obs = classify_market(MINT, rows)
    assert obs.outcome == "POOLS" and obs.malformed_rows == 2 and obs.rows_returned == 4
    by = {p.pair_address: p.identity for p in obs.pools}
    assert by == {"not a pool!": "MALFORMED", P1: "EXACT_BASE"}
    assert obs.primary_pool == P1


# --- pinned pools ----------------------------------------------------------------------------


def test_a_pinned_pool_stays_the_anchor_over_a_larger_pool(tmp_path: Path) -> None:
    rows = [pair_row(P1, liquidity=50_000.0), pair_row(P2, liquidity=500_000.0)]
    svc, clock, _ = setup(tmp_path, rows)
    svc.pin_pool(MINT, P1)
    clock.advance(1)
    body = snap(svc)
    mk = m(body)
    assert mk["primary_pool"]["value"]["anchor"] == "PINNED"
    assert mk["primary_pool"]["value"]["address"] == P1
    assert mk["primary_liquidity_usd"]["value"] == 50_000.0
    assert [p["pair_address"] for p in mk["alternative_eligible_pools"]] == [P2]
    assert flags(body)["LOW_LIQUIDITY"] == T and flag(body, "LOW_LIQUIDITY")["severity"] == "medium"
    assert flags(body)["PRIMARY_MARKET_UNCLEAR"] == N
    pins = svc.repo.db().execute("SELECT COUNT(*) FROM safety_target_pools").fetchone()[0]
    assert pins == 1  # building never mutates the pin


def test_a_pin_learned_after_as_of_is_ignored(tmp_path: Path) -> None:
    rows = [pair_row(P1, liquidity=50_000.0), pair_row(P2, liquidity=500_000.0)]
    svc, clock, _ = setup(tmp_path, rows)
    first = asyncio.run(svc.snapshot(CID, market=True))
    clock.advance(10)
    svc.pin_pool(MINT, P1)
    old = svc.build(CID, ts(first.as_of))
    assert m(old)["primary_pool"]["value"]["address"] == P2
    assert m(old)["primary_pool"]["value"]["anchor"] == "SELECTED"
    assert svc.rebuild(first.snapshot_id or 0).status == "REPRODUCED"
    assert m(svc.build(CID, ts(clock.now())))["primary_pool"]["value"]["address"] == P1


def test_a_missing_never_observed_pin_is_not_reported_never_closed(tmp_path: Path) -> None:
    svc, clock, rpc = setup(tmp_path, [pair_row(P2)])
    svc.pin_pool(MINT, P3)
    clock.advance(1)
    body = snap(svc)
    mk = m(body)
    assert mk["pool_presence"]["state"] == "NOT_REPORTED"
    assert "not proof of closure" in mk["pool_presence"]["reason"]
    assert "MARKET_CLOSED_ON_CHAIN" not in flags(body)  # a pin alone isn't corroboration
    assert "MARKET_CLOSED_ON_CHAIN" in out_of_scope(body)
    assert rpc.calls == ["getAccountInfo", "dex.tokenPairs"]  # never observed: no RPC check


# --- primary selection -----------------------------------------------------------------------


def test_deterministic_primary_among_alternatives(tmp_path: Path) -> None:
    rows = [pair_row(P2, liquidity=50_000.0), pair_row(P1, liquidity=200_000.0)]
    svc, _, _ = setup(tmp_path, rows)
    mk = m(snap(svc))
    assert mk["primary_pool"]["value"]["address"] == P1 and mk["primary_clear"]["value"] is True
    assert [p["pair_address"] for p in mk["alternative_eligible_pools"]] == [P2]


def test_an_ambiguous_market_stays_visible(tmp_path: Path) -> None:
    rows = [pair_row(P1, liquidity=200_000.0), pair_row(P2, liquidity=150_000.0)]
    svc, _, _ = setup(tmp_path, rows)
    body = snap(svc)
    assert m(body)["primary_clear"]["value"] is False
    assert m(body)["ambiguity"] and "75%" in m(body)["ambiguity"][0]
    assert flags(body)["PRIMARY_MARKET_UNCLEAR"] == T


def test_no_eligible_exact_market(tmp_path: Path) -> None:
    svc, _, _ = setup(tmp_path, [pair_row(P1, liquidity=500.0)])
    body = snap(svc)
    f = flags(body)
    assert f["NO_ELIGIBLE_MARKET"] == T and f["PRIMARY_MARKET_UNCLEAR"] == N
    assert m(body)["primary_pool"]["status"] == "UNAVAILABLE"
    assert f["LOW_LIQUIDITY"] == U  # no tracked pool: never a fabricated zero


def test_two_pins_make_the_market_unclear(tmp_path: Path) -> None:
    svc, clock, _ = setup(tmp_path, [pair_row(P1), pair_row(P2)])
    svc.pin_pool(MINT, P2)
    clock.advance(1)
    svc.pin_pool(MINT, P1)
    clock.advance(1)
    body = snap(svc)
    assert m(body)["primary_pool"]["value"]["address"] == P2  # the earliest pin
    assert flags(body)["PRIMARY_MARKET_UNCLEAR"] == T


# --- liquidity -------------------------------------------------------------------------------


def _facts(**kw: Any) -> MarketFacts:
    base: dict[str, Any] = dict(status="AVAILABLE", reason=None, presence="REPORTED",
                                anchor=P1, any_eligible=True, has_primary=True, clear=True)  # fmt: skip
    return MarketFacts(**(base | kw))


def _rule(facts: MarketFacts, rule: str) -> tuple[str, str]:
    r = next(r for r in evaluate_market(facts) if r.id == rule)
    return r.outcome, r.severity


@pytest.mark.parametrize(("usd", "expected"), [
    (24_999.99, (T, "high")), (25_000.0, (T, "medium")), (99_999.99, (T, "medium")),
    (100_000.0, (N, "medium")), (5_000_000.0, (N, "medium")),
])  # fmt: skip
def test_low_liquidity_boundaries(usd: float, expected: tuple[str, str]) -> None:
    assert _rule(_facts(liquidity_usd=usd), "LOW_LIQUIDITY") == expected


def test_dex_failure_leaves_liquidity_undetermined_never_zero(tmp_path: Path) -> None:
    svc, _, _ = setup(tmp_path, ("http", 503), max_retries=0)
    body = snap(svc)
    mk = m(body)
    assert mk["status"] == "PROVIDER_UNAVAILABLE"
    assert mk["primary_liquidity_usd"]["status"] == "PROVIDER_UNAVAILABLE"
    assert mk["primary_liquidity_usd"]["value"] is None
    assert all(flags(body)[r] == U for r in MARKET_RULE_IDS if r != "MARKET_CLOSED_ON_CHAIN")
    assert "MARKET_CLOSED_ON_CHAIN" in out_of_scope(body)  # no corroborated pool to check
    assert body["authority"]["mint_authority"]["status"] == "AVAILABLE"
    assert body["assessment"]["coverage"] == "PARTIAL"


# --- history / collapse ----------------------------------------------------------------------


def _history(
    tmp_path: Path, steps: list[tuple[float, Any]], **kw: Any
) -> tuple[Any, Clock, FakeRpc]:
    """Collect at each (hours since start, dex answer)."""
    svc, clock, rpc = setup(tmp_path, steps[0][1], **kw)
    elapsed = 0.0
    for hours, rows in steps:
        clock.advance((hours - elapsed) * HOUR)
        elapsed = hours
        rpc.dex = rows
        collect(svc)
    return svc, clock, rpc


@pytest.mark.parametrize(("now", "outcome"), [
    (30_000.0, T),  # -85 %
    (40_000.0, T),  # exactly -80 %
    (40_000.01, N),
    (50_000.0, N),  # -75 %
])  # fmt: skip
def test_same_pool_collapse(tmp_path: Path, now: float, outcome: str) -> None:
    svc, clock, _ = _history(tmp_path, [(0, [pair_row(P1, liquidity=200_000.0)]),
                                        (1, [pair_row(P1, liquidity=now)])])  # fmt: skip
    body = svc.build(CID, ts(clock.now()))
    assert flags(body)["LIQUIDITY_COLLAPSE"] == outcome
    assert m(body)["liquidity_change_pct_24h"]["value"] == round(100 * (now - 2e5) / 2e5, 6)


def test_collapse_uses_the_24h_high_of_the_same_pool(tmp_path: Path) -> None:
    svc, clock, _ = _history(tmp_path, [
        (0, [pair_row(P1, liquidity=900_000.0)]),  # older than 24 h: outside the window
        (20, [pair_row(P1, liquidity=300_000.0)]),
        (25, [pair_row(P1, liquidity=150_000.0)]),
        (30, [pair_row(P1, liquidity=100_000.0)]),
    ])  # fmt: skip
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["liquidity_change_pct_24h"]["value"] == round(100 * (1e5 - 3e5) / 3e5, 6)
    assert flags(body)["LIQUIDITY_COLLAPSE"] == N


def test_another_pools_history_is_never_bridged(tmp_path: Path) -> None:
    svc, clock, _ = _history(tmp_path, [
        (0, [pair_row(P2, liquidity=5_000_000.0)]),  # the previous primary
        (1, [pair_row(P1, liquidity=200_000.0)]),  # a different pool now
    ])  # fmt: skip
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["liquidity_change_pct_24h"]["status"] == "UNAVAILABLE"
    assert flags(body)["LIQUIDITY_COLLAPSE"] == U


def test_one_point_or_a_point_after_as_of_is_not_history(tmp_path: Path) -> None:
    svc, clock, _ = _history(tmp_path, [(0, [pair_row(P1, liquidity=200_000.0)]),
                                        (1, [pair_row(P1, liquidity=10_000.0)])])  # fmt: skip
    earlier = svc.build(CID, ts(T0))  # only the first observation is known
    assert flags(earlier)["LIQUIDITY_COLLAPSE"] == U
    assert m(earlier)["primary_liquidity_usd"]["value"] == 200_000.0
    assert flags(svc.build(CID, ts(clock.now())))["LIQUIDITY_COLLAPSE"] == T


# --- not reported ----------------------------------------------------------------------------


def _miss_history(tmp_path: Path, misses: list[float]) -> tuple[Any, Clock, FakeRpc]:
    steps: list[tuple[float, Any]] = [(0, [pair_row(P1)])] + [(h, []) for h in misses]
    svc, clock, rpc = setup(tmp_path, steps[0][1])
    rpc.accounts[P1] = EXISTS  # the pool account still exists on-chain
    elapsed = 0.0
    for hours, rows in steps:
        clock.advance((hours - elapsed) * HOUR)
        elapsed = hours
        rpc.dex = rows
        collect(svc)
    return svc, clock, rpc


@pytest.mark.parametrize(("misses", "outcome"), [
    ([1.0], U),  # one miss
    ([1.0, 1.0 + 1 / HOUR], U),  # two immediate misses, a second apart
    ([1.0, 6.99], U),  # just under 6 h apart
    ([1.0, 7.0], T),  # 2 misses exactly 6 h apart
    ([1.0, 3.0, 9.0], T),
])  # fmt: skip
def test_market_not_reported_needs_repeated_misses_over_time(
    tmp_path: Path, misses: list[float], outcome: str
) -> None:
    svc, clock, _ = _miss_history(tmp_path, misses)
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["pool_presence"]["state"] == "NOT_REPORTED"
    assert flags(body)["MARKET_NOT_REPORTED"] == outcome
    assert flags(body)["MARKET_CLOSED_ON_CHAIN"] == N  # the account exists
    assert m(body)["not_reported"]["misses"] == len(misses)
    assert MARKET_NOT_REPORTED_MIN_DURATION_S == 6 * 3600


def test_provider_failures_are_not_misses(tmp_path: Path) -> None:
    svc, clock, rpc = _miss_history(tmp_path, [1.0])
    for hours in (3, 9):  # failed reads 2 h and 8 h after the miss
        clock.advance(hours * HOUR)
        rpc.dex = ("http", 503)
        collect(svc)
    body = svc.build(CID, ts(clock.now()))
    assert flags(body)["MARKET_NOT_REPORTED"] == U  # the latest read failed
    assert m(body)["pool_presence"]["state"] == "UNAVAILABLE"
    rpc.dex = []
    collect(svc)  # a second successful miss, > 6 h after the first
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["not_reported"]["misses"] == 2
    assert flags(body)["MARKET_NOT_REPORTED"] == T


def test_a_pool_reported_again_resets_not_reported(tmp_path: Path) -> None:
    svc, clock, rpc = _miss_history(tmp_path, [1.0, 8.0])
    assert flags(svc.build(CID, ts(clock.now())))["MARKET_NOT_REPORTED"] == T
    clock.advance(HOUR)
    rpc.dex = [pair_row(P1)]
    collect(svc)
    body = svc.build(CID, ts(clock.now()))
    assert flags(body)["MARKET_NOT_REPORTED"] == N
    assert m(body)["not_reported"]["misses"] == 0


# --- on-chain closure ------------------------------------------------------------------------


def test_a_previously_reported_pool_with_no_account_is_closed_on_chain(tmp_path: Path) -> None:
    svc, clock, rpc = _history(tmp_path, [(0, [pair_row(P1)]), (1, [])])
    assert rpc.calls.count("getAccountInfo") == 3  # mint x2 + the pool once
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["pool_presence"]["state"] == "CLOSED_ON_CHAIN"
    assert m(body)["pool_account"]["outcome"] == "ACCOUNT_MISSING"
    assert flags(body)["MARKET_CLOSED_ON_CHAIN"] == T
    assert flags(body)["MARKET_NOT_REPORTED"] == U  # one miss is still only one miss


def test_an_existing_pool_account_is_not_closed(tmp_path: Path) -> None:
    svc, clock, _ = _miss_history(tmp_path, [1.0])
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["pool_account"]["outcome"] == "EXISTS"
    assert m(body)["pool_account"]["program_owner"] == OTHER_PROGRAM
    assert flags(body)["MARKET_CLOSED_ON_CHAIN"] == N


def test_a_failed_pool_account_read_is_undetermined(tmp_path: Path) -> None:
    svc, clock, rpc = setup(tmp_path, [pair_row(P1)], max_retries=0)
    collect(svc)
    clock.advance(HOUR)
    rpc.dex = []
    rpc.accounts[P1] = ("http", 503)
    got = collect(svc)
    assert got.pool_account_outcome == "PROVIDER_FAILED"
    assert svc.repo.collection(got.collection_id)[0] == "DONE"
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["pool_presence"]["state"] == "NOT_REPORTED"
    assert flags(body)["MARKET_CLOSED_ON_CHAIN"] == U


def test_an_unobserved_pool_can_never_be_closed_on_chain() -> None:
    """Even a stored ACCOUNT_MISSING for a pinned pool never reported by the DEX."""
    obs = (ObservedMarket(1, 7, 10.0, "NO_POOLS", None, ()),)
    pin = PoolPin(1, CID, P3, None, 5.0, "manual")
    check = AccountCheck(1, 7, P3, 10.0, "ACCOUNT_MISSING", None, None)
    body = build_body(CID, MINT, 10.0, None, code_fingerprints(),
                      market=MarketInputs(obs, (pin,), (check,)))  # fmt: skip
    assert m(body)["pool_presence"]["state"] == "NOT_REPORTED"
    assert "MARKET_CLOSED_ON_CHAIN" not in flags(body)
    assert "MARKET_CLOSED_ON_CHAIN" in out_of_scope(body)


# --- pool age --------------------------------------------------------------------------------


@pytest.mark.parametrize(("age_hours", "outcome", "severity"), [
    (24 * 30, N, "medium"), (72, N, "medium"), (71.5, T, "medium"), (24, T, "medium"),
    (23.5, T, "high"), (0, T, "high"),
])  # fmt: skip
def test_pool_age_boundaries(tmp_path: Path, age_hours: float, outcome: str, severity: str) -> None:
    svc, _, _ = setup(tmp_path, [pair_row(P1, created=T0 - timedelta(hours=age_hours))])
    body = snap(svc)
    assert m(body)["pool_age_hours"]["value"] == age_hours
    assert (flag(body, "VERY_NEW_POOL")["outcome"],
            flag(body, "VERY_NEW_POOL")["severity"]) == (outcome, severity)  # fmt: skip


@pytest.mark.parametrize("created", [T0 + timedelta(seconds=1), None])
def test_future_or_missing_pool_creation_is_undetermined(tmp_path: Path, created: Any) -> None:
    svc, _, _ = setup(tmp_path, [pair_row(P1, created=created)])
    body = snap(svc)
    age = m(body)["pool_age_hours"]
    assert age["value"] is None
    assert age["status"] == ("UNKNOWN" if created else "UNAVAILABLE")
    assert flags(body)["VERY_NEW_POOL"] == U


# --- coverage --------------------------------------------------------------------------------


def test_a_snapshot_without_market_evidence_stays_reproducible(tmp_path: Path) -> None:
    svc, clock, rpc = setup(tmp_path, [pair_row(P1)])
    old = asyncio.run(svc.snapshot(CID))  # mint only: no market observation
    assert {o["id"] for o in old.body["coverage"]["out_of_scope"]} >= set(MARKET_RULE_IDS)
    assert old.body["assessment"]["coverage"] == "COMPLETE"
    assert old.body["coverage"]["components"]["market"] == "UNAVAILABLE"
    clock.advance(60)
    collect(svc)  # market evidence arrives later
    assert svc.rebuild(old.snapshot_id or 0).status == "REPRODUCED"
    assert "dex.tokenPairs" in rpc.calls


def test_a_market_observation_activates_the_market_rules(tmp_path: Path) -> None:
    svc, _, _ = setup(tmp_path, [pair_row(P1)])
    body = snap(svc)
    active = set(MARKET_RULE_IDS) - {"MARKET_CLOSED_ON_CHAIN"}  # healthy: closure N/A
    assert active <= set(body["coverage"]["decision_rules"])
    assert out_of_scope(body) & set(MARKET_RULE_IDS) == {"MARKET_CLOSED_ON_CHAIN"}
    assert body["assessment"]["coverage"] == "PARTIAL"  # one point: collapse undetermined


def test_missing_optional_metadata_never_forces_partial(tmp_path: Path) -> None:
    row = pair_row(P1, volume=None)
    svc, clock, _ = _history(tmp_path, [(0, [row]), (1, [row])])
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["volume_24h"]["status"] == "UNAVAILABLE"
    assert body["assessment"]["coverage"] == "COMPLETE"  # closure out of scope, no RPC


# --- causality -------------------------------------------------------------------------------


def test_observation_at_exactly_as_of_is_used_and_future_inputs_raise() -> None:
    pool = classify_market(MINT, [pair_row(P1)]).pools
    obs = ObservedMarket(1, 1, 10.0, "POOLS", None, pool)
    ok = build_body(CID, MINT, 10.0, None, code_fingerprints(), market=MarketInputs((obs,)))
    assert m(ok)["pool_presence"]["state"] == "REPORTED"
    with pytest.raises(SafetyCausalityError, match="market observation"):
        build_body(CID, MINT, 9.9, None, code_fingerprints(), market=MarketInputs((obs,)))
    check = AccountCheck(1, 1, P1, 11.0, "EXISTS", OTHER_PROGRAM, None)
    with pytest.raises(SafetyCausalityError, match="pool account"):
        build_body(CID, MINT, 10.0, None, code_fingerprints(),
                   market=MarketInputs((obs,), accounts=(check,)))  # fmt: skip
    pin = PoolPin(1, CID, P1, None, 11.0, "manual")
    with pytest.raises(SafetyCausalityError, match="pool pin"):
        build_body(CID, MINT, 10.0, None, code_fingerprints(),
                   market=MarketInputs((obs,), pins=(pin,)))  # fmt: skip


def test_later_market_evidence_never_changes_an_older_snapshot(tmp_path: Path) -> None:
    svc, clock, rpc = setup(tmp_path, [pair_row(P1, liquidity=200_000.0)])
    first = asyncio.run(svc.snapshot(CID, market=True))
    for liquidity in (100_000.0, 10_000.0):
        clock.advance(HOUR)
        rpc.dex = [pair_row(P1, liquidity=liquidity)]
        collect(svc)
    clock.advance(HOUR)
    rpc.dex = []
    collect(svc)
    assert svc.rebuild(first.snapshot_id or 0).status == "REPRODUCED"
    assert svc.build(CID, ts(first.as_of)) == first.body


# --- hashing / fingerprints ------------------------------------------------------------------


def test_same_market_inputs_give_the_same_hash(tmp_path: Path) -> None:
    hashes = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        svc, _, _ = setup(tmp_path / name, [pair_row(P1), pair_row(P2, liquidity=20_000.0)])
        res = asyncio.run(svc.snapshot(CID, market=True))
        assert svc.rebuild(res.snapshot_id or 0).status == "REPRODUCED"
        hashes.append(res.body_hash)
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize(("attr", "name", "key"), [
    ("SOLANA_DEX", "solana_dex.py", "solana_dex_source"),
    ("DEXSCREENER", "dexscreener.py", "dexscreener_source"),
])  # fmt: skip
def test_a_changed_dex_helper_refuses_exact_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attr: str, name: str, key: str
) -> None:
    svc, _, _ = setup(tmp_path, [pair_row(P1)])
    res = asyncio.run(svc.snapshot(CID, market=True))
    changed = tmp_path / name
    changed.write_bytes(getattr(features, attr).read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(features, attr, changed)
    features.code_fingerprints.cache_clear()
    svc._fingerprints = features.code_fingerprints
    try:
        rb = svc.rebuild(res.snapshot_id or 0)
    finally:
        monkeypatch.undo()
        features.code_fingerprints.cache_clear()
    assert rb.status == "FINGERPRINT_MISMATCH" and set(rb.mismatched) == {key}


def test_market_module_is_fingerprinted() -> None:
    assert "market.py" in features.FINGERPRINTED


# --- failures / durability / bounds ----------------------------------------------------------


def test_market_failure_keeps_mint_and_holder_evidence(tmp_path: Path) -> None:
    from tests.safety_v2_fakes import HolderChain, num_addr

    chain = HolderChain()
    for i in range(150):
        chain.add(num_addr("Hdr", i), 1_000_000)
    rpc = FakeRpc({MINT: mint_value()}, holders=chain, dex=("timeout",))
    svc, _, _ = make_service(tmp_path, rpc, dex=True, helius=True, max_retries=1)
    svc.add_target(MINT)
    res = asyncio.run(svc.snapshot(CID, holders=True, market=True))
    assert res.collected is not None
    assert (res.collected.outcome, res.collected.holder_outcome,
            res.collected.market_outcome) == ("MINT", "COLLECTED", "PROVIDER_FAILED")  # fmt: skip
    comps = res.body["coverage"]["components"]
    assert (comps["mint_account"], comps["holders"], comps["market"]) == (
        "AVAILABLE", "AVAILABLE", "PROVIDER_UNAVAILABLE")  # fmt: skip
    assert svc.repo.collection(res.collected.collection_id)[0] == "DONE"
    assert rpc.calls.count("dex.tokenPairs") == 2  # one retry, then failed


def test_exhausted_budget_makes_no_dex_request(tmp_path: Path) -> None:
    svc, _, rpc = setup(tmp_path, [pair_row(P1)], daily_request_budget=1)
    body = snap(svc)
    assert rpc.calls == ["getAccountInfo"]
    assert m(body)["status"] == "NOT_COLLECTED"
    assert m(body)["pool_presence"]["state"] == "NOT_COLLECTED"
    assert all(flags(body)[r] == U for r in MARKET_RULE_IDS if r != "MARKET_CLOSED_ON_CHAIN")
    assert "MARKET_CLOSED_ON_CHAIN" in out_of_scope(body)  # no corroborated pool to check


def test_no_market_provider_configured_is_not_collected(tmp_path: Path) -> None:
    svc, _, rpc = make_service(tmp_path, FakeRpc({MINT: mint_value()}))  # no DEX URL
    svc.add_target(MINT)
    got = asyncio.run(svc.collect(CID, market=True))
    assert got.market_outcome == "NOT_COLLECTED" and "dex.tokenPairs" not in rpc.calls


def test_unexpected_market_error_aborts_the_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, _, _ = setup(tmp_path, [pair_row(P1)])

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(service_module, "classify_market", boom)
    with pytest.raises(RuntimeError, match="bug"):
        collect(svc)
    statuses = [r[0] for r in svc.repo.db().execute("SELECT status FROM safety_collections")]
    assert statuses == ["ABORTED"]


def test_market_tables_are_append_only(tmp_path: Path) -> None:
    import sqlite3

    svc, _, _ = _history(tmp_path, [(0, [pair_row(P1)]), (1, [])])
    conn = svc.repo.db()
    for table in ("safety_market_observations", "safety_market_pools",
                  "safety_pool_account_observations"):  # fmt: skip
        for sql in (f"UPDATE {table} SET rowid = rowid", f"DELETE FROM {table}"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(sql)
    with pytest.raises(sqlite3.IntegrityError, match="RUNNING"):
        conn.execute(
            "INSERT INTO safety_market_pools (observation_id, rank, pair_address, identity, "
            "chain, dex, base_mint, quote_address, quote_kind, eligible, rejections_json) "
            "VALUES (1, 99, ?, 'OTHER_MINT', 'solana', 'x', 'y', 'z', 'other', 0, '[\"r\"]')",
            (P3,),
        )
    assert (
        conn.execute("SELECT COUNT(*) FROM safety_collections WHERE status = 'RUNNING'").fetchone()[
            0
        ]
        == 0
    )


def test_dishonest_market_rows_are_rejected(tmp_path: Path) -> None:
    import sqlite3

    svc, _, _ = setup(tmp_path, [pair_row(P1)])
    cid = svc.repo.start_collection(CID, ts(T0))
    conn = svc.repo.db()
    bad = [
        # a failed read with a primary pool / raw hash
        ("INSERT INTO safety_market_observations (canonical_id, collection_id, fetched_at, "
         "provider, origin, outcome, reason, raw_hash, rows_returned, malformed_rows, "
         "primary_pool, primary_clear, reasons_json) VALUES (?, ?, ?, 'x', 'o', "
         "'PROVIDER_FAILED', 'r', 'h', 0, 0, NULL, NULL, '[]')", (CID, cid, ts(T0))),
        # an existing account without an owner
        ("INSERT INTO safety_pool_account_observations (canonical_id, collection_id, "
         "pool_address, fetched_at, provider, outcome, raw_hash) VALUES (?, ?, ?, ?, 'x', "
         "'EXISTS', 'h')", (CID, cid, P1, ts(T0))),
        # a missing account with lamports
        ("INSERT INTO safety_pool_account_observations (canonical_id, collection_id, "
         "pool_address, fetched_at, provider, outcome, lamports, raw_hash) VALUES "
         "(?, ?, ?, ?, 'x', 'ACCOUNT_MISSING', 5, 'h')", (CID, cid, P1, ts(T0))),
    ]  # fmt: skip
    for sql, args in bad:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, args)
    mid = svc.repo.record_market_observation(CID, cid, ts(T0), "x",
                                             classify_market(MINT, [pair_row(P1)]))  # fmt: skip
    with pytest.raises(sqlite3.IntegrityError, match="target mint"):
        conn.execute(
            "INSERT INTO safety_market_pools (observation_id, rank, pair_address, identity, "
            "chain, dex, base_mint, quote_address, quote_kind, eligible, rejections_json) "
            "VALUES (?, 9, ?, 'EXACT_BASE', 'solana', 'x', ?, 'z', 'sol', 0, '[]')",
            (mid, P3, addr("MpostorAA")),
        )
    svc.repo.finish_collection(cid, "DONE", ts(T0), 0, [])


def test_market_requests_are_bounded(tmp_path: Path) -> None:
    svc, clock, rpc = setup(tmp_path, [pair_row(P1)])
    healthy = collect(svc)
    assert healthy.requests == 2 and rpc.calls == ["getAccountInfo", "dex.tokenPairs"]
    clock.advance(HOUR)
    rpc.dex = []
    rpc.calls.clear()
    missing = collect(svc)
    assert missing.requests == 3  # mint + DEX + exactly one pool check
    assert rpc.calls == ["getAccountInfo", "dex.tokenPairs", "getAccountInfo"]


def test_market_collection_emits_no_production_evidence(tmp_path: Path) -> None:
    received: list[str] = []

    class Sink:
        def submit(self, kind: str, obj: Any, component: str, extra: Any) -> None:
            received.append(kind)

    previous = hooks.installed()
    hooks.install(Sink())
    try:
        svc, _, _ = setup(tmp_path, [pair_row(P1)])
        snap(svc)
    finally:
        hooks.install(previous)
    assert received == []


def test_cli_market_flags_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                  monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    import json

    from upscale.services.safety_v2.cli import main

    for key in ("UPSCALE_HELIUS_API_KEY", "UPSCALE_SOLANA_RPC_URL", "UPSCALE_SAFETY_V2_DEX_URL"):
        monkeypatch.delenv(key, raising=False)
    db = ["--db", str(tmp_path / "s.sqlite3")]
    assert main([*db, "targets", "add", "--token", MINT]) == 0
    assert main([*db, "collect", "--token", MINT, "--json"]) == 0
    got = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert got["market_outcome"] == "NOT_COLLECTED"  # no DEX URL: nothing requested
    assert main([*db, "collect", "--token", MINT, "--no-market", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["market_outcome"] is None


# --- closure-rule applicability (hardening) --------------------------------------------------


def test_healthy_reported_market_is_complete_without_a_pool_rpc(tmp_path: Path) -> None:
    svc, clock, rpc = _history(tmp_path, [(0, [pair_row(P1)]), (1, [pair_row(P1)])])
    body = svc.build(CID, ts(clock.now()))
    assert rpc.calls.count("getAccountInfo") == 2  # the mint, twice; never the pool
    assert m(body)["pool_presence"]["state"] == "REPORTED"
    assert "MARKET_CLOSED_ON_CHAIN" in out_of_scope(body)
    assert "MARKET_CLOSED_ON_CHAIN" not in flags(body)
    assert body["assessment"]["coverage"] == "COMPLETE"


def test_missing_pool_with_missing_account_is_closed(tmp_path: Path) -> None:
    svc, clock, rpc = _history(tmp_path, [(0, [pair_row(P1)]), (1, [])])
    body = svc.build(CID, ts(clock.now()))
    assert rpc.calls.count("getAccountInfo") == 3  # mint x2 + the pool exactly once
    assert m(body)["closure_check"]["applicable"] is True
    assert (m(body)["pool_presence"]["state"], flags(body)["MARKET_CLOSED_ON_CHAIN"]) == (
        "CLOSED_ON_CHAIN", T)  # fmt: skip
    assert "MARKET_CLOSED_ON_CHAIN" not in out_of_scope(body)


def test_missing_pool_with_existing_account_stays_not_reported(tmp_path: Path) -> None:
    svc, clock, rpc = _miss_history(tmp_path, [1.0])
    body = svc.build(CID, ts(clock.now()))
    assert rpc.calls.count("getAccountInfo") == 3
    assert (m(body)["pool_presence"]["state"], flags(body)["MARKET_CLOSED_ON_CHAIN"]) == (
        "NOT_REPORTED", N)  # fmt: skip


def test_missing_pool_with_failed_rpc_is_undetermined_and_partial(tmp_path: Path) -> None:
    svc, clock, rpc = setup(tmp_path, [pair_row(P1)], max_retries=0)
    collect(svc)
    clock.advance(HOUR)
    rpc.dex = []
    rpc.accounts[P1] = ("http", 503)
    collect(svc)
    body = svc.build(CID, ts(clock.now()))
    assert flags(body)["MARKET_CLOSED_ON_CHAIN"] == U
    assert body["assessment"]["coverage"] == "PARTIAL"


@pytest.mark.parametrize("kind", ["no_rpc_provider", "dex_failed_after_miss"])
def test_missing_pool_without_usable_rpc_evidence_is_never_not_triggered(
    tmp_path: Path, kind: str
) -> None:
    if kind == "no_rpc_provider":
        rpc = FakeRpc({MINT: mint_value()}, dex=[pair_row(P1)])
        svc, clock, rpc = make_service(tmp_path, rpc, dex=True, provider=False)
        svc.add_target(MINT)
        collect(svc)
        clock.advance(HOUR)
        rpc.dex = []
        got = collect(svc)
        assert got.pool_account_outcome == "NOT_COLLECTED"
    else:
        svc, clock, rpc = _miss_history(tmp_path, [1.0])
        clock.advance(HOUR)
        rpc.dex = ("http", 503)
        collect(svc)  # the latest DEX read failed: no check in this collection
    body = svc.build(CID, ts(clock.now()))
    assert m(body)["closure_check"]["applicable"] is True
    assert flags(body)["MARKET_CLOSED_ON_CHAIN"] == U


def test_reported_again_returns_closure_to_out_of_scope(tmp_path: Path) -> None:
    svc, clock, rpc = _miss_history(tmp_path, [1.0, 8.0])
    assert flags(svc.build(CID, ts(clock.now())))["MARKET_CLOSED_ON_CHAIN"] == N
    clock.advance(HOUR)
    rpc.dex = [pair_row(P1)]
    rpc.calls.clear()
    collect(svc)
    body = svc.build(CID, ts(clock.now()))
    assert "MARKET_CLOSED_ON_CHAIN" in out_of_scope(body)
    assert flags(body)["MARKET_NOT_REPORTED"] == N
    assert rpc.calls == ["getAccountInfo", "dex.tokenPairs"]  # no pool check


def test_not_reported_and_closed_on_chain_stay_independent(tmp_path: Path) -> None:
    svc, clock, _ = _miss_history(tmp_path, [1.0, 7.5])  # missing >= 6 h, account exists
    body = svc.build(CID, ts(clock.now()))
    assert flags(body)["MARKET_NOT_REPORTED"] == T
    assert flags(body)["MARKET_CLOSED_ON_CHAIN"] == N
    assert m(body)["pool_presence"]["state"] == "NOT_REPORTED"
