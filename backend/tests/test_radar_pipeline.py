"""Radar V1 pipeline end to end over a fake chain: target -> holders -> incremental pool
scan -> flows -> consecutive-snapshot changes -> repeated wallets / timing -> snapshot.
Covers causality, determinism, unavailable != zero, creator semantics, partial coverage,
budget enforcement and optional enrichment. Offline."""

import asyncio
import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest

from tests.radar_fakes import MINT, POOL, T0, Clock, FakeChain, addr, make_service, tx
from upscale.services.radar.features import timing_clusters
from upscale.services.radar.models import RadarCausalityError, ts
from upscale.services.radar.parsing import parse_transaction
from upscale.services.radar.repository import FlowRow, encode_body
from upscale.services.radar.service import RadarService
from upscale.services.radar.wallet_history import describe_wallet

A, B, C, D = addr("WaAAA"), addr("WaBBB"), addr("WaCCC"), addr("WaDDD")
DEP, CRE, FUNDER = addr("DepAAA"), addr("CreAAA"), addr("FundAAA")
CREATED = T0 - timedelta(hours=1)
CID = f"solana:{MINT}"


def base_chain(init_mint: bool = True) -> FakeChain:
    chain = FakeChain(holders={A: 300_000, B: 100_000, C: 100_000, POOL: 500_000})
    chain.add(MINT, tx("sigMint1", CREATED - timedelta(minutes=5), DEP,
                       init_mint=MINT if init_mint else None))  # fmt: skip
    s = timedelta(seconds=1)
    chain.add(
        POOL,
        tx("sigP1", CREATED, CRE, pre=[(CRE, 1_000_000)], post=[(CRE, 0), (POOL, 1_000_000)]),
        tx("sigP2", CREATED + 10 * s, A, pre=[(POOL, 1_000_000)], post=[(POOL, 700_000), (A, 300_000)]),
        tx("sigP3", CREATED + 12 * s, B, pre=[(POOL, 700_000)], post=[(POOL, 600_000), (B, 100_000)]),
        tx("sigP4", CREATED + 13 * s, C, pre=[(POOL, 600_000)], post=[(POOL, 500_000), (C, 100_000)]),
    )  # fmt: skip
    return chain


def second_round(chain: FakeChain, clock: Clock) -> None:
    clock.advance(3600)
    s = timedelta(seconds=1)
    chain.add(
        POOL,
        tx("sigP5", T0 + 600 * s, A, pre=[(POOL, 500_000), (A, 300_000)], post=[(POOL, 700_000), (A, 100_000)]),
        tx("sigP6", T0 + 900 * s, D, pre=[(POOL, 700_000)], post=[(POOL, 650_000), (D, 50_000)]),
    )  # fmt: skip
    chain.holders = {A: 100_000, B: 100_000, C: 100_000, D: 50_000, POOL: 650_000}


def setup(
    tmp_path: Any, chain: FakeChain | None = None, **kw: Any
) -> tuple[RadarService, Clock, FakeChain]:
    clock = Clock()
    chain = chain or base_chain()
    svc = make_service(tmp_path, chain, clock, **kw)
    svc.add_target(MINT, POOL, "pumpswap", CREATED)
    return svc, clock, chain


def metrics(body: Any) -> Iterator[dict[str, Any]]:
    if isinstance(body, dict):
        if set(body) == {"status", "value", "lower_bound", "reason"}:
            yield body
        else:
            for v in body.values():
                yield from metrics(v)
    elif isinstance(body, list):
        for v in body:
            yield from metrics(v)


def test_first_snapshot_end_to_end(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    r = asyncio.run(svc.snapshot(CID))
    b = r.body
    assert b["schema_version"] == "radar.snapshot.v1" and r.saved
    assert b["identity"]["mint"] == MINT and b["identity"]["pool_address"] == POOL
    h = b["holders"]
    assert h["source"] == "full_scan"
    assert h["holder_count"] == {
        "status": "AVAILABLE",
        "value": 3,
        "lower_bound": False,
        "reason": None,
    }
    assert h["top1_pct"]["value"] == 30.0 and h["top10_pct"]["value"] == 50.0
    assert h["top10_change_pp"]["status"] == "NOT_COLLECTED"  # no baseline yet: not zero
    a = b["activity"]
    assert a["interacting_wallets"]["status"] == "AVAILABLE"  # whole (short) history listed
    assert a["token_inflow_wallets"]["value"] == 3  # A, B, C
    assert a["token_outflow_wallets"]["value"] == 1  # CRE seeding the pool
    assert a["tracked_pool_counterparty_inflow_wallets"]["value"] == 3
    e = b["early_activity"]
    assert (
        e["early_wallet_count"]["status"] == "AVAILABLE" and e["early_wallet_count"]["value"] == 4
    )
    # Timing: A, B, C entered within 3 seconds.
    assert b["clusters"]["coordinated_timing"]["count"]["value"] == 1
    assert b["clusters"]["coordinated_timing"]["largest_wallets"] == 3
    assert b["clusters"]["funding"]["count"]["status"] == "NOT_COLLECTED"
    # Creator semantics: a candidate from the pool, a deployer only from the mint init tx.
    cand, dep = b["creators"]["pool_creator_candidate"], b["creators"]["token_deployer"]
    assert (cand["status"], cand["identity"]) == ("CANDIDATE", CRE)
    assert cand["method"] == "FEE_PAYER_OF_OLDEST_SUCCESSFUL_POOL_TX"
    assert (dep["status"], dep["identity"]) == ("VERIFIED", DEP)
    assert dep["method"] == "FEE_PAYER_OF_MINT_INITIALIZATION_TX"
    assert dep["signature"] == "sigMint1"
    assert b["creators"]["candidate_matches_verified_deployer"] is False
    assert b["creators"]["sell_detected"]["status"] == "NOT_SUPPORTED"
    assert (
        b["creators"]["pool_creator_candidate"]["token_flows"]["token_outflow_events"]["value"] == 1
    )
    # Bounded request count: holders 5 + activity 1+4 + early 1+1 + deployer 1+1.
    assert r.requests == 14 and sum(chain.calls.values()) == 14
    assert {k: v.requests for k, v in r.steps.items()} == {
        "holders": 5, "activity": 5, "early": 2, "deployer": 2}  # fmt: skip
    # Nothing raw is stored.
    conn = sqlite3.connect(svc.settings.db_path)
    cols = {
        c[1]
        for t in ("radar_tx", "radar_wallet_flows")
        for c in conn.execute(f"PRAGMA table_info({t})")
    }
    assert not any("json" in c or "raw_tx" in c for c in cols)


def test_second_snapshot_changes_and_incremental_scan(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    second_round(chain, clock)
    chain.calls.clear()
    chain.params.clear()
    r = asyncio.run(svc.snapshot(CID))
    b = r.body
    # Incremental: only signatures after the cursor, no new early / deployer work.
    sig_calls = [p for m, p in chain.params if m == "getSignaturesForAddress"]
    assert sig_calls == [[POOL, {"limit": 1000, "commitment": "confirmed", "until": "sigP4"}]]
    assert chain.calls["getTransaction"] == 2 and r.requests == 8
    h = b["holders"]
    assert h["holder_count"]["value"] == 4
    assert h["holder_count_change"] == {
        "status": "AVAILABLE",
        "value": 1,
        "lower_bound": False,
        "reason": None,
    }
    assert h["top1_change_pp"]["value"] == -20.0
    assert h["top10_change_pp"]["value"] == -15.0
    lg = b["large_holders"]
    assert lg["large_holder_accumulation_count"]["value"] == 1
    assert lg["large_holder_reduction_count"]["value"] == 1
    # A and D signed pool transactions Radar read: proven wallets.
    assert lg["large_wallet_accumulation_count"]["value"] == 1
    assert lg["large_wallet_reduction_count"]["value"] == 1
    assert lg["large_wallet_exit_count"] == {"status": "AVAILABLE", "value": 0,
                                             "lower_bound": False, "reason": None}  # fmt: skip
    kinds = {c["owner"]: (c["kind"], c["change_is_lower_bound"]) for c in lg["changes"]}
    assert kinds[A] == ("LARGE_HOLDER_REDUCTION", False)
    assert kinds[D] == ("LARGE_HOLDER_ACCUMULATION", True)  # was below the threshold before
    a = b["activity"]
    assert a["token_inflow_wallets"]["value"] == 1 and a["token_outflow_wallets"]["value"] == 1
    assert a["net_inflow_wallets"]["value"] == 0  # a real, complete net of zero
    assert a["tracked_pool_counterparty_outflow_wallets"]["value"] == 1


def test_causality_snapshot_never_sees_later_data(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    first = asyncio.run(svc.snapshot(CID))
    second_round(chain, clock)
    asyncio.run(svc.snapshot(CID))
    mid = T0 + timedelta(minutes=30)
    body = svc.build(CID, mid)
    assert body["holders"]["holder_count"]["value"] == 3  # D's holder data came later
    inp = svc.repo.load_inputs(CID, ts(mid))
    assert D not in {f.wallet for f in inp.flows}
    assert all(f.fetched_at <= ts(mid) for f in inp.flows)
    # Audit-style lookup: the latest snapshot observed at or before the decision.
    got = svc.repo.snapshot_as_of(CID, ts(mid))
    assert got is not None and got[0] == ts(first.observed_at)
    assert svc.repo.snapshot_as_of(CID, ts(T0) - 1) is None
    with pytest.raises(RadarCausalityError):
        svc.build(CID, T0 - timedelta(days=1))  # before the target existed


def test_causality_guard_rejects_future_rows(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, clock, _ = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    real = svc.repo._holder_rows
    monkeypatch.setattr(svc.repo, "_holder_rows", lambda c, a, n: real(c, a + 10_000, n))
    with pytest.raises(RadarCausalityError):
        svc.repo.load_inputs(CID, ts(T0) - 5)
    with pytest.raises(RadarCausalityError):
        future = parse_transaction(tx("future", T0 + timedelta(hours=1), A), MINT, POOL)
        assert future is not None
        svc.repo.record_tx(CID, future, ts(T0), "x", None, None)  # block time after fetch


def test_deterministic_snapshot(tmp_path: Any) -> None:
    svc, clock, _ = setup(tmp_path)
    r = asyncio.run(svc.snapshot(CID))
    run = {k: v.as_dict() for k, v in r.steps.items()}
    again = svc.build(CID, r.observed_at, run)
    assert again == r.body
    assert encode_body(again)[2] == r.body_hash
    stored = svc.repo.snapshot_as_of(CID, ts(r.observed_at))
    assert stored is not None and stored[1] == r.body
    with pytest.raises(sqlite3.IntegrityError):  # immutable
        with sqlite3.connect(svc.settings.db_path) as conn:
            conn.execute("UPDATE radar_snapshots SET coverage = 'X'")


def test_unavailable_is_never_zero(tmp_path: Any) -> None:
    clock = Clock()
    svc = make_service(tmp_path, None, clock)  # no provider configured
    svc.add_target(MINT, POOL)
    r = asyncio.run(svc.snapshot(CID))
    b = r.body
    assert r.requests == 0
    assert b["holders"]["holder_count"]["status"] == "PROVIDER_UNAVAILABLE"
    assert b["activity"]["interacting_wallets"]["status"] == "PROVIDER_UNAVAILABLE"
    assert b["early_activity"]["early_wallet_count"]["status"] == "UNAVAILABLE"  # creation unknown
    assert b["coverage"]["overall"] == "EMPTY"
    for m in metrics(b):
        assert m["value"] is None and m["status"] not in ("AVAILABLE", "PARTIAL"), m


def test_partial_holder_scan_is_lower_bound_and_changes_unavailable(tmp_path: Any) -> None:
    chain = base_chain()
    chain.das_pages_limit = 1
    svc, clock, _ = setup(tmp_path, chain, holder_max_pages=2)
    b = asyncio.run(svc.snapshot(CID)).body
    assert b["holders"]["source"] == "partial_scan"
    hc = b["holders"]["holder_count"]
    assert hc["status"] == "PARTIAL" and hc["lower_bound"] and hc["value"] == 3
    assert b["holders"]["top10_pct"]["status"] == "PARTIAL"
    assert chain.calls["getTokenAccounts"] == 2  # Radar's own page cap
    clock.advance(3600)
    b2 = asyncio.run(svc.snapshot(CID)).body
    assert b2["holders"]["top10_change_pp"]["status"] == "UNAVAILABLE"
    assert b2["holders"]["holder_count_change"]["status"] == "UNAVAILABLE"
    assert b2["large_holders"]["large_holder_accumulation_count"]["status"] == "PARTIAL"
    assert b2["large_holders"]["large_wallet_exit_count"]["status"] == "UNAVAILABLE"


def test_history_cap_makes_early_and_creator_unavailable(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, signature_page_size=2, early_max_sig_pages=1,
                              deployer_max_sig_pages=1)  # fmt: skip
    chain.add(MINT, tx("sigMint2", CREATED, A))  # 3 mint signatures: beyond one page of 2
    chain.add(MINT, tx("sigMint3", CREATED + timedelta(seconds=1), A))
    b = asyncio.run(svc.snapshot(CID)).body
    assert b["activity"]["interacting_wallets"]["status"] == "PARTIAL"
    assert "first scan" in (b["activity"]["interacting_wallets"]["reason"] or "")
    cand = b["creators"]["pool_creator_candidate"]
    assert cand["status"] == "UNAVAILABLE" and cand["identity"] is None
    assert cand["provenance"]["reason"] == "HISTORY_CAP_REACHED"
    dep = b["creators"]["token_deployer"]
    assert dep["status"] == "UNAVAILABLE" and dep["identity"] is None
    assert b["creators"]["candidate_matches_verified_deployer"] is None
    # pool_created_at is known, so the early window exists, but only partly observed.
    assert b["early_activity"]["early_wallet_count"]["status"] == "PARTIAL"


def test_deployer_requires_mint_initialization(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, base_chain(init_mint=False))
    b = asyncio.run(svc.snapshot(CID)).body
    dep = b["creators"]["token_deployer"]
    assert dep["status"] == "UNAVAILABLE" and dep["identity"] is None
    assert dep["provenance"]["reason"] == "OLDEST_MINT_TX_DOES_NOT_INITIALIZE_MINT"
    assert b["creators"]["pool_creator_candidate"]["identity"] == CRE  # never promoted
    with sqlite3.connect(svc.settings.db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO radar_creators VALUES (?, 'POOL_CREATOR_CANDIDATE', 'VERIFIED', ?, 'm', "
            "NULL, NULL, 0, 'p', '{}')",
            ("solana:x", CRE),
        )


def test_provider_failures_are_isolated_per_step(tmp_path: Any) -> None:
    chain = base_chain()
    chain.fail = lambda m: "timeout" if m == "getTransaction" else None
    svc, clock, _ = setup(tmp_path, chain, max_retries=1)
    r = asyncio.run(svc.snapshot(CID))
    assert r.steps["holders"].status == "AVAILABLE"
    assert r.steps["activity"].status == "PROVIDER_UNAVAILABLE"
    assert svc.repo.get_target(CID).last_signature is None  # type: ignore[union-attr]  # retried next time
    assert r.body["holders"]["holder_count"]["value"] == 3
    assert r.body["activity"]["interacting_wallets"]["status"] in (
        "PROVIDER_UNAVAILABLE",
        "PARTIAL",
    )


def test_429_mid_run_cools_down_remaining_steps(tmp_path: Any) -> None:
    chain = base_chain()
    chain.fail = lambda m: 429 if m == "getSignaturesForAddress" else None
    svc, clock, _ = setup(tmp_path, chain)
    r = asyncio.run(svc.snapshot(CID))
    assert r.steps["holders"].status == "AVAILABLE"
    assert r.steps["activity"].status == "PROVIDER_UNAVAILABLE"
    assert r.steps["early"].status == "PROVIDER_UNAVAILABLE"
    assert chain.calls["getSignaturesForAddress"] == 1  # the cooldown stopped the rest
    assert svc.guard.cooldown_until() is not None


def test_budget_enforced_end_to_end(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, daily_request_budget=7)
    r = asyncio.run(svc.snapshot(CID))
    assert sum(chain.calls.values()) == 7 and r.requests == 7
    assert r.steps["deployer"].status == "NOT_COLLECTED"
    later = asyncio.run(svc.snapshot_active(5))
    assert later[0] == (CID, "NOT_COLLECTED: daily request budget spent")
    assert sum(chain.calls.values()) == 7


def test_repeated_wallets_and_groups_across_targets(tmp_path: Any) -> None:
    chain = base_chain()
    mint2, pool2 = addr("MintBBB"), addr("PooLBBB")
    s = timedelta(seconds=1)
    created2 = CREATED - timedelta(hours=5)
    chain.add(pool2, *[
        tx(f"q{i}", created2 + i * 60 * s, w, pre=[(pool2, 100)], post=[(pool2, 90), (w, 10)], mint=mint2)
        for i, w in enumerate([A, B, C, D])
    ])  # fmt: skip
    chain.add(mint2, tx("m2", created2, DEP, init_mint=mint2))
    svc, clock, _ = setup(tmp_path, chain)
    svc.add_target(mint2, pool2, "raydium", created2)
    asyncio.run(svc.snapshot(f"solana:{mint2}"))
    clock.advance(60)
    b = asyncio.run(svc.snapshot(CID)).body
    rp = b["repeated"]
    assert rp["repeated_wallet_count"]["status"] == "PARTIAL"  # Radar targets only
    assert rp["repeated_wallet_count"]["value"] == 3  # A, B, C (D only in the other token)
    assert rp["repeated_wallet_group_count"]["value"] == 1
    assert rp["groups"][0] == {"label": "REPEATED_WALLET_GROUP", "other_token": f"solana:{mint2}",
                               "shared_wallets": 3}  # fmt: skip
    assert b["early_activity"]["repeated_early_wallet_count"]["value"] == 3
    assert b["wallet_history"]["wallets_with_history"]["value"] == 3
    assert b["wallet_history"]["wallets_meeting_minimum_unique_tokens"]["value"] == 0


def test_timing_clusters_are_deterministic_windows() -> None:
    def f(t: float, w: str, d: str = "TOKEN_INFLOW") -> FlowRow:
        return FlowRow("s", w, d, 1, "UNVERIFIED", t, t, None)

    flows = [f(0, A), f(2, B), f(4, C), f(4, A), f(30, A), f(31, B), f(100, D, "TOKEN_OUTFLOW")]
    assert timing_clusters(flows, 5, 3) == [(0.0, 3)]
    assert timing_clusters(flows, 5, 2) == [(0.0, 3), (30.0, 2)]
    assert timing_clusters(list(reversed(flows)), 5, 2) == [(0.0, 3), (30.0, 2)]


def test_optional_enrichment_off_makes_no_wallet_requests(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    r = asyncio.run(svc.snapshot(CID))
    queried = {p[0] for m, p in chain.params if m == "getSignaturesForAddress"}
    assert queried == {POOL, MINT}
    assert "wallet_profiles" not in r.steps
    assert r.body["wallet_age"]["new_wallet_count"]["status"] == "NOT_COLLECTED"


def test_optional_first_funder_builds_funding_clusters(tmp_path: Any) -> None:
    chain = base_chain()
    for i, w in enumerate((A, B, C)):
        chain.add(w, tx(f"fund{i}", CREATED - timedelta(days=1), FUNDER, funds=(FUNDER, w)))
    svc, clock, _ = setup(tmp_path, chain, wallet_age=True, first_funder=True)
    r = asyncio.run(svc.snapshot(CID))
    assert r.steps["wallet_profiles"].status == "AVAILABLE"
    fund = r.body["clusters"]["funding"]
    assert fund["label"] == "FUNDING_CLUSTER" and fund["count"]["value"] == 1
    assert fund["clusters"] == [{"funder": FUNDER, "wallets": 3}]
    assert r.body["wallet_age"]["profiled_wallets"]["value"] >= 3


def test_wallet_history_minimum_sample_and_no_label_leakage(tmp_path: Any) -> None:
    svc, clock, _ = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    scout = tmp_path / "scout.sqlite3"
    with sqlite3.connect(scout) as conn:
        conn.execute(
            "CREATE TABLE scout_outcome_observations (id INTEGER PRIMARY KEY, canonical_id TEXT, observed_at REAL)"
        )
        conn.execute(
            "CREATE TABLE scout_outcome_horizons (observation_id INTEGER, horizon TEXT, return_pct REAL, finalized_at REAL)"
        )
        conn.execute(
            "INSERT INTO scout_outcome_observations VALUES (1, ?, ?)", (CID, ts(CREATED) + 60)
        )
        conn.execute(
            "INSERT INTO scout_outcome_horizons VALUES (1, '4h', 80.0, ?)", (ts(T0) + 7200,)
        )
    early = describe_wallet(svc.repo, svc.settings, A, T0 + timedelta(hours=1), str(scout))
    assert early["outcomes"]["observations"] == 0  # label finalized after as_of: unused
    later = describe_wallet(svc.repo, svc.settings, A, T0 + timedelta(hours=3), str(scout))
    assert later["outcomes"]["observations"] == 1 and later["outcomes"]["hit_rate"] == 1.0
    assert later["meets_minimum_sample"] is False  # one win never qualifies
    assert later["unique_tokens"] == 1
