"""Radar V1 hardening: participant classification (programs / pools / routers / unknown
never feed wallet statistics), retroactive-history anti-lookahead, Raydium AMM v4
counterparty, creator semantics, and import isolation from Audit / Shadow. Offline."""

import ast
import asyncio
import json
import re
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.radar_fakes import MINT, POOL, T0, Clock, FakeChain, addr, make_service, tx
from tests.test_radar_pipeline import CID, CRE, CREATED, DEP, A, B, base_chain, setup
from upscale.services.radar.models import RadarCausalityError, ts
from upscale.services.radar.parsing import KNOWN_ROUTERS, classify, parse_transaction
from upscale.services.radar.repository import RadarRepository
from upscale.services.radar.wallet_history import describe_wallet
from upscale.services.solana_chain import RAYDIUM_AMM_V4_AUTHORITY, TOKEN_PROGRAM

BACKEND = Path(__file__).resolve().parents[1]
RADAR = BACKEND / "upscale" / "services" / "radar"
JUP = sorted(KNOWN_ROUTERS)[0]
PROG = addr("ProgXYZ")  # a custom program the transaction invokes
RELAY = addr("ReLayXYZ")  # a configured intermediary
UNK = addr("UnkXYZ")  # a non-signer nobody can identify
BOT = addr("BotXYZ")  # fee payer with no token movement


# --- 1. classification ----------------------------------------------------------------------


def test_classify_uses_positive_evidence_only() -> None:
    other_pool = addr("PooLBBB")

    def c(owner: str) -> str:
        return classify(owner, signers={A}, invoked_programs={PROG}, pool_address=POOL,
                        known_pools={other_pool}, known_intermediaries={RELAY})  # fmt: skip

    assert c(A) == "NORMAL_WALLET"
    assert c(POOL) == c(other_pool) == c(RAYDIUM_AMM_V4_AUTHORITY) == "POOL_OR_VAULT"
    assert c(JUP) == c(RELAY) == "ROUTER_OR_INTERMEDIARY"
    assert c(PROG) == c(TOKEN_PROGRAM) == "PROGRAM"
    assert c(UNK) == "UNKNOWN"  # never assumed to be a wallet


def _noise_tx(sig: str, when: Any, other_pool: str, mint: str = MINT) -> dict[str, Any]:
    """Tokens moving only through non-wallet / unknown accounts, signed by a bot that
    holds none: no wallet took part."""
    owners = (JUP, PROG, other_pool, UNK, RELAY)
    return tx(sig, when, BOT, pre=[(o, 100) for o in owners], post=[(o, 150) for o in owners],
              mint=mint, programs=[PROG])  # fmt: skip


def test_non_wallets_cannot_create_fake_repeated_or_timing_signals(tmp_path: Any) -> None:
    mint2, pool2 = addr("MintBBB"), addr("PooLBBB")
    chain = FakeChain(holders={A: 10, POOL: 10})
    s = timedelta(seconds=1)
    chain.add(MINT, tx("m1", CREATED - 60 * s, DEP, init_mint=MINT))
    chain.add(mint2, tx("m2", CREATED - 60 * s, DEP, init_mint=mint2))
    # Same non-wallet / unknown participants, same seconds, in both tokens.
    chain.add(POOL, *[_noise_tx(f"n{i}", CREATED + i * s, pool2) for i in range(4)])
    chain.add(pool2, *[_noise_tx(f"k{i}", CREATED + i * s, POOL, mint2) for i in range(4)])
    clock = Clock()
    svc = make_service(tmp_path, chain, clock, known_intermediaries=(RELAY,))
    svc.add_target(mint2, pool2, "raydium", CREATED)
    svc.add_target(MINT, POOL, "raydium", CREATED)
    asyncio.run(svc.snapshot(f"solana:{mint2}"))
    clock.advance(60)
    b = asyncio.run(svc.snapshot(CID)).body

    a = b["activity"]
    assert a["participants"] == {"POOL_OR_VAULT": 1, "PROGRAM": 1,
                                 "ROUTER_OR_INTERMEDIARY": 2, "UNKNOWN": 1}  # fmt: skip
    assert (
        a["interacting_wallets"]["value"] == 0 and a["interacting_wallets"]["status"] == "PARTIAL"
    )
    assert "couldn't be proven to be wallets" in a["interacting_wallets"]["reason"]
    assert b["clusters"]["coordinated_timing"]["count"]["value"] == 0
    assert b["repeated"]["repeated_wallet_count"]["status"] == "NOT_COLLECTED"
    assert b["repeated"]["groups"] == []
    assert b["early_activity"]["early_wallet_count"]["value"] == 0
    assert b["early_activity"]["early_wallet_count"]["status"] == "PARTIAL"
    assert b["wallet_history"]["wallets_with_history"]["status"] == "NOT_COLLECTED"
    pc = b["coverage"]["participant_classification"]
    assert pc["status"] == "INCOMPLETE" and pc["participants_by_type"]["UNKNOWN"] == 1
    # Raw typed evidence is kept, but only NORMAL / UNKNOWN participants get roll-ups.
    with sqlite3.connect(svc.settings.db_path) as conn:
        typed = dict(conn.execute(
            "SELECT participant, COUNT(DISTINCT wallet) FROM radar_wallet_flows "
            "WHERE canonical_id = ? GROUP BY participant", (CID,)).fetchall())  # fmt: skip
        rolled = {r[0] for r in conn.execute("SELECT DISTINCT wallet FROM radar_wallet_entries")}
    assert typed["ROUTER_OR_INTERMEDIARY"] == 2 and typed["PROGRAM"] == 1
    assert rolled == {UNK}


def test_unknown_becomes_wallet_only_once_proven_and_only_from_then(tmp_path: Any) -> None:
    chain = base_chain()
    s = timedelta(seconds=1)
    # A transfer to UNK (UNK doesn't sign): UNK is UNKNOWN.
    chain.add(POOL, tx("u1", CREATED + 30 * s, A, pre=[(A, 300_000), (UNK, 0), (POOL, 1)],
                       post=[(A, 299_000), (UNK, 1_000), (POOL, 1)]))  # fmt: skip
    svc, clock, _ = setup(tmp_path, chain)
    first = asyncio.run(svc.snapshot(CID))
    assert first.body["activity"]["participants"]["UNKNOWN"] == 1
    assert first.body["activity"]["interacting_wallets"]["status"] == "PARTIAL"
    clock.advance(3600)
    chain.add(
        POOL,
        tx("u2", T0 + 60 * s, UNK, pre=[(UNK, 1_000), (POOL, 1)], post=[(UNK, 0), (POOL, 1_001)]),
    )
    later = asyncio.run(svc.snapshot(CID))
    assert (
        "UNKNOWN"
        not in later.body["coverage"]["participant_classification"]["participants_by_type"]
    )
    # As of the first snapshot UNK is still unproven: rebuilding it is unchanged.
    run = {k: v.as_dict() for k, v in first.steps.items()}
    assert svc.build(CID, first.observed_at, run) == first.body


# --- 2. retroactive history ------------------------------------------------------------------


def test_deeper_history_never_rewrites_what_radar_knew(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    W, X = addr("WaWWW"), addr("WaXXX")
    b0 = CREATED + timedelta(seconds=5)  # W's real first activity
    b1 = T0 - timedelta(minutes=10)  # what Radar sees first
    chain = FakeChain(holders={W: 10, X: 10, POOL: 10})
    chain.add(POOL,
              tx("h0", b0, W, pre=[(POOL, 100)], post=[(POOL, 90), (W, 10)]),
              tx("h1", CREATED + timedelta(seconds=20), X, pre=[(POOL, 90)], post=[(POOL, 80), (X, 10)]),
              tx("h2", b1, W, pre=[(POOL, 80), (W, 10)], post=[(POOL, 70), (W, 20)]))  # fmt: skip
    clock = Clock()
    common = {"signature_page_size": 1, "verify_deployer": False}
    svc1 = make_service(tmp_path, chain, clock, early_max_sig_pages=0, **common)
    svc1.add_target(MINT, POOL, "pumpswap", CREATED)
    s1 = asyncio.run(svc1.snapshot(CID))  # T1: sees only h2
    t1 = ts(s1.observed_at)
    run1 = {k: v.as_dict() for k, v in s1.steps.items()}

    clock.advance(3600)  # T2: a deeper history query finds h0
    svc2 = make_service(tmp_path, chain, clock, early_max_sig_pages=5, early_max_tx=10, **common)
    svc2.repo = svc1.repo
    asyncio.run(svc2.snapshot(CID))
    t2 = ts(clock.now())

    at1, at2 = svc1.repo.wallet_entries(W, t1)[0], svc1.repo.wallet_entries(W, t2)[0]
    assert (at1.first_block_time, at1.block_time_known_at) == (ts(b1), t1)
    assert (at2.first_block_time, at2.block_time_known_at) == (ts(b0), t2)
    assert at1.first_fetched_at == at2.first_fetched_at == t1  # first sighting never moves
    # The decision-time feature set at T1 is exactly what was known at T1.
    assert svc1.build(CID, s1.observed_at, run1) == s1.body
    assert ts(b0) not in {f.block_time for f in svc1.repo.load_inputs(CID, t1).flows}
    w1 = describe_wallet(svc1.repo, svc1.settings, W, s1.observed_at)
    assert w1["tokens"][0]["first_block_time"] == b1.isoformat()
    assert w1["flows_in_retention"] == 1

    # A resolution bug that leaked the revision must fail loudly, not leak.
    monkeypatch.setattr(RadarRepository, "_resolve_entry",
                        staticmethod(lambda r, as_of: (r[5], r[6], r[7])))  # fmt: skip
    with pytest.raises(RadarCausalityError):
        svc1.repo.wallet_entries(W, t1)
    with pytest.raises(RadarCausalityError):
        svc1.repo.load_inputs(CID, t1)
    # And the storage refuses a revision learned before the first sighting.
    with sqlite3.connect(svc1.settings.db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE radar_wallet_entries SET revised_at = ? WHERE wallet = ?", (t1 - 1, W))


# --- 3. Raydium AMM v4 -----------------------------------------------------------------------


def test_raydium_v4_counterparty_not_supported_and_never_buy_sell() -> None:
    v4_pool = addr("AmmVFour")  # the pool id holds no tokens: its vaults sit under the authority
    t = tx("r1", T0, A, pre=[(RAYDIUM_AMM_V4_AUTHORITY, 1000), (A, 0)],
           post=[(RAYDIUM_AMM_V4_AUTHORITY, 900), (A, 100)])  # fmt: skip
    p = parse_transaction(t, MINT, v4_pool)
    assert p is not None
    flows = {f.wallet: (f.direction, f.counterparty, f.participant) for f in p.flows}
    assert flows[A] == ("TOKEN_INFLOW", "NOT_SUPPORTED", "NORMAL_WALLET")
    assert flows[RAYDIUM_AMM_V4_AUTHORITY][2] == "POOL_OR_VAULT"


def test_raydium_v4_counterparty_metric_is_not_supported(tmp_path: Any) -> None:
    v4_pool = addr("AmmVFour")
    chain = FakeChain(holders={A: 100, RAYDIUM_AMM_V4_AUTHORITY: 900})
    chain.add(v4_pool, tx("r1", CREATED, A, pre=[(RAYDIUM_AMM_V4_AUTHORITY, 1000)],
                          post=[(RAYDIUM_AMM_V4_AUTHORITY, 900), (A, 100)]))  # fmt: skip
    clock = Clock()
    svc = make_service(tmp_path, chain, clock, verify_deployer=False)
    svc.add_target(MINT, v4_pool, "raydium", CREATED)
    b = asyncio.run(svc.snapshot(CID)).body
    a = b["activity"]
    assert a["token_inflow_wallets"]["value"] == 1
    assert a["tracked_pool_counterparty_inflow_wallets"]["status"] == "NOT_SUPPORTED"
    assert a["tracked_pool_counterparty_inflow_wallets"]["value"] is None
    assert not re.search(r"\b(BUY|SELL)\b", json.dumps(b))


# --- 5. creator semantics --------------------------------------------------------------------


def test_candidate_never_verified_even_when_it_is_the_deployer(tmp_path: Any) -> None:
    chain = base_chain()
    chain.history[MINT] = [tx("mi", CREATED - timedelta(minutes=5), CRE, init_mint=MINT)]
    svc, clock, _ = setup(tmp_path, chain)
    c = asyncio.run(svc.snapshot(CID)).body["creators"]
    assert c["pool_creator_candidate"]["status"] == "CANDIDATE"
    assert c["token_deployer"]["status"] == "VERIFIED"
    assert c["candidate_matches_verified_deployer"] is True  # a fact, not a promotion
    collector_src = (RADAR / "collector.py").read_text()
    calls = re.findall(r'"POOL_CREATOR_CANDIDATE",\s*"(\w+)"', collector_src)
    assert calls and set(calls) <= {"CANDIDATE", "UNAVAILABLE"}


def test_verified_deployer_requires_init_of_the_exact_mint(tmp_path: Any) -> None:
    chain = base_chain()
    chain.history[MINT] = [
        tx("mo", CREATED - timedelta(minutes=5), DEP, init_mint=addr("ZtherMint"))
    ]
    svc, clock, _ = setup(tmp_path, chain)
    dep = asyncio.run(svc.snapshot(CID)).body["creators"]["token_deployer"]
    assert dep["status"] == "UNAVAILABLE" and dep["identity"] is None


def test_creator_movement_is_never_a_sell(tmp_path: Any) -> None:
    chain = base_chain()
    chain.history[MINT] = [tx("mi", CREATED - timedelta(minutes=5), CRE, init_mint=MINT)]
    svc, clock, _ = setup(tmp_path, chain)
    asyncio.run(svc.snapshot(CID))
    clock.advance(3600)
    chain.holders[CRE] = 0
    # The creator sends tokens into the tracked pool (exactly matched counterparty).
    chain.add(POOL, tx("c1", T0 + timedelta(minutes=5), CRE, pre=[(CRE, 50), (POOL, 500_000)],
                       post=[(CRE, 0), (POOL, 500_050)]))  # fmt: skip
    b = asyncio.run(svc.snapshot(CID)).body
    for role in ("pool_creator_candidate", "token_deployer"):
        flows = b["creators"][role]["token_flows"]
        assert flows["token_outflow_events"]["value"] == 2  # seeding + this transfer
    assert b["creators"]["sell_detected"]["status"] == "NOT_SUPPORTED"
    assert b["creators"]["sell_detected"]["value"] is None
    assert not re.search(r"\b(BUY|SELL)\b", json.dumps(b))


# --- 4. import isolation ---------------------------------------------------------------------


def _imports(folder: Path) -> set[str]:
    out: set[str] = set()
    for f in folder.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
            elif isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
    return out


def test_radar_and_audit_never_import_each_other() -> None:
    radar = {m for m in _imports(RADAR) if m.startswith("upscale")}
    for banned in ("upscale.services.audit", "upscale.services.shadow",
                   "upscale.services.calibration", "upscale.services.retention",
                   "upscale.services.evidence_archive", "upscale.main"):  # fmt: skip
        assert not any(m == banned or m.startswith(banned + ".") for m in radar), banned
    assert radar <= {
        "upscale", "upscale.services.chains", "upscale.services.market_data",
        "upscale.services.scout.normalize", "upscale.services.solana_chain",
    } | {m for m in radar if m.startswith("upscale.services.radar")}  # fmt: skip
    for pkg in ("audit", "shadow", "calibration", "retention", "scout"):
        assert not any("radar" in m for m in _imports(BACKEND / "upscale" / "services" / pkg)), pkg


# --- large-holder vs large-wallet metrics ----------------------------------------------------


def test_large_wallet_metrics_only_count_proven_wallets(tmp_path: Any) -> None:
    C, D = addr("WaCCC"), addr("WaDDD")
    H, P = addr("HoLderUnk"), addr("ProgXwned")  # never signs / program-owned account
    chain = base_chain()
    chain.holders.update({H: 20_000, P: 30_000})
    chain.program_owned = {P}
    svc, clock, _ = setup(tmp_path, chain)
    asyncio.run(svc.snapshot(CID))
    clock.advance(3600)
    s = timedelta(seconds=1)
    chain.add(
        POOL, tx("d1", T0 + 900 * s, D, pre=[(POOL, 500_000)], post=[(POOL, 450_000), (D, 50_000)])
    )
    # A reduces, C exits, D (proven by d1) accumulates, H (UNKNOWN) accumulates, P exits.
    chain.holders = {A: 100_000, B: 100_000, D: 50_000, H: 80_000, POOL: 450_000}
    lg = asyncio.run(svc.snapshot(CID)).body["large_holders"]

    types = {c["owner"]: c["owner_type"] for c in lg["changes"]}
    assert types == {A: "NORMAL_WALLET", C: "NORMAL_WALLET", D: "NORMAL_WALLET",
                     H: "UNKNOWN", P: "PROGRAM"}  # fmt: skip
    # Holder metrics: NORMAL_WALLET + UNKNOWN, never the program-owned account.
    assert lg["large_holder_accumulation_count"]["value"] == 2  # D, H
    assert lg["large_holder_reduction_count"]["value"] == 2  # A, C
    assert lg["large_holder_accumulation_count"]["status"] == "AVAILABLE"
    # Wallet metrics: proven NORMAL_WALLET only; H's possible wallet-ness makes them PARTIAL.
    acc, red, ex = (lg[k] for k in ("large_wallet_accumulation_count",
                                    "large_wallet_reduction_count", "large_wallet_exit_count"))  # fmt: skip
    assert (acc["value"], red["value"], ex["value"]) == (1, 2, 1)  # D / A, C / C
    assert all(
        m["status"] == "PARTIAL" and "aren't proven wallets" in m["reason"] for m in (acc, red, ex)
    )
    cov = lg["coverage"]
    assert cov["wallet_metrics_include"] == ["NORMAL_WALLET"]
    assert cov["holder_metrics_include"] == ["NORMAL_WALLET", "UNKNOWN"]
    assert (cov["unknown_owners"], cov["excluded_known_non_wallet_owners"]) == (1, 1)


# --- UNKNOWN never enters a wallet-named feature until proven --------------------------------


def test_unknown_excluded_from_wallet_features_until_proven(tmp_path: Any) -> None:
    U = [addr("UnkAAA"), addr("UnkBBB"), addr("UnkCCC")]
    mint2, pool2 = addr("MintBBB"), addr("PooLBBB")
    s = timedelta(seconds=1)

    def burst(sig: str, pool: str, mint: str) -> dict[str, Any]:
        # Three non-signing recipients in one transaction: a perfect timing burst if counted.
        return tx(sig, CREATED + 5 * s, BOT, pre=[(pool, 100)] + [(u, 0) for u in U],
                  post=[(pool, 70)] + [(u, 10) for u in U], mint=mint)  # fmt: skip

    chain = FakeChain(holders={POOL: 10})
    chain.add(POOL, burst("b1", POOL, MINT))
    chain.add(pool2, burst("b2", pool2, mint2))
    clock = Clock()
    svc = make_service(tmp_path, chain, clock, verify_deployer=False)
    svc.add_target(mint2, pool2, "raydium", CREATED)
    svc.add_target(MINT, POOL, "raydium", CREATED)
    asyncio.run(svc.snapshot(f"solana:{mint2}"))
    clock.advance(60)
    s1 = asyncio.run(svc.snapshot(CID))
    b = s1.body
    assert b["repeated"]["repeated_wallet_count"]["status"] == "NOT_COLLECTED"
    assert b["repeated"]["groups"] == []
    assert b["early_activity"]["early_wallet_count"]["value"] == 0
    assert b["early_activity"]["repeated_early_wallet_count"]["value"] == 0
    assert b["clusters"]["coordinated_timing"]["count"]["value"] == 0
    assert b["wallet_history"]["wallets_with_history"]["status"] == "NOT_COLLECTED"

    # Later each one signs a transaction Radar reads: from then on they are wallets.
    clock.advance(3600)
    chain.add(POOL, *[tx(f"p{i}", T0 + i * s, u, pre=[(u, 10), (POOL, 70)],
                         post=[(u, 0), (POOL, 80)]) for i, u in enumerate(U)])  # fmt: skip
    b2 = asyncio.run(svc.snapshot(CID)).body
    assert b2["repeated"]["repeated_wallet_count"]["value"] == 3
    assert b2["repeated"]["repeated_wallet_group_count"]["value"] == 1
    assert b2["early_activity"]["repeated_early_wallet_count"]["value"] == 3
    assert b2["clusters"]["coordinated_timing"]["count"]["value"] == 1
    assert b2["wallet_history"]["wallets_with_history"]["value"] == 3
    # ...but never before that: the decision-time features at T1 are unchanged.
    run = {k: v.as_dict() for k, v in s1.steps.items()}
    assert svc.build(CID, s1.observed_at, run) == b


# --- coverage.overall is headline coverage, not feature coverage ------------------------------


def test_complete_headline_coverage_coexists_with_incomplete_features(tmp_path: Any) -> None:
    from upscale.services.radar.cli import snapshot_text
    from upscale.services.radar.features import OVERALL_BASIS, OVERALL_NOTE

    H = addr("HoLderUnk")  # a large holder that never signs: never a proven wallet
    chain = base_chain()
    chain.holders[H] = 20_000
    # As in the live run: trading starts days after the pool was created, and a 2-signature
    # history cap means the pool's first transaction is never reached.
    clock = Clock()
    svc = make_service(tmp_path, chain, clock, signature_page_size=2, early_max_sig_pages=1)
    svc.add_target(MINT, POOL, "pumpswap", CREATED - timedelta(days=3))
    first = asyncio.run(svc.snapshot(CID)).body
    assert first["coverage"]["overall"] == "PARTIAL"  # the capped first activity scan
    clock.advance(3600)
    r = asyncio.run(svc.snapshot(CID))  # quiet pool, unchanged holders
    b = r.body
    assert b["coverage"]["overall"] == "COMPLETE"
    for path in OVERALL_BASIS:
        section, key = path.split(".")
        assert b[section][key]["status"] == "AVAILABLE"
    assert b["activity"]["signatures_listed"] == 0
    # ...while other features are not complete.
    assert b["early_activity"]["early_wallet_count"]["status"] == "UNAVAILABLE"
    assert b["coverage"]["early_status"] == "UNAVAILABLE"
    for k in ("large_wallet_accumulation_count", "large_wallet_reduction_count",
              "large_wallet_exit_count"):  # fmt: skip
        assert b["large_holders"][k]["status"] == "PARTIAL"
    assert b["large_holders"]["large_holder_accumulation_count"]["status"] == "AVAILABLE"
    assert b["clusters"]["funding"]["count"]["status"] == "NOT_COLLECTED"
    # The basis is stated in the body, and the DB column keeps the same value.
    assert b["coverage"]["overall_basis"] == [
        "holders.holder_count", "holders.top10_pct", "activity.interacting_wallets"
    ]  # fmt: skip
    assert b["coverage"]["overall_note"] == OVERALL_NOTE
    assert "feature-level coverage" in OVERALL_NOTE
    assert b["schema_version"] == "radar.snapshot.v1"
    with sqlite3.connect(svc.settings.db_path) as conn:
        stored = conn.execute("SELECT coverage FROM radar_snapshots ORDER BY id").fetchall()
    assert stored == [("PARTIAL",), ("COMPLETE",)]
    assert "headline coverage: COMPLETE" in snapshot_text(b)
    # Run steps are execution status: a skipped step is AVAILABLE though its data isn't.
    assert b["coverage"]["run"]["early"]["status"] == "AVAILABLE"
    assert b["coverage"]["run"]["early"]["reasons"] == ["early history already collected"]
