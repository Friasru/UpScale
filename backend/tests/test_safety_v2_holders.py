"""Safety V2 Phase 2: holder evidence, owner classification, lower-bound rules, Radar
read-only wallet proof, anti-lookahead, failures and rebuilds. Offline: every request is
answered by `FakeRpc`, and Radar databases are local fixtures."""

import asyncio
import hashlib
import sqlite3
from datetime import UTC, datetime
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
from upscale.services.radar.repository import RadarRepository
from upscale.services.safety_v2 import features, registry
from upscale.services.safety_v2 import service as service_module
from upscale.services.safety_v2.features import HolderInputs, build_body, code_fingerprints
from upscale.services.safety_v2.models import (
    Evidence,
    SafetyCausalityError,
    SafetyStateError,
    ts,
)
from upscale.services.safety_v2.provider import OwnerFact
from upscale.services.safety_v2.repository import HolderRow, PoolPin, owners_hash
from upscale.services.safety_v2.rules import HolderFacts, evaluate_holders
from upscale.services.safety_v2.sources import WalletProof, WalletProofs, radar_wallet_proofs
from upscale.services.solana_chain import (
    INCINERATOR,
    RAYDIUM_AMM_V4_AUTHORITY,
    SYSTEM_PROGRAM,
)

CID = f"solana:{MINT}"
# After registry v1's knowledge epoch (2026-10-08), so registry entries apply.
T2 = datetime(2026, 10, 9, tzinfo=UTC)
SUPPLY = 1_000_000_000
PCT = SUPPLY // 100  # 1 % of supply in raw units
WHALE = addr("WhaLe")
POOL = addr("PooLAAA")
STRANGER = addr("StrangerAA")
T, N, U = "TRIGGERED", "NOT_TRIGGERED", "UNDETERMINED"
HOLDER_RULES = ("TOP1_CONCENTRATION", "TOP10_CONCENTRATION", "FEW_HOLDERS",
                "LARGE_UNKNOWN_OWNER", "LARGE_UNCLASSIFIED_PROGRAM_OWNER")  # fmt: skip


def flags(body: dict[str, Any]) -> dict[str, str]:
    return {f["id"]: f["outcome"] for f in body["flags"]}


def severity(body: dict[str, Any], rule: str) -> str:
    return next(f["severity"] for f in body["flags"] if f["id"] == rule)


def spread(chain: HolderChain, owners: int, amount: int, prefix: str = "Hdr") -> list[str]:
    out = [num_addr(prefix, i) for i in range(owners)]
    for o in out:
        chain.add(o, amount)
    return out


def setup(
    tmp_path: Path, chain: HolderChain, helius: bool = True, **kw: Any
) -> tuple[Any, Clock, FakeRpc]:
    rpc = FakeRpc({MINT: mint_value()}, holders=chain)
    kw.setdefault("clock", Clock(T2))
    svc, clock, rpc = make_service(tmp_path, rpc, helius=helius, **kw)
    svc.add_target(MINT)
    return svc, clock, rpc


def snap(svc: Any, **kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = asyncio.run(svc.snapshot(CID, holders=True, **kw)).body
    return body


def radar_db(path: Path, proofs: list[tuple[str, float]], schema: str = "3") -> Path:
    """A Radar database (real Radar schema) with signer flows for (wallet, fetched_at)."""
    repo = RadarRepository(path)
    conn = repo.db()
    with conn:
        for i, (wallet, fetched_at) in enumerate(proofs):
            conn.execute(
                "INSERT INTO radar_wallet_flows (canonical_id, signature, wallet, direction, "
                "amount_raw, counterparty, participant, signer, block_time, fetched_at, "
                "provider) VALUES (?, ?, ?, 'TOKEN_INFLOW', '1', 'UNVERIFIED', "
                "'NORMAL_WALLET', 1, NULL, ?, 'fake')",
                ("solana:other", f"sig{i}", wallet, fetched_at),
            )
        if schema != "3":
            conn.execute("UPDATE radar_meta SET value = ? WHERE key = 'schema_version'", (schema,))
    repo.close()
    return path


def fingerprint(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


# --- full holders ----------------------------------------------------------------------------


def test_complete_scan_gives_exact_values_and_not_triggered_below_thresholds(
    tmp_path: Path,
) -> None:
    chain = HolderChain()
    spread(chain, 200, PCT // 10)  # 200 owners at 0.1 % each
    svc, _, rpc = setup(tmp_path, chain)
    body = snap(svc)
    h = body["holders"]
    assert (h["status"], h["source"], h["lower_bound"]) == ("AVAILABLE", "full_scan", False)
    assert h["top1_pct"] == {"status": "AVAILABLE", "value": 0.1, "lower_bound": False,
                             "reason": None}  # fmt: skip
    assert h["top5_pct"]["value"] == 0.5 and h["top10_pct"]["value"] == 1.0
    assert h["holder_count"] == {"status": "AVAILABLE", "value": 200, "lower_bound": False,
                                 "reason": None}  # fmt: skip
    assert h["meaningful_holder_count"]["value"] == 200
    f = flags(body)
    assert {k: f[k] for k in HOLDER_RULES} == dict.fromkeys(HOLDER_RULES, N)
    assert body["assessment"]["coverage"] == "COMPLETE"
    assert body["coverage"]["components"]["holders"] == "AVAILABLE"
    assert not {o["id"] for o in body["coverage"]["out_of_scope"]} & set(HOLDER_RULES)
    assert set(HOLDER_RULES) <= set(body["coverage"]["decision_rules"])
    # Bounded: supply, largest, their accounts, one page, the owners.
    assert rpc.calls == ["getAccountInfo", "getTokenSupply", "getTokenLargestAccounts",
                         "getMultipleAccounts", "getTokenAccounts", "getMultipleAccounts"]  # fmt: skip


def test_complete_scan_triggers_above_thresholds(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 30 * PCT)
    spread(chain, 19, PCT)  # 20 owners: top1 30 %, top10 39 %
    svc, _, _ = setup(tmp_path, chain)
    body = snap(svc)
    f = flags(body)
    assert f["TOP1_CONCENTRATION"] == T and severity(body, "TOP1_CONCENTRATION") == "high"
    assert f["TOP10_CONCENTRATION"] == T and severity(body, "TOP10_CONCENTRATION") == "medium"
    assert body["holders"]["top10_pct"]["value"] == 39.0
    assert f["FEW_HOLDERS"] == T and severity(body, "FEW_HOLDERS") == "high"  # 20 < 30
    assert f["LARGE_UNKNOWN_OWNER"] == T
    assert body["holders"]["unknown_large_owners"][0]["owner"] == WHALE
    assert body["assessment"]["band"] == "ELEVATED_EVIDENCE"


def test_owner_aggregation_and_zero_balances(tmp_path: Path) -> None:
    chain = HolderChain()
    for i in range(3):
        chain.add(WHALE, 4 * PCT, num_addr("Wacc", i))  # 3 accounts, 12 % together
    zero = chain.add(STRANGER, 0)
    spread(chain, 150, PCT // 10)
    chain.largest = [num_addr("Wacc", i) for i in range(3)] + [zero]
    svc, _, _ = setup(tmp_path, chain)
    h = snap(svc)["holders"]
    top = h["top_owners"][0]
    assert (top["owner"], top["pct"], top["token_accounts"]) == (WHALE, 12.0, 3)
    assert h["top1_pct"]["value"] == 12.0
    assert h["holder_count"]["value"] == 151  # the zero-balance owner isn't a holder
    assert STRANGER not in {o["owner"] for o in h["top_owners"]}


# --- partial holders ---------------------------------------------------------------------------


def _big_chain(whale_pct: float) -> HolderChain:
    chain = HolderChain()
    spread(chain, 1500, 10_000)  # 1,500 owners: more than one 1,000-row page
    chain.add(WHALE, int(whale_pct * PCT))
    return chain


def test_page_cap_partial_lower_bound_above_threshold_triggers(tmp_path: Path) -> None:
    svc, _, rpc = setup(tmp_path, _big_chain(40), holder_max_pages=1)
    body = snap(svc)
    h = body["holders"]
    assert (h["status"], h["source"], h["lower_bound"]) == ("PARTIAL", "partial_scan", True)
    assert h["pages_read"] == h["max_pages"] == 1
    assert rpc.calls.count("getTokenAccounts") == 1
    top10 = h["top10_pct"]
    assert top10["status"] == "PARTIAL" and top10["lower_bound"] and top10["value"] >= 40
    assert "page cap" in top10["reason"]
    f = flags(body)
    assert f["TOP10_CONCENTRATION"] == T and f["TOP1_CONCENTRATION"] == T
    # A partial count of >= 100 owners still can't prove FEW_HOLDERS false.
    assert h["holder_count"]["status"] == "PARTIAL" and h["holder_count"]["value"] >= 1000
    assert f["FEW_HOLDERS"] == U
    assert body["assessment"]["coverage"] == "PARTIAL"


def test_partial_lower_bound_below_threshold_is_undetermined(tmp_path: Path) -> None:
    svc, _, _ = setup(tmp_path, _big_chain(10), holder_max_pages=1)
    body = snap(svc)
    assert body["holders"]["top10_pct"]["value"] < 35
    f = flags(body)
    assert f["TOP10_CONCENTRATION"] == U  # never NOT_TRIGGERED from a lower bound
    assert f["TOP1_CONCENTRATION"] == T  # 10 % lower bound already crosses 10 %
    assert f["LARGE_UNKNOWN_OWNER"] == T
    assert f["LARGE_UNCLASSIFIED_PROGRAM_OWNER"] == U


def test_full_scan_over_two_pages_is_complete(tmp_path: Path) -> None:
    svc, _, rpc = setup(tmp_path, _big_chain(1), holder_max_pages=2)
    h = snap(svc)["holders"]
    assert (h["status"], h["source"], h["pages_read"]) == ("AVAILABLE", "full_scan", 2)
    assert h["holder_count"]["value"] == 1501
    assert rpc.calls.count("getTokenAccounts") == 2


def test_largest_accounts_only_is_partial_and_has_no_holder_count(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 50 * PCT)
    spread(chain, 40, PCT // 10)
    svc, _, rpc = setup(tmp_path, chain, helius=False)
    body = snap(svc)
    h = body["holders"]
    assert (h["status"], h["source"], h["pages_read"]) == ("PARTIAL", "largest_accounts", 0)
    assert "getTokenAccounts" not in rpc.calls
    assert h["top1_pct"]["lower_bound"] and h["top1_pct"]["value"] == 50.0
    assert h["holder_count"]["status"] == "UNAVAILABLE"
    f = flags(body)
    assert f["TOP1_CONCENTRATION"] == T and f["FEW_HOLDERS"] == U


def test_a_scan_missing_a_largest_account_is_not_complete(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 50, PCT // 10)
    chain.add(WHALE, 5 * PCT, addr("HiddenAcct"))
    chain.extra_rows = []
    rpc = FakeRpc({MINT: mint_value()}, holders=chain)
    original = chain.result

    def hide(method: str, params: Any) -> Any:
        out = original(method, params)
        if method == "getTokenAccounts":
            out["token_accounts"] = [r for r in out["token_accounts"]
                                     if r.get("address") != addr("HiddenAcct")]  # fmt: skip
        return out

    chain.result = hide  # type: ignore[method-assign]
    svc, _, _ = make_service(tmp_path, rpc, helius=True)
    svc.add_target(MINT)
    h = snap(svc)["holders"]
    assert h["source"] == "partial_scan" and h["status"] == "PARTIAL"
    assert any("missed" in r for r in h["reasons"])


def test_malformed_scan_rows_make_the_scan_partial(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 50, PCT // 10)
    chain.extra_rows = [{"address": "not-base58", "owner": STRANGER, "amount": 5, "mint": MINT}]
    svc, _, _ = setup(tmp_path, chain)
    h = snap(svc)["holders"]
    assert h["source"] == "partial_scan"
    assert any("malformed" in r for r in h["reasons"])


# --- lower-bound rule mathematics ------------------------------------------------------------


def _facts(status: str, top10_pct: float, count: Evidence | None = None) -> HolderFacts:
    return HolderFacts(status=status,  # type: ignore[arg-type]
                       reason=None if status == "AVAILABLE" else "partial",
                       supply=SUPPLY, top1_amount=int(top10_pct * PCT / 10),
                       top10_amount=int(top10_pct * PCT), meaningful_count=count)  # fmt: skip


def _outcomes(facts: HolderFacts) -> dict[str, str]:
    return {r.id: r.outcome for r in evaluate_holders(facts)}


@pytest.mark.parametrize(("status", "pct", "expected"), [
    ("AVAILABLE", 22, N),  # complete, below 35
    ("AVAILABLE", 35, T),  # exactly at the threshold
    ("PARTIAL", 40, T),  # a lower bound past the threshold proves it
    ("PARTIAL", 42, T),
    ("PARTIAL", 22, U),  # a lower bound below can't prove anything
    ("PARTIAL", 20, U),
    ("PROVIDER_UNAVAILABLE", 0, U),
])  # fmt: skip
def test_top10_lower_bound_mathematics(status: str, pct: float, expected: str) -> None:
    assert _outcomes(_facts(status, pct))["TOP10_CONCENTRATION"] == expected


@pytest.mark.parametrize(("count", "expected"), [
    (Evidence(status="PARTIAL", value=150, lower_bound=True, reason="page cap"), U),
    (Evidence(status="PARTIAL", value=10, lower_bound=True, reason="page cap"), U),
    (Evidence(status="AVAILABLE", value=150), N),
    (Evidence(status="AVAILABLE", value=99), T),
    (Evidence(status="UNAVAILABLE", reason="largest accounts only"), U),
])  # fmt: skip
def test_few_holders_is_decided_only_by_a_complete_count(count: Evidence, expected: str) -> None:
    assert _outcomes(_facts("AVAILABLE", 1, count))["FEW_HOLDERS"] == expected


def test_threshold_comparison_is_exact_integer_arithmetic() -> None:
    just_below = HolderFacts(status="AVAILABLE", reason=None, supply=SUPPLY,
                             top1_amount=10 * PCT - 1, top10_amount=0)  # fmt: skip
    at = HolderFacts(status="AVAILABLE", reason=None, supply=SUPPLY, top1_amount=10 * PCT,
                     top10_amount=0)  # fmt: skip
    assert _outcomes(just_below)["TOP1_CONCENTRATION"] == N
    assert _outcomes(at)["TOP1_CONCENTRATION"] == T


def test_dust_owners_count_as_holders_but_not_as_meaningful_holders(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 40, PCT // 10)  # 40 meaningful owners (0.1 % each)
    spread(chain, 200, 1, prefix="Dst")  # 200 dust owners (1e-9 of supply each)
    svc, _, _ = setup(tmp_path, chain)
    body = snap(svc)
    h = body["holders"]
    assert h["holder_count"]["value"] == 240
    assert h["meaningful_holder_count"]["value"] == 40
    assert h["meaningful_holder_min_fraction"] == "1/1000000"
    few = next(f for f in body["flags"] if f["id"] == "FEW_HOLDERS")
    assert few["evidence"] == ["holders.meaningful_holder_count"]
    assert (few["outcome"], few["severity"]) == (T, "medium")  # 40 < 100, despite 240 holders
    assert "1/1000000" in few["reason"]
    sid = svc.repo.latest_snapshot_id(CID)
    assert sid is not None and svc.rebuild(sid).status == "REPRODUCED"


def test_meaningful_threshold_is_exact_at_the_boundary(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 120, SUPPLY // 1_000_000)  # exactly 1e-6 each: meaningful
    spread(chain, 5, SUPPLY // 1_000_000 - 1, prefix="Dst")  # just below
    svc, _, _ = setup(tmp_path, chain)
    body = snap(svc)
    assert body["holders"]["holder_count"]["value"] == 125
    assert body["holders"]["meaningful_holder_count"]["value"] == 120
    assert flags(body)["FEW_HOLDERS"] == N  # complete count >= 100


def test_partial_meaningful_count_cannot_prove_few_holders_false(tmp_path: Path) -> None:
    svc, _, _ = setup(tmp_path, _big_chain(1), holder_max_pages=1)
    body = snap(svc)
    m = body["holders"]["meaningful_holder_count"]
    assert m["status"] == "PARTIAL" and m["lower_bound"] and m["value"] >= 100
    assert flags(body)["FEW_HOLDERS"] == U


# --- exclusions -------------------------------------------------------------------------------


def test_burn_address_is_excluded(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(INCINERATOR, 50 * PCT)
    spread(chain, 150, PCT // 10)
    svc, _, _ = setup(tmp_path, chain)
    h = snap(svc)["holders"]
    assert [(e["owner"], e["classification"], e["pct"]) for e in h["excluded"]] == [
        (INCINERATOR, "BURN", 50.0)
    ]
    assert h["top1_pct"]["value"] == 0.1  # supply stays the denominator
    assert h["holder_count"]["value"] == 150


def test_only_an_exactly_pinned_pool_is_excluded(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(POOL, 30 * PCT)
    chain.add(STRANGER, 20 * PCT)  # looks just like a pool, but nothing proves it
    spread(chain, 150, PCT // 10)
    svc, clock, _ = setup(tmp_path, chain)
    assert svc.pin_pool(MINT, POOL) and not svc.pin_pool(MINT, POOL)
    clock.advance(1)
    body = snap(svc)
    h = body["holders"]
    assert [(e["owner"], e["classification"]) for e in h["excluded"]] == [(POOL, "POOL_OR_VAULT")]
    assert h["top_owners"][0]["owner"] == STRANGER
    assert h["top_owners"][0]["classification"] == "UNKNOWN"
    assert h["top1_pct"]["value"] == 20.0
    assert any(i["component"] == "pool_pin" for i in body["provenance"]["inputs"])


def test_raydium_authority_needs_a_pinned_raydium_pool(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(RAYDIUM_AMM_V4_AUTHORITY, 30 * PCT)
    spread(chain, 150, PCT // 10)
    svc, clock, _ = setup(tmp_path, chain)
    h = snap(svc)["holders"]
    assert h["excluded"] == []
    assert h["top_owners"][0]["classification"] == "UNKNOWN"
    assert "corroborates" in h["top_owners"][0]["reason"]
    svc.pin_pool(MINT, POOL, "raydium-amm-v4")
    clock.advance(1)
    h = snap(svc, fetch=False)["holders"]
    assert [e["classification"] for e in h["excluded"]] == ["POOL_OR_VAULT"]


def test_program_owned_owner_is_never_a_wallet(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 8 * PCT)
    chain.owner_programs[WHALE] = OTHER_PROGRAM
    spread(chain, 150, PCT // 10)
    svc, _, _ = setup(tmp_path, chain)
    body = snap(svc)
    h = body["holders"]
    assert h["top_owners"][0]["classification"] == "PROGRAM_OWNED"
    assert h["program_large_owners"][0]["role_identified"] is False
    assert h["top1_pct"]["value"] == 8.0  # counted in general concentration, disclosed
    f = flags(body)
    assert f["LARGE_UNCLASSIFIED_PROGRAM_OWNER"] == T
    assert f["LARGE_UNKNOWN_OWNER"] == N  # separate conditions, never conflated
    assert h["unknown_large_owners"] == []


def test_large_unknown_owner_is_not_a_program_flag(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 8 * PCT)
    chain.owner_programs[WHALE] = SYSTEM_PROGRAM  # a system account: not proof of a wallet
    spread(chain, 150, PCT // 10)
    svc, _, _ = setup(tmp_path, chain)
    body = snap(svc)
    f = flags(body)
    assert (f["LARGE_UNKNOWN_OWNER"], f["LARGE_UNCLASSIFIED_PROGRAM_OWNER"]) == (T, N)
    assert body["holders"]["unknown_large_owners"][0]["classification"] == "UNKNOWN"


# --- owner classification / Radar ------------------------------------------------------------


def test_radar_signer_proof_makes_a_normal_wallet(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 8 * PCT)
    spread(chain, 150, PCT // 10)
    radar = radar_db(tmp_path / "radar.sqlite3", [(WHALE, ts(T2) - 60)])
    before = fingerprint(radar)
    svc, _, _ = setup(tmp_path, chain, radar_db_path=str(radar))
    body = snap(svc)
    h = body["holders"]
    top = h["top_owners"][0]
    assert top["classification"] == "NORMAL_WALLET" and top["wallet_proof"]["signature"] == "sig0"
    assert flags(body)["LARGE_UNKNOWN_OWNER"] == N
    # Other owners are UNKNOWN / UNRESOLVED and could still be top-10 wallets: lower bound.
    wallet = h["top10_wallet_pct"]
    assert wallet["status"] == "PARTIAL" and wallet["lower_bound"] and wallet["value"] == 8.0
    wp = next(i for i in body["provenance"]["inputs"] if i["component"] == "wallet_proofs")
    assert (wp["status"], wp["proofs"]) == ("AVAILABLE", 1)
    assert fingerprint(radar) == before  # byte-identical, untouched mtime


def test_radar_proof_learned_after_as_of_is_ignored(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 8 * PCT)
    spread(chain, 150, PCT // 10)
    radar = radar_db(tmp_path / "radar.sqlite3", [(WHALE, ts(T2) + 1)])
    svc, clock, _ = setup(tmp_path, chain, radar_db_path=str(radar))
    body = snap(svc)
    assert body["holders"]["top_owners"][0]["classification"] == "UNKNOWN"
    assert flags(body)["LARGE_UNKNOWN_OWNER"] == T
    clock.advance(2)  # later, the proof is known
    later = snap(svc, fetch=False)
    assert later["holders"]["top_owners"][0]["classification"] == "NORMAL_WALLET"


@pytest.mark.parametrize("kind", ["schema_2", "missing", "not_radar", "unconfigured"])
def test_unusable_radar_gives_no_wallet_proof_and_no_failure(tmp_path: Path, kind: str) -> None:
    chain = HolderChain()
    chain.add(WHALE, 8 * PCT)
    spread(chain, 150, PCT // 10)
    path: Path | None = tmp_path / "radar.sqlite3"
    assert path is not None
    if kind == "schema_2":
        radar_db(path, [(WHALE, 0.0)], schema="2")
    elif kind == "not_radar":
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE scout_tokens (x)")
        conn.close()
    elif kind == "unconfigured":
        path = None
    before = fingerprint(path) if path is not None and path.exists() else None
    svc, _, _ = setup(tmp_path, chain, radar_db_path=str(path) if path else None)
    body = snap(svc)
    status = {"schema_2": "INCOMPATIBLE", "missing": "UNAVAILABLE", "not_radar": "INCOMPATIBLE",
              "unconfigured": "NOT_CONFIGURED"}[kind]  # fmt: skip
    wp = next(i for i in body["provenance"]["inputs"] if i["component"] == "wallet_proofs")
    assert wp["status"] == status
    assert body["holders"]["top_owners"][0]["classification"] == "UNKNOWN"
    assert body["holders"]["top10_wallet_pct"]["status"] == "UNKNOWN"
    if kind == "missing":
        assert path is not None and not path.exists()  # never created
    if before is not None and path is not None:
        assert fingerprint(path) == before


def test_conflicting_program_owner_and_signer_proof_is_unresolved(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 8 * PCT)
    chain.owner_programs[WHALE] = OTHER_PROGRAM
    spread(chain, 150, PCT // 10)
    radar = radar_db(tmp_path / "radar.sqlite3", [(WHALE, 0.0)])
    svc, _, _ = setup(tmp_path, chain, radar_db_path=str(radar))
    body = snap(svc)
    assert body["holders"]["top_owners"][0]["classification"] == "UNRESOLVED"
    assert flags(body)["LARGE_UNKNOWN_OWNER"] == T


def test_owners_beyond_the_lookup_cap_are_unresolved_never_wallets(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 150, PCT // 10)
    svc, _, _ = setup(tmp_path, chain, holder_owner_lookups=0)
    h = snap(svc)["holders"]
    assert h["owner_classes"]["UNRESOLVED"] == 150 and h["owner_classes"]["NORMAL_WALLET"] == 0


def test_radar_reader_filters_by_knowledge_time_directly(tmp_path: Path) -> None:
    radar = radar_db(tmp_path / "r.sqlite3", [(WHALE, 10.0), (WHALE, 5.0), (STRANGER, 20.0)])
    got = radar_wallet_proofs(str(radar), [WHALE, STRANGER, POOL], 10.0)
    assert got.status == "AVAILABLE" and set(got.proofs) == {WHALE}
    assert got.proofs[WHALE].fetched_at == 5.0  # the earliest proof
    assert radar_wallet_proofs(str(radar), [], 10.0).status == "NOT_CONSULTED"


# --- wallet-only concentration -----------------------------------------------------------------


def test_wallet_concentration_counts_only_proven_wallets(tmp_path: Path) -> None:
    chain = HolderChain()
    wallets = [num_addr("Wqt", i) for i in range(10)]
    for w in wallets:
        chain.add(w, 3 * PCT)  # ten proven wallets, 30 %
    chain.add(STRANGER, 2 * PCT)  # unknown, smaller than every wallet
    radar = radar_db(tmp_path / "radar.sqlite3", [(w, 0.0) for w in wallets])
    svc, _, _ = setup(tmp_path, chain, radar_db_path=str(radar))
    h = snap(svc)["holders"]
    assert h["top10_wallet_pct"] == {"status": "AVAILABLE", "value": 30.0,
                                     "lower_bound": False, "reason": None}  # fmt: skip
    assert h["top10_pct"]["value"] == 30.0


def test_a_large_unknown_owner_makes_wallet_concentration_a_lower_bound(tmp_path: Path) -> None:
    chain = HolderChain()
    wallets = [num_addr("Wqt", i) for i in range(10)]
    for w in wallets:
        chain.add(w, 3 * PCT)
    chain.add(STRANGER, 9 * PCT)  # unknown, larger than the 10th wallet
    radar = radar_db(tmp_path / "radar.sqlite3", [(w, 0.0) for w in wallets])
    svc, _, _ = setup(tmp_path, chain, radar_db_path=str(radar))
    h = snap(svc)["holders"]
    wallet = h["top10_wallet_pct"]
    assert wallet["status"] == "PARTIAL" and wallet["lower_bound"]
    assert wallet["value"] == 30.0  # the unknown owner is never in the wallet metric
    assert "could still be top-10 wallets" in wallet["reason"]
    assert h["top10_pct"]["value"] == 36.0  # but it is in general concentration


# --- causality ---------------------------------------------------------------------------------


def _row(fetched_at: float, balances: list[OwnerFact], **kw: Any) -> HolderRow:
    base: dict[str, Any] = dict(
        id=1, canonical_id=CID, collection_id=1, fetched_at=fetched_at, provider="fake",
        origin="safety_v2.collect_holders", outcome="COLLECTED", reason=None,
        source="full_scan", supply_raw=str(SUPPLY), decimals=6, pages_read=1, max_pages=2,
        token_accounts_seen=len(balances), largest_accounts_seen=len(balances), skipped_rows=0,
        holder_count=len(balances), holder_count_complete=True, reliable=True, reasons=[],
        owners_hash=owners_hash(balances),
    )  # fmt: skip
    base.update(kw)
    return HolderRow(**base)


def _facts_list(*pairs: tuple[str, int]) -> list[OwnerFact]:
    return [OwnerFact(o, o, (num_addr("Tacc", i),), a, "MISSING")
            for i, (o, a) in enumerate(pairs)]  # fmt: skip


def test_holder_observation_at_exactly_as_of_is_accepted_and_after_is_rejected() -> None:
    balances = _facts_list((WHALE, 5 * PCT))
    ok = build_body(CID, MINT, 100.0, None, code_fingerprints(),
                    HolderInputs(_row(100.0, balances), tuple(balances)))  # fmt: skip
    assert ok["holders"]["status"] == "AVAILABLE"
    with pytest.raises(SafetyCausalityError, match="holder observation"):
        build_body(CID, MINT, 99.9, None, code_fingerprints(),
                   HolderInputs(_row(100.0, balances), tuple(balances)))  # fmt: skip


def test_future_wallet_proof_or_pool_pin_passed_to_build_is_rejected() -> None:
    balances = _facts_list((WHALE, 5 * PCT))
    row = _row(10.0, balances)
    proof = WalletProofs("AVAILABLE", None, {WHALE: WalletProof(WHALE, 11.0, "s", "c")})
    with pytest.raises(SafetyCausalityError, match="wallet proof"):
        build_body(CID, MINT, 10.0, None, code_fingerprints(),
                   HolderInputs(row, tuple(balances), proofs=proof))  # fmt: skip
    pin = PoolPin(1, CID, POOL, None, 11.0, "manual")
    with pytest.raises(SafetyCausalityError, match="pool pin"):
        build_body(CID, MINT, 10.0, None, code_fingerprints(),
                   HolderInputs(row, tuple(balances), pools=(pin,)))  # fmt: skip


def test_tampered_balances_are_refused() -> None:
    balances = _facts_list((WHALE, 5 * PCT))
    row = _row(10.0, balances)
    tampered = (OwnerFact(WHALE, WHALE, balances[0].accounts, 1, "MISSING"),)
    with pytest.raises(SafetyStateError, match="hash"):
        build_body(CID, MINT, 10.0, None, code_fingerprints(), HolderInputs(row, tampered))


def test_registry_entry_needs_knowledge_time_not_just_validity() -> None:
    balances = _facts_list((INCINERATOR, 50 * PCT), (WHALE, 5 * PCT))
    inputs = HolderInputs(_row(1.0, balances), tuple(balances))
    epoch = registry.KNOWLEDGE_EPOCH_V1
    burn = registry.lookup(INCINERATOR, 4e9)
    assert burn is not None and burn.valid_since < burn.known_since == epoch
    # Objectively valid (after 2020) but before the ruleset knew it: not applied.
    before = build_body(CID, MINT, epoch - 1, None, code_fingerprints(), inputs)["holders"]
    assert before["excluded"] == [] and before["top_owners"][0]["classification"] == "UNKNOWN"
    at = build_body(CID, MINT, epoch, None, code_fingerprints(), inputs)["holders"]
    assert [e["classification"] for e in at["excluded"]] == ["BURN"]


def test_registry_lookup_requires_both_times() -> None:
    entry = registry.RegistryEntry(STRANGER, "BURN", "test", valid_since=100.0,
                                   known_since=200.0)  # fmt: skip
    table = (entry,)
    assert registry.lookup(STRANGER, 150.0, table) is None  # valid, not yet known
    assert registry.lookup(STRANGER, 200.0, table) is entry  # both <= as_of
    late_valid = registry.RegistryEntry(STRANGER, "BURN", "t", valid_since=300.0,
                                        known_since=200.0)  # fmt: skip
    assert registry.lookup(STRANGER, 250.0, (late_valid,)) is None  # known, not yet valid


def test_registry_entries_are_never_backdated_before_the_knowledge_epoch() -> None:
    assert {e.known_since for e in registry.ENTRIES} == {registry.KNOWLEDGE_EPOCH_V1}


def test_registry_knowledge_after_as_of_is_ignored() -> None:
    balances = _facts_list((INCINERATOR, 50 * PCT), (WHALE, 5 * PCT))
    inputs = HolderInputs(_row(1.0, balances), tuple(balances))
    old = build_body(CID, MINT, 1.0, None, code_fingerprints(), inputs)["holders"]  # 1970
    assert old["excluded"] == []
    assert old["top_owners"][0]["classification"] == "UNKNOWN"
    now = build_body(CID, MINT, 4e9, None, code_fingerprints(), inputs)["holders"]
    assert [e["classification"] for e in now["excluded"]] == ["BURN"]


def test_pins_and_holder_observations_after_as_of_are_not_selected(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(POOL, 30 * PCT)
    spread(chain, 150, PCT // 10)
    svc, clock, _ = setup(tmp_path, chain)
    first = asyncio.run(svc.snapshot(CID, holders=True))
    clock.advance(10)
    svc.pin_pool(MINT, POOL)
    clock.advance(10)
    asyncio.run(svc.collect(CID, holders=True))
    old = svc.build(CID, ts(first.as_of))
    assert old["holders"]["excluded"] == [] and old == first.body
    assert svc.rebuild(first.snapshot_id or 0).status == "REPRODUCED"
    assert svc.build(CID, ts(clock.now()))["holders"]["excluded"][0]["owner"] == POOL


# --- failures / durability ---------------------------------------------------------------------


def test_holder_provider_failure_keeps_the_mint_component(tmp_path: Path) -> None:
    chain = HolderChain(fail={"getTokenLargestAccounts": ("http", 503)})
    spread(chain, 10, PCT)
    svc, _, _ = setup(tmp_path, chain, max_retries=1)
    res = asyncio.run(svc.snapshot(CID, holders=True))
    body = res.body
    assert res.collected is not None
    assert (res.collected.outcome, res.collected.holder_outcome) == ("MINT", "PROVIDER_FAILED")
    assert body["coverage"]["components"] | {} == {
        "mint_account": "AVAILABLE", "holders": "PROVIDER_UNAVAILABLE",
        "market": "UNAVAILABLE", "creator": "NOT_CONFIGURED",
    }  # fmt: skip
    assert body["authority"]["mint_authority"]["status"] == "AVAILABLE"
    f = flags(body)
    assert all(f[r] == U for r in HOLDER_RULES)
    assert body["assessment"]["coverage"] == "PARTIAL"
    status, *_ = svc.repo.collection(res.collected.collection_id)
    assert status == "DONE"
    assert svc.repo.counts()["safety_holder_balances"] == 0  # partial data is never kept


def test_budget_too_small_for_holders_is_not_collected_without_requests(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 10, PCT)
    svc, _, rpc = setup(tmp_path, chain, daily_request_budget=5)  # mint 1 + holders up to 6
    body = snap(svc)
    assert rpc.calls == ["getAccountInfo"]
    assert body["holders"]["status"] == "NOT_COLLECTED"
    assert "budget" in body["holders"]["reason"]
    assert all(flags(body)[r] == U for r in HOLDER_RULES)


def test_no_provider_records_a_deliberate_not_collected(tmp_path: Path) -> None:
    svc, _, rpc = make_service(tmp_path, FakeRpc(), provider=False)
    svc.add_target(MINT)
    got = asyncio.run(svc.collect(CID, holders=True))
    assert (got.outcome, got.holder_outcome) == ("NOT_COLLECTED", "NOT_COLLECTED")
    assert rpc.calls == []
    assert svc.repo.collection(got.collection_id)[0] == "DONE"


def test_mint_only_collection_never_shadows_holder_evidence(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 150, PCT // 10)
    svc, clock, _ = setup(tmp_path, chain)
    asyncio.run(svc.collect(CID, holders=True))
    clock.advance(5)
    body = asyncio.run(svc.snapshot(CID)).body  # mint-only refresh
    assert body["holders"]["status"] == "AVAILABLE"


def test_unexpected_holder_error_aborts_the_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = HolderChain()
    svc, _, _ = setup(tmp_path, chain)

    async def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(service_module, "collect_holders", boom)
    with pytest.raises(RuntimeError, match="bug"):
        asyncio.run(svc.collect(CID, holders=True))
    statuses = [r[0] for r in svc.repo.db().execute("SELECT status FROM safety_collections")]
    assert statuses == ["ABORTED"]


def test_holder_tables_are_append_only(tmp_path: Path) -> None:
    chain = HolderChain()
    spread(chain, 5, PCT)
    svc, _, _ = setup(tmp_path, chain)
    got = asyncio.run(svc.collect(CID, holders=True))
    svc.pin_pool(MINT, POOL)
    conn = svc.repo.db()
    for sql in ("UPDATE safety_holder_observations SET provider = 'x'",
                "DELETE FROM safety_holder_observations",
                "UPDATE safety_holder_balances SET amount_raw = '1'",
                "DELETE FROM safety_holder_balances",
                "UPDATE safety_target_pools SET dex = 'x'",
                "DELETE FROM safety_target_pools"):  # fmt: skip
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(sql)
    with pytest.raises(sqlite3.IntegrityError, match="RUNNING"):
        conn.execute(
            "INSERT INTO safety_holder_balances (observation_id, rank, owner_key, owner, "
            "accounts_json, amount_raw, lookup) VALUES (?, 99, ?, ?, '[]', '5', 'MISSING')",
            (got.holder_observation_id, STRANGER, STRANGER),
        )
    assert (
        svc.repo.db()
        .execute("SELECT COUNT(*) FROM safety_collections WHERE status = 'RUNNING'")
        .fetchone()[0]
        == 0
    )


def test_bounded_requests_on_a_large_token(tmp_path: Path) -> None:
    svc, _, rpc = setup(tmp_path, _big_chain(1), holder_max_pages=1)
    got = asyncio.run(svc.collect(CID, holders=True))
    assert got.requests == len(rpc.calls) == 1 + 4 + 1


# --- hashing / rebuild -------------------------------------------------------------------------


def test_same_evidence_same_hash_and_rebuild_reproduces(tmp_path: Path) -> None:
    hashes = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        chain = HolderChain()
        chain.add(WHALE, 8 * PCT)
        spread(chain, 150, PCT // 10)
        radar = radar_db(tmp_path / name / "radar.sqlite3", [(WHALE, 0.0)])
        svc, _, _ = setup(tmp_path / name, chain, radar_db_path=str(radar))
        res = asyncio.run(svc.snapshot(CID, holders=True))
        assert svc.rebuild(res.snapshot_id or 0).status == "REPRODUCED"
        hashes.append(res.body_hash)
    assert hashes[0] == hashes[1]


def test_later_radar_proof_does_not_leak_into_an_older_rebuild(tmp_path: Path) -> None:
    chain = HolderChain()
    chain.add(WHALE, 8 * PCT)
    spread(chain, 150, PCT // 10)
    radar = tmp_path / "radar.sqlite3"
    radar_db(radar, [])
    svc, clock, _ = setup(tmp_path, chain, radar_db_path=str(radar))
    res = asyncio.run(svc.snapshot(CID, holders=True))
    clock.advance(60)
    conn = sqlite3.connect(radar)
    with conn:
        conn.execute(
            "INSERT INTO radar_wallet_flows (canonical_id, signature, wallet, direction, "
            "amount_raw, counterparty, participant, signer, block_time, fetched_at, provider) "
            "VALUES ('solana:x', 'late', ?, 'TOKEN_INFLOW', '1', 'UNVERIFIED', 'NORMAL_WALLET', "
            "1, NULL, ?, 'fake')",
            (WHALE, ts(clock.now())),
        )
    conn.close()
    assert svc.rebuild(res.snapshot_id or 0).status == "REPRODUCED"


def test_changed_registry_refuses_exact_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = HolderChain()
    spread(chain, 150, PCT // 10)
    svc, _, _ = setup(tmp_path, chain)
    res = asyncio.run(svc.snapshot(CID, holders=True))
    copy = tmp_path / "src"
    copy.mkdir()
    for name in features.FINGERPRINTED:
        (copy / name).write_bytes((features._PACKAGE / name).read_bytes())
    (copy / "registry.py").write_bytes((copy / "registry.py").read_bytes() + b"\n# new entry\n")
    monkeypatch.setattr(features, "_PACKAGE", copy)
    features.code_fingerprints.cache_clear()
    svc._fingerprints = features.code_fingerprints
    try:
        rb = svc.rebuild(res.snapshot_id or 0)
    finally:
        monkeypatch.undo()
        features.code_fingerprints.cache_clear()
    assert rb.status == "FINGERPRINT_MISMATCH" and set(rb.mismatched) == {"safety_v2_source"}


def test_registry_and_radar_reader_are_fingerprinted() -> None:
    assert {"registry.py", "sources.py"} <= set(features.FINGERPRINTED)


def test_no_holder_observation_keeps_holder_rules_out_of_scope(tmp_path: Path) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
    svc.add_target(MINT)
    body = asyncio.run(svc.snapshot(CID)).body
    assert body["holders"]["status"] == "UNAVAILABLE"
    assert set(HOLDER_RULES) <= {o["id"] for o in body["coverage"]["out_of_scope"]}
    assert not set(HOLDER_RULES) & set(flags(body))
    assert body["assessment"]["coverage"] == "COMPLETE"


def test_unknown_dex_and_invalid_pool_are_refused(tmp_path: Path) -> None:
    from upscale.services.market_data import InvalidRequestError

    svc, _, _ = make_service(tmp_path, FakeRpc())
    svc.add_target(MINT)
    with pytest.raises(InvalidRequestError, match="unknown dex"):
        svc.pin_pool(MINT, POOL, "orca")
    with pytest.raises(InvalidRequestError, match="valid Solana"):
        svc.pin_pool(MINT, "0xabc")
    with pytest.raises(InvalidRequestError, match="mint itself"):
        svc.pin_pool(MINT, MINT)


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_collects_holders_by_default_and_pins_pools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from upscale.services.safety_v2.cli import main

    for key in ("UPSCALE_HELIUS_API_KEY", "UPSCALE_SOLANA_RPC_URL", "UPSCALE_SAFETY_V2_RADAR_DB"):
        monkeypatch.delenv(key, raising=False)
    db = ["--db", str(tmp_path / "s.sqlite3")]
    assert main([*db, "targets", "add", "--token", MINT]) == 0
    assert main([*db, "targets", "pin-pool", "--token", MINT, "--pool", POOL]) == 0
    assert "pinned" in capsys.readouterr().out
    assert main([*db, "targets", "pin-pool", "--token", MINT, "--pool", POOL, "--dex", "x"]) == 2
    assert main([*db, "collect", "--token", MINT, "--json"]) == 0
    capsys.readouterr()
    assert main([*db, "snapshot", "--token", MINT, "--no-fetch", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["body"]["holders"]["status"] == "NOT_COLLECTED"
    assert main([*db, "rebuild", "--id", str(out["snapshot_id"])]) == 0
    assert "REPRODUCED" in capsys.readouterr().out
    assert main([*db, "collect", "--token", MINT, "--no-holders", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["holder_outcome"] is None
