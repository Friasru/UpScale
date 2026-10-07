"""Radar V1: identity (Solana case sensitivity), settings defaults, and transaction
normalization into neutral TOKEN_INFLOW / TOKEN_OUTFLOW flows. Offline."""

from datetime import timedelta

import pytest

from tests.radar_fakes import MINT, POOL, T0, addr, tx
from upscale.services.market_data import InvalidRequestError
from upscale.services.radar.config import load_settings
from upscale.services.radar.models import Metric, parse_canonical, solana_identity
from upscale.services.radar.parsing import first_funder, parse_signatures, parse_transaction
from upscale.services.solana_chain import RAYDIUM_AMM_V4_AUTHORITY

A, B, C = addr("WaAAA"), addr("WaBBB"), addr("WaCCC")


# --- identity -------------------------------------------------------------------------------


def test_solana_identity_keeps_case() -> None:
    upper = "So11111111111111111111111111111111111111112"
    lower = "so11111111111111111111111111111111111111112"
    assert solana_identity(upper) == (f"solana:{upper}", upper)
    assert solana_identity(lower)[0] == f"solana:{lower}"
    assert solana_identity(upper)[0] != solana_identity(lower)[0]
    assert solana_identity(f"  {upper} ")[1] == upper  # trimmed, never lowercased


def test_parse_canonical_rejects_evm_and_garbage() -> None:
    assert parse_canonical(f"solana:{MINT}") == ("solana", MINT)
    assert parse_canonical(MINT) == ("solana", MINT)
    with pytest.raises(InvalidRequestError):
        parse_canonical("ethereum:0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
    with pytest.raises(InvalidRequestError):
        solana_identity("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")  # no EVM lowercasing path
    with pytest.raises(InvalidRequestError):
        solana_identity("not-an-address")


# --- settings -------------------------------------------------------------------------------


def test_conservative_defaults_and_overrides() -> None:
    s = load_settings({})
    assert (s.enabled, s.daily_request_budget, s.max_rps, s.concurrency) == (False, 2000, 1.0, 1)
    assert not (s.deep_backfill or s.wallet_age or s.first_funder or s.funding_clusters_enabled)
    assert (s.tx_retention_days, s.holder_balance_retention_days) == (7, 7)
    assert (s.snapshot_retention_days, s.request_retention_days) == (30, 90)
    o = load_settings({
        "UPSCALE_RADAR": "1", "UPSCALE_RADAR_DAILY_REQUEST_BUDGET": "50",
        "UPSCALE_RADAR_MAX_RPS": "0.5", "UPSCALE_RADAR_CONCURRENCY": "2",
        "UPSCALE_RADAR_FIRST_FUNDER": "on", "UPSCALE_RADAR_TX_RETENTION_DAYS": "1",
        "UPSCALE_RADAR_SNAPSHOT_RETENTION_DAYS": "2",
    })  # fmt: skip
    assert (o.enabled, o.daily_request_budget, o.max_rps, o.concurrency) == (True, 50, 0.5, 2)
    assert o.funding_clusters_enabled
    assert (o.tx_retention_days, o.snapshot_retention_days) == (3, 14)  # floors
    with pytest.raises(ValueError):
        load_settings({"UPSCALE_RADAR": "maybe"})


def test_metric_never_zero_fills() -> None:
    with pytest.raises(ValueError):
        Metric(status="UNAVAILABLE", value=0)
    with pytest.raises(ValueError):
        Metric(status="AVAILABLE")
    with pytest.raises(ValueError):
        Metric(status="PARTIAL", value=3)  # must be flagged as a lower bound


# --- transactions ---------------------------------------------------------------------------


def test_flows_are_per_owner_and_neutral() -> None:
    # A swaps against the tracked pool: exact opposite pool move, signed by A.
    t = tx("s1", T0, A, pre=[(POOL, 1000), (A, 0)], post=[(POOL, 700), (A, 300)])
    p = parse_transaction(t, MINT, POOL)
    assert p is not None and not p.failed
    assert [(f.wallet, f.direction, f.amount_raw, f.counterparty) for f in p.flows] == [
        (A, "TOKEN_INFLOW", 300, "TRACKED_POOL_COUNTERPARTY")
    ]
    assert p.fee_payer == A and p.block_time == T0.timestamp()


def test_counterparty_unverified_unless_proven() -> None:
    # Two wallets change: a plain transfer between them is never matched to the pool.
    t = tx("s2", T0, A, pre=[(A, 500), (B, 0)], post=[(A, 200), (B, 300)])
    p = parse_transaction(t, MINT, POOL)
    assert p is not None
    assert {(f.wallet, f.direction, f.counterparty) for f in p.flows} == {
        (A, "TOKEN_OUTFLOW", "UNVERIFIED"), (B, "TOKEN_INFLOW", "UNVERIFIED"),
    }  # fmt: skip
    # Pool moves the opposite amount but the wallet didn't sign (fee payer is someone else).
    t = tx("s3", T0, C, pre=[(POOL, 1000), (A, 0)], post=[(POOL, 900), (A, 100)], signers=[C])
    p = parse_transaction(t, MINT, POOL)
    assert p is not None and p.flows[0].counterparty == "UNVERIFIED"


def test_pool_side_and_other_mints_ignored_and_failed_has_no_flows() -> None:
    t = tx("s4", T0, A, pre=[(POOL, 10), (RAYDIUM_AMM_V4_AUTHORITY, 5)],
           post=[(POOL, 0), (RAYDIUM_AMM_V4_AUTHORITY, 15)])  # fmt: skip
    p = parse_transaction(t, MINT, POOL)
    assert p is not None
    # The shared vault authority is kept as typed evidence, never as a wallet.
    assert [(f.wallet, f.participant, f.counterparty) for f in p.flows] == [
        (RAYDIUM_AMM_V4_AUTHORITY, "POOL_OR_VAULT", "UNVERIFIED")
    ]
    other = tx("s5", T0, A, pre=[(A, 0)], post=[(A, 5)], mint=addr("ZtherMint"))
    assert parse_transaction(other, MINT, POOL).flows == ()  # type: ignore[union-attr]
    failed = tx("s6", T0, A, pre=[(A, 0)], post=[(A, 5)], err={"InstructionError": [0, "x"]})
    p = parse_transaction(failed, MINT, POOL)
    assert p is not None and p.failed and p.flows == ()


def test_owner_accounts_are_summed() -> None:
    t = tx("s7", T0, A, pre=[(A, 100)], post=[(A, 100)])
    t["meta"]["postTokenBalances"].append(dict(t["meta"]["postTokenBalances"][0], accountIndex=5,
                                               uiTokenAmount={"amount": "40"}))  # fmt: skip
    p = parse_transaction(t, MINT, POOL)
    assert p is not None and [(f.direction, f.amount_raw) for f in p.flows] == [
        ("TOKEN_INFLOW", 40)
    ]
    t["meta"]["postTokenBalances"].append({"mint": MINT, "uiTokenAmount": {"amount": "1"}})
    assert parse_transaction(t, MINT, POOL).unattributed_balances == 1  # type: ignore[union-attr]


def test_mint_initialization_detected_in_inner_instructions_only_for_this_mint() -> None:
    assert parse_transaction(tx("s8", T0, A, init_mint=MINT), MINT, POOL).initializes_mint  # type: ignore[union-attr]
    assert not parse_transaction(
        tx("s9", T0, A, init_mint=addr("ZtherMint")), MINT, POOL
    ).initializes_mint  # type: ignore[union-attr]
    assert parse_transaction({"transaction": {}}, MINT, POOL) is None


def test_first_funder_and_signature_rows() -> None:
    funder = addr("FunderA")
    assert first_funder(tx("f", T0, funder, funds=(funder, A)), A) == funder
    assert first_funder(tx("f", T0, funder, funds=(funder, B)), A) is None
    rows = parse_signatures([
        {"signature": "x", "slot": 5, "blockTime": int((T0 - timedelta(seconds=1)).timestamp()), "err": None},
        {"signature": "y", "slot": 6, "blockTime": None, "err": {"e": 1}}, {"bad": 1}, "junk",
    ])  # fmt: skip
    assert [(r.signature, r.failed, r.block_time is None) for r in rows] == [
        ("x", False, False),
        ("y", True, True),
    ]
