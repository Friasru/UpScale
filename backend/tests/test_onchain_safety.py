"""Solana on-chain safety: RPC/Helius parsing, holder methodology and exclusions, the
caching service, the agent, and how the profile, Risk and Opportunity use it.

All offline: the Solana RPC is replaced by `FakeSolanaRpc` / MockTransport.
"""

import asyncio
from datetime import timedelta
from typing import Any

import httpx2
import pytest

from upscale.agents import AgentContext, OnchainSafetyAgent
from upscale.orchestrator import Orchestrator, with_profile_agents
from upscale.routing import route
from upscale.schemas import AgentResult, ChatMessage, ChatRequest
from upscale.services import opportunity as opportunity_service
from upscale.services import risk as risk_service
from upscale.services.asset_profile import AssetIdentity, build_profile
from upscale.services.market_data import (
    AssetNotFoundError,
    InvalidRequestError,
    MarketDataUnavailableError,
)
from upscale.services.risk import OnchainRiskConfig, collect_inputs, observations, onchain_factors
from upscale.services.solana_chain import (
    INCINERATOR,
    RAYDIUM_AMM_V4_AUTHORITY,
    SYSTEM_PROGRAM,
    TOKEN_2022_PROGRAM,
    TOKEN_PROGRAM,
    HeliusProvider,
    KnownPool,
    SolanaRpcProvider,
    SolanaSafetyService,
)

from .conftest import DEX_NOW, FakeDexScreener, FakeSolanaRpc
from .test_opportunity import full_buy
from .test_risk_agent import market
from .test_solana_dex import BONK_MINT, MINT, OTHER_MINT, pair

NOW = DEX_NOW
POOL = "PoolSo1111111111111111111111111111111111111"  # the DEX pair address in `pair()`
DECIMALS = 6
SUPPLY = 1_000_000_000  # UI units
PROGRAM = "Prog111111111111111111111111111111111111111"  # an unidentified program


# --- Builders -------------------------------------------------------------------------------


def mint_account(
    *,
    mint_authority: str | None = None,
    freeze_authority: str | None = None,
    supply: int = SUPPLY,
    program: str = TOKEN_PROGRAM,
    extensions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "decimals": DECIMALS,
        "freezeAuthority": freeze_authority,
        "isInitialized": True,
        "mintAuthority": mint_authority,
        "supply": str(supply * 10**DECIMALS),
    }
    if extensions is not None:
        info["extensions"] = extensions
    return {
        "owner": program,
        "lamports": 1,
        "executable": False,
        "data": {
            "program": "spl-token-2022" if program == TOKEN_2022_PROGRAM else "spl-token",
            "parsed": {"type": "mint", "info": info},
        },
    }


def token_account(owner: str, mint: str = MINT) -> dict[str, Any]:
    return {
        "owner": TOKEN_PROGRAM,
        "lamports": 1,
        "executable": False,
        "data": {
            "program": "spl-token",
            "parsed": {"type": "account", "info": {"owner": owner, "mint": mint}},
        },
    }


def owner_account(program: str = SYSTEM_PROGRAM) -> dict[str, Any]:
    return {"owner": program, "lamports": 1, "executable": False, "data": ["", "base64"]}


def wallet(n: int) -> str:
    return f"Wa11et{n:02d}".ljust(43, "1")


def holding(
    fake: FakeSolanaRpc,
    owner: str,
    pct: float,
    *,
    n: int,
    mint: str = MINT,
    owner_program: str | None = SYSTEM_PROGRAM,
) -> str:
    """A token account holding `pct`% of supply, owned by `owner`. It appears in the Helius
    scan and, if among the 20 biggest, in getTokenLargestAccounts."""
    address = f"TokAcc{n:04d}".ljust(43, "1")
    raw = int(SUPPLY * pct / 100 * 10**DECIMALS)
    fake.holders.setdefault(MINT, []).append(
        {"address": address, "mint": mint, "owner": owner, "amount": raw}
    )
    fake.accounts[address] = token_account(owner, mint)
    if owner_program is not None:
        fake.accounts.setdefault(owner, owner_account(owner_program))
    return address


def setup_token(
    fake: FakeSolanaRpc,
    holders: list[tuple[str, float]] | None = None,
    **mint_kw: Any,
) -> None:
    """A mint plus its token accounts: by default a liquidity pool vault (40%) and
    spread-out wallets (3% each)."""
    fake.mints[MINT] = mint_account(**mint_kw)
    rows = (
        holders if holders is not None else [(POOL, 40.0)] + [(wallet(i), 3.0) for i in range(12)]
    )
    for n, (owner, pct) in enumerate(rows):
        holding(fake, owner, pct, n=n, owner_program=PROGRAM if owner == POOL else SYSTEM_PROGRAM)


def helius(fake: FakeSolanaRpc, **kw: Any) -> HeliusProvider:
    return HeliusProvider("secret-test-key", transport=fake.transport(), **kw)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def rpc(fake: FakeSolanaRpc) -> SolanaRpcProvider:
    """A generic Solana RPC (no Helius): largest accounts only, no holder scan."""
    return SolanaRpcProvider("https://rpc.example/?api-key=hidden", transport=fake.transport())


def service(
    fake: FakeSolanaRpc,
    clock: Clock | None = None,
    provider: Any = None,
    **kw: Any,
) -> SolanaSafetyService:
    return SolanaSafetyService(
        provider or helius(fake), clock=clock or Clock(), now=lambda: NOW, **kw
    )


POOL_REF = KnownPool(address=POOL, dex="raydium", eligible=True)


def snapshot(
    fake: FakeSolanaRpc, pools: tuple[KnownPool, ...] = (POOL_REF,), provider: Any = None
) -> Any:
    return asyncio.run(service(fake, provider=provider).get_snapshot(MINT, pools))


def dex(fake_dex: FakeDexScreener, **kw: Any) -> AgentResult:
    from upscale.agents import DexMarketAgent
    from upscale.services.dexscreener import DexScreenerProvider
    from upscale.services.solana_dex import SolanaDexService

    fake_dex.pairs[MINT] = [pair(**kw)]
    svc = SolanaDexService(DexScreenerProvider(transport=fake_dex.transport()), now=lambda: NOW)
    ctx = AgentContext(
        query="", assets=[MINT], asset_identity=AssetIdentity(chain="solana", address=MINT)
    )
    return asyncio.run(DexMarketAgent(svc).run(ctx))


def onchain(
    fake: FakeSolanaRpc, prior: dict[Any, AgentResult] | None = None, provider: Any = None
) -> AgentResult:
    ctx = AgentContext(
        query="",
        assets=[MINT],
        asset_identity=AssetIdentity(chain="solana", address=MINT),
        prior_results=prior or {},
    )
    return asyncio.run(OnchainSafetyAgent(service(fake, provider=provider)).run(ctx))


def results(*rs: AgentResult) -> dict[Any, AgentResult]:
    return {r.agent: r for r in rs}


def token_prior(
    fake: FakeSolanaRpc, fake_dex: FakeDexScreener, provider: Any = None, **dex_kw: Any
) -> dict[Any, AgentResult]:
    d = dex(fake_dex, **dex_kw)
    return results(d, onchain(fake, results(d), provider))


def profile_for(prior: dict[Any, AgentResult], asset: str = "NEWT", mint: str = MINT) -> Any:
    ident = AssetIdentity(chain="solana", address=mint)
    return build_profile(
        ident,
        observations(collect_inputs(prior, asset)),
        now=NOW,
        integrated=frozenset({"candles", "market_snapshot", "news", "dex", "onchain"}),
    )


def review(prior: dict[Any, AgentResult], asset: str = "NEWT", **kw: Any) -> Any:
    return risk_service.assess(prior, asset, profile=profile_for(prior, asset), **kw)


def factor(r: Any, fid: str) -> Any:
    return next((f for f in r.factors if f.id == fid), None)


def decide(prior: dict[Any, AgentResult], asset: str = "NEWT") -> Any:
    p = profile_for(prior, asset)
    r = risk_service.assess(prior, asset, profile=p)
    risk = AgentResult(agent="risk", mock=False, summary="r", findings=r.model_dump(mode="json"))
    return opportunity_service.assess(prior | {"risk": risk}, asset, profile=p)


# --- Provider: mint account -----------------------------------------------------------------


def test_reads_authorities_supply_decimals_and_program(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90), freeze_authority=wallet(91))
    s = snapshot(fake_solana_rpc)
    assert s.canonical_id == f"solana:{MINT}" and s.mint == MINT
    assert (s.token_program, s.token_program_id) == ("spl_token", TOKEN_PROGRAM)
    assert (s.decimals, s.supply) == (DECIMALS, SUPPLY)
    assert s.mint_authority == wallet(90) and s.mint_authority_active
    assert s.freeze_authority == wallet(91) and s.freeze_authority_active
    assert s.provider == "Helius" and s.fetched_at == NOW


def test_revoked_authorities(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    s = snapshot(fake_solana_rpc)
    assert not s.mint_authority_active and not s.freeze_authority_active


def test_token_2022_extensions(fake_solana_rpc) -> None:
    exts = [
        {
            "extension": "transferFeeConfig",
            "state": {"newerTransferFee": {"transferFeeBasisPoints": 100}},
        },
        {"extension": "permanentDelegate", "state": {"delegate": wallet(92)}},
    ]
    setup_token(fake_solana_rpc, program=TOKEN_2022_PROGRAM, extensions=exts)
    s = snapshot(fake_solana_rpc)
    assert s.token_program == "token_2022"
    assert s.extensions == ["transferFeeConfig", "permanentDelegate"]
    assert s.extension_state["permanentDelegate"] == {"delegate": wallet(92)}


@pytest.mark.parametrize(
    "setup",
    [
        lambda f: None,  # no account at all
        lambda f: f.accounts.update({MINT: owner_account()}),  # a wallet, not a mint
    ],
)
def test_non_mints_are_not_found(fake_solana_rpc, setup) -> None:
    setup(fake_solana_rpc)
    with pytest.raises(AssetNotFoundError):
        snapshot(fake_solana_rpc)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx2.Response(429), "rate limit"),
        (httpx2.Response(401), "rejected the API key"),
        (httpx2.Response(500), "HTTP 500"),
        (httpx2.Response(200, content=b"not json"), "invalid JSON"),
        (httpx2.Response(200, json=[1, 2]), "unexpected response"),
        (
            httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32005, "message": "node is behind"},
                },
            ),
            "node is behind",
        ),
    ],
)
def test_rpc_failures_are_errors_without_leaking_the_key(
    fake_solana_rpc, response, message
) -> None:
    fake_solana_rpc.handler = lambda r: response
    with pytest.raises(MarketDataUnavailableError, match=message) as exc:
        snapshot(fake_solana_rpc)
    assert "secret-test-key" not in str(exc.value)
    assert "secret-test-key" not in repr(helius(fake_solana_rpc))


def test_authorities_survive_malformed_holder_data(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90))
    fake_solana_rpc.largest[MINT] = [{"address": 5, "amount": "x"}]
    s = snapshot(fake_solana_rpc)
    assert s.authorities_available and s.mint_authority == wallet(90)
    assert not s.holders_available and "malformed" in s.holders_error
    assert s.top1_pct is None and s.concentration_source == "unavailable"
    assert not s.concentration_authoritative


def test_authorities_survive_a_rate_limited_holder_endpoint(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    fake_solana_rpc.fail["getTokenLargestAccounts"] = 429
    s = snapshot(fake_solana_rpc)
    assert s.authorities_available and s.mint_authority_active is False
    assert s.freeze_authority_active is False
    assert not s.holders_available and "rate limit" in s.holders_error
    assert any("holder data is unavailable" in r for r in s.incomplete_reasons)


def test_holders_survive_an_unreadable_mint_account(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [(POOL, 40.0), (wallet(1), 12.0)])
    fake_solana_rpc.fail["getAccountInfo"] = "node is behind"
    s = snapshot(fake_solana_rpc)
    assert not s.authorities_available and "node is behind" in s.authorities_error
    assert s.mint_authority is None and s.mint_authority_active is None  # unknown, not revoked
    assert s.holders_available and s.top1_pct == pytest.approx(12.0)  # supply via getTokenSupply
    assert any(r["method"] == "getTokenSupply" for r in fake_solana_rpc.requests)


def test_both_components_succeed(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    s = snapshot(fake_solana_rpc)
    assert s.authorities_available and s.holders_available
    assert s.authorities_error is None and s.holders_error is None


def test_both_components_failing_is_an_error_and_not_cached(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    fake_solana_rpc.fail = {"getAccountInfo": 429, "getTokenLargestAccounts": 429}
    svc = service(fake_solana_rpc)
    with pytest.raises(MarketDataUnavailableError, match="authorities.*holders"):
        asyncio.run(svc.get_snapshot(MINT))
    fake_solana_rpc.fail = {}
    assert asyncio.run(svc.get_snapshot(MINT)).holders_available  # retried, not cached


def test_partial_results_are_retried_not_cached(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    fake_solana_rpc.fail["getTokenLargestAccounts"] = 429
    svc = service(fake_solana_rpc)
    assert not asyncio.run(svc.get_snapshot(MINT)).holders_available
    del fake_solana_rpc.fail["getTokenLargestAccounts"]
    assert asyncio.run(svc.get_snapshot(MINT)).holders_available


def test_partial_authorities_reach_risk_and_keep_the_holder_blocker(
    fake_solana_rpc, fake_dexscreener
) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90))
    fake_solana_rpc.fail["getTokenLargestAccounts"] = 429
    prior = token_prior(fake_solana_rpc, fake_dexscreener, liquidity=5e6, age=timedelta(days=20))
    p = profile_for(prior)
    assert p.evidence_status("token_authorities").status == "available"
    assert p.evidence_status("holder_concentration").status == "unavailable"
    r = review(prior)
    assert factor(r, "onchain_mint_authority").severity == "high"  # the authority fact survived
    assert factor(r, "onchain_holders_incomplete").severity == "high"
    a = decide(prior)
    assert a.action == "wait"
    reasons = " ".join(f.reason for f in a.blocking_factors)
    assert "holder concentration" in reasons


# --- Holder methodology ---------------------------------------------------------------------


def test_pool_vaults_and_burns_are_excluded_only_when_identified(fake_solana_rpc) -> None:
    rows = [
        (POOL, 30.0),  # this mint's usable DEX pool (from DEX data)
        (RAYDIUM_AMM_V4_AUTHORITY, 10.0),  # Raydium AMM v4 vaults; DEX data reports Raydium
        (INCINERATOR, 5.0),  # burned
        (wallet(1), 8.0),
    ]
    setup_token(fake_solana_rpc, rows)
    s = snapshot(fake_solana_rpc)
    kinds = {e.owner: e.kind for e in s.excluded}
    assert kinds == {
        POOL: "liquidity_pool",
        RAYDIUM_AMM_V4_AUTHORITY: "liquidity_pool",
        INCINERATOR: "burn",
    }
    assert s.excluded_pct == pytest.approx(45.0)
    assert s.pool_addresses_used == sorted([POOL, RAYDIUM_AMM_V4_AUTHORITY])
    assert [e.owner for e in s.holders] == [wallet(1)]


def test_raydium_authority_needs_a_corroborating_raydium_pool(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [(RAYDIUM_AMM_V4_AUTHORITY, 30.0), (wallet(1), 8.0)])
    orca_only = (KnownPool(address=POOL, dex="orca", eligible=True),)
    s = snapshot(fake_solana_rpc, pools=orca_only)
    top = s.holders[0]
    assert top.owner == RAYDIUM_AMM_V4_AUTHORITY and top.kind == "holder"
    assert "no usable Raydium pool" in top.label
    assert s.top1_pct == pytest.approx(30.0)


def test_rejected_pools_are_not_excluded(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [(POOL, 30.0), (wallet(1), 8.0)])
    s = snapshot(fake_solana_rpc, pools=(KnownPool(address=POOL, dex="raydium", eligible=False),))
    assert s.holders[0].owner == POOL and "isn't a usable market" in s.holders[0].label
    assert s.excluded == []


def test_unknown_program_account_remains_counted(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [(POOL, 30.0), (PROGRAM, 12.0), (wallet(1), 8.0)])
    fake_solana_rpc.accounts[PROGRAM] = owner_account("SomeProgram11111111111111111111111111111111")
    s = snapshot(fake_solana_rpc)
    program = s.holders[0]
    assert program.owner == PROGRAM and program.kind == "holder" and program.program_owned
    assert "program-owned account, not identified (counted)" == program.label
    assert s.top1_pct == pytest.approx(12.0)


def test_pool_is_counted_as_a_holder_without_dex_pool_addresses(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [(POOL, 30.0), (wallet(1), 8.0)])
    s = snapshot(fake_solana_rpc, pools=())
    assert s.top1_pct == pytest.approx(30.0)  # overstated, never understated
    assert any("no usable DEX pool" in r for r in s.incomplete_reasons)
    assert not s.holder_data_complete and s.concentration_reliable


def test_one_owner_with_many_token_accounts_is_one_holder(fake_solana_rpc) -> None:
    # A whale splits 18% of supply over 30 small accounts, none among the 20 largest.
    fake_solana_rpc.mints[MINT] = mint_account()
    holding(fake_solana_rpc, POOL, 40.0, n=0, owner_program=PROGRAM)
    for i in range(20):
        holding(fake_solana_rpc, wallet(i), 2.0, n=1 + i)
    for k in range(30):
        holding(fake_solana_rpc, wallet(99), 0.6, n=100 + k)
    s = snapshot(fake_solana_rpc)
    assert s.concentration_source == "full_scan" and s.concentration_authoritative
    whale = s.holders[0]
    assert whale.owner == wallet(99) and len(whale.token_accounts) == 30
    assert whale.pct_of_supply == pytest.approx(18.0)
    assert s.top1_pct == pytest.approx(18.0)
    assert s.holder_count == 21  # 20 wallets + the whale, counted once


def test_account_in_both_sources_is_counted_once(fake_solana_rpc) -> None:
    """The Helius scan and getTokenLargestAccounts overlap: dedupe by token-account
    address before summing by owner."""
    fake_solana_rpc.mints[MINT] = mint_account()
    holding(fake_solana_rpc, POOL, 40.0, n=0, owner_program=PROGRAM)
    shared = holding(fake_solana_rpc, wallet(1), 10.0, n=1)  # in the scan...
    second = holding(fake_solana_rpc, wallet(1), 3.0, n=2)  # same owner, different account
    holding(fake_solana_rpc, wallet(2), 5.0, n=3)
    raw = int(SUPPLY * 10.0 / 100 * 10**DECIMALS)
    fake_solana_rpc.largest[MINT] = [  # ...and the exact same address in the largest list
        {"address": shared, "amount": str(raw), "decimals": DECIMALS}
    ]
    s = snapshot(fake_solana_rpc)
    by_owner = {e.owner: e for e in s.holders}
    assert fake_solana_rpc._largest(MINT)[0]["address"] == shared
    assert s.token_accounts_seen == 4  # four distinct token accounts, not five
    assert by_owner[wallet(1)].pct_of_supply == pytest.approx(13.0)  # 10 once + 3, not 23
    assert sorted(by_owner[wallet(1)].token_accounts) == sorted([shared, second])
    assert s.top1_pct == pytest.approx(13.0)
    assert s.holder_count == 2


def test_top10_is_computed_after_owner_aggregation(fake_solana_rpc) -> None:
    fake_solana_rpc.mints[MINT] = mint_account()
    for k in range(5):  # wallet 1: five accounts of 2% = 10%
        holding(fake_solana_rpc, wallet(1), 2.0, n=k)
    for i in range(2, 14):  # twelve single-account wallets of 3%
        holding(fake_solana_rpc, wallet(i), 3.0, n=10 + i)
    holding(fake_solana_rpc, POOL, 20.0, n=99, owner_program=PROGRAM)
    s = snapshot(fake_solana_rpc)
    assert s.holders[0].owner == wallet(1)
    assert s.top10_pct == pytest.approx(10.0 + 9 * 3.0)  # not five 2% entries
    assert s.holder_count == 13


def test_largest_accounts_fallback_is_labelled_incomplete(fake_solana_rpc) -> None:
    fake_solana_rpc.mints[MINT] = mint_account()
    for i in range(25):
        holding(fake_solana_rpc, wallet(i), 1.0, n=i)
    for k in range(30):  # the whale's small accounts are invisible without a scan
        holding(fake_solana_rpc, wallet(99), 0.5, n=100 + k)
    s = snapshot(fake_solana_rpc, provider=rpc(fake_solana_rpc))
    assert s.concentration_source == "largest_accounts" and s.concentration_lower_bound
    assert not s.concentration_authoritative
    assert s.holder_count is None and s.largest_accounts_seen == 20
    assert s.top1_pct == pytest.approx(1.0)  # a lower bound: the 15% whale isn't visible
    assert any("no full holder scan" in r for r in s.incomplete_reasons)
    assert not any(r["method"] == "getTokenAccounts" for r in fake_solana_rpc.requests)


def test_accounts_of_another_mint_are_never_counted(fake_solana_rpc) -> None:
    fake_solana_rpc.mints[MINT] = mint_account()
    stray = holding(fake_solana_rpc, wallet(1), 50.0, n=0, mint=OTHER_MINT)
    holding(fake_solana_rpc, wallet(2), 4.0, n=1)
    fake_solana_rpc.largest[MINT] = [{"address": stray, "amount": "1", "decimals": 6}]
    s = snapshot(fake_solana_rpc)
    assert [e.owner for e in s.holders] == [wallet(2)]


def test_unresolved_owners_make_concentration_unreliable(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [(wallet(1), 9.0), (wallet(2), 4.0)])
    first = fake_solana_rpc.holders[MINT][0]["address"]
    del fake_solana_rpc.accounts[first]
    s = snapshot(fake_solana_rpc, provider=rpc(fake_solana_rpc))
    assert not s.concentration_reliable
    assert s.top1_pct is None and s.top10_pct is None  # no number rather than a wrong one
    assert any("couldn't be resolved" in r for r in s.incomplete_reasons)


def test_helius_scan_paginates(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [])
    fake_solana_rpc.holders[MINT] = [
        {
            "address": f"a{i}",
            "mint": MINT,
            "owner": f"o{i % 1500}",
            "amount": 10**DECIMALS * (1000 if i < 40 else 1),
        }
        for i in range(1600)
    ]
    s = snapshot(fake_solana_rpc)
    assert s.concentration_source == "full_scan" and s.scan_pages_read == 2
    assert s.holder_count == 1500 and s.holder_count_complete
    assert s.meaningful_holder_count == 40  # 1,000+ tokens = at least 1e-6 of supply
    das = [r for r in fake_solana_rpc.requests if r["method"] == "getTokenAccounts"]
    assert [r["params"]["page"] for r in das] == [1, 2]
    assert all(r["params"]["limit"] == 1000 and r["params"]["mint"] == MINT for r in das)
    assert das[0]["params"]["options"] == {"showZeroBalance": False}


def test_page_cap_truncation_is_a_lower_bound(fake_solana_rpc) -> None:
    fake_solana_rpc.mints[MINT] = mint_account()
    fake_solana_rpc.holders[MINT] = [
        {"address": f"a{i}", "mint": MINT, "owner": f"o{i}", "amount": 5} for i in range(2500)
    ]
    capped = SolanaSafetyService(helius(fake_solana_rpc, max_pages=2), now=lambda: NOW)
    s = asyncio.run(capped.get_snapshot(MINT, (POOL_REF,)))
    assert s.concentration_source == "partial_scan" and s.concentration_lower_bound
    assert (s.scan_pages_read, s.scan_max_pages) == (2, 2)
    assert s.holder_count == 2000 and not s.holder_count_complete
    assert not s.concentration_authoritative
    assert any("page cap" in r for r in s.incomplete_reasons)


def test_scan_missing_largest_accounts_is_not_treated_as_complete(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)  # 13 accounts on chain
    visible = fake_solana_rpc._largest(MINT)
    fake_solana_rpc.largest[MINT] = visible
    fake_solana_rpc.holders[MINT] = []  # the holder index returns nothing (lagging)
    s = snapshot(fake_solana_rpc)
    assert s.concentration_source == "partial_scan" and not s.concentration_authoritative
    assert any("missed 13 of the largest" in r for r in s.incomplete_reasons)
    assert s.top1_pct == pytest.approx(3.0)  # largest accounts still counted


def test_malformed_scan_rows_are_skipped_and_flagged(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    fake_solana_rpc.holders[MINT].append({"mint": MINT, "owner": "x"})  # no address/amount
    s = snapshot(fake_solana_rpc)
    assert not s.holder_count_complete and s.concentration_lower_bound
    assert any("malformed" in r for r in s.incomplete_reasons)


# --- Service --------------------------------------------------------------------------------


def test_chain_data_is_cached_and_reanalyzed_with_new_pools(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    clock = Clock()
    svc = service(fake_solana_rpc, clock, cache_ttl=120)
    without = asyncio.run(svc.get_snapshot(MINT, ()))
    calls = len(fake_solana_rpc.requests)
    with_pools = asyncio.run(svc.get_snapshot(MINT, (POOL_REF,)))
    assert len(fake_solana_rpc.requests) == calls  # cached
    assert without.top1_pct == pytest.approx(40.0) and with_pools.top1_pct == pytest.approx(3.0)
    clock.t += 121
    asyncio.run(svc.get_snapshot(MINT))
    assert len(fake_solana_rpc.requests) > calls


def test_concurrent_requests_are_deduplicated(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    svc = service(fake_solana_rpc)

    async def many() -> list[Any]:
        return await asyncio.gather(*(svc.get_snapshot(MINT) for _ in range(5)))

    asyncio.run(many())
    assert sum(r["method"] == "getAccountInfo" for r in fake_solana_rpc.requests) == 1


def test_failures_are_not_cached_but_missing_mints_are(fake_solana_rpc) -> None:
    svc = service(fake_solana_rpc)
    fake_solana_rpc.handler = lambda r: httpx2.Response(429)
    with pytest.raises(MarketDataUnavailableError):
        asyncio.run(svc.get_snapshot(MINT))
    fake_solana_rpc.handler = fake_solana_rpc.rpc
    setup_token(fake_solana_rpc)
    assert asyncio.run(svc.get_snapshot(MINT)).mint == MINT  # retried

    before = len(fake_solana_rpc.requests)
    for _ in range(2):
        with pytest.raises(AssetNotFoundError):
            asyncio.run(svc.get_snapshot(OTHER_MINT))
    assert len(fake_solana_rpc.requests) - before <= 2  # one attempt (two parallel calls)


def test_rate_limit_and_invalid_mint(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    svc = service(fake_solana_rpc, max_calls_per_minute=1)
    asyncio.run(svc.get_snapshot(MINT))
    with pytest.raises(MarketDataUnavailableError, match="request limit"):
        asyncio.run(svc.get_snapshot(OTHER_MINT))
    with pytest.raises(InvalidRequestError):
        asyncio.run(svc.get_snapshot("BTC"))


# --- Agent ----------------------------------------------------------------------------------


def test_agent_uses_dex_pool_addresses_for_exclusions(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc)
    prior = token_prior(fake_solana_rpc, fake_dexscreener)
    r = prior["onchain_safety"]
    assert r.status == "ok" and r.findings["canonical_id"] == f"solana:{MINT}"
    snap = r.findings["snapshot"]
    assert snap["pool_addresses_used"] == [POOL]
    assert snap["top1_pct"] == pytest.approx(3.0)
    text = " ".join([r.summary, *r.evidence])
    assert "Mint authority revoked" in text and "Top 10 non-pool holders control" in text
    assert "not a verdict" in text
    assert "scam" not in text.replace("safe or a scam", "") and "rug" not in text.lower()


def test_agent_without_a_provider_or_mint(fake_solana_rpc) -> None:
    no_mint = asyncio.run(
        OnchainSafetyAgent(service(fake_solana_rpc)).run(AgentContext(query="", assets=["BTC"]))
    )
    assert no_mint.findings["snapshot"] is None and fake_solana_rpc.requests == []


def test_agent_reports_no_provider_when_unconfigured() -> None:
    ctx = AgentContext(
        query="", assets=[MINT], asset_identity=AssetIdentity(chain="solana", address=MINT)
    )
    r = asyncio.run(OnchainSafetyAgent().run(ctx))
    assert r.status == "ok" and "no Solana RPC provider is configured" in r.summary


def test_agent_outage_is_an_isolated_error(fake_solana_rpc) -> None:
    fake_solana_rpc.handler = lambda r: httpx2.Response(503)
    assert onchain(fake_solana_rpc).status == "error"


# --- Profile --------------------------------------------------------------------------------


def test_onchain_data_satisfies_the_safety_evidence(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc)
    p = profile_for(token_prior(fake_solana_rpc, fake_dexscreener))
    cap = p.capability("onchain")
    assert cap.status == "available" and cap.verified
    assert p.evidence_status("token_authorities").status == "available"
    assert p.evidence_status("holder_concentration").status == "available"
    # Only the token's own pool candles (not part of this fixture) are still missing.
    assert [e.evidence for e in p.unavailable_critical()] == ["technical_structure"]


def test_largest_accounts_fallback_does_not_satisfy_the_requirement(
    fake_solana_rpc, fake_dexscreener
) -> None:
    setup_token(fake_solana_rpc)
    prior = token_prior(fake_solana_rpc, fake_dexscreener, provider=rpc(fake_solana_rpc))
    p = profile_for(prior)
    status = p.evidence_status("holder_concentration")
    assert status.status == "insufficient" and "no full holder scan" in status.reason
    assert {e.evidence for e in p.unavailable_critical()} == {
        "holder_concentration",
        "technical_structure",  # no pool candles in this fixture
    }
    assert p.evidence_status("token_authorities").status == "available"


def test_truncated_scan_does_not_satisfy_the_requirement(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc)
    d = dex(fake_dexscreener)
    capped = helius(fake_solana_rpc, max_pages=0)
    p = profile_for(results(d, onchain(fake_solana_rpc, results(d), capped)))
    assert p.evidence_status("holder_concentration").status == "insufficient"


def test_onchain_data_for_another_mint_is_ignored(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc)
    prior = token_prior(fake_solana_rpc, fake_dexscreener)
    p = profile_for(prior, mint=OTHER_MINT)
    assert p.capability("onchain").verified is False  # the snapshot is another mint's
    status = p.evidence_status("token_authorities")
    assert status is None or status.status != "available"
    r = risk_service.assess(prior, MINT, profile=p)
    assert not any(f.category == "onchain" for f in r.factors)


def test_assets_without_a_solana_mint_have_no_onchain_capability(fake_solana_rpc) -> None:
    for symbol in ("BTC", "SOL"):
        p = build_profile(symbol)
        assert p is not None and p.capability("onchain").status == "unavailable"
        assert "no Solana mint" in p.capability("onchain").reason
    pepe = build_profile("PEPE")  # an Ethereum token: no EVM on-chain provider yet
    assert pepe is not None and pepe.capability("onchain").status == "unavailable"
    assert "integrated for Ethereum" in pepe.capability("onchain").reason


def test_unconfigured_onchain_says_how_to_configure() -> None:
    p = build_profile(AssetIdentity(chain="solana", address=MINT))
    assert p is not None and "UPSCALE_HELIUS_API_KEY" in p.capability("onchain").reason


# --- Routing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "agents"),
    [
        (f"Should I buy {MINT}?", ["dex_market", "onchain_safety", "risk", "opportunity"]),
        ("Should I buy BONK?", None),  # checked below: includes both token agents
        (
            "Should I buy BTC?",
            ["technical_analysis", "market", "news_sentiment", "risk", "opportunity"],
        ),
    ],
)
def test_onchain_agent_is_selected_by_profile_when_configured(
    fake_solana_rpc, query, agents
) -> None:
    d = route(query, has_images=False)
    ident = AssetIdentity(chain="solana", address=d.token_address) if d.token_address else None
    selected = with_profile_agents(d, ident).agents
    if agents is None:
        assert {"dex_market", "onchain_safety"} <= set(selected)
    else:
        assert selected == agents


def test_onchain_agent_is_not_selected_without_a_provider() -> None:
    d = route(f"Should I buy {MINT}?", has_images=False)
    selected = with_profile_agents(d, AssetIdentity(chain="solana", address=MINT)).agents
    assert "onchain_safety" not in selected


# --- Risk -----------------------------------------------------------------------------------


def test_active_authorities_on_a_new_dex_token_are_high_risk(
    fake_solana_rpc, fake_dexscreener
) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90), freeze_authority=wallet(91))
    r = review(token_prior(fake_solana_rpc, fake_dexscreener))
    mint_f, freeze_f = factor(r, "onchain_mint_authority"), factor(r, "onchain_freeze_authority")
    assert mint_f.severity == freeze_f.severity == "high"
    assert "Mint authority still enabled" in mint_f.explanation
    assert "Freeze authority enabled" in freeze_f.explanation
    assert r.overall_risk == "high"
    text = " ".join(f.explanation for f in r.factors).lower()
    assert "scam" not in text and "rug" not in text


def test_authority_severity_depends_on_the_kind_of_asset(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90))
    s = snapshot(fake_solana_rpc)
    cfg = OnchainRiskConfig()
    sev = {
        c: onchain_factors(s, cfg, c)[0].severity
        for c in ("new_dex_token", "established_memecoin", "stablecoin")
    }
    assert sev == {"new_dex_token": "high", "established_memecoin": "medium", "stablecoin": "low"}


@pytest.mark.parametrize(
    ("ext", "state", "severity"),
    [
        ("permanentDelegate", {"delegate": "x"}, "high"),
        ("transferFeeConfig", {}, "medium"),
        ("defaultAccountState", {"accountState": "frozen"}, "high"),
        ("defaultAccountState", {"accountState": "initialized"}, None),
        ("metadataPointer", {}, None),
    ],
)
def test_token_2022_extension_rules(fake_solana_rpc, ext, state, severity) -> None:
    setup_token(
        fake_solana_rpc, program=TOKEN_2022_PROGRAM, extensions=[{"extension": ext, "state": state}]
    )
    f = next(
        (
            f
            for f in onchain_factors(
                snapshot(fake_solana_rpc), OnchainRiskConfig(), "new_dex_token"
            )
            if f.id == "onchain_token_extension"
        ),
        None,
    )
    assert (f.severity if f else None) == severity


@pytest.mark.parametrize(
    ("top", "top1", "top10"),
    [(25.0, "high", None), (12.0, "medium", None), (5.0, None, None)],  # top 10 = top + 1%
)
def test_single_wallet_concentration(fake_solana_rpc, top, top1, top10) -> None:
    setup_token(fake_solana_rpc, [(POOL, 40.0), (wallet(1), top), (wallet(2), 1.0)])
    fs = {f.id: f for f in onchain_factors(snapshot(fake_solana_rpc), OnchainRiskConfig(), None)}
    assert (fs["onchain_top_holder"].severity if "onchain_top_holder" in fs else None) == top1
    assert (fs["onchain_top10"].severity if "onchain_top10" in fs else None) == top10
    if top1:
        assert f"one non-pool wallet controls {top:.1f}%" in fs["onchain_top_holder"].headline


@pytest.mark.parametrize(("each", "severity"), [(7.0, "high"), (4.0, "medium"), (2.0, None)])
def test_top10_concentration(fake_solana_rpc, each, severity) -> None:
    setup_token(fake_solana_rpc, [(POOL, 20.0)] + [(wallet(i), each) for i in range(10)])
    fs = {
        f.id: f.severity
        for f in onchain_factors(snapshot(fake_solana_rpc), OnchainRiskConfig(), "new_dex_token")
    }
    assert fs.get("onchain_top10") == severity


def test_few_holders_only_when_the_count_is_complete(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc, [(POOL, 40.0)] + [(wallet(i), 2.0) for i in range(20)])
    fs = {
        f.id: f.severity
        for f in onchain_factors(snapshot(fake_solana_rpc), OnchainRiskConfig(), None)
    }
    assert fs["onchain_few_holders"] == "high"  # 20 meaningful holders, complete count
    capped = asyncio.run(
        SolanaSafetyService(helius(fake_solana_rpc, max_pages=0), now=lambda: NOW).get_snapshot(
            MINT, (POOL_REF,)
        )
    )
    assert "onchain_few_holders" not in {
        f.id for f in onchain_factors(capped, OnchainRiskConfig(), None)
    }


def test_truncated_data_raises_uncertainty_but_still_flags_concentration(
    fake_solana_rpc, fake_dexscreener
) -> None:
    setup_token(fake_solana_rpc, [(POOL, 40.0), (wallet(1), 25.0), (wallet(2), 4.0)])
    d = dex(fake_dexscreener)
    prior = results(d, onchain(fake_solana_rpc, results(d), helius(fake_solana_rpc, max_pages=0)))
    r = review(prior)
    inc = factor(r, "onchain_holders_incomplete")
    assert inc.affects == "uncertainty" and inc.severity == "medium"
    top = factor(r, "onchain_top_holder")
    assert top.severity == "high" and "at least 25.0%" in top.headline  # lower bound still flags
    assert r.uncertainty_level == "high"  # holder evidence essential for a new token is missing


def test_unresolvable_holders_never_fabricate_safety(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc, [(wallet(1), 25.0), (wallet(2), 4.0)])
    del fake_solana_rpc.accounts[fake_solana_rpc.holders[MINT][0]["address"]]
    prior = token_prior(fake_solana_rpc, fake_dexscreener, provider=rpc(fake_solana_rpc))
    r = review(prior)
    inc = factor(r, "onchain_holders_incomplete")
    assert inc.severity == "high"
    assert factor(r, "onchain_top_holder") is None and factor(r, "onchain_top10") is None
    assert r.uncertainty_level == "high"


def test_missing_pool_addresses_are_medium_uncertainty(fake_solana_rpc) -> None:
    setup_token(fake_solana_rpc)
    s = asyncio.run(service(fake_solana_rpc).get_snapshot(MINT, ()))
    inc = next(
        f
        for f in onchain_factors(s, OnchainRiskConfig(), None)
        if f.id == "onchain_holders_incomplete"
    )
    assert inc.severity == "medium"


def test_onchain_config_is_validated_and_configurable(fake_solana_rpc) -> None:
    with pytest.raises(ValueError):
        OnchainRiskConfig(top1_medium_pct=30.0, top1_high_pct=20.0)
    setup_token(fake_solana_rpc, [(POOL, 40.0), (wallet(1), 12.0)])
    strict = OnchainRiskConfig(top1_medium_pct=5.0, top1_high_pct=10.0)
    fs = {f.id: f.severity for f in onchain_factors(snapshot(fake_solana_rpc), strict, None)}
    assert fs["onchain_top_holder"] == "high"


def test_dex_caveat_disappears_once_safety_was_checked(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc)
    prior = token_prior(fake_solana_rpc, fake_dexscreener, liquidity=20_000.0)
    f = factor(review(prior), "dex_low_liquidity")
    assert "aren't checked yet" not in f.explanation
    dex_only = {"dex_market": prior["dex_market"]}
    assert "aren't checked yet" in factor(review(dex_only), "dex_low_liquidity").explanation


# --- Opportunity ----------------------------------------------------------------------------


def _clean_prior(
    fake: FakeSolanaRpc, fake_dex: FakeDexScreener, provider: Any = None
) -> dict[Any, AgentResult]:
    setup_token(fake)
    for i in range(500):  # a broad base of small, meaningful holders
        holding(fake, f"Sma11{i:04d}".ljust(43, "1"), 0.001, n=1000 + i)
    return token_prior(
        fake,
        fake_dex,
        provider,
        liquidity=5e6,
        age=timedelta(days=20),
        change={"m5": 0.1, "h1": 1.0},
    )


def test_fallback_holder_data_keeps_the_holder_blocker(fake_solana_rpc, fake_dexscreener) -> None:
    a = decide(_clean_prior(fake_solana_rpc, fake_dexscreener, rpc(fake_solana_rpc)))
    assert a.action == "wait"
    reasons = " ".join(f.reason for f in a.blocking_factors)
    assert "holder concentration" in reasons and "token mint/freeze authorities" not in reasons


def test_clean_safety_evidence_still_waits_for_mint_specific_candles(
    fake_solana_rpc, fake_dexscreener
) -> None:
    a = decide(_clean_prior(fake_solana_rpc, fake_dexscreener))
    assert a.action == "wait"
    ids = {f.id for f in a.blocking_factors}
    assert "profile_evidence_missing" not in ids  # authorities and holders are satisfied
    assert ids == {"no_technical"}
    assert "no price candles for this exact mint" in a.blocking_factors[0].reason


def test_safety_findings_become_blockers_and_cautions(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(
        fake_solana_rpc,
        [(POOL, 40.0)] + [(wallet(i), 4.0) for i in range(10)],
        mint_authority=wallet(90),
    )
    a = decide(
        token_prior(fake_solana_rpc, fake_dexscreener, liquidity=5e6, age=timedelta(days=20))
    )
    assert a.action == "wait"
    assert "onchain_mint_authority" in {f.id for f in a.blocking_factors}
    assert "onchain_top10" in {f.id for f in a.cautions}
    assert any(f.reason.startswith("On-chain:") for f in a.blocking_factors)


def test_ticker_candles_never_unblock_a_mint(fake_solana_rpc, fake_dexscreener) -> None:
    prior = _clean_prior(fake_solana_rpc, fake_dexscreener)
    prior |= {r.agent: r for r in full_buy(symbol="NEWT")[:1]}  # ticker-keyed candles
    a = decide(prior)
    assert a.action == "wait" and a.bullish_evidence == []


# --- End to end and unchanged behavior ------------------------------------------------------


def ask(content: str) -> Any:
    request = ChatRequest(messages=[ChatMessage(role="user", content=content)])
    return asyncio.run(Orchestrator().respond(request))


def test_mint_query_end_to_end(fake_solana_rpc, fake_dexscreener) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90))
    fake_dexscreener.pairs[MINT] = [pair()]
    response = ask(f"Should I buy {MINT}?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert list(by_agent) == [
        "dex_market",
        "onchain_safety",
        "technical_analysis",
        "risk",
        "opportunity",
    ]
    assert by_agent["onchain_safety"].findings["snapshot"]["pool_addresses_used"] == [POOL]
    decision = by_agent["opportunity"].findings
    assert decision["action"] == "wait"
    assert "onchain_mint_authority" in {f["id"] for f in decision["blocking_factors"]}
    assert "Mint authority still enabled" in response.message.content


def test_btc_never_reads_the_chain(fake_solana_rpc) -> None:
    ask("Should I buy BTC?")
    assert fake_solana_rpc.requests == []


@pytest.mark.parametrize("symbol", ["BTC", "SOL"])
def test_unrelated_onchain_results_leave_btc_and_sol_unchanged(fake_solana_rpc, symbol) -> None:
    setup_token(fake_solana_rpc, mint_authority=wallet(90))
    base = {r.agent: r for r in full_buy(symbol=symbol)}
    base["market"] = market(symbol=symbol, price=84_445.0)
    with_chain = base | {"onchain_safety": onchain(fake_solana_rpc)}
    before = opportunity_service.assess(
        base, symbol, profile=risk_service.profile_from_results(base, symbol)
    )
    after = opportunity_service.assess(
        with_chain, symbol, profile=risk_service.profile_from_results(with_chain, symbol)
    )
    assert after.model_dump(exclude={"asset_profile", "inputs"}) == before.model_dump(
        exclude={"asset_profile", "inputs"}
    )
    assert before.action == "buy"
    assert (
        risk_service.assess(with_chain, symbol).factors == risk_service.assess(base, symbol).factors
    )


def test_bonk_uses_its_registry_mint(fake_solana_rpc) -> None:
    fake_solana_rpc.mints[BONK_MINT] = mint_account()
    ctx = AgentContext(query="", assets=["BONK"])
    r = asyncio.run(OnchainSafetyAgent(service(fake_solana_rpc)).run(ctx))
    assert r.findings["mint"] == BONK_MINT and r.findings["snapshot"] is not None
