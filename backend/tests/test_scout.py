"""Scout: discovery providers, exact identity, snapshots, growth features, filters, and the
request gate. All offline: GeckoTerminal and DEX Screener are MockTransport fakes.
"""

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from upscale.services.market_data import MarketDataUnavailableError
from upscale.services.scout import (
    DexScreenerDiscoveryProvider,
    DiscoveryNotSupportedError,
    GeckoTerminalDiscoveryProvider,
    RequestGate,
    ScoutConfig,
    ScoutConfigError,
    ScoutService,
    ScoutSnapshot,
    ScoutSnapshotStore,
    load_scout_config,
)
from upscale.services.scout.config import ScoutFeatureConfig, ScoutProviderLimits
from upscale.services.scout.features import compare_with, window_acceleration
from upscale.services.scout.models import ScoutMarketMetrics, ScoutWindow
from upscale.services.scout.normalize import Listing, build_candidates
from upscale.services.scout.store import snapshot_of

from .test_solana_dex import pair

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WETH_BASE = "0x4200000000000000000000000000000000000006"
MINT_A = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
MINT_B = "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"
MINT_C = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
EVM_TOKEN = "0xAbCdEf0123456789aBcDeF0123456789AbCdEf01"
POOL_1 = "Pool1111111111111111111111111111111111111111"
POOL_2 = "Pool2222222222222222222222222222222222222222"
POOL_3 = "Pool3333333333333333333333333333333333333333"


# --- Fakes and builders ---------------------------------------------------------------------


class Now:
    """A shared, movable wall clock (datetime) and monotonic clock (seconds)."""

    def __init__(self) -> None:
        self.at = NOW

    def __call__(self) -> datetime:
        return self.at

    def mono(self) -> float:
        return self.at.timestamp()

    def advance(self, **kw: float) -> None:
        self.at += timedelta(**kw)


def gt_pool(
    pool: str,
    token: str,
    *,
    network: str = "solana",
    symbol: str = "NEWT",
    quote: str = SOL,
    dex: str = "raydium",
    reserve: str | None = "25000.5",
    price: str | None = "0.0042",
    fdv: str | None = "900000",
    mcap: str | None = None,
    created: datetime | None = NOW - timedelta(hours=3),
    txns: dict[str, tuple[int, int]] | None = None,
    volume: dict[str, str] | None = None,
    change: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A GeckoTerminal pool row and its base-token row (for `included`)."""
    txns = txns or {
        "m5": (30, 10),
        "m15": (60, 30),
        "m30": (90, 60),
        "h1": (150, 100),
        "h6": (400, 300),
        "h24": (800, 700),
    }
    volume = volume or {
        "m5": "3000",
        "m15": "6000",
        "m30": "9000",
        "h1": "12000",
        "h6": "40000",
        "h24": "90000",
    }
    change = change or {"m5": "2.5", "m15": "4", "m30": "5", "h1": "6", "h6": "12", "h24": "30"}
    attrs: dict[str, Any] = {
        "address": pool,
        "name": f"{symbol} / Q",
        "base_token_price_quote_token": "0.00003",
        "fdv_usd": fdv,
        "market_cap_usd": mcap,
        "reserve_in_usd": reserve,
        "pool_created_at": created.isoformat().replace("+00:00", "Z") if created else None,
        "transactions": {
            w: {"buys": b, "sells": s, "buyers": b // 2, "sellers": s // 2}
            for w, (b, s) in txns.items()
        },
        "volume_usd": volume,
        "price_change_percentage": change,
    }
    if price is not None:
        attrs["base_token_price_usd"] = price
    row = {
        "id": f"{network}_{pool}",
        "type": "pool",
        "attributes": attrs,
        "relationships": {
            "base_token": {"data": {"id": f"{network}_{token}", "type": "token"}},
            "quote_token": {"data": {"id": f"{network}_{quote}", "type": "token"}},
            "dex": {"data": {"id": dex, "type": "dex"}},
        },
    }
    token_row = {
        "id": f"{network}_{token}",
        "type": "token",
        "attributes": {"address": token, "name": f"{symbol} Token", "symbol": symbol},
    }
    return row, token_row


def gt_doc(*items: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    tokens = {t["id"]: t for _, t in items}
    return {"data": [p for p, _ in items], "included": list(tokens.values())}


class FakeGeckoTerminal:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.listings: dict[str, Any] = {}  # path -> JSON body
        self.tokens: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        self.fail: int | None = None

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.fail:
            return httpx2.Response(self.fail)
        path = request.url.path.removeprefix("/api/v2")
        if "/tokens/multi/" in path:
            addresses = path.rsplit("/", 1)[1].split(",")
            pools = [p for a in addresses for p, _ in self.tokens.get(a, [])]
            token_rows = [t for a in addresses for _, t in self.tokens.get(a, [])[:1]]
            return httpx2.Response(200, json={"data": token_rows, "included": pools})
        if path in self.listings:
            return httpx2.Response(200, json=self.listings[path])
        return httpx2.Response(200, json={"data": []})

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)


class FakeDexScreener:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.profiles: list[Any] = []
        self.pairs: dict[str, list[dict[str, Any]]] = {}  # token address -> pairs
        self.fail: int | None = None

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.fail:
            return httpx2.Response(self.fail)
        path = request.url.path
        if path == "/token-profiles/latest/v1":
            return httpx2.Response(200, json=self.profiles)
        if path.startswith("/tokens/v1/"):
            addresses = path.rsplit("/", 1)[1].split(",")
            return httpx2.Response(200, json=[p for a in addresses for p in self.pairs.get(a, [])])
        return httpx2.Response(404)

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)


def limits(**kw: Any) -> ScoutProviderLimits:
    return ScoutProviderLimits(**({"calls_per_minute": 1000} | kw))


def providers(
    gt: FakeGeckoTerminal, ds: FakeDexScreener, now: Now, config: ScoutConfig | None = None
) -> tuple[GeckoTerminalDiscoveryProvider, DexScreenerDiscoveryProvider]:
    config = config or ScoutConfig()
    return (
        GeckoTerminalDiscoveryProvider(
            config,
            RequestGate("GeckoTerminal", limits(), now.mono),
            transport=gt.transport(),
            now=now,
        ),
        DexScreenerDiscoveryProvider(
            config,
            RequestGate("DEX Screener", limits(), now.mono),
            transport=ds.transport(),
            now=now,
        ),
    )


def make_service(
    tmp_path: Path,
    gt: FakeGeckoTerminal,
    ds: FakeDexScreener,
    now: Now,
    config: ScoutConfig | None = None,
    **kw: Any,
) -> ScoutService:
    config = config or ScoutConfig(chains=("solana", "base"))
    return ScoutService(
        list(providers(gt, ds, now, config)),
        ScoutSnapshotStore(tmp_path / "scout.sqlite3"),
        config,
        now=now,
        **kw,
    )


def listing(kind: Any = "new", provider: str = "GeckoTerminal") -> Listing:
    return Listing(provider, kind, "new_pools", NOW)


def dex_pools(*rows: dict[str, Any]) -> list[Any]:
    from upscale.services.dexscreener import parse_pair

    return [p for r in rows if (p := parse_pair(r)) is not None]


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- Discovery and normalization ------------------------------------------------------------


def test_new_pools_become_candidates_with_exact_identity(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/solana/new_pools"] = gt_doc(gt_pool(POOL_1, MINT_A))
    gt_provider, _ = providers(gt, ds, now)
    result = run(gt_provider.discover_new_tokens("solana", 10))

    [c] = result.candidates
    assert c.canonical_id == f"solana:{MINT_A}"
    assert (c.chain, c.address, c.symbol, c.name) == ("solana", MINT_A, "NEWT", "NEWT Token")
    assert c.pool.address == POOL_1 and c.pool.dex == "raydium" and c.pool.quote_kind == "SOL"
    assert c.pool.age_hours == pytest.approx(3.0)
    assert c.token_created_at is None and c.token_age_hours is None  # never invented
    assert c.oldest_pool_created_at == NOW - timedelta(hours=3)
    [source] = c.sources
    assert (source.provider, source.kind, source.listing, source.position) == (
        "GeckoTerminal",
        "new",
        "new_pools",
        0,
    )
    m = c.metrics
    assert m.price_usd == 0.0042 and m.liquidity_usd == 25000.5
    assert [w.window for w in m.windows] == ["m5", "m15", "m30", "h1", "h6", "h24"]
    m15 = m.window("m15")
    assert m15 is not None
    assert (m15.buys, m15.sells, m15.txns, m15.buyers, m15.sellers) == (60, 30, 90, 30, 15)
    assert m15.volume_usd == 6000 and m15.price_change_pct == 4
    [req] = gt.requests
    assert req.url.params["include"] == "base_token,quote_token,dex"


def test_missing_market_cap_stays_missing_and_fdv_is_not_used_instead() -> None:
    [pool] = run(_gt_pools(gt_pool(POOL_1, MINT_A, mcap=None, fdv="900000")))
    result = build_candidates([pool], listing(), ScoutConfig())
    [c] = result.candidates
    assert c.metrics.market_cap_usd is None
    assert c.metrics.fdv_usd == 900000
    assert c.risk_flags.market_cap_missing and c.risk_flags.fdv_only


def test_market_cap_and_fdv_are_kept_distinct() -> None:
    pools = dex_pools(pair(POOL_1, mint=MINT_A, mcap=1_000_000.0, fdv=5_000_000.0))
    [c] = build_candidates(pools, listing(), ScoutConfig()).candidates
    assert (c.metrics.market_cap_usd, c.metrics.fdv_usd) == (1_000_000.0, 5_000_000.0)
    assert c.risk_flags.fdv_far_above_market_cap and not c.risk_flags.fdv_only
    assert "FDV is 5.0x the reported market cap" in c.risk_flags.notes


def test_duplicate_tickers_stay_separate_tokens() -> None:
    pools = dex_pools(
        pair(POOL_1, mint=MINT_A, symbol="XYZ"), pair(POOL_2, mint=MINT_B, symbol="XYZ")
    )
    result = build_candidates(pools, listing(), ScoutConfig())
    assert sorted(c.canonical_id for c in result.candidates) == sorted(
        [f"solana:{MINT_A}", f"solana:{MINT_B}"]
    )
    assert {c.symbol for c in result.candidates} == {"XYZ"}


def test_multiple_pools_of_one_token_become_one_candidate() -> None:
    pools = dex_pools(
        pair(POOL_1, mint=MINT_A, liquidity=40_000.0),
        pair(POOL_2, mint=MINT_A, liquidity=90_000.0, quote=USDC),
        # biggest, but quoted in an unrecognized token: never preferred
        pair(
            POOL_3,
            mint=MINT_A,
            liquidity=500_000.0,
            quote="FakeQuote1111111111111111111111111111111111",
        ),
    )
    [c] = build_candidates(pools, listing(), ScoutConfig()).candidates
    assert c.pool.address == POOL_2 and c.pool.quote_kind == "USDC"
    assert c.pool_count == 3
    assert c.recognized_quote_liquidity_usd == 130_000.0
    assert not c.risk_flags.unrecognized_quote


def test_evm_addresses_are_case_insensitive_and_chains_are_separate() -> None:
    lower, upper = EVM_TOKEN.lower(), EVM_TOKEN.upper().replace("0X", "0x")
    pools = dex_pools(
        pair("0x" + "1" * 40, mint=lower, chain="base", quote=WETH_BASE),
        pair("0x" + "2" * 40, mint=upper, chain="base", quote=WETH_BASE),
        pair("0x" + "3" * 40, mint=lower, chain="ethereum", quote=WETH_BASE),
    )
    result = build_candidates(pools, listing(), ScoutConfig())
    ids = sorted(c.canonical_id for c in result.candidates)
    assert ids == [f"base:{lower}", f"ethereum:{lower}"]
    base = next(c for c in result.candidates if c.chain == "base")
    assert base.pool_count == 2 and base.address == lower


def test_solana_mints_are_case_sensitive() -> None:
    other_case = MINT_A[0] + MINT_A[1].upper() + MINT_A[2:]  # 7xK... vs 7XK...
    pools = dex_pools(pair(POOL_1, mint=MINT_A), pair(POOL_2, mint=other_case))
    result = build_candidates(pools, listing(), ScoutConfig())
    assert len(result.candidates) == 2


def test_unverifiable_identity_and_quote_assets_are_rejected() -> None:
    pools = dex_pools(
        pair(POOL_1, mint="not-a-mint"),
        pair(POOL_2, mint=SOL, quote=USDC),  # the base is a quote asset
        pair("0x" + "4" * 40, mint=EVM_TOKEN, chain="fantom"),  # chain UpScale can't verify
    )
    result = build_candidates(pools, listing(), ScoutConfig())
    assert result.candidates == []
    reasons = [r for x in result.rejected for r in x.reasons]
    assert any("not a valid Solana token address" in r for r in reasons)
    assert any("quote asset" in r for r in reasons)
    assert any("can't verify token addresses on fantom" in r for r in reasons)


def test_no_liquidity_and_no_trades_are_rejected_with_reasons() -> None:
    pools = dex_pools(
        pair(POOL_1, mint=MINT_A, liquidity=50.0),
        pair(POOL_2, mint=MINT_B, txns={"h24": (0, 0)}),
        pair(POOL_3, mint=MINT_C, price=None),
    )
    result = build_candidates(pools, listing(), ScoutConfig())
    assert result.candidates == []
    by_id = {r.canonical_id: " ".join(r.reasons) for r in result.rejected}
    assert "below $1,000" in by_id[f"solana:{MINT_A}"]
    assert "0 trades in 24h" in by_id[f"solana:{MINT_B}"]
    assert "no USD price" in by_id[f"solana:{MINT_C}"]


def test_filter_thresholds_come_from_config() -> None:
    pools = dex_pools(pair(POOL_1, mint=MINT_A, liquidity=50.0))
    cfg = ScoutConfig(filters={"min_liquidity_usd": 10.0})
    assert len(build_candidates(pools, listing(), cfg).candidates) == 1


def test_malformed_provider_rows_are_counted_not_guessed() -> None:
    good = gt_pool(POOL_1, MINT_A)
    broken = gt_pool(POOL_2, MINT_B)
    del broken[0]["relationships"]["dex"]
    doc = gt_doc(good, broken)
    doc["data"].append("garbage")
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/solana/new_pools"] = doc
    result = run(providers(gt, ds, now)[0].discover_new_tokens("solana", 10))
    assert [c.address for c in result.candidates] == [MINT_A]
    assert any("2 malformed row(s)" in r for x in result.rejected for r in x.reasons)


def test_malformed_document_is_a_provider_error() -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/solana/new_pools"] = {"unexpected": True}
    with pytest.raises(MarketDataUnavailableError, match="unexpected response"):
        run(providers(gt, ds, now)[0].discover_new_tokens("solana", 10))


def test_listing_limit_is_respected() -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/solana/new_pools"] = gt_doc(
        gt_pool(POOL_1, MINT_A), gt_pool(POOL_2, MINT_B), gt_pool(POOL_3, MINT_C)
    )
    result = run(providers(gt, ds, now)[0].discover_new_tokens("solana", 2))
    assert [c.address for c in result.candidates] == [MINT_A, MINT_B]


def test_active_and_trending_listings_use_their_endpoints() -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/base/pools"] = gt_doc(
        gt_pool("0x" + "5" * 40, EVM_TOKEN.lower(), network="base", quote=WETH_BASE)
    )
    gt.listings["/networks/solana/trending_pools"] = gt_doc(gt_pool(POOL_1, MINT_A))
    gt_provider = providers(gt, ds, now)[0]
    [active] = run(gt_provider.discover_active_tokens("base", 5)).candidates
    [trending] = run(gt_provider.discover_trending_tokens("solana", 5)).candidates
    assert active.canonical_id == f"base:{EVM_TOKEN.lower()}" and active.pool.quote_kind == "WETH"
    assert trending.sources[0].kind == "trending"
    params = {r.url.path: dict(r.url.params) for r in gt.requests}
    assert params["/api/v2/networks/base/pools"]["sort"] == "h24_tx_count_desc"
    assert params["/api/v2/networks/solana/trending_pools"]["duration"] == "1h"


def test_unsupported_kinds_and_chains_are_never_faked() -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt_provider, ds_provider = providers(gt, ds, now)
    with pytest.raises(DiscoveryNotSupportedError, match="trending"):
        run(ds_provider.discover_trending_tokens("solana", 5))
    with pytest.raises(DiscoveryNotSupportedError, match="active"):
        run(ds_provider.discover_active_tokens("solana", 5))
    with pytest.raises(DiscoveryNotSupportedError):
        run(gt_provider.discover_new_tokens("fantom", 5))
    assert gt.requests == [] and ds.requests == []


def test_dexscreener_new_profiles_are_looked_up_in_batches() -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    mints = [_mint(i) for i in range(35)]
    ds.profiles = [{"chainId": "solana", "tokenAddress": m} for m in mints] + [
        {"chainId": "base", "tokenAddress": EVM_TOKEN},
        {"no": "address"},
    ]
    for i, m in enumerate(mints):
        ds.pairs[m] = [pair(f"Pool{i:040d}"[:44], mint=m, symbol=f"T{i}")]
    result = run(providers(gt, ds, now)[1].discover_new_tokens("solana", 40))
    assert len(result.candidates) == 35
    lookups = [r for r in ds.requests if r.url.path.startswith("/tokens/v1/solana/")]
    assert [len(r.url.path.rsplit("/", 1)[1].split(",")) for r in lookups] == [30, 5]
    assert result.candidates[0].sources[0].note and "paid" in result.candidates[0].sources[0].note
    assert any("1 malformed row(s)" in r for x in result.rejected for r in x.reasons)


def test_exact_lookup_ignores_pools_where_the_token_is_only_the_quote() -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    ds.pairs[MINT_A] = [
        pair(POOL_1, mint=MINT_A),
        pair(POOL_2, mint=MINT_B, quote=MINT_A),  # another token quoted in MINT_A
    ]
    result = run(providers(gt, ds, now)[1].lookup_exact_token("solana", MINT_A))
    assert [c.canonical_id for c in result.candidates] == [f"solana:{MINT_A}"]


def test_geckoterminal_batched_lookup_parses_multi_token_documents() -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.tokens[MINT_A] = [gt_pool(POOL_1, MINT_A, mcap="700000")]
    gt.tokens[MINT_B] = [gt_pool(POOL_2, MINT_B, symbol="OTHER")]
    result = run(providers(gt, ds, now)[0].lookup_exact_tokens("solana", [MINT_A, MINT_B]))
    by_id = {c.address: c for c in result.candidates}
    assert by_id[MINT_A].metrics.market_cap_usd == 700000
    assert by_id[MINT_B].symbol == "OTHER"
    [req] = gt.requests
    assert req.url.params["include"] == "top_pools"


# --- Service: merging, enrichment, failures -------------------------------------------------


def test_discover_merges_sources_and_enriches_with_exact_lookup(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/solana/new_pools"] = gt_doc(gt_pool(POOL_1, MINT_A))
    gt.listings["/networks/solana/trending_pools"] = gt_doc(
        gt_pool(POOL_1, MINT_A), gt_pool(POOL_3, MINT_C)
    )
    # DEX Screener knows every pool of MINT_A (and nothing about MINT_C yet)
    ds.pairs[MINT_A] = [
        pair(POOL_1, mint=MINT_A, liquidity=25_000.0),
        pair(POOL_2, mint=MINT_A, liquidity=60_000.0, quote=USDC),
    ]
    svc = make_service(tmp_path, gt, ds, now)
    scout = run(svc.discover(chains=["solana"]))

    by_id = {c.address: c for c in scout.candidates}
    assert set(by_id) == {MINT_A, MINT_C}
    a = by_id[MINT_A]
    assert a.market_provider == "DEX Screener" and a.pool.address == POOL_2 and a.pool_count == 2
    assert {(s.provider, s.kind) for s in a.sources} == {
        ("GeckoTerminal", "new"),
        ("GeckoTerminal", "trending"),
        ("DEX Screener", "lookup"),
    }
    c = by_id[MINT_C]  # unknown to the lookup: keeps what discovery observed
    assert c.market_provider == "GeckoTerminal" and c.metrics.window("m15") is not None
    assert "not a recommendation" in scout.disclaimer


def test_a_failing_provider_does_not_stop_discovery(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    ds.fail = 503
    gt.listings["/networks/solana/new_pools"] = gt_doc(gt_pool(POOL_1, MINT_A))
    scout = run(make_service(tmp_path, gt, ds, now).discover(kinds=["new"], chains=["solana"]))
    assert [c.address for c in scout.candidates] == [MINT_A]
    assert {(e.provider, e.kind) for e in scout.errors} == {
        ("DEX Screener", "new"),
        ("DEX Screener", "lookup"),
    }
    assert all("HTTP 503" in e.error for e in scout.errors)


def test_every_provider_failing_gives_an_empty_run_with_errors(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.fail = ds.fail = 429
    scout = run(make_service(tmp_path, gt, ds, now).discover(chains=["solana"]))
    assert scout.candidates == []
    assert scout.errors and all("rate limit" in e.error for e in scout.errors)


def test_exact_token_lookup_falls_back_to_the_next_provider(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    ds.fail = 500
    gt.tokens[MINT_A] = [gt_pool(POOL_1, MINT_A)]
    scout = run(make_service(tmp_path, gt, ds, now).lookup_exact_token("solana", MINT_A))
    [c] = scout.candidates
    assert c.market_provider == "GeckoTerminal" and c.first_seen_at == NOW
    assert [e.provider for e in scout.errors] == ["DEX Screener"]


# --- Snapshot storage -----------------------------------------------------------------------


def _candidate(**metric_kw: Any) -> Any:
    [pool] = run(_gt_pools(gt_pool(POOL_1, MINT_A, **metric_kw)))
    [c] = build_candidates([pool], listing(), ScoutConfig()).candidates
    return c


def test_snapshots_persist_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "scout.sqlite3"
    store = ScoutSnapshotStore(path)
    c = _candidate(mcap="700000")
    assert run(store.record_seen(c)) == NOW
    assert run(store.save_snapshot(snapshot_of(c), 30))
    store.close()

    reopened = ScoutSnapshotStore(path)
    [snap] = run(reopened.history(c.canonical_id))
    assert snap.observed_at == NOW and snap.provider == "GeckoTerminal"
    assert snap.pool_address == POOL_1
    assert snap.metrics == c.metrics  # every window, market cap and FDV round-trip
    assert snap.holder_count is None and snap.social is None
    assert run(reopened.first_seen(c.canonical_id)) == NOW
    assert run(reopened.tracked_tokens(NOW - timedelta(hours=1))) == [("solana", MINT_A)]


def test_same_observation_is_stored_once_and_min_interval_applies(tmp_path: Path) -> None:
    store = ScoutSnapshotStore(tmp_path / "s.sqlite3")
    c = _candidate()
    run(store.record_seen(c))
    assert run(store.save_snapshot(snapshot_of(c), 0))
    assert not run(store.save_snapshot(snapshot_of(c), 0))  # cached data seen again
    soon = c.model_copy(update={"observed_at": NOW + timedelta(seconds=10)})
    assert not run(store.save_snapshot(snapshot_of(soon), 30))
    later = c.model_copy(update={"observed_at": NOW + timedelta(seconds=45)})
    assert run(store.save_snapshot(snapshot_of(later), 30))
    assert len(run(store.history(c.canonical_id))) == 2


def test_first_seen_is_kept_across_sightings(tmp_path: Path) -> None:
    store = ScoutSnapshotStore(tmp_path / "s.sqlite3")
    c = _candidate()
    run(store.record_seen(c))
    later = c.model_copy(update={"observed_at": NOW + timedelta(hours=2), "symbol": None})
    assert run(store.record_seen(later)) == NOW


def test_nearest_snapshot_respects_tolerance_and_never_interpolates(tmp_path: Path) -> None:
    store = ScoutSnapshotStore(tmp_path / "s.sqlite3")
    c = _candidate()
    run(store.record_seen(c))
    for minutes in (0, 14, 60):
        snap = snapshot_of(c).model_copy(update={"observed_at": NOW + timedelta(minutes=minutes)})
        run(store.save_snapshot(snap, 0))
    now = NOW + timedelta(minutes=30)
    near = run(
        store.nearest_snapshot(
            c.canonical_id, now - timedelta(minutes=15), timedelta(minutes=4), before=now
        )
    )
    assert near is not None and near.observed_at == NOW + timedelta(minutes=14)
    assert (
        run(
            store.nearest_snapshot(
                c.canonical_id, now - timedelta(minutes=5), timedelta(minutes=1), before=now
            )
        )
        is None
    )
    # never a snapshot from the future of the observation being compared
    assert (
        run(
            store.nearest_snapshot(
                c.canonical_id, NOW, timedelta(minutes=90), before=NOW, provider="GeckoTerminal"
            )
        )
        is None
    )
    assert run(store.prune(NOW + timedelta(minutes=30))) == 2


def test_in_memory_store_works(tmp_path: Path) -> None:
    store = ScoutSnapshotStore(":memory:")
    c = _candidate()
    run(store.record_seen(c))
    run(store.save_snapshot(snapshot_of(c), 0))
    assert len(run(store.history(c.canonical_id))) == 1


# --- Growth features ------------------------------------------------------------------------


def metrics(**windows: tuple[float | None, int | None, int | None, float | None]) -> Any:
    return ScoutMarketMetrics(
        price_usd=1.0,
        liquidity_usd=10_000.0,
        windows=[
            ScoutWindow(window=w, volume_usd=v, buys=b, sells=s, price_change_pct=p)
            for w, (v, b, s, p) in windows.items()
        ],
    )


def test_window_acceleration_compares_per_minute_rates() -> None:
    cfg = ScoutFeatureConfig(window_pairs=(("m5", "h1"), ("h1", "h24")))
    m = metrics(
        m5=(3_000.0, 40, 10, 2.0),  # 600 $/min, 10 trades/min, 80% buys
        h1=(12_000.0, 150, 150, 6.0),  # 200 $/min, 5 trades/min, 50% buys
        h24=(None, 0, 0, 30.0),
    )
    [fast, day] = window_acceleration(m, cfg)
    assert fast.volume_rate_ratio == pytest.approx(3.0)
    assert fast.txn_rate_ratio == pytest.approx(2.0)
    assert fast.buy_share_change == pytest.approx(0.3)
    assert fast.price_velocity_change_pct_per_hour == pytest.approx(24.0 - 6.0)
    # missing or zero bases give None, never a guess
    assert day.volume_rate_ratio is None and day.txn_rate_ratio is None
    assert day.buy_share_change is None


def test_history_comparison_measures_changes() -> None:
    c = _candidate(
        reserve="30000", price="0.005", mcap="1000000", fdv="2000000",
        volume={"h1": "24000"}, txns={"h1": (300, 100), "h24": (900, 500)},
    )  # fmt: skip
    then = ScoutSnapshot(
        canonical_id=c.canonical_id,
        observed_at=NOW - timedelta(minutes=16),
        provider="GeckoTerminal",
        pool_address=POOL_1,
        dex="raydium",
        metrics=ScoutMarketMetrics(
            price_usd=0.004,
            market_cap_usd=800_000.0,
            fdv_usd=None,
            liquidity_usd=20_000.0,
            windows=[ScoutWindow(window="h1", volume_usd=8_000.0, buys=100, sells=100)],
        ),
    )
    h = compare_with(c, then, 15, ScoutFeatureConfig())
    assert h.elapsed_minutes == pytest.approx(16)
    assert h.same_pool and h.volume_window == "h1"
    assert h.price_change_pct == pytest.approx(25.0)
    assert h.liquidity_change_pct == pytest.approx(50.0)
    assert h.market_cap_change_pct == pytest.approx(25.0)
    assert h.fdv_change_pct is None  # FDV missing earlier: no change reported
    assert h.volume_ratio == pytest.approx(3.0)
    assert h.txn_ratio == pytest.approx(2.0)
    assert h.buy_share_change == pytest.approx(0.25)


def test_history_comparison_skips_pool_figures_when_the_pool_changed() -> None:
    c = _candidate()
    then = snapshot_of(c).model_copy(
        update={"observed_at": NOW - timedelta(minutes=5), "pool_address": POOL_2}
    )
    h = compare_with(c, then, 5, ScoutFeatureConfig())
    assert not h.same_pool and h.price_change_pct == 0
    assert h.liquidity_change_pct is None and h.volume_ratio is None
    assert "pool changed" in h.notes[0]


def test_acceleration_from_stored_history_end_to_end(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    cfg = ScoutConfig(chains=("solana",), kinds=("new",))
    svc = make_service(tmp_path, gt, ds, now, cfg, enrichment_provider=None)
    path = "/networks/solana/new_pools"

    gt.listings[path] = gt_doc(gt_pool(POOL_1, MINT_A, volume={"h1": "10000", "h24": "50000"}))
    first = run(svc.discover())
    [c0] = first.candidates
    assert c0.features is not None and c0.features.history == []
    assert c0.features.missing_lookbacks == [5, 15, 30, 60, 360, 1440]  # no fake history
    assert c0.features.window_acceleration  # measurable from one observation

    now.advance(minutes=15)
    gt.listings[path] = gt_doc(
        gt_pool(POOL_1, MINT_A, reserve="50001", volume={"h1": "30000", "h24": "80000"})
    )
    [c1] = run(svc.discover()).candidates
    assert c1.first_seen_at == NOW
    assert c1.features is not None
    [h] = c1.features.history
    assert h.lookback_minutes == 15 and h.elapsed_minutes == pytest.approx(15)
    assert h.volume_ratio == pytest.approx(3.0)
    assert h.liquidity_change_pct == pytest.approx(100.0, rel=1e-3)
    assert 15 not in c1.features.missing_lookbacks
    assert len(run(svc.store.history(c1.canonical_id))) == 2


def test_refresh_tracked_extends_history_with_batched_lookups(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/solana/new_pools"] = gt_doc(
        gt_pool(POOL_1, MINT_A), gt_pool(POOL_2, MINT_B)
    )
    cfg = ScoutConfig(chains=("solana",), kinds=("new",))
    svc = make_service(tmp_path, gt, ds, now, cfg)
    run(svc.discover())
    ds.pairs[MINT_A] = [pair(POOL_1, mint=MINT_A)]
    ds.pairs[MINT_B] = [pair(POOL_2, mint=MINT_B)]
    now.advance(minutes=5)
    refreshed = run(svc.refresh_tracked(timedelta(hours=1)))
    assert {c.address for c in refreshed.candidates} == {MINT_A, MINT_B}
    lookups = [r for r in ds.requests if r.url.path.startswith("/tokens/v1/")]
    assert len(lookups) == 2  # one during discovery enrichment, one batched refresh


# --- Request gate: caching, deduplication, limits, timeouts -----------------------------------


def test_listings_are_cached_until_the_ttl_expires(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    gt.listings["/networks/solana/new_pools"] = gt_doc(gt_pool(POOL_1, MINT_A))
    gt_provider = providers(gt, ds, now)[0]
    run(gt_provider.discover_new_tokens("solana", 10))
    run(gt_provider.discover_new_tokens("solana", 10))
    assert len(gt.requests) == 1
    now.advance(seconds=31)
    run(gt_provider.discover_new_tokens("solana", 10))
    assert len(gt.requests) == 2


def test_dexscreener_profiles_are_fetched_once_for_all_chains(tmp_path: Path) -> None:
    gt, ds, now = FakeGeckoTerminal(), FakeDexScreener(), Now()
    ds.profiles = [{"chainId": "solana", "tokenAddress": MINT_A}]
    ds.pairs[MINT_A] = [pair(POOL_1, mint=MINT_A)]
    svc = make_service(tmp_path, gt, ds, now, enrichment_provider=None)
    run(svc.discover(kinds=["new"], chains=["solana", "base", "ethereum"]))
    assert sum(r.url.path == "/token-profiles/latest/v1" for r in ds.requests) == 1


def test_concurrent_identical_requests_are_deduplicated() -> None:
    now = Now()
    gate = RequestGate("P", limits(), now.mono)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return "body"

    async def main() -> list[str]:
        return list(await asyncio.gather(*(gate.run("k", fetch) for _ in range(5))))

    assert run(main()) == ["body"] * 5
    assert calls == 1 and gate.requests_made == 1


def test_failures_are_not_cached_and_rate_limit_fails_fast() -> None:
    now = Now()
    gate = RequestGate("P", limits(calls_per_minute=2, cache_ttl_seconds=60), now.mono)

    async def boom() -> str:
        raise MarketDataUnavailableError("down")

    async def ok() -> str:
        return "fine"

    with pytest.raises(MarketDataUnavailableError, match="down"):
        run(gate.run("k", boom))
    assert run(gate.run("k", ok)) == "fine"  # the failure wasn't cached
    with pytest.raises(MarketDataUnavailableError, match="request limit"):
        run(gate.run("other", ok))
    assert run(gate.run("k", ok)) == "fine"  # cached answers cost no budget
    now.advance(seconds=61)
    assert run(gate.run("other", ok)) == "fine"


def test_slow_requests_time_out() -> None:
    gate = RequestGate("P", limits(timeout_seconds=0.01))

    async def slow() -> str:
        await asyncio.sleep(1)
        return "late"

    with pytest.raises(MarketDataUnavailableError, match="timed out"):
        run(gate.run("k", slow))


def test_concurrency_is_capped_per_provider() -> None:
    gate = RequestGate("P", limits(max_concurrency=2))
    running = peak = 0

    def fetch_factory() -> Callable[[], Any]:
        async def fetch() -> None:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

        return fetch

    async def main() -> None:
        await asyncio.gather(*(gate.run(f"k{i}", fetch_factory()) for i in range(6)))

    run(main())
    assert peak == 2


# --- Config ---------------------------------------------------------------------------------


def test_config_is_validated() -> None:
    with pytest.raises(ValidationError):
        ScoutConfig(filters={"min_liquidity_usd": -1})
    with pytest.raises(ValidationError):
        ScoutConfig(chains=("fantom",))
    with pytest.raises(ValidationError):
        ScoutConfig(features={"window_pairs": [["h1", "m5"]]})
    with pytest.raises(ValidationError):
        ScoutConfig(unknown_threshold=1)
    with pytest.raises(ScoutConfigError):
        load_scout_config("{not json")
    cfg = load_scout_config(json.dumps({"chains": ["base"], "filters": {"min_txns_h24": 5}}))
    assert cfg.chains == ("base",) and cfg.filters.min_txns_h24 == 5
    assert load_scout_config(None) == ScoutConfig()


def test_scout_is_wired_without_touching_the_disk_at_import() -> None:
    from upscale.services import scout_service

    assert {p.name for p in scout_service.providers} == {"GeckoTerminal", "DEX Screener"}
    assert scout_service.store._conn is None


# --- helpers ----------------------------------------------------------------------------------


async def _gt_pools(*items: tuple[dict[str, Any], dict[str, Any]]) -> list[Any]:
    from upscale.services.scout.providers import parse_gt_document

    pools, malformed = parse_gt_document(gt_doc(*items), "solana", "solana")
    assert malformed == 0
    return pools


def _mint(i: int) -> str:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return "Mint" + "".join(alphabet[(i * 7 + k) % len(alphabet)] for k in range(40))
