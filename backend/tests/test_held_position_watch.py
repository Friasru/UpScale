"""Held-position watch (production side): what it looks up, how it identifies the exact
pool, what it archives, its priority and quota discipline, and the end-to-end path into
an EVIDENCE_AWARE_V2 Shadow run. DEX Screener answers through a mock transport behind the
real provider, gate and lane limiter; databases are temporary."""

import asyncio
import json
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

import upscale.main
from upscale.held_position_watch import (
    WATCH_LANE,
    DexScreenerLookup,
    GeckoTerminalLookup,
    HeldPositionWatch,
    WatchSettings,
    held_pools,
    load_watch_settings,
)
from upscale.services.evidence_archive import hooks as evidence_hooks
from upscale.services.evidence_archive.recorder import EvidenceRecorder
from upscale.services.evidence_archive.store import WATCH_COMPONENT
from upscale.services.quota import INTERACTIVE_LANE, LaneLimiter
from upscale.services.scout.config import ScoutConfig, ScoutProviderLimits
from upscale.services.scout.gate import RequestGate
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.scout.providers import (
    DexScreenerDiscoveryProvider,
    GeckoTerminalDiscoveryProvider,
)
from upscale.services.scout.service import REFRESH_LANE

from .test_shadow import T0, Harness, strat

SA = "Tok1" + "A" * 40  # valid Solana mints (base58, 44 characters)
SB = "Tok1" + "B" * 40
SOL = "So11111111111111111111111111111111111111112"
EVM = "0x" + "ab" * 20
EVM_POOL = "0xAbCdEf0123456789aBcDeF0123456789AbCdEf01"  # mixed case, as a provider spells it
V2, V1 = "EVIDENCE_AWARE_V2", "LEGACY_V1"


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[GrowthScoutResult, GrowthCandidate]:
    from .test_outcomes import ranked

    result, store = ranked(tmp_path_factory.mktemp("template"))
    store.close()
    return result, next(c for c in result.candidates if c.stage == "ACCELERATING")


@pytest.fixture
def h(tmp_path: Path, template: Any) -> Iterator[Harness]:
    harness = Harness(tmp_path, template)
    yield harness
    harness.close()


@pytest.fixture
def recorder(h: Harness) -> Iterator[EvidenceRecorder]:
    """The production recorder, writing into the harness archive."""
    rec = EvidenceRecorder(h.writer)
    previous = evidence_hooks.installed()
    evidence_hooks.install(rec)
    yield rec
    rec.flush()
    evidence_hooks.install(previous)


def m(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def cand(h: Harness, mint: str, provider: str = "DEX Screener", chain: str = "solana",
         **kw: Any) -> GrowthCandidate:  # fmt: skip
    g = h.cand(mint, **kw)
    g = g.model_copy(update={"market": g.market.model_copy(update={"market_provider": provider})})
    if chain != "solana":
        g = g.model_copy(update={"chain": chain, "canonical_id": f"{chain}:{mint}"})
    return g


def pair(chain: str, token: str, pool: str, price: float, liquidity: float = 80_000.0) -> Any:
    return {"chainId": chain, "dexId": "raydium", "pairAddress": pool,
            "baseToken": {"address": token, "symbol": "TOK"},
            "quoteToken": {"address": SOL, "symbol": "SOL"},
            "priceUsd": str(price), "liquidity": {"usd": liquidity}}  # fmt: skip


class FakeDex:
    """DEX Screener's /tokens/v1 endpoint over a mock transport."""

    def __init__(self) -> None:
        self.pairs: dict[str, list[Any]] = {}
        self.calls: list[str] = []
        self.fail: int | None = None  # an HTTP status to answer with

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(request.url.path)
        if self.fail is not None:
            return httpx2.Response(self.fail, json={"error": "x"})
        _, _, _, chain, addrs = request.url.path.split("/", 4)
        rows = [p for a in addrs.split(",") for p in self.pairs.get(a, [])]
        return httpx2.Response(200, json=rows)


def ds_lookup(dex: FakeDex, limiter: LaneLimiter | None = None, keep_free: int = 10) -> Any:
    config = ScoutConfig()
    # No response cache: test cycles run seconds apart (production ones 15 minutes).
    limits = config.dexscreener.model_copy(update={"cache_ttl_seconds": 0.0})
    gate = RequestGate(
        "DEX Screener", limits,
        limiter=limiter or LaneLimiter(50, 60.0, {"refresh": 10}),
    )  # fmt: skip
    provider = DexScreenerDiscoveryProvider(
        config, gate=gate, transport=httpx2.MockTransport(dex.handle)
    )
    return DexScreenerLookup(provider, keep_free)


def make(h: Harness, lookups: list[Any], defer: Any = lambda: None, **kw: Any) -> HeldPositionWatch:
    return HeldPositionWatch(
        WatchSettings(**kw), lookups, shadow_db=lambda: str(h.shadow_path),
        evidence_store=lambda: h.reader, defer=defer, now=lambda: h.now or T0,
    )  # fmt: skip


def cycle(h: Harness, w: HeldPositionWatch, rec: EvidenceRecorder, at: datetime) -> Any:
    h.now = at
    h._observed = at.timestamp()
    r = asyncio.run(w.run_once())
    assert rec.flush()
    return r


def watch_records(h: Harness) -> list[Any]:
    return [r for r in h.reader.records(kind="market") if r.component == WATCH_COMPONENT]


def held(h: Harness, *mints: str, policy: str = V2, run: str = "t", **kw: Any) -> None:
    h.scan(T0, *(cand(h, x, **kw) for x in mints))
    h.now = m(1)
    h.run(strat(exit={"max_hold_minutes": 1440.0}), run_id=run, policy=policy)  # type: ignore[arg-type]


# --- what is held ----------------------------------------------------------------------------------


def test_only_open_positions_of_active_evidence_aware_runs(h: Harness) -> None:
    held(h, SA)
    held(h, SB, run="legacy", policy=V1)  # LEGACY_V1 ignores watch records: not watched
    pools = held_pools(str(h.shadow_path), m(2))
    assert [(p.chain, p.token, p.pool) for p in pools] == [("solana", SA, f"pool-{SA}")]
    assert held_pools(str(h.shadow_path.parent / "missing.sqlite3"), m(2)) == []


def test_several_strategies_holding_one_pool_make_one_lookup(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    h.scan(T0, cand(h, SA))
    h.now = m(1)
    long = {"max_hold_minutes": 1440.0}
    h.run(strat("s1", exit=long), strat("s2", exit=long), strat("s3", exit=long), policy=V2)
    (pool,) = held_pools(str(h.shadow_path), m(2))
    assert pool.holders == 3
    dex = FakeDex()
    dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", 1.05)]
    r = cycle(h, make(h, [ds_lookup(dex)]), recorder, m(30))
    assert r.status == "completed" and len(dex.calls) == 1 and r.looked_up == 1
    (rec,) = watch_records(h)
    assert rec.pool_address == f"pool-{SA}" and rec.payload["watch"]["holders"] == 3


# --- what is archived, and how Shadow uses it --------------------------------------------------------


def test_dropped_from_discovery_but_priced_by_the_watch_end_to_end(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    held(h, SA)  # Scout never lists the token again
    dex = FakeDex()
    w = make(h, [ds_lookup(dex)])
    for i, price in enumerate([1.02, 1.01, 0.99, 0.97, 0.95, 0.94, 0.93, 0.92, 0.85], 1):
        dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", price)]
        assert cycle(h, w, recorder, m(45 * i)).status == "completed"
    assert len(watch_records(h)) == 9
    h.now = m(45 * 9 + 1)
    h.engine.run("t")
    (t,) = h.trades()
    assert t["exit_reason"] == "STOP_LOSS" and t["exit_price"] == 0.85
    assert t["exit_at"] == m(45 * 9).timestamp()
    # 6.75 hours without any Scout price: LEGACY_V1 would have written it off at 2 hours.


def test_fresh_scout_price_skips_the_lookup_and_no_duplicate_evidence(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    held(h, SA)
    dex = FakeDex()
    dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", 1.0)]
    w = make(h, [ds_lookup(dex)])
    assert cycle(h, w, recorder, m(5)).fresh == 1 and dex.calls == []  # Scout priced it
    cycle(h, w, recorder, m(30))
    assert len(dex.calls) == 1 and len(watch_records(h)) == 1
    # Again soon after, and after a restart (a new watch instance): the archived watch
    # price is fresh, so nothing is requested or archived twice.
    cycle(h, w, recorder, m(31))
    cycle(h, make(h, [ds_lookup(dex)]), recorder, m(32))
    assert len(dex.calls) == 1 and len(watch_records(h)) == 1


def test_authoritative_not_found_and_not_listed(h: Harness, recorder: EvidenceRecorder) -> None:
    h.scan(T0, cand(h, SA), cand(h, SB, provider="Some Other Provider"))
    h.now = m(1)
    h.run(strat(exit={"max_hold_minutes": 1440.0}), policy=V2)
    dex = FakeDex()
    dex.pairs[SA] = [pair("solana", SA, "a-different-pool", 1.0)]  # our pool is gone
    cycle(h, make(h, [ds_lookup(dex)]), recorder, m(30))
    got = {r.asset_id: (r.availability, r.payload["watch"]["status"],
                        r.payload["watch"]["authoritative"]) for r in watch_records(h)}  # fmt: skip
    assert got == {
        f"solana:{SA}": ("NOT_AVAILABLE", "NOT_FOUND", True),  # DEX Screener priced it
        f"solana:{SB}": ("NOT_AVAILABLE", "NOT_LISTED", False),  # someone else did
    }


def test_provider_failure_is_archived_but_own_budget_is_not(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    held(h, SA)
    dex = FakeDex()
    dex.fail = 500
    cycle(h, make(h, [ds_lookup(dex)]), recorder, m(30))
    (r,) = watch_records(h)
    assert r.availability == "PROVIDER_FAILED" and r.payload["watch"]["status"] == "PROVIDER_FAILED"
    dex.fail = 429
    cycle(h, make(h, [ds_lookup(dex)]), recorder, m(60))
    assert watch_records(h)[-1].availability == "RATE_LIMITED"


def test_solana_pool_identity_is_case_sensitive(h: Harness, recorder: EvidenceRecorder) -> None:
    held(h, SA, pool="PoolCaseSensitive1111111111111111")
    dex = FakeDex()
    dex.pairs[SA] = [pair("solana", SA, "poolcasesensitive1111111111111111", 2.0)]
    cycle(h, make(h, [ds_lookup(dex)]), recorder, m(30))
    (r,) = watch_records(h)
    assert r.payload["watch"]["status"] == "NOT_FOUND" and "candidate" not in r.payload


def test_evm_addresses_are_normalized_and_archived_as_the_position_spells_them(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    held(h, EVM, chain="base", pool=EVM_POOL)
    dex = FakeDex()
    dex.pairs[EVM] = [pair("base", EVM.upper().replace("0X", "0x"), EVM_POOL.lower(), 0.8)]
    cycle(h, make(h, [ds_lookup(dex)]), recorder, m(30))
    (r,) = watch_records(h)
    assert r.payload["watch"]["status"] == "PRICED"
    assert r.asset_id == f"base:{EVM}" and r.pool_address == EVM_POOL  # Shadow's exact key
    h.now = m(31)
    h.engine.run("t")
    (t,) = h.trades()
    assert t["exit_reason"] == "STOP_LOSS" and t["exit_price"] == 0.8


# --- priority and quota ------------------------------------------------------------------------------


def test_deferred_while_anything_outranks_it(h: Harness, recorder: EvidenceRecorder) -> None:
    held(h, SA)
    dex = FakeDex()
    w = make(h, [ds_lookup(dex)], defer=lambda: "Analyze is active")
    r = cycle(h, w, recorder, m(30))
    assert r.status == "deferred" and dex.calls == [] and watch_records(h) == []
    assert w.status()["deferrals_by_reason"] == {"Analyze is active": 1}


def test_production_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    main = upscale.main
    monkeypatch.setattr(main, "_last_analyze", -1e9)
    assert main._watch_defer_reason() is None
    monkeypatch.setattr(main, "_interactive", 1)
    assert main._watch_defer_reason() == "Analyze is active"
    monkeypatch.setattr(main, "_interactive", 0)
    monkeypatch.setattr(main.background_scout, "running", True)
    assert main._watch_defer_reason() == "a Scout scan is running"
    monkeypatch.setattr(main.background_scout, "running", False)
    monkeypatch.setattr(main.services.safety_enrichment, "running", True, raising=False)
    assert main._watch_defer_reason() == "safety enrichment is running"


def test_quota_is_respected_and_reservations_are_never_used(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    held(h, SA)
    dex = FakeDex()
    dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", 1.0)]
    limiter = LaneLimiter(50, 60.0, {REFRESH_LANE: 10, INTERACTIVE_LANE: 5})
    # Other work used the window down to exactly the watch's keep-free margin.
    for _ in range(50 - 10 - 5 - 10):
        assert limiter.try_acquire("default")
    assert limiter.available(WATCH_LANE) == 10
    w = make(h, [ds_lookup(dex, limiter)])
    r = cycle(h, w, recorder, m(30))
    assert dex.calls == [] and r.waiting == 1 and watch_records(h) == []
    # One call of headroom beyond the margin: the watch takes it, and only from unreserved
    # capacity (the refresh and interactive reservations are untouched).
    limiter.reset()
    for _ in range(50 - 10 - 5 - 11):
        assert limiter.try_acquire("default")
    cycle(h, make(h, [ds_lookup(dex, limiter)]), recorder, m(31))
    assert len(dex.calls) == 1 and limiter.used(WATCH_LANE) == 1
    assert limiter.available(REFRESH_LANE) >= 10 and limiter.available(INTERACTIVE_LANE) >= 5


def test_geckoterminal_only_for_its_own_pools_and_within_one_request(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    held(h, SA, provider="GeckoTerminal")
    dex = FakeDex()  # DEX Screener doesn't list the pool
    gt_calls: list[str] = []

    def gt_handle(request: httpx2.Request) -> httpx2.Response:
        gt_calls.append(request.url.path)
        return httpx2.Response(200, json={"data": [], "included": []})

    config = ScoutConfig()
    gt_limits = ScoutProviderLimits(
        calls_per_minute=6, reservations={"refresh": 2, "interactive": 2}
    )
    gt = GeckoTerminalDiscoveryProvider(
        config, gate=RequestGate("GeckoTerminal", gt_limits,
                                 limiter=LaneLimiter(6, 60.0, {"refresh": 2, "interactive": 2})),
        transport=httpx2.MockTransport(gt_handle),
    )  # fmt: skip
    cycle(h, make(h, [ds_lookup(dex), GeckoTerminalLookup(gt)]), recorder, m(30))
    assert len(dex.calls) == 1 and len(gt_calls) == 1
    (r,) = watch_records(h)
    assert r.provider == "GeckoTerminal" and r.payload["watch"]["status"] == "NOT_FOUND"
    assert r.payload["watch"]["authoritative"] is True


def test_settings_and_status(h: Harness) -> None:
    assert load_watch_settings(None, None) == WatchSettings()
    assert load_watch_settings("0", "1").enabled is False
    assert load_watch_settings("1", "1").interval_minutes == 5.0
    assert load_watch_settings("1", "nope").interval_minutes == 15.0
    w = make(h, [])
    assert asyncio.run(w.run_once()).status == "idle"  # nothing held
    json.dumps(w.status(), default=str)


def test_watch_makes_no_request_outside_its_providers(
    h: Harness, recorder: EvidenceRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("no real network request")

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", refuse)
    held(h, SA)
    dex = FakeDex()
    dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", 1.0)]
    assert cycle(h, make(h, [ds_lookup(dex)]), recorder, m(30)).status == "completed"
