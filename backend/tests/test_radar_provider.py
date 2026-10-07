"""Radar V1 provider guard: Radar's own budget, limiter, 429 cooldown, timeout / retry and
failure handling. Offline (MockTransport)."""

import asyncio
from datetime import timedelta
from typing import Any

import httpx2
import pytest

from tests.radar_fakes import MINT, POOL, Clock, FakeChain, make_service
from upscale.services.radar.models import (
    RadarBudgetExhaustedError,
    RadarCoolingDownError,
    RadarRateLimitedError,
    RadarTimeoutError,
    RadarUnavailableError,
)
from upscale.services.radar.provider import (
    COOLDOWN_KEY,
    RadarHeliusProvider,
    RadarRpcProvider,
    RequestGuard,
)


def _provider(tmp_path: Any, chain: FakeChain, clock: Clock, **kw: Any) -> RadarHeliusProvider:
    svc = make_service(tmp_path, chain, clock, **kw)
    assert isinstance(svc.provider, RadarHeliusProvider)
    return svc.provider


def test_429_starts_persisted_cooldown_without_retry(tmp_path: Any) -> None:
    clock, chain = Clock(), FakeChain()
    chain.fail = lambda m: 429
    p = _provider(tmp_path, chain, clock, cooldown_seconds=600, max_cooldown_seconds=1500)
    with pytest.raises(RadarRateLimitedError):
        asyncio.run(p.get_signatures(POOL, limit=10))
    assert chain.calls["getSignaturesForAddress"] == 1  # never retried through a 429
    # Fails fast during the cooldown: no HTTP request at all.
    with pytest.raises(RadarCoolingDownError):
        asyncio.run(p.fetch_mint(MINT))
    assert sum(chain.calls.values()) == 1
    # Persisted: a fresh guard on the same database refuses too.
    fresh = _provider(tmp_path, chain, clock)
    with pytest.raises(RadarCoolingDownError):
        asyncio.run(fresh.fetch_mint(MINT))
    # After the cooldown a second 429 doubles it (capped).
    clock.advance(601)
    with pytest.raises(RadarRateLimitedError):
        asyncio.run(p.fetch_mint(MINT))
    until = p.guard.cooldown_until()
    assert until is not None and until - clock.now() == timedelta(seconds=1200)
    clock.advance(1201)
    with pytest.raises(RadarRateLimitedError):
        asyncio.run(p.fetch_mint(MINT))
    until = p.guard.cooldown_until()
    assert until is not None and until - clock.now() == timedelta(seconds=1500)
    # Success resets the streak.
    clock.advance(1501)
    chain.fail = None
    asyncio.run(p.fetch_mint(MINT))
    assert p.guard.ledger.get_meta("provider.consecutive_429") == "0"
    rows = {r[1]: r for r in p.guard.ledger.requests_by_day("2000-01-01")}  # type: ignore[attr-defined]
    assert rows["getAccountInfo"][4] == 2  # rate_limited column


def test_rate_limit_in_jsonrpc_body_is_a_429(tmp_path: Any) -> None:
    clock = Clock()

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": 1,
                                          "error": {"code": -32429, "message": "rate limited"}})  # fmt: skip

    svc = make_service(tmp_path, FakeChain(), clock)
    assert svc.provider is not None
    svc.provider._transport = httpx2.MockTransport(handler)
    with pytest.raises(RadarRateLimitedError):
        asyncio.run(svc.provider.get_transaction("sig"))
    assert svc.repo.get_meta(COOLDOWN_KEY) is not None


def test_timeouts_retry_with_backoff_then_provider_unavailable(tmp_path: Any) -> None:
    clock, chain = Clock(), FakeChain()
    chain.fail = lambda m: "timeout"
    p = _provider(tmp_path, chain, clock, max_retries=2, backoff_base_seconds=1.0, max_rps=50)
    with pytest.raises(RadarTimeoutError) as err:
        asyncio.run(p.get_transaction("sig"))
    assert err.value.status == "PROVIDER_UNAVAILABLE"
    assert chain.calls["getTransaction"] == 3
    backoffs = [s for s in clock.sleeps if s >= 1.0]
    assert backoffs == [1.0, 2.0]  # exponential (jitter fixed at 0 in tests)
    assert p.guard.used_today() == 3  # every attempt counts against the budget
    assert p.guard.cooldown_until() is None  # timeouts never start a cooldown


def test_5xx_retries_then_succeeds_and_4xx_does_not_retry(tmp_path: Any) -> None:
    clock, chain = Clock(), FakeChain()
    seen: list[str] = []

    def flaky(method: str) -> int | None:
        seen.append(method)
        return 503 if len(seen) == 1 else None

    chain.fail = flaky
    p = _provider(tmp_path, chain, clock)
    assert asyncio.run(p.get_signatures(POOL, limit=5)) == []
    assert chain.calls["getSignaturesForAddress"] == 2
    chain.fail = lambda m: 401
    with pytest.raises(RadarUnavailableError, match="rejected the API key"):
        asyncio.run(p.get_signatures(POOL, limit=5))
    assert chain.calls["getSignaturesForAddress"] == 3


def test_budget_blocks_before_any_request(tmp_path: Any) -> None:
    clock, chain = Clock(), FakeChain()
    p = _provider(tmp_path, chain, clock, daily_request_budget=2)
    asyncio.run(p.get_signatures(POOL, limit=5))
    asyncio.run(p.get_signatures(POOL, limit=5))
    with pytest.raises(RadarBudgetExhaustedError) as err:
        asyncio.run(p.get_signatures(POOL, limit=5))
    assert err.value.status == "NOT_COLLECTED"
    assert chain.calls["getSignaturesForAddress"] == 2
    clock.advance(86400)  # a new UTC day: a new budget
    asyncio.run(p.get_signatures(POOL, limit=5))
    assert chain.calls["getSignaturesForAddress"] == 3


def test_limiter_spaces_requests_by_max_rps(tmp_path: Any) -> None:
    clock, chain = Clock(), FakeChain()
    p = _provider(tmp_path, chain, clock, max_rps=0.5)
    for _ in range(3):
        asyncio.run(p.get_signatures(POOL, limit=5))
    assert clock.sleeps == [2.0, 2.0]  # 1 / max_rps between starts


def test_concurrency_bound(tmp_path: Any) -> None:
    clock = Clock()
    active = peak = 0

    async def attempt() -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return 1

    async def run() -> None:
        svc = make_service(tmp_path, FakeChain(), clock, concurrency=2, max_rps=50)
        guard = svc.guard
        await asyncio.gather(*(guard.run("m", attempt) for _ in range(6)))

    asyncio.run(run())
    assert peak <= 2


def test_key_never_in_errors_and_plain_rpc_has_no_holder_scan(tmp_path: Any) -> None:
    clock, chain = Clock(), FakeChain()
    chain.fail = lambda m: 500
    p = _provider(tmp_path, chain, clock, max_retries=0)
    with pytest.raises(RadarUnavailableError) as err:
        asyncio.run(p.fetch_mint(MINT))
    assert "test-key" not in str(err.value) and "test-key" not in repr(p)
    svc = make_service(tmp_path, None, clock)
    guard = RequestGuard(svc.settings, svc.repo, now=clock.now)
    rpc = RadarRpcProvider(guard, "https://rpc.example/?api-key=secret", chain.transport())
    assert asyncio.run(rpc.scan_token_accounts(MINT)) is None
