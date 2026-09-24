import asyncio
import json

import httpx2
import pytest

from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.market_data import (
    AssetNotFoundError,
    MarketDataService,
    MarketDataUnavailableError,
    RateLimiter,
)

from .conftest import FakeCoinGecko, coingecko_row


@pytest.fixture
def fake() -> FakeCoinGecko:
    return FakeCoinGecko()


def provider(fake: FakeCoinGecko, **kwargs) -> CoinGeckoProvider:
    return CoinGeckoProvider(transport=fake.transport(), **kwargs)


def fetch(fake: FakeCoinGecko, symbol: str, **kwargs):
    return asyncio.run(provider(fake, **kwargs).fetch_snapshot(symbol))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- Provider: success --------------------------------------------------------------


def test_fetch_snapshot_parses_all_fields(fake):
    snap = fetch(fake, "btc")
    assert snap.symbol == "BTC"
    assert snap.name == "Bitcoin"
    assert snap.provider == "CoinGecko"
    assert snap.provider_id == "bitcoin"
    assert snap.price_usd == 64_000.0
    assert snap.change_24h_pct == 1.5
    assert snap.change_24h_usd == pytest.approx(960.0)
    assert snap.high_24h_usd == pytest.approx(65_280.0)
    assert snap.low_24h_usd == pytest.approx(62_080.0)
    assert snap.volume_24h_usd == 32_000_000_000.0
    assert snap.market_cap_usd == 1_216_000_000_000.0
    assert snap.last_updated is not None and snap.last_updated.year == 2026

    [request] = fake.requests
    assert request.url.path == "/api/v3/coins/markets"
    assert request.url.params["vs_currency"] == "usd"
    assert request.url.params["ids"] == "bitcoin"


@pytest.mark.parametrize(("symbol", "coin_id"), [("ETH", "ethereum"), ("SOL", "solana")])
def test_fetch_snapshot_requests_the_asked_for_asset(fake, symbol, coin_id):
    snap = fetch(fake, symbol)
    assert snap.symbol == symbol
    assert fake.requests[0].url.params["ids"] == coin_id


def test_unmapped_symbol_is_looked_up_by_symbol(fake):
    fake.handler = lambda r: httpx2.Response(
        200, content=json.dumps([coingecko_row("dogwifcoin", "wif", "dogwifhat", 2.5)])
    )
    assert fetch(fake, "WIF").provider_id == "dogwifcoin"
    params = fake.requests[0].url.params
    assert params["symbols"] == "wif"
    assert params["include_tokens"] == "top"
    assert "ids" not in params


def test_missing_optional_fields_stay_none(fake):
    row = coingecko_row("bitcoin", "btc", "Bitcoin", 64_000.0, market_cap=0, high_24h=None)
    fake.handler = lambda r: httpx2.Response(200, content=json.dumps([row]))
    snap = fetch(fake, "BTC")
    assert snap.market_cap_usd is None
    assert snap.high_24h_usd is None


def test_api_key_is_sent_when_configured(fake):
    fetch(fake, "BTC", api_key="demo-key")
    assert fake.requests[0].headers["x-cg-demo-api-key"] == "demo-key"


# --- Provider: failures -------------------------------------------------------------


def _raise(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda r: httpx2.Response(500), "HTTP 500"),
        (lambda r: httpx2.Response(429), "rate limit"),
        (_raise(httpx2.ReadTimeout("slow")), "timed out"),
        (_raise(httpx2.ConnectError("down")), "could not reach"),
        (lambda r: httpx2.Response(200, content=b"<html>"), "invalid JSON"),
        (lambda r: httpx2.Response(200, content=b'{"error": "x"}'), "unexpected response"),
        (
            lambda r: httpx2.Response(
                200,
                content=json.dumps(
                    [coingecko_row("bitcoin", "btc", "Bitcoin", 1.0) | {"current_price": None}]
                ),
            ),
            "no current price",
        ),
    ],
    ids=["http-500", "http-429", "timeout", "connect-error", "bad-json", "not-a-list", "no-price"],
)
def test_api_failures_raise_unavailable(fake, handler, message):
    fake.handler = handler
    with pytest.raises(MarketDataUnavailableError, match=message):
        fetch(fake, "BTC")


def test_unknown_asset_raises_not_found(fake):
    fake.handler = lambda r: httpx2.Response(200, content=b"[]")
    with pytest.raises(AssetNotFoundError, match="NOTACOIN"):
        fetch(fake, "NOTACOIN")


def test_row_for_a_different_symbol_is_not_accepted(fake):
    fake.handler = lambda r: httpx2.Response(
        200, content=json.dumps([coingecko_row("ethereum", "eth", "Ethereum", 3000.0)])
    )
    with pytest.raises(AssetNotFoundError):
        fetch(fake, "BTC")


# --- Service: caching and rate limiting --------------------------------------------


def test_service_caches_until_ttl_expires(fake):
    clock = FakeClock()
    service = MarketDataService(provider(fake), cache_ttl=60, clock=clock)

    async def scenario():
        first = await service.get_snapshot("BTC")
        assert await service.get_snapshot("btc") is first
        assert len(fake.requests) == 1
        clock.now += 61
        await service.get_snapshot("BTC")
        assert len(fake.requests) == 2

    asyncio.run(scenario())


def test_service_deduplicates_concurrent_requests(fake):
    service = MarketDataService(provider(fake))

    async def scenario():
        return await asyncio.gather(*(service.get_snapshot("ETH") for _ in range(5)))

    results = asyncio.run(scenario())
    assert len(fake.requests) == 1
    assert all(r is results[0] for r in results)


def test_service_caches_unknown_assets_but_not_outages(fake):
    service = MarketDataService(provider(fake))
    fake.handler = lambda r: httpx2.Response(200, content=b"[]")
    for _ in range(2):
        with pytest.raises(AssetNotFoundError):
            asyncio.run(service.get_snapshot("NOTACOIN"))
    assert len(fake.requests) == 1

    fake.handler = lambda r: httpx2.Response(503)
    with pytest.raises(MarketDataUnavailableError):
        asyncio.run(service.get_snapshot("BTC"))
    fake.handler = fake.markets
    assert asyncio.run(service.get_snapshot("BTC")).price_usd == 64_000.0  # retried, not cached


def test_service_rate_limits_provider_calls(fake):
    clock = FakeClock()
    service = MarketDataService(provider(fake), max_calls_per_minute=2, clock=clock)
    asyncio.run(service.get_snapshot("BTC"))
    asyncio.run(service.get_snapshot("ETH"))
    with pytest.raises(MarketDataUnavailableError, match="limit"):
        asyncio.run(service.get_snapshot("SOL"))
    assert len(fake.requests) == 2
    assert asyncio.run(service.get_snapshot("BTC")).symbol == "BTC"  # cache still served
    clock.now += 60
    asyncio.run(service.get_snapshot("SOL"))
    assert len(fake.requests) == 3


def test_rate_limiter_window():
    clock = FakeClock()
    limiter = RateLimiter(max_calls=1, period=10, clock=clock)
    assert limiter.try_acquire()
    assert not limiter.try_acquire()
    clock.now += 10
    assert limiter.try_acquire()
