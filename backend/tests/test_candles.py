import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx2
import pytest

from upscale.services import market_data_service
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.market_data import (
    MAX_CANDLES,
    TIMEFRAMES,
    AssetNotFoundError,
    Candle,
    CandleSeries,
    InsufficientDataError,
    InvalidRequestError,
    MarketDataService,
    MarketDataUnavailableError,
    Timeframe,
    UnsupportedTimeframeError,
)

from .conftest import FOUR_HOURS_MS, LAST_CLOSE_MS, FakeCoinGecko, coingecko_row, ohlc_rows

LAST_OPEN = datetime.fromtimestamp(LAST_CLOSE_MS / 1000, UTC) - timedelta(hours=4)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fake() -> FakeCoinGecko:
    return FakeCoinGecko()


def make_service(fake: FakeCoinGecko, **kwargs) -> MarketDataService:
    return MarketDataService(CoinGeckoProvider(transport=fake.transport()), **kwargs)


def candles(service: MarketDataService, symbol="ETH", timeframe="4h", **kwargs) -> CandleSeries:
    return asyncio.run(service.get_candles(symbol, timeframe, **kwargs))


def respond(fake: FakeCoinGecko, status=200, body=None):
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    fake.handler = lambda request: httpx2.Response(status, content=content)


# --- Parsing ------------------------------------------------------------------------


def test_candles_are_parsed_with_open_time_timestamps(fake):
    series = candles(make_service(fake), limit=3)
    assert series.symbol == "ETH"
    assert series.provider == "CoinGecko"
    assert series.provider_id == "ethereum"
    assert series.timeframe == "4h"
    assert series.interval == timedelta(hours=4)
    assert len(series.candles) == 3
    # CoinGecko's timestamp is the close time; the model stores the open time, in UTC, oldest first.
    assert [c.timestamp for c in series.candles] == [
        LAST_OPEN - timedelta(hours=8),
        LAST_OPEN - timedelta(hours=4),
        LAST_OPEN,
    ]
    assert all(c.timestamp.tzinfo == UTC for c in series.candles)
    assert LAST_OPEN == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_candle_ohlcv_values(fake):
    respond(fake, body=[[LAST_CLOSE_MS, 100.5, 110.25, 95.0, 105.75]])
    [candle] = candles(make_service(fake), limit=1).candles
    assert candle == Candle(
        timestamp=LAST_OPEN, open=100.5, high=110.25, low=95.0, close=105.75, volume=None
    )


def test_coingecko_candles_have_no_volume(fake):
    series = candles(make_service(fake), limit=5)
    assert series.volume_available is False
    assert all(c.volume is None for c in series.candles)


def test_duplicate_rows_are_collapsed_and_sorted(fake):
    rows = ohlc_rows(3)
    respond(fake, body=[rows[2], rows[0], rows[1], rows[2]])
    series = candles(make_service(fake), limit=3)
    assert [c.open for c in series.candles] == [100, 101, 102]


@pytest.mark.parametrize(
    ("limit", "days"), [(1, "7"), (42, "7"), (43, "14"), (84, "14"), (180, "30")]
)
def test_smallest_sufficient_range_is_requested(fake, limit, days):
    series = candles(make_service(fake), limit=limit)
    [request] = fake.requests
    assert request.url.path == "/api/v3/coins/ethereum/ohlc"
    assert request.url.params["days"] == days
    assert request.url.params["vs_currency"] == "usd"
    assert len(series.candles) == limit


def test_unmapped_symbol_resolves_coin_id_first(fake):
    def handler(request):
        if request.url.path.endswith("/coins/markets"):
            return httpx2.Response(
                200, content=json.dumps([coingecko_row("dogwifcoin", "wif", "dogwifhat", 2.0)])
            )
        return httpx2.Response(200, content=json.dumps(ohlc_rows(42)))

    fake.handler = handler
    provider = CoinGeckoProvider(transport=fake.transport())
    series = asyncio.run(provider.fetch_candles("WIF", "4h", 10))
    assert series.provider_id == "dogwifcoin"
    assert fake.requests[1].url.path == "/api/v3/coins/dogwifcoin/ohlc"
    asyncio.run(provider.fetch_candles("WIF", "4h", 10))
    assert len(fake.requests) == 3  # coin id lookup is remembered


# --- Timeframes and limits ----------------------------------------------------------


def test_coingecko_supports_only_4h():
    assert CoinGeckoProvider.supported_timeframes == {"4h"}
    assert market_data_service.supported_timeframes == ["4h"]


@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "1h", "1d"])
def test_unsupported_timeframes_are_rejected_without_a_request(fake, timeframe):
    with pytest.raises(UnsupportedTimeframeError, match="available timeframes: 4h"):
        candles(make_service(fake), timeframe=timeframe)
    assert fake.requests == []


def test_unknown_timeframe_is_rejected(fake):
    with pytest.raises(UnsupportedTimeframeError, match="unknown timeframe '2h'"):
        candles(make_service(fake), timeframe="2h")


def test_provider_rejects_unsupported_timeframe_directly(fake):
    provider = CoinGeckoProvider(transport=fake.transport())
    with pytest.raises(UnsupportedTimeframeError):
        asyncio.run(provider.fetch_candles("BTC", "1h", 10))
    assert fake.requests == []


@pytest.mark.parametrize("limit", [0, -1, MAX_CANDLES + 1])
def test_limit_bounds(fake, limit):
    with pytest.raises(InvalidRequestError):
        candles(make_service(fake), limit=limit)
    assert fake.requests == []


def test_min_candles_must_not_exceed_limit(fake):
    with pytest.raises(InvalidRequestError):
        candles(make_service(fake), limit=10, min_candles=11)


def test_more_history_than_coingecko_offers_is_rejected_up_front(fake):
    with pytest.raises(InsufficientDataError, match="at most 180 4h candles"):
        candles(make_service(fake), limit=181)
    assert fake.requests == []


# --- Insufficient data ----------------------------------------------------------------


def test_insufficient_history_raises(fake):
    respond(fake, body=ohlc_rows(10))  # e.g. a newly listed coin
    with pytest.raises(InsufficientDataError, match="only 10 4h candle"):
        candles(make_service(fake), limit=50)


def test_min_candles_accepts_partial_history(fake):
    respond(fake, body=ohlc_rows(10))
    series = candles(make_service(fake), limit=50, min_candles=5)
    assert len(series.candles) == 10


def test_empty_history_raises_insufficient(fake):
    respond(fake, body=[])
    with pytest.raises(InsufficientDataError, match="only 0"):
        candles(make_service(fake), limit=1)


# --- Failures -------------------------------------------------------------------------


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
    ],
    ids=["http-500", "http-429", "timeout", "connect-error"],
)
def test_api_failures(fake, handler, message):
    fake.handler = handler
    with pytest.raises(MarketDataUnavailableError, match=message):
        candles(make_service(fake), limit=5)


@pytest.mark.parametrize(
    "body",
    [
        b"<html>",
        {"error": "x"},
        [[LAST_CLOSE_MS, 1, 2, 0.5]],  # missing close
        [[LAST_CLOSE_MS, "1", 2, 0.5, 1]],  # string value
        [[LAST_CLOSE_MS, 1, 2, 0.5, None]],
        [[LAST_CLOSE_MS, 1, 0.9, 0.5, 1]],  # high below open
        [[LAST_CLOSE_MS, 1, 2, 1.5, 1.2]],  # low above open
        [[LAST_CLOSE_MS, 0, 0, 0, 0]],  # non-positive prices
        [{"t": LAST_CLOSE_MS}],
        [[1e30, 1, 2, 0.5, 1]],  # timestamp out of range
    ],
    ids=[
        "not-json",
        "not-a-list",
        "short-row",
        "string",
        "null",
        "high-too-low",
        "low-too-high",
        "zero-price",
        "object-row",
        "bad-timestamp",
    ],
)
def test_malformed_responses(fake, body):
    respond(fake, body=body)
    with pytest.raises(MarketDataUnavailableError, match="malformed|invalid JSON|unexpected"):
        candles(make_service(fake), limit=1)


def test_wrongly_spaced_candles_are_not_relabeled(fake):
    respond(fake, body=ohlc_rows(48, step_ms=FOUR_HOURS_MS // 8))  # 30m candles
    with pytest.raises(MarketDataUnavailableError, match="not spaced 4h apart"):
        candles(make_service(fake), limit=10)


def test_unknown_asset(fake):
    service = make_service(fake)
    for _ in range(2):
        with pytest.raises(AssetNotFoundError, match="NOTACOIN"):
            candles(service, symbol="NOTACOIN", limit=5)
    assert len(fake.requests) == 1  # symbol lookup via /coins/markets, then remembered


def test_unknown_coin_id_404(fake):
    provider = CoinGeckoProvider(transport=fake.transport())
    provider._resolved_ids["GONE"] = "delisted-coin"
    with pytest.raises(AssetNotFoundError, match="GONE"):
        asyncio.run(provider.fetch_candles("GONE", "4h", 5))
    assert fake.requests[0].url.path == "/api/v3/coins/delisted-coin/ohlc"


# --- Caching and rate limiting -------------------------------------------------------


def test_candles_are_cached(fake):
    clock = FakeClock()
    service = make_service(fake, clock=clock)
    first = candles(service, limit=40)
    assert candles(service, limit=40) == first
    assert len(candles(service, limit=10).candles) == 10  # served from the larger cached fetch
    assert len(fake.requests) == 1

    candles(service, limit=80)  # needs more history than was fetched
    assert len(fake.requests) == 2

    clock.now += 301  # 4h candles are cached for 5 minutes
    candles(service, limit=10)
    assert len(fake.requests) == 3


def test_concurrent_candle_requests_share_one_call(fake):
    service = make_service(fake)

    async def scenario():
        return await asyncio.gather(*(service.get_candles("BTC", "4h", 20) for _ in range(5)))

    asyncio.run(scenario())
    assert len(fake.requests) == 1


def test_candle_and_price_requests_share_the_rate_limit(fake):
    clock = FakeClock()
    service = make_service(fake, max_calls_per_minute=2, clock=clock)
    asyncio.run(service.get_snapshot("BTC"))
    candles(service, symbol="BTC", limit=5)
    with pytest.raises(MarketDataUnavailableError, match="limit was reached"):
        candles(service, symbol="ETH", limit=5)
    assert len(fake.requests) == 2
    assert len(candles(service, symbol="BTC", limit=5).candles) == 5  # cache still served
    clock.now += 60
    candles(service, symbol="ETH", limit=5)
    assert len(fake.requests) == 3


def test_failures_are_not_cached(fake):
    service = make_service(fake)
    respond(fake, status=503)
    with pytest.raises(MarketDataUnavailableError):
        candles(service, limit=5)
    fake.handler = fake.markets
    assert len(candles(service, limit=5).candles) == 5


# --- Multiple providers ---------------------------------------------------------------


class HourlyProvider:
    """Stand-in for a future provider that offers timeframes CoinGecko can't."""

    name = "Hourly"
    supported_timeframes: frozenset[Timeframe] = frozenset({"1h"})

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    async def fetch_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        self.calls.append((symbol, timeframe, limit))
        start = datetime(2026, 9, 24, tzinfo=UTC)
        return CandleSeries(
            symbol=symbol,
            provider=self.name,
            provider_id=symbol.lower(),
            timeframe=timeframe,
            candles=[
                Candle(
                    timestamp=start + timedelta(hours=i),
                    open=1,
                    high=2,
                    low=0.5,
                    close=1.5,
                    volume=10,
                )
                for i in range(limit)
            ],
            volume_available=True,
            fetched_at=start,
        )


def test_each_timeframe_is_served_by_a_provider_that_supports_it(fake):
    coingecko = CoinGeckoProvider(transport=fake.transport())
    hourly = HourlyProvider()
    service = MarketDataService(coingecko, candle_providers=[coingecko, hourly])
    assert service.supported_timeframes == ["1h", "4h"]

    assert candles(service, symbol="BTC", timeframe="1h", limit=3).provider == "Hourly"
    assert candles(service, symbol="BTC", timeframe="4h", limit=3).provider == "CoinGecko"
    assert hourly.calls == [("BTC", "1h", 3)]
    assert len(fake.requests) == 1
    with pytest.raises(UnsupportedTimeframeError, match="available timeframes: 1h, 4h"):
        candles(service, timeframe="1d")


def test_service_without_candle_providers():
    class PriceOnly:
        name = "PriceOnly"

        async def fetch_snapshot(self, symbol):  # pragma: no cover - not called
            raise AssertionError

    service = MarketDataService(PriceOnly())
    assert service.supported_timeframes == []
    for timeframe in TIMEFRAMES:
        with pytest.raises(UnsupportedTimeframeError, match="available timeframes: none"):
            candles(service, timeframe=timeframe)
