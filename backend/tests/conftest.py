import asyncio
import base64
import json
import struct
import zlib
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from upscale.main import app
from upscale.services import market_data_service, vision_service
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.vision import ChartReading, CheckedImage


def make_png(width: int, height: int) -> bytes:
    """A valid solid-white RGB PNG of the given size."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


CHART_PNG = base64.b64encode(make_png(320, 200)).decode()

PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()


def coingecko_row(coin_id: str, symbol: str, name: str, price: float, **overrides: Any) -> dict:
    row = {
        "id": coin_id,
        "symbol": symbol.lower(),
        "name": name,
        "current_price": price,
        "market_cap": price * 19_000_000,
        "total_volume": price * 500_000,
        "high_24h": price * 1.02,
        "low_24h": price * 0.97,
        "price_change_24h": price * 0.015,
        "price_change_percentage_24h": 1.5,
        "last_updated": "2026-09-24T10:00:00.000Z",
    }
    return row | overrides


COINS = {
    "bitcoin": coingecko_row("bitcoin", "btc", "Bitcoin", 64_000.0),
    "ethereum": coingecko_row("ethereum", "eth", "Ethereum", 3_100.0),
    "solana": coingecko_row("solana", "sol", "Solana", 150.0),
}


FOUR_HOURS_MS = 4 * 60 * 60 * 1000
LAST_CLOSE_MS = 1_790_265_600_000  # 2026-09-24T16:00:00Z, a 4h boundary


def ohlc_rows(count: int, step_ms: int = FOUR_HOURS_MS, start_price: float = 100.0) -> list[list]:
    """CoinGecko-style `[close_time_ms, open, high, low, close]` rows, oldest first."""
    rows = []
    for i in range(count):
        o = start_price + i
        rows.append([LAST_CLOSE_MS - (count - 1 - i) * step_ms, o, o + 2, o - 1, o + 1])
    return rows


class FakeCoinGecko:
    """Stands in for api.coingecko.com. Records requests; `handler` can be replaced per test."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.handler: Callable[[httpx2.Request], httpx2.Response] = self.markets

    def markets(self, request: httpx2.Request) -> httpx2.Response:
        """Serves /coins/markets and /coins/{id}/ohlc (6 candles per day, like CoinGecko's 4h)."""
        path = request.url.path
        if path.endswith("/ohlc"):
            coin_id = path.split("/")[-2]
            if coin_id not in COINS:
                return httpx2.Response(404, content=b'{"error":"coin not found"}')
            days = int(request.url.params["days"])
            return httpx2.Response(200, content=json.dumps(ohlc_rows(days * 6)))
        ids = request.url.params.get("ids", "")
        rows = [COINS[i] for i in ids.split(",") if i in COINS]
        return httpx2.Response(200, content=json.dumps(rows))

    def transport(self) -> httpx2.MockTransport:
        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.handler(request)

        return httpx2.MockTransport(handle)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything tries a real HTTP request (MockTransport is unaffected)."""

    async def blocked(self: object, request: httpx2.Request) -> httpx2.Response:
        raise RuntimeError(f"real network access in tests: {request.url}")

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", blocked)


@pytest.fixture(autouse=True)
def fake_coingecko() -> Iterator[FakeCoinGecko]:
    """Point the app's shared market data service (prices and candles) at a fake API."""
    fake = FakeCoinGecko()
    original = (market_data_service.provider, market_data_service.candle_providers)
    provider = CoinGeckoProvider(transport=fake.transport())
    market_data_service.provider = provider
    market_data_service.candle_providers = [provider]
    market_data_service.reset()
    yield fake
    market_data_service.provider, market_data_service.candle_providers = original
    market_data_service.reset()


def chart_reading_data(**overrides: Any) -> dict[str, Any]:
    """What a vision model might return for a BTC/USDT 4h TradingView screenshot."""
    data: dict[str, Any] = {
        "is_price_chart": True,
        "asset": {
            "symbol": "BTC",
            "pair": "BTC/USDT",
            "exchange": "Binance",
            "basis": "visible",
            "evidence": "Symbol header top-left",
        },
        "timeframe": {"label": "4h", "basis": "visible", "evidence": "Interval button in toolbar"},
        "displayed_price": {"value": 64210.5, "basis": "visible", "evidence": "Last price tag"},
        "chart_type": "candlestick",
        "indicators": [
            {
                "name": "RSI",
                "settings": "14",
                "values": [{"label": None, "value": 58.2}],
                "basis": "visible",
                "evidence": "Lower pane legend",
            },
            {
                "name": "EMA",
                "settings": None,
                "values": [],
                "basis": "inferred",
                "evidence": "A smooth line that might be a moving average",
            },
        ],
        "support_levels": [
            {
                "price": 62000.0,
                "label": None,
                "source": "drawn_line",
                "basis": "visible",
                "evidence": "Green horizontal line",
            },
            {
                "price": None,
                "label": "support?",
                "source": "price_structure",
                "basis": "inferred",
                "evidence": "Cluster of lows",
            },
        ],
        "resistance_levels": [
            {
                "price": 66000.0,
                "label": "66k",
                "source": "price_label",
                "basis": "visible",
                "evidence": "Red line labeled 66k",
            }
        ],
        "trend_lines": [
            {
                "kind": "channel",
                "direction": "rising",
                "description": "Parallel rising channel drawn from the recent lows",
                "basis": "visible",
                "evidence": "Two parallel drawn lines",
            }
        ],
        "drawn_levels": [{"price": 63000.0, "label": "entry zone", "evidence": "Dashed line"}],
        "patterns": [
            {"name": "ascending triangle", "clarity": "clear", "evidence": "Flat top, rising lows"},
            {"name": "double top", "clarity": "tentative", "evidence": "Two similar highs"},
        ],
        "observations": ["Last three candles have long upper wicks."],
        "uncertainties": ["Volume pane is cropped."],
    }
    return data | overrides


class FakeVisionModel:
    """Stands in for the Claude vision call. Set `reading`, `error` or `delay` per test."""

    name = "fake-vision"

    def __init__(self) -> None:
        self.reading: ChartReading = ChartReading.model_validate(chart_reading_data())
        self.error: Exception | None = None
        self.delay = 0.0
        self.calls: list[CheckedImage] = []

    async def read_chart(self, image: CheckedImage) -> ChartReading:
        self.calls.append(image)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.reading


@pytest.fixture(autouse=True)
def fake_vision() -> Iterator[FakeVisionModel]:
    """Point the app's shared vision service at a fake model. No test calls Claude."""
    fake = FakeVisionModel()
    original = vision_service.model
    vision_service.model = fake
    vision_service.reset()
    yield fake
    vision_service.model = original
    vision_service.reset()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def png_attachment() -> dict[str, str]:
    return {"name": "chart.png", "media_type": "image/png", "data": CHART_PNG}
