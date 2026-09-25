import asyncio
import base64
import json
import math
import struct
import zlib
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any
from xml.sax.saxutils import escape

import httpx2
import pytest
from fastapi.testclient import TestClient

import upscale.agents.education
from upscale.main import app
from upscale.services import market_data_service, news_service, solana_dex_service, vision_service
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.dexscreener import DexScreenerProvider
from upscale.services.kraken import KrakenProvider
from upscale.services.news_sentiment_model import ArticleAssessment, ArticleInput
from upscale.services.rss_news import DEFAULT_FEEDS, RssNewsProvider
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


class FakeDexScreener:
    """Stands in for api.dexscreener.com. `pairs` maps a mint to the pair objects its
    /token-pairs endpoint returns (unknown mints get an empty list, as the real API does)."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.pairs: dict[str, list[Any]] = {}
        self.handler: Callable[[httpx2.Request], httpx2.Response] = self.token_pairs

    def token_pairs(self, request: httpx2.Request) -> httpx2.Response:
        mint = request.url.path.rsplit("/", 1)[-1]
        return httpx2.Response(200, content=json.dumps(self.pairs.get(mint, [])))

    def transport(self) -> httpx2.MockTransport:
        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.handler(request)

        return httpx2.MockTransport(handle)


# Wall-clock "now" for DEX pool ages in tests.
DEX_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def fake_dexscreener() -> Iterator[FakeDexScreener]:
    """Point the app's shared Solana DEX service at a fake DEX Screener."""
    fake = FakeDexScreener()
    original = (solana_dex_service.provider, solana_dex_service.now)
    solana_dex_service.provider = DexScreenerProvider(transport=fake.transport())
    solana_dex_service.now = lambda: DEX_NOW
    solana_dex_service.reset()
    yield fake
    solana_dex_service.provider, solana_dex_service.now = original
    solana_dex_service.reset()


# Open time (s) of the in-progress candle in every fake Kraken response: 2026-09-24T00:00Z,
# a boundary of every Kraken interval UpScale uses (1m through 1d, all UTC-aligned).
KRAKEN_NOW_S = 1_790_208_000
KRAKEN_PAIRS = {"XBTUSD": "XXBTZUSD", "ETHUSD": "XETHZUSD", "SOLUSD": "SOLUSD"}
KRAKEN_START_PRICES = {"XBTUSD": 64_000.0, "ETHUSD": 3_100.0, "SOLUSD": 150.0}


def kraken_rows(
    count: int, interval_s: int, start_price: float = 100.0, end_open_s: int = KRAKEN_NOW_S
) -> list[list]:
    """Kraken-style `[time, open, high, low, close, vwap, volume, trades]` rows, oldest first.

    The last row opens at `end_open_s` and plays the in-progress candle. Prices oscillate
    so indicators and levels exist; volumes vary and are real base-asset amounts.
    """
    rows: list[list] = []
    previous = start_price
    for i in range(count):
        close = start_price * (1 + 0.05 * math.sin(2 * math.pi * i / 20))
        open_ = previous
        high, low = max(open_, close) * 1.001, min(open_, close) * 0.999
        volume = 10.0 + (i % 7) * 1.5
        rows.append(
            [
                end_open_s - (count - 1 - i) * interval_s,
                f"{open_:.2f}",
                f"{high:.2f}",
                f"{low:.2f}",
                f"{close:.2f}",
                f"{(high + low) / 2:.2f}",
                f"{volume:.8f}",
                40 + i % 5,
            ]
        )
        previous = close
    return rows


def kraken_body(rows: list[list], key: str = "XXBTZUSD") -> bytes:
    return json.dumps(
        {"error": [], "result": {key: rows, "last": rows[-1][0] if rows else 0}}
    ).encode()


def kraken_error(*errors: str) -> httpx2.Response:
    return httpx2.Response(200, content=json.dumps({"error": list(errors)}).encode())


class FakeKraken:
    """Stands in for api.kraken.com/0/public/OHLC (720 rows per request, like Kraken)."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.handler: Callable[[httpx2.Request], httpx2.Response] = self.ohlc

    def ohlc(self, request: httpx2.Request) -> httpx2.Response:
        pair = request.url.params["pair"]
        if pair not in KRAKEN_PAIRS:
            return kraken_error("EQuery:Unknown asset pair")
        interval_s = int(request.url.params["interval"]) * 60
        rows = kraken_rows(720, interval_s, KRAKEN_START_PRICES[pair])
        return httpx2.Response(200, content=kraken_body(rows, KRAKEN_PAIRS[pair]))

    def transport(self) -> httpx2.MockTransport:
        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.handler(request)

        return httpx2.MockTransport(handle)


@pytest.fixture
def fake_kraken(fake_coingecko: FakeCoinGecko) -> Iterator[FakeKraken]:
    """Make the app's shared service prefer a fake Kraken for candles, as in production."""
    fake = FakeKraken()
    coingecko = market_data_service.provider
    assert isinstance(coingecko, CoinGeckoProvider)
    market_data_service.candle_providers = [KrakenProvider(transport=fake.transport()), coingecko]
    market_data_service.reset()
    yield fake


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


def chart_transcript_data(**overrides: Any) -> dict[str, Any]:
    """The flat transport form of `chart_reading_data()`, as Claude returns it."""
    data: dict[str, Any] = {
        "is_price_chart": True,
        "chart_type": "candlestick",
        "symbol": "BTC",
        "pair": "BTC/USDT",
        "exchange": "Binance",
        "asset_inferred": False,
        "asset_evidence": "Symbol header top-left",
        "timeframe": "4h",
        "timeframe_inferred": False,
        "timeframe_evidence": "Interval button in toolbar",
        "price": 64210.5,
        "price_inferred": False,
        "price_evidence": "Last price tag",
        "indicators": [
            {
                "name": "RSI",
                "settings": "14",
                "values": [{"label": "", "value": 58.2}],
                "inferred": False,
                "evidence": "Lower pane legend",
            },
            {
                "name": "EMA",
                "settings": "",
                "values": [],
                "inferred": True,
                "evidence": "A smooth line that might be a moving average",
            },
        ],
        "levels": [
            {
                "kind": "support",
                "price": 62000.0,
                "label": "",
                "source": "drawn_line",
                "inferred": False,
                "evidence": "Green horizontal line",
            },
            {
                "kind": "support",
                "price": 0,
                "label": "support?",
                "source": "price_structure",
                "inferred": True,
                "evidence": "Cluster of lows",
            },
            {
                "kind": "resistance",
                "price": 66000.0,
                "label": "66k",
                "source": "price_label",
                "inferred": False,
                "evidence": "Red line labeled 66k",
            },
            {
                "kind": "user_drawn",
                "price": 63000.0,
                "label": "entry zone",
                "source": "drawn_line",
                "inferred": False,
                "evidence": "Dashed line",
            },
        ],
        "lines": [
            {
                "kind": "channel",
                "direction": "rising",
                "description": "Parallel rising channel drawn from the recent lows",
                "inferred": False,
                "evidence": "Two parallel drawn lines",
            }
        ],
        "patterns": [
            {"name": "ascending triangle", "clear": True, "evidence": "Flat top, rising lows"},
            {"name": "double top", "clear": False, "evidence": "Two similar highs"},
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


# --- News -------------------------------------------------------------------------------------


def news_item(
    title: str | None,
    link: str | None,
    hours_ago: float | None,
    description: str | None = None,
    categories: Sequence[str] = (),
    pub_date: str | None = None,
) -> dict[str, Any]:
    """One RSS item. `pub_date` overrides the date computed from `hours_ago`."""
    if pub_date is None and hours_ago is not None:
        pub_date = format_datetime(datetime.now(UTC) - timedelta(hours=hours_ago))
    return {
        "title": title,
        "link": link,
        "pubDate": pub_date,
        "description": description,
        "categories": list(categories),
    }


def rss_feed(items: Sequence[dict[str, Any]]) -> bytes:
    parts = ['<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>']
    for item in items:
        parts.append("<item>")
        for tag in ("title", "link", "pubDate", "description"):
            if item.get(tag) is not None:
                parts.append(f"<{tag}>{escape(item[tag])}</{tag}>")
        parts += [f"<category>{escape(c)}</category>" for c in item["categories"]]
        parts.append("</item>")
    parts.append("</channel></rss>")
    return "".join(parts).encode()


FEED_URLS = {feed.name: feed.url for feed in DEFAULT_FEEDS}


def default_news() -> dict[str, list[dict[str, Any]]]:
    """A realistic snapshot of the default feeds, with times relative to now."""
    return {
        "CoinDesk": [
            news_item(
                "Bitcoin ETF inflows hit $1B as institutions add exposure",
                "https://www.coindesk.com/markets/2026/09/24/bitcoin-etf-inflows-hit-1b",
                3,
                "Spot bitcoin ETFs recorded their largest daily inflow in months.",
                ["Markets", "Bitcoin"],
            ),
            news_item(
                "Ethereum developers set date for next network upgrade",
                "https://www.coindesk.com/tech/2026/09/24/ethereum-upgrade-date",
                5,
                "Core developers agreed on a mainnet activation date.",
                ["Tech", "Ethereum"],
            ),
            news_item(
                "SEC delays decision on crypto custody rules",
                "https://www.coindesk.com/policy/2026/09/24/sec-delays-custody-rules",
                10,
                "The regulator pushed back its timeline by 45 days.",
                ["Policy"],
            ),
            news_item(
                "Crypto market liquidations top $500M in 24 hours",
                "https://www.coindesk.com/markets/2026/09/24/liquidations-500m",
                4,
                "Leveraged positions were wiped out across major exchanges.",
                ["Markets"],
            ),
            news_item(
                "Bitcoin hits record high",
                "https://www.coindesk.com/markets/2026/09/14/bitcoin-record-high",
                24 * 10,
                "An old story that should be ignored.",
                ["Bitcoin"],
            ),
        ],
        "Decrypt": [
            news_item(
                "Bitcoin ETF inflows hit $1B as institutions add exposure",
                "https://decrypt.co/379300/bitcoin-etf-inflows-1b?utm_source=rss",
                2,
                '<p style="float:right"><img src="https://img/x.jpg"></p>'
                "<p>Spot bitcoin ETFs saw inflows &amp; rising volume.</p>",
                ["Markets"],
            ),
            news_item(
                "Bitcoin miners face pressure as hashprice drops to yearly low",
                "https://decrypt.co/379200/bitcoin-miners-hashprice-low",
                8,
                "Miner revenue per unit of hashrate fell to its lowest level this year.",
            ),
            news_item(
                "Solana network suffers brief outage",
                "https://decrypt.co/379100/solana-network-outage",
                20,
                "Block production halted for about an hour before validators restarted.",
            ),
            news_item(
                "Meta launches AI keychain gadget",
                "https://decrypt.co/379253/meta-ai-keychain",
                1,
                "A palm-sized device for talking to an AI assistant.",
                ["Artificial Intelligence"],
            ),
            news_item(
                "Bitcoin whale moves 10,000 BTC to an exchange",
                "https://decrypt.co/379001/bitcoin-whale-exchange",
                60,
                "On-chain data shows a large transfer from a long-dormant wallet.",
                ["Bitcoin"],
            ),
        ],
    }


class FakeNewsFeeds:
    """Stands in for the default feeds' RSS endpoints. Set `items`, or `responses[source]` to
    an `httpx2.Response` / exception to make one feed fail."""

    def __init__(self) -> None:
        self.items = default_news()
        self.responses: dict[str, httpx2.Response | Exception] = {}
        self.requests: list[httpx2.Request] = []

    def all_urls(self) -> set[str]:
        return {item["link"] for items in self.items.values() for item in items}

    def all_titles(self) -> set[str]:
        return {item["title"] for items in self.items.values() for item in items}

    def fail_all(self, response: httpx2.Response | Exception) -> None:
        for source in FEED_URLS:
            self.responses[source] = response

    def transport(self) -> httpx2.MockTransport:
        by_url = {url: name for name, url in FEED_URLS.items()}

        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            source = by_url.get(str(request.url))
            if source is None:
                return httpx2.Response(404)
            override = self.responses.get(source)
            if isinstance(override, Exception):
                raise override
            if override is not None:
                return override
            return httpx2.Response(200, content=rss_feed(self.items.get(source, [])))

        return httpx2.MockTransport(handle)


# Title keyword -> (sentiment, impact, reason) used by FakeSentimentModel.
DEFAULT_LABELS: dict[str, tuple[str, str, str]] = {
    "inflows": ("bullish", "high", "Large ETF inflows may be relevant to demand for BTC."),
    "hashprice": ("bearish", "medium", "Miner stress may be relevant to BTC supply."),
    "whale": ("bearish", "low", "A large exchange deposit may be relevant to supply."),
    "liquidations": ("bearish", "medium", "Liquidations may be relevant to market leverage."),
    "sec delays": ("neutral", "medium", "The delay may be relevant to custody regulation."),
    "upgrade": ("bullish", "medium", "The upgrade may be relevant to Ethereum's roadmap."),
    "outage": ("bearish", "high", "The outage may be relevant to Solana's reliability."),
}


class FakeSentimentModel:
    """Stands in for the Claude sentiment call. Labels by title keyword; set `labels`,
    `error`, `delay` or `respond` per test."""

    name = "fake-sentiment"

    def __init__(self) -> None:
        self.labels = dict(DEFAULT_LABELS)
        self.error: Exception | None = None
        self.delay = 0.0
        self.respond: Callable[[str, list[ArticleInput]], list[ArticleAssessment]] | None = None
        self.calls: list[tuple[str, list[ArticleInput]]] = []

    async def classify(
        self, subject: str, articles: Sequence[ArticleInput]
    ) -> list[ArticleAssessment]:
        self.calls.append((subject, list(articles)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        if self.respond:
            return self.respond(subject, list(articles))
        out = []
        for a in articles:
            sentiment, impact, reason = next(
                (v for k, v in self.labels.items() if k in a.title.lower()),
                ("neutral", "low", "Routine coverage that may be of limited relevance."),
            )
            out.append(
                ArticleAssessment(id=a.id, sentiment=sentiment, impact=impact, reason=reason)
            )
        return out


@pytest.fixture(autouse=True)
def fake_news() -> Iterator[FakeNewsFeeds]:
    """Point the app's shared news service at fake feeds. No test reads real feeds."""
    fake = FakeNewsFeeds()
    original = news_service.provider
    news_service.provider = RssNewsProvider(transport=fake.transport())
    news_service.reset()
    yield fake
    news_service.provider = original
    news_service.reset()


@pytest.fixture(autouse=True)
def fake_sentiment() -> Iterator[FakeSentimentModel]:
    """Point the app's shared news service at a fake sentiment model. No test calls Claude."""
    fake = FakeSentimentModel()
    original = news_service.model
    news_service.model = fake
    yield fake
    news_service.model = original


# --- Education ---------------------------------------------------------------------------


class FakeExplainerModel:
    """Stands in for the Claude explanation call. Set `answer` or `error` per test."""

    name = "fake-explainer"

    def __init__(self) -> None:
        self.answer = "A concise explanation of the concept."
        self.error: Exception | None = None
        self.calls: list[str] = []

    async def explain(self, question: str) -> str:
        self.calls.append(question)
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture(autouse=True)
def fake_explainer(monkeypatch: pytest.MonkeyPatch) -> FakeExplainerModel:
    """Point the education agent's shared model at a fake. No test calls Claude."""
    fake = FakeExplainerModel()
    monkeypatch.setattr(upscale.agents.education, "explainer_model", fake)
    return fake
