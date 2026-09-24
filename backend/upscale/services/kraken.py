"""Kraken candle provider (public spot market data REST API; no API key, no account access).

Docs: https://docs.kraken.com/api/docs/rest-api/get-ohlc-data

`GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1` returns up to the 720
most recent candles of a native interval (1, 5, 15, 30, 60, 240 or 1440 minutes, among
others) as `[open_time_s, "open", "high", "low", "close", "vwap", "volume", trades]`,
plus a `last` cursor. The final row is always the current, still-open candle, so it is
dropped. Volume is in the base asset. Errors come back as HTTP 200 with a non-empty
`error` list (e.g. "EQuery:Unknown asset pair", "EGeneral:Too many requests").
"""

import itertools
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2

from upscale.services.market_data import (
    TIMEFRAME_SECONDS,
    AssetNotFoundError,
    Candle,
    CandleSeries,
    InsufficientDataError,
    MarketDataUnavailableError,
    Timeframe,
    UnsupportedTimeframeError,
)

PUBLIC_BASE_URL = "https://api.kraken.com/0/public"

# Kraken's `interval` parameter (minutes) for each UpScale timeframe. All are native.
KRAKEN_INTERVALS: dict[Timeframe, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}
# Kraken returns at most 720 rows, the last of which is the unfinished candle.
MAX_COMPLETED_CANDLES = 719

QUOTE = "USD"
# Kraken's own codes for assets whose ticker differs from the common one.
KRAKEN_BASE_CODES: dict[str, str] = {"BTC": "XBT", "DOGE": "XDG"}
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,10}$")


@dataclass(frozen=True)
class KrakenPair:
    symbol: str  # UpScale ticker, e.g. "BTC"
    pair: str  # human-readable pair, e.g. "BTC/USD"
    request_pair: str  # what Kraken's `pair` parameter expects, e.g. "XBTUSD"


def resolve_pair(symbol: str) -> KrakenPair:
    """Map an UpScale ticker to its Kraken USD spot pair, e.g. BTC -> BTC/USD (XBTUSD).

    Any other well-formed ticker maps to `<TICKER>USD`; Kraken itself decides whether that
    pair exists (an unknown pair is reported as `AssetNotFoundError` by the provider).
    """
    symbol = symbol.upper()
    if not _SYMBOL_RE.fullmatch(symbol):
        raise AssetNotFoundError(f"{symbol!r} is not a valid ticker for Kraken")
    base = KRAKEN_BASE_CODES.get(symbol, symbol)
    return KrakenPair(symbol=symbol, pair=f"{symbol}/{QUOTE}", request_pair=f"{base}{QUOTE}")


class KrakenProvider:
    name = "Kraken"
    supported_timeframes: frozenset[Timeframe] = frozenset(KRAKEN_INTERVALS)

    def __init__(
        self,
        base_url: str = PUBLIC_BASE_URL,
        timeout: float = 8.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._transport = transport  # injectable for tests

    async def fetch_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        if timeframe not in KRAKEN_INTERVALS:
            raise UnsupportedTimeframeError(f"{self.name} does not provide {timeframe} candles")
        if limit > MAX_COMPLETED_CANDLES:
            raise InsufficientDataError(
                f"{self.name} provides at most {MAX_COMPLETED_CANDLES} {timeframe} candles"
            )
        market = resolve_pair(symbol)
        result = await self._get(
            "/OHLC",
            {"pair": market.request_pair, "interval": str(KRAKEN_INTERVALS[timeframe])},
            market,
        )
        rows = [value for key, value in result.items() if key != "last"]
        if len(rows) != 1 or not isinstance(rows[0], list):
            raise MarketDataUnavailableError(f"{self.name} returned an unexpected response")
        return CandleSeries(
            symbol=market.symbol,
            provider=self.name,
            provider_id=market.request_pair,
            pair=market.pair,
            timeframe=timeframe,
            # The last row is the candle still in progress; only completed candles are used.
            candles=self._parse_candles(rows[0][:-1], timeframe),
            volume_available=True,
            fetched_at=datetime.now(UTC),
        )

    def _parse_candles(self, rows: list[Any], timeframe: Timeframe) -> list[Candle]:
        """Validate Kraken rows into candles; duplicates, gaps or bad values reject the lot."""
        malformed = MarketDataUnavailableError(f"{self.name} returned malformed candle data")
        interval = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        candles: list[Candle] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 7:
                raise malformed
            opened_s = row[0]
            values = [_decimal(v) for v in row[1:5]]
            volume = _decimal(row[6])
            if isinstance(opened_s, bool) or not isinstance(opened_s, int):
                raise malformed
            if volume is None or volume < 0 or any(v is None or v <= 0 for v in values):
                raise malformed
            open_, high, low, close = (v for v in values if v is not None)
            if not (low <= min(open_, close) and high >= max(open_, close)):
                raise malformed
            try:
                opened = datetime.fromtimestamp(opened_s, UTC)
            except (ValueError, OverflowError, OSError) as exc:
                raise malformed from exc
            if opened.timestamp() % interval.total_seconds():
                raise MarketDataUnavailableError(
                    f"{self.name} returned a candle not aligned to {timeframe} boundaries"
                )
            candles.append(
                Candle(timestamp=opened, open=open_, high=high, low=low, close=close, volume=volume)
            )

        pairs = list(itertools.pairwise(candles))
        if any(b.timestamp <= a.timestamp for a, b in pairs):
            raise MarketDataUnavailableError(
                f"{self.name} returned duplicate or out-of-order {timeframe} candles"
            )
        for a, b in pairs:
            if b.timestamp - a.timestamp != interval:
                # Missing candles are never filled in, and other intervals are never relabeled.
                raise MarketDataUnavailableError(
                    f"{self.name} returned {timeframe} candles with a gap "
                    f"({a.timestamp:%Y-%m-%d %H:%M} to {b.timestamp:%Y-%m-%d %H:%M} UTC)"
                )
        return candles

    async def _get(self, path: str, params: dict[str, str], market: KrakenPair) -> dict[str, Any]:
        """GET a Kraken public endpoint and return its `result` object."""
        try:
            async with httpx2.AsyncClient(
                base_url=self.base_url,
                headers={"accept": "application/json"},
                timeout=self.timeout,
                transport=self._transport,
            ) as client:
                response = await client.get(path, params=params)
        except httpx2.TimeoutException as exc:
            raise MarketDataUnavailableError(f"{self.name} request timed out") from exc
        except httpx2.HTTPError as exc:
            raise MarketDataUnavailableError(f"could not reach {self.name}") from exc

        if response.status_code == 429:
            raise MarketDataUnavailableError(f"{self.name} rate limit reached")
        if response.status_code != 200:
            raise MarketDataUnavailableError(f"{self.name} returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MarketDataUnavailableError(f"{self.name} returned invalid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("error", []), list):
            raise MarketDataUnavailableError(f"{self.name} returned an unexpected response")

        errors = [str(e) for e in data.get("error", [])]
        if any("Unknown asset pair" in e for e in errors):
            raise AssetNotFoundError(
                f"{market.symbol} ({market.pair}) is not traded on {self.name}"
            )
        if any("Too many requests" in e or "Rate limit" in e for e in errors):
            raise MarketDataUnavailableError(f"{self.name} rate limit reached")
        if errors:
            raise MarketDataUnavailableError(f"{self.name} returned an error: {', '.join(errors)}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise MarketDataUnavailableError(f"{self.name} returned an unexpected response")
        return result


def _decimal(value: Any) -> float | None:
    """Kraken sends prices and volumes as decimal strings; anything else is rejected."""
    if not isinstance(value, str):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None
