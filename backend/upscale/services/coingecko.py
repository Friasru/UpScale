"""CoinGecko market data provider (public REST API, read-only, no account access).

Docs: https://docs.coingecko.com/reference/coins-markets
      https://docs.coingecko.com/reference/coins-id-ohlc

Candle limits of the public/demo API (verified against the live API): `/coins/{id}/ohlc`
only accepts days = 1, 7, 14, 30, 90, 180, 365, picks the candle size itself (1 day ->
30m, 7-30 days -> 4h, 90+ days -> 4 days), has no volume, and rejects the paid-plan
`interval` parameter. Of UpScale's timeframes that leaves only 4h, capped at 30 days.
"""

import itertools
import math
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
    MarketSnapshot,
    Timeframe,
    UnsupportedTimeframeError,
)

PUBLIC_BASE_URL = "https://api.coingecko.com/api/v3"

# CoinGecko coin ids for the tickers UpScale's router recognizes. Symbols aren't unique
# on CoinGecko, so known ids are used when possible; other symbols fall back to the
# largest coin by market cap with that symbol.
COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "BNB": "binancecoin",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "LTC": "litecoin",
    "TRX": "tron",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
    "LINK": "chainlink",
    "DOT": "polkadot",
    "TON": "the-open-network",
    "OP": "optimism",
    "ARB": "arbitrum",
    "SUI": "sui",
    "NEAR": "near",
    "APT": "aptos",
}


# Timeframe -> the `days` values whose native candle size is that timeframe, smallest first.
OHLC_DAYS: dict[Timeframe, tuple[int, ...]] = {"4h": (7, 14, 30)}


class CoinGeckoProvider:
    name = "CoinGecko"
    supported_timeframes: frozenset[Timeframe] = frozenset(OHLC_DAYS)

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = PUBLIC_BASE_URL,
        timeout: float = 8.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {"accept": "application/json"}
        if api_key:
            self.headers["x-cg-demo-api-key"] = api_key
        self._transport = transport  # injectable for tests
        self._resolved_ids: dict[str, str] = {}

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        symbol = symbol.upper()
        params = {"vs_currency": "usd", "price_change_percentage": "24h"}
        if coin_id := COINGECKO_IDS.get(symbol):
            params["ids"] = coin_id
        else:
            params.update(symbols=symbol.lower(), include_tokens="top")

        rows = await self._get("/coins/markets", params)
        row = next(
            (r for r in rows if isinstance(r, dict) and str(r.get("symbol", "")).upper() == symbol),
            None,
        )
        if row is None:
            raise AssetNotFoundError(f"{symbol} is not recognized by {self.name}")
        return self._parse(symbol, row)

    async def fetch_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        symbol = symbol.upper()
        if timeframe not in OHLC_DAYS:
            raise UnsupportedTimeframeError(f"{self.name} does not provide {timeframe} candles")
        per_day = 86_400 // TIMEFRAME_SECONDS[timeframe]
        days = next((d for d in OHLC_DAYS[timeframe] if d * per_day >= limit), None)
        if days is None:
            most = OHLC_DAYS[timeframe][-1]
            raise InsufficientDataError(
                f"{self.name} provides at most {most * per_day} {timeframe} candles ({most} days)"
            )

        coin_id = await self._coin_id(symbol)
        rows = await self._get(
            f"/coins/{coin_id}/ohlc",
            {"vs_currency": "usd", "days": str(days)},
            missing=f"{symbol} is not recognized by {self.name}",
        )
        return CandleSeries(
            symbol=symbol,
            provider=self.name,
            provider_id=coin_id,
            timeframe=timeframe,
            candles=self._parse_candles(rows, timeframe),
            volume_available=False,
            fetched_at=datetime.now(UTC),
        )

    async def _coin_id(self, symbol: str) -> str:
        if coin_id := COINGECKO_IDS.get(symbol) or self._resolved_ids.get(symbol):
            return coin_id
        coin_id = (await self.fetch_snapshot(symbol)).provider_id
        if not coin_id:
            raise MarketDataUnavailableError(f"{self.name} returned no id for {symbol}")
        self._resolved_ids[symbol] = coin_id
        return coin_id

    def _parse_candles(self, rows: list[Any], timeframe: Timeframe) -> list[Candle]:
        """Validate CoinGecko `[close_time_ms, o, h, l, c]` rows into open-time candles."""
        malformed = MarketDataUnavailableError(f"{self.name} returned malformed candle data")
        interval = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        by_time: dict[datetime, Candle] = {}
        for row in rows:
            if not isinstance(row, list) or len(row) != 5:
                raise malformed
            values = [_number(v) for v in row]
            if any(v is None or not math.isfinite(v) for v in values):
                raise malformed
            ts_ms, open_, high, low, close = (v for v in values if v is not None)
            if not (low <= min(open_, close) and high >= max(open_, close)):
                raise malformed
            try:
                # CoinGecko timestamps mark the candle's close; UpScale uses the open time.
                opened = datetime.fromtimestamp(ts_ms / 1000, UTC) - interval
                by_time[opened] = Candle(
                    timestamp=opened, open=open_, high=high, low=low, close=close
                )
            except (ValueError, OverflowError) as exc:
                raise malformed from exc

        candles = [by_time[t] for t in sorted(by_time)]
        # Refuse to relabel data: every gap must equal the requested timeframe.
        if any(b.timestamp - a.timestamp != interval for a, b in itertools.pairwise(candles)):
            raise MarketDataUnavailableError(
                f"{self.name} returned candles that are not spaced {timeframe} apart"
            )
        return candles

    async def _get(
        self, path: str, params: dict[str, str], missing: str | None = None
    ) -> list[Any]:
        """GET a JSON list. A 404 raises `AssetNotFoundError(missing)` when `missing` is set."""
        try:
            async with httpx2.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                transport=self._transport,
            ) as client:
                response = await client.get(path, params=params)
        except httpx2.TimeoutException as exc:
            raise MarketDataUnavailableError(f"{self.name} request timed out") from exc
        except httpx2.HTTPError as exc:
            raise MarketDataUnavailableError(f"could not reach {self.name}") from exc

        if response.status_code == 404 and missing:
            raise AssetNotFoundError(missing)
        if response.status_code == 429:
            raise MarketDataUnavailableError(f"{self.name} rate limit reached")
        if response.status_code != 200:
            raise MarketDataUnavailableError(f"{self.name} returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise MarketDataUnavailableError(f"{self.name} returned invalid JSON") from exc
        if not isinstance(data, list):
            raise MarketDataUnavailableError(f"{self.name} returned an unexpected response")
        return data

    def _parse(self, symbol: str, row: dict[str, Any]) -> MarketSnapshot:
        price = _number(row.get("current_price"))
        if price is None:
            raise MarketDataUnavailableError(f"{self.name} has no current price for {symbol}")
        return MarketSnapshot(
            symbol=symbol,
            name=str(row.get("name") or symbol),
            provider=self.name,
            provider_id=str(row.get("id", "")),
            price_usd=price,
            change_24h_usd=_number(row.get("price_change_24h")),
            change_24h_pct=_number(row.get("price_change_percentage_24h")),
            high_24h_usd=_number(row.get("high_24h")),
            low_24h_usd=_number(row.get("low_24h")),
            volume_24h_usd=_number(row.get("total_volume")),
            market_cap_usd=_number(row.get("market_cap")) or None,  # 0 means "unknown"
            last_updated=_timestamp(row.get("last_updated")),
            fetched_at=datetime.now(UTC),
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
