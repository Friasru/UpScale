"""DEX Screener provider: every pool listed for a token, by chain and exact token address.

Uses the official public API (https://docs.dexscreener.com/api/reference), no key needed:

* `GET https://api.dexscreener.com/token-pairs/v1/{chainId}/{tokenAddress}` returns a JSON
  list of pair objects (the token may be either side).
* `GET https://api.dexscreener.com/latest/dex/search?q={query}` returns
  `{"pairs": [...]}` matching a ticker, name or address on any chain (used to discover
  which token a ticker means, never to pick one silently).

Both are rate-limited by DEX Screener to 300 requests per minute. Prices arrive as strings, everything else as numbers; any field may
be missing, and a missing field is left as None here, never filled in.
"""

import math
from datetime import UTC, datetime
from typing import Any

import httpx2

from upscale.services.market_data import MarketDataUnavailableError
from upscale.services.solana_dex import WINDOWS, DexPool, TokenRef, WindowStats

PUBLIC_BASE_URL = "https://api.dexscreener.com"


class DexScreenerProvider:
    name = "DEX Screener"

    def __init__(
        self,
        base_url: str = PUBLIC_BASE_URL,
        timeout: float = 8.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._transport = transport  # injectable for tests

    async def fetch_token_pools(self, chain: str, token_address: str) -> list[DexPool]:
        rows = await self._get(f"/token-pairs/v1/{chain}/{token_address}")
        pools = [pool for row in rows if (pool := parse_pair(row)) is not None]
        if rows and not pools:
            raise MarketDataUnavailableError(f"{self.name} returned malformed pair data")
        return pools

    async def search_pools(self, query: str) -> list[DexPool]:
        body = await self._request("/latest/dex/search", {"q": query})
        rows = body.get("pairs") if isinstance(body, dict) else None
        if rows is None:
            return []
        if not isinstance(rows, list):
            raise MarketDataUnavailableError(f"{self.name} returned an unexpected response")
        return [pool for row in rows if (pool := parse_pair(row)) is not None]

    async def _get(self, path: str) -> list[Any]:
        data = await self._request(path)
        if not isinstance(data, list):
            raise MarketDataUnavailableError(f"{self.name} returned an unexpected response")
        return data

    async def _request(self, path: str, params: dict[str, str] | None = None) -> Any:
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
        if response.status_code == 404:
            return []  # unknown token: no pools
        if response.status_code != 200:
            raise MarketDataUnavailableError(f"{self.name} returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise MarketDataUnavailableError(f"{self.name} returned invalid JSON") from exc


def parse_pair(row: Any) -> DexPool | None:
    """One DEX Screener pair object, or None if it lacks what identifies a pool."""
    if not isinstance(row, dict):
        return None
    base, quote = _token(row.get("baseToken")), _token(row.get("quoteToken"))
    chain, dex, pair = row.get("chainId"), row.get("dexId"), row.get("pairAddress")
    if not (isinstance(chain, str) and isinstance(dex, str) and isinstance(pair, str)):
        return None
    if base is None or quote is None:
        return None
    liquidity = _dict(row.get("liquidity"))
    labels = row.get("labels")
    return DexPool(
        chain=chain,
        dex=dex,
        pair_address=pair,
        url=row["url"] if isinstance(row.get("url"), str) else None,
        labels=[str(label) for label in labels] if isinstance(labels, list) else [],
        base=base,
        quote=quote,
        price_usd=_positive(_number(row.get("priceUsd"))),
        price_native=_positive(_number(row.get("priceNative"))),
        liquidity_usd=_non_negative(_number(liquidity.get("usd"))),
        liquidity_base=_non_negative(_number(liquidity.get("base"))),
        liquidity_quote=_non_negative(_number(liquidity.get("quote"))),
        market_cap_usd=_positive(_number(row.get("marketCap"))),
        fdv_usd=_positive(_number(row.get("fdv"))),
        pair_created_at=_millis(row.get("pairCreatedAt")),
        windows=_windows(row),
    )


def _token(value: Any) -> TokenRef | None:
    if not isinstance(value, dict) or not isinstance(value.get("address"), str):
        return None
    symbol, name = value.get("symbol"), value.get("name")
    return TokenRef(
        address=value["address"],
        symbol=symbol if isinstance(symbol, str) and symbol else None,
        name=name if isinstance(name, str) and name else None,
    )


def _windows(row: dict[str, Any]) -> list[WindowStats]:
    txns, volume = _dict(row.get("txns")), _dict(row.get("volume"))
    change = _dict(row.get("priceChange"))
    out: list[WindowStats] = []
    for w in WINDOWS:
        counts = _dict(txns.get(w))
        stats = WindowStats(
            window=w,
            buys=_count(counts.get("buys")),
            sells=_count(counts.get("sells")),
            volume_usd=_non_negative(_number(volume.get(w))),
            price_change_pct=_number(change.get(w)),
        )
        if any(
            v is not None
            for v in (stats.buys, stats.sells, stats.volume_usd, stats.price_change_pct)
        ):
            out.append(stats)
    return out


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    """A finite number from a JSON number or numeric string; anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _positive(value: float | None) -> float | None:
    return value if value is not None and value > 0 else None


def _non_negative(value: float | None) -> float | None:
    return value if value is not None and value >= 0 else None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _millis(value: Any) -> datetime | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number / 1000, UTC)
    except (ValueError, OverflowError, OSError):
        return None
