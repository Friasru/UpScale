"""GeckoTerminal provider: OHLCV candles for one exact DEX pool (free public API, no key).

`GET https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool}/ohlcv/{period}`
with `aggregate`, `limit` (max 1000), `currency=usd` and `token=base` returns
`data.attributes.ohlcv_list`: `[timestamp_s, open, high, low, close, volume_usd]` rows,
newest first. Supported: 1m, 5m, 15m (minute), 1h, 4h (hour), 1d (day); there is no 30m.
The public API allows roughly 30 requests per minute.

Pools only have candles for intervals with trades, so a quiet pool has gaps. Missing
candles are never filled in: only the latest run of consecutive closed candles is used,
and the trim is recorded in the series' notes. The in-progress candle is dropped.
"""

import asyncio
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2

from upscale.services.chains import GECKOTERMINAL_NETWORKS, chain_label
from upscale.services.market_data import (
    TIMEFRAME_SECONDS,
    AssetNotFoundError,
    Candle,
    CandleSeries,
    MarketDataUnavailableError,
    ProviderRateLimitedError,
    Timeframe,
    UnsupportedTimeframeError,
)
from upscale.services.quota import INTERACTIVE_LANE, LaneLimiter

PUBLIC_BASE_URL = "https://api.geckoterminal.com/api/v2"
GT_TIMEFRAMES: dict[Timeframe, tuple[str, int]] = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
    "4h": ("hour", 4),
    "1d": ("day", 1),
}
MAX_LIMIT = 1000


class GeckoTerminalProvider:
    name = "GeckoTerminal"
    supported_timeframes: frozenset[Timeframe] = frozenset(GT_TIMEFRAMES)

    def __init__(
        self,
        base_url: str = PUBLIC_BASE_URL,
        timeout: float = 8.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._transport = transport

    async def fetch_pool_candles(
        self,
        chain: str,
        pool: str,
        timeframe: Timeframe,
        limit: int,
        *,
        symbol: str,
        canonical_id: str | None,
        now: datetime,
        before: datetime | None = None,
        contiguous: bool = True,
    ) -> CandleSeries:
        """`before`: only candles opened before it (the API's `before_timestamp`), for
        history older than the latest `limit`. `contiguous=False` keeps every closed
        candle instead of only the latest consecutive run (gaps stay gaps, never filled)."""
        network = GECKOTERMINAL_NETWORKS.get(chain)
        if network is None:
            raise UnsupportedTimeframeError(
                f"{self.name} has no pool candles for {chain_label(chain)}"
            )
        if timeframe not in GT_TIMEFRAMES:
            raise UnsupportedTimeframeError(f"{self.name} does not provide {timeframe} candles")
        period, aggregate = GT_TIMEFRAMES[timeframe]
        params = {
            "aggregate": str(aggregate),
            "limit": str(min(limit + 1, MAX_LIMIT)),
            "currency": "usd",
            "token": "base",
        }
        if before is not None:
            params["before_timestamp"] = str(math.ceil(before.timestamp()))
        body = await self._get(f"/networks/{network}/pools/{pool}/ohlcv/{period}", params)
        rows = _ohlcv_rows(body)
        if rows is None:
            raise MarketDataUnavailableError(f"{self.name} returned malformed candle data")
        candles, notes = parse_pool_candles(rows, timeframe, now, self.name, contiguous)
        return CandleSeries(
            symbol=symbol,
            provider=self.name,
            provider_id=pool,
            pair=f"{chain_label(chain)} pool {pool}",
            timeframe=timeframe,
            candles=candles[-limit:],
            volume_available=True,
            fetched_at=now,
            canonical_id=canonical_id,
            volume_unit="USD",
            as_of=now,
            notes=notes,
        )

    async def _get(self, path: str, params: dict[str, str]) -> Any:
        try:
            async with httpx2.AsyncClient(
                base_url=self.base_url,
                headers={"accept": "application/json;version=20230302"},
                timeout=self.timeout,
                transport=self._transport,
            ) as client:
                response = await client.get(path, params=params)
        except httpx2.TimeoutException as exc:
            raise MarketDataUnavailableError(f"{self.name} request timed out") from exc
        except httpx2.HTTPError as exc:
            raise MarketDataUnavailableError(f"could not reach {self.name}") from exc
        if response.status_code == 404:
            raise AssetNotFoundError(f"{self.name} doesn't know this pool")
        if response.status_code == 429:
            raise ProviderRateLimitedError(f"{self.name} rate limit reached")
        if response.status_code != 200:
            raise MarketDataUnavailableError(f"{self.name} returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise MarketDataUnavailableError(f"{self.name} returned invalid JSON") from exc


def _ohlcv_rows(body: Any) -> list[Any] | None:
    data = body.get("data") if isinstance(body, dict) else None
    attributes = data.get("attributes") if isinstance(data, dict) else None
    rows = attributes.get("ohlcv_list") if isinstance(attributes, dict) else None
    return rows if isinstance(rows, list) else None


def _number(value: Any) -> float | None:
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


def parse_pool_candles(
    rows: list[Any], timeframe: Timeframe, now: datetime, provider: str, contiguous: bool = True
) -> tuple[list[Candle], list[str]]:
    """Closed, consecutive candles (oldest first) and notes on anything left out.
    `contiguous=False`: every closed candle (intervals without trades stay missing)."""
    malformed = MarketDataUnavailableError(f"{provider} returned malformed candle data")
    interval = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    by_time: dict[datetime, Candle] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            raise malformed
        values = [_number(v) for v in row[:6]]
        if any(v is None for v in values):
            raise malformed
        ts, open_, high, low, close, volume = (v for v in values if v is not None)
        if min(open_, high, low, close) <= 0 or volume < 0:
            raise malformed
        if not (low <= min(open_, close) and high >= max(open_, close)):
            raise malformed
        try:
            opened = datetime.fromtimestamp(ts, UTC)
        except (ValueError, OverflowError, OSError) as exc:
            raise malformed from exc
        if opened.timestamp() % interval.total_seconds():
            raise MarketDataUnavailableError(
                f"{provider} returned a candle not aligned to {timeframe} boundaries"
            )
        by_time[opened] = Candle(
            timestamp=opened, open=open_, high=high, low=low, close=close, volume=volume
        )

    closed = [by_time[t] for t in sorted(by_time) if t + interval <= now]
    notes: list[str] = []
    if not contiguous:
        return closed, notes
    start = len(closed)
    while start > 0 and (
        start == len(closed) or closed[start].timestamp - closed[start - 1].timestamp == interval
    ):
        start -= 1
    if start > 0:
        notes.append(
            f"{provider} has no {timeframe} candles for intervals without trades before "
            f"{closed[start].timestamp:%Y-%m-%d %H:%M} UTC; only the latest "
            f"{len(closed) - start} consecutive candles were used (gaps are never filled in)"
        )
    return closed[start:], notes


class DexCandleService:
    """Caches pool candles, rate-limits, de-duplicates concurrent requests; failures are
    never cached.

    `limiter`: the provider's quota shared with every other UpScale consumer of it (e.g.
    Scout's discovery and refresh); requests count against `lane` ("interactive": a user
    is waiting, so they may use the capacity held for interactive work). Without one, the
    service keeps its own limit of `max_calls_per_minute`."""

    def __init__(
        self,
        provider: GeckoTerminalProvider,
        max_cache_ttl: float = 60.0,
        max_calls_per_minute: int = 20,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        limiter: LaneLimiter | None = None,
        lane: str = INTERACTIVE_LANE,
    ):
        self.provider = provider
        self.max_cache_ttl = max_cache_ttl
        self._clock = clock
        self.now = now
        self._max_calls = max_calls_per_minute
        self._shared = limiter is not None
        self.limiter = limiter or LaneLimiter(max_calls_per_minute, 60.0, clock=clock)
        self.lane = lane
        self._cache: dict[tuple[str, str, str, int], tuple[float, CandleSeries]] = {}
        self._locks: dict[tuple[str, str, str, int], asyncio.Lock] = {}

    @property
    def supported_timeframes(self) -> frozenset[Timeframe]:
        return self.provider.supported_timeframes

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def covers(self, chain: str) -> bool:
        return chain in GECKOTERMINAL_NETWORKS

    def reset(self) -> None:
        self._cache.clear()
        self._locks.clear()
        if self._shared:
            self.limiter.reset()
        else:
            self.limiter = LaneLimiter(self._max_calls, 60.0, clock=self._clock)

    async def get_candles(
        self,
        chain: str,
        pool: str,
        timeframe: Timeframe,
        limit: int,
        *,
        symbol: str,
        canonical_id: str | None,
    ) -> CandleSeries:
        key = (chain, pool, timeframe, limit)
        if (hit := self._cached(key)) is not None:
            return hit.model_copy(update={"symbol": symbol})
        async with self._locks.setdefault(key, asyncio.Lock()):
            if (hit := self._cached(key)) is not None:
                return hit.model_copy(update={"symbol": symbol})
            if not self.limiter.try_acquire(self.lane):
                raise ProviderRateLimitedError(
                    f"UpScale's {self.provider.name} request limit was reached; try again "
                    "in about a minute"
                )
            try:
                series = await self.provider.fetch_pool_candles(
                    chain,
                    pool,
                    timeframe,
                    limit,
                    symbol=symbol,
                    canonical_id=canonical_id,
                    now=self.now(),
                )
            except ProviderRateLimitedError:
                self.limiter.note_rate_limited()  # the provider pushed back: shared
                raise
            ttl = min(self.max_cache_ttl, TIMEFRAME_SECONDS[timeframe] * 0.25)
            self._cache[key] = (self._clock() + ttl, series)
            return series

    async def get_window(
        self,
        chain: str,
        pool: str,
        timeframe: Timeframe,
        limit: int,
        *,
        before: datetime,
        lane: str,
        canonical_id: str | None,
    ) -> CandleSeries:
        """Every closed candle of one exact pool opened before `before` (the latest `limit`),
        gaps kept as gaps, counted against `lane` of the shared quota (background work
        passes its own lane, so it can never use the capacity held for Analyze)."""
        key = (chain, pool, f"{timeframe}<{math.ceil(before.timestamp())}", limit)
        if (hit := self._cached(key)) is not None:
            return hit
        async with self._locks.setdefault(key, asyncio.Lock()):
            if (hit := self._cached(key)) is not None:
                return hit
            if not self.limiter.try_acquire(lane):
                raise ProviderRateLimitedError(
                    f"UpScale's {self.provider.name} request limit was reached for {lane} work"
                )
            try:
                series = await self.provider.fetch_pool_candles(
                    chain,
                    pool,
                    timeframe,
                    limit,
                    symbol=pool,
                    canonical_id=canonical_id,
                    now=self.now(),
                    before=before,
                    contiguous=False,
                )
            except ProviderRateLimitedError:
                self.limiter.note_rate_limited()  # the provider pushed back: shared
                raise
            ttl = min(self.max_cache_ttl, TIMEFRAME_SECONDS[timeframe] * 0.25)
            self._cache[key] = (self._clock() + ttl, series)
            return series

    def _cached(self, key: tuple[str, str, str, int]) -> CandleSeries | None:
        entry = self._cache.get(key)
        return entry[1] if entry and entry[0] > self._clock() else None
