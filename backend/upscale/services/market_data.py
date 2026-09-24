"""Provider-agnostic market data: models, errors, provider interfaces, and a caching service.

Agents depend on `MarketDataService`. Providers are plain objects implementing
`MarketDataProvider` (current prices) and/or `CandleProvider` (historical OHLCV); the
service can combine several candle providers so each timeframe comes from one that
actually offers it, trying them in order of preference. Providers never synthesize,
resample or relabel candles they don't have.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Protocol, TypeVar, get_args, runtime_checkable

from pydantic import BaseModel, Field

T = TypeVar("T")

Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
TIMEFRAMES: tuple[Timeframe, ...] = get_args(Timeframe)
TIMEFRAME_SECONDS: dict[Timeframe, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}
DEFAULT_CANDLES = 100
MAX_CANDLES = 500  # hard cap per request, whatever the provider could return
# Candles are cached for this fraction of their timeframe (capped by the service), so a
# new 1m candle is picked up within ~15 s and a new 5m candle within ~75 s.
CANDLE_CACHE_FRACTION = 0.25


class MarketSnapshot(BaseModel):
    """Current market data for one asset, quoted in USD. Missing values stay None."""

    symbol: str
    name: str
    provider: str
    provider_id: str
    price_usd: float
    change_24h_usd: float | None = None
    change_24h_pct: float | None = None
    high_24h_usd: float | None = None
    low_24h_usd: float | None = None
    volume_24h_usd: float | None = None
    market_cap_usd: float | None = None
    last_updated: datetime | None = None  # as reported by the provider
    fetched_at: datetime  # when UpScale retrieved it


class Candle(BaseModel):
    """One OHLCV candle in USD. `timestamp` is the candle's open time (UTC)."""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    # Base-asset volume traded during the candle; None when the provider doesn't supply it.
    volume: float | None = Field(default=None, ge=0)


class CandleSeries(BaseModel):
    """Consecutive completed candles, oldest first."""

    symbol: str
    provider: str
    provider_id: str  # the provider's own id for the market, e.g. "bitcoin" or "XBTUSD"
    pair: str | None = None  # traded pair when the provider quotes one, e.g. "BTC/USD"
    timeframe: Timeframe
    candles: list[Candle]
    volume_available: bool
    fetched_at: datetime
    # Why preferred providers were skipped before this one served the candles, if any.
    fallback_notes: list[str] = Field(default_factory=list)

    @property
    def interval(self) -> timedelta:
        return timedelta(seconds=TIMEFRAME_SECONDS[self.timeframe])


class MarketDataError(Exception):
    """Live market data could not be retrieved."""


class AssetNotFoundError(MarketDataError):
    """The provider does not recognize the requested asset."""


class MarketDataUnavailableError(MarketDataError):
    """The provider failed, timed out, rate-limited us, or returned unusable data."""


class UnsupportedTimeframeError(MarketDataError):
    """No configured provider offers candles for the requested timeframe."""


class InsufficientDataError(MarketDataError):
    """Fewer candles exist (or can be fetched) than the caller requires."""


class InvalidRequestError(MarketDataError):
    """The request itself is out of bounds (e.g. too many candles)."""


class MarketDataProvider(Protocol):
    name: str

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return a fresh snapshot or raise a `MarketDataError`."""
        ...


@runtime_checkable
class CandleProvider(Protocol):
    name: str
    supported_timeframes: frozenset[Timeframe]

    async def fetch_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        """Return at least the `limit` most recent completed candles if they exist.

        May return more (the service trims), or fewer when the asset lacks history.
        Raises `InsufficientDataError` up front if `limit` exceeds what the provider
        can ever return for this timeframe, and never fabricates or resamples candles.
        """
        ...


class RateLimiter:
    """Sliding-window limit on outgoing requests to one provider (shared across assets)."""

    def __init__(self, max_calls: int, period: float, clock: Callable[[], float] = time.monotonic):
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._calls: deque[float] = deque()

    def try_acquire(self) -> bool:
        now = self._clock()
        while self._calls and now - self._calls[0] >= self.period:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            return False
        self._calls.append(now)
        return True


@dataclass
class _CandleCacheEntry:
    expires_at: float
    requested: int  # how many candles the provider was asked for
    series: CandleSeries


class MarketDataService:
    """Caches results, rate-limits provider calls, and de-duplicates concurrent requests.

    Candle providers are listed in order of preference: a timeframe is served by the first
    provider that offers it, and a later provider offering the same timeframe is only tried
    when an earlier one fails. Only real provider data is ever returned; if every provider
    fails a `MarketDataError` is raised.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        candle_providers: Sequence[CandleProvider] | None = None,
        cache_ttl: float = 60.0,
        max_candle_cache_ttl: float = 300.0,
        not_found_ttl: float = 600.0,
        max_calls_per_minute: int = 20,
        provider_calls_per_minute: Mapping[str, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.provider = provider
        # By default the snapshot provider also serves candles if it can.
        if candle_providers is None:
            candle_providers = [provider] if isinstance(provider, CandleProvider) else []
        self.candle_providers = list(candle_providers)
        self.cache_ttl = cache_ttl
        self.max_candle_cache_ttl = max_candle_cache_ttl
        self.not_found_ttl = not_found_ttl
        self._clock = clock
        # Each provider has its own request budget; unlisted providers get the default.
        self.max_calls_per_minute = max_calls_per_minute
        self.provider_calls_per_minute = dict(provider_calls_per_minute or {})
        self._limiters: dict[str, RateLimiter] = {}
        self._snapshots: dict[str, tuple[float, MarketSnapshot]] = {}
        # (provider, symbol, timeframe) -> entry; the entry records how many candles were
        # fetched, so a request for up to that many is served from it.
        self._candles: dict[tuple[str, str, Timeframe], _CandleCacheEntry] = {}
        self._not_found: dict[tuple[str, str], float] = {}  # (provider, symbol) -> expiry
        self._locks: dict[object, asyncio.Lock] = {}

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def supported_timeframes(self) -> list[Timeframe]:
        offered = {tf for p in self.candle_providers for tf in p.supported_timeframes}
        return [tf for tf in TIMEFRAMES if tf in offered]

    def reset(self) -> None:
        """Forget cached data and rate-limit history (for tests and provider swaps)."""
        self._snapshots.clear()
        self._candles.clear()
        self._not_found.clear()
        self._locks.clear()
        self._limiters.clear()

    def candle_cache_ttl(self, timeframe: Timeframe) -> float:
        return min(self.max_candle_cache_ttl, TIMEFRAME_SECONDS[timeframe] * CANDLE_CACHE_FRACTION)

    def _limiter(self, provider_name: str) -> RateLimiter:
        if provider_name not in self._limiters:
            calls = self.provider_calls_per_minute.get(provider_name, self.max_calls_per_minute)
            self._limiters[provider_name] = RateLimiter(calls, 60.0, self._clock)
        return self._limiters[provider_name]

    # --- Current price -----------------------------------------------------------------

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        symbol = symbol.upper()
        provider = self.provider

        def cached() -> MarketSnapshot | None:
            entry = self._snapshots.get(symbol)
            return entry[1] if entry and entry[0] > self._clock() else None

        async def fetch() -> MarketSnapshot:
            snapshot = await provider.fetch_snapshot(symbol)
            self._snapshots[symbol] = (self._clock() + self.cache_ttl, snapshot)
            return snapshot

        return await self._fetch_once(("snapshot", symbol), provider.name, symbol, cached, fetch)

    # --- Historical candles ------------------------------------------------------------

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_CANDLES,
        min_candles: int | None = None,
    ) -> CandleSeries:
        """The `limit` most recent completed candles, oldest first.

        Providers offering `timeframe` are tried in order of preference; a provider that
        fails (outage, rate limit, unknown asset, bad data, too little history) is skipped
        and the reason recorded in `fallback_notes`. Raises `InsufficientDataError` if no
        provider has at least `min_candles` (default: `limit`) candles.
        """
        symbol = symbol.upper()
        tf = self._validate_timeframe(timeframe)
        if not 1 <= limit <= MAX_CANDLES:
            raise InvalidRequestError(f"limit must be between 1 and {MAX_CANDLES} candles")
        required = limit if min_candles is None else min_candles
        if not 1 <= required <= limit:
            raise InvalidRequestError("min_candles must be between 1 and limit")

        providers = [p for p in self.candle_providers if tf in p.supported_timeframes]
        failures: list[tuple[str, MarketDataError]] = []
        for provider in providers:
            try:
                series = await self._provider_candles(provider, symbol, tf, limit, required)
            except (MarketDataUnavailableError, AssetNotFoundError, InsufficientDataError) as exc:
                failures.append((provider.name, exc))
                continue
            notes = [f"{name} {tf} candles were unavailable ({exc})" for name, exc in failures]
            return series.model_copy(
                update={"candles": series.candles[-limit:], "fallback_notes": notes}
            )
        raise _combined_failure(failures)

    async def _provider_candles(
        self, provider: CandleProvider, symbol: str, tf: Timeframe, limit: int, required: int
    ) -> CandleSeries:
        key = (provider.name, symbol, tf)
        ttl = self.candle_cache_ttl(tf)

        def cached() -> CandleSeries | None:
            entry = self._candles.get(key)
            if entry and entry.expires_at > self._clock() and entry.requested >= limit:
                return entry.series
            return None

        async def fetch() -> CandleSeries:
            series = await provider.fetch_candles(symbol, tf, limit)
            self._candles[key] = _CandleCacheEntry(self._clock() + ttl, limit, series)
            return series

        series = await self._fetch_once(("candles", *key), provider.name, symbol, cached, fetch)
        if len(series.candles) < required:
            raise InsufficientDataError(
                f"{provider.name} has only {len(series.candles)} {tf} candle(s) for {symbol}; "
                f"{required} required"
            )
        return series

    def _validate_timeframe(self, timeframe: str) -> Timeframe:
        supported = self.supported_timeframes
        available = ", ".join(supported) or "none"
        for tf in TIMEFRAMES:
            if tf == timeframe:
                if tf not in supported:
                    raise UnsupportedTimeframeError(
                        f"{tf} candles are not available from the configured market data "
                        f"provider(s); available timeframes: {available}"
                    )
                return tf
        raise UnsupportedTimeframeError(
            f"unknown timeframe {timeframe!r}; UpScale timeframes are {', '.join(TIMEFRAMES)}"
        )

    # --- Shared plumbing ---------------------------------------------------------------

    async def _fetch_once(
        self,
        key: object,
        provider_name: str,
        symbol: str,
        cached: Callable[[], T | None],
        fetch: Callable[[], Awaitable[T]],
    ) -> T:
        """Serve from cache, else make one rate-limited provider call per key at a time."""
        self._raise_if_known_unknown(provider_name, symbol)
        if (hit := cached()) is not None:
            return hit
        # Concurrent callers for the same key wait for the first one and reuse its result.
        async with self._locks.setdefault(key, asyncio.Lock()):
            self._raise_if_known_unknown(provider_name, symbol)
            if (hit := cached()) is not None:
                return hit
            if not self._limiter(provider_name).try_acquire():
                raise MarketDataUnavailableError(
                    f"UpScale's {provider_name} request limit was reached; try again in a minute"
                )
            try:
                return await fetch()
            except AssetNotFoundError:
                self._not_found[(provider_name, symbol)] = self._clock() + self.not_found_ttl
                raise

    def _raise_if_known_unknown(self, provider_name: str, symbol: str) -> None:
        if self._not_found.get((provider_name, symbol), 0.0) > self._clock():
            raise AssetNotFoundError(f"{symbol} is not recognized by {provider_name}")


def _combined_failure(failures: list[tuple[str, MarketDataError]]) -> MarketDataError:
    """One error for a request every candidate provider failed, of the most telling type."""
    if len(failures) == 1:
        return failures[0][1]
    message = "; ".join(f"{name}: {exc}" for name, exc in failures)
    kinds = {type(exc) for _, exc in failures}
    if kinds == {AssetNotFoundError}:
        return AssetNotFoundError(message)
    if kinds <= {AssetNotFoundError, InsufficientDataError}:
        return InsufficientDataError(message)
    return MarketDataUnavailableError(message)
