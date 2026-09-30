"""Historical pool candles: fetched once, cached, and served point-in-time.

**Fetching.** Candles are requested from GeckoTerminal (the production
`GeckoTerminalProvider` and parser) in fixed, aligned chunks of 1,000 candles per
timeframe: chunk k covers [k x span, (k + 1) x span) with span = 1,000 x interval. A chunk
is requested with ``before_timestamp`` = its end and ``limit`` = 1,000, so it returns every
candle the pool has in that span (quiet intervals have none: gaps are never filled). Any
sample needing any part of a chunk reuses it, so one pool's samples across days cost a few
requests per timeframe instead of several per sample.

**Caching.** Historical candles are immutable: a chunk that had fully closed when fetched
is cached for good (key: provider, chain, token, pool, timeframe, start, end). A chunk
reaching past the fetch time is cached as incomplete and only serves ranges that had
closed by then. Every request goes through `ReplayGate` (lowest priority); a cache hit
costs nothing.

**Serving.** `PointInTimeCandles` is what the decision phase sees: only candles that had
*closed* by T, re-checked against the historical clock. The Technical agent gets them
through `PointInTimeRegistry`, which mimics the live `get_candles` exactly (the latest
`limit` candles opened before T, the in-progress one dropped, only the latest consecutive
run kept, by the production `parse_pool_candles`). It serves only the sample's own pool:
other pools' historical state is unknown, so the live alternate-pool fallback can't apply.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from upscale.services.chains import GECKOTERMINAL_NETWORKS, chain_label, normalize_address
from upscale.services.geckoterminal import GT_TIMEFRAMES, GeckoTerminalProvider, parse_pool_candles
from upscale.services.market_data import (
    TIMEFRAME_SECONDS,
    AssetNotFoundError,
    Candle,
    CandleSeries,
    MarketDataError,
    MarketDataUnavailableError,
    ProviderRateLimitedError,
    Timeframe,
)
from upscale.services.replay_lab.clock import HistoricalClock, LookaheadError, closed_by, utc
from upscale.services.replay_lab.quota import ReplayDeferred, ReplayGate
from upscale.services.replay_lab.store import ReplayStore

CHUNK_CANDLES = 1000
PROVIDER = "GeckoTerminal"
# Cache key of candles priced for the exact token (never mixed with base-priced ones).
TOKEN_PRICED_KEY = "GeckoTerminal:token-priced"


class PoolHistoryUnavailableError(Exception):
    """The provider doesn't know the pool (dead, migrated or never listed)."""


def interval_of(timeframe: Timeframe) -> timedelta:
    return timedelta(seconds=TIMEFRAME_SECONDS[timeframe])


def chunk_bounds(
    timeframe: Timeframe, start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """The aligned chunks covering [start, end)."""
    span = TIMEFRAME_SECONDS[timeframe] * CHUNK_CANDLES
    first = math.floor(start.timestamp() / span)
    last = math.ceil(end.timestamp() / span)
    return [(utc(k * span), utc((k + 1) * span)) for k in range(first, max(last, first + 1))]


@dataclass
class ProviderUsage:
    requests: int = 0
    cache_hits: int = 0
    rate_limited: int = 0
    failures: int = 0
    deferrals: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": PROVIDER,
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "rate_limited": self.rate_limited,
            "failures": self.failures,
            "deferrals": dict(self.deferrals),
        }


class HistoricalCandleFetcher:
    def __init__(
        self,
        store: ReplayStore,
        gate: ReplayGate,
        provider: GeckoTerminalProvider | None = None,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.gate = gate
        self.provider = provider or GeckoTerminalProvider()
        self.wall_clock = wall_clock
        self.usage = ProviderUsage()

    async def range(
        self,
        chain: str,
        token: str,
        pool: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        token_priced: bool = False,
    ) -> list[Candle]:
        """Every candle of this exact pool opened in [start, end), oldest first (gaps stay
        gaps). May raise `ReplayDeferred` (try later) or `PoolHistoryUnavailableError`.

        `token_priced`: priced in USD for this exact token, as every replay use requests
        them (Technical input, reference prices, outcomes, like live Technical and outcome
        collection). Without it: the provider's own pool base (kept only for candles cached
        before exact-token pricing; never mixed: cached under another key)."""
        if chain not in GECKOTERMINAL_NETWORKS or timeframe not in GT_TIMEFRAMES:
            raise PoolHistoryUnavailableError(f"{PROVIDER} has no {timeframe} candles for {chain}")
        out: dict[datetime, Candle] = {}
        for lo, hi in chunk_bounds(timeframe, start, end):
            for c in await self._chunk(chain, token, pool, timeframe, lo, hi, end, token_priced):
                if start <= c.timestamp < end:
                    out[c.timestamp] = c
        return [out[t] for t in sorted(out)]

    async def _chunk(
        self,
        chain: str,
        token: str,
        pool: str,
        timeframe: Timeframe,
        lo: datetime,
        hi: datetime,
        needed_until: datetime,
        token_priced: bool = False,
    ) -> list[Candle]:
        key = TOKEN_PRICED_KEY if token_priced else PROVIDER
        cached = self.store.cached_candles(key, chain, token, pool, timeframe, lo, hi)
        if cached is not None:
            candles, fetched_at, complete = cached
            if complete or min(hi, needed_until).timestamp() <= fetched_at:
                self.usage.cache_hits += 1
                return candles
        try:
            await self.gate.acquire()
        except ReplayDeferred as exc:
            self.usage.deferrals[exc.reason] = self.usage.deferrals.get(exc.reason, 0) + 1
            raise
        self.usage.requests += 1
        fetched = self.wall_clock()
        try:
            series = await self.provider.fetch_pool_candles(
                chain, pool, timeframe, CHUNK_CANDLES, symbol=pool, canonical_id=f"{chain}:{token}",
                now=utc(fetched), before=hi, contiguous=False,
                token=token if token_priced else None,
            )  # fmt: skip
        except ProviderRateLimitedError as exc:
            self.usage.rate_limited += 1
            self.gate.note_rate_limited()
            raise ReplayDeferred(
                f"{PROVIDER} answered 'rate limited'", self.gate.cooldown, production=True
            ) from exc
        except AssetNotFoundError as exc:
            self.usage.failures += 1
            raise PoolHistoryUnavailableError(f"{PROVIDER} doesn't know pool {pool}") from exc
        except MarketDataError:
            self.usage.failures += 1
            raise
        candles = [c for c in series.candles if lo <= c.timestamp < hi]
        complete = hi.timestamp() <= fetched
        self.store.cache_candles(
            key, chain, token, pool, timeframe, lo, hi, candles, fetched, complete
        )
        return candles


class PointInTimeCandles:
    """One sample's candles, fetched ahead of the decision, served strictly by the clock."""

    def __init__(self, clock: HistoricalClock, chain: str, token: str, pool: str):
        self.clock = clock
        self.chain, self.token, self.pool = chain, token, pool
        self._loaded: dict[Timeframe, list[Candle]] = {}

    def load(self, timeframe: Timeframe, candles: Sequence[Candle]) -> None:
        merged = {c.timestamp: c for c in self._loaded.get(timeframe, [])}
        merged |= {c.timestamp: c for c in candles}
        self._loaded[timeframe] = [merged[t] for t in sorted(merged)]

    def loaded(self, timeframe: Timeframe) -> bool:
        return timeframe in self._loaded

    def closed(self, timeframe: Timeframe) -> list[Candle]:
        """Candles closed by the clock's visible boundary."""
        interval = interval_of(timeframe)
        out = closed_by(self._loaded.get(timeframe, []), interval, self.clock.visible_until)
        self.clock.check_candles(out, interval, f"{timeframe} candles")
        return out

    def window(self, timeframe: Timeframe, start: datetime, end: datetime) -> list[Candle]:
        """Candles lying entirely inside [start, end] (outcome windows: reveal phase)."""
        self.clock.check_request(end, f"{timeframe} candles")
        interval = interval_of(timeframe)
        out = [
            c
            for c in self._loaded.get(timeframe, [])
            if c.timestamp >= start and c.timestamp + interval <= end
        ]
        self.clock.check_candles(out, interval, f"{timeframe} candles")
        return out

    def series_at(
        self, timeframe: Timeframe, limit: int, symbol: str, canonical_id: str | None
    ) -> CandleSeries:
        """The series the live `DexCandleService.get_candles` would have returned at T."""
        self.clock.check_decision_phase("a decision-time candle series")
        t = self.clock.decision_at
        interval = interval_of(timeframe)
        opened = [c for c in self._loaded.get(timeframe, []) if c.timestamp < t][-(limit + 1) :]
        rows = [
            [c.timestamp.timestamp(), c.open, c.high, c.low, c.close, c.volume or 0.0]
            for c in opened
        ]
        candles, notes = parse_pool_candles(rows, timeframe, t, PROVIDER, contiguous=True)
        self.clock.check_candles(candles, interval, "technical candles")
        return CandleSeries(
            symbol=symbol,
            provider=PROVIDER,
            provider_id=self.pool,
            pair=f"{chain_label(self.chain)} pool {self.pool}",
            timeframe=timeframe,
            candles=candles[-limit:],
            volume_available=True,
            fetched_at=t,
            canonical_id=canonical_id,
            volume_unit="USD",
            as_of=t,
            notes=notes,
        )


class PointInTimeCandleSource:
    """A `PoolCandleSource` over one sample's point-in-time candles (no network)."""

    provider_name = PROVIDER
    supported_timeframes: frozenset[Timeframe] = frozenset(GT_TIMEFRAMES)

    def __init__(self, candles: PointInTimeCandles):
        self.candles = candles
        self.requests: list[tuple[str, Timeframe, int]] = []

    def covers(self, chain: str) -> bool:
        return chain in GECKOTERMINAL_NETWORKS

    async def get_candles(
        self,
        chain: str,
        pool: str,
        timeframe: Timeframe,
        limit: int,
        *,
        symbol: str,
        canonical_id: str | None,
        token: str,
    ) -> CandleSeries:
        self.requests.append((pool, timeframe, limit))
        if chain != self.candles.chain or pool != self.candles.pool:
            raise MarketDataUnavailableError(
                "historical replay only has candles for the sample's own pool (other pools' "
                "history at the decision time is unknown)"
            )
        own = normalize_address(chain, self.candles.token) or self.candles.token
        if (normalize_address(chain, token) or token) != own:
            raise MarketDataUnavailableError(
                "historical replay only has candles priced for the sample's exact token"
            )
        if not self.candles.loaded(timeframe):
            raise MarketDataUnavailableError(f"{timeframe} candles were not loaded for replay")
        series = self.candles.series_at(timeframe, limit, symbol, canonical_id)
        if not series.candles:
            raise MarketDataUnavailableError(
                f"{PROVIDER} had no closed {timeframe} candles for this pool before the decision time"
            )
        return series


class PointInTimeRegistry:
    """What `TechnicalAnalysisAgent` needs from the provider registry, at time T."""

    def __init__(self, source: PointInTimeCandleSource):
        self._source = source

    @property
    def dex_candles(self) -> PointInTimeCandleSource:
        return self._source

    async def pool_candles(
        self,
        chain: str,
        pool: str,
        timeframe: Timeframe,
        limit: int,
        *,
        symbol: str,
        canonical_id: str | None,
        token: str,
    ) -> CandleSeries:
        if not self._source.covers(chain):
            raise MarketDataUnavailableError(f"no pool-candle provider covers {chain}")
        return await self._source.get_candles(
            chain, pool, timeframe, limit, symbol=symbol, canonical_id=canonical_id, token=token
        )


def assert_closed(candles: Sequence[Candle], timeframe: Timeframe, at: datetime, what: str) -> None:
    interval = interval_of(timeframe)
    for c in candles:
        if c.timestamp + interval > at:
            raise LookaheadError(f"{what}: a {timeframe} candle closing after {at.isoformat()}")
