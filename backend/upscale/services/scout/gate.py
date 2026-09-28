"""One gate per provider: every outgoing request passes through it.

* **cache**: successful responses are kept for `cache_ttl` seconds (failures never are);
* **deduplication**: concurrent callers asking for the same key share one request;
* **rate limit**: a sliding-window budget per provider; over budget fails fast. The limit
  is hard: nothing ever sends more than `calls_per_minute` in any 60 seconds;
* **reservations**: part of that budget can be reserved for a named lane (e.g. tracked-token
  refresh), set per call with `request_lane`. Other lanes can't use a reservation while it
  is held; the lane itself may also use free unreserved capacity. `release` hands what is
  left of it back to everyone (e.g. once the refresh is done), `arm` restores it;
* **concurrency**: at most `max_concurrency` requests in flight per provider;
* **timeout**: a request running longer than `timeout` seconds fails.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from upscale.services.market_data import MarketDataUnavailableError
from upscale.services.scout.config import ScoutProviderLimits

T = TypeVar("T")

DEFAULT_LANE = "default"
_LANE: ContextVar[str] = ContextVar("upscale_request_lane", default=DEFAULT_LANE)


@contextmanager
def request_lane(lane: str) -> Iterator[None]:
    """Every gated request made inside this block (including tasks it starts) counts
    against `lane`, and may use the capacity reserved for it."""
    token = _LANE.set(lane)
    try:
        yield
    finally:
        _LANE.reset(token)


class LaneLimiter:
    """A sliding-window request limit shared by lanes, with per-lane reservations.

    Over any `period`: total calls <= `max_calls` (hard). While a reservation is held,
    calls from other lanes leave room for what the lane hasn't used of it yet."""

    def __init__(
        self,
        max_calls: int,
        period: float,
        reservations: dict[str, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_calls = max_calls
        self.period = period
        self.reservations = dict(reservations or {})
        self._clock = clock
        self._calls: deque[tuple[float, str]] = deque()
        self._held = set(self.reservations)

    def _trim(self) -> None:
        now = self._clock()
        while self._calls and now - self._calls[0][0] >= self.period:
            self._calls.popleft()

    def _outstanding(self, lane: str) -> int:
        """Reserved capacity other lanes must leave free for `lane`'s competitors."""
        held = 0
        for other in self._held:
            if other == lane:
                continue
            used = sum(1 for _, x in self._calls if x == other)
            held += max(0, self.reservations[other] - used)
        return held

    def available(self, lane: str = DEFAULT_LANE) -> int:
        self._trim()
        return max(0, self.max_calls - len(self._calls) - self._outstanding(lane))

    def used(self, lane: str) -> int:
        self._trim()
        return sum(1 for _, x in self._calls if x == lane)

    def try_acquire(self, lane: str = DEFAULT_LANE) -> bool:
        if self.available(lane) <= 0:
            return False
        self._calls.append((self._clock(), lane))
        return True

    def release(self, lane: str) -> None:
        self._held.discard(lane)

    def arm(self, lane: str) -> None:
        if lane in self.reservations:
            self._held.add(lane)


class RateLimitReachedError(MarketDataUnavailableError):
    """UpScale's own budget for a provider (or the provider's limit) was reached."""


class RequestGate:
    def __init__(
        self,
        name: str,
        limits: ScoutProviderLimits,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.limits = limits
        self._clock = clock
        self._limiter = self._new_limiter()
        self._semaphore = asyncio.Semaphore(limits.max_concurrency)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self.requests_made = 0  # outgoing requests actually started (for tests / metrics)
        self.cache_hits = 0  # calls answered from the cache (for usage reporting)

    def reset(self) -> None:
        self._cache.clear()
        self._inflight.clear()
        self._limiter = self._new_limiter()
        self._semaphore = asyncio.Semaphore(self.limits.max_concurrency)
        self.requests_made = 0
        self.cache_hits = 0

    def _new_limiter(self) -> LaneLimiter:
        return LaneLimiter(
            self.limits.calls_per_minute, 60.0, dict(self.limits.reservations), self._clock
        )

    def available(self, lane: str | None = None) -> int:
        """Requests the rate limit would let start right now in `lane` (default: the
        current lane)."""
        return self._limiter.available(lane or _LANE.get())

    def used(self, lane: str) -> int:
        """Requests `lane` started within the current rate-limit window."""
        return self._limiter.used(lane)

    def reserved(self, lane: str) -> int:
        return self.limits.reservations.get(lane, 0)

    def release(self, lane: str) -> None:
        """Hand what `lane` hasn't used of its reservation back to every lane."""
        self._limiter.release(lane)

    def arm(self, lane: str) -> None:
        """Hold `lane`'s reservation again (e.g. at the start of a run)."""
        self._limiter.arm(lane)

    async def run(self, key: str, fetch: Callable[[], Awaitable[T]]) -> T:
        entry = self._cache.get(key)
        if entry is not None and entry[0] > self._clock():
            cached: T = entry[1]
            self.cache_hits += 1
            return cached
        task = self._inflight.get(key)
        if task is None:
            if not self._limiter.try_acquire(_LANE.get()):
                raise RateLimitReachedError(
                    f"UpScale's {self.name} request limit was reached; try again in a minute"
                )
            task = asyncio.ensure_future(self._fetch(key, fetch))
            # Mark the outcome retrieved even if every waiter was cancelled.
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
            self._inflight[key] = task
        # shield: one caller being cancelled must not cancel the request for the others
        result: T = await asyncio.shield(task)
        return result

    async def _fetch(self, key: str, fetch: Callable[[], Awaitable[T]]) -> T:
        try:
            async with self._semaphore:
                self.requests_made += 1
                try:
                    value = await asyncio.wait_for(fetch(), self.limits.timeout_seconds)
                except TimeoutError as exc:
                    raise MarketDataUnavailableError(f"{self.name} request timed out") from exc
            if self.limits.cache_ttl_seconds > 0:
                self._cache[key] = (self._clock() + self.limits.cache_ttl_seconds, value)
            return value
        finally:
            self._inflight.pop(key, None)
