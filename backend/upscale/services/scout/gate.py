"""One gate per provider: every outgoing request passes through it.

* **cache**: successful responses are kept for `cache_ttl` seconds (failures never are);
* **deduplication**: concurrent callers asking for the same key share one request;
* **rate limit**: a sliding-window budget per provider; over budget fails fast;
* **concurrency**: at most `max_concurrency` requests in flight per provider;
* **timeout**: a request running longer than `timeout` seconds fails.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from upscale.services.market_data import MarketDataUnavailableError, RateLimiter
from upscale.services.scout.config import ScoutProviderLimits

T = TypeVar("T")


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
        self._limiter = RateLimiter(limits.calls_per_minute, 60.0, clock)
        self._semaphore = asyncio.Semaphore(limits.max_concurrency)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self.requests_made = 0  # outgoing requests actually started (for tests / metrics)

    def reset(self) -> None:
        self._cache.clear()
        self._inflight.clear()
        self._limiter = RateLimiter(self.limits.calls_per_minute, 60.0, self._clock)
        self._semaphore = asyncio.Semaphore(self.limits.max_concurrency)
        self.requests_made = 0

    async def run(self, key: str, fetch: Callable[[], Awaitable[T]]) -> T:
        entry = self._cache.get(key)
        if entry is not None and entry[0] > self._clock():
            cached: T = entry[1]
            return cached
        task = self._inflight.get(key)
        if task is None:
            if not self._limiter.try_acquire():
                raise MarketDataUnavailableError(
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
