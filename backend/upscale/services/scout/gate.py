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
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from upscale.services.market_data import MarketDataUnavailableError, ProviderRateLimitedError
from upscale.services.quota import _LANE, DEFAULT_LANE, LaneLimiter, request_lane
from upscale.services.scout.config import ScoutProviderLimits

__all__ = ["DEFAULT_LANE", "LaneLimiter", "RateLimitReachedError", "RequestGate", "request_lane"]

T = TypeVar("T")


class RateLimitReachedError(ProviderRateLimitedError):
    """UpScale's own budget for a provider (or the provider's limit) was reached."""


class RequestGate:
    """`limiter`: the provider's shared quota (see `upscale.services.quota`), when other
    UpScale features call the same provider; otherwise the gate keeps its own."""

    def __init__(
        self,
        name: str,
        limits: ScoutProviderLimits,
        clock: Callable[[], float] = time.monotonic,
        limiter: LaneLimiter | None = None,
    ):
        self.name = name
        self.limits = limits
        self._clock = clock
        self._shared = limiter is not None
        self._limiter = limiter if limiter is not None else self._new_limiter()
        self._semaphore = asyncio.Semaphore(limits.max_concurrency)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self.requests_made = 0  # outgoing requests actually started (for tests / metrics)
        self.cache_hits = 0  # calls answered from the cache (for usage reporting)

    def reset(self) -> None:
        self._cache.clear()
        self._inflight.clear()
        if self._shared:
            self._limiter.reset()
        else:
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
        return self._limiter.reservations.get(lane, 0)

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
