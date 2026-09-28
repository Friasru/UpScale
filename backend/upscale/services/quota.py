"""Provider request quotas shared across UpScale.

A real external quota (e.g. GeckoTerminal's free API: a burst of ~6, then ~1 request per
10 s, per client) is shared by every UpScale feature that calls the provider, so UpScale's
accounting must be shared too: one `LaneLimiter` per provider, handed to every consumer
(Scout's request gate, Analyze's candle service, ...).

Requests run in named lanes (`request_lane`, or an explicit lane). A provider's limits may
reserve part of its budget for a lane: other lanes can't use a held reservation, the lane
itself may also use free unreserved capacity, and the total never exceeds the limit.

* "interactive": what a user is waiting for (Analyze). Held permanently: background work
  never borrows it, so an Analyze right after a Scout scan still gets capacity.
* "refresh": Scout's tracked-token refresh; released after the refresh, so discovery can
  use what it left.
* "default": everything else (e.g. Scout discovery): unreserved capacity only.
"""

import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

DEFAULT_LANE = "default"
# What a user is waiting for right now (Analyze): its reservation is never lent out.
INTERACTIVE_LANE = "interactive"
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
        reservations: Mapping[str, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_calls = max_calls
        self.period = period
        self.reservations = dict(reservations or {})
        self._clock = clock
        self._calls: deque[tuple[float, str]] = deque()
        self._held = set(self.reservations)

    def reset(self) -> None:
        """Forget every call and hold every reservation again (e.g. between tests)."""
        self._calls.clear()
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


def current_lane() -> str:
    return _LANE.get()
