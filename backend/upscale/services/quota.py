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

A lane may *outrank* another (`outranks`): it may then use that lane's held reservation too
(e.g. due outcome measurements outrank Scout refresh, which otherwise keeps its reservation
held between scans). Nothing outranks "interactive" unless configured, and the total is
still hard.

A limiter also remembers when the provider itself last answered "rate limited" (HTTP 429),
so optional background work can stay away from a provider that is pushing back.
"""

import time
from collections import deque
from collections.abc import Callable, Collection, Iterator, Mapping
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
    calls from other lanes leave room for what the lane hasn't used of it yet, except lanes
    that outrank it (`outranks`: lane -> the lanes whose reservations it may use)."""

    def __init__(
        self,
        max_calls: int,
        period: float,
        reservations: Mapping[str, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        outranks: Mapping[str, Collection[str]] | None = None,
    ):
        self.max_calls = max_calls
        self.period = period
        self.reservations = dict(reservations or {})
        self.outranks = {lane: frozenset(lower) for lane, lower in (outranks or {}).items()}
        self._clock = clock
        self._calls: deque[tuple[float, str]] = deque()
        self._held = set(self.reservations)
        self._rate_limited_at: float | None = None

    def reset(self) -> None:
        """Forget every call and hold every reservation again (e.g. between tests)."""
        self._calls.clear()
        self._held = set(self.reservations)
        self._rate_limited_at = None

    def _trim(self) -> None:
        now = self._clock()
        while self._calls and now - self._calls[0][0] >= self.period:
            self._calls.popleft()

    def _consumed(self) -> dict[str, int]:
        """Calls counted against each held reservation: its own lane's, plus what a lane
        that outranks it took beyond the unreserved capacity (a borrowed call uses up the
        reservation it came from, so that reservation isn't also held for anyone else)."""
        counts: dict[str, int] = {}
        for _, x in self._calls:
            counts[x] = counts.get(x, 0) + 1
        consumed = {r: min(self.reservations[r], counts.get(r, 0)) for r in self._held}
        unreserved = self.max_calls - sum(self.reservations[r] for r in self._held)
        for x, n in counts.items():
            if x not in self.outranks:  # borrowers are placed below
                unreserved -= n - consumed.get(x, 0)
        for lane in sorted(self.outranks):
            rest = counts.get(lane, 0) - consumed.get(lane, 0)
            take = min(rest, max(0, unreserved))
            unreserved -= take
            rest -= take
            for lower in sorted(self.outranks[lane] & self._held):
                took = min(rest, self.reservations[lower] - consumed[lower])
                consumed[lower] += took
                rest -= took
        return consumed

    def _outstanding(self, lane: str) -> int:
        """Reserved capacity `lane` must leave free for other lanes."""
        consumed = self._consumed()
        lower = self.outranks.get(lane, frozenset())
        return sum(
            self.reservations[other] - consumed[other]
            for other in self._held
            if other != lane and other not in lower
        )

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

    def note_rate_limited(self) -> None:
        """The provider itself rejected a request for its rate limit (e.g. HTTP 429)."""
        self._rate_limited_at = self._clock()

    def rate_limited_within(self, seconds: float) -> bool:
        at = self._rate_limited_at
        return at is not None and self._clock() - at < seconds

    def rate_limited_ago(self) -> float | None:
        """Seconds since the provider last answered "rate limited" (None: never)."""
        at = self._rate_limited_at
        return None if at is None else self._clock() - at

    def snapshot(self, lanes: Collection[str] = ()) -> dict[str, object]:
        """The current window (for diagnostics): calls per lane, what each of `lanes` (and
        every reserved lane) could start now, and the held reservations."""
        self._trim()
        used: dict[str, int] = {}
        for _, lane in self._calls:
            used[lane] = used.get(lane, 0) + 1
        ago = self.rate_limited_ago()
        return {
            "limit": self.max_calls,
            "period_seconds": self.period,
            "used": used,
            "available": {x: self.available(x) for x in sorted({*lanes, *self.reservations})},
            "reservations": dict(self.reservations),
            "held": sorted(self._held),
            "outranks": {k: sorted(v) for k, v in self.outranks.items()},
            "provider_429_seconds_ago": None if ago is None else round(ago, 1),
        }


def current_lane() -> str:
    return _LANE.get()
