"""Replay's provider access: the lowest priority of all UpScale work.

Priority on every shared provider quota:

1. Interactive Analyze
2. Due live outcome collection
3. Manual Scout refresh
4. Background Scout
5. Historical Replay

Replay requests go through the production `LaneLimiter` in their own lane (`replay`),
which holds no reservation: they can never use capacity reserved for Analyze or Scout
refresh. On top of that, `ReplayGate` defers (raises `ReplayDeferred`, never waits in a
loop) whenever:

* production work is active (`production_busy`: in the CLI, a local UpScale backend is
  running, since its in-memory quota can't be shared with another process; in-process
  callers pass their own activity signals);
* the provider answered "rate limited" (HTTP 429) recently;
* any other lane used the provider in the current window (someone else needs it);
* replay already used its share of the current window (pacing, a short wait).
"""

import time
from collections.abc import Awaitable, Callable
from datetime import datetime

import httpx2

from upscale.services.quota import LaneLimiter
from upscale.services.scout.config import ScoutProviderLimits

REPLAY_LANE = "replay"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


class ReplayDeferred(Exception):
    """Replay must wait: `retry_after` seconds (a hint), `production` when higher-priority
    work caused it (the job is PAUSED), otherwise it's replay's own pacing."""

    def __init__(self, reason: str, retry_after: float, production: bool):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after
        self.production = production


def production_limiter(
    limits: ScoutProviderLimits, clock: Callable[[], float] = time.monotonic
) -> LaneLimiter:
    """A limiter with the production GeckoTerminal budget and reservations (held), so the
    replay lane only ever sees the unreserved share."""
    return LaneLimiter(limits.calls_per_minute, 60.0, limits.reservations, clock=clock)


async def _idle() -> str | None:
    return None


class ReplayGate:
    def __init__(
        self,
        limiter: LaneLimiter,
        production_busy: Callable[[], Awaitable[str | None]] = _idle,
        lane: str = REPLAY_LANE,
        rate_limit_cooldown_seconds: float = 600.0,
        busy_retry_seconds: float = 300.0,
    ):
        if lane in limiter.reservations:
            raise ValueError("the replay lane must not hold a reservation")
        self.limiter = limiter
        self.production_busy = production_busy
        self.lane = lane
        self.cooldown = rate_limit_cooldown_seconds
        self.busy_retry = busy_retry_seconds

    async def check(self) -> None:
        """Raise `ReplayDeferred` when replay must not send a request now."""
        busy = await self.production_busy()
        if busy is not None:
            raise ReplayDeferred(busy, self.busy_retry, production=True)
        ago = self.limiter.rate_limited_ago()
        if ago is not None and ago < self.cooldown:
            raise ReplayDeferred(
                "the provider answered 'rate limited' recently", self.cooldown - ago, True
            )
        snapshot = self.limiter.snapshot([self.lane])
        used = snapshot["used"]
        assert isinstance(used, dict)
        others = {lane: n for lane, n in used.items() if lane != self.lane and n}
        if others:
            raise ReplayDeferred(
                f"provider capacity in use by other work ({', '.join(sorted(others))})",
                self.limiter.period,
                production=True,
            )
        if self.limiter.available(self.lane) <= 0:
            raise ReplayDeferred(
                "replay's share of the provider window is used", self.limiter.period / 2, False
            )

    async def acquire(self) -> None:
        await self.check()
        if not self.limiter.try_acquire(self.lane):
            raise ReplayDeferred("provider quota unavailable", self.limiter.period / 2, False)

    def note_rate_limited(self) -> None:
        self.limiter.note_rate_limited()


class LocalBackendProbe:
    """Is a local UpScale backend running? It owns the provider quota on this machine
    (Analyze, outcome collection, Scout, Background Scout), which a separate replay process
    can't share, so replay pauses while it answers. Railway runs elsewhere (its own quota)
    and is never contacted."""

    def __init__(
        self,
        url: str = DEFAULT_BACKEND_URL,
        cache_seconds: float = 30.0,
        transport: httpx2.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.url = url.rstrip("/")
        self.cache_seconds = cache_seconds
        self._transport = transport
        self._clock = clock
        self._last: tuple[float, str | None] | None = None

    async def __call__(self) -> str | None:
        now = self._clock()
        if self._last is not None and now - self._last[0] < self.cache_seconds:
            return self._last[1]
        reason = None
        try:
            async with httpx2.AsyncClient(timeout=1.5, transport=self._transport) as client:
                response = await client.get(f"{self.url}/health")
            if response.status_code == 200:
                reason = (
                    f"a local UpScale backend is running at {self.url}: Analyze, outcome "
                    "collection and Scout have priority on the shared provider quota"
                )
        except httpx2.HTTPError:
            reason = None
        self._last = (now, reason)
        return reason


def describe(limiter: LaneLimiter, at: datetime | None = None) -> dict[str, object]:
    snap = limiter.snapshot([REPLAY_LANE])
    if at is not None:
        snap["at"] = at.isoformat()
    return snap
