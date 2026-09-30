"""UpScale's single authoritative "now" for domain logic.

Live code always gets the wall clock. Historical replay (`upscale.services.replay_lab`)
freezes it at the historical decision time T for the duration of one analysis, so logic
that measures ages or freshness against "now" (e.g. a pool's age in the asset profile)
sees T, never the present. The override is a context variable: it applies only to the
code (and the tasks it starts) running inside `frozen_now`, never to other requests.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

_FROZEN: ContextVar[datetime | None] = ContextVar("upscale_frozen_now", default=None)


def utcnow() -> datetime:
    """The current time for domain logic: the wall clock, or the frozen replay time."""
    frozen = _FROZEN.get()
    return frozen if frozen is not None else datetime.now(UTC)


def frozen_time() -> datetime | None:
    """The frozen replay time, or None when running live."""
    return _FROZEN.get()


@contextmanager
def frozen_now(at: datetime) -> Iterator[None]:
    """Inside this block (and tasks started in it) `utcnow()` returns `at`."""
    if at.tzinfo is None:
        raise ValueError("a frozen time must be timezone-aware (UTC)")
    token = _FROZEN.set(at.astimezone(UTC))
    try:
        yield
    finally:
        _FROZEN.reset(token)
