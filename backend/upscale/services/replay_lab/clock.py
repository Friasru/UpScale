"""The historical clock: one authoritative time boundary per replay sample.

A sample is analyzed as of its decision time T. Two phases, in this order only:

1. **DECIDING**: evidence may be read up to T and no later. Candles must have *closed*
   by T (a candle that opened before T but closed after it contains post-T trades).
2. **REVEALED**: entered only with a `DecisionReceipt`, the proof that the frozen
   decision was committed to the replay database. Only then may data after T be read,
   and only up to T + the longest outcome horizon.

Every accessor checks its boundary explicitly and raises `LookaheadError` (never
silently filters data it should not have been asked for). Filters that legitimately
select "what existed at T" from a larger cached range are separate, named functions.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from upscale.services.market_data import Candle

Phase = Literal["DECIDING", "REVEALED"]


class LookaheadError(RuntimeError):
    """Information from after the decision time reached (or was requested for) the
    historical decision. Replay must fail loudly rather than record a contaminated
    decision."""


@dataclass(frozen=True)
class DecisionReceipt:
    """Issued by the replay store after a decision row was committed (immutable)."""

    sample_id: int
    decision_at: datetime
    record_hash: str


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


class HistoricalClock:
    def __init__(self, decision_at: datetime, reveal_horizon: timedelta):
        if decision_at.tzinfo is None:
            raise ValueError("decision time must be timezone-aware (UTC)")
        self.decision_at = decision_at.astimezone(UTC)
        self.reveal_horizon = reveal_horizon
        self._receipt: DecisionReceipt | None = None

    @property
    def phase(self) -> Phase:
        return "DECIDING" if self._receipt is None else "REVEALED"

    @property
    def visible_until(self) -> datetime:
        """The latest instant whose information may be read in the current phase."""
        if self._receipt is None:
            return self.decision_at
        return self.decision_at + self.reveal_horizon

    def reveal(self, receipt: DecisionReceipt) -> None:
        if receipt.decision_at != self.decision_at:
            raise LookaheadError("a receipt for another decision time cannot reveal this sample")
        if not receipt.record_hash:
            raise LookaheadError("future data can only be revealed after the decision is stored")
        self._receipt = receipt

    @property
    def receipt(self) -> DecisionReceipt | None:
        return self._receipt

    # --- explicit boundary checks -------------------------------------------------------------

    def check_time(self, at: datetime, what: str) -> None:
        if at > self.visible_until:
            raise LookaheadError(
                f"{what} at {at.isoformat()} is after the visible boundary "
                f"{self.visible_until.isoformat()} ({self.phase})"
            )

    def check_request(self, end: datetime, what: str) -> None:
        """A request for data up to `end` (exclusive)."""
        if end > self.visible_until:
            raise LookaheadError(
                f"{what} requested up to {end.isoformat()}, beyond the visible boundary "
                f"{self.visible_until.isoformat()} ({self.phase})"
            )

    def check_candles(self, candles: Iterable[Candle], interval: timedelta, what: str) -> None:
        """Every candle must have closed by the visible boundary."""
        for c in candles:
            if c.timestamp + interval > self.visible_until:
                raise LookaheadError(
                    f"{what}: candle opened {c.timestamp.isoformat()} closes after the visible "
                    f"boundary {self.visible_until.isoformat()} ({self.phase})"
                )

    def check_decision_phase(self, what: str) -> None:
        if self.phase != "DECIDING":
            raise LookaheadError(f"{what} can only run before the future is revealed")


def closed_by(candles: Sequence[Candle], interval: timedelta, at: datetime) -> list[Candle]:
    """The candles that had closed by `at` (oldest first): what existed at time `at`."""
    return sorted((c for c in candles if c.timestamp + interval <= at), key=lambda c: c.timestamp)
