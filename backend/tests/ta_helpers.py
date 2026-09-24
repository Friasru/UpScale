import math
from datetime import UTC, datetime, timedelta

from upscale.services.market_data import Candle, CandleSeries

START = datetime(2026, 9, 1, tzinfo=UTC)


def make_series(
    closes: list[float],
    *,
    spread: float = 0.5,
    volume_available: bool = False,
    symbol: str = "BTC",
    timeframe: str = "4h",
) -> CandleSeries:
    """Consecutive 4h candles: open = previous close, high/low = body +/- spread."""
    candles = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i else close
        candles.append(
            Candle(
                timestamp=START + timedelta(hours=4 * i),
                open=open_,
                high=max(open_, close) + spread,
                low=min(open_, close) - spread,
                close=close,
                volume=100.0 if volume_available else None,
            )
        )
    return CandleSeries(
        symbol=symbol,
        provider="TestProvider",
        provider_id=symbol.lower(),
        timeframe=timeframe,
        candles=candles,
        volume_available=volume_available,
        fetched_at=START,
    )


def wave(count: int, mid: float = 100.0, amplitude: float = 10.0, period: int = 20) -> list[float]:
    """A price oscillating between roughly mid - amplitude and mid + amplitude."""
    return [mid + amplitude * math.sin(2 * math.pi * i / period) for i in range(count)]
