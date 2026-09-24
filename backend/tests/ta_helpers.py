import math
from datetime import UTC, datetime, timedelta

from upscale.services.market_data import TIMEFRAME_SECONDS, Candle, CandleSeries

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


def swing_series(
    base: float,
    noise: float,
    swings: list[float],
    *,
    last_close: float | None = None,
    timeframe: str = "1m",
) -> CandleSeries:
    """Candles whose true range is `noise` (high/low = base +/- noise/2, close = base), with
    one isolated wick per swing price: above the band it is a swing high, below it a swing
    low (for pivot windows up to 2). 60 trailing noise candles keep ATR 14 close to `noise`.
    The final candle closes at `last_close`. Timestamps are consecutive for `timeframe`.
    """
    step = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    top, bottom = base + noise / 2, base - noise / 2
    quiet = (top, bottom, base)
    rows: list[tuple[float, float, float]] = [quiet] * 15
    for swing in swings:
        assert not bottom <= swing <= top, "swing must lie outside the noise band"
        rows += [(max(top, swing), min(bottom, swing), base), quiet, quiet]
    close = base if last_close is None else last_close
    rows += [quiet] * 60 + [(max(top, close), min(bottom, close), close)]
    candles = [
        Candle(timestamp=START + step * i, open=base, high=high, low=low, close=c)
        for i, (high, low, c) in enumerate(rows)
    ]
    return CandleSeries(
        symbol="BTC",
        provider="TestProvider",
        provider_id="btc",
        timeframe=timeframe,
        candles=candles,
        volume_available=False,
        fetched_at=START,
    )
