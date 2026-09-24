"""Pure, deterministic indicator math over price lists (oldest first).

Every function returns a list aligned with its input: position i holds the indicator
value at candle i, or None where there isn't enough history yet. No I/O, no guessing.
"""

from collections.abc import Sequence


def _check_period(period: int) -> None:
    if period < 1:
        raise ValueError("period must be at least 1")


def sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average: mean of the last `period` values. First value at index period-1."""
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    window_sum = 0.0
    for i, value in enumerate(values):
        window_sum += value
        if i >= period:
            window_sum -= values[i - period]
        if i >= period - 1:
            out[i] = window_sum / period
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average with alpha = 2 / (period + 1).

    Seeded with the SMA of the first `period` values (the common TA-Lib convention), so
    the first value is at index period-1; then EMA_t = alpha * x_t + (1 - alpha) * EMA_t-1.
    """
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2 / (period + 1)
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = alpha * values[i] + (1 - alpha) * current
        out[i] = current
    return out


def rsi(values: Sequence[float], period: int) -> list[float | None]:
    """Wilder's Relative Strength Index. First value at index `period` (needs period+1 closes).

    Average gain/loss start as the simple mean of the first `period` changes, then use
    Wilder smoothing: avg_t = (avg_t-1 * (period - 1) + change_t) / period.
    RSI = 100 - 100 / (1 + avg_gain / avg_loss); 100 when there are no losses. When
    prices didn't move at all (no gains and no losses) RSI is undefined and stays None.
    """
    _check_period(period)
    out: list[float | None] = [None] * len(values)
    if len(values) < period + 1:
        return out
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    avg_gain = sum(max(c, 0.0) for c in changes[:period]) / period
    avg_loss = sum(max(-c, 0.0) for c in changes[:period]) / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period, len(changes)):
        change = changes[i]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        out[i + 1] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float | None:
    if avg_loss == 0:
        return None if avg_gain == 0 else 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def macd(
    values: Sequence[float], fast: int, slow: int, signal: int
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """MACD line (EMA fast - EMA slow), signal line (EMA of the MACD line), histogram.

    The MACD line starts at index slow-1; the signal line and histogram at slow+signal-2.
    """
    _check_period(signal)
    if not 1 <= fast < slow:
        raise ValueError("MACD fast period must be shorter than the slow period")
    if len(values) < slow:
        empty: list[float | None] = [None] * len(values)
        return empty, list(empty), list(empty)
    fast_ema, slow_ema = ema(values, fast), ema(values, slow)
    line: list[float | None] = [
        f - s if f is not None and s is not None else None
        for f, s in zip(fast_ema, slow_ema, strict=True)
    ]
    start = slow - 1
    signal_tail = ema([v for v in line[start:] if v is not None], signal)
    padding: list[float | None] = [None] * start
    signal_line = padding + signal_tail
    signal_line += [None] * (len(values) - len(signal_line))
    histogram: list[float | None] = [
        m - s if m is not None and s is not None else None
        for m, s in zip(line, signal_line, strict=True)
    ]
    return line, signal_line, histogram
