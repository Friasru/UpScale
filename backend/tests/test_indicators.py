import math

import pytest

from upscale.services.indicators import ema, macd, rsi, sma

# Worked example from StockCharts' RSI article (Wilder's method).
STOCKCHARTS_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
    45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21,
]  # fmt: skip
# StockCharts rounds intermediate averages to 2 decimals, so allow a small tolerance.
STOCKCHARTS_RSI = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93]


def approx_list(values, **kwargs):
    return [None if v is None else pytest.approx(v, **kwargs) for v in values]


# --- SMA --------------------------------------------------------------------------------


def test_sma():
    assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2, 3, 4]
    assert sma([2, 4, 9], 1) == [2, 4, 9]


def test_sma_not_enough_values():
    assert sma([1, 2], 3) == [None, None]
    assert sma([], 3) == []


# --- EMA --------------------------------------------------------------------------------


def test_ema_is_seeded_with_sma_then_smoothed():
    # alpha = 2 / (3 + 1) = 0.5; seed = mean(2, 4, 6) = 4
    assert ema([2, 4, 6, 8, 12], 3) == [None, None, 4, 6, 9]


def test_ema_period_one_is_the_input():
    assert ema([3, 1, 4], 1) == [3, 1, 4]


def test_ema_constant_series():
    assert ema([5.0] * 10, 4)[-1] == 5.0


def test_ema_not_enough_values():
    assert ema([1, 2], 3) == [None, None]


@pytest.mark.parametrize("fn", [sma, ema, rsi])
def test_period_must_be_positive(fn):
    with pytest.raises(ValueError):
        fn([1, 2, 3], 0)


# --- RSI --------------------------------------------------------------------------------


def test_rsi_matches_stockcharts_reference():
    values = [v for v in rsi(STOCKCHARTS_CLOSES, 14) if v is not None]
    assert values == [pytest.approx(v, abs=0.1) for v in STOCKCHARTS_RSI]


def test_rsi_hand_calculation():
    # changes +1, -1, +1; first avg gain = avg loss = 0.5 -> 50
    # then gain = (0.5 + 1) / 2 = 0.75, loss = (0.5 + 0) / 2 = 0.25 -> RS 3 -> 75
    assert rsi([1, 2, 1, 2], 2) == [None, None, 50, 75]


def test_rsi_first_value_needs_period_plus_one_closes():
    assert rsi(list(range(14)), 14) == [None] * 14
    assert rsi(list(range(15)), 14)[-1] == 100


def test_rsi_extremes_and_flat_prices():
    assert rsi([float(i) for i in range(30)], 14)[-1] == 100
    assert rsi([float(30 - i) for i in range(30)], 14)[-1] == 0
    assert rsi([10.0] * 30, 14)[-1] is None  # undefined, not invented


def test_rsi_stays_in_bounds():
    closes = [100 + 10 * math.sin(i / 3) + i % 7 for i in range(200)]
    assert all(0 <= v <= 100 for v in rsi(closes, 14) if v is not None)


# --- MACD -------------------------------------------------------------------------------


def test_macd_hand_calculation():
    # fast=1 -> EMA is the input; slow=2 -> alpha 2/3, seed mean(1, 2) = 1.5
    # slow EMA at idx 2 = 2/3 * 4 + 1/3 * 1.5 = 3.1667 -> line = 0.5, 0.8333
    line, signal, hist = macd([1, 2, 4], fast=1, slow=2, signal=2)
    assert line == approx_list([None, 0.5, 0.8333333])
    assert signal == approx_list([None, None, 0.6666667])  # seed mean(0.5, 0.8333)
    assert hist == approx_list([None, None, 0.1666667])


def test_macd_alignment_with_default_periods():
    line, signal, hist = macd([float(i) for i in range(40)], 12, 26, 9)
    assert line[24] is None and line[25] is not None  # first at slow - 1
    assert signal[32] is None and signal[33] is not None  # first at slow + signal - 2
    assert hist[32] is None and hist[33] is not None


def test_macd_of_linear_ramp_is_constant_lag_difference():
    # For a ramp of slope 1, an SMA-seeded EMA lags by (period - 1) / 2 exactly,
    # so the MACD line is (26 - 1) / 2 - (12 - 1) / 2 = 7 and the histogram is 0.
    line, signal, hist = macd([float(i) for i in range(100)], 12, 26, 9)
    assert line[-1] == pytest.approx(7)
    assert signal[-1] == pytest.approx(7)
    assert hist[-1] == pytest.approx(0, abs=1e-9)


def test_macd_constant_series_is_zero():
    line, signal, hist = macd([50.0] * 60, 12, 26, 9)
    assert (line[-1], signal[-1], hist[-1]) == (0, 0, 0)


def test_macd_not_enough_values():
    line, signal, hist = macd([1.0] * 10, 12, 26, 9)
    assert line == signal == hist == [None] * 10


def test_macd_requires_fast_shorter_than_slow():
    with pytest.raises(ValueError):
        macd([1.0] * 50, 26, 12, 9)
