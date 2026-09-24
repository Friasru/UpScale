import asyncio
import json
from datetime import timedelta

import httpx2
import pytest

from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.market_data import Candle, MarketDataService, UnsupportedTimeframeError
from upscale.services.technical_analysis import (
    InvalidCandleDataError,
    TechnicalAnalysisConfig,
    TechnicalAnalysisService,
    analyze_series,
    classify_trend,
    recent_range,
    support_resistance,
    validate_candles,
)

from .conftest import FakeCoinGecko, ohlc_rows
from .ta_helpers import START, make_series, wave

CONFIG = TechnicalAnalysisConfig()


# --- Full analysis ------------------------------------------------------------------------


def test_analysis_reports_all_default_indicators():
    closes = [100 + i for i in range(180)]
    a = analyze_series(make_series(closes))
    assert [i.name for i in a.indicators] == [
        "SMA 20",
        "SMA 50",
        "EMA 20",
        "EMA 50",
        "RSI 14",
        "MACD 12/26/9",
    ]
    assert all(i.available for i in a.indicators)
    assert a.indicator("SMA 20").value == pytest.approx(sum(closes[-20:]) / 20)
    assert a.indicator("SMA 50").value == pytest.approx(sum(closes[-50:]) / 50)
    assert a.indicator("RSI 14").value == 100  # only gains
    macd = a.indicator("MACD 12/26/9")
    assert macd.value == pytest.approx(7) and macd.signal == pytest.approx(7)
    assert macd.params == {"fast": 12, "slow": 26, "signal": 9}


def test_analysis_keeps_candle_metadata():
    a = analyze_series(make_series([100.0 + i for i in range(60)], symbol="SOL"))
    assert (a.symbol, a.timeframe, a.provider, a.provider_id) == (
        "SOL",
        "4h",
        "TestProvider",
        "sol",
    )
    assert a.candle_count == 60
    assert a.first_candle_at == START
    assert a.last_candle_at == START + timedelta(hours=4 * 59)
    assert a.last_close == 159


def test_periods_are_configurable():
    config = TechnicalAnalysisConfig(
        sma_periods=(5,), ema_periods=(8, 13), rsi_period=7, macd_fast=3, macd_slow=6, macd_signal=4
    )
    a = analyze_series(make_series(wave(40)), config)
    assert [i.name for i in a.indicators] == ["SMA 5", "EMA 8", "EMA 13", "RSI 7", "MACD 3/6/4"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rsi_period": 0},
        {"sma_periods": (0,)},
        {"macd_fast": 26, "macd_slow": 12},
        {"trend_fast_sma": 50, "trend_slow_sma": 20},
        {"sr_lookback": 4, "sr_pivot_window": 3},
        {"sr_cluster_pct": 0},
        {"candles_to_fetch": 501},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TechnicalAnalysisConfig(**kwargs)


# --- Recent high/low ------------------------------------------------------------------------


def test_recent_high_low():
    closes = [100.0] * 60
    closes[5] = 130.0  # outside the last-50 window (candles 10-59)
    closes[30] = 120.0
    closes[45] = 90.0
    series = make_series(closes, spread=0)
    r = recent_range(series, 50)
    assert r.available
    assert r.high == 120.0 and r.high_at == series.candles[30].timestamp  # first of the tied highs
    assert r.low == 90.0 and r.low_at == series.candles[45].timestamp
    assert r.close_position_pct == pytest.approx(100 * (100 - 90) / (120 - 90))


def test_recent_range_needs_full_lookback():
    r = recent_range(make_series([100.0] * 30), 50)
    assert not r.available
    assert r.unavailable_reason == "needs 50 candles, only 30 available"


def test_recent_range_flat_prices_has_no_position():
    r = recent_range(make_series([100.0] * 50, spread=0), 50)
    assert r.high == r.low == 100.0
    assert r.close_position_pct is None


# --- Trend ------------------------------------------------------------------------------------


def test_uptrend():
    t = classify_trend([100.0 + i for i in range(60)], 20, 50)
    assert t.label == "uptrend"
    assert t.reasons == ["Close is above SMA 20.", "SMA 20 is above SMA 50."]


def test_downtrend():
    t = classify_trend([200.0 - i for i in range(60)], 20, 50)
    assert t.label == "downtrend"
    assert t.reasons == ["Close is below SMA 20.", "SMA 20 is below SMA 50."]


def test_mixed_trend_after_a_sharp_drop():
    closes = [100.0 + i for i in range(59)] + [100.0]  # rising, then last close falls back
    t = classify_trend(closes, 20, 50)
    assert t.label == "mixed"
    assert t.reasons == ["Close is below SMA 20.", "SMA 20 is above SMA 50."]


def test_flat_prices_are_mixed():
    t = classify_trend([100.0] * 60, 20, 50)
    assert t.label == "mixed"
    assert "equal to" in t.reasons[0]


def test_trend_needs_slow_sma_history():
    t = classify_trend([100.0 + i for i in range(49)], 20, 50)
    assert not t.available and t.label is None
    assert t.unavailable_reason == "needs 50 candles, only 49 available"
    assert "SMA 20" in t.method and "SMA 50" in t.method


# --- Support / resistance ----------------------------------------------------------------------


def test_support_and_resistance_from_oscillating_prices():
    closes = wave(120, mid=100, amplitude=10, period=20) + [100.0]
    s = support_resistance(make_series(closes, spread=0.2), CONFIG)
    assert s.available
    assert "approximate" in s.method
    [support] = s.support
    [resistance] = s.resistance
    assert support.price == pytest.approx(89.8, abs=0.5)  # wave low minus spread
    assert resistance.price == pytest.approx(110.2, abs=0.5)
    assert support.touches >= 4 and resistance.touches >= 4
    assert support.low <= support.price <= support.high
    assert support.distance_pct < 0 < resistance.distance_pct


def test_levels_are_sorted_nearest_first_and_capped():
    # Swing lows at 80, 85, 90, 95 and swing highs at 105, 110, 115, 120, all with close at 100.
    closes = []
    for low, high in [(80, 120), (85, 115), (90, 110), (95, 105)] * 2:
        closes += [100, low, 100, 100, high, 100, 100]
    closes += [100.0] * 10
    config = TechnicalAnalysisConfig(sr_lookback=len(closes), sr_pivot_window=1, sr_max_levels=3)
    s = support_resistance(make_series(closes, spread=0), config)
    assert [round(lv.price) for lv in s.support] == [95, 90, 85]
    assert [round(lv.price) for lv in s.resistance] == [105, 110, 115]
    assert all(lv.touches == 2 for lv in s.support + s.resistance)


def test_nearby_swing_points_are_grouped_within_tolerance():
    closes = []
    for low in (90.0, 90.5, 95.0):
        closes += [100, 100, low, 100, 100]
    closes += [100.0] * 5
    config = TechnicalAnalysisConfig(sr_lookback=len(closes), sr_pivot_window=2, sr_cluster_pct=1.0)
    s = support_resistance(make_series(closes, spread=0), config)
    assert [(round(lv.price, 2), lv.touches) for lv in s.support] == [(95.0, 1), (90.25, 2)]


def test_no_levels_in_a_steady_ramp():
    s = support_resistance(make_series([100.0 + i for i in range(120)]), CONFIG)
    assert s.available and s.support == [] and s.resistance == []


def test_support_resistance_needs_lookback():
    s = support_resistance(make_series(wave(60)), CONFIG)
    assert not s.available
    assert s.unavailable_reason == "needs 100 candles, only 60 available"


# --- Insufficient data -------------------------------------------------------------------------


def test_short_history_marks_indicators_unavailable_instead_of_guessing():
    a = analyze_series(make_series(wave(30)))
    by_name = {i.name: i for i in a.indicators}
    assert by_name["SMA 20"].available and by_name["EMA 20"].available
    assert by_name["RSI 14"].available
    for name, need in [("SMA 50", 50), ("EMA 50", 50), ("MACD 12/26/9", 34)]:
        assert not by_name[name].available
        assert by_name[name].value is None
        assert by_name[name].unavailable_reason == f"needs {need} candles, only 30 available"
    assert not a.trend.available
    assert not a.recent_range.available
    assert not a.levels.available


def test_very_short_history():
    a = analyze_series(make_series([100.0, 101.0]))
    assert not any(i.available for i in a.indicators)
    assert a.indicator("RSI 14").unavailable_reason == "needs 15 candles, only 2 available"


def test_flat_prices_make_rsi_undefined_not_fifty():
    rsi = analyze_series(make_series([100.0] * 60)).indicator("RSI 14")
    assert not rsi.available and rsi.value is None
    assert "did not change" in rsi.unavailable_reason


# --- Volume ------------------------------------------------------------------------------------


def test_missing_volume_is_reported_and_not_used():
    a = analyze_series(make_series(wave(60), volume_available=False))
    assert a.volume_available is False
    assert "does not supply per-candle volume" in a.volume_note
    assert {i.kind for i in a.indicators} <= {"sma", "ema", "rsi", "macd"}


def test_available_volume_is_reported_but_no_volume_indicators_exist_yet():
    a = analyze_series(make_series(wave(60), volume_available=True))
    assert a.volume_available is True
    assert "no volume-based indicators are implemented" in a.volume_note


# --- Invalid candles -----------------------------------------------------------------------------


def _with_candle(series, index, **changes):
    candles = list(series.candles)
    # model_construct skips validation so we can build candles a buggy provider might send.
    candles[index] = Candle.model_construct(**{**candles[index].model_dump(), **changes})
    return series.model_copy(update={"candles": candles})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"high": 50.0}, "inconsistent high/low"),
        ({"low": 500.0}, "inconsistent high/low"),
        ({"close": float("nan")}, "invalid prices"),
        ({"open": float("inf")}, "invalid prices"),
        ({"low": -1.0}, "invalid prices"),
    ],
)
def test_invalid_candle_values_are_rejected(changes, message):
    series = _with_candle(make_series(wave(30)), 10, **changes)
    with pytest.raises(InvalidCandleDataError, match=message):
        analyze_series(series)


def test_gapped_or_unordered_candles_are_rejected():
    series = make_series(wave(30))
    gapped = series.model_copy(update={"candles": series.candles[:10] + series.candles[11:]})
    with pytest.raises(InvalidCandleDataError, match="not consecutive"):
        validate_candles(gapped)
    reversed_ = series.model_copy(update={"candles": series.candles[::-1]})
    with pytest.raises(InvalidCandleDataError, match="not consecutive"):
        validate_candles(reversed_)


def test_wrong_timeframe_label_is_rejected():
    with pytest.raises(InvalidCandleDataError, match="not consecutive 1h candles"):
        validate_candles(make_series(wave(10), timeframe="1h"))  # spaced 4h apart


def test_empty_series_is_rejected():
    series = make_series(wave(5)).model_copy(update={"candles": []})
    with pytest.raises(InvalidCandleDataError, match="no candles"):
        analyze_series(series)


# --- Service --------------------------------------------------------------------------------------


@pytest.fixture
def fake() -> FakeCoinGecko:
    return FakeCoinGecko()


def make_service(fake, **config) -> TechnicalAnalysisService:
    market = MarketDataService(CoinGeckoProvider(transport=fake.transport()))
    return TechnicalAnalysisService(market, TechnicalAnalysisConfig(**config))


def test_service_fetches_candles_and_analyzes(fake):
    a = asyncio.run(make_service(fake).analyze("ETH", "4h"))
    assert a.symbol == "ETH" and a.provider == "CoinGecko" and a.provider_id == "ethereum"
    assert a.candle_count == 180
    assert a.volume_available is False
    [request] = fake.requests
    assert request.url.params["days"] == "30"


def test_service_analyzes_partial_history(fake):
    fake.handler = lambda r: httpx2.Response(200, content=json.dumps(ohlc_rows(20)))
    a = asyncio.run(make_service(fake).analyze("ETH", "4h"))
    assert a.candle_count == 20
    assert a.indicator("SMA 20").available and not a.indicator("SMA 50").available


@pytest.mark.parametrize(
    ("requested", "expected", "has_note"),
    [
        (None, "4h", False),
        ("4h", "4h", False),
        ("1h", "4h", True),
        ("1d", "4h", True),
        ("2h", "4h", True),
    ],
)
def test_resolve_timeframe(fake, requested, expected, has_note):
    timeframe, note = make_service(fake).resolve_timeframe(requested)
    assert timeframe == expected
    assert (note is not None) == has_note
    if note:
        assert note.startswith(f"{requested} candles are not available") and "4h" in note


def test_resolve_timeframe_without_candle_providers():
    class PriceOnly:
        name = "PriceOnly"

        async def fetch_snapshot(self, symbol):  # pragma: no cover
            raise AssertionError

    service = TechnicalAnalysisService(MarketDataService(PriceOnly()))
    with pytest.raises(UnsupportedTimeframeError):
        service.resolve_timeframe("4h")
