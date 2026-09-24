import asyncio
import json
import random
from datetime import timedelta

import httpx2
import pytest

from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.indicators import atr
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
from .ta_helpers import START, make_series, swing_series, wave

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
        {"sr_zone_atr_multiple": 0},
        {"sr_lookback": 14, "atr_period": 14},
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
    assert support.lower <= support.price <= support.upper
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


BASE = 84_000.0
# BTC-like ATR 14 per timeframe, measured on live Kraken candles around $84k.
BTC_ATR = {"1m": 28.0, "5m": 106.0, "15m": 262.0, "1h": 602.0, "4h": 1078.0}


def _levels(series):
    config = TechnicalAnalysisConfig(
        sr_lookback=len(series.candles), sr_pivot_window=2, sr_max_levels=10
    )
    return support_resistance(series, config)


def _bounds(levels):
    return [(lv.lower, lv.upper, lv.touches) for lv in levels]


def test_zone_tolerance_is_one_atr_at_the_last_candle():
    series = swing_series(BASE, 28.0, [BASE + 50, BASE - 50])
    s = support_resistance(series, TechnicalAnalysisConfig(sr_lookback=len(series.candles)))
    c = series.candles
    expected = atr([x.high for x in c], [x.low for x in c], [x.close for x in c], 14)[-1]
    assert s.atr_period == 14
    assert s.atr == pytest.approx(expected)
    assert s.zone_tolerance == pytest.approx(expected)
    assert s.atr == pytest.approx(28.0, rel=0.05)  # the series' typical candle range
    assert "ATR 14" in s.method
    doubled = TechnicalAnalysisConfig(sr_lookback=len(c), sr_zone_atr_multiple=2.0)
    assert support_resistance(series, doubled).zone_tolerance == pytest.approx(2 * expected)


@pytest.mark.parametrize("timeframe", list(BTC_ATR))
def test_distinct_zones_scale_with_each_timeframes_volatility(timeframe):
    r = BTC_ATR[timeframe]
    far_support = [BASE - 5.4 * r, BASE - 5.0 * r]
    near_support = [BASE - 2.6 * r, BASE - 2.3 * r, BASE - 2.0 * r]
    near_resistance = [BASE + 1.5 * r, BASE + 1.9 * r]
    far_resistance = [BASE + 4.0 * r]
    swings = far_support + near_support + near_resistance + far_resistance
    s = _levels(swing_series(BASE, r, swings, timeframe=timeframe))

    assert s.zone_tolerance == pytest.approx(r, rel=0.05)
    # Bounds are actual swing prices; distant clusters (2.4+ ATR apart) stay separate.
    assert _bounds(s.support) == [
        (near_support[0], near_support[-1], 3),
        (far_support[0], far_support[-1], 2),
    ]
    assert _bounds(s.resistance) == [
        (near_resistance[0], near_resistance[-1], 2),
        (far_resistance[0], far_resistance[0], 1),
    ]
    assert s.containing is None
    if timeframe == "1m":
        # Every swing here spans < 1% of price: the old fixed 1% rule merged them into one zone.
        assert max(swings) - min(swings) < 0.01 * min(swings)


def test_quiet_1m_zones_are_much_tighter_than_4h():
    swings = [BASE - 3 * BTC_ATR["4h"], BASE + 3 * BTC_ATR["4h"]]
    quiet_1m = _levels(swing_series(BASE, BTC_ATR["1m"], swings, timeframe="1m"))
    h4 = _levels(swing_series(BASE, BTC_ATR["4h"], swings, timeframe="4h"))
    assert quiet_1m.zone_tolerance is not None and h4.zone_tolerance is not None
    assert quiet_1m.zone_tolerance < h4.zone_tolerance / 20
    assert quiet_1m.zone_tolerance < 0.0005 * BASE  # well under the old 1% ($840)


def test_high_volatility_allows_wider_zones_than_low_volatility():
    swings = [BASE + 200, BASE + 230, BASE + 260]  # 30 apart
    quiet = _levels(swing_series(BASE, 20.0, swings))
    volatile = _levels(swing_series(BASE, 150.0, swings))
    assert _bounds(quiet.resistance) == [(p, p, 1) for p in swings]
    assert _bounds(volatile.resistance) == [(BASE + 200, BASE + 260, 3)]


def test_zones_do_not_chain_wider_than_the_tolerance():
    r = 28.0
    ladder = [BASE + r * (1.0 + 0.7 * i) for i in range(6)]  # each step within one ATR
    s = _levels(swing_series(BASE, r, ladder))
    assert s.zone_tolerance is not None
    assert _bounds(s.resistance) == [
        (ladder[0], ladder[1], 2),
        (ladder[2], ladder[3], 2),
        (ladder[4], ladder[5], 2),
    ]
    assert all(lv.upper - lv.lower <= s.zone_tolerance for lv in s.resistance)


def test_zone_containing_the_close_is_neither_support_nor_resistance():
    # Mean ~84,021 is below the 84,022.40 close, so a mean-based rule would call it support.
    r = 28.0
    zone = [BASE + 0.6 * r, BASE + 0.9 * r]
    s = _levels(swing_series(BASE, r, [BASE - 0.6 * r, *zone], last_close=BASE + 0.8 * r))
    assert s.resistance == []
    assert _bounds(s.support) == [(BASE - 0.6 * r, BASE - 0.6 * r, 1)]
    assert s.containing is not None
    assert (s.containing.lower, s.containing.upper) == (zone[0], zone[1])
    assert s.containing.price < BASE + 0.8 * r


@pytest.mark.parametrize("seed", range(20))
def test_levels_lie_wholly_on_their_side_of_the_close(seed):
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(149):
        closes.append(closes[-1] * (1 + rng.uniform(-0.01, 0.01)))
    s = support_resistance(make_series(closes, spread=0.3), CONFIG)
    close = closes[-1]
    for lv in [*s.support, *s.resistance, *([s.containing] if s.containing else [])]:
        assert lv.lower <= lv.price <= lv.upper
    assert all(lv.upper < close for lv in s.support)
    assert all(lv.lower > close for lv in s.resistance)
    assert s.containing is None or s.containing.lower <= close <= s.containing.upper


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


def test_available_volume_is_analyzed():
    series = make_series(wave(60), volume_available=True)
    volumes = [100.0] * 59 + [250.0]
    series.candles = [
        c.model_copy(update={"volume": v}) for c, v in zip(series.candles, volumes, strict=True)
    ]
    a = analyze_series(series)
    assert a.volume_available is True
    assert a.volume_note == "Per-candle volume from TestProvider (BTC)."
    v = a.volume
    assert v.available and v.unit == "BTC" and v.lookback == 20
    assert v.last_volume == 250.0
    assert v.average_volume == pytest.approx(100.0)
    assert v.relative_volume == pytest.approx(2.5)
    window = series.candles[-21:]
    up = sum(c.volume for c in window if c.close > c.open)
    down = sum(c.volume for c in window if c.close < c.open)
    total = sum(c.volume for c in window)
    assert v.up_volume_pct == pytest.approx(100 * up / total)
    assert v.down_volume_pct == pytest.approx(100 * down / total)


def test_volume_needs_enough_candles_and_nonzero_history():
    short = analyze_series(make_series(wave(15), volume_available=True))
    assert not short.volume.available
    assert short.volume.unavailable_reason == "needs 21 candles, only 15 available"
    assert "relative volume is unavailable" in short.volume_note

    series = make_series(wave(30), volume_available=True)
    series.candles = [c.model_copy(update={"volume": 0.0}) for c in series.candles]
    idle = analyze_series(series)
    assert not idle.volume.available
    assert "no volume was traded" in idle.volume.unavailable_reason


@pytest.mark.parametrize("volume", [float("inf"), float("nan"), None])
def test_invalid_or_missing_volume_is_rejected(volume):
    series = make_series(wave(30), volume_available=True)
    series.candles[5] = series.candles[5].model_copy(update={"volume": volume})
    with pytest.raises(InvalidCandleDataError, match="volume"):
        analyze_series(series)


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
