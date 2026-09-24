"""Deterministic technical analysis over `CandleSeries` from the market data service.

Computes SMA, EMA, RSI, MACD, a recent high/low range, a rule-based trend label and
approximate support/resistance zones. Any calculation without enough candles is
reported as unavailable with the reason, never estimated. No volume-based indicators
are computed; volume availability is only reported.
"""

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from upscale.services import indicators
from upscale.services.market_data import (
    MAX_CANDLES,
    TIMEFRAMES,
    CandleSeries,
    MarketDataService,
    Timeframe,
    UnsupportedTimeframeError,
)

SR_METHOD = (
    "Swing highs/lows (a candle's high or low is the extreme of the {w} candles on each "
    "side) over the last {n} candles, grouped when within {tol}% of each other. "
    "Levels are approximate zones, not exact prices."
)


class InvalidCandleDataError(ValueError):
    """The candle series is unusable (unordered, gapped, or internally inconsistent)."""


@dataclass(frozen=True)
class TechnicalAnalysisConfig:
    """Every period and window used by the analysis. Change these, not the math."""

    sma_periods: tuple[int, ...] = (20, 50)
    ema_periods: tuple[int, ...] = (20, 50)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Trend rule compares the close with these two SMAs.
    trend_fast_sma: int = 20
    trend_slow_sma: int = 50
    high_low_lookback: int = 50
    sr_lookback: int = 100
    sr_pivot_window: int = 3
    sr_cluster_pct: float = 1.0
    sr_max_levels: int = 3
    # Candles requested from the market data service (180 = CoinGecko's 4h maximum).
    candles_to_fetch: int = 180
    default_timeframe: Timeframe = "4h"

    def __post_init__(self) -> None:
        periods = [
            *self.sma_periods,
            *self.ema_periods,
            self.rsi_period,
            self.macd_signal,
            self.high_low_lookback,
            self.sr_pivot_window,
            self.sr_max_levels,
        ]
        if any(p < 1 for p in periods):
            raise ValueError("all periods, windows and counts must be at least 1")
        if not 1 <= self.macd_fast < self.macd_slow:
            raise ValueError("MACD fast period must be shorter than the slow period")
        if not 1 <= self.trend_fast_sma < self.trend_slow_sma:
            raise ValueError("trend fast SMA must be shorter than the slow SMA")
        if self.sr_lookback < 2 * self.sr_pivot_window + 1:
            raise ValueError("sr_lookback must fit at least one pivot window")
        if self.sr_cluster_pct <= 0:
            raise ValueError("sr_cluster_pct must be positive")
        if not 1 <= self.candles_to_fetch <= MAX_CANDLES:
            raise ValueError(f"candles_to_fetch must be between 1 and {MAX_CANDLES}")


# --- Result models -----------------------------------------------------------------------


class Indicator(BaseModel):
    name: str  # e.g. "SMA 20", "MACD 12/26/9"
    kind: Literal["sma", "ema", "rsi", "macd"]
    params: dict[str, int]
    available: bool
    value: float | None = None  # SMA/EMA/RSI value, or the MACD line
    signal: float | None = None  # MACD only
    histogram: float | None = None  # MACD only
    unavailable_reason: str | None = None


class RecentRange(BaseModel):
    lookback: int
    available: bool
    high: float | None = None
    high_at: datetime | None = None
    low: float | None = None
    low_at: datetime | None = None
    # Where the last close sits within the range: 0 = at the low, 100 = at the high.
    close_position_pct: float | None = None
    unavailable_reason: str | None = None


class Trend(BaseModel):
    method: str
    available: bool
    label: Literal["uptrend", "downtrend", "mixed"] | None = None
    reasons: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class Level(BaseModel):
    price: float  # mean of the grouped swing points
    low: float  # lowest swing point in the zone
    high: float  # highest swing point in the zone
    touches: int
    last_touched: datetime
    distance_pct: float  # from the last close; negative = below


class SupportResistance(BaseModel):
    method: str
    lookback: int
    available: bool
    support: list[Level] = Field(default_factory=list)  # nearest first
    resistance: list[Level] = Field(default_factory=list)  # nearest first
    unavailable_reason: str | None = None


class TechnicalAnalysis(BaseModel):
    symbol: str
    timeframe: Timeframe
    provider: str
    provider_id: str
    candle_count: int
    first_candle_at: datetime
    last_candle_at: datetime  # open time of the most recent candle
    last_close: float
    volume_available: bool
    volume_note: str
    indicators: list[Indicator]
    recent_range: RecentRange
    trend: Trend
    levels: SupportResistance

    def indicator(self, name: str) -> Indicator | None:
        return next((i for i in self.indicators if i.name == name), None)


# --- Analysis ------------------------------------------------------------------------------


def validate_candles(series: CandleSeries) -> None:
    """Reject series that would make indicator math meaningless."""
    candles = series.candles
    if not candles:
        raise InvalidCandleDataError("no candles to analyze")
    for c in candles:
        prices = (c.open, c.high, c.low, c.close)
        if not all(math.isfinite(p) and p > 0 for p in prices):
            raise InvalidCandleDataError(f"candle at {c.timestamp} has invalid prices")
        if c.low > min(c.open, c.close) or c.high < max(c.open, c.close) or c.low > c.high:
            raise InvalidCandleDataError(f"candle at {c.timestamp} has inconsistent high/low")
    for a, b in itertools.pairwise(candles):
        if b.timestamp - a.timestamp != series.interval:
            raise InvalidCandleDataError(
                f"candles are not consecutive {series.timeframe} candles "
                f"(gap between {a.timestamp} and {b.timestamp})"
            )


def analyze_series(
    series: CandleSeries, config: TechnicalAnalysisConfig | None = None
) -> TechnicalAnalysis:
    """Run every configured calculation on a validated candle series."""
    config = config or TechnicalAnalysisConfig()
    validate_candles(series)
    closes = [c.close for c in series.candles]
    indicator_list = [
        *(_moving_average(closes, "sma", p) for p in config.sma_periods),
        *(_moving_average(closes, "ema", p) for p in config.ema_periods),
        _rsi(closes, config.rsi_period),
        _macd(closes, config),
    ]
    volume_note = (
        f"{series.provider} supplies per-candle volume, but no volume-based indicators are "
        "implemented yet."
        if series.volume_available
        else f"{series.provider} does not supply per-candle volume, so no volume-based "
        "analysis was performed."
    )
    return TechnicalAnalysis(
        symbol=series.symbol,
        timeframe=series.timeframe,
        provider=series.provider,
        provider_id=series.provider_id,
        candle_count=len(series.candles),
        first_candle_at=series.candles[0].timestamp,
        last_candle_at=series.candles[-1].timestamp,
        last_close=closes[-1],
        volume_available=series.volume_available,
        volume_note=volume_note,
        indicators=indicator_list,
        recent_range=recent_range(series, config.high_low_lookback),
        trend=classify_trend(closes, config.trend_fast_sma, config.trend_slow_sma),
        levels=support_resistance(series, config),
    )


def _needs(count: int, have: int) -> str:
    return f"needs {count} candles, only {have} available"


def _moving_average(closes: list[float], kind: Literal["sma", "ema"], period: int) -> Indicator:
    name = f"{kind.upper()} {period}"
    params = {"period": period}
    if len(closes) < period:
        return Indicator(
            name=name,
            kind=kind,
            params=params,
            available=False,
            unavailable_reason=_needs(period, len(closes)),
        )
    series = indicators.sma(closes, period) if kind == "sma" else indicators.ema(closes, period)
    return Indicator(name=name, kind=kind, params=params, available=True, value=series[-1])


def _rsi(closes: list[float], period: int) -> Indicator:
    name, params = f"RSI {period}", {"period": period}
    if len(closes) < period + 1:
        reason = _needs(period + 1, len(closes))
        return Indicator(
            name=name, kind="rsi", params=params, available=False, unavailable_reason=reason
        )
    value = indicators.rsi(closes, period)[-1]
    if value is None:
        reason = f"undefined: price did not change over the last {period} candles"
        return Indicator(
            name=name, kind="rsi", params=params, available=False, unavailable_reason=reason
        )
    return Indicator(name=name, kind="rsi", params=params, available=True, value=value)


def _macd(closes: list[float], config: TechnicalAnalysisConfig) -> Indicator:
    fast, slow, signal = config.macd_fast, config.macd_slow, config.macd_signal
    name = f"MACD {fast}/{slow}/{signal}"
    params = {"fast": fast, "slow": slow, "signal": signal}
    required = slow + signal - 1
    if len(closes) < required:
        reason = _needs(required, len(closes))
        return Indicator(
            name=name, kind="macd", params=params, available=False, unavailable_reason=reason
        )
    line, signal_line, histogram = indicators.macd(closes, fast, slow, signal)
    return Indicator(
        name=name,
        kind="macd",
        params=params,
        available=True,
        value=line[-1],
        signal=signal_line[-1],
        histogram=histogram[-1],
    )


def recent_range(series: CandleSeries, lookback: int) -> RecentRange:
    """Highest high and lowest low of the last `lookback` candles."""
    candles = series.candles
    if len(candles) < lookback:
        return RecentRange(
            lookback=lookback, available=False, unavailable_reason=_needs(lookback, len(candles))
        )
    window = candles[-lookback:]
    top = max(window, key=lambda c: c.high)
    bottom = min(window, key=lambda c: c.low)
    span = top.high - bottom.low
    close = candles[-1].close
    position = 100 * (close - bottom.low) / span if span > 0 else None
    return RecentRange(
        lookback=lookback,
        available=True,
        high=top.high,
        high_at=top.timestamp,
        low=bottom.low,
        low_at=bottom.timestamp,
        close_position_pct=position,
    )


def classify_trend(closes: list[float], fast: int, slow: int) -> Trend:
    """Uptrend if close > SMA fast > SMA slow, downtrend if close < SMA fast < SMA slow."""
    method = (
        f"Close vs SMA {fast} vs SMA {slow}: uptrend if close > SMA {fast} > SMA {slow}, "
        f"downtrend if close < SMA {fast} < SMA {slow}, otherwise mixed."
    )
    if len(closes) < slow:
        return Trend(method=method, available=False, unavailable_reason=_needs(slow, len(closes)))
    close = closes[-1]
    fast_value = indicators.sma(closes, fast)[-1]
    slow_value = indicators.sma(closes, slow)[-1]
    assert fast_value is not None and slow_value is not None  # guaranteed by the length check

    def relation(a: float, b: float) -> str:
        return "above" if a > b else "below" if a < b else "equal to"

    reasons = [
        f"Close is {relation(close, fast_value)} SMA {fast}.",
        f"SMA {fast} is {relation(fast_value, slow_value)} SMA {slow}.",
    ]
    if close > fast_value > slow_value:
        label: Literal["uptrend", "downtrend", "mixed"] = "uptrend"
    elif close < fast_value < slow_value:
        label = "downtrend"
    else:
        label = "mixed"
    return Trend(method=method, available=True, label=label, reasons=reasons)


def support_resistance(series: CandleSeries, config: TechnicalAnalysisConfig) -> SupportResistance:
    """Approximate levels from clustered swing highs/lows of recent price structure."""
    w, lookback, tol = config.sr_pivot_window, config.sr_lookback, config.sr_cluster_pct
    method = SR_METHOD.format(w=w, n=lookback, tol=tol)
    candles = series.candles
    if len(candles) < lookback:
        return SupportResistance(
            method=method,
            lookback=lookback,
            available=False,
            unavailable_reason=_needs(lookback, len(candles)),
        )

    window = candles[-lookback:]
    pivots: list[tuple[float, datetime]] = []
    # The last `w` candles can't be confirmed as swing points yet, so they are skipped.
    for i in range(w, len(window) - w):
        highs = [c.high for c in window[i - w : i + w + 1]]
        lows = [c.low for c in window[i - w : i + w + 1]]
        # index() == w keeps only the first candle when neighbours tie.
        if highs.index(max(highs)) == w:
            pivots.append((window[i].high, window[i].timestamp))
        if lows.index(min(lows)) == w:
            pivots.append((window[i].low, window[i].timestamp))

    close = candles[-1].close
    clusters: list[list[tuple[float, datetime]]] = []
    for price, at in sorted(pivots):
        if clusters and price <= clusters[-1][0][0] * (1 + tol / 100):
            clusters[-1].append((price, at))
        else:
            clusters.append([(price, at)])

    levels = []
    for cluster in clusters:
        prices = [p for p, _ in cluster]
        mean = sum(prices) / len(prices)
        levels.append(
            Level(
                price=mean,
                low=min(prices),
                high=max(prices),
                touches=len(cluster),
                last_touched=max(at for _, at in cluster),
                distance_pct=100 * (mean - close) / close,
            )
        )
    support = sorted((lv for lv in levels if lv.price < close), key=lambda lv: -lv.price)
    resistance = sorted((lv for lv in levels if lv.price > close), key=lambda lv: lv.price)
    return SupportResistance(
        method=method,
        lookback=lookback,
        available=True,
        support=support[: config.sr_max_levels],
        resistance=resistance[: config.sr_max_levels],
    )


# --- Service ---------------------------------------------------------------------------------


class TechnicalAnalysisService:
    """Fetches candles through `MarketDataService` and analyzes them."""

    def __init__(
        self, market_data: MarketDataService, config: TechnicalAnalysisConfig | None = None
    ):
        self.market_data = market_data
        self.config = config or TechnicalAnalysisConfig()

    def resolve_timeframe(self, requested: str | None) -> tuple[Timeframe, str | None]:
        """Pick the timeframe to analyze, plus a note if the requested one isn't available."""
        supported = self.market_data.supported_timeframes
        if not supported:
            raise UnsupportedTimeframeError("no market data provider offers candle data")
        match = next((tf for tf in TIMEFRAMES if tf == requested), None)
        if match in supported:
            return match, None
        fallback = (
            self.config.default_timeframe
            if self.config.default_timeframe in supported
            else supported[0]
        )
        if requested is None:
            return fallback, None
        note = (
            f"{requested} candles are not available from the configured market data provider; "
            f"analyzed {fallback} instead (available: {', '.join(supported)})."
        )
        return fallback, note

    async def analyze(self, symbol: str, timeframe: Timeframe) -> TechnicalAnalysis:
        """Raises `MarketDataError` if candles can't be fetched, `InvalidCandleDataError`
        if they're unusable. Partial history yields per-indicator "unavailable" entries."""
        series = await self.market_data.get_candles(
            symbol, timeframe, limit=self.config.candles_to_fetch, min_candles=1
        )
        return analyze_series(series, self.config)
