"""Deterministic technical analysis over `CandleSeries` from the market data service.

Computes SMA, EMA, RSI, MACD, a recent high/low range, a rule-based trend label and
approximate support/resistance zones, plus relative volume when the candle provider
supplies per-candle volume. Any calculation without enough candles (or without volume)
is reported as unavailable with the reason, never estimated.
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
    "side) over the last {n} candles. Sorted by price, a swing point joins the current zone "
    "when it is within {k:g} x ATR {atr_period} (the average true range at the last candle, "
    "{tol}) of the zone's lowest swing point, so no zone is wider than that. Each zone spans "
    "its lowest to highest grouped swing point; support lies wholly below the last close and "
    "resistance wholly above it. Levels are approximate zones, not exact prices."
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
    # Swing points group into one zone when within this many ATRs of the zone's lowest swing
    # point: one ATR is the typical full range of a single candle on this series, so swings
    # closer than that are the same area on the chart. Scales with timeframe and volatility.
    atr_period: int = 14
    sr_zone_atr_multiple: float = 1.0
    sr_max_levels: int = 3
    # Relative volume compares the last candle with the average of the ones before it.
    volume_lookback: int = 20
    # Candles requested from the market data service (180 = CoinGecko's 4h maximum, the
    # fallback when Kraken is unavailable; Kraken offers up to 719 per timeframe).
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
            self.atr_period,
            self.sr_max_levels,
            self.volume_lookback,
        ]
        if any(p < 1 for p in periods):
            raise ValueError("all periods, windows and counts must be at least 1")
        if not 1 <= self.macd_fast < self.macd_slow:
            raise ValueError("MACD fast period must be shorter than the slow period")
        if not 1 <= self.trend_fast_sma < self.trend_slow_sma:
            raise ValueError("trend fast SMA must be shorter than the slow SMA")
        if self.sr_lookback < 2 * self.sr_pivot_window + 1:
            raise ValueError("sr_lookback must fit at least one pivot window")
        if self.sr_zone_atr_multiple <= 0:
            raise ValueError("sr_zone_atr_multiple must be positive")
        if self.sr_lookback <= self.atr_period:
            raise ValueError("sr_lookback must exceed atr_period so the ATR is available")
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
    """A support/resistance zone built from grouped swing points."""

    price: float  # representative price: mean of the grouped swing points
    lower: float  # lower bound: lowest swing point in the zone
    upper: float  # upper bound: highest swing point in the zone
    touches: int
    last_touched: datetime
    distance_pct: float  # from the last close; negative = below


class SupportResistance(BaseModel):
    method: str
    lookback: int
    available: bool
    # Volatility used to group swing points: ATR at the last candle and the resulting
    # maximum zone width (atr * multiple), both in price units.
    atr_period: int | None = None
    atr: float | None = None
    zone_tolerance: float | None = None
    support: list[Level] = Field(default_factory=list)  # nearest first
    resistance: list[Level] = Field(default_factory=list)  # nearest first
    # The zone whose bounds contain the last close, if any: price is trading inside it, so
    # it is neither support below nor resistance above.
    containing: Level | None = None
    unavailable_reason: str | None = None


class VolumeAnalysis(BaseModel):
    """Per-candle volume in the base asset (e.g. BTC), as supplied by the candle provider."""

    lookback: int
    available: bool
    unit: str | None = None
    last_volume: float | None = None  # the most recent completed candle
    average_volume: float | None = None  # mean of the `lookback` candles before it
    relative_volume: float | None = None  # last / average
    # Share of the window's volume (last `lookback` + 1 candles) on candles closing above
    # their open, and below it; the remainder traded on unchanged candles.
    up_volume_pct: float | None = None
    down_volume_pct: float | None = None
    unavailable_reason: str | None = None


def _volume_not_computed() -> VolumeAnalysis:
    return VolumeAnalysis(lookback=0, available=False, unavailable_reason="not computed")


class TechnicalAnalysis(BaseModel):
    symbol: str
    timeframe: Timeframe
    provider: str
    provider_id: str
    pair: str | None = None  # e.g. "BTC/USD" when the provider quotes a traded pair
    # Why preferred candle providers were skipped before `provider` served the candles.
    fallback_notes: list[str] = Field(default_factory=list)
    candle_count: int
    first_candle_at: datetime
    last_candle_at: datetime  # open time of the most recent candle
    last_close: float
    # Close of the candle before the last one, so a break of a zone on the last close can be
    # told apart from price that was already beyond it. None with a single candle.
    previous_close: float | None = None
    volume_available: bool
    volume_note: str
    volume: VolumeAnalysis = Field(default_factory=_volume_not_computed)
    indicators: list[Indicator]
    recent_range: RecentRange
    trend: Trend
    levels: SupportResistance
    canonical_id: str | None = None  # exact asset, for contract/pool-keyed candles
    as_of: datetime | None = None  # see `CandleSeries.as_of`
    notes: list[str] = Field(default_factory=list)

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
        if c.volume is not None and not (math.isfinite(c.volume) and c.volume >= 0):
            raise InvalidCandleDataError(f"candle at {c.timestamp} has invalid volume")
        if series.volume_available and c.volume is None:
            raise InvalidCandleDataError(f"candle at {c.timestamp} is missing its volume")
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
    volume = volume_analysis(series, config.volume_lookback)
    if not series.volume_available:
        volume_note = (
            f"{series.provider} does not supply per-candle volume, so no volume-based "
            "analysis was performed."
        )
    elif volume.available:
        volume_note = f"Per-candle volume from {series.provider} ({volume.unit})."
    else:
        volume_note = (
            f"{series.provider} supplies per-candle volume, but relative volume is "
            f"unavailable: {volume.unavailable_reason}."
        )
    return TechnicalAnalysis(
        symbol=series.symbol,
        timeframe=series.timeframe,
        provider=series.provider,
        provider_id=series.provider_id,
        pair=series.pair,
        fallback_notes=series.fallback_notes,
        candle_count=len(series.candles),
        first_candle_at=series.candles[0].timestamp,
        last_candle_at=series.candles[-1].timestamp,
        last_close=closes[-1],
        previous_close=closes[-2] if len(closes) > 1 else None,
        volume_available=series.volume_available,
        volume_note=volume_note,
        volume=volume,
        indicators=indicator_list,
        recent_range=recent_range(series, config.high_low_lookback),
        trend=classify_trend(closes, config.trend_fast_sma, config.trend_slow_sma),
        levels=support_resistance(series, config),
        canonical_id=series.canonical_id,
        as_of=series.as_of,
        notes=series.notes,
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


def volume_analysis(series: CandleSeries, lookback: int) -> VolumeAnalysis:
    """Last candle's volume relative to the preceding `lookback` candles, and the up/down
    split of volume over the window. Only computed from provider-supplied volume."""
    unit = series.volume_unit or series.symbol
    if not series.volume_available:
        return VolumeAnalysis(
            lookback=lookback,
            available=False,
            unavailable_reason=f"{series.provider} does not supply per-candle volume",
        )
    candles = series.candles
    if len(candles) < lookback + 1:
        return VolumeAnalysis(
            lookback=lookback,
            available=False,
            unit=unit,
            unavailable_reason=_needs(lookback + 1, len(candles)),
        )
    window = candles[-(lookback + 1) :]
    volumes = [c.volume or 0.0 for c in window]
    last = volumes[-1]
    average = sum(volumes[:-1]) / lookback
    total = sum(volumes)
    if average <= 0 or total <= 0:
        return VolumeAnalysis(
            lookback=lookback,
            available=False,
            unit=unit,
            last_volume=last,
            unavailable_reason=f"no volume was traded in the previous {lookback} candles",
        )
    up = sum(v for c, v in zip(window, volumes, strict=True) if c.close > c.open)
    down = sum(v for c, v in zip(window, volumes, strict=True) if c.close < c.open)
    return VolumeAnalysis(
        lookback=lookback,
        available=True,
        unit=unit,
        last_volume=last,
        average_volume=average,
        relative_volume=last / average,
        up_volume_pct=100 * up / total,
        down_volume_pct=100 * down / total,
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
    w, lookback, k = config.sr_pivot_window, config.sr_lookback, config.sr_zone_atr_multiple
    candles = series.candles

    def method(tol: str) -> str:
        return SR_METHOD.format(w=w, n=lookback, k=k, atr_period=config.atr_period, tol=tol)

    if len(candles) < lookback:
        return SupportResistance(
            method=method("unavailable"),
            lookback=lookback,
            available=False,
            atr_period=config.atr_period,
            unavailable_reason=_needs(lookback, len(candles)),
        )

    atr = indicators.atr(
        [c.high for c in candles],
        [c.low for c in candles],
        [c.close for c in candles],
        config.atr_period,
    )[-1]
    if atr is None:  # unreachable: len(candles) >= sr_lookback > atr_period
        raise ValueError("ATR unavailable")
    tolerance = k * atr

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
        # Anchored to the zone's lowest swing point, so zones can't chain wider than `tolerance`.
        if clusters and price - clusters[-1][0][0] <= tolerance:
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
                lower=min(prices),
                upper=max(prices),
                touches=len(cluster),
                last_touched=max(at for _, at in cluster),
                distance_pct=100 * (mean - close) / close,
            )
        )
    # Clusters are disjoint price ranges, so at most one can contain the close. A zone is
    # classified by its bounds, not its mean: one straddling the close is neither side.
    support = sorted((lv for lv in levels if lv.upper < close), key=lambda lv: -lv.price)
    resistance = sorted((lv for lv in levels if lv.lower > close), key=lambda lv: lv.price)
    containing = next((lv for lv in levels if lv.lower <= close <= lv.upper), None)
    return SupportResistance(
        method=method(f"{tolerance:,.6g} in price units"),
        lookback=lookback,
        available=True,
        atr_period=config.atr_period,
        atr=atr,
        zone_tolerance=tolerance,
        support=support[: config.sr_max_levels],
        resistance=resistance[: config.sr_max_levels],
        containing=containing,
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
