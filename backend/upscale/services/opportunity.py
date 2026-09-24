"""Deterministic BUY / SELL / WAIT decision from the evidence other agents already produced.

No network I/O and no model calls. Inputs are the structured findings of the Technical
Analysis, Market, News & Sentiment, Risk and Vision agents for one asset; evidence about
any other asset is ignored. The decision is made on the timeframe Technical Analysis
actually analyzed, and every trigger and invalidation is a close on that timeframe at a
real zone bound or moving average it computed. Nothing is estimated: no probabilities, no
targets, no percentage stops.

Actions:

* ``buy``: a confirmed bullish setup, strong enough to act on or prepare an entry.
* ``sell``: a confirmed bearish setup: reduce or exit a long position. It never means
  opening a short; shorting is not assessed.
* ``wait``: evidence is conflicting, incomplete, too risky, or not confirmed yet. WAIT is a
  full answer, not a failure, and is returned whenever BUY/SELL isn't clearly supported.

Rules (thresholds in `OpportunityConfig`):

1. The trend sets the only candidate direction: uptrend -> buy, downtrend -> sell, mixed or
   unavailable -> wait. News, volume or one indicator can never create a candidate.
2. Evidence scores points for its side: trend 2; MACD line vs signal 1; RSI on the trend's
   side of 50 and not stretched 1; a zone broken by the last close 2; volume skewed to up
   (or down) candles 1; the last candle's volume surging in its direction 1; overall news
   tone 1; a fresh high-impact story 1 (news is capped at 2 so it can't dominate).
3. BUY/SELL needs: no blocking factor for that side, at least `min_score` points for it
   (one more without a Risk review), and at least `min_margin` more than the other side.
4. Blocking factors: mixed/unavailable trend, MACD against the trend or unavailable, price
   inside a zone, price within `room_atr` ATRs of the next opposing zone (break not
   confirmed), a break on weak volume, RSI at an extreme in the trade's direction,
   fresh high-impact news against the setup or on both sides, high overall risk, high
   uncertainty, and a live price already beyond the invalidation level.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from upscale.formatting import usd, usd_zone
from upscale.schemas import AgentName, AgentResult, Level
from upscale.services.news import Story
from upscale.services.risk import InputState, RiskAssessment, collect_inputs
from upscale.services.technical_analysis import Level as PriceLevel
from upscale.services.technical_analysis import TechnicalAnalysis

Action = Literal["buy", "sell", "wait"]
Side = Literal["bullish", "bearish"]
AppliesTo = Literal["buy", "sell", "both"]
Setup = Literal["breakout", "breakdown", "trend_continuation"]
SignalCategory = Literal["trend", "momentum", "structure", "volume", "news"]
RiskLevel = Literal["low", "medium", "high", "unknown", "unavailable"]
UncertaintyLevel = Literal["low", "medium", "high", "unavailable"]
TriggerBasis = Literal[
    "resistance_zone_upper",
    "support_zone_lower",
    "containing_zone_upper",
    "containing_zone_lower",
    "last_close",
]

SELL_MEANING = (
    "SELL means reduce or exit a long position. It is not a signal to open a short position."
)
RULES = (
    "Deterministic rules: the trend sets the candidate (uptrend -> BUY, downtrend -> SELL, "
    "mixed -> WAIT). Evidence scores points (trend 2, MACD 1, RSI 1, zone break 2, volume "
    "up to 2, news up to 2). BUY/SELL needs no blocking factor, enough points and a clear "
    "margin over the opposing side; otherwise WAIT. Triggers and invalidations are closes "
    "on the analyzed timeframe at computed zone bounds or moving averages. No "
    "probabilities, targets or percentage stops."
)
_LEVELS: tuple[Level, ...] = ("low", "medium", "high")

TREND_POINTS = 2
MACD_POINTS = 1
RSI_POINTS = 1
BREAK_POINTS = 2
VOLUME_BIAS_POINTS = 1
VOLUME_SURGE_POINTS = 1
NEWS_TONE_POINTS = 1
NEWS_IMPACT_POINTS = 1


@dataclass(frozen=True)
class OpportunityConfig:
    """Every threshold used by the rules. Change these, not the logic."""

    min_score: int = 4  # points the candidate side needs (trend + MACD + one more)
    no_risk_extra_score: int = 1  # added to min_score when no Risk review is available
    min_margin: int = 3  # candidate points minus opposing points
    strong_margin: int = 6  # margin for high confidence
    rsi_stretched: float = 70.0  # RSI >= this (<= 100 - this for SELL): caution
    rsi_extreme: float = 80.0  # RSI >= this (<= 100 - this for SELL): blocks
    volume_bias_pct: float = 20.0  # up-volume % minus down-volume % needed for a side
    strong_relative_volume: float = 1.5  # last candle volume vs average: a surge
    weak_relative_volume: float = 0.7  # below this: weak volume
    room_atr: float = 1.0  # opposing zone within this many ATRs: break not confirmed

    def __post_init__(self) -> None:
        if self.min_score < 1 or self.min_margin < 1 or self.strong_margin < self.min_margin:
            raise ValueError("scores must be positive and strong_margin >= min_margin")
        if not 50 < self.rsi_stretched <= self.rsi_extreme < 100:
            raise ValueError("RSI thresholds must satisfy 50 < stretched <= extreme < 100")
        if not 0 < self.weak_relative_volume < 1 < self.strong_relative_volume:
            raise ValueError("relative volume thresholds must satisfy weak < 1 < strong")
        if self.room_atr < 0 or self.volume_bias_pct <= 0:
            raise ValueError("room_atr must be >= 0 and volume_bias_pct positive")


# --- Result models -----------------------------------------------------------------------


class Signal(BaseModel):
    """One piece of evidence pointing one way, with the points it scores."""

    id: str
    side: Side
    points: int
    category: SignalCategory
    clause: str  # short, for the summary
    source: AgentName


class Factor(BaseModel):
    """A reason to WAIT (blocker) or to lower confidence (caution)."""

    id: str
    kind: Literal["blocker", "caution"]
    applies_to: AppliesTo
    reason: str
    source: AgentName


class PriceZone(BaseModel):
    lower: float
    upper: float


class Trigger(BaseModel):
    """The close that would confirm (or did confirm) an action."""

    action: Literal["buy", "sell"]
    condition: str
    price: float  # the zone bound (or last close, for a confirmed continuation)
    basis: TriggerBasis
    zone: PriceZone | None = None
    confirmed: bool = False
    # Whether the live price is already beyond `price` (a close is still needed).
    live_price_beyond: bool | None = None


class Invalidation(BaseModel):
    condition: str
    price: float
    basis: Literal[
        "broken_zone_lower",
        "broken_zone_upper",
        "support_zone_lower",
        "resistance_zone_upper",
        "trend_average",
    ]
    timeframe: str


class OpportunityAssessment(BaseModel):
    asset: str | None
    timeframe: str | None  # the timeframe the decision is made on (as analyzed)
    requested_timeframe: str | None
    action: Action
    confidence: Level
    setup: Setup | None = None  # BUY/SELL only
    confirmed: bool  # True for BUY/SELL: the setup has closed; WAIT never is
    summary: str  # one sentence: why this action
    bullish_score: int
    bearish_score: int
    bullish_evidence: list[Signal]
    bearish_evidence: list[Signal]
    blocking_factors: list[Factor]
    cautions: list[Factor]
    bullish_trigger: Trigger | None
    bearish_trigger: Trigger | None
    # BUY: where to enter; SELL: where to exit a long (never a short entry).
    entry_zone: PriceZone | None = None
    entry_basis: str | None = None
    invalidation: Invalidation | None = None
    other_invalidations: list[Invalidation] = Field(default_factory=list)
    risk_level: RiskLevel
    uncertainty_level: UncertaintyLevel
    missing_evidence: list[str]
    last_close: float | None = None
    live_price: float | None = None
    context: list[str] = Field(default_factory=list)  # market/screenshot context, not scored
    inputs: list[InputState] = Field(default_factory=list)
    sell_meaning: str = SELL_MEANING
    rules: str = RULES


# --- Evidence ----------------------------------------------------------------------------


@dataclass
class _Evidence:
    asset: str
    technical: TechnicalAnalysis | None
    requested_timeframe: str | None
    live_price: float | None
    news_stories: list[Story] | None  # fresh, classified, about the asset; None = no report
    news_overall: str | None
    risk: RiskAssessment | None
    context: list[str]
    missing: list[str]
    inputs: list[InputState]


def _risk_review(prior: Mapping[AgentName, AgentResult], asset: str) -> RiskAssessment | None:
    result = prior.get("risk")
    if result is None or result.mock or result.status != "ok":
        return None
    try:
        review = RiskAssessment.model_validate(result.findings)
    except ValidationError:
        return None
    return review if review.asset == asset else None


def _gather(prior: Mapping[AgentName, AgentResult], asset: str) -> _Evidence:
    inputs = collect_inputs(prior, asset)
    context: list[str] = []
    missing: list[str] = []

    technical = inputs.technical
    extra = inputs.technical_extra or {}
    if technical is None:
        missing.append("Live technical analysis for this asset")

    snapshot = inputs.snapshot
    live_price = snapshot.price_usd if snapshot is not None else None
    if snapshot is None:
        missing.append("Live market snapshot")
    else:
        line = f"Live price {usd(snapshot.price_usd)} ({snapshot.provider})"
        if snapshot.low_24h_usd is not None and snapshot.high_24h_usd is not None:
            line += f"; 24h range {usd(snapshot.low_24h_usd)} – {usd(snapshot.high_24h_usd)}"
        context.append(line + ".")

    news = inputs.news if inputs.news is not None and inputs.news.asset == asset else None
    stories: list[Story] | None = None
    if news is None:
        missing.append("News & sentiment for this asset")
    else:
        stories = [s for s in news.focus if s.sentiment is not None and not s.stale]
        if not stories:
            missing.append("Fresh classified news for this asset")

    risk = _risk_review(prior, asset)
    if risk is None:
        missing.append("Risk review")

    chart = inputs.chart
    if chart is not None:
        shot_asset = chart.reading.asset.symbol
        if shot_asset not in (None, asset):
            context.append(f"The screenshot shows {shot_asset}, not {asset}; it was not used.")
        else:
            shot = chart.reading.displayed_price.value
            label = chart.normalized_timeframe or chart.reading.timeframe.label or "unknown"
            line = f"Screenshot ({label} chart)"
            if shot is not None:
                line += f" shows ~{usd(shot)}"
            context.append(line + "; supporting context only, live data decides.")
            if technical is not None and label != technical.timeframe:
                context.append(
                    f"The screenshot's timeframe ({label}) differs from the analyzed "
                    f"{technical.timeframe} candles."
                )

    requested = extra.get("requested_timeframe")
    return _Evidence(
        asset=asset,
        technical=technical,
        requested_timeframe=requested if isinstance(requested, str) else None,
        live_price=live_price,
        news_stories=stories,
        news_overall=news.overall if news is not None else None,
        risk=risk,
        context=context,
        missing=missing,
        inputs=list(inputs.states.values()),
    )


# --- Rules -------------------------------------------------------------------------------


@dataclass
class _Read:
    signals: list[Signal]
    factors: list[Factor]
    candidate: Literal["buy", "sell"] | None
    broken_up: PriceLevel | None = None  # zone the last close broke above
    broken_down: PriceLevel | None = None  # zone the last close broke below

    def add(
        self,
        id: str,
        side: Side,
        points: int,
        cat: SignalCategory,
        clause: str,
        source: AgentName = "technical_analysis",
    ) -> None:
        self.signals.append(
            Signal(id=id, side=side, points=points, category=cat, clause=clause, source=source)
        )

    def block(
        self, id: str, applies_to: AppliesTo, reason: str, source: AgentName = "technical_analysis"
    ) -> None:
        self.factors.append(
            Factor(id=id, kind="blocker", applies_to=applies_to, reason=reason, source=source)
        )

    def caution(
        self, id: str, applies_to: AppliesTo, reason: str, source: AgentName = "technical_analysis"
    ) -> None:
        self.factors.append(
            Factor(id=id, kind="caution", applies_to=applies_to, reason=reason, source=source)
        )


def _zone(lv: PriceLevel) -> str:
    return usd_zone(lv.lower, lv.upper)


def _technical_rules(t: TechnicalAnalysis, cfg: OpportunityConfig) -> _Read:
    tf = t.timeframe
    read = _Read(signals=[], factors=[], candidate=None)

    trend = t.trend.label if t.trend.available else None
    if trend == "uptrend":
        read.candidate = "buy"
        read.add("uptrend", "bullish", TREND_POINTS, "trend", "is in an uptrend")
    elif trend == "downtrend":
        read.candidate = "sell"
        read.add("downtrend", "bearish", TREND_POINTS, "trend", "is in a downtrend")
    elif trend == "mixed":
        read.block("mixed_trend", "both", f"the {tf} trend is mixed")
    else:
        read.block("trend_unavailable", "both", f"the {tf} trend is unavailable")

    macd = next((i for i in t.indicators if i.kind == "macd" and i.available), None)
    if macd is None or macd.value is None or macd.signal is None:
        read.block("macd_unavailable", "both", "MACD is unavailable")
    elif macd.value != macd.signal:
        up = macd.value > macd.signal
        side: Side = "bullish" if up else "bearish"
        read.add(
            "macd",
            side,
            MACD_POINTS,
            "momentum",
            f"MACD is {'above' if up else 'below'} its signal line",
        )
        if trend in ("uptrend", "downtrend") and up != (trend == "uptrend"):
            read.block(
                "macd_against_trend",
                "both",
                f"MACD is {'above' if up else 'below'} its signal line, against the {trend}",
            )

    rsi = next((i for i in t.indicators if i.kind == "rsi" and i.available), None)
    if rsi is not None and rsi.value is not None:
        r = rsi.value
        low_stretch, low_extreme = 100 - cfg.rsi_stretched, 100 - cfg.rsi_extreme
        if 50 < r < cfg.rsi_stretched:
            read.add(
                "rsi", "bullish", RSI_POINTS, "momentum", f"RSI {r:.0f} is firm, not stretched"
            )
        elif low_stretch < r < 50:
            read.add(
                "rsi", "bearish", RSI_POINTS, "momentum", f"RSI {r:.0f} is weak, not stretched"
            )
        if r >= cfg.rsi_extreme:
            read.block("rsi_extreme", "buy", f"RSI {r:.0f} is extremely overbought")
        elif r >= cfg.rsi_stretched:
            read.caution("rsi_stretched", "buy", f"RSI {r:.0f} is overbought")
        if r <= low_extreme:
            read.block("rsi_extreme", "sell", f"RSI {r:.0f} is extremely oversold")
        elif r <= low_stretch:
            read.caution("rsi_stretched", "sell", f"RSI {r:.0f} is oversold")

    _structure_rules(t, cfg, read)
    _volume_rules(t, cfg, read)
    return read


def _structure_rules(t: TechnicalAnalysis, cfg: OpportunityConfig, read: _Read) -> None:
    tf, close, prev, lv = t.timeframe, t.last_close, t.previous_close, t.levels
    if not lv.available:
        read.caution("levels_unavailable", "both", "support/resistance zones are unavailable")
        return
    support = lv.support[0] if lv.support else None
    resistance = lv.resistance[0] if lv.resistance else None
    if lv.containing is not None:
        read.block("inside_zone", "both", f"price is inside the zone {_zone(lv.containing)}")

    if prev is not None and support is not None and prev <= support.upper < close:
        read.broken_up = support
        read.add(
            "break_above",
            "bullish",
            BREAK_POINTS,
            "structure",
            f"closed above the zone {_zone(support)}",
        )
    if prev is not None and resistance is not None and prev >= resistance.lower > close:
        read.broken_down = resistance
        read.add(
            "break_below",
            "bearish",
            BREAK_POINTS,
            "structure",
            f"closed below the zone {_zone(resistance)}",
        )

    atr = lv.atr
    if atr is None:
        read.caution("atr_unavailable", "both", "no ATR to judge room to the next zone")
        return
    room = cfg.room_atr * atr
    if resistance is not None and lv.containing is None and resistance.lower - close <= room:
        read.block(
            "breakout_not_confirmed",
            "buy",
            f"price is just under resistance {_zone(resistance)}; no {tf} close above "
            f"~{usd(resistance.upper)} yet",
        )
    if support is not None and lv.containing is None and close - support.upper <= room:
        if read.broken_up is not support:  # a zone just broken upward is the BUY's retest area
            read.block(
                "breakdown_not_confirmed",
                "sell",
                f"price is just above support {_zone(support)}; no {tf} close below "
                f"~{usd(support.lower)} yet",
            )


def _volume_rules(t: TechnicalAnalysis, cfg: OpportunityConfig, read: _Read) -> None:
    v = t.volume
    if not (
        v.available
        and v.relative_volume is not None
        and v.up_volume_pct is not None
        and v.down_volume_pct is not None
    ):
        return
    rel = v.relative_volume
    bias = v.up_volume_pct - v.down_volume_pct
    if bias >= cfg.volume_bias_pct:
        read.add(
            "volume_bias",
            "bullish",
            VOLUME_BIAS_POINTS,
            "volume",
            f"{v.up_volume_pct:.0f}% of recent volume traded on up candles",
        )
    elif bias <= -cfg.volume_bias_pct:
        read.add(
            "volume_bias",
            "bearish",
            VOLUME_BIAS_POINTS,
            "volume",
            f"{v.down_volume_pct:.0f}% of recent volume traded on down candles",
        )

    prev = t.previous_close
    direction = 0 if prev is None else (t.last_close > prev) - (t.last_close < prev)
    if rel >= cfg.strong_relative_volume and direction:
        up = direction > 0
        read.add(
            "volume_surge",
            "bullish" if up else "bearish",
            VOLUME_SURGE_POINTS,
            "volume",
            f"the last {'up' if up else 'down'} close came on {rel:.1f}x average volume",
        )
    if rel < cfg.weak_relative_volume:
        weak = f"{rel:.1f}x the {v.lookback}-candle average"
        if read.broken_up is not None:
            read.block("weak_breakout_volume", "buy", f"the break came on weak volume ({weak})")
        if read.broken_down is not None:
            read.block("weak_breakdown_volume", "sell", f"the break came on weak volume ({weak})")
        read.caution("weak_volume", "both", f"volume is weak ({weak})")


def _quote(s: Story) -> str:
    return f"'{s.title}' ({s.source})"


def _news_rules(stories: list[Story], overall: str | None, read: _Read) -> None:
    if not stories:
        return
    tone: dict[str, Side] = {"bullish": "bullish", "bearish": "bearish"}
    if overall in tone:
        read.add(
            "news_tone",
            tone[overall],
            NEWS_TONE_POINTS,
            "news",
            f"recent news is {overall}",
            "news_sentiment",
        )
    elif overall == "mixed":
        read.caution("news_mixed", "both", "recent news is mixed", "news_sentiment")

    high_bull = [s for s in stories if s.sentiment == "bullish" and s.impact == "high"]
    high_bear = [s for s in stories if s.sentiment == "bearish" and s.impact == "high"]
    if high_bull and high_bear:
        read.block(
            "news_conflict",
            "both",
            f"fresh high-impact news points both ways ({_quote(high_bull[0])} vs "
            f"{_quote(high_bear[0])})",
            "news_sentiment",
        )
        return
    pairs: tuple[tuple[Side, list[Story], AppliesTo], ...] = (
        ("bullish", high_bull, "sell"),
        ("bearish", high_bear, "buy"),
    )
    for side, found, against in pairs:
        if found:
            read.add(
                "news_high_impact",
                side,
                NEWS_IMPACT_POINTS,
                "news",
                f"high-impact {side} story {_quote(found[0])}",
                "news_sentiment",
            )
            read.block(
                "news_against",
                against,
                f"fresh high-impact {side} news ({_quote(found[0])}) runs against it",
                "news_sentiment",
            )
    if overall == "bearish":
        read.caution("news_against", "buy", "recent news is bearish", "news_sentiment")
    elif overall == "bullish":
        read.caution("news_against", "sell", "recent news is bullish", "news_sentiment")


def _risk_rules(risk: RiskAssessment | None, read: _Read) -> None:
    if risk is None:
        return
    if risk.overall_risk in ("high", "unknown"):
        why = "; ".join(risk.overall_reasons[:2])
        read.block("risk_high", "both", f"overall risk is {risk.overall_risk} ({why})", "risk")
    if risk.uncertainty_level == "high":
        why = "; ".join(risk.uncertainty_reasons[:2]) or "evidence is unreliable"
        read.block("uncertainty_high", "both", f"uncertainty is high ({why})", "risk")


# Most decisive first: the summary names the first two blockers.
_BLOCKER_ORDER = (
    "no_technical",
    "risk_high",
    "uncertainty_high",
    "live_beyond_invalidation",
    "news_conflict",
    "news_against",
    "trend_unavailable",
    "mixed_trend",
    "inside_zone",
    "macd_unavailable",
    "macd_against_trend",
    "breakout_not_confirmed",
    "breakdown_not_confirmed",
    "weak_breakout_volume",
    "weak_breakdown_volume",
    "rsi_extreme",
    "balanced_evidence",
    "not_enough_evidence",
)


def _blocker_rank(f: Factor) -> int:
    return _BLOCKER_ORDER.index(f.id) if f.id in _BLOCKER_ORDER else len(_BLOCKER_ORDER)


# --- Triggers and invalidation -----------------------------------------------------------


def _fast_average(t: TechnicalAnalysis) -> tuple[str, float] | None:
    smas = [i for i in t.indicators if i.kind == "sma" and i.available and i.value is not None]
    if not smas:
        return None
    fast = min(smas, key=lambda i: i.params.get("period", 0))
    assert fast.value is not None
    return fast.name, fast.value


def _beyond(live: float | None, price: float, above: bool) -> bool | None:
    if live is None:
        return None
    return live > price if above else live < price


def _reference_triggers(
    t: TechnicalAnalysis, read: _Read, live: float | None
) -> tuple[Trigger | None, Trigger | None]:
    """The closes that would confirm a BUY / SELL from here (zone bounds only)."""
    lv, tf = t.levels, t.timeframe
    if not lv.available:
        return None, None
    macd_bull = any(s.id == "macd" and s.side == "bullish" for s in read.signals)
    macd_bear = any(s.id == "macd" and s.side == "bearish" for s in read.signals)

    def trigger(action: Literal["buy", "sell"], level: PriceLevel, containing: bool) -> Trigger:
        above = action == "buy"
        price = level.upper if above else level.lower
        kind = "price" if containing else "resistance" if above else "support"
        basis: TriggerBasis = (
            ("containing_zone_upper" if above else "containing_zone_lower")
            if containing
            else ("resistance_zone_upper" if above else "support_zone_lower")
        )
        condition = (
            f"{tf} close {'above' if above else 'below'} ~{usd(price)} "
            f"({'upper' if above else 'lower'} bound of the {kind} zone {_zone(level)})"
        )
        wants = "above" if above else "below"
        if not (macd_bull if above else macd_bear):
            condition += f", with MACD turning {wants} its signal line"
        if t.volume.available:
            condition += " and supporting volume"
        return Trigger(
            action=action,
            condition=condition,
            price=price,
            basis=basis,
            zone=PriceZone(lower=level.lower, upper=level.upper),
            live_price_beyond=_beyond(live, price, above),
        )

    if lv.containing is not None:
        return trigger("buy", lv.containing, True), trigger("sell", lv.containing, True)
    up = trigger("buy", lv.resistance[0], False) if lv.resistance else None
    down = trigger("sell", lv.support[0], False) if lv.support else None
    return up, down


@dataclass
class _Plan:
    setup: Setup
    trigger: Trigger
    invalidation: Invalidation | None
    others: list[Invalidation]
    entry_zone: PriceZone | None = None
    entry_basis: str | None = None


def _plan(t: TechnicalAnalysis, read: _Read, action: Literal["buy", "sell"]) -> _Plan:
    """Confirmed trigger, entry zone and invalidation for a BUY/SELL candidate."""
    tf, close, lv = t.timeframe, t.last_close, t.levels
    buy = action == "buy"
    broken = read.broken_up if buy else read.broken_down
    average = _fast_average(t)
    trend_inv = (
        Invalidation(
            condition=(
                f"{tf} close {'below' if buy else 'above'} {average[0]} (~{usd(average[1])}), "
                f"ending the {'uptrend' if buy else 'downtrend'} classification"
            ),
            price=average[1],
            basis="trend_average",
            timeframe=tf,
        )
        if average is not None
        else None
    )

    if broken is not None:
        bound = broken.upper if buy else broken.lower
        trigger = Trigger(
            action=action,
            condition=(
                f"confirmed: {tf} close {usd(close)} {'above' if buy else 'below'} "
                f"~{usd(bound)} ({'upper' if buy else 'lower'} bound of {_zone(broken)})"
            ),
            price=bound,
            basis="resistance_zone_upper" if buy else "support_zone_lower",
            zone=PriceZone(lower=broken.lower, upper=broken.upper),
            confirmed=True,
        )
        inv_price = broken.lower if buy else broken.upper
        broken_inv = Invalidation(
            condition=(
                f"{tf} close back {'below' if buy else 'above'} ~{usd(inv_price)} "
                f"({'lower' if buy else 'upper'} bound of the broken zone {_zone(broken)})"
            ),
            price=inv_price,
            basis="broken_zone_lower" if buy else "broken_zone_upper",
            timeframe=tf,
        )
        return _Plan(
            setup="breakout" if buy else "breakdown",
            trigger=trigger,
            invalidation=broken_inv,
            others=[trend_inv] if trend_inv else [],
            entry_zone=PriceZone(lower=broken.lower, upper=broken.upper),
            entry_basis=(
                "retest of the broken zone" if buy else "retest of the broken zone, to exit longs"
            ),
        )

    trigger = Trigger(
        action=action,
        condition=(
            f"confirmed: {tf} close {usd(close)} in a {'uptrend' if buy else 'downtrend'} "
            f"with MACD {'above' if buy else 'below'} its signal line"
        ),
        price=close,
        basis="last_close",
        confirmed=True,
    )
    level = (
        (lv.support[0] if lv.support else None)
        if buy
        else (lv.resistance[0] if lv.resistance else None)
    )
    if level is not None:
        inv_price = level.lower if buy else level.upper
        invalidation: Invalidation | None = Invalidation(
            condition=(
                f"{tf} close {'below' if buy else 'above'} ~{usd(inv_price)} "
                f"({'lower' if buy else 'upper'} bound of the "
                f"{'support' if buy else 'resistance'} zone {_zone(level)})"
            ),
            price=inv_price,
            basis="support_zone_lower" if buy else "resistance_zone_upper",
            timeframe=tf,
        )
        others = [trend_inv] if trend_inv else []
    else:
        invalidation, others = trend_inv, []
    return _Plan(
        setup="trend_continuation", trigger=trigger, invalidation=invalidation, others=others
    )


# --- Decision ----------------------------------------------------------------------------


def _confidence(
    action: Action,
    margin: int,
    cautions: Sequence[Factor],
    ev: _Evidence,
    cfg: OpportunityConfig,
) -> Level:
    t, risk = ev.technical, ev.risk
    news_ok = bool(ev.news_stories)
    volume_ok = t is not None and t.volume.available
    if action == "wait":
        complete = t is not None and risk is not None and ev.live_price is not None and news_ok
        solid = complete and risk is not None and risk.uncertainty_level != "high"
        return "medium" if solid else "low"

    rank = 2 if margin >= cfg.strong_margin else 1
    rank -= len(cautions)
    rank -= sum(
        [
            risk is not None and risk.uncertainty_level == "medium",
            ev.live_price is None,
            not news_ok,
            not volume_ok,
        ]
    )
    if risk is None or (t is not None and ev.requested_timeframe not in (None, t.timeframe)):
        rank = min(rank, 0)
    elif risk.overall_risk == "medium":
        rank = min(rank, 1)
    return _LEVELS[max(0, min(2, rank))]


def _summary(
    asset: str, tf: str | None, action: Action, signals: list[Signal], blockers: list[Factor]
) -> str:
    subject = f"{asset} {tf}" if tf else asset
    if action == "wait":
        reasons = [b.reason for b in blockers[:2]] or ["the evidence does not confirm a setup"]
        return f"{subject}: {'; '.join(reasons)}."
    ranked = sorted(signals, key=lambda s: -s.points)[:3]
    return f"{subject} " + ", ".join(s.clause for s in ranked) + "."


def assess(
    prior: Mapping[AgentName, AgentResult],
    asset: str | None,
    config: OpportunityConfig | None = None,
) -> OpportunityAssessment:
    cfg = config or OpportunityConfig()
    if asset is None:
        return OpportunityAssessment(
            asset=None,
            timeframe=None,
            requested_timeframe=None,
            action="wait",
            confidence="low",
            confirmed=False,
            summary="No specific cryptocurrency was identified, so there is nothing to decide on.",
            bullish_score=0,
            bearish_score=0,
            bullish_evidence=[],
            bearish_evidence=[],
            blocking_factors=[
                Factor(
                    id="no_asset",
                    kind="blocker",
                    applies_to="both",
                    reason="no specific cryptocurrency was identified",
                    source="opportunity",
                )
            ],
            cautions=[],
            bullish_trigger=None,
            bearish_trigger=None,
            risk_level="unavailable",
            uncertainty_level="unavailable",
            missing_evidence=["An asset to analyze (name a coin, e.g. BTC)"],
        )

    ev = _gather(prior, asset)
    t, risk = ev.technical, ev.risk
    risk_level: RiskLevel = risk.overall_risk if risk else "unavailable"
    uncertainty: UncertaintyLevel = risk.uncertainty_level if risk else "unavailable"

    if t is None:
        blocker = Factor(
            id="no_technical",
            kind="blocker",
            applies_to="both",
            reason=f"there is no live technical analysis for {asset}",
            source="opportunity",
        )
        return OpportunityAssessment(
            asset=asset,
            timeframe=None,
            requested_timeframe=ev.requested_timeframe,
            action="wait",
            confidence="low",
            confirmed=False,
            summary=_summary(asset, None, "wait", [], [blocker]),
            bullish_score=0,
            bearish_score=0,
            bullish_evidence=[],
            bearish_evidence=[],
            blocking_factors=[blocker],
            cautions=[],
            bullish_trigger=None,
            bearish_trigger=None,
            risk_level=risk_level,
            uncertainty_level=uncertainty,
            missing_evidence=ev.missing,
            live_price=ev.live_price,
            context=ev.context,
            inputs=ev.inputs,
        )

    read = _technical_rules(t, cfg)
    if ev.news_stories:
        _news_rules(ev.news_stories, ev.news_overall, read)
    _risk_rules(risk, read)
    if ev.requested_timeframe not in (None, t.timeframe):
        read.caution(
            "timeframe_fallback",
            "both",
            f"{ev.requested_timeframe} was requested but {t.timeframe} candles were analyzed",
        )
    if not t.volume.available:
        ev.missing.append("Per-candle volume")

    bull = sum(s.points for s in read.signals if s.side == "bullish")
    bear = sum(s.points for s in read.signals if s.side == "bearish")
    candidate = read.candidate
    plan = _plan(t, read, candidate) if candidate else None

    if candidate and plan and plan.invalidation is not None and ev.live_price is not None:
        inv = plan.invalidation
        below = candidate == "buy"
        if _beyond(ev.live_price, inv.price, not below):
            read.block(
                "live_beyond_invalidation",
                candidate,
                f"the live price {usd(ev.live_price)} is already "
                f"{'below' if below else 'above'} the invalidation level ~{usd(inv.price)}",
                "market",
            )

    action: Action = "wait"
    margin = 0
    if candidate is not None:
        ours, theirs = (bull, bear) if candidate == "buy" else (bear, bull)
        margin = ours - theirs
        needed = cfg.min_score + (cfg.no_risk_extra_score if risk is None else 0)
        blocked = any(
            f.kind == "blocker" and f.applies_to in (candidate, "both") for f in read.factors
        )
        if blocked:
            pass
        elif ours < needed:
            read.block(
                "not_enough_evidence",
                candidate,
                f"not enough supporting evidence yet ({ours} of {needed} points)",
                "opportunity",
            )
        elif margin < cfg.min_margin:
            read.block(
                "balanced_evidence",
                candidate,
                f"bullish and bearish evidence are roughly balanced ({bull} vs {bear} points)",
                "opportunity",
            )
        else:
            action = candidate

    # Without a candidate (no clear trend) only blockers against any action are relevant.
    relevant = (candidate, "both") if candidate else ("both",)
    blockers = sorted(
        (f for f in read.factors if f.kind == "blocker" and f.applies_to in relevant),
        key=_blocker_rank,
    )
    cautions = [f for f in read.factors if f.kind == "caution" and f.applies_to in relevant]
    side_signals = [
        s for s in read.signals if s.side == ("bullish" if action == "buy" else "bearish")
    ]
    bull_trigger, bear_trigger = _reference_triggers(t, read, ev.live_price)
    if action == "buy" and plan is not None:
        bull_trigger = plan.trigger
    if action == "sell" and plan is not None:
        bear_trigger = plan.trigger

    acting = action != "wait" and plan is not None
    return OpportunityAssessment(
        asset=asset,
        timeframe=t.timeframe,
        requested_timeframe=ev.requested_timeframe,
        action=action,
        confidence=_confidence(action, margin, cautions, ev, cfg),
        setup=plan.setup if acting and plan else None,
        confirmed=acting,
        summary=_summary(asset, t.timeframe, action, side_signals, blockers),
        bullish_score=bull,
        bearish_score=bear,
        bullish_evidence=[s for s in read.signals if s.side == "bullish"],
        bearish_evidence=[s for s in read.signals if s.side == "bearish"],
        blocking_factors=blockers if action == "wait" else [],
        cautions=cautions,
        bullish_trigger=bull_trigger,
        bearish_trigger=bear_trigger,
        entry_zone=plan.entry_zone if acting and plan else None,
        entry_basis=plan.entry_basis if acting and plan else None,
        invalidation=plan.invalidation if acting and plan else None,
        other_invalidations=plan.others if acting and plan else [],
        risk_level=risk_level,
        uncertainty_level=uncertainty,
        missing_evidence=ev.missing,
        last_close=t.last_close,
        live_price=ev.live_price,
        context=ev.context,
        inputs=ev.inputs,
    )
