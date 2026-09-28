"""ScoutMomentumScore, the growth stage, risk flags and the reasons a candidate surfaced.
Pure and deterministic: the same evidence always gives the same result.

**Score.** Five independent families, each normalized to 0..1 (see `signals`):

    base  = 100 x Σ weight_f x score_f          (weights normalized to sum to 1)
    score = clamp(base + stage_adjustment - risk_penalty, 0, 100)

Every term is exposed (`ScoutMomentumScore.components`) so a move up or down can be
explained.

* market_activity: volume and trade acceleration, buy pressure (level and change,
  trades and distinct wallets), 1h price momentum, lightweight technical context.
* liquidity_quality: absolute depth (log-capped), growth, stability, liquidity relative
  to market cap / FDV. A tiny market cap never makes thin liquidity look good.
* social_momentum: supporting evidence only. Missing or quiet social is `neutral`, never
  zero; positive evidence is scaled down by weak attribution (ticker-only chatter) and
  spam; raw mention counts are never used.
* earliness: how early we appear to be in a real, developing move: how much of the move
  already happened (a move largely made caps the family; a decline is neutral), pool age
  (very new is not the best), whether activity only recently picked up, and market size
  as minor context credited only for a healthy market (skipped in ALL_TRENDING mode).
  Little stored history damps credit above neutral; a MARKET_COLLAPSE gets none.
* cross_confirmation: how many independent market indicators agree, plus social
  agreeing (worth at most 15% of this family: market-only evidence stays competitive);
  contradictions (volume up while buyers fade) subtract.

A family with no evidence at all uses a documented stand-in (market: below flat, since
unmeasured activity is uncertainty; social: neutral) and is marked `available=False`.

**Stage adjustment** (validated config): ACCELERATING > EARLY > promising NEW > STEADY >
CROWDED > FADING > INSUFFICIENT_DATA, so a fading token doesn't outrank a similarly strong
accelerating one on stale earliness / liquidity points. The stage is stabilized first: a
one-run reversal needs stronger evidence (`stabilize`).

**Flow** is read from trade counts, never dollar flow: a very weak buy count while
activity / price rise (divergence), many buys from few wallets, or buys without a price
response put the flow in doubt: no "buyers strengthening", capped buy pressure, a
cross-confirmation contradiction and (divergence) a penalty.

**Risk penalty** (points, capped): market collapse, critical / high on-chain findings,
flow divergence, thin-market pumps,
social-only hype, distribution, heavy spam, draining liquidity, unrecognized quotes,
missing safety data (uncertainty), and smaller data-quality flags.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.growth.models import (
    FamilyName,
    FamilyScore,
    GrowthCandidate,
    GrowthMarket,
    GrowthMomentum,
    GrowthQuality,
    GrowthStage,
    RiskFlag,
    SafetyStatus,
    ScoutMomentumScore,
    Severity,
    SubSignal,
    VerificationStatus,
)
from upscale.services.scout.growth.signals import (
    MarketEvidence,
    SocialEvidence,
    clamp,
    linear_score,
    log_band_score,
    mean,
    pct_score,
    ratio_score,
)
from upscale.services.scout.models import ScoutCandidate
from upscale.services.scout.normalize import canonical_id, identity_problem
from upscale.services.solana_chain import OnchainSafetySnapshot

MISSING_MARKET_SCORE = 0.3  # no trusted acceleration measurement: below flat (uncertainty)
_VERIFY_IDENTITY = ScoutConfig()


# --- Qualitative reading of the evidence ----------------------------------------------------


@dataclass
class Reading:
    """Directional judgments shared by the stage, cross-confirmation, risks and reasons."""

    volume_rising: bool = False
    volume_falling: bool = False
    txns_rising: bool = False
    txns_falling: bool = False
    buyers_strengthening: bool = False
    buyers_weakening: bool = False
    price_rising: bool = False
    price_weakening: bool = False
    vertical: bool = False
    liquidity_confirming: bool = False
    liquidity_draining: bool = False
    distribution: bool = False
    social_only_hype: bool = False
    thin_pump: bool = False
    market_flat: bool = False
    timescales_rising: list[str] = field(default_factory=list)
    timescales_falling: list[str] = field(default_factory=list)
    collapse: list[str] = field(default_factory=list)  # MARKET_COLLAPSE evidence
    flow_doubts: list[str] = field(default_factory=list)  # why trade-count flow is in doubt
    flow_divergent: bool = False  # weak buy count while volume / trades / price rise
    flow_quality: str = "unknown"

    @property
    def collapsed(self) -> bool:
        return bool(self.collapse)

    @property
    def indicators_rising(self) -> int:
        return self.volume_rising + self.txns_rising + self.buyers_strengthening


def _geo_mean(values: list[float]) -> float | None:
    positive = [v for v in values if v > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def read(
    c: ScoutCandidate, ev: MarketEvidence, social: SocialEvidence, cfg: GrowthConfig
) -> Reading:
    mc, rc = cfg.market, cfg.risk
    r = Reading()
    for metric in ("volume", "txns"):
        gm = _geo_mean([v for _, v in ev.ratios(metric)])
        rising = gm is not None and gm >= mc.rising_ratio
        falling = gm is not None and gm <= mc.falling_ratio
        if metric == "volume":
            r.volume_rising, r.volume_falling = rising, falling
        else:
            r.txns_rising, r.txns_falling = rising, falling
    for t in ev.timescales:
        ratios = [x for x in (t.volume_ratio, t.txn_ratio) if x is not None]
        if ratios and max(ratios) >= mc.rising_ratio:
            r.timescales_rising.append(t.name)
        if ratios and max(ratios) <= mc.falling_ratio:
            r.timescales_falling.append(t.name)
    all_ratios = [v for m in ("volume", "txns") for _, v in ev.ratios(m)]
    r.market_flat = bool(all_ratios) and max(all_ratios) <= mc.flat_ratio

    change = ev.buy_share_change
    if change is not None:
        r.buyers_strengthening = change >= mc.buyers_strengthening_change
        r.buyers_weakening = change <= mc.buyers_weakening_change
    elif ev.buy_share is not None:
        r.buyers_strengthening = ev.buy_share >= 0.55
        r.buyers_weakening = ev.buy_share < 0.40

    if ev.price_h1 is not None:
        r.price_rising = ev.price_h1 >= 2.0
        r.price_weakening = ev.price_h1 <= -5.0 or (
            ev.price_h1 < 0 and (ev.price_velocity_change or 0) < 0
        )
        liquidity_kept_up = ev.liquidity_change_pct is not None and (
            ev.liquidity_change_pct >= ev.price_h1 / 4
        )
        r.vertical = ev.price_h1 >= cfg.stage.vertical_h1_pct and not liquidity_kept_up

    liquidity = c.metrics.liquidity_usd
    if ev.liquidity_min_change_pct is not None:
        r.liquidity_draining = ev.liquidity_min_change_pct <= -rc.liquidity_drain_pct
    if liquidity is not None and not r.liquidity_draining:
        growing = ev.liquidity_change_pct is not None and ev.liquidity_change_pct >= 10
        steady = ev.liquidity_change_pct is None or ev.liquidity_change_pct >= -5
        r.liquidity_confirming = growing or (liquidity >= cfg.liquidity.healthy_min_usd and steady)

    _read_flow(ev, r, cfg)
    r.collapse = collapse_evidence(c, ev, cfg)

    short = next((t for t in ev.timescales if t.name == "short"), None)
    sells_accelerating = short is not None and (short.sells_ratio or 0) >= mc.rising_ratio
    r.distribution = r.volume_rising and r.buyers_weakening and sells_accelerating
    r.social_only_hype = social.rising and r.market_flat
    moved = max(ev.price_h24 or 0.0, ev.price_h1 or 0.0) >= rc.thin_pump_price_h24_pct or (
        r.vertical
    )
    r.thin_pump = (
        moved
        and liquidity is not None
        and liquidity < rc.thin_pump_liquidity_usd
        and (ev.txns_h24 or 0) < rc.thin_pump_txns_h24
    )
    return r


def _read_flow(ev: MarketEvidence, r: Reading, cfg: GrowthConfig) -> None:
    """Trade counts are not dollar flow: bots and very different trade sizes distort them.
    Buyer strength needs a real buy level, wallets that agree and a price response;
    otherwise the flow is in doubt, never read as healthy buying."""
    fc = cfg.flow
    share, wallets = ev.buy_share, ev.buyer_share
    if share is None:
        r.flow_quality = "unknown"
        return
    moving = r.volume_rising or r.txns_rising or r.price_rising
    if share <= fc.weak_buy_share and moving:
        r.flow_divergent = True
        r.flow_doubts.append(
            f"only {share:.0%} of trades are buys while activity / price rise: the move "
            "isn't carried by visible buyers (distribution, or counts distorted)"
        )
    if share >= fc.bot_buy_share and wallets is not None and wallets <= fc.bot_max_buyer_share:
        r.flow_doubts.append(
            f"{share:.0%} of trades are buys but only {wallets:.0%} of wallets buy: many "
            "small buys from few wallets"
        )
    if (
        share >= fc.unanswered_buy_share
        and ev.price_h1 is not None
        and ev.price_h1 <= 0
        and not (ev.liquidity_change_pct or 0) > 0
    ):
        r.flow_doubts.append(f"{share:.0%} buys without any price response")
    if share < fc.min_strength_buy_share or r.flow_doubts:
        r.buyers_strengthening = False
    r.flow_quality = (
        "divergent" if r.flow_doubts else "count_only" if wallets is None else "consistent"
    )


def collapse_evidence(c: ScoutCandidate, ev: MarketEvidence, cfg: GrowthConfig) -> list[str]:
    """MARKET_COLLAPSE evidence (empty: none). A young token with little history is not a
    collapse: every test needs a real drop, dying activity or a stored liquidity peak."""
    k = cfg.collapse
    out = []
    drops = [(w, p) for w, p in (("1h", ev.price_h1), ("6h", ev.price_h6), ("24h", ev.price_h24))
             if p is not None and p <= -k.price_drop_pct]  # fmt: skip
    if drops:
        w, p = min(drops, key=lambda x: x[1])
        out.append(f"price collapsed {p:+.0f}% over {w}")
    if (
        ev.liquidity_min_change_pct is not None
        and ev.liquidity_min_change_pct <= -k.liquidity_drop_pct
    ):
        out.append(f"liquidity collapsed {ev.liquidity_min_change_pct:+.0f}% at a stored lookback")
    age = ev.pool_age_hours
    h1, day = c.metrics.window("h1"), c.metrics.window("h24")
    if (
        age is not None
        and age >= k.activity_min_age_hours
        and h1 is not None
        and h1.txns is not None
        and day is not None
        and day.txns is not None
        and day.txns >= cfg.stage.min_txns_h24
    ):
        lifetime_rate = day.txns / min(1440.0, age * 60)
        if h1.txns / 60 <= k.dead_activity_ratio * lifetime_rate:
            out.append(f"activity dying: {h1.txns} trades in 1h against {day.txns} in 24h")
    liq, peak = c.metrics.liquidity_usd, ev.liquidity_peak_usd
    if (
        liq is not None
        and peak is not None
        and liq < k.usable_liquidity_usd
        and peak >= liq * k.peak_multiple
    ):
        out.append(f"liquidity ${liq:,.0f}, down from a stored ${peak:,.0f}: no longer usable")
    return out


# --- Families -------------------------------------------------------------------------------


def _family(
    name: FamilyName,
    signals: list[SubSignal],
    weight: float,
    fallback: float,
    notes: list[str] | None = None,
) -> FamilyScore:
    total = sum(s.weight for s in signals)
    available = total > 0
    score = sum(s.score * s.weight for s in signals) / total if available else fallback
    return FamilyScore(
        family=name,
        score=round(score, 4),
        weight=weight,
        contribution=0.0,
        available=available,
        signals=signals,
        notes=notes or [],
    )


def _sub(
    name: str, score: float | None, weight: float, raw: float | None, detail: str
) -> list[SubSignal]:
    if score is None or weight <= 0:
        return []
    return [SubSignal(name=name, score=round(score, 4), weight=weight, raw=raw, detail=detail)]


def market_family(
    ev: MarketEvidence, cfg: GrowthConfig, weight: float, r: Reading | None = None
) -> FamilyScore:
    mc = cfg.market
    signals: list[SubSignal] = []
    for metric, w, label in (
        ("volume", mc.volume_weight, "volume"),
        ("txns", mc.txn_weight, "trades"),
    ):
        ratios = ev.ratios(metric)
        score = mean([ratio_score(v, mc.ratio_full) for _, v in ratios])
        detail = ", ".join(f"{b}: {v:.2f}x" for b, v in ratios)
        signals += _sub(f"{label}_acceleration", score, w, _geo_mean([v for _, v in ratios]),
                        f"{label} rate ratios ({detail})" if ratios else "")  # fmt: skip
    has_activity = bool(signals)
    change = ev.buy_share_change
    pressure = mean(
        [
            linear_score(ev.buy_share, mc.buy_share_low, mc.buy_share_high),
            linear_score(ev.buyer_share, mc.buy_share_low, mc.buy_share_high),
            (clamp(0.5 + 0.5 * change / mc.buy_share_change_full) if change is not None else None),
        ]
    )
    doubtful = r is not None and bool(r.flow_doubts)
    if doubtful and pressure is not None:
        pressure = min(pressure, cfg.flow.doubtful_pressure_cap)
    signals += _sub("buy_pressure", pressure, mc.buy_pressure_weight, ev.buy_share,
                    f"h1 buy share (trade counts) {_fmt(ev.buy_share)}, distinct-buyer share "
                    f"{_fmt(ev.buyer_share)}, change {_fmt(change, signed=True)}"
                    + ("; capped: flow in doubt" if doubtful else ""))  # fmt: skip
    signals += _sub("price_momentum", pct_score(ev.price_h1, mc.price_h1_full_pct),
                    mc.price_weight, ev.price_h1, f"1h price change {_pct(ev.price_h1)}")  # fmt: skip
    t = ev.technical
    if t is not None:
        tech = {"up": 0.7, "flat": 0.5, "down": 0.25}[t.trend]
        if t.breakout and t.volume_confirmed:
            tech = 0.9
        if t.higher_lows:
            tech += 0.1
        signals += _sub("technical", clamp(tech), mc.technical_weight, t.change_pct,
                        f"{t.trend} over {t.snapshots} snapshots ({t.change_pct:+.1f}%), "
                        f"breakout {t.breakout}, volume confirmed {t.volume_confirmed}, "
                        f"higher lows {t.higher_lows}")  # fmt: skip
    if not has_activity:
        return _family(
            "market_activity",
            [],
            weight,
            MISSING_MARKET_SCORE,
            ["no trusted volume / trade acceleration: scored below flat (uncertainty)"],
        )
    return _family("market_activity", signals, weight, MISSING_MARKET_SCORE)


def liquidity_family(
    c: ScoutCandidate, ev: MarketEvidence, cap: float | None, cfg: GrowthConfig, weight: float
) -> FamilyScore:
    lc = cfg.liquidity
    liq = c.metrics.liquidity_usd
    signals = _sub("depth", log_band_score(liq, lc.depth_floor_usd, lc.depth_full_usd),
                   lc.depth_weight, liq, f"${liq or 0:,.0f} usable liquidity (absolute, log-capped)")  # fmt: skip
    signals += _sub("growth", pct_score(ev.liquidity_change_pct, lc.growth_full_pct), lc.growth_weight,
                    ev.liquidity_change_pct,
                    f"liquidity {_pct(ev.liquidity_change_pct)} {ev.liquidity_change_basis or ''}")  # fmt: skip
    worst = ev.liquidity_min_change_pct
    stability = (
        clamp(1 + min(0.0, worst) / lc.stability_drop_full_pct) if worst is not None else None
    )
    signals += _sub("stability", stability, lc.stability_weight, worst,
                    f"worst liquidity change over stored lookbacks {_pct(worst)}")  # fmt: skip
    ratio = liq / cap if liq is not None and cap else None
    signals += _sub("liquidity_to_cap", log_band_score(ratio, lc.ratio_low, lc.ratio_high),
                    lc.ratio_weight, ratio, f"liquidity / market cap (or FDV) {_fmt(ratio)}")  # fmt: skip
    return _family("liquidity_quality", signals, weight, 0.0)


def social_family(social: SocialEvidence, cfg: GrowthConfig, weight: float) -> FamilyScore:
    sc = cfg.social
    notes: list[str] = []
    base = {
        "SOCIAL_UNAVAILABLE": sc.neutral,
        "SOCIAL_QUIET": sc.neutral,
        "SOCIAL_EMERGING": sc.emerging,
        "SOCIAL_ACCELERATING": sc.accelerating,
        "SOCIAL_STRONG": sc.strong,
        "SOCIAL_STEADY": sc.steady,
        "SOCIAL_SATURATED": sc.saturated,
        "SOCIAL_FADING": sc.fading,
    }[social.status]
    if social.status == "SOCIAL_UNAVAILABLE":
        notes.append(
            f"social unavailable ({social.unavailable_reason}): neutral, not zero "
            "(many real tokens have no discussion)"
        )
        return _family("social_momentum", [], weight, sc.neutral, notes)
    if social.status == "SOCIAL_QUIET":
        notes.append("searched, no meaningful discussion: neutral (social isn't required)")
        if social.age_minutes is not None and social.age_minutes >= 1:
            notes.append(f"last searched {social.age_minutes:.0f} minutes ago (not this run)")
        return _family(
            "social_momentum",
            [SubSignal(name="state", score=sc.neutral, weight=1.0, detail="quiet")],
            weight,
            sc.neutral,
            notes,
        )
    if social.age_minutes is not None and social.age_minutes >= 1:
        notes.append(f"last searched {social.age_minutes:.0f} minutes ago (not this run)")
    trend = social.trend
    author = (
        ratio_score(trend.unique_authors.acceleration_ratio, sc.author_ratio_full)
        if trend is not None
        else None
    )
    blended = (
        base
        if author is None
        else ((1 - sc.author_growth_weight) * base + sc.author_growth_weight * author)
    )
    positive = blended - sc.neutral
    if positive > 0:
        strong = social.strong_share if social.strong_share is not None else 0.0
        factor = sc.probable_only_factor + (1 - sc.probable_only_factor) * strong
        if factor < 1:
            notes.append(f"weak attribution: positive evidence x{factor:.2f}")
        positive *= factor
        if social.spam_risk == "medium":
            positive *= sc.medium_spam_factor
            notes.append(f"medium spam risk: positive evidence x{sc.medium_spam_factor}")
        if social.corroborated:
            positive += sc.corroborated_bonus
            notes.append("confirmed on several platforms")
        if social.organic == "high":
            positive += sc.organic_high_bonus
    score = clamp(sc.neutral + positive)
    if social.spam_risk == "high":
        score = min(score, sc.high_spam_cap)
        notes.append(f"high spam risk: capped at {sc.high_spam_cap}")
    signals = [
        SubSignal(
            name="state",
            score=round(score, 4),
            weight=1.0,
            raw=None,
            detail=f"{social.status} ({social.reason or ''})",
        ),  # fmt: skip
    ]
    if author is not None and trend is not None:
        signals.append(
            SubSignal(
                name="unique_author_growth",
                score=round(author, 4),
                weight=0.0,  # already inside `state`; shown for explanation
                raw=trend.unique_authors.acceleration_ratio,
                detail=f"unique authors {trend.unique_authors.previous:g} → "
                f"{trend.unique_authors.recent:g}",
            )
        )
    return FamilyScore(
        family="social_momentum",
        score=round(score, 4),
        weight=weight,
        contribution=0.0,
        available=True,
        signals=signals,
        notes=notes,
    )


def earliness_family(
    c: ScoutCandidate,
    ev: MarketEvidence,
    r: Reading,
    cap: float | None,
    cfg: GrowthConfig,
    weight: float,
) -> FamilyScore:
    """How early we appear to be in a real, developing move: how much of the move already
    happened, pool age, whether activity only recently picked up, and (minor context)
    market size, only credited for a healthy market. Thin history pulls it toward
    neutral. A collapsed market gets none."""
    ec = cfg.earliness
    if r.collapsed:
        return _family(
            "earliness",
            [
                SubSignal(
                    name="market_collapse", score=0.0, weight=1.0, detail="; ".join(r.collapse)
                )
            ],
            weight,
            0.0,
            ["MARKET_COLLAPSE: no earliness credit (a collapse is not an early move)"],
        )
    extent = ev.move_extent_pct
    move: float | None = None
    if extent is not None:
        if extent < 0:
            move = ec.decline_move_score  # no up-move made, none developing either
        else:
            early = math.log(1 + ec.early_move_pct / 100)
            late = math.log(ec.late_move_multiple)
            done = math.log(1 + extent / 100)
            move = 1.0 if done <= early else clamp(1 - (done - early) / (late - early))
    signals = _sub("move_already_made", move, ec.move_weight, extent,
                   f"price already moved {_pct(extent)} (24h / since first seen)"
                   + (": a decline, neutral" if extent is not None and extent < 0 else ""))  # fmt: skip
    age = ev.oldest_pool_age_hours if ev.oldest_pool_age_hours is not None else ev.pool_age_hours
    age_score: float | None = None
    if age is not None:
        if age < ec.young_hours:
            age_score = ec.young_score
        elif age <= ec.prime_hours:
            age_score = 1.0
        else:
            age_score = clamp(
                1 - math.log(age / ec.prime_hours) / math.log(ec.old_days * 24 / ec.prime_hours)
            )
    signals += _sub("age", age_score, ec.age_weight, age,
                    f"oldest pool {age:.1f}h old" if age is not None else "")  # fmt: skip
    regime = ev.activity_h6_vs_h24
    signals += _sub("activity_regime", ratio_score(regime, ec.activity_ratio_full),
                    ec.activity_weight, regime,
                    f"6h trade rate {_fmt(regime)}x the 24h rate (recent pick-up is early)")  # fmt: skip
    notes = []
    if cfg.mode == "ALL_TRENDING":
        notes.append("ALL_TRENDING mode: market size not scored")
    else:
        size = log_band_score(cap, ec.small_cap_usd, ec.large_cap_usd)
        context = 1 - size if size is not None else None
        liq = c.metrics.liquidity_usd
        healthy = liq is not None and liq >= cfg.liquidity.healthy_min_usd
        if context is not None and not healthy and context > ec.unhealthy_maturity_cap:
            context = ec.unhealthy_maturity_cap
            notes.append("small market with thin liquidity: size earns no extra earliness")
        signals += _sub("market_size_context", context, ec.maturity_weight, cap,
                        f"market cap (FDV if none) ${cap or 0:,.0f}")  # fmt: skip
    family = _family("earliness", signals, weight, 0.5, notes)
    # A move largely made already caps the whole family: pool age or a recent pick-up
    # can't make a token that already ran look early.
    if move is not None and extent is not None and extent > ec.early_move_pct:
        factor = 0.5 + 0.5 * move
        family.score = round(family.score * factor, 4)
        family.notes.append(f"price already up {extent:.0f}%: earliness x{factor:.2f}")
    # History depth: with little stored history, credit above neutral is less certain
    # (never the other way: missing history doesn't make a late token look earlier).
    depth = min(1.0, ev.stored_snapshots / cfg.technical.min_snapshots)
    confidence = ec.min_confidence + (1 - ec.min_confidence) * depth
    if family.available and confidence < 1 and family.score > 0.5:
        family.score = round(0.5 + (family.score - 0.5) * confidence, 4)
        family.notes.append(
            f"{ev.stored_snapshots} stored snapshot(s): credit above neutral x{confidence:.2f}"
        )
    return family


def cross_family(
    r: Reading, social: SocialEvidence, cfg: GrowthConfig, weight: float
) -> FamilyScore:
    cc = cfg.cross
    confirmations = {
        "volume accelerating": r.volume_rising,
        "trades accelerating": r.txns_rising,
        "buyers strengthening": r.buyers_strengthening,
        "liquidity healthy / growing": r.liquidity_confirming,
        "price rising (not vertical)": r.price_rising and not r.vertical,
    }
    agreeing = [k for k, v in confirmations.items() if v]
    social_agrees = social.rising and social.spam_risk != "high" and not r.social_only_hype
    score = cc.market_share * min(1.0, len(agreeing) / cc.full_market_confirmations)
    score += (1 - cc.market_share) * social_agrees
    notes = []
    if r.distribution:
        score -= cc.contradiction_penalty
        notes.append("contradiction: volume up while buyers weaken and sells accelerate")
    if r.volume_rising and r.buyers_weakening and not r.distribution:
        score -= cc.contradiction_penalty / 2
        notes.append("contradiction: volume up while buy pressure weakens")
    if r.flow_divergent:
        score -= cc.contradiction_penalty
        notes.append("contradiction: activity / price rising on a very weak buy count")
    if r.collapsed:
        score = 0.0
        notes.append("MARKET_COLLAPSE: nothing confirms a developing move")
    signals = [
        SubSignal(name=name, score=1.0 if ok else 0.0, weight=0.0, detail="agrees" if ok else "no")
        for name, ok in confirmations.items()
    ] + [SubSignal(name="social agrees", score=float(social_agrees), weight=0.0,
                   detail=social.status)]  # fmt: skip
    return FamilyScore(
        family="cross_confirmation",
        score=round(clamp(score), 4),
        weight=weight,
        contribution=0.0,
        available=True,
        signals=signals,
        notes=[f"{len(agreeing)} independent market confirmation(s)", *notes],
    )


# --- Safety and risk ------------------------------------------------------------------------


def safety_status(
    c: ScoutCandidate, snap: OnchainSafetySnapshot | None
) -> tuple[SafetyStatus, list[str]]:
    if c.chain != "solana":
        return "INSUFFICIENT_SAFETY_DATA", [
            f"no on-chain safety source for {c.chain} yet (authorities, holders unknown)"
        ]
    if snap is None:
        return "INSUFFICIENT_SAFETY_DATA", ["on-chain safety not checked yet"]
    missing = []
    if not snap.authorities_available:
        missing.append(f"token authorities unknown ({snap.authorities_error or 'unavailable'})")
    if not snap.holders_available:
        missing.append(f"holder concentration unknown ({snap.holders_error or 'unavailable'})")
    elif not snap.concentration_authoritative:
        missing.append("holder scan incomplete: concentration figures are lower bounds")
    missing += [r for r in snap.incomplete_reasons if r not in missing]
    if snap.authorities_available and snap.concentration_authoritative:
        return "SAFETY_CHECKS_COMPLETE", missing
    if snap.authorities_available or snap.holders_available:
        return "SAFETY_CHECKS_PARTIAL", missing
    return "INSUFFICIENT_SAFETY_DATA", missing


def risk_flags(
    c: ScoutCandidate,
    ev: MarketEvidence,
    r: Reading,
    social: SocialEvidence,
    snap: OnchainSafetySnapshot | None,
    safety: SafetyStatus,
    cfg: GrowthConfig,
) -> list[RiskFlag]:
    rc = cfg.risk
    flags: list[RiskFlag] = []

    def add(code: str, severity: Severity, detail: str, penalty: float = 0.0) -> None:
        flags.append(RiskFlag(code=code, severity=severity, detail=detail, penalty=penalty))

    if snap is not None and c.chain == "solana":
        if snap.freeze_authority_active:
            add("freeze_authority_active", "critical",
                "a freeze authority is set: holders' tokens can be frozen", rc.freeze_authority_active)  # fmt: skip
        if snap.mint_authority_active:
            add("mint_authority_active", "high",
                "a mint authority is set: supply can still be increased", rc.mint_authority_active)  # fmt: skip
        bound = " (at least; holder scan incomplete)" if snap.concentration_lower_bound else ""
        if snap.top1_pct is not None and snap.top1_pct >= rc.top1_holder_pct:
            add("holder_concentration_top1", "high",
                f"largest holder owns {snap.top1_pct:.1f}% of supply{bound}", rc.top1_holder)  # fmt: skip
        if snap.top10_pct is not None and snap.top10_pct >= rc.top10_holders_pct:
            add("holder_concentration_top10", "high",
                f"top 10 holders own {snap.top10_pct:.1f}% of supply{bound}", rc.top10_holders)  # fmt: skip
    if safety == "INSUFFICIENT_SAFETY_DATA":
        add("safety_data_missing", "caution",
            "on-chain safety (authorities, holders) unknown", rc.insufficient_safety_data)  # fmt: skip
    elif safety == "SAFETY_CHECKS_PARTIAL":
        add("safety_data_partial", "info", "holder data incomplete", rc.partial_safety_data)
    if r.collapsed:
        add("market_collapse", "critical",
            "MARKET_COLLAPSE: " + "; ".join(r.collapse), rc.market_collapse)  # fmt: skip
    if r.flow_divergent:
        add("flow_divergence", "high",
            "; ".join(d for d in r.flow_doubts if "visible buyers" in d), rc.flow_divergence)  # fmt: skip
    for doubt in (d for d in r.flow_doubts if "visible buyers" not in d):
        add("flow_in_doubt", "caution", doubt)
    if r.thin_pump:
        add("thin_market_pump", "high",
            f"price {_pct(max(ev.price_h24 or 0, ev.price_h1 or 0))} on ${c.metrics.liquidity_usd or 0:,.0f} "
            f"liquidity and {ev.txns_h24 or 0} trades in 24h", rc.thin_market_pump)  # fmt: skip
    if r.social_only_hype:
        add("social_only_hype", "high",
            "attention rising while trading volume and trade count are flat", rc.social_only_hype)  # fmt: skip
    elif social.rising and not ev.timescales:
        add("social_without_market_evidence", "caution",
            "attention rising; no trusted market acceleration to confirm it")  # fmt: skip
    if r.distribution:
        add("distribution", "high",
            "volume rising while buy pressure weakens and sells accelerate", rc.distribution)  # fmt: skip
    if social.spam_risk == "high":
        add(
            "social_spam",
            "caution",
            "social mentions look manipulated (high spam risk)",
            rc.spam_high,
        )
    if r.liquidity_draining:
        add("liquidity_draining", "high",
            f"liquidity fell {_pct(ev.liquidity_min_change_pct)} at a stored lookback", rc.liquidity_draining)  # fmt: skip
    f = c.risk_flags
    if f.unrecognized_quote:
        add("unrecognized_quote", "caution",
            "priced against an unrecognized quote: USD liquidity may be inflated", rc.unrecognized_quote)  # fmt: skip
    if f.high_volume_to_liquidity:
        add("high_volume_to_liquidity", "caution",
            "very high volume relative to liquidity (thin, easy to move)", rc.high_volume_to_liquidity)  # fmt: skip
    if f.primary_pool_unclear:
        add("primary_pool_unclear", "caution", "several comparable pools; market choice unclear",
            rc.primary_pool_unclear)  # fmt: skip
    if f.fdv_far_above_market_cap:
        add("fdv_far_above_market_cap", "caution", "large locked / unissued supply",
            rc.fdv_far_above_market_cap)  # fmt: skip
    if f.very_new_pool:
        add("very_new", "info", "very new pool: little history", rc.very_new_pool)
    if f.pool_age_unknown:
        add("pool_age_unknown", "info", "pool age not reported")
    if f.market_cap_missing:
        add("market_cap_missing", "info", "no market cap reported (FDV kept separately)")
    if social.status == "SOCIAL_UNAVAILABLE":
        add("social_unavailable", "info",
            f"social data unavailable ({social.unavailable_reason}); not counted against it")  # fmt: skip
    return flags


# --- Stage ----------------------------------------------------------------------------------


def classify_stage(
    ev: MarketEvidence, r: Reading, social: SocialEvidence, cfg: GrowthConfig
) -> tuple[GrowthStage, list[str]]:
    """First match wins: INSUFFICIENT_DATA, FADING, CROWDED, NEW, ACCELERATING, EARLY,
    STEADY. A price spike alone never makes a token ACCELERATING: that needs volume /
    trades / buyers rising on several independent time scales."""
    sc = cfg.stage
    age = ev.pool_age_hours
    young = age is not None and age < sc.new_pool_hours
    if ev.txns_h24 is None or ev.txns_h24 < sc.min_txns_h24:
        return "INSUFFICIENT_DATA", [
            f"only {ev.txns_h24 or 0} trades in 24h (need {sc.min_txns_h24})"
        ]
    if not ev.timescales and not young:
        return "INSUFFICIENT_DATA", [
            "no trusted acceleration measurement (windows too thin or partial, no history)",
            *ev.ignored[:3],
        ]
    if r.collapsed:
        return "FADING", ["MARKET_COLLAPSE", *r.collapse]

    fading = []
    if r.volume_falling:
        fading.append("volume declining")
    if r.txns_falling:
        fading.append("trades declining")
    if r.buyers_weakening:
        fading.append("buy pressure weakening")
    if r.price_weakening:
        fading.append("price momentum weakening")
    if (r.volume_falling or r.txns_falling) and social.status == "SOCIAL_FADING":
        fading.append("social attention fading")
    if (
        ev.timescales
        and len(fading) >= sc.fading_min_signs
        and (r.volume_falling or r.txns_falling)
    ):
        return "FADING", fading

    extent = ev.move_extent_pct
    if extent is not None and extent >= sc.crowded_move_pct:
        signs = []
        if not r.volume_rising:
            signs.append("volume acceleration flattening")
        if r.buyers_weakening:
            signs.append("buy pressure weakening")
        if social.status in ("SOCIAL_SATURATED", "SOCIAL_FADING"):
            signs.append(f"social attention high but not growing ({social.status})")
        if r.vertical:
            signs.append("price vertical while liquidity lags")
        lagging = (
            ev.price_change_same_lookback is not None
            and ev.liquidity_change_pct is not None
            and ev.price_change_same_lookback >= 50
            and ev.liquidity_change_pct < ev.price_change_same_lookback / 4
        )
        if lagging:
            signs.append("liquidity not keeping pace with price")
        if r.thin_pump:
            signs.append("thin market: liquidity and trade count far behind the move")
        move = f"price already up {extent:.0f}%"
        if extent >= sc.extreme_move_pct:
            return "CROWDED", [move + " (most of the move has likely happened)", *signs]
        if len(signs) >= sc.crowded_min_signs:
            return "CROWDED", [move, *signs]

    if young or not ev.timescales:
        return "NEW", [
            f"pool {age:.1f}h old: too little history to confirm acceleration"
            if age is not None
            else "too little history to confirm acceleration"
        ]
    rising = [
        n
        for n, ok in (
            ("volume", r.volume_rising),
            ("trades", r.txns_rising),
            ("buyers", r.buyers_strengthening),
        )
        if ok
    ]
    if (
        len(rising) >= sc.accelerating_min_indicators
        and (r.volume_rising or r.txns_rising)
        and len(r.timescales_rising) >= sc.accelerating_min_timescales
        and not r.timescales_falling
        and not r.distribution
    ):
        reasons = [
            f"{', '.join(rising)} rising on {len(r.timescales_rising)} time scales "
            f"({', '.join(r.timescales_rising)})"
        ]
        if social.rising:
            reasons.append(f"social {social.status.removeprefix('SOCIAL_').lower()} too")
        return "ACCELERATING", reasons
    strengthening = (r.volume_rising or r.txns_rising or r.timescales_rising) or (
        r.buyers_strengthening and not r.volume_falling
    )
    if strengthening and r.timescales_falling:
        # Rising on one time scale, falling on another: a mixed picture, not early growth.
        return "STEADY", [
            f"mixed: rising on {', '.join(r.timescales_rising) or 'none'}, falling on "
            f"{', '.join(r.timescales_falling)}"
        ]
    if strengthening:
        why = rising or ["buy pressure"]
        detail = (
            f"on {', '.join(r.timescales_rising)} only" if r.timescales_rising else "not yet broad"
        )
        return "EARLY", [f"{', '.join(why)} beginning to strengthen ({detail})"]
    return "STEADY", ["activity present but neither strengthening nor deteriorating"]


_RISING_STAGES = ("EARLY", "ACCELERATING")


def stabilize(
    stage: GrowthStage,
    reasons: list[str],
    r: Reading,
    previous: Sequence[tuple[datetime, str]],
    now: datetime,
    cfg: GrowthConfig,
) -> tuple[GrowthStage, list[str], GrowthStage | None]:
    """Hysteresis: a one-run reversal (rising stage -> FADING, FADING -> ACCELERATING)
    needs stronger evidence, otherwise the intermediate stage is shown until confirmed.
    Returns (stage, reasons, the unconfirmed stage or None)."""
    sc = cfg.stability
    if not sc.enabled or r.collapsed:
        return stage, reasons, None
    recent = [s for at, s in previous if (now - at).total_seconds() <= sc.memory_minutes * 60]
    if not recent:
        return stage, reasons, None
    before = recent[-1]
    if stage == "FADING" and before in _RISING_STAGES:
        if len(r.timescales_falling) >= sc.fading_min_timescales:
            return stage, reasons, None
        return (
            "STEADY",
            [
                f"was {before}; now {', '.join(reasons)} on too few time scales: a one-run "
                "reversal, awaiting confirmation",
            ],
            stage,
        )
    if stage == "ACCELERATING" and before == "FADING":
        need = cfg.stage.accelerating_min_timescales + sc.accelerating_extra_timescales
        if len(r.timescales_rising) >= need:
            return stage, reasons, None
        return (
            "EARLY",
            [
                f"was FADING; {', '.join(reasons)}: a one-run reversal, awaiting confirmation",
            ],
            stage,
        )
    return stage, reasons, None


def stage_adjustment(stage: GrowthStage, r: Reading, cfg: GrowthConfig) -> float:
    a = cfg.stage_adjustment
    if stage == "NEW":
        promising = (
            (r.buyers_strengthening or r.price_rising)
            and r.liquidity_confirming
            and not (r.thin_pump or r.collapsed or r.flow_doubts or r.vertical)
        )
        return a.new_promising if promising else a.new
    return {
        "ACCELERATING": a.accelerating,
        "EARLY": a.early,
        "STEADY": a.steady,
        "CROWDED": a.crowded,
        "FADING": a.fading,
        "INSUFFICIENT_DATA": a.insufficient_data,
    }[stage]


def maturity(
    age_hours: float | None, cap: float | None, liquidity: float | None, cfg: GrowthConfig
) -> float | None:
    """0 (young / small / shallow) .. 1 (established): weighted mean of what's known."""
    m, e = cfg.eligibility.maturity, cfg.eligibility
    parts = [
        (m.age_weight, log_band_score(age_hours, cfg.earliness.prime_hours,
                                      e.established_pool_age_days * 24)),
        (m.size_weight, log_band_score(cap, cfg.earliness.small_cap_usd, e.established_cap_usd)),
        (m.depth_weight, log_band_score(liquidity, cfg.liquidity.depth_full_usd,
                                        m.mature_liquidity_usd)),
    ]  # fmt: skip
    known = [(w, v) for w, v in parts if v is not None and w > 0]
    total = sum(w for w, _ in known)
    return round(sum(w * v for w, v in known) / total, 4) if total else None


# --- Reasons --------------------------------------------------------------------------------


def reasons_surfaced(
    ev: MarketEvidence, r: Reading, social: SocialEvidence, stage: GrowthStage
) -> list[str]:
    out = []
    if r.volume_rising:
        basis, value = max(ev.ratios("volume"), key=lambda x: x[1])
        out.append(f"volume accelerating ({value:.1f}x, {basis})")
    if r.txns_rising:
        basis, value = max(ev.ratios("txns"), key=lambda x: x[1])
        out.append(f"trade count accelerating ({value:.1f}x, {basis})")
    if r.buyers_strengthening:
        change = ev.buy_share_change
        out.append(
            "buyer activity increasing"
            + (f" (buy share {change:+.0%})" if change is not None else "")
        )
    if (
        r.liquidity_confirming
        and ev.liquidity_change_pct is not None
        and ev.liquidity_change_pct >= 10
    ):
        out.append(
            f"liquidity growing ({_pct(ev.liquidity_change_pct)} {ev.liquidity_change_basis})"
        )
    elif r.liquidity_confirming:
        out.append("liquidity healthy")
    if r.price_rising and not r.vertical:
        out.append(f"price rising ({_pct(ev.price_h1)} in 1h)")
    t = ev.technical
    if t is not None and t.breakout and t.volume_confirmed:
        out.append("breaking above stored highs with rising volume")
    if social.rising and social.spam_risk != "high" and not r.social_only_hype:
        attribution = {"exact": ", exact-contract mentions", "strong": ", strong attribution"}
        out.append(
            f"social {social.status.removeprefix('SOCIAL_').lower()}"
            + attribution.get(social.attribution, "")
            + (", confirmed on several platforms" if social.corroborated else "")
        )
    extent = ev.move_extent_pct
    # A price that fell hasn't made an early move: never call a drop "still early".
    if stage in ("EARLY", "ACCELERATING") and extent is not None and 0 <= extent < 100:
        out.append(f"still early: price {_pct(extent)} so far")
    return out or ["observed market activity (no acceleration yet)"]


# --- Candidate ------------------------------------------------------------------------------


def trusted_market_cap(c: ScoutCandidate) -> tuple[float | None, str | None]:
    m = c.metrics
    if m.market_cap_usd is None:
        return None, "no market cap reported" + (" (FDV only)" if m.fdv_usd else "")
    if c.risk_flags.unrecognized_quote:
        return None, "priced against an unrecognized quote: market cap not trusted"
    if m.fdv_usd is not None and m.market_cap_usd > m.fdv_usd * 1.05:
        return None, "reported market cap exceeds FDV: not trusted"
    return m.market_cap_usd, None


def evaluate(
    c: ScoutCandidate,
    ev: MarketEvidence,
    social: SocialEvidence,
    snap: OnchainSafetySnapshot | None,
    cfg: GrowthConfig,
    now: datetime,
    previous_stages: Sequence[tuple[datetime, str]] = (),
    stale_minutes: float | None = None,
) -> GrowthCandidate:
    """`stale_minutes`: the candidate is carried on a last good observation this old (a
    provider failure kept it from being refreshed): labeled, penalized, never current."""
    cap, cap_note = trusted_market_cap(c)
    cap_for_scale = cap if cap is not None else c.metrics.fdv_usd
    r = read(c, ev, social, cfg)
    stage, stage_reasons = classify_stage(ev, r, social, cfg)
    stage, stage_reasons, unconfirmed = stabilize(
        stage, stage_reasons, r, previous_stages, now, cfg
    )
    safety, safety_missing = safety_status(c, snap)
    flags = risk_flags(c, ev, r, social, snap, safety, cfg)
    if stale_minutes is not None:
        flags.insert(0, RiskFlag(
            code="stale_market_data", severity="high",
            detail=f"not refreshed this run (provider unavailable): market evidence is "
                   f"{stale_minutes:.0f} minutes old",
            penalty=cfg.risk.stale_market_data,
        ))  # fmt: skip
        stage_reasons = [f"stale: evidence {stale_minutes:.0f} minutes old", *stage_reasons]

    weights = cfg.weights.as_dict()
    total = sum(weights.values())
    families = [
        market_family(ev, cfg, weights["market_activity"] / total, r),
        liquidity_family(c, ev, cap_for_scale, cfg, weights["liquidity_quality"] / total),
        social_family(social, cfg, weights["social_momentum"] / total),
        earliness_family(c, ev, r, cap_for_scale, cfg, weights["earliness"] / total),
        cross_family(r, social, cfg, weights["cross_confirmation"] / total),
    ]
    for f in families:
        f.contribution = round(100 * f.score * f.weight, 2)
    base = sum(f.contribution for f in families)
    adjustment = stage_adjustment(stage, r, cfg)
    penalty = min(cfg.risk.max_total, sum(f.penalty for f in flags))
    score = ScoutMomentumScore(
        score=round(clamp(base + adjustment - penalty, 0, 100), 1),
        base=round(base, 1),
        stage_adjustment=adjustment,
        risk_penalty=round(penalty, 1),
        families=families,
    )

    identity_ok = identity_problem(
        c.chain, c.address, _VERIFY_IDENTITY
    ) is None and c.canonical_id == canonical_id(c.chain, c.address)
    market_confirmed = (
        c.primary_clear
        and not c.risk_flags.unrecognized_quote
        and c.metrics.price_usd is not None
        and (ev.txns_h24 or 0) >= cfg.stage.min_txns_h24
    )
    verification: list[VerificationStatus] = []
    if identity_ok:
        verification.append("VERIFIED_IDENTITY")
    if market_confirmed:
        verification.append("MARKET_CONFIRMED")
    verification.append(safety)

    liq = c.metrics.liquidity_usd
    trend = social.trend
    first_seen = c.first_seen_at
    return GrowthCandidate(
        canonical_id=c.canonical_id,
        symbol=c.symbol,
        name=c.name,
        chain=c.chain,
        address=c.address,
        observed_at=c.observed_at,
        data_status="CURRENT" if stale_minutes is None else "STALE_CARRIED",
        snapshot_age_minutes=round(stale_minutes, 1) if stale_minutes is not None else None,
        stage=stage,
        stage_reasons=stage_reasons,
        unconfirmed_stage=unconfirmed,
        market=GrowthMarket(
            price_usd=c.metrics.price_usd,
            market_cap_usd=cap,
            market_cap_note=cap_note,
            fdv_usd=c.metrics.fdv_usd,
            liquidity_usd=liq,
            pool_age_hours=ev.pool_age_hours,
            oldest_pool_age_hours=ev.oldest_pool_age_hours,
            first_seen_at=first_seen,
            tracked_hours=(
                round((now - first_seen).total_seconds() / 3600, 2) if first_seen else None
            ),
            selected_pool=c.pool,
            market_provider=c.market_provider,
            pool_count=c.pool_count,
        ),
        momentum=GrowthMomentum(
            volume_acceleration=_first(ev.ratios("volume")),
            volume_acceleration_basis=_first_basis(ev.ratios("volume")),
            txn_acceleration=_first(ev.ratios("txns")),
            txn_acceleration_basis=_first_basis(ev.ratios("txns")),
            buy_share=ev.buy_share,
            buy_pressure_change=ev.buy_share_change,
            buyer_share=ev.buyer_share,
            price_change_h1_pct=ev.price_h1,
            price_change_h24_pct=ev.price_h24,
            price_velocity_change_pct_per_hour=ev.price_velocity_change,
            liquidity_change_pct=ev.liquidity_change_pct,
            liquidity_change_basis=ev.liquidity_change_basis,
            market_cap_change_pct=ev.market_cap_change_pct,
            move_since_first_seen_pct=ev.move_since_first_seen_pct,
            timescales_rising=r.timescales_rising,
            technical=ev.technical,
            social_status=social.status,
            social_state=social.state,
            social_reason=social.reason or social.unavailable_reason,
            mention_acceleration=trend.mentions.acceleration_ratio if trend else None,
            unique_author_acceleration=trend.unique_authors.acceleration_ratio if trend else None,
            engagement_acceleration=trend.engagement.acceleration_ratio if trend else None,
            cross_platform_corroborated=social.corroborated,
            platforms_active=social.platforms_active,
        ),
        quality=GrowthQuality(
            verification=verification,
            identity_status="VERIFIED_IDENTITY" if identity_ok else "UNVERIFIED",
            market_confirmed=market_confirmed,
            safety_status=safety,
            safety_missing=safety_missing,
            social_attribution=social.attribution,
            exact_mention_share=social.exact_share,
            spam_risk=social.spam_risk,
            organic_signal=social.organic,
            holder_top1_pct=snap.top1_pct if snap else None,
            holder_top10_pct=snap.top10_pct if snap else None,
            holder_data_lower_bound=snap.concentration_lower_bound if snap else None,
            mint_authority_active=snap.mint_authority_active if snap else None,
            freeze_authority_active=snap.freeze_authority_active if snap else None,
            liquidity_quality=(
                "unknown"
                if liq is None
                else "draining"
                if r.liquidity_draining
                else "thin"
                if liq < cfg.liquidity.healthy_min_usd
                else "healthy"
            ),
            market_status="MARKET_COLLAPSE" if r.collapsed else "OK",
            collapse_evidence=r.collapse,
            flow_quality=r.flow_quality,  # type: ignore[arg-type]
            flow_notes=r.flow_doubts,
            maturity=maturity(
                ev.oldest_pool_age_hours
                if ev.oldest_pool_age_hours is not None
                else ev.pool_age_hours,
                cap_for_scale,
                liq,
                cfg,
            ),
        ),
        scout_momentum=score,
        reasons_surfaced=reasons_surfaced(ev, r, social, stage),
        risk_flags=flags,
    )


def eligibility(g: GrowthCandidate, cfg: GrowthConfig) -> list[str]:
    """Why a candidate isn't ranked (empty: it is). Applied before ranking."""
    reasons = []
    if g.quality.identity_status != "VERIFIED_IDENTITY":
        reasons.append("identity can't be verified (chain + contract / mint)")
    if g.stage == "INSUFFICIENT_DATA" and not cfg.eligibility.rank_insufficient_data:
        reasons.append("insufficient data to rank: " + "; ".join(g.stage_reasons[:1]))
    if cfg.mode == "NEW_AND_EARLY":
        e = cfg.eligibility
        cap = g.market.market_cap_usd or g.market.fdv_usd
        age = g.market.oldest_pool_age_hours or g.market.pool_age_hours
        if cap is not None and cap >= e.established_cap_usd:
            reasons.append(f"established market (${cap:,.0f} cap): outside NEW_AND_EARLY")
        elif age is not None and age >= e.established_pool_age_days * 24:
            reasons.append(f"established market ({age / 24:.0f} days old): outside NEW_AND_EARLY")
        elif (
            g.quality.maturity is not None
            and g.quality.maturity >= e.maturity.threshold
            and g.stage != "ACCELERATING"
        ):
            reasons.append(
                f"mature market (maturity {g.quality.maturity:.2f}) without a new acceleration "
                "regime: outside NEW_AND_EARLY"
            )
    return reasons


# --- Formatting helpers ---------------------------------------------------------------------


def _first(items: list[tuple[str, float]]) -> float | None:
    return round(items[0][1], 4) if items else None


def _first_basis(items: list[tuple[str, float]]) -> str | None:
    return items[0][0] if items else None


def _fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.0f}%"
