"""Attention measurements and the deterministic momentum state. Nothing here ranks tokens
or produces a trade decision.

**Coverage.** A window is only measured when successful searches fully covered it (the
union of every stored check's [covered_from, covered_to] for that provider). A window
nobody looked at is absent, never zero; a provider that failed contributes nothing.

**Trends** (per window W, per metric: weighted mentions, unique authors, engagement):
recent = [now - W, now], previous = [now - 2W, now - W], baseline = the average of the
`baseline_periods` periods before that (only when covered). Velocity is per hour;
`acceleration_ratio` = (recent + s) / (previous + s) and `baseline_ratio` = (previous + s) /
(baseline + s), with smoothing s so tiny counts can't make huge ratios. Absolute size alone
never implies momentum: 10,000 → 9,800 → 9,400 is FADING, 50 → 120 → 400 is STRONG.

**Momentum state** (on the combined trend of `state_window`, first match wins):

1. UNAVAILABLE: no social provider could be searched this run (none configured, or all
   failed).
2. INSUFFICIENT_DATA: the recent and previous periods aren't both covered.
3. QUIET: fewer than `min_mentions` weighted mentions in both periods.
4. EMERGING: attention appearing from (almost) nothing: previous (and baseline, when
   known) at most `emerging_prior_max`, recent at least `min_mentions`.
5. FADING: recent / previous below `fading_ratio`, or declining for two periods in a row
   (both ratios below 1).
6. STRONG: accelerating (below) **and** sustained (previous / baseline at least
   `strong_baseline_ratio`) **and** broad (at least `strong_min_authors` unique authors)
   **and** spam risk not high **and** not "several platforms checked, only one active".
7. ACCELERATING: mentions ratio at least `accelerating_ratio`, unique-author ratio at
   least `author_accelerating_ratio`, and recent at least `min_mentions`.
8. SATURATED: at least `saturated_min_mentions` in the recent period, but not accelerating.
9. STABLE: anything else.

STRONG / ACCELERATING describe attention only; they never mean BUY.
"""

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta

from upscale.services.scout.models import WINDOW_MINUTES, ScoutGrowthFeatures
from upscale.services.scout.social.attribution import COUNTED_LEVELS
from upscale.services.scout.social.config import SocialConfig
from upscale.services.scout.social.models import (
    CrossPlatformConfirmation,
    Level,
    MarketCrossCheck,
    MetricTrend,
    MomentumState,
    SocialEvent,
    SocialQuality,
    SocialSourceSnapshot,
    SocialWindowStats,
    WindowTrend,
)
from upscale.services.scout.social.store import CoverageCheck
from upscale.services.scout.social.text import hamming
from upscale.services.solana_dex import Window

RISING_STATES = frozenset({"EMERGING", "ACCELERATING", "STRONG"})


# --- Coverage -------------------------------------------------------------------------------


def covered(checks: Sequence[CoverageCheck], start: datetime, end: datetime) -> bool:
    """Whether successful searches together cover [start, end] without a gap."""
    spans = sorted(
        (c.covered_from, c.covered_to)
        for c in checks
        if c.covered_from is not None and c.covered_to is not None
    )
    reached = start
    for lo, hi in spans:
        assert lo is not None and hi is not None
        if lo > reached:
            break
        reached = max(reached, hi)
        if reached >= end:
            return True
    return reached >= end


# --- Windows and trends ---------------------------------------------------------------------


def weight(event: SocialEvent, cfg: SocialConfig) -> float:
    if event.promoted or event.attribution_level not in COUNTED_LEVELS:
        return 0.0
    return cfg.attribution.probable_weight if event.attribution_level == "PROBABLE" else 1.0


def _in(events: Sequence[SocialEvent], start: datetime, end: datetime) -> list[SocialEvent]:
    # Periods are half-open (start, end] so a post is never counted in two periods.
    return [e for e in events if start < e.posted_at <= end]


def window_stats(
    window: Window,
    events: Sequence[SocialEvent],
    ambiguous: int,
    cfg: SocialConfig,
) -> SocialWindowStats:
    counted = [e for e in events if weight(e, cfg) > 0]
    return SocialWindowStats(
        window=window,
        mentions=sum(weight(e, cfg) for e in events),
        unique_authors=len({e.author_key for e in counted}),
        engagement=sum(e.engagement or 0 for e in counted),
        exact_mentions=sum(e.attribution_level == "EXACT" for e in counted),
        attributable_posts=len(counted),
        promoted_posts=sum(
            bool(e.promoted) and e.attribution_level in COUNTED_LEVELS for e in events
        ),
        ambiguous_posts=ambiguous,
        sources=len({e.platform for e in counted}),
    )


def _metric(
    recent: float, previous: float, baseline_total: float | None, minutes: int, cfg: SocialConfig
) -> MetricTrend:
    hours = minutes / 60
    s = cfg.momentum.smoothing
    baseline = (
        baseline_total / cfg.momentum.baseline_periods if baseline_total is not None else None
    )
    velocity, previous_velocity = recent / hours, previous / hours
    return MetricTrend(
        recent=recent,
        previous=previous,
        baseline=baseline,
        velocity_per_hour=velocity,
        previous_velocity_per_hour=previous_velocity,
        acceleration_per_hour=(velocity - previous_velocity) / hours,
        acceleration_ratio=(recent + s) / (previous + s),
        baseline_ratio=(previous + s) / (baseline + s) if baseline is not None else None,
    )


def window_trend(
    window: Window,
    events: Sequence[SocialEvent],
    now: datetime,
    baseline_covered: bool,
    cfg: SocialConfig,
) -> WindowTrend:
    minutes = WINDOW_MINUTES[window]
    w = timedelta(minutes=minutes)
    periods = cfg.momentum.baseline_periods
    recent = window_stats(window, _in(events, now - w, now), 0, cfg)
    previous = window_stats(window, _in(events, now - 2 * w, now - w), 0, cfg)
    base = (
        window_stats(window, _in(events, now - (2 + periods) * w, now - 2 * w), 0, cfg)
        if baseline_covered
        else None
    )
    base_authors = (
        len(
            {
                e.author_key
                for e in _in(events, now - (2 + periods) * w, now - 2 * w)
                if weight(e, cfg) > 0
            }
        )
        if baseline_covered
        else None
    )
    return WindowTrend(
        window=window,
        mentions=_metric(
            recent.mentions, previous.mentions, base.mentions if base else None, minutes, cfg
        ),
        unique_authors=_metric(
            recent.unique_authors,
            previous.unique_authors,
            float(base_authors) if base_authors is not None else None,
            minutes,
            cfg,
        ),
        engagement=_metric(
            recent.engagement, previous.engagement, base.engagement if base else None, minutes, cfg
        ),
    )


def measure(
    events: Sequence[SocialEvent],
    ambiguous_events: Sequence[SocialEvent],
    checks: Sequence[CoverageCheck],
    now: datetime,
    cfg: SocialConfig,
) -> tuple[list[SocialWindowStats], list[WindowTrend]]:
    """Windows and trends from the events of the providers whose `checks` are given; only
    fully covered windows are produced."""
    windows, trends = [], []
    for window in cfg.momentum.windows:
        w = timedelta(minutes=WINDOW_MINUTES[window])
        if not covered(checks, now - w, now):
            continue
        windows.append(
            window_stats(
                window, _in(events, now - w, now), len(_in(ambiguous_events, now - w, now)), cfg
            )
        )
        if covered(checks, now - 2 * w, now):
            baseline_ok = covered(checks, now - (2 + cfg.momentum.baseline_periods) * w, now)
            trends.append(window_trend(window, events, now, baseline_ok, cfg))
    return windows, trends


def is_accelerating(trend: WindowTrend | None, cfg: SocialConfig) -> bool:
    if trend is None:
        return False
    m = cfg.momentum
    return (
        trend.mentions.recent >= m.min_mentions
        and trend.mentions.acceleration_ratio >= m.accelerating_ratio
        and trend.unique_authors.acceleration_ratio >= m.author_accelerating_ratio
    )


# --- Quality --------------------------------------------------------------------------------


def social_quality(
    events: Sequence[SocialEvent],
    trend: WindowTrend | None,
    cross: CrossPlatformConfirmation | None,
    market_state: str | None,
    cfg: SocialConfig,
) -> SocialQuality:
    q = cfg.quality
    posts = [e for e in events if e.attribution_level in COUNTED_LEVELS]
    rejected_contracts = sum(
        e.attribution_level == "REJECTED" and "different contract" in e.attribution_reason
        for e in events
    )
    if len(posts) < q.min_posts:
        return SocialQuality(
            spam_risk="unknown",
            organic_signal_strength="unknown",
            reasons=[f"only {len(posts)} attributable post(s): too few to judge quality"],
            posts=len(posts),
            unique_authors=len({e.author_key for e in posts}),
            conflicting_contracts=rejected_contracts,
        )
    by_author = Counter(e.author_key for e in posts)
    authors = len(by_author)
    top_share = by_author.most_common(1)[0][1] / len(posts)
    per_mention = authors / len(posts)
    duplicates = _duplicate_count(posts, q.near_duplicate_bits)
    duplicate_share = duplicates / len(posts)
    contract_posts = [e for e in posts if e.has_contract]
    contract_by_author = Counter(e.author_key for e in contract_posts)
    repeated = sum(n for n in contract_by_author.values() if n >= q.contract_repeats_per_author)
    repeated_share = repeated / len(contract_posts) if contract_posts else 0.0
    promoted_share = sum(bool(e.promoted) for e in posts) / len(posts)

    high: list[str] = []
    medium: list[str] = []
    if top_share >= q.top_author_share_high:
        high.append(f"one author wrote {top_share:.0%} of the mentions")
    elif top_share >= q.top_author_share_medium:
        medium.append(f"one author wrote {top_share:.0%} of the mentions")
    if duplicate_share >= q.duplicate_share_high:
        high.append(f"{duplicate_share:.0%} of the mentions are (near-)duplicate texts")
    elif duplicate_share >= q.duplicate_share_medium:
        medium.append(f"{duplicate_share:.0%} of the mentions are (near-)duplicate texts")
    if repeated_share >= q.repeated_contract_share_medium:
        medium.append(
            f"{repeated_share:.0%} of contract mentions come from authors repeating it "
            f"{q.contract_repeats_per_author}+ times"
        )
    if promoted_share >= q.promoted_share_high:
        high.append(f"{promoted_share:.0%} of the mentions are paid / promoted")
    elif promoted_share >= q.promoted_share_medium:
        medium.append(f"{promoted_share:.0%} of the mentions are paid / promoted")
    if is_accelerating(trend, cfg) and per_mention < q.low_diversity_authors_per_mention:
        medium.append(
            f"a burst with little author diversity ({per_mention:.2f} authors per mention)"
        )
    if rejected_contracts >= 2:
        medium.append(f"{rejected_contracts} posts use this ticker with a different contract")
    if market_state == "UNCONFIRMED_SOCIAL_SPIKE":
        medium.append("attention is rising without matching market activity")

    spam: Level = "high" if high or len(medium) >= 3 else "medium" if medium else "low"
    single_or_corroborated = cross is None or cross.corroborated or cross.single_source_available
    organic: Level
    if spam == "high" or authors < q.organic_medium_authors:
        organic = "low"
    elif (
        spam == "low"
        and authors >= q.organic_high_authors
        and per_mention >= q.organic_high_authors_per_mention
        and single_or_corroborated
    ):
        organic = "high"
    else:
        organic = "medium"
    return SocialQuality(
        spam_risk=spam,
        organic_signal_strength=organic,
        reasons=[*high, *medium] or ["no manipulation pattern found (not a guarantee)"],
        posts=len(posts),
        unique_authors=authors,
        top_author_share=top_share,
        authors_per_mention=per_mention,
        duplicate_share=duplicate_share,
        repeated_contract_share=repeated_share if contract_posts else None,
        promoted_share=promoted_share,
        conflicting_contracts=rejected_contracts,
    )


def _duplicate_count(posts: Sequence[SocialEvent], bits: int) -> int:
    """Posts whose text duplicates (or nearly duplicates) another post's."""
    dup = [False] * len(posts)
    by_fp = Counter(e.fingerprint for e in posts)
    for i, e in enumerate(posts):
        if by_fp[e.fingerprint] > 1:
            dup[i] = True
    for i in range(len(posts)):
        if dup[i]:
            continue
        for j in range(len(posts)):
            if i != j and hamming(posts[i].simhash, posts[j].simhash) <= bits:
                dup[i] = dup[j] = True
                break
    return sum(dup)


# --- Cross-platform and market --------------------------------------------------------------


def cross_platform(
    sources: Sequence[SocialSourceSnapshot], configured: int, cfg: SocialConfig
) -> CrossPlatformConfirmation:
    checked = [s for s in sources if s.status in ("PROVIDER_OK", "PROVIDER_CHECKED_ZERO_MATCHES")]
    active = {s.platform for s in checked if s.active}
    accelerating = {s.platform for s in checked if s.accelerating}
    platforms_checked = {s.platform for s in checked}
    notes: list[str] = []
    single = len(platforms_checked) == 1
    only_one = len(platforms_checked) >= 2 and len(active) == 1
    if configured == 0:
        notes.append("no social provider is configured")
    elif single:
        notes.append(
            "only one platform could be checked: corroboration isn't possible (not a penalty)"
        )
    if only_one:
        notes.append(
            f"{len(platforms_checked)} platforms checked; only {next(iter(active))} shows activity"
        )
    return CrossPlatformConfirmation(
        providers_configured=configured,
        providers_checked=len(checked),
        platforms_with_activity=len(active),
        platforms_accelerating=len(accelerating),
        source_diversity=len(active) / len(platforms_checked) if platforms_checked else None,
        corroborated=len(accelerating) >= cfg.cross_platform.min_platforms,
        single_source_available=single,
        only_one_active_of_several=only_one,
        notes=notes,
    )


def market_cross_check(
    state: MomentumState, features: ScoutGrowthFeatures | None, cfg: SocialConfig
) -> MarketCrossCheck:
    mc = cfg.market
    if features is None:
        return MarketCrossCheck(state="INSUFFICIENT_DATA", reasons=["no Scout market features"])
    accel = None
    for short, long in mc.window_pairs:
        accel = next(
            (
                a
                for a in features.window_acceleration
                if a.short == short
                and a.long == long
                and (a.volume_rate_ratio is not None or a.txn_rate_ratio is not None)
            ),
            None,
        )
        if accel:
            break
    history = sorted(
        (h for h in features.history if h.lookback_minutes <= mc.liquidity_lookback_minutes),
        key=lambda h: -h.lookback_minutes,
    )
    liquidity = next(
        (h.liquidity_change_pct for h in history if h.liquidity_change_pct is not None), None
    )
    check = MarketCrossCheck(
        state="INSUFFICIENT_DATA",
        market_window=f"{accel.short}/{accel.long}" if accel else None,
        volume_rate_ratio=accel.volume_rate_ratio if accel else None,
        txn_rate_ratio=accel.txn_rate_ratio if accel else None,
        liquidity_change_pct=liquidity,
    )
    ratios = [r for r in (check.volume_rate_ratio, check.txn_rate_ratio) if r is not None]
    market_rising = bool(ratios) and max(ratios) >= mc.rising_ratio
    social_rising = state in RISING_STATES
    if social_rising and liquidity is not None and liquidity <= -mc.liquidity_drop_pct:
        check.state = "CAUTION_LIQUIDITY_FALLING"
        check.reasons.append(f"attention rising while liquidity changed {liquidity:.0f}%")
    elif social_rising and market_rising:
        check.state = "CORROBORATED"
        check.reasons.append("attention and market activity are rising together")
    elif social_rising and ratios:
        check.state = "UNCONFIRMED_SOCIAL_SPIKE"
        check.reasons.append(
            f"attention rising but market activity ratio is {max(ratios):.2f}"
            + (" (flat)" if max(ratios) <= mc.flat_ratio else "")
        )
    elif social_rising:
        check.reasons.append("attention rising; no market activity ratios to compare")
    elif state in ("UNAVAILABLE", "INSUFFICIENT_DATA"):
        check.reasons.append("no usable social measurement")
    elif market_rising:
        check.state = "MARKET_WITHOUT_SOCIAL"
        check.reasons.append("market activity rising without rising attention")
    else:
        check.state = "NO_SIGNAL"
        check.reasons.append("neither attention nor market activity is rising")
    return check


# --- State ----------------------------------------------------------------------------------


def momentum_state(
    trend: WindowTrend | None,
    any_checked: bool,
    quality: SocialQuality,
    cross: CrossPlatformConfirmation,
    cfg: SocialConfig,
) -> tuple[MomentumState, list[str]]:
    m = cfg.momentum
    if not any_checked:
        return "UNAVAILABLE", ["no social provider could be searched"]
    if trend is None:
        return "INSUFFICIENT_DATA", [
            f"the last two {m.state_window} periods aren't both covered by searches"
        ]
    t = trend.mentions
    ratio_text = f"{t.previous:g} → {t.recent:g} mentions (ratio {t.acceleration_ratio:.2f})"
    if t.recent < m.min_mentions and t.previous < m.min_mentions:
        return "QUIET", [f"fewer than {m.min_mentions:g} mentions per period: {ratio_text}"]
    if (
        t.previous <= m.emerging_prior_max
        and (t.baseline is None or t.baseline <= m.emerging_prior_max)
        and t.recent >= m.min_mentions
    ):
        return "EMERGING", [f"attention appearing from almost nothing: {ratio_text}"]
    declining_twice = (
        t.acceleration_ratio < 1 and t.baseline_ratio is not None and t.baseline_ratio < 1
    )
    if t.acceleration_ratio < m.fading_ratio or declining_twice:
        return "FADING", [
            f"attention declining: {ratio_text}"
            + (f", baseline {t.baseline:g}" if t.baseline is not None else "")
        ]
    if is_accelerating(trend, cfg):
        blockers = []
        if t.baseline_ratio is None or t.baseline_ratio < m.strong_baseline_ratio:
            blockers.append("not yet sustained over the baseline")
        if trend.unique_authors.recent < m.strong_min_authors:
            blockers.append(f"fewer than {m.strong_min_authors} unique authors")
        if quality.spam_risk == "high":
            blockers.append("spam risk is high")
        if cross.only_one_active_of_several:
            blockers.append("several platforms checked, activity on only one")
        reasons = [
            f"accelerating: {ratio_text}; unique authors "
            f"{trend.unique_authors.previous:g} → {trend.unique_authors.recent:g}"
        ]
        if blockers:
            return "ACCELERATING", [*reasons, "not STRONG: " + "; ".join(blockers)]
        return "STRONG", [*reasons, "sustained over the baseline and broad"]
    if t.recent >= m.saturated_min_mentions:
        return "SATURATED", [f"high attention that is no longer growing: {ratio_text}"]
    return "STABLE", [f"attention roughly steady: {ratio_text}"]
