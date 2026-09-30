"""SocialScoutService: search → attribute → store → measure → momentum.

One run, for a set of tracked tokens:

1. Every configured provider is searched concurrently, tokens batched into as few queries
   as its API allows. Each token's search starts where its last successful search ended
   (minus a small overlap), or `search_span_hours` back the first time. A provider with a
   per-run budget or rate limit (X, Neynar) searches tokens in priority order (how long
   each has waited, never-searched and provisional-rank bonuses; see `fair_order`), in
   waves its remaining budget and rate limit can afford; tokens it doesn't reach are
   PROVIDER_UNAVAILABLE (deferred), never a zero, and have waited longer next run. After
   an HTTP 402 (credits exhausted) nothing more is sent to that provider this run.
2. Every returned post is attributed (EXACT / STRONG / PROBABLE / AMBIGUOUS / REJECTED)
   and stored once; a post seen again is not rewritten.
3. Each (token, provider) search outcome is stored: PROVIDER_OK,
   PROVIDER_CHECKED_ZERO_MATCHES, PROVIDER_UNAVAILABLE (with the error) or
   PROVIDER_NOT_CONFIGURED. Coverage of successful searches is what makes a zero a real
   zero.
4. Per token: windows and trends per provider and combined, spam / quality evidence,
   cross-platform confirmation, the market cross-check (when Scout market features are
   given), and the momentum state. Snapshots and momentum are appended to history.

Social failures never raise out of a run: a failing provider is reported and the others
still count. This service is independent of Scout's market discovery, which keeps working
whatever happens here. Nothing here decides BUY / SELL / WAIT.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from upscale.services.evidence_archive import hooks as evidence
from upscale.services.market_data import MarketDataError
from upscale.services.scout.gate import RateLimitReachedError
from upscale.services.scout.models import WINDOW_MINUTES, ScoutCandidate, ScoutGrowthFeatures
from upscale.services.scout.social.analysis import (
    covered,
    cross_platform,
    is_accelerating,
    market_cross_check,
    measure,
    momentum_state,
    social_quality,
    window_stats,
    window_trend,
)
from upscale.services.scout.social.attribution import COUNTED_LEVELS, AttributionIndex
from upscale.services.scout.social.config import SchedulingConfig, SocialConfig
from upscale.services.scout.social.models import (
    Attribution,
    ProviderCheck,
    ProviderStatus,
    SocialEvent,
    SocialMomentum,
    SocialPost,
    SocialRun,
    SocialSearchResult,
    SocialSourceSnapshot,
    SocialWindowStats,
    TokenIdentity,
    WindowTrend,
)
from upscale.services.scout.social.providers import SocialTrendProvider, plan_queries
from upscale.services.scout.social.store import CoverageCheck, SocialStore
from upscale.services.scout.social.text import author_key, fingerprint, simhash
from upscale.services.scout.store import ScoutSnapshotStore

CHECKED: frozenset[ProviderStatus] = frozenset({"PROVIDER_OK", "PROVIDER_CHECKED_ZERO_MATCHES"})


def identities_from_candidates(candidates: Sequence[ScoutCandidate]) -> list[TokenIdentity]:
    return [
        TokenIdentity(
            canonical_id=c.canonical_id,
            chain=c.chain,
            address=c.address,
            symbol=c.symbol,
            name=c.name,
        )
        for c in candidates
    ]


async def known_identities(store: ScoutSnapshotStore) -> list[TokenIdentity]:
    """Every token Scout has ever seen: the directory used to detect competing tickers."""
    return [
        TokenIdentity(canonical_id=cid, chain=chain, address=address, symbol=symbol, name=name)
        for cid, chain, address, symbol, name in await store.tokens()
    ]


class SocialScoutService:
    def __init__(
        self,
        providers: Sequence[SocialTrendProvider],
        store: SocialStore,
        config: SocialConfig | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.providers = list(providers)
        self.store = store
        self.config = config or SocialConfig()
        self.now = now

    async def observe(
        self,
        tracked: Sequence[TokenIdentity],
        universe: Sequence[TokenIdentity] = (),
        market: Mapping[str, ScoutGrowthFeatures] | None = None,
        priority: Mapping[str, int] | None = None,
    ) -> SocialRun:
        """`priority`: each token's provisional rank (0 = best), used by budgeted
        providers to spend scarce searches on the leading candidates first, within the
        anti-starvation rules of `fair_order`."""
        cfg = self.config
        started = self.now()
        await self.store.prune(started - timedelta(hours=cfg.event_retention_hours))
        salt = await self.store.author_salt()

        def keyer(platform: str, author_id: str) -> str:
            return author_key(salt, platform, author_id)

        index = AttributionIndex(tracked, universe, cfg.attribution)
        statuses: dict[tuple[str, str], tuple[ProviderStatus, str | None]] = {}
        checks: list[ProviderCheck] = []
        counts = {"ambiguous": 0, "rejected": 0}

        async def run_provider(provider: SocialTrendProvider) -> None:
            if not provider.configured:
                checks.append(
                    ProviderCheck(
                        provider=provider.name,
                        platform=provider.platform,
                        status="PROVIDER_NOT_CONFIGURED",
                        checked_at=started,
                        requirement=provider.requirement,
                    )
                )
                for t in tracked:
                    statuses[(t.canonical_id, provider.name)] = ("PROVIDER_NOT_CONFIGURED", None)
                    await self.store.record_check(
                        t.canonical_id, provider.name, provider.platform, started,
                        "PROVIDER_NOT_CONFIGURED", None, None, provider.requirement,
                    )  # fmt: skip
                return
            start_run = getattr(provider, "start_run", None)
            daily = getattr(provider, "max_results_per_day", None)
            if isinstance(daily, int):
                used = await self.store.results_today(provider.name, started)
                if used >= daily:  # stop cleanly: nothing searched, nothing reported as zero
                    error = f"{provider.name} daily result budget ({daily}) was reached"
                    checks.append(
                        ProviderCheck(
                            provider=provider.name,
                            platform=provider.platform,
                            status="PROVIDER_UNAVAILABLE",
                            checked_at=started,
                            error=error,
                            requests=0,
                            results=0,
                        )
                    )
                    for t in tracked:
                        statuses[(t.canonical_id, provider.name)] = ("PROVIDER_UNAVAILABLE", error)
                        await self.store.record_check(
                            t.canonical_id, provider.name, provider.platform, started,
                            "PROVIDER_UNAVAILABLE", None, None, error,
                        )  # fmt: skip
                    return
                if callable(start_run):
                    start_run(daily - used)
            elif callable(start_run):
                start_run()
            last_covered = {
                t.canonical_id: await self.store.last_covered_to(t.canonical_id, provider.name)
                for t in tracked
            }
            since_by_token = {cid: self._since(last, started) for cid, last in last_covered.items()}
            capacity = getattr(provider, "search_capacity", None)
            budgeted = callable(capacity) and capacity() is not None
            order = list(tracked)
            if budgeted:
                # A budget may not reach every token: search the stalest first (see
                # `fair_order`), never in the order tokens happened to be listed.
                waiting = {
                    cid: await self.store.first_unavailable_at(cid, provider.name)
                    for cid, last in last_covered.items()
                    if last is None
                }
                order = fair_order(
                    tracked,
                    last_covered,
                    await self.store.next_schedule_round(provider.name),
                    waiting,
                    now=started,
                    priority=priority,
                    config=cfg.scheduling,
                )
            # A provider may shape its own terms from the token's references (e.g. to
            # avoid paying for posts attribution could never count); otherwise the
            # address and cashtag.
            search_terms = getattr(provider, "search_terms", None)
            plans = plan_queries(
                provider,
                {
                    t.canonical_id: search_terms(index.references(t))
                    if callable(search_terms)
                    else index.terms_for(t)
                    for t in order
                },
            )

            async def search(plan: tuple[list[str], list[str]]) -> SocialSearchResult:
                ids, terms = plan
                return await provider.search(terms, min(since_by_token[c] for c in ids), keyer)

            outcomes: list[SocialSearchResult | BaseException]
            if budgeted:
                assert callable(capacity)
                outcomes = await _in_budget_waves(provider.name, capacity, plans, search)
            else:
                outcomes = list(
                    await asyncio.gather(*(search(p) for p in plans), return_exceptions=True)
                )
            # A token's terms may be split over several queries: its outcome is only
            # judged once all of them are in. Any failed query makes the token
            # unavailable (a partial search never counts as a real zero); coverage is
            # the span every query covered.
            errors: list[str] = []
            failed: dict[str, list[str]] = {}
            done: dict[str, list[tuple[SocialSearchResult, set[str]]]] = {}
            for (ids, _), outcome in zip(plans, outcomes, strict=True):
                if isinstance(outcome, MarketDataError):
                    errors.append(str(outcome))
                    for cid in ids:
                        failed.setdefault(cid, []).append(str(outcome))
                    continue
                if isinstance(outcome, BaseException):
                    raise outcome  # a bug, never hidden
                events = self._events(outcome, index)
                counts["ambiguous"] += sum(e.attribution_level == "AMBIGUOUS" for e in events)
                counts["rejected"] += sum(e.attribution_level == "REJECTED" for e in events)
                await self.store.add_events(events)
                for cid in ids:
                    matched = {
                        e.content_id
                        for e in events
                        if e.canonical_id == cid and e.attribution_level in COUNTED_LEVELS
                    }
                    done.setdefault(cid, []).append((outcome, matched))
            for cid in dict.fromkeys(c for ids, _ in plans for c in ids):
                if cid in failed:
                    error = "; ".join(sorted(set(failed[cid])))
                    statuses[(cid, provider.name)] = ("PROVIDER_UNAVAILABLE", error)
                    await self.store.record_check(
                        cid, provider.name, provider.platform, started,
                        "PROVIDER_UNAVAILABLE", None, None, error,
                    )  # fmt: skip
                    continue
                results = done[cid]
                found = len(set().union(*(m for _, m in results)))
                status: ProviderStatus = "PROVIDER_OK" if found else "PROVIDER_CHECKED_ZERO_MATCHES"
                statuses[(cid, provider.name)] = (status, None)
                await self.store.record_check(
                    cid, provider.name, provider.platform,
                    min(r.checked_at for r, _ in results), status,
                    max(r.complete_since for r, _ in results), found, None,
                )  # fmt: skip
            ok = len(errors) < len(plans) or not plans
            usage_of = getattr(provider, "usage", None)
            usage = usage_of() if callable(usage_of) else {}
            if usage:
                await self.store.record_usage(provider.name, started, usage)
            checks.append(
                ProviderCheck(
                    provider=provider.name,
                    platform=provider.platform,
                    status="PROVIDER_OK" if ok else "PROVIDER_UNAVAILABLE",
                    checked_at=self.now(),
                    error="; ".join(sorted(set(errors))) or None,
                    **usage,
                )
            )

        await asyncio.gather(*(run_provider(p) for p in self.providers))
        configured = sum(p.configured for p in self.providers)
        momentum = await asyncio.gather(
            *(
                self._momentum(
                    t.canonical_id,
                    statuses,
                    configured,
                    (market or {}).get(t.canonical_id),
                    started,
                )
                for t in tracked
            )
        )
        return SocialRun(
            started_at=started,
            momentum=list(momentum),
            providers=sorted(checks, key=lambda c: c.provider),
            ambiguous_mentions=counts["ambiguous"],
            rejected_mentions=counts["rejected"],
        )

    # --- internals ------------------------------------------------------------------------

    def _since(self, last_covered: datetime | None, now: datetime) -> datetime:
        cfg = self.config
        earliest = now - timedelta(hours=cfg.search_span_hours)
        if last_covered is None:
            return earliest
        return max(earliest, last_covered - timedelta(minutes=cfg.overlap_minutes))

    def _events(self, result: SocialSearchResult, index: AttributionIndex) -> list[SocialEvent]:
        events: list[SocialEvent] = []
        for post in result.posts:
            for a in index.attribute(post):
                events.append(_event(post, result.checked_at, a))
        return events

    async def _momentum(
        self,
        cid: str,
        statuses: Mapping[tuple[str, str], tuple[ProviderStatus, str | None]],
        configured: int,
        features: ScoutGrowthFeatures | None,
        now: datetime,
    ) -> SocialMomentum:
        """Momentum as of `now`, the run's start: every search of this run began after it,
        so its windows are fully covered (a later clock reading would never be)."""
        cfg = self.config
        longest = max(WINDOW_MINUTES[w] for w in cfg.momentum.windows)
        span_start = now - timedelta(minutes=longest * (2 + cfg.momentum.baseline_periods))
        events = await self.store.events(cid, span_start, now)
        ambiguous = [
            e for e in await self.store.ambiguous_events(span_start) if cid in e.candidates
        ]
        all_checks = await self.store.checks(cid, span_start)

        sources: list[SocialSourceSnapshot] = []
        per_provider: list[tuple[list[SocialEvent], list[SocialEvent], list[CoverageCheck]]] = []
        for provider in self.providers:
            status, error = statuses.get(
                (cid, provider.name), ("PROVIDER_UNAVAILABLE", "not searched")
            )
            p_events = [e for e in events if e.provider == provider.name]
            p_ambiguous = [e for e in ambiguous if e.provider == provider.name]
            p_checks = [c for c in all_checks if c.provider == provider.name]
            windows, trends = (
                measure(p_events, p_ambiguous, p_checks, now, cfg)
                if status in CHECKED
                else ([], [])
            )
            state_trend = next((t for t in trends if t.window == cfg.momentum.state_window), None)
            state_window = next((w for w in windows if w.window == cfg.momentum.state_window), None)
            snapshot = SocialSourceSnapshot(
                canonical_id=cid,
                provider=provider.name,
                platform=provider.platform,
                observed_at=now,
                status=status,
                windows=windows,
                trends=trends,
                accelerating=is_accelerating(state_trend, cfg),
                active=bool(state_window and state_window.mentions > 0),
                error=error,
            )
            sources.append(snapshot)
            await self.store.add_snapshot(snapshot)
            if status in CHECKED:
                per_provider.append((p_events, p_ambiguous, p_checks))

        windows, trends = self._combined(per_provider, now)
        trend = next((t for t in trends if t.window == cfg.momentum.state_window), None)
        w = timedelta(minutes=WINDOW_MINUTES[cfg.momentum.state_window])
        judged = [e for evs, _, _ in per_provider for e in evs if now - 2 * w < e.posted_at <= now]
        cross = cross_platform(sources, configured, cfg)
        any_checked = bool(per_provider)
        preliminary = social_quality(judged, trend, cross, None, cfg)
        state, _ = momentum_state(trend, any_checked, preliminary, cross, cfg)
        market = market_cross_check(state, features, cfg)
        quality = social_quality(judged, trend, cross, market.state, cfg)
        state, reasons = momentum_state(trend, any_checked, quality, cross, cfg)
        momentum = SocialMomentum(
            canonical_id=cid,
            computed_at=now,
            state=state,
            reasons=reasons,
            window=cfg.momentum.state_window,
            trend=trend,
            windows=windows,
            trends=trends,
            quality=quality,
            cross_platform=cross,
            market=market,
            sources=sources,
        )
        await self.store.add_momentum(momentum)
        evidence.emit("social", momentum)  # archive (metrics only; never raises)
        return momentum

    def _combined(
        self,
        per_provider: Sequence[tuple[list[SocialEvent], list[SocialEvent], list[CoverageCheck]]],
        now: datetime,
    ) -> tuple[list[SocialWindowStats], list[WindowTrend]]:
        """Across platforms: each window sums the providers that fully cover it."""
        cfg = self.config
        windows, trends = [], []
        for window in cfg.momentum.windows:
            w = timedelta(minutes=WINDOW_MINUTES[window])
            recent = [p for p in per_provider if covered(p[2], now - w, now)]
            if not recent:
                continue
            windows.append(
                window_stats(
                    window,
                    [e for p in recent for e in p[0] if now - w < e.posted_at <= now],
                    sum(1 for p in recent for e in p[1] if now - w < e.posted_at <= now),
                    cfg,
                )
            )
            both = [p for p in per_provider if covered(p[2], now - 2 * w, now)]
            if not both:
                continue
            base_start = now - (2 + cfg.momentum.baseline_periods) * w
            baseline_ok = all(covered(p[2], base_start, now) for p in both)
            trends.append(
                window_trend(window, [e for p in both for e in p[0]], now, baseline_ok, cfg)
            )
        return windows, trends


class BudgetDeferredError(RateLimitReachedError):
    """A search this run's budget couldn't afford: deferred, never reported as zero."""


def fair_order(
    tracked: Sequence[TokenIdentity],
    last_covered: Mapping[str, datetime | None],
    schedule_round: int,
    waiting_since: Mapping[str, datetime | None] | None = None,
    *,
    now: datetime,
    priority: Mapping[str, int] | None = None,
    config: SchedulingConfig | None = None,
) -> list[TokenIdentity]:
    """The order a budgeted provider searches tokens in, independent of list order.
    Highest priority first, in minutes (see `SchedulingConfig`):

        waited + never-searched bonus + provisional-rank bonus

    * `waited`: since the last successful search; never searched, since the first time
      the token was deferred (`waiting_since`), 0 when brand new;
    * the never-searched bonus puts first-time coverage ahead of routine refreshes;
    * the rank bonus (1 for the top provisional candidate, falling linearly to 0 for the
      last; none without `priority`) spends scarce searches on the leading candidates.

    Bonuses are bounded while `waited` keeps growing, so a deferred token overtakes
    every fresher one eventually, however low it ranks and however many new tokens
    discovery adds: nothing starves. Ties by canonical id, rotated by one position per
    run (`schedule_round`), so the same token never always wins a tie.
    """
    cfg = config or SchedulingConfig()
    ids = sorted({t.canonical_id for t in tracked})
    shift = schedule_round % len(ids) if ids else 0
    rotation = {cid: (i - shift) % len(ids) for i, cid in enumerate(ids)}
    waiting = waiting_since or {}
    ranks = priority or {}
    ranked = len(ranks)

    def minutes(since: datetime | None) -> float:
        return max(0.0, (now - since).total_seconds() / 60) if since else 0.0

    def score(t: TokenIdentity) -> float:
        cid = t.canonical_id
        last = last_covered.get(cid)
        value = minutes(last) if last else minutes(waiting.get(cid))
        if last is None:
            value += cfg.never_searched_bonus_minutes
        if cid in ranks and ranked:
            value += cfg.rank_bonus_minutes * (1 - ranks[cid] / max(1, ranked - 1))
        return value

    return sorted(tracked, key=lambda t: (-score(t), rotation[t.canonical_id]))


async def _in_budget_waves(
    provider: str,
    capacity: Callable[[], tuple[int, str] | None],
    plans: Sequence[tuple[list[str], list[str]]],
    search: Callable[[tuple[list[str], list[str]]], Awaitable[SocialSearchResult]],
) -> list[SocialSearchResult | BaseException]:
    """Run plans in priority order, each wave only as large as the budget can still afford
    (so a concurrent reservation never crowds out later searches); once it can afford
    none, the rest are deferred. Budgets count what searches actually consumed, so small
    incremental searches leave room for more tokens."""
    outcomes: list[SocialSearchResult | BaseException] = []
    while len(outcomes) < len(plans):
        room, limit = capacity() or (len(plans), "")
        if room <= 0:
            error = BudgetDeferredError(
                f"{provider} credits are exhausted (HTTP 402); no more {provider} requests "
                "this run: deferred to a later run"
                if limit == "credits"
                else f"UpScale's {provider} {limit} for this run was reached; deferred to a "
                "later run (tokens that waited longest go first)"
            )
            outcomes += [error] * (len(plans) - len(outcomes))
            break
        wave = plans[len(outcomes) : len(outcomes) + room]
        outcomes += await asyncio.gather(*(search(p) for p in wave), return_exceptions=True)
    return outcomes


def _event(post: SocialPost, fetched_at: datetime, attribution: Attribution) -> SocialEvent:
    return SocialEvent(
        canonical_id=attribution.canonical_id,
        provider=post.provider,
        platform=post.platform,
        posted_at=post.created_at,
        fetched_at=fetched_at,
        content_id=post.post_id,
        author_key=post.author_key,
        token_reference=attribution.token_reference,
        likes=post.likes,
        replies=post.replies,
        reposts=post.reposts,
        quotes=post.quotes,
        views=post.views,
        engagement=post.engagement,
        attribution_level=attribution.level,
        attribution_reason=attribution.reason,
        candidates=attribution.candidates,
        fingerprint=fingerprint(post.text),
        simhash=simhash(post.text),
        has_contract=attribution.level == "EXACT",
        promoted=post.promoted,
        source_url=post.source_url,
        author_quality=post.author_quality,
    )
