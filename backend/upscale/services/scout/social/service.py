"""SocialScoutService: search → attribute → store → measure → momentum.

One run, for a set of tracked tokens:

1. Every configured provider is searched concurrently, tokens batched into as few queries
   as its API allows. Each token's search starts where its last successful search ended
   (minus a small overlap), or `search_span_hours` back the first time.
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
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from upscale.services.market_data import MarketDataError
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
from upscale.services.scout.social.config import SocialConfig
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
    ) -> SocialRun:
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
            since_by_token = {
                t.canonical_id: await self._since(t.canonical_id, provider.name, started)
                for t in tracked
            }
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
                    for t in tracked
                },
            )
            outcomes = await asyncio.gather(
                *(
                    provider.search(terms, min(since_by_token[c] for c in ids), keyer)
                    for ids, terms in plans
                ),
                return_exceptions=True,
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

    async def _since(self, cid: str, provider: str, now: datetime) -> datetime:
        cfg = self.config
        earliest = now - timedelta(hours=cfg.search_span_hours)
        last = await self.store.last_covered_to(cid, provider)
        if last is None:
            return earliest
        return max(earliest, last - timedelta(minutes=cfg.overlap_minutes))

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
