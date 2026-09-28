"""ScoutService: discovery → exact identity → one market per token → filters → snapshots →
growth features.

1. Every provider is asked, concurrently, for each discovery kind and chain it serves.
   A failing source is reported in the run's `errors`; the others still count.
2. Candidates are merged by canonical id (`<chain>:<address>`), so a token found by
   several listings or pools is one candidate with every source kept.
3. Optionally, candidates are **enriched** with an exact, batched lookup of every pool
   the token has (default: DEX Screener, the source the Analyze pipeline's DEX market
   uses, so Scout and Analyze agree on the token's primary pool). A token the lookup
   doesn't know keeps its discovery data.
4. Basic filters reject what is clearly unusable (reasons are kept).
5. Accepted candidates are recorded (first sighting) and snapshotted, and their growth
   features are computed against stored history.

Tracked tokens are re-observed with `refresh_tokens` in the "refresh" request lane: each
provider may reserve part of its rate limit for it (`ScoutProviderLimits.reservations`),
which discovery can't consume while held (`hold_reservations`); `release_reservations`
hands what is left back to discovery (e.g. to retry listings that failed for capacity).
Provider rate limits stay hard throughout.

Discovery order is the order tokens were collected; nothing here ranks candidates.
"""

import asyncio
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from upscale.services.market_data import MarketDataError
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.features import compute_features
from upscale.services.scout.filters import candidate_problems
from upscale.services.scout.gate import RequestGate, request_lane
from upscale.services.scout.models import (
    DiscoveryKind,
    ScoutCandidate,
    ScoutFeedReport,
    ScoutRejection,
    ScoutRun,
    ScoutSourceError,
)
from upscale.services.scout.normalize import DiscoveryResult, canonical_id, merge_candidates
from upscale.services.scout.providers import DiscoveryProvider
from upscale.services.scout.store import ScoutSnapshotStore, snapshot_of

LISTING_KINDS: tuple[DiscoveryKind, ...] = ("new", "active", "trending")
# The request lane tracked-token refreshes run in (providers may reserve capacity for it).
REFRESH_LANE = "refresh"


class ScoutService:
    def __init__(
        self,
        providers: Sequence[DiscoveryProvider],
        store: ScoutSnapshotStore,
        config: ScoutConfig | None = None,
        enrichment_provider: str | None = "DEX Screener",
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.providers = list(providers)
        self.store = store
        self.config = config or ScoutConfig()
        self.enrichment_provider = enrichment_provider
        self.now = now

    async def discover(
        self,
        kinds: Iterable[DiscoveryKind] | None = None,
        chains: Iterable[str] | None = None,
        only: Collection[tuple[str, str, str]] | None = None,
    ) -> ScoutRun:
        """`only`: just these (provider, kind, chain) listings, e.g. to retry the ones that
        failed once capacity is available again."""
        started = self.now()
        wanted_kinds = [k for k in (kinds or self.config.kinds) if k in LISTING_KINDS]
        wanted_chains = list(chains or self.config.chains)
        calls = [
            (provider, kind, chain)
            for provider in self.providers
            for kind in wanted_kinds
            if kind in provider.kinds
            for chain in wanted_chains
            if chain in provider.chains
            if only is None or (provider.name, kind, chain) in only
        ]
        calls, reports = await self._schedule_feeds(calls, started)
        outcomes = await asyncio.gather(
            *(self._listing(p, k, c) for p, k, c in calls), return_exceptions=True
        )
        collected = DiscoveryResult()
        errors: list[ScoutSourceError] = []
        for (provider, kind, chain), outcome in zip(calls, outcomes, strict=True):
            if isinstance(outcome, MarketDataError):
                report = next(r for r in reports if r.provider == provider.name)
                report.failed.append(f"{kind}:{chain}")
                errors.append(
                    ScoutSourceError(
                        provider=provider.name, kind=kind, chain=chain, error=str(outcome)
                    )
                )
            elif isinstance(outcome, BaseException):
                raise outcome  # a bug, not a provider outage: never hidden
            else:
                collected.candidates.extend(outcome.candidates)
                collected.rejected.extend(outcome.rejected)
        candidates = merge_candidates(collected.candidates)
        candidates, enrich_errors = await self._enrich(candidates)
        run = await self._finish(started, candidates, collected.rejected, errors + enrich_errors)
        run.feeds = reports
        return run

    async def _schedule_feeds(
        self, calls: list[tuple[DiscoveryProvider, DiscoveryKind, str]], at: datetime
    ) -> tuple[list[tuple[DiscoveryProvider, DiscoveryKind, str]], list[ScoutFeedReport]]:
        """Fair discovery feed scheduling (stride scheduling), per provider.

        A feed is `<kind>:<chain>`. Each has a persisted pass value; the feeds with the
        lowest pass run first (ties: higher weight, then the feed name, so input order
        never matters), as many as the provider's discovery capacity allows right now
        (its rate limit minus any held reservation, e.g. for tracked-token refresh). A feed
        that runs advances its pass by 1 / weight: a weight-2 feed comes round about twice
        as often as a weight-1 feed, and every feed's turn comes, so none starves. Feeds
        not sent are deferred, never reported as failed. A newly offered feed starts at
        the lowest pass among the provider's feeds (no burst of priority), feeds of one
        kind staggered across their stride so kinds interleave within a run."""
        weights = self.config.feed_weights
        by_provider: dict[str, list[tuple[DiscoveryProvider, DiscoveryKind, str]]] = {}
        for call in calls:
            by_provider.setdefault(call[0].name, []).append(call)
        selected: list[tuple[DiscoveryProvider, DiscoveryKind, str]] = []
        reports: list[ScoutFeedReport] = []
        for name, group in by_provider.items():
            provider = group[0][0]
            gate = getattr(provider, "gate", None)
            stored = await self.store.feed_passes(name)
            feeds = {f"{k}:{c}": (p, k, c) for p, k, c in group}
            known = [stored[f] for f in feeds if f in stored]
            start = min(known) if known else 0.0
            # Feeds of one kind start staggered across its stride (the i-th of n at i / n of
            # it), so kinds interleave instead of moving in lockstep.
            siblings: dict[str, list[str]] = {}
            for f in sorted(feeds):
                siblings.setdefault(f.split(":", 1)[0], []).append(f)
            passes = {
                f: stored.get(f, start + i / len(same) / _feed_weight(f, weights))
                for same in siblings.values()
                for i, f in enumerate(same)
            }

            capacity = gate.available() if isinstance(gate, RequestGate) else len(feeds)
            queue = _feed_order(passes, weights)
            run, deferred = queue[:capacity], queue[capacity:]
            for feed in run:
                passes[feed] += 1 / _feed_weight(feed, weights)
            await self.store.save_feed_passes(name, passes, run, at)
            nominal = (
                gate.limits.calls_per_minute - sum(gate.limits.reservations.values())
                if isinstance(gate, RequestGate)
                else len(feeds)
            )
            selected += [feeds[f] for f in run]
            reports.append(
                ScoutFeedReport(
                    provider=name,
                    available=sorted(feeds),
                    executed=run,
                    deferred=deferred,
                    requests=len(run),
                    next_scheduled=_feed_order(passes, weights)[:nominal],
                )
            )
        return selected, reports

    async def lookup_exact_token(self, chain: str, address: str) -> ScoutRun:
        """One exact token (chain + contract / mint), from the first provider that knows it."""
        return await self.lookup_exact_tokens(chain, [address])

    async def lookup_exact_tokens(
        self,
        chain: str,
        addresses: Sequence[str],
        first: str | None = None,
        only: Collection[str] | None = None,
        exclude: Collection[str] = (),
    ) -> ScoutRun:
        """Exact lookup by chain + address, provider by provider until every token is
        found (`first`: the provider to ask first, e.g. the one that last priced them;
        `only` / `exclude`: restrict which providers are asked)."""
        started = self.now()
        wanted = {canonical_id(chain, a) for a in addresses}
        found: list[ScoutCandidate] = []
        rejected: list[ScoutRejection] = []
        errors: list[ScoutSourceError] = []
        for provider in self._lookup_providers(chain, first):
            if (only is not None and provider.name not in only) or provider.name in exclude:
                continue
            missing = [a for a in addresses if canonical_id(chain, a) in wanted]
            if not missing:
                break
            try:
                result = await provider.lookup_exact_tokens(chain, missing)
            except MarketDataError as exc:
                errors.append(
                    ScoutSourceError(
                        provider=provider.name, kind="lookup", chain=chain, error=str(exc)
                    )
                )
                continue
            found.extend(result.candidates)
            rejected.extend(result.rejected)
            wanted -= {c.canonical_id for c in result.candidates}
        return await self._finish(started, merge_candidates(found), rejected, errors)

    async def refresh_tokens(
        self, tokens: Sequence[tuple[str, str, str | None]], lane: str = REFRESH_LANE
    ) -> "RefreshOutcome":
        """Re-observe tracked tokens `(chain, address, provider that last priced it)` by
        exact address, batched per provider and chain, in the request lane `lane` (so they
        use the capacity providers reserve for it).

        `unresolved`: tokens whose planned provider failed (not refreshed, but not known to
        be gone). A token its planned provider answered for without a usable market is
        gone / unusable, whatever a fallback provider did."""
        groups: dict[tuple[str, str], list[str]] = {}
        for chain, address, provider in tokens:
            able = self._lookup_providers(chain, provider)
            if able:
                groups.setdefault((able[0].name, chain), []).append(address)
        with request_lane(lane):
            # 1. Each token from the provider it was planned for (within that provider's
            #    reservation); fallbacks can't take capacity planned for another group.
            runs = list(
                await asyncio.gather(
                    *(self.lookup_exact_tokens(ch, a, only={p}) for (p, ch), a in groups.items())
                )
            )
            found = {c.canonical_id for r in runs for c in r.candidates}
            failed_groups = {
                (e.provider, e.chain) for r in runs for e in r.errors if e.chain is not None
            }
            # 2. Then whatever is still missing, from the other providers, on what's left.
            retry = {
                key: [a for a in addrs if canonical_id(key[1], a) not in found]
                for key, addrs in groups.items()
            }
            runs += await asyncio.gather(
                *(self.lookup_exact_tokens(ch, a, exclude={p}) for (p, ch), a in retry.items() if a)
            )
        found = {c.canonical_id for r in runs for c in r.candidates}
        unresolved = {
            canonical_id(ch, a)
            for (p, ch), addrs in groups.items()
            if (p, ch) in failed_groups
            for a in addrs
            if canonical_id(ch, a) not in found
        }
        return RefreshOutcome(
            run=ScoutRun(
                started_at=min((r.started_at for r in runs), default=self.now()),
                candidates=[c for r in runs for c in r.candidates],
                rejected=[x for r in runs for x in r.rejected],
                errors=[e for r in runs for e in r.errors],
            ),
            unresolved=unresolved,
        )

    def _gates(self) -> list[RequestGate]:
        return [g for p in self.providers if isinstance(g := getattr(p, "gate", None), RequestGate)]

    def hold_reservations(self, lane: str = REFRESH_LANE) -> None:
        """Hold every provider's reservation for `lane` (other lanes can't use it)."""
        for gate in self._gates():
            gate.arm(lane)

    def release_reservations(self, lane: str = REFRESH_LANE) -> None:
        """Hand what `lane` left unused back to every lane (e.g. discovery retries)."""
        for gate in self._gates():
            gate.release(lane)

    def refresh_capacity(self, lane: str = REFRESH_LANE) -> dict[str, int]:
        """Requests each lookup provider reserves per minute for `lane`."""
        return {
            p.name: g.reserved(lane)
            for p in self.providers
            if "lookup" in p.kinds and isinstance(g := getattr(p, "gate", None), RequestGate)
        }

    async def refresh_tracked(self, seen_within: timedelta) -> ScoutRun:
        """Re-observe every token seen within `seen_within`, in batched exact lookups, to
        extend its snapshot history."""
        tracked = await self.store.tracked_tokens(self.now() - seen_within)
        by_chain: dict[str, list[str]] = {}
        for chain, address in tracked:
            by_chain.setdefault(chain, []).append(address)
        runs = await asyncio.gather(
            *(self.lookup_exact_tokens(chain, addrs) for chain, addrs in by_chain.items())
        )
        return ScoutRun(
            started_at=min((r.started_at for r in runs), default=self.now()),
            candidates=[c for r in runs for c in r.candidates],
            rejected=[x for r in runs for x in r.rejected],
            errors=[e for r in runs for e in r.errors],
        )

    async def run_periodically(
        self,
        interval: timedelta,
        stop: asyncio.Event,
        track_for: timedelta = timedelta(hours=6),
    ) -> None:
        """Discover and refresh tracked tokens every `interval` until `stop` is set, so
        snapshots accumulate at a steady cadence."""
        while not stop.is_set():
            await self.discover()
            await self.refresh_tracked(track_for)
            try:
                await asyncio.wait_for(stop.wait(), interval.total_seconds())
            except TimeoutError:
                pass

    # --- internals ------------------------------------------------------------------------

    async def _listing(
        self, provider: DiscoveryProvider, kind: DiscoveryKind, chain: str
    ) -> DiscoveryResult:
        limit = self.config.max_per_listing
        if kind == "new":
            return await provider.discover_new_tokens(chain, limit)
        if kind == "active":
            return await provider.discover_active_tokens(chain, limit)
        return await provider.discover_trending_tokens(chain, limit)

    def _lookup_providers(self, chain: str, first: str | None = None) -> list[DiscoveryProvider]:
        able = [p for p in self.providers if "lookup" in p.kinds and chain in p.chains]
        # `first`, then the enrichment provider, so exact lookups match the Analyze pipeline.
        return sorted(able, key=lambda p: (p.name != first, p.name != self.enrichment_provider))

    async def _enrich(
        self, candidates: list[ScoutCandidate]
    ) -> tuple[list[ScoutCandidate], list[ScoutSourceError]]:
        provider = next(
            (
                p
                for p in self.providers
                if p.name == self.enrichment_provider and "lookup" in p.kinds
            ),
            None,
        )
        if provider is None or not candidates:
            return candidates, []
        by_chain: dict[str, list[str]] = {}
        for c in candidates:
            if c.chain in provider.chains:
                by_chain.setdefault(c.chain, []).append(c.address)
        chains = list(by_chain)
        outcomes = await asyncio.gather(
            *(provider.lookup_exact_tokens(ch, by_chain[ch]) for ch in chains),
            return_exceptions=True,
        )
        full: dict[str, ScoutCandidate] = {}
        errors: list[ScoutSourceError] = []
        for chain, outcome in zip(chains, outcomes, strict=True):
            if isinstance(outcome, MarketDataError):
                errors.append(
                    ScoutSourceError(
                        provider=provider.name, kind="lookup", chain=chain, error=str(outcome)
                    )
                )
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                full.update({c.canonical_id: c for c in outcome.candidates})
        enriched = []
        for c in candidates:
            market = full.get(c.canonical_id)
            if market is None:
                enriched.append(c)  # unknown to the lookup: keep what discovery observed
                continue
            oldest = [t for t in (c.oldest_pool_created_at, market.oldest_pool_created_at) if t]
            enriched.append(
                market.model_copy(
                    update={
                        "sources": [*c.sources, *market.sources],
                        "oldest_pool_created_at": min(oldest) if oldest else None,
                        "symbol": market.symbol or c.symbol,
                        "name": market.name or c.name,
                    }
                )
            )
        return enriched, errors

    async def _finish(
        self,
        started: datetime,
        candidates: list[ScoutCandidate],
        rejected: list[ScoutRejection],
        errors: list[ScoutSourceError],
    ) -> ScoutRun:
        accepted: list[ScoutCandidate] = []
        for c in candidates:
            problems = candidate_problems(c, self.config)
            if problems:
                rejected.append(
                    ScoutRejection(
                        canonical_id=c.canonical_id,
                        chain=c.chain,
                        address=c.address,
                        symbol=c.symbol,
                        reasons=problems,
                        provider=c.market_provider,
                    )
                )
            else:
                accepted.append(c)
        accepted = list(await asyncio.gather(*(self._persist(c) for c in accepted)))
        accepted_ids = {c.canonical_id for c in accepted}
        return ScoutRun(
            started_at=started,
            candidates=accepted,
            rejected=_unique_rejections(r for r in rejected if r.canonical_id not in accepted_ids),
            errors=errors,
        )

    async def _persist(self, candidate: ScoutCandidate) -> ScoutCandidate:
        first_seen = await self.store.record_seen(candidate)
        features = await compute_features(candidate, self.store, self.config.features)
        await self.store.save_snapshot(
            snapshot_of(candidate), self.config.min_snapshot_interval_seconds
        )
        full = candidate.model_copy(update={"first_seen_at": first_seen, "features": features})
        await self.store.save_latest(full)  # the last good observation, for a short grace
        return full


def _feed_weight(feed: str, weights: dict[str, float]) -> float:
    return weights.get(feed.split(":", 1)[0], 1.0)


def _feed_order(passes: dict[str, float], weights: dict[str, float]) -> list[str]:
    """Lowest pass first; ties: higher weight, then the feed name (never input order)."""
    return sorted(passes, key=lambda feed: (passes[feed], -_feed_weight(feed, weights), feed))


@dataclass
class RefreshOutcome:
    run: ScoutRun
    unresolved: set[str] = field(default_factory=set)  # planned provider failed


def _unique_rejections(rejections: Iterable[ScoutRejection]) -> list[ScoutRejection]:
    seen: set[tuple[str | None, tuple[str, ...]]] = set()
    out = []
    for r in rejections:
        key = (r.canonical_id, tuple(r.reasons))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
