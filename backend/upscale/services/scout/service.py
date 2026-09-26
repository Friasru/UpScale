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

Discovery order is the order tokens were collected; nothing here ranks candidates.
"""

import asyncio
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta

from upscale.services.market_data import MarketDataError
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.features import compute_features
from upscale.services.scout.filters import candidate_problems
from upscale.services.scout.models import (
    DiscoveryKind,
    ScoutCandidate,
    ScoutRejection,
    ScoutRun,
    ScoutSourceError,
)
from upscale.services.scout.normalize import DiscoveryResult, canonical_id, merge_candidates
from upscale.services.scout.providers import DiscoveryProvider
from upscale.services.scout.store import ScoutSnapshotStore, snapshot_of

LISTING_KINDS: tuple[DiscoveryKind, ...] = ("new", "active", "trending")


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
    ) -> ScoutRun:
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
        ]
        outcomes = await asyncio.gather(
            *(self._listing(p, k, c) for p, k, c in calls), return_exceptions=True
        )
        collected = DiscoveryResult()
        errors: list[ScoutSourceError] = []
        for (provider, kind, chain), outcome in zip(calls, outcomes, strict=True):
            if isinstance(outcome, MarketDataError):
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
        return await self._finish(started, candidates, collected.rejected, errors + enrich_errors)

    async def lookup_exact_token(self, chain: str, address: str) -> ScoutRun:
        """One exact token (chain + contract / mint), from the first provider that knows it."""
        return await self.lookup_exact_tokens(chain, [address])

    async def lookup_exact_tokens(self, chain: str, addresses: Sequence[str]) -> ScoutRun:
        started = self.now()
        wanted = {canonical_id(chain, a) for a in addresses}
        found: list[ScoutCandidate] = []
        rejected: list[ScoutRejection] = []
        errors: list[ScoutSourceError] = []
        for provider in self._lookup_providers(chain):
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

    def _lookup_providers(self, chain: str) -> list[DiscoveryProvider]:
        able = [p for p in self.providers if "lookup" in p.kinds and chain in p.chains]
        # The enrichment provider first, so exact lookups match the Analyze pipeline.
        return sorted(able, key=lambda p: p.name != self.enrichment_provider)

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
        return candidate.model_copy(update={"first_seen_at": first_seen, "features": features})


def _unique_rejections(rejections: Iterable[ScoutRejection]) -> list[ScoutRejection]:
    seen: set[tuple[str | None, tuple[str, ...]]] = set()
    out = []
    for r in rejections:
        key = (r.canonical_id, tuple(r.reasons))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
