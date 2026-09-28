"""GrowthScoutService: rank Scout's discovery candidates by early, cross-confirmed momentum.

Lightweight by design: it reuses what Scout already measured (market windows, growth
features, stored snapshots), the social layer's momentum (handed in, or the latest stored
one when fresh) and on-chain safety snapshots (handed in, or looked up for at most
`safety.top_k` top Solana candidates, cached by the safety service). It never runs the
News / Technical / Risk / Opportunity / Vision pipeline: that happens in Analyze, when a
candidate is chosen.

1. Candidates are deduplicated by canonical id (the latest observation wins); two tokens
   with the same ticker are never merged.
2. Each is evaluated (stage, ScoutMomentumScore, flags, reasons) from its evidence; the
   stage is stabilized against the stages stored by recent runs (`scan` stores them).
3. Eligibility filters apply (identity, enough data, the discovery mode); ineligible
   candidates are reported in `unranked` with the reason.
4. Eligible candidates are sorted: score, then cross-confirmation, then canonical id
   (deterministic); the top `limit` are returned. Stored history is never pruned here.

Never BUY / SELL: the output is discovery ranking only.
"""

import asyncio
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from upscale.services.market_data import MarketDataError
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.growth.models import (
    GrowthCandidate,
    GrowthScoutResult,
    UniverseReport,
)
from upscale.services.scout.growth.scoring import eligibility, evaluate
from upscale.services.scout.growth.signals import market_evidence, social_evidence
from upscale.services.scout.models import (
    ScoutCandidate,
    ScoutFeedReport,
    ScoutRun,
    ScoutSnapshot,
    ScoutSourceError,
)
from upscale.services.scout.service import RefreshOutcome, ScoutService
from upscale.services.scout.social.models import SocialMomentum
from upscale.services.scout.social.service import (
    SocialScoutService,
    identities_from_candidates,
    known_identities,
)
from upscale.services.scout.social.store import SocialStore
from upscale.services.scout.store import ScoutSnapshotStore, TrackedToken
from upscale.services.solana_chain import KnownPool, OnchainSafetySnapshot


class SafetySource(Protocol):
    """What Growth Scout needs from `SolanaSafetyService` (cached, rate-limited)."""

    async def get_snapshot(
        self, mint: str, pools: Sequence[KnownPool] = ()
    ) -> OnchainSafetySnapshot: ...


class GrowthScoutService:
    def __init__(
        self,
        store: ScoutSnapshotStore,
        config: GrowthConfig | None = None,
        social_store: SocialStore | None = None,
        safety: SafetySource | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.store = store
        self.config = config or GrowthConfig()
        self.social_store = social_store
        self.safety = safety
        self.now = now

    async def rank(
        self,
        candidates: Sequence[ScoutCandidate],
        social: Mapping[str, SocialMomentum] | None = None,
        safety: Mapping[str, OnchainSafetySnapshot] | None = None,
        limit: int | None = None,
        carried: Mapping[str, float] | None = None,
    ) -> GrowthScoutResult:
        """`carried`: canonical id -> minutes old, for candidates ranked on a last good
        observation (labeled STALE_CARRIED and penalized)."""
        cfg = self.config
        stale = dict(carried or {})
        now = self.now()
        limit = limit if limit is not None else cfg.default_limit
        unique = _latest_per_token(candidates)
        histories = await asyncio.gather(*(self._history(c, now) for c in unique))
        stages = await self.store.recent_stages(
            [c.canonical_id for c in unique],
            now - timedelta(minutes=cfg.stability.memory_minutes),
        )
        momentum = dict(social or {})
        notes: list[str] = []
        if self.social_store is not None:
            # A token no provider could search this run (e.g. deferred by X's budget) keeps
            # its latest real measurement; `social_evidence` drops it once too old.
            missing = [
                c.canonical_id
                for c in unique
                if c.canonical_id not in momentum or momentum[c.canonical_id].state == "UNAVAILABLE"
            ]
            stored = await asyncio.gather(*(self._stored_momentum(cid) for cid in missing))
            momentum |= {
                m.canonical_id: m
                for m in stored
                if m is not None and (m.state != "UNAVAILABLE" or m.canonical_id not in momentum)
            }
        snapshots = dict(safety or {})
        evidence = {
            c.canonical_id: (c, market_evidence(c, h, cfg), social_evidence(momentum.get(c.canonical_id), now, cfg))
            for c, h in zip(unique, histories, strict=True)
        }  # fmt: skip

        def assess(cid: str) -> GrowthCandidate:
            c, ev, soc = evidence[cid]
            g = evaluate(
                c, ev, soc, snapshots.get(cid), cfg, now, stages.get(cid, ()), stale.get(cid)
            )
            g.ineligible_reasons = eligibility(g, cfg)
            g.eligible = not g.ineligible_reasons
            return g

        assessed = {cid: assess(cid) for cid in evidence}
        # On-chain safety for the leading Solana candidates (bounded; cached upstream).
        if self.safety is not None and cfg.safety.top_k > 0:
            leaders = [
                g
                for g in _ordered(assessed.values())
                if g.eligible and g.chain == "solana" and g.canonical_id not in snapshots
            ][: cfg.safety.top_k]
            fetched = await asyncio.gather(
                *(self._safety_snapshot(evidence[g.canonical_id][0]) for g in leaders),
                return_exceptions=True,
            )
            failed = 0
            for g, outcome in zip(leaders, fetched, strict=True):
                if isinstance(outcome, OnchainSafetySnapshot):
                    snapshots[g.canonical_id] = outcome
                    assessed[g.canonical_id] = assess(g.canonical_id)
                elif isinstance(outcome, MarketDataError):
                    failed += 1
                elif isinstance(outcome, BaseException):
                    raise outcome  # a bug, never hidden
            if failed:
                notes.append(
                    f"on-chain safety lookup failed for {failed} candidate(s): their safety "
                    "stays INSUFFICIENT_SAFETY_DATA (never assumed safe)"
                )

        ranked = _ordered(g for g in assessed.values() if g.eligible)
        for i, g in enumerate(ranked, start=1):
            g.rank = i
        unranked = sorted(
            (g for g in assessed.values() if not g.eligible), key=lambda g: g.canonical_id
        )
        if len(candidates) != len(unique):
            notes.append(
                f"{len(candidates) - len(unique)} duplicate observation(s) of the same token "
                "merged by canonical id (latest kept)"
            )
        return GrowthScoutResult(
            computed_at=now,
            mode=cfg.mode,
            evaluated=len(unique),
            eligible=len(ranked),
            limit=limit,
            candidates=ranked[:limit],
            unranked=unranked,
            notes=notes,
        )

    async def scan(
        self,
        scout: ScoutService,
        social: SocialScoutService | None = None,
        limit: int | None = None,
    ) -> GrowthScoutResult:
        """One ranking run:

        1. discovery, in the providers' unreserved capacity (the refresh reservation is
           held, so discovery can't consume it);
        2. refresh of tracked tokens (surfaced by discovery within the horizon, not this
           run) in the reserved capacity: leaders, rising stages and the stalest first,
           rotating so every tracked token is revisited; each asked first of the provider
           that last priced it;
        3. the reservation is released and discovery listings that failed for capacity are
           retried on whatever the refresh left unused;
        4. tracked tokens a provider failure kept from being refreshed are carried on their
           last good observation for a short grace (labeled STALE_CARRIED, penalized);
           tokens every lookup confirmed gone / unusable are dropped;
        5. provisional market-only ranking, social observation of the current rankable
           candidates (paid searches by provisional rank, fairly), the final ranking; each
           token's stage and rank are stored (stage stability, refresh priority).
        """
        tc = self.config.tracking
        now = self.now()
        hold = getattr(scout, "hold_reservations", None)
        if callable(hold):
            hold()
        run = await scout.discover()
        discovered_ids = {c.canonical_id for c in run.candidates}
        plan = await self._plan_refresh(scout, discovered_ids, now)
        refresh, planned_failed = await self._run_refresh(scout, plan.selected)
        release = getattr(scout, "release_reservations", None)
        if callable(release):
            release()
        retried: list[ScoutCandidate] = []
        # Feeds discovery deferred (over capacity) or that failed, retried on whatever the
        # refresh left of its reservation (never before the refresh is done).
        failed_listings: set[tuple[str, str, str]] = {
            (e.provider, e.kind, e.chain) for e in run.errors if e.kind != "lookup" and e.chain
        }
        for report in run.feeds:
            for feed in report.deferred:
                kind, chain = feed.split(":", 1)
                failed_listings.add((report.provider, kind, chain))
        retry_errors: list[ScoutSourceError] = []
        feeds = list(run.feeds)
        retried_feeds = 0
        if failed_listings and callable(release):
            retry = await scout.discover(only=failed_listings)
            retried = [c for c in retry.candidates if c.canonical_id not in discovered_ids]
            retry_errors = retry.errors
            discovered_ids |= {c.canonical_id for c in retry.candidates}
            feeds = _merge_feed_reports(run.feeds, retry.feeds)
            # Feeds the retry really sent (with no capacity left it defers them again).
            retried_feeds = (
                sum(len(r.executed) for r in retry.feeds)
                if retry.feeds
                else len(
                    {(e.provider, e.kind, e.chain) for e in retry.errors if e.kind != "lookup"}
                )
            )
        refreshed = [c for c in refresh.candidates if c.canonical_id not in discovered_ids]
        found = {c.canonical_id for c in refreshed} | discovered_ids
        missing = [t for t in plan.selected if t.canonical_id not in found]
        # Unresolved: the provider the token was planned for failed (not known to be gone).
        # A token that provider answered for without a usable market is gone / unusable.
        unresolved = [t for t in missing if t.canonical_id in planned_failed]
        gone = [t.canonical_id for t in missing if t.canonical_id not in planned_failed]
        unusable = len(gone)
        if gone:  # confirmed by the provider that prices them: stop spending refreshes
            await self.store.untrack(gone)
        carried, carried_age = await self._carry_forward(unresolved, now)
        current = [*run.candidates, *retried, *refreshed]
        momentum: dict[str, SocialMomentum] = {}
        social_checks = []
        provisional = await self._provisional(current) if social is not None else []
        if social is not None and provisional:
            observed = await social.observe(
                identities_from_candidates(provisional),
                await known_identities(scout.store),
                {c.canonical_id: c.features for c in provisional if c.features},
                priority={c.canonical_id: i for i, c in enumerate(provisional)},
            )
            momentum = {m.canonical_id: m for m in observed.momentum}
            social_checks = observed.providers
        result = await self.rank([*current, *carried], momentum, limit=limit, carried=carried_age)
        result.social_checks = list(social_checks)
        evaluated = [
            g for g in [*result.candidates, *result.unranked] if g.data_status == "CURRENT"
        ]
        await self.store.record_stages(
            result.computed_at,
            {g.canonical_id: g.stage for g in evaluated},
            {g.canonical_id: (g.rank, g.score) for g in evaluated},
        )
        horizon = now - timedelta(hours=tc.horizon_hours)
        result.universe = UniverseReport(
            discovered=len(discovered_ids),
            discovery_retried=retried_feeds,
            feeds=feeds,
            refreshed=len({c.canonical_id for c in refreshed}),
            carried_stale=len(carried),
            expired=await self.store.expired_count(horizon, horizon),
            unusable=unusable,
            unresolved=len(unresolved),
            deferred=plan.deferred,
            horizon_hours=tc.horizon_hours,
            tracked=plan.tracked,
            refresh_capacity_requests=plan.capacity,
            refresh_capacity_tokens=plan.capacity_tokens,
            estimated_max_revisit_minutes=plan.revisit_minutes,
            horizon_covered=plan.horizon_covered,
        )
        if not plan.horizon_covered:
            result.notes.append(
                f"tracking horizon is {tc.horizon_hours:g}h, but refresh capacity "
                f"({plan.capacity_tokens} tokens per run) revisits {plan.tracked} tracked "
                f"tokens only about every {plan.revisit_minutes:.0f} minutes"
            )
        errors = [*run.errors, *refresh.errors, *retry_errors]
        if errors:
            result.notes.append(
                f"{len(errors)} discovery / refresh source(s) failed: "
                + "; ".join(sorted({f"{e.provider} {e.kind}" for e in errors}))
            )
        return result

    # --- internals ------------------------------------------------------------------------

    async def _provisional(self, candidates: Sequence[ScoutCandidate]) -> list[ScoutCandidate]:
        """The rankable candidates in provisional order: ranked from market evidence alone
        (no network: no social, no safety lookups)."""
        cfg, now = self.config, self.now()
        unique = _latest_per_token(candidates)
        histories = await asyncio.gather(*(self._history(c, now) for c in unique))
        stages = await self.store.recent_stages(
            [c.canonical_id for c in unique],
            now - timedelta(minutes=cfg.stability.memory_minutes),
        )
        unmeasured = social_evidence(None, now, cfg)
        by_id = {c.canonical_id: c for c in unique}
        assessed = []
        for c, h in zip(unique, histories, strict=True):
            g = evaluate(c, market_evidence(c, h, cfg), unmeasured, None, cfg, now,
                         stages.get(c.canonical_id, ()))  # fmt: skip
            if not eligibility(g, cfg):
                assessed.append(g)
        return [by_id[g.canonical_id] for g in _ordered(assessed)]

    async def _plan_refresh(
        self, scout: ScoutService, discovered: set[str], now: datetime
    ) -> "RefreshPlan":
        """Which tracked tokens this run refreshes, within the providers' reserved refresh
        capacity (batched lookups per provider and chain), highest priority first."""
        tc = self.config.tracking
        tracked = [
            t
            for t in await self.store.tracked_state(now - timedelta(hours=tc.horizon_hours))
            if t.canonical_id not in discovered
        ]
        latest = await self.store.latest_growth([t.canonical_id for t in tracked])
        rotation_round = await self.store.ranking_runs()
        capacity_of = getattr(scout, "refresh_capacity", None)
        capacity: dict[str, int] = capacity_of() if callable(capacity_of) else {}
        scout_config = getattr(scout, "config", None)
        batch = scout_config.lookup_batch_size if scout_config is not None else 30
        default_provider = getattr(scout, "enrichment_provider", None)

        def priority(t: TrackedToken) -> float:
            value = max(0.0, (now - t.last_seen_at).total_seconds() / 60)
            last = latest.get(t.canonical_id)
            if last is not None:
                _, stage, rank = last
                if rank is not None and rank <= tc.top_n:
                    value += tc.top_bonus_minutes
                if stage in ("ACCELERATING", "EARLY"):
                    value += tc.rising_stage_bonus_minutes
            if t.last_seen_at <= t.last_discovered_at:  # not refreshed since discovered
                value += tc.never_refreshed_bonus_minutes
            return value

        ids = sorted(t.canonical_id for t in tracked)
        shift = rotation_round % len(ids) if ids else 0
        rotation = {cid: (i - shift) % len(ids) for i, cid in enumerate(ids)}
        ordered = sorted(tracked, key=lambda t: (-priority(t), rotation[t.canonical_id]))
        selected: list[TrackedToken] = []
        per_group: dict[tuple[str, str], int] = {}
        used: dict[str, int] = {}
        for t in ordered:
            if len(selected) >= tc.max_refresh:
                break
            provider = t.market_provider or default_provider or ""
            if capacity and provider in capacity:
                count = per_group.get((provider, t.chain), 0)
                extra = 1 if count % batch == 0 else 0  # a new batch request is needed
                if used.get(provider, 0) + extra > capacity[provider]:
                    continue
                used[provider] = used.get(provider, 0) + extra
                per_group[(provider, t.chain)] = count + 1
            selected.append(t)
        # How far tracking really reaches: per provider, the batched requests needed to
        # revisit every token it prices, over what it reserves per run; the slowest
        # provider, times the typical interval between runs.
        routed: dict[tuple[str, str], int] = {}
        for t in tracked:
            key = (t.market_provider or default_provider or "", t.chain)
            routed[key] = routed.get(key, 0) + 1
        runs_needed = 1 if tracked else 0
        for provider in {q for q, _ in routed}:
            cap = capacity.get(provider) if capacity else None
            if cap is None:
                continue
            requests = sum(math.ceil(m / batch) for (q, _), m in routed.items() if q == provider)
            runs_needed = max(runs_needed, math.ceil(requests / cap) if cap else 10**6)
        if tc.max_refresh:
            runs_needed = max(runs_needed, math.ceil(len(tracked) / tc.max_refresh))
        capacity_tokens = (
            min(tc.max_refresh, sum(capacity.values()) * batch) if capacity else tc.max_refresh
        )
        if not capacity:
            runs_needed = math.ceil(len(tracked) / capacity_tokens) if capacity_tokens else 10**6
        times = await self.store.ranking_run_times(6)
        gaps = sorted((a - b).total_seconds() / 60 for a, b in zip(times, times[1:], strict=False))
        interval = gaps[len(gaps) // 2] if gaps else tc.expected_interval_minutes
        revisit = runs_needed * interval if runs_needed < 10**6 else None
        return RefreshPlan(
            selected=selected,
            tracked=len(tracked),
            deferred=len(tracked) - len(selected),
            capacity=capacity,
            capacity_tokens=capacity_tokens,
            revisit_minutes=round(revisit, 1) if revisit is not None else None,
            horizon_covered=revisit is not None and revisit <= tc.horizon_hours * 60,
        )

    async def _run_refresh(
        self, scout: ScoutService, selected: Sequence[TrackedToken]
    ) -> tuple[ScoutRun, set[str]]:
        """The refresh run, and the tokens whose planned provider failed."""
        if not selected:
            return ScoutRun(started_at=self.now(), candidates=[]), set()
        refresh = getattr(scout, "refresh_tokens", None)
        if callable(refresh):
            outcome: RefreshOutcome = await refresh(
                [(t.chain, t.address, t.market_provider) for t in selected]
            )
            return outcome.run, outcome.unresolved
        by_chain: dict[str, list[str]] = {}
        for t in selected:
            by_chain.setdefault(t.chain, []).append(t.address)
        runs = await asyncio.gather(
            *(scout.lookup_exact_tokens(chain, addrs) for chain, addrs in by_chain.items())
        )
        run = ScoutRun(
            started_at=self.now(),
            candidates=[c for r in runs for c in r.candidates],
            rejected=[x for r in runs for x in r.rejected],
            errors=[e for r in runs for e in r.errors],
        )
        # Without a planned provider, a token on a chain whose lookup failed is unresolved.
        failed = {e.chain for e in run.errors}
        found = {c.canonical_id for c in run.candidates}
        return run, {
            t.canonical_id for t in selected if t.chain in failed and t.canonical_id not in found
        }

    async def _carry_forward(
        self, unresolved: Sequence[TrackedToken], now: datetime
    ) -> tuple[list[ScoutCandidate], dict[str, float]]:
        """Last good observations of unresolved tokens, while within the grace period
        (measured from the observation itself, so it never extends)."""
        grace = self.config.tracking.grace_minutes
        if not unresolved or grace <= 0:
            return [], {}
        latest = await self.store.latest_candidates([t.canonical_id for t in unresolved])
        carried, ages = [], {}
        for c in latest.values():
            age = (now - c.observed_at).total_seconds() / 60
            if age <= grace:
                carried.append(c)
                ages[c.canonical_id] = age
        return carried, ages

    async def _history(self, c: ScoutCandidate, now: datetime) -> list[ScoutSnapshot]:
        hours = max(
            self.config.technical.lookback_hours,
            self.config.earliness.first_seen_lookback_hours,
        )
        return await self.store.history(c.canonical_id, since=now - timedelta(hours=hours))

    async def _stored_momentum(self, cid: str) -> SocialMomentum | None:
        """The latest stored momentum from a run that could search the token (the latest
        of any kind when none could)."""
        assert self.social_store is not None
        history = await self.social_store.momentum_history(cid)
        measured = [m for m in history if m.state != "UNAVAILABLE"]
        return (measured or history)[-1] if history else None

    async def _safety_snapshot(self, c: ScoutCandidate) -> OnchainSafetySnapshot:
        assert self.safety is not None
        pools = [KnownPool(address=c.pool.address, dex=c.pool.dex, eligible=True)]
        return await self.safety.get_snapshot(c.address, pools)


def _latest_per_token(candidates: Sequence[ScoutCandidate]) -> list[ScoutCandidate]:
    latest: dict[str, ScoutCandidate] = {}
    for c in candidates:
        seen = latest.get(c.canonical_id)
        if seen is None or c.observed_at >= seen.observed_at:
            latest[c.canonical_id] = c
    return sorted(latest.values(), key=lambda c: c.canonical_id)


def _ordered(items: Iterable[GrowthCandidate]) -> list[GrowthCandidate]:
    def key(g: GrowthCandidate) -> tuple[float, float, str]:
        cross = next(f.score for f in g.scout_momentum.families if f.family == "cross_confirmation")
        return (-g.score, -cross, g.canonical_id)

    return sorted(items, key=key)


@dataclass
class RefreshPlan:
    selected: list[TrackedToken]
    tracked: int
    deferred: int
    capacity: dict[str, int] = field(default_factory=dict)
    capacity_tokens: int = 0
    revisit_minutes: float | None = None
    horizon_covered: bool = True


def _merge_feed_reports(
    first: Sequence[ScoutFeedReport], retry: Sequence[ScoutFeedReport]
) -> list[ScoutFeedReport]:
    """One report per provider across discovery and its capacity retry."""
    later = {r.provider: r for r in retry}
    merged = []
    for r in first:
        again = later.get(r.provider)
        if again is None:
            merged.append(r)
            continue
        retried = set(again.executed) | set(again.deferred)
        merged.append(
            ScoutFeedReport(
                provider=r.provider,
                available=r.available,
                executed=[*r.executed, *again.executed],
                deferred=again.deferred + [f for f in r.deferred if f not in retried],
                failed=[*(f for f in r.failed if f not in again.executed), *again.failed],
                requests=r.requests + again.requests,
                next_scheduled=again.next_scheduled,
            )
        )
    return merged
