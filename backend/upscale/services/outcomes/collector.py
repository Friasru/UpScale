"""The outcome collector: measures due horizons, lowest priority on every shared quota.

Each cycle:

1. **What is due**: every PENDING horizon whose window ended at least `settle_seconds`
   ago (all of them each cycle, so priority is global). Nothing is measured early: a
   horizon is never completed before its window has passed.
2. **Stored evidence first (no network)**: Scout's snapshots of the exact token *and*
   exact pool (the horizon-end market state, and a snapshot-only price fallback), and
   Scout's stored stage / rank / score nearest the horizon end.
3. **Provider evidence, only if needed and affordable**: batched exact-pool lookups (DEX
   Screener, several tokens per request) for a horizon-end market state no snapshot
   provides, while it can still describe the horizon end; and candles of the exact pool
   (GeckoTerminal) or exact exchange market (Kraken), one request per market and
   timeframe for every window due. Requests run in the "outcomes" lane: they can never
   use the capacity reserved for Analyze ("interactive") or Scout refresh, leave
   `min_free_calls` of each provider's window to Scout discovery, stop at
   `max_requests_per_cycle`, and are skipped entirely while Scout or Analyze is running
   (`busy`). Deferred work waits for a later cycle.
4. **Finalize or wait**: COMPLETE when the price path (candles) and the horizon-end market
   state are both measured; a confirmed terminal state (pool gone, liquidity collapsed,
   no trades) finalizes as soon as it is known; otherwise the horizon stays PENDING until
   `retry_minutes` after it was due, then becomes PARTIAL (stored snapshots only, or one
   part missing) or UNAVAILABLE, with every missing part's reason. Nothing is ever
   dropped or deleted.

The scheduler sleeps until the next horizon becomes collectable (`next_wake`), at most
`max_sleep_seconds`, and wakes early when new observations are anchored.
"""

import asyncio
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from upscale.services.chains import normalize_address
from upscale.services.geckoterminal import MAX_LIMIT, DexCandleService
from upscale.services.market_data import (
    MAX_CANDLES,
    TIMEFRAME_SECONDS,
    AssetNotFoundError,
    Candle,
    CandleSeries,
    MarketDataError,
    MarketDataService,
    ProviderRateLimitedError,
    Timeframe,
)
from upscale.services.outcomes.config import HorizonSpec, OutcomeConfig
from upscale.services.outcomes.metrics import (
    first_event,
    in_window,
    market_at_horizon,
    market_status,
    path_from_candles,
    path_from_points,
    trigger_outcome,
)
from upscale.services.outcomes.models import (
    DecisionObservation,
    MarketAtHorizon,
    MarketSource,
    ObservedMarket,
    PricePath,
    PriceRef,
    ScoutObservation,
    TriggerOutcome,
)
from upscale.services.outcomes.store import DueHorizon, HorizonUpdate, OutcomeStore
from upscale.services.quota import request_lane
from upscale.services.scout.models import ScoutMarketMetrics, ScoutSnapshot
from upscale.services.scout.normalize import canonical_id, metrics_from_pool
from upscale.services.scout.providers import (
    DexScreenerDiscoveryProvider,
    GeckoTerminalDiscoveryProvider,
)
from upscale.services.scout.store import ScoutSnapshotStore
from upscale.services.solana_dex import DexPool

# The request lane outcome work runs in: no provider reserves capacity for it.
OUTCOME_LANE = "outcomes"
logger = logging.getLogger("upscale.outcomes")


# --- Evidence sources -----------------------------------------------------------------------


class CandleSource(Protocol):
    def provider(self, ref: PriceRef) -> str: ...

    def headroom(self, ref: PriceRef) -> int:
        """Requests the outcome lane may start for this market right now."""
        ...

    async def window(
        self, ref: PriceRef, timeframe: Timeframe, start: datetime, end: datetime, now: datetime
    ) -> CandleSeries:
        """Candles of exactly this market covering [start, end] (raises MarketDataError)."""
        ...


PoolKey = tuple[str, str]  # (token address, pool address)


class PoolSource(Protocol):
    """Current state of exact pools. A source is authoritative for the pools it priced
    itself: only then does "not listed" mean the pool is gone."""

    name: str

    def headroom(self, chain: str) -> int: ...

    def requests(self, count: int) -> int:
        """Requests needed to look up `count` pools of one chain."""
        ...

    async def pools(self, chain: str, keys: Sequence[PoolKey]) -> dict[PoolKey, DexPool | None]:
        """Each exact pool of each exact token, or None when this source answered without
        it (a pool of another token never counts). Raises MarketDataError."""
        ...


class ProviderCandles:
    """Pool candles from the shared GeckoTerminal service; exchange candles from the
    market data service. Both are counted against their shared limits."""

    def __init__(self, dex: DexCandleService, exchange: MarketDataService, min_free: int):
        self.dex = dex
        self.exchange = exchange
        self.min_free = min_free

    def provider(self, ref: PriceRef) -> str:
        return self.dex.provider_name if ref.kind == "dex_pool" else ref.provider or "exchange"

    def headroom(self, ref: PriceRef) -> int:
        if ref.kind == "dex_pool":
            if ref.chain is None or not self.dex.covers(ref.chain):
                return 0
            return self.dex.limiter.available(OUTCOME_LANE) - self.min_free
        if ref.provider is None:
            return 0
        return self.exchange.available_calls(ref.provider) - self.min_free

    async def window(
        self, ref: PriceRef, timeframe: Timeframe, start: datetime, end: datetime, now: datetime
    ) -> CandleSeries:
        interval = TIMEFRAME_SECONDS[timeframe]
        if ref.kind == "dex_pool":
            assert ref.chain is not None and ref.pool_address is not None
            need = math.ceil((end - start).total_seconds() / interval) + 2
            return await self.dex.get_window(
                ref.chain,
                ref.pool_address,
                timeframe,
                min(need, MAX_LIMIT),
                before=end + timedelta(seconds=interval),
                lane=OUTCOME_LANE,
                canonical_id=canonical_id(ref.chain, ref.token_address or ""),
            )
        # Exchange candles: the latest `limit` completed ones, which must reach back to
        # `start` (older windows are beyond what the provider returns).
        need = math.ceil((now - start).total_seconds() / interval) + 2
        if need > MAX_CANDLES or ref.symbol is None:
            raise AssetNotFoundError(
                "the window is older than the exchange candle history UpScale can request"
            )
        serving = next(
            (p.name for p in self.exchange.candle_providers if timeframe in p.supported_timeframes),
            None,
        )
        if serving != ref.provider:
            # Checked before any request: another provider's candles are another market.
            raise AssetNotFoundError(
                f"{timeframe} candles come from {serving or 'no provider'}, not the decision's "
                f"market ({ref.provider} {ref.pair or ''}): never stitched"
            )
        series = await self.exchange.get_candles(ref.symbol, timeframe, need)
        if series.provider != ref.provider or (ref.pair and series.pair != ref.pair):
            raise AssetNotFoundError(
                f"candles now come from {series.provider} {series.pair or ''}, not the "
                f"decision's market ({ref.provider} {ref.pair or ''}): never stitched"
            )
        return series


class DexScreenerPools:
    """Exact-pool state from DEX Screener's batched token lookup (through its gate)."""

    name = "DEX Screener"

    def __init__(self, provider: DexScreenerDiscoveryProvider, min_free: int):
        self.provider = provider
        self.min_free = min_free

    def headroom(self, chain: str) -> int:
        if chain not in self.provider.chains:
            return 0
        return self.provider.gate.available(OUTCOME_LANE) - self.min_free

    def requests(self, count: int) -> int:
        return math.ceil(count / self.provider.config.lookup_batch_size)

    async def pools(self, chain: str, keys: Sequence[PoolKey]) -> dict[PoolKey, DexPool | None]:
        with request_lane(OUTCOME_LANE):
            found = await self.provider.token_pools(chain, sorted({t for t, _ in keys}))
        out: dict[PoolKey, DexPool | None] = {}
        for token, pool in keys:
            pools = found.get(canonical_id(chain, token), [])
            out[(token, pool)] = next(
                (p for p in pools if _same(chain, p.pair_address, pool)), None
            )
        return out


class GeckoTerminalPools:
    """Exact-pool state from GeckoTerminal's batched pool lookup (through its gate, on the
    process-wide GeckoTerminal quota)."""

    name = "GeckoTerminal"

    def __init__(self, provider: GeckoTerminalDiscoveryProvider, min_free: int):
        self.provider = provider
        self.min_free = min_free

    def headroom(self, chain: str) -> int:
        if chain not in self.provider.chains:
            return 0
        return self.provider.gate.available(OUTCOME_LANE) - self.min_free

    def requests(self, count: int) -> int:
        return math.ceil(count / self.provider.config.lookup_batch_size)

    async def pools(self, chain: str, keys: Sequence[PoolKey]) -> dict[PoolKey, DexPool | None]:
        with request_lane(OUTCOME_LANE):
            found = await self.provider.pools_by_address(chain, sorted({p for _, p in keys}))
        out: dict[PoolKey, DexPool | None] = {}
        for token, pool in keys:
            out[(token, pool)] = next(
                (
                    p
                    for p in found
                    if _same(chain, p.pair_address, pool) and _same(chain, p.base.address, token)
                ),
                None,
            )
        return out


# --- The collector ------------------------------------------------------------------------


@dataclass
class CycleReport:
    at: datetime
    due: int = 0
    finalized: dict[str, int] = field(default_factory=dict)
    still_pending: int = 0
    requests: dict[str, int] = field(default_factory=dict)
    deferred_for_quota: int = 0
    waiting_to_share_candles: int = 0
    network_skipped: bool = False
    reused_snapshots: int = 0


@dataclass
class _Item:
    due: DueHorizon
    spec: HorizonSpec
    start: datetime
    end: datetime
    tolerance: timedelta
    ref: PriceRef
    reference_price: float | None
    token_id: str | None  # canonical chain:address, when known
    observed: ObservedMarket | None
    reference_liquidity: float | None
    rank: int  # candle priority (decisions first, then Scout rank)
    pool_source: str | None  # the provider that priced the observed pool
    snapshots: list[ScoutSnapshot] = field(default_factory=list)
    market: MarketAtHorizon | None = None
    series: CandleSeries | None = None
    stored_price: PricePath | None = None  # a candle path stored by an earlier attempt
    candles_final: bool = False  # no point asking again (e.g. the provider doesn't know it)
    provider_failed: bool = False
    missing: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.due.kind


class OutcomeCollector:
    def __init__(
        self,
        store: OutcomeStore,
        config: OutcomeConfig,
        scout_store: ScoutSnapshotStore | None = None,
        candles: CandleSource | None = None,
        pools: Sequence[PoolSource] = (),
        busy: Callable[[], bool] = lambda: False,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.store = store
        self.config = config
        self.scout_store = scout_store
        self.candles = candles
        self.pools = list(pools)
        self.busy = busy
        self.now = now
        self.last_cycle: CycleReport | None = None
        self._wake = asyncio.Event()

    def wake(self) -> None:
        """New observations were anchored: re-plan the next wake-up."""
        self._wake.set()

    async def run_forever(self, stop: asyncio.Event) -> None:
        cc = self.config.collector
        await self.store.ensure_horizons(self.config.horizons)
        while not stop.is_set():
            try:
                await self.collect()
            except Exception:  # background work must never take the app down
                logger.exception("outcome collection cycle failed")
            wake_at = await self.store.next_wake(cc.settle_seconds, cc.retry_seconds)
            delay = (
                cc.max_sleep_seconds
                if wake_at is None
                else min(cc.max_sleep_seconds, (wake_at - self.now()).total_seconds())
            )
            delay = max(cc.min_sleep_seconds, delay)
            self._wake.clear()
            waiters = [asyncio.ensure_future(stop.wait()), asyncio.ensure_future(self._wake.wait())]
            try:
                await asyncio.wait(waiters, timeout=delay, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waiters:
                    w.cancel()

    async def collect(self) -> CycleReport:
        cc = self.config.collector
        now = self.now()
        report = CycleReport(at=now)
        # Every ended PENDING horizon is considered each cycle (retry spacing only decides
        # when the collector wakes), so priority is global: work that became eligible a few
        # seconds after lower-priority work can't be starved by it.
        due = await self.store.due(now, cc.settle_seconds, 0.0, cc.max_due_per_cycle)
        report.due = len(due)
        # Other ended-but-pending horizons of the same observations join this cycle, so one
        # candle request can serve them all.
        due += await self.store.companions(due, now - timedelta(seconds=cc.settle_seconds))
        items = [self._item(d) for d in due]
        for item in items:
            await self._stored_evidence(item, report)
        if items and self.busy():
            report.network_skipped = True
        elif items:
            await self._lookup_pools(items, now, report)
            await self._fetch_candles(items, now, report)
        for item in items:
            status = await self._finish(item, now)
            if status is None:
                report.still_pending += 1
            else:
                report.finalized[status] = report.finalized.get(status, 0) + 1
        self.last_cycle = report
        return report

    # --- per item ---------------------------------------------------------------------------

    def _spec(self, d: DueHorizon) -> HorizonSpec:
        spec = self.config.horizon(d.horizon.horizon)
        if spec is not None:
            return spec
        # A horizon no longer configured: still measured, with defaults for its length.
        minutes = d.horizon.horizon_minutes
        tf: Timeframe = (
            "1m"
            if minutes <= 60
            else "5m"
            if minutes <= 240
            else "15m"
            if minutes <= 1440
            else "1h"
        )
        return HorizonSpec(label=d.horizon.horizon, minutes=minutes, candles=tf,
                           retry_minutes=max(120, minutes))  # fmt: skip

    def _item(self, d: DueHorizon) -> _Item:
        cc = self.config.collector
        spec = self._spec(d)
        o = d.observation
        seconds = d.horizon.horizon_minutes * 60
        tolerance = timedelta(
            seconds=max(cc.min_market_tolerance_seconds, cc.market_tolerance * seconds)
        )
        if isinstance(o, ScoutObservation):
            start = o.observed_at
            item = _Item(
                due=d, spec=spec, start=start, end=start + timedelta(seconds=seconds),
                tolerance=tolerance, ref=o.price_ref, reference_price=o.market.price_usd,
                token_id=o.canonical_id, observed=o.market,
                reference_liquidity=o.market.liquidity_usd, rank=1_000 + o.rank,
                pool_source=o.market_provider,
            )  # fmt: skip
        else:
            assert isinstance(o, DecisionObservation)
            start = o.analyzed_at
            token = o.asset_id if o.chain and o.address else None
            item = _Item(
            due=d, spec=spec, start=start, end=start + timedelta(seconds=seconds),
            tolerance=tolerance, ref=o.price_ref, reference_price=o.reference_price,
            token_id=token, observed=None, reference_liquidity=o.liquidity_usd, rank=0,
            pool_source="DEX Screener",  # Analyze's DEX market and its pools come from it
            )  # fmt: skip
        # Parts an earlier attempt already measured are kept (first write wins).
        item.market = d.horizon.market
        if d.horizon.price is not None and d.horizon.price.source == "candles":
            item.stored_price = d.horizon.price
        return item

    async def _stored_evidence(self, item: _Item, report: CycleReport) -> None:
        """Scout's stored snapshots of the exact token and pool, and its stored stage."""
        if self.scout_store is None or item.token_id is None or item.ref.kind != "dex_pool":
            return
        history = await self.scout_store.history(item.token_id, since=item.start, limit=2000)
        pool = item.ref.pool_address
        item.snapshots = [
            s
            for s in history
            if s.pool_address == pool and s.observed_at <= item.end + item.tolerance
        ]
        if other := {s.pool_address for s in history if s.pool_address != pool}:
            item.missing.append(
                f"Scout observed {len(other)} other pool(s) of this token after the "
                "observation; only the observed pool is used (never stitched)"
            )
        near = [
            s
            for s in item.snapshots
            if s.observed_at > item.start
            and abs((s.observed_at - item.end).total_seconds()) <= item.tolerance.total_seconds()
        ]
        if near and item.market is None:
            s = min(
                near, key=lambda x: (abs((x.observed_at - item.end).total_seconds()), x.observed_at)
            )
            item.market = self._market(
                item, s.metrics, "scout_snapshot", s.provider, s.observed_at, True
            )
            report.reused_snapshots += 1  # fmt: skip

    def _market(
        self,
        item: _Item,
        metrics: ScoutMarketMetrics | None,
        source: MarketSource,
        provider: str,
        at: datetime,
        found: bool,
    ) -> MarketAtHorizon:
        cfg = self.config.collapse
        return market_at_horizon(
            metrics,
            item.observed,
            source=source,
            provider=provider,
            observed_at=at,
            pool_found=found,
            liquidity_drop_pct=cfg.liquidity_drop_pct,
            liquidity_floor_usd=cfg.liquidity_floor_usd,
            reference_liquidity=item.reference_liquidity,
        )

    # --- network ----------------------------------------------------------------------------

    def _budget(self, provider: str) -> int:
        cc = self.config.collector
        return cc.request_budgets.get(provider, cc.max_requests_per_cycle)

    async def _lookup_pools(self, items: list[_Item], now: datetime, report: CycleReport) -> None:
        """Horizon-end state of exact pools no Scout snapshot described, batched per source
        and chain, while a current state can still describe the horizon end."""
        if not self.pools:
            return
        by_name = {p.name: p for p in self.pools}
        groups: dict[tuple[str, str], list[_Item]] = {}
        for i in items:
            if not (
                i.market is None
                and i.ref.kind == "dex_pool"
                and i.ref.chain
                and i.ref.token_address
                and i.ref.pool_address
                and now <= i.end + i.tolerance
            ):
                continue
            source = by_name.get(i.pool_source or "") or self.pools[0]
            groups.setdefault((source.name, i.ref.chain or ""), []).append(i)
        for (name, chain), group in sorted(groups.items()):
            source = by_name[name]
            keys = sorted({(i.ref.token_address or "", i.ref.pool_address or "") for i in group})
            batches = source.requests(len(keys))
            used = report.requests.get(name, 0)
            if used + batches > self._budget(name) or source.headroom(chain) < batches:
                report.deferred_for_quota += len(group)
                for i in group:
                    i.missing.append(f"{name} pool lookup deferred: background quota in use")
                continue
            report.requests[name] = used + batches
            try:
                found = await source.pools(chain, keys)
            except MarketDataError as exc:
                for i in group:
                    i.provider_failed = True
                    i.missing.append(f"{name} pool lookup failed: {exc}")
                continue
            at = self.now()
            for i in group:
                key = (i.ref.token_address or "", i.ref.pool_address or "")
                if key not in found:
                    i.missing.append(f"{name} did not answer for this pool")
                    continue
                pool = found[key]
                if pool is None and i.pool_source != name:
                    # Not proof the pool is gone: this source never priced it.
                    i.missing.append(
                        f"{name} does not list this pool (priced by {i.pool_source}): "
                        "horizon-end market state unknown"
                    )
                    continue
                metrics = metrics_from_pool(pool) if pool is not None else None
                i.market = self._market(i, metrics, "pool_lookup", name, at, pool is not None)

    async def _fetch_candles(self, items: list[_Item], now: datetime, report: CycleReport) -> None:
        if self.candles is None:
            for i in items:
                i.missing.append("no candle source configured")
            return
        groups: dict[tuple[str, ...], list[_Item]] = {}
        for i in items:
            if i.stored_price is not None:
                continue
            if (longer := self._shares_later(i, now)) is not None:
                report.waiting_to_share_candles += 1
                i.missing.append(f"candles wait for the {longer} horizon (one request serves both)")
                continue
            key = (i.ref.kind, i.ref.chain or "", i.ref.pool_address or "", i.ref.provider or "",
                   i.ref.pair or "", i.ref.symbol or "", i.spec.candles)  # fmt: skip
            groups.setdefault(key, []).append(i)
        limited: set[str] = set()
        ordered = sorted(
            groups.values(), key=lambda g: (min(i.rank for i in g), min(i.end for i in g))
        )
        for group in ordered:
            ref, tf = group[0].ref, group[0].spec.candles
            provider = self.candles.provider(ref)
            for chunk in _chunks_by_span(group, TIMEFRAME_SECONDS[tf], MAX_LIMIT - 2):
                used = report.requests.get(provider, 0)
                budget = self._budget(provider)
                if provider in limited or used >= budget or self.candles.headroom(ref) <= 0:
                    report.deferred_for_quota += len(chunk)
                    why = (
                        "rate limited this cycle"
                        if provider in limited
                        else "background quota in use"
                    )
                    for i in chunk:
                        i.missing.append(f"{provider} candles deferred: {why}")
                    continue
                start, end = min(i.start for i in chunk), max(i.end for i in chunk)
                report.requests[provider] = used + 1
                try:
                    series = await self.candles.window(ref, tf, start, end, now)
                except ProviderRateLimitedError as exc:
                    limited.add(provider)
                    for i in chunk:
                        i.missing.append(f"{provider} candles rate limited: {exc}")
                    continue
                except AssetNotFoundError as exc:
                    for i in chunk:
                        i.candles_final = True
                        i.missing.append(f"{provider} candles unavailable: {exc}")
                    continue
                except MarketDataError as exc:
                    for i in chunk:
                        i.provider_failed = True
                        i.missing.append(f"{provider} candles failed: {exc}")
                    continue
                for i in chunk:
                    i.series = series

    def _shares_later(self, item: _Item, now: datetime) -> str | None:
        """A longer horizon measured on the same candle timeframe whose window ends in time
        for this horizon's retry window: its request will cover this window too."""
        cc = self.config.collector
        if not cc.coalesce_candles:
            return None
        deadline = item.end + timedelta(minutes=item.spec.retry_minutes)
        for spec in sorted(self.config.horizons, key=lambda h: -h.minutes):  # longest first
            if spec.candles != item.spec.candles or spec.minutes <= item.spec.minutes:
                continue
            ready = item.start + timedelta(minutes=spec.minutes, seconds=cc.settle_seconds)
            if now < ready and ready + timedelta(seconds=cc.retry_seconds) <= deadline:
                return spec.label
        return None

    # --- finalize ---------------------------------------------------------------------------

    async def _finish(self, item: _Item, now: datetime) -> str | None:
        cfg = self.config
        u = HorizonUpdate(missing=list(item.missing))
        price: PricePath | None = None
        inside: list[Candle] = []
        if item.reference_price is None:
            u.missing.append("no reference price at observation: price outcome not measurable")
        elif item.stored_price is not None:
            price = item.stored_price
        elif item.series is not None:
            interval = item.series.interval
            price = path_from_candles(
                item.series.candles, interval, item.reference_price, item.start, item.end,
                provider=item.series.provider, timeframe=item.series.timeframe,
                price_drop_pct=cfg.collapse.price_drop_pct,
            )  # fmt: skip
            inside = in_window(item.series.candles, interval, item.start, item.end)
            if price.points == 0:
                u.missing.append("no trades in the window: price path not measurable")
        if item.token_id is not None and self.scout_store is not None:
            stage = await self.scout_store.growth_near(item.token_id, item.end, item.tolerance)
            if stage is not None:
                u.future_stage_at, u.future_stage, u.future_rank, u.future_score = stage
            elif item.kind == "scout":
                u.missing.append("no Growth Scout run near the horizon end: future stage unknown")
        o = item.due.observation
        # Trigger touches come from the candles themselves (stored with the price path).
        if isinstance(o, DecisionObservation) and inside and price is not None:
            u.triggers = _decision_triggers(o, inside, item.reference_price)
            u.first_trigger_event = first_event(u.triggers)
            u.price_position = _price_position(o, price.end_price)
        exchange = item.ref.kind == "exchange"
        market_seen = item.market is not None
        if not market_seen and not exchange and now > item.end + item.tolerance:
            u.missing.append(
                "horizon-end market state not observed (no Scout snapshot of this pool within "
                f"{item.tolerance.total_seconds() / 60:.0f} min of the horizon end)"
            )
        u.market = item.market
        priced = price is not None and price.points > 0
        candles_done = priced or item.candles_final or (price is not None and price.points == 0)
        # A lookup not listing the pool is a fact about the lookup; the pool is only gone
        # when the candles don't show it trading up to the horizon end.
        step = TIMEFRAME_SECONDS[item.spec.candles]
        slack = timedelta(seconds=max(2 * step, 0.1 * item.spec.minutes * 60))
        trading_to_end = bool(
            priced and price and price.end_price_at and price.end_price_at >= item.end - slack
        )
        unlisted = item.market is not None and not item.market.pool_found
        contradicted = unlisted and trading_to_end
        gone = unlisted and candles_done and not trading_to_end
        listed = item.market is not None and item.market.pool_found
        if contradicted and item.market is not None:
            u.missing.append(
                f"{item.market.provider} no longer lists this pool, but its candles show trades "
                "up to the horizon end: not treated as gone (market state unknown)"
            )
        collapsed = listed and bool(item.market and item.market.liquidity_collapsed)
        market_known = listed or gone
        market_done = market_known or exchange or now > item.end + item.tolerance
        deadline = item.end + timedelta(minutes=item.spec.retry_minutes)
        if priced and (market_known or exchange):
            u.finalize = "COMPLETE"
        elif ((gone or collapsed) and candles_done) or (candles_done and market_done):
            u.finalize = "PARTIAL" if (priced or market_known) else "UNAVAILABLE"
        elif now >= deadline:
            if price is None and item.reference_price is not None and item.snapshots:
                price = path_from_points(
                    [(s.observed_at, s.metrics.price_usd) for s in item.snapshots
                     if s.metrics.price_usd is not None],
                    item.reference_price, item.start, item.end, item.tolerance,
                    provider=item.snapshots[-1].provider,
                    price_drop_pct=cfg.collapse.price_drop_pct,
                )  # fmt: skip
                if price.end_price is None and price.points == 0:
                    price = None
            u.finalize = "PARTIAL" if (price is not None or market_known) else "UNAVAILABLE"
            if not priced:
                u.missing.append(
                    f"candle price path not collected within {item.spec.retry_minutes} min "
                    "of the horizon end"
                )
        # A candle path is final evidence (stored now); a snapshot-only path only on
        # finalization, so later candles can still replace it before then.
        if price is not None and (price.source == "candles" or u.finalize is not None):
            u.price = price
        u.market_status = market_status(
            None if contradicted else item.market, price, item.provider_failed
        )
        await self.store.update_horizon(
            item.due.kind, item.due.horizon.observation_id, item.due.horizon.horizon, u, now
        )
        return u.finalize


def _same(chain: str, a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return (normalize_address(chain, a) or a) == (normalize_address(chain, b) or b)


def _chunks_by_span(items: list[_Item], interval: int, max_candles: int) -> list[list[_Item]]:
    """Consecutive windows sharing one request while their span fits one response."""
    ordered = sorted(items, key=lambda i: i.start)
    chunks: list[list[_Item]] = []
    for i in ordered:
        if chunks:
            first = chunks[-1][0].start
            end = max(max(x.end for x in chunks[-1]), i.end)
            if (end - first).total_seconds() / interval <= max_candles:
                chunks[-1].append(i)
                continue
        chunks.append([i])
    return chunks


def _decision_triggers(
    d: DecisionObservation, candles: list[Candle], reference: float | None
) -> dict[str, TriggerOutcome]:
    out: dict[str, TriggerOutcome] = {}
    if d.buy_trigger is not None:
        out["buy_trigger"] = trigger_outcome(candles, d.buy_trigger.price, "above", reference)
    if d.sell_trigger is not None:
        out["sell_trigger"] = trigger_outcome(candles, d.sell_trigger.price, "below", reference)
    if d.invalidation is not None:
        out["invalidation"] = trigger_outcome(
            candles, d.invalidation.price, d.invalidation.direction, reference
        )
    return out


def _price_position(d: DecisionObservation, price: float | None) -> str | None:
    if price is None:
        return None
    parts = []
    for name, level in (
        ("buy trigger", d.buy_trigger.price if d.buy_trigger else None),
        ("sell trigger", d.sell_trigger.price if d.sell_trigger else None),
        ("invalidation", d.invalidation.price if d.invalidation else None),
    ):
        if level is not None:
            parts.append(
                f"{'above' if price > level else 'below' if price < level else 'at'} {name}"
            )
    return "; ".join(parts) or None
