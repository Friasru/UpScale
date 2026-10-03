"""Held-position watch: exact-pool prices for the positions open in Shadow runs.

Shadow replays archived evidence and never calls a provider. Its exit prices used to come
only as a by-product of Scout discovery: a held token that dropped out of Scout's listings
(or a Scout scan that was deferred for hours) left the position without a price, and the
book closed it as MARKET_UNAVAILABLE although the market was there. This production-side
task closes that gap:

    open Shadow position (exact chain + token + pool, read-only from the shadow database)
      -> low-priority exact-pool lookup (this module, through the providers' shared gates)
      -> Evidence Archive (kind ``market``, component ``shadow_watch``)
      -> Shadow consumes the archived price later, in time order (anti-lookahead intact)

What it watches: the open positions of active runs (no end time, or one in the future)
whose availability policy reads watch evidence (EVIDENCE_AWARE_V2), and the pending
REALISTIC_V1 entry intents of those runs (an entry fills only at a later observation of
its exact pool; it stops being watched once filled or cancelled). Pending REALISTIC_V1 exit
intents belong to open positions, so they are always covered. LEGACY_V1 runs ignore watch
records, so their positions are not looked up. One lookup per exact (chain, token, pool),
however many strategies, runs, positions or intents hold it. Observations are archived at
the time they are made: nothing is backdated or reconstructed. A pool priced by any source within
`fresh_minutes` is skipped (no duplicate evidence, no wasted request).

How: DEX Screener first (one batched token lookup per chain and 30 tokens; every
watched chain), matched on the exact pool address (EVM compared lowercased, Solana
case-sensitive) and the exact base token. GeckoTerminal (an exact pool lookup) only for
pools GeckoTerminal priced that DEX Screener did not list, and only while its scarce shared
quota has room. A lookup that finds the pool archives its price (``PRICED``, or
``LIQUIDITY_COLLAPSE`` below $1,000 of liquidity: still a real price). One that answers
without it archives ``NOT_FOUND`` when the provider asked is the one that priced the pool
(authoritative: the pool is gone) or ``NOT_LISTED`` otherwise (not proof of anything); a
failure archives ``PROVIDER_FAILED`` / ``RATE_LIMITED``. UpScale's own budget being used
up archives nothing (not evidence about the provider): the pool waits for the next cycle.

Priority (lowest of all production work): a cycle is deferred while Analyze is active, a
Scout scan (manual or background) or safety enrichment is running. Requests run in their
own lane (``shadow_watch``, no reservation): they can never use capacity reserved for
Analyze ("interactive") or Scout refresh, leave `keep_free_calls` of DEX Screener's window
free for everyone else, use GeckoTerminal only when both of discovery's unreserved calls
are free, stay away from a provider that answered HTTP 429 within the cooldown, and stop
at `max_requests_per_cycle`. Every cycle runs at most every `interval_minutes` (default
15): with the default 360-minute staleness rule, a pool would need ~24 consecutive failed
cycles before its price is that old.
"""

import asyncio
import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from upscale.services.chains import normalize_address
from upscale.services.evidence_archive import hooks as evidence
from upscale.services.evidence_archive.payloads import PoolWatchObservation, WatchResult
from upscale.services.evidence_archive.store import WATCH_COMPONENT, EvidenceStore
from upscale.services.market_data import MarketDataError, ProviderRateLimitedError
from upscale.services.quota import request_lane
from upscale.services.scout.gate import RateLimitReachedError, RequestGate
from upscale.services.scout.normalize import canonical_id
from upscale.services.scout.providers import (
    DexScreenerDiscoveryProvider,
    GeckoTerminalDiscoveryProvider,
)
from upscale.services.shadow.config import run_policy
from upscale.services.shadow.store import ShadowStore
from upscale.services.solana_dex import DexPool

logger = logging.getLogger("upscale.shadow.watch")

WATCH_LANE = "shadow_watch"
DEX_SCREENER = "DEX Screener"
GECKOTERMINAL = "GeckoTerminal"
LIQUIDITY_COLLAPSE_USD = 1_000.0
DEFAULT_INTERVAL_MINUTES = 15.0
MIN_INTERVAL_MINUTES = 5.0
_OFF = ("0", "false", "off", "no")

PoolKey = tuple[str, str]  # (token address, pool address), as the position stores them


@dataclass(frozen=True)
class WatchSettings:
    enabled: bool = True
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES
    startup_delay_minutes: float = 6.0  # after background Scout's first run is due
    retry_minutes: float = 2.0  # after a deferral
    fresh_minutes: float = 10.0  # a pool priced this recently (any source) is skipped
    keep_free_calls: int = 10  # of DEX Screener's window, left to every other workload
    max_requests_per_cycle: int = 10
    max_geckoterminal_requests_per_cycle: int = 1
    rate_limit_cooldown_minutes: float = 10.0


def load_watch_settings(enabled: str | None, interval_minutes: str | None) -> WatchSettings:
    """From ``UPSCALE_SHADOW_WATCH`` (default on; 0 / false / off disables it; it only has
    work while EVIDENCE_AWARE_V2 runs hold positions) and
    ``UPSCALE_SHADOW_WATCH_INTERVAL_MINUTES`` (default 15, at least 5)."""
    on = (enabled or "1").strip().lower() not in _OFF
    interval = DEFAULT_INTERVAL_MINUTES
    if interval_minutes and interval_minutes.strip():
        try:
            interval = float(interval_minutes)
        except ValueError:
            interval = math.nan
        if not math.isfinite(interval) or interval <= 0:
            interval = DEFAULT_INTERVAL_MINUTES
        interval = max(interval, MIN_INTERVAL_MINUTES)
    return WatchSettings(enabled=on, interval_minutes=interval)


# --- What is held --------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldPool:
    """One exact (chain, token, pool) held by at least one open position."""

    chain: str
    token: str  # as the position stores it
    pool: str
    asset_id: str
    holders: int  # open positions and pending entry intents sharing it (strategies x runs)
    provider: str | None  # the provider whose observation priced the position's pool
    variants: tuple[tuple[str, str], ...] = ()  # other (asset_id, pool) spellings held
    positions: int = 0  # open positions
    pending_entries: int = 0  # REALISTIC_V1 entry intents waiting for a price
    pending_exits: int = 0  # open positions with an exit intent waiting for a price
    runs: tuple[str, ...] = ()

    @property
    def key(self) -> PoolKey:
        return (self.token, self.pool)


def _norm(chain: str, address: str) -> str:
    return normalize_address(chain, address) or address


def held_pools(shadow_db: str, now: datetime) -> list[HeldPool]:
    """Open positions and pending REALISTIC_V1 entry intents of active EVIDENCE_AWARE_V2
    runs, one entry per exact pool (read-only; a missing shadow database holds nothing).
    Open positions come first, so a pool's archived spelling is a position's when any
    position holds it."""
    if not Path(shadow_db).expanduser().exists():
        return []
    store = ShadowStore(shadow_db, read_only=True)
    try:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

        def add(row: dict[str, Any]) -> None:
            chain, token, pool = row["chain"], row["address"], row["pool"]
            if chain and token and pool:
                groups.setdefault((chain, _norm(chain, token), _norm(chain, pool)), []).append(row)

        for run in store.runs():
            if run_policy(run) != "EVIDENCE_AWARE_V2":
                continue
            if run["until_ts"] is not None and run["until_ts"] <= now.timestamp():
                continue
            for p in store.positions(run_id=run["run_id"], status="OPEN"):
                add(p | {"kind": "position", "run_id": run["run_id"]})
        for run in store.runs():
            if run_policy(run) != "EVIDENCE_AWARE_V2":
                continue
            if run["until_ts"] is not None and run["until_ts"] <= now.timestamp():
                continue
            for book in store.checkpoint(run["run_id"])["books"].values():
                for e in ((book.get("state") or {}).get("pending_entries") or {}).values():
                    add({"chain": e["chain"], "address": e["address"], "pool": e["pool"],
                         "asset_id": e["asset_id"], "kind": "pending_entry",
                         "pending_exit": None, "run_id": run["run_id"]})  # fmt: skip
    finally:
        store.close()
    out = []
    for (chain, _, _), rows in sorted(groups.items()):
        first = rows[0]
        spellings = sorted({(r["asset_id"], r["pool"]) for r in rows})
        positions = [r for r in rows if r["kind"] == "position"]
        out.append(HeldPool(
            chain=chain, token=first["address"], pool=first["pool"], asset_id=first["asset_id"],
            holders=len(rows), provider=None,
            variants=tuple(s for s in spellings if s != (first["asset_id"], first["pool"])),
            positions=len(positions), pending_entries=len(rows) - len(positions),
            pending_exits=sum(1 for r in positions if r.get("pending_exit")),
            runs=tuple(sorted({r["run_id"] for r in rows})),
        ))  # fmt: skip
    return out


def targets_summary(held: Sequence[HeldPool]) -> dict[str, Any]:
    """What the watch looks after, for status: pools by kind, pending-entry pools listed."""
    return {
        "pools": len(held),
        "open_position_pools": sum(1 for h in held if h.positions),
        "pending_entry_pools": [
            {"chain": h.chain, "token": h.token, "pool": h.pool, "pending_entries":
             h.pending_entries, "open_positions": h.positions, "runs": list(h.runs)}
            for h in held if h.pending_entries
        ],
        "pending_exit_intents": sum(h.pending_exits for h in held),
        "note": "pending exit intents belong to open positions and are watched with them",
    }  # fmt: skip


# --- Exact-pool lookups ----------------------------------------------------------------------


class PoolLookup(Protocol):
    name: str
    chains: frozenset[str]

    def headroom(self, chain: str) -> int:
        """Requests the watch may start for this chain right now."""
        ...

    def requests(self, count: int) -> int: ...

    def rate_limited(self, seconds: float) -> bool: ...

    async def pools(self, chain: str, keys: Sequence[PoolKey]) -> dict[PoolKey, DexPool | None]:
        """Each exact pool of each exact token, or None when the provider answered without
        it. Raises MarketDataError."""
        ...


class _GateLookup:
    keep_free = 0

    def __init__(self, gate: RequestGate, chains: frozenset[str], batch: int):
        self.gate = gate
        self.chains = chains
        self.batch = batch

    def headroom(self, chain: str) -> int:
        if chain not in self.chains:
            return 0
        return self.gate.available(WATCH_LANE) - self.keep_free

    def requests(self, count: int) -> int:
        return math.ceil(count / self.batch)

    def rate_limited(self, seconds: float) -> bool:
        return self.gate.rate_limited_within(seconds)


class DexScreenerLookup(_GateLookup):
    name = DEX_SCREENER

    def __init__(self, provider: DexScreenerDiscoveryProvider, keep_free: int):
        super().__init__(provider.gate, provider.chains, provider.config.lookup_batch_size)
        self.provider = provider
        self.keep_free = keep_free

    async def pools(self, chain: str, keys: Sequence[PoolKey]) -> dict[PoolKey, DexPool | None]:
        with request_lane(WATCH_LANE):
            found = await self.provider.token_pools(chain, sorted({t for t, _ in keys}))
        return {
            (token, pool): next(
                (
                    p
                    for p in found.get(canonical_id(chain, token), [])
                    if _norm(chain, p.pair_address) == _norm(chain, pool)
                ),
                None,
            )  # fmt: skip
            for token, pool in keys
        }


class GeckoTerminalLookup(_GateLookup):
    """On the process-wide GeckoTerminal quota: only when both of discovery's unreserved
    calls are free (it then takes one, leaving the other)."""

    name = GECKOTERMINAL

    def __init__(self, provider: GeckoTerminalDiscoveryProvider):
        super().__init__(provider.gate, provider.chains, provider.config.lookup_batch_size)
        self.provider = provider
        self.keep_free = 1

    async def pools(self, chain: str, keys: Sequence[PoolKey]) -> dict[PoolKey, DexPool | None]:
        with request_lane(WATCH_LANE):
            found = await self.provider.pools_by_address(chain, sorted({p for _, p in keys}))
        return {
            (token, pool): next(
                (
                    p
                    for p in found
                    if _norm(chain, p.pair_address) == _norm(chain, pool)
                    and _norm(chain, p.base.address) == _norm(chain, token)
                ),
                None,
            )  # fmt: skip
            for token, pool in keys
        }


# --- The watch -------------------------------------------------------------------------------


def _by_chain(pools: Sequence[HeldPool]) -> list[tuple[str, list[HeldPool]]]:
    out: dict[str, list[HeldPool]] = {}
    for h in pools:
        out.setdefault(h.chain, []).append(h)
    return sorted(out.items())


@dataclass
class CycleResult:
    at: datetime
    status: str  # completed / deferred / idle / failed
    reason: str | None = None
    held: int = 0
    fresh: int = 0  # skipped: priced recently by another source
    looked_up: int = 0
    results: dict[str, int] = field(default_factory=dict)
    requests: dict[str, int] = field(default_factory=dict)
    waiting: int = 0  # due, but no provider capacity this cycle
    duration_seconds: float = 0.0


class HeldPositionWatch:
    def __init__(
        self,
        settings: WatchSettings,
        lookups: Sequence[PoolLookup],
        shadow_db: Callable[[], str],
        evidence_store: Callable[[], EvidenceStore | None],
        defer: Callable[[], str | None],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.settings = settings
        self.lookups = {x.name: x for x in lookups}
        self._shadow_db = shadow_db
        self._evidence = evidence_store
        self._defer = defer
        self.now = now
        self.running = False
        self.last_cycle: CycleResult | None = None
        self._current: CycleResult | None = None
        self.last_completed_at: datetime | None = None
        self.last_targets: dict[str, Any] | None = None
        self.next_run: datetime | None = None
        self.counts: Counter[str] = Counter()
        self.deferrals: Counter[str] = Counter()
        self.results: Counter[str] = Counter()
        self.requests: Counter[str] = Counter()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # --- one cycle ---------------------------------------------------------------------------

    def _provider_of(self, store: EvidenceStore | None, h: HeldPool, now: datetime) -> str | None:
        """Which provider priced the position's pool (the latest archived observation of
        the exact pool): only that provider's "not listed" means the pool is gone."""
        if store is None:
            return None
        for kind in ("market", "scout"):
            r = store.latest(kind, h.asset_id, until=now, pool=h.pool)
            if r is not None and r.component != WATCH_COMPONENT and r.provider:
                return r.provider
        return None

    def _fresh(self, store: EvidenceStore | None, h: HeldPool, now: datetime) -> bool:
        if store is None:
            return False
        since = now - timedelta(minutes=self.settings.fresh_minutes)
        for kind in ("market", "scout"):
            r = store.latest(kind, h.asset_id, until=now, since=since, pool=h.pool)
            if r is not None and r.availability == "AVAILABLE":
                c = r.payload.get("candidate") or {}
                if kind == "scout" and c.get("data_status") != "CURRENT":
                    continue  # carried from an earlier observation: not a fresh price
                price = (c.get("metrics") or c.get("market") or {}).get("price_usd")
                if isinstance(price, int | float) and price > 0:
                    return True
        return False

    def _emit(self, h: HeldPool, provider: str, status: WatchResult, authoritative: bool,
              pool: DexPool | None, error: str | None, at: datetime) -> None:  # fmt: skip
        self.results[status] += 1
        if self._current is not None:
            self._current.results[status] = self._current.results.get(status, 0) + 1
        with evidence.component(WATCH_COMPONENT):
            for asset_id, raw_pool in ((h.asset_id, h.pool), *h.variants):
                token = asset_id.split(":", 1)[1] if ":" in asset_id else h.token
                evidence.emit("pool_watch", PoolWatchObservation(
                    chain=h.chain, token=token, pool_address=raw_pool, provider=provider,
                    observed_at=at, status=status, authoritative=authoritative, pool=pool,
                    error=error, holders=h.holders,
                ))  # fmt: skip

    async def run_once(self) -> CycleResult:
        """One cycle (never raises): deferred if anything outranks it, else every held pool
        not priced recently is looked up, within each provider's headroom."""
        started = self.now()
        if self.running:
            return self._finish(CycleResult(started, "deferred", "a watch cycle is running"))
        try:
            reason = self._defer()
        except Exception:
            logger.exception("held-position watch could not check priorities")
            reason = "priority check failed"
        if reason is not None:
            self.deferrals[reason] += 1
            return self._finish(CycleResult(started, "deferred", reason))
        self.running = True
        try:
            return self._finish(await self._cycle(started))
        except Exception as exc:  # background work never takes the app down
            logger.exception("held-position watch cycle failed")
            return self._finish(CycleResult(started, "failed", type(exc).__name__))
        finally:
            self.running = False

    async def _cycle(self, now: datetime) -> CycleResult:
        s = self.settings
        result = self._current = CycleResult(now, "completed")
        held = await asyncio.to_thread(held_pools, self._shadow_db(), now)
        result.held = len(held)
        self.last_targets = {"at": now} | targets_summary(held)
        if not held:
            result.status = "idle"
            return result
        store = self._evidence()
        due: list[HeldPool] = []
        for h in held:
            if self._fresh(store, h, now):
                result.fresh += 1
            else:
                due.append(replace(h, provider=self._provider_of(store, h, now)))
        budget = s.max_requests_per_cycle
        cooldown = s.rate_limit_cooldown_minutes * 60
        # 1. DEX Screener for every due pool of a chain it serves (one request per 30 tokens).
        ask_gt: list[HeldPool] = []
        ds_answered: set[HeldPool] = set()
        ds = self.lookups.get(DEX_SCREENER)
        for chain, group in _by_chain(due):
            if ds is None or chain not in ds.chains:
                ask_gt += group
                continue
            got = await self._lookup(ds, chain, group, budget, cooldown, result)
            if got is None:
                result.waiting += len(group)
                continue
            budget -= got[0]
            for h, pool in got[1].items():
                ds_answered.add(h)
                if pool is not None:
                    continue
                if h.provider == GECKOTERMINAL:
                    ask_gt.append(h)  # not proof: GeckoTerminal priced it, ask it next
                else:
                    gone = h.provider == DEX_SCREENER
                    self._emit(h, DEX_SCREENER, "NOT_FOUND" if gone else "NOT_LISTED", gone,
                               None, None, self.now())  # fmt: skip
        # 2. GeckoTerminal, only for what DEX Screener can't answer for (bounded per cycle).
        gt = self.lookups.get(GECKOTERMINAL)
        gt_budget = min(budget, s.max_geckoterminal_requests_per_cycle)
        for chain, group in _by_chain(ask_gt):
            got = None
            if gt is not None and chain in gt.chains:
                got = await self._lookup(gt, chain, group, gt_budget, cooldown, result)
            if got is None:
                result.waiting += len(group)
                for h in group:
                    if h in ds_answered:  # what DEX Screener did say (not authoritative)
                        self._emit(h, DEX_SCREENER, "NOT_LISTED", False, None, None, self.now())
                continue
            gt_budget -= got[0]
            for h, pool in got[1].items():
                if pool is None:
                    gone = h.provider == GECKOTERMINAL
                    self._emit(h, GECKOTERMINAL, "NOT_FOUND" if gone else "NOT_LISTED", gone,
                               None, None, self.now())  # fmt: skip
        result.duration_seconds = (self.now() - now).total_seconds()
        return result

    async def _lookup(
        self,
        source: PoolLookup,
        chain: str,
        group: list[HeldPool],
        budget: int,
        cooldown: float,
        result: CycleResult,
    ) -> tuple[int, dict[HeldPool, DexPool | None]] | None:
        """Look `group` up in `source` when its headroom, the cycle budget and the 429
        cooldown allow; prices are archived here. None: not attempted, or refused by
        UpScale's own budget (no evidence either way)."""
        n = source.requests(len(group))
        if n > budget or source.headroom(chain) < n or source.rate_limited(cooldown):
            return None
        at = self.now()
        try:
            found = await source.pools(chain, [h.key for h in group])
        except RateLimitReachedError:
            return None  # UpScale's own budget: not evidence about the provider
        except MarketDataError as exc:
            # A provider HTTP 429 is recorded on its gate as it happens.
            limited = isinstance(exc, ProviderRateLimitedError) or source.rate_limited(5.0)
            status: WatchResult = "RATE_LIMITED" if limited else "PROVIDER_FAILED"
            self._count_requests(source.name, n, result)
            for h in group:
                self._emit(h, source.name, status, False, None, str(exc)[:300], at)
            return (n, {})
        self._count_requests(source.name, n, result)
        result.looked_up += len(group)
        out: dict[HeldPool, DexPool | None] = {}
        for h in group:
            pool = found.get(h.key)
            if pool is None or pool.price_usd is None or pool.price_usd <= 0:
                out[h] = None
                continue
            low = pool.liquidity_usd is not None and pool.liquidity_usd < LIQUIDITY_COLLAPSE_USD
            self._emit(h, source.name, "LIQUIDITY_COLLAPSE" if low else "PRICED",
                       h.provider == source.name, pool, None, at)  # fmt: skip
            out[h] = pool
        return (n, out)

    def _count_requests(self, name: str, n: int, result: CycleResult) -> None:
        result.requests[name] = result.requests.get(name, 0) + n
        self.requests[name] += n

    def _finish(self, r: CycleResult) -> CycleResult:
        self.last_cycle = r
        self.counts[r.status] += 1
        if r.status in ("completed", "idle"):
            self.last_completed_at = r.at
        return r

    # --- schedule ------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        s = self.settings
        c = self.last_cycle
        return {
            "enabled": s.enabled,
            "interval_minutes": s.interval_minutes,
            "running": self.running,
            "next_run": self.next_run if self.active else None,
            "last_completed_at": self.last_completed_at,
            "cycles": dict(self.counts),
            "deferrals_by_reason": dict(self.deferrals),
            "archived_by_result": dict(self.results),
            "requests_by_provider": dict(self.requests),
            "last_cycle": c.__dict__ if c is not None else None,
            "targets": self.last_targets,
            "lane": WATCH_LANE,
            "note": "lowest priority: never runs during Analyze, a Scout scan or safety "
            "enrichment; never uses reserved capacity",
        }

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def run_forever(self) -> None:
        s = self.settings
        self.next_run = self.now() + timedelta(minutes=s.startup_delay_minutes)
        while not self._stop.is_set():
            delay = max(0.0, (self.next_run - self.now()).total_seconds())
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except TimeoutError:
                pass
            r = await self.run_once()
            minutes = s.retry_minutes if r.status == "deferred" else s.interval_minutes
            self.next_run = self.now() + timedelta(minutes=minutes)

    def start(self) -> None:
        if not self.settings.enabled or self.active:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), 15.0)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.next_run = None
