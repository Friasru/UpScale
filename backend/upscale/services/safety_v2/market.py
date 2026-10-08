"""Safety V2 market evidence (Phase 3): exact pool identity, primary selection and the
as-of market view. Pure: no request, no clock, no database.

Identity. A DEX pool can be this token's market only when it is on Solana, its pair address
is a well-formed Solana address, and its *base* token is byte-for-byte the target mint
(base58 is case-sensitive and never trimmed: a case or whitespace variant is another
mint). Everything else the provider returned is kept for audit with its identity:
``QUOTE_SIDE`` (the mint is the quote token), ``OTHER_MINT`` (e.g. a matching symbol on a
different mint), ``OTHER_CHAIN`` or ``MALFORMED``. Only ``EXACT_BASE`` pools reach
`solana_dex.select_primary_pool`, with the selection thresholds frozen here.

The tracked pool (`MarketView.anchor`) as of a snapshot:

1. the target's earliest pool pin known at ``as_of`` (``PINNED``): never replaced by a
   higher-liquidity pool, which is reported as an alternative instead;
2. else the primary selected from the latest successful observation (``SELECTED``);
3. else the primary of the most recent earlier successful observation that had one
   (``PREVIOUSLY_SELECTED``): the pool that was the market and may have disappeared.

``MARKET_CLOSED_ON_CHAIN`` is applicable (`MarketView.closure_applicable`) only when the
tracked pool was corroborated (reported by a successful observation; a pin alone is not)
and the latest successful observation no longer reports it. Otherwise it is out of scope:
DEX presence is never treated as on-chain proof, and no RPC is spent on a healthy market.

Presence of the tracked pool: ``REPORTED`` (in the latest successful observation, which is
the latest observation), ``NOT_REPORTED`` (a successful observation didn't return it: a
DEX miss, never "closed"), ``CLOSED_ON_CHAIN`` (only: the pool was previously reported by
a successful observation **and** a successful ``getAccountInfo(pool)`` in the same
collection as the latest observation returned no account), ``UNAVAILABLE`` (the latest
market read failed, or no exact pool is known) or ``NOT_COLLECTED``.

All amounts are PROVIDER_REPORTED (DEX Screener's figures), never verified on-chain.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from upscale.services.chains import SOLANA, is_solana_address
from upscale.services.safety_v2.models import PoolIdentity, PoolPresence
from upscale.services.solana_dex import (
    DexPool,
    PoolSelection,
    PoolSelectionConfig,
    TokenRef,
    WindowStats,
    select_primary_pool,
)

# Copied from production's `PoolSelectionConfig` defaults and frozen under RULES_VERSION 3
# (never read from the mutable production object).
SELECTION = PoolSelectionConfig(
    min_liquidity_usd=1_000.0, min_txns_24h=1, competing_liquidity_ratio=0.5,
    max_price_divergence_pct=10.0,
)  # fmt: skip
SUCCESS = frozenset({"POOLS", "NO_POOLS"})

AnchorSource = Literal["PINNED", "SELECTED", "PREVIOUSLY_SELECTED"]


@dataclass(frozen=True)
class MarketPool:
    """One pool as the DEX provider reported it, with its identity relative to the mint."""

    pair_address: str
    identity: PoolIdentity
    chain: str
    dex: str
    base_mint: str
    base_symbol: str | None
    quote_address: str
    quote_symbol: str | None
    quote_kind: str
    liquidity_usd: float | None
    volume_24h: float | None
    txns_24h: int | None
    price_usd: float | None
    pair_created_at: float | None  # provider metadata (epoch seconds), not chain time
    eligible: bool = False
    rejections: tuple[str, ...] = ()


def pool_identity(pool: DexPool, mint: str) -> tuple[PoolIdentity, str | None]:
    """(identity, why it isn't EXACT_BASE)."""
    if pool.chain != SOLANA:
        return "OTHER_CHAIN", f"on {pool.chain}, not solana"
    if pool.pair_address != pool.pair_address.strip() or not is_solana_address(pool.pair_address):
        return "MALFORMED", "the pair address isn't a valid Solana address"
    if pool.base.address == mint:
        return "EXACT_BASE", None
    if pool.quote.address == mint:
        return "QUOTE_SIDE", "the mint is this pool's quote token, not its base"
    return "OTHER_MINT", f"the base token is {pool.base.address!r}, not the exact mint"


def from_dex_pool(pool: DexPool, mint: str) -> MarketPool:
    identity, why = pool_identity(pool, mint)
    day = pool.window("h24")
    created = pool.pair_created_at.timestamp() if pool.pair_created_at else None
    return MarketPool(
        pair_address=pool.pair_address, identity=identity, chain=pool.chain, dex=pool.dex,
        base_mint=pool.base.address, base_symbol=pool.base.symbol,
        quote_address=pool.quote.address, quote_symbol=pool.quote.symbol,
        quote_kind=pool.quote_kind if identity == "EXACT_BASE" else "other",
        liquidity_usd=pool.liquidity_usd, volume_24h=day.volume_usd if day else None,
        txns_24h=day.txns if day else None, price_usd=pool.price_usd,
        pair_created_at=created, rejections=(why,) if why else (),
    )  # fmt: skip


def to_dex_pool(p: MarketPool) -> DexPool:
    """The stored pool as a `DexPool` (what selection reads: price, liquidity, 24h counts
    and volume, quote). ``txns_24h`` is carried as buys so ``.txns`` round-trips."""
    windows = []
    if p.txns_24h is not None or p.volume_24h is not None:
        windows.append(WindowStats(window="h24", buys=p.txns_24h,
                                   sells=0 if p.txns_24h is not None else None,
                                   volume_usd=p.volume_24h))  # fmt: skip
    created = datetime.fromtimestamp(p.pair_created_at, UTC) if p.pair_created_at else None
    return DexPool(
        chain=p.chain, dex=p.dex, pair_address=p.pair_address,
        base=TokenRef(address=p.base_mint, symbol=p.base_symbol),
        quote=TokenRef(address=p.quote_address, symbol=p.quote_symbol),
        price_usd=p.price_usd, liquidity_usd=p.liquidity_usd, pair_created_at=created,
        windows=windows,
    )  # fmt: skip


def select(pools: Sequence[MarketPool], mint: str) -> PoolSelection:
    """`select_primary_pool` over the EXACT_BASE pools only (frozen `SELECTION`)."""
    exact = [to_dex_pool(p) for p in pools if p.identity == "EXACT_BASE"]
    return select_primary_pool(exact, mint, SELECTION, SOLANA)


def with_selection(pools: Sequence[MarketPool], mint: str) -> tuple[MarketPool, ...]:
    """The pools with eligibility and rejection reasons filled from the selection."""
    by_address = {c.pair_address: c for c in select(pools, mint).candidates}
    out = []
    for p in pools:
        c = by_address.get(p.pair_address) if p.identity == "EXACT_BASE" else None
        if c is None:
            out.append(p)
        else:
            out.append(MarketPool(**{**p.__dict__, "eligible": c.eligible,
                                     "rejections": tuple(c.rejected_because)}))  # fmt: skip
    return tuple(out)


# --- the as-of view -------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedMarket:
    """One stored market observation with its pools (a build input)."""

    id: int
    collection_id: int
    fetched_at: float
    outcome: str
    reason: str | None
    pools: tuple[MarketPool, ...]
    provider: str = ""

    @property
    def successful(self) -> bool:
        return self.outcome in SUCCESS

    def exact(self, address: str) -> MarketPool | None:
        return next((p for p in self.pools
                     if p.identity == "EXACT_BASE" and p.pair_address == address), None)  # fmt: skip


@dataclass(frozen=True)
class AccountCheck:
    """One stored getAccountInfo(pool) observation (a build input)."""

    id: int
    collection_id: int
    pool_address: str
    fetched_at: float
    outcome: str
    program_owner: str | None
    reason: str | None


@dataclass(frozen=True)
class MarketView:
    latest: ObservedMarket
    selection: PoolSelection | None  # of the latest observation, when successful
    anchor: str | None
    anchor_source: AnchorSource | None
    anchor_dex: str | None
    pins: tuple[str, ...]  # pinned pools known at as_of, earliest first
    reported: MarketPool | None  # the anchor in the latest observation
    previously_reported: bool  # the anchor was in an earlier successful observation
    presence: PoolPresence
    presence_reason: str
    account: AccountCheck | None  # the check in the latest observation's collection
    misses: tuple[ObservedMarket, ...]  # successful observations missing the anchor
    # MARKET_CLOSED_ON_CHAIN applies only when the tracked pool was corroborated (reported
    # by a successful observation) and the latest successful observation lacks it.
    closure_applicable: bool = False
    closure_reason: str = ""


def market_view(
    mint: str,
    observations: Sequence[ObservedMarket],
    pins: Sequence[tuple[str, str | None, float]],
    accounts: Sequence[AccountCheck],
) -> MarketView:
    """`observations` (all, at or before as_of) oldest first; `pins` (pool, dex,
    pinned_at) earliest first; `accounts` at or before as_of."""
    latest = observations[-1]
    successful = [o for o in observations if o.successful]
    selection = select(latest.pools, mint) if latest.successful else None
    anchor: str | None = None
    source: AnchorSource | None = None
    dex: str | None = None
    since = float("-inf")
    if pins:
        anchor, dex, since = pins[0]
        source = "PINNED"
    elif selection is not None and selection.primary is not None:
        anchor, dex, source = selection.primary.pair_address, selection.primary.dex, "SELECTED"
    else:
        for o in reversed(successful):
            if o is latest:
                continue
            primary = select(o.pools, mint).primary
            if primary is not None:
                anchor, dex = primary.pair_address, primary.dex
                source = "PREVIOUSLY_SELECTED"
                break

    reported = latest.exact(anchor) if anchor is not None and latest.successful else None
    if reported is not None and dex is None:
        dex = reported.dex
    earlier = [o for o in successful if o is not latest]
    previously = anchor is not None and any(o.exact(anchor) for o in earlier)
    account = None
    if anchor is not None:
        same = [a for a in accounts
                if a.pool_address == anchor and a.collection_id == latest.collection_id]  # fmt: skip
        account = max(same, key=lambda a: (a.fetched_at, a.id)) if same else None

    presence: PoolPresence
    if latest.outcome == "NOT_COLLECTED":
        presence, why = "NOT_COLLECTED", f"market observation {latest.id}: {latest.reason}"
    elif not latest.successful:
        presence, why = "UNAVAILABLE", f"market observation {latest.id} failed: {latest.reason}"
    elif anchor is None:
        presence, why = "UNAVAILABLE", "no exact-mint pool is known for this token"
    elif reported is not None:
        presence, why = "REPORTED", "the exact pool is in the latest successful DEX response"
    elif account is not None and account.outcome == "ACCOUNT_MISSING" and previously:
        presence, why = (
            "CLOSED_ON_CHAIN",
            (
                f"the previously reported pool wasn't reported, and getAccountInfo({anchor}) "
                "succeeded with no account"
            ),
        )
    else:
        presence = "NOT_REPORTED"
        why = (
            "the latest successful DEX response didn't return the exact pool (not proof of closure)"
        )
        if account is not None:
            why += f"; on-chain check: {account.outcome}"

    misses: list[ObservedMarket] = []
    if anchor is not None:
        for o in successful:
            if o.fetched_at < since:
                continue  # before the pin: Safety V2 wasn't tracking this pool yet
            if o.exact(anchor) is not None:
                misses = []  # only misses after the last report count
            else:
                misses.append(o)
    last_success = successful[-1] if successful else None
    corroborated = anchor is not None and any(o.exact(anchor) for o in successful)
    closure_applicable = False
    if anchor is None:
        closure_why = "no exact pool is tracked"
    elif last_success is None:
        closure_why = "no successful market observation"
    elif last_success.exact(anchor) is not None:
        closure_why = ("the tracked pool is reported by the latest successful DEX response: no "
                       "evidence warrants an on-chain closure check")  # fmt: skip
    elif not corroborated:
        closure_why = ("the tracked pool was never reported by a successful DEX response, so it "
                       "isn't corroborated as this token's market (a pin alone isn't)")  # fmt: skip
    else:
        closure_applicable = True
        closure_why = "the corroborated pool is missing from the latest successful DEX response"
    return MarketView(latest, selection, anchor, source, dex, tuple(p for p, _, _ in pins),
                      reported, previously, presence, why, account, tuple(misses),
                      closure_applicable, closure_why)  # fmt: skip


def pool_record(p: MarketPool) -> dict[str, Any]:
    """A pool as it appears in a snapshot body (PROVIDER_REPORTED values)."""
    return {
        "pair_address": p.pair_address, "dex": p.dex, "quote_symbol": p.quote_symbol,
        "quote_kind": p.quote_kind, "liquidity_usd": p.liquidity_usd,
        "volume_24h": p.volume_24h, "txns_24h": p.txns_24h, "price_usd": p.price_usd,
        "eligible": p.eligible, "rejections": list(p.rejections),
    }  # fmt: skip
