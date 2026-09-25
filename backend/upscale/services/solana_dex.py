"""Solana DEX market data by exact mint address: models, pool selection, and a caching service.

Agents depend on `SolanaDexService`; providers (e.g. `DexScreenerProvider`) implement
`DexPoolProvider` and only return pools exactly as the source reported them. Nothing here
is estimated: a field the source didn't supply stays None.

A token is identified only by its mint (`solana:<mint>`), never by its ticker: many tokens
share a ticker, and a pool for another mint is never used.

Primary pool selection (thresholds in `PoolSelectionConfig`), deterministic:

1. A pool is a **candidate** when it is on Solana and the mint is its *base* token (a pool
   quoting the mint prices something else).
2. A candidate is **eligible** to be the primary market when it reports a USD price, USD
   liquidity of at least `min_liquidity_usd`, and at least `min_txns_24h` trades in 24h.
   Tiny, dead or unpriced pools are kept in the findings with the reasons they were
   rejected, but never become the main market.
3. Pools quoted in a recognized asset (SOL, USDC, USDT) are preferred: an unrecognized
   quote token can inflate the USD liquidity a pool appears to have. Other quotes are
   used only when no recognized-quote pool is eligible.
4. Among the preferred group: highest USD liquidity, then highest 24h volume, then pair
   address (so ties are broken the same way every time).
5. The primary market is **unclear** when another eligible pool has at least
   `competing_liquidity_ratio` of the primary's liquidity, or when eligible pools' USD
   prices differ by more than `max_price_divergence_pct`.
"""

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from upscale.services.market_data import (
    AssetNotFoundError,
    InvalidRequestError,
    MarketDataUnavailableError,
    RateLimiter,
)

CHAIN = "solana"
Window = Literal["m5", "h1", "h6", "h24"]
WINDOWS: tuple[Window, ...] = ("m5", "h1", "h6", "h24")
QuoteKind = Literal["SOL", "USDC", "USDT", "other"]

# Mints of the quote assets UpScale recognizes (wrapped SOL, Circle USDC, Tether USDT).
KNOWN_QUOTES: dict[str, QuoteKind] = {
    "So11111111111111111111111111111111111111112": "SOL",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
}
# Solana addresses are base58-encoded 32-byte keys: 32 to 44 characters, no 0/O/I/l.
_MINT_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def is_solana_address(value: str) -> bool:
    return bool(_MINT_RE.fullmatch(value))


# --- Models ---------------------------------------------------------------------------------


class TokenRef(BaseModel):
    address: str
    symbol: str | None = None
    name: str | None = None


class WindowStats(BaseModel):
    """Trading over one rolling window, as reported. Missing values stay None."""

    window: Window
    buys: int | None = None
    sells: int | None = None
    volume_usd: float | None = None
    price_change_pct: float | None = None

    @property
    def txns(self) -> int | None:
        if self.buys is None or self.sells is None:
            return None
        return self.buys + self.sells


class DexPool(BaseModel):
    """One pool (pair) exactly as the provider reported it."""

    chain: str
    dex: str
    pair_address: str
    url: str | None = None
    labels: list[str] = Field(default_factory=list)
    base: TokenRef
    quote: TokenRef
    price_usd: float | None = None
    price_native: float | None = None  # base token priced in the quote token
    liquidity_usd: float | None = None
    liquidity_base: float | None = None
    liquidity_quote: float | None = None
    market_cap_usd: float | None = None
    fdv_usd: float | None = None
    pair_created_at: datetime | None = None
    windows: list[WindowStats] = Field(default_factory=list)  # only windows reported

    def window(self, name: Window) -> WindowStats | None:
        return next((w for w in self.windows if w.window == name), None)

    @property
    def quote_kind(self) -> QuoteKind:
        return KNOWN_QUOTES.get(self.quote.address, "other")


class PoolCandidate(BaseModel):
    pair_address: str
    dex: str
    quote_symbol: str | None
    quote_kind: QuoteKind
    liquidity_usd: float | None
    volume_24h_usd: float | None
    txns_24h: int | None
    price_usd: float | None
    pair_created_at: datetime | None
    eligible: bool
    rejected_because: list[str] = Field(default_factory=list)
    primary: bool = False


class SolanaDexSnapshot(BaseModel):
    """The token's primary Solana DEX market, plus every pool that was considered."""

    canonical_id: str  # "solana:<mint>"
    mint: str
    symbol: str | None
    name: str | None
    provider: str
    dex: str
    pair_address: str
    pair_url: str | None
    quote_symbol: str | None
    quote_address: str
    quote_kind: QuoteKind
    price_usd: float
    price_native: float | None
    liquidity_usd: float
    # Reported by the provider (supply x price); never used as identity or size proof.
    market_cap_usd: float | None
    fdv_usd: float | None
    pair_created_at: datetime | None
    pool_age_hours: float | None  # primary pool, at retrieval time
    # Oldest pool for this mint: a lower bound on the token's age (pools can migrate).
    first_pool_created_at: datetime | None
    windows: list[WindowStats]
    fetched_at: datetime
    candidates: list[PoolCandidate]
    primary_clear: bool
    ambiguity: list[str] = Field(default_factory=list)

    def window(self, name: Window) -> WindowStats | None:
        return next((w for w in self.windows if w.window == name), None)


# --- Pool selection -------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolSelectionConfig:
    """Every threshold used to pick the primary pool. Change these, not the logic."""

    min_liquidity_usd: float = 1_000.0  # below this a pool is too small to be the market
    min_txns_24h: int = 1  # a pool with no trades in 24h isn't an active market
    competing_liquidity_ratio: float = 0.5  # runner-up liquidity / primary liquidity
    max_price_divergence_pct: float = 10.0  # between eligible pools' USD prices


@dataclass(frozen=True)
class PoolSelection:
    primary: DexPool | None
    candidates: list[PoolCandidate]
    clear: bool
    ambiguity: list[str]


def _rejections(pool: DexPool, mint: str, cfg: PoolSelectionConfig) -> list[str]:
    reasons: list[str] = []
    if pool.chain != CHAIN:
        reasons.append(f"on {pool.chain}, not Solana")
    if pool.base.address != mint:
        reasons.append("the mint is not this pool's base token")
    if pool.price_usd is None:
        reasons.append("no USD price reported")
    if pool.liquidity_usd is None:
        reasons.append("no USD liquidity reported")
    elif pool.liquidity_usd < cfg.min_liquidity_usd:
        reasons.append(
            f"liquidity ${pool.liquidity_usd:,.0f} is below ${cfg.min_liquidity_usd:,.0f}"
        )
    day = pool.window("h24")
    txns = day.txns if day else None
    if txns is None:
        reasons.append("no 24h trade counts reported")
    elif txns < cfg.min_txns_24h:
        reasons.append(f"{txns} trades in 24h (inactive)")
    return reasons


def _rank_key(pool: DexPool) -> tuple[float, float, str]:
    day = pool.window("h24")
    volume = day.volume_usd if day and day.volume_usd is not None else 0.0
    return (-(pool.liquidity_usd or 0.0), -volume, pool.pair_address)


def select_primary_pool(
    pools: Sequence[DexPool], mint: str, config: PoolSelectionConfig | None = None
) -> PoolSelection:
    """Pick the token's primary market (see the module docstring for the rules)."""
    cfg = config or PoolSelectionConfig()
    relevant = [p for p in pools if p.chain == CHAIN and p.base.address == mint]
    rejections = {p.pair_address: _rejections(p, mint, cfg) for p in relevant}
    eligible = [p for p in relevant if not rejections[p.pair_address]]
    preferred = [p for p in eligible if p.quote_kind != "other"] or eligible
    ranked = sorted(preferred, key=_rank_key)
    primary = ranked[0] if ranked else None

    ambiguity: list[str] = []
    if primary is not None and primary.liquidity_usd:
        rivals = [p for p in sorted(eligible, key=_rank_key) if p is not primary]
        if rivals and (rivals[0].liquidity_usd or 0) >= (
            cfg.competing_liquidity_ratio * primary.liquidity_usd
        ):
            r = rivals[0]
            ambiguity.append(
                f"{r.dex} pool {r.pair_address} has ${r.liquidity_usd:,.0f} liquidity, "
                f"{100 * (r.liquidity_usd or 0) / primary.liquidity_usd:.0f}% of the primary "
                f"pool's ${primary.liquidity_usd:,.0f}"
            )
        prices = [p.price_usd for p in eligible if p.price_usd]
        if len(prices) >= 2:
            spread = 100 * (max(prices) - min(prices)) / min(prices)
            if spread > cfg.max_price_divergence_pct:
                ambiguity.append(
                    f"eligible pools' USD prices differ by {spread:.1f}% "
                    f"(more than {cfg.max_price_divergence_pct:g}%)"
                )

    candidates = [
        PoolCandidate(
            pair_address=p.pair_address,
            dex=p.dex,
            quote_symbol=p.quote.symbol,
            quote_kind=p.quote_kind,
            liquidity_usd=p.liquidity_usd,
            volume_24h_usd=(w.volume_usd if (w := p.window("h24")) else None),
            txns_24h=(w.txns if (w := p.window("h24")) else None),
            price_usd=p.price_usd,
            pair_created_at=p.pair_created_at,
            eligible=not rejections[p.pair_address],
            rejected_because=rejections[p.pair_address]
            or (
                ["an eligible pool quoted in SOL, USDC or USDT was preferred"]
                if primary is not None and p is not primary and p not in preferred
                else []
            ),
            primary=p is primary,
        )
        for p in sorted(relevant, key=_rank_key)
    ]
    return PoolSelection(
        primary=primary, candidates=candidates, clear=not ambiguity, ambiguity=ambiguity
    )


def build_snapshot(
    mint: str,
    pools: Sequence[DexPool],
    provider: str,
    fetched_at: datetime,
    config: PoolSelectionConfig | None = None,
) -> SolanaDexSnapshot:
    """Normalize the provider's pools into one snapshot, or raise if none is usable."""
    selection = select_primary_pool(pools, mint, config)
    p = selection.primary
    if p is None:
        if not selection.candidates:
            raise AssetNotFoundError(f"no Solana pool with base token {mint} on {provider}")
        raise NoUsablePoolError(mint, selection.candidates)
    assert p.price_usd is not None and p.liquidity_usd is not None  # eligibility guarantees it
    created = [c.pair_created_at for c in selection.candidates if c.pair_created_at]
    age = (
        max(0.0, (fetched_at - p.pair_created_at).total_seconds() / 3600)
        if p.pair_created_at
        else None
    )
    return SolanaDexSnapshot(
        canonical_id=f"{CHAIN}:{mint}",
        mint=mint,
        symbol=p.base.symbol,
        name=p.base.name,
        provider=provider,
        dex=p.dex,
        pair_address=p.pair_address,
        pair_url=p.url,
        quote_symbol=p.quote.symbol,
        quote_address=p.quote.address,
        quote_kind=p.quote_kind,
        price_usd=p.price_usd,
        price_native=p.price_native,
        liquidity_usd=p.liquidity_usd,
        market_cap_usd=p.market_cap_usd,
        fdv_usd=p.fdv_usd,
        pair_created_at=p.pair_created_at,
        pool_age_hours=age,
        first_pool_created_at=min(created) if created else None,
        windows=p.windows,
        fetched_at=fetched_at,
        candidates=selection.candidates,
        primary_clear=selection.clear,
        ambiguity=selection.ambiguity,
    )


class NoUsablePoolError(AssetNotFoundError):
    """Pools exist for the mint, but none is liquid, active and priced enough to use."""

    def __init__(self, mint: str, candidates: list[PoolCandidate]):
        self.mint = mint
        self.candidates = candidates
        super().__init__(
            f"{len(candidates)} Solana pool(s) found for {mint}, but none has enough "
            "liquidity, trading activity and a USD price to be treated as its market"
        )


# --- Provider interface and service ---------------------------------------------------------


class DexPoolProvider(Protocol):
    name: str

    async def fetch_token_pools(self, chain: str, token_address: str) -> list[DexPool]:
        """All pools the source lists for the token (any side), or raise a
        `MarketDataError`. An unknown token returns an empty list."""
        ...


class SolanaDexService:
    """Caches snapshots, rate-limits provider calls, and de-duplicates concurrent requests.

    Failures (outages, rate limits, malformed data) are never cached; a mint with no pools
    is remembered for `not_found_ttl` seconds so repeated lookups don't spend the budget.
    """

    def __init__(
        self,
        provider: DexPoolProvider,
        cache_ttl: float = 30.0,
        not_found_ttl: float = 120.0,
        max_calls_per_minute: int = 60,
        selection: PoolSelectionConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.provider = provider
        self.cache_ttl = cache_ttl
        self.not_found_ttl = not_found_ttl
        self.selection = selection
        self._clock = clock
        self.now = now  # wall clock for pool ages (replaceable in tests)
        self._limiter = RateLimiter(max_calls_per_minute, 60.0, clock)
        self._cache: dict[str, tuple[float, SolanaDexSnapshot]] = {}
        self._not_found: dict[str, tuple[float, AssetNotFoundError]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def reset(self) -> None:
        self._cache.clear()
        self._not_found.clear()
        self._locks.clear()
        self._limiter = RateLimiter(self._limiter.max_calls, 60.0, self._clock)

    async def get_snapshot(self, mint: str) -> SolanaDexSnapshot:
        mint = mint.strip()
        if not is_solana_address(mint):
            raise InvalidRequestError(f"{mint!r} is not a valid Solana mint address")
        if (hit := self._cached(mint)) is not None:
            return hit
        async with self._locks.setdefault(mint, asyncio.Lock()):
            if (hit := self._cached(mint)) is not None:
                return hit
            if not self._limiter.try_acquire():
                raise MarketDataUnavailableError(
                    f"UpScale's {self.provider.name} request limit was reached; try again "
                    "in a minute"
                )
            try:
                pools = await self.provider.fetch_token_pools(CHAIN, mint)
                snapshot = build_snapshot(
                    mint, pools, self.provider.name, self.now(), self.selection
                )
            except AssetNotFoundError as exc:
                self._not_found[mint] = (self._clock() + self.not_found_ttl, exc)
                raise
            self._cache[mint] = (self._clock() + self.cache_ttl, snapshot)
            return snapshot

    def _cached(self, mint: str) -> SolanaDexSnapshot | None:
        missing = self._not_found.get(mint)
        if missing and missing[0] > self._clock():
            raise missing[1]
        entry = self._cache.get(mint)
        return entry[1] if entry and entry[0] > self._clock() else None
