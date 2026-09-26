"""Capability-based data routing: agents ask for a *kind* of data about an asset, and the
registry picks the provider that can serve it for that asset's chain and market.

Capabilities and today's providers (each behind a normalized model, so a provider can be
replaced without touching Technical, Risk or Opportunity):

* ``candles``: exchange candles by ticker (Kraken, then CoinGecko) for listed assets;
  pool candles by exact pool address (GeckoTerminal) for DEX tokens.
* ``market_snapshot``: CoinGecko, by ticker, for listed assets.
* ``dex_market``: DEX Screener, by chain + token address (Solana and EVM chains).
* ``onchain_safety``: a Solana RPC (Helius or any standard RPC), by mint. No EVM provider
  is integrated yet: the capability reports unavailable rather than guessing.
* ``news``: publisher RSS feeds, by ticker/name (registered assets only).
* ``derivatives``, ``social``: no provider integrated yet.

Nothing here decides anything; it only fetches normalized evidence. Timeframes and tokens
are never substituted silently: a fallback is recorded in the evidence it produces.
"""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from upscale.services.chains import DEX_CHAINS, GECKOTERMINAL_NETWORKS, SOLANA
from upscale.services.geckoterminal import DexCandleService
from upscale.services.market_data import (
    CandleSeries,
    MarketDataError,
    MarketDataService,
    MarketSnapshot,
    Timeframe,
)
from upscale.services.solana_chain import KnownPool, OnchainSafetySnapshot, SolanaSafetyService
from upscale.services.solana_dex import DexMarketService, SolanaDexSnapshot

Capability = Literal[
    "candles", "market_snapshot", "dex_market", "onchain_safety", "news", "derivatives", "social"
]
ONCHAIN_CHAINS = frozenset({SOLANA})


class CapabilityUnavailableError(MarketDataError):
    """No integrated provider can serve this capability for this asset."""


@dataclass
class ProviderRegistry:
    market_data: MarketDataService
    dex: DexMarketService
    dex_candles: DexCandleService
    # Looked up on each call: the on-chain provider exists only when configured.
    onchain: Callable[[], SolanaSafetyService | None] = field(default=lambda: None)

    def providers_for(self, capability: Capability, chain: str | None = None) -> list[str]:
        """Which providers would serve a capability (for plans and reporting)."""
        if capability == "candles":
            names = [p.name for p in self.market_data.candle_providers]
            if chain in GECKOTERMINAL_NETWORKS:
                names.append(self.dex_candles.provider.name)
            return names
        if capability == "market_snapshot":
            return [self.market_data.provider_name]
        if capability == "dex_market":
            return [self.dex.provider_name] if chain in DEX_CHAINS else []
        if capability == "onchain_safety":
            service = self.onchain()
            return [service.provider_name] if service and chain in ONCHAIN_CHAINS else []
        return []

    async def exchange_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        return await self.market_data.get_candles(symbol, timeframe, limit=limit, min_candles=1)

    async def pool_candles(
        self,
        chain: str,
        pool: str,
        timeframe: Timeframe,
        limit: int,
        *,
        symbol: str,
        canonical_id: str | None,
    ) -> CandleSeries:
        if chain not in GECKOTERMINAL_NETWORKS:
            raise CapabilityUnavailableError(f"no pool-candle provider covers {chain}")
        return await self.dex_candles.get_candles(
            chain, pool, timeframe, limit, symbol=symbol, canonical_id=canonical_id
        )

    async def market_snapshot(self, symbol: str) -> MarketSnapshot:
        return await self.market_data.get_snapshot(symbol)

    async def dex_market(self, chain: str, address: str) -> SolanaDexSnapshot:
        if chain not in DEX_CHAINS:
            raise CapabilityUnavailableError(f"no DEX market provider covers {chain}")
        return await self.dex.get_snapshot(address, chain)

    async def onchain_safety(
        self, chain: str, address: str, pools: Sequence[KnownPool] = ()
    ) -> OnchainSafetySnapshot:
        service = self.onchain()
        if chain not in ONCHAIN_CHAINS:
            raise CapabilityUnavailableError(
                f"no on-chain safety provider is integrated for {chain}"
            )
        if service is None:
            raise CapabilityUnavailableError("no Solana RPC provider is configured")
        return await service.get_snapshot(address, pools)

    async def derivatives(self, symbol: str) -> None:
        raise CapabilityUnavailableError("no derivatives provider is integrated yet")

    async def social(self, symbol: str) -> None:
        raise CapabilityUnavailableError("no social data provider is integrated yet")

    async def prefetch(self, chain: str | None, address: str | None) -> None:
        """Warm the caches for a token's independent data while the first agents run:
        DEX pools and on-chain facts don't depend on each other, so their network calls
        overlap. Agents then read the cached (or in-flight, de-duplicated) results.
        Failures are ignored here; the agents report them."""
        if not chain or not address:
            return
        tasks: list[Coroutine[Any, Any, object]] = []
        if chain in DEX_CHAINS:
            tasks.append(self.dex.get_snapshot(address, chain))
        service = self.onchain()
        if service is not None and chain in ONCHAIN_CHAINS:
            tasks.append(service.get_chain_data(address))
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)
