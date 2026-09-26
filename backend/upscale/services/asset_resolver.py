"""Works out exactly which asset a request means, or asks when it can't tell.

Order of precedence, deterministic:

1. **Addresses.** A Solana mint or EVM contract address is an exact identity. An EVM
   address names its chain in the text ("on base"), or the chain is discovered from DEX
   pools for that exact address; if it trades on several chains, UpScale asks which.
2. **Known assets.** Tickers and names in UpScale's registry (BTC, "bitcoin", BONK,
   "dogwifhat"...) resolve to their reviewed identity.
3. **Other tickers** (`$USELESS`, or an all-caps word like TRUMP that isn't a common
   word): DEX pools are searched for tokens with exactly that ticker. Tokens are ranked by
   real 24h trading volume, not by reported liquidity: a pool's liquidity figure is easy
   to inflate (live, a copycat $USELESS pool reported $247M of liquidity with 20 trades a
   day). One token clearly dominating the ticker's trading, with real liquidity and
   activity, is used; several plausible tokens produce a short clarification listing
   them; none produces a "couldn't find it" note. A ticker is never assumed unique.
4. **Follow-ups.** With no asset in the message, the most recent earlier user message that
   named one is reused (with its timeframe), so "Analyze SOL 5m" then "Should I sell?"
   stays on SOL 5m.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from upscale.routing import detect_assets, detect_solana_mint, detect_timeframe
from upscale.services.asset_profile import AssetIdentity
from upscale.services.asset_registry import DEFAULT_REGISTRY, AssetRegistry
from upscale.services.chains import (
    CHAIN_MENTIONS,
    DEX_CHAINS,
    EVM_CHAINS,
    chain_label,
    normalize_address,
)
from upscale.services.market_data import MarketDataError
from upscale.services.solana_dex import DexPool, WindowStats

Source = Literal["address", "registry", "discovered", "history"]
Confidence = Literal["exact", "registry", "discovered"]

_EVM_ADDRESS_RE = re.compile(r"(?<![0-9A-Za-z])0x[0-9a-fA-F]{40}(?![0-9A-Za-z])")
_CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z][A-Za-z0-9]{1,14})\b")
_CAPS_RE = re.compile(r"(?<![A-Za-z0-9$])([A-Z][A-Z0-9]{1,9})(?![A-Za-z0-9])")
# All-caps words that are common in trading questions but aren't tickers.
# fmt: off
_NOT_TICKERS = frozenset({
    "I", "A", "OK", "BUY", "SELL", "WAIT", "HOLD", "LONG", "SHORT", "NOW", "TODAY", "RSI",
    "MACD", "EMA", "SMA", "ATR", "VWAP", "ATH", "ATL", "USD", "EUR", "TA", "DCA", "FOMO",
    "FUD", "ETF", "SEC", "CPI", "FED", "FOMC", "AM", "PM", "UTC", "US", "UK", "EU", "AI",
    "NFT", "DEX", "CEX", "LP", "TP", "SL", "ROI", "PNL", "IMO", "LOL", "CEO", "GM", "GN",
    "NO", "YES", "NEW", "TOP", "THE", "AND", "OR", "IS", "IT", "IN", "ON", "AT", "TO", "MY",
    "ME", "BE", "DO", "IF", "SO", "UP", "OF", "AN", "AS", "BY", "HODL", "APE", "PUMP", "DUMP",
    "MOON", "WHAT", "WHY", "HOW", "WHEN", "SHOULD", "ANALYZE", "CHART", "COIN", "TOKEN",
    "CRYPTO", "PLS", "PLEASE", "ASAP", "OTC", "KYC", "API", "JSON", "OHLC", "OHLCV", "MA",
})
# fmt: on


class PoolSearch(Protocol):
    async def search_pools(self, query: str) -> list[DexPool]: ...


@dataclass(frozen=True)
class ResolverConfig:
    # A discovered token is used only if its pools hold at least this much liquidity,
    min_established_liquidity_usd: float = 100_000.0
    # trade at least this often (a liquidity figure with no trading isn't a market),
    min_established_txns_24h: int = 100
    # and it trades at least this many times the 24h volume of the next token with the
    # same ticker.
    dominance_ratio: float = 10.0
    max_listed_candidates: int = 3


@dataclass(frozen=True)
class Candidate:
    symbol: str
    name: str | None
    chain: str
    address: str
    liquidity_usd: float
    pools: int
    volume_24h_usd: float = 0.0
    txns_24h: int = 0

    def describe(self) -> str:
        short = f"{self.address[:4]}…{self.address[-4:]}"
        name = f"{self.name} " if self.name else ""
        return (
            f"{name}(${self.symbol}) on {chain_label(self.chain)}, {short}, "
            f"24h volume ${self.volume_24h_usd:,.0f}, liquidity ${self.liquidity_usd:,.0f}"
        )


@dataclass(frozen=True)
class ResolvedAsset:
    label: str  # what agents call it (ticker, or the address until DEX data names it)
    identity: AssetIdentity
    source: Source
    confidence: Confidence
    registered: bool  # in UpScale's reviewed registry
    note: str | None = None

    @property
    def is_contract(self) -> bool:
        """Identified by chain + address and not a registered asset: ticker-keyed data
        (exchange candles, CoinGecko, news) can't be attributed to it."""
        return bool(self.identity.address) and not self.registered


@dataclass
class Resolution:
    assets: list[ResolvedAsset] = field(default_factory=list)
    timeframe: str | None = None  # from the message, or reused with a follow-up asset
    clarification: str | None = None  # ask the user instead of analyzing a guess
    notes: list[str] = field(default_factory=list)

    @property
    def primary(self) -> ResolvedAsset | None:
        return self.assets[0] if self.assets else None


def ticker_candidates(text: str, registry: AssetRegistry = DEFAULT_REGISTRY) -> list[str]:
    """Unknown tickers the text may name: cashtags, and all-caps words that aren't common
    words, timeframes or registered assets."""
    found: list[str] = []
    for match in _CASHTAG_RE.finditer(text):
        found.append(match.group(1).upper())
    for match in _CAPS_RE.finditer(text):
        word = match.group(1)
        if word in _NOT_TICKERS or detect_timeframe(word.lower()) or word.isdigit():
            continue
        found.append(word)
    unique = list(dict.fromkeys(found))
    return [t for t in unique if registry.by_symbol(t) is None]


class AssetResolver:
    def __init__(
        self,
        search: PoolSearch | None,
        registry: AssetRegistry = DEFAULT_REGISTRY,
        config: ResolverConfig | None = None,
    ):
        self.search = search
        self.registry = registry
        self.config = config or ResolverConfig()

    async def resolve(self, text: str, history: Sequence[str] = ()) -> Resolution:
        resolution = await self._resolve_text(text)
        resolution.timeframe = detect_timeframe(text)
        if resolution.assets or resolution.clarification:
            return resolution
        # A follow-up: reuse the last asset (and timeframe) the conversation was about.
        for earlier in reversed(history):
            previous = await self._resolve_text(earlier)
            if previous.clarification:
                break  # the conversation never settled on an asset
            if previous.assets:
                resolution.assets = [
                    ResolvedAsset(
                        label=a.label,
                        identity=a.identity,
                        source="history",
                        confidence=a.confidence,
                        registered=a.registered,
                        note=f"Continuing with {a.label} from earlier in the conversation.",
                    )
                    for a in previous.assets
                ]
                resolution.timeframe = resolution.timeframe or detect_timeframe(earlier)
                break
        return resolution

    async def _resolve_text(self, text: str) -> Resolution:
        out = Resolution()
        lowered = text.lower()

        if mint := detect_solana_mint(text):
            out.assets.append(self._address_asset("solana", mint))
        for match in _EVM_ADDRESS_RE.finditer(text):
            resolved = await self._evm_asset(match.group(0), lowered, out)
            if resolved is not None:
                out.assets.append(resolved)
        if out.clarification:
            return out

        for symbol in detect_assets(text):
            entry = self.registry.by_symbol(symbol)
            if entry is None or any(a.label == symbol for a in out.assets):
                continue
            out.assets.append(
                ResolvedAsset(
                    label=symbol,
                    identity=AssetIdentity(symbol=symbol),
                    source="registry",
                    confidence="registry",
                    registered=True,
                )
            )

        if out.assets:
            return out  # an address or known asset was named; don't guess at extra words
        for ticker in ticker_candidates(text, self.registry):
            resolved = await self._discover(ticker, out)
            if out.clarification:
                return out
            if resolved is not None:
                out.assets.append(resolved)
                break
        return out

    def _address_asset(self, chain: str, address: str) -> ResolvedAsset:
        entry = self.registry.by_address(chain, address)
        if entry is not None:
            return ResolvedAsset(
                label=entry.symbol,
                identity=AssetIdentity(symbol=entry.symbol),
                source="address",
                confidence="exact",
                registered=True,
            )
        normalized = normalize_address(chain, address) or address
        return ResolvedAsset(
            label=normalized,
            identity=AssetIdentity(chain=chain, address=normalized),
            source="address",
            confidence="exact",
            registered=False,
        )

    async def _evm_asset(self, address: str, lowered: str, out: Resolution) -> ResolvedAsset | None:
        registered = [c for c in sorted(EVM_CHAINS) if self.registry.by_address(c, address)]
        if len(registered) == 1:
            return self._address_asset(registered[0], address)  # the registry knows its chain
        mentioned = [c for phrase, c in CHAIN_MENTIONS.items() if phrase in lowered]
        chains = [c for c in mentioned if c in EVM_CHAINS]
        if len(chains) == 1:
            return self._address_asset(chains[0], address)
        found = await self._chains_for_address(address, out)
        if found is None:
            return None
        if len(found) == 1:
            return self._address_asset(found[0], address)
        if not found:
            out.clarification = (
                f"I couldn't find DEX markets for {address} on a supported chain. Which chain "
                "is it on (e.g. Ethereum, Base, BNB Chain)?"
            )
        else:
            names = ", ".join(chain_label(c) for c in found)
            out.clarification = (
                f"{address} trades on several chains ({names}). Which one do you mean?"
            )
        return None

    async def _chains_for_address(self, address: str, out: Resolution) -> list[str] | None:
        if self.search is None:
            out.clarification = f"Which chain is {address} on (e.g. Ethereum, Base, BNB Chain)?"
            return None
        try:
            pools = await self.search.search_pools(address)
        except MarketDataError as exc:
            out.notes.append(f"Token lookup failed: {exc}.")
            out.clarification = f"I couldn't look up {address} right now. Which chain is it on?"
            return None
        target = address.lower()
        chains = {
            p.chain
            for p in pools
            if p.chain in DEX_CHAINS and p.chain in EVM_CHAINS and p.base.address.lower() == target
        }
        return sorted(chains)

    async def _discover(self, ticker: str, out: Resolution) -> ResolvedAsset | None:
        if self.search is None:
            out.notes.append(f"${ticker} isn't a known asset, and token discovery is unavailable.")
            return None
        try:
            pools = await self.search.search_pools(ticker)
        except MarketDataError as exc:
            out.notes.append(f"Couldn't look up ${ticker} right now ({exc}).")
            return None
        candidates = rank_candidates(ticker, pools)
        if not candidates:
            out.notes.append(f"No DEX market was found for a token with the ticker ${ticker}.")
            return None
        cfg = self.config
        top = candidates[0]
        runner_up = candidates[1].volume_24h_usd if len(candidates) > 1 else 0.0
        dominant = (
            top.liquidity_usd >= cfg.min_established_liquidity_usd
            and top.txns_24h >= cfg.min_established_txns_24h
            and top.volume_24h_usd > 0
            and (runner_up == 0 or top.volume_24h_usd >= cfg.dominance_ratio * runner_up)
        )
        if not dominant:
            listed = candidates[: cfg.max_listed_candidates]
            options = "\n".join(f"{i}. {c.describe()}" for i, c in enumerate(listed, 1))
            more = len(candidates) - len(listed)
            out.clarification = (
                f"Several tokens use the ticker ${ticker}:\n{options}"
                + (f"\n(and {more} more)" if more > 0 else "")
                + "\nPaste the contract or mint address of the one you mean."
            )
            return None
        return ResolvedAsset(
            label=top.symbol,
            identity=AssetIdentity(
                symbol=top.symbol, name=top.name, chain=top.chain, address=top.address
            ),
            source="discovered",
            confidence="discovered",
            registered=False,
            note=(
                f"${ticker} matched {top.describe()}: it carries most of this ticker's DEX "
                "trading. Paste an address to analyze a different token."
            ),
        )


def rank_candidates(ticker: str, pools: Sequence[DexPool]) -> list[Candidate]:
    """Tokens whose ticker is exactly `ticker`, by total 24h volume of their pools (then
    liquidity), since reported liquidity alone is easy to fake."""
    groups: dict[tuple[str, str], list[DexPool]] = {}
    for p in pools:
        symbol = (p.base.symbol or "").lstrip("$").upper()
        if symbol != ticker.upper() or p.chain not in DEX_CHAINS:
            continue
        key = (p.chain, normalize_address(p.chain, p.base.address) or p.base.address)
        groups.setdefault(key, []).append(p)
    out = [
        Candidate(
            symbol=ticker.upper(),
            name=items[0].base.name,
            chain=chain,
            address=address,
            liquidity_usd=sum(p.liquidity_usd or 0.0 for p in items),
            pools=len(items),
            volume_24h_usd=sum(_day(p).volume_usd or 0.0 for p in items),
            txns_24h=sum(_day(p).txns or 0 for p in items),
        )
        for (chain, address), items in groups.items()
    ]
    out.sort(key=lambda c: (-c.volume_24h_usd, -c.liquidity_usd, c.chain, c.address))
    return out


def _day(pool: DexPool) -> WindowStats:
    return pool.window("h24") or WindowStats(window="h24")
