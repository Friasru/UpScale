"""Reviewed facts about the crypto assets UpScale knows by name.

These are *static classification metadata* (market-cap tier, launch date, tags, where it
trades), not behavior: `upscale.services.asset_profile` turns them into a category with
the same rules it applies to any other asset, so an asset is never special-cased by its
ticker.

They are **not current market data**. Seeds are approximate and reviewed by hand (see
`REGISTRY_AS_OF`); they are used only against coarse bands (market-cap tiers, ages of
months or years), never as prices, market caps or liquidity. Live values (market cap,
liquidity, candle history) only ever come from providers, and the registry refuses
entries that carry them.
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from upscale.services.kraken import resolve_pair

REGISTRY_AS_OF = "2026-09"


class AssetMetadata(BaseModel):
    """What is known about one asset before any live data is fetched.

    Every field except `symbol` may be None: unknown stays unknown, it is never guessed.
    Registry entries fill in only static classification metadata; providers (e.g. a DEX
    pool lookup) fill in the live facts for the exact token they looked up.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str | None = None
    # Network the asset lives on ("bitcoin", "ethereum", "solana", ...) and, for tokens,
    # its contract / mint address there. Native coins have no address.
    chain: str | None = None
    address: str | None = None
    coingecko_id: str | None = None
    kraken_pair: str | None = None  # Kraken's request pair, e.g. "XBTUSD", when listed
    tags: frozenset[str] = Field(default_factory=frozenset)  # e.g. {"meme"}, {"stablecoin"}
    peg: str | None = None  # stablecoins: the currency it tracks, e.g. "USD"
    # Static classification metadata: a coarse, hand-reviewed size tier (see
    # asset_profile.MarketCapClass). Never read as the asset's current market cap.
    classification_cap_tier: str | None = None
    launched: date | None = None  # approximate genesis / token launch date
    cex_listed: bool | None = None  # traded on at least one centralized exchange
    dex_listed: bool | None = None  # traded in at least one DEX pool
    # Provider-supplied facts about the exact token (never set by the registry).
    market_cap_usd: float | None = None
    liquidity_usd: float | None = None  # DEX tokens: main pool liquidity
    pool_created_at: datetime | None = None  # DEX tokens: when the main pool was created
    candle_history: int | None = None  # candles available on the analysis timeframe
    source: str = "caller"  # where these facts came from


# Fields that describe the market right now; only providers may set them.
LIVE_FIELDS = ("market_cap_usd", "liquidity_usd", "candle_history")


def _entry(
    symbol: str,
    name: str,
    chain: str,
    coingecko_id: str,
    cap: str,
    launched: date,
    *,
    tags: frozenset[str] = frozenset(),
    kraken: bool = True,
    address: str | None = None,
    peg: str | None = None,
    dex: bool | None = None,
) -> AssetMetadata:
    return AssetMetadata(
        symbol=symbol,
        name=name,
        chain=chain,
        address=address,
        coingecko_id=coingecko_id,
        kraken_pair=resolve_pair(symbol).request_pair if kraken else None,
        tags=tags,
        peg=peg,
        classification_cap_tier=cap,
        launched=launched,
        cex_listed=True,
        dex_listed=dex,
        source=f"UpScale asset registry, reviewed {REGISTRY_AS_OF}",
    )


MEME = frozenset({"meme"})
STABLE = frozenset({"stablecoin"})

# fmt: off
# `kraken=False` where a Kraken USD pair wasn't confirmed at review time: the profile then
# reports Kraken candles as unknown rather than claiming them.
KNOWN_ASSETS: tuple[AssetMetadata, ...] = (
    _entry("BTC", "Bitcoin", "bitcoin", "bitcoin", "mega", date(2009, 1, 3)),
    _entry("ETH", "Ethereum", "ethereum", "ethereum", "mega", date(2015, 7, 30)),
    _entry("SOL", "Solana", "solana", "solana", "large", date(2020, 3, 16)),
    _entry("XRP", "XRP", "xrp-ledger", "ripple", "large", date(2012, 6, 2)),
    _entry("BNB", "BNB", "bnb-chain", "binancecoin", "large", date(2017, 7, 25), kraken=False),
    _entry("ADA", "Cardano", "cardano", "cardano", "large", date(2017, 9, 29)),
    _entry("TRX", "TRON", "tron", "tron", "large", date(2017, 9, 1)),
    _entry("AVAX", "Avalanche", "avalanche", "avalanche-2", "mid", date(2020, 9, 21)),
    _entry("LTC", "Litecoin", "litecoin", "litecoin", "mid", date(2011, 10, 7)),
    _entry("LINK", "Chainlink", "ethereum", "chainlink", "mid", date(2017, 9, 19),
           address="0x514910771af9ca656af840dff83e8264ecf986ca", dex=True),
    _entry("DOT", "Polkadot", "polkadot", "polkadot", "mid", date(2020, 5, 26)),
    _entry("TON", "Toncoin", "ton", "the-open-network", "mid", date(2021, 1, 1), kraken=False),
    _entry("OP", "Optimism", "optimism", "optimism", "mid", date(2022, 5, 31)),
    _entry("ARB", "Arbitrum", "arbitrum", "arbitrum", "mid", date(2023, 3, 23)),
    _entry("SUI", "Sui", "sui", "sui", "mid", date(2023, 5, 3)),
    _entry("NEAR", "NEAR Protocol", "near", "near", "mid", date(2020, 10, 13)),
    _entry("APT", "Aptos", "aptos", "aptos", "mid", date(2022, 10, 17)),
    _entry("DOGE", "Dogecoin", "dogecoin", "dogecoin", "large", date(2013, 12, 6), tags=MEME),
    _entry("SHIB", "Shiba Inu", "ethereum", "shiba-inu", "mid", date(2020, 8, 1), tags=MEME,
           address="0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce", dex=True),
    _entry("PEPE", "Pepe", "ethereum", "pepe", "mid", date(2023, 4, 14), tags=MEME,
           address="0x6982508145454ce325ddbe47a25d4ec3d2311933", dex=True),
    _entry("USDT", "Tether", "ethereum", "tether", "mega", date(2014, 10, 6), tags=STABLE,
           address="0xdac17f958d2ee523a2206206994597c13d831ec7", peg="USD", dex=True),
    _entry("USDC", "USD Coin", "ethereum", "usd-coin", "large", date(2018, 9, 26), tags=STABLE,
           address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", peg="USD", dex=True),
    _entry("DAI", "Dai", "ethereum", "dai", "mid", date(2019, 11, 18), tags=STABLE, peg="USD",
           dex=True),
)
# fmt: on


class AssetRegistry:
    """Lookup of reviewed assets by ticker, Kraken pair, or chain + address.

    A ticker resolves only when exactly one entry has it; tokens found by chain + address
    never match an entry by ticker alone, so two tokens sharing a ticker stay separate.
    """

    def __init__(self, entries: tuple[AssetMetadata, ...] = KNOWN_ASSETS):
        for e in entries:
            live = [f for f in LIVE_FIELDS if getattr(e, f) is not None]
            if live:
                raise ValueError(
                    f"registry entry {e.symbol} sets live market data ({', '.join(live)}); "
                    "the registry holds static classification metadata only"
                )
        self.entries = entries
        by_symbol: dict[str, list[AssetMetadata]] = {}
        for e in entries:
            by_symbol.setdefault(e.symbol.upper(), []).append(e)
        self._by_symbol = by_symbol
        self._by_pair = {e.kraken_pair: e for e in entries if e.kraken_pair}
        self._by_address = {
            (e.chain, normalize_address(e.chain, e.address)): e
            for e in entries
            if e.chain and e.address
        }

    def by_symbol(self, symbol: str) -> AssetMetadata | None:
        matches = self._by_symbol.get(symbol.upper(), [])
        return matches[0] if len(matches) == 1 else None

    def by_kraken_pair(self, pair: str) -> AssetMetadata | None:
        return self._by_pair.get(pair.upper())

    def by_address(self, chain: str, address: str) -> AssetMetadata | None:
        return self._by_address.get((chain, normalize_address(chain, address)))

    def has_symbol(self, symbol: str) -> bool:
        return symbol.upper() in self._by_symbol


# EVM addresses are case-insensitive hex; Solana mints (base58) and others are not.
_EVM_CHAINS = {"ethereum", "bnb-chain", "arbitrum", "optimism", "base", "polygon", "avalanche"}


def normalize_address(chain: str | None, address: str | None) -> str | None:
    if address is None:
        return None
    address = address.strip()
    return address.lower() if chain in _EVM_CHAINS else address


DEFAULT_REGISTRY = AssetRegistry()
