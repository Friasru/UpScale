"""External data services used by agents. All network I/O lives here, never in agents."""

from upscale.config import (
    COINGECKO_API_KEY,
    EXPLAINER_MODEL,
    HELIUS_API_KEY,
    HELIUS_MAX_HOLDER_PAGES,
    NEWS_DISABLED_FEEDS,
    NEWS_FEEDS,
    NEWS_MODEL,
    SCOUT_CONFIG,
    SCOUT_DB_PATH,
    SOLANA_RPC_URL,
    VISION_MODEL,
)
from upscale.services.asset_profile import set_capability_enabled
from upscale.services.asset_resolver import AssetResolver
from upscale.services.capabilities import ProviderRegistry
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.dexscreener import DexScreenerProvider
from upscale.services.explainer import ClaudeExplainerModel
from upscale.services.geckoterminal import DexCandleService, GeckoTerminalProvider
from upscale.services.kraken import KrakenProvider
from upscale.services.market_data import MarketDataService
from upscale.services.news import NewsService
from upscale.services.news_sentiment_model import ClaudeNewsSentimentModel
from upscale.services.rss_news import RssNewsProvider, configured_feeds
from upscale.services.scout import (
    DexScreenerDiscoveryProvider,
    GeckoTerminalDiscoveryProvider,
    ScoutService,
    ScoutSnapshotStore,
    load_scout_config,
)
from upscale.services.solana_chain import (
    HeliusProvider,
    SolanaChainProvider,
    SolanaRpcProvider,
    SolanaSafetyService,
)
from upscale.services.solana_dex import SolanaDexService
from upscale.services.technical_analysis import TechnicalAnalysisService
from upscale.services.vision import ClaudeVisionModel, VisionService

# Shared across requests so the cache and rate limiter actually apply. CoinGecko serves
# current prices. Candles come from Kraken (native 1m-1d intervals with volume) first;
# CoinGecko is the fallback for the one timeframe it genuinely offers (4h).
_coingecko = CoinGeckoProvider(api_key=COINGECKO_API_KEY)
_kraken = KrakenProvider()
market_data_service = MarketDataService(
    _coingecko,
    candle_providers=[_kraken, _coingecko],
    # Kraken's public API allows roughly one request per second; stay well below it.
    provider_calls_per_minute={_kraken.name: 30},
)
technical_analysis_service = TechnicalAnalysisService(market_data_service)
# Solana DEX pools by exact mint. DEX Screener allows 300 requests per minute; stay well below.
solana_dex_service = SolanaDexService(DexScreenerProvider(), max_calls_per_minute=60)


def _solana_chain_provider() -> SolanaChainProvider | None:
    if HELIUS_API_KEY:
        return HeliusProvider(HELIUS_API_KEY, max_pages=HELIUS_MAX_HOLDER_PAGES)
    if SOLANA_RPC_URL:
        return SolanaRpcProvider(SOLANA_RPC_URL)
    return None


# On-chain token safety needs a configured Solana RPC (see upscale.config); None without one.
_chain_provider = _solana_chain_provider()
solana_safety_service = SolanaSafetyService(_chain_provider) if _chain_provider else None
set_capability_enabled("onchain", solana_safety_service is not None)

# DEX pool candles (GeckoTerminal allows ~30 requests per minute; stay below it).
dex_candle_service = DexCandleService(GeckoTerminalProvider(), max_calls_per_minute=20)
# Agents ask this registry for data capabilities instead of calling providers directly.
provider_registry = ProviderRegistry(
    market_data=market_data_service,
    dex=solana_dex_service,
    dex_candles=dex_candle_service,
    onchain=lambda: solana_safety_service,
)
# Works out which exact asset a request means (DEX search disambiguates unknown tickers).
asset_resolver = AssetResolver(search=solana_dex_service)
vision_service = VisionService(ClaudeVisionModel(model=VISION_MODEL))
news_service = NewsService(
    RssNewsProvider(feeds=configured_feeds(NEWS_FEEDS, NEWS_DISABLED_FEEDS)),
    ClaudeNewsSentimentModel(model=NEWS_MODEL),
)
explainer_model = ClaudeExplainerModel(model=EXPLAINER_MODEL)

# Scout: token discovery (not wired into chat or UI yet). The snapshot store is opened on
# first use, so importing this module never touches the disk.
scout_config = load_scout_config(SCOUT_CONFIG)
scout_service = ScoutService(
    [
        GeckoTerminalDiscoveryProvider(scout_config),
        DexScreenerDiscoveryProvider(scout_config),
    ],
    ScoutSnapshotStore(SCOUT_DB_PATH),
    scout_config,
)
