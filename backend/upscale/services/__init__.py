"""External data services used by agents. All network I/O lives here, never in agents."""

from upscale.config import (
    COINGECKO_API_KEY,
    EXPLAINER_MODEL,
    GROWTH_CONFIG,
    HELIUS_API_KEY,
    HELIUS_MAX_HOLDER_PAGES,
    NEWS_DISABLED_FEEDS,
    NEWS_FEEDS,
    NEWS_MODEL,
    NEYNAR_API_KEY,
    OUTCOME_CONFIG,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    SCOUT_CONFIG,
    SCOUT_DB_PATH,
    SOCIAL_CONFIG,
    SOCIAL_FORUMS,
    SOLANA_RPC_URL,
    VISION_MODEL,
    X_BEARER_TOKEN,
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
from upscale.services.outcomes import (
    OUTCOME_LANE,
    DexScreenerPools,
    GeckoTerminalPools,
    OutcomeCollector,
    OutcomeStore,
    ProviderCandles,
    load_outcome_config,
)
from upscale.services.quota import LaneLimiter
from upscale.services.rss_news import RssNewsProvider, configured_feeds
from upscale.services.scout import (
    DexScreenerDiscoveryProvider,
    GeckoTerminalDiscoveryProvider,
    ScoutService,
    ScoutSnapshotStore,
    load_scout_config,
)
from upscale.services.scout.gate import RequestGate
from upscale.services.scout.growth import GrowthScoutService, load_growth_config
from upscale.services.scout.service import REFRESH_LANE
from upscale.services.scout.social import (
    DiscourseForumProvider,
    NeynarFarcasterProvider,
    RedditProvider,
    SocialScoutService,
    SocialStore,
    XRecentSearchProvider,
    load_social_config,
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
# Scout's settings are loaded first: they own the GeckoTerminal quota (below).
scout_config = load_scout_config(SCOUT_CONFIG)
# ONE GeckoTerminal quota for the whole process: its real limit is shared by Scout's
# discovery and refresh and Analyze's pool candles, so UpScale counts them together. Part
# is held for interactive work (Analyze) and never lent to background Scout traffic.
# Priority: Analyze > due outcome collection > Scout refresh > Scout discovery, so outcome
# work may also use Scout refresh's reservation (held from startup until a scan releases
# it); it never touches the interactive one.
geckoterminal_quota = LaneLimiter(
    scout_config.geckoterminal.calls_per_minute,
    60.0,
    scout_config.geckoterminal.reservations,
    outranks={OUTCOME_LANE: {REFRESH_LANE}},
)
dex_candle_service = DexCandleService(GeckoTerminalProvider(), limiter=geckoterminal_quota)
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

# Scout: token discovery. The snapshot store is opened on first use, so importing this
# module never touches the disk.
_dexscreener_discovery = DexScreenerDiscoveryProvider(scout_config)
_geckoterminal_discovery = GeckoTerminalDiscoveryProvider(
    scout_config,
    gate=RequestGate("GeckoTerminal", scout_config.geckoterminal, limiter=geckoterminal_quota),
)
scout_service = ScoutService(
    [_geckoterminal_discovery, _dexscreener_discovery],
    ScoutSnapshotStore(SCOUT_DB_PATH),
    scout_config,
)

# Scout social / attention evidence (not wired into chat or UI yet; never a trade signal).
# Providers without credentials report "not configured"; the store opens on first use.
social_config = load_social_config(SOCIAL_CONFIG)
social_scout_service = SocialScoutService(
    [
        RedditProvider(
            REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, social_config.reddit
        ),
        NeynarFarcasterProvider(NEYNAR_API_KEY, social_config.farcaster),
        XRecentSearchProvider(
            X_BEARER_TOKEN,
            social_config.x_allow_paid,
            social_config.x_max_reads_per_run,
            social_config.x,
            fetch_usernames=social_config.x_fetch_usernames,
        ),
        *(
            DiscourseForumProvider(url.strip(), social_config.discourse)
            for url in (SOCIAL_FORUMS or "").split(",")
            if url.strip()
        ),
    ],
    SocialStore(SCOUT_DB_PATH),
    social_config,
)

# Growth Scout: ranks Scout's candidates (not wired into chat or UI yet; never BUY / SELL).
# Reuses Scout's snapshots, stored social momentum and the cached on-chain safety service.
growth_scout_service = GrowthScoutService(
    scout_service.store,
    load_growth_config(GROWTH_CONFIG),
    social_store=social_scout_service.store,
    safety=solana_safety_service,
)

# Outcome tracking: immutable Scout / decision observations and what happened afterward
# (same database file, separate tables; opened on first use). The collector is background
# work in its own request lane: it never uses capacity reserved for Analyze, outranks Scout
# (refresh and discovery) on the shared GeckoTerminal quota, and is started by the API app
# (upscale.main).
outcome_config = load_outcome_config(OUTCOME_CONFIG)
outcome_store = OutcomeStore(SCOUT_DB_PATH)
outcome_collector = OutcomeCollector(
    outcome_store,
    outcome_config,
    scout_store=scout_service.store,
    candles=ProviderCandles(
        dex_candle_service, market_data_service, outcome_config.collector.min_free_calls
    ),
    pools=[
        DexScreenerPools(_dexscreener_discovery, outcome_config.collector.min_free_calls),
        GeckoTerminalPools(_geckoterminal_discovery, outcome_config.collector.min_free_calls),
    ],
)
