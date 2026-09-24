"""External data services used by agents. All network I/O lives here, never in agents."""

from upscale.config import (
    COINGECKO_API_KEY,
    NEWS_DISABLED_FEEDS,
    NEWS_FEEDS,
    NEWS_MODEL,
    VISION_MODEL,
)
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.kraken import KrakenProvider
from upscale.services.market_data import MarketDataService
from upscale.services.news import NewsService
from upscale.services.news_sentiment_model import ClaudeNewsSentimentModel
from upscale.services.rss_news import RssNewsProvider, configured_feeds
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
vision_service = VisionService(ClaudeVisionModel(model=VISION_MODEL))
news_service = NewsService(
    RssNewsProvider(feeds=configured_feeds(NEWS_FEEDS, NEWS_DISABLED_FEEDS)),
    ClaudeNewsSentimentModel(model=NEWS_MODEL),
)
