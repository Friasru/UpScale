import os
from pathlib import Path

from dotenv import load_dotenv

# Load the repo-root .env if present; real environment variables take precedence.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_CORS_ORIGINS = ",".join(
    [
        "http://localhost:1420",  # Vite dev server / `tauri dev`
        "http://127.0.0.1:1420",
        "tauri://localhost",  # Tauri production build (macOS/Linux)
        "http://tauri.localhost",  # Tauri production build (Windows)
    ]
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("UPSCALE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

# Market data (CoinGecko). Works without a key; a free "demo" key raises the rate limit.
COINGECKO_API_KEY = os.getenv("UPSCALE_COINGECKO_API_KEY") or None

# Vision (chart screenshots). The Anthropic SDK reads ANTHROPIC_API_KEY itself; without
# credentials the Vision agent reports that screenshot analysis is unavailable.
VISION_MODEL = os.getenv("UPSCALE_VISION_MODEL") or "claude-opus-5"

# News (publisher RSS feeds; no key needed). Optional override of the feed list, with an
# optional per-feed AI policy (none / headline / description; default headline):
# UPSCALE_NEWS_FEEDS="Name|https://feed-url,Name|https://feed-url|none".
NEWS_FEEDS = os.getenv("UPSCALE_NEWS_FEEDS") or None
# Model that labels each article's sentiment and potential impact. It uses the same Anthropic
# credentials as Vision; without them articles are still reported, just unclassified.
NEWS_MODEL = os.getenv("UPSCALE_NEWS_MODEL") or "claude-opus-5"
# Comma-separated feed names to switch off, e.g. UPSCALE_NEWS_DISABLED_FEEDS="Decrypt".
NEWS_DISABLED_FEEDS = os.getenv("UPSCALE_NEWS_DISABLED_FEEDS") or None
