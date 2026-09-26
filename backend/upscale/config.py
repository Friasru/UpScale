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
# Model that answers general educational questions ("What is RSI?"). Same Anthropic
# credentials as Vision; without them those questions get an "unavailable" reply.
EXPLAINER_MODEL = os.getenv("UPSCALE_EXPLAINER_MODEL") or "claude-opus-5"

# On-chain token safety (Solana). Helius is preferred (its API also counts holders); any
# other Solana JSON-RPC URL works too, without holder counts. Without either, on-chain
# safety is reported as unavailable. Keys are never logged or shown in errors.
HELIUS_API_KEY = os.getenv("UPSCALE_HELIUS_API_KEY") or None
SOLANA_RPC_URL = os.getenv("UPSCALE_SOLANA_RPC_URL") or None
# Pages of 1,000 token accounts Helius may scan per mint for holder concentration. A token
# with more holders gets lower-bound (incomplete) figures, flagged as such.
HELIUS_MAX_HOLDER_PAGES = int(os.getenv("UPSCALE_HELIUS_MAX_HOLDER_PAGES") or 10)

# Scout (token discovery). Snapshots are stored in a local SQLite file, created on first use.
SCOUT_DB_PATH = os.getenv("UPSCALE_SCOUT_DB") or str(Path.home() / ".upscale" / "scout.sqlite3")
# Optional JSON overriding Scout's validated thresholds, e.g.
# UPSCALE_SCOUT_CONFIG='{"chains": ["solana", "base"], "filters": {"min_liquidity_usd": 5000}}'
SCOUT_CONFIG = os.getenv("UPSCALE_SCOUT_CONFIG") or None
