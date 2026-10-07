import os
from pathlib import Path

from dotenv import load_dotenv

from upscale import log_safety

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

# Scout social / attention evidence (never a trade signal). Every provider is optional;
# without credentials it is reported as "not configured" and Scout keeps working.
# Reddit: an API application approved under Reddit's Responsible Builder Policy.
REDDIT_CLIENT_ID = os.getenv("UPSCALE_REDDIT_CLIENT_ID") or None
REDDIT_CLIENT_SECRET = os.getenv("UPSCALE_REDDIT_CLIENT_SECRET") or None
REDDIT_USER_AGENT = os.getenv("UPSCALE_REDDIT_USER_AGENT") or None
# Farcaster cast search through Neynar (credit-based plans).
NEYNAR_API_KEY = os.getenv("UPSCALE_NEYNAR_API_KEY") or None
# X recent search is billed per post read: also needs "x_allow_paid": true in the config.
X_BEARER_TOKEN = os.getenv("UPSCALE_X_BEARER_TOKEN") or None
# Comma-separated public Discourse forum URLs whose terms allow automated public reads.
SOCIAL_FORUMS = os.getenv("UPSCALE_SOCIAL_FORUMS") or None
# Optional JSON overriding social thresholds, e.g. '{"momentum": {"min_mentions": 5}}'.
SOCIAL_CONFIG = os.getenv("UPSCALE_SOCIAL_CONFIG") or None

# Growth Scout (discovery ranking; never BUY / SELL). Optional JSON overriding its validated
# weights and thresholds, e.g. '{"mode": "ALL_TRENDING", "safety": {"top_k": 5}}'.
GROWTH_CONFIG = os.getenv("UPSCALE_GROWTH_CONFIG") or None

# Outcome tracking (measurement only; never changes Scout or decisions). Stored in the Scout
# database. Optional JSON overriding its horizons / policies, e.g.
# UPSCALE_OUTCOME_CONFIG='{"observation": {"min_score_change": 15}}'. The background
# collector can be switched off with UPSCALE_OUTCOMES=0 (anchors are still recorded).
OUTCOME_CONFIG = os.getenv("UPSCALE_OUTCOME_CONFIG") or None
OUTCOMES_COLLECTOR = (os.getenv("UPSCALE_OUTCOMES") or "1").strip().lower() not in (
    "0",
    "false",
    "off",
)

# Background Scout: the regular Scout scan, run automatically while the backend is alive
# (lowest priority on every provider quota; see upscale.background_scout). On by default;
# UPSCALE_BACKGROUND_SCOUT=0 disables it. Interval in minutes (default 30, at least 5).
BACKGROUND_SCOUT = os.getenv("UPSCALE_BACKGROUND_SCOUT") or None
BACKGROUND_SCOUT_INTERVAL_MINUTES = os.getenv("UPSCALE_BACKGROUND_SCOUT_INTERVAL_MINUTES") or None
# Past interval + this many minutes without a completed scan, only Analyze / a running scan
# still defer background Scout (default 60, at least 5).
BACKGROUND_SCOUT_MAX_DEFERRAL_MINUTES = (
    os.getenv("UPSCALE_BACKGROUND_SCOUT_MAX_DEFERRAL_MINUTES") or None
)

# Point-in-Time Evidence Archive: an append-only record of the evidence production already
# fetched (market, DEX, on-chain safety, social, Growth Scout, Analyze), for Replay Lab.
# Archiving adds no provider requests. On by default; UPSCALE_EVIDENCE_ARCHIVE=0 disables
# it. Stored next to the Scout database by default (on Railway: the same /data volume).
EVIDENCE_ARCHIVE = (os.getenv("UPSCALE_EVIDENCE_ARCHIVE") or "1").strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)
EVIDENCE_DB_PATH = os.getenv("UPSCALE_EVIDENCE_DB") or str(
    Path(SCOUT_DB_PATH).expanduser().parent / "evidence.sqlite3"
)
# Optional bounded on-chain safety enrichment of Scout candidates (NEW Solana RPC / Helius
# requests, lowest production priority). Off by default; at most this many tokens per scan.
EVIDENCE_SAFETY_ENRICHMENT = os.getenv("UPSCALE_EVIDENCE_SAFETY_ENRICHMENT") or None
EVIDENCE_SAFETY_MAX_PER_REFRESH = os.getenv("UPSCALE_EVIDENCE_SAFETY_MAX_PER_REFRESH") or None

# Shadow / Paper Strategy Engine: simulated strategies over archived evidence (no orders,
# keys or provider requests). OFF by default; UPSCALE_SHADOW=1 runs it in the background at
# the lowest priority. Stored in UPSCALE_SHADOW_DB (default shadow.sqlite3 next to the Scout
# database). The background run id (default "production") and its start time (default and
# minimum: the clean-data cutoff 2026-09-30T05:50:00Z).
SHADOW = os.getenv("UPSCALE_SHADOW") or None
SHADOW_INTERVAL_MINUTES = os.getenv("UPSCALE_SHADOW_INTERVAL_MINUTES") or None
SHADOW_RUN = os.getenv("UPSCALE_SHADOW_RUN") or None
SHADOW_SINCE = os.getenv("UPSCALE_SHADOW_SINCE") or None
# Comma-separated run ids advanced together (e.g. continuous-v2,continuous-v2-realistic).
# Unset: only UPSCALE_SHADOW_RUN, exactly as before. Listed runs must already exist.
SHADOW_RUNS = os.getenv("UPSCALE_SHADOW_RUNS") or None
# Held-position watch: exact-pool prices for open positions of EVIDENCE_AWARE_V2 Shadow runs
# (upscale.held_position_watch). On by default (idle without such positions);
# UPSCALE_SHADOW_WATCH=0 disables it. Interval in minutes (default 15, at least 5).
SHADOW_WATCH = os.getenv("UPSCALE_SHADOW_WATCH") or None
SHADOW_WATCH_INTERVAL_MINUTES = os.getenv("UPSCALE_SHADOW_WATCH_INTERVAL_MINUTES") or None
# Data retention (Scout and Evidence Archive databases; Shadow is never pruned). Background
# passes are OFF unless UPSCALE_RETENTION_ENABLED is 1 (delete) or dry-run (log only);
# UPSCALE_EVIDENCE_RETENTION_DAYS / UPSCALE_SCOUT_SNAPSHOT_RETENTION_DAYS /
# UPSCALE_SOCIAL_RETENTION_DAYS (default 7, at least 3) and UPSCALE_OUTCOME_RETENTION_DAYS
# (default 30, at least 14). Read by upscale.services.retention.config.

# Secret-safe logging for the whole process (every logger and level): the credentials above
# and any credential-looking query parameter or header are masked in log output.
log_safety.install()
log_safety.register_secrets(
    [
        COINGECKO_API_KEY,
        HELIUS_API_KEY,
        REDDIT_CLIENT_SECRET,
        NEYNAR_API_KEY,
        X_BEARER_TOKEN,
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("ANTHROPIC_AUTH_TOKEN"),
    ]
)
log_safety.register_url(SOLANA_RPC_URL)
