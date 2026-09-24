"""Decides which agents handle a request. Keyword-based for now; can become model-based later."""

import re
from dataclasses import dataclass, field

from upscale.schemas import AgentName

# fmt: off
# Tickers that are unambiguous even in lowercase.
ASSET_ALIASES: dict[str, str] = {
    "btc": "BTC", "bitcoin": "BTC",
    "eth": "ETH", "ethereum": "ETH", "ether": "ETH",
    "sol": "SOL", "solana": "SOL",
    "xrp": "XRP", "ripple": "XRP",
    "bnb": "BNB",
    "doge": "DOGE", "dogecoin": "DOGE",
    "ada": "ADA", "cardano": "ADA",
    "avax": "AVAX", "avalanche": "AVAX",
    "ltc": "LTC", "litecoin": "LTC",
    "trx": "TRX", "tron": "TRX",
    "shib": "SHIB", "pepe": "PEPE",
    "chainlink": "LINK", "polkadot": "DOT", "toncoin": "TON",
}
# Tickers that are also common English words: only matched as uppercase or with a $ prefix.
AMBIGUOUS_TICKERS = {"LINK", "DOT", "TON", "OP", "ARB", "SUI", "NEAR", "APT"}

CRYPTO_TERMS = {
    "crypto", "cryptocurrency", "coin", "coins", "token", "tokens", "altcoin", "altcoins",
    "defi", "blockchain", "stablecoin", "memecoin", "bull", "bullish", "bear", "bearish",
}

INTENT_KEYWORDS: dict[AgentName, set[str]] = {
    "technical_analysis": {
        "chart", "technical", "ta", "rsi", "macd", "ema", "sma", "moving average", "support",
        "resistance", "trend", "trendline", "breakout", "breakdown", "pattern", "indicator",
        "indicators", "candle", "candles", "fibonacci", "bollinger", "timeframe",
    },
    "market": {
        "price", "volume", "market cap", "marketcap", "liquidity", "dominance", "funding",
        "open interest", "order book", "volatility", "volatile", "market",
    },
    "news_sentiment": {
        "news", "sentiment", "headline", "headlines", "twitter", "social", "fear", "greed",
        "hype", "rumor", "rumors", "announcement", "etf", "regulation", "sec", "why",
    },
    "opportunity": {
        "opportunity", "opportunities", "setup", "scenario", "scenarios", "outlook",
        "potential", "upside", "downside", "entry", "target", "buy", "sell", "long", "short",
        "trade", "prediction", "predict", "forecast", "should i", "best move", "what to do",
        "what should i do", "good entry", "buy or wait", "buy or sell", "sell or hold",
    },
}
# An explicit request for an action. These run the full evidence pipeline, news included,
# because current news can materially change a decision.
DECISION_KEYWORDS = {
    "buy", "sell", "should i", "best move", "what to do", "what should i do", "entry",
    "good entry", "buy or wait", "buy or sell", "sell or hold",
}
# Decision phrases unambiguous enough to route without a coin or crypto word ("What should
# I do?"); the opportunity agent then asks for an asset if none can be identified.
STANDALONE_DECISION_PHRASES = {
    "what should i do", "best move", "buy or wait", "buy or sell", "sell or hold",
    "good entry", "what to do", "should i buy", "should i sell",
}
DECISION_AGENTS: tuple[AgentName, ...] = (
    "technical_analysis", "market", "news_sentiment", "opportunity",
)

# Agents used for a crypto question that doesn't ask for anything specific.
GENERAL_CRYPTO_AGENTS: tuple[AgentName, ...] = (
    "technical_analysis", "market", "news_sentiment", "opportunity",
)
# Risk reviews the evidence agents; opportunity decides last, after the risk review.
AGENT_ORDER: tuple[AgentName, ...] = (
    "vision", "technical_analysis", "market", "news_sentiment", "risk", "opportunity",
)
# fmt: on

_WORD_RE = re.compile(r"\$?[A-Za-z][A-Za-z0-9]*")
_TIMEFRAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("30m", re.compile(r"\b30[ -]?(?:m|min|mins|minutes?)\b")),
    ("15m", re.compile(r"\b15[ -]?(?:m|min|mins|minutes?)\b")),
    ("5m", re.compile(r"\b5[ -]?(?:m|min|mins|minutes?)\b")),
    ("1m", re.compile(r"\b1[ -]?(?:m|min|minute)\b")),
    ("4h", re.compile(r"\b(?:4[ -]?(?:h|hr|hrs|hours?)|four[ -]hours?)\b")),
    ("1h", re.compile(r"\b(?:1[ -]?(?:h|hr|hour)|hourly)\b")),
    ("1d", re.compile(r"\b(?:1[ -]?(?:d|day)|daily)\b")),
)


@dataclass
class RoutingDecision:
    assets: list[str] = field(default_factory=list)
    # Chart timeframe the user asked for ("1m", "5m", "15m", "30m", "1h", "4h", "1d"), if any.
    timeframe: str | None = None
    # Selected agents (in AGENT_ORDER) mapped to the reason each was selected.
    reasons: dict[AgentName, str] = field(default_factory=dict)

    @property
    def agents(self) -> list[AgentName]:
        return list(self.reasons)


def detect_assets(text: str) -> list[str]:
    found: list[str] = []
    for token in _WORD_RE.findall(text):
        bare = token.lstrip("$")
        symbol = ASSET_ALIASES.get(bare.lower())
        if (
            symbol is None
            and bare.upper() in AMBIGUOUS_TICKERS
            and (token.startswith("$") or bare.isupper())
        ):
            symbol = bare.upper()
        if symbol and symbol not in found:
            found.append(symbol)
    return found


def detect_timeframe(text: str) -> str | None:
    """First timeframe mentioned, e.g. "BTC 4h chart" -> "4h", "daily RSI" -> "1d"."""
    lowered = text.lower()
    matches = [(m.start(), tf) for tf, rx in _TIMEFRAME_PATTERNS if (m := rx.search(lowered))]
    return min(matches)[1] if matches else None


def _matched_keywords(text: str, words: set[str], keywords: set[str]) -> list[str]:
    return sorted(kw for kw in keywords if (kw in text if " " in kw else kw in words))


def route(query: str, has_images: bool) -> RoutingDecision:
    lowered = query.lower()
    words = {token.lstrip("$") for token in _WORD_RE.findall(lowered)}
    assets = detect_assets(query)
    is_crypto = bool(assets) or bool(words & CRYPTO_TERMS)

    reasons: dict[AgentName, str] = {}
    if has_images:
        reasons["vision"] = "Message includes a screenshot."
        reasons["technical_analysis"] = "Screenshots are treated as charts to analyze."
        reasons["market"] = "Live market data to compare with the screenshot."
        reasons["opportunity"] = "A chart analysis ends with a BUY / SELL / WAIT read."

    intents = {
        agent: matched
        for agent, keywords in INTENT_KEYWORDS.items()
        if (matched := _matched_keywords(lowered, words, keywords))
    }
    decision = _matched_keywords(lowered, words, DECISION_KEYWORDS)
    standalone = _matched_keywords(lowered, words, STANDALONE_DECISION_PHRASES)
    # Intent keywords alone ("price", "trend") only count when the request is about crypto
    # or comes with a chart, so small talk doesn't trigger the whole pipeline.
    if is_crypto or has_images or standalone:
        if decision:
            why = f"Decision request ({', '.join(decision)}): needs full evidence."
            for agent in DECISION_AGENTS:
                reasons.setdefault(agent, why)
        for agent, matched in intents.items():
            reasons.setdefault(agent, f"Request mentions: {', '.join(matched)}.")
        if is_crypto and not intents:
            for agent in GENERAL_CRYPTO_AGENTS:
                reasons.setdefault(agent, "General crypto question.")

    if "opportunity" in reasons:
        # A BUY / SELL / WAIT read is made from live technical and market evidence.
        for agent in ("technical_analysis", "market"):
            reasons.setdefault(agent, "The opportunity decision needs live evidence.")
    if reasons:
        reasons["risk"] = "Risk review runs whenever other agents do."

    ordered = {agent: reasons[agent] for agent in AGENT_ORDER if agent in reasons}
    return RoutingDecision(assets=assets, timeframe=detect_timeframe(query), reasons=ordered)
