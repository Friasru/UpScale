"""Provider-agnostic crypto news: article models, errors, provider interface, relevance,
de-duplication, recency, sentiment aggregation, and a caching service.

Agents depend on `NewsService`. A `NewsProvider` returns real, retrieved articles; UpScale
never writes headlines, links or dates itself. Everything that decides which stories are
shown (relevance to the asset, duplicates, age) is deterministic code in this module. Only
the per-article sentiment/impact labels come from a `NewsSentimentModel`, which sees just
the retrieved article text and refers to articles by id, so it cannot add stories.
"""

import asyncio
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from upscale.routing import AMBIGUOUS_TICKERS, ASSET_ALIASES
from upscale.schemas import Level
from upscale.services.market_data import RateLimiter
from upscale.services.news_sentiment_model import (
    ArticleAssessment,
    ArticleInput,
    ArticleSentiment,
    NewsSentimentModel,
    SentimentModelError,
)

# --- Tunables ----------------------------------------------------------------------------------

STALE_HOURS = 48.0  # older than this is flagged as stale
MAX_AGE = timedelta(days=7)  # older articles are ignored entirely
FUTURE_TOLERANCE = timedelta(minutes=15)  # clock skew allowed on publication times
MAX_ASSET_STORIES = 6
MAX_MENTIONED_STORIES = 2  # articles that only mention the asset in passing
MAX_MARKET_STORIES = 3  # market-wide stories shown alongside asset-specific ones
MAX_GENERAL_STORIES = 8  # stories for a general "crypto news" question (no asset)
DUPLICATE_SIMILARITY = 0.75  # Jaccard similarity of title words for syndicated copies
# Independent write-ups of one event: at least this many shared title words, covering at
# least half of the shorter title, published within this many hours of each other.
SAME_EVENT_SHARED_WORDS = 5
SAME_EVENT_HOURS = 12.0

# --- Models ------------------------------------------------------------------------------------

Scope = Literal["asset", "market"]
# How an article relates to the requested asset:
#   headline  - the asset is named in the title
#   tagged    - the publisher tagged the article with the asset
#   mentioned - the asset is only named in the summary
#   market    - no asset-specific link, but a market-wide topic (regulation, rates, ETFs, ...)
Relevance = Literal["headline", "tagged", "mentioned", "market"]
# What the sentiment model may see from an article, set per source by the provider:
#   none        - nothing; the article is reported but never sent to the model
#   headline    - headline, publisher and publication time
#   description - also the publisher's short feed description
ModelUse = Literal["none", "headline", "description"]
OverallSentiment = Literal[
    "bullish", "bearish", "neutral", "mixed", "insufficient_data", "unavailable"
]


class NewsArticle(BaseModel):
    """One article exactly as a provider returned it. Missing values stay None/empty."""

    title: str
    source: str  # publisher, e.g. "CoinDesk"
    url: str
    published_at: datetime  # UTC, as published by the source
    summary: str | None = None
    # Asset tickers the provider itself tagged the article with (never guessed).
    tickers: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    # Nothing is sent to a model unless the provider says the source allows it.
    model_use: ModelUse = "none"


class NewsBatch(BaseModel):
    """Everything one provider call returned."""

    provider: str
    sources: list[str]  # sources that returned data
    failed_sources: list[dict[str, str]] = Field(default_factory=list)  # {"source", "reason"}
    articles: list[NewsArticle]
    skipped_items: int = 0  # items dropped for missing/invalid title, link or date
    fetched_at: datetime


class Story(BaseModel):
    """A selected, de-duplicated article with UpScale's analysis attached."""

    id: str  # short per-report id, e.g. "n1"; the only handle the sentiment model gets
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str | None
    tickers: list[str]
    also_reported_by: list[str]  # other sources that carried the same story
    scope: Scope
    relevance: Relevance
    age_hours: float
    stale: bool
    model_use: ModelUse
    # None = not classified (no model, model failure, or the model skipped it).
    sentiment: ArticleSentiment | None = None
    impact: Level | None = None
    impact_reason: str | None = None


class NewsReport(BaseModel):
    asset: str | None  # None = the overall crypto market
    provider: str
    sources: list[str]
    failed_sources: list[dict[str, str]]
    retrieved_at: datetime  # when the articles were fetched from the provider
    analyzed_at: datetime  # when ages/staleness were computed
    model: str | None  # sentiment model used, if any
    stories: list[Story]
    asset_story_count: int
    # Tone of the asset-specific stories (or of the market stories when asset is None).
    overall_sentiment: OverallSentiment
    market_news_sentiment: OverallSentiment
    # Why the aggregate differs from a plain article count, when it does (else None).
    sentiment_basis: str | None = None
    market_sentiment_basis: str | None = None
    sentiment_counts: dict[str, int]
    conflicts: list[str]
    uncertainties: list[str]
    excluded: list[str]  # preprocessing notes: old, future-dated or duplicate articles
    discarded: list[str]  # sentiment-model output UpScale refused to use


# --- Errors and provider interface ---------------------------------------------------------------


class NewsError(Exception):
    """News could not be retrieved."""


class NewsUnavailableError(NewsError):
    """The provider failed, timed out, rate-limited us, or returned unusable data."""


class NewsProvider(Protocol):
    name: str
    # True if `fetch_articles` narrows results to `symbols` itself. False for a general feed
    # the service filters locally, so a single fetch serves every asset.
    filters_by_asset: bool

    async def fetch_articles(self, symbols: Sequence[str]) -> NewsBatch:
        """Return real retrieved articles or raise `NewsUnavailableError`."""
        ...


# --- Asset matching ------------------------------------------------------------------------------

# Extra names matched case-sensitively, for tickers the router knows without a name alias.
EXTRA_NAMES: dict[str, tuple[str, ...]] = {
    "OP": ("Optimism",),
    "ARB": ("Arbitrum",),
    "SUI": ("Sui",),
    "NEAR": ("NEAR Protocol", "Near Protocol"),
    "APT": ("Aptos",),
    "BNB": ("BNB Chain", "Binance Coin"),
    "TON": ("Toncoin", "The Open Network"),
}
# Router aliases that are also everyday words: only matched capitalized.
CAPITALIZED_ONLY = {"ripple", "avalanche"}
# Names that start a different asset's name ("Bitcoin Cash") or a common phrase.
NOT_FOLLOWED_BY = {
    "bitcoin": r"(?!\s+(?:cash|sv|gold)\b)",
    "ethereum": r"(?!\s+classic\b)",
    "ripple": r"(?!\s+effects?\b)",
}
KNOWN_SYMBOLS = frozenset(ASSET_ALIASES.values()) | AMBIGUOUS_TICKERS


def _asset_patterns() -> dict[str, re.Pattern[str]]:
    names: dict[str, set[str]] = defaultdict(set)
    for alias, symbol in ASSET_ALIASES.items():
        if alias.upper() != symbol:
            names[symbol].add(alias)
    patterns: dict[str, re.Pattern[str]] = {}
    for symbol in sorted(KNOWN_SYMBOLS):
        # Tickers are case-sensitive ("SOL", "$sol" is also fine), so "sol" or "link" in
        # ordinary text doesn't count.
        parts = [
            rf"(?<![A-Za-z0-9])\$(?i:{re.escape(symbol)})(?![A-Za-z0-9])",
            rf"(?<![A-Za-z0-9$]){re.escape(symbol)}(?![A-Za-z0-9])",
        ]
        for name in sorted(names[symbol]):
            tail = NOT_FOLLOWED_BY.get(name, "")
            if name in CAPITALIZED_ONLY:
                parts.append(rf"\b{re.escape(name.capitalize())}\b{tail}")
            else:
                parts.append(rf"(?i:\b{re.escape(name)}\b{tail})")
        parts += [rf"\b{re.escape(n)}\b" for n in EXTRA_NAMES.get(symbol, ())]
        patterns[symbol] = re.compile("|".join(parts))
    return patterns


_ASSET_PATTERNS = _asset_patterns()


def mentioned_assets(text: str) -> set[str]:
    """Tickers of the assets a piece of text names, e.g. "Bitcoin ETF inflows" -> {"BTC"}."""
    return {symbol for symbol, rx in _ASSET_PATTERNS.items() if rx.search(text)}


def tag_tickers(categories: Sequence[str]) -> list[str]:
    """Tickers for publisher tags that exactly name an asset ("Bitcoin", "ETH")."""
    by_name = {alias: symbol for alias, symbol in ASSET_ALIASES.items()}
    by_name |= {n.lower(): s for s, names in EXTRA_NAMES.items() for n in names}
    found: list[str] = []
    for category in categories:
        tag = category.strip()
        symbol = by_name.get(tag.lower())
        if symbol is None and tag.isupper() and tag in KNOWN_SYMBOLS:
            symbol = tag
        if symbol and symbol not in found:
            found.append(symbol)
    return found


# Topics that can move the whole crypto market, so they're relevant to any asset.
_MARKET_TOPICS = re.compile(
    r"\b(?:crypto(?:currency)?\s+(?:market|markets|prices|sell-?off|rally|crash|regulation|"
    r"regulators?|bill|legislation|tax|policy)|digital[- ]asset\s+(?:market|regulation|bill)|"
    r"spot\s+(?:bitcoin\s+|ether\s+|crypto\s+)?ETFs?|(?:crypto\s+)?ETF\s+(?:flows?|inflows?|"
    r"outflows?|approvals?)|stablecoin\s+(?:bill|law|regulation|rules)|Federal\s+Reserve|"
    r"interest\s+rates?|rate\s+(?:cut|cuts|hike|hikes)|inflation|liquidations?)\b",
    re.IGNORECASE,
)
_MARKET_ACRONYMS = re.compile(r"\b(?:SEC|CFTC|FOMC|CPI|Fed)\b")


def is_market_wide(article: NewsArticle) -> bool:
    # Headline only: summaries mention regulators and ETFs in passing far too often.
    return bool(_MARKET_TOPICS.search(article.title) or _MARKET_ACRONYMS.search(article.title))


def relevance(article: NewsArticle, asset: str | None) -> tuple[Scope, Relevance] | None:
    """How `article` relates to `asset` (None = the whole market), or None if it doesn't."""
    in_title = mentioned_assets(article.title)
    if asset is not None:
        if asset in in_title:
            return "asset", "headline"
        if asset in article.tickers:
            return "asset", "tagged"
        if article.summary and asset in mentioned_assets(article.summary):
            return "asset", "mentioned"
    # A market-wide story shouldn't be about some other coin. Bitcoin is the exception:
    # it's the market's bellwether, so BTC headlines count as market news for other assets.
    other_assets = in_title - {asset, "BTC"}
    if other_assets:
        return None
    if is_market_wide(article) or (asset is None and "BTC" in in_title):
        return "market", "market"
    return None


# --- De-duplication ------------------------------------------------------------------------------

_STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "as", "is", "at", "by"}


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}"


# Word pairs that make two otherwise similar headlines different stories.
_OPPOSITES = [
    frozenset(p)
    for p in (
        ("inflows", "outflows"),
        ("inflow", "outflow"),
        ("rises", "falls"),
        ("gains", "losses"),
        ("up", "down"),
        ("above", "below"),
        ("bullish", "bearish"),
        ("buy", "sell"),
        ("buys", "sells"),
        ("approves", "rejects"),
        ("surges", "plunges"),
        ("high", "low"),
    )
]


def title_words(title: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9$]+", title.lower())
    return frozenset(w for w in words if len(w) > 1 and w not in _STOPWORDS)


def same_story(a: frozenset[str], b: frozenset[str], hours_apart: float = 0.0) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    differing = (a - b) | (b - a)
    if any(pair <= differing for pair in _OPPOSITES):
        return False  # e.g. "ETF inflows" vs "ETF outflows"
    shared = len(a & b)
    if min(len(a), len(b)) >= 4 and shared / len(a | b) >= DUPLICATE_SIMILARITY:
        return True  # syndicated or lightly edited copy
    return (
        hours_apart <= SAME_EVENT_HOURS
        and shared >= SAME_EVENT_SHARED_WORDS
        and shared / min(len(a), len(b)) >= 0.5
    )


def dedupe(articles: Sequence[NewsArticle]) -> list[tuple[NewsArticle, list[str]]]:
    """Collapse the same story from several feeds (syndicated copies, or several outlets'
    write-ups of one event) into the earliest copy, recording who else carried it."""
    # Each cluster: original, other sources, and the URLs/title words of every member.
    clusters: list[tuple[NewsArticle, list[str], set[str], list[frozenset[str]]]] = []
    for article in sorted(articles, key=lambda a: a.published_at):
        url, words = canonical_url(article.url), title_words(article.title)
        for original, also, urls, member_words in clusters:
            hours = (article.published_at - original.published_at).total_seconds() / 3600
            if url in urls or any(same_story(words, w, hours) for w in member_words):
                if article.source != original.source and article.source not in also:
                    also.append(article.source)
                urls.add(url)
                member_words.append(words)
                break
        else:
            clusters.append((article, [], {url}, [words]))
    return [(original, also) for original, also, _, _ in clusters]


# --- Selection -------------------------------------------------------------------------------------

_RANK: dict[Relevance, int] = {"headline": 0, "tagged": 1, "mentioned": 2, "market": 3}


def select_stories(
    articles: Sequence[NewsArticle], asset: str | None, now: datetime
) -> tuple[list[Story], list[str]]:
    """Deterministically pick the stories to report: recent, relevant, de-duplicated."""
    excluded: list[str] = []
    too_old = [a for a in articles if now - a.published_at > MAX_AGE]
    future = [a for a in articles if a.published_at - now > FUTURE_TOLERANCE]
    if too_old:
        excluded.append(f"Ignored {len(too_old)} article(s) older than {MAX_AGE.days} days.")
    if future:
        excluded.append(f"Ignored {len(future)} article(s) with a publication time in the future.")
    usable = [a for a in articles if a not in too_old and a not in future]

    unique = dedupe(usable)
    if merged := len(usable) - len(unique):
        excluded.append(f"Merged {merged} duplicate or syndicated article(s).")

    candidates: list[tuple[NewsArticle, list[str], Scope, Relevance, float]] = []
    for article, also in unique:
        if (match := relevance(article, asset)) is not None:
            age = max(0.0, (now - article.published_at).total_seconds() / 3600)
            candidates.append((article, also, *match, age))

    def order(c: tuple[NewsArticle, list[str], Scope, Relevance, float]) -> tuple[bool, int, float]:
        return (c[4] > STALE_HOURS, _RANK[c[3]], c[4])  # fresh first, then relevance, newest

    candidates.sort(key=order)
    asset_items = [c for c in candidates if c[2] == "asset"]
    mentioned = [c for c in asset_items if c[3] == "mentioned"][:MAX_MENTIONED_STORIES]
    asset_items = [c for c in asset_items if c[3] != "mentioned" or c in mentioned]
    market_items = [c for c in candidates if c[2] == "market"]
    chosen = (
        asset_items[:MAX_ASSET_STORIES] + market_items[:MAX_MARKET_STORIES]
        if asset is not None
        else market_items[:MAX_GENERAL_STORIES]
    )

    stories = [
        Story(
            id=f"n{i}",
            title=article.title,
            source=article.source,
            url=article.url,
            published_at=article.published_at,
            summary=article.summary,
            tickers=article.tickers,
            also_reported_by=also,
            scope=scope,
            relevance=rel,
            age_hours=round(age, 1),
            stale=age > STALE_HOURS,
            model_use=article.model_use,
        )
        for i, (article, also, scope, rel, age) in enumerate(chosen, start=1)
    ]
    return stories, excluded


# --- Sentiment ---------------------------------------------------------------------------------------

# Reasons that assert causation, certainty or predictions are dropped (the label is kept).
_OVERCLAIM = re.compile(
    r"\b(?:caus(?:ed|es|ing)|guarantee[ds]?|proves?|certain(?:ly)?\s+to|definitely|"
    r"will\s+(?:surge|soar|rise|climb|rally|pump|fall|drop|crash|plunge|dump|tank)|"
    r"(?:should|must)\s+(?:buy|sell)|buy\s+signal|sell\s+signal)\b",
    re.IGNORECASE,
)
_IMPACT_WEIGHT: dict[Level, float] = {"low": 1.0, "medium": 2.0, "high": 3.0}


def apply_assessments(
    stories: list[Story], assessments: Sequence[ArticleAssessment]
) -> tuple[list[Story], list[str]]:
    """Attach model labels to stories by id. Unknown or repeated ids are discarded, so the
    model can label retrieved articles but never introduce new ones."""
    by_id = {s.id: s for s in stories}
    labeled: dict[str, Story] = {}
    discarded: list[str] = []
    for a in assessments:
        if a.id not in by_id:
            discarded.append(f"Ignored a sentiment label for unknown article id {a.id!r}.")
            continue
        if a.id in labeled:
            discarded.append(f"Ignored a repeated sentiment label for article {a.id}.")
            continue
        reason: str | None = " ".join(a.reason.split()) or None
        if reason and _OVERCLAIM.search(reason):
            discarded.append(
                f"Dropped the impact explanation for article {a.id}: it asserted causation "
                "or a price prediction."
            )
            reason = None
        labeled[a.id] = by_id[a.id].model_copy(
            update={"sentiment": a.sentiment, "impact": a.impact, "impact_reason": reason}
        )
    return [labeled.get(s.id, s) for s in stories], discarded


def _impact_weight(s: Story) -> float:
    return _IMPACT_WEIGHT[s.impact or "low"]


def _recency_weight(s: Story) -> float:
    return 0.5 if s.stale else 1.0


def _full_weight(s: Story) -> float:
    return _impact_weight(s) * _recency_weight(s)


def _verdict(classified: Sequence[Story], weight: Callable[[Story], float]) -> OverallSentiment:
    totals: dict[str, float] = defaultdict(float)
    for s in classified:
        totals[s.sentiment or ""] += weight(s)
    bull, bear = totals["bullish"], totals["bearish"]
    everything = sum(totals.values())
    if bull + bear < everything / 3:  # mostly neutral/mixed coverage
        return "mixed" if totals["mixed"] > totals["neutral"] else "neutral"
    # Mixed unless one side outweighs the other at least 2:1.
    if bull and bear and min(bull, bear) > 0.5 * max(bull, bear):
        return "mixed"
    return "bullish" if bull > bear else "bearish"


def aggregate_sentiment(stories: Sequence[Story]) -> OverallSentiment:
    """Overall tone of a set of stories, weighted by impact and recency.

    Based only on article labels: price movement never enters into it.
    """
    if not stories:
        return "insufficient_data"
    classified = [s for s in stories if s.sentiment is not None]
    if not classified:
        return "unavailable"
    return _verdict(classified, _full_weight)


def sentiment_basis(stories: Sequence[Story], label: str = "Overall sentiment") -> str | None:
    """Plain-language reason for the aggregate sentiment, when weighting by impact and
    recency gives a different answer than simply counting articles; None otherwise."""
    overall = aggregate_sentiment(stories)
    classified = [s for s in stories if s.sentiment is not None]
    if overall not in ("bullish", "bearish", "neutral", "mixed"):
        return None
    by_count = _verdict(classified, lambda s: 1.0)
    if by_count == overall:
        return None
    if _verdict(classified, _impact_weight) == overall:
        factor = "impact"
    elif _verdict(classified, _recency_weight) == overall:
        factor = "recency"
    else:
        factor = "impact and recency"
    if overall in ("bullish", "bearish"):
        other = "bearish" if overall == "bullish" else "bullish"
        mine = sum(s.sentiment == overall for s in classified)
        theirs = sum(s.sentiment == other for s in classified)
        if mine == theirs:
            despite = "despite an even bullish/bearish article count"
        elif mine < theirs:
            despite = f"despite more {other} than {overall} articles"
        else:
            despite = f"although the article count alone would suggest {by_count}"
        weight = {
            "impact": "impact weight",
            "recency": "weight because stale stories count less",
            "impact and recency": "weight once impact and staleness are considered",
        }[factor]
        return f"{label} is {overall} because {overall} coverage carries more {weight} {despite}."
    why = {
        "impact": "stories are weighted by potential impact",
        "recency": "stale stories count less",
        "impact and recency": "stories are weighted by potential impact and stale ones count less",
    }[factor]
    return (
        f"{label} is {overall} rather than {by_count} (as the article count alone would "
        f"suggest) because {why}."
    )


def find_conflicts(stories: Sequence[Story]) -> list[str]:
    bullish = [s for s in stories if s.sentiment == "bullish"]
    bearish = [s for s in stories if s.sentiment == "bearish"]
    if not bullish or not bearish:
        return []
    b, s = bullish[0], bearish[0]
    return [
        f"Coverage points both ways: {len(bullish)} bullish vs {len(bearish)} bearish "
        f'article(s), e.g. "{b.title}" ({b.source}) vs "{s.title}" ({s.source}).'
    ]


# --- Service -----------------------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NewsService:
    """Fetches, filters, classifies and caches news. Only real provider articles are
    reported; a failed provider raises `NewsError` and nothing is substituted."""

    def __init__(
        self,
        provider: NewsProvider,
        model: NewsSentimentModel | None = None,
        cache_ttl: float = 300.0,
        partial_cache_ttl: float = 60.0,
        max_calls_per_minute: int = 6,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _utcnow,
    ):
        self.provider = provider
        self.model = model
        self.cache_ttl = cache_ttl
        # A batch where some sources failed is kept briefly so they're retried soon.
        self.partial_cache_ttl = partial_cache_ttl
        self._clock = clock
        self._now = now
        self._limiter = RateLimiter(max_calls_per_minute, 60.0, clock)
        self._batches: dict[tuple[str, ...], tuple[float, NewsBatch]] = {}
        self._reports: dict[tuple[str | None, str], tuple[float, NewsReport]] = {}
        self._locks: dict[object, asyncio.Lock] = {}

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def reset(self) -> None:
        """Forget cached data and rate-limit history (for tests and provider swaps)."""
        self._batches.clear()
        self._reports.clear()
        self._locks.clear()
        self._limiter = RateLimiter(self._limiter.max_calls, self._limiter.period, self._clock)

    async def get_articles(self, symbols: Sequence[str] = ()) -> NewsBatch:
        """Raw articles from the provider: cached, rate-limited, one request per key at a time."""
        key = tuple(sorted({s.upper() for s in symbols})) if self.provider.filters_by_asset else ()

        def cached() -> NewsBatch | None:
            entry = self._batches.get(key)
            return entry[1] if entry and entry[0] > self._clock() else None

        if (hit := cached()) is not None:
            return hit
        # Concurrent callers (e.g. several assets in one question) share one provider call.
        async with self._locks.setdefault(("batch", key), asyncio.Lock()):
            if (hit := cached()) is not None:
                return hit
            if not self._limiter.try_acquire():
                raise NewsUnavailableError(
                    "UpScale's news request limit was reached; try again in a minute"
                )
            batch = await self.provider.fetch_articles(list(key))  # failures are not cached
            ttl = self.partial_cache_ttl if batch.failed_sources else self.cache_ttl
            self._batches[key] = (self._clock() + ttl, batch)
            return batch

    async def get_report(self, asset: str | None) -> NewsReport:
        """Recent relevant news about `asset` (None = the crypto market) with sentiment."""
        asset = asset.upper() if asset else None
        batch = await self.get_articles([asset] if asset else [])
        key = (asset, f"{batch.provider}@{batch.fetched_at.isoformat()}")

        def cached() -> NewsReport | None:
            entry = self._reports.get(key)
            return entry[1] if entry and entry[0] > self._clock() else None

        if (hit := cached()) is not None:
            return hit
        async with self._locks.setdefault(("report", key), asyncio.Lock()):
            if (hit := cached()) is not None:
                return hit
            report, cacheable = await self._analyze(asset, batch)
            now = self._clock()
            self._reports = {k: v for k, v in self._reports.items() if v[0] > now}
            if cacheable:
                self._reports[key] = (now + self.cache_ttl, report)
            return report

    async def _analyze(self, asset: str | None, batch: NewsBatch) -> tuple[NewsReport, bool]:
        now = self._now()
        stories, excluded = select_stories(batch.articles, asset, now)
        if batch.skipped_items:
            excluded.append(
                f"Skipped {batch.skipped_items} feed item(s) missing a title, link or valid date."
            )
        uncertainties: list[str] = []
        discarded: list[str] = []
        cacheable = True
        model_name: str | None = None

        # Only articles whose source allows it are sent, and only the fields it allows.
        eligible = [s for s in stories if s.model_use != "none"]
        if withheld := len(stories) - len(eligible):
            uncertainties.append(
                f"{withheld} article(s) come from sources configured without AI processing, "
                "so they are shown but not classified."
            )
        if eligible and self.model is None:
            uncertainties.append("No sentiment model is configured, so articles are unclassified.")
        elif eligible and self.model is not None:
            model_name = self.model.name
            subject = asset or "the overall crypto market"
            inputs = [
                ArticleInput(
                    id=s.id,
                    title=s.title,
                    source=s.source,
                    published_at=s.published_at,
                    summary=s.summary if s.model_use == "description" else None,
                    scope=s.scope,
                )
                for s in eligible
            ]
            try:
                assessments = await self.model.classify(subject, inputs)
            except SentimentModelError as exc:
                uncertainties.append(f"Article sentiment could not be classified: {exc}.")
                cacheable = False  # retry the classification next time
            else:
                # Labels can only attach to articles that were actually sent.
                labeled, discarded = apply_assessments(eligible, assessments)
                by_id = {s.id: s for s in labeled}
                stories = [by_id.get(s.id, s) for s in stories]
                if missing := [s.id for s in labeled if s.sentiment is None]:
                    uncertainties.append(
                        f"{len(missing)} article(s) were not classified by the sentiment model."
                    )

        asset_stories = [s for s in stories if s.scope == "asset"]
        market_stories = [s for s in stories if s.scope == "market"]
        focus = asset_stories if asset is not None else market_stories
        uncertainties += _coverage_notes(asset, batch, stories, asset_stories)

        counts = {k: 0 for k in ("bullish", "bearish", "neutral", "mixed", "unclassified")}
        for s in focus:
            counts[s.sentiment or "unclassified"] += 1

        report = NewsReport(
            asset=asset,
            provider=batch.provider,
            sources=batch.sources,
            failed_sources=batch.failed_sources,
            retrieved_at=batch.fetched_at,
            analyzed_at=now,
            model=model_name,
            stories=stories,
            asset_story_count=len(asset_stories),
            overall_sentiment=aggregate_sentiment(focus),
            market_news_sentiment=aggregate_sentiment(market_stories),
            sentiment_basis=sentiment_basis(focus),
            market_sentiment_basis=sentiment_basis(market_stories, "Market-wide sentiment"),
            sentiment_counts=counts,
            conflicts=find_conflicts(focus),
            uncertainties=uncertainties,
            excluded=excluded,
            discarded=discarded,
        )
        return report, cacheable


def _coverage_notes(
    asset: str | None, batch: NewsBatch, stories: list[Story], asset_stories: list[Story]
) -> list[str]:
    notes: list[str] = []
    subject = asset or "the crypto market"
    if not stories:
        notes.append(
            f"No recent relevant news about {subject} was found in "
            f"{', '.join(batch.sources) or 'the configured sources'}."
        )
    elif asset is not None and not asset_stories:
        notes.append(
            f"No recent articles specifically about {asset} were found; only market-wide "
            "stories are shown."
        )
    elif len([s for s in stories if not s.stale]) < 2:
        notes.append(f"Very little recent coverage of {subject}; the news picture is thin.")
    if any(s.stale for s in stories):
        notes.append(f"Some stories are older than {STALE_HOURS:g} hours and may be outdated.")
    if batch.failed_sources:
        failed = ", ".join(f"{f['source']} ({f['reason']})" for f in batch.failed_sources)
        notes.append(f"Some news sources were unavailable: {failed}.")
    return notes
