"""News from crypto publishers' public RSS/Atom feeds (no account or API key needed).

Feeds carry headlines, links, publication times and short descriptions but no reliable
per-coin filtering, so this provider returns everything (`filters_by_asset = False`) and
`NewsService` selects relevant stories. One fetch therefore serves every asset.

What is read: only the feed document itself. From each item UpScale keeps the headline,
link, publication time, the publisher's short description and its category tags; other
fields (full-text `content:encoded`, images, authors) are ignored, and article pages are
never fetched. An item missing a title, an http(s) link or a timezone-aware publication
time is skipped, never repaired. One feed failing doesn't fail the others.

What the sentiment model may see is set per feed (`FeedSource.model_use`); by default it
is the headline, publisher name and publication time only.

Source policy for the defaults (terms reviewed 2026-09-24; not legal advice):
- CoinDesk (Terms of Use effective 2025-11-14) and Decrypt (Terms of Service last updated
  2020-10-12) limit use to personal, non-commercial purposes and restrict automated
  collection. Neither addresses AI processing. They are defaults for personal, local use
  only; no permission for commercial or AI use is claimed.
- Cointelegraph and The Block are not defaults: Cointelegraph's terms prohibit using its
  content for the "operation of artificial intelligence systems ... large language models"
  without written consent, and The Block's terms prohibit processing its content with AI/ML.
"""

import asyncio
import html
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import get_args
from urllib.parse import urlsplit

import httpx2

from upscale.services.news import (
    ModelUse,
    NewsArticle,
    NewsBatch,
    NewsUnavailableError,
    tag_tickers,
)


@dataclass(frozen=True)
class FeedSource:
    name: str
    url: str
    # What the sentiment model may see from this feed's articles (see `ModelUse`).
    model_use: ModelUse = "headline"
    enabled: bool = True


DEFAULT_FEEDS: tuple[FeedSource, ...] = (
    FeedSource("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss"),
    FeedSource("Decrypt", "https://decrypt.co/feed"),
)

MAX_FEED_BYTES = 3 * 1024 * 1024
MAX_SUMMARY_CHARS = 400
USER_AGENT = "UpScale/0.1 (local crypto analysis app; RSS reader)"
ATOM = "{http://www.w3.org/2005/Atom}"
_MODEL_USES: dict[str, ModelUse] = {use: use for use in get_args(ModelUse)}


def parse_feed_setting(value: str | None) -> tuple[FeedSource, ...]:
    """Feeds from `UPSCALE_NEWS_FEEDS`, else the defaults.

    Format: "Name|https://url[|model_use],..." where model_use is none, headline (the
    default) or description.
    """
    if not value or not value.strip():
        return DEFAULT_FEEDS
    feeds: list[FeedSource] = []
    for entry in value.split(","):
        if not entry.strip():
            continue
        parts = [part.strip() for part in entry.split("|")]
        model_use = _MODEL_USES.get(parts[2] if len(parts) == 3 else "headline")
        if len(parts) not in (2, 3) or not parts[0] or not _is_http_url(parts[1]) or not model_use:
            raise ValueError(
                f"invalid UPSCALE_NEWS_FEEDS entry {entry.strip()!r}; expected "
                f"Name|https://url or Name|https://url|{'/'.join(_MODEL_USES)}"
            )
        feeds.append(FeedSource(parts[0], parts[1], model_use))
    return tuple(feeds)


def configured_feeds(feeds: str | None, disabled: str | None) -> tuple[FeedSource, ...]:
    """Feeds from `UPSCALE_NEWS_FEEDS`, minus those named in `UPSCALE_NEWS_DISABLED_FEEDS`
    (comma-separated, case-insensitive)."""
    sources = parse_feed_setting(feeds)
    names = {n.strip().lower() for n in (disabled or "").split(",") if n.strip()}
    if unknown := names - {f.name.lower() for f in sources}:
        known = ", ".join(f.name for f in sources)
        raise ValueError(
            f"UPSCALE_NEWS_DISABLED_FEEDS names unknown feed(s) {sorted(unknown)}; "
            f"configured feeds: {known}"
        )
    return tuple(replace(f, enabled=f.name.lower() not in names) for f in sources)


class FeedError(Exception):
    """One feed could not be used."""


class RssNewsProvider:
    name = "Publisher RSS feeds"
    filters_by_asset = False

    def __init__(
        self,
        feeds: Sequence[FeedSource] = DEFAULT_FEEDS,
        timeout: float = 8.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self.feeds = tuple(f for f in feeds if f.enabled)
        self.timeout = timeout
        self._transport = transport  # injectable for tests

    async def fetch_articles(self, symbols: Sequence[str]) -> NewsBatch:
        if not self.feeds:
            raise NewsUnavailableError("no news sources are enabled")
        async with httpx2.AsyncClient(
            headers={
                "user-agent": USER_AGENT,
                "accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            },
            timeout=self.timeout,
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            outcomes = await asyncio.gather(
                *(self._fetch_feed(client, feed) for feed in self.feeds), return_exceptions=True
            )

        articles: list[NewsArticle] = []
        sources: list[str] = []
        failed: list[dict[str, str]] = []
        skipped = 0
        for feed, outcome in zip(self.feeds, outcomes, strict=True):
            if isinstance(outcome, FeedError):
                failed.append({"source": feed.name, "reason": str(outcome)})
            elif isinstance(outcome, BaseException):
                raise outcome  # unexpected bug: surface it
            else:
                items, bad = outcome
                articles += items
                skipped += bad
                sources.append(feed.name)

        if not sources:
            reasons = "; ".join(f"{f['source']}: {f['reason']}" for f in failed)
            raise NewsUnavailableError(f"no news source could be read ({reasons})")
        return NewsBatch(
            provider=self.name,
            sources=sources,
            failed_sources=failed,
            articles=articles,
            skipped_items=skipped,
            fetched_at=datetime.now(UTC),
        )

    async def _fetch_feed(
        self, client: httpx2.AsyncClient, feed: FeedSource
    ) -> tuple[list[NewsArticle], int]:
        try:
            response = await client.get(feed.url)
        except httpx2.TimeoutException as exc:
            raise FeedError("timed out") from exc
        except httpx2.HTTPError as exc:
            raise FeedError("could not be reached") from exc
        if response.status_code == 429:
            raise FeedError("rate limited")
        if response.status_code != 200:
            raise FeedError(f"HTTP {response.status_code}")
        if len(response.content) > MAX_FEED_BYTES:
            raise FeedError("response too large")
        return parse_feed(feed.name, response.content, feed.model_use)


def parse_feed(
    source: str, content: bytes, model_use: ModelUse = "headline"
) -> tuple[list[NewsArticle], int]:
    """Articles in an RSS 2.0 or Atom document, plus the number of unusable items skipped."""
    # Feeds never need a DTD; refusing them rules out entity-expansion tricks.
    if b"<!DOCTYPE" in content[:4096].upper() or b"<!ENTITY" in content.upper():
        raise FeedError("returned a document with a DTD")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise FeedError("returned malformed XML") from exc

    if root.tag == "rss":
        items = root.findall("./channel/item")
        parsed = [_rss_item(source, item) for item in items]
    elif root.tag == f"{ATOM}feed":
        items = root.findall(f"./{ATOM}entry")
        parsed = [_atom_entry(source, entry) for entry in items]
    else:
        raise FeedError("did not return an RSS or Atom feed")

    articles = [a.model_copy(update={"model_use": model_use}) for a in parsed if a is not None]
    if items and not articles:
        raise FeedError("returned no usable articles")
    return articles, len(items) - len(articles)


def _rss_item(source: str, item: ET.Element) -> NewsArticle | None:
    categories = [_clean(c.text) for c in item.findall("category") if _clean(c.text)]
    return _article(
        source,
        title=item.findtext("title"),
        url=item.findtext("link"),
        published=_rfc822(item.findtext("pubDate")),
        summary=item.findtext("description"),
        categories=categories,
    )


def _atom_entry(source: str, entry: ET.Element) -> NewsArticle | None:
    links = entry.findall(f"{ATOM}link")
    link = next((lk.get("href") for lk in links if lk.get("rel", "alternate") == "alternate"), None)
    categories = [
        _clean(c.get("term")) for c in entry.findall(f"{ATOM}category") if _clean(c.get("term"))
    ]
    return _article(
        source,
        title=entry.findtext(f"{ATOM}title"),
        url=link,
        published=_iso(entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated")),
        summary=entry.findtext(f"{ATOM}summary"),
        categories=categories,
    )


def _article(
    source: str,
    title: str | None,
    url: str | None,
    published: datetime | None,
    summary: str | None,
    categories: list[str],
) -> NewsArticle | None:
    title, url = _clean(title), (url or "").strip()
    if not title or not _is_http_url(url) or published is None:
        return None
    return NewsArticle(
        title=title,
        source=source,
        url=url,
        published_at=published,
        summary=_truncate(_clean(summary)) or None,
        tickers=tag_tickers(categories),
        categories=categories,
    )


def _clean(text: str | None) -> str:
    """Plain text from a feed field that may contain HTML markup and entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", " ", text)
    return " ".join(html.unescape(text).split())


def _truncate(text: str) -> str:
    if len(text) <= MAX_SUMMARY_CHARS:
        return text
    cut = text[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:.") + "…"


def _is_http_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _aware_utc(value: datetime) -> datetime | None:
    # A time without a timezone can't be placed reliably, so it's treated as missing.
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(UTC)


def _rfc822(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    try:
        parsed = parsedate_to_datetime(value.strip())
    except (TypeError, ValueError, IndexError):
        return None
    # "-0000" means "timezone unknown" in RFC 2822; Python returns it as naive.
    return _aware_utc(parsed)


def _iso(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    try:
        return _aware_utc(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except ValueError:
        return None
