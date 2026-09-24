"""RssNewsProvider: feed parsing, normalization and per-feed failure handling (no network)."""

import asyncio
from datetime import UTC

import httpx2
import pytest

from upscale.services.news import NewsBatch, NewsUnavailableError
from upscale.services.rss_news import (
    DEFAULT_FEEDS,
    FeedError,
    FeedSource,
    RssNewsProvider,
    configured_feeds,
    parse_feed,
    parse_feed_setting,
)

from .conftest import FakeNewsFeeds, news_item, rss_feed


def fetch(fake: FakeNewsFeeds) -> NewsBatch:
    return asyncio.run(RssNewsProvider(transport=fake.transport()).fetch_articles([]))


def by_title(batch: NewsBatch, title: str, source: str):
    return next(a for a in batch.articles if a.title == title and a.source == source)


# --- Normalization ------------------------------------------------------------------------------


def test_reads_every_default_feed(fake_news):
    batch = fetch(fake_news)
    assert batch.provider == "Publisher RSS feeds"
    assert batch.sources == ["CoinDesk", "Decrypt"]
    assert batch.failed_sources == []
    assert len(batch.articles) == 10
    # Only the feed documents are requested, never the article pages they link to.
    assert sorted(str(r.url) for r in fake_news.requests) == sorted(f.url for f in DEFAULT_FEEDS)
    assert not {str(r.url) for r in fake_news.requests} & fake_news.all_urls()


def test_default_sources_exclude_publishers_that_prohibit_ai_processing():
    names = [f.name for f in DEFAULT_FEEDS]
    assert names == ["CoinDesk", "Decrypt"]
    assert not any("theblock" in f.url or "cointelegraph" in f.url for f in DEFAULT_FEEDS)
    # Defaults send the model headlines only, never the feed descriptions.
    assert {f.model_use for f in DEFAULT_FEEDS} == {"headline"}
    assert all(f.enabled for f in DEFAULT_FEEDS)


def test_articles_are_normalized_without_inventing_anything(fake_news):
    batch = fetch(fake_news)
    etf = by_title(batch, "Bitcoin ETF inflows hit $1B as institutions add exposure", "CoinDesk")
    assert etf.url == "https://www.coindesk.com/markets/2026/09/24/bitcoin-etf-inflows-hit-1b"
    assert etf.summary == "Spot bitcoin ETFs recorded their largest daily inflow in months."
    assert etf.published_at.tzinfo == UTC
    assert etf.categories == ["Markets", "Bitcoin"]
    assert etf.tickers == ["BTC"]  # from the publisher's own "Bitcoin" tag

    # HTML and entities are stripped from descriptions; the link is kept exactly.
    dc = by_title(batch, "Bitcoin ETF inflows hit $1B as institutions add exposure", "Decrypt")
    assert dc.summary == "Spot bitcoin ETFs saw inflows & rising volume."
    assert dc.url == "https://decrypt.co/379300/bitcoin-etf-inflows-1b?utm_source=rss"
    assert dc.tickers == []  # "Markets" names no asset; nothing is guessed
    assert {a.model_use for a in batch.articles} == {"headline"}

    assert {a.url for a in batch.articles} <= fake_news.all_urls()
    assert {a.title for a in batch.articles} <= fake_news.all_titles()


def test_items_missing_required_fields_are_skipped_not_repaired(fake_news):
    fake_news.items = {
        "CoinDesk": [
            news_item("Valid Bitcoin story", "https://coindesk.com/a", 1),
            news_item(None, "https://coindesk.com/b", 1),  # no title
            news_item("No link", None, 1),
            news_item("Bad link", "javascript:alert(1)", 1),
            news_item("No date", "https://coindesk.com/c", None),
            news_item("Garbage date", "https://coindesk.com/d", None, pub_date="yesterday"),
            news_item(  # RFC 2822 "-0000" means the timezone is unknown
                "Unknown zone",
                "https://coindesk.com/e",
                None,
                pub_date="Thu, 24 Sep 2026 10:00:00 -0000",
            ),
        ]
    }
    batch = fetch(fake_news)
    assert [a.title for a in batch.articles] == ["Valid Bitcoin story"]
    assert batch.skipped_items == 6


def test_missing_description_stays_none(fake_news):
    fake_news.items = {"CoinDesk": [news_item("Bitcoin story", "https://coindesk.com/a", 1)]}
    [article] = fetch(fake_news).articles
    assert article.summary is None


def test_long_descriptions_are_truncated_not_rewritten():
    long_text = "word " * 200
    xml = rss_feed([news_item("Bitcoin story", "https://x.co/a", 1, long_text)])
    [article], _ = parse_feed("X", xml)
    assert len(article.summary) <= 401 and article.summary.endswith("…")
    assert long_text.startswith(article.summary[:-1])


def test_atom_feeds_are_supported():
    xml = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Ethereum staking hits record</title>
        <link rel="alternate" href="https://example.com/eth-staking"/>
        <published>2026-09-24T10:00:00Z</published>
        <summary>More ETH is staked than ever.</summary>
        <category term="Ethereum"/>
      </entry>
    </feed>"""
    [article], skipped = parse_feed("Example", xml)
    assert (article.title, article.url, article.tickers) == (
        "Ethereum staking hits record",
        "https://example.com/eth-staking",
        ["ETH"],
    )
    assert article.published_at.isoformat() == "2026-09-24T10:00:00+00:00"
    assert skipped == 0


def test_redirects_are_followed(fake_news):
    """CoinDesk's feed URL answers with a redirect in practice."""
    feeds = [FeedSource("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/")]
    moved = httpx2.Response(308, headers={"location": DEFAULT_FEEDS[0].url})
    inner = fake_news.transport()

    def handle(request):
        if str(request.url).endswith("/rss/"):
            return moved
        return inner.handle_request(request)

    provider = RssNewsProvider(feeds=feeds, transport=httpx2.MockTransport(handle))
    batch = asyncio.run(provider.fetch_articles([]))
    assert batch.sources == ["CoinDesk"] and len(batch.articles) == 5


def test_only_feed_level_fields_are_read():
    """Full-text and media fields in a feed item are ignored; only headline, link, time,
    the short description and category tags are kept."""
    xml = b"""<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"><channel><item>
        <title>Bitcoin story</title>
        <link>https://x.co/a</link>
        <pubDate>Thu, 24 Sep 2026 10:00:00 +0000</pubDate>
        <description>Short teaser.</description>
        <content:encoded>FULL ARTICLE BODY that must not be kept</content:encoded>
        <dc:creator>Some Author</dc:creator>
        <category>Bitcoin</category>
    </item></channel></rss>"""
    [article], _ = parse_feed("X", xml)
    assert set(article.model_dump()) == {
        "title",
        "source",
        "url",
        "published_at",
        "summary",
        "tickers",
        "categories",
        "model_use",
    }
    assert "FULL ARTICLE BODY" not in article.model_dump_json()
    assert "Some Author" not in article.model_dump_json()


@pytest.mark.parametrize("model_use", ["none", "headline", "description"])
def test_feed_policy_is_stamped_on_every_article(fake_news, model_use):
    feeds = [FeedSource("CoinDesk", DEFAULT_FEEDS[0].url, model_use)]
    provider = RssNewsProvider(feeds=feeds, transport=fake_news.transport())
    batch = asyncio.run(provider.fetch_articles([]))
    assert {a.model_use for a in batch.articles} == {model_use}


def test_disabled_feeds_are_not_requested(fake_news):
    feeds = configured_feeds(None, "decrypt")
    assert [(f.name, f.enabled) for f in feeds] == [("CoinDesk", True), ("Decrypt", False)]
    batch = asyncio.run(RssNewsProvider(feeds, transport=fake_news.transport()).fetch_articles([]))
    assert batch.sources == ["CoinDesk"]
    assert [str(r.url) for r in fake_news.requests] == [DEFAULT_FEEDS[0].url]


def test_all_feeds_disabled_is_unavailable(fake_news):
    feeds = configured_feeds(None, "CoinDesk, Decrypt")
    provider = RssNewsProvider(feeds, transport=fake_news.transport())
    with pytest.raises(NewsUnavailableError, match="no news sources are enabled"):
        asyncio.run(provider.fetch_articles([]))
    assert fake_news.requests == []


def test_disabling_an_unknown_feed_is_a_config_error():
    with pytest.raises(ValueError, match="unknown feed"):
        configured_feeds(None, "The Block")


# --- Malformed data ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"<rss><channel><item>", "malformed XML"),
        (b"<html><body>Not a feed</body></html>", "did not return an RSS or Atom feed"),
        (b'<!DOCTYPE rss [<!ENTITY a "aaaa">]><rss/>', "DTD"),
        (
            b"<rss><channel><item><title>No link or date</title></item></channel></rss>",
            "no usable articles",
        ),
        (b"", "malformed XML"),
    ],
    ids=["broken-xml", "html", "dtd", "no-usable-items", "empty"],
)
def test_malformed_feed_fails_only_that_source(fake_news, content, reason):
    fake_news.responses["CoinDesk"] = httpx2.Response(200, content=content)
    batch = fetch(fake_news)
    assert batch.sources == ["Decrypt"]
    [failure] = batch.failed_sources
    assert failure["source"] == "CoinDesk" and reason in failure["reason"]
    assert all(a.source != "CoinDesk" for a in batch.articles)


def test_empty_feed_is_not_an_error():
    articles, skipped = parse_feed("X", rss_feed([]))
    assert (articles, skipped) == ([], 0)


def test_parse_feed_rejects_entity_declarations_anywhere():
    with pytest.raises(FeedError, match="DTD"):
        parse_feed("X", b"<rss>" + b" " * 5000 + b"<!ENTITY x 'y'></rss>")


# --- Provider failures -----------------------------------------------------------------------------


def _raise(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (httpx2.Response(429), "rate limited"),
        (httpx2.Response(503), "HTTP 503"),
        (httpx2.Response(403), "HTTP 403"),
        (httpx2.ReadTimeout("slow"), "timed out"),
        (httpx2.ConnectError("down"), "could not be reached"),
    ],
    ids=["429", "503", "403", "timeout", "connect"],
)
def test_one_failing_feed_does_not_fail_the_others(fake_news, failure, reason):
    fake_news.responses["Decrypt"] = failure
    batch = fetch(fake_news)
    assert batch.failed_sources == [{"source": "Decrypt", "reason": reason}]
    assert batch.sources == ["CoinDesk"] and batch.articles


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (httpx2.Response(429), "rate limited"),
        (httpx2.ReadTimeout("slow"), "timed out"),
        (httpx2.Response(200, content=b"not xml"), "malformed XML"),
        (httpx2.Response(500), "HTTP 500"),
    ],
    ids=["rate-limited", "timeout", "malformed", "server-error"],
)
def test_all_feeds_failing_raises_unavailable(fake_news, failure, reason):
    fake_news.fail_all(failure)
    with pytest.raises(NewsUnavailableError, match=reason) as info:
        fetch(fake_news)
    assert "no news source could be read" in str(info.value)


def test_oversized_response_is_rejected(fake_news, monkeypatch):
    monkeypatch.setattr("upscale.services.rss_news.MAX_FEED_BYTES", 100)
    fake_news.fail_all(httpx2.Response(200, content=b"<rss>" + b" " * 200 + b"</rss>"))
    with pytest.raises(NewsUnavailableError, match="too large"):
        fetch(fake_news)


def test_requests_identify_upscale(fake_news):
    fetch(fake_news)
    assert all("UpScale" in r.headers["user-agent"] for r in fake_news.requests)


# --- Configuration ----------------------------------------------------------------------------------


def test_feed_setting_defaults_and_overrides():
    assert parse_feed_setting(None) == DEFAULT_FEEDS
    assert parse_feed_setting("  ") == DEFAULT_FEEDS
    assert parse_feed_setting(
        "A|https://a.example/rss, B|http://b.example/feed|none, C|https://c.example|description"
    ) == (
        FeedSource("A", "https://a.example/rss", "headline"),
        FeedSource("B", "http://b.example/feed", "none"),
        FeedSource("C", "https://c.example", "description"),
    )
    custom = configured_feeds("A|https://a.example/rss,B|https://b.example/rss", "b")
    assert [(f.name, f.enabled) for f in custom] == [("A", True), ("B", False)]


@pytest.mark.parametrize(
    "value",
    [
        "https://a.example/rss",
        "A|ftp://x",
        "|https://a.example",
        "A|https://a.example|everything",
        "A|https://a.example|none|extra",
    ],
)
def test_invalid_feed_setting_is_rejected(value):
    with pytest.raises(ValueError, match="UPSCALE_NEWS_FEEDS"):
        parse_feed_setting(value)
