"""NewsService: relevance, de-duplication, recency, sentiment, caching and failures."""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx2
import pytest

from upscale.services.news import (
    NewsArticle,
    NewsReport,
    NewsService,
    NewsUnavailableError,
    Story,
    aggregate_sentiment,
    dedupe,
    relevance,
    sentiment_basis,
)
from upscale.services.news_sentiment_model import ArticleAssessment, SentimentModelError
from upscale.services.rss_news import FeedSource, RssNewsProvider

from .conftest import FEED_URLS, FakeSentimentModel, news_item


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def model() -> FakeSentimentModel:
    return FakeSentimentModel()


@pytest.fixture
def service(fake_news, model, clock) -> NewsService:
    return NewsService(RssNewsProvider(transport=fake_news.transport()), model, clock=clock)


def report(service: NewsService, asset: str | None) -> NewsReport:
    return asyncio.run(service.get_report(asset))


def titles(r: NewsReport) -> list[str]:
    return [s.title for s in r.stories]


def article(title: str, summary: str | None = None, tickers=(), hours_ago=1.0, **kw) -> NewsArticle:
    return NewsArticle(
        title=title,
        source=kw.get("source", "CoinDesk"),
        url=kw.get("url", f"https://example.com/{abs(hash(title))}"),
        published_at=datetime.now(UTC) - timedelta(hours=hours_ago),
        summary=summary,
        tickers=list(tickers),
        model_use=kw.get("model_use", "headline"),
    )


# --- Relevance ----------------------------------------------------------------------------------


def test_btc_report_contains_only_relevant_recent_stories(service):
    r = report(service, "BTC")
    assert r.asset == "BTC"
    assert titles(r) == [
        # asset-specific, freshest first; the 60h-old whale story is stale so it comes last
        "Bitcoin ETF inflows hit $1B as institutions add exposure",
        "Bitcoin miners face pressure as hashprice drops to yearly low",
        "Bitcoin whale moves 10,000 BTC to an exchange",
        # market-wide stories after asset-specific ones
        "Crypto market liquidations top $500M in 24 hours",
        "SEC delays decision on crypto custody rules",
    ]
    assert [s.scope for s in r.stories] == ["asset"] * 3 + ["market"] * 2
    assert [s.relevance for s in r.stories] == ["headline"] * 3 + ["market"] * 2
    assert [s.id for s in r.stories] == ["n1", "n2", "n3", "n4", "n5"]
    assert r.asset_story_count == 3
    # Unrelated (Meta AI), other-coin (Ethereum, Solana) and 10-day-old stories are left out.
    assert not {"Meta launches AI keychain gadget", "Bitcoin hits record high"} & set(titles(r))
    assert any("older than 7 days" in e for e in r.excluded)


def test_other_assets_get_their_own_stories(service):
    eth = report(service, "ETH")
    assert titles(eth)[0] == "Ethereum developers set date for next network upgrade"
    assert eth.stories[0].relevance == "headline"
    # Bitcoin-ETF and liquidation headlines count as market-wide context for ETH ...
    assert {s.scope for s in eth.stories[1:]} == {"market"}
    # ... but a Bitcoin-miner story with no market-wide topic does not.
    assert "Bitcoin miners face pressure as hashprice drops to yearly low" not in titles(eth)

    sol = report(service, "SOL")
    assert titles(sol)[0] == "Solana network suffers brief outage"
    assert sol.asset_story_count == 1


def test_asset_without_specific_news_gets_market_stories_only(service):
    r = report(service, "DOGE")
    assert r.asset_story_count == 0
    assert {s.scope for s in r.stories} == {"market"}
    assert r.overall_sentiment == "insufficient_data"  # no DOGE-specific evidence
    assert r.market_news_sentiment != "insufficient_data"
    assert any("No recent articles specifically about DOGE" in u for u in r.uncertainties)


def test_general_market_report_without_an_asset(service):
    r = report(service, None)
    assert r.asset is None
    assert {s.scope for s in r.stories} == {"market"}
    assert "Crypto market liquidations top $500M in 24 hours" in titles(r)
    assert "Meta launches AI keychain gadget" not in titles(r)
    assert r.overall_sentiment == r.market_news_sentiment


def test_relevance_rules():
    assert relevance(article("Bitcoin ETF sees inflows"), "BTC") == ("asset", "headline")
    assert relevance(article("Miners struggle", tickers=["BTC"]), "BTC") == ("asset", "tagged")
    assert relevance(article("Fund goes onchain", "Issued on Ethereum."), "ETH") == (
        "asset",
        "mentioned",
    )
    assert relevance(article("Fed signals rate cuts"), "SOL") == ("market", "market")
    assert relevance(article("SEC sues exchange"), "ETH") == ("market", "market")
    # A story about another coin isn't market-wide context, even with a market keyword.
    assert relevance(article("XRP ETF approval expected"), "SOL") is None
    assert relevance(article("New AI gadget launched"), "BTC") is None
    # Lookalikes don't count.
    assert relevance(article("Bitcoin Cash hard fork scheduled"), "BTC") is None
    assert relevance(article("Click the link to vote"), "LINK") is None
    assert relevance(article("LINK surges after partnership"), "LINK") == ("asset", "headline")


def test_mentioned_only_stories_are_capped(fake_news, service):
    fake_news.items = {
        "CoinDesk": [
            news_item(title, f"https://c.com/{i}", i + 1, "Built on Ethereum.")
            for i, title in enumerate(
                [
                    "Asset manager tokenizes money market fund",
                    "Game studio launches onchain collectibles",
                    "Insurer pilots parametric policies",
                    "Bank issues digital bond",
                    "Marketplace adds wallet logins",
                ]
            )
        ]
    }
    r = report(service, "ETH")
    assert [s.relevance for s in r.stories] == ["mentioned", "mentioned"]


# --- De-duplication -----------------------------------------------------------------------------


def test_syndicated_duplicates_are_merged(service):
    r = report(service, "BTC")
    [etf] = [s for s in r.stories if "ETF inflows" in s.title]
    # The earliest copy is kept and the other outlet is credited.
    assert etf.source == "CoinDesk"
    assert etf.also_reported_by == ["Decrypt"]
    assert any("Merged 1 duplicate" in e for e in r.excluded)


def test_dedupe_by_url_and_near_identical_titles():
    a = article(
        "Bitcoin ETF inflows hit $1B as institutions pile in", hours_ago=3, url="https://a.com/x"
    )
    b = article(
        "Bitcoin ETF inflows hit $1B as institutions pile in!", hours_ago=2, source="Decrypt"
    )
    c = article("Totally different story about bitcoin", hours_ago=1, url="https://www.a.com/x/")
    d = article("Bitcoin ETF outflows hit $1B", hours_ago=1)
    kept = dedupe([d, c, b, a])
    assert [(k.title, also) for k, also in kept] == [
        ("Bitcoin ETF inflows hit $1B as institutions pile in", ["Decrypt"]),
        ("Bitcoin ETF outflows hit $1B", []),
    ]


def test_independent_write_ups_of_one_event_are_merged():
    kept = dedupe(
        [
            article(
                "Solana Foundation Hires Binance's Former Global CMO for Institutional Push",
                hours_ago=7.4,
                source="Outlet A",
            ),
            article(
                "Solana Foundation hires Binance, Polygon veterans as it ramps up tokenized "
                "finance push",
                hours_ago=7.3,
                source="Outlet B",
            ),
            article(
                "Solana Foundation hires ex-Binance CMO and payments exec as new partnerships "
                "expand",
                hours_ago=7.2,
                source="Outlet C",
            ),
        ]
    )
    [(original, also)] = kept
    assert original.source == "Outlet A" and also == ["Outlet B", "Outlet C"]


def test_similar_headlines_with_opposite_meaning_are_kept_apart():
    kept = dedupe(
        [
            article("Bitcoin ETF inflows hit $1B as institutions add exposure", hours_ago=2),
            article("Bitcoin ETF outflows hit $1B as institutions cut exposure", hours_ago=1),
        ]
    )
    assert len(kept) == 2


def test_same_words_far_apart_in_time_are_different_events():
    kept = dedupe(
        [
            article("Solana Foundation hires Binance executives for payments push", hours_ago=40),
            article(
                "Solana Foundation hires Binance veterans in new institutional push", hours_ago=2
            ),
        ]
    )
    assert len(kept) == 2


def test_market_topics_must_be_in_the_headline():
    """Summaries mention regulators and ETFs in passing, which isn't enough."""
    story = article(
        "Fintech firms form coalition for tokenized stocks",
        "The group follows the U.S. SEC's innovation exemption.",
    )
    assert relevance(story, "BTC") is None


# --- Time ------------------------------------------------------------------------------------------


def test_ages_and_staleness_are_exposed(service):
    r = report(service, "BTC")
    whale = next(s for s in r.stories if "whale" in s.title)
    assert whale.stale is True and 59.5 <= whale.age_hours <= 60.5
    fresh = r.stories[0]
    assert fresh.stale is False and 2.5 <= fresh.age_hours <= 3.5
    assert r.retrieved_at.tzinfo is not None and r.analyzed_at >= r.retrieved_at
    assert any("older than 48 hours" in u for u in r.uncertainties)


def test_future_dated_articles_are_ignored(fake_news, service):
    fake_news.items = {
        "CoinDesk": [
            news_item("Bitcoin story from the future", "https://c.com/f", -5),
            news_item("Bitcoin story", "https://c.com/n", 1),
        ]
    }
    r = report(service, "BTC")
    assert titles(r) == ["Bitcoin story"]
    assert any("in the future" in e for e in r.excluded)


def test_only_stale_news_is_flagged(fake_news, service):
    fake_news.items = {"CoinDesk": [news_item("Bitcoin treasury update", "https://c.com/1", 100)]}
    r = report(service, "BTC")
    assert [s.stale for s in r.stories] == [True]
    assert any("thin" in u for u in r.uncertainties)


# --- No results ------------------------------------------------------------------------------------


def test_no_relevant_articles_is_insufficient_data(fake_news, service, model):
    fake_news.items = {"Decrypt": [news_item("Meta launches AI gadget", "https://d.co/1", 1)]}
    r = report(service, "BTC")
    assert r.stories == []
    assert r.overall_sentiment == r.market_news_sentiment == "insufficient_data"
    assert any("No recent relevant news about BTC" in u for u in r.uncertainties)
    assert model.calls == []  # nothing to classify, so the model isn't asked


def test_empty_feeds_are_insufficient_data_not_an_error(fake_news, service):
    fake_news.items = {}
    r = report(service, "ETH")
    assert r.stories == [] and r.overall_sentiment == "insufficient_data"


# --- Sentiment ---------------------------------------------------------------------------------------


def _all(sentiment, impact="medium"):
    def respond(subject, articles):
        return [
            ArticleAssessment(
                id=a.id, sentiment=sentiment, impact=impact, reason="May be relevant."
            )
            for a in articles
        ]

    return respond


@pytest.mark.parametrize("sentiment", ["bullish", "bearish", "neutral", "mixed"])
def test_uniform_labels_give_that_overall_sentiment(service, model, sentiment):
    model.respond = _all(sentiment)
    r = report(service, "BTC")
    assert r.overall_sentiment == sentiment
    assert r.sentiment_counts[sentiment] == r.asset_story_count == 3


def test_labels_attach_to_the_right_stories(service):
    r = report(service, "BTC")
    by_title = {s.title: s for s in r.stories}
    etf = by_title["Bitcoin ETF inflows hit $1B as institutions add exposure"]
    assert (etf.sentiment, etf.impact) == ("bullish", "high")
    assert etf.impact_reason == "Large ETF inflows may be relevant to demand for BTC."
    miners = by_title["Bitcoin miners face pressure as hashprice drops to yearly low"]
    assert (miners.sentiment, miners.impact) == ("bearish", "medium")


def test_conflicting_news_is_reported_as_mixed(service):
    # Default labels: ETF inflows bullish/high vs miners bearish/medium + stale whale bearish.
    r = report(service, "BTC")
    assert r.overall_sentiment == "mixed"
    [conflict] = r.conflicts
    assert "1 bullish vs 2 bearish" in conflict
    assert "Bitcoin ETF inflows" in conflict and "(CoinDesk)" in conflict


def _story(sentiment, impact="medium", stale=False, scope="asset") -> Story:
    return Story(
        id="n1",
        title="t",
        source="s",
        url="https://x.com",
        published_at=datetime.now(UTC),
        summary=None,
        tickers=[],
        also_reported_by=[],
        scope=scope,
        relevance="headline",
        age_hours=60.0 if stale else 1.0,
        stale=stale,
        model_use="headline",
        sentiment=sentiment,
        impact=impact,
    )


def test_aggregation_weights_impact_and_recency():
    assert aggregate_sentiment([]) == "insufficient_data"
    assert aggregate_sentiment([_story(None, None)]) == "unavailable"
    # One high-impact bullish story outweighs one low-impact bearish one.
    assert aggregate_sentiment([_story("bullish", "high"), _story("bearish", "low")]) == "bullish"
    # Equal weights both ways are mixed.
    assert aggregate_sentiment([_story("bullish"), _story("bearish")]) == "mixed"
    # A single directional story among many neutral ones leaves the picture neutral.
    assert aggregate_sentiment([_story("bearish", "low")] + [_story("neutral")] * 3) == "neutral"
    # Stale stories count half: fresh bearish (3) vs stale bullish (1.5) is 2:1 -> bearish.
    assert (
        aggregate_sentiment([_story("bearish", "high"), _story("bullish", "high", stale=True)])
        == "bearish"
    )


def test_sentiment_basis_explains_weighted_outcomes():
    # Even count, but the bearish side carries more impact: explained without numbers.
    even = (
        [_story("bullish", "low")] * 2 + [_story("bearish", "high")] * 2 + [_story("neutral")] * 2
    )
    assert aggregate_sentiment(even) == "bearish"
    assert sentiment_basis(even) == (
        "Overall sentiment is bearish because bearish coverage carries more impact weight "
        "despite an even bullish/bearish article count."
    )
    # Stale articles count less, so a count majority of stale stories can be evened out.
    fresh = [_story("bullish")] + [_story("bearish", stale=True)] * 2
    assert sentiment_basis(fresh) == (
        "Overall sentiment is mixed rather than bearish (as the article count alone would "
        "suggest) because stale stories count less."
    )
    outnumbered = [_story("bullish", "high")] * 2 + [_story("bearish", "low")] * 3
    assert "despite more bearish than bullish articles" in (sentiment_basis(outnumbered) or "")
    # Weighting pulls a count-directional picture back to mixed.
    evened = [_story("bullish")] * 2 + [_story("bearish", "high")]
    assert sentiment_basis(evened) == (
        "Overall sentiment is mixed rather than bullish (as the article count alone would "
        "suggest) because stories are weighted by potential impact."
    )
    # No explanation when the count already tells the same story, or with no data.
    assert sentiment_basis([_story("bullish"), _story("bearish")]) is None
    assert sentiment_basis([]) is None
    assert sentiment_basis([_story(None, None)]) is None
    assert sentiment_basis(even, "Market-wide sentiment").startswith("Market-wide sentiment is")


def test_model_only_sees_retrieved_article_data(service, model):
    r = report(service, "BTC")
    [(subject, inputs)] = model.calls
    assert subject == "BTC"
    assert [a.id for a in inputs] == [s.id for s in r.stories]
    for a, s in zip(inputs, r.stories, strict=True):
        assert set(a.model_dump()) == {"id", "title", "source", "published_at", "summary", "scope"}
        assert (a.title, a.source, a.published_at) == (s.title, s.source, s.published_at)
        # Default feeds send headlines only: the feed description stays out of the prompt.
        assert a.summary is None and s.summary is not None


def _service_with(fake_news, model, clock, **policies) -> NewsService:
    feeds = [
        FeedSource(name, FEED_URLS[name], policies.get(name.lower(), "headline"))
        for name in ("CoinDesk", "Decrypt")
    ]
    return NewsService(RssNewsProvider(feeds, transport=fake_news.transport()), model, clock=clock)


def test_description_is_sent_only_for_feeds_that_allow_it(fake_news, model, clock):
    service = _service_with(fake_news, model, clock, coindesk="description")
    report(service, "BTC")
    [(_, inputs)] = model.calls
    by_title = {a.title: a for a in inputs}
    etf = by_title["Bitcoin ETF inflows hit $1B as institutions add exposure"]  # CoinDesk copy
    assert etf.summary == "Spot bitcoin ETFs recorded their largest daily inflow in months."
    miners = by_title["Bitcoin miners face pressure as hashprice drops to yearly low"]  # Decrypt
    assert miners.summary is None


def test_feeds_without_ai_processing_are_shown_but_never_sent(fake_news, model, clock):
    service = _service_with(fake_news, model, clock, decrypt="none")

    def respond(subject, articles):  # also tries to label a withheld article
        labels = [
            ArticleAssessment(id=a.id, sentiment="bullish", impact="low", reason="May matter.")
            for a in articles
        ]
        return labels + [
            ArticleAssessment(id="n2", sentiment="bearish", impact="high", reason="Sneaky.")
        ]

    model.respond = respond
    r = report(service, "BTC")
    [(_, inputs)] = model.calls
    sent = {a.title for a in inputs}
    decrypt = [s for s in r.stories if s.source == "Decrypt"]
    assert decrypt and not {s.title for s in decrypt} & sent
    assert all(s.sentiment is None and s.model_use == "none" for s in decrypt)
    assert all(s.sentiment == "bullish" for s in r.stories if s.source == "CoinDesk")
    assert any("unknown article id 'n2'" in d for d in r.discarded)
    assert any("configured without AI processing" in u for u in r.uncertainties)


def test_no_model_call_when_no_source_allows_it(fake_news, model, clock):
    service = _service_with(fake_news, model, clock, coindesk="none", decrypt="none")
    r = report(service, "BTC")
    assert r.stories and model.calls == []
    assert r.overall_sentiment == "unavailable"


def test_model_cannot_add_or_rename_articles(service, model, fake_news):
    def respond(subject, articles):
        first = articles[0]
        return [
            ArticleAssessment(
                id=first.id, sentiment="bullish", impact="high", reason="May matter."
            ),
            ArticleAssessment(id=first.id, sentiment="bearish", impact="low", reason="Repeat."),
            ArticleAssessment(id="n99", sentiment="bullish", impact="high", reason="Invented."),
        ]

    model.respond = respond
    r = report(service, "BTC")
    assert len(r.stories) == 5
    assert {s.title for s in r.stories} <= fake_news.all_titles()
    assert {s.url for s in r.stories} <= fake_news.all_urls()
    assert r.stories[0].sentiment == "bullish"  # the first label wins
    assert any("unknown article id 'n99'" in d for d in r.discarded)
    assert any("repeated sentiment label" in d for d in r.discarded)
    assert all(s.sentiment is None for s in r.stories[1:])
    assert any("4 article(s) were not classified" in u for u in r.uncertainties)


@pytest.mark.parametrize(
    "reason",
    [
        "This news caused the rally.",
        "BTC will surge after this.",
        "Traders should buy now.",
        "This guarantees higher prices.",
    ],
)
def test_overclaiming_reasons_are_dropped(service, model, reason):
    model.respond = lambda s, arts: [
        ArticleAssessment(id=a.id, sentiment="bullish", impact="medium", reason=reason)
        for a in arts
    ]
    r = report(service, "BTC")
    assert all(s.impact_reason is None and s.sentiment == "bullish" for s in r.stories)
    assert any("asserted causation or a price prediction" in d for d in r.discarded)


def test_model_failure_keeps_articles_unclassified(service, model):
    model.error = SentimentModelError("the sentiment model timed out")
    r = report(service, "BTC")
    assert len(r.stories) == 5
    assert all(s.sentiment is None for s in r.stories)
    assert r.overall_sentiment == "unavailable"
    assert any(
        "could not be classified: the sentiment model timed out" in u for u in r.uncertainties
    )


def test_without_a_model_articles_are_unclassified(fake_news, clock):
    service = NewsService(RssNewsProvider(transport=fake_news.transport()), None, clock=clock)
    r = report(service, "BTC")
    assert r.model is None and r.overall_sentiment == "unavailable"
    assert any("No sentiment model is configured" in u for u in r.uncertainties)


# --- Caching and rate limits -------------------------------------------------------------------------


def test_reports_and_feeds_are_cached(service, model, fake_news, clock):
    report(service, "BTC")
    report(service, "BTC")
    assert len(fake_news.requests) == 2  # one request per feed
    assert len(model.calls) == 1

    clock.now += 301  # past the 5-minute cache
    report(service, "BTC")
    assert len(fake_news.requests) == 4
    assert len(model.calls) == 2


def test_one_fetch_serves_every_asset(service, fake_news, model):
    report(service, "BTC")
    report(service, "ETH")
    report(service, None)
    assert len(fake_news.requests) == 2
    assert [subject for subject, _ in model.calls] == ["BTC", "ETH", "the overall crypto market"]


def test_concurrent_requests_are_deduplicated(service, fake_news, model):
    model.delay = 0.01

    async def many():
        return await asyncio.gather(*(service.get_report(a) for a in ["BTC"] * 5 + ["ETH"] * 3))

    results = asyncio.run(many())
    assert len(fake_news.requests) == 2  # a single provider fetch
    assert sorted(s for s, _ in model.calls) == ["BTC", "ETH"]  # one classification per asset
    assert all(r is results[0] for r in results[:5])


def test_provider_failures_are_not_cached(service, fake_news):
    fake_news.fail_all(httpx2.Response(503))
    with pytest.raises(NewsUnavailableError):
        report(service, "BTC")
    fake_news.responses.clear()
    assert report(service, "BTC").stories  # retried immediately, not served from cache


def test_model_failures_are_not_cached(service, model):
    model.error = SentimentModelError("the sentiment model is rate limited")
    assert report(service, "BTC").overall_sentiment == "unavailable"
    model.error = None
    assert report(service, "BTC").overall_sentiment == "mixed"
    assert len(model.calls) == 2
    report(service, "BTC")
    assert len(model.calls) == 2  # the successful result is cached


def test_partial_results_are_cached_briefly(service, fake_news, clock):
    fake_news.responses["Decrypt"] = httpx2.Response(503)
    r = report(service, "BTC")
    assert r.failed_sources == [{"source": "Decrypt", "reason": "HTTP 503"}]
    assert any("Decrypt (HTTP 503)" in u for u in r.uncertainties)
    report(service, "BTC")
    assert len(fake_news.requests) == 2
    clock.now += 61  # partial batches expire after a minute so the failed feed is retried
    fake_news.responses.clear()
    assert report(service, "BTC").failed_sources == []
    assert len(fake_news.requests) == 4


def test_rate_limit_stops_excess_provider_calls(fake_news, model, clock):
    service = NewsService(
        RssNewsProvider(transport=fake_news.transport()),
        model,
        cache_ttl=0,
        max_calls_per_minute=1,
        clock=clock,
    )
    report(service, "BTC")
    with pytest.raises(NewsUnavailableError, match="request limit"):
        report(service, "BTC")
    assert len(fake_news.requests) == 2
    clock.now += 61
    assert report(service, "BTC").stories


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (httpx2.ReadTimeout("slow"), "timed out"),
        (httpx2.Response(429), "rate limited"),
        (httpx2.Response(200, content=b"{not xml"), "malformed XML"),
    ],
    ids=["timeout", "rate-limited", "malformed"],
)
def test_provider_failure_raises_without_substitutes(service, fake_news, failure, message):
    fake_news.fail_all(failure)
    with pytest.raises(NewsUnavailableError, match=message):
        report(service, "BTC")
