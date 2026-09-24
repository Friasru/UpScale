"""NewsSentimentAgent output, plus orchestrator and routing integration (fake feeds/model)."""

import asyncio

import httpx2
import pytest

from upscale.agents import AgentContext, NewsSentimentAgent
from upscale.orchestrator import Orchestrator
from upscale.routing import route
from upscale.schemas import AgentResult, ChatMessage, ChatRequest, ImageAttachment
from upscale.services.news_sentiment_model import ArticleAssessment, SentimentModelError

from .conftest import CHART_PNG, news_item
from .test_agents import BUY_SELL_WORDS


def run(*assets: str, prior=None) -> AgentResult:
    context = AgentContext(query="news", assets=list(assets), prior_results=prior or {})
    return asyncio.run(NewsSentimentAgent().run(context))


def ask(content: str, images: int = 0):
    attachments = [
        ImageAttachment(name="chart.png", media_type="image/png", data=CHART_PNG)
        for _ in range(images)
    ]
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=content, attachments=attachments)]
    )
    return asyncio.run(Orchestrator().respond(request))


def market_result(symbol: str, change: float) -> AgentResult:
    return AgentResult(
        agent="market",
        mock=False,
        summary="market",
        findings={
            "snapshots": [{"symbol": symbol, "change_24h_pct": change, "provider": "CoinGecko"}]
        },
    )


# --- Agent output ----------------------------------------------------------------------------------


def test_btc_news_is_real_structured_evidence():
    result = run("BTC")
    assert result.status == "ok" and result.mock is False
    assert result.summary.startswith("News for BTC: 5 recent article(s) from")
    assert "overall news sentiment: mixed" in result.summary
    [report] = result.findings["reports"]
    for key in (
        "asset",
        "overall_news_sentiment",
        "stories",
        "bullish_evidence",
        "bearish_evidence",
        "neutral_mixed_evidence",
        "impact_events",
        "conflicting_evidence",
        "uncertainty",
        "retrieved_at",
    ):
        assert key in report
    assert report["asset"] == "BTC"
    assert report["provider"] == "Publisher RSS feeds"
    assert report["sentiment_model"] == "fake-sentiment"

    story = report["stories"][0]
    for key in ("title", "source", "url", "published_at", "age_hours", "age", "stale"):
        assert story[key] is not None
    assert (
        len(report["bullish_evidence"]) == 1
        and "Bitcoin ETF inflows" in report["bullish_evidence"][0]
    )
    assert len(report["bearish_evidence"]) == 3  # miners, whale (stale), liquidations
    assert report["impact_events"][0].startswith("[bullish, high impact]")
    assert report["conflicting_evidence"]


def test_evidence_includes_source_and_publication_time():
    result = run("BTC")
    lines = [e for e in result.evidence if e.startswith("[")]
    assert lines
    for line in lines:
        assert " UTC, " in line and "ago" in line
        assert any(src in line for src in ("CoinDesk", "Decrypt"))
    assert any("also reported by Decrypt" in line for line in lines)
    assert any(", stale)" in line for line in lines)
    assert result.evidence[0].startswith("News sentiment for BTC: mixed")
    assert "not a price forecast" in result.evidence[0]


def test_no_invented_articles(fake_news):
    result = run("BTC", "ETH", "SOL")
    for report in result.findings["reports"]:
        for story in report["stories"]:
            assert story["url"] in fake_news.all_urls()
            assert story["title"] in fake_news.all_titles()


def test_multiple_assets_share_one_fetch(fake_news, fake_sentiment):
    result = run("BTC", "ETH", "SOL")
    assert [r["asset"] for r in result.findings["reports"]] == ["BTC", "ETH", "SOL"]
    assert len(fake_news.requests) == 2
    assert sorted(subject for subject, _ in fake_sentiment.calls) == ["BTC", "ETH", "SOL"]
    assert "News for BTC" in result.summary and "News for SOL" in result.summary


def test_no_asset_reports_the_overall_market():
    result = run()
    [report] = result.findings["reports"]
    assert report["asset"] is None and report["subject"] == "the crypto market"
    assert result.summary.startswith("News for the crypto market")


def test_news_sentiment_is_separate_from_price_movement(fake_sentiment):
    """A sharp price drop doesn't turn uniformly bullish news bearish, and vice versa."""
    fake_sentiment.respond = lambda s, arts: [
        ArticleAssessment(id=a.id, sentiment="bullish", impact="medium", reason="May matter.")
        for a in arts
    ]
    result = run("BTC", prior={"market": market_result("BTC", -12.5)})
    [report] = result.findings["reports"]
    assert report["overall_news_sentiment"] == "bullish"
    assert result.findings["price_context"][0]["change_24h_pct"] == -12.5
    assert "not used to label news sentiment" in result.findings["price_note"]
    assert (
        "Price context (separate from news sentiment): BTC moved -12.50% over 24h (CoinGecko)."
        in result.evidence
    )
    assert any("caused a price move" in r.description for r in result.risks)


def test_output_contains_no_recommendations_or_causal_claims():
    result = run("BTC", "ETH")
    text = " ".join([result.summary, *result.evidence, *(r.description for r in result.risks)])
    # The only mention of buying/selling or causation is UpScale's own disclaimer.
    text = text.lower().replace("news sentiment is not a buy or sell signal", "")
    assert not any(word in text for word in BUY_SELL_WORDS)
    assert "caused" not in text


def test_no_relevant_news_is_insufficient_data(fake_news):
    fake_news.items = {"Decrypt": [news_item("Meta launches AI gadget", "https://d.co/1", 1)]}
    result = run("BTC")
    assert result.status == "ok"
    assert "insufficient data" in result.summary
    [report] = result.findings["reports"]
    assert report["stories"] == [] and report["overall_news_sentiment"] == "insufficient_data"
    assert "no headlines are shown rather than invented" in result.evidence[0]


def test_provider_down_is_a_clear_error(fake_news):
    fake_news.fail_all(httpx2.Response(503))
    result = run("BTC")
    assert result.status == "error" and result.mock is False
    assert "could not be retrieved" in result.summary
    assert "HTTP 503" in result.error
    assert result.findings["reports"] == [] and result.evidence == []


def test_classification_failure_still_reports_real_articles(fake_sentiment):
    fake_sentiment.error = SentimentModelError("the sentiment model timed out")
    result = run("BTC")
    assert result.status == "ok"
    [report] = result.findings["reports"]
    assert report["overall_news_sentiment"] == "unavailable"
    assert all(s["sentiment"] is None for s in report["stories"])
    assert any("unclassified" in e for e in result.evidence)
    assert any("could not be classified" in r.description for r in result.risks)


def test_partial_source_failure_is_disclosed(fake_news):
    fake_news.responses["Decrypt"] = httpx2.Response(429)
    result = run("BTC")
    assert result.status == "ok"
    assert any("Decrypt (rate limited)" in r.description for r in result.risks)


def test_agent_declares_its_dependencies():
    agent = NewsSentimentAgent()
    assert agent.depends_on == ("vision", "market")
    assert agent.timeout and agent.timeout > 30


# --- Orchestrator integration ------------------------------------------------------------------------


def test_news_question_uses_the_real_agent():
    analysis = ask("Any news on bitcoin?").analysis
    assert analysis.agents_used == ["news_sentiment", "risk"]
    news = analysis.agent_results[0]
    assert news.status == "ok" and news.mock is False
    assert any(
        e.source == "news_sentiment" and "Bitcoin ETF inflows" in e.statement
        for e in analysis.evidence
    )
    risk = analysis.agent_results[1]
    assert risk.findings["reviewed_agents"] == ["news_sentiment"]
    assert "news_sentiment" not in {r.agent for r in analysis.agent_results if r.mock}


def test_general_question_gets_price_context_from_the_market_agent():
    analysis = ask("What's up with BTC?").analysis
    news = next(r for r in analysis.agent_results if r.agent == "news_sentiment")
    assert news.status == "ok"
    [context] = news.findings["price_context"]
    assert context["symbol"] == "BTC" and context["change_24h_pct"] == 1.5  # fake CoinGecko
    response = ask("What's up with BTC?")
    assert "News for BTC" in response.message.content


def test_news_failure_does_not_break_other_agents(fake_news):
    fake_news.fail_all(httpx2.ReadTimeout("slow"))
    response = ask("What's up with ETH?")
    by_agent = {r.agent: r for r in response.analysis.agent_results}
    assert by_agent["news_sentiment"].status == "error"
    assert by_agent["market"].status == "ok"
    assert any("news_sentiment agent failed" in n for n in response.analysis.uncertainty.notes)
    assert "news_sentiment: failed" in response.message.content


def test_slow_news_is_isolated_by_the_orchestrator_timeout(fake_sentiment):
    fake_sentiment.delay = 1.0
    orchestrator = Orchestrator()
    orchestrator.agents["news_sentiment"].timeout = 0.05  # this instance only
    request = ChatRequest(messages=[ChatMessage(role="user", content="BTC news")])
    analysis = asyncio.run(orchestrator.respond(request)).analysis
    news = analysis.agent_results[0]
    assert news.status == "error" and news.error == "timed out"
    assert analysis.agent_results[1].status == "ok"  # risk still ran


def test_screenshot_asset_reaches_the_news_agent():
    analysis = ask("any news about this coin?", images=1).analysis
    assert "news_sentiment" in analysis.agents_used
    news = next(r for r in analysis.agent_results if r.agent == "news_sentiment")
    assert news.findings["reports"][0]["asset"] == "BTC"  # read from the (fake) screenshot


# --- Routing -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "assets"),
    [
        ("Any news on bitcoin?", ["BTC"]),
        ("why is SOL down today", ["SOL"]),
        ("ETH and SOL headlines", ["ETH", "SOL"]),
        ("what's the sentiment in crypto right now", []),
        ("SEC news for XRP", ["XRP"]),
    ],
)
def test_news_questions_route_to_the_news_agent(query, assets):
    decision = route(query, has_images=False)
    assert "news_sentiment" in decision.agents
    assert decision.assets == assets


def test_non_crypto_news_is_not_routed():
    assert route("any news about the weather?", has_images=False).agents == []
