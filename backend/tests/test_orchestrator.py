import asyncio
import json
from datetime import UTC, datetime

import httpx2
import pytest

from upscale.agents import Agent, AgentContext
from upscale.orchestrator import Orchestrator
from upscale.routing import detect_assets, route
from upscale.schemas import AgentResult, ChatMessage, ChatRequest, ImageAttachment

from .conftest import PNG_1PX, coingecko_row
from .test_technical_agent import serve_wave


def ask(orchestrator: Orchestrator, content: str, images: int = 0):
    attachments = [
        ImageAttachment(name=f"c{i}.png", media_type="image/png", data=PNG_1PX)
        for i in range(images)
    ]
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=content, attachments=attachments)]
    )
    return asyncio.run(orchestrator.respond(request))


# --- Routing ----------------------------------------------------------------------


def test_detect_assets():
    assert detect_assets("bitcoin vs $ETH and sol") == ["BTC", "ETH", "SOL"]
    # Ambiguous tickers only count when written as tickers.
    assert detect_assets("click the link") == []
    assert detect_assets("LINK and $dot") == ["LINK", "DOT"]


def test_route_general_crypto_question():
    decision = route("What do you think about ETH?", has_images=False)
    assert decision.agents == [
        "technical_analysis",
        "market",
        "news_sentiment",
        "risk",
        "opportunity",
    ]


def test_route_specific_intent_selects_subset():
    decision = route("Any news on bitcoin?", has_images=False)
    assert decision.agents == ["news_sentiment", "risk"]
    decision = route("BTC RSI and support levels", has_images=False)
    assert decision.agents == ["technical_analysis", "risk"]


def test_route_screenshot():
    assert route("", has_images=True).agents == [
        "vision",
        "technical_analysis",
        "market",
        "risk",
        "opportunity",
    ]


def test_route_ignores_non_crypto_text():
    assert route("what's the price of coffee", has_images=False).agents == []
    assert route("hi", has_images=False).agents == []


# --- Execution --------------------------------------------------------------------


def test_orchestrator_runs_selected_agents_and_combines_results():
    response = ask(Orchestrator(), "BTC outlook?")
    analysis = response.analysis
    assert analysis is not None
    assert [r.agent for r in analysis.agent_results] == analysis.agents_used
    assert analysis.evidence and all(e.source in analysis.agents_used for e in analysis.evidence)
    assert analysis.risks and all(r.source for r in analysis.risks)
    assert analysis.mock is False
    # The reply leads with the decision block.
    assert response.message.content.split("\n", 1)[0] in ("BUY", "SELL", "WAIT")
    assert "Prototype mode" not in response.message.content
    assert "not built yet" not in response.message.content


def test_orchestrator_passes_dependency_results_downstream():
    risk = next(
        r for r in ask(Orchestrator(), "", images=1).analysis.agent_results if r.agent == "risk"
    )
    assert risk.findings["reviewed_agents"] == ["market", "technical_analysis", "vision"]


class RecordingAgent(Agent):
    def __init__(self, name, depends_on=(), fail=False):
        self.name = name
        self.depends_on = depends_on
        self.fail = fail
        self.description = "test"
        self.seen: list[str] = []

    async def run(self, context: AgentContext) -> AgentResult:
        self.seen = sorted(context.prior_results)
        if self.fail:
            raise RuntimeError("boom")
        return AgentResult(
            agent=self.name, mock=False, summary=f"{self.name} ok", evidence=[f"{self.name} fact"]
        )


def test_orchestrator_isolates_agent_failures():
    market = RecordingAgent("market", fail=True)
    risk = RecordingAgent("risk", depends_on=("market",))
    orchestrator = Orchestrator(agents=[market, risk])
    results = asyncio.run(orchestrator.run_agents(["market", "risk"], AgentContext(query="")))
    assert [r.status for r in results] == ["error", "ok"]
    assert "boom" in results[0].error
    assert risk.seen == ["market"]


def test_orchestrator_times_out_slow_agents():
    class SlowAgent(RecordingAgent):
        async def run(self, context):
            await asyncio.sleep(1)
            return await super().run(context)

    orchestrator = Orchestrator(agents=[SlowAgent("market")], timeout=0.01)
    [result] = asyncio.run(orchestrator.run_agents(["market"], AgentContext(query="")))
    assert result.status == "error"
    assert result.error == "timed out"


def test_orchestrator_skips_agents_that_are_not_registered():
    orchestrator = Orchestrator(agents=[RecordingAgent("market")])
    response = ask(orchestrator, "What about BTC?")
    assert response.analysis.agents_used == ["market"]
    assert response.analysis.mock is False
    assert response.analysis.uncertainty.level == "medium"


def test_detect_timeframe():
    from upscale.routing import detect_timeframe

    assert detect_timeframe("BTC 4h chart") == "4h"
    assert detect_timeframe("daily RSI on eth") == "1d"
    assert detect_timeframe("15min SOL") == "15m"
    assert detect_timeframe("BTC 30m chart") == "30m"
    assert detect_timeframe("30 minutes candles on eth") == "30m"
    assert detect_timeframe("BTC 1m scalp") == "1m"
    assert detect_timeframe("1 hour candles") == "1h"
    assert detect_timeframe("what about btc") is None
    assert route("ETH 4h", has_images=False).timeframe == "4h"


# --- Uncertainty: mock placeholders don't raise it ------------------------------------------


def _result(agent, *, mock=False, status="ok", **findings):
    return AgentResult(
        agent=agent,
        mock=mock,
        status=status,
        summary=f"{'[Mock] ' if mock else ''}{agent}",
        findings=findings,
        error="boom" if status == "error" else None,
    )


def _review(level, reviewed=("market", "technical_analysis"), reasons=()):
    return _result(
        "risk",
        uncertainty_level=level,
        uncertainty_reasons=list(reasons),
        reviewed_agents=list(reviewed),
    )


def _uncertainty(*results):
    decision = route("What about BTC?", has_images=False)
    return Orchestrator().synthesize(decision, list(results)).uncertainty


MOCK_OPPORTUNITY = _result("opportunity", mock=True)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_mock_opportunity_does_not_raise_uncertainty(level):
    uncertainty = _uncertainty(_result("market"), _review(level), MOCK_OPPORTUNITY)
    assert uncertainty.level == level
    assert any(n.startswith("Not built yet: opportunity") for n in uncertainty.notes)


def test_risk_review_reasons_become_uncertainty_notes():
    uncertainty = _uncertainty(
        _result("market"), _review("medium", reasons=["news is missing"]), MOCK_OPPORTUNITY
    )
    assert "Risk review: news is missing." in uncertainty.notes


def test_real_failure_outside_the_risk_review_is_at_least_medium():
    uncertainty = _uncertainty(
        _result("market"), _result("opportunity", status="error"), _review("low")
    )
    assert uncertainty.level == "medium"
    # A failure the risk review already assessed is left to its level.
    covered = _uncertainty(
        _result("market", status="error"), _result("news_sentiment"), _review("low")
    )
    assert covered.level == "low"


def test_uncertainty_without_a_risk_review():
    assert _uncertainty(_result("market"), MOCK_OPPORTUNITY).level == "medium"
    assert _uncertainty(_result("market"), _result("news_sentiment", status="error")).level == (
        "high"
    )


def test_only_mock_output_is_high_uncertainty():
    assert _uncertainty(MOCK_OPPORTUNITY).level == "high"
    assert _uncertainty().level == "high"


def test_unreadable_risk_uncertainty_is_treated_as_high():
    assert _uncertainty(_result("market"), _review("unclear")).level == "high"


def test_general_question_uncertainty_follows_real_evidence(fake_coingecko):
    # Consistent, fresh live data: candles around $100 and a matching current price.
    serve_wave(fake_coingecko)
    candles = fake_coingecko.handler
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    btc = coingecko_row("bitcoin", "btc", "Bitcoin", 100.0, last_updated=now)

    def handler(request):
        if request.url.path.endswith("/markets"):
            return httpx2.Response(200, content=json.dumps([btc]))
        return candles(request)

    fake_coingecko.handler = handler
    response = ask(Orchestrator(), "What's up with BTC?")
    analysis = response.analysis
    by_agent = {r.agent: r for r in analysis.agent_results}
    opportunity = by_agent["opportunity"]
    assert opportunity.mock is False and opportunity.status == "ok"
    assert analysis.mock is False
    risk_level = by_agent["risk"].findings["uncertainty_level"]
    assert risk_level in ("low", "medium")
    assert analysis.uncertainty.level == risk_level
    assert opportunity.findings["uncertainty_level"] == risk_level
    assert response.message.content.startswith(opportunity.summary)
