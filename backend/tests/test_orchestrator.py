import asyncio

from upscale.agents import Agent, AgentContext
from upscale.orchestrator import Orchestrator
from upscale.routing import detect_assets, route
from upscale.schemas import AgentResult, ChatMessage, ChatRequest, ImageAttachment

from .conftest import PNG_1PX


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
        "opportunity",
        "risk",
    ]


def test_route_specific_intent_selects_subset():
    decision = route("Any news on bitcoin?", has_images=False)
    assert decision.agents == ["news_sentiment", "risk"]
    decision = route("BTC RSI and support levels", has_images=False)
    assert decision.agents == ["technical_analysis", "risk"]


def test_route_screenshot():
    assert route("", has_images=True).agents == ["vision", "technical_analysis", "market", "risk"]


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
    assert analysis.scenarios
    assert analysis.risks and all(r.source for r in analysis.risks)
    assert analysis.uncertainty.level == "high"
    assert response.message.content.startswith("Prototype mode")
    assert "No AI model is connected" not in response.message.content
    assert "not built yet (opportunity, risk)" in response.message.content


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
    assert detect_timeframe("1 hour candles") == "1h"
    assert detect_timeframe("what about btc") is None
    assert route("ETH 4h", has_images=False).timeframe == "4h"
