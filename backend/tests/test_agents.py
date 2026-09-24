import asyncio

from upscale.agents import Agent, AgentContext, default_agents
from upscale.schemas import AgentResult

BUY_SELL_WORDS = ("buy", "sell", "go long", "go short", "entry price", "take profit", "stop loss")


def run(agent: Agent, context: AgentContext) -> AgentResult:
    return asyncio.run(agent.run(context))


def test_default_agents_cover_every_role_once():
    names = [agent.name for agent in default_agents()]
    assert sorted(names) == sorted(
        ["vision", "technical_analysis", "market", "news_sentiment", "opportunity", "risk"]
    )


def test_no_default_agent_is_a_mock():
    for agent in default_agents():
        assert agent.description
        assert "mock" not in type(agent).__name__.lower()


def test_opportunity_decides_after_every_evidence_agent():
    agents = {a.name: a for a in default_agents()}
    assert set(agents["opportunity"].depends_on) == {
        "vision",
        "technical_analysis",
        "market",
        "news_sentiment",
        "risk",
    }
    assert "opportunity" not in agents["risk"].depends_on
