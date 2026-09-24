import asyncio

import pytest

from upscale.agents import (
    Agent,
    AgentContext,
    MockOpportunityAgent,
    MockRiskAgent,
    default_agents,
)
from upscale.schemas import AgentResult

BUY_SELL_WORDS = ("buy", "sell", "go long", "go short", "entry price", "take profit", "stop loss")


def run(agent: Agent, context: AgentContext) -> AgentResult:
    return asyncio.run(agent.run(context))


def test_default_agents_cover_every_role_once():
    names = [agent.name for agent in default_agents()]
    assert sorted(names) == sorted(
        ["vision", "technical_analysis", "market", "news_sentiment", "opportunity", "risk"]
    )


LIVE_AGENTS = {"vision", "market", "technical_analysis", "news_sentiment"}  # own test modules
MOCK_AGENTS = [a for a in default_agents() if a.name not in LIVE_AGENTS]


@pytest.mark.parametrize("agent", MOCK_AGENTS, ids=lambda a: a.name)
def test_every_mock_agent_returns_labeled_mock_result(agent):
    result = run(agent, AgentContext(query="BTC outlook", assets=["BTC"]))
    assert isinstance(result, AgentResult)
    assert result.agent == agent.name
    assert result.status == "ok"
    assert result.mock is True
    assert result.summary.startswith("[Mock]")
    assert agent.description


def test_opportunity_agent_gives_scenarios_not_recommendations():
    result = run(MockOpportunityAgent(), AgentContext(query="should I buy BTC?", assets=["BTC"]))
    assert len(result.scenarios) == 3
    assert result.findings["recommendation"] is None
    text = " ".join(
        f"{s.name} {s.description} {' '.join(s.conditions)}" for s in result.scenarios
    ).lower()
    assert not any(word in text for word in BUY_SELL_WORDS)


def test_risk_agent_flags_failed_agents():
    failed = AgentResult(agent="market", status="error", mock=False, summary="x", error="boom")
    result = run(MockRiskAgent(), AgentContext(query="", prior_results={"market": failed}))
    assert any("market" in risk.description for risk in result.risks)
