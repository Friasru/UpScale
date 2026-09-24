"""Specialized agents. Each is called by the orchestrator, never directly by the API.

The vision, market, technical analysis, news & sentiment and risk agents are real (via
`upscale.services`): a vision model reads screenshots, live data and deterministic
calculations do the rest, news comes from real publishers' feeds, and the risk agent
applies deterministic rules to the other agents' evidence. The opportunity agent is
still a mock implementation that returns clearly labeled placeholder output and makes no
AI or network calls. Replace an agent by writing a new `Agent` subclass with the same
`name` and swapping it in `default_agents()`.
"""

from upscale.agents.base import Agent, AgentContext
from upscale.agents.market import MarketAgent
from upscale.agents.news_sentiment import NewsSentimentAgent
from upscale.agents.opportunity import MockOpportunityAgent
from upscale.agents.risk import RiskAgent
from upscale.agents.technical import TechnicalAnalysisAgent
from upscale.agents.vision import VisionAgent


def default_agents() -> list[Agent]:
    return [
        VisionAgent(),
        TechnicalAnalysisAgent(),
        MarketAgent(),
        NewsSentimentAgent(),
        MockOpportunityAgent(),
        RiskAgent(),
    ]


__all__ = [
    "Agent",
    "AgentContext",
    "MarketAgent",
    "MockOpportunityAgent",
    "NewsSentimentAgent",
    "RiskAgent",
    "TechnicalAnalysisAgent",
    "VisionAgent",
    "default_agents",
]
