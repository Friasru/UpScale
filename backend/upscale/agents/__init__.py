"""Specialized agents. Each is called by the orchestrator, never directly by the API.

All agents are real (via `upscale.services`): a vision model reads screenshots, live data
and deterministic calculations do the rest, news comes from real publishers' feeds, and
the risk and opportunity agents apply deterministic rules to the other agents' evidence.
Replace an agent by writing a new `Agent` subclass with the same
`name` and swapping it in `default_agents()`.
"""

from upscale.agents.base import Agent, AgentContext
from upscale.agents.market import MarketAgent
from upscale.agents.news_sentiment import NewsSentimentAgent
from upscale.agents.opportunity import OpportunityAgent
from upscale.agents.risk import RiskAgent
from upscale.agents.technical import TechnicalAnalysisAgent
from upscale.agents.vision import VisionAgent


def default_agents() -> list[Agent]:
    return [
        VisionAgent(),
        TechnicalAnalysisAgent(),
        MarketAgent(),
        NewsSentimentAgent(),
        RiskAgent(),
        OpportunityAgent(),
    ]


__all__ = [
    "Agent",
    "AgentContext",
    "MarketAgent",
    "NewsSentimentAgent",
    "OpportunityAgent",
    "RiskAgent",
    "TechnicalAnalysisAgent",
    "VisionAgent",
    "default_agents",
]
