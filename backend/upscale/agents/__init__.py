"""Specialized agents. Each is called by the orchestrator, never directly by the API.

The vision, market and technical analysis agents are real (via `upscale.services`):
a vision model reads screenshots, live data and deterministic calculations do the rest.
The others are still mock implementations that return clearly
labeled placeholder output and make no AI or network calls. Replace an agent by writing a new `Agent` subclass with the same
`name` and swapping it in `default_agents()`.
"""

from upscale.agents.base import Agent, AgentContext
from upscale.agents.market import MarketAgent
from upscale.agents.news_sentiment import MockNewsSentimentAgent
from upscale.agents.opportunity import MockOpportunityAgent
from upscale.agents.risk import MockRiskAgent
from upscale.agents.technical import TechnicalAnalysisAgent
from upscale.agents.vision import VisionAgent


def default_agents() -> list[Agent]:
    return [
        VisionAgent(),
        TechnicalAnalysisAgent(),
        MarketAgent(),
        MockNewsSentimentAgent(),
        MockOpportunityAgent(),
        MockRiskAgent(),
    ]


__all__ = [
    "Agent",
    "AgentContext",
    "MarketAgent",
    "MockNewsSentimentAgent",
    "MockOpportunityAgent",
    "MockRiskAgent",
    "TechnicalAnalysisAgent",
    "VisionAgent",
    "default_agents",
]
