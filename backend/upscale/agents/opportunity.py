from upscale.agents.base import Agent, AgentContext
from upscale.schemas import AgentResult, Scenario


class MockOpportunityAgent(Agent):
    """Placeholder for scenario building.

    Describes conditional scenarios from the other agents' output. It must never
    produce buy/sell recommendations, entries, targets or position sizes.
    """

    name = "opportunity"
    description = "Builds conditional bullish/neutral/bearish scenarios. No trade recommendations."
    depends_on = ("vision", "technical_analysis", "market", "news_sentiment")

    async def run(self, context: AgentContext) -> AgentResult:
        asset = context.primary_asset or "the asset"
        return AgentResult(
            agent=self.name,
            mock=True,
            summary=f"[Mock] Three placeholder scenarios for {asset}; these are not recommendations.",
            findings={
                "asset": context.primary_asset,
                "based_on": sorted(context.prior_results),
                "recommendation": None,
            },
            scenarios=[
                Scenario(
                    name="Bullish continuation (mock)",
                    description=f"{asset} holds support and momentum improves.",
                    conditions=["Price stays above support", "Rising volume on up moves"],
                    invalidation="A decisive close below support.",
                ),
                Scenario(
                    name="Range (mock)",
                    description=f"{asset} keeps trading between support and resistance.",
                    conditions=["No clear break of either level", "Declining volume"],
                    invalidation="A decisive break of either level.",
                ),
                Scenario(
                    name="Bearish breakdown (mock)",
                    description=f"{asset} loses support and sentiment weakens.",
                    conditions=["Close below support", "Negative news flow"],
                    invalidation="A quick recovery back above support.",
                ),
            ],
            evidence=["Scenarios are templates, not derived from real data (mock)."],
        )
