from upscale.agents.base import Agent, AgentContext
from upscale.schemas import AgentResult, Risk


class MockNewsSentimentAgent(Agent):
    """Placeholder for news and social sentiment. No feeds are fetched."""

    name = "news_sentiment"
    description = "Recent headlines and overall market/social sentiment."

    async def run(self, context: AgentContext) -> AgentResult:
        asset = context.primary_asset or "crypto market"
        return AgentResult(
            agent=self.name,
            mock=True,
            summary=f"[Mock] News & sentiment for {asset}: no news or social sources are connected.",
            findings={
                "asset": context.primary_asset,
                "headlines": [
                    {
                        "title": f"[Mock] Placeholder headline about {asset}",
                        "source": None,
                        "sentiment": "neutral",
                    }
                ],
                "overall_sentiment": "neutral (mock)",
                "sentiment_score": None,
            },
            evidence=["Sentiment is a placeholder; no headlines were retrieved (mock)."],
            risks=[
                Risk(
                    description="Recent news that could move the market is not being tracked yet.",
                    severity="medium",
                )
            ],
        )
