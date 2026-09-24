from upscale.agents.base import Agent, AgentContext
from upscale.schemas import AgentResult, Risk


class MockRiskAgent(Agent):
    """Placeholder for risk assessment. Reviews what the other agents produced."""

    name = "risk"
    description = "Highlights risks and data gaps across all other agents' output."
    depends_on = ("vision", "technical_analysis", "market", "news_sentiment", "opportunity")

    async def run(self, context: AgentContext) -> AgentResult:
        reviewed = sorted(context.prior_results)
        mocked = sorted(name for name, result in context.prior_results.items() if result.mock)
        risks = [
            Risk(
                description=f"Mock (placeholder) data from: {', '.join(mocked + [self.name])}.",
                severity="high",
            ),
            Risk(
                description="Crypto assets are highly volatile; any scenario can fail quickly.",
                severity="high",
            ),
        ]
        failed = [
            name for name, result in context.prior_results.items() if result.status == "error"
        ]
        if failed:
            risks.append(
                Risk(
                    description=f"Some agents failed: {', '.join(sorted(failed))}.",
                    severity="medium",
                )
            )
        return AgentResult(
            agent=self.name,
            mock=True,
            summary=f"[Mock] Reviewed output from {len(reviewed)} agent(s); overall risk level is unknown.",
            findings={"reviewed_agents": reviewed, "overall_risk": "unknown (mock)"},
            evidence=["Risk level is not computed yet (mock)."],
            risks=risks,
        )
