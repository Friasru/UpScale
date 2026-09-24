from upscale.agents.base import Agent, AgentContext
from upscale.schemas import AgentResult
from upscale.services import explainer_model
from upscale.services.explainer import ExplainerError, ExplainerModel


class EducationAgent(Agent):
    """Explains general trading and crypto concepts ("What is RSI?").

    Runs alone, for questions that ask what a concept means rather than what a market is
    doing: no live data, no risk review and no BUY / SELL / WAIT decision.
    """

    name = "education"
    description = "Concise educational explanations of trading and crypto concepts."
    timeout = 45.0

    def __init__(self, model: ExplainerModel | None = None):
        self._model = model

    @property
    def model(self) -> ExplainerModel:
        # Resolved per call so the shared model can be swapped (e.g. in tests).
        return self._model or explainer_model

    async def run(self, context: AgentContext) -> AgentResult:
        try:
            text = await self.model.explain(context.query)
        except ExplainerError as exc:
            return AgentResult(
                agent=self.name,
                status="error",
                mock=False,
                summary="The explanation could not be generated.",
                findings={"model": self.model.name},
                error=str(exc),
            )
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=text,
            findings={"model": self.model.name},
        )
