import asyncio
import dataclasses
from collections.abc import Iterable, Mapping

from upscale.agents import Agent, AgentContext, default_agents
from upscale.routing import RoutingDecision, route
from upscale.schemas import (
    AgentName,
    AgentResult,
    Analysis,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Evidence,
    Risk,
    Scenario,
    Uncertainty,
)

AGENT_TIMEOUT_SECONDS = 30.0
DISCLAIMER = (
    "UpScale explains evidence, scenarios, risks and uncertainty. It does not give "
    "buy/sell recommendations and is not financial advice."
)


class Orchestrator:
    """Entry point for answering a chat turn.

    Routes the request to the relevant agents, runs them (respecting each agent's
    `depends_on`), and merges their results into one structured `Analysis`.
    """

    def __init__(
        self, agents: Iterable[Agent] | None = None, timeout: float = AGENT_TIMEOUT_SECONDS
    ):
        self.agents: dict[AgentName, Agent] = {
            agent.name: agent for agent in (agents or default_agents())
        }
        self.timeout = timeout

    async def respond(self, request: ChatRequest) -> ChatResponse:
        latest = request.messages[-1]
        query = latest.content.strip()
        decision = route(query, has_images=bool(latest.attachments))
        base_context = AgentContext(
            query=query,
            attachments=latest.attachments,
            history=request.messages[:-1],
            assets=decision.assets,
            assets_source="user" if decision.assets else None,
            timeframe=decision.timeframe,
            timeframe_source="user" if decision.timeframe else None,
        )
        results = await self.run_agents(decision.agents, base_context)
        final_context = apply_vision(base_context, {r.agent: r for r in results})
        analysis = self.synthesize(decision, results, assets=final_context.assets)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=render_text(analysis)),
            analysis=analysis,
        )

    async def run_agents(self, names: list[AgentName], context: AgentContext) -> list[AgentResult]:
        """Run the selected agents in dependency waves; agents in the same wave run concurrently."""
        pending = [name for name in names if name in self.agents]
        results: dict[AgentName, AgentResult] = {}
        while pending:
            ready = [
                name
                for name in pending
                if all(dep in results or dep not in pending for dep in self.agents[name].depends_on)
            ]
            if not ready:  # dependency cycle: run the rest without waiting on each other
                ready = pending
            wave_context = apply_vision(context, results)
            wave = await asyncio.gather(
                *(self._run_one(self.agents[n], wave_context) for n in ready)
            )
            results.update((result.agent, result) for result in wave)
            pending = [name for name in pending if name not in results]
        return [results[name] for name in names if name in results]

    async def _run_one(self, agent: Agent, context: AgentContext) -> AgentResult:
        try:
            timeout = agent.timeout or self.timeout
            return await asyncio.wait_for(agent.run(context), timeout=timeout)
        except Exception as exc:  # one failing agent must not break the whole response
            reason = (
                "timed out" if isinstance(exc, TimeoutError) else f"{type(exc).__name__}: {exc}"
            )
            return AgentResult(
                agent=agent.name,
                status="error",
                mock=False,
                summary=f"{agent.name} agent failed.",
                error=reason,
            )

    def synthesize(
        self,
        decision: RoutingDecision,
        results: list[AgentResult],
        assets: list[str] | None = None,
    ) -> Analysis:
        ok = [r for r in results if r.status == "ok"]
        failed = [r for r in results if r.status == "error"]
        mock = any(r.mock for r in results)

        evidence = [Evidence(source=r.agent, statement=s) for r in ok for s in r.evidence]
        scenarios: list[Scenario] = [
            s.model_copy(update={"source": r.agent}) for r in ok for s in r.scenarios
        ]
        risks: list[Risk] = []
        seen: set[str] = set()
        for r in ok:
            for risk in r.risks:
                if risk.description not in seen:
                    seen.add(risk.description)
                    risks.append(risk.model_copy(update={"source": risk.source or r.agent}))

        notes: list[str] = []
        if mocked := [r.agent for r in results if r.mock]:
            notes.append(f"Mock placeholder output from: {', '.join(mocked)}.")
        notes += [f"{r.agent} agent failed ({r.error})." for r in failed]
        if not results:
            notes.append("No agents were relevant to this request.")

        return Analysis(
            mock=mock,
            summary=_summary(decision, ok, assets if assets is not None else decision.assets),
            assets=assets if assets is not None else decision.assets,
            agents_used=[r.agent for r in results],
            routing={
                r.agent: decision.reasons[r.agent] for r in results if r.agent in decision.reasons
            },
            evidence=evidence,
            scenarios=scenarios,
            risks=risks,
            uncertainty=Uncertainty(
                level="high" if mock or failed or not ok else "medium", notes=notes
            ),
            agent_results=results,
            disclaimer=DISCLAIMER,
        )


def apply_vision(context: AgentContext, results: Mapping[AgentName, AgentResult]) -> AgentContext:
    """Context for the next agents: prior results, plus the screenshot's asset/timeframe
    when the user's text didn't specify them (explicit user requests always win)."""
    updates: dict[str, object] = {"prior_results": dict(results)}
    vision = results.get("vision")
    if vision and vision.status == "ok" and not vision.mock:
        asset = vision.findings.get("detected_asset")
        if not context.assets and isinstance(asset, str):
            updates |= {"assets": [asset], "assets_source": "screenshot"}
        # Keep an unrecognized label (e.g. "1W") so Technical can say it isn't available.
        timeframe = vision.findings.get("detected_timeframe") or vision.findings.get(
            "detected_timeframe_label"
        )
        if not context.timeframe and isinstance(timeframe, str):
            updates |= {"timeframe": timeframe, "timeframe_source": "screenshot"}
    return dataclasses.replace(context, **updates)  # type: ignore[arg-type]


def _summary(decision: RoutingDecision, ok: list[AgentResult], assets: list[str]) -> str:
    if not decision.agents:
        return (
            "I couldn't find a crypto question or chart screenshot in your message. "
            "Ask about a coin (e.g. BTC, ETH) or attach a chart screenshot."
        )
    subject = ", ".join(assets) if assets else "your request"
    mocked = sum(r.mock for r in ok)
    prefix = "" if not mocked else "[Mock] " if mocked == len(ok) else "[Partly mock] "
    return f"{prefix}Combined analysis of {subject} from {len(ok)} agent(s)."


def _mock_notice(analysis: Analysis) -> str:
    live = [r.agent for r in analysis.agent_results if not r.mock and r.status == "ok"]
    notice = "Prototype mode: results marked [Mock] are placeholders. No AI model is connected."
    if live:
        notice += f" Live data: {', '.join(live)}."
    return notice


def render_text(analysis: Analysis) -> str:
    """Plain-text version of the analysis for the chat UI (which renders text, not markdown)."""
    if not analysis.agent_results:
        return analysis.summary

    sections: list[str] = []
    if analysis.mock:
        sections.append(_mock_notice(analysis))
    sections.append(analysis.summary)
    sections.append(
        "Agents:\n"
        + "\n".join(
            f"• {r.summary if r.status == 'ok' else f'{r.agent}: failed ({r.error})'}"
            for r in analysis.agent_results
        )
    )
    if analysis.evidence:
        sections.append("Evidence:\n" + "\n".join(f"• {e.statement}" for e in analysis.evidence))
    if analysis.scenarios:
        sections.append(
            "Scenarios:\n" + "\n".join(f"• {s.name}: {s.description}" for s in analysis.scenarios)
        )
    if analysis.risks:
        sections.append(
            "Risks:\n" + "\n".join(f"• [{r.severity}] {r.description}" for r in analysis.risks)
        )
    sections.append(
        f"Uncertainty: {analysis.uncertainty.level}"
        + "".join(f"\n• {note}" for note in analysis.uncertainty.notes)
    )
    sections.append(analysis.disclaimer)
    return "\n\n".join(sections)
