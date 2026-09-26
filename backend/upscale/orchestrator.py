import asyncio
import dataclasses
import logging
import time
from collections.abc import Iterable, Mapping
from typing import cast

import upscale.services as services
from upscale.agents import Agent, AgentContext, default_agents
from upscale.routing import (
    RoutingDecision,
    add_agent,
    is_concept_question,
    is_trading_follow_up,
    route,
)
from upscale.schemas import (
    AgentName,
    AgentResult,
    Analysis,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Evidence,
    Level,
    Risk,
    Scenario,
    Uncertainty,
)
from upscale.services.asset_profile import (
    AssetIdentity,
    build_profile,
    integrated_capabilities,
    wants_dex_data,
    wants_onchain_data,
)
from upscale.services.asset_registry import DEFAULT_REGISTRY
from upscale.services.asset_resolver import AssetResolver, Resolution, ResolvedAsset
from upscale.services.chains import DEX_CHAINS
from upscale.services.trade_context import (
    IdentityConfidence,
    VenueRequest,
    build_trade_context,
    requested_venue,
    trader_context,
)

AGENT_TIMEOUT_SECONDS = 30.0
_LEVEL_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
DISCLAIMER = (
    "BUY / SELL / WAIT reads are rule-based decision support from the evidence shown, not "
    "guarantees or financial advice. SELL means reduce or exit a long position, not open "
    "a short."
)
EDUCATION_DISCLAIMER = "General educational explanation, not financial advice."
logger = logging.getLogger("upscale.performance")


class Orchestrator:
    """Entry point for answering a chat turn.

    Routes the request to the relevant agents, runs them (respecting each agent's
    `depends_on`), and merges their results into one structured `Analysis`.
    """

    def __init__(
        self,
        agents: Iterable[Agent] | None = None,
        timeout: float = AGENT_TIMEOUT_SECONDS,
        resolver: AssetResolver | None = None,
    ):
        self.agents: dict[AgentName, Agent] = {
            agent.name: agent for agent in (agents or default_agents())
        }
        self.timeout = timeout
        self._resolver = resolver

    @property
    def resolver(self) -> AssetResolver:
        return self._resolver or services.asset_resolver

    async def respond(self, request: ChatRequest) -> ChatResponse:
        """Resolve the exact asset, build the trading context, run the relevant agents
        (independent ones concurrently), and return BUY / SELL / WAIT with its evidence."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        latest = request.messages[-1]
        query = latest.content.strip()
        has_images = bool(latest.attachments)
        history = [m.content for m in request.messages[:-1] if m.role == "user"]

        resolution: Resolution | None = None
        if is_concept_question(query, has_images):
            decision = route(query, has_images)
        else:
            t0 = time.perf_counter()
            # A screenshot names its own asset (Vision reads it), so the conversation's
            # earlier asset is only reused for text follow-ups.
            follow_up = history if is_trading_follow_up(query) and not has_images else []
            resolution = await self.resolver.resolve(query, follow_up)
            timings["asset_resolution"] = _ms(t0)
            if resolution.clarification:
                return _reply(resolution.clarification, timings, started)
            if not resolution.assets and resolution.notes and not has_images:
                return _reply(" ".join(resolution.notes), timings, started)
            decision = route(query, has_images, resolution)

        primary = resolution.primary if resolution else None
        venue = requested_venue(query, primary.identity.address if primary else None)
        identity = _trade_identity(primary, venue)
        venue_kind = (
            "dex"
            if identity is not None and venue.kind != "cex"
            else (venue.kind if venue.explicit else None)
        )
        decision = with_profile_agents(decision, identity, venue.kind)
        profile = build_profile(
            identity or (decision.assets[0] if decision.assets else None), venue=venue_kind
        )
        trade = build_trade_context(
            profile,
            _confidence(primary),
            trader_context(query, history, decision.timeframe),
            decision.timeframe,
            venue=venue,
        )
        base_context = AgentContext(
            query=query,
            attachments=latest.attachments,
            history=request.messages[:-1],
            assets=decision.assets,
            assets_source="mint"
            if primary is not None and primary.is_contract
            else "user"
            if decision.assets
            else None,
            asset_identity=identity,
            trade=trade,
            timeframe=decision.timeframe,
            timeframe_source="user" if decision.timeframe else None,
        )
        # Token data that doesn't depend on other agents starts downloading right away.
        prefetch = (
            asyncio.create_task(
                services.provider_registry.prefetch(identity.chain, identity.address)
            )
            if identity is not None
            else None
        )
        results = await self.run_agents(decision.agents, base_context, timings)
        if prefetch is not None:
            await prefetch
        final_context = apply_vision(base_context, {r.agent: r for r in results})
        analysis = self.synthesize(decision, results, assets=final_context.assets)
        if primary is not None and primary.note:
            analysis.uncertainty.notes.insert(0, primary.note)
        timings["total"] = _ms(started)
        analysis.timings = timings
        logger.info("decision latency (ms): %s", timings)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=render_text(analysis)),
            analysis=analysis,
        )

    async def run_agents(
        self,
        names: list[AgentName],
        context: AgentContext,
        timings: dict[str, float] | None = None,
    ) -> list[AgentResult]:
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
                *(self._run_one(self.agents[n], wave_context, timings) for n in ready)
            )
            results.update((result.agent, result) for result in wave)
            pending = [name for name in pending if name not in results]
        return [results[name] for name in names if name in results]

    async def _run_one(
        self, agent: Agent, context: AgentContext, timings: dict[str, float] | None = None
    ) -> AgentResult:
        started = time.perf_counter()
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
        finally:
            if timings is not None:
                timings[f"agent.{agent.name}"] = _ms(started)

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

        review = next((r for r in ok if r.agent == "risk" and not r.mock), None)
        notes: list[str] = []
        notes += [f"{r.agent} agent failed ({r.error})." for r in failed]
        if review is not None:
            notes += [
                f"Risk review: {reason}."
                for reason in review.findings.get("uncertainty_reasons", [])
            ]
        if not results:
            notes.append("No agents were relevant to this request.")
        if mocked := [r.agent for r in results if r.mock]:
            notes.append(
                f"Not built yet: {', '.join(mocked)} (placeholder output, not used as evidence "
                "and not counted in this uncertainty level)."
            )

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
            uncertainty=Uncertainty(level=_uncertainty_level(results, review), notes=notes),
            agent_results=results,
            disclaimer=EDUCATION_DISCLAIMER if _is_education(decision) else DISCLAIMER,
        )


def _uncertainty_level(results: list[AgentResult], review: AgentResult | None) -> Level:
    """Overall uncertainty from real agents only; mock placeholders never raise it.

    * No real agent produced output: high.
    * With the real risk review: its deterministic uncertainty level (which already covers
      its input agents' failures and missing evidence), raised to at least medium if some
      other real agent failed.
    * Without a risk review: high if any real agent failed, otherwise medium.
    """
    real = [r for r in results if not r.mock]
    if not any(r.status == "ok" for r in real):
        return "high"
    failed = {r.agent for r in real if r.status == "error"}
    if review is None:
        return "high" if failed else "medium"
    level = review.findings.get("uncertainty_level")
    if level not in _LEVEL_RANK:
        level = "high"
    uncovered = failed - set(review.findings.get("reviewed_agents", []))
    if uncovered and _LEVEL_RANK[level] < _LEVEL_RANK["medium"]:
        level = "medium"
    return cast(Level, level)


def _trade_identity(primary: ResolvedAsset | None, venue: VenueRequest) -> AssetIdentity | None:
    """The chain + address identity agents should use for a DEX trade, else None.

    An unregistered token is always traded by its address. A registered token (PEPE,
    BONK...) is a DEX trade when the trader pasted its address or asked about a DEX
    market; otherwise it keeps its exchange path (and named exchanges always do)."""
    if primary is None:
        return None
    if primary.is_contract:
        return primary.identity
    if venue.kind == "cex" or not (venue.kind == "dex" or primary.source == "address"):
        return None
    entry = DEFAULT_REGISTRY.by_symbol(primary.label)
    if entry is None or entry.chain not in DEX_CHAINS or not entry.address:
        return None
    return AssetIdentity(symbol=entry.symbol, chain=entry.chain, address=entry.address)


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def _confidence(asset: object) -> "IdentityConfidence":
    source = getattr(asset, "source", None)
    confidence = getattr(asset, "confidence", None)
    if confidence == "exact":
        return "exact"
    if source == "discovered":
        return "discovered"
    return "registry" if asset is not None else "ticker_only"


def _reply(text: str, timings: dict[str, float], started: float) -> ChatResponse:
    """A short answer without analysis, e.g. asking which token the user means."""
    timings["total"] = _ms(started)
    analysis = Analysis(
        mock=False,
        summary=text,
        uncertainty=Uncertainty(level="high", notes=[]),
        disclaimer=DISCLAIMER,
        timings=timings,
    )
    return ChatResponse(message=ChatMessage(role="assistant", content=text), analysis=analysis)


def with_profile_agents(
    decision: RoutingDecision, identity: AssetIdentity | None, venue: str | None = None
) -> RoutingDecision:
    """Add agents the asset's profile calls for, for Solana tokens identified by mint:
    DEX market data (new DEX tokens, established memecoins) and on-chain safety data (when
    token authorities / holder concentration matter and a Solana RPC is configured). BTC
    and other assets without a Solana mint never get either."""
    if not decision.agents or _is_education(decision) or venue == "cex":
        return decision  # an exchange trade needs no DEX pool or token-safety data
    target: AssetIdentity | str | None = identity or (
        decision.assets[0] if decision.assets else None
    )
    profile = build_profile(target)
    if profile is None:
        return decision
    what = f"{profile.symbol} is a {profile.category_label} with a Solana mint"
    if wants_dex_data(profile):
        decision = add_agent(decision, "dex_market", f"{what}, so DEX pool data is relevant.")
    # A pasted mint is a Solana token by definition, before any data says what kind.
    exact_mint = identity is not None and identity.chain == "solana" and bool(identity.address)
    if wants_onchain_data(profile) or (exact_mint and "onchain" in integrated_capabilities()):
        decision = add_agent(
            decision,
            "onchain_safety",
            f"{what}: token authorities and holder concentration are required evidence.",
        )
    return decision


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
    dex = results.get("dex_market")
    if context.assets_source == "mint" and dex and dex.status == "ok" and not dex.mock:
        # The mint stays the identity; the symbol DEX data reports is just its label.
        snapshot = dex.findings.get("snapshot")
        symbol = snapshot.get("symbol") if isinstance(snapshot, dict) else None
        if isinstance(symbol, str) and symbol:
            updates["assets"] = [symbol]
        if context.trade is not None and isinstance(snapshot, dict):
            # The market is now known: the selected pool, its DEX and quote asset.
            market = context.trade.market.model_copy(
                update={
                    "pool": snapshot.get("pair_address"),
                    "venue": snapshot.get("dex"),
                    "quote_asset": snapshot.get("quote_symbol"),
                    "venue_kind": "dex",
                    "market_type": "dex",
                }
            )
            updates["trade"] = context.trade.model_copy(update={"market": market})
    return dataclasses.replace(context, **updates)  # type: ignore[arg-type]


def _is_education(decision: RoutingDecision) -> bool:
    return decision.agents == ["education"]


def _summary(decision: RoutingDecision, ok: list[AgentResult], assets: list[str]) -> str:
    if _is_education(decision):
        # The explanation is the whole reply.
        if ok:
            return ok[0].summary
        return (
            "I couldn't generate an explanation right now. Please try again in a moment, "
            "or ask about a specific coin for a live analysis."
        )
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
    real = [r.agent for r in analysis.agent_results if not r.mock and r.status == "ok"]
    mocked = [r.agent for r in analysis.agent_results if r.mock]
    notice = (
        "Prototype mode: results marked [Mock] are placeholders from agents that are not "
        f"built yet ({', '.join(mocked)})."
    )
    if real:
        notice += f" Real agents (live data or AI model): {', '.join(real)}."
    return notice


def _agent_line(result: AgentResult) -> str:
    if result.agent == "opportunity" and not result.mock:
        # The full decision block is shown above; list only the action line here.
        return f"Opportunity: {result.summary.splitlines()[0]}"
    return result.summary


def render_text(analysis: Analysis) -> str:
    """Plain-text version of the analysis for the chat UI (which renders text, not markdown)."""
    if not analysis.agent_results or analysis.agents_used == ["education"]:
        # No analysis pipeline ran: the reply is just the summary (or the explanation).
        return analysis.summary

    sections: list[str] = []
    if analysis.mock:
        sections.append(_mock_notice(analysis))
    # The decision leads: fast decision support is the point of the reply.
    decision = next(
        (
            r
            for r in analysis.agent_results
            if r.agent == "opportunity" and r.status == "ok" and not r.mock
        ),
        None,
    )
    if decision is not None:
        sections.append(decision.summary)
    sections.append(analysis.summary)
    sections.append(
        "Agents:\n"
        + "\n".join(
            f"• {_agent_line(r)}" if r.status == "ok" else f"• {r.agent}: failed ({r.error})"
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
