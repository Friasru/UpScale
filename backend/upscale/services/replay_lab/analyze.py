"""Analyze at historical time T, through the production pipeline.

This mirrors `Orchestrator.respond` for an exact token handed over by Scout (the
`ChatRequest.asset` path), step for step and with the same functions: exact resolution,
routing, the asset profile, the trade context, the dependency-ordered agent run and
`synthesize`; the decision is extracted with the live outcome module's
`decision_from_response`. The differences are only where data comes from:

* `DexMarketAgent` gets a `DexMarketService` whose provider returns the point-in-time
  pool (production pool selection still runs), with `now` = T;
* `TechnicalAnalysisAgent` gets a registry serving the pool's candles closed by T;
* `RiskAgent` and `OpportunityAgent` are unchanged (they only read the other agents);
* agents whose evidence has no historical source (on-chain safety, news, exchange market
  data, vision) are not run: they're reported unavailable, and Risk / Opportunity apply
  their production rules for missing evidence, as they do live when a provider is down;
* the whole run happens under `frozen_now(T)`, so anything measuring age against "now"
  (e.g. the asset profile's pool age) sees T.

The live `respond` also prefetches current DEX / on-chain data from the global registry;
replay deliberately doesn't (current provider state must never reach a historical run).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from upscale.agents import DexMarketAgent, OpportunityAgent, RiskAgent, TechnicalAnalysisAgent
from upscale.agents.base import AgentContext
from upscale.orchestrator import (
    Orchestrator,
    _confidence,
    _trade_identity,
    apply_vision,
    render_text,
    with_profile_agents,
)
from upscale.routing import route
from upscale.schemas import AgentResult, AssetRef, ChatMessage, ChatRequest, ChatResponse
from upscale.services.asset_profile import build_profile
from upscale.services.asset_resolver import AssetResolver
from upscale.services.chains import chain_label, same_address
from upscale.services.clock import frozen_now
from upscale.services.market_data import MarketDataUnavailableError, Timeframe
from upscale.services.outcomes.models import DecisionObservation
from upscale.services.outcomes.observe import decision_from_response
from upscale.services.replay_lab.candles import (
    PointInTimeCandles,
    PointInTimeCandleSource,
    PointInTimeRegistry,
)
from upscale.services.replay_lab.clock import HistoricalClock
from upscale.services.replay_lab.models import AgentOutput
from upscale.services.solana_dex import DexMarketService, DexPool
from upscale.services.trade_context import build_trade_context, requested_venue, trader_context

UNAVAILABLE_AGENTS = {
    "onchain_safety": "token authorities and holder concentration at T have no historical source",
    "news_sentiment": "news available at T can't be reconstructed reliably",
    "market": "exchange market data (CoinGecko) isn't used for DEX tokens",
    "vision": "no screenshot in a historical replay",
    "education": "not part of a decision",
}


class PointInTimePoolProvider:
    """A `DexPoolProvider` that knows only the pools recorded at T."""

    def __init__(self, name: str, pools: list[DexPool], clock: HistoricalClock):
        self.name = name
        self._pools = pools
        self._clock = clock

    async def fetch_token_pools(self, chain: str, token_address: str) -> list[DexPool]:
        self._clock.check_decision_phase("a point-in-time pool lookup")
        return [
            p
            for p in self._pools
            if p.chain == chain and same_address(chain, p.base.address, token_address)
        ]

    async def search_pools(self, query: str) -> list[DexPool]:
        raise MarketDataUnavailableError("historical replay can't search pools by text")


@dataclass
class AnalyzeAtResult:
    request: ChatRequest
    response: ChatResponse
    decision: DecisionObservation | None
    agents: dict[str, AgentOutput]
    unavailable: dict[str, str]
    routed: list[str]
    candle_requests: list[tuple[str, Timeframe, int]] = field(default_factory=list)


def scout_query(symbol: str | None, token: str, chain: str) -> str:
    """The text Scout's Analyze sends (app/src/lib/scout.ts)."""
    return f"Analyze {symbol or token}: {token} on {chain_label(chain)}"


async def analyze_at(
    clock: HistoricalClock,
    *,
    chain: str,
    token: str,
    symbol: str | None,
    name: str | None,
    pool: DexPool | None,
    market_provider: str,
    candles: PointInTimeCandles,
) -> AnalyzeAtResult:
    clock.check_decision_phase("Analyze")
    t: datetime = clock.decision_at
    query = scout_query(symbol, token, chain)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=query)],
        asset=AssetRef(
            chain=chain,
            address=token,
            symbol=symbol,
            name=name,
            pool_address=pool.pair_address if pool else None,
        ),
    )
    dex_service = DexMarketService(
        PointInTimePoolProvider(market_provider, [pool] if pool else [], clock),
        cache_ttl=0.0,
        not_found_ttl=0.0,
        max_calls_per_minute=10_000,
        now=lambda: t,
    )
    source = PointInTimeCandleSource(candles)
    agents = [
        DexMarketAgent(service=dex_service),
        TechnicalAnalysisAgent(registry=PointInTimeRegistry(source)),
        RiskAgent(),
        OpportunityAgent(),
    ]
    resolver = AssetResolver(search=None)
    orchestrator = Orchestrator(agents=agents, resolver=resolver)
    with frozen_now(t):
        # Orchestrator.respond, the `request.asset` path (no network prefetch).
        resolution = resolver.exact(chain, token, query)
        decision = route(query, False, resolution)
        primary = resolution.primary
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
            trader_context(query, [], decision.timeframe),
            decision.timeframe,
            venue=venue,
        )
        context = AgentContext(
            query=query,
            attachments=[],
            history=[],
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
        results = await orchestrator.run_agents(decision.agents, context)
        final = apply_vision(context, {r.agent: r for r in results})
        analysis = orchestrator.synthesize(decision, results, assets=final.assets)
    response = ChatResponse(
        message=ChatMessage(role="assistant", content=render_text(analysis)), analysis=analysis
    )
    clock.check_decision_phase("decision extraction")
    observation = decision_from_response(request, response, t)
    unavailable: dict[str, str] = {
        str(name): UNAVAILABLE_AGENTS.get(name, "no historical source")
        for name in decision.agents
        if name not in orchestrator.agents
    }
    return AnalyzeAtResult(
        request=request,
        response=response,
        decision=observation,
        agents={r.agent: _output(r) for r in results},
        unavailable=unavailable,
        routed=list(decision.agents),
        candle_requests=list(source.requests),
    )


def _output(r: AgentResult) -> AgentOutput:
    findings: dict[str, Any] = dict(r.findings)
    return AgentOutput(
        agent=r.agent, status=r.status, summary=r.summary, findings=findings, error=r.error
    )
