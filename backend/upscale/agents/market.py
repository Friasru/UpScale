import asyncio

from upscale.agents.base import Agent, AgentContext
from upscale.formatting import usd as _usd
from upscale.formatting import usd_compact as _usd_compact
from upscale.schemas import AgentResult, Risk
from upscale.services import market_data_service
from upscale.services.market_data import MarketDataError, MarketDataService, MarketSnapshot

MAX_ASSETS = 3
LARGE_MOVE_PCT = 10.0


class MarketAgent(Agent):
    """Live market data (price, 24h change/range/volume, market cap) for the requested assets.

    Uses whatever assets the router detected; it never guesses one. All I/O goes through
    `MarketDataService`, and values are only ever reported as the provider returned them.
    """

    name = "market"
    description = "Current price, 24h change, 24h high/low, 24h volume and market cap."
    # Waits for vision so a coin read from a screenshot can be looked up.
    depends_on = ("vision",)

    def __init__(self, service: MarketDataService | None = None):
        self.service = service or market_data_service

    async def run(self, context: AgentContext) -> AgentResult:
        provider = self.service.provider_name
        symbols = context.assets[:MAX_ASSETS]
        if not symbols:
            return AgentResult(
                agent=self.name,
                mock=False,
                summary="No specific cryptocurrency was identified, so no market data was requested.",
                findings={"provider": provider, "snapshots": [], "unavailable": []},
                evidence=["Name a coin (e.g. BTC, ETH, SOL) to get live market data."],
            )

        outcomes = await asyncio.gather(
            *(self.service.get_snapshot(s) for s in symbols), return_exceptions=True
        )
        snapshots: list[MarketSnapshot] = []
        unavailable: list[dict[str, str]] = []
        for symbol, outcome in zip(symbols, outcomes, strict=True):
            if isinstance(outcome, MarketSnapshot):
                snapshots.append(outcome)
            elif isinstance(outcome, MarketDataError):
                unavailable.append({"symbol": symbol, "reason": str(outcome)})
            else:
                raise outcome  # unexpected bug: let the orchestrator record the failure

        findings = {
            "provider": provider,
            "quote_currency": "USD",
            "snapshots": [s.model_dump(mode="json") for s in snapshots],
            "unavailable": unavailable,
        }
        missing = ", ".join(f"{u['symbol']} ({u['reason']})" for u in unavailable)
        if not snapshots:
            return AgentResult(
                agent=self.name,
                status="error",
                mock=False,
                summary=f"Live market data could not be retrieved for {', '.join(symbols)}.",
                findings=findings,
                error=missing,
            )

        risks = [
            Risk(
                description=(
                    f"{s.symbol} moved {s.change_24h_pct:+.1f}% in 24h, indicating high volatility."
                ),
                severity="medium",
            )
            for s in snapshots
            if s.change_24h_pct is not None and abs(s.change_24h_pct) >= LARGE_MOVE_PCT
        ]
        if unavailable:
            risks.append(
                Risk(description=f"Live market data unavailable for {missing}.", severity="medium")
            )
        risks.append(
            Risk(
                description=f"Market data is a point-in-time snapshot from {provider} and may lag.",
                severity="low",
            )
        )
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=f"Live market data from {provider} for {', '.join(s.symbol for s in snapshots)}.",
            findings=findings,
            evidence=[line for s in snapshots for line in _describe(s)],
            risks=risks,
        )


def _describe(s: MarketSnapshot) -> list[str]:
    as_of = (s.last_updated or s.fetched_at).strftime("%Y-%m-%d %H:%M UTC")
    headline = f"{s.name} ({s.symbol}) price: {_usd(s.price_usd)}"
    if s.change_24h_pct is not None:
        headline += f", {s.change_24h_pct:+.2f}% over 24h"
    lines = [f"{headline} ({s.provider}, as of {as_of})."]
    details = []
    if s.low_24h_usd is not None and s.high_24h_usd is not None:
        details.append(f"24h range {_usd(s.low_24h_usd)} – {_usd(s.high_24h_usd)}")
    if s.volume_24h_usd is not None:
        details.append(f"24h volume {_usd_compact(s.volume_24h_usd)}")
    if s.market_cap_usd is not None:
        details.append(f"market cap {_usd_compact(s.market_cap_usd)}")
    if details:
        lines.append(f"{s.symbol} " + "; ".join(details) + ".")
    return lines
