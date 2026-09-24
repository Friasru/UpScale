import asyncio
import json

import httpx2
import pytest

from upscale.agents import AgentContext, MarketAgent
from upscale.orchestrator import Orchestrator
from upscale.schemas import ChatMessage, ChatRequest
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.market_data import MarketDataService

from .conftest import FakeCoinGecko, coingecko_row


@pytest.fixture
def fake() -> FakeCoinGecko:
    return FakeCoinGecko()


@pytest.fixture
def agent(fake) -> MarketAgent:
    return MarketAgent(MarketDataService(CoinGeckoProvider(transport=fake.transport())))


def run(agent: MarketAgent, *assets: str):
    return asyncio.run(agent.run(AgentContext(query="", assets=list(assets))))


def test_market_agent_reports_live_data(agent):
    result = run(agent, "ETH")
    assert result.status == "ok"
    assert result.mock is False
    assert "CoinGecko" in result.summary and "ETH" in result.summary
    [snapshot] = result.findings["snapshots"]
    assert snapshot["symbol"] == "ETH"
    assert snapshot["price_usd"] == 3_100.0
    assert result.findings["unavailable"] == []
    assert result.evidence[0].startswith(
        "Ethereum (ETH) price: $3,100.00, +1.50% over 24h (CoinGecko"
    )
    assert "24h range" in result.evidence[1] and "market cap" in result.evidence[1]


def test_market_agent_handles_multiple_assets(agent, fake):
    result = run(agent, "BTC", "SOL")
    assert [s["symbol"] for s in result.findings["snapshots"]] == ["BTC", "SOL"]
    assert len(fake.requests) == 2


def test_market_agent_without_asset_makes_no_request(agent, fake):
    result = run(agent)
    assert result.status == "ok"
    assert result.findings["snapshots"] == []
    assert fake.requests == []


def test_market_agent_api_failure_returns_error_without_values(agent, fake):
    fake.handler = lambda r: httpx2.Response(500)
    result = run(agent, "BTC")
    assert result.status == "error"
    assert result.mock is False
    assert "could not be retrieved" in result.summary
    assert "HTTP 500" in result.error
    assert result.findings["snapshots"] == []
    assert result.evidence == []


def test_market_agent_unknown_asset(agent, fake):
    fake.handler = lambda r: httpx2.Response(200, content=b"[]")
    result = run(agent, "NOTACOIN")
    assert result.status == "error"
    assert "not recognized" in result.error
    assert result.findings["unavailable"][0]["symbol"] == "NOTACOIN"


def test_market_agent_partial_failure_keeps_available_data(agent):
    result = run(agent, "BTC", "NOTACOIN")
    assert result.status == "ok"
    assert [s["symbol"] for s in result.findings["snapshots"]] == ["BTC"]
    assert any("NOTACOIN" in r.description for r in result.risks)


def test_market_agent_flags_large_moves(agent, fake):
    row = coingecko_row("solana", "sol", "Solana", 150.0, price_change_percentage_24h=-12.3)
    fake.handler = lambda r: httpx2.Response(200, content=json.dumps([row]))
    result = run(agent, "SOL")
    assert any("-12.3%" in r.description for r in result.risks)


def test_market_agent_formats_small_prices(agent, fake):
    row = coingecko_row("pepe", "pepe", "Pepe", 0.00001234)
    fake.handler = lambda r: httpx2.Response(200, content=json.dumps([row]))
    assert "$0.00001234" in run(agent, "PEPE").evidence[0]


# --- Through the orchestrator and API (uses the autouse fake_coingecko fixture) --------


def ask(content: str):
    request = ChatRequest(messages=[ChatMessage(role="user", content=content)])
    return asyncio.run(Orchestrator().respond(request))


def test_orchestrator_includes_live_market_data(fake_coingecko):
    analysis = ask("What's up with SOL?").analysis
    market = next(r for r in analysis.agent_results if r.agent == "market")
    assert market.status == "ok" and market.mock is False
    assert any(e.source == "market" and "$150.00" in e.statement for e in analysis.evidence)
    assert analysis.summary.startswith("[Partly mock]")
    markets = [r for r in fake_coingecko.requests if r.url.path.endswith("/coins/markets")]
    assert markets[0].url.params["ids"] == "solana"


def test_orchestrator_continues_when_market_agent_fails(fake_coingecko):
    fake_coingecko.handler = lambda r: httpx2.Response(503)
    response = ask("What's up with BTC?")
    analysis = response.analysis
    by_agent = {r.agent: r for r in analysis.agent_results}
    assert by_agent["market"].status == "error"
    # Technical analysis uses the same (failing) provider; everything else still runs.
    assert by_agent["technical_analysis"].status == "error"
    assert all(
        r.status == "ok"
        for name, r in by_agent.items()
        if name not in ("market", "technical_analysis")
    )
    assert analysis.scenarios and analysis.risks
    assert not any(e.source == "market" for e in analysis.evidence)
    assert any("market agent failed" in n for n in analysis.uncertainty.notes)
    assert any("Live market data failed" in r.description for r in analysis.risks)
    assert "market: failed" in response.message.content


def test_chat_endpoint_with_market_outage(client, fake_coingecko):
    fake_coingecko.handler = lambda r: httpx2.Response(500)
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "ETH price?"}]})
    assert response.status_code == 200
    market = next(r for r in response.json()["analysis"]["agent_results"] if r["agent"] == "market")
    assert market["status"] == "error"
    assert "could not be retrieved" in market["summary"]


def test_chat_endpoint_caches_market_data(client, fake_coingecko):
    for _ in range(3):
        client.post("/chat", json={"messages": [{"role": "user", "content": "BTC price"}]})
    assert len(fake_coingecko.requests) == 1


def test_default_orchestrator_uses_live_market_agent():
    assert isinstance(Orchestrator().agents["market"], MarketAgent)
