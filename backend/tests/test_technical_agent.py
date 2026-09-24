import asyncio
import json

import httpx2
import pytest

from upscale.agents import AgentContext, TechnicalAnalysisAgent
from upscale.orchestrator import Orchestrator
from upscale.schemas import AgentResult, ChatMessage, ChatRequest
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.market_data import Candle, CandleSeries, MarketDataService, Timeframe
from upscale.services.technical_analysis import TechnicalAnalysisService

from .conftest import FOUR_HOURS_MS, LAST_CLOSE_MS, FakeCoinGecko
from .ta_helpers import make_series, wave

BUY_SELL_WORDS = (
    "buy", "sell", "long", "short", "entry", "target", "take profit", "stop loss", "should",
)  # fmt: skip


def wave_rows(count: int) -> list[list[float]]:
    """CoinGecko-format rows for an oscillating price, so levels and indicators all exist."""
    closes = wave(count, mid=100, amplitude=10, period=20)
    rows = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i else close
        ts = LAST_CLOSE_MS - (count - 1 - i) * FOUR_HOURS_MS
        rows.append([ts, open_, max(open_, close) + 0.2, min(open_, close) - 0.2, close])
    return rows


def serve_wave(fake: FakeCoinGecko, count: int = 180) -> None:
    def handler(request):
        if request.url.path.endswith("/ohlc"):
            return httpx2.Response(200, content=json.dumps(wave_rows(count)))
        return fake.markets(request)

    fake.handler = handler


@pytest.fixture
def fake() -> FakeCoinGecko:
    return FakeCoinGecko()


@pytest.fixture
def agent(fake) -> TechnicalAnalysisAgent:
    market = MarketDataService(CoinGeckoProvider(transport=fake.transport()))
    return TechnicalAnalysisAgent(TechnicalAnalysisService(market))


def run(agent, *assets, timeframe=None, prior=None, attachments=None) -> AgentResult:
    context = AgentContext(
        query="",
        assets=list(assets),
        timeframe=timeframe,
        prior_results=prior or {},
        attachments=attachments or [],
    )
    return asyncio.run(agent.run(context))


# --- Technical agent -------------------------------------------------------------------------


def test_technical_agent_turns_calculations_into_evidence(agent, fake):
    serve_wave(fake)
    result = run(agent, "BTC")
    assert result.status == "ok"
    assert result.mock is False
    assert result.summary.startswith("Technical analysis of BTC 4h from 180 CoinGecko candles")

    f = result.findings
    assert (f["symbol"], f["timeframe"], f["provider"], f["candle_count"]) == (
        "BTC",
        "4h",
        "CoinGecko",
        180,
    )
    assert f["volume_available"] is False
    assert {i["name"] for i in f["indicators"]} == {
        "SMA 20", "SMA 50", "EMA 20", "EMA 50", "RSI 14", "MACD 12/26/9",
    }  # fmt: skip
    assert f["levels"]["support"] and f["levels"]["resistance"]

    text = "\n".join(result.evidence)
    for expected in (
        "BTC 4h: 180 candles from CoinGecko",
        "SMA 20: $",
        "EMA 50: $",
        "RSI 14: ",
        "MACD 12/26/9: line",
        "50-candle range: low $",
        "Trend (rule-based):",
        "Approximate support: ~$",
        "does not supply per-candle volume",
    ):
        assert expected in text
    assert "unavailable" not in text


def test_technical_agent_scenarios_are_conditional_not_recommendations(agent, fake):
    serve_wave(fake)
    result = run(agent, "BTC")
    names = [s.name for s in result.scenarios]
    assert names == ["Range holds", "Break above resistance", "Break below support"]
    assert all(s.invalidation and s.conditions for s in result.scenarios)
    text = " ".join(
        f"{s.name} {s.description} {' '.join(s.conditions)} {s.invalidation}"
        for s in result.scenarios
    ).lower()
    assert not any(word in text for word in BUY_SELL_WORDS)
    assert "recommend" not in " ".join(result.evidence).lower()


def test_technical_agent_risks_mention_lag_levels_and_volume(agent, fake):
    serve_wave(fake)
    descriptions = " ".join(r.description for r in run(agent, "BTC").risks)
    assert "lag" in descriptions
    assert "approximate" in descriptions
    assert "volume" in descriptions


def test_technical_agent_requests_the_asked_for_asset(agent, fake):
    serve_wave(fake)
    run(agent, "SOL")
    assert fake.requests[0].url.path == "/api/v3/coins/solana/ohlc"


def test_market_data_failure_means_no_analysis(agent, fake):
    fake.handler = lambda r: httpx2.Response(500)
    result = run(agent, "BTC")
    assert result.status == "error"
    assert result.mock is False
    assert (
        result.summary
        == "Technical analysis could not be performed for BTC: CoinGecko returned HTTP 500."
    )
    assert result.findings == {"symbol": "BTC"}
    assert result.evidence == [] and result.scenarios == []


def test_rate_limited_or_timed_out_market_data(agent, fake):
    fake.handler = lambda r: httpx2.Response(429)
    assert "rate limit" in run(agent, "BTC").error


def test_unknown_asset(agent, fake):
    result = run(agent, "NOTACOIN")
    assert result.status == "error"
    assert "not recognized" in result.error


def test_no_asset_means_no_request(agent, fake):
    result = run(agent)
    assert result.status == "ok"
    assert "No specific cryptocurrency" in result.summary
    assert fake.requests == []


def test_insufficient_history_is_explained(agent, fake):
    serve_wave(fake, count=30)
    result = run(agent, "BTC")
    assert result.status == "ok"
    text = "\n".join(result.evidence)
    assert "SMA 50 unavailable: needs 50 candles, only 30 available." in text
    assert "MACD 12/26/9 unavailable: needs 34 candles, only 30 available." in text
    assert "Trend unavailable" in text and "Support/resistance unavailable" in text
    assert "SMA 20: $" in text
    assert result.scenarios == []  # no levels, so no level-based scenarios
    assert any(
        "Not enough history for: SMA 50, EMA 50, MACD 12/26/9" in r.description
        for r in result.risks
    )


def test_unsupported_timeframe_falls_back_and_says_so(agent, fake):
    serve_wave(fake)
    result = run(agent, "BTC", timeframe="1h")
    assert result.findings["requested_timeframe"] == "1h"
    assert result.findings["timeframe"] == "4h"
    note = result.findings["timeframe_note"]
    assert note.startswith("1h candles are not available")
    assert note in result.evidence
    assert any(r.description == note for r in result.risks)


def test_screenshot_timeframe_fallback_names_the_screenshot(agent, fake):
    serve_wave(fake)
    context = AgentContext(query="", assets=["BTC"], timeframe="1h", timeframe_source="screenshot")
    result = asyncio.run(agent.run(context))
    assert result.findings["requested_timeframe"] == "1h"
    assert result.findings["requested_timeframe_source"] == "screenshot"
    assert result.findings["timeframe"] == "4h"
    assert result.findings["timeframe_note"].startswith(
        "The screenshot's timeframe is 1h, but 1h candles are not available"
    )


def test_unrecognized_screenshot_timeframe_is_preserved(agent, fake):
    serve_wave(fake)
    context = AgentContext(query="", assets=["BTC"], timeframe="1W", timeframe_source="screenshot")
    result = asyncio.run(agent.run(context))
    assert result.findings["requested_timeframe"] == "1W"
    assert "1W candles are not available" in result.findings["timeframe_note"]


def test_only_the_primary_asset_is_analyzed(agent, fake):
    serve_wave(fake)
    result = run(agent, "BTC", "ETH")
    assert result.findings["not_analyzed"] == ["ETH"]
    assert "not analyzed: ETH" in result.evidence[-1]
    assert len(fake.requests) == 1


class BrokenCandles:
    """A provider that returns internally inconsistent candles."""

    name = "Broken"
    supported_timeframes: frozenset[Timeframe] = frozenset({"4h"})

    async def fetch_snapshot(self, symbol):  # pragma: no cover
        raise AssertionError

    async def fetch_candles(self, symbol, timeframe, limit) -> CandleSeries:
        series = make_series(wave(60))
        candles = list(series.candles)
        candles[5] = Candle.model_construct(**{**candles[5].model_dump(), "high": 1.0})
        return series.model_copy(update={"candles": candles})


def test_invalid_candles_mean_no_analysis():
    agent = TechnicalAnalysisAgent(TechnicalAnalysisService(MarketDataService(BrokenCandles())))
    result = run(agent, "BTC")
    assert result.status == "error"
    assert "inconsistent high/low" in result.error


# --- Orchestrator and API (use the autouse fake_coingecko fixture) -----------------------------


def ask(content: str):
    request = ChatRequest(messages=[ChatMessage(role="user", content=content)])
    return asyncio.run(Orchestrator().respond(request))


def test_orchestrator_uses_real_technical_analysis(fake_coingecko):
    serve_wave(fake_coingecko)
    analysis = ask("BTC RSI and support levels").analysis
    assert analysis.agents_used == ["technical_analysis", "risk"]
    technical = analysis.agent_results[0]
    assert technical.status == "ok" and technical.mock is False
    assert any(
        e.source == "technical_analysis" and e.statement.startswith("RSI 14:")
        for e in analysis.evidence
    )
    assert {s.source for s in analysis.scenarios} == {"technical_analysis"}
    assert analysis.summary.startswith("[Partly mock]")


def test_general_question_runs_technical_and_market_together(fake_coingecko):
    serve_wave(fake_coingecko)
    analysis = ask("What's up with ETH?").analysis
    by_agent = {r.agent: r for r in analysis.agent_results}
    assert by_agent["technical_analysis"].status == "ok"
    assert by_agent["market"].status == "ok"
    assert by_agent["opportunity"].mock and by_agent["risk"].mock  # still mocks
    assert "technical_analysis" in by_agent["risk"].findings["reviewed_agents"]


def test_orchestrator_continues_when_market_data_is_down(fake_coingecko):
    fake_coingecko.handler = lambda r: httpx2.Response(503)
    response = ask("What's up with BTC?")
    analysis = response.analysis
    by_agent = {r.agent: r for r in analysis.agent_results}
    assert by_agent["technical_analysis"].status == "error"
    assert by_agent["market"].status == "error"
    assert by_agent["news_sentiment"].status == "ok"
    assert not any(e.source == "technical_analysis" for e in analysis.evidence)
    assert any("technical_analysis agent failed" in n for n in analysis.uncertainty.notes)
    assert "technical_analysis: failed" in response.message.content


def test_timeframe_in_the_question_reaches_the_agent(fake_coingecko):
    serve_wave(fake_coingecko)
    analysis = ask("SOL 1h chart").analysis
    technical = next(r for r in analysis.agent_results if r.agent == "technical_analysis")
    assert technical.findings["requested_timeframe"] == "1h"
    assert technical.findings["timeframe"] == "4h"


def test_chat_endpoint_with_technical_analysis(client, fake_coingecko):
    serve_wave(fake_coingecko)
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "ETH macd"}]})
    assert response.status_code == 200
    content = response.json()["message"]["content"]
    assert "MACD 12/26/9: line" in content
    assert "Break above resistance" in content


def test_screenshot_without_asset_does_not_analyze(
    client, fake_coingecko, fake_vision, png_attachment
):
    fake_vision.reading = fake_vision.reading.model_copy(
        update={
            "asset": fake_vision.reading.asset.model_copy(
                update={"symbol": None, "pair": None, "basis": "unknown"}
            )
        }
    )
    message = {"role": "user", "content": "", "attachments": [png_attachment]}
    response = client.post("/chat", json={"messages": [message]})
    technical = response.json()["analysis"]["agent_results"][1]
    assert technical["agent"] == "technical_analysis"
    assert "No specific cryptocurrency" in technical["summary"]
    assert fake_coingecko.requests == []
