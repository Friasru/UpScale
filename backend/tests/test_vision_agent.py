import asyncio
import base64

import pytest

from upscale.agents import AgentContext, VisionAgent
from upscale.orchestrator import Orchestrator
from upscale.schemas import ChatMessage, ChatRequest, ImageAttachment
from upscale.services.vision import (
    ChartReading,
    MalformedVisionOutputError,
    VisionRefusedError,
    VisionService,
    VisionUnavailableError,
)

from .conftest import CHART_PNG, FakeVisionModel, chart_reading_data, make_png
from .test_technical_agent import serve_wave

BUY_SELL_WORDS = ("buy", "sell", "go long", "go short", "take profit", "stop loss", "recommend")
UNKNOWN_ASSET = {
    "symbol": None,
    "pair": None,
    "exchange": None,
    "basis": "unknown",
    "evidence": None,
}
UNKNOWN_TIMEFRAME = {"label": None, "basis": "unknown", "evidence": None}


def chart(name="chart.png", data=CHART_PNG) -> ImageAttachment:
    return ImageAttachment(name=name, media_type="image/png", data=data)


@pytest.fixture
def model() -> FakeVisionModel:
    return FakeVisionModel()


@pytest.fixture
def agent(model) -> VisionAgent:
    return VisionAgent(VisionService(model))


def run(agent, *images, assets=(), assets_source=None):
    context = AgentContext(
        query="", attachments=list(images), assets=list(assets), assets_source=assets_source
    )
    return asyncio.run(agent.run(context))


# --- Vision agent -------------------------------------------------------------------------


def test_successful_reading_becomes_structured_evidence(agent):
    result = run(agent, chart())
    assert result.status == "ok" and result.mock is False
    assert result.summary == "Screenshot shows a candlestick chart of BTC/USDT (4h)."
    assert result.findings["detected_asset"] == "BTC"
    assert result.findings["detected_timeframe"] == "4h"
    assert result.findings["detected_timeframe_label"] == "4h"
    [chart_finding] = result.findings["charts"]
    assert chart_finding["reading"]["indicators"][0]["values"] == [{"label": None, "value": 58.2}]

    text = "\n".join(result.evidence)
    for expected in (
        "chart.png: asset BTC/USDT (visible), timeframe 4h = 4h (visible), candlestick chart.",
        "Price shown on the screenshot: $64,210.50 (read from the screenshot, not live data).",
        "Indicator on screenshot: RSI 14: 58.2 (read from the screenshot, not live data).",
        "Support marked on screenshot near $62,000.00 (visible; drawn line).",
        "Resistance marked on screenshot near $66,000.00 (visible; price label).",
        "User-drawn horizontal level 'entry zone' at $63,000.00.",
        "Channel (rising, visible): Parallel rising channel drawn from the recent lows.",
        "Pattern visible: ascending triangle",
        "Visual observation: Last three candles have long upper wicks.",
        "Could not determine: Volume pane is cropped.",
    ):
        assert expected in text


def test_invisible_items_are_not_reported(agent):
    text = "\n".join(run(agent, chart()).evidence)
    assert "Indicator on screenshot: EMA" not in text
    assert "Pattern visible: double top" not in text
    assert "Possible double top, but not clear enough to confirm" in text
    assert "Discarded from vision output: Dropped indicator 'EMA'" in text


def test_no_indicator_values_are_invented(agent, model):
    data = chart_reading_data(
        indicators=[
            {
                "name": "MACD",
                "settings": "12 26 9",
                "values": [],
                "basis": "visible",
                "evidence": "pane",
            }
        ]
    )
    model.reading = ChartReading.model_validate(data)
    text = "\n".join(run(agent, chart()).evidence)
    assert "Indicator on screenshot: MACD 12 26 9 (no value printed)" in text


def test_unknown_asset_and_timeframe(agent, model):
    model.reading = ChartReading.model_validate(
        chart_reading_data(
            asset=UNKNOWN_ASSET,
            timeframe=UNKNOWN_TIMEFRAME,
            displayed_price={"value": None, "basis": "unknown", "evidence": None},
        )
    )
    result = run(agent, chart())
    assert result.status == "ok"
    assert result.findings["detected_asset"] is None
    assert result.findings["detected_timeframe"] is None
    assert (
        result.summary
        == "Screenshot shows a candlestick chart of unknown asset (unknown timeframe)."
    )
    assert result.evidence[0] == "chart.png: asset unknown, timeframe unknown, candlestick chart."
    assert not any("Price shown" in e for e in result.evidence)


def test_not_a_chart(agent, model):
    model.reading = ChartReading.model_validate(chart_reading_data(is_price_chart=False))
    result = run(agent, chart())
    assert result.status == "ok"
    assert result.summary == "The screenshot does not appear to be a price chart."
    assert result.findings["detected_asset"] is None
    assert result.evidence[0] == "chart.png: not recognized as a price chart."


def test_no_screenshot(agent, model):
    result = run(agent)
    assert result.status == "ok"
    assert "No screenshot" in result.summary
    assert model.calls == []


def test_invalid_image_fails_cleanly(agent, model):
    tiny = chart("tiny.png", base64.b64encode(make_png(4, 4)).decode())
    result = run(agent, tiny)
    assert result.status == "error"
    assert result.summary == "The screenshot could not be analyzed."
    assert "tiny.png: tiny.png is too small to read" in result.error
    assert model.calls == []


@pytest.mark.parametrize(
    "error",
    [
        VisionUnavailableError("the vision model is not configured; set ANTHROPIC_API_KEY"),
        VisionUnavailableError("the vision model timed out"),
        VisionRefusedError("the vision model declined to analyze this image"),
        MalformedVisionOutputError(
            "the vision model's output did not match the expected chart schema"
        ),
    ],
)
def test_model_failures_fail_cleanly(agent, model, error):
    model.error = error
    result = run(agent, chart())
    assert result.status == "error"
    assert str(error) in result.error
    assert result.evidence == [] and result.findings["detected_asset"] is None


def test_partial_failure_keeps_readable_screenshots(agent):
    tiny = chart("tiny.png", base64.b64encode(make_png(4, 4)).decode())
    result = run(agent, chart(), tiny)
    assert result.status == "ok"
    assert result.findings["failed"][0]["image"] == "tiny.png"
    assert "Read 1 of 2 screenshots." in result.summary
    assert any(
        "Some screenshots could not be read: tiny.png" in r.description for r in result.risks
    )


def test_asset_mismatch_with_the_users_question(agent):
    result = run(agent, chart(), assets=["ETH"], assets_source="user")
    note = "The screenshot shows BTC, but you asked about ETH; live analysis uses ETH."
    assert note in result.evidence
    assert any(r.description == note for r in result.risks)


def test_vision_never_recommends(agent):
    result = run(agent, chart())
    text = " ".join(
        [result.summary, *result.evidence, *(r.description for r in result.risks)]
    ).lower()
    # "entry zone" is the user's own drawn label; everything else must be neutral.
    text = text.replace("'entry zone'", "")
    assert not any(word in text for word in BUY_SELL_WORDS)
    assert result.scenarios == []


def test_screenshot_values_are_flagged_as_not_live(agent):
    risks = " ".join(r.description for r in run(agent, chart()).risks)
    assert "live data takes precedence" in risks


# --- Orchestrator integration (autouse fake_vision + fake_coingecko) ---------------------------------


def ask(content: str, *images: ImageAttachment):
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=content, attachments=list(images))]
    )
    return asyncio.run(Orchestrator().respond(request))


def by_agent(response):
    return {r.agent: r for r in response.analysis.agent_results}


def test_screenshot_asset_and_timeframe_flow_downstream(fake_coingecko):
    serve_wave(fake_coingecko)
    response = ask("", chart())
    results = by_agent(response)
    assert response.analysis.agents_used == ["vision", "technical_analysis", "market", "risk"]
    assert response.analysis.assets == ["BTC"]
    assert results["vision"].status == "ok"
    technical = results["technical_analysis"]
    assert technical.status == "ok"
    assert technical.findings["symbol"] == "BTC"
    assert technical.findings["requested_timeframe"] == "4h"
    assert technical.findings["requested_timeframe_source"] == "screenshot"
    assert technical.findings["timeframe_note"] is None
    market = results["market"]
    assert market.findings["snapshots"][0]["symbol"] == "BTC"
    paths = {r.url.path for r in fake_coingecko.requests}
    assert paths == {"/api/v3/coins/bitcoin/ohlc", "/api/v3/coins/markets"}


def test_combined_evidence_keeps_visual_and_live_values_apart(fake_coingecko):
    serve_wave(fake_coingecko)
    analysis = ask("", chart()).analysis
    sources = {e.source for e in analysis.evidence}
    assert {"vision", "technical_analysis", "market"} <= sources
    vision_rsi = next(
        e.statement
        for e in analysis.evidence
        if e.statement.startswith("Indicator on screenshot: RSI")
    )
    live_rsi = next(e.statement for e in analysis.evidence if e.statement.startswith("RSI 14:"))
    assert "not live data" in vision_rsi
    assert "58.2" in vision_rsi and "58.2" not in live_rsi  # Technical computes its own RSI


def test_unsupported_screenshot_timeframe_falls_back_and_is_preserved(fake_coingecko, fake_vision):
    serve_wave(fake_coingecko)
    fake_vision.reading = ChartReading.model_validate(
        chart_reading_data(timeframe={"label": "15", "basis": "visible", "evidence": "toolbar"})
    )
    response = ask("", chart())
    technical = by_agent(response)["technical_analysis"]
    assert technical.findings["requested_timeframe"] == "15m"
    assert technical.findings["timeframe"] == "4h"
    note = technical.findings["timeframe_note"]
    assert note.startswith("The screenshot's timeframe is 15m, but 15m candles are not available")
    assert note in response.message.content


def test_user_timeframe_and_asset_win_over_the_screenshot(fake_coingecko):
    serve_wave(fake_coingecko)
    response = ask("ETH on the daily", chart())
    results = by_agent(response)
    assert response.analysis.assets == ["ETH"]
    technical = results["technical_analysis"]
    assert technical.findings["symbol"] == "ETH"
    assert technical.findings["requested_timeframe"] == "1d"
    assert technical.findings["requested_timeframe_source"] == "user"
    assert any("shows BTC, but you asked about ETH" in e for e in results["vision"].evidence)


def test_vision_failure_does_not_break_the_response(fake_vision, fake_coingecko):
    fake_vision.error = VisionUnavailableError(
        "the vision model is not configured; set ANTHROPIC_API_KEY"
    )
    response = ask("", chart())
    results = by_agent(response)
    assert results["vision"].status == "error"
    assert results["technical_analysis"].status == "ok"
    assert "No specific cryptocurrency" in results["technical_analysis"].summary
    assert results["risk"].status == "ok"
    assert any("vision agent failed" in n for n in response.analysis.uncertainty.notes)
    assert "vision: failed" in response.message.content
    assert fake_coingecko.requests == []


def test_vision_failure_with_named_asset_still_gets_live_analysis(fake_vision, fake_coingecko):
    serve_wave(fake_coingecko)
    fake_vision.error = MalformedVisionOutputError(
        "the vision model's output did not match the expected chart schema"
    )
    results = by_agent(ask("SOL chart", chart()))
    assert results["vision"].status == "error"
    assert results["technical_analysis"].findings["symbol"] == "SOL"
    assert results["market"].status == "ok"


def test_slow_vision_times_out_without_blocking_everything(fake_vision, fake_coingecko):
    class QuickVision(VisionAgent):
        timeout = 0.05

    serve_wave(fake_coingecko)
    fake_vision.delay = 1.0
    orchestrator = Orchestrator()
    orchestrator.agents["vision"] = QuickVision()
    request = ChatRequest(messages=[ChatMessage(role="user", content="BTC", attachments=[chart()])])
    results = {
        r.agent: r for r in asyncio.run(orchestrator.respond(request)).analysis.agent_results
    }
    assert results["vision"].error == "timed out"
    assert results["technical_analysis"].status == "ok"


def test_unknown_screenshot_asset_means_no_live_lookup(fake_vision, fake_coingecko):
    fake_vision.reading = ChartReading.model_validate(chart_reading_data(asset=UNKNOWN_ASSET))
    response = ask("", chart())
    assert response.analysis.assets == []
    assert "No specific cryptocurrency" in by_agent(response)["market"].summary
    assert fake_coingecko.requests == []


def test_chat_endpoint_with_screenshot(client, fake_coingecko, png_attachment):
    serve_wave(fake_coingecko)
    message = {"role": "user", "content": "what do you see?", "attachments": [png_attachment]}
    response = client.post("/chat", json={"messages": [message]})
    assert response.status_code == 200
    content = response.json()["message"]["content"]
    assert "Screenshot shows a candlestick chart of BTC/USDT (4h)." in content
    assert "Price shown on the screenshot: $64,210.50" in content
    assert "Technical analysis of BTC 4h" in content
