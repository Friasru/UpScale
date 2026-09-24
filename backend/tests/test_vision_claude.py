"""ClaudeVisionModel through the real Anthropic SDK, with HTTP served by a MockTransport."""

import asyncio
import json

import anthropic
import httpx2
import pytest
from anthropic import DefaultAsyncHttpxClient

from upscale.services.vision import (
    ChartReading,
    CheckedImage,
    ClaudeVisionModel,
    MalformedVisionOutputError,
    VisionRefusedError,
    VisionUnavailableError,
    strict_json_schema,
)

from .conftest import CHART_PNG, chart_reading_data

IMAGE = CheckedImage(
    name="chart.png",
    media_type="image/png",
    data=CHART_PNG,
    size_bytes=100,
    width=320,
    height=200,
    sha256="0" * 64,
)


def message(text: str | None, stop_reason: str = "end_turn") -> dict:
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def api_error(status: int, kind: str) -> httpx2.Response:
    body = {"type": "error", "error": {"type": kind, "message": f"{kind} happened"}}
    return httpx2.Response(status, json=body)


class FakeAnthropicAPI:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.response = lambda request: httpx2.Response(
            200, json=message(json.dumps(chart_reading_data()))
        )

    def model(self) -> ClaudeVisionModel:
        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.response(request)

        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handle)),
        )
        return ClaudeVisionModel(client=client)


@pytest.fixture
def api() -> FakeAnthropicAPI:
    return FakeAnthropicAPI()


def read(api: FakeAnthropicAPI) -> ChartReading:
    return asyncio.run(api.model().read_chart(IMAGE))


# --- Success ----------------------------------------------------------------------------------


def test_successful_structured_reading(api):
    reading = read(api)
    assert reading.asset.pair == "BTC/USDT"
    assert reading.timeframe.label == "4h"
    assert reading.indicators[0].name == "RSI"


def test_request_shape(api):
    read(api)
    [request] = api.requests
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    assert body["model"] == "claude-opus-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"]["type"] == "json_schema"
    image_block, text_block = body["messages"][0]["content"]
    assert image_block["type"] == "image"
    assert image_block["source"] == {"type": "base64", "media_type": "image/png", "data": CHART_PNG}
    assert text_block["type"] == "text"
    assert "Never fill in values" in body["system"]


def test_output_schema_is_strict():
    schema = strict_json_schema(ChartReading)

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                yield node
            for value in node.values():
                yield from objects(value)
        elif isinstance(node, list):
            for item in node:
                yield from objects(item)

    found = list(objects(schema))
    assert len(found) > 5
    for obj in found:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    text = json.dumps(schema)
    assert '"default"' not in text and '"minimum"' not in text


def test_model_name_is_configurable(api):
    model = api.model()
    model.name = "claude-sonnet-5"
    asyncio.run(model.read_chart(IMAGE))
    assert json.loads(api.requests[0].content)["model"] == "claude-sonnet-5"


# --- Malformed output ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I think this is a Bitcoin chart.",  # not JSON
        '{"is_price_chart": true}',  # missing fields
        json.dumps(chart_reading_data(chart_type="hologram")),  # invalid enum
        json.dumps(chart_reading_data(indicators=[{"name": "RSI"}])),  # incomplete nested item
        "[]",
    ],
    ids=["prose", "missing-fields", "bad-enum", "bad-nested", "array"],
)
def test_malformed_output(api, text):
    api.response = lambda r: httpx2.Response(200, json=message(text))
    with pytest.raises(MalformedVisionOutputError, match="did not match"):
        read(api)


def test_no_text_block(api):
    api.response = lambda r: httpx2.Response(200, json=message(None))
    with pytest.raises(MalformedVisionOutputError, match="no text output"):
        read(api)


def test_truncated_output(api):
    api.response = lambda r: httpx2.Response(
        200, json=message('{"is_price', stop_reason="max_tokens")
    )
    with pytest.raises(MalformedVisionOutputError, match="cut off"):
        read(api)


def test_refusal(api):
    api.response = lambda r: httpx2.Response(200, json=message(None, stop_reason="refusal"))
    with pytest.raises(VisionRefusedError):
        read(api)


# --- Provider failures ---------------------------------------------------------------------------------


def _raise(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.parametrize(
    ("response", "message_part"),
    [
        (lambda r: api_error(429, "rate_limit_error"), "rate limited"),
        (lambda r: api_error(401, "authentication_error"), "check ANTHROPIC_API_KEY"),
        (lambda r: api_error(403, "permission_error"), "check ANTHROPIC_API_KEY"),
        (lambda r: api_error(400, "invalid_request_error"), "rejected the request"),
        (lambda r: api_error(500, "api_error"), "HTTP 500"),
        (lambda r: api_error(529, "overloaded_error"), "HTTP 529"),
        (_raise(httpx2.ReadTimeout("slow")), "timed out"),
        (_raise(httpx2.ConnectError("down")), "could not reach"),
    ],
    ids=["429", "401", "403", "400", "500", "529", "timeout", "connect"],
)
def test_provider_failures(api, response, message_part):
    api.response = response
    with pytest.raises(VisionUnavailableError, match=message_part):
        read(api)


def test_missing_credentials():
    class NoCredentials:
        class beta:
            class messages:
                @staticmethod
                async def create(**kwargs):
                    raise anthropic.AnthropicError("no API key or profile found")

    model = ClaudeVisionModel(client=NoCredentials())  # type: ignore[arg-type]
    with pytest.raises(VisionUnavailableError, match="not configured; set ANTHROPIC_API_KEY"):
        asyncio.run(model.read_chart(IMAGE))


def test_no_credentials_anywhere(monkeypatch, tmp_path):
    """The real SDK client with no key, token or profile: reported as not configured."""
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile on disk
    model = ClaudeVisionModel(max_retries=0)  # builds its own real client
    with pytest.raises(VisionUnavailableError, match="not configured; set ANTHROPIC_API_KEY"):
        asyncio.run(model.read_chart(IMAGE))


def test_unrelated_type_errors_are_not_hidden():
    class Broken:
        class beta:
            class messages:
                @staticmethod
                async def create(**kwargs):
                    raise TypeError("unexpected keyword argument 'foo'")

    with pytest.raises(TypeError, match="foo"):
        asyncio.run(ClaudeVisionModel(client=Broken()).read_chart(IMAGE))  # type: ignore[arg-type]


def test_default_client_is_blocked_from_the_network_in_tests(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = ClaudeVisionModel(max_retries=0)
    with pytest.raises(VisionUnavailableError):
        asyncio.run(model.read_chart(IMAGE))
