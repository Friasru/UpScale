"""ClaudeVisionModel through the real Anthropic SDK, with HTTP served by a MockTransport."""

import asyncio
import json

import anthropic
import httpx2
import pytest
from anthropic import DefaultAsyncHttpxClient

from upscale.services.vision import (
    ChartReading,
    ChartTranscript,
    CheckedImage,
    ClaudeVisionModel,
    MalformedVisionOutputError,
    VisionRefusedError,
    VisionUnavailableError,
    parse_reading,
    sanitize,
    strict_json_schema,
)

from .conftest import CHART_PNG, chart_reading_data, chart_transcript_data

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
            200, json=message(json.dumps(chart_transcript_data()))
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


def _nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _nodes(item)


def _objects(schema):
    return [n for n in _nodes(schema) if n.get("type") == "object" and "properties" in n]


def test_output_schema_is_strict():
    schema = strict_json_schema(ChartTranscript)
    found = _objects(schema)
    assert len(found) > 5
    for obj in found:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    text = json.dumps(schema)
    assert '"default"' not in text and '"minimum"' not in text


def test_request_sends_the_transport_schema(api):
    read(api)
    sent = json.loads(api.requests[0].content)["output_config"]["format"]["schema"]
    assert sent == strict_json_schema(ChartTranscript)


def test_transport_schema_stays_small():
    """Regression: the richer ChartReading schema compiled to a grammar Anthropic rejected
    ("The compiled grammar is too large"). Unions (nullable fields), enums, objects and
    nesting are what grow the grammar, so each is capped well below the old schema's
    14 unions / 11 enums / 10 objects / 47 properties."""
    schema = strict_json_schema(ChartTranscript)
    nodes = list(_nodes(schema))
    unions = [n for n in nodes if {"anyOf", "oneOf", "allOf"} & n.keys()]
    type_lists = [n for n in nodes if isinstance(n.get("type"), list)]
    nulls = [n for n in nodes if n.get("type") == "null"]
    enums = [n for n in nodes if "enum" in n or "const" in n]
    objects = _objects(schema)
    properties = sum(len(o["properties"]) for o in objects)

    assert unions == [] and type_lists == [] and nulls == []
    assert len(enums) <= 5
    assert sum(len(e.get("enum", [None])) for e in enums) <= 20
    assert len(objects) <= 6
    assert properties <= 42
    assert len(json.dumps(schema, separators=(",", ":"))) <= 4000

    defs = schema.get("$defs", {})

    def depth(node, seen=()):
        """Nesting of objects, following $refs (the schema must not be recursive)."""
        if isinstance(node, dict):
            if (ref := node.get("$ref")) is not None:
                name = ref.rsplit("/", 1)[-1]
                assert name not in seen, "recursive schema"
                return depth(defs[name], (*seen, name))
            own = 1 if node.get("type") == "object" else 0
            children = [v for k, v in node.items() if k != "$defs"]
            return own + max((depth(c, seen) for c in children), default=0)
        if isinstance(node, list):
            return max((depth(c, seen) for c in node), default=0)
        return 0

    assert depth(schema) <= 3  # root -> indicator -> printed value


def test_model_name_is_configurable(api):
    model = api.model()
    model.name = "claude-sonnet-5"
    asyncio.run(model.read_chart(IMAGE))
    assert json.loads(api.requests[0].content)["model"] == "claude-sonnet-5"


# --- Transport -> ChartReading --------------------------------------------------------------------


def test_transcript_expands_to_the_same_reading_as_before():
    """The fixture transcript maps onto the richer reading the rest of UpScale consumes."""
    reading = parse_reading(json.dumps(chart_transcript_data()))
    assert reading == ChartReading.model_validate(chart_reading_data())


def test_blank_transcript_fields_become_unknown():
    reading = parse_reading(
        json.dumps(
            chart_transcript_data(
                symbol=" ",
                pair="",
                exchange="",
                asset_inferred=True,
                timeframe="",
                timeframe_inferred=False,
                price=0,
                price_inferred=False,
            )
        )
    )
    assert (reading.asset.symbol, reading.asset.pair) == (None, None)
    assert reading.asset.basis == "unknown"
    assert (reading.timeframe.label, reading.timeframe.basis) == (None, "unknown")
    assert (reading.displayed_price.value, reading.displayed_price.basis) == (None, "unknown")


def test_inferred_flags_stay_inferred():
    reading = parse_reading(
        json.dumps(
            chart_transcript_data(asset_inferred=True, timeframe_inferred=True, price_inferred=True)
        )
    )
    assert reading.asset.basis == reading.timeframe.basis == reading.displayed_price.basis
    assert reading.asset.basis == "inferred"
    cleaned, discarded = sanitize(reading)
    assert [i.name for i in cleaned.indicators] == ["RSI"]
    assert any("EMA" in d for d in discarded)


def test_negative_prices_are_unreadable():
    level = {
        "kind": "resistance",
        "price": -5,
        "label": "",
        "source": "price_label",
        "inferred": False,
        "evidence": "scale",
    }
    reading = parse_reading(json.dumps(chart_transcript_data(price=-1, levels=[level])))
    assert reading.displayed_price.value is None
    cleaned, discarded = sanitize(reading)
    assert cleaned.resistance_levels == []
    assert "Dropped a resistance level without a readable price." in discarded


def test_inferred_user_drawn_level_is_an_uncertainty():
    level = {
        "kind": "user_drawn",
        "price": 61000,
        "label": "maybe",
        "source": "drawn_line",
        "inferred": True,
        "evidence": "faint line",
    }
    reading = parse_reading(json.dumps(chart_transcript_data(levels=[level])))
    assert reading.drawn_levels == []
    assert any("Possible user-drawn level 'maybe'" in u for u in reading.uncertainties)


def test_non_chart_transcript_yields_no_chart_data():
    reading = parse_reading(json.dumps(chart_transcript_data(is_price_chart=False)))
    cleaned, discarded = sanitize(reading)
    assert cleaned.asset.symbol is None and cleaned.indicators == [] and cleaned.patterns == []
    assert cleaned.support_levels == cleaned.resistance_levels == cleaned.drawn_levels == []
    assert "does not look like a price chart" in discarded[0]


# --- Malformed output ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I think this is a Bitcoin chart.",  # not JSON
        '{"is_price_chart": true}',  # missing fields
        json.dumps(chart_transcript_data(chart_type="hologram")),  # invalid enum
        json.dumps(chart_transcript_data(indicators=[{"name": "RSI"}])),  # incomplete nested item
        json.dumps(chart_transcript_data(price=None)),  # null where the schema has none
        json.dumps(chart_reading_data()),  # the old nested shape is no longer accepted
        "[]",
    ],
    ids=["prose", "missing-fields", "bad-enum", "bad-nested", "null", "old-shape", "array"],
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
