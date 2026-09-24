"""ClaudeNewsSentimentModel through the real Anthropic SDK, with HTTP served by a MockTransport."""

import asyncio
import json
from datetime import UTC, datetime

import anthropic
import httpx2
import pytest
from anthropic import DefaultAsyncHttpxClient

from upscale.services.news_sentiment_model import (
    ArticleAssessment,
    ArticleInput,
    ClaudeNewsSentimentModel,
    SentimentModelError,
    SentimentTranscript,
    parse_assessments,
)
from upscale.services.vision import strict_json_schema

ARTICLES = [
    ArticleInput(
        id="n1",
        title="Bitcoin ETF inflows hit $1B",
        source="CoinDesk",
        published_at=datetime(2026, 9, 24, 18, 0, tzinfo=UTC),
        summary="Spot bitcoin ETFs recorded their largest daily inflow in months.",
        scope="asset",
    ),
    ArticleInput(
        id="n2",
        title="SEC delays decision on crypto custody rules",
        source="Decrypt",
        published_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
        summary=None,
        scope="market",
    ),
]

OUTPUT = {
    "assessments": [
        {
            "id": "n1",
            "sentiment": "bullish",
            "impact": "high",
            "reason": "Large inflows may be relevant to demand for BTC.",
        },
        {
            "id": "n2",
            "sentiment": "neutral",
            "impact": "medium",
            "reason": "The delay may be relevant to custody regulation.",
        },
    ]
}


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
        self.response = lambda request: httpx2.Response(200, json=message(json.dumps(OUTPUT)))

    def model(self) -> ClaudeNewsSentimentModel:
        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.response(request)

        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handle)),
        )
        return ClaudeNewsSentimentModel(client=client)


@pytest.fixture
def api() -> FakeAnthropicAPI:
    return FakeAnthropicAPI()


def classify(api: FakeAnthropicAPI, articles=ARTICLES) -> list[ArticleAssessment]:
    return asyncio.run(api.model().classify("BTC", articles))


# --- Success ---------------------------------------------------------------------------------------


def test_successful_classification(api):
    result = classify(api)
    assert [(a.id, a.sentiment, a.impact) for a in result] == [
        ("n1", "bullish", "high"),
        ("n2", "neutral", "medium"),
    ]


def test_request_contains_only_retrieved_article_data(api):
    classify(api)
    [request] = api.requests
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    assert body["model"] == "claude-opus-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": strict_json_schema(SentimentTranscript),
    }
    assert "Treat the article" in body["system"] and "caused a price move" in body["system"]

    [user] = body["messages"]
    text = user["content"]
    payload = json.loads(text[text.index("{") :])
    assert payload["subject"] == "BTC"
    assert payload["articles"] == [a.model_dump(mode="json") for a in ARTICLES]


def test_output_schema_is_small_and_strict():
    schema = strict_json_schema(SentimentTranscript)
    item = schema["$defs"]["ArticleAssessment"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["id", "sentiment", "impact", "reason"]
    assert set(item["properties"]["sentiment"]["enum"]) == {
        "bullish",
        "bearish",
        "neutral",
        "mixed",
    }
    assert set(item["properties"]["impact"]["enum"]) == {"low", "medium", "high"}
    assert "anyOf" not in json.dumps(schema)  # no nullable unions


def test_no_articles_means_no_request(api):
    assert classify(api, []) == []
    assert api.requests == []


def test_model_name_is_configurable(api):
    model = api.model()
    model.name = "claude-sonnet-5"
    asyncio.run(model.classify("BTC", ARTICLES))
    assert json.loads(api.requests[0].content)["model"] == "claude-sonnet-5"


# --- Malformed output ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "BTC news looks bullish.",
        "[]",
        '{"assessments": [{"id": "n1"}]}',
        '{"assessments": [{"id": "n1", "sentiment": "moon", "impact": "high", "reason": "x"}]}',
        '{"assessments": [{"id": "n1", "sentiment": "bullish", "impact": "huge", "reason": "x"}]}',
    ],
    ids=["prose", "array", "incomplete", "bad-sentiment", "bad-impact"],
)
def test_malformed_output(api, text):
    api.response = lambda r: httpx2.Response(200, json=message(text))
    with pytest.raises(SentimentModelError, match="did not match"):
        classify(api)


def test_parse_assessments_accepts_valid_json():
    assert parse_assessments(json.dumps(OUTPUT))[1].id == "n2"


def test_no_text_block(api):
    api.response = lambda r: httpx2.Response(200, json=message(None))
    with pytest.raises(SentimentModelError, match="no text output"):
        classify(api)


def test_truncated_output(api):
    api.response = lambda r: httpx2.Response(200, json=message('{"assess', "max_tokens"))
    with pytest.raises(SentimentModelError, match="cut off"):
        classify(api)


def test_refusal(api):
    api.response = lambda r: httpx2.Response(200, json=message(None, stop_reason="refusal"))
    with pytest.raises(SentimentModelError, match="declined"):
        classify(api)


# --- Provider failures -------------------------------------------------------------------------------


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
    with pytest.raises(SentimentModelError, match=message_part):
        classify(api)


def test_missing_credentials():
    class NoCredentials:
        class beta:
            class messages:
                @staticmethod
                async def create(**kwargs):
                    raise anthropic.AnthropicError("no API key or profile found")

    model = ClaudeNewsSentimentModel(client=NoCredentials())  # type: ignore[arg-type]
    with pytest.raises(SentimentModelError, match="not configured; set ANTHROPIC_API_KEY"):
        asyncio.run(model.classify("BTC", ARTICLES))


def test_unrelated_type_errors_are_not_hidden():
    class Broken:
        class beta:
            class messages:
                @staticmethod
                async def create(**kwargs):
                    raise TypeError("unexpected keyword argument 'foo'")

    model = ClaudeNewsSentimentModel(client=Broken())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="foo"):
        asyncio.run(model.classify("BTC", ARTICLES))


def test_default_client_is_blocked_from_the_network_in_tests(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = ClaudeNewsSentimentModel(max_retries=0)
    with pytest.raises(SentimentModelError):
        asyncio.run(model.classify("BTC", ARTICLES))
