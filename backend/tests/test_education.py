import pytest

from upscale.orchestrator import Orchestrator
from upscale.routing import route
from upscale.services.explainer import ExplainerError

from .conftest import FakeExplainerModel
from .test_orchestrator import ask

CONCEPT_QUESTIONS = [
    "What is RSI?",
    "What is MACD?",
    "What is support and resistance?",
    "What does volume mean?",
    "What is a candlestick?",
    "What does overbought mean?",
    "What's a moving average?",
    "what’s a stop loss?",
    "Explain Bollinger bands",
    "How does leverage work?",
    "What are altcoins?",
    "Can you explain what a limit order is?",
]

DECISION_QUESTIONS = [
    "Should I buy BTC?",
    "What's the best move on ETH 5m?",
    "Analyze this chart and tell me what to do",
    "What should I do?",
    "BTC buy or wait?",
]


# --- Routing ----------------------------------------------------------------------


@pytest.mark.parametrize("question", CONCEPT_QUESTIONS)
def test_concept_questions_route_to_education_only(question):
    decision = route(question, has_images=False)
    assert decision.agents == ["education"]
    assert decision.assets == []
    assert decision.timeframe is None


@pytest.mark.parametrize("question", DECISION_QUESTIONS)
def test_decision_questions_keep_the_full_pipeline(question):
    agents = route(question, has_images=False).agents
    assert "education" not in agents
    assert agents == ["technical_analysis", "market", "news_sentiment", "risk", "opportunity"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # A named coin makes it a live question about that coin.
        ("What is the RSI on BTC?", ["technical_analysis", "risk"]),
        ("What's ETH's support and resistance?", ["technical_analysis", "risk"]),
        # A chart screenshot is always analyzed.
        ("What is RSI?", None),
        # Questions about the market right now need live data, not a lesson.
        ("What is the crypto market doing?", ["market", "risk"]),
    ],
)
def test_live_questions_are_not_treated_as_concepts(question, expected):
    if expected is None:
        agents = route(question, has_images=True).agents
        assert agents[0] == "vision" and "opportunity" in agents
    else:
        assert route(question, has_images=False).agents == expected


@pytest.mark.parametrize(
    "text",
    ["What is love?", "what's the price of coffee", "What is the volume right now?", "hi"],
)
def test_non_trading_or_live_text_is_not_a_concept_question(text):
    assert "education" not in route(text, has_images=False).agents


# --- Response ---------------------------------------------------------------------


@pytest.mark.parametrize("question", ["What is RSI?", "What does overbought mean?"])
def test_concept_question_gets_a_plain_explanation(question, fake_explainer):
    fake_explainer.answer = "RSI is a momentum oscillator from 0 to 100."
    response = ask(Orchestrator(), question)
    assert fake_explainer.calls == [question]
    assert response.message.content == "RSI is a momentum oscillator from 0 to 100."
    analysis = response.analysis
    assert analysis is not None
    assert analysis.agents_used == ["education"]
    assert analysis.assets == []
    assert "BUY / SELL / WAIT" not in analysis.disclaimer
    assert not any(r.agent == "opportunity" for r in analysis.agent_results)


def test_concept_question_runs_no_market_data(fake_coingecko, fake_news):
    ask(Orchestrator(), "What is support and resistance?")
    assert fake_coingecko.requests == []
    assert fake_news.requests == []


def test_explainer_failure_is_reported_plainly(fake_explainer: FakeExplainerModel):
    fake_explainer.error = ExplainerError("the explanation model is not configured")
    response = ask(Orchestrator(), "What is MACD?")
    assert response.message.content.startswith("I couldn't generate an explanation right now")
    [result] = response.analysis.agent_results
    assert result.status == "error"
    assert result.error == "the explanation model is not configured"


def test_decision_question_still_returns_a_decision(fake_explainer):
    response = ask(Orchestrator(), "Should I buy BTC?")
    assert fake_explainer.calls == []
    assert response.message.content.split("\n", 1)[0] in ("BUY", "SELL", "WAIT")


def test_chat_endpoint_answers_concept_question(client, fake_explainer):
    fake_explainer.answer = "A candlestick shows open, high, low and close for one period."
    response = client.post(
        "/chat", json={"messages": [{"role": "user", "content": "What is a candlestick?"}]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"] == fake_explainer.answer
    assert data["analysis"]["agents_used"] == ["education"]
