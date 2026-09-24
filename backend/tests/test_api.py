from upscale.schemas import ChatResponse


def post_chat(client, content="", attachments=None):
    message = {"role": "user", "content": content, "attachments": attachments or []}
    return client.post("/chat", json={"messages": [message]})


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_text(client):
    response = post_chat(client, "What's up with BTC?")
    assert response.status_code == 200
    body = ChatResponse.model_validate(response.json())  # full structured response is valid
    assert body.message.role == "assistant"
    assert body.analysis is not None
    assert body.analysis.mock is True
    assert body.analysis.assets == ["BTC"]
    assert body.analysis.agents_used == [
        "technical_analysis",
        "market",
        "news_sentiment",
        "opportunity",
        "risk",
    ]
    assert "Mock" in body.message.content
    assert "Bitcoin (BTC) price: $64,000.00" in body.message.content  # from the fake CoinGecko


def test_chat_structured_response_fields(client):
    data = post_chat(client, "Give me the outlook for ETH").json()
    analysis = data["analysis"]
    for key in (
        "summary",
        "evidence",
        "scenarios",
        "risks",
        "uncertainty",
        "agent_results",
        "disclaimer",
    ):
        assert key in analysis
    assert analysis["uncertainty"]["level"] == "high"
    assert analysis["scenarios"] and all(
        s["source"] == "opportunity" for s in analysis["scenarios"]
    )
    assert {r["agent"]: r["mock"] for r in analysis["agent_results"]} == {
        "opportunity": True,
        "risk": True,
    }
    assert set(analysis["routing"]) == set(analysis["agents_used"])


def test_chat_with_image_only(client, png_attachment):
    response = post_chat(client, attachments=[png_attachment])
    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["agents_used"] == ["vision", "technical_analysis", "market", "risk"]
    vision = analysis["agent_results"][0]
    assert vision["agent"] == "vision" and vision["mock"] is False
    [chart] = vision["findings"]["charts"]
    assert (chart["image_name"], chart["media_type"]) == ("chart.png", "image/png")
    assert (chart["width"], chart["height"]) == (320, 200)
    assert vision["findings"]["detected_asset"] == "BTC"
    assert vision["findings"]["detected_timeframe"] == "4h"
    assert analysis["assets"] == ["BTC"]  # taken from the screenshot
    assert "Price shown on the screenshot: $64,210.50" in response.json()["message"]["content"]


def test_chat_screenshot_with_question(client, png_attachment):
    analysis = post_chat(client, "Is this SOL chart bullish?", [png_attachment]).json()["analysis"]
    assert analysis["agents_used"][0] == "vision"
    assert "risk" in analysis["agents_used"]
    assert analysis["assets"] == ["SOL"]  # the user's explicit asset wins over the screenshot
    vision = analysis["agent_results"][0]
    assert vision["findings"]["detected_asset"] == "BTC"
    assert any("shows BTC, but you asked about SOL" in e for e in vision["evidence"])


def test_chat_non_crypto_message(client):
    data = post_chat(client, "hello there").json()
    assert data["analysis"]["agents_used"] == []
    assert data["analysis"]["agent_results"] == []
    assert "crypto" in data["message"]["content"]


def test_chat_uses_only_latest_message_for_routing(client):
    messages = [
        {"role": "user", "content": "Tell me about BTC"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "thanks"},
    ]
    response = client.post("/chat", json={"messages": messages})
    assert response.status_code == 200
    assert response.json()["analysis"]["agents_used"] == []


def test_chat_rejects_empty_message(client):
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "  "}]})
    assert response.status_code == 422


def test_chat_rejects_bad_base64(client):
    response = post_chat(
        client, attachments=[{"name": "x.png", "media_type": "image/png", "data": "not base64!"}]
    )
    assert response.status_code == 422
