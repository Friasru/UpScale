import base64
import binascii
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_ATTACHMENTS_PER_MESSAGE = 4
ImageMediaType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]


class ImageAttachment(BaseModel):
    name: str = Field(max_length=255)
    media_type: ImageMediaType
    data: str = Field(description="Base64-encoded image bytes (no data: URL prefix).")

    @field_validator("data")
    @classmethod
    def check_base64_and_size(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("data must be valid base64") from exc
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit")
        return value


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=20_000)
    attachments: list[ImageAttachment] = Field(
        default_factory=list, max_length=MAX_ATTACHMENTS_PER_MESSAGE
    )


class ChatRequest(BaseModel):
    # Full conversation so far, oldest first; the last message is the new user turn.
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)

    @field_validator("messages")
    @classmethod
    def last_message_is_from_user(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        last = value[-1]
        if last.role != "user":
            raise ValueError("last message must be from the user")
        if not last.content.strip() and not last.attachments:
            raise ValueError("last message must have text or at least one image")
        return value


# --- Agent analysis -------------------------------------------------------------

AgentName = Literal[
    "vision",
    "technical_analysis",
    "market",
    "dex_market",
    "news_sentiment",
    "opportunity",
    "risk",
    "education",
]
Level = Literal["low", "medium", "high"]


class Evidence(BaseModel):
    source: AgentName
    statement: str


class Scenario(BaseModel):
    """A conditional "if X then Y" outlook. Never a buy/sell recommendation."""

    name: str
    description: str
    conditions: list[str] = Field(default_factory=list)
    invalidation: str | None = None
    source: AgentName | None = None


class Risk(BaseModel):
    description: str
    severity: Level
    source: AgentName | None = None


class Uncertainty(BaseModel):
    level: Level
    notes: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    """What every agent returns, so the orchestrator can merge results generically."""

    agent: AgentName
    status: Literal["ok", "error"] = "ok"
    mock: bool = Field(description="True when the output is placeholder data, not real analysis.")
    summary: str
    # Agent-specific structured output (e.g. detected timeframe, indicators, prices).
    findings: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    error: str | None = None


class Analysis(BaseModel):
    mock: bool
    summary: str
    assets: list[str] = Field(default_factory=list)
    agents_used: list[AgentName] = Field(default_factory=list)
    # Why the orchestrator selected each agent.
    routing: dict[AgentName, str] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    uncertainty: Uncertainty
    agent_results: list[AgentResult] = Field(default_factory=list)
    disclaimer: str


class ChatResponse(BaseModel):
    # Plain-text rendering of the analysis; this is what the chat UI displays.
    message: ChatMessage
    analysis: Analysis | None = None
