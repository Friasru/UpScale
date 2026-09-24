import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from upscale.schemas import AgentName, AgentResult, ChatMessage, ImageAttachment


@dataclass(frozen=True)
class AgentContext:
    """Everything an agent may use to answer one chat turn."""

    query: str
    attachments: list[ImageAttachment] = field(default_factory=list)
    # Earlier messages in the conversation, oldest first (excludes the current turn).
    history: list[ChatMessage] = field(default_factory=list)
    # Asset symbols the orchestrator detected in the request, e.g. ["BTC", "ETH"].
    assets: list[str] = field(default_factory=list)
    # Chart timeframe to analyze, if any (e.g. "4h"), and where it came from.
    timeframe: str | None = None
    timeframe_source: Literal["user", "screenshot"] | None = None
    # Where `assets` came from: the user's text, or a screenshot when the text named none.
    assets_source: Literal["user", "screenshot"] | None = None
    # Results from agents that ran earlier in this turn, keyed by agent name.
    prior_results: dict[AgentName, AgentResult] = field(default_factory=dict)

    @property
    def primary_asset(self) -> str | None:
        return self.assets[0] if self.assets else None


def image_size_bytes(image: ImageAttachment) -> int:
    return len(base64.b64decode(image.data))


class Agent(ABC):
    """Base class for all agents.

    To add or replace an agent, subclass this, set `name` and `description`, and
    implement `run`. `depends_on` lists agents whose results this one wants in
    `context.prior_results`; the orchestrator runs those first when they are selected.
    """

    name: ClassVar[AgentName]
    description: ClassVar[str]
    depends_on: ClassVar[tuple[AgentName, ...]] = ()
    # Per-agent time limit in seconds; None uses the orchestrator's default.
    timeout: ClassVar[float | None] = None

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult: ...
