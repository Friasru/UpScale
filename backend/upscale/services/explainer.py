"""Short educational explanations of trading and crypto concepts, behind `ExplainerModel`.

Used for general questions such as "What is RSI?" that need no live data and no decision.
The model only explains concepts: it is told not to quote current prices or market
conditions and not to recommend trades. `ClaudeExplainerModel` is the default implementation.
"""

from typing import Protocol

import anthropic

MAX_QUESTION_CHARS = 2_000


class ExplainerError(Exception):
    """The explainer model failed, refused, or returned no usable text."""


class ExplainerModel(Protocol):
    name: str

    async def explain(self, question: str) -> str:
        """A concise plain-text explanation. Raises `ExplainerError` on failure."""
        ...


SYSTEM_PROMPT = """\
You are UpScale's trading tutor. Users of a crypto analysis app ask general questions about
trading and crypto concepts (indicators, chart reading, order types, market terms).

Answer the question with a concise educational explanation:
- Plain text only (the chat shows raw text, not markdown): no headings, bold, or tables.
  Short "- " bullet lines are fine when they help.
- About 60-150 words: what it is, how it is read or used, and one common caveat or
  limitation. Use typical default settings as examples where relevant (e.g. RSI 14).
- Explain the concept in general. Do not quote current prices, levels or market
  conditions, and do not recommend buying, selling or any specific trade.
- If the question is not about trading, markets or crypto, say briefly that you can only
  explain trading and crypto concepts.
"""


class ClaudeExplainerModel:
    """Explains concepts with Claude via the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 1,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.name = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile).
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key, timeout=self._timeout, max_retries=self._max_retries
            )
        return self._client

    async def explain(self, question: str) -> str:
        try:
            response = await self._get_client().beta.messages.create(
                model=self.name,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": question[:MAX_QUESTION_CHARS]}],
                # A short definition doesn't need deep reasoning; keep replies fast.
                output_config={"effort": "low"},
                # On a safety decline, let the API retry on its recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise ExplainerError("the explanation model is rate limited") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ExplainerError(
                "the explanation model rejected UpScale's credentials; check ANTHROPIC_API_KEY"
            ) from exc
        except anthropic.BadRequestError as exc:
            raise ExplainerError(
                f"the explanation model rejected the request: {exc.message}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ExplainerError(f"the explanation model returned HTTP {exc.status_code}") from exc
        except anthropic.APITimeoutError as exc:
            raise ExplainerError("the explanation model timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise ExplainerError("could not reach the explanation model") from exc
        except anthropic.AnthropicError as exc:  # e.g. a credentials/profile problem
            raise ExplainerError(
                "the explanation model is not configured; set ANTHROPIC_API_KEY"
            ) from exc
        except TypeError as exc:
            # The SDK raises a plain TypeError when no API key, token or profile exists.
            if "authentication method" not in str(exc):
                raise
            raise ExplainerError(
                "the explanation model is not configured; set ANTHROPIC_API_KEY"
            ) from exc

        if response.stop_reason == "refusal":
            raise ExplainerError("the explanation model declined to answer")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise ExplainerError("the explanation model returned no text")
        return text
