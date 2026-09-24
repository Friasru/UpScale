"""Chart screenshot reading: structured schema, image checks, model interface, guardrails.

`VisionService` validates the image, asks a `VisionModel` to transcribe what the chart
shows into `ChartReading` (Claude fills the flat `ChartTranscript`, which code expands), then enforces the "never invent" rules deterministically:
only visible indicators are kept, unclear patterns are demoted to uncertainties,
timeframes are normalized by code (not by the model), and unknowns stay unknown.
The model sits behind the `VisionModel` protocol; `ClaudeVisionModel` is the default.
"""

import asyncio
import base64
import binascii
import hashlib
import json
import math
import re
import struct
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import anthropic
from pydantic import BaseModel, Field, ValidationError

from upscale.schemas import ImageAttachment, ImageMediaType
from upscale.services.market_data import Timeframe

# --- Schema: what the model must return ------------------------------------------------------

Basis = Literal["visible", "inferred"]


class AssetReading(BaseModel):
    symbol: str | None = Field(description="Base asset ticker, e.g. BTC. Null if not shown.")
    pair: str | None = Field(description="Pair exactly as displayed, e.g. BTC/USDT or BTCUSDT.")
    exchange: str | None = Field(description="Exchange or data source if displayed.")
    basis: Literal["visible", "inferred", "unknown"]
    evidence: str | None = Field(description="Where on the chart this was read.")


class TimeframeReading(BaseModel):
    label: str | None = Field(description="Timeframe exactly as displayed, e.g. 4h, 240, 1D.")
    basis: Literal["visible", "inferred", "unknown"]
    evidence: str | None


class PriceReading(BaseModel):
    value: float | None = Field(description="Current/last price printed on the chart.")
    basis: Literal["visible", "inferred", "unknown"]
    evidence: str | None


class IndicatorValue(BaseModel):
    label: str | None = Field(description="Line or output name as displayed, e.g. signal.")
    value: float


class IndicatorReading(BaseModel):
    name: str = Field(description="Indicator name as displayed, e.g. RSI, EMA, MACD.")
    settings: str | None = Field(description="Settings as displayed, e.g. 14 or 12 26 9.")
    values: list[IndicatorValue] = Field(description="Only values printed on the chart.")
    basis: Basis
    evidence: str


class LevelReading(BaseModel):
    price: float | None
    label: str | None
    source: Literal["drawn_line", "price_label", "price_structure"]
    basis: Basis
    evidence: str


class LineReading(BaseModel):
    kind: Literal["trendline", "channel"]
    direction: Literal["rising", "falling", "horizontal", "unclear"]
    description: str
    basis: Basis
    evidence: str


class DrawnLevel(BaseModel):
    price: float | None
    label: str | None
    evidence: str


class PatternReading(BaseModel):
    name: str
    clarity: Literal["clear", "tentative"]
    evidence: str


ChartType = Literal["candlestick", "heikin_ashi", "bar", "line", "area", "other", "unknown"]


class ChartReading(BaseModel):
    """Everything read from one screenshot. Values are as displayed, not live data."""

    is_price_chart: bool
    asset: AssetReading
    timeframe: TimeframeReading
    displayed_price: PriceReading
    chart_type: ChartType
    indicators: list[IndicatorReading]
    support_levels: list[LevelReading]
    resistance_levels: list[LevelReading]
    trend_lines: list[LineReading]
    drawn_levels: list[DrawnLevel]
    patterns: list[PatternReading]
    observations: list[str]
    uncertainties: list[str]


class ChartVision(BaseModel):
    """The sanitized reading of one screenshot plus deterministic post-processing."""

    image_name: str
    media_type: ImageMediaType
    size_bytes: int
    width: int | None
    height: int | None
    model: str
    reading: ChartReading
    # UpScale timeframe the displayed label maps to; None when unknown or not one of ours.
    normalized_timeframe: Timeframe | None
    # Items the model returned that UpScale dropped or demoted, with the reason.
    discarded: list[str]
    analyzed_at: datetime


# --- Transport schema: what the model is constrained to emit ------------------------------------
#
# Structured outputs compile the JSON schema into a grammar, and nullable fields (`anyOf` with
# null), repeated enums and objects nested inside arrays make that grammar grow quickly; the
# richer `ChartReading` schema was rejected as too large. So the model fills this deliberately
# flat shape instead: no unions, no nulls ("" or 0 mean "not readable"), one boolean in place of
# each visible/inferred basis, and a single list for all horizontal levels. `transcript_to_reading`
# maps it onto `ChartReading`, and `sanitize` applies UpScale's rules as before.


class TranscriptValue(BaseModel):
    label: str = Field(description='Output name as displayed, e.g. "signal"; "" if unlabeled.')
    value: float


class TranscriptIndicator(BaseModel):
    name: str = Field(description="Indicator name as displayed, e.g. RSI, EMA, MACD.")
    settings: str = Field(description='Settings as displayed, e.g. "14"; "" if not shown.')
    values: list[TranscriptValue] = Field(description="Only numbers printed on the chart.")
    inferred: bool
    evidence: str


class TranscriptLevel(BaseModel):
    kind: Literal["support", "resistance", "user_drawn"]
    price: float = Field(description="Price read from a label or the price scale; 0 if unreadable.")
    label: str
    source: Literal["drawn_line", "price_label", "price_structure"]
    inferred: bool
    evidence: str


class TranscriptLine(BaseModel):
    kind: Literal["trendline", "channel"]
    direction: Literal["rising", "falling", "horizontal", "unclear"]
    description: str
    inferred: bool
    evidence: str


class TranscriptPattern(BaseModel):
    name: str
    clear: bool = Field(description="True only if the visual evidence is unambiguous.")
    evidence: str


class ChartTranscript(BaseModel):
    """Flat model output for one screenshot; converted into `ChartReading` by code."""

    is_price_chart: bool
    chart_type: ChartType
    symbol: str = Field(description='Base ticker as displayed, e.g. "BTC"; "" if not shown.')
    pair: str = Field(description='Pair exactly as displayed, e.g. "BTC/USDT"; "" if not shown.')
    exchange: str = Field(description='Exchange or data source if displayed; "" otherwise.')
    asset_inferred: bool
    asset_evidence: str
    timeframe: str = Field(description='Interval exactly as displayed, e.g. "4h", "240"; "".')
    timeframe_inferred: bool
    timeframe_evidence: str
    price: float = Field(description="Current/last price printed on the chart; 0 if unreadable.")
    price_inferred: bool
    price_evidence: str
    indicators: list[TranscriptIndicator]
    levels: list[TranscriptLevel]
    lines: list[TranscriptLine]
    patterns: list[TranscriptPattern]
    observations: list[str]
    uncertainties: list[str]


def _text(value: str) -> str | None:
    return value.strip() or None


def _price(value: float) -> float | None:
    return value if math.isfinite(value) and value > 0 else None


def _basis(present: bool, inferred: bool) -> Literal["visible", "inferred", "unknown"]:
    if not present:
        return "unknown"
    return "inferred" if inferred else "visible"


def transcript_to_reading(t: ChartTranscript) -> ChartReading:
    """Expand the flat transport output into `ChartReading` (before `sanitize`).

    Empty strings and zero prices become None/"unknown", so nothing unreadable looks read.
    """
    symbol, pair, exchange = _text(t.symbol), _text(t.pair), _text(t.exchange)
    timeframe = _text(t.timeframe)
    price = _price(t.price)

    def level(lv: TranscriptLevel) -> LevelReading:
        return LevelReading(
            price=_price(lv.price),
            label=_text(lv.label),
            source=lv.source,
            basis="inferred" if lv.inferred else "visible",
            evidence=lv.evidence,
        )

    return ChartReading(
        is_price_chart=t.is_price_chart,
        asset=AssetReading(
            symbol=symbol,
            pair=pair,
            exchange=exchange,
            basis=_basis(bool(symbol or pair), t.asset_inferred),
            evidence=_text(t.asset_evidence),
        ),
        timeframe=TimeframeReading(
            label=timeframe,
            basis=_basis(timeframe is not None, t.timeframe_inferred),
            evidence=_text(t.timeframe_evidence),
        ),
        displayed_price=PriceReading(
            value=price,
            basis=_basis(price is not None, t.price_inferred),
            evidence=_text(t.price_evidence),
        ),
        chart_type=t.chart_type,
        indicators=[
            IndicatorReading(
                name=ind.name,
                settings=_text(ind.settings),
                values=[IndicatorValue(label=_text(v.label), value=v.value) for v in ind.values],
                basis="inferred" if ind.inferred else "visible",
                evidence=ind.evidence,
            )
            for ind in t.indicators
        ],
        support_levels=[level(lv) for lv in t.levels if lv.kind == "support"],
        resistance_levels=[level(lv) for lv in t.levels if lv.kind == "resistance"],
        trend_lines=[
            LineReading(
                kind=ln.kind,
                direction=ln.direction,
                description=ln.description,
                basis="inferred" if ln.inferred else "visible",
                evidence=ln.evidence,
            )
            for ln in t.lines
        ],
        drawn_levels=[
            DrawnLevel(price=_price(lv.price), label=_text(lv.label), evidence=lv.evidence)
            for lv in t.levels
            if lv.kind == "user_drawn" and not lv.inferred
        ],
        patterns=[
            PatternReading(
                name=p.name, clarity="clear" if p.clear else "tentative", evidence=p.evidence
            )
            for p in t.patterns
        ],
        observations=t.observations,
        # A "user-drawn" level the model only inferred is not a drawn line UpScale can rely on.
        uncertainties=t.uncertainties
        + [
            f"Possible user-drawn level{' ' + repr(lv.label) if lv.label else ''}, "
            f"but not clearly drawn ({lv.evidence})."
            for lv in t.levels
            if lv.kind == "user_drawn" and lv.inferred
        ],
    )


# --- Errors ------------------------------------------------------------------------------------


class VisionError(Exception):
    """A screenshot could not be analyzed."""


class InvalidImageError(VisionError):
    """The upload is not a decodable, supported, reasonably sized image."""


class VisionUnavailableError(VisionError):
    """The vision model is not configured, unreachable, rate-limited or failed."""


class VisionRefusedError(VisionError):
    """The vision model declined to analyze the image."""


class MalformedVisionOutputError(VisionError):
    """The vision model's output did not match the expected schema."""


# --- Image checks ----------------------------------------------------------------------------

MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 8000  # Claude's per-side limit for image input


class CheckedImage(BaseModel):
    name: str
    media_type: ImageMediaType
    data: str  # base64
    size_bytes: int
    width: int | None
    height: int | None
    sha256: str


def sniff_media_type(raw: bytes) -> ImageMediaType | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_dimensions(raw: bytes, media_type: ImageMediaType) -> tuple[int, int] | None:
    """Width/height from the file header, or None when it can't be read cheaply."""
    try:
        if media_type == "image/png" and raw[12:16] == b"IHDR":
            width, height = struct.unpack(">II", raw[16:24])
            return int(width), int(height)
        if media_type == "image/gif":
            width, height = struct.unpack("<HH", raw[6:10])
            return int(width), int(height)
        if media_type == "image/jpeg":
            i = 2
            while i + 9 < len(raw):
                if raw[i] != 0xFF:
                    return None
                marker = raw[i + 1]
                length = struct.unpack(">H", raw[i + 2 : i + 4])[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    height, width = struct.unpack(">HH", raw[i + 5 : i + 9])
                    return int(width), int(height)
                i += 2 + length
    except struct.error:
        return None
    return None


def check_image(image: ImageAttachment) -> CheckedImage:
    """Decode and sanity-check an upload. Raises `InvalidImageError`."""
    try:
        raw = base64.b64decode(image.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"{image.name} is not valid base64 image data") from exc
    if not raw:
        raise InvalidImageError(f"{image.name} is empty")
    actual = sniff_media_type(raw)
    if actual is None:
        raise InvalidImageError(f"{image.name} is not a PNG, JPEG, WebP or GIF image")
    size = image_dimensions(raw, actual)
    if size is not None:
        width, height = size
        if min(width, height) < MIN_IMAGE_SIDE:
            raise InvalidImageError(
                f"{image.name} is too small to read ({width}x{height}px; minimum "
                f"{MIN_IMAGE_SIDE}px per side)"
            )
        if max(width, height) > MAX_IMAGE_SIDE:
            raise InvalidImageError(
                f"{image.name} is too large ({width}x{height}px; maximum {MAX_IMAGE_SIDE}px per side)"
            )
    return CheckedImage(
        name=image.name,
        # Trust the bytes over the declared type, so a mislabeled upload still works.
        media_type=actual,
        data=image.data,
        size_bytes=len(raw),
        width=size[0] if size else None,
        height=size[1] if size else None,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


# --- Model interface ---------------------------------------------------------------------------


class VisionModel(Protocol):
    name: str

    async def read_chart(self, image: CheckedImage) -> ChartReading:
        """Transcribe the chart. Raises `VisionError` subclasses on failure."""
        ...


SYSTEM_PROMPT = """\
You transcribe trading chart screenshots into structured data for a crypto analysis app.
Your output is treated as visual evidence, never as live market data.

Rules:
- Report only what is shown in the image. Never fill in values from memory or estimation.
- Use "" for any text you cannot read and 0 for any price you cannot read.
- Set an `inferred` flag to false only for things explicitly printed or drawn (labels,
  legends, price scales, drawn lines). Set it to true for things you deduce from what is
  drawn, such as support implied by repeated lows, and explain the deduction in `evidence`.
- List an indicator only if it is actually on the chart (legend, pane or plotted line).
  Include a value only if its number is printed on the chart; otherwise leave `values`
  empty. Copy numbers exactly as displayed, without thousands separators.
- `timeframe` is the interval exactly as displayed (e.g. "4h", "240", "1D", "15").
- `levels` holds horizontal levels: kind "support" or "resistance" for levels on the chart,
  "user_drawn" for horizontal lines the user drew themselves. Use price 0 if you cannot
  read the level's price from a label or the price scale.
- `lines` holds trendlines and channels.
- Set a pattern's `clear` to true only if the visual evidence is unambiguous. Do not list
  patterns you are merely speculating about.
- Put anything relevant that you could not determine in `uncertainties`.
- If the image is not a price chart, set is_price_chart to false and leave the lists empty.
- Do not give trading advice, recommendations, targets or probabilities.
"""

USER_PROMPT = "Transcribe this chart screenshot into the required JSON structure."


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema adapted for structured outputs: every object is closed and
    every property required (optional values are expressed as nullable)."""
    schema = model.model_json_schema()

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                fix(value)
        elif isinstance(node, list):
            for item in node:
                fix(item)

    fix(schema)
    return schema


class ClaudeVisionModel:
    """Reads charts with Claude via the Anthropic SDK and structured outputs."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        timeout: float = 75.0,
        max_retries: int = 1,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.name = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._schema = strict_json_schema(ChartTranscript)

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile).
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key, timeout=self._timeout, max_retries=self._max_retries
            )
        return self._client

    async def read_chart(self, image: CheckedImage) -> ChartReading:
        try:
            response = await self._get_client().beta.messages.create(
                model=self.name,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image.media_type,
                                    "data": image.data,
                                },
                            },
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": self._schema}},
                # On a safety decline, let the API retry on its recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise VisionUnavailableError(
                "the vision model is rate limited; try again shortly"
            ) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise VisionUnavailableError(
                "the vision model rejected UpScale's credentials; check ANTHROPIC_API_KEY"
            ) from exc
        except anthropic.BadRequestError as exc:
            raise VisionUnavailableError(
                f"the vision model rejected the request: {exc.message}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise VisionUnavailableError(
                f"the vision model returned HTTP {exc.status_code}"
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise VisionUnavailableError("the vision model timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise VisionUnavailableError("could not reach the vision model") from exc
        except anthropic.AnthropicError as exc:  # e.g. a credentials/profile problem
            raise VisionUnavailableError(
                "the vision model is not configured; set ANTHROPIC_API_KEY"
            ) from exc
        except TypeError as exc:
            # The SDK raises a plain TypeError when no API key, token or profile exists.
            if "authentication method" not in str(exc):
                raise
            raise VisionUnavailableError(
                "the vision model is not configured; set ANTHROPIC_API_KEY"
            ) from exc

        if response.stop_reason == "refusal":
            raise VisionRefusedError("the vision model declined to analyze this image")
        if response.stop_reason == "max_tokens":
            raise MalformedVisionOutputError("the vision model's output was cut off")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise MalformedVisionOutputError("the vision model returned no text output")
        return parse_reading(text)


def parse_reading(text: str) -> ChartReading:
    """Validate the model's JSON against `ChartTranscript` and expand it to `ChartReading`."""
    try:
        return transcript_to_reading(ChartTranscript.model_validate(json.loads(text)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise MalformedVisionOutputError(
            "the vision model's output did not match the expected chart schema"
        ) from exc


# --- Deterministic post-processing ----------------------------------------------------------------

_TIMEFRAME_MINUTES: dict[int, Timeframe] = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    240: "4h",
    1440: "1d",
}
_UNIT_MINUTES = {"m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1, "h": 60, "hr": 60,
                 "hrs": 60, "hour": 60, "hours": 60, "d": 1440, "day": 1440, "days": 1440}  # fmt: skip
QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "FDUSD", "USD", "EUR", "BTC", "ETH", "PERP")
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,12}$")


def normalize_timeframe(label: str | None) -> Timeframe | None:
    """Map a displayed interval ("4h", "240", "1D", "H1", "15 min") to an UpScale timeframe."""
    if not label:
        return None
    s = label.strip().lower().replace(" ", "")
    minutes: int | None = None
    if s in ("daily", "hourly"):
        minutes = 1440 if s == "daily" else 60
    elif s.isdigit():  # TradingView shows intraday intervals as bare minutes
        minutes = int(s)
    elif m := re.fullmatch(r"(\d*)([a-z]+)", s):
        count, unit = int(m.group(1) or 1), m.group(2)
        if unit in _UNIT_MINUTES:
            minutes = count * _UNIT_MINUTES[unit]
    elif m := re.fullmatch(r"([mhd])(\d+)", s):  # MetaTrader style: M15, H1, D1
        minutes = int(m.group(2)) * _UNIT_MINUTES[m.group(1)]
    return _TIMEFRAME_MINUTES.get(minutes) if minutes else None


def base_symbol(asset: AssetReading) -> str | None:
    """Ticker from the reading, deriving it from a displayed pair when needed.

    "BTC" -> BTC, "BTC/USDT" -> BTC, "BINANCE:ETHUSDT" -> ETH, "SOL-PERP" -> SOL.
    """
    if asset.symbol:
        text, from_pair = asset.symbol, False
    elif asset.pair:
        text, from_pair = asset.pair, True
    else:
        return None
    text = text.upper().split(":")[-1].strip()
    parts = re.split(r"[/\-_ .]", text)
    if len(parts) > 1:
        text = parts[0]
    elif from_pair:
        for quote in QUOTE_SUFFIXES:
            if text.endswith(quote) and len(text) > len(quote):
                text = text[: -len(quote)]
                break
    return text if _SYMBOL_RE.fullmatch(text) else None


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def sanitize(reading: ChartReading) -> tuple[ChartReading, list[str]]:
    """Enforce the no-invention rules on raw model output."""
    discarded: list[str] = []
    uncertainties = list(reading.uncertainties)

    if not reading.is_price_chart:
        empty = ChartReading(
            is_price_chart=False,
            asset=AssetReading(
                symbol=None, pair=None, exchange=None, basis="unknown", evidence=None
            ),
            timeframe=TimeframeReading(label=None, basis="unknown", evidence=None),
            displayed_price=PriceReading(value=None, basis="unknown", evidence=None),
            chart_type="unknown",
            indicators=[],
            support_levels=[],
            resistance_levels=[],
            trend_lines=[],
            drawn_levels=[],
            patterns=[],
            observations=reading.observations,
            uncertainties=uncertainties,
        )
        return empty, ["The image does not look like a price chart; no chart data was extracted."]

    asset = reading.asset
    symbol = base_symbol(asset) if asset.basis != "unknown" else None
    if symbol is None:
        if asset.basis != "unknown":
            discarded.append("Asset could not be determined reliably; reported as unknown.")
        asset = AssetReading(
            symbol=None, pair=None, exchange=asset.exchange, basis="unknown", evidence=None
        )
    else:
        asset = asset.model_copy(update={"symbol": symbol})

    timeframe = reading.timeframe
    if timeframe.basis == "unknown" or not timeframe.label:
        timeframe = TimeframeReading(label=None, basis="unknown", evidence=None)

    price = reading.displayed_price
    if price.basis == "unknown" or not _finite_positive(price.value):
        price = PriceReading(value=None, basis="unknown", evidence=None)

    indicators = []
    for ind in reading.indicators:
        if ind.basis != "visible":
            discarded.append(
                f"Dropped indicator '{ind.name}': not explicitly visible on the chart."
            )
            continue
        values = [v for v in ind.values if math.isfinite(v.value)]
        indicators.append(ind.model_copy(update={"values": values}))

    def levels(items: list[LevelReading], kind: str) -> list[LevelReading]:
        kept = []
        for level in items:
            if _finite_positive(level.price):
                kept.append(level)
            else:
                discarded.append(f"Dropped a {kind} level without a readable price.")
        return kept

    drawn = []
    for d in reading.drawn_levels:
        if _finite_positive(d.price):
            drawn.append(d)
        else:
            discarded.append("Dropped a drawn level without a readable price.")

    patterns = []
    for p in reading.patterns:
        if p.clarity == "clear":
            patterns.append(p)
        else:
            uncertainties.append(
                f"Possible {p.name}, but not clear enough to confirm ({p.evidence})."
            )

    cleaned = reading.model_copy(
        update={
            "asset": asset,
            "timeframe": timeframe,
            "displayed_price": price,
            "indicators": indicators,
            "support_levels": levels(reading.support_levels, "support"),
            "resistance_levels": levels(reading.resistance_levels, "resistance"),
            "drawn_levels": drawn,
            "patterns": patterns,
            "uncertainties": uncertainties,
        }
    )
    return cleaned, discarded


# --- Service ----------------------------------------------------------------------------------------


class VisionService:
    """Validates screenshots, calls the vision model, sanitizes and caches the results."""

    def __init__(
        self,
        model: VisionModel,
        cache_ttl: float = 3600.0,
        cache_size: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.model = model
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self._clock = clock
        self._cache: OrderedDict[str, tuple[float, ChartVision]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def reset(self) -> None:
        self._cache.clear()
        self._locks.clear()

    async def analyze(self, image: ImageAttachment) -> ChartVision:
        """Raises `VisionError` subclasses; never returns unvalidated model output."""
        checked = check_image(image)
        key = f"{self.model.name}:{checked.sha256}"
        if (hit := self._cached(key)) is not None:
            return hit.model_copy(update={"image_name": image.name})
        async with self._locks.setdefault(key, asyncio.Lock()):
            if (hit := self._cached(key)) is not None:
                return hit.model_copy(update={"image_name": image.name})
            reading, discarded = sanitize(await self.model.read_chart(checked))
            result = ChartVision(
                image_name=checked.name,
                media_type=checked.media_type,
                size_bytes=checked.size_bytes,
                width=checked.width,
                height=checked.height,
                model=self.model.name,
                reading=reading,
                normalized_timeframe=normalize_timeframe(reading.timeframe.label),
                discarded=discarded,
                analyzed_at=datetime.now(UTC),
            )
            self._cache[key] = (self._clock() + self.cache_ttl, result)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return result

    def _cached(self, key: str) -> ChartVision | None:
        entry = self._cache.get(key)
        if entry and entry[0] > self._clock():
            self._cache.move_to_end(key)
            return entry[1]
        return None
