import asyncio
import base64
import struct

import pytest

from upscale.schemas import ImageAttachment
from upscale.services.vision import (
    AssetReading,
    ChartReading,
    InvalidImageError,
    VisionService,
    VisionUnavailableError,
    base_symbol,
    check_image,
    image_dimensions,
    normalize_timeframe,
    sanitize,
    sniff_media_type,
)

from .conftest import CHART_PNG, FakeVisionModel, chart_reading_data, make_png


def attachment(raw: bytes, media_type="image/png", name="chart.png") -> ImageAttachment:
    return ImageAttachment(name=name, media_type=media_type, data=base64.b64encode(raw).decode())


def reading(**overrides) -> ChartReading:
    return ChartReading.model_validate(chart_reading_data(**overrides))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- Image checks -------------------------------------------------------------------------


def test_valid_png_passes_with_dimensions():
    checked = check_image(attachment(make_png(320, 200)))
    assert (checked.media_type, checked.width, checked.height) == ("image/png", 320, 200)
    assert checked.size_bytes == len(make_png(320, 200))
    assert len(checked.sha256) == 64


def test_empty_upload_is_rejected():
    with pytest.raises(InvalidImageError, match="empty"):
        check_image(ImageAttachment(name="x.png", media_type="image/png", data=""))


def test_non_image_bytes_are_rejected():
    with pytest.raises(InvalidImageError, match="not a PNG, JPEG, WebP or GIF"):
        check_image(attachment(b"%PDF-1.7 definitely not an image"))


def test_mislabeled_image_uses_the_real_type():
    gif = b"GIF89a" + struct.pack("<HH", 100, 80) + b"\x00" * 20
    checked = check_image(attachment(gif, media_type="image/png"))
    assert checked.media_type == "image/gif"
    assert (checked.width, checked.height) == (100, 80)


@pytest.mark.parametrize(("size", "message"), [((1, 1), "too small"), ((20, 400), "too small")])
def test_tiny_images_are_rejected(size, message):
    with pytest.raises(InvalidImageError, match=message):
        check_image(attachment(make_png(*size)))


def test_oversized_images_are_rejected():
    header_only = make_png(10, 10)[:16] + struct.pack(">II", 9000, 400)
    with pytest.raises(InvalidImageError, match="too large"):
        check_image(attachment(header_only))


def test_jpeg_dimensions_are_read_from_sof():
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
        + b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 600, 1024) + b"\x00" * 10
    )  # fmt: skip
    assert sniff_media_type(jpeg) == "image/jpeg"
    assert image_dimensions(jpeg, "image/jpeg") == (1024, 600)


def test_webp_is_accepted_without_dimensions():
    webp = b"RIFF" + b"\x00" * 4 + b"WEBPVP8 " + b"\x00" * 20
    checked = check_image(attachment(webp, media_type="image/webp"))
    assert checked.media_type == "image/webp"
    assert checked.width is None


# --- Timeframe and asset normalization ------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("4h", "4h"), ("4H", "4h"), ("240", "4h"), ("4 hours", "4h"),
        ("1h", "1h"), ("60", "1h"), ("H1", "1h"), ("hourly", "1h"),
        ("15", "15m"), ("15m", "15m"), ("15 min", "15m"), ("M15", "15m"),
        ("1", "1m"), ("5m", "5m"),
        ("1D", "1d"), ("D", "1d"), ("daily", "1d"), ("D1", "1d"),
        ("1W", None), ("2h", None), ("3m", None), ("", None), (None, None), ("??", None),
    ],
)  # fmt: skip
def test_normalize_timeframe(label, expected):
    assert normalize_timeframe(label) == expected


@pytest.mark.parametrize(
    ("symbol", "pair", "expected"),
    [
        ("BTC", None, "BTC"), ("eth", None, "ETH"), (None, "BTC/USDT", "BTC"),
        (None, "BINANCE:ETHUSDT", "ETH"), (None, "SOL-PERP", "SOL"), (None, "XRPUSD", "XRP"),
        (None, None, None), ("???", None, None), ("A VERY LONG NAME THAT IS NOT A TICKER", None, None),
    ],
)  # fmt: skip
def test_base_symbol(symbol, pair, expected):
    asset = AssetReading(symbol=symbol, pair=pair, exchange=None, basis="visible", evidence=None)
    assert base_symbol(asset) == expected


# --- Guardrails ---------------------------------------------------------------------------------


def test_sanitize_keeps_visible_information():
    cleaned, _ = sanitize(reading())
    assert cleaned.asset.symbol == "BTC" and cleaned.asset.pair == "BTC/USDT"
    assert cleaned.timeframe.label == "4h"
    assert cleaned.displayed_price.value == 64210.5
    assert [i.name for i in cleaned.indicators] == ["RSI"]
    assert cleaned.indicators[0].values[0].value == 58.2
    assert [lv.price for lv in cleaned.support_levels] == [62000.0]
    assert [lv.price for lv in cleaned.resistance_levels] == [66000.0]
    assert [d.price for d in cleaned.drawn_levels] == [63000.0]
    assert cleaned.trend_lines[0].kind == "channel"


def test_indicators_that_are_not_visible_are_dropped():
    cleaned, discarded = sanitize(reading())
    assert "EMA" not in [i.name for i in cleaned.indicators]
    assert "Dropped indicator 'EMA': not explicitly visible on the chart." in discarded


def test_levels_without_a_readable_price_are_dropped():
    cleaned, discarded = sanitize(reading())
    assert all(lv.price is not None for lv in cleaned.support_levels)
    assert "Dropped a support level without a readable price." in discarded


def test_tentative_patterns_are_not_claimed():
    cleaned, _ = sanitize(reading())
    assert [p.name for p in cleaned.patterns] == ["ascending triangle"]
    assert any(u.startswith("Possible double top, but not clear") for u in cleaned.uncertainties)


def test_unknown_asset_and_timeframe_stay_unknown():
    raw = reading(
        asset={
            "symbol": None,
            "pair": None,
            "exchange": None,
            "basis": "unknown",
            "evidence": None,
        },
        timeframe={"label": None, "basis": "unknown", "evidence": None},
        displayed_price={"value": None, "basis": "unknown", "evidence": None},
    )
    cleaned, _ = sanitize(raw)
    assert cleaned.asset.symbol is None and cleaned.asset.basis == "unknown"
    assert cleaned.timeframe.label is None
    assert cleaned.displayed_price.value is None


def test_values_marked_unknown_are_not_kept_even_if_filled_in():
    raw = reading(
        asset={
            "symbol": "BTC",
            "pair": None,
            "exchange": None,
            "basis": "unknown",
            "evidence": None,
        },
        timeframe={"label": "4h", "basis": "unknown", "evidence": None},
    )
    cleaned, _ = sanitize(raw)
    assert cleaned.asset.symbol is None
    assert cleaned.timeframe.label is None


def test_unreadable_asset_is_reported_as_unknown():
    raw = reading(
        asset={
            "symbol": "?!",
            "pair": None,
            "exchange": None,
            "basis": "inferred",
            "evidence": "blurry",
        }
    )
    cleaned, discarded = sanitize(raw)
    assert cleaned.asset.symbol is None and cleaned.asset.basis == "unknown"
    assert any("Asset could not be determined" in d for d in discarded)


@pytest.mark.parametrize("value", [0.0, -5.0, float("nan"), float("inf")])
def test_invalid_displayed_price_is_dropped(value):
    raw = reading(displayed_price={"value": value, "basis": "visible", "evidence": "tag"})
    assert sanitize(raw)[0].displayed_price.value is None


def test_non_chart_images_yield_no_chart_data():
    cleaned, discarded = sanitize(reading(is_price_chart=False))
    assert cleaned.asset.symbol is None
    assert cleaned.indicators == cleaned.support_levels == cleaned.patterns == []
    assert discarded == ["The image does not look like a price chart; no chart data was extracted."]


# --- Service -------------------------------------------------------------------------------------


def chart(name="chart.png") -> ImageAttachment:
    return ImageAttachment(name=name, media_type="image/png", data=CHART_PNG)


def test_service_returns_sanitized_structured_result():
    model = FakeVisionModel()
    result = asyncio.run(VisionService(model).analyze(chart()))
    assert result.model == "fake-vision"
    assert result.normalized_timeframe == "4h"
    assert result.reading.asset.symbol == "BTC"
    assert (result.width, result.height) == (320, 200)
    assert result.discarded  # the inferred EMA and priceless support level
    assert model.calls[0].media_type == "image/png"


def test_service_maps_unsupported_timeframe_label_to_none():
    model = FakeVisionModel()
    model.reading = reading(timeframe={"label": "1W", "basis": "visible", "evidence": "toolbar"})
    result = asyncio.run(VisionService(model).analyze(chart()))
    assert result.reading.timeframe.label == "1W"
    assert result.normalized_timeframe is None


def test_invalid_images_never_reach_the_model():
    model = FakeVisionModel()
    tiny = ImageAttachment(
        name="tiny.png", media_type="image/png", data=base64.b64encode(make_png(2, 2)).decode()
    )
    with pytest.raises(InvalidImageError):
        asyncio.run(VisionService(model).analyze(tiny))
    assert model.calls == []


def test_model_errors_propagate():
    model = FakeVisionModel()
    model.error = VisionUnavailableError("the vision model timed out")
    with pytest.raises(VisionUnavailableError, match="timed out"):
        asyncio.run(VisionService(model).analyze(chart()))


def test_results_are_cached_per_image_content():
    clock = FakeClock()
    model = FakeVisionModel()
    service = VisionService(model, cache_ttl=60, clock=clock)
    asyncio.run(service.analyze(chart("a.png")))
    second = asyncio.run(service.analyze(chart("b.png")))
    assert len(model.calls) == 1
    assert second.image_name == "b.png"
    clock.now += 61
    asyncio.run(service.analyze(chart()))
    assert len(model.calls) == 2


def test_failures_are_not_cached():
    model = FakeVisionModel()
    service = VisionService(model)
    model.error = VisionUnavailableError("down")
    with pytest.raises(VisionUnavailableError):
        asyncio.run(service.analyze(chart()))
    model.error = None
    assert asyncio.run(service.analyze(chart())).reading.asset.symbol == "BTC"


def test_cache_is_bounded():
    model = FakeVisionModel()
    service = VisionService(model, cache_size=2)
    for size in (40, 41, 42):
        img = ImageAttachment(
            name="x.png",
            media_type="image/png",
            data=base64.b64encode(make_png(size, size)).decode(),
        )
        asyncio.run(service.analyze(img))
    assert len(service._cache) == 2
