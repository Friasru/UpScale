"""Kraken candles: provider parsing/validation, service precedence and fallback, and the
Technical and Risk agents running on real intraday timeframes. All offline."""

import asyncio
import itertools
import json
from datetime import UTC, datetime, timedelta

import httpx2
import pytest

from upscale.orchestrator import Orchestrator
from upscale.schemas import ChatMessage, ChatRequest, ImageAttachment
from upscale.services.coingecko import CoinGeckoProvider
from upscale.services.kraken import (
    KRAKEN_INTERVALS,
    MAX_COMPLETED_CANDLES,
    KrakenPair,
    KrakenProvider,
    resolve_pair,
)
from upscale.services.market_data import (
    TIMEFRAME_SECONDS,
    TIMEFRAMES,
    AssetNotFoundError,
    CandleSeries,
    InsufficientDataError,
    MarketDataService,
    MarketDataUnavailableError,
    UnsupportedTimeframeError,
)
from upscale.services.vision import ChartReading

from .conftest import (
    CHART_PNG,
    KRAKEN_NOW_S,
    FakeCoinGecko,
    FakeKraken,
    chart_reading_data,
    kraken_body,
    kraken_error,
    kraken_rows,
)
from .test_technical_agent import serve_wave

NOW = datetime.fromtimestamp(KRAKEN_NOW_S, UTC)  # open time of the in-progress candle


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def kraken() -> FakeKraken:
    return FakeKraken()


@pytest.fixture
def gecko() -> FakeCoinGecko:
    return FakeCoinGecko()


def make_service(kraken: FakeKraken, gecko: FakeCoinGecko, **kwargs) -> MarketDataService:
    coingecko = CoinGeckoProvider(transport=gecko.transport())
    return MarketDataService(
        coingecko,
        candle_providers=[KrakenProvider(transport=kraken.transport()), coingecko],
        **kwargs,
    )


def candles(service: MarketDataService, symbol="BTC", timeframe="1m", **kwargs) -> CandleSeries:
    return asyncio.run(service.get_candles(symbol, timeframe, **kwargs))


def fetch(kraken: FakeKraken, symbol="BTC", timeframe="1m", limit=10) -> CandleSeries:
    provider = KrakenProvider(transport=kraken.transport())
    return asyncio.run(provider.fetch_candles(symbol, timeframe, limit))


def respond(kraken: FakeKraken, rows: list[list] | None = None, *, body=None, status=200):
    content = kraken_body(rows) if rows is not None else body
    if not isinstance(content, bytes):
        content = json.dumps(content).encode()
    kraken.handler = lambda request: httpx2.Response(status, content=content)


def _raise(exc):
    def handler(request):
        raise exc

    return handler


# --- Symbol mapping ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "pair", "request_pair"),
    [
        ("BTC", "BTC/USD", "XBTUSD"),
        ("btc", "BTC/USD", "XBTUSD"),
        ("ETH", "ETH/USD", "ETHUSD"),
        ("SOL", "SOL/USD", "SOLUSD"),
        ("DOGE", "DOGE/USD", "XDGUSD"),
        ("LINK", "LINK/USD", "LINKUSD"),  # any other ticker maps generically
    ],
)
def test_symbol_mapping(symbol, pair, request_pair):
    assert resolve_pair(symbol) == KrakenPair(symbol.upper(), pair, request_pair)


@pytest.mark.parametrize("symbol", ["", "B", "BTC/USD", "BTC USD", "A" * 11, "BTC&x=1"])
def test_invalid_tickers_are_rejected_without_a_request(kraken, symbol):
    with pytest.raises(AssetNotFoundError):
        fetch(kraken, symbol)
    assert kraken.requests == []


# --- Provider: timeframes and parsing ---------------------------------------------------


def test_kraken_supports_every_upscale_timeframe_natively():
    assert KrakenProvider.supported_timeframes == frozenset(TIMEFRAMES)
    assert TIMEFRAMES == ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
    for tf, minutes in KRAKEN_INTERVALS.items():
        assert minutes * 60 == TIMEFRAME_SECONDS[tf]


@pytest.mark.parametrize(
    ("timeframe", "interval"),
    [
        ("1m", "1"),
        ("5m", "5"),
        ("15m", "15"),
        ("30m", "30"),
        ("1h", "60"),
        ("4h", "240"),
        ("1d", "1440"),
    ],
)
def test_btc_candles_for_each_timeframe(kraken, timeframe, interval):
    series = fetch(kraken, "BTC", timeframe, limit=100)
    [request] = kraken.requests
    assert request.url.host == "api.kraken.com"
    assert request.url.path == "/0/public/OHLC"
    assert dict(request.url.params) == {"pair": "XBTUSD", "interval": interval}
    step = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    assert series.timeframe == timeframe
    assert series.interval == step
    assert (series.provider, series.provider_id, series.pair) == ("Kraken", "XBTUSD", "BTC/USD")
    assert series.volume_available is True
    # 720 rows minus the in-progress one; the newest completed candle closes at NOW.
    assert len(series.candles) == MAX_COMPLETED_CANDLES
    assert series.candles[-1].timestamp == NOW - step
    assert all(b.timestamp - a.timestamp == step for a, b in itertools.pairwise(series.candles))


def test_in_progress_candle_is_dropped(kraken):
    rows = kraken_rows(3, 60)
    respond(kraken, rows)
    series = fetch(kraken, limit=2)
    assert [c.timestamp for c in series.candles] == [
        NOW - timedelta(minutes=2),
        NOW - timedelta(minutes=1),
    ]


def test_ohlcv_values_and_real_volume(kraken):
    ts = KRAKEN_NOW_S - 60
    respond(
        kraken,
        [
            [ts, "64000.1", "64100.5", "63950.0", "64050.25", "64020.0", "12.34567890", 321],
            [KRAKEN_NOW_S, "64050.25", "64060.0", "64040.0", "64055.0", "64050.0", "0.5", 7],
        ],
    )
    [candle] = fetch(kraken, limit=1).candles
    assert candle.timestamp == NOW - timedelta(minutes=1)
    assert (candle.open, candle.high, candle.low, candle.close) == (
        64000.1,
        64100.5,
        63950.0,
        64050.25,
    )
    assert candle.volume == 12.3456789


def test_zero_volume_candles_are_valid(kraken):
    rows = kraken_rows(4, 60)
    rows[1][6] = "0.00000000"
    respond(kraken, rows)
    assert fetch(kraken, limit=3).candles[1].volume == 0.0


@pytest.mark.parametrize(("symbol", "key"), [("ETH", "XETHZUSD"), ("SOL", "SOLUSD")])
def test_eth_and_sol(kraken, symbol, key):
    series = fetch(kraken, symbol, "5m", limit=50)
    assert series.symbol == symbol
    assert series.pair == f"{symbol}/USD"
    assert kraken.requests[0].url.params["pair"] == f"{symbol}USD"
    assert series.candles[-1].close > 0


def test_unsupported_timeframe_is_rejected_without_a_request(kraken):
    with pytest.raises(UnsupportedTimeframeError):
        asyncio.run(KrakenProvider(transport=kraken.transport()).fetch_candles("BTC", "2h", 5))  # type: ignore[arg-type]
    assert kraken.requests == []


def test_more_candles_than_kraken_offers_is_rejected_up_front(kraken):
    with pytest.raises(InsufficientDataError, match="at most 719"):
        fetch(kraken, limit=720)
    assert kraken.requests == []


# --- Provider: failures and bad data ----------------------------------------------------


def test_unknown_pair_is_not_found(kraken):
    with pytest.raises(
        AssetNotFoundError, match=r"NOTACOIN \(NOTACOIN/USD\) is not traded on Kraken"
    ):
        fetch(kraken, "NOTACOIN")


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda r: httpx2.Response(500), "HTTP 500"),
        (lambda r: httpx2.Response(429), "rate limit"),
        (lambda r: kraken_error("EGeneral:Too many requests"), "rate limit"),
        (lambda r: kraken_error("EAPI:Rate limit exceeded"), "rate limit"),
        (lambda r: kraken_error("EService:Unavailable"), "EService:Unavailable"),
        (_raise(httpx2.ReadTimeout("slow")), "timed out"),
        (_raise(httpx2.ConnectError("down")), "could not reach"),
        (lambda r: httpx2.Response(200, content=b"<html>"), "invalid JSON"),
        (lambda r: httpx2.Response(200, content=b"[]"), "unexpected response"),
        (lambda r: httpx2.Response(200, content=b'{"error": []}'), "unexpected response"),
        (lambda r: httpx2.Response(200, content=b'{"error": "x", "result": {}}'), "unexpected"),
        (
            lambda r: httpx2.Response(200, content=b'{"error": [], "result": {"last": 1}}'),
            "unexpected response",
        ),
        (
            lambda r: httpx2.Response(
                200, content=b'{"error": [], "result": {"A": [], "B": [], "last": 1}}'
            ),
            "unexpected response",
        ),
    ],
    ids=[
        "http-500",
        "http-429",
        "too-many-requests",
        "rate-limit-exceeded",
        "service-error",
        "timeout",
        "connect-error",
        "bad-json",
        "not-an-object",
        "no-result",
        "error-not-a-list",
        "no-pair-rows",
        "two-pairs",
    ],
)
def test_api_failures(kraken, handler, message):
    kraken.handler = handler
    with pytest.raises(MarketDataUnavailableError, match=message):
        fetch(kraken)


def _row(ts=KRAKEN_NOW_S - 60, o="10", h="12", lo="9", c="11", vol="5"):
    return [ts, o, h, lo, c, "10.5", vol, 3]


LIVE = _row(ts=KRAKEN_NOW_S)  # the in-progress candle, dropped before validation


@pytest.mark.parametrize(
    "row",
    [
        [KRAKEN_NOW_S - 60, "10", "12", "9", "11", "10.5"],  # no volume column
        _row(o=10.0),  # number instead of decimal string
        _row(c=None),
        _row(c="abc"),
        _row(h="9.5"),  # high below the open
        _row(lo="10.5"),  # low above the open
        _row(o="0", h="0", lo="0", c="0"),
        _row(lo="-1"),
        _row(c="NaN"),
        _row(h="inf"),
        _row(vol="-1"),
        _row(vol="nan"),
        _row(vol=None),
        _row(ts="1790265540"),  # time as a string
        _row(ts=1.5e30),
        {"time": KRAKEN_NOW_S - 60},
    ],
    ids=[
        "short-row",
        "number-price",
        "null-close",
        "text-close",
        "high-too-low",
        "low-too-high",
        "zero-prices",
        "negative-low",
        "nan-close",
        "inf-high",
        "negative-volume",
        "nan-volume",
        "null-volume",
        "string-time",
        "huge-time",
        "object-row",
    ],
)
def test_malformed_candles_are_rejected(kraken, row):
    respond(kraken, [row, LIVE])
    with pytest.raises(MarketDataUnavailableError, match="malformed"):
        fetch(kraken, limit=1)


def test_misaligned_candles_are_not_relabeled(kraken):
    respond(kraken, [_row(ts=KRAKEN_NOW_S - 90), LIVE])
    with pytest.raises(MarketDataUnavailableError, match="not aligned to 1m"):
        fetch(kraken, limit=1)


def test_duplicate_timestamps_are_rejected(kraken):
    rows = kraken_rows(6, 60)
    rows.insert(2, list(rows[2]))
    respond(kraken, rows)
    with pytest.raises(MarketDataUnavailableError, match="duplicate or out-of-order"):
        fetch(kraken, limit=3)


def test_out_of_order_candles_are_rejected(kraken):
    rows = kraken_rows(6, 60)
    rows[1], rows[2] = rows[2], rows[1]
    respond(kraken, rows)
    with pytest.raises(MarketDataUnavailableError, match="duplicate or out-of-order"):
        fetch(kraken, limit=3)


def test_missing_candle_is_a_gap_not_filled_in(kraken):
    rows = kraken_rows(10, 60)
    del rows[4]
    respond(kraken, rows)
    with pytest.raises(MarketDataUnavailableError, match="1m candles with a gap"):
        fetch(kraken, limit=5)


def test_candles_of_another_interval_are_not_relabeled(kraken):
    respond(kraken, kraken_rows(10, 5 * 60))  # 5m rows for a 1m request
    with pytest.raises(MarketDataUnavailableError, match="gap"):
        fetch(kraken, "BTC", "1m", limit=5)


def test_only_the_live_candle_means_no_history(kraken, gecko):
    respond(kraken, [LIVE])
    with pytest.raises(InsufficientDataError, match="Kraken has only 0 1m candle"):
        candles(make_service(kraken, gecko), limit=1)


# --- Service: precedence, fallback, caching, rate limits, dedup -------------------------


def test_intraday_timeframes_come_from_kraken(kraken, gecko):
    service = make_service(kraken, gecko)
    assert service.supported_timeframes == list(TIMEFRAMES)
    for tf in ("1m", "5m", "15m", "30m", "1h", "1d"):
        series = candles(service, timeframe=tf, limit=100)
        assert (series.provider, series.timeframe, series.fallback_notes) == ("Kraken", tf, [])
        assert len(series.candles) == 100
    assert gecko.requests == []


def test_4h_prefers_kraken(kraken, gecko):
    series = candles(make_service(kraken, gecko), timeframe="4h", limit=100)
    assert series.provider == "Kraken"
    assert series.volume_available is True
    assert gecko.requests == []


def test_4h_falls_back_to_coingecko_when_kraken_fails(kraken, gecko):
    kraken.handler = _raise(httpx2.ReadTimeout("slow"))
    series = candles(make_service(kraken, gecko), timeframe="4h", limit=100)
    assert series.provider == "CoinGecko"
    assert series.timeframe == "4h"
    assert series.volume_available is False
    assert series.fallback_notes == [
        "Kraken 4h candles were unavailable (Kraken request timed out)"
    ]


def test_4h_falls_back_to_coingecko_for_assets_kraken_lacks(kraken, gecko):
    series = candles(make_service(kraken, gecko), symbol="ETH", timeframe="4h", limit=10)
    assert series.provider == "Kraken"
    kraken.handler = lambda r: kraken_error("EQuery:Unknown asset pair")
    series = candles(make_service(kraken, gecko), symbol="SOL", timeframe="4h", limit=10)
    assert series.provider == "CoinGecko"
    assert "not traded on Kraken" in series.fallback_notes[0]


def test_1m_never_falls_back_to_another_timeframe(kraken, gecko):
    kraken.handler = lambda r: kraken_error("EGeneral:Too many requests")
    with pytest.raises(MarketDataUnavailableError, match="Kraken rate limit reached"):
        candles(make_service(kraken, gecko), timeframe="1m", limit=10)
    assert gecko.requests == []  # CoinGecko has no 1m candles, so it isn't asked


def test_unknown_asset_everywhere_is_not_found(kraken, gecko):
    gecko.handler = lambda r: httpx2.Response(200, content=b"[]")
    with pytest.raises(AssetNotFoundError, match="Kraken: .*; CoinGecko: .*NOTACOIN"):
        candles(make_service(kraken, gecko), symbol="NOTACOIN", timeframe="4h", limit=5)


def test_both_providers_down_is_unavailable(kraken, gecko):
    kraken.handler = lambda r: httpx2.Response(502)
    gecko.handler = lambda r: httpx2.Response(503)
    with pytest.raises(MarketDataUnavailableError, match="Kraken: .*502; CoinGecko: .*503"):
        candles(make_service(kraken, gecko), timeframe="4h", limit=5)


def test_bad_kraken_data_falls_back_rather_than_being_used(kraken, gecko):
    rows = kraken_rows(200, 4 * 3600)
    del rows[50]
    respond(kraken, rows)
    series = candles(make_service(kraken, gecko), timeframe="4h", limit=100)
    assert series.provider == "CoinGecko"
    assert "gap" in series.fallback_notes[0]


def test_short_kraken_history_falls_back_when_more_is_required(kraken, gecko):
    respond(kraken, kraken_rows(31, 4 * 3600))  # 30 completed candles
    service = make_service(kraken, gecko)
    assert candles(service, timeframe="4h", limit=100, min_candles=10).provider == "Kraken"
    series = candles(service, timeframe="4h", limit=100, min_candles=50)
    assert series.provider == "CoinGecko"
    assert "only 30" in series.fallback_notes[0]


def test_candles_are_cached_per_provider_asset_and_timeframe(kraken, gecko):
    clock = FakeClock()
    service = make_service(kraken, gecko, clock=clock)
    first = candles(service, limit=100)
    assert candles(service, limit=100) == first
    assert len(candles(service, limit=20).candles) == 20  # served from the larger fetch
    assert len(kraken.requests) == 1
    candles(service, limit=200)  # more candles than were cached
    candles(service, timeframe="5m", limit=100)
    candles(service, symbol="ETH", limit=100)
    assert len(kraken.requests) == 4


@pytest.mark.parametrize(
    ("timeframe", "ttl"),
    [("1m", 15), ("5m", 75), ("15m", 225), ("30m", 300), ("1h", 300), ("4h", 300), ("1d", 300)],
)
def test_intraday_cache_ttls_are_short(kraken, gecko, timeframe, ttl):
    clock = FakeClock()
    service = make_service(kraken, gecko, clock=clock)
    assert service.candle_cache_ttl(timeframe) == ttl
    candles(service, timeframe=timeframe, limit=10)
    clock.now += ttl - 1
    candles(service, timeframe=timeframe, limit=10)
    assert len(kraken.requests) == 1
    clock.now += 1
    candles(service, timeframe=timeframe, limit=10)
    assert len(kraken.requests) == 2


def test_failures_are_not_cached(kraken, gecko):
    service = make_service(kraken, gecko)
    kraken.handler = _raise(httpx2.ReadTimeout("slow"))
    with pytest.raises(MarketDataUnavailableError):
        candles(service, limit=5)
    kraken.handler = kraken.ohlc
    assert candles(service, limit=5).provider == "Kraken"
    assert len(kraken.requests) == 2


def test_unknown_pairs_are_remembered(kraken, gecko):
    service = make_service(kraken, gecko)
    for _ in range(2):
        with pytest.raises(AssetNotFoundError):
            candles(service, symbol="NOTACOIN", timeframe="1m", limit=5)
    assert len(kraken.requests) == 1


def test_concurrent_identical_requests_share_one_call(kraken, gecko):
    service = make_service(kraken, gecko)

    async def scenario():
        return await asyncio.gather(*(service.get_candles("BTC", "1m", 50) for _ in range(5)))

    results = asyncio.run(scenario())
    assert len(kraken.requests) == 1
    assert all(r == results[0] for r in results)


def test_kraken_has_its_own_rate_limit(kraken, gecko):
    clock = FakeClock()
    service = make_service(
        kraken, gecko, clock=clock, max_calls_per_minute=5, provider_calls_per_minute={"Kraken": 2}
    )
    candles(service, symbol="BTC", limit=5)
    candles(service, symbol="ETH", limit=5)
    with pytest.raises(MarketDataUnavailableError, match="Kraken request limit was reached"):
        candles(service, symbol="SOL", limit=5)
    assert len(kraken.requests) == 2
    assert candles(service, symbol="BTC", limit=5).provider == "Kraken"  # cache still served
    asyncio.run(service.get_snapshot("BTC"))  # CoinGecko's budget is separate
    clock.now += 60
    candles(service, symbol="SOL", limit=5)
    assert len(kraken.requests) == 3


def test_kraken_rate_limited_locally_falls_back_for_4h(kraken, gecko):
    service = make_service(kraken, gecko, provider_calls_per_minute={"Kraken": 1})
    candles(service, symbol="BTC", timeframe="4h", limit=5)
    series = candles(service, symbol="ETH", timeframe="4h", limit=5)
    assert series.provider == "CoinGecko"
    assert "request limit was reached" in series.fallback_notes[0]


# --- Technical and Risk agents via the orchestrator (shared service, fake Kraken) --------


def ask(content: str, *images: ImageAttachment):
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=content, attachments=list(images))]
    )
    return asyncio.run(Orchestrator().respond(request))


def by_agent(response):
    return {r.agent: r for r in response.analysis.agent_results}


def chart() -> ImageAttachment:
    return ImageAttachment(name="chart.png", media_type="image/png", data=CHART_PNG)


def factor_ids(risk) -> set[str]:
    return {f["id"] for f in risk.findings["factors"]}


def test_technical_analyzes_the_requested_1m_timeframe(fake_kraken):
    response = ask("BTC 1m RSI and support levels")
    technical = by_agent(response)["technical_analysis"]
    f = technical.findings
    assert technical.status == "ok"
    assert (f["timeframe"], f["requested_timeframe"], f["timeframe_note"]) == ("1m", "1m", None)
    assert f["candle_source"] == {
        "provider": "Kraken",
        "pair": "BTC/USD",
        "requested_timeframe": "1m",
        "actual_timeframe": "1m",
        "candle_count": 180,
        "volume_available": True,
        "fallback_notes": [],
    }
    assert technical.evidence[0].startswith(
        "BTC 1m: 180 candles from Kraken (BTC/USD) with per-candle volume"
    )
    assert fake_kraken.requests[0].url.params["interval"] == "1"
    assert not any("not available" in line for line in technical.evidence)


def test_technical_uses_real_volume_from_kraken(fake_kraken):
    technical = by_agent(ask("BTC 5m chart"))["technical_analysis"]
    f = technical.findings
    assert f["volume_available"] is True
    assert f["volume"]["available"] is True
    assert f["volume"]["unit"] == "BTC"
    assert f["volume"]["relative_volume"] > 0
    text = " ".join(technical.evidence + [r.description for r in technical.risks])
    assert "Last 5m candle volume" in text
    assert "does not supply per-candle volume" not in text
    assert "volume is unavailable" not in text


def test_technical_1m_reports_kraken_failure_instead_of_switching_to_4h(
    fake_kraken, fake_coingecko
):
    fake_kraken.handler = lambda r: kraken_error("EGeneral:Too many requests")
    technical = by_agent(ask("BTC 1m chart"))["technical_analysis"]
    assert technical.status == "error"
    assert "Kraken rate limit reached" in technical.summary
    assert not any(r.url.path.endswith("/ohlc") for r in fake_coingecko.requests)


def test_technical_4h_falls_back_to_coingecko_and_says_so(fake_kraken, fake_coingecko):
    serve_wave(fake_coingecko)
    fake_kraken.handler = lambda r: httpx2.Response(503)
    technical = by_agent(ask("BTC 4h chart"))["technical_analysis"]
    f = technical.findings
    assert (f["provider"], f["timeframe"]) == ("CoinGecko", "4h")
    assert f["candle_source"]["fallback_notes"] == [
        "Kraken 4h candles were unavailable (Kraken returned HTTP 503)"
    ]
    assert (
        "Kraken 4h candles were unavailable (Kraken returned HTTP 503); used CoinGecko instead."
        in technical.evidence
    )


def test_1m_screenshot_is_analyzed_on_1m_and_risk_sees_no_mismatch(fake_kraken, fake_vision):
    fake_vision.reading = ChartReading.model_validate(
        chart_reading_data(timeframe={"label": "1m", "basis": "visible", "evidence": "toolbar"})
    )
    response = ask("", chart())
    results = by_agent(response)
    technical = results["technical_analysis"]
    assert technical.findings["requested_timeframe_source"] == "screenshot"
    assert (technical.findings["requested_timeframe"], technical.findings["timeframe"]) == (
        "1m",
        "1m",
    )
    assert technical.findings["timeframe_note"] is None
    ids = factor_ids(results["risk"])
    assert "timeframe_mismatch" not in ids
    assert "unsupported_timeframe" not in ids
    assert "volume_unavailable" not in ids
    assert "1m/4h" not in response.message.content


def test_without_kraken_a_1m_screenshot_still_falls_back_and_risk_flags_it(
    fake_coingecko, fake_vision
):
    """Control: CoinGecko-only configuration keeps the old, clearly-labelled 4h fallback."""
    serve_wave(fake_coingecko)
    fake_vision.reading = ChartReading.model_validate(
        chart_reading_data(timeframe={"label": "1m", "basis": "visible", "evidence": "toolbar"})
    )
    results = by_agent(ask("", chart()))
    assert results["technical_analysis"].findings["timeframe"] == "4h"
    assert "timeframe_mismatch" in factor_ids(results["risk"])
