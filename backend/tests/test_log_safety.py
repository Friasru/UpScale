"""Secret-safe logging: credentials never reach log output, at any level, from any logger.

Only fake secrets are used. Assertions about the real configured credentials compare
counts, so a failure can never print a real value.
"""

import asyncio
import logging
from collections.abc import Iterator

import httpx2
import pytest

from upscale import log_safety
from upscale.services.solana_chain import HeliusProvider

FAKE_HELIUS = "fake-helius-key-0123456789abcdef"
FAKE_BEARER = "fakeBearerToken_ABCdef0123456789"
FAKE_NEYNAR = "FAKE-NEYNAR-KEY-9876543210"
FAKE_UNREGISTERED = "unregisteredSecret42XYZ"


@pytest.fixture(autouse=True)
def fake_secrets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The process registry, plus fake secrets, restored afterwards."""
    monkeypatch.setattr(log_safety, "_secrets", set(log_safety._secrets))
    log_safety.register_secrets([FAKE_HELIUS, FAKE_BEARER, FAKE_NEYNAR])
    yield


@pytest.fixture
def debug_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)  # debug mode must be secret-safe too
    return caplog


def assert_clean(text: str, *secrets: str) -> None:
    leaked = [i for i, s in enumerate(secrets) if s in text]
    assert leaked == [], f"secret(s) #{leaked} leaked into logs"


def test_redaction_is_installed_process_wide() -> None:
    assert log_safety._installed  # by upscale.config, before any provider exists


def test_helius_request_log_masks_the_url_api_key(debug_logs: pytest.LogCaptureFixture) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "result": {"value": {"amount": "5", "decimals": 6}}},
        )

    provider = HeliusProvider(FAKE_HELIUS, transport=httpx2.MockTransport(handle))
    assert asyncio.run(provider.fetch_supply("So11111111111111111111111111111111111111112")) == (
        5,
        6,
    )
    text = debug_logs.text
    assert "HTTP Request: POST https://mainnet.helius-rpc.com/?api-key=***" in text
    assert_clean(text, FAKE_HELIUS)
    for r in debug_logs.records:  # nothing left that could re-render the secret
        assert_clean(repr(r.args), FAKE_HELIUS)


def test_unregistered_credentials_in_query_parameters_are_masked(
    debug_logs: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("httpx2")
    for name in ("api-key", "api_key", "apikey", "key", "token", "access_token", "secret",
                 "client_secret", "password", "signature"):  # fmt: skip
        log.info("GET %s", f"https://rpc.example/v1?chain=sol&{name}={FAKE_UNREGISTERED}&x=1")
    assert_clean(debug_logs.text, FAKE_UNREGISTERED)
    assert debug_logs.text.count("chain=sol") == 10 and debug_logs.text.count("&x=1") == 10


def test_authorization_and_api_key_headers_are_masked(debug_logs: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("httpcore.http11")
    log.debug("send_request_headers.started headers=%r",
              [(b"Authorization", f"Bearer {FAKE_UNREGISTERED}".encode()), (b"x-api-key", b"k" * 20)])  # fmt: skip
    log.debug("headers %s", {"authorization": f"Basic {FAKE_UNREGISTERED}=="})
    log.debug("headers %s", {"X-API-Key": FAKE_UNREGISTERED, "accept": "application/json"})
    log.debug("headers %s", {"x-cg-demo-api-key": FAKE_UNREGISTERED})
    log.debug("Authorization: Bearer %s", FAKE_UNREGISTERED)
    log.debug("retrying with bearer %s", FAKE_UNREGISTERED)
    text = debug_logs.text
    assert_clean(text, FAKE_UNREGISTERED, "k" * 20)
    assert "application/json" in text and "Bearer ***" in text


def test_real_provider_headers_logged_at_debug_are_masked(
    debug_logs: pytest.LogCaptureFixture,
) -> None:
    async def go() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json={})),
            headers={"x-api-key": FAKE_NEYNAR, "Authorization": f"Bearer {FAKE_BEARER}"},
        ) as client:
            response = await client.get("https://api.neynar.com/v2/farcaster/cast/search?q=x")
            logging.getLogger("upscale.debug").debug(
                "request headers: %s", dict(response.request.headers)
            )

    asyncio.run(go())
    assert_clean(debug_logs.text, FAKE_NEYNAR, FAKE_BEARER)
    assert "api.neynar.com" in debug_logs.text  # the provider stays identifiable


def test_tracebacks_and_chained_errors_are_masked(debug_logs: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("upscale.outcomes")
    try:
        try:
            raise httpx2.ConnectError(
                f"failed https://mainnet.helius-rpc.com/?api-key={FAKE_HELIUS}"
            )
        except httpx2.ConnectError as exc:
            raise RuntimeError(
                f"could not reach Helius ({FAKE_UNREGISTERED}?token=abc12345678)"
            ) from exc
    except RuntimeError:
        log.exception("outcome collection cycle failed")
    text = debug_logs.text
    assert_clean(text, FAKE_HELIUS)
    assert "could not reach Helius" in text and "Traceback" in text
    # The formatted record carries only the redacted traceback.
    formatted = logging.Formatter().format(debug_logs.records[-1])
    assert_clean(formatted, FAKE_HELIUS, "abc12345678")


def test_credential_urls_are_masked_including_path_tokens() -> None:
    url = "https://solana-mainnet.quiknode.example/abcdEFGHijklMNOP1234/?token=ZZZ999"
    log_safety.register_url(url)
    out = log_safety.redact(f"POST {url} and later abcdEFGHijklMNOP1234 again")
    assert "abcdEFGHijklMNOP1234" not in out and "ZZZ999" not in out


def test_ordinary_text_is_not_redacted() -> None:
    text = (
        "Basic safety checks complete; token address solana:So111 is the exact token; "
        "bearer of news; key levels ~$1.2"
    )
    assert log_safety.redact(text) == text


def test_every_configured_credential_is_masked_without_printing_it() -> None:
    import upscale.config as config

    configured = [
        v
        for v in (config.HELIUS_API_KEY, config.NEYNAR_API_KEY, config.X_BEARER_TOKEN,
                  config.REDDIT_CLIENT_SECRET, config.COINGECKO_API_KEY)
        if v and len(v) >= log_safety.MIN_SECRET_LENGTH
    ]  # fmt: skip
    leaked = sum(1 for v in configured if v in log_safety.redact(f"GET https://x/?q={v} {v}"))
    assert leaked == 0


def test_no_test_can_reach_the_real_scout_database() -> None:
    from pathlib import Path

    import upscale.services as services
    from upscale.config import SCOUT_DB_PATH

    real = Path.home() / ".upscale" / "scout.sqlite3"
    stores = [services.scout_service.store.path, services.social_scout_service.store.path,
              services.outcome_store.path, SCOUT_DB_PATH]  # fmt: skip
    assert all(Path(p).resolve() != real.resolve() for p in stores)


def test_formatters_reading_args_keep_working(debug_logs: pytest.LogCaptureFixture) -> None:
    from uvicorn.logging import AccessFormatter

    log = logging.getLogger("uvicorn.access")
    log.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET",
             f"/outcomes/status?token={FAKE_UNREGISTERED}", "1.1", 200)  # fmt: skip
    log.info("%d requests, %.1f%% cached", 12, 50.0)
    record = debug_logs.records[0]
    assert isinstance(record.args, tuple) and record.args[-1] == 200
    line = AccessFormatter('%(client_addr)s "%(request_line)s" %(status_code)s').format(record)
    assert "GET /outcomes/status?token=*** HTTP/1.1" in line and "200" in line
    assert "12 requests, 50.0% cached" in debug_logs.text
    assert_clean(debug_logs.text + line, FAKE_UNREGISTERED)


def test_a_secret_split_across_template_and_argument_is_masked(
    debug_logs: pytest.LogCaptureFixture,
) -> None:
    logging.getLogger("upscale.debug").info(
        "calling https://rpc.example/?api-key=%s", FAKE_UNREGISTERED
    )
    assert_clean(debug_logs.text, FAKE_UNREGISTERED)
    assert "api-key=***" in debug_logs.text


def test_a_malformed_log_call_still_redacts_its_traceback(
    debug_logs: pytest.LogCaptureFixture,
) -> None:
    try:
        raise httpx2.ConnectError(f"https://mainnet.helius-rpc.com/?api-key={FAKE_HELIUS}")
    except httpx2.ConnectError:
        logging.getLogger("upscale.debug").exception("two placeholders %s %s", "only one")
    assert_clean(debug_logs.text, FAKE_HELIUS)
    assert "Traceback" in debug_logs.text
