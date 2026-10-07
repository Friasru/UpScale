"""Radar's own Solana provider access: a request budget, limiter, cooldown and retry policy
that belong to Radar alone.

Production's `SolanaSafetyService` is never used (it would archive Radar's lookups as
production "safety" evidence and spend production's limiter). Instead Radar subclasses
the stateless providers in `solana_chain` (unchanged) and replaces only their `_call`, so
the mint / largest-accounts / DAS parsing is reused while every HTTP request goes through
`RequestGuard`:

* **Daily budget** (persisted per UTC day in ``radar_requests``): every attempt, retries
  included, counts. When it's spent, calls fail with `RadarBudgetExhaustedError` and the
  work is reported ``NOT_COLLECTED``.
* **Limiter**: at most ``max_rps`` request starts per second and ``concurrency`` in flight.
* **429**: never retried. It starts a persisted Radar cooldown (``cooldown_seconds``,
  doubling per consecutive 429 up to ``max_cooldown_seconds``); until it ends every call
  fails fast with `RadarCoolingDownError`. A successful call resets the streak.
* **Timeouts, transport errors and HTTP 5xx**: retried up to ``max_retries`` with
  exponential backoff and jitter, then reported ``PROVIDER_UNAVAILABLE``.

The URL can contain an API key: it never appears in an error message.
"""

import asyncio
import itertools
import random
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import httpx2

from upscale.services.market_data import AssetNotFoundError
from upscale.services.radar.config import RadarSettings
from upscale.services.radar.models import (
    RadarBudgetExhaustedError,
    RadarCoolingDownError,
    RadarRateLimitedError,
    RadarTimeoutError,
    RadarUnavailableError,
    SignatureInfo,
)
from upscale.services.radar.parsing import parse_signatures
from upscale.services.solana_chain import HeliusProvider, SolanaRpcProvider

Outcome = Literal["ok", "rate_limited", "timeout", "failed"]
COOLDOWN_KEY = "provider.cooldown_until"
STREAK_KEY = "provider.consecutive_429"


class RequestLedger(Protocol):
    def requests_on(self, day: str) -> int: ...

    def record_request(self, day: str, method: str, outcome: Outcome) -> None: ...

    def get_meta(self, key: str) -> str | None: ...

    def set_meta(self, key: str, value: str) -> None: ...


class _Retryable(Exception):
    def __init__(self, error: RadarUnavailableError):
        super().__init__(str(error))
        self.error = error


class RequestGuard:
    def __init__(
        self,
        settings: RadarSettings,
        ledger: RequestLedger,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self.settings = settings
        self.ledger = ledger
        self._now = now
        self._monotonic = monotonic
        self._sleep = sleep
        self._rng = rng
        self._sem = asyncio.Semaphore(settings.concurrency)
        self._spacing = asyncio.Lock()
        self._next_start = 0.0
        self.used_this_run = 0

    def day(self) -> str:
        return self._now().astimezone(UTC).strftime("%Y-%m-%d")

    def used_today(self) -> int:
        return self.ledger.requests_on(self.day())

    def remaining_today(self) -> int:
        return max(0, self.settings.daily_request_budget - self.used_today())

    def cooldown_until(self) -> datetime | None:
        raw = self.ledger.get_meta(COOLDOWN_KEY)
        if not raw:
            return None
        until = datetime.fromtimestamp(float(raw), UTC)
        return until if until > self._now() else None

    def check(self) -> None:
        """Fail fast (no request) during a cooldown or once the budget is spent."""
        until = self.cooldown_until()
        if until is not None:
            raise RadarCoolingDownError(
                f"Radar is cooling down after a provider rate limit until {until.isoformat()}"
            )
        if self.remaining_today() <= 0:
            raise RadarBudgetExhaustedError(
                f"Radar's daily request budget ({self.settings.daily_request_budget}) is spent"
            )

    def start_cooldown(self) -> datetime:
        streak = int(self.ledger.get_meta(STREAK_KEY) or 0) + 1
        seconds = min(
            self.settings.cooldown_seconds * 2 ** (streak - 1), self.settings.max_cooldown_seconds
        )
        until = self._now() + timedelta(seconds=seconds)
        self.ledger.set_meta(STREAK_KEY, str(streak))
        self.ledger.set_meta(COOLDOWN_KEY, repr(until.timestamp()))
        return until

    def succeeded(self) -> None:
        if self.ledger.get_meta(STREAK_KEY) not in (None, "0"):
            self.ledger.set_meta(STREAK_KEY, "0")

    async def _space(self) -> None:
        async with self._spacing:
            wait = self._next_start - self._monotonic()
            if wait > 0:
                await self._sleep(wait)
            self._next_start = (
                max(self._next_start, self._monotonic()) + 1.0 / self.settings.max_rps
            )

    async def run(self, method: str, attempt: Callable[[], Awaitable[Any]]) -> Any:
        """Run one logical call: budget/cooldown checks, limiter, retries with backoff."""
        tries = 0
        while True:
            self.check()
            async with self._sem:
                await self._space()
                day = self.day()
                self.used_this_run += 1
                try:
                    result = await attempt()
                except RadarRateLimitedError:
                    self.ledger.record_request(day, method, "rate_limited")
                    until = self.start_cooldown()
                    raise RadarRateLimitedError(
                        f"provider rate limit (429) on {method}; Radar cools down until "
                        f"{until.isoformat()}"
                    ) from None
                except _Retryable as exc:
                    timeout = isinstance(exc.error, RadarTimeoutError)
                    self.ledger.record_request(day, method, "timeout" if timeout else "failed")
                    if tries >= self.settings.max_retries:
                        raise exc.error from None
                except BaseException:
                    self.ledger.record_request(day, method, "failed")
                    raise
                else:
                    self.ledger.record_request(day, method, "ok")
                    self.succeeded()
                    return result
            delay = self.settings.backoff_base_seconds * 2**tries * (1 + 0.25 * self._rng())
            tries += 1
            await self._sleep(delay)


class _RadarRpc:
    """Replaces `SolanaRpcProvider._call` with a guarded one and adds history methods."""

    name: str
    timeout: float
    guard: RequestGuard
    _url: str
    _transport: httpx2.AsyncBaseTransport | None
    _ids: "itertools.count[int]"

    async def _call(self, method: str, params: Any) -> Any:
        return await self.guard.run(method, lambda: self._attempt(method, params))

    async def _attempt(self, method: str, params: Any) -> Any:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        try:
            async with httpx2.AsyncClient(
                timeout=self.timeout, transport=self._transport
            ) as client:
                response = await client.post(self._url, json=payload)
        except httpx2.TimeoutException:
            raise _Retryable(RadarTimeoutError(f"{self.name} {method} timed out")) from None
        except httpx2.HTTPError:
            raise _Retryable(RadarUnavailableError(f"could not reach {self.name}")) from None
        if response.status_code == 429:
            raise RadarRateLimitedError(f"{self.name} rate limit reached")
        if response.status_code in (401, 403):
            raise RadarUnavailableError(f"{self.name} rejected the API key")
        if response.status_code >= 500:
            raise _Retryable(
                RadarUnavailableError(f"{self.name} returned HTTP {response.status_code}")
            )
        if response.status_code != 200:
            raise RadarUnavailableError(f"{self.name} returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise RadarUnavailableError(f"{self.name} returned invalid JSON") from None
        if not isinstance(body, dict):
            raise RadarUnavailableError(f"{self.name} returned an unexpected response")
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", "error"))
            if error.get("code") in (429, -32429) or "rate limit" in message.lower():
                raise RadarRateLimitedError(f"{self.name} rate limit reached")
            if "not a Token mint" in message or "Invalid param" in message:
                raise AssetNotFoundError(f"{method}: {message}")
            raise RadarUnavailableError(f"{self.name} {method} failed: {message}")
        if "result" not in body:
            raise RadarUnavailableError(f"{self.name} returned no result")
        return body["result"]

    async def get_signatures(
        self,
        address: str,
        *,
        limit: int,
        before: str | None = None,
        until: str | None = None,
    ) -> list[SignatureInfo]:
        """One page of the address's signatures, newest first."""
        options: dict[str, Any] = {"limit": limit, "commitment": "confirmed"}
        if before:
            options["before"] = before
        if until:
            options["until"] = until
        result = await self._call("getSignaturesForAddress", [address, options])
        if not isinstance(result, list):
            raise RadarUnavailableError(f"{self.name} returned malformed signatures")
        return parse_signatures(result)

    async def get_transaction(self, signature: str) -> dict[str, Any] | None:
        """The jsonParsed transaction, or None when the provider has no record of it."""
        result = await self._call(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0,
                    "commitment": "confirmed",
                },
            ],
        )
        if result is None:
            return None
        if not isinstance(result, dict):
            raise RadarUnavailableError(f"{self.name} returned a malformed transaction")
        return result


class RadarRpcProvider(_RadarRpc, SolanaRpcProvider):
    def __init__(
        self,
        guard: RequestGuard,
        url: str,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        SolanaRpcProvider.__init__(self, url, guard.settings.timeout_seconds, transport)
        self.guard = guard


class RadarHeliusProvider(_RadarRpc, HeliusProvider):
    def __init__(
        self,
        guard: RequestGuard,
        api_key: str,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        HeliusProvider.__init__(
            self,
            api_key,
            max_pages=guard.settings.holder_max_pages,
            timeout=guard.settings.timeout_seconds,
            transport=transport,
        )
        self.guard = guard


RadarProvider = RadarRpcProvider | RadarHeliusProvider


def build_provider(
    guard: RequestGuard,
    helius_api_key: str | None,
    rpc_url: str | None,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> RadarProvider | None:
    """Helius when a key is configured (holder scans), else a plain RPC, else None."""
    if helius_api_key:
        return RadarHeliusProvider(guard, helius_api_key, transport)
    if rpc_url:
        return RadarRpcProvider(guard, rpc_url, transport)
    return None
