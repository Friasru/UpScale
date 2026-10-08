"""Safety V2's own guarded Solana provider access and the mint-account classifier.

Production's `SolanaSafetyService` is never used (it would archive these lookups as
production evidence and spend production's limiter), and Radar's provider isn't imported
(Radar is a separate, frozen component). Every request goes through `RequestGuard`:

* **Daily budget** (persisted per UTC day in ``safety_requests``): every attempt, retries
  included, counts. Once spent, calls fail with `SafetyBudgetExhaustedError` before any
  request, and the evidence is ``NOT_COLLECTED``.
* **Limiter**: at most ``max_rps`` request starts per second, one in flight.
* **429**: never retried; it starts a persisted cooldown (``cooldown_seconds``, doubling
  per consecutive 429 up to ``max_cooldown_seconds``) during which calls fail fast with
  `SafetyCoolingDownError` (``NOT_COLLECTED``).
* **Timeouts, transport errors, HTTP 5xx**: retried up to ``max_retries`` with backoff,
  then ``PROVIDER_UNAVAILABLE``.

The URL may contain an API key: it never appears in an error, a repr or the database.

`classify_mint_account` turns one raw ``getAccountInfo`` result into a `MintObservation`
that distinguishes MINT / NOT_A_MINT / ACCOUNT_MISSING / MALFORMED. Anything it can't
vouch for is MALFORMED, never a default: in particular a malformed authority value is
never read as a revoked (null) authority.
"""

import asyncio
import hashlib
import itertools
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import httpx2

from upscale.services.chains import is_solana_address
from upscale.services.market_data import MarketDataError
from upscale.services.safety_v2.config import SafetySettings
from upscale.services.safety_v2.models import (
    MintOutcome,
    SafetyBudgetExhaustedError,
    SafetyCoolingDownError,
    SafetyProviderError,
    SafetyRateLimitedError,
    SafetyTimeoutError,
)
from upscale.services.solana_chain import TOKEN_2022_PROGRAM, TOKEN_PROGRAM, parse_mint

RequestOutcome = Literal["ok", "rate_limited", "timeout", "failed"]
COOLDOWN_KEY = "provider.cooldown_until"
STREAK_KEY = "provider.consecutive_429"
TOKEN_PROGRAMS: dict[str, str] = {TOKEN_PROGRAM: "spl_token", TOKEN_2022_PROGRAM: "token_2022"}


class RequestLedger(Protocol):
    def requests_on(self, day: str) -> int: ...

    def record_request(self, day: str, method: str, outcome: RequestOutcome) -> None: ...

    def get_meta(self, key: str) -> str | None: ...

    def set_meta(self, key: str, value: str) -> None: ...


class _Retryable(Exception):
    def __init__(self, error: SafetyProviderError):
        super().__init__(str(error))
        self.error = error


class RequestGuard:
    def __init__(
        self,
        settings: SafetySettings,
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
        self._lock = asyncio.Lock()
        self._next_start = 0.0
        self.used_this_run = 0

    def day(self) -> str:
        return self._now().astimezone(UTC).strftime("%Y-%m-%d")

    def remaining_today(self) -> int:
        return max(0, self.settings.daily_request_budget - self.ledger.requests_on(self.day()))

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
            raise SafetyCoolingDownError(
                f"Safety V2 is cooling down after a provider rate limit until {until.isoformat()}"
            )
        if self.remaining_today() <= 0:
            raise SafetyBudgetExhaustedError(
                f"Safety V2's daily request budget ({self.settings.daily_request_budget}) is spent"
            )

    def _start_cooldown(self) -> datetime:
        streak = int(self.ledger.get_meta(STREAK_KEY) or 0) + 1
        seconds = min(
            self.settings.cooldown_seconds * 2 ** (streak - 1), self.settings.max_cooldown_seconds
        )
        until = self._now() + timedelta(seconds=seconds)
        self.ledger.set_meta(STREAK_KEY, str(streak))
        self.ledger.set_meta(COOLDOWN_KEY, repr(until.timestamp()))
        return until

    async def run(self, method: str, attempt: Callable[[], Awaitable[Any]]) -> Any:
        """One logical call: budget / cooldown checks, limiter, retries with backoff."""
        tries = 0
        while True:
            self.check()
            async with self._lock:
                wait = self._next_start - self._monotonic()
                if wait > 0:
                    await self._sleep(wait)
                self._next_start = self._monotonic() + 1.0 / self.settings.max_rps
                day = self.day()
                self.used_this_run += 1
                try:
                    result = await attempt()
                except SafetyRateLimitedError:
                    self.ledger.record_request(day, method, "rate_limited")
                    until = self._start_cooldown()
                    raise SafetyRateLimitedError(
                        f"provider rate limit (429) on {method}; Safety V2 cools down until "
                        f"{until.isoformat()}"
                    ) from None
                except _Retryable as exc:
                    timeout = isinstance(exc.error, SafetyTimeoutError)
                    self.ledger.record_request(day, method, "timeout" if timeout else "failed")
                    if tries >= self.settings.max_retries:
                        raise exc.error from None
                except BaseException:
                    self.ledger.record_request(day, method, "failed")
                    raise
                else:
                    self.ledger.record_request(day, method, "ok")
                    if self.ledger.get_meta(STREAK_KEY) not in (None, "0"):
                        self.ledger.set_meta(STREAK_KEY, "0")
                    return result
            delay = self.settings.backoff_base_seconds * 2**tries * (1 + 0.25 * self._rng())
            tries += 1
            await self._sleep(delay)


class SafetyRpcProvider:
    """Guarded Solana JSON-RPC: only what Phase 1 needs (``getAccountInfo``)."""

    def __init__(
        self,
        guard: RequestGuard,
        url: str,
        name: str = "Solana RPC",
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self.guard = guard
        self.name = name
        self._url = url
        self._transport = transport
        self._ids = itertools.count(1)

    def __repr__(self) -> str:  # never show the URL (it may hold a key)
        return f"{type(self).__name__}(name={self.name!r})"

    async def get_account_info(self, address: str) -> Any:
        """The raw ``result`` of ``getAccountInfo`` (jsonParsed, confirmed)."""
        params = [address, {"encoding": "jsonParsed", "commitment": "confirmed"}]
        return await self.guard.run(
            "getAccountInfo", lambda: self._attempt("getAccountInfo", params)
        )

    async def _attempt(self, method: str, params: Any) -> Any:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        try:
            async with httpx2.AsyncClient(
                timeout=self.guard.settings.timeout_seconds, transport=self._transport
            ) as client:
                response = await client.post(self._url, json=payload)
        except httpx2.TimeoutException:
            raise _Retryable(SafetyTimeoutError(f"{self.name} {method} timed out")) from None
        except httpx2.HTTPError:
            raise _Retryable(SafetyProviderError(f"could not reach {self.name}")) from None
        if response.status_code == 429:
            raise SafetyRateLimitedError(f"{self.name} rate limit reached")
        if response.status_code >= 500:
            raise _Retryable(
                SafetyProviderError(f"{self.name} returned HTTP {response.status_code}")
            )
        if response.status_code != 200:
            raise SafetyProviderError(f"{self.name} returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise SafetyProviderError(f"{self.name} returned invalid JSON") from None
        if not isinstance(body, dict):
            raise SafetyProviderError(f"{self.name} returned an unexpected response")
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", "error"))
            if error.get("code") in (429, -32429) or "rate limit" in message.lower():
                raise SafetyRateLimitedError(f"{self.name} rate limit reached")
            raise SafetyProviderError(f"{self.name} {method} failed: {message}")
        if "result" not in body:
            raise SafetyProviderError(f"{self.name} returned no result")
        return body["result"]


def build_provider(
    guard: RequestGuard,
    helius_api_key: str | None,
    rpc_url: str | None,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> SafetyRpcProvider | None:
    """Helius when a key is configured, else a plain RPC, else None (nothing collected)."""
    if helius_api_key:
        url = f"https://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        return SafetyRpcProvider(guard, url, "Helius", transport)
    if rpc_url:
        return SafetyRpcProvider(guard, rpc_url, "Solana RPC", transport)
    return None


# --- mint account classification ----------------------------------------------------------


@dataclass(frozen=True)
class MintObservation:
    """What one getAccountInfo(mint) established. Authority fields mean something only
    when `outcome` is MINT (the database enforces that they are NULL otherwise)."""

    outcome: MintOutcome
    reason: str | None
    raw_hash: str | None = None
    context_slot: int | None = None
    program_owner: str | None = None
    token_program: str | None = None
    decimals: int | None = None
    supply_raw: str | None = None
    mint_authority: str | None = None
    freeze_authority: str | None = None
    extensions: tuple[tuple[str, Any], ...] | None = None


def raw_hash(result: Any) -> str:
    text = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def not_observed(exc: SafetyProviderError) -> MintObservation:
    outcome: MintOutcome = "NOT_COLLECTED" if exc.status == "NOT_COLLECTED" else "PROVIDER_FAILED"
    return MintObservation(outcome=outcome, reason=str(exc) or type(exc).__name__)


def _authority(info: Mapping[str, Any], key: str) -> tuple[bool, str | None]:
    """(well-formed, address). Only an explicit null / absent key is "no authority"."""
    value = info.get(key)
    if value is None:
        return True, None
    return (True, value) if isinstance(value, str) and is_solana_address(value) else (False, None)


def classify_mint_account(mint: str, result: Any, provider: str) -> MintObservation:
    """Classify a raw ``getAccountInfo`` result (jsonParsed) for `mint`. A malformed RPC
    envelope is a provider failure (nothing is learned about the account)."""
    if not isinstance(result, dict) or "value" not in result:
        return MintObservation(
            "PROVIDER_FAILED", f"{provider} returned a malformed getAccountInfo envelope"
        )
    digest = raw_hash(result)
    context = result.get("context")
    slot = context.get("slot") if isinstance(context, dict) else None
    slot = slot if isinstance(slot, int) and not isinstance(slot, bool) else None
    value = result["value"]
    base: dict[str, Any] = {"raw_hash": digest, "context_slot": slot}
    if value is None:
        return MintObservation("ACCOUNT_MISSING", f"no account exists at {mint}", **base)
    if not isinstance(value, dict) or not isinstance(value.get("owner"), str):
        return MintObservation("MALFORMED", "the account has no readable owner program", **base)
    owner: str = value["owner"]
    program = TOKEN_PROGRAMS.get(owner)
    if program is None:
        return MintObservation(
            "NOT_A_MINT", f"the account is owned by {owner}, not a token program",
            program_owner=owner, **base,
        )  # fmt: skip
    data = value.get("data")
    parsed = data.get("parsed") if isinstance(data, dict) else None
    info = parsed.get("info") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or not isinstance(info, dict):
        return MintObservation(
            "MALFORMED", "a token-program account without parsed account data",
            program_owner=owner, **base,
        )  # fmt: skip
    kind = parsed.get("type")
    if kind != "mint":
        return MintObservation(
            "NOT_A_MINT", f"a token-program {kind!r} account, not a mint",
            program_owner=owner, **base,
        )  # fmt: skip

    def malformed(why: str) -> MintObservation:
        return MintObservation("MALFORMED", why, program_owner=owner, **base)

    if info.get("isInitialized") is not True:
        return malformed("the mint isn't marked initialized")
    mint_ok, mint_auth = _authority(info, "mintAuthority")
    freeze_ok, freeze_auth = _authority(info, "freezeAuthority")
    if not (mint_ok and freeze_ok):
        return malformed("an authority isn't null or a valid Solana address")
    raw_supply = info.get("supply")
    if not isinstance(raw_supply, str) or not raw_supply.isdigit():
        return malformed("the supply isn't a decimal string")
    raw_extensions = info.get("extensions")
    if raw_extensions is not None:
        if program != "token_2022":
            return malformed("an SPL Token mint reports Token-2022 extensions")
        if not isinstance(raw_extensions, list) or not all(
            isinstance(e, dict) and isinstance(e.get("extension"), str) for e in raw_extensions
        ):
            return malformed("the Token-2022 extension list is malformed")
    try:
        parsed_mint = parse_mint(mint, value, provider)
    except MarketDataError as exc:
        return malformed(f"the mint data can't be parsed ({exc})")
    if not 0 <= parsed_mint.decimals <= 255:
        return malformed(f"decimals {parsed_mint.decimals} are out of range")
    extensions = tuple((e["extension"], e.get("state")) for e in raw_extensions or ())
    return MintObservation(
        outcome="MINT",
        reason=None,
        program_owner=owner,
        token_program=program,
        decimals=parsed_mint.decimals,
        supply_raw=raw_supply,
        mint_authority=mint_auth,
        freeze_authority=freeze_auth,
        extensions=extensions,
        **base,
    )
