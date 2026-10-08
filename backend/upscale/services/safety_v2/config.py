"""Safety V2 settings (Solana token-safety evidence; descriptive features only).

Safety V2 is CLI-only: nothing in the app imports it, and it never feeds Scout, the
Opportunity Model, Shadow, Calibration or execution. It has its own SQLite database and
its own provider budget, cooldown and retry policy; it never shares Radar's or
production's Solana limits.

Environment (all optional):

* ``UPSCALE_SAFETY_V2_DB`` (default ``~/.upscale/safety_v2.sqlite3``)
* ``UPSCALE_SAFETY_V2_DAILY_REQUEST_BUDGET`` (500), ``UPSCALE_SAFETY_V2_MAX_RPS`` (1),
  ``UPSCALE_SAFETY_V2_TIMEOUT_SECONDS`` (10), ``UPSCALE_SAFETY_V2_MAX_RETRIES`` (2),
  ``UPSCALE_SAFETY_V2_COOLDOWN_SECONDS`` (900), ``UPSCALE_SAFETY_V2_MAX_COOLDOWN_SECONDS``
  (7200)
* ``UPSCALE_SAFETY_V2_HOLDER_MAX_PAGES`` (2): DAS ``getTokenAccounts`` pages (1,000 token
  accounts each) one holder collection may read; past it the scan is PARTIAL.
* ``UPSCALE_SAFETY_V2_HOLDER_OWNER_LOOKUPS`` (30): largest owners whose own accounts are
  read (one ``getMultipleAccounts``) to tell program-owned owners apart.
* ``UPSCALE_SAFETY_V2_RADAR_DB`` (unset): a Radar database read **read-only** for positive
  wallet proof (a signer). Unset, missing or incompatible: no owner is proven a wallet.

* ``UPSCALE_SAFETY_V2_DEX_URL`` (unset): the DEX Screener API base URL (e.g.
  ``https://api.dexscreener.com``). Unset: market evidence is NOT_COLLECTED, so nothing
  ever reaches a market provider unless it is configured explicitly.

One holder collection makes at most ``4 + holder_max_pages`` logical requests
(``getTokenSupply``, ``getTokenLargestAccounts``, one ``getMultipleAccounts`` for those
accounts, the scan pages, one ``getMultipleAccounts`` for the owners). One market
collection makes at most 2: the DEX token-pairs request and, only when a previously
observed or pinned pool wasn't reported, one ``getAccountInfo(pool)``.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

SNAPSHOT_SCHEMA = "safety.snapshot.v2"
# 1: Phase 1 tables. 2: + safety_target_pools, safety_holder_observations,
# safety_holder_balances. 3: + safety_market_observations, safety_market_pools,
# safety_pool_account_observations. A database of another version is refused, never
# migrated.
DB_SCHEMA_VERSION = 3
# Bump whenever a rule's logic, threshold or wording changes: a snapshot built under
# another rules version is never claimed to be an exact reproduction.
# "2": Phase 2 holder rules (concentration, few holders, large unknown / program owners)
# and holder-scoped coverage. "3": Phase 3 market rules (liquidity, collapse, closure,
# not-reported history, eligibility, primary clarity, pool age) and market-scoped coverage.
RULES_VERSION = "3"

ENV_PREFIX = "UPSCALE_SAFETY_V2_"


def default_db_path() -> str:
    return os.getenv(ENV_PREFIX + "DB") or str(Path.home() / ".upscale" / "safety_v2.sqlite3")


class SafetySettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    db_path: str = Field(default_factory=default_db_path)

    # --- Provider budget (Safety V2's own; never shared) ---
    daily_request_budget: int = Field(default=500, ge=0)
    max_rps: float = Field(default=1.0, gt=0, le=50)
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=5)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    cooldown_seconds: float = Field(default=900.0, ge=0)  # after a 429; doubles per repeat
    max_cooldown_seconds: float = Field(default=7200.0, ge=0)

    # --- Holder collection (Phase 2) ---
    holder_max_pages: int = Field(default=2, ge=1, le=20)
    holder_owner_lookups: int = Field(default=30, ge=0, le=100)  # one getMultipleAccounts
    radar_db_path: str | None = None  # read-only wallet proof; None: not consulted

    # --- Market collection (Phase 3) ---
    dex_url: str | None = None  # DEX Screener base URL; None: no market provider

    @model_validator(mode="after")
    def _caps(self) -> "SafetySettings":
        if self.max_cooldown_seconds < self.cooldown_seconds:
            raise ValueError("max cooldown must be at least the base cooldown")
        return self


_INT = ("daily_request_budget", "max_retries", "holder_max_pages", "holder_owner_lookups")
_FLOAT = ("max_rps", "timeout_seconds", "backoff_base_seconds", "cooldown_seconds",
          "max_cooldown_seconds")  # fmt: skip


def load_settings(env: Mapping[str, str] | None = None) -> SafetySettings:
    """Settings from ``UPSCALE_SAFETY_V2_*`` (a malformed value is an error, not ignored)."""
    source = os.environ if env is None else env
    values: dict[str, object] = {}
    if raw := source.get(ENV_PREFIX + "DB"):
        values["db_path"] = raw
    if raw := source.get(ENV_PREFIX + "RADAR_DB"):
        values["radar_db_path"] = raw
    if raw := source.get(ENV_PREFIX + "DEX_URL"):
        values["dex_url"] = raw
    for name in _INT + _FLOAT:
        raw = source.get(ENV_PREFIX + name.upper())
        if raw is None or not raw.strip():
            continue
        try:
            values[name] = int(raw) if name in _INT else float(raw)
        except ValueError:
            raise ValueError(f"{ENV_PREFIX}{name.upper()}={raw!r} isn't a number") from None
    return SafetySettings.model_validate(values)
