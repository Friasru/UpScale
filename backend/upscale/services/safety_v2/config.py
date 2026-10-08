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
"""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

SNAPSHOT_SCHEMA = "safety.snapshot.v2"
DB_SCHEMA_VERSION = 1
# Bump whenever a rule's logic, threshold or wording changes: a snapshot built under
# another rules version is never claimed to be an exact reproduction.
RULES_VERSION = "1"

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

    @model_validator(mode="after")
    def _caps(self) -> "SafetySettings":
        if self.max_cooldown_seconds < self.cooldown_seconds:
            raise ValueError("max cooldown must be at least the base cooldown")
        return self


_INT = ("daily_request_budget", "max_retries")
_FLOAT = ("max_rps", "timeout_seconds", "backoff_base_seconds", "cooldown_seconds",
          "max_cooldown_seconds")  # fmt: skip


def load_settings(env: Mapping[str, str] | None = None) -> SafetySettings:
    """Settings from ``UPSCALE_SAFETY_V2_*`` (a malformed value is an error, not ignored)."""
    source = os.environ if env is None else env
    values: dict[str, object] = {}
    if raw := source.get(ENV_PREFIX + "DB"):
        values["db_path"] = raw
    for name in _INT + _FLOAT:
        raw = source.get(ENV_PREFIX + name.upper())
        if raw is None or not raw.strip():
            continue
        try:
            values[name] = int(raw) if name in _INT else float(raw)
        except ValueError:
            raise ValueError(f"{ENV_PREFIX}{name.upper()}={raw!r} isn't a number") from None
    return SafetySettings.model_validate(values)
