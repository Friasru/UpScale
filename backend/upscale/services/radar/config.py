"""Radar V1 settings (Solana wallet / holder intelligence; descriptive only).

Radar is CLI-only in V1: nothing starts it from ``main.py`` or any background loop, and
``UPSCALE_RADAR`` (default OFF) is reserved for a later, separately reviewed background
step. Radar has its own SQLite database and its own provider budget, limiter, cooldown and
retry policy; it never shares or borrows production's Solana limits.

Every expensive enrichment (wallet age, first funder / FUNDING_CLUSTER, deep early-history
backfill) is OFF by default and reported as ``NOT_COLLECTED`` until enabled.

Environment (all optional):

* ``UPSCALE_RADAR`` (default 0), ``UPSCALE_RADAR_DB`` (default ``~/.upscale/radar.sqlite3``)
* ``UPSCALE_RADAR_DAILY_REQUEST_BUDGET`` (2000), ``UPSCALE_RADAR_MAX_RPS`` (1),
  ``UPSCALE_RADAR_CONCURRENCY`` (1), ``UPSCALE_RADAR_TIMEOUT_SECONDS`` (10),
  ``UPSCALE_RADAR_MAX_RETRIES`` (2), ``UPSCALE_RADAR_COOLDOWN_SECONDS`` (900),
  ``UPSCALE_RADAR_MAX_COOLDOWN_SECONDS`` (7200)
* ``UPSCALE_RADAR_HOLDER_MAX_PAGES`` (2), ``UPSCALE_RADAR_MAX_TX_PER_SNAPSHOT`` (40),
  ``UPSCALE_RADAR_EARLY_MAX_SIG_PAGES`` (1), ``UPSCALE_RADAR_EARLY_MAX_TX`` (20),
  ``UPSCALE_RADAR_VERIFY_DEPLOYER`` (1), ``UPSCALE_RADAR_DEPLOYER_MAX_SIG_PAGES`` (1)
* ``UPSCALE_RADAR_DEEP_BACKFILL`` (0), ``UPSCALE_RADAR_WALLET_AGE`` (0),
  ``UPSCALE_RADAR_FIRST_FUNDER`` (0), ``UPSCALE_RADAR_WALLET_PROFILE_MAX`` (20)
* ``UPSCALE_RADAR_KNOWN_INTERMEDIARIES``: comma-separated router / intermediary addresses
* ``UPSCALE_RADAR_TX_RETENTION_DAYS`` (7), ``UPSCALE_RADAR_HOLDER_BALANCE_RETENTION_DAYS``
  (7), ``UPSCALE_RADAR_SNAPSHOT_RETENTION_DAYS`` (30),
  ``UPSCALE_RADAR_REQUEST_RETENTION_DAYS`` (90)
"""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

SNAPSHOT_SCHEMA = "radar.snapshot.v1"
DB_SCHEMA_VERSION = 1

# Retention floors: an override below these is raised to the floor.
TX_RETENTION_FLOOR_DAYS = 3
SNAPSHOT_RETENTION_FLOOR_DAYS = 14

_OFF = ("0", "false", "off", "no")
_ON = ("1", "true", "on", "yes")


def default_db_path() -> str:
    return os.getenv("UPSCALE_RADAR_DB") or str(Path.home() / ".upscale" / "radar.sqlite3")


class RadarSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False  # reserved for a future background loop; V1 is CLI-only
    db_path: str = Field(default_factory=default_db_path)

    # --- Provider budget (Radar's own; never shared with production) ---
    daily_request_budget: int = Field(default=2000, ge=0)
    max_rps: float = Field(default=1.0, gt=0, le=50)
    concurrency: int = Field(default=1, ge=1, le=8)
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=5)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    cooldown_seconds: float = Field(default=900.0, ge=0)  # after a 429; doubles per repeat
    max_cooldown_seconds: float = Field(default=7200.0, ge=0)

    # --- Collection caps (bounded per target per run) ---
    holder_max_pages: int = Field(default=2, ge=1, le=20)  # DAS pages of 1,000 accounts
    owner_lookups: int = Field(default=30, ge=0, le=100)
    signature_page_size: int = Field(default=1000, ge=1, le=1000)
    activity_max_sig_pages: int = Field(default=1, ge=1, le=10)
    max_tx_per_snapshot: int = Field(default=40, ge=0, le=1000)
    early_max_sig_pages: int = Field(default=1, ge=0, le=10)
    early_max_tx: int = Field(default=20, ge=0, le=500)
    verify_deployer: bool = True
    deployer_max_sig_pages: int = Field(default=1, ge=1, le=10)

    # --- Optional expensive enrichment (OFF by default) ---
    deep_backfill: bool = False
    deep_early_max_sig_pages: int = Field(default=10, ge=1, le=100)
    deep_early_max_tx: int = Field(default=100, ge=1, le=2000)
    wallet_age: bool = False
    first_funder: bool = False
    wallet_profile_max: int = Field(default=20, ge=0, le=200)  # wallets profiled per target
    wallet_profile_max_sig_pages: int = Field(default=1, ge=1, le=10)

    # Extra addresses known to be routers / intermediaries (never counted as wallets).
    known_intermediaries: tuple[str, ...] = ()

    # --- Feature thresholds (descriptive; never trading thresholds) ---
    large_holder_min_pct: float = Field(default=0.5, gt=0, le=100)  # % of supply
    large_change_min_pp: float = Field(default=0.1, gt=0)  # percentage points of supply
    early_window_minutes: float = Field(default=60.0, gt=0)
    timing_window_seconds: float = Field(default=5.0, gt=0)
    timing_min_wallets: int = Field(default=3, ge=2)
    repeated_group_min_wallets: int = Field(default=3, ge=2)
    funding_cluster_min_wallets: int = Field(default=3, ge=2)
    history_min_unique_tokens: int = Field(default=10, ge=2)
    history_min_observations: int = Field(default=10, ge=2)
    history_horizon: str = "4h"
    catastrophic_return_pct: float = Field(default=-30.0, lt=0)

    # --- Radar-local retention ---
    tx_retention_days: int = Field(default=7, ge=TX_RETENTION_FLOOR_DAYS)
    holder_balance_retention_days: int = Field(default=7, ge=TX_RETENTION_FLOOR_DAYS)
    snapshot_retention_days: int = Field(default=30, ge=SNAPSHOT_RETENTION_FLOOR_DAYS)
    request_retention_days: int = Field(default=90, ge=1)

    @model_validator(mode="after")
    def _caps(self) -> "RadarSettings":
        if self.max_cooldown_seconds < self.cooldown_seconds:
            raise ValueError("max cooldown must be at least the base cooldown")
        return self

    @property
    def funding_clusters_enabled(self) -> bool:
        """FUNDING_CLUSTER needs first-funder discovery: it has no other data source."""
        return self.first_funder

    @property
    def effective_early_sig_pages(self) -> int:
        return self.deep_early_max_sig_pages if self.deep_backfill else self.early_max_sig_pages

    @property
    def effective_early_max_tx(self) -> int:
        return self.deep_early_max_tx if self.deep_backfill else self.early_max_tx


def _flag(raw: str | None, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise ValueError(f"invalid boolean {raw!r} (use 1/0, true/false, on/off)")


_INTS = {
    "daily_request_budget": "UPSCALE_RADAR_DAILY_REQUEST_BUDGET",
    "concurrency": "UPSCALE_RADAR_CONCURRENCY",
    "max_retries": "UPSCALE_RADAR_MAX_RETRIES",
    "holder_max_pages": "UPSCALE_RADAR_HOLDER_MAX_PAGES",
    "max_tx_per_snapshot": "UPSCALE_RADAR_MAX_TX_PER_SNAPSHOT",
    "early_max_sig_pages": "UPSCALE_RADAR_EARLY_MAX_SIG_PAGES",
    "early_max_tx": "UPSCALE_RADAR_EARLY_MAX_TX",
    "deployer_max_sig_pages": "UPSCALE_RADAR_DEPLOYER_MAX_SIG_PAGES",
    "wallet_profile_max": "UPSCALE_RADAR_WALLET_PROFILE_MAX",
    "tx_retention_days": "UPSCALE_RADAR_TX_RETENTION_DAYS",
    "holder_balance_retention_days": "UPSCALE_RADAR_HOLDER_BALANCE_RETENTION_DAYS",
    "snapshot_retention_days": "UPSCALE_RADAR_SNAPSHOT_RETENTION_DAYS",
    "request_retention_days": "UPSCALE_RADAR_REQUEST_RETENTION_DAYS",
}
_FLOATS = {
    "max_rps": "UPSCALE_RADAR_MAX_RPS",
    "timeout_seconds": "UPSCALE_RADAR_TIMEOUT_SECONDS",
    "cooldown_seconds": "UPSCALE_RADAR_COOLDOWN_SECONDS",
    "max_cooldown_seconds": "UPSCALE_RADAR_MAX_COOLDOWN_SECONDS",
}
_FLAGS = {
    "enabled": ("UPSCALE_RADAR", False),
    "verify_deployer": ("UPSCALE_RADAR_VERIFY_DEPLOYER", True),
    "deep_backfill": ("UPSCALE_RADAR_DEEP_BACKFILL", False),
    "wallet_age": ("UPSCALE_RADAR_WALLET_AGE", False),
    "first_funder": ("UPSCALE_RADAR_FIRST_FUNDER", False),
}
_FLOORED = {
    "tx_retention_days": TX_RETENTION_FLOOR_DAYS,
    "holder_balance_retention_days": TX_RETENTION_FLOOR_DAYS,
    "snapshot_retention_days": SNAPSHOT_RETENTION_FLOOR_DAYS,
}


def load_settings(env: Mapping[str, str] | None = None) -> RadarSettings:
    """Settings from the environment (or `env`). Retention below its floor is raised to it."""
    source = os.environ if env is None else env
    values: dict[str, object] = {}
    if source.get("UPSCALE_RADAR_DB"):
        values["db_path"] = source["UPSCALE_RADAR_DB"]
    for key, name in _INTS.items():
        raw = source.get(name)
        if raw is not None and raw.strip():
            try:
                number = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
            values[key] = max(number, _FLOORED.get(key, number))
    for key, name in _FLOATS.items():
        raw = source.get(name)
        if raw is not None and raw.strip():
            try:
                values[key] = float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    raw = source.get("UPSCALE_RADAR_KNOWN_INTERMEDIARIES")
    if raw:
        values["known_intermediaries"] = tuple(
            sorted({a.strip() for a in raw.split(",") if a.strip()})
        )
    for key, (name, default) in _FLAGS.items():
        values[key] = _flag(source.get(name), default)
    return RadarSettings.model_validate(values)
