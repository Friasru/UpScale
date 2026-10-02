"""Shadow / Paper Strategy settings: immutable, versioned strategy configurations.

A strategy is identified by ``(strategy_id, version)`` and fingerprinted by the hash of its
rules (`StrategyConfig.config_hash`). Once registered it never changes: different rules
under the same id need a new version (the store refuses anything else), and a run freezes
the exact versions and hashes it uses.

None of these thresholds is claimed to be optimal. The baselines (`strategies.py`) are
deliberately simple, conservative comparison rules.
"""

import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import upscale.config  # noqa: F401  (loads the repo-root .env before reading variables)

# The first production observation recorded after the exact-token pricing / orientation
# fixes. Earlier observations are contaminated: production shadow runs never start before it.
CLEAN_DATA_CUTOFF = datetime(2026, 9, 30, 5, 50, tzinfo=UTC)

ExecutionModel = Literal["IDEALIZED_NO_FEES"]
EXECUTION_NOTE = (
    "IDEALIZED_NO_FEES: fills at the exact pool's observed Scout price, no fees, no "
    "slippage, no latency, no price impact. Results are simulated and not real profit."
)
NOT_REAL_PROFIT = (
    "Shadow / paper simulation only: no order was placed and no money was at risk. "
    "Simulated P/L is not real profit, not an expected return and not a recommendation."
)
Action = Literal["ENTER", "HOLD", "EXIT", "NO_ACTION"]
ExitReason = Literal[
    "TAKE_PROFIT",
    "STOP_LOSS",
    "TRAILING_STOP",
    "MAX_HOLD_TIME",
    "SIGNAL_EXIT",
    "MARKET_UNAVAILABLE",
]
# How a run's books treat missing price evidence (a run-level setting, frozen when the run
# is created; never part of a strategy's config hash):
#
# * LEGACY_V1 (every run created before this setting existed): no exact-pool price for
#   `market_unavailable_after_minutes`, or a triggered exit without a price within
#   `max_exit_delay_minutes`, closes the position as MARKET_UNAVAILABLE without a price.
#   Only Scout-derived prices are read (held-position watch records are ignored), so such
#   a run behaves exactly as it always did.
# * EVIDENCE_AWARE_V2 (new runs): missing evidence never closes a position. A position
#   without a recent price stays open (PRICE_STALE / EVIDENCE_GAP / PROVIDER_UNAVAILABLE,
#   reporting only); a triggered exit waits for the next observed exact-pool price. Only an
#   authoritative MARKET_NOT_FOUND observation of the exact pool, confirmed (see
#   `book.NOT_FOUND_CONFIRMATIONS`) over at least `max_exit_delay_minutes` with no valid
#   price in between, closes it as MARKET_UNAVAILABLE. Held-position watch prices count.
AvailabilityPolicy = Literal["LEGACY_V1", "EVIDENCE_AWARE_V2"]
AVAILABILITY_POLICIES: tuple[AvailabilityPolicy, ...] = ("LEGACY_V1", "EVIDENCE_AWARE_V2")
LEGACY_POLICY: AvailabilityPolicy = "LEGACY_V1"
DEFAULT_POLICY: AvailabilityPolicy = "EVIDENCE_AWARE_V2"  # for runs created from now on


def run_policy(run: dict[str, object]) -> AvailabilityPolicy:
    """A stored run's policy: runs created before the setting existed are LEGACY_V1."""
    args = run.get("args")
    raw = args.get("availability_policy") if isinstance(args, dict) else None
    for p in AVAILABILITY_POLICIES:
        if raw == p:
            return p
    return LEGACY_POLICY


MissingLabel = Literal[
    "SAFETY_NOT_AVAILABLE",
    "SOCIAL_NOT_AVAILABLE",
    "MARKET_CAP_NOT_AVAILABLE",
    "LIQUIDITY_NOT_AVAILABLE",
    "TECHNICAL_NOT_AVAILABLE",
]
MISSING_LABELS: tuple[MissingLabel, ...] = (
    "SAFETY_NOT_AVAILABLE", "SOCIAL_NOT_AVAILABLE", "MARKET_CAP_NOT_AVAILABLE",
    "LIQUIDITY_NOT_AVAILABLE", "TECHNICAL_NOT_AVAILABLE",
)  # fmt: skip
STAGES = (
    "NEW", "EARLY", "ACCELERATING", "CROWDED", "FADING", "STEADY", "INSUFFICIENT_DATA",
)  # fmt: skip
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
SPAM_ORDER = {"low": 0, "medium": 1, "high": 2}
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


def _scout_dir() -> Path:
    scout = os.getenv("UPSCALE_SCOUT_DB") or str(Path.home() / ".upscale" / "scout.sqlite3")
    return Path(scout).expanduser().parent


def default_shadow_db() -> str:
    """``UPSCALE_SHADOW_DB`` (Railway: ``/data/shadow.sqlite3``), else ``shadow.sqlite3``
    next to the Scout database (``~/.upscale/shadow.sqlite3`` by default)."""
    return os.getenv("UPSCALE_SHADOW_DB") or str(_scout_dir() / "shadow.sqlite3")


def default_evidence_db() -> str:
    """The Evidence Archive (read-only here)."""
    return os.getenv("UPSCALE_EVIDENCE_DB") or str(_scout_dir() / "evidence.sqlite3")


def _frozen() -> ConfigDict:
    return ConfigDict(frozen=True, extra="forbid")


# --- entry -----------------------------------------------------------------------------------


class TechnicalRules(BaseModel):
    """Conditions on Scout's snapshot-based Technical context (``momentum.technical``: the
    exact token's own stored pool snapshots up to the decision time; no candles)."""

    model_config = _frozen()

    allowed_trends: tuple[Literal["up", "down", "flat"], ...] = ("up",)
    min_snapshots: int = Field(default=3, ge=1)
    require_breakout: bool = False
    require_volume_confirmed: bool = False  # None (unknown) does not satisfy it
    require_higher_lows: bool = False


class SocialRules(BaseModel):
    """Only applied when social evidence exists; whether missing social evidence blocks an
    entry is `EntryRules.allowed_missing` (SOCIAL_NOT_AVAILABLE)."""

    model_config = _frozen()

    allowed_statuses: tuple[str, ...] | None = None  # None: any measured status
    max_spam_risk: Literal["low", "medium", "high"] | None = None


class AnalyzeRules(BaseModel):
    """Require a recent archived Analyze decision (Opportunity / Technical / Risk) for the
    exact asset, observed at or before the Scout decision time and at most `max_age`."""

    model_config = _frozen()

    max_age_minutes: float = Field(default=60.0, gt=0)
    allowed_actions: tuple[Literal["buy", "sell", "wait"], ...] = ("buy",)
    min_confidence: Literal["low", "medium", "high"] = "medium"
    max_risk_level: Literal["low", "medium", "high"] | None = "medium"
    allowed_technical_trends: tuple[Literal["uptrend", "downtrend", "mixed"], ...] | None = None


class EntryRules(BaseModel):
    model_config = _frozen()

    min_scout_score: float = Field(default=60.0, ge=0, le=100)
    allowed_stages: tuple[str, ...] = ("EARLY", "ACCELERATING")
    require_eligible: bool = True
    # STALE_CARRIED candidates are ranked on an old observation: never an entry price.
    require_current_data: bool = True
    min_liquidity_usd: float = Field(default=50_000.0, ge=0)
    max_risk_penalty: float | None = Field(default=10.0, ge=0)
    blocking_flag_severities: tuple[Literal["info", "caution", "high", "critical"], ...] = (
        "critical",
    )
    required_safety: Literal["ANY", "PARTIAL_OR_COMPLETE", "COMPLETE"] = "ANY"
    block_active_authorities: bool = False  # mint / freeze authority known to be active
    max_holder_top10_pct: float | None = Field(default=None, gt=0, le=100)
    # Missing evidence that does NOT block an entry (anything else missing does).
    allowed_missing: tuple[MissingLabel, ...] = MISSING_LABELS
    technical: TechnicalRules | None = None
    social: SocialRules | None = None
    analyze: AnalyzeRules | None = None
    # Deterministic random baseline: of the candidates passing every rule above, enter
    # only a pseudo-random fraction (hash of seed, strategy, asset and decision time).
    random_fraction: float | None = Field(default=None, gt=0, le=1)
    random_seed: int = 0

    @field_validator("allowed_stages")
    @classmethod
    def _stages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [s for s in value if s not in STAGES]
        if unknown or not value:
            raise ValueError(f"allowed_stages must be non-empty Growth Scout stages: {unknown}")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "EntryRules":
        if self.required_safety != "ANY" and "SAFETY_NOT_AVAILABLE" in self.allowed_missing:
            raise ValueError(
                "required_safety needs safety evidence: remove SAFETY_NOT_AVAILABLE from "
                "allowed_missing"
            )
        if self.technical is not None and "TECHNICAL_NOT_AVAILABLE" in self.allowed_missing:
            raise ValueError(
                "technical rules need Technical evidence: remove TECHNICAL_NOT_AVAILABLE "
                "from allowed_missing"
            )
        return self


# --- exit ------------------------------------------------------------------------------------


class TakeProfitLevel(BaseModel):
    model_config = _frozen()

    gain_pct: float = Field(gt=0)
    # Share of the ORIGINAL quantity sold at this level; the last level sells the rest.
    fraction: float = Field(default=1.0, gt=0, le=1)


class ExitRules(BaseModel):
    """Exits fill at the exact pool's observed price that triggered them (never at the
    level itself: discrete observations can gap through a level)."""

    model_config = _frozen()

    take_profit: tuple[TakeProfitLevel, ...] = (TakeProfitLevel(gain_pct=30.0),)
    stop_loss_pct: float | None = Field(default=15.0, gt=0, lt=100)
    trailing_stop_pct: float | None = Field(default=None, gt=0, lt=100)
    trailing_activation_pct: float = Field(default=0.0, ge=0)  # gain that arms the trail
    max_hold_minutes: float = Field(default=1440.0, gt=0)
    signal_exit_stages: tuple[str, ...] = ("FADING",)
    signal_exit_below_score: float | None = Field(default=None, ge=0, le=100)
    signal_exit_on_market_collapse: bool = True
    # No observed price of the exact pool for this long: MARKET_UNAVAILABLE (no price).
    market_unavailable_after_minutes: float = Field(default=360.0, gt=0)
    # A time- or signal-triggered exit fills at the next observed price of the exact pool
    # within this delay; otherwise MARKET_UNAVAILABLE (no price is fabricated).
    max_exit_delay_minutes: float = Field(default=120.0, gt=0)

    @field_validator("signal_exit_stages")
    @classmethod
    def _stages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [s for s in value if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown signal exit stages: {unknown}")
        return value

    @model_validator(mode="after")
    def _levels(self) -> "ExitRules":
        gains = [lv.gain_pct for lv in self.take_profit]
        if gains != sorted(gains) or len(set(gains)) != len(gains):
            raise ValueError("take-profit levels must have strictly increasing gains")
        if sum(lv.fraction for lv in self.take_profit[:-1]) >= 1.0 + 1e-9:
            raise ValueError("take-profit fractions before the last level must sum below 1")
        return self


# --- risk ------------------------------------------------------------------------------------


class RiskRules(BaseModel):
    model_config = _frozen()

    initial_capital_usd: float = Field(default=10_000.0, gt=0)
    position_size_usd: float = Field(default=500.0, gt=0)
    max_allocation_pct: float = Field(default=5.0, gt=0, le=100)  # of equity, per entry
    min_position_usd: float = Field(default=50.0, gt=0)
    max_open_positions: int = Field(default=10, ge=1)
    max_exposure_per_asset_pct: float = Field(default=10.0, gt=0, le=100)
    max_gross_exposure_pct: float = Field(default=100.0, gt=0, le=100)
    # Duplicate / correlation control: repeated Scout scans of one asset are not new trades.
    allow_scaling: bool = False
    max_positions_per_asset: int = Field(default=1, ge=1)
    entry_cooldown_minutes: float = Field(default=240.0, ge=0)  # after an exit of the asset
    min_entry_spacing_minutes: float = Field(default=60.0, ge=0)  # between entries
    max_entries_per_asset_per_day: int = Field(default=2, ge=1)  # UTC day

    @model_validator(mode="after")
    def _scaling(self) -> "RiskRules":
        if not self.allow_scaling and self.max_positions_per_asset != 1:
            raise ValueError("max_positions_per_asset > 1 requires allow_scaling")
        if self.min_position_usd > self.position_size_usd:
            raise ValueError("min_position_usd exceeds position_size_usd")
        return self


# --- strategy ----------------------------------------------------------------------------------


class StrategyConfig(BaseModel):
    model_config = _frozen()

    strategy_id: str
    version: int = Field(ge=1)
    name: str
    description: str = ""
    created_at: datetime
    entry: EntryRules = EntryRules()
    exit: ExitRules = ExitRules()
    risk: RiskRules = RiskRules()
    execution_model: ExecutionModel = "IDEALIZED_NO_FEES"

    @field_validator("strategy_id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG.match(value):
            raise ValueError("strategy_id must be a lowercase slug (a-z, 0-9, _ or -)")
        return value

    @field_validator("created_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @property
    def key(self) -> str:
        return f"{self.strategy_id}@v{self.version}"

    def rules_json(self) -> str:
        """The rules only (not name, description or creation time): what the hash covers."""
        body = self.model_dump(mode="json", exclude={"name", "description", "created_at"})
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.rules_json().encode()).hexdigest()[:24]


# --- background service settings ------------------------------------------------------------------

DEFAULT_INTERVAL_MINUTES = 15.0
MIN_INTERVAL_MINUTES = 5.0
DEFAULT_RUN_ID = "production"
# Evidence observed less than this long ago is not processed yet: the archive writer
# queue may still hold records of the same moment (keeps processing strictly ordered).
SETTLE_SECONDS = 600.0
_OFF = ("", "0", "false", "off", "no")


class ShadowSettings(BaseModel):
    model_config = _frozen()

    enabled: bool = False
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES
    run_id: str = DEFAULT_RUN_ID
    since: datetime = CLEAN_DATA_CUTOFF
    startup_delay_minutes: float = 5.0
    retry_minutes: float = 5.0


# --- rejection diagnostics storage --------------------------------------------------------------

DiagnosticsDetail = Literal["full", "sampled", "aggregate"]
DETAIL_MODES: tuple[DiagnosticsDetail, ...] = ("full", "sampled", "aggregate")
DEFAULT_DETAIL: DiagnosticsDetail = "sampled"
# ~1.2 KB per stored rejection row: 20 per strategy, primary reason and UTC day is at most a
# few hundred rows (well under 1 MB) a day for the four baselines, against ~25 MB a day
# for one full row per rejected evaluation.
DEFAULT_SAMPLE_PER_REASON = 20


class DiagnosticsSettings(BaseModel):
    """How much rejection detail a processing step stores. The aggregate counters (and so
    every funnel / diagnostics count) are exact in every mode; only the per-candidate
    ``shadow_rejections`` rows differ:

    * ``full``: one row per rejected evaluation (one-off research runs);
    * ``sampled`` (default): at most `sample_per_reason` rows per strategy, primary reason
      and UTC day, spread over the day (a per-hour quota of ceil(N / 24));
    * ``aggregate``: no rows, counters only.

    `retention_days` (None: disabled, the default) deletes sampled rows whose decision
    time is older than that many days before the processed time. It never touches rows
    stored before aggregate counters existed for a run, rows of a ``full`` step's run, or
    any other table."""

    model_config = _frozen()

    detail: DiagnosticsDetail = DEFAULT_DETAIL
    sample_per_reason: int = Field(default=DEFAULT_SAMPLE_PER_REASON, ge=1)
    retention_days: float | None = Field(default=None, gt=0)

    @property
    def hourly_quota(self) -> int:
        return max(1, math.ceil(self.sample_per_reason / 24))


def load_diagnostics_settings(
    detail: str | None = None,
    sample_per_reason: str | None = None,
    retention_days: str | None = None,
) -> DiagnosticsSettings:
    """From ``UPSCALE_SHADOW_DIAGNOSTICS_DETAIL`` (full / sampled / aggregate; default
    sampled), ``UPSCALE_SHADOW_REJECTION_SAMPLE_PER_REASON`` (default 20) and
    ``UPSCALE_SHADOW_REJECTION_RETENTION_DAYS`` (unset or 0: retention disabled). Arguments
    override the environment; invalid values fall back to the safe defaults."""
    raw_detail = (detail or os.getenv("UPSCALE_SHADOW_DIAGNOSTICS_DETAIL") or "").strip().lower()
    mode: DiagnosticsDetail = DEFAULT_DETAIL
    for m in DETAIL_MODES:
        if raw_detail == m:
            mode = m
    n = DEFAULT_SAMPLE_PER_REASON
    raw_n = sample_per_reason or os.getenv("UPSCALE_SHADOW_REJECTION_SAMPLE_PER_REASON")
    if raw_n and raw_n.strip():
        try:
            n = int(raw_n)
        except ValueError:
            n = DEFAULT_SAMPLE_PER_REASON
        if n < 1:
            n = DEFAULT_SAMPLE_PER_REASON
    days: float | None = None
    raw_days = retention_days or os.getenv("UPSCALE_SHADOW_REJECTION_RETENTION_DAYS")
    if raw_days and raw_days.strip():
        try:
            days = float(raw_days)
        except ValueError:
            days = None
        if days is not None and (not math.isfinite(days) or days <= 0):
            days = None
    return DiagnosticsSettings(detail=mode, sample_per_reason=n, retention_days=days)


def parse_timestamp(raw: str) -> datetime:
    """An ISO-8601 timestamp with a timezone (``Z`` or an offset)."""
    try:
        value = datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {raw!r} (e.g. 2026-09-30T05:50:00Z)") from exc
    if value.tzinfo is None:
        raise ValueError(f"timestamp {raw!r} has no timezone: add Z or an offset")
    return value.astimezone(UTC)


def load_settings(
    enabled: str | None,
    interval_minutes: str | None = None,
    run_id: str | None = None,
    since: str | None = None,
) -> ShadowSettings:
    """From ``UPSCALE_SHADOW`` (default OFF: only 1 / true / on / yes enable it),
    ``UPSCALE_SHADOW_INTERVAL_MINUTES`` (default 15, at least 5), ``UPSCALE_SHADOW_RUN``
    (default ``production``) and ``UPSCALE_SHADOW_SINCE`` (default: the clean-data
    cutoff; an earlier value is raised to it)."""
    on = (enabled or "").strip().lower() not in _OFF
    interval = DEFAULT_INTERVAL_MINUTES
    if interval_minutes and interval_minutes.strip():
        try:
            interval = float(interval_minutes)
        except ValueError:
            interval = math.nan
        if not math.isfinite(interval) or interval <= 0:
            interval = DEFAULT_INTERVAL_MINUTES
        interval = max(interval, MIN_INTERVAL_MINUTES)
    start = CLEAN_DATA_CUTOFF
    if since and since.strip():
        try:
            start = max(parse_timestamp(since), CLEAN_DATA_CUTOFF)
        except ValueError:
            start = CLEAN_DATA_CUTOFF
    run = (run_id or "").strip() or DEFAULT_RUN_ID
    if not _SLUG.match(run):
        run = DEFAULT_RUN_ID
    return ShadowSettings(enabled=on, interval_minutes=interval, run_id=run, since=start)
