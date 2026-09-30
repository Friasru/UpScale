"""Calibration Engine settings. Every threshold that decides what counts as evidence lives
here (validated), never in the analysis code."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import upscale.config  # noqa: F401  (loads the repo-root .env before reading variables)

Origin = Literal["LIVE_FORWARD", "HISTORICAL_REPLAY", "SHADOW"]
ORIGINS: tuple[Origin, ...] = ("LIVE_FORWARD", "HISTORICAL_REPLAY", "SHADOW")
Split = Literal["CALIBRATION", "VALIDATION", "HOLDOUT"]
Strength = Literal["INSUFFICIENT_SAMPLE", "WEAK_EVIDENCE", "MODERATE_EVIDENCE", "STRONGER_EVIDENCE"]
HORIZONS: tuple[str, ...] = ("5m", "15m", "1h", "4h", "24h")


def _scout_dir() -> Path:
    scout = os.getenv("UPSCALE_SCOUT_DB") or str(Path.home() / ".upscale" / "scout.sqlite3")
    return Path(scout).expanduser().parent


def default_calibration_db() -> str:
    return os.getenv("UPSCALE_CALIBRATION_DB") or str(_scout_dir() / "calibration.sqlite3")


def default_live_db() -> str:
    """The live Scout / outcome database (read-only here)."""
    return os.getenv("UPSCALE_SCOUT_DB") or str(_scout_dir() / "scout.sqlite3")


def default_replay_db() -> str:
    return os.getenv("UPSCALE_REPLAY_DB") or str(_scout_dir() / "replay.sqlite3")


def default_shadow_db() -> str:
    """The Shadow / Paper Strategy database (read-only here)."""
    return os.getenv("UPSCALE_SHADOW_DB") or str(_scout_dir() / "shadow.sqlite3")


class SampleRules(BaseModel):
    """Minimum support. Below `descriptive` a cohort is INSUFFICIENT_SAMPLE (counts only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    descriptive_n: int = Field(default=20, gt=0)
    descriptive_assets: int = Field(default=5, gt=0)
    candidate_n: int = Field(default=50, gt=0)
    candidate_assets: int = Field(default=10, gt=0)
    strong_n: int = Field(default=100, gt=0)
    strong_assets: int = Field(default=20, gt=0)
    outer_percentiles_n: int = Field(default=50, gt=0)  # p10 / p90

    @model_validator(mode="after")
    def _ordered(self) -> "SampleRules":
        if not self.descriptive_n <= self.candidate_n <= self.strong_n:
            raise ValueError("sample thresholds must increase: descriptive <= candidate <= strong")
        if not self.descriptive_assets <= self.candidate_assets <= self.strong_assets:
            raise ValueError("asset thresholds must increase")
        return self


class CorrelationRules(BaseModel):
    """Nearby observations of one token are not independent evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_per_asset: int = Field(default=10, gt=0)  # per origin and split
    min_spacing_minutes: float = Field(default=60.0, ge=0)


class LiveSplitPolicy(BaseModel):
    """Live observations get a sticky, chronological split by UTC calendar day: the days
    since `anchor` form cycles of `cycle_days`; within each, the first
    `calibration_days` are CALIBRATION, the next `validation_days` VALIDATION, the rest
    HOLDOUT. A day's split never changes as data grows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anchor: str = "2026-01-01"
    cycle_days: int = Field(default=20, gt=2)
    calibration_days: int = Field(default=14, gt=0)
    validation_days: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _fits(self) -> "LiveSplitPolicy":
        if self.calibration_days + self.validation_days >= self.cycle_days:
            raise ValueError("the cycle must leave at least one HOLDOUT day")
        return self


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: SampleRules = SampleRules()
    correlation: CorrelationRules = CorrelationRules()
    live_split: LiveSplitPolicy = LiveSplitPolicy()
    seed: int = 7
    bootstrap_resamples: int = Field(default=400, ge=0, le=5000)
    # Interactions: at most this many conditions per pattern and patterns evaluated.
    max_interaction_size: int = Field(default=3, ge=1, le=3)
    max_patterns: int = Field(default=200, gt=0)
    # Bounded weight perturbations (relative), per family, normalization preserved.
    weight_steps: tuple[float, ...] = (-0.10, -0.05, 0.05, 0.10)
    # A difference smaller than this (percentage points) is "no material difference".
    material_return_pp: float = Field(default=1.0, ge=0)
    material_mae_pp: float = Field(default=2.0, ge=0)
    material_rank_correlation: float = Field(default=0.05, ge=0)
    # Thresholds explored by sensitivity analysis (never deployed).
    liquidity_floors_usd: tuple[float, ...] = (10_000.0, 20_000.0, 50_000.0, 100_000.0)
    score_floors: tuple[float, ...] = (50.0, 55.0, 60.0, 65.0, 70.0)
    top10_ceilings_pct: tuple[float, ...] = (30.0, 40.0, 50.0)
    prior_move_ceilings_pct: tuple[float, ...] = (20.0, 40.0, 60.0)
    # A candidate filter must keep at least this share of the base sample.
    min_retention: float = Field(default=0.3, gt=0, le=1)


class TimeFilter(BaseModel):
    """Only observations whose decision time is in [since, until) (`since` inclusive,
    `until` exclusive), for the origins listed (LIVE_FORWARD by default: a live cutoff such
    as a data fix never removes replay samples unless asked). Cohort membership uses the
    observation / decision time, never when its outcome was completed. Inactive when both
    bounds are None: every command then behaves exactly as without a filter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    since: datetime | None = None
    until: datetime | None = None
    origins: tuple[Origin, ...] = ("LIVE_FORWARD",)

    @field_validator("since", "until")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must include a timezone, e.g. 2026-09-30T05:50:00Z")
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def _ordered(self) -> "TimeFilter":
        if self.since and self.until and self.until <= self.since:
            raise ValueError("--until must be after --since")
        if not self.origins:
            raise ValueError("the time filter needs at least one origin")
        return self

    @property
    def active(self) -> bool:
        return self.since is not None or self.until is not None

    def keeps(self, origin: str, at: datetime) -> bool:
        if not self.active or origin not in self.origins:
            return True
        return (self.since is None or at >= self.since) and (self.until is None or at < self.until)

    def describe(self) -> dict[str, object]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
            "origins": list(self.origins),
        }


def parse_timestamp(raw: str) -> datetime:
    """An ISO-8601 timestamp with a timezone (``Z`` or an offset)."""
    try:
        value = datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {raw!r} (e.g. 2026-09-30T05:50:00Z)") from exc
    if value.tzinfo is None:
        raise ValueError(
            f"timestamp {raw!r} has no timezone: add Z or an offset, e.g. 2026-09-30T05:50:00Z"
        )
    return value.astimezone(UTC)
