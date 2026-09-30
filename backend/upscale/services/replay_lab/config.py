"""Replay Lab settings: where replay data lives and what a replay job asks for.

Replay data is HISTORICAL_REPLAY and lives in its own database
(`UPSCALE_REPLAY_DB`, default ``~/.upscale/replay.sqlite3``), never in the live Scout /
outcome database (LIVE_FORWARD). The live Scout database may be *read* (read-only) as a
historical archive of what Scout actually recorded at the time.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import upscale.config  # noqa: F401  (loads the repo-root .env before reading variables)
from upscale.services.outcomes.config import DEFAULT_HORIZONS, HorizonSpec

HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
LIVE_FORWARD = "LIVE_FORWARD"  # the live outcome tables (never written by Replay Lab)

Mode = Literal["MARKET_ONLY", "MARKET_PLUS_SOCIAL"]
# RECORDED: sample times are snapshots Scout really stored (full market evidence at T).
# CANDLES: sample times on a grid; only pool candles are historical (no trade counts,
# liquidity, market cap or FDV), so production Scout can't be reconstructed.
Evidence = Literal["RECORDED", "CANDLES"]
Split = Literal["CALIBRATION", "VALIDATION", "HOLDOUT"]
SPLITS: tuple[Split, ...] = ("CALIBRATION", "VALIDATION", "HOLDOUT")
JobStatus = Literal["PENDING", "RUNNING", "PAUSED", "COMPLETE", "FAILED"]
SampleStatus = Literal["PLANNED", "DECIDED", "COMPLETE", "SKIPPED", "FAILED"]


def default_replay_db() -> str:
    return os.getenv("UPSCALE_REPLAY_DB") or str(Path.home() / ".upscale" / "replay.sqlite3")


def default_evidence_db() -> str:
    """The Point-in-Time Evidence Archive (read-only here), next to the Scout database."""
    return os.getenv("UPSCALE_EVIDENCE_DB") or str(
        Path(default_archive_db()).expanduser().parent / "evidence.sqlite3"
    )


def default_archive_db() -> str:
    """The live Scout database, opened read-only as a historical archive."""
    return os.getenv("UPSCALE_SCOUT_DB") or str(Path.home() / ".upscale" / "scout.sqlite3")


class SplitConfig(BaseModel):
    """Time-based split of one job's samples (oldest first). HOLDOUT is the latest slice
    and is never used for calibration findings, thresholds or strategy selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["time"] = "time"
    calibration_pct: float = Field(default=70.0, gt=0, lt=100)
    validation_pct: float = Field(default=15.0, ge=0, lt=100)
    holdout_pct: float = Field(default=15.0, gt=0, lt=100)

    @model_validator(mode="after")
    def _sums_to_100(self) -> "SplitConfig":
        total = self.calibration_pct + self.validation_pct + self.holdout_pct
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"split percentages must sum to 100 (got {total:g})")
        return self


class AssetSpec(BaseModel):
    """`chain:token` (filter recorded samples) or `chain:token:pool` (an exact pool; the
    only way to replay candles for an asset the Scout archive doesn't know)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chain: str
    token: str
    pool: str | None = None

    @classmethod
    def parse(cls, raw: str) -> "AssetSpec":
        parts = raw.strip().split(":")
        if len(parts) not in (2, 3) or not all(parts):
            raise ValueError(f"asset {raw!r} must be chain:token or chain:token:pool")
        return cls(chain=parts[0], token=parts[1], pool=parts[2] if len(parts) == 3 else None)

    @property
    def canonical_id(self) -> str:
        from upscale.services.scout.normalize import canonical_id

        return canonical_id(self.chain, self.token)


class ReplayJobConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    origin: Literal["HISTORICAL_REPLAY"] = "HISTORICAL_REPLAY"
    start: datetime
    end: datetime
    chains: tuple[str, ...] = ("solana",)
    mode: Mode = "MARKET_ONLY"
    evidence: Evidence = "RECORDED"
    max_samples: int = Field(default=100, gt=0, le=100_000)
    min_spacing_minutes: float = Field(default=60.0, ge=0)
    # No asset may contribute more than this many samples (correlated observations).
    max_per_asset: int = Field(default=10, gt=0)
    assets: tuple[AssetSpec, ...] = ()
    split: SplitConfig = SplitConfig()
    provider: Literal["GeckoTerminal"] = "GeckoTerminal"
    horizons: tuple[HorizonSpec, ...] = DEFAULT_HORIZONS
    # Minutes after the longest horizon before a sample can be measured (candles close).
    settle_minutes: float = Field(default=5.0, ge=0)
    seed: int = 0

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @model_validator(mode="after")
    def _ordered(self) -> "ReplayJobConfig":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if not self.chains:
            raise ValueError("at least one chain is required")
        return self

    @property
    def max_horizon_minutes(self) -> int:
        return max(h.minutes for h in self.horizons)
