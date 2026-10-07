"""Audit / Learning V1 settings: descriptive labels, buckets and sample minimums.

Everything here only shapes a read-only report. Nothing is read by production, and no
setting here changes Scout, Shadow, Calibration or any threshold they use.
"""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

HORIZONS: tuple[str, ...] = ("5m", "15m", "1h", "4h", "24h")
POPULATIONS = ("scout", "shadow")


def _scout_db() -> str:
    return os.getenv("UPSCALE_SCOUT_DB") or str(Path.home() / ".upscale" / "scout.sqlite3")


def default_scout_db() -> str:
    """The Scout database (Scout anchors and their outcome horizons; read-only here)."""
    return _scout_db()


def default_evidence_db() -> str:
    return os.getenv("UPSCALE_EVIDENCE_DB") or str(
        Path(_scout_db()).expanduser().parent / "evidence.sqlite3"
    )


def default_shadow_db() -> str:
    return os.getenv("UPSCALE_SHADOW_DB") or str(
        Path(_scout_db()).expanduser().parent / "shadow.sqlite3"
    )


class Labels(BaseModel):
    """Descriptive outcome labels (percent). A label is a fact about the future, used only
    to describe outcomes, never as a decision-time feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    large_winner_pct: float = 25.0  # return >= +25%
    catastrophic_pct: float = -30.0  # return <= -30%
    high_mfe_pct: float = 25.0  # MFE >= +25%
    high_mae_pct: float = -20.0  # MAE <= -20%

    @model_validator(mode="after")
    def _signs(self) -> "Labels":
        if self.large_winner_pct <= 0 or self.high_mfe_pct <= 0:
            raise ValueError("winner / MFE labels must be positive percentages")
        if self.catastrophic_pct >= 0 or self.high_mae_pct >= 0:
            raise ValueError("catastrophic / MAE labels must be negative percentages")
        return self


class Buckets(BaseModel):
    """Bucket edges (lower bound inclusive). Values below the first edge form their own
    bucket; a missing value is its own UNAVAILABLE bucket, never zero."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: tuple[float, ...] = (40.0, 60.0, 75.0, 90.0)
    liquidity_usd: tuple[float, ...] = (10_000.0, 50_000.0, 100_000.0, 250_000.0, 1_000_000.0)
    market_cap_usd: tuple[float, ...] = (100_000.0, 1_000_000.0, 10_000_000.0, 100_000_000.0)
    risk_penalty: tuple[float, ...] = (0.000001, 5.0, 10.0, 20.0)  # 0 is its own bucket
    execution_delay_seconds: tuple[float, ...] = (60.000001, 300.0, 900.0, 3600.0)


class Minimums(BaseModel):
    """A group below any minimum is INSUFFICIENT_SAMPLE: its figures are shown, labeled,
    and never turned into a hypothesis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    measured: int = Field(default=30, ge=1)  # observations with a measured label
    unique_assets: int = Field(default=10, ge=1)
    effective_sample: float = Field(default=10.0, ge=1.0)


class HypothesisRules(BaseModel):
    """When a difference is strong enough to be listed as a hypothesis for Calibration.

    Both sides must meet `Minimums`; the difference must be at least `min_rate_effect`
    (rates, percentage points) or `min_return_effect` (mean MAE / clipped mean return,
    percentage points), and at least `z` asset-clustered standard errors away from zero.
    `z` defaults to 2.576 (two-sided 99%) because many comparisons are made."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    z: float = Field(default=2.576, gt=0)
    min_rate_effect: float = Field(default=5.0, ge=0)
    min_return_effect: float = Field(default=2.0, ge=0)
    return_clip_pct: float = Field(default=100.0, gt=0)  # mean return clipped to +-100%
    max_listed: int = Field(default=12, ge=1)


class AuditConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: Labels = Labels()
    buckets: Buckets = Buckets()
    minimums: Minimums = Minimums()
    hypotheses: HypothesisRules = HypothesisRules()
    primary_horizon: str = "4h"
    # Analyze Technical evidence (RSI / MACD / support-resistance) counts only when it was
    # observed at or before the decision and at most this long before it.
    analyze_max_age_minutes: float = Field(default=360.0, gt=0)
    trim: float = Field(default=0.10, ge=0, lt=0.5)
    # A Scout decision whose market observation is older than this is reported as a data
    # quality note (its outcome window starts at the market observation).
    decision_lag_note_minutes: float = Field(default=15.0, gt=0)
    batch: int = Field(default=500, ge=1, le=900)  # SQLite IN (...) chunk
    # Keep free-text features (Scout reasons) in snapshots: `feature-audit` only.
    text_features: bool = False

    @model_validator(mode="after")
    def _horizon(self) -> "AuditConfig":
        if self.primary_horizon not in HORIZONS:
            raise ValueError(f"primary horizon must be one of {', '.join(HORIZONS)}")
        return self
