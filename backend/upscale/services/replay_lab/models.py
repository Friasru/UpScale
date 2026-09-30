"""Replay Lab's typed records.

* `PlannedSample`: one historical decision time T for one exact pool, with its split.
* `ReplayDecisionRecord`: everything UpScale concluded at T from evidence available at T
  (features, Scout, Technical, Risk, Opportunity, the decision). Frozen and immutable once
  stored: outcomes can never change it.
* `ReplayHorizonOutcome`: what the market did over one horizon after T, measured with the
  production outcome definitions. Factual metrics only: not realized profit, and no fill,
  fee, slippage, size or latency is assumed.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from upscale.services.outcomes.models import (
    DecisionObservation,
    HorizonStatus,
    MarketAtHorizon,
    MarketStatus,
    PricePath,
    TriggerOutcome,
)
from upscale.services.replay_lab.config import Evidence, Mode, Split

RECORD_VERSION = 1

NOT_A_BACKTEST = (
    "Historical replay of UpScale's production logic on evidence available at each decision "
    "time. Outcomes are market measurements, not realized trading profit: no entry, fill, "
    "fee, slippage, position size or latency is assumed, and replay alone never shows a "
    "setup is profitable."
)

Availability = str  # "AVAILABLE" | "IMMUTABLE_METADATA" | "UNAVAILABLE: <why>" | "DEFAULTED: <why>"


class PlannedSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_key: str
    asset_id: str  # canonical id: <chain>:<token address>
    chain: str
    token_address: str
    pool_address: str
    symbol: str | None = None
    decision_at: datetime
    evidence: Evidence
    universe_basis: str  # why this asset was in the universe at T (bias is labeled)
    split: Split
    purged: bool  # outcome window crosses into a later split: kept out of findings
    cohort_at: datetime  # T bucket: the candidates competing at the same moment
    plan_order: int
    snapshot_provider: str | None = None  # RECORDED: whose snapshot is the evidence
    # Set when the sample reproduces an archived Scout decision: `decision_at` is then its
    # decision time D (when it was final), this is the market observation time T <= D.
    market_observed_at: datetime | None = None


class AgentOutput(BaseModel):
    agent: str
    status: str
    summary: str
    findings: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ScoutReplay(BaseModel):
    """Production Growth Scout's evaluation at T (or why it couldn't be reconstructed)."""

    status: Literal["RECONSTRUCTED", "NOT_RECONSTRUCTIBLE"]
    reason: str | None = None
    stage: str | None = None
    unconfirmed_stage: str | None = None
    stage_reasons: list[str] = Field(default_factory=list)
    score: float | None = None
    base: float | None = None
    stage_adjustment: float | None = None
    risk_penalty: float | None = None
    families: dict[str, float] = Field(default_factory=dict)  # contribution per family
    eligible: bool | None = None
    ineligible_reasons: list[str] = Field(default_factory=list)
    risk_flags: list[dict[str, Any]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    social_status: str | None = None
    safety_status: str | None = None
    candidate: dict[str, Any] | None = None  # the full GrowthCandidate, for later research


class ReplayDecisionRecord(BaseModel):
    """Frozen state at T. Stored once; the replay database refuses updates and deletes."""

    origin: Literal["HISTORICAL_REPLAY"] = "HISTORICAL_REPLAY"
    record_version: int = RECORD_VERSION
    sample_key: str
    asset_id: str
    chain: str
    token_address: str
    pool_address: str
    symbol: str | None
    decision_at: datetime
    evidence: Evidence
    mode: Mode
    # Market evidence time T; `decision_at` is the decision time D (>= T). Equal for samples
    # without an archived decision to reproduce.
    market_observed_at: datetime | None = None
    decision_basis: str | None = None
    # The newest timestamp of any evidence used (always <= decision_at).
    evidence_latest_at: datetime
    reference_price: float | None
    reference_basis: Literal["recorded_snapshot_price", "last_closed_candle"] | None
    features: dict[str, Any] = Field(default_factory=dict)
    availability: dict[str, Availability] = Field(default_factory=dict)
    social: dict[str, Any] = Field(default_factory=dict)
    scout: ScoutReplay
    agents: dict[str, AgentOutput] = Field(default_factory=dict)
    agents_unavailable: dict[str, str] = Field(default_factory=dict)
    decision: DecisionObservation | None = None
    action: str | None = None
    confidence: str | None = None
    risk_level: str | None = None
    uncertainty_level: str | None = None
    warnings: list[str] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    # Future position-aware evaluation (entry, exposure, prior recommendation...). None in
    # v1: production has no position model yet, and none is invented here.
    position_context: dict[str, Any] | None = None


class ReplayHorizonOutcome(BaseModel):
    horizon: str
    horizon_minutes: int
    window_start: datetime
    window_end: datetime
    status: HorizonStatus
    market_status: MarketStatus | None = None
    price: PricePath | None = None
    market: MarketAtHorizon | None = None
    future_stage: str | None = None
    future_stage_at: datetime | None = None
    future_score: float | None = None
    triggers: dict[str, TriggerOutcome] = Field(default_factory=dict)
    first_trigger_event: str | None = None
    price_position: str | None = None
    missing: list[str] = Field(default_factory=list)


class ReplayFidelity(BaseModel):
    """Revealed after the decision: what the live system itself recorded for the run that
    used this snapshot (RECORDED samples only), to check replay reproduces production."""

    live_stage: str | None = None
    live_score: float | None = None
    live_rank: int | None = None
    live_run_at: datetime | None = None
    stage_matches: bool | None = None
    score_delta: float | None = None
    notes: list[str] = Field(default_factory=list)
