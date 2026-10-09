"""Opportunity Model V1 decision types (``opportunity.decision.v1``).

ENTER means "this opportunity currently satisfies the entry evidence requirements". It is
never an order, a size, a price instruction or a transaction: those don't exist here.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from upscale.services.opportunity_model.models import CorrelationGroup, Layer, OpportunityError

Decision = Literal["SKIP", "WATCH", "ENTER"]
Quality = Literal["COMPLETE", "PARTIAL", "INSUFFICIENT"]
SetupBand = Literal["NONE", "WEAK", "MODERATE", "STRONG"]
TechnicalBand = Literal["CONFIRMING", "NEUTRAL", "CONTRADICTING", "UNAVAILABLE"]
SocialBand = Literal["SUPPORTING", "NEUTRAL", "CONTRADICTING", "UNAVAILABLE"]
RiskTier = Literal["CLEAN", "ELEVATED_1", "ELEVATED_2_PLUS"]
SkipBasis = Literal["HARD_VETO", "REQUIRED_SAFETY_NOT_READY", "NO_SETUP"]

SETUP_ORDER: dict[SetupBand, int] = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}
DECISION_ORDER: dict[Decision, int] = {"SKIP": 0, "WATCH": 1, "ENTER": 2}

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class OpportunityDecisionError(OpportunityError):
    """The input violates an O2 invariant: an integrity failure, never a decision."""


class Reason(BaseModel):
    """One deterministic, structured explanation entry (templated, never generated)."""

    model_config = _FROZEN

    code: str
    group: CorrelationGroup | None
    source: Layer | Literal["engine"]
    source_record_id: str | None
    evidence_paths: tuple[str, ...]
    status: str  # rule state / severity / band / source status
    message: str


class UpgradeItem(BaseModel):
    """One failed ENTER gate. Every listed gate must pass; fixing one guarantees nothing."""

    model_config = _FROZEN

    code: str
    gate: str
    message: str


class OpportunityDecision(BaseModel):
    model_config = _FROZEN

    schema_version: str
    rules_version: str
    canonical_id: str
    decision_at: datetime
    origin: str
    input_hash: str
    thresholds: dict[str, float]

    decision: Decision
    # Why a SKIP: HARD_VETO (evidence of a disqualifying risk), REQUIRED_SAFETY_NOT_READY
    # (required Safety domains never assessed: not entry-eligible, not proof of malice) and /
    # or NO_SETUP. Empty unless SKIP.
    skip_basis: tuple[SkipBasis, ...]
    quality: Quality
    safety_entry_readiness: dict[str, str]  # IDENTITY / HOLDERS / MARKET -> READY / ...
    setup_band: SetupBand
    technical_band: TechnicalBand
    social_band: SocialBand
    risk_tier: RiskTier

    positive_groups: tuple[CorrelationGroup, ...]
    risk_groups: tuple[CorrelationGroup, ...]

    vetoes: tuple[Reason, ...]
    ineligible: tuple[Reason, ...]  # required Safety evidence missing (non-risk gating)
    blockers: tuple[Reason, ...]
    positive_reasons: tuple[Reason, ...]
    risks: tuple[Reason, ...]
    missing_evidence: tuple[Reason, ...]
    upgrade_path: tuple[UpgradeItem, ...]
    note: str = (
        "ENTER means the entry evidence requirements are currently met; it is not an order, "
        "a size or a transaction"
    )

    def body(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        return json.dumps(self.body(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def decision_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
