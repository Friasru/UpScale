"""Opportunity Model V1 input types: one canonical, immutable `OpportunityInput` per token
and decision time. Inputs only: nothing here decides SKIP / WATCH / ENTER.

* Every source carries a `SourceRef`: its status, the store and record it came from, its
  fingerprint / hash, when it was observed and how old it was at ``decision_at``.
* Every piece of evidence is a `Fact` with exactly one owning layer per aspect (see
  `ownership`). A copy of an aspect seen in another layer's record is kept as a
  DIAGNOSTIC fact: visible, never scoring.
* Missing evidence is a NOT_AVAILABLE fact with a reason and no value: never 0, False or a
  neutral stand-in.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator, model_validator

from upscale.services.opportunity_model.config import INPUT_SCHEMA

SourceStatus = Literal[
    "AVAILABLE",  # read, fresh and compatible
    "STALE",  # read and compatible, but older than its freshness limit at decision_at
    "NOT_COLLECTED",  # nothing was recorded for this token at or before decision_at
    "UNAVAILABLE",  # the store couldn't be read, or the record says the source failed
    "INCOMPATIBLE",  # a record exists but its version / shape can't be used by V1
    "NOT_SUPPORTED",  # V1 doesn't cover this (e.g. a non-Solana token)
]
SOURCE_STATUSES: tuple[SourceStatus, ...] = (
    "AVAILABLE", "STALE", "NOT_COLLECTED", "UNAVAILABLE", "INCOMPATIBLE", "NOT_SUPPORTED",
)  # fmt: skip
OpportunityOrigin = Literal["LIVE_FORWARD", "HISTORICAL_REPLAY"]
CorrelationGroup = Literal[
    "IDENTITY",
    "AUTHORITY",
    "HOLDERS",
    "LIQUIDITY",
    "MARKET_STRUCTURE",
    "FLOW",
    "PRICE",
    "ATTENTION",
    "NEWS",
]
CORRELATION_GROUPS: tuple[CorrelationGroup, ...] = (
    "IDENTITY", "AUTHORITY", "HOLDERS", "LIQUIDITY", "MARKET_STRUCTURE", "FLOW", "PRICE",
    "ATTENTION", "NEWS",
)  # fmt: skip
Layer = Literal["scout", "technical", "social", "news", "safety"]
# SCORING: may feed a later decision. CONTEXT: a state / gate input, never additive.
# DIAGNOSTIC: a copy of another owner's aspect (or an unowned figure): shown, never used.
FactRole = Literal["SCORING", "CONTEXT", "DIAGNOSTIC"]
FactStatus = Literal["AVAILABLE", "NOT_AVAILABLE"]
SafetyRuleState = Literal[
    "TRIGGERED", "NOT_TRIGGERED", "UNDETERMINED", "OUT_OF_SCOPE", "NOT_SUPPORTED"
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class OpportunityError(Exception):
    """An Opportunity input can't be built honestly."""


class OpportunityCausalityError(OpportunityError):
    """A source selected for an input was observed after its decision time."""


class OpportunityIdentityError(OpportunityError):
    """A source describes a different chain or token than the input."""


class OwnershipError(OpportunityError):
    """A fact violates the ownership map (two scoring owners, wrong group or layer)."""


def _utc(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
        raise ValueError("times must be timezone-aware UTC")
    return value


class SourceRef(BaseModel):
    """Where a source's evidence came from and how usable it is at ``decision_at``."""

    model_config = _FROZEN

    status: SourceStatus
    store: str | None = None
    record_id: str | None = None
    fingerprint: str | None = None  # archive fingerprint / payload hash, when one exists
    observed_at: datetime | None = None
    age_seconds: float | None = None
    reason: str | None = None

    _times = field_validator("observed_at")(_utc)

    @model_validator(mode="after")
    def _honest(self) -> "SourceRef":
        if self.status != "AVAILABLE" and not self.reason:
            raise ValueError(f"a {self.status} source must say why")
        if self.status in ("AVAILABLE", "STALE") and (
            self.record_id is None or self.observed_at is None or self.age_seconds is None
        ):
            raise ValueError("a read source carries its record, observed_at and age")
        if self.age_seconds is not None and self.age_seconds < 0:
            raise ValueError("a source can't be observed after its decision time")
        return self


class SafetySourceRef(SourceRef):
    """A Safety V2 snapshot reference: ``observed_at`` is the snapshot's ``as_of``."""

    snapshot_id: int | None = None
    as_of: datetime | None = None
    rules_version: str | None = None
    body_hash: str | None = None

    _as_of = field_validator("as_of")(_utc)

    @model_validator(mode="after")
    def _snapshot(self) -> "SafetySourceRef":
        if self.status in ("AVAILABLE", "STALE") and (
            self.snapshot_id is None or self.as_of is None or self.body_hash is None
        ):
            raise ValueError("a read Safety snapshot carries its id, as_of and body hash")
        return self


class Fact(BaseModel):
    """One piece of evidence about one aspect. ``layer`` is where this copy came from;
    ``scoring_owner`` is the layer the ownership map gives the aspect (None: unowned in V1).
    """

    model_config = _FROZEN

    aspect: str
    group: CorrelationGroup
    layer: Layer
    role: FactRole
    status: FactStatus
    value: JsonValue = None
    reason: str | None = None
    path: str  # where in the source record the evidence was read
    scoring_owner: Layer | None

    @model_validator(mode="after")
    def _honest(self) -> "Fact":
        if self.status == "NOT_AVAILABLE" and (self.value is not None or not self.reason):
            raise ValueError("a NOT_AVAILABLE fact has no value and says why")
        if self.role != "DIAGNOSTIC" and self.scoring_owner != self.layer:
            raise ValueError(f"{self.aspect}: only its owner may hold a {self.role} fact")
        return self


class ScoutMarketView(BaseModel):
    """Opportunity-owned Scout evidence: per-sub-signal market / discovery facts only.
    Scout's composite score, family scores and every copy of another layer's evidence are
    in `ScoutFacts.diagnostics`, never here."""

    model_config = _FROZEN

    facts: tuple[Fact, ...] = ()


class ScoutFacts(BaseModel):
    model_config = _FROZEN

    ref: SourceRef
    market_observed_at: datetime | None = None  # when Scout's market evidence was fetched
    market_view: ScoutMarketView = ScoutMarketView()
    context: tuple[Fact, ...] = ()
    diagnostics: tuple[Fact, ...] = ()

    _times = field_validator("market_observed_at")(_utc)


class TechnicalFacts(BaseModel):
    """Price structure. ``ref``: Scout's snapshot TechnicalContext (archived inside the
    Scout record); ``analyze_ref``: the archived Analyze decision record, if any. Analyze's
    final action, confidence and overall risk are never loaded (they combine layers)."""

    model_config = _FROZEN

    ref: SourceRef
    analyze_ref: SourceRef
    facts: tuple[Fact, ...] = ()


class SocialFacts(BaseModel):
    model_config = _FROZEN

    ref: SourceRef
    facts: tuple[Fact, ...] = ()


class NewsFacts(BaseModel):
    """Context only in V1. Story sentiment / impact labels may come from a language model
    (``sentiment_models``); nothing here is ever a positive input."""

    model_config = _FROZEN

    ref: SourceRef
    llm_labelled: bool | None = None  # None: no report was read
    sentiment_models: tuple[str, ...] = ()
    facts: tuple[Fact, ...] = ()


class SafetyRule(BaseModel):
    """One Safety V2 rule exactly as the snapshot reported it (never flattened)."""

    model_config = _FROZEN

    id: str
    state: SafetyRuleState
    severity: str | None = None
    reason: str | None = None
    needs: str | None = None
    evidence: tuple[str, ...] = ()


class SafetyFacts(BaseModel):
    """The Safety V2 snapshot's sections, preserved as stored (read-only JSON objects)."""

    model_config = _FROZEN

    ref: SafetySourceRef
    snapshot_schema: str | None = None
    identity: dict[str, JsonValue] | None = None
    authority: dict[str, JsonValue] | None = None
    holders: dict[str, JsonValue] | None = None
    market: dict[str, JsonValue] | None = None
    creator: dict[str, JsonValue] | None = None
    changes: dict[str, JsonValue] | None = None
    coverage: dict[str, JsonValue] | None = None
    assessment: dict[str, JsonValue] | None = None
    rules: tuple[SafetyRule, ...] = ()
    facts: tuple[Fact, ...] = ()


class Sources(BaseModel):
    model_config = _FROZEN

    scout: ScoutFacts
    technical: TechnicalFacts
    social: SocialFacts
    news: NewsFacts
    safety: SafetyFacts


class OpportunityInput(BaseModel):
    """Everything Opportunity V1 may know about one token at ``decision_at``."""

    model_config = _FROZEN

    schema_version: str = INPUT_SCHEMA
    canonical_id: str
    chain: str
    address: str
    decision_at: datetime
    origin: OpportunityOrigin
    sources: Sources

    _times = field_validator("decision_at")(_utc)

    @model_validator(mode="after")
    def _consistent(self) -> "OpportunityInput":
        if self.schema_version != INPUT_SCHEMA:
            raise ValueError(f"unknown input schema {self.schema_version!r}")
        if self.canonical_id != f"{self.chain}:{self.address}":
            raise ValueError("canonical_id must be <chain>:<address>")
        from upscale.services.opportunity_model.ownership import check

        check(self.facts())
        return self

    def facts(self) -> tuple[Fact, ...]:
        s = self.sources
        return (
            *s.scout.market_view.facts, *s.scout.context, *s.scout.diagnostics,
            *s.technical.facts, *s.social.facts, *s.news.facts, *s.safety.facts,
        )  # fmt: skip

    def scoring_facts(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts() if f.role == "SCORING")

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def input_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
