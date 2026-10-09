"""Frozen orchestration decisions for later phases (B2 / B3). B1 implements none of the
state machine; these are recorded here so the boundaries can't drift.

A. Infrastructure failure is not a token decision. When B3 collects Safety evidence, only
   *observed token evidence* may proceed to a Safety snapshot and an Opportunity decision.
   An infrastructure / orchestration failure leaves the job DEFERRED, RETRY_WAIT or
   FAILED and records NO Opportunity decision, so outages never contaminate Opportunity
   outcome / calibration data.

B. A Safety budget preflight is a conservative estimate, never a guarantee: retries and the
   optional pool-account check vary, another Safety process may spend budget, providers may
   fail. Safety's own RequestGuard stays the final authority.

C. Deferred B3 design items (unresolved; frozen components are not touched in B1):
   see `DEFERRED_B3_ITEMS`.
"""

from typing import Literal

# Observed token evidence: may lead to a Safety snapshot + Opportunity decision.
TOKEN_EVIDENCE_OUTCOMES: tuple[str, ...] = (
    "NOT_A_MINT",  # a genuine mint-account result (NOT_A_TOKEN_MINT)
    "ACCOUNT_MISSING",  # the actual on-chain result
    "NO_POOLS",  # -> NO_ELIGIBLE_MARKET
    "CLOSED_ON_CHAIN",  # confirmed closed market
    "HOLDERS_COMPLETE",
    "HOLDERS_PARTIAL",
    "OTHER_OBSERVED_SAFETY_FACT",
)
# Infrastructure / orchestration failures: never an automatic Opportunity SKIP.
INFRASTRUCTURE_FAILURES: tuple[str, ...] = (
    "PROVIDER_UNAVAILABLE",
    "PROVIDER_TIMEOUT_AFTER_RETRIES",
    "SAFETY_BUDGET_EXHAUSTED",
    "SAFETY_COOLDOWN",
    "PROVIDER_NOT_CONFIGURED",
    "DATABASE_LOCK_OR_READ_FAILURE",
    "COLLECTION_EXCEPTION_BEFORE_EVIDENCE",
)
InfrastructureState = Literal["DEFERRED", "RETRY_WAIT", "FAILED"]
PreflightLabel = Literal[
    "CONSERVATIVE_PREFLIGHT_OK", "CONSERVATIVE_PREFLIGHT_INSUFFICIENT", "COST_BOUND_UNKNOWN"
]
DEFERRED_B3_ITEMS: tuple[str, ...] = (
    "Safety's isolation test forbids importing Safety V2 outside its package",
    "Safety has no public service factory (credentials are wired in its private CLI helper)",
    "Safety exposes no exact / worst-case collection request-cost API",
    "the live bridge host needs direct access to the archive, Safety and Opportunity DBs",
    "Opportunity runs carry no orchestrator caller / job tag",
    "a concurrent manual Safety CLI run can race a bridge budget preflight",
)
