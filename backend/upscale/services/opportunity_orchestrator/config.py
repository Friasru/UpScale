"""Orchestration policy values (frozen for B1). They decide only which Scout candidates are
worth considering for Safety budget; they never change Scout, Safety or Opportunity rules.
"""

from dataclasses import dataclass

# Stages capable of reaching Opportunity's current entry setup (approved, high level only).
ADMISSIBLE_STAGES: tuple[str, ...] = ("ACCELERATING", "EARLY")
STAGE_PRIORITY: dict[str, int] = {"ACCELERATING": 0, "EARLY": 1}
# Scout's own values (GrowthQuality / GrowthCandidate), read as stored.
VERIFIED_IDENTITY = "VERIFIED_IDENTITY"
CURRENT_DATA = "CURRENT"
MARKET_COLLAPSE = "MARKET_COLLAPSE"
DIVERGENT_FLOW = "divergent"
MIN_SCOUT_TIMING_VERSION = 2

# Safety snapshots the reuse preview can read (the version Opportunity V1 requires).
SAFETY_RULES_VERSION = "4"
SAFETY_SNAPSHOT_SCHEMA = "safety.snapshot.v2"
# Safety's request-ledger budget limit, only when explicitly configured (never assumed).
SAFETY_BUDGET_ENV = "UPSCALE_SAFETY_V2_DAILY_REQUEST_BUDGET"
SAFETY_COOLDOWN_META_KEY = "provider.cooldown_until"


@dataclass(frozen=True)
class OrchestratorPolicy:
    max_rank: int = 10
    admission_market_age_s: float = 15 * 60  # Scout market evidence at admission
    settle_s: float = 60.0  # archive quiet period before a Scout run is considered
    safety_reuse_max_age_s: float = 45 * 60  # Opportunity's 60 min rule stays authoritative
    run_lookback_s: float = 3 * 3600  # how far back to look for Scout runs
    archive_scan_limit: int = 20_000  # records read per scan (truncation is reported)

    def __post_init__(self) -> None:
        if self.max_rank < 1 or min(self.admission_market_age_s, self.settle_s,
                                    self.safety_reuse_max_age_s, self.run_lookback_s) < 0:  # fmt: skip
            raise ValueError("orchestration policy values must be positive")


POLICY = OrchestratorPolicy()
