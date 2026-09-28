"""Growth Scout: ranks Scout's discovery candidates whose real market activity (and,
when present, attention) is beginning to accelerate together, and filters obvious weak or
manipulated setups.

Discovery ranking only: ScoutMomentumScore is not a probability of profit or a BUY
confidence, and nothing here outputs BUY / SELL. The Analyze pipeline decides afterward.
"""

from upscale.services.scout.growth.config import (
    GrowthConfig,
    GrowthConfigError,
    load_growth_config,
)
from upscale.services.scout.growth.models import (
    NOT_A_TRADE_SIGNAL,
    GrowthCandidate,
    GrowthScoutResult,
    GrowthStage,
    ScoutMomentumScore,
    SocialStatus,
)
from upscale.services.scout.growth.service import GrowthScoutService

__all__ = [
    "NOT_A_TRADE_SIGNAL",
    "GrowthCandidate",
    "GrowthConfig",
    "GrowthConfigError",
    "GrowthScoutResult",
    "GrowthScoutService",
    "GrowthStage",
    "ScoutMomentumScore",
    "SocialStatus",
    "load_growth_config",
]
