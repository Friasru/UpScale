"""Per-category analysis strategies: every tunable threshold the technical, risk and decision
rules use, chosen by the asset's profile instead of hard-coded for BTC.

Every category currently uses the same, existing defaults, so BTC and SOL behave exactly as
before. Tuning a category means giving it its own `Strategy` here (with tests), never
editing the rules or branching on a ticker.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from upscale.services.asset_profile import CATEGORY_LABELS, Category, CryptoAssetProfile
from upscale.services.opportunity import OpportunityConfig
from upscale.services.risk import DexRiskConfig, OnchainRiskConfig, RiskConfig
from upscale.services.technical_analysis import TechnicalAnalysisConfig
from upscale.services.technical_pool import TechnicalPoolConfig


@dataclass(frozen=True)
class Strategy:
    technical: TechnicalAnalysisConfig = field(default_factory=TechnicalAnalysisConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    dex_risk: DexRiskConfig = field(default_factory=DexRiskConfig)
    onchain_risk: OnchainRiskConfig = field(default_factory=OnchainRiskConfig)
    opportunity: OpportunityConfig = field(default_factory=OpportunityConfig)
    technical_pool: TechnicalPoolConfig = field(default_factory=TechnicalPoolConfig)


DEFAULT_STRATEGY = Strategy()
# One entry per category; all share the current defaults until a category is tuned.
STRATEGIES: Mapping[Category, Strategy] = MappingProxyType(
    {category: DEFAULT_STRATEGY for category in CATEGORY_LABELS}
)


def strategy_for(
    profile: CryptoAssetProfile | None, strategies: Mapping[Category, Strategy] = STRATEGIES
) -> Strategy:
    """The strategy for this asset's category (the default without a profile). Category
    risk overrides from the profile are applied on top of the strategy's risk config."""
    if profile is None:
        return DEFAULT_STRATEGY
    base = strategies.get(profile.category, DEFAULT_STRATEGY)
    if not profile.risk_overrides:
        return base
    return dataclasses.replace(base, risk=dataclasses.replace(base.risk, **profile.risk_overrides))
