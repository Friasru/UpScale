"""Scout: discovery of newly launched and accelerating tokens, before analysis.

Scout never decides: it finds tokens by exact identity (chain + contract / mint), observes
their real market activity, stores snapshots, and measures acceleration. A candidate is
handed to the existing UpScale pipeline for BUY / SELL / WAIT.
"""

from upscale.services.scout.config import ScoutConfig, ScoutConfigError, load_scout_config
from upscale.services.scout.gate import RequestGate
from upscale.services.scout.models import (
    NOT_A_RECOMMENDATION,
    ScoutCandidate,
    ScoutGrowthFeatures,
    ScoutMarketMetrics,
    ScoutRiskFlags,
    ScoutRun,
    ScoutSnapshot,
    ScoutSourceEvidence,
)
from upscale.services.scout.providers import (
    DexScreenerDiscoveryProvider,
    DiscoveryNotSupportedError,
    DiscoveryProvider,
    GeckoTerminalDiscoveryProvider,
)
from upscale.services.scout.service import ScoutService
from upscale.services.scout.store import ScoutSnapshotStore

__all__ = [
    "NOT_A_RECOMMENDATION",
    "DexScreenerDiscoveryProvider",
    "DiscoveryNotSupportedError",
    "DiscoveryProvider",
    "GeckoTerminalDiscoveryProvider",
    "RequestGate",
    "ScoutCandidate",
    "ScoutConfig",
    "ScoutConfigError",
    "ScoutGrowthFeatures",
    "ScoutMarketMetrics",
    "ScoutRiskFlags",
    "ScoutRun",
    "ScoutService",
    "ScoutSnapshot",
    "ScoutSnapshotStore",
    "ScoutSourceEvidence",
    "load_scout_config",
]
