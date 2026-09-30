"""Replay Lab: calibration research by replaying historical setups through UpScale's
production logic with no lookahead, then measuring the known outcomes (5m ... 24h).

HISTORICAL_REPLAY data only, in its own database: it never writes live (LIVE_FORWARD)
Scout / outcome data, never changes production configuration, never trades, and never
touches wallets or keys. Results are research evidence, not proof of profitability.
See `engine` for the pipeline and `clock` for the anti-lookahead rules.
"""

from upscale.services.replay_lab.clock import DecisionReceipt, HistoricalClock, LookaheadError
from upscale.services.replay_lab.config import HISTORICAL_REPLAY, ReplayJobConfig, SplitConfig
from upscale.services.replay_lab.engine import ReplayRunner
from upscale.services.replay_lab.store import ReplayIsolationError, ReplayStore

__all__ = [
    "HISTORICAL_REPLAY",
    "DecisionReceipt",
    "HistoricalClock",
    "LookaheadError",
    "ReplayIsolationError",
    "ReplayJobConfig",
    "ReplayRunner",
    "ReplayStore",
    "SplitConfig",
]
