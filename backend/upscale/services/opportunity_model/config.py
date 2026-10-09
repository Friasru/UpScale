"""Opportunity Model V1 input settings, frozen. Change these values, not the logic.

Freshness limits come from limits UpScale already uses: Scout runs every 15 minutes (20
allows one cycle plus slack); the Evidence Archive status treats safety and social
evidence older than 60 minutes as stale; Shadow ignores Analyze decisions older than 60
minutes.
"""

import os
from dataclasses import dataclass
from pathlib import Path

INPUT_SCHEMA = "opportunity.input.v1"
SUPPORTED_CHAINS: tuple[str, ...] = ("solana",)

# Safety V2 snapshots Opportunity V1 understands. Older rules versions lack the creator /
# holder-change sections and are INCOMPATIBLE (never read as partial evidence). Frozen here,
# never imported from Safety V2, so a later Safety version can't silently change V1 inputs.
SAFETY_RULES_VERSION = "4"
SAFETY_SNAPSHOT_SCHEMA = "safety.snapshot.v2"

# Archived Scout records before timing version 2 used the ranking run's start as
# observed_at (not the decision time): their timing can't support causal replay.
MIN_SCOUT_TIMING_VERSION = 2

# Analyze agent result keys in an archived ``decision`` record.
TECHNICAL_AGENT = "technical_analysis"
NEWS_AGENT = "news_sentiment"

ARCHIVE_STORE = "evidence_archive"
SAFETY_STORE = "safety"


@dataclass(frozen=True)
class Freshness:
    """Maximum age, in seconds before ``decision_at``, at which a source is AVAILABLE. A
    source exactly at the limit is fresh; older is STALE."""

    scout_s: float = 20 * 60
    safety_s: float = 60 * 60
    social_s: float = 60 * 60
    analyze_s: float = 60 * 60  # Analyze-derived technical and news

    def __post_init__(self) -> None:
        if min(self.scout_s, self.safety_s, self.social_s, self.analyze_s) <= 0:
            raise ValueError("freshness limits must be positive")


FRESHNESS = Freshness()


def evidence_db_path() -> str:
    """The Evidence Archive (read-only here): ``UPSCALE_EVIDENCE_DB``, else next to Scout."""
    scout = os.getenv("UPSCALE_SCOUT_DB") or str(Path.home() / ".upscale" / "scout.sqlite3")
    return os.getenv("UPSCALE_EVIDENCE_DB") or str(
        Path(scout).expanduser().parent / "evidence.sqlite3"
    )


def safety_db_path() -> str | None:
    """The Safety V2 database (read-only here), from the same variable Safety V2 uses. No
    guessed default: unset, the Safety source is UNAVAILABLE (Safety V2's isolation rule
    keeps its package name out of other modules)."""
    return os.getenv("UPSCALE_SAFETY_V2_DB") or None
