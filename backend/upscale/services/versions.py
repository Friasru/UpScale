"""Stable fingerprints of the production implementation that produced a record.

Each fingerprint hashes the relevant configuration (as loaded, env overrides included)
and the source of the modules holding the rules, so two records carry the same value only
when the same scoring / risk / decision implementation produced them. No secret is ever
part of a fingerprint: only configuration models and source files are hashed.
"""

import hashlib
import json
import os
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

FEATURE_SCHEMA_VERSION = "1"

_ROOT = Path(__file__).resolve().parents[1]  # the `upscale` package
_SOURCES: dict[str, tuple[str, ...]] = {
    "scout_scoring": (
        "services/scout/growth/scoring.py",
        "services/scout/growth/signals.py",
        "services/scout/growth/service.py",
        "services/scout/features.py",
        "services/scout/normalize.py",
    ),
    "risk": ("services/risk.py", "services/asset_profile.py"),
    "opportunity": ("services/opportunity.py",),
    "technical": ("services/technical_analysis.py", "agents/technical.py"),
    "onchain_safety": ("services/solana_chain.py",),
    "outcomes": ("services/outcomes/metrics.py",),
}


def _digest(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(
            p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=repr).encode()
        )
    return h.hexdigest()[:16]


def _source(name: str) -> bytes:
    out = b""
    for rel in _SOURCES[name]:
        path = _ROOT / rel
        out += path.read_bytes() if path.exists() else rel.encode()
    return out


def code_revision() -> str:
    """The deployed commit (Railway sets RAILWAY_GIT_COMMIT_SHA), else the local git HEAD."""
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )  # fmt: skip
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@cache
def fingerprints() -> dict[str, str]:
    from upscale.config import GROWTH_CONFIG, OUTCOME_CONFIG, SCOUT_CONFIG
    from upscale.services.outcomes.config import load_outcome_config
    from upscale.services.scout.config import load_scout_config
    from upscale.services.scout.growth.config import load_growth_config
    from upscale.services.solana_chain import HolderAnalysisConfig
    from upscale.services.strategy import STRATEGIES

    strategies = {k: repr(v) for k, v in sorted(STRATEGIES.items())}
    return {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "code": code_revision(),
        "scout_scoring": _digest(
            load_growth_config(GROWTH_CONFIG).model_dump(mode="json"), _source("scout_scoring")
        ),
        "scout_config": _digest(load_scout_config(SCOUT_CONFIG).model_dump(mode="json")),
        "risk": _digest(
            [repr(v.risk) + repr(v.dex_risk) + repr(v.onchain_risk) for v in STRATEGIES.values()],
            _source("risk"),
        ),
        "opportunity": _digest(
            [repr(v.opportunity) for v in STRATEGIES.values()], _source("opportunity")
        ),
        "technical": _digest(
            [repr(v.technical) + repr(v.technical_pool) for v in STRATEGIES.values()],
            _source("technical"),
        ),
        "onchain_safety": _digest(repr(HolderAnalysisConfig()), _source("onchain_safety")),
        "outcomes": _digest(
            load_outcome_config(OUTCOME_CONFIG).model_dump(mode="json"), _source("outcomes")
        ),
        "strategies": _digest(strategies),
    }
