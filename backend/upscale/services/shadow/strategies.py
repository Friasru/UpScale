"""Baseline comparison strategies (v1). Comparison rules, not production recommendations,
and deliberately not tuned: they share the same exit and risk rules, so only the entry
filter differs between them.

A. ``scout_threshold``: Growth Scout score and stage, a liquidity floor, a risk-penalty cap.
B. ``scout_technical``: A + Scout's snapshot Technical context confirming an up-trend.
C. ``scout_safety_technical``: B + on-chain safety evidence (no active mint / freeze
   authority, bounded holder concentration).
D. ``random_eligible``: a deterministic pseudo-random 10% of eligible, liquid candidates
   (fixed seed): the "is the selection better than chance?" reference.
"""

from datetime import UTC, datetime

from upscale.services.shadow.config import (
    MISSING_LABELS,
    EntryRules,
    ExitRules,
    RiskRules,
    StrategyConfig,
    TechnicalRules,
)

BASELINES_CREATED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
EXIT = ExitRules()  # +30% take profit, -15% stop, 24h max hold, exit on FADING
RISK = RiskRules()  # $10,000 capital, $500 (<= 5%) per entry, 10 open max, 1 per asset
_NO_TECH_MISSING = tuple(m for m in MISSING_LABELS if m != "TECHNICAL_NOT_AVAILABLE")
_NO_SAFETY_TECH_MISSING = tuple(m for m in _NO_TECH_MISSING if m != "SAFETY_NOT_AVAILABLE")

_A = EntryRules(
    min_scout_score=60.0,
    allowed_stages=("EARLY", "ACCELERATING"),
    min_liquidity_usd=50_000.0,
    max_risk_penalty=10.0,
)
_B = _A.model_copy(
    update={
        "technical": TechnicalRules(allowed_trends=("up",)),
        "allowed_missing": _NO_TECH_MISSING,
    }
)
_C = _B.model_copy(
    update={
        "required_safety": "PARTIAL_OR_COMPLETE",
        "block_active_authorities": True,
        "max_holder_top10_pct": 50.0,
        "allowed_missing": _NO_SAFETY_TECH_MISSING,
    }
)
_D = EntryRules(
    min_scout_score=0.0,
    allowed_stages=("NEW", "EARLY", "ACCELERATING", "CROWDED", "FADING", "STEADY"),
    min_liquidity_usd=50_000.0,
    max_risk_penalty=None,
    random_fraction=0.10,
    random_seed=20260930,
)


def _strategy(sid: str, name: str, description: str, entry: EntryRules) -> StrategyConfig:
    # model_validate re-runs the validators on the copied entry rules.
    return StrategyConfig.model_validate(
        {
            "strategy_id": sid,
            "version": 1,
            "name": name,
            "description": description,
            "created_at": BASELINES_CREATED_AT,
            "entry": entry.model_dump(),
            "exit": EXIT.model_dump(),
            "risk": RISK.model_dump(),
        }
    )


BASELINES: tuple[StrategyConfig, ...] = (
    _strategy(
        "scout_threshold",
        "A. Scout threshold baseline",
        "Score >= 60, stage EARLY / ACCELERATING, liquidity >= $50k, risk penalty <= 10.",
        _A,
    ),
    _strategy(
        "scout_technical",
        "B. Scout + Technical confirmation",
        "A + Scout's snapshot Technical context: up-trend over at least 3 stored snapshots.",
        _B,
    ),
    _strategy(
        "scout_safety_technical",
        "C. Scout + safety + Technical confirmation",
        "B + safety checks at least partial, no active mint / freeze authority, top-10 "
        "holders <= 50%.",
        _C,
    ),
    _strategy(
        "random_eligible",
        "D. Random eligible-entry baseline",
        "A deterministic pseudo-random 10% of eligible candidates with liquidity >= $50k "
        "(seed 20260930).",
        _D,
    ),
)
