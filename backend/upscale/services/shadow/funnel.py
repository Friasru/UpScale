"""Sequential entry funnel: where each strategy's Scout evaluations actually drop out.

A strategy checks every entry rule on every evaluation (`Book._evaluate` collects every
failed rule, it never stops at the first), so the rules have no inherent order. This
module fixes ONE documented diagnostic order (`GATES`) and assigns each rejected
evaluation to the FIRST gate in that order that one of its reason codes belongs to. Every
evaluation therefore fails at most one gate: several raw reasons never double-count a
funnel failure. The order changes nothing about entries; it only decides where a
candidate that fails several rules is counted.

Diagnostic order (only the gates relevant to a strategy are shown)::

    TOTAL_EVALUATED
    CURRENT_AND_ELIGIBLE        SCOUT_NOT_ELIGIBLE, STALE_DATA, CURRENT_PRICE_UNAVAILABLE
    MARKET_NOT_COLLAPSED        MARKET_COLLAPSE
    STAGE_ALLOWED               STAGE_NOT_ALLOWED
    SCORE_PASS                  SCORE_BELOW_MIN
    LIQUIDITY_PASS              LIQUIDITY_NOT_AVAILABLE, LIQUIDITY_BELOW_MIN
    MARKET_CAP_AVAILABLE        MARKET_CAP_NOT_AVAILABLE
    RISK_PASS                   RISK_PENALTY_TOO_HIGH
    NO_BLOCKING_FLAG            BLOCKING_RISK_FLAG
    TECHNICAL_AVAILABLE         TECHNICAL_NOT_AVAILABLE
    TECHNICAL_SNAPSHOTS_PASS    TECHNICAL_TOO_FEW_SNAPSHOTS
    TECHNICAL_TREND_PASS        TECHNICAL_TREND_NOT_ALLOWED
    TECHNICAL_OTHER_REQUIREMENTS_PASS  breakout / volume / higher-lows requirements
    SAFETY_AVAILABLE            SAFETY_NOT_AVAILABLE
    SAFETY_LEVEL_PASS           SAFETY_LEVEL_INSUFFICIENT
    AUTHORITY_PASS              MINT_AUTHORITY_ACTIVE, FREEZE_AUTHORITY_ACTIVE
    HOLDER_CONCENTRATION_PASS   HOLDER_CONCENTRATION_TOO_HIGH
    SOCIAL_AVAILABLE            SOCIAL_NOT_AVAILABLE
    SOCIAL_PASS                 SOCIAL_REQUIREMENT_FAILED
    ANALYZE_PASS                ANALYZE_REQUIREMENT_FAILED
    OTHER_RULES_PASS            any code no gate above knows (a future rule)
    RANDOM_SELECTION_PASS       RANDOM_BASELINE_NOT_SELECTED (only ever the sole reason)
    ENTRY_QUALIFIED
    ENTERED                     failed = qualified but blocked by position / risk control

The input of every computation is a list of evaluation outcomes with counts
(``(outcome, reasons, count)``), exactly what the aggregate counters store, so a funnel
never needs one row per rejected candidate.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from upscale.services.shadow.config import STAGES, StrategyConfig

# The outcome of one Scout evaluation by one strategy (a held asset is not evaluated).
OUTCOMES = ("ENTERED", "BLOCKED", "REJECTED", "HELD")
Combo = tuple[str, tuple[str, ...], int]  # (outcome, reasons, count)


@dataclass(frozen=True)
class Gate:
    name: str
    codes: tuple[str, ...]
    group: str
    # A gate that is only measurable once `requires` passed (a Technical trend needs
    # Technical evidence): conditional counts use that as the denominator.
    requires: str | None = None


GATES: tuple[Gate, ...] = (
    Gate("CURRENT_AND_ELIGIBLE",
         ("SCOUT_NOT_ELIGIBLE", "STALE_DATA", "CURRENT_PRICE_UNAVAILABLE"), "SCOUT"),
    Gate("MARKET_NOT_COLLAPSED", ("MARKET_COLLAPSE",), "SCOUT"),
    Gate("STAGE_ALLOWED", ("STAGE_NOT_ALLOWED",), "SCOUT"),
    Gate("SCORE_PASS", ("SCORE_BELOW_MIN",), "SCOUT"),
    Gate("LIQUIDITY_PASS", ("LIQUIDITY_NOT_AVAILABLE", "LIQUIDITY_BELOW_MIN"), "SCOUT"),
    Gate("MARKET_CAP_AVAILABLE", ("MARKET_CAP_NOT_AVAILABLE",), "SCOUT"),
    Gate("RISK_PASS", ("RISK_PENALTY_TOO_HIGH",), "SCOUT"),
    Gate("NO_BLOCKING_FLAG", ("BLOCKING_RISK_FLAG",), "SCOUT"),
    Gate("TECHNICAL_AVAILABLE", ("TECHNICAL_NOT_AVAILABLE",), "TECHNICAL"),
    Gate("TECHNICAL_SNAPSHOTS_PASS", ("TECHNICAL_TOO_FEW_SNAPSHOTS",), "TECHNICAL",
         "TECHNICAL_AVAILABLE"),
    Gate("TECHNICAL_TREND_PASS", ("TECHNICAL_TREND_NOT_ALLOWED",), "TECHNICAL",
         "TECHNICAL_AVAILABLE"),
    Gate("TECHNICAL_OTHER_REQUIREMENTS_PASS",
         ("TECHNICAL_BREAKOUT_REQUIRED", "TECHNICAL_VOLUME_NOT_CONFIRMED",
          "TECHNICAL_HIGHER_LOWS_REQUIRED"), "TECHNICAL", "TECHNICAL_AVAILABLE"),
    Gate("SAFETY_AVAILABLE", ("SAFETY_NOT_AVAILABLE",), "SAFETY"),
    Gate("SAFETY_LEVEL_PASS", ("SAFETY_LEVEL_INSUFFICIENT",), "SAFETY", "SAFETY_AVAILABLE"),
    Gate("AUTHORITY_PASS", ("MINT_AUTHORITY_ACTIVE", "FREEZE_AUTHORITY_ACTIVE"), "SAFETY"),
    Gate("HOLDER_CONCENTRATION_PASS", ("HOLDER_CONCENTRATION_TOO_HIGH",), "SAFETY"),
    Gate("SOCIAL_AVAILABLE", ("SOCIAL_NOT_AVAILABLE",), "SOCIAL"),
    Gate("SOCIAL_PASS", ("SOCIAL_REQUIREMENT_FAILED",), "SOCIAL", "SOCIAL_AVAILABLE"),
    Gate("ANALYZE_PASS", ("ANALYZE_REQUIREMENT_FAILED",), "ANALYZE"),
    Gate("OTHER_RULES_PASS", (), "OTHER"),
    Gate("RANDOM_SELECTION_PASS", ("RANDOM_BASELINE_NOT_SELECTED",), "RANDOM"),
)  # fmt: skip
GROUP_TITLES = {
    "SCOUT": "Scout", "TECHNICAL": "Technical", "SAFETY": "safety", "SOCIAL": "social",
    "ANALYZE": "Analyze", "OTHER": "other rules", "RANDOM": "random selection",
}  # fmt: skip
_GATE_OF_CODE = {code: g.name for g in GATES for code in g.codes}
_CODE_RANK = {code: i for i, code in enumerate(c for g in GATES for c in g.codes)}


def gate_of(code: str) -> str:
    return _GATE_OF_CODE.get(code, "OTHER_RULES_PASS")


def primary_reason(reasons: Iterable[str]) -> str:
    """The code that makes the evaluation fail its first gate (unknown codes last)."""
    return min(reasons, key=lambda c: (_CODE_RANK.get(c, len(_CODE_RANK)), c))


def reasons_key(reasons: Iterable[str]) -> str:
    """Canonical storage form of a reason set (order-free, duplicate-free)."""
    return ",".join(sorted(set(reasons)))


def parse_reasons_key(key: str) -> tuple[str, ...]:
    return tuple(key.split(",")) if key else ()


def relevant_gates(cfg: StrategyConfig) -> set[str]:
    """The gates this strategy's configuration can fail (the rest are omitted, unless
    data shows a failure there anyway)."""
    e = cfg.entry
    allowed = set(e.allowed_missing)
    out = {"CURRENT_AND_ELIGIBLE", "MARKET_NOT_COLLAPSED"}  # the exact-pool price is required
    if set(e.allowed_stages) != set(STAGES):
        out.add("STAGE_ALLOWED")
    if e.min_scout_score > 0:
        out.add("SCORE_PASS")
    if e.min_liquidity_usd > 0 or "LIQUIDITY_NOT_AVAILABLE" not in allowed:
        out.add("LIQUIDITY_PASS")
    if "MARKET_CAP_NOT_AVAILABLE" not in allowed:
        out.add("MARKET_CAP_AVAILABLE")
    if e.max_risk_penalty is not None:
        out.add("RISK_PASS")
    if e.blocking_flag_severities:
        out.add("NO_BLOCKING_FLAG")
    if "TECHNICAL_NOT_AVAILABLE" not in allowed:
        out.add("TECHNICAL_AVAILABLE")
    if e.technical is not None:
        out |= {"TECHNICAL_SNAPSHOTS_PASS", "TECHNICAL_TREND_PASS"}
        t = e.technical
        if t.require_breakout or t.require_volume_confirmed or t.require_higher_lows:
            out.add("TECHNICAL_OTHER_REQUIREMENTS_PASS")
    if "SAFETY_NOT_AVAILABLE" not in allowed or e.required_safety != "ANY":
        out.add("SAFETY_AVAILABLE")
    if e.required_safety != "ANY":
        out.add("SAFETY_LEVEL_PASS")
    if e.block_active_authorities:
        out.add("AUTHORITY_PASS")
    if e.max_holder_top10_pct is not None:
        out.add("HOLDER_CONCENTRATION_PASS")
    if "SOCIAL_NOT_AVAILABLE" not in allowed:
        out.add("SOCIAL_AVAILABLE")
    if e.social is not None:
        out.add("SOCIAL_PASS")
    if e.analyze is not None:
        out.add("ANALYZE_PASS")
    if e.random_fraction is not None:
        out.add("RANDOM_SELECTION_PASS")
    return out


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def _gate_names(reasons: Iterable[str]) -> set[str]:
    return {gate_of(c) for c in reasons}


def funnel(cfg: StrategyConfig, combos: Sequence[Combo]) -> dict[str, Any]:
    """The sequential funnel and conditional counts of one strategy (exact: pure counting
    over the outcome counts)."""
    by_outcome: Counter[str] = Counter()
    blocked_by: Counter[str] = Counter()
    raw: Counter[str] = Counter()
    first_fail: Counter[str] = Counter()
    fail_codes: dict[str, Counter[str]] = {}
    seen: set[str] = set()
    for outcome, reasons, n in combos:
        by_outcome[outcome] += n
        if outcome == "BLOCKED":
            for code in reasons:
                blocked_by[code] += n
        if outcome != "REJECTED":
            continue
        for code in reasons:
            raw[code] += n
        gates = _gate_names(reasons) or {"OTHER_RULES_PASS"}
        seen |= gates
        first = next(g.name for g in GATES if g.name in gates)
        first_fail[first] += n
        codes = fail_codes.setdefault(first, Counter())
        for code in reasons:
            if gate_of(code) == first:
                codes[code] += n
    shown = [g for g in GATES if g.name in relevant_gates(cfg) | seen]
    entered, blocked = by_outcome["ENTERED"], by_outcome["BLOCKED"]
    rejected = by_outcome["REJECTED"]
    total = entered + blocked + rejected
    rows: list[dict[str, Any]] = [
        {"gate": "TOTAL_EVALUATED", "input": total, "passed": total, "failed": 0,
         "pct_of_previous": _pct(total, total), "pct_of_evaluated": _pct(total, total)},
    ]  # fmt: skip
    passed = total
    for g in shown:
        failed = first_fail[g.name]
        row = {
            "gate": g.name, "input": passed, "passed": passed - failed, "failed": failed,
            "pct_of_previous": _pct(passed - failed, passed),
            "pct_of_evaluated": _pct(passed - failed, total),
        }  # fmt: skip
        if fail_codes.get(g.name) and len(g.codes) != 1:  # several codes (or unknown ones)
            row["failed_by_reason"] = dict(sorted(fail_codes[g.name].items(),
                                                  key=lambda kv: (-kv[1], kv[0])))  # fmt: skip
        rows.append(row)
        passed -= failed
    qualified = entered + blocked
    assert passed == qualified, "funnel accounting mismatch"
    rows.append({"gate": "ENTRY_QUALIFIED", "input": qualified, "passed": qualified,
                 "failed": 0, "pct_of_previous": _pct(qualified, qualified),
                 "pct_of_evaluated": _pct(qualified, total)})  # fmt: skip
    entered_row: dict[str, Any] = {
        "gate": "ENTERED", "input": qualified, "passed": entered, "failed": blocked,
        "pct_of_previous": _pct(entered, qualified), "pct_of_evaluated": _pct(entered, total),
    }  # fmt: skip
    if blocked:
        entered_row["failed_by_reason"] = dict(sorted(blocked_by.items(),
                                                      key=lambda kv: (-kv[1], kv[0])))  # fmt: skip
    rows.append(entered_row)
    return {
        "evaluated": total,
        "entered": entered,
        "blocked": blocked,
        "rejected": rejected,
        "held": by_outcome["HELD"],
        "blocked_by": dict(sorted(blocked_by.items())),
        "reasons": dict(sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))),
        "funnel": rows,
        "conditional": _conditional(shown, combos),
    }


def _conditional(shown: Sequence[Gate], combos: Sequence[Combo]) -> list[dict[str, Any]]:
    """Per rule group: among the evaluations passing every earlier group, how many pass
    each gate of this group on its own (not sequentially), e.g. "among candidates that
    passed the Scout rules: Technical available X/Y, trend up X/Y"."""
    evaluated = [(set(r), n) for o, r, n in combos if o in ("ENTERED", "BLOCKED", "REJECTED")]
    names = {g.name for g in shown}
    groups = list(dict.fromkeys(g.group for g in shown))
    out = []
    for k, group in enumerate(groups):
        earlier = {g.name for g in shown if g.group in groups[:k]}
        base = [(gates, n) for r, n in evaluated if not ((gates := _gate_names(r)) & earlier)]
        title = ("all evaluated" if k == 0 else
                 "passing " + " + ".join(GROUP_TITLES[x] for x in groups[:k]))  # fmt: skip
        members = [g for g in shown if g.group == group]
        checks = []
        for g in members:
            pool = [(gates, n) for gates, n in base
                    if g.requires is None or g.requires not in names or g.requires not in gates]  # fmt: skip
            of = sum(n for _, n in pool)
            ok = sum(n for gates, n in pool if g.name not in gates)
            checks.append({"gate": g.name, "passed": ok, "of": of, "pct": _pct(ok, of),
                           **({"given": g.requires} if g.requires in names else {})})  # fmt: skip
        of = sum(n for _, n in base)
        group_names = {g.name for g in members}
        ok = sum(n for gates, n in base if not gates & group_names)
        out.append({
            "among": title, "base": of, "group": group,
            "all_group_rules": {"passed": ok, "of": of, "pct": _pct(ok, of)},
            "checks": checks,
        })  # fmt: skip
    return out
