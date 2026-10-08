"""Safety V2 rules (RULES_VERSION 2): deterministic, evidence-backed, three outcomes.

Every rule is decision-bearing and returns TRIGGERED / NOT_TRIGGERED /
UNDETERMINED with a severity, the evidence paths it read and a deterministic reason.
NOT_TRIGGERED needs a successful read that establishes the opposite: a failed, missing,
unknown or not-collected read always leaves a rule UNDETERMINED.

| rule                     | sev.     | TRIGGERED                         | NOT_TRIGGERED            |
|--------------------------|----------|-----------------------------------|--------------------------|
| NOT_A_TOKEN_MINT         | critical | no account / not a mint account   | parsed mint              |
| UNEXPECTED_TOKEN_PROGRAM | critical | account owned by a non-token prog | token-program owner      |
| MALFORMED_MINT_ACCOUNT   | critical | token-program mint data malformed | well-formed account read |
| IDENTITY_MISMATCH        | critical | identity MISMATCH                 | identity VERIFIED        |
| MINT_AUTHORITY_ACTIVE    | high     | parsed mint, authority set        | parsed mint, null        |
| FREEZE_AUTHORITY_ACTIVE  | high     | parsed mint, authority set        | parsed mint, null        |

Phase 2 holder rules (evaluated only when a holder observation exists at or before
``as_of``; otherwise listed in ``coverage.out_of_scope``). Thresholds are frozen here
(copied from production's ``OnchainRiskConfig`` defaults, never imported); percentages are
of total supply and compared in exact integer arithmetic:

| rule                             | sev.        | TRIGGERED                    | NOT_TRIGGERED   |
|----------------------------------|-------------|------------------------------|-----------------|
| TOP1_CONCENTRATION               | med >=10 %, | top-1 share (or its lower    | complete scan,  |
|                                  | high >=20 % | bound) at/above threshold    | share below     |
| TOP10_CONCENTRATION              | med >=35 %, | top-10 share (or its lower   | complete scan,  |
|                                  | high >=60 % | bound) at/above threshold    | share below     |
| FEW_HOLDERS                      | med <100,   | complete count of meaningful | complete count  |
|                                  | high <30    | holders below threshold      | >= 100          |
| LARGE_UNKNOWN_OWNER              | medium      | an UNKNOWN / UNRESOLVED      | complete scan,  |
|                                  |             | owner holds >= 5 %           | none >= 5 %     |
| LARGE_UNCLASSIFIED_PROGRAM_OWNER | medium      | a PROGRAM_OWNED owner no     | complete scan,  |
|                                  |             | registry identifies >= 5 %   | none, all typed |

Lower bounds (PARTIAL) can prove ``value >= threshold`` only: a partial share below a
threshold is UNDETERMINED, and a partial holder count never decides FEW_HOLDERS (it is a
lower bound, so it can't show a thin base, and partial evidence never yields
NOT_TRIGGERED). ``LARGE_UNKNOWN_OWNER`` and ``LARGE_UNCLASSIFIED_PROGRAM_OWNER`` are
separate: an unknown owner isn't a program and an unidentified program isn't unknown. The
5 % large-owner line is half the TOP1 medium threshold: one owner of unproven type that
large is worth naming before it alone reaches a concentration threshold.

Token-2022 extension rules are deferred (see `models.DEFERRED_RULES`): the parsed
extension list is reported as evidence, but its risk depends on extension state that needs
interpretation (a delegate may be unset, a hook program may be null, ...).

Coverage counts applicable decision-bearing rules only: deferred / NOT_SUPPORTED rules
never make a snapshot PARTIAL. An identity that isn't VERIFIED (mint unread, unresolved or
proven not to be a mint) is INSUFFICIENT; otherwise any UNDETERMINED decision-bearing rule
is PARTIAL; COMPLETE needs every applicable decision-bearing rule evaluated.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from upscale.services.safety_v2.models import (
    SEVERITIES,
    Band,
    Coverage,
    Evidence,
    IdentityStatus,
    RuleResult,
    Severity,
    Status,
)
from upscale.services.safety_v2.provider import TOKEN_PROGRAMS

READ_NEEDED = "a successful read of the mint account"


def _rule(
    id: str,
    severity: Severity,
    evidence: Sequence[str],
    triggered: bool | None,
    reason: str,
    needs: str = READ_NEEDED,
) -> RuleResult:
    return RuleResult(
        id=id,
        outcome="UNDETERMINED"
        if triggered is None
        else "TRIGGERED"
        if triggered
        else "NOT_TRIGGERED",
        severity=severity,
        decision_bearing=True,
        evidence=tuple(evidence),
        reason=reason,
        needs=needs if triggered is None else None,
    )


def _unread(outcome: str | None, why: str | None) -> str:
    if outcome is None:
        return "no mint observation of this identity was fetched at or before as_of"
    return f"the mint account wasn't read ({outcome}: {why})"


def evaluate(
    identity_status: IdentityStatus,
    outcome: str | None,
    outcome_reason: str | None,
    program_owner: str | None,
    authority: Mapping[str, Evidence],
) -> list[RuleResult]:
    """The Phase 1 rules for one mint observation (`outcome` None: no observation)."""
    read = outcome in ("MINT", "NOT_A_MINT", "ACCOUNT_MISSING", "MALFORMED")
    unread = _unread(outcome, outcome_reason)
    obs = ("provenance.inputs.mint_account",)
    out: list[RuleResult] = []

    # NOT_A_TOKEN_MINT
    if outcome in ("ACCOUNT_MISSING", "NOT_A_MINT"):
        out.append(_rule("NOT_A_TOKEN_MINT", "critical", obs, True, str(outcome_reason)))
    elif outcome == "MINT":
        out.append(_rule("NOT_A_TOKEN_MINT", "critical", obs, False,
                         "the account parses as an initialized token mint"))  # fmt: skip
    else:
        why = f"the account's mint data couldn't be trusted ({outcome_reason})" if read else unread
        out.append(_rule("NOT_A_TOKEN_MINT", "critical", obs, None, why))

    # UNEXPECTED_TOKEN_PROGRAM
    if program_owner is not None and program_owner not in TOKEN_PROGRAMS:
        out.append(_rule("UNEXPECTED_TOKEN_PROGRAM", "critical", obs, True,
                         f"the account is owned by {program_owner}, not SPL Token or Token-2022"))  # fmt: skip
    elif program_owner is not None:
        out.append(_rule("UNEXPECTED_TOKEN_PROGRAM", "critical", obs, False,
                         f"the account is owned by the token program {program_owner}"))  # fmt: skip
    elif outcome == "ACCOUNT_MISSING":
        out.append(_rule("UNEXPECTED_TOKEN_PROGRAM", "critical", obs, None,
                         "no account exists, so it has no owner program",
                         "an existing account at the mint address"))  # fmt: skip
    else:
        why = "the account's owner program couldn't be read" if read else unread
        out.append(_rule("UNEXPECTED_TOKEN_PROGRAM", "critical", obs, None, why))

    # MALFORMED_MINT_ACCOUNT
    if outcome == "MALFORMED":
        out.append(_rule("MALFORMED_MINT_ACCOUNT", "critical", obs, True, str(outcome_reason)))
    elif read:
        out.append(_rule("MALFORMED_MINT_ACCOUNT", "critical", obs, False,
                         "the account read was well-formed"))  # fmt: skip
    else:
        out.append(_rule("MALFORMED_MINT_ACCOUNT", "critical", obs, None, unread))

    # IDENTITY_MISMATCH
    path = ("identity.token_mint",)
    if identity_status == "MISMATCH":
        out.append(_rule("IDENTITY_MISMATCH", "critical", path, True,
                         f"solana:<mint> isn't a token mint: {outcome_reason}"))  # fmt: skip
    elif identity_status == "VERIFIED":
        out.append(_rule("IDENTITY_MISMATCH", "critical", path, False,
                         "the exact address is an initialized token mint"))  # fmt: skip
    else:
        out.append(_rule("IDENTITY_MISMATCH", "critical", path, None,
                         f"the identity is unverified: {unread if not read else outcome_reason}"))  # fmt: skip

    # Authorities: only a parsed mint (AVAILABLE) decides them, either way.
    for rule_id, key, label in (
        ("MINT_AUTHORITY_ACTIVE", "mint_authority", "mint"),
        ("FREEZE_AUTHORITY_ACTIVE", "freeze_authority", "freeze"),
    ):
        field = authority[key]
        evidence = (f"authority.{key}",)
        if field.status == "AVAILABLE":
            address = field.value["address"]
            if field.value["active"]:
                out.append(_rule(rule_id, "high", evidence, True,
                                 f"the {label} authority is set ({address})"))  # fmt: skip
            else:
                out.append(_rule(rule_id, "high", evidence, False,
                                 f"the parsed mint has no {label} authority (revoked)"))  # fmt: skip
        else:
            out.append(_rule(rule_id, "high", evidence, None,
                             f"{field.status}: {field.reason}",
                             "a successfully parsed mint account"))  # fmt: skip
    return sorted(out, key=lambda r: r.id)


def assess(
    results: Sequence[RuleResult], identity_status: IdentityStatus, identity_reason: str
) -> tuple[dict[str, Any], Coverage, list[str]]:
    """(assessment, coverage, coverage reasons). The band describes triggered evidence only:
    it's never a verdict, and coverage is always reported next to it."""
    decision = [r for r in results if r.decision_bearing]
    triggered = [r for r in decision if r.outcome == "TRIGGERED"]
    undetermined = [r for r in decision if r.outcome == "UNDETERMINED"]
    reasons: list[str] = []
    coverage: Coverage
    if identity_status != "VERIFIED":
        coverage = "INSUFFICIENT"
        reasons.append(
            f"the exact token identity isn't verified ({identity_status}): {identity_reason}"
        )
    elif undetermined:
        coverage = "PARTIAL"
    else:
        coverage = "COMPLETE"
    reasons += [f"{r.id} is undetermined: {r.reason}" for r in undetermined]
    band: Band = (
        "CRITICAL_EVIDENCE"
        if any(r.severity == "critical" for r in triggered)
        else "ELEVATED_EVIDENCE"
        if triggered
        else "NO_TRIGGERED_FLAGS"
    )
    assessment = {
        "band": band,
        "coverage": coverage,
        "triggered": {s: sum(r.severity == s for r in triggered) for s in SEVERITIES},
        "not_triggered_count": sum(r.outcome == "NOT_TRIGGERED" for r in decision),
        "undetermined_count": len(undetermined),
        "note": "the band summarizes triggered evidence only; read it with coverage",
    }
    return assessment, coverage, reasons


# --- holder rules (Phase 2) ----------------------------------------------------------------

TOP1_MEDIUM_PCT, TOP1_HIGH_PCT = 10, 20
TOP10_MEDIUM_PCT, TOP10_HIGH_PCT = 35, 60
# A meaningful holder holds at least this fraction of total supply (numerator, denominator;
# compared exactly). FEW_HOLDERS reads meaningful_holder_count only, never holder_count, so
# dust / airdrop owners can't make a thin holder base look broad.
MEANINGFUL_HOLDER_FRACTION = (1, 1_000_000)
FEW_HOLDERS_MEDIUM, FEW_HOLDERS_HIGH = 100, 30  # meaningful holders
LARGE_OWNER_PCT = 5
HOLDER_RULE_IDS = ("FEW_HOLDERS", "LARGE_UNCLASSIFIED_PROGRAM_OWNER", "LARGE_UNKNOWN_OWNER",
                   "TOP10_CONCENTRATION", "TOP1_CONCENTRATION")  # fmt: skip
HOLDERS_NEEDED = "a complete holder scan (every token-account page read, owners resolved)"


@dataclass(frozen=True)
class HolderFacts:
    """What the holder rules read. `status` is AVAILABLE (complete), PARTIAL (balances are
    lower bounds) or why there are no usable balances; amounts are raw base units."""

    status: Status
    reason: str | None
    supply: int = 0
    top1_amount: int = 0
    top10_amount: int = 0
    meaningful_count: Evidence | None = None
    unknown_large: tuple[str, ...] = ()
    unidentified_program_large: tuple[str, ...] = ()
    unresolved_large: tuple[str, ...] = ()  # large owners whose type couldn't be read


def _at_least(amount: int, supply: int, pct: int) -> bool:
    return amount * 100 >= pct * supply


def _share_rule(
    id: str, facts: HolderFacts, amount: int, medium: int, high: int, label: str
) -> RuleResult:
    path = (f"holders.{'top1_pct' if id == 'TOP1_CONCENTRATION' else 'top10_pct'}",)
    if facts.status not in ("AVAILABLE", "PARTIAL"):
        return _rule(id, "medium", path, None, f"{facts.status}: {facts.reason}", HOLDERS_NEEDED)
    pct = round(100 * amount / facts.supply, 6)
    bound = "" if facts.status == "AVAILABLE" else "at least "
    if _at_least(amount, facts.supply, medium):
        sev: Severity = "high" if _at_least(amount, facts.supply, high) else "medium"
        line = high if sev == "high" else medium
        return _rule(id, sev, path, True,
                     f"{label} hold {bound}{pct}% of supply (>= {line}%)")  # fmt: skip
    if facts.status == "AVAILABLE":
        return _rule(id, "medium", path, False,
                     f"{label} hold {pct}% of supply (< {medium}%), complete scan")  # fmt: skip
    return _rule(id, "medium", path, None,
                 f"{label} hold at least {pct}% of supply; a lower bound below {medium}% "
                 "can't show the share is below it", HOLDERS_NEEDED)  # fmt: skip


def _few_holders(facts: HolderFacts) -> RuleResult:
    path = ("holders.meaningful_holder_count",)
    field = facts.meaningful_count
    if field is None or field.status != "AVAILABLE":
        why = (
            f"{facts.status}: {facts.reason}" if field is None
            else f"the meaningful-holder count is {field.status}: {field.reason}"
        )  # fmt: skip
        return _rule("FEW_HOLDERS", "medium", path, None, why, HOLDERS_NEEDED)
    n = int(field.value)
    if n < FEW_HOLDERS_MEDIUM:
        sev: Severity = "high" if n < FEW_HOLDERS_HIGH else "medium"
        line = FEW_HOLDERS_HIGH if sev == "high" else FEW_HOLDERS_MEDIUM
        return _rule("FEW_HOLDERS", sev, path, True,
                     f"{n} meaningful holders (>= {_fraction()} of supply each) < {line}, "
                     "complete scan")  # fmt: skip
    return _rule("FEW_HOLDERS", "medium", path, False,
                 f"{n} meaningful holders (>= {_fraction()} of supply each) >= "
                 f"{FEW_HOLDERS_MEDIUM}, complete scan")  # fmt: skip


def _fraction() -> str:
    num, den = MEANINGFUL_HOLDER_FRACTION
    return f"{num}/{den}"


def is_meaningful(amount: int, supply: int) -> bool:
    num, den = MEANINGFUL_HOLDER_FRACTION
    return amount * den >= num * supply


def _large_rule(
    id: str, facts: HolderFacts, hits: tuple[str, ...], open_: tuple[str, ...], what: str
) -> RuleResult:
    path = ("holders.unknown_large_owners" if id == "LARGE_UNKNOWN_OWNER"
            else "holders.program_large_owners",)  # fmt: skip
    if facts.status not in ("AVAILABLE", "PARTIAL"):
        return _rule(id, "medium", path, None, f"{facts.status}: {facts.reason}", HOLDERS_NEEDED)
    if hits:
        return _rule(id, "medium", path, True,
                     f"{what} holding at least {LARGE_OWNER_PCT}% of supply each: {len(hits)}")  # fmt: skip
    if facts.status == "PARTIAL":
        return _rule(id, "medium", path, None,
                     f"no {what} at {LARGE_OWNER_PCT}% among the owners read, but the holder "
                     "evidence is partial", HOLDERS_NEEDED)  # fmt: skip
    if open_:
        return _rule(id, "medium", path, None,
                     f"large owners whose own account couldn't be read: {len(open_)}",
                     "the large owners' own accounts")  # fmt: skip
    return _rule(id, "medium", path, False,
                 f"no {what} holds {LARGE_OWNER_PCT}% of supply, complete scan")  # fmt: skip


def evaluate_holders(facts: HolderFacts) -> list[RuleResult]:
    """The Phase 2 holder rules (only for a snapshot with a holder observation)."""
    return [
        _share_rule("TOP1_CONCENTRATION", facts, facts.top1_amount, TOP1_MEDIUM_PCT,
                    TOP1_HIGH_PCT, "the largest non-pool, non-burn owner"),
        _share_rule("TOP10_CONCENTRATION", facts, facts.top10_amount, TOP10_MEDIUM_PCT,
                    TOP10_HIGH_PCT, "the 10 largest non-pool, non-burn owners"),
        _few_holders(facts),
        _large_rule("LARGE_UNKNOWN_OWNER", facts, facts.unknown_large, (),
                    "owners of unproven type"),
        _large_rule("LARGE_UNCLASSIFIED_PROGRAM_OWNER", facts,
                    facts.unidentified_program_large, facts.unresolved_large,
                    "program-owned owners without an identified role"),
    ]  # fmt: skip
