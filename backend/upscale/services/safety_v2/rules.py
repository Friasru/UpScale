"""Safety V2 Phase 1 rules: deterministic, evidence-backed, three outcomes.

Every rule is decision-bearing in Phase 1 and returns TRIGGERED / NOT_TRIGGERED /
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

Token-2022 extension rules are deferred (see `models.DEFERRED_RULES`): the parsed
extension list is reported as evidence, but its risk depends on extension state that needs
interpretation (a delegate may be unset, a hook program may be null, ...).

Coverage counts applicable decision-bearing rules only: deferred / NOT_SUPPORTED rules
never make a snapshot PARTIAL. An identity that isn't VERIFIED (mint unread, unresolved or
proven not to be a mint) is INSUFFICIENT; otherwise any UNDETERMINED decision-bearing rule
is PARTIAL; COMPLETE needs every applicable decision-bearing rule evaluated.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from upscale.services.safety_v2.models import (
    SEVERITIES,
    Band,
    Coverage,
    Evidence,
    IdentityStatus,
    RuleResult,
    Severity,
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
