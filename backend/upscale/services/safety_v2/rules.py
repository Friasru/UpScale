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

Phase 3 market rules (evaluated only when a market observation exists at or before
``as_of``; otherwise listed in ``coverage.out_of_scope``). Liquidity and age are
PROVIDER_REPORTED; thresholds are frozen here:

| rule                   | sev.          | TRIGGERED                       | NOT_TRIGGERED          |
|------------------------|---------------|---------------------------------|------------------------|
| LOW_LIQUIDITY          | high <$25k,   | tracked pool's liquidity below  | >= $100k               |
|                        | med <$100k    | (DexRiskConfig)                 |                        |
| LIQUIDITY_COLLAPSE     | high          | same pool: now <= 20 % of its   | 2+ same-pool points,   |
|                        |               | 24 h high (CollapseConfig 80 %) | drop < 80 %            |
| MARKET_CLOSED_ON_CHAIN | high          | corroborated pool missing from  | its account exists     |
|                        |               | DEX, getAccountInfo: none       | on-chain               |
| MARKET_NOT_REPORTED    | medium        | >= 2 successful misses of the   | the pool is reported   |
|                        |               | same pool spanning >= 6 h       |                        |
| NO_ELIGIBLE_MARKET     | high          | no eligible exact-mint pool     | an eligible pool       |
| PRIMARY_MARKET_UNCLEAR | medium        | selection ambiguous / >1 pin    | clear, pinned, or none |
| VERY_NEW_POOL          | high <24 h,   | provider pair_created_at that   | >= 72 h                |
|                        | med <72 h     | recent (DexRiskConfig)          |                        |

A failed or missing market read never decides a rule NOT_TRIGGERED, and a DEX miss alone
never means closed. MARKET_CLOSED_ON_CHAIN is applicable only when a corroborated pool
(reported by an earlier successful observation) is missing from the latest successful
one; otherwise (healthy reported market, pin never corroborated, no pool) it is out of
scope, never NOT_TRIGGERED from DEX presence.

Phase 4B rules. VERIFIED_DEPLOYER_HOLDS_SUPPLY (medium >= 5 %, high >= 10 %) applies only
when a VERIFIED token deployer is known at ``as_of``; complete holder evidence decides it
either way (absent from a complete scan = 0 %), partial evidence can only trigger it.
Deployer TOKEN_OUTFLOW events are evidence only (no defensible amount threshold exists, and
Radar observes only transactions touching the tracked pool, so zero observed outflows is
never proof of none). Holder-change rules compare the current holder observation with the
latest strictly earlier *complete* one; with no earlier holder observation they are out of
scope, and with no complete pair they are UNDETERMINED:

* CONCENTRATION_RISING (medium): top-10 share up >= 10 percentage points.
* RAPID_HOLDER_LOSS (medium): **meaningful** holders down >= 30 % within 7 days (out of
  scope for a pair further apart or a previous meaningful count below 30).
* LARGE_HOLDER_EXIT (medium): an owner with >= 5 % in the previous complete scan is absent
  (zero balance) from the current complete scan; a reduction is never an exit. Proven
  pool / burn owners (classified as of each scan) never count; UNKNOWN and PROGRAM_OWNED
  owners do (no wallet proof is required).

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
from fractions import Fraction
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


# --- market rules (Phase 3) ----------------------------------------------------------------

# Copied from production's `DexRiskConfig` defaults (same facts), frozen here.
LIQUIDITY_HIGH_USD, LIQUIDITY_MEDIUM_USD = 25_000, 100_000  # tracked pool below: high / med
POOL_AGE_HIGH_HOURS, POOL_AGE_MEDIUM_HOURS = 24, 72  # tracked pool younger: high / medium
# Copied from production's outcomes `CollapseConfig.liquidity_drop_pct`.
LIQUIDITY_COLLAPSE_DROP_PCT = 80
LIQUIDITY_WINDOW_S = 24 * 3600
# No production equivalent: chosen for Safety V2. A DEX index commonly drops a pool for
# minutes (re-indexing, outages); a pool still missing across successful reads spanning six
# hours is a material, persistent disappearance. Two immediate polls never qualify.
MARKET_NOT_REPORTED_MIN_MISSES = 2
MARKET_NOT_REPORTED_MIN_DURATION_S = 6 * 3600
MARKET_RULE_IDS = ("LIQUIDITY_COLLAPSE", "LOW_LIQUIDITY", "MARKET_CLOSED_ON_CHAIN",
                   "MARKET_NOT_REPORTED", "NO_ELIGIBLE_MARKET", "PRIMARY_MARKET_UNCLEAR",
                   "VERY_NEW_POOL")  # fmt: skip
MARKET_NEEDED = "a successful market observation reporting the tracked exact pool"


@dataclass(frozen=True)
class MarketFacts:
    """What the market rules read (see `features.market_section`)."""

    status: Status  # AVAILABLE when the latest market observation succeeded
    reason: str | None
    presence: str = "UNAVAILABLE"
    presence_reason: str = ""
    account_outcome: str | None = None  # the tracked pool's getAccountInfo, same collection
    anchor: str | None = None
    liquidity_usd: float | None = None  # the tracked pool, PROVIDER_REPORTED
    liquidity_reason: str = ""
    window_max_usd: float | None = None  # max same-pool liquidity in the preceding 24 h
    change_reason: str = ""
    age_s: float | None = None  # as_of - provider pair_created_at (never negative)
    age_reason: str = ""
    any_eligible: bool | None = None
    pinned: int = 0
    has_primary: bool = False
    clear: bool | None = None
    ambiguity: tuple[str, ...] = ()
    misses: int = 0
    miss_span_s: float = 0.0
    closure_applicable: bool = False
    closure_reason: str = ""


def _ok(facts: MarketFacts) -> bool:
    return facts.status == "AVAILABLE"


def _failed(id: str, sev: Severity, path: str, facts: MarketFacts) -> RuleResult:
    return _rule(id, sev, (path,), None, f"{facts.status}: {facts.reason}", MARKET_NEEDED)


def evaluate_market(facts: MarketFacts) -> list[RuleResult]:
    """The Phase 3 market rules (only for a snapshot with a market observation)."""
    out: list[RuleResult] = []

    # LOW_LIQUIDITY (strictly below a threshold triggers)
    path = "market.primary_liquidity_usd"
    if facts.liquidity_usd is None:
        why = f"{facts.status}: {facts.reason}" if not _ok(facts) else facts.liquidity_reason
        out.append(_rule("LOW_LIQUIDITY", "medium", (path,), None, why, MARKET_NEEDED))
    else:
        v = facts.liquidity_usd
        if v < LIQUIDITY_MEDIUM_USD:
            sev: Severity = "high" if v < LIQUIDITY_HIGH_USD else "medium"
            line = LIQUIDITY_HIGH_USD if sev == "high" else LIQUIDITY_MEDIUM_USD
            out.append(_rule("LOW_LIQUIDITY", sev, (path,), True,
                             f"provider-reported liquidity ${v:,.2f} is below ${line:,}"))  # fmt: skip
        else:
            out.append(_rule("LOW_LIQUIDITY", "medium", (path,), False,
                             f"provider-reported liquidity ${v:,.2f} is at least "
                             f"${LIQUIDITY_MEDIUM_USD:,}"))  # fmt: skip

    # LIQUIDITY_COLLAPSE (same pool, preceding 24 h; a drop of >= 80 % triggers)
    path = "market.liquidity_change_pct_24h"
    if facts.liquidity_usd is None or facts.window_max_usd is None:
        why = f"{facts.status}: {facts.reason}" if not _ok(facts) else facts.change_reason
        out.append(_rule("LIQUIDITY_COLLAPSE", "high", (path,), None, why,
                         "two successful same-pool liquidity observations within 24 h"))  # fmt: skip
    else:
        now, peak = facts.liquidity_usd, facts.window_max_usd
        collapsed = now * 100 <= peak * (100 - LIQUIDITY_COLLAPSE_DROP_PCT)
        out.append(_rule("LIQUIDITY_COLLAPSE", "high", (path,), collapsed,
                         f"provider-reported liquidity ${now:,.2f} vs a 24 h same-pool high of "
                         f"${peak:,.2f} (collapse: a drop of {LIQUIDITY_COLLAPSE_DROP_PCT}% or "
                         "more)"))  # fmt: skip

    # MARKET_CLOSED_ON_CHAIN: only when applicable (a corroborated pool went missing from
    # the DEX); otherwise the caller lists it out of scope. DEX presence never decides it.
    path = "market.pool_presence"
    if facts.closure_applicable:
        if facts.presence == "CLOSED_ON_CHAIN":
            out.append(_rule("MARKET_CLOSED_ON_CHAIN", "high", (path,), True,
                             facts.presence_reason))  # fmt: skip
        elif facts.account_outcome == "EXISTS":
            out.append(_rule("MARKET_CLOSED_ON_CHAIN", "high", (path,), False,
                             "getAccountInfo found the missing pool's account on-chain"))  # fmt: skip
        else:
            why = (f"no usable on-chain check of the missing pool "
                   f"({facts.account_outcome or 'none in the latest collection'})")  # fmt: skip
            out.append(_rule("MARKET_CLOSED_ON_CHAIN", "high", (path,), None, why,
                             "a successful getAccountInfo of the missing corroborated pool"))  # fmt: skip

    # MARKET_NOT_REPORTED (repeated successful misses over a minimum duration)
    path = "market.not_reported"
    if not _ok(facts):
        out.append(_failed("MARKET_NOT_REPORTED", "medium", path, facts))
    elif facts.presence == "REPORTED":
        out.append(_rule("MARKET_NOT_REPORTED", "medium", (path,), False,
                         "the tracked exact pool is in the latest successful DEX response"))  # fmt: skip
    elif facts.anchor is None:
        out.append(_rule("MARKET_NOT_REPORTED", "medium", (path,), None,
                         "no exact pool is known to track", "a previously reported or pinned pool"))  # fmt: skip
    elif (facts.misses >= MARKET_NOT_REPORTED_MIN_MISSES
          and facts.miss_span_s >= MARKET_NOT_REPORTED_MIN_DURATION_S):  # fmt: skip
        out.append(_rule("MARKET_NOT_REPORTED", "medium", (path,), True,
                         f"the tracked pool {facts.anchor} was missing from {facts.misses} "
                         f"successful DEX responses over {facts.miss_span_s / 3600:.2f} h"))  # fmt: skip
    else:
        out.append(_rule("MARKET_NOT_REPORTED", "medium", (path,), None,
                         f"{facts.misses} successful misses over {facts.miss_span_s / 3600:.2f} h "
                         f"(needs >= {MARKET_NOT_REPORTED_MIN_MISSES} over >= "
                         f"{MARKET_NOT_REPORTED_MIN_DURATION_S // 3600} h)",
                         "repeated successful DEX misses over the minimum duration"))  # fmt: skip

    # NO_ELIGIBLE_MARKET
    path = "market.alternative_eligible_pools"
    if facts.any_eligible is None:
        out.append(_failed("NO_ELIGIBLE_MARKET", "high", path, facts))
    else:
        out.append(_rule("NO_ELIGIBLE_MARKET", "high", (path,), not facts.any_eligible,
                         "an eligible exact-mint pool is reported" if facts.any_eligible
                         else "no exact-mint pool is liquid, active and priced enough to be "
                         "the market"))  # fmt: skip

    # PRIMARY_MARKET_UNCLEAR
    path = "market.primary_clear"
    if not _ok(facts):
        out.append(_failed("PRIMARY_MARKET_UNCLEAR", "medium", path, facts))
    elif facts.pinned > 1:
        out.append(_rule("PRIMARY_MARKET_UNCLEAR", "medium", (path,), True,
                         f"{facts.pinned} pools are pinned; the earliest is tracked"))  # fmt: skip
    elif facts.pinned == 1:
        out.append(_rule("PRIMARY_MARKET_UNCLEAR", "medium", (path,), False,
                         "the market is anchored to the target's pinned pool"))  # fmt: skip
    elif not facts.has_primary:
        out.append(_rule("PRIMARY_MARKET_UNCLEAR", "medium", (path,), False,
                         "no eligible exact-mint pool competes to be the market "
                         "(see NO_ELIGIBLE_MARKET)"))  # fmt: skip
    else:
        out.append(_rule("PRIMARY_MARKET_UNCLEAR", "medium", (path,), not facts.clear,
                         "; ".join(facts.ambiguity) if not facts.clear
                         else "one eligible exact-mint pool is clearly the primary market"))  # fmt: skip

    # VERY_NEW_POOL (strictly younger than a threshold triggers)
    path = "market.pool_age_hours"
    if facts.age_s is None:
        why = f"{facts.status}: {facts.reason}" if not _ok(facts) else facts.age_reason
        out.append(_rule("VERY_NEW_POOL", "medium", (path,), None, why,
                         "a provider-reported pool creation time at or before as_of"))  # fmt: skip
    else:
        hours = facts.age_s / 3600
        if facts.age_s < POOL_AGE_MEDIUM_HOURS * 3600:
            sev = "high" if facts.age_s < POOL_AGE_HIGH_HOURS * 3600 else "medium"
            line = POOL_AGE_HIGH_HOURS if sev == "high" else POOL_AGE_MEDIUM_HOURS
            out.append(_rule("VERY_NEW_POOL", sev, (path,), True,
                             f"the tracked pool is {hours:.2f} h old (< {line} h, "
                             "provider-reported creation time)"))  # fmt: skip
        else:
            out.append(_rule("VERY_NEW_POOL", "medium", (path,), False,
                             f"the tracked pool is {hours:.2f} h old (>= "
                             f"{POOL_AGE_MEDIUM_HOURS} h, provider-reported creation time)"))  # fmt: skip
    return out


# --- creator / deployer and holder-change rules (Phase 4B) ---------------------------------

# No production equivalent exists for any of these; they are chosen for Safety V2 and frozen
# here. They reuse Safety V2's own Phase 2 lines where the fact is the same:
# * a verified deployer holding >= LARGE_OWNER_PCT (5 %) is a large owner (medium), and
#   >= TOP1_MEDIUM_PCT (10 %) is as much as the top-1 concentration line (high);
# * a "large holder" for LARGE_HOLDER_EXIT is the same LARGE_OWNER_PCT (5 %) owner.
DEPLOYER_HOLDING_MEDIUM_PCT, DEPLOYER_HOLDING_HIGH_PCT = LARGE_OWNER_PCT, TOP1_MEDIUM_PCT
# Top-10 share up by >= 10 percentage points between two complete scans: a quarter of the
# 35 % -> 60 % gap between the TOP10 medium and high lines (top10_change_pp, never %).
CONCENTRATION_RISING_PP = 10
# Meaningful holders down >= 30 % between two complete scans at most 7 days apart. A pair
# further apart isn't "rapid" (out of scope); a baseline below FEW_HOLDERS_HIGH (30) is
# too small to read a percentage from (FEW_HOLDERS covers thin holder bases).
RAPID_HOLDER_LOSS_PCT = 30
RAPID_HOLDER_LOSS_MAX_WINDOW_S = 7 * 24 * 3600
RAPID_HOLDER_LOSS_MIN_BASELINE = FEW_HOLDERS_HIGH
CHANGE_RULE_IDS = ("CONCENTRATION_RISING", "LARGE_HOLDER_EXIT", "RAPID_HOLDER_LOSS")
CREATOR_RULE_IDS = ("VERIFIED_DEPLOYER_HOLDS_SUPPLY",)
CHANGE_NEEDED = "two complete holder scans (full scan, every owner resolved)"


@dataclass(frozen=True)
class DeployerFacts:
    """A VERIFIED deployer's holding. `status`: AVAILABLE (complete scan: exact, absent =
    0), PARTIAL (observed in partial evidence: a lower bound) or why there is none."""

    status: Status
    reason: str | None
    amount: int = 0
    supply: int = 0


def evaluate_creator(facts: DeployerFacts) -> list[RuleResult]:
    path = ("creator.deployer_holding_pct",)
    if facts.status not in ("AVAILABLE", "PARTIAL"):
        return [_rule("VERIFIED_DEPLOYER_HOLDS_SUPPLY", "medium", path, None,
                      f"{facts.status}: {facts.reason}",
                      "a complete holder scan, or the deployer observed in partial evidence")]  # fmt: skip
    pct = round(100 * facts.amount / facts.supply, 6)
    bound = "" if facts.status == "AVAILABLE" else "at least "
    if _at_least(facts.amount, facts.supply, DEPLOYER_HOLDING_MEDIUM_PCT):
        high = _at_least(facts.amount, facts.supply, DEPLOYER_HOLDING_HIGH_PCT)
        line = DEPLOYER_HOLDING_HIGH_PCT if high else DEPLOYER_HOLDING_MEDIUM_PCT
        return [_rule("VERIFIED_DEPLOYER_HOLDS_SUPPLY", "high" if high else "medium", path, True,
                      f"the verified token deployer holds {bound}{pct}% of supply "
                      f"(>= {line}%)")]  # fmt: skip
    if facts.status == "AVAILABLE":
        return [_rule("VERIFIED_DEPLOYER_HOLDS_SUPPLY", "medium", path, False,
                      f"the verified token deployer holds {pct}% of supply "
                      f"(< {DEPLOYER_HOLDING_MEDIUM_PCT}%), complete scan")]  # fmt: skip
    return [_rule("VERIFIED_DEPLOYER_HOLDS_SUPPLY", "medium", path, None,
                  f"the verified token deployer holds at least {pct}%; a lower bound below "
                  f"{DEPLOYER_HOLDING_MEDIUM_PCT}% can't show the holding is below it",
                  "a complete holder scan")]  # fmt: skip


@dataclass(frozen=True)
class ChangeFacts:
    """An exact comparison of two complete holder scans (`exact`), or why there is none.
    Raw integers: top-10 amounts and supplies, meaningful holder counts."""

    exact: bool
    reason: str | None
    elapsed_s: float = 0.0
    prev_top10: int = 0
    prev_supply: int = 1
    cur_top10: int = 0
    cur_supply: int = 1
    prev_meaningful: int = 0
    cur_meaningful: int = 0
    exits: int = 0


def top10_change_pp(f: ChangeFacts) -> Fraction:
    return 100 * (Fraction(f.cur_top10, f.cur_supply) - Fraction(f.prev_top10, f.prev_supply))


def holder_loss_applicable(f: ChangeFacts) -> tuple[bool, str]:
    """RAPID_HOLDER_LOSS is out of scope for a pair too far apart or a tiny baseline."""
    if not f.exact:
        return True, ""
    if f.elapsed_s > RAPID_HOLDER_LOSS_MAX_WINDOW_S:
        return False, (f"the two complete scans are {f.elapsed_s / 3600:.1f} h apart, more than "
                       f"{RAPID_HOLDER_LOSS_MAX_WINDOW_S // 3600} h: not a rapid-change window")  # fmt: skip
    if f.prev_meaningful < RAPID_HOLDER_LOSS_MIN_BASELINE:
        return False, (f"the previous scan had {f.prev_meaningful} meaningful holders, below the "
                       f"{RAPID_HOLDER_LOSS_MIN_BASELINE} baseline a percentage needs "
                       "(FEW_HOLDERS covers thin holder bases)")  # fmt: skip
    return True, ""


def evaluate_changes(f: ChangeFacts) -> list[RuleResult]:
    """Holder-change rules (only when a prior holder observation exists)."""
    out: list[RuleResult] = []
    if not f.exact:
        why = str(f.reason)
        out.append(_rule("CONCENTRATION_RISING", "medium", ("changes.top10_change_pp",), None,
                         why, CHANGE_NEEDED))  # fmt: skip
        out.append(_rule("LARGE_HOLDER_EXIT", "medium", ("changes.large_holder_exits",), None,
                         why, CHANGE_NEEDED))  # fmt: skip
        out.append(_rule("RAPID_HOLDER_LOSS", "medium",
                         ("changes.meaningful_holder_count_change_pct",), None, why,
                         CHANGE_NEEDED))  # fmt: skip
        return out
    pp = top10_change_pp(f)
    out.append(_rule("CONCENTRATION_RISING", "medium", ("changes.top10_change_pp",),
                     pp >= CONCENTRATION_RISING_PP,
                     f"the top-10 share changed by {float(pp):+.4f} percentage points between two "
                     f"complete scans (rising: >= +{CONCENTRATION_RISING_PP} pp)"))  # fmt: skip
    out.append(_rule("LARGE_HOLDER_EXIT", "medium", ("changes.large_holder_exits",),
                     f.exits > 0,
                     f"owners holding >= {LARGE_OWNER_PCT}% in the previous complete scan that "
                     f"are absent (zero balance) from the current complete scan: {f.exits}"
                     if f.exits else f"no owner holding >= {LARGE_OWNER_PCT}% in the previous "
                     "complete scan is absent from the current one"))  # fmt: skip
    applicable, _ = holder_loss_applicable(f)
    if applicable:
        lost = f.prev_meaningful - f.cur_meaningful
        rapid = lost * 100 >= RAPID_HOLDER_LOSS_PCT * f.prev_meaningful
        out.append(_rule("RAPID_HOLDER_LOSS", "medium",
                         ("changes.meaningful_holder_count_change_pct",), rapid,
                         f"meaningful holders (>= {_fraction()} of supply each) went "
                         f"{f.prev_meaningful} -> {f.cur_meaningful} in "
                         f"{f.elapsed_s / 3600:.1f} h (rapid loss: a decline of >= "
                         f"{RAPID_HOLDER_LOSS_PCT}% within "
                         f"{RAPID_HOLDER_LOSS_MAX_WINDOW_S // 3600} h)"))  # fmt: skip
    return out
