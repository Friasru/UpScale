"""Safety V2 models: field statuses, honest evidence fields, rule results, identity, errors.

Field statuses (never silently turned into zero / false / "revoked"):

* ``AVAILABLE``: evidence exists and the field is fully established.
* ``PARTIAL``: evidence exists but is incomplete; a numeric value is a lower bound
  (``lower_bound`` is then required).
* ``UNKNOWN``: evidence exists, but its classification / meaning can't be resolved (e.g. a
  token-program account whose mint data is malformed).
* ``UNAVAILABLE``: the evidence needed to determine the field doesn't exist in the stored
  inputs (e.g. no mint observation at or before ``as_of``, or the account isn't a mint).
* ``PROVIDER_UNAVAILABLE``: a provider failure prevented collection.
* ``NOT_COLLECTED``: collection was deliberately not attempted (no provider configured,
  budget spent, cooling down).
* ``NOT_SUPPORTED``: Safety V2 doesn't derive this field (yet).

Only ``AVAILABLE`` and ``PARTIAL`` carry a value.

Rule outcomes are ``TRIGGERED`` / ``NOT_TRIGGERED`` / ``UNDETERMINED``. Evidence that is
missing, partial, unknown or failed can prove a ``>= threshold`` rule true (a lower bound
already past the threshold) but never false: it can't produce ``NOT_TRIGGERED``.

Holder evidence (Phase 2): owner types are never assumed (see `OwnerClass`), a page-capped
or largest-accounts-only holder read is PARTIAL with ``lower_bound`` set, and partial
holder evidence can trigger a ``>= threshold`` rule but never yields NOT_TRIGGERED.
``LARGE_UNKNOWN_OWNER`` (owner type can't be proven) and
``LARGE_UNCLASSIFIED_PROGRAM_OWNER`` (a program-owned account no registry entry identifies)
are separate conditions and are never conflated.

Market evidence (Phase 3): ``MARKET_CLOSED_ON_CHAIN`` needs an exact pool that was
previously observed and a *successful* ``getAccountInfo(pool)`` returning no account; a
provider failure or a DEX miss never means closed. ``MARKET_NOT_REPORTED`` needs repeated
successful misses of the same pool *and* a minimum elapsed duration: two immediate
consecutive polls never trigger it. Liquidity, volume and price are PROVIDER_REPORTED,
never presented as verified on-chain.

Invariants reserved for later phases (not implemented yet; see `DEFERRED_RULES`):

* Holders: a verified deployer *absent* from a PARTIAL holder scan is not 0% (the field is
  UNAVAILABLE); a verified deployer *observed* in a partial scan is a PARTIAL lower bound.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from upscale.services.chains import SOLANA, is_evm_address, is_solana_address
from upscale.services.market_data import InvalidRequestError

Status = Literal[
    "AVAILABLE",
    "PARTIAL",
    "UNKNOWN",
    "UNAVAILABLE",
    "PROVIDER_UNAVAILABLE",
    "NOT_COLLECTED",
    "NOT_SUPPORTED",
]
VALUED: frozenset[str] = frozenset({"AVAILABLE", "PARTIAL"})

Outcome = Literal["TRIGGERED", "NOT_TRIGGERED", "UNDETERMINED"]
Severity = Literal["critical", "high", "medium", "low"]
SEVERITIES: tuple[Severity, ...] = ("critical", "high", "medium", "low")
IdentityStatus = Literal["VERIFIED", "MISMATCH", "UNVERIFIED"]
Band = Literal["CRITICAL_EVIDENCE", "ELEVATED_EVIDENCE", "NO_TRIGGERED_FLAGS"]
Coverage = Literal["COMPLETE", "PARTIAL", "INSUFFICIENT"]

# What one getAccountInfo(mint) observation established.
MintOutcome = Literal[
    "MINT",  # a parsed, initialized mint of the SPL Token or Token-2022 program
    "NOT_A_MINT",  # an account exists but isn't a token mint (other program / other type)
    "ACCOUNT_MISSING",  # a successful read: no account exists at the address
    "MALFORMED",  # the account looks like a token mint but its state can't be trusted
    "PROVIDER_FAILED",  # the read failed (no evidence about the account)
    "NOT_COLLECTED",  # the read wasn't attempted (no provider, budget spent, cooldown)
]
MINT_OUTCOMES: tuple[MintOutcome, ...] = (
    "MINT", "NOT_A_MINT", "ACCOUNT_MISSING", "MALFORMED", "PROVIDER_FAILED", "NOT_COLLECTED",
)  # fmt: skip

# What one holder collection established (balances exist only for COLLECTED).
HolderOutcome = Literal["COLLECTED", "PROVIDER_FAILED", "NOT_COLLECTED"]
HOLDER_OUTCOMES: tuple[HolderOutcome, ...] = ("COLLECTED", "PROVIDER_FAILED", "NOT_COLLECTED")
# full_scan: every DAS page read and consistent with the largest accounts; partial_scan:
# stopped at the page cap (or inconsistent); largest_accounts: no scan (<= 20 accounts).
HolderSource = Literal["full_scan", "partial_scan", "largest_accounts"]
HOLDER_SOURCES: tuple[HolderSource, ...] = ("full_scan", "partial_scan", "largest_accounts")
# What the owner's own account looked like (one getMultipleAccounts at collection time).
OwnerLookup = Literal["FOUND", "MISSING", "NOT_LOOKED_UP"]
OWNER_LOOKUPS: tuple[OwnerLookup, ...] = ("FOUND", "MISSING", "NOT_LOOKED_UP")

# An owner's type as of a snapshot, from positive evidence only:
# * NORMAL_WALLET: proven a keypair (it signed a transaction; Radar evidence <= as_of).
# * PROGRAM_OWNED: the owner's account is owned by a program other than System, or the
#   registry names it a program.
# * POOL_OR_VAULT: the target's pinned pool, or a registry pool authority corroborated by a
#   pinned pool of that DEX.
# * BURN: a registry burn address (the incinerator).
# * UNRESOLVED: the owner's address or account couldn't be read, or evidence conflicts.
# * UNKNOWN: read, but nothing proves what it is. Never treated as a wallet.
OwnerClass = Literal[
    "NORMAL_WALLET", "PROGRAM_OWNED", "POOL_OR_VAULT", "BURN", "UNRESOLVED", "UNKNOWN"
]
OWNER_CLASSES: tuple[OwnerClass, ...] = (
    "NORMAL_WALLET", "PROGRAM_OWNED", "POOL_OR_VAULT", "BURN", "UNRESOLVED", "UNKNOWN",
)  # fmt: skip
EXCLUDED_CLASSES: frozenset[str] = frozenset({"POOL_OR_VAULT", "BURN"})

# What one market (DEX provider) observation established. POOLS: a successful response with
# at least one exact-mint (base) pool; NO_POOLS: successful, none for this exact mint.
MarketOutcome = Literal["POOLS", "NO_POOLS", "PROVIDER_FAILED", "NOT_COLLECTED"]
MARKET_OUTCOMES: tuple[MarketOutcome, ...] = ("POOLS", "NO_POOLS", "PROVIDER_FAILED",
                                              "NOT_COLLECTED")  # fmt: skip
# How a returned pool relates to the exact target identity. Only EXACT_BASE pools can be
# the market; the others are kept for audit with their reason.
PoolIdentity = Literal["EXACT_BASE", "QUOTE_SIDE", "OTHER_MINT", "OTHER_CHAIN", "MALFORMED"]
POOL_IDENTITIES: tuple[PoolIdentity, ...] = ("EXACT_BASE", "QUOTE_SIDE", "OTHER_MINT",
                                             "OTHER_CHAIN", "MALFORMED")  # fmt: skip
# One getAccountInfo(pool): exact on-chain presence evidence (never inferred from a DEX).
PoolAccountOutcome = Literal["EXISTS", "ACCOUNT_MISSING", "PROVIDER_FAILED", "NOT_COLLECTED"]
POOL_ACCOUNT_OUTCOMES: tuple[PoolAccountOutcome, ...] = (
    "EXISTS", "ACCOUNT_MISSING", "PROVIDER_FAILED", "NOT_COLLECTED",
)  # fmt: skip
# The tracked pool's presence as of a snapshot. NOT_REPORTED is a DEX-provider miss and
# never means closed; CLOSED_ON_CHAIN needs a successful getAccountInfo(pool) returning no
# account for a pool Safety V2 previously observed.
PoolPresence = Literal["REPORTED", "NOT_REPORTED", "CLOSED_ON_CHAIN", "UNAVAILABLE",
                       "NOT_COLLECTED"]  # fmt: skip

# Rules Safety V2 deliberately doesn't evaluate yet. They are reported NOT_SUPPORTED and never
# count toward coverage (an optional, unsupported rule can't make every snapshot PARTIAL).
DEFERRED_RULES: dict[str, str] = {
    "TOKEN_2022_EXTENSION_RISK": (
        "Token-2022 extension semantics (delegates, hooks, fees, pause, default state) need "
        "interpretation beyond the parsed extension list; extensions are reported as evidence"
    ),
    "VERIFIED_DEPLOYER_HOLDS_SUPPLY": "creator / deployer evidence is a later phase",
}


class SafetyError(Exception):
    pass


class SafetyCausalityError(SafetyError):
    """An input was fetched after the snapshot's ``as_of`` (lookahead)."""


class SafetySchemaError(SafetyError):
    """The database was created by another (or an unknown) Safety V2 schema version."""


class SafetyStateError(SafetyError):
    """A stored-state invariant would be broken (e.g. finishing a finished collection)."""


class SafetyIdentityError(SafetyError):
    """An input doesn't belong to the exact identity being built."""


class SafetyProviderError(Exception):
    """A provider call failed or wasn't attempted; `status` says which."""

    status: Status = "PROVIDER_UNAVAILABLE"


class SafetyNotCollectedError(SafetyProviderError):
    status: Status = "NOT_COLLECTED"


class SafetyBudgetExhaustedError(SafetyNotCollectedError):
    pass


class SafetyCoolingDownError(SafetyNotCollectedError):
    pass


class SafetyRateLimitedError(SafetyProviderError):
    pass


class SafetyTimeoutError(SafetyProviderError):
    pass


class Evidence(BaseModel):
    """One evidence field. The validator makes dishonest combinations unrepresentable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Status
    value: Any = None
    lower_bound: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def _honest(self) -> "Evidence":
        if self.status in VALUED and self.value is None:
            raise ValueError(f"a {self.status} field needs a value")
        if self.status not in VALUED and self.value is not None:
            raise ValueError(f"a {self.status} field can't carry a value (never zero-filled)")
        if self.lower_bound and self.status != "PARTIAL":
            raise ValueError("only a PARTIAL field can be a lower bound")
        if self.status == "PARTIAL":
            numeric = isinstance(self.value, int | float) and not isinstance(self.value, bool)
            if numeric and not self.lower_bound:
                raise ValueError("a PARTIAL numeric field is a lower bound")
            if not self.reason:
                raise ValueError("a PARTIAL field needs a reason")
        if self.status != "AVAILABLE" and not self.reason:
            raise ValueError(f"a {self.status} field needs a reason")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def available(value: Any) -> Evidence:
    return Evidence(status="AVAILABLE", value=value)


def missing(status: Status, reason: str) -> Evidence:
    if status in VALUED:
        raise ValueError(f"{status} isn't a missing-evidence status")
    return Evidence(status=status, reason=reason)


class RuleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    outcome: Outcome
    severity: Severity
    decision_bearing: bool
    evidence: tuple[str, ...]  # paths into the snapshot body
    reason: str
    needs: str | None = None  # UNDETERMINED only: what evidence would decide it

    @model_validator(mode="after")
    def _needs(self) -> "RuleResult":
        if (self.outcome == "UNDETERMINED") != (self.needs is not None):
            raise ValueError("exactly an UNDETERMINED rule says what it needs")
        return self


def solana_identity(raw: str) -> tuple[str, str]:
    """(``solana:<mint>``, mint) from ``solana:<mint>`` or a bare mint.

    Solana only; base58 is case-sensitive and never normalized. A parseable address is only
    a well-formed identity: whether it is a token mint is established by evidence."""
    value = raw.strip()
    chain, sep, address = value.partition(":")
    if not sep:
        chain, address = SOLANA, value
    if chain != SOLANA:
        raise InvalidRequestError(f"Safety V2 supports Solana only (got chain {chain!r})")
    if is_evm_address(address):
        raise InvalidRequestError(f"{address!r} is an EVM address, not a Solana mint")
    if address != address.strip() or not is_solana_address(address):
        raise InvalidRequestError(f"{address!r} is not a valid Solana address")
    return f"{SOLANA}:{address}", address


def ts(value: datetime) -> float:
    if value.tzinfo is None:
        raise ValueError("Safety V2 timestamps must be timezone-aware")
    return value.timestamp()


def iso(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, UTC).isoformat() if value is not None else None
