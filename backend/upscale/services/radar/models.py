"""Radar V1 vocabulary: availability states, flow / creator labels, identity, metrics.

Missing data is never zero: every metric carries a status, and its value is None unless
the status is ``AVAILABLE`` or ``PARTIAL`` (``PARTIAL`` values are lower bounds of what a
complete scan would show, flagged ``lower_bound``).

Labels are deliberately neutral. A wallet's token balance going up or down is a
``TOKEN_INFLOW`` / ``TOKEN_OUTFLOW``, never a buy or a sale; the earliest pool
transaction's fee payer is a ``POOL_CREATOR_CANDIDATE``, never the token deployer; a
``TOKEN_DEPLOYER`` is only recorded from a verified mint-initialization transaction.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from upscale.services.chains import SOLANA, is_solana_address
from upscale.services.market_data import InvalidRequestError, MarketDataUnavailableError
from upscale.services.scout.normalize import canonical_id

Status = Literal[
    "AVAILABLE",
    "PARTIAL",  # a lower bound: the scan behind it hit a cap or a gap
    "UNAVAILABLE",  # the data can't be established (e.g. history cap reached)
    "NOT_SUPPORTED",  # V1 can't measure this honestly at all
    "PROVIDER_UNAVAILABLE",  # provider failed, timed out, rate-limited or isn't configured
    "NOT_COLLECTED",  # not attempted (feature off, budget exhausted, no baseline yet)
]
VALUED: frozenset[str] = frozenset({"AVAILABLE", "PARTIAL"})

FlowDirection = Literal["TOKEN_INFLOW", "TOKEN_OUTFLOW"]
# How sure Radar is about the other side of a wallet's flow. "TRACKED_POOL_COUNTERPARTY":
# the tracked pool's own token accounts moved by exactly the opposite amount in the same
# successful transaction, signed by the wallet, with no other participant's balance
# changing. "NOT_SUPPORTED": the tracked pool's vaults are held by a shared authority
# (Raydium AMM v4), so its side of the transaction can't be attributed. Neither is ever
# called a buy or sell.
Counterparty = Literal["TRACKED_POOL_COUNTERPARTY", "UNVERIFIED", "NOT_SUPPORTED"]
# Who a balance-change owner is, only from positive evidence (see `parsing.classify`).
# Only NORMAL_WALLET identities feed wallet statistics; UNKNOWN is never assumed a wallet.
Participant = Literal[
    "NORMAL_WALLET",  # signed a transaction: a keypair, so not a program or PDA
    "PROGRAM",  # a known program id, or a program invoked by the transaction itself
    "POOL_OR_VAULT",  # the tracked pool, another Radar target's pool, a known vault authority
    "ROUTER_OR_INTERMEDIARY",  # a known aggregator / router program or configured address
    "UNKNOWN",  # none of the above could be proven
]
PARTICIPANTS: tuple[Participant, ...] = (
    "NORMAL_WALLET", "PROGRAM", "POOL_OR_VAULT", "ROUTER_OR_INTERMEDIARY", "UNKNOWN",
)  # fmt: skip
CreatorRole = Literal["POOL_CREATOR_CANDIDATE", "TOKEN_DEPLOYER"]
CreatorStatus = Literal["CANDIDATE", "VERIFIED", "UNAVAILABLE"]


class RadarError(Exception):
    pass


class RadarCausalityError(RadarError):
    """An input was fetched after the moment a snapshot claims to describe."""


class RadarUnavailableError(MarketDataUnavailableError):
    """Radar's own guard refused or the provider failed (never production's limits)."""

    status: Status = "PROVIDER_UNAVAILABLE"


class RadarBudgetExhaustedError(RadarUnavailableError):
    status: Status = "NOT_COLLECTED"


class RadarCoolingDownError(RadarUnavailableError):
    status: Status = "PROVIDER_UNAVAILABLE"


class RadarRateLimitedError(RadarUnavailableError):
    status: Status = "PROVIDER_UNAVAILABLE"


class RadarTimeoutError(RadarUnavailableError):
    status: Status = "PROVIDER_UNAVAILABLE"


class RadarTxUnavailableError(RadarUnavailableError):
    """One transaction can't be read in a usable form (unsupported version, malformed,
    unknown to the provider) while the provider itself works: safe to skip on its own."""

    status: Status = "UNAVAILABLE"


def solana_identity(mint: str) -> tuple[str, str]:
    """(canonical id, mint) for a Solana mint. Base58 is case-sensitive: never lowercased."""
    value = mint.strip()
    if not is_solana_address(value):
        raise InvalidRequestError(f"{mint!r} is not a valid Solana address")
    return canonical_id(SOLANA, value), value


def parse_canonical(value: str) -> tuple[str, str]:
    """(chain, address) of ``solana:<mint>`` or a bare Solana mint. Other chains: V1 can't."""
    raw = value.strip()
    chain, sep, address = raw.partition(":")
    if not sep:
        chain, address = SOLANA, raw
    if chain != SOLANA:
        raise InvalidRequestError(f"Radar V1 supports Solana only (got chain {chain!r})")
    return chain, solana_identity(address)[1]


def ts(value: datetime) -> float:
    if value.tzinfo is None:
        raise ValueError("Radar timestamps must be timezone-aware")
    return value.timestamp()


def dt(value: float | None) -> datetime | None:
    return datetime.fromtimestamp(value, UTC) if value is not None else None


def iso(value: float | None) -> str | None:
    d = dt(value)
    return d.isoformat() if d else None


class Metric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Status
    value: float | int | None = None
    lower_bound: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def _honest(self) -> "Metric":
        if self.status in VALUED and self.value is None:
            raise ValueError(f"a {self.status} metric needs a value")
        if self.status not in VALUED and self.value is not None:
            raise ValueError(f"a {self.status} metric can't carry a value (never zero-filled)")
        if self.status == "PARTIAL" and not self.lower_bound:
            raise ValueError("a PARTIAL metric is a lower bound")
        return self


def available(value: float | int) -> Metric:
    return Metric(status="AVAILABLE", value=value)


def partial(value: float | int, reason: str) -> Metric:
    return Metric(status="PARTIAL", value=value, lower_bound=True, reason=reason)


def missing(status: Status, reason: str) -> Metric:
    if status in VALUED:
        raise ValueError("use available() / partial() for valued metrics")
    return Metric(status=status, reason=reason)


@dataclass(frozen=True)
class Target:
    canonical_id: str
    chain: str
    mint: str
    pool_address: str
    dex: str | None
    pool_created_at: float | None  # from Scout (provider-reported), or None
    pool_created_source: str | None
    source: str  # "scout" | "manual"
    selected_at: float
    status: str  # "ACTIVE" | "INACTIVE"
    last_signature: str | None  # newest pool signature already listed (activity cursor)
    early_status: str | None
    snapshots_taken: int
    last_scan_at: float | None = None  # last activity scan that finished (not aborted)


@dataclass(frozen=True)
class SignatureInfo:
    signature: str
    slot: int | None
    block_time: float | None
    failed: bool


@dataclass(frozen=True)
class WalletDelta:
    wallet: str
    direction: FlowDirection
    amount_raw: int  # absolute change in base units
    counterparty: Counterparty
    participant: Participant
    signer: bool


@dataclass(frozen=True)
class ParsedTx:
    signature: str
    slot: int | None
    block_time: float | None
    fee_payer: str | None
    failed: bool
    flows: tuple[WalletDelta, ...]
    initializes_mint: bool
    unattributed_balances: int  # token balance rows of the mint without an owner
