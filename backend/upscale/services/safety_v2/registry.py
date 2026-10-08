"""Safety V2 known-address registry (versioned; part of the body-affecting fingerprint).

Only addresses whose role is a protocol fact are listed: never an address guessed from its
shape, its balance, or its absence from a list. Each entry carries two times:

* ``valid_since``: when the address's role objectively became true (e.g. the program's
  mainnet deployment). It never stands in for knowledge.
* ``known_since``: when *this Safety V2 ruleset* gained the classification. Entries are
  never back-dated: an entry added in a later code version gets that version's knowledge
  epoch.

An entry applies to a snapshot only when ``known_since <= as_of`` **and** ``valid_since <=
as_of``. So a build at an ``as_of`` before the ruleset knew an entry never applies it,
even if the role was already objectively valid, and knowledge added later never changes
an older ``as_of``. Any change to this file also changes the ``safety_v2_source``
fingerprint (and must bump `REGISTRY_VERSION`), so a stored snapshot is never "reproduced"
under other knowledge.

Kinds:

* ``BURN``: no private key exists; tokens held there are out of circulation.
* ``PROGRAM``: a program itself (``PROGRAM_OWNED`` with a resolved role).
* ``SHARED_POOL_AUTHORITY``: a PDA that owns the vaults of many pools of one DEX. It is a
  ``POOL_OR_VAULT`` for a target only when that target has a pinned pool of the same DEX
  (`corroborating_dex`) known at ``as_of``; otherwise its role for this token is
  unproven and it stays ``UNKNOWN`` (never globally excluded).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from upscale.services.solana_chain import (
    INCINERATOR,
    RAYDIUM_AMM_V4_AUTHORITY,
    SYSTEM_PROGRAM,
    TOKEN_2022_PROGRAM,
    TOKEN_PROGRAM,
)

REGISTRY_VERSION = "1"
RAYDIUM_AMM_V4 = "raydium-amm-v4"

RegistryKind = Literal["BURN", "PROGRAM", "SHARED_POOL_AUTHORITY"]


@dataclass(frozen=True)
class RegistryEntry:
    address: str
    kind: RegistryKind
    reason: str
    valid_since: float  # epoch seconds: the role became objectively true
    known_since: float  # epoch seconds: this ruleset gained the classification
    corroborating_dex: str | None = None  # SHARED_POOL_AUTHORITY only


def _at(day: str) -> float:
    return datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()


# The knowledge epoch of registry version 1 (Safety V2 Phase 2): every entry introduced by
# this ruleset is known from this instant, whatever its protocol history.
KNOWLEDGE_EPOCH_V1 = _at("2026-10-08")
# valid_since: conservative dates on or after each address's mainnet deployment.
_MAINNET_BETA = _at("2020-03-16")

ENTRIES: tuple[RegistryEntry, ...] = (
    RegistryEntry(INCINERATOR, "BURN", "the Solana incinerator (no private key exists)",
                  _MAINNET_BETA, KNOWLEDGE_EPOCH_V1),
    RegistryEntry(SYSTEM_PROGRAM, "PROGRAM", "the System program", _MAINNET_BETA,
                  KNOWLEDGE_EPOCH_V1),
    RegistryEntry(TOKEN_PROGRAM, "PROGRAM", "the SPL Token program", _MAINNET_BETA,
                  KNOWLEDGE_EPOCH_V1),
    RegistryEntry(TOKEN_2022_PROGRAM, "PROGRAM", "the Token-2022 program", _at("2023-01-01"),
                  KNOWLEDGE_EPOCH_V1),
    RegistryEntry(
        RAYDIUM_AMM_V4_AUTHORITY, "SHARED_POOL_AUTHORITY",
        "the Raydium AMM v4 authority (owns the vaults of every Raydium AMM v4 pool)",
        _at("2021-03-01"), KNOWLEDGE_EPOCH_V1, corroborating_dex=RAYDIUM_AMM_V4,
    ),
)  # fmt: skip
_BY_ADDRESS = {e.address: e for e in ENTRIES}
assert len(_BY_ADDRESS) == len(ENTRIES), "duplicate registry address"


def applies(entry: RegistryEntry, as_of: float) -> bool:
    """Known to this ruleset and objectively valid at `as_of`."""
    return entry.known_since <= as_of and entry.valid_since <= as_of


def lookup(
    address: str, as_of: float, entries: tuple[RegistryEntry, ...] | None = None
) -> RegistryEntry | None:
    """The entry for `address` applicable at `as_of`, or None (future knowledge is
    ignored). `entries` overrides the registry (tests)."""
    table = _BY_ADDRESS if entries is None else {e.address: e for e in entries}
    entry = table.get(address)
    return entry if entry is not None and applies(entry, as_of) else None
