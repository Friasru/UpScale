"""Safety V2 snapshot body: pure, deterministic, from stored inputs only.

`build_body` re-checks every selected input: an input fetched after ``as_of`` raises
`SafetyCausalityError`, and an input of another identity raises `SafetyIdentityError`
(base58 is case-sensitive: a case variant is another identity).

The body holds no wall-clock data other than ``as_of`` and the inputs' ``fetched_at``.
Its canonical JSON (`repository.encode_body`) is hashed, so the same stored inputs, the
same ``as_of``, the same ``RULES_VERSION`` and the same code fingerprints give an
identical body and hash. The fingerprints are part of the body: a rebuild under other
rules or helper code is reported as a fingerprint mismatch, never as a reproduction.

Authority honesty: only a successfully parsed mint (observation outcome MINT) makes an
authority AVAILABLE. A failed, missing, not-collected or unreadable read is never a
revoked (null) authority.
"""

import hashlib
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from upscale.services.safety_v2.config import RULES_VERSION, SNAPSHOT_SCHEMA
from upscale.services.safety_v2.models import (
    DEFERRED_RULES,
    Evidence,
    IdentityStatus,
    SafetyCausalityError,
    SafetyIdentityError,
    SafetyStateError,
    Status,
    available,
    iso,
    missing,
)
from upscale.services.safety_v2.provider import TOKEN_PROGRAMS
from upscale.services.safety_v2.repository import MintRow
from upscale.services.safety_v2.rules import assess, evaluate

_PACKAGE = Path(__file__).resolve().parent
# The Safety V2 modules whose code decides a body (parsing, fields, rules, encoding).
FINGERPRINTED = ("config.py", "features.py", "models.py", "provider.py", "repository.py",
                 "rules.py")  # fmt: skip
SOLANA_CHAIN = _PACKAGE.parent / "solana_chain.py"
# A mint observation that isn't MINT: what its authority fields are (never "revoked").
_UNREAD_STATUS: dict[str, Status] = {
    "NOT_COLLECTED": "NOT_COLLECTED", "PROVIDER_FAILED": "PROVIDER_UNAVAILABLE",
    "MALFORMED": "UNKNOWN", "ACCOUNT_MISSING": "UNAVAILABLE", "NOT_A_MINT": "UNAVAILABLE",
}  # fmt: skip
# How far the mint-account read itself got (a read proving "no mint here" is AVAILABLE).
_COMPONENT_STATUS: dict[str, Status] = {
    "MINT": "AVAILABLE", "NOT_A_MINT": "AVAILABLE", "ACCOUNT_MISSING": "AVAILABLE",
    "MALFORMED": "UNKNOWN", "PROVIDER_FAILED": "PROVIDER_UNAVAILABLE",
    "NOT_COLLECTED": "NOT_COLLECTED",
}  # fmt: skip
AUTHORITY_FIELDS = (
    "token_program", "mint_authority", "freeze_authority", "supply_raw", "decimals", "extensions",
)  # fmt: skip


def _digest(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


@cache
def code_fingerprints() -> dict[str, str]:
    """RULES_VERSION plus SHA-256 of the Safety V2 rule-bearing sources and of
    ``solana_chain.py`` (whose `parse_mint` and program ids Safety V2 reuses)."""
    return {
        "rules_version": RULES_VERSION,
        "safety_v2_source": _digest(*(_PACKAGE / f for f in FINGERPRINTED)),
        "solana_chain_source": _digest(SOLANA_CHAIN),
    }


def authority_fields(row: MintRow | None) -> dict[str, Evidence]:
    if row is None:
        why = "no mint observation of this identity was fetched at or before as_of"
        return {k: missing("UNAVAILABLE", why) for k in AUTHORITY_FIELDS}
    if row.outcome != "MINT":
        status = _UNREAD_STATUS.get(row.outcome, "UNAVAILABLE")
        why = f"mint observation {row.id} is {row.outcome}: {row.reason}"
        return {k: missing(status, why) for k in AUTHORITY_FIELDS}
    assert row.token_program and row.decimals is not None and row.supply_raw is not None
    return {
        "token_program": available({"program": row.token_program, "program_id": row.program_owner}),
        "mint_authority": available(
            {"active": row.mint_authority is not None, "address": row.mint_authority}
        ),
        "freeze_authority": available(
            {"active": row.freeze_authority is not None, "address": row.freeze_authority}
        ),
        "supply_raw": available(row.supply_raw),
        "decimals": available(row.decimals),
        "extensions": available(list(row.extensions or [])),
    }


def _owned_as_named(row: MintRow) -> bool:
    """A MINT row is owned by exactly the token program it names (the classifier and the
    schema guarantee it; a row breaking it is never VERIFIED)."""
    return (
        row.program_owner is not None and TOKEN_PROGRAMS.get(row.program_owner) == row.token_program
    )


def _inconsistent(row: MintRow) -> str:
    return (
        f"mint observation {row.id} is MINT but owned by {row.program_owner}, not the "
        f"{row.token_program} program: stored evidence is inconsistent"
    )


def token_identity(row: MintRow | None) -> tuple[IdentityStatus, str]:
    if row is None:
        return "UNVERIFIED", "no mint observation at or before as_of"
    if row.outcome == "MINT":
        if not _owned_as_named(row):
            raise SafetyStateError(_inconsistent(row))
        return "VERIFIED", f"an initialized {row.token_program} mint exists at this exact address"
    if row.outcome in ("ACCOUNT_MISSING", "NOT_A_MINT"):
        return "MISMATCH", str(row.reason)
    return "UNVERIFIED", f"{row.outcome}: {row.reason}"


def _mint_component(row: MintRow | None) -> Status:
    if row is None:
        return "UNAVAILABLE"
    return _COMPONENT_STATUS[row.outcome]


def build_body(
    canonical_id: str,
    mint: str,
    as_of: float,
    mint_row: MintRow | None,
    fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    """The ``safety.snapshot.v2`` body for one exact identity as of `as_of`."""
    if canonical_id != f"solana:{mint}":
        raise SafetyIdentityError(f"{canonical_id} isn't the identity of mint {mint}")
    inputs: list[dict[str, Any]] = []
    if mint_row is not None:
        if mint_row.canonical_id != canonical_id:
            raise SafetyIdentityError(
                f"mint observation {mint_row.id} belongs to {mint_row.canonical_id}, not "
                f"{canonical_id}"
            )
        if mint_row.outcome == "MINT" and not _owned_as_named(mint_row):
            raise SafetyStateError(_inconsistent(mint_row))
        if mint_row.fetched_at > as_of:
            raise SafetyCausalityError(
                f"mint observation {mint_row.id} was fetched at {iso(mint_row.fetched_at)}, "
                f"after as_of {iso(as_of)}"
            )
        inputs.append({
            "component": "mint_account",
            "observation_id": mint_row.id,
            "collection_id": mint_row.collection_id,
            "fetched_at": iso(mint_row.fetched_at),
            "provider": mint_row.provider,
            "outcome": mint_row.outcome,
            "raw_hash": mint_row.raw_hash,
            "context_slot": mint_row.context_slot,
        })  # fmt: skip

    authority = authority_fields(mint_row)
    status, why = token_identity(mint_row)
    results = evaluate(
        status,
        mint_row.outcome if mint_row else None,
        mint_row.reason if mint_row else None,
        mint_row.program_owner if mint_row else None,
        authority,
    )
    assessment, coverage, coverage_reasons = assess(results, status, why)
    deferred_why = "not collected in Safety V2 Phase 1"
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "rules_version": RULES_VERSION,
        "as_of": iso(as_of),
        "identity": {
            "canonical_id": canonical_id,
            "chain": "solana",
            "mint": mint,
            "address_format": "VALID_SOLANA_BASE58",
            "token_mint": {"status": status, "reason": why},
            "pool_match": {
                "status": "UNVERIFIED",
                "evidence": missing(
                    "NOT_COLLECTED", "pool / market collection is a later phase"
                ).as_dict(),
            },
        },
        "provenance": {"inputs": inputs, "fingerprints": dict(sorted(fingerprints.items()))},
        "authority": {k: v.as_dict() for k, v in authority.items()},
        "flags": [r.model_dump(mode="json", exclude={"needs"}) for r in results],
        "undetermined": [
            {"id": r.id, "needs": r.needs, "reason": r.reason}
            for r in results
            if r.outcome == "UNDETERMINED"
        ],  # fmt: skip
        "assessment": assessment,
        "coverage": {
            "coverage": coverage,
            "reasons": coverage_reasons,
            "decision_rules": [r.id for r in results if r.decision_bearing],
            "components": {
                "mint_account": _mint_component(mint_row),
                "holders": "NOT_SUPPORTED",
                "market": "NOT_SUPPORTED",
                "creator": "NOT_SUPPORTED",
            },
            "component_notes": {k: deferred_why for k in ("creator", "holders", "market")},
            "not_supported": [{"id": k, "reason": v} for k, v in sorted(DEFERRED_RULES.items())],
        },
    }
