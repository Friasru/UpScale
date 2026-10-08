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

Holders (Phase 2; `holder_section`). Owners are classified per ``as_of`` from positive
evidence only (`classify_owner`): registry entries with ``known_since <= as_of``, the
target's pool pins with ``pinned_at <= as_of``, the owner's own account as collected, and
Radar signer proof with ``fetched_at <= as_of``. Semantics:

* **Denominator** of every percentage: the token's total supply (``getTokenSupply``, same
  collection), never reduced by excluded balances.
* **top1 / top5 / top10**: the N largest owners not proven ``POOL_OR_VAULT`` or ``BURN``
  (so NORMAL_WALLET, UNKNOWN, UNRESOLVED and PROGRAM_OWNED are counted; program-owned
  owners are disclosed separately). Owner-aggregated before ranking.
* **top10_wallet_pct**: the 10 largest proven ``NORMAL_WALLET`` owners only.
* **holder_count**: every owner-aggregated owner with a non-zero balance, except owners
  proven ``POOL_OR_VAULT`` / ``BURN`` (dust owners included).
* **meaningful_holder_count**: those of them holding at least
  ``rules.MEANINGFUL_HOLDER_FRACTION`` (1/1,000,000) of total supply; the only input of
  ``FEW_HOLDERS``. The fraction is disclosed in the section.
* AVAILABLE needs a ``full_scan`` with every token account's owner resolved and the
  balances consistent with supply. A ``partial_scan`` or ``largest_accounts`` read only
  ever undercounts balances, so its shares are PARTIAL lower bounds; the wallet share is
  also a lower bound while an UNKNOWN / UNRESOLVED owner could still enter the wallet
  top 10. A holder count is a lower bound only from a partial scan whose owners are all
  resolved (an unresolved account could repeat a counted owner).
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from upscale.services.safety_v2.config import RULES_VERSION, SNAPSHOT_SCHEMA
from upscale.services.safety_v2.models import (
    DEFERRED_RULES,
    EXCLUDED_CLASSES,
    OWNER_CLASSES,
    Evidence,
    IdentityStatus,
    OwnerClass,
    SafetyCausalityError,
    SafetyIdentityError,
    SafetyStateError,
    Status,
    available,
    iso,
    missing,
)
from upscale.services.safety_v2.provider import TOKEN_PROGRAMS, OwnerFact
from upscale.services.safety_v2.registry import REGISTRY_VERSION, lookup
from upscale.services.safety_v2.repository import HolderRow, MintRow, PoolPin, owners_hash
from upscale.services.safety_v2.rules import (
    HOLDER_RULE_IDS,
    LARGE_OWNER_PCT,
    MEANINGFUL_HOLDER_FRACTION,
    HolderFacts,
    assess,
    evaluate,
    evaluate_holders,
    is_meaningful,
)
from upscale.services.safety_v2.sources import NOT_CONSULTED, WalletProof, WalletProofs
from upscale.services.solana_chain import SYSTEM_PROGRAM

_PACKAGE = Path(__file__).resolve().parent
# The Safety V2 modules whose code decides a body (parsing, fields, rules, encoding).
FINGERPRINTED = ("config.py", "features.py", "models.py", "provider.py", "registry.py",
                 "repository.py", "rules.py", "sources.py")  # fmt: skip
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
    """RULES_VERSION plus SHA-256 of the Safety V2 rule-bearing sources (``registry.py``
    and the Radar reader ``sources.py`` included, so any registry change is a fingerprint
    mismatch) and of ``solana_chain.py`` (whose `parse_mint`, holder helpers and program ids
    Safety V2 reuses)."""
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
    holders: "HolderInputs | None" = None,
) -> dict[str, Any]:
    """The ``safety.snapshot.v2`` body for one exact identity as of `as_of`. Without a
    holder observation at or before `as_of`, the holder rules are out of scope (listed in
    ``coverage.out_of_scope``, never counted toward coverage)."""
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

    hin = holders or HolderInputs(None)
    inputs += _check_holder_inputs(canonical_id, as_of, hin)
    section, facts = holder_section(as_of, hin)

    authority = authority_fields(mint_row)
    status, why = token_identity(mint_row)
    results = evaluate(
        status,
        mint_row.outcome if mint_row else None,
        mint_row.reason if mint_row else None,
        mint_row.program_owner if mint_row else None,
        authority,
    )
    in_scope = hin.observation is not None
    if in_scope:
        results = sorted(results + evaluate_holders(facts), key=lambda r: r.id)
    assessment, coverage, coverage_reasons = assess(results, status, why)
    deferred_why = "not collected yet in Safety V2"
    notes = {k: deferred_why for k in ("creator", "market")}
    if not in_scope:
        notes["holders"] = HOLDERS_OUT_OF_SCOPE
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
        "holders": section,
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
                "holders": section["status"],
                "market": "NOT_SUPPORTED",
                "creator": "NOT_SUPPORTED",
            },
            "component_notes": dict(sorted(notes.items())),
            "not_supported": [{"id": k, "reason": v} for k, v in sorted(DEFERRED_RULES.items())],
            "out_of_scope": []
            if in_scope
            else [{"id": r, "reason": HOLDERS_OUT_OF_SCOPE} for r in HOLDER_RULE_IDS],
        },
    }


# --- holders (Phase 2) ---------------------------------------------------------------------

HOLDERS_OUT_OF_SCOPE = (
    "no holder collection of this identity at or before as_of: holder rules are out of scope"
)
_TOP_N = (1, 5, 10)
_LISTED = 10
_STATUS_OF: dict[str, Status] = {"NOT_COLLECTED": "NOT_COLLECTED",
                                 "PROVIDER_FAILED": "PROVIDER_UNAVAILABLE"}  # fmt: skip
HOLDER_FIELDS = ("top1_pct", "top5_pct", "top10_pct", "top10_wallet_pct", "holder_count",
                 "meaningful_holder_count")  # fmt: skip


@dataclass(frozen=True)
class HolderInputs:
    """Everything a holder section reads: the selected observation (None: out of scope),
    its balances, the target's pool pins and the Radar wallet proofs, all <= as_of."""

    observation: HolderRow | None
    balances: tuple[OwnerFact, ...] = ()
    pools: tuple[PoolPin, ...] = ()
    proofs: WalletProofs = field(default_factory=lambda: NOT_CONSULTED)


@dataclass(frozen=True)
class Classified:
    fact: OwnerFact
    cls: OwnerClass
    reason: str
    identified: bool  # the owner's role is resolved (registry / pool pin), not just its type
    proof: WalletProof | None = None


def _check_holder_inputs(canonical_id: str, as_of: float, h: HolderInputs) -> list[dict[str, Any]]:
    """Re-check identity and causality of every holder input; their provenance entries."""
    out: list[dict[str, Any]] = []
    obs = h.observation
    if obs is None:
        if h.balances:
            raise SafetyStateError("holder balances without a holder observation")
        return out
    if obs.canonical_id != canonical_id:
        raise SafetyIdentityError(
            f"holder observation {obs.id} belongs to {obs.canonical_id}, not {canonical_id}"
        )
    if obs.fetched_at > as_of:
        raise SafetyCausalityError(
            f"holder observation {obs.id} was fetched at {iso(obs.fetched_at)}, after as_of "
            f"{iso(as_of)}"
        )
    if obs.outcome == "COLLECTED" and owners_hash(list(h.balances)) != obs.owners_hash:
        raise SafetyStateError(f"holder observation {obs.id}'s balances don't match its hash")
    if obs.outcome != "COLLECTED" and h.balances:
        raise SafetyStateError(f"holder observation {obs.id} is {obs.outcome} but has balances")
    out.append({
        "component": "holders",
        "observation_id": obs.id,
        "collection_id": obs.collection_id,
        "fetched_at": iso(obs.fetched_at),
        "provider": obs.provider,
        "origin": obs.origin,
        "outcome": obs.outcome,
        "source": obs.source,
        "owners_hash": obs.owners_hash,
    })  # fmt: skip
    for pin in h.pools:
        if pin.canonical_id != canonical_id:
            raise SafetyIdentityError(f"pool pin {pin.id} belongs to {pin.canonical_id}")
        if pin.pinned_at > as_of:
            raise SafetyCausalityError(
                f"pool pin {pin.id} was learned at {iso(pin.pinned_at)}, after as_of {iso(as_of)}"
            )
        out.append({"component": "pool_pin", "pool_address": pin.pool_address, "dex": pin.dex,
                    "pinned_at": iso(pin.pinned_at), "source": pin.source})  # fmt: skip
    for proof in h.proofs.proofs.values():
        if proof.fetched_at > as_of:
            raise SafetyCausalityError(
                f"Radar wallet proof for {proof.wallet} was learned at {iso(proof.fetched_at)}, "
                f"after as_of {iso(as_of)}"
            )
    out.append({"component": "wallet_proofs", "source": "radar", "status": h.proofs.status,
                "reason": h.proofs.reason, "proofs": len(h.proofs.proofs)})  # fmt: skip
    return out


def classify_owner(
    fact: OwnerFact, as_of: float, pools: Sequence[PoolPin], proofs: WalletProofs
) -> Classified:
    """The owner's type as of `as_of`, from positive evidence only (see `OwnerClass`)."""
    owner = fact.owner
    if owner is None:
        return Classified(fact, "UNRESOLVED", "the token account's owner couldn't be read", False)
    entry = lookup(owner, as_of)
    if entry is not None:
        if entry.kind == "BURN":
            return Classified(fact, "BURN", entry.reason, True)
        if entry.kind == "PROGRAM":
            return Classified(fact, "PROGRAM_OWNED", entry.reason, True)
        dexes = {p.dex for p in pools}
        if entry.corroborating_dex in dexes:
            return Classified(fact, "POOL_OR_VAULT", f"{entry.reason}; the target has a pinned "
                              f"{entry.corroborating_dex} pool", True)  # fmt: skip
        return Classified(fact, "UNKNOWN", f"{entry.reason}, but no pinned "
                          f"{entry.corroborating_dex} pool corroborates it for this token",
                          False)  # fmt: skip
    if owner in {p.pool_address for p in pools}:
        return Classified(fact, "POOL_OR_VAULT", "the target's pinned pool (owns its vaults)",
                          True)  # fmt: skip
    proof = proofs.proofs.get(owner)
    program_owned = fact.lookup == "FOUND" and fact.owner_program != SYSTEM_PROGRAM
    if program_owned and proof is not None:
        return Classified(fact, "UNRESOLVED", f"conflicting evidence: its account is owned by "
                          f"{fact.owner_program}, yet Radar saw it sign", False)  # fmt: skip
    if program_owned:
        return Classified(fact, "PROGRAM_OWNED", f"its account is owned by the program "
                          f"{fact.owner_program}; its role isn't identified", False)  # fmt: skip
    if proof is not None:
        return Classified(fact, "NORMAL_WALLET", "signed a transaction (Radar signer evidence)",
                          False, proof)  # fmt: skip
    if fact.lookup == "NOT_LOOKED_UP":
        return Classified(fact, "UNRESOLVED", "its account wasn't read (beyond the owner "
                          "lookup cap)", False)  # fmt: skip
    if proofs.status == "AVAILABLE":
        why = "not program-owned, but no signer evidence proves it is a wallet"
    else:
        why = f"not program-owned; wallet proof unavailable ({proofs.status})"
    return Classified(fact, "UNKNOWN", why, False)


def _pct(amount: int, supply: int) -> float:
    return round(100 * amount / supply, 6)


def _owner_entry(c: Classified, supply: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "owner": c.fact.owner, "owner_key": c.fact.key, "classification": c.cls,
        "reason": c.reason, "pct": _pct(c.fact.amount_raw, supply),
        "amount_raw": str(c.fact.amount_raw), "token_accounts": len(c.fact.accounts),
    }  # fmt: skip
    if c.proof is not None:
        entry["wallet_proof"] = {"source": "radar", "fetched_at": iso(c.proof.fetched_at),
                                 "signature": c.proof.signature}  # fmt: skip
    return entry


def _unavailable_section(
    status: Status, why: str, obs: HolderRow | None
) -> tuple[dict[str, Any], HolderFacts]:
    fields = {k: missing(status, why).as_dict() for k in HOLDER_FIELDS}
    section: dict[str, Any] = {
        "status": status, "reason": why,
        "source": obs.source if obs else None,
        "fetched_at": iso(obs.fetched_at) if obs else None,
        "lower_bound": False, **fields,
        "unknown_large_owners": [], "program_large_owners": [], "top_owners": [],
        "excluded": [], "owner_classes": {}, "reasons": list(obs.reasons) if obs else [],
    }  # fmt: skip
    return section, HolderFacts(status=status, reason=why)


def holder_section(as_of: float, h: HolderInputs) -> tuple[dict[str, Any], HolderFacts]:
    """The body's ``holders`` section and the facts the holder rules read."""
    obs = h.observation
    if obs is None:
        return _unavailable_section("UNAVAILABLE", HOLDERS_OUT_OF_SCOPE, None)
    if obs.outcome != "COLLECTED":
        failed = _STATUS_OF.get(obs.outcome, "UNAVAILABLE")
        return _unavailable_section(failed, f"holder observation {obs.id} is {obs.outcome}: "
                                    f"{obs.reason}", obs)  # fmt: skip
    assert obs.supply_raw is not None and obs.source is not None
    supply = int(obs.supply_raw)
    total = sum(f.amount_raw for f in h.balances)
    if not obs.reliable or supply == 0 or not h.balances or total > supply:
        why = "; ".join(obs.reasons) or "the holder data is inconsistent with the supply"
        return _unavailable_section("UNKNOWN", f"holder observation {obs.id} can't be "
                                    f"trusted: {why}", obs)  # fmt: skip

    ranked = sorted(h.balances, key=lambda f: (-f.amount_raw, f.key))
    owners = [classify_owner(f, as_of, h.pools, h.proofs) for f in ranked]
    counted = [c for c in owners if c.cls not in EXCLUDED_CLASSES]
    excluded = [c for c in owners if c.cls in EXCLUDED_CLASSES]
    resolved = all(f.owner is not None for f in ranked)
    full = obs.source == "full_scan" and resolved
    partial_why = (
        "; ".join(obs.reasons) or "the holder evidence is incomplete, so shares are lower bounds"
    )

    def share(amount: int, n_counted: int) -> Evidence:
        if full:
            return available(_pct(amount, supply))
        if not n_counted:
            return missing("UNKNOWN", f"no counted owner was observed: {partial_why}")
        return Evidence(status="PARTIAL", value=_pct(amount, supply), lower_bound=True,
                        reason=partial_why)  # fmt: skip

    tops = {n: sum(c.fact.amount_raw for c in counted[:n]) for n in _TOP_N}
    wallets = [c for c in counted if c.cls == "NORMAL_WALLET"]
    floor = wallets[9].fact.amount_raw if len(wallets) >= 10 else 0
    contenders = [c for c in counted
                  if c.cls in ("UNKNOWN", "UNRESOLVED") and c.fact.amount_raw > floor]  # fmt: skip
    wallet_amount = sum(c.fact.amount_raw for c in wallets[:10])
    wallet: Evidence
    if full and not contenders:
        wallet = available(_pct(wallet_amount, supply))
    elif not wallets:
        wallet = missing("UNKNOWN", f"no owner is a proven wallet, and {len(contenders)} "
                         f"UNKNOWN / UNRESOLVED owners could be ({h.proofs.status} wallet "
                         "proof)")  # fmt: skip
    else:
        gaps = [] if full else [partial_why]
        if contenders:
            gaps.append(f"{len(contenders)} UNKNOWN / UNRESOLVED owners could still be "
                        "top-10 wallets")  # fmt: skip
        wallet = Evidence(status="PARTIAL", value=_pct(wallet_amount, supply), lower_bound=True,
                          reason="; ".join(gaps))  # fmt: skip

    meaningful = sum(is_meaningful(c.fact.amount_raw, supply) for c in counted)

    def count(value: int) -> Evidence:
        if obs.source == "largest_accounts":
            return missing("UNAVAILABLE", "only the largest token accounts were read: they say "
                           "nothing about how many holders exist")  # fmt: skip
        if not resolved:
            return missing("UNKNOWN", "some token accounts' owners are unresolved and could "
                           "repeat a counted owner, so the count isn't even a lower bound")  # fmt: skip
        if full:
            return available(value)
        return Evidence(status="PARTIAL", value=value, lower_bound=True, reason=partial_why)

    large = [c for c in counted if c.fact.amount_raw * 100 >= LARGE_OWNER_PCT * supply]
    unknown_large = [c for c in large if c.cls in ("UNKNOWN", "UNRESOLVED")]
    program_large = [c for c in large if c.cls == "PROGRAM_OWNED"]
    classes = {k: sum(c.cls == k for c in owners) for k in OWNER_CLASSES}
    fields = {
        "top1_pct": share(tops[1], len(counted)),
        "top5_pct": share(tops[5], len(counted)),
        "top10_pct": share(tops[10], len(counted)),
        "top10_wallet_pct": wallet,
        "holder_count": count(len(counted)),
        "meaningful_holder_count": count(meaningful),
    }
    status: Status = "AVAILABLE" if full else "PARTIAL"
    reason = None if full else partial_why
    section = {
        "status": status,
        "reason": reason,
        "source": obs.source,
        "fetched_at": iso(obs.fetched_at),
        "lower_bound": not full,
        "supply_raw": obs.supply_raw,
        "decimals": obs.decimals,
        "pages_read": obs.pages_read,
        "max_pages": obs.max_pages,
        "token_accounts_seen": obs.token_accounts_seen,
        **{k: v.as_dict() for k, v in fields.items()},
        "unknown_large_owners": [_owner_entry(c, supply) for c in unknown_large],
        "program_large_owners": [
            dict(_owner_entry(c, supply), role_identified=c.identified) for c in program_large
        ],
        "top_owners": [_owner_entry(c, supply) for c in counted[:_LISTED]],
        "excluded": [_owner_entry(c, supply) for c in excluded],
        "owner_classes": classes,
        "large_owner_min_pct": LARGE_OWNER_PCT,
        "meaningful_holder_min_fraction": "{}/{}".format(*MEANINGFUL_HOLDER_FRACTION),
        "registry_version": REGISTRY_VERSION,
        "semantics": {
            "denominator": "total supply (getTokenSupply, same collection)",
            "concentration": "largest owners not proven POOL_OR_VAULT / BURN, owner-aggregated",
            "wallet_concentration": "proven NORMAL_WALLET owners only",
            "holder_count": "owners with a non-zero balance, minus proven POOL_OR_VAULT / BURN",
            "meaningful_holder_count": "holder_count owners holding >= "
            "meaningful_holder_min_fraction of total supply (the FEW_HOLDERS input)",
        },
        "reasons": list(obs.reasons),
    }
    facts = HolderFacts(
        status=status,
        reason=reason,
        supply=supply,
        top1_amount=tops[1],
        top10_amount=tops[10],
        meaningful_count=fields["meaningful_holder_count"],
        unknown_large=tuple(c.fact.key for c in unknown_large),
        unidentified_program_large=tuple(c.fact.key for c in program_large if not c.identified),
        unresolved_large=tuple(c.fact.key for c in large if c.cls == "UNRESOLVED"),
    )
    return section, facts
