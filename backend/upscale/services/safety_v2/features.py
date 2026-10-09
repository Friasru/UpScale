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
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from upscale.services.safety_v2.config import RULES_VERSION, SNAPSHOT_SCHEMA
from upscale.services.safety_v2.market import (
    AccountCheck,
    ObservedMarket,
    market_view,
    pool_record,
    with_selection,
)
from upscale.services.safety_v2.models import (
    DEFERRED_RULES,
    EXCLUDED_CLASSES,
    OWNER_CLASSES,
    VALUED,
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
    CHANGE_RULE_IDS,
    CREATOR_RULE_IDS,
    HOLDER_RULE_IDS,
    LARGE_OWNER_PCT,
    LIQUIDITY_WINDOW_S,
    MARKET_NOT_REPORTED_MIN_DURATION_S,
    MARKET_NOT_REPORTED_MIN_MISSES,
    MARKET_RULE_IDS,
    MEANINGFUL_HOLDER_FRACTION,
    ChangeFacts,
    DeployerFacts,
    HolderFacts,
    MarketFacts,
    assess,
    evaluate,
    evaluate_changes,
    evaluate_creator,
    evaluate_holders,
    evaluate_market,
    holder_loss_applicable,
    is_meaningful,
    top10_change_pp,
)
from upscale.services.safety_v2.sources import (
    NOT_CONSULTED,
    RADAR_SCHEMA_VERSION,
    WalletProof,
    WalletProofs,
)
from upscale.services.solana_chain import SYSTEM_PROGRAM

_PACKAGE = Path(__file__).resolve().parent
# The Safety V2 modules whose code decides a body (parsing, fields, rules, encoding).
FINGERPRINTED = ("config.py", "features.py", "market.py", "models.py", "provider.py",
                 "registry.py", "repository.py", "rules.py", "sources.py")  # fmt: skip
SOLANA_CHAIN = _PACKAGE.parent / "solana_chain.py"
# Pool eligibility and primary selection (`select_primary_pool`) and DEX row parsing
# (`parse_pair`) decide market feature values, so both are fingerprinted.
SOLANA_DEX = _PACKAGE.parent / "solana_dex.py"
DEXSCREENER = _PACKAGE.parent / "dexscreener.py"
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
        "solana_dex_source": _digest(SOLANA_DEX),
        "dexscreener_source": _digest(DEXSCREENER),
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
    market: "MarketInputs | None" = None,
    creator: "CreatorInputs | None" = None,
    changes: "ChangeInputs | None" = None,
) -> dict[str, Any]:
    """The ``safety.snapshot.v2`` body for one exact identity as of `as_of`. Without a
    holder (market) observation at or before `as_of`, the holder (market) rules are out of
    scope (listed in ``coverage.out_of_scope``, never counted toward coverage)."""
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
    min_ = market or MarketInputs()
    inputs += _check_market_inputs(canonical_id, as_of, min_)
    msection, mfacts = market_section(mint, as_of, min_)
    cin = creator or CreatorInputs()
    inputs += _check_creator_inputs(canonical_id, as_of, cin)
    csection, dfacts, deployer_scope = creator_section(as_of, cin, hin)
    chin = changes or ChangeInputs()
    inputs += _check_change_inputs(canonical_id, as_of, hin, chin)
    chsection, chfacts = changes_section(as_of, hin, chin)

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
    market_scope = bool(min_.observations)
    out_of_scope: list[dict[str, str]] = []
    if in_scope:
        results += evaluate_holders(facts)
    if market_scope:
        results += evaluate_market(mfacts)
        if not mfacts.closure_applicable:
            out_of_scope.append({"id": "MARKET_CLOSED_ON_CHAIN",
                                 "reason": mfacts.closure_reason or str(mfacts.reason)})  # fmt: skip
    if deployer_scope is None:
        results += evaluate_creator(dfacts)
    else:
        out_of_scope += [{"id": r, "reason": deployer_scope} for r in CREATOR_RULE_IDS]
    if chin.had_prior:
        results += evaluate_changes(chfacts)
        loss_ok, loss_why = holder_loss_applicable(chfacts)
        if not loss_ok:
            results = [r for r in results if r.id != "RAPID_HOLDER_LOSS"]
            out_of_scope.append({"id": "RAPID_HOLDER_LOSS", "reason": loss_why})
    else:
        out_of_scope += [{"id": r, "reason": CHANGES_OUT_OF_SCOPE} for r in CHANGE_RULE_IDS]
    results = sorted(results, key=lambda r: r.id)
    assessment, coverage, coverage_reasons = assess(results, status, why)
    notes: dict[str, str] = {}
    if csection["source_status"] == "NOT_CAPTURED":
        notes["creator"] = CREATOR_OUT_OF_SCOPE
    if not in_scope:
        notes["holders"] = HOLDERS_OUT_OF_SCOPE
        out_of_scope += [{"id": r, "reason": HOLDERS_OUT_OF_SCOPE} for r in HOLDER_RULE_IDS]
    if not market_scope:
        notes["market"] = MARKET_OUT_OF_SCOPE
        out_of_scope += [{"id": r, "reason": MARKET_OUT_OF_SCOPE} for r in MARKET_RULE_IDS]
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
            "pool_match": pool_match(min_, msection),
        },
        "provenance": {"inputs": inputs, "fingerprints": dict(sorted(fingerprints.items()))},
        "authority": {k: v.as_dict() for k, v in authority.items()},
        "holders": section,
        "market": msection,
        "creator": csection,
        "changes": chsection,
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
                "market": msection["status"],
                "creator": csection["source_status"],
            },
            "component_notes": dict(sorted(notes.items())),
            "not_supported": [{"id": k, "reason": v} for k, v in sorted(DEFERRED_RULES.items())],
            "out_of_scope": sorted(out_of_scope, key=lambda r: r["id"]),
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
        if proof.captured_at is not None and proof.captured_at > as_of:
            raise SafetyCausalityError(
                f"wallet proof for {proof.wallet} was captured at {iso(proof.captured_at)}, "
                f"after as_of {iso(as_of)}"
            )
    out.append({"component": "wallet_proofs", "source": "radar", "status": h.proofs.status,
                "reason": h.proofs.reason, "proofs": len(h.proofs.proofs),
                "capture_id": h.proofs.capture_id})  # fmt: skip
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
        entry["wallet_proof"] = {
            "source": "radar", "radar_schema_version": RADAR_SCHEMA_VERSION,
            "fetched_at": iso(c.proof.fetched_at), "signature": c.proof.signature,
            "radar_target": c.proof.canonical_id, "source_key": c.proof.source_key,
            "captured_at": iso(c.proof.captured_at), "capture_row_id": c.proof.capture_row_id,
        }  # fmt: skip
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


# --- market (Phase 3) ----------------------------------------------------------------------

MARKET_OUT_OF_SCOPE = (
    "no market observation of this identity at or before as_of: market rules are out of scope"
)
PROVIDER_REPORTED = "PROVIDER_REPORTED"
MARKET_FIELDS = ("primary_pool", "primary_dex", "primary_liquidity_usd", "pool_age_hours",
                 "primary_clear", "liquidity_change_pct_24h", "volume_24h", "txns_24h",
                 "price_usd")  # fmt: skip


@dataclass(frozen=True)
class MarketInputs:
    """Every market observation, the pool pins and the pool-account checks, all <= as_of."""

    observations: tuple[ObservedMarket, ...] = ()
    pins: tuple[PoolPin, ...] = ()
    accounts: tuple[AccountCheck, ...] = ()


def _check_market_inputs(canonical_id: str, as_of: float, m: MarketInputs) -> list[dict[str, Any]]:
    """Re-check causality of every market input; their provenance entries."""
    if not m.observations:
        return []
    for o in m.observations:
        if o.fetched_at > as_of:
            raise SafetyCausalityError(
                f"market observation {o.id} was fetched at {iso(o.fetched_at)}, after as_of "
                f"{iso(as_of)}"
            )
        bad = [p.pair_address for p in o.pools
               if p.identity == "EXACT_BASE" and f"solana:{p.base_mint}" != canonical_id]  # fmt: skip
        if bad:
            raise SafetyIdentityError(f"market observation {o.id} has pools of another mint")
    for a in m.accounts:
        if a.fetched_at > as_of:
            raise SafetyCausalityError(
                f"pool account observation {a.id} was fetched at {iso(a.fetched_at)}, after "
                f"as_of {iso(as_of)}"
            )
    for pin in m.pins:
        if pin.canonical_id != canonical_id:
            raise SafetyIdentityError(f"pool pin {pin.id} belongs to {pin.canonical_id}")
        if pin.pinned_at > as_of:
            raise SafetyCausalityError(
                f"pool pin {pin.id} was learned at {iso(pin.pinned_at)}, after as_of {iso(as_of)}"
            )
    latest = m.observations[-1]
    if any(o.fetched_at > latest.fetched_at for o in m.observations):
        raise SafetyStateError("market observations must be ordered oldest first")
    return [{
        "component": "market",
        "observation_id": latest.id,
        "collection_id": latest.collection_id,
        "fetched_at": iso(latest.fetched_at),
        "provider": latest.provider,
        "outcome": latest.outcome,
        "history_observations": len(m.observations),
        "pool_account_checks": len(m.accounts),
    }]  # fmt: skip


def _empty_market(status: Status, why: str) -> tuple[dict[str, Any], MarketFacts]:
    fields = {k: missing(status, why).as_dict() for k in MARKET_FIELDS}
    section: dict[str, Any] = {
        "status": status, "reason": why, "basis": PROVIDER_REPORTED, "fetched_at": None,
        **fields, "pool_presence": {"state": "UNAVAILABLE" if status != "NOT_COLLECTED"
                                    else "NOT_COLLECTED", "reason": why},
        "alternative_eligible_pools": [], "pinned_pools": [], "other_pools": [],
        "not_reported": None, "pool_account": None, "ambiguity": [],
        "closure_check": {"applicable": False, "reason": why},
    }  # fmt: skip
    return section, MarketFacts(status=status, reason=why)


def pool_match(m: MarketInputs, section: dict[str, Any]) -> dict[str, Any]:
    """Whether a provider-reported pool has this exact mint as its base (never on-chain
    verification: the status stays UNVERIFIED)."""
    if not m.observations:
        return {"status": "UNVERIFIED", "evidence": missing("NOT_COLLECTED",
                MARKET_OUT_OF_SCOPE).as_dict()}  # fmt: skip
    presence = section["pool_presence"]["state"]
    if presence == "REPORTED":
        ev = available({"pool": section["primary_pool"]["value"]["address"],
                        "basis": "the provider reports this exact mint as the pool's base"})  # fmt: skip
    else:
        ev = missing("UNAVAILABLE", f"no exact pool is reported ({presence})")
    return {"status": "UNVERIFIED", "evidence": ev.as_dict()}


def market_section(mint: str, as_of: float, m: MarketInputs) -> tuple[dict[str, Any], MarketFacts]:
    """The body's ``market`` section and the facts the market rules read."""
    if not m.observations:
        return _empty_market("UNAVAILABLE", MARKET_OUT_OF_SCOPE)
    pins = sorted(m.pins, key=lambda p: (p.pinned_at, p.id))
    view = market_view(mint, m.observations, [(p.pool_address, p.dex, p.pinned_at) for p in pins],
                       m.accounts)  # fmt: skip
    latest = view.latest
    ok = latest.successful
    status: Status = (
        "AVAILABLE" if ok
        else "NOT_COLLECTED" if latest.outcome == "NOT_COLLECTED"
        else "PROVIDER_UNAVAILABLE"
    )  # fmt: skip
    reason = None if ok else f"market observation {latest.id} is {latest.outcome}: {latest.reason}"
    if not ok:
        section, _ = _empty_market(status, str(reason))
    else:
        section = {"status": status, "reason": None}
    section |= {"basis": PROVIDER_REPORTED, "fetched_at": iso(latest.fetched_at),
                "pool_presence": {"state": view.presence, "reason": view.presence_reason}}  # fmt: skip

    reported = view.reported
    no_pool = f"the tracked pool isn't reported ({view.presence})"
    if not ok:
        tracked_why = str(reason)
    elif view.anchor is None:
        tracked_why = "no exact-mint pool is known for this token"
    else:
        tracked_why = no_pool

    def pool_field(value: Any) -> Evidence:
        if reported is None:
            return missing("UNAVAILABLE" if ok else status, tracked_why)
        if value is None:
            return missing("UNAVAILABLE", "the provider didn't report this field")
        return available(value)

    if view.anchor is not None:
        section["primary_pool"] = available({"address": view.anchor, "dex": view.anchor_dex,
                                             "anchor": view.anchor_source}).as_dict()  # fmt: skip
        section["primary_dex"] = (available(view.anchor_dex) if view.anchor_dex
                                  else missing("UNAVAILABLE", "no DEX known")).as_dict()  # fmt: skip
    else:
        for k in ("primary_pool", "primary_dex"):
            section[k] = missing("UNAVAILABLE" if ok else status, tracked_why).as_dict()

    liquidity = pool_field(reported.liquidity_usd if reported else None)
    section["primary_liquidity_usd"] = liquidity.as_dict()
    for k, attr in (("volume_24h", "volume_24h"), ("txns_24h", "txns_24h"),
                    ("price_usd", "price_usd")):  # fmt: skip
        section[k] = pool_field(getattr(reported, attr) if reported else None).as_dict()

    # Pool age: as_of - provider pair_created_at, never negative.
    age_s: float | None = None
    if reported is None:
        age = missing("UNAVAILABLE" if ok else status, tracked_why)
    elif reported.pair_created_at is None:
        age = missing("UNAVAILABLE", "the provider reported no pool creation time")
    elif reported.pair_created_at > as_of:
        age = missing("UNKNOWN", "the provider-reported creation time is after as_of")
    else:
        age_s = as_of - reported.pair_created_at
        age = available(round(age_s / 3600, 6))
    section["pool_age_hours"] = age.as_dict()

    # Liquidity change: the same exact pool, latest vs its 24 h high (never bridged).
    window_max: float | None = None
    if reported is None or reported.liquidity_usd is None:
        change = missing("UNAVAILABLE" if ok else status, tracked_why)
    else:
        points = [
            p.liquidity_usd for o in m.observations
            if o is not latest and o.successful
            and latest.fetched_at - LIQUIDITY_WINDOW_S <= o.fetched_at < latest.fetched_at
            and (p := o.exact(reported.pair_address)) is not None and p.liquidity_usd is not None
        ]  # fmt: skip
        if not points:
            change = missing("UNAVAILABLE", "fewer than 2 same-pool liquidity observations in "
                             "the preceding 24 h")  # fmt: skip
        elif max(points) <= 0:
            change = missing("UNAVAILABLE", "the same-pool 24 h high is zero")
        else:
            window_max = max(points)
            change = available(round(100 * (reported.liquidity_usd - window_max) / window_max, 6))
    section["liquidity_change_pct_24h"] = change.as_dict()

    sel = view.selection
    if not ok:
        clear = missing(status, str(reason))
    elif pins:
        clear = available(len(pins) == 1)
    elif sel is not None and sel.primary is not None:
        clear = available(sel.clear)
    else:
        clear = missing("UNAVAILABLE", "no eligible exact-mint pool to select from")
    section["primary_clear"] = clear.as_dict()
    # Eligibility is re-derived from the stored figures (never trusted from collection).
    current = with_selection(latest.pools, mint) if ok else ()
    eligible = [p for p in current if p.identity == "EXACT_BASE" and p.eligible]
    section["alternative_eligible_pools"] = [pool_record(p) for p in eligible
                                             if p.pair_address != view.anchor]  # fmt: skip
    section["pinned_pools"] = list(view.pins)
    section["other_pools"] = [
        {"pair_address": p.pair_address, "identity": p.identity, "base_mint": p.base_mint,
         "base_symbol": p.base_symbol, "reasons": list(p.rejections)}
        for p in latest.pools if p.identity != "EXACT_BASE"
    ]  # fmt: skip
    section["ambiguity"] = list(sel.ambiguity) if sel is not None else []
    first = view.misses[0].fetched_at if view.misses else None
    last = view.misses[-1].fetched_at if view.misses else None
    span = (last - first) if first is not None and last is not None else 0.0
    section["closure_check"] = {"applicable": view.closure_applicable,
                                "reason": view.closure_reason}  # fmt: skip
    section["not_reported"] = {
        "misses": len(view.misses), "first_miss_at": iso(first), "last_miss_at": iso(last),
        "min_misses": MARKET_NOT_REPORTED_MIN_MISSES,
        "min_duration_s": MARKET_NOT_REPORTED_MIN_DURATION_S,
    }  # fmt: skip
    a = view.account
    section["pool_account"] = None if a is None else {
        "outcome": a.outcome, "fetched_at": iso(a.fetched_at), "program_owner": a.program_owner,
        "reason": a.reason,
    }  # fmt: skip

    facts = MarketFacts(
        status=status, reason=reason, presence=view.presence,
        presence_reason=view.presence_reason, account_outcome=a.outcome if a else None,
        anchor=view.anchor,
        liquidity_usd=reported.liquidity_usd if reported else None,
        liquidity_reason=str(liquidity.reason or ""),
        window_max_usd=window_max, change_reason=str(change.reason or ""),
        age_s=age_s, age_reason=str(age.reason or ""),
        any_eligible=bool(eligible) if ok else None,
        pinned=len(pins), has_primary=sel is not None and sel.primary is not None,
        clear=sel.clear if sel is not None else None,
        ambiguity=tuple(sel.ambiguity) if sel is not None else (),
        misses=len(view.misses), miss_span_s=span,
        closure_applicable=view.closure_applicable, closure_reason=view.closure_reason,
    )  # fmt: skip
    return section, facts


# --- creator / verified deployer (Phase 4B) ------------------------------------------------

CREATOR_OUT_OF_SCOPE = "no Radar capture at or before as_of: creator evidence is out of scope"
POOL_ONLY_COVERAGE = (
    "Radar observes only transactions touching the tracked pool: a deployer transfer that "
    "doesn't touch the pool is invisible to it, so zero observed TOKEN_OUTFLOW events never "
    "proves there were none"
)


@dataclass(frozen=True)
class CreatorInputs:
    """Safety-captured Radar evidence usable at as_of (``source_time <= as_of`` and
    ``captured_at <= as_of``): the latest capture attempt and the captured rows."""

    capture: tuple[int, str, str | None, float] | None = None
    creators: tuple[dict[str, Any], ...] = ()
    flows: tuple[dict[str, Any], ...] = ()
    coverage: tuple[dict[str, Any], ...] = ()


def _check_creator_inputs(
    canonical_id: str, as_of: float, c: CreatorInputs
) -> list[dict[str, Any]]:
    if c.capture is None:
        if c.creators or c.flows or c.coverage:
            raise SafetyStateError("captured Radar facts without a capture at or before as_of")
        return []
    if c.capture[3] > as_of:
        raise SafetyCausalityError(
            f"Radar capture {c.capture[0]} was made at {iso(c.capture[3])}, after as_of {iso(as_of)}"
        )
    for kind, rows in (("creator", c.creators), ("flow", c.flows), ("coverage", c.coverage)):
        for r in rows:
            if r["canonical_id"] != canonical_id:
                raise SafetyIdentityError(f"captured {kind} row {r['id']} is of another token")
            for col in ("source_time", "captured_at"):
                if r[col] > as_of:
                    raise SafetyCausalityError(
                        f"captured {kind} row {r['id']} has {col} {iso(r[col])}, after as_of "
                        f"{iso(as_of)}"
                    )
    return [{
        "component": "radar_capture", "capture_id": c.capture[0], "status": c.capture[1],
        "captured_at": iso(c.capture[3]),
        "creator_rows": sorted(r["id"] for r in c.creators),
        "flow_rows": sorted(r["id"] for r in c.flows),
        "coverage_rows": sorted(r["id"] for r in c.coverage),
    }]  # fmt: skip


def _role_fact(row: dict[str, Any] | None, missing_status: str, why: str) -> dict[str, Any]:
    if row is None:
        return {"status": missing_status, "address": None, "reason": why}
    return {
        "status": row["status"], "address": row["address"], "method": row["method"],
        "reason": json.loads(row["provenance_json"]).get("reason")
        or json.loads(row["provenance_json"]).get("note"),
        "radar_signature": row["signature"], "source_system": row["source_system"],
        "radar_schema_version": row["radar_schema_version"],
        "source_key": row["source_key"], "source_time": iso(row["source_time"]),
        "captured_at": iso(row["captured_at"]), "capture_row_id": row["id"],
    }  # fmt: skip


def _latest(rows: Sequence[dict[str, Any]], role: str) -> dict[str, Any] | None:
    """The latest knowable fact of one role: (source_time, Safety row id), never iteration
    order. Roles are never mixed."""
    mine = [r for r in rows if r["role"] == role]
    return max(mine, key=lambda r: (r["source_time"], r["id"])) if mine else None


def _outflow_coverage(rows: Sequence[dict[str, Any]]) -> tuple[Status, list[str]]:
    """Whether captured activity coverage could support an outflow count. Never COMPLETE:
    Radar's pool-only listing can't see every deployer transfer."""
    if not rows:
        return "UNAVAILABLE", ["no Radar activity coverage was captured"]
    reasons = [POOL_ONLY_COVERAGE]
    gaps: dict[int, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: (r["source_time"], r["id"])):
        if r["kind"] == "GAP":
            gaps[r["radar_id"]] = r  # the latest captured state of each gap
    open_gaps = sum(g["status"] == "OPEN" for g in gaps.values())
    if open_gaps:
        reasons.append(f"{open_gaps} Radar activity gaps are open (signatures not traversed)")
    scans = [json.loads(r["facts_json"]) for r in rows if r["kind"] == "SCAN"]
    if not any(s.get("kind") == "activity" for s in scans):
        reasons.append("no Radar activity scan was captured")
    if any(s.get("status") != "AVAILABLE" for s in scans):
        reasons.append("some captured Radar scans weren't complete (status not AVAILABLE)")
    if any((s.get("txs_skipped") or 0) or (s.get("txs_beyond_cap") or 0) for s in scans):
        reasons.append("some captured Radar scans skipped transactions or hit a parse cap")
    if any(s.get("kind") == "activity" and s.get("head_listing_complete") == 0 for s in scans):
        reasons.append("some captured Radar activity listings were incomplete")
    return "PARTIAL", reasons


def creator_section(
    as_of: float, c: CreatorInputs, h: "HolderInputs"
) -> tuple[dict[str, Any], DeployerFacts, str | None]:
    """(``creator`` section, deployer-holding facts, out-of-scope reason or None)."""
    source_why: str | None
    if c.capture is None:
        source, source_why = "NOT_CAPTURED", CREATOR_OUT_OF_SCOPE
    else:
        source, source_why = c.capture[1], c.capture[2]
    captured = source == "CAPTURED"
    if not captured:
        absent = source_why or f"Radar capture {source}"
        cand = _role_fact(None, "NOT_COLLECTED", absent)
        dep = _role_fact(None, "NOT_COLLECTED", absent)
        cand_row = dep_row = None
    else:
        cand_row = _latest(c.creators, "POOL_CREATOR_CANDIDATE")
        dep_row = _latest(c.creators, "TOKEN_DEPLOYER")
        cand = _role_fact(cand_row, "NOT_COLLECTED",
                          "Radar had no pool-creator determination captured by as_of")  # fmt: skip
        dep = _role_fact(dep_row, "UNVERIFIED",
                         "Radar had no token-deployer determination captured by as_of "
                         "(not proof that none exists)")  # fmt: skip
    verified: str | None = None  # the VERIFIED deployer's exact address
    if dep_row is not None and dep_row["status"] == "VERIFIED" and dep_row["address"]:
        verified = str(dep_row["address"])
    match: bool | None = None
    if (verified is not None and cand_row is not None and cand_row["status"] == "CANDIDATE"
            and cand_row["address"]):  # fmt: skip
        match = cand_row["address"] == verified  # exact, case-sensitive

    # Holding: only for a VERIFIED deployer, from the latest holder observation.
    scope: str | None = None
    facts = DeployerFacts("UNAVAILABLE", "no verified token deployer")
    holding: Evidence = missing("UNAVAILABLE", "no verified token deployer at as_of")
    if not verified:
        scope = (f"no VERIFIED token deployer at as_of (Radar capture: {source}; deployer: "
                 f"{dep['status']})")  # fmt: skip
    else:
        address = verified
        obs = h.observation
        if obs is None:
            holding = missing("UNAVAILABLE", "no holder observation at or before as_of")
        elif obs.outcome != "COLLECTED":
            holding = missing(_STATUS_OF.get(obs.outcome, "UNAVAILABLE"),
                              f"holder observation {obs.id} is {obs.outcome}")  # fmt: skip
        elif not obs.reliable or not obs.supply_raw or int(obs.supply_raw) == 0:
            holding = missing("UNKNOWN", f"holder observation {obs.id} can't be trusted")
        else:
            supply = int(obs.supply_raw)
            amount = sum(b.amount_raw for b in h.balances if b.owner == address)
            full = obs.source == "full_scan" and all(b.owner is not None for b in h.balances)
            pct = round(100 * amount / supply, 6)
            if full:
                holding = available(pct)  # absent from a complete scan: exactly 0
            elif amount > 0:
                holding = Evidence(status="PARTIAL", value=pct, lower_bound=True,
                                   reason=f"observed in {obs.source} evidence: a lower bound")  # fmt: skip
            else:
                holding = missing("UNKNOWN", f"absent from {obs.source} holder evidence, which "
                                  "is incomplete: not proof of zero")  # fmt: skip
            if holding.status in VALUED:
                facts = DeployerFacts(holding.status, holding.reason, amount, supply)
        if holding.status not in VALUED:
            facts = DeployerFacts(holding.status, holding.reason)

    # Outflows: evidence only.
    outflows: dict[str, Any]
    if not verified:
        outflows = {"status": "UNAVAILABLE", "reason": "no verified token deployer at as_of"}
    else:
        events = sorted((f for f in c.flows if f["wallet"] == verified
                         and f["direction"] == "TOKEN_OUTFLOW"),
                        key=lambda f: (f["source_time"], f["id"]))  # fmt: skip
        cov_status, cov_reasons = _outflow_coverage(c.coverage)
        times = [f["block_time"] if f["block_time"] is not None else f["source_time"]
                 for f in events]  # fmt: skip
        outflows = {
            "status": cov_status,
            "kind": "TOKEN_OUTFLOW",
            "event_count": Evidence(status="PARTIAL", value=len(events), lower_bound=True,
                                    reason=cov_reasons[0]).as_dict()
            if cov_status == "PARTIAL"
            else missing("UNAVAILABLE", cov_reasons[0]).as_dict(),
            "total_amount_raw": str(sum(int(f["amount_raw"]) for f in events)),
            "first_event_time": iso(min(times)) if times else None,
            "last_event_time": iso(max(times)) if times else None,
            "events": [{"signature": f["signature"], "amount_raw": f["amount_raw"],
                        "block_time": iso(f["block_time"]), "source_time": iso(f["source_time"]),
                        "capture_row_id": f["id"]} for f in events],
            "coverage_reasons": cov_reasons,
            "note": "observed TOKEN_OUTFLOW events (balance decreases), not trades",
        }  # fmt: skip
    section = {
        "source_status": source,
        "source_reason": source_why,
        "pool_creator_candidate": cand,
        "token_deployer": dep,
        "candidate_matches_verified_deployer": match,
        "deployer_holding_pct": holding.as_dict(),
        "deployer_outflows": outflows,
    }
    return section, facts, scope


# --- holder changes (Phase 4B) --------------------------------------------------------------

CHANGES_OUT_OF_SCOPE = (
    "no earlier holder observation at or before as_of: holder-change rules are out of scope"
)


@dataclass(frozen=True)
class ChangeInputs:
    """`had_prior`: an earlier holder observation exists. `previous`: the latest strictly
    earlier (in (fetched_at, id) order) *complete* one, with its pins and wallet proofs as
    knowable at its own fetched_at (historical classification), or None."""

    had_prior: bool = False
    previous: "HolderInputs | None" = None


def is_complete(obs: HolderRow | None, balances: Sequence[OwnerFact]) -> bool:
    """A holder observation exact comparison can use: a reliable full scan, every owner
    resolved, supply known and > 0, balances consistent with supply."""
    return (
        obs is not None and obs.outcome == "COLLECTED" and obs.reliable
        and obs.source == "full_scan" and obs.supply_raw is not None
        and int(obs.supply_raw) > 0 and all(b.owner is not None for b in balances)
        and sum(b.amount_raw for b in balances) <= int(obs.supply_raw)
    )  # fmt: skip


def _check_change_inputs(
    canonical_id: str, as_of: float, cur: "HolderInputs", ch: ChangeInputs
) -> list[dict[str, Any]]:
    prev = ch.previous
    if prev is None or prev.observation is None:
        return []
    p, c = prev.observation, cur.observation
    if c is None or (p.fetched_at, p.id) >= (c.fetched_at, c.id):
        raise SafetyStateError("the previous holder observation must precede the current one")
    # Its own causality, as of its own fetched_at (historical classification inputs).
    out = _check_holder_inputs(canonical_id, p.fetched_at, prev)
    for entry in out:
        entry["component"] = "previous_" + entry["component"]
    return out


def _change_pct(cur: int, prev: int) -> Evidence:
    if prev == 0:
        return missing(
            "UNAVAILABLE", "the previous count is zero: a percentage change is undefined"
        )
    return available(round(100 * (cur - prev) / prev, 6))


def changes_section(
    as_of: float, cur: "HolderInputs", ch: ChangeInputs
) -> tuple[dict[str, Any], ChangeFacts]:
    obs = cur.observation
    fields = ("top10_change_pp", "holder_count_change_pct", "meaningful_holder_count_change_pct")
    if not ch.had_prior or obs is None:
        why = CHANGES_OUT_OF_SCOPE
        section: dict[str, Any] = {
            "status": "UNAVAILABLE", "reason": why, "current_holder_observation_id":
            obs.id if obs else None, "previous_holder_observation_id": None,
            **{k: missing("UNAVAILABLE", why).as_dict() for k in fields},
            "large_holder_exits": missing("UNAVAILABLE", why).as_dict(),
        }  # fmt: skip
        return section, ChangeFacts(False, why)
    prev = ch.previous
    if not is_complete(obs, cur.balances):
        why = f"the current holder observation {obs.id} isn't a complete scan"
    elif prev is None or prev.observation is None:
        why = (
            "no earlier complete holder scan to compare with (earlier scans are partial or failed)"
        )
    else:
        why = ""
    if why:
        section = {
            "status": "UNKNOWN", "reason": why, "current_holder_observation_id": obs.id,
            "previous_holder_observation_id": prev.observation.id
            if prev and prev.observation else None,
            **{k: missing("UNKNOWN", why).as_dict() for k in fields},
            "large_holder_exits": missing("UNKNOWN", why).as_dict(),
        }  # fmt: skip
        return section, ChangeFacts(False, why)

    assert prev is not None and prev.observation is not None
    p = prev.observation
    # Each scan as a snapshot at its own time would report it: previous owners classified
    # with what was knowable at the previous fetched_at, current with what is knowable now.
    psec, pf = holder_section(p.fetched_at, prev)
    csec, cf = holder_section(as_of, cur)
    pp = top10_change_pp(ChangeFacts(True, None, prev_top10=pf.top10_amount,
                                     prev_supply=pf.supply, cur_top10=cf.top10_amount,
                                     cur_supply=cf.supply))  # fmt: skip
    present = {b.owner for b in cur.balances if b.amount_raw > 0}
    prev_owners = [classify_owner(f, p.fetched_at, prev.pools, prev.proofs)
                   for f in sorted(prev.balances, key=lambda f: (-f.amount_raw, f.key))]  # fmt: skip
    exits = [
        {"owner": c.fact.owner, "previous_pct": _pct(c.fact.amount_raw, pf.supply),
         "previous_amount_raw": str(c.fact.amount_raw), "current_amount_raw": "0",
         "classification_at_previous": c.cls, "classification_reason": c.reason,
         "previous_holder_observation_id": p.id, "current_holder_observation_id": obs.id}
        for c in prev_owners
        if c.cls not in EXCLUDED_CLASSES
        and c.fact.amount_raw * 100 >= LARGE_OWNER_PCT * pf.supply
        and c.fact.owner not in present
    ]  # fmt: skip
    hc_prev, hc_cur = psec["holder_count"]["value"], csec["holder_count"]["value"]
    mc_prev = psec["meaningful_holder_count"]["value"]
    mc_cur = csec["meaningful_holder_count"]["value"]
    section = {
        "status": "AVAILABLE", "reason": None,
        "current_holder_observation_id": obs.id, "previous_holder_observation_id": p.id,
        "current_fetched_at": iso(obs.fetched_at), "previous_fetched_at": iso(p.fetched_at),
        "elapsed_seconds": obs.fetched_at - p.fetched_at,
        "top10_change_pp": available(round(float(pp), 6)).as_dict(),
        "previous_top10_pct": psec["top10_pct"]["value"], "current_top10_pct": csec["top10_pct"]["value"],
        "holder_count_change_pct": _change_pct(hc_cur, hc_prev).as_dict(),
        "previous_holder_count": hc_prev, "current_holder_count": hc_cur,
        "meaningful_holder_count_change_pct": _change_pct(mc_cur, mc_prev).as_dict(),
        "previous_meaningful_holder_count": mc_prev, "current_meaningful_holder_count": mc_cur,
        "large_holder_exits": available(exits).as_dict(),
        "semantics": {
            "top10_change_pp": "current top10_pct - previous top10_pct, percentage points",
            "holder_count_change_pct": "(current - previous) / previous holder_count, %",
            "meaningful_holder_count_change_pct": "(current - previous) / previous "
            "meaningful_holder_count, % (the RAPID_HOLDER_LOSS input)",
            "large_holder_exits": f"owners >= {LARGE_OWNER_PCT}% in the previous complete scan "
            "absent from the current complete scan (not reductions); pool / burn excluded",
        },
    }  # fmt: skip
    facts = ChangeFacts(True, None, obs.fetched_at - p.fetched_at, pf.top10_amount, pf.supply,
                        cf.top10_amount, cf.supply, int(mc_prev), int(mc_cur), len(exits))  # fmt: skip
    return section, facts
