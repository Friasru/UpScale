"""Pure normalization of ``getTransaction`` (jsonParsed) results into token flows.

A flow is one owner's net change in the tracked mint across one transaction, from the
transaction's own pre/post token balances, summed per owner (an owner with several token
accounts is one participant). Nothing is inferred beyond that:

* An increase is ``TOKEN_INFLOW`` and a decrease ``TOKEN_OUTFLOW``: never a buy or a sale.
* Every owner is classified (`classify`) only from positive evidence in the transaction or
  in Radar's own known address sets. A signer is a ``NORMAL_WALLET`` (programs and PDAs
  can't sign); known programs / pools / vault authorities / routers are typed as such; and
  everything else stays ``UNKNOWN``, never assumed to be a wallet. Wallet statistics use
  ``NORMAL_WALLET`` identities only (see `features`).
* The tracked pool's own balance is the pool side of the transaction and isn't a flow.
* ``TRACKED_POOL_COUNTERPARTY`` is set only when the transaction succeeded, exactly one
  participant's balance changed, it is a signing wallet, and the tracked pool moved by
  exactly the opposite amount. When the tracked pool's vaults sit under a shared vault
  authority (Raydium AMM v4) the pool side can't be attributed: ``NOT_SUPPORTED``.
  Everything else is ``UNVERIFIED``.
* Failed transactions change no token balance and yield no flows.

Raw payloads are never stored; only the normalized result is.
"""

from collections.abc import Collection, Iterator, Mapping
from typing import Any

from upscale.services.radar.models import (
    Counterparty,
    ParsedTx,
    Participant,
    SignatureInfo,
    WalletDelta,
)
from upscale.services.solana_chain import (
    INCINERATOR,
    RAYDIUM_AMM_V4_AUTHORITY,
    SYSTEM_PROGRAM,
    TOKEN_2022_PROGRAM,
    TOKEN_PROGRAM,
)

# Well-known program ids (a token balance "owned" by one of these is never a wallet).
KNOWN_PROGRAMS = frozenset({
    SYSTEM_PROGRAM, TOKEN_PROGRAM, TOKEN_2022_PROGRAM, INCINERATOR,
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # associated token account program
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM v4
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",  # Raydium CPMM
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",  # Raydium CLMM
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",  # Orca Whirlpool
    "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",  # Meteora DLMM
    "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB",  # Meteora dynamic AMM
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",  # pump.fun
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",  # PumpSwap AMM
})  # fmt: skip
# Aggregator / router programs.
KNOWN_ROUTERS = frozenset({
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter v6
    "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB",  # Jupiter v4
})  # fmt: skip
# Authorities that hold the vaults of many pools at once.
SHARED_VAULT_AUTHORITIES = frozenset({RAYDIUM_AMM_V4_AUTHORITY})


def classify(
    owner: str,
    *,
    signers: Collection[str],
    invoked_programs: Collection[str],
    pool_address: str,
    known_pools: Collection[str] = (),
    known_intermediaries: Collection[str] = (),
) -> Participant:
    """The owner's type from positive evidence only; UNKNOWN when nothing proves it."""
    if owner == pool_address or owner in known_pools or owner in SHARED_VAULT_AUTHORITIES:
        return "POOL_OR_VAULT"
    if owner in KNOWN_ROUTERS or owner in known_intermediaries:
        return "ROUTER_OR_INTERMEDIARY"
    if owner in KNOWN_PROGRAMS or owner in invoked_programs:
        return "PROGRAM"
    if owner in signers:
        return "NORMAL_WALLET"
    return "UNKNOWN"


INIT_MINT_TYPES = frozenset({"initializeMint", "initializeMint2"})
SYSTEM_TRANSFER_TYPES = frozenset({"transfer", "transferWithSeed", "createAccount"})


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def parse_signatures(rows: Any) -> list[SignatureInfo]:
    """``getSignaturesForAddress`` rows (newest first), malformed rows skipped."""
    out: list[SignatureInfo] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("signature"), str):
            continue
        slot = _int(row.get("slot"))
        bt = row.get("blockTime")
        out.append(
            SignatureInfo(
                signature=row["signature"],
                slot=slot,
                block_time=float(bt) if isinstance(bt, int) and not isinstance(bt, bool) else None,
                failed=row.get("err") is not None,
            )
        )
    return out


def account_keys(tx: Mapping[str, Any]) -> list[tuple[str, bool]]:
    """(pubkey, signer) in message order; index 0 is the fee payer."""
    message = (tx.get("transaction") or {}).get("message") or {}
    out: list[tuple[str, bool]] = []
    for key in message.get("accountKeys") or []:
        if isinstance(key, dict) and isinstance(key.get("pubkey"), str):
            out.append((key["pubkey"], bool(key.get("signer"))))
        elif isinstance(key, str):  # non-parsed encoding: signers unknown
            out.append((key, False))
    return out


def instructions(tx: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every top-level and inner (CPI) instruction."""
    message = (tx.get("transaction") or {}).get("message") or {}
    for ix in message.get("instructions") or []:
        if isinstance(ix, dict):
            yield ix
    meta = tx.get("meta") or {}
    for group in meta.get("innerInstructions") or []:
        for ix in (group or {}).get("instructions") or [] if isinstance(group, dict) else []:
            if isinstance(ix, dict):
                yield ix


def invoked_programs(tx: Mapping[str, Any]) -> set[str]:
    return {ix["programId"] for ix in instructions(tx) if isinstance(ix.get("programId"), str)}


def initializes_mint(tx: Mapping[str, Any], mint: str) -> bool:
    for ix in instructions(tx):
        parsed = ix.get("parsed")
        if (
            isinstance(parsed, dict)
            and parsed.get("type") in INIT_MINT_TYPES
            and isinstance(parsed.get("info"), dict)
            and parsed["info"].get("mint") == mint
        ):
            return True
    return False


def _balances(rows: Any, mint: str) -> tuple[dict[str, int], int]:
    per_owner: dict[str, int] = {}
    unattributed = 0
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("mint") != mint:
            continue
        owner = row.get("owner")
        ui = row.get("uiTokenAmount")
        amount = _int(ui.get("amount")) if isinstance(ui, dict) else None
        if not isinstance(owner, str) or amount is None:
            unattributed += 1
            continue
        per_owner[owner] = per_owner.get(owner, 0) + amount
    return per_owner, unattributed


def parse_transaction(
    tx: Mapping[str, Any],
    mint: str,
    pool_address: str,
    known_pools: Collection[str] = (),
    known_intermediaries: Collection[str] = (),
) -> ParsedTx | None:
    """The normalized transaction, or None when the payload has no signature."""
    sigs = (tx.get("transaction") or {}).get("signatures")
    if not isinstance(sigs, list) or not sigs or not isinstance(sigs[0], str):
        return None
    meta = tx.get("meta") or {}
    keys = account_keys(tx)
    signers = {k for k, s in keys if s}
    programs = invoked_programs(tx)
    failed = meta.get("err") is not None
    bt = tx.get("blockTime")
    block_time = float(bt) if isinstance(bt, int) and not isinstance(bt, bool) else None

    pre, un_pre = _balances(meta.get("preTokenBalances"), mint)
    post, un_post = _balances(meta.get("postTokenBalances"), mint)
    deltas = {o: post.get(o, 0) - pre.get(o, 0) for o in sorted(set(pre) | set(post))}
    changed = {o: d for o, d in deltas.items() if d != 0 and o != pool_address}
    pool_delta = deltas.get(pool_address, 0)
    types = {
        o: classify(o, signers=signers, invoked_programs=programs, pool_address=pool_address,
                    known_pools=known_pools, known_intermediaries=known_intermediaries)
        for o in changed
    }  # fmt: skip

    flows: list[WalletDelta] = []
    if not failed:
        shared_vault = pool_delta == 0 and any(o in SHARED_VAULT_AUTHORITIES for o in changed)
        only = next(iter(changed.items()), None)
        matched = (
            len(changed) == 1
            and only is not None
            and types[only[0]] == "NORMAL_WALLET"
            and pool_delta != 0
            and only[1] == -pool_delta
        )
        for owner, delta in changed.items():
            counterparty: Counterparty = (
                "TRACKED_POOL_COUNTERPARTY" if matched
                else "NOT_SUPPORTED" if shared_vault
                else "UNVERIFIED"
            )  # fmt: skip
            flows.append(
                WalletDelta(
                    wallet=owner,
                    direction="TOKEN_INFLOW" if delta > 0 else "TOKEN_OUTFLOW",
                    amount_raw=abs(delta),
                    counterparty=counterparty,
                    participant=types[owner],
                    signer=owner in signers,
                )
            )
    return ParsedTx(
        signature=sigs[0],
        slot=_int(tx.get("slot")),
        block_time=block_time,
        fee_payer=keys[0][0] if keys else None,
        failed=failed,
        flows=tuple(flows),
        initializes_mint=initializes_mint(tx, mint),
        unattributed_balances=un_pre + un_post,
    )


def first_funder(tx: Mapping[str, Any], wallet: str) -> str | None:
    """The source of the first System Program SOL transfer into `wallet` in this
    transaction, or None. Only meaningful for the wallet's own oldest transaction."""
    for ix in instructions(tx):
        parsed = ix.get("parsed")
        if ix.get("program") != "system" or not isinstance(parsed, dict):
            continue
        info = parsed.get("info")
        if parsed.get("type") not in SYSTEM_TRANSFER_TYPES or not isinstance(info, dict):
            continue
        dest = info.get("destination") or info.get("newAccount")
        source = info.get("source")
        if dest == wallet and isinstance(source, str) and source != wallet:
            return source
    return None
