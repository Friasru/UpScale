"""Solana on-chain token safety data by exact mint: models, providers, holder analysis, and a
caching service.

No estimates: every value comes from the chain. Agents depend on `SolanaSafetyService`;
providers implement `SolanaChainProvider`:

* `SolanaRpcProvider`: any standard Solana JSON-RPC endpoint (`getAccountInfo`,
  `getTokenLargestAccounts`, `getMultipleAccounts`). It can't list every holder.
* `HeliusProvider`: the same RPC on Helius (API key), plus Helius' DAS `getTokenAccounts`,
  paginated, to scan every token account of the mint up to `max_pages` pages.

Holder methodology (`analyze`):

1. **Token accounts.** With Helius, every token account of the mint is scanned (page by
   page, up to the page cap). The RPC's 20 largest token accounts are always read too and
   merged in by account address, so the biggest balances are never missed even when the
   scan stops early. Accounts whose parsed mint isn't exactly this mint are ignored.
2. **Owners.** Balances are summed per owner wallet, so one wallet with many token
   accounts is one holder. Shares, top-1, top-10 and holder counts are all computed after
   this aggregation. Percentages are of total supply.
3. **Exclusions**, only when identified with high confidence:
   * ``burn``: the owner is the Solana incinerator (no private key exists for it).
   * ``liquidity_pool``: the owner is a pool the DEX agent reported *for this mint and
     accepted as a usable market* (pool vaults are owned by the pool account on Orca,
     Raydium CLMM/CPMM, Meteora, PumpSwap and pump.fun), or the Raydium AMM v4 authority
     when DEX data also reports a usable Raydium pool for this mint.
   Everything else is counted: pools the DEX agent rejected, the Raydium v4 authority
   without a corroborating Raydium pool, and program-owned accounts that can't be
   identified (vesting, team vaults and pools look alike) are labeled but still counted.
4. **Completeness.** ``full_scan`` (every page read) is authoritative. ``partial_scan``
   (page cap reached) and ``largest_accounts`` (no Helius) give lower bounds: an owner's
   other accounts may be missing. Every gap is listed in `incomplete_reasons`, and every
   fallback overstates concentration rather than understating it, so truncated data can
   never make a token look safer.
"""

import asyncio
import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import httpx2
from pydantic import BaseModel, Field

from upscale.services.chains import is_solana_address
from upscale.services.market_data import (
    AssetNotFoundError,
    InvalidRequestError,
    MarketDataError,
    MarketDataUnavailableError,
    RateLimiter,
)

TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
INCINERATOR = "1nc1nerator11111111111111111111111111111111"
# Owns the token vaults of every Raydium AMM v4 pool. Only excluded when DEX data reports
# a usable Raydium pool for the mint (see `analyze`).
RAYDIUM_AMM_V4_AUTHORITY = "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1"
MAX_LARGEST_ACCOUNTS = 20  # what getTokenLargestAccounts returns at most
SCAN_PAGE_SIZE = 1000  # DAS getTokenAccounts maximum

TokenProgram = Literal["spl_token", "token_2022", "other"]
HolderKind = Literal["holder", "liquidity_pool", "burn"]
ConcentrationSource = Literal["full_scan", "partial_scan", "largest_accounts", "unavailable"]


# --- Models ---------------------------------------------------------------------------------


class MintInfo(BaseModel):
    mint: str
    program_id: str
    program: TokenProgram
    decimals: int
    supply_raw: int  # base units
    mint_authority: str | None  # None: revoked (no one can mint more)
    freeze_authority: str | None  # None: no one can freeze holders' accounts
    extensions: list[str] = Field(default_factory=list)  # Token-2022 extension names
    extension_state: dict[str, Any] = Field(default_factory=dict)

    @property
    def supply(self) -> float:
        return self.supply_raw / float(10**self.decimals)


class TokenAccountBalance(BaseModel):
    address: str
    amount_raw: int


class AccountRecord(BaseModel):
    """What the chain says about one account (None fields: not a parsed token account)."""

    address: str
    program_owner: str  # the program that owns this account
    token_owner: str | None = None  # token accounts: the wallet that controls it
    token_mint: str | None = None  # token accounts: which mint it holds


class ScannedAccount(BaseModel):
    address: str
    owner: str
    amount_raw: int


class TokenAccountScan(BaseModel):
    """Token accounts of the mint listed page by page (Helius DAS)."""

    accounts: list[ScannedAccount]
    pages_read: int
    max_pages: int
    complete: bool  # the last page was reached (not stopped by the page cap)
    skipped_rows: int = 0  # rows without an address, owner or amount


class KnownPool(BaseModel):
    """A DEX pool the DEX agent reported for this mint."""

    address: str
    dex: str
    eligible: bool  # accepted as a usable market (liquid, active, priced)


class ChainData(BaseModel):
    """Raw facts fetched for one mint; analysis happens separately so it can be redone
    with different pools without refetching."""

    address: str  # the mint
    # Two independent components: one failing never discards the other.
    mint: MintInfo | None  # authorities, program, supply; None when it couldn't be read
    mint_error: str | None = None
    largest: list[TokenAccountBalance] | None  # None when holder data couldn't be read
    holders_error: str | None = None
    # Supply / decimals from getTokenSupply, when the mint account itself couldn't be read.
    supply_raw: int | None = None
    decimals: int | None = None
    token_accounts: dict[str, AccountRecord | None] = Field(default_factory=dict)
    scan: TokenAccountScan | None = None  # None: the provider can't list every account
    owners: dict[str, AccountRecord | None] = Field(default_factory=dict)
    provider: str
    fetched_at: datetime


class HolderEntry(BaseModel):
    owner: str | None  # None when the owner couldn't be resolved
    token_accounts: list[str]
    amount: float  # UI units
    pct_of_supply: float
    kind: HolderKind
    label: str | None = None  # why excluded, or why counted despite looking special
    program_owned: bool | None = None  # owner is an account owned by a program, not a wallet


class OnchainSafetySnapshot(BaseModel):
    canonical_id: str  # "solana:<mint>"
    mint: str
    provider: str
    fetched_at: datetime
    # Authorities component (the mint account). When unavailable, the fields below are None
    # and mean "unknown", never "revoked".
    authorities_available: bool
    authorities_error: str | None = None
    token_program: TokenProgram | None
    token_program_id: str | None
    decimals: int | None
    supply: float | None
    mint_authority: str | None
    freeze_authority: str | None
    extensions: list[str] = Field(default_factory=list)
    extension_state: dict[str, Any] = Field(default_factory=dict)
    # Holder component (largest accounts / full scan).
    holders_available: bool = True
    holders_error: str | None = None
    holders: list[HolderEntry]  # largest non-excluded owners (up to `listed_holders`)
    excluded: list[HolderEntry]  # pools and burn
    top1_pct: float | None  # largest non-excluded owner, % of supply
    top10_pct: float | None  # ten largest non-excluded owners combined
    excluded_pct: float
    holder_count: int | None  # non-excluded owners with a balance (scan only)
    meaningful_holder_count: int | None
    holder_count_complete: bool
    concentration_source: ConcentrationSource
    # True unless a complete scan backs the numbers: shares are then at-least values.
    concentration_lower_bound: bool
    token_accounts_seen: int
    scan_pages_read: int | None
    scan_max_pages: int | None
    largest_accounts_seen: int
    pool_addresses_used: list[str]  # owners whose balances were excluded as pool vaults
    # False when owners couldn't be resolved: concentration can't be attributed.
    concentration_reliable: bool
    holder_data_complete: bool
    incomplete_reasons: list[str] = Field(default_factory=list)

    @property
    def mint_authority_active(self) -> bool | None:
        """None when the authorities couldn't be read (unknown, not revoked)."""
        return self.mint_authority is not None if self.authorities_available else None

    @property
    def freeze_authority_active(self) -> bool | None:
        return self.freeze_authority is not None if self.authorities_available else None

    @property
    def concentration_authoritative(self) -> bool:
        """A complete holder scan with every owner resolved: enough to satisfy the
        holder-concentration evidence requirement."""
        return self.concentration_source == "full_scan" and self.concentration_reliable


# --- Holder analysis ------------------------------------------------------------------------


@dataclass(frozen=True)
class HolderAnalysisConfig:
    # Owners holding at least this fraction of supply count as meaningful holders.
    meaningful_min_supply_fraction: float = 1e-6
    top_n: int = 10
    listed_holders: int = 20  # non-excluded owners kept in the snapshot
    owner_lookups: int = 30  # biggest owners whose own accounts are read (program-owned?)


@dataclass(frozen=True)
class OwnerBalance:
    key: str  # the owner, or "unresolved:<token account>"
    owner: str | None
    accounts: tuple[str, ...]
    amount_raw: int


def merge_accounts(
    mint: str,
    largest: Sequence[TokenAccountBalance],
    token_accounts: Mapping[str, AccountRecord | None],
    scan: TokenAccountScan | None,
) -> dict[str, tuple[str | None, int]]:
    """Token account -> (owner, balance), from the largest accounts and the scan."""
    merged: dict[str, tuple[str | None, int]] = {}
    for bal in largest:
        record = token_accounts.get(bal.address)
        if record is not None and record.token_mint not in (None, mint):
            continue  # not an account of this mint: never counted
        merged[bal.address] = (record.token_owner if record else None, bal.amount_raw)
    for acc in scan.accounts if scan else ():
        merged[acc.address] = (acc.owner, acc.amount_raw)
    return merged


def aggregate_owners(merged: Mapping[str, tuple[str | None, int]]) -> list[OwnerBalance]:
    """Balances summed per owner wallet, largest first (ties by owner, deterministically)."""
    groups: dict[str, list[tuple[str, int]]] = {}
    owners: dict[str, str | None] = {}
    for address, (owner, amount) in merged.items():
        key = owner or f"unresolved:{address}"
        groups.setdefault(key, []).append((address, amount))
        owners[key] = owner
    out = [
        OwnerBalance(
            key=key,
            owner=owners[key],
            accounts=tuple(sorted(a for a, _ in items)),
            amount_raw=sum(v for _, v in items),
        )
        for key, items in groups.items()
    ]
    out.sort(key=lambda o: (-o.amount_raw, o.key))
    return out


def _classify(
    owner: str | None,
    eligible_pools: set[str],
    other_pools: set[str],
    raydium_pool: bool,
    program_owned: bool | None,
) -> tuple[HolderKind, str | None]:
    if owner is None:
        return "holder", "owner couldn't be resolved (counted)"
    if owner == INCINERATOR:
        return "burn", "Solana incinerator (burned)"
    if owner in eligible_pools:
        return "liquidity_pool", f"vault of DEX pool {owner}, reported for this mint"
    if owner == RAYDIUM_AMM_V4_AUTHORITY:
        if raydium_pool:
            return "liquidity_pool", "Raydium AMM v4 vaults (DEX data reports a Raydium pool)"
        return (
            "holder",
            "Raydium AMM v4 authority, but no usable Raydium pool is reported (counted)",
        )
    if owner in other_pools:
        return "holder", "DEX pool that isn't a usable market (counted)"
    if program_owned:
        return "holder", "program-owned account, not identified (counted)"
    return "holder", None


def analyze(
    data: ChainData,
    pools: Sequence[KnownPool] = (),
    config: HolderAnalysisConfig | None = None,
) -> OnchainSafetySnapshot:
    """Deterministic holder breakdown (see the module docstring for the rules)."""
    cfg = config or HolderAnalysisConfig()
    mint = data.mint
    supply_raw = mint.supply_raw if mint else data.supply_raw
    decimals = mint.decimals if mint else data.decimals
    holders_ok = data.largest is not None and supply_raw is not None and decimals is not None
    largest = data.largest or []
    eligible = {p.address for p in pools if p.eligible}
    others = {p.address for p in pools if not p.eligible} - eligible
    raydium = any(p.eligible and p.dex.lower() == "raydium" for p in pools)
    merged = (
        merge_accounts(data.address, largest, data.token_accounts, data.scan) if holders_ok else {}
    )
    supply_raw = supply_raw or 0
    decimals = decimals or 0

    entries: list[HolderEntry] = []
    for o in aggregate_owners(merged):
        record = data.owners.get(o.owner) if o.owner else None
        program_owned = record.program_owner != SYSTEM_PROGRAM if record is not None else None
        kind, label = _classify(o.owner, eligible, others, raydium, program_owned)
        entries.append(
            HolderEntry(
                owner=o.owner,
                token_accounts=list(o.accounts),
                amount=o.amount_raw / 10**decimals,
                pct_of_supply=100 * o.amount_raw / supply_raw if supply_raw else 0.0,
                kind=kind,
                label=label,
                program_owned=program_owned,
            )
        )
    holders = [e for e in entries if e.kind == "holder"]
    excluded = [e for e in entries if e.kind != "holder"]
    unresolved = sum(e.owner is None for e in entries)

    scan = data.scan if holders_ok else None
    reasons: list[str] = []
    scan_complete = scan is not None and scan.complete
    if scan is not None and scan.complete:
        scanned = {a.address for a in scan.accounts}
        missed = [
            b.address
            for b in largest
            if b.amount_raw > 0 and b.address in merged and b.address not in scanned
        ]
        if missed:
            # E.g. the provider's index lags the chain: a "complete" scan that lacks
            # accounts the RPC shows can't be trusted as complete.
            reasons.append(
                f"the holder scan missed {len(missed)} of the largest token accounts, so it "
                "isn't treated as complete"
            )
            scan_complete = False
    source: ConcentrationSource = (
        "unavailable"
        if not holders_ok
        else "largest_accounts"
        if scan is None
        else "full_scan"
        if scan_complete
        else "partial_scan"
    )
    reliable = holders_ok
    if not holders_ok:
        why = data.holders_error or "total supply unknown"
        reasons.append(f"holder data is unavailable ({why})")
    elif supply_raw == 0:
        reasons.append("total supply is zero, so holder shares can't be computed")
        reliable = False
    elif not merged:
        reasons.append("no token accounts could be read")
        reliable = False
    if unresolved:
        reasons.append(f"{unresolved} of the largest accounts' owners couldn't be resolved")
        reliable = False
    if source == "largest_accounts":
        reasons.append(
            f"no full holder scan is available ({data.provider}): concentration comes from "
            f"the {len(largest)} largest token accounts only, and one wallet can own "
            "several accounts, so shares are lower bounds"
        )
    elif source == "partial_scan" and scan is not None and not scan.complete:
        reasons.append(
            f"the holder scan stopped at the page cap ({scan.pages_read} pages, "
            f"{len(scan.accounts):,} accounts): shares and holder counts are lower bounds"
        )
    if scan is not None and scan.skipped_rows:
        reasons.append(f"{scan.skipped_rows} scanned rows were malformed and skipped")
    if not eligible and holders_ok:
        reasons.append(
            "no usable DEX pool was reported for this mint, so no pool vaults were excluded "
            "and concentration may be overstated"
        )

    counted = [e for e in holders if e.amount > 0]
    supply = supply_raw / float(10**decimals) if holders_ok or mint else None
    threshold = (supply or 0.0) * cfg.meaningful_min_supply_fraction
    count_complete = source == "full_scan" and not (scan and scan.skipped_rows)
    return OnchainSafetySnapshot(
        canonical_id=f"solana:{data.address}",
        mint=data.address,
        provider=data.provider,
        fetched_at=data.fetched_at,
        authorities_available=mint is not None,
        authorities_error=data.mint_error,
        token_program=mint.program if mint else None,
        token_program_id=mint.program_id if mint else None,
        decimals=mint.decimals if mint else data.decimals,
        supply=mint.supply if mint else supply,
        mint_authority=mint.mint_authority if mint else None,
        freeze_authority=mint.freeze_authority if mint else None,
        extensions=mint.extensions if mint else [],
        extension_state=mint.extension_state if mint else {},
        holders_available=holders_ok,
        holders_error=data.holders_error,
        holders=holders[: cfg.listed_holders],
        excluded=excluded,
        top1_pct=holders[0].pct_of_supply if holders and reliable else None,
        top10_pct=(
            sum(e.pct_of_supply for e in holders[: cfg.top_n]) if holders and reliable else None
        ),
        excluded_pct=sum(e.pct_of_supply for e in excluded),
        holder_count=len(counted) if scan is not None else None,
        meaningful_holder_count=(
            sum(e.amount >= threshold for e in counted) if scan is not None else None
        ),
        holder_count_complete=count_complete,
        concentration_source=source,
        concentration_lower_bound=not count_complete,
        token_accounts_seen=len(merged),
        scan_pages_read=scan.pages_read if scan else None,
        scan_max_pages=scan.max_pages if scan else None,
        largest_accounts_seen=len(largest),
        pool_addresses_used=sorted(
            {e.owner for e in excluded if e.kind == "liquidity_pool" and e.owner}
        ),
        concentration_reliable=reliable,
        holder_data_complete=not reasons,
        incomplete_reasons=reasons,
    )


# --- Providers ------------------------------------------------------------------------------


class SolanaChainProvider(Protocol):
    name: str

    async def fetch_mint(self, mint: str) -> MintInfo:
        """The mint account, or `AssetNotFoundError` if it doesn't exist / isn't a mint."""
        ...

    async def fetch_largest_accounts(self, mint: str) -> list[TokenAccountBalance]: ...

    async def fetch_accounts(self, addresses: Sequence[str]) -> dict[str, AccountRecord | None]:
        """Accounts by address; None for accounts that don't exist."""
        ...

    async def scan_token_accounts(self, mint: str) -> TokenAccountScan | None:
        """Every token account of the mint (up to the provider's page cap), or None when
        the provider can't list them."""
        ...

    async def fetch_supply(self, mint: str) -> tuple[int, int]:
        """(total supply in base units, decimals)."""
        ...


class SolanaRpcProvider:
    """Standard Solana JSON-RPC. The URL may contain an API key: it is never put in errors."""

    name = "Solana RPC"

    def __init__(
        self,
        url: str,
        timeout: float = 10.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self._url = url
        self.timeout = timeout
        self._transport = transport
        self._ids = itertools.count(1)

    def __repr__(self) -> str:  # never show the URL (it may hold a key)
        return f"{type(self).__name__}(name={self.name!r})"

    async def fetch_mint(self, mint: str) -> MintInfo:
        result = await self._call("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        value = result.get("value") if isinstance(result, dict) else None
        if value is None:
            raise AssetNotFoundError(f"no account exists at {mint}")
        return parse_mint(mint, value, self.name)

    async def fetch_largest_accounts(self, mint: str) -> list[TokenAccountBalance]:
        result = await self._call("getTokenLargestAccounts", [mint, {"commitment": "confirmed"}])
        rows = result.get("value") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            raise MarketDataUnavailableError(f"{self.name} returned malformed largest accounts")
        out: list[TokenAccountBalance] = []
        for row in rows:
            amount = _int(row.get("amount")) if isinstance(row, dict) else None
            address = row.get("address") if isinstance(row, dict) else None
            if not isinstance(address, str) or amount is None:
                raise MarketDataUnavailableError(f"{self.name} returned a malformed account")
            out.append(TokenAccountBalance(address=address, amount_raw=amount))
        return out

    async def fetch_accounts(self, addresses: Sequence[str]) -> dict[str, AccountRecord | None]:
        unique = list(dict.fromkeys(addresses))
        out: dict[str, AccountRecord | None] = {}
        for chunk in (unique[i : i + 100] for i in range(0, len(unique), 100)):
            result = await self._call("getMultipleAccounts", [chunk, {"encoding": "jsonParsed"}])
            values = result.get("value") if isinstance(result, dict) else None
            if not isinstance(values, list) or len(values) != len(chunk):
                raise MarketDataUnavailableError(f"{self.name} returned malformed accounts")
            for address, value in zip(chunk, values, strict=True):
                out[address] = parse_account(address, value)
        return out

    async def scan_token_accounts(self, mint: str) -> TokenAccountScan | None:
        return None  # standard RPC has no efficient way to list every holder

    async def fetch_supply(self, mint: str) -> tuple[int, int]:
        result = await self._call("getTokenSupply", [mint])
        value = result.get("value") if isinstance(result, dict) else None
        amount = _int(value.get("amount")) if isinstance(value, dict) else None
        decimals = value.get("decimals") if isinstance(value, dict) else None
        if amount is None or not isinstance(decimals, int) or isinstance(decimals, bool):
            raise MarketDataUnavailableError(f"{self.name} returned a malformed token supply")
        return amount, decimals

    async def _call(self, method: str, params: Any) -> Any:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        try:
            async with httpx2.AsyncClient(
                timeout=self.timeout, transport=self._transport
            ) as client:
                response = await client.post(self._url, json=payload)
        except httpx2.TimeoutException as exc:
            raise MarketDataUnavailableError(f"{self.name} request timed out") from exc
        except httpx2.HTTPError as exc:
            raise MarketDataUnavailableError(f"could not reach {self.name}") from exc
        if response.status_code == 429:
            raise MarketDataUnavailableError(f"{self.name} rate limit reached")
        if response.status_code in (401, 403):
            raise MarketDataUnavailableError(f"{self.name} rejected the API key")
        if response.status_code != 200:
            raise MarketDataUnavailableError(f"{self.name} returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise MarketDataUnavailableError(f"{self.name} returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise MarketDataUnavailableError(f"{self.name} returned an unexpected response")
        if isinstance(body.get("error"), dict):
            message = str(body["error"].get("message", "error"))
            if "not a Token mint" in message or "Invalid param" in message:
                raise AssetNotFoundError(f"{method}: {message}")
            raise MarketDataUnavailableError(f"{self.name} {method} failed: {message}")
        if "result" not in body:
            raise MarketDataUnavailableError(f"{self.name} returned no result")
        return body["result"]


class HeliusProvider(SolanaRpcProvider):
    """Helius mainnet RPC (needs an API key) plus DAS `getTokenAccounts`, paginated, to
    list every token account of a mint."""

    name = "Helius"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://mainnet.helius-rpc.com",
        max_pages: int = 10,
        timeout: float = 10.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        super().__init__(f"{base_url}/?api-key={api_key}", timeout, transport)
        self.max_pages = max_pages

    async def scan_token_accounts(self, mint: str) -> TokenAccountScan:
        accounts: list[ScannedAccount] = []
        skipped = pages = 0
        complete = False
        for page in range(1, self.max_pages + 1):
            result = await self._call(
                "getTokenAccounts",
                {
                    "mint": mint,
                    "limit": SCAN_PAGE_SIZE,
                    "page": page,
                    "options": {"showZeroBalance": False},
                },
            )
            rows = result.get("token_accounts") if isinstance(result, dict) else None
            if not isinstance(rows, list):
                raise MarketDataUnavailableError(f"{self.name} returned malformed token accounts")
            pages += 1
            for row in rows:
                if isinstance(row, dict) and row.get("mint") not in (None, mint):
                    continue  # another mint's account: never counted
                fields = row if isinstance(row, dict) else {}
                address, owner = fields.get("address"), fields.get("owner")
                amount = _int(fields.get("amount"))
                if not isinstance(address, str) or not isinstance(owner, str) or amount is None:
                    skipped += 1
                    continue
                accounts.append(ScannedAccount(address=address, owner=owner, amount_raw=amount))
            if len(rows) < SCAN_PAGE_SIZE:  # a short page is the last one
                complete = True
                break
        return TokenAccountScan(
            accounts=accounts,
            pages_read=pages,
            max_pages=self.max_pages,
            complete=complete,
            skipped_rows=skipped,
        )


def parse_mint(mint: str, value: Mapping[str, Any], provider: str) -> MintInfo:
    program_id = value.get("owner")
    data = value.get("data")
    parsed = data.get("parsed") if isinstance(data, dict) else None
    info = parsed.get("info") if isinstance(parsed, dict) else None
    if not isinstance(program_id, str):
        raise MarketDataUnavailableError(f"{provider} returned a malformed account")
    if program_id not in (TOKEN_PROGRAM, TOKEN_2022_PROGRAM):
        raise AssetNotFoundError(f"{mint} is not a token mint (owned by {program_id})")
    if not isinstance(parsed, dict) or parsed.get("type") != "mint" or not isinstance(info, dict):
        raise AssetNotFoundError(f"{mint} is not a token mint")
    decimals, supply = info.get("decimals"), _int(info.get("supply"))
    if not isinstance(decimals, int) or isinstance(decimals, bool) or supply is None:
        raise MarketDataUnavailableError(f"{provider} returned a malformed mint account")
    extensions: list[str] = []
    state: dict[str, Any] = {}
    for ext in info.get("extensions") or []:
        if isinstance(ext, dict) and isinstance(ext.get("extension"), str):
            extensions.append(ext["extension"])
            state[ext["extension"]] = ext.get("state")
    return MintInfo(
        mint=mint,
        program_id=program_id,
        program="token_2022" if program_id == TOKEN_2022_PROGRAM else "spl_token",
        decimals=decimals,
        supply_raw=supply,
        mint_authority=_str(info.get("mintAuthority")),
        freeze_authority=_str(info.get("freezeAuthority")),
        extensions=extensions,
        extension_state=state,
    )


def parse_account(address: str, value: Any) -> AccountRecord | None:
    if not isinstance(value, dict) or not isinstance(value.get("owner"), str):
        return None
    data = value.get("data")
    parsed = data.get("parsed") if isinstance(data, dict) else None
    info = parsed.get("info") if isinstance(parsed, dict) else None
    token_owner = token_mint = None
    if isinstance(parsed, dict) and parsed.get("type") == "account" and isinstance(info, dict):
        token_owner, token_mint = _str(info.get("owner")), _str(info.get("mint"))
    return AccountRecord(
        address=address,
        program_owner=value["owner"],
        token_owner=token_owner,
        token_mint=token_mint,
    )


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


# --- Service --------------------------------------------------------------------------------


@dataclass
class _Entry:
    expires_at: float
    data: ChainData


@dataclass
class SolanaSafetyService:
    """Caches chain data per mint, rate-limits, and de-duplicates concurrent requests.

    Failures are never cached; a mint that doesn't exist is remembered for `not_found_ttl`.
    Analysis (which depends on the pools to exclude) runs on every call.
    """

    provider: SolanaChainProvider
    cache_ttl: float = 120.0
    not_found_ttl: float = 300.0
    max_calls_per_minute: int = 20
    analysis: HolderAnalysisConfig = field(default_factory=HolderAnalysisConfig)
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.max_calls_per_minute, 60.0, self.clock)
        self._cache: dict[str, _Entry] = {}
        self._not_found: dict[str, tuple[float, AssetNotFoundError]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def reset(self) -> None:
        self._cache.clear()
        self._not_found.clear()
        self._locks.clear()
        self._limiter = RateLimiter(self.max_calls_per_minute, 60.0, self.clock)

    async def get_snapshot(
        self, mint: str, pools: Sequence[KnownPool] = ()
    ) -> OnchainSafetySnapshot:
        return analyze(await self.get_chain_data(mint), pools, self.analysis)

    async def get_chain_data(self, mint: str) -> ChainData:
        mint = mint.strip()
        if not is_solana_address(mint):
            raise InvalidRequestError(f"{mint!r} is not a valid Solana mint address")
        if (hit := self._cached(mint)) is not None:
            return hit
        async with self._locks.setdefault(mint, asyncio.Lock()):
            if (hit := self._cached(mint)) is not None:
                return hit
            if not self._limiter.try_acquire():
                raise MarketDataUnavailableError(
                    f"UpScale's {self.provider.name} request limit was reached; try again "
                    "in a minute"
                )
            try:
                data = await self._fetch(mint)
            except AssetNotFoundError as exc:
                self._not_found[mint] = (self.clock() + self.not_found_ttl, exc)
                raise
            if data.mint is not None and data.largest is not None:
                # Only complete data is cached: a failed component is retried next time.
                self._cache[mint] = _Entry(self.clock() + self.cache_ttl, data)
            return data

    async def _fetch(self, mint: str) -> ChainData:
        """Authorities and holders are fetched independently: one failing (e.g. a rate
        limit on the holder endpoint) never discards the other. Only when both fail, or the
        address isn't a mint, is the request an error."""
        info_res, largest_res = await asyncio.gather(
            self.provider.fetch_mint(mint),
            self.provider.fetch_largest_accounts(mint),
            return_exceptions=True,
        )
        for res in (info_res, largest_res):
            if isinstance(res, BaseException) and not isinstance(res, MarketDataError):
                raise res  # a bug, not a provider failure
        if isinstance(info_res, AssetNotFoundError):
            raise info_res  # not a mint at all
        info = info_res if isinstance(info_res, MintInfo) else None
        mint_error = str(info_res) if isinstance(info_res, MarketDataError) else None

        data = ChainData(
            address=mint,
            mint=info,
            mint_error=mint_error,
            largest=None,
            provider=self.provider.name,
            fetched_at=self.now(),
        )
        if isinstance(largest_res, MarketDataError):
            data.holders_error = str(largest_res)
        else:
            assert isinstance(largest_res, list)
            try:
                data = await self._holders(data, largest_res)
            except MarketDataError as exc:
                data.holders_error = str(exc)  # partial holder data is never kept
        if data.mint is None and data.largest is None:
            raise MarketDataUnavailableError(
                f"on-chain data unavailable (authorities: {data.mint_error}; holders: "
                f"{data.holders_error})"
            )
        return data

    async def _holders(self, data: ChainData, largest: list[TokenAccountBalance]) -> ChainData:
        mint = data.address
        token_accounts = await self.provider.fetch_accounts([a.address for a in largest])
        scan = await self.provider.scan_token_accounts(mint)
        # The biggest owners' own accounts tell wallets from program-owned accounts.
        ranked = aggregate_owners(merge_accounts(mint, largest, token_accounts, scan))
        lookups = [o.owner for o in ranked if o.owner][: self.analysis.owner_lookups]
        owners = await self.provider.fetch_accounts(lookups) if lookups else {}
        supply_raw = decimals = None
        if data.mint is None:  # shares need the total supply
            supply_raw, decimals = await self.provider.fetch_supply(mint)
        return data.model_copy(
            update={
                "largest": largest,
                "token_accounts": token_accounts,
                "scan": scan,
                "owners": owners,
                "supply_raw": supply_raw,
                "decimals": decimals,
            }
        )

    def _cached(self, mint: str) -> ChainData | None:
        missing = self._not_found.get(mint)
        if missing and missing[0] > self.clock():
            raise missing[1]
        entry = self._cache.get(mint)
        return entry.data if entry and entry.expires_at > self.clock() else None
