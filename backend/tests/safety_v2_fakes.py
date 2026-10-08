"""Offline fakes for Safety V2 tests: a scriptable Solana JSON-RPC (MockTransport) and a
clock. No test using these reaches a network: every request is answered by `FakeRpc`."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2

from upscale.services.safety_v2.config import SafetySettings
from upscale.services.safety_v2.provider import RequestGuard
from upscale.services.safety_v2.repository import SafetyRepository
from upscale.services.safety_v2.service import SafetyService
from upscale.services.solana_chain import TOKEN_2022_PROGRAM, TOKEN_PROGRAM

T0 = datetime(2026, 10, 1, tzinfo=UTC)
RPC_URL = "https://rpc.invalid/?api-key=SECRETKEY"
DEX_URL = "https://dex.invalid"
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def addr(tag: str) -> str:
    """A valid base58 address starting with `tag` (no 0 O I l)."""
    value = (tag + "1" * 44)[:44]
    assert not set(value) & set("0OIl"), tag
    return value


MINT = addr("MintAAA")
AUTH = addr("AuthAAA")
FREEZER = addr("FreezeAAA")
OTHER_PROGRAM = addr("ProgramXX")


class Clock:
    def __init__(self, start: datetime = T0):
        self.t = start
        self.mono = 0.0
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.mono += seconds


def mint_value(
    *,
    program: str = TOKEN_PROGRAM,
    mint_authority: Any = None,
    freeze_authority: Any = None,
    supply: Any = "1000000000",
    decimals: Any = 6,
    initialized: Any = True,
    kind: str = "mint",
    extensions: Any = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "decimals": decimals,
        "freezeAuthority": freeze_authority,
        "isInitialized": initialized,
        "mintAuthority": mint_authority,
        "supply": supply,
    }
    if extensions is not None:
        info["extensions"] = extensions
    name = "spl-token-2022" if program == TOKEN_2022_PROGRAM else "spl-token"
    return {
        "data": {"parsed": {"info": info, "type": kind}, "program": name, "space": 82},
        "executable": False,
        "lamports": 1461600,
        "owner": program,
        "rentEpoch": 18446744073709551615,
        "space": 82,
    }


_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def num_addr(prefix: str, i: int) -> str:
    """A unique valid base58 address for (prefix, i), fixed width so none collide."""
    digits = ""
    while True:
        i, r = divmod(i, 58)
        digits = _B58[r] + digits
        if i == 0:
            break
    return addr(prefix + digits.rjust(6, "1"))


@dataclass
class HolderChain:
    """What the holder RPC methods answer for MINT. `accounts`: token account -> (owner,
    amount); `largest`: the token accounts getTokenLargestAccounts returns (None: the 20
    largest of `accounts`); `owner_programs`: owner -> program owning its account (missing
    key: no account). `fail`: method -> an instruction as for `FakeRpc.accounts`."""

    supply: int = 1_000_000_000
    decimals: int = 6
    accounts: dict[str, tuple[str, int]] = field(default_factory=dict)
    largest: list[str] | None = None
    owner_programs: dict[str, str] = field(default_factory=dict)
    fail: dict[str, Any] = field(default_factory=dict)
    extra_rows: list[Any] = field(default_factory=list)  # appended to the scan (malformed...)

    def add(self, owner: str, amount: int, account: str | None = None) -> str:
        account = account or num_addr("Tacc", len(self.accounts))
        self.accounts[account] = (owner, amount)
        return account

    def result(self, method: str, params: Any) -> Any:
        ctx = {"context": {"slot": 123}}
        if method == "getTokenSupply":
            return ctx | {"value": {"amount": str(self.supply), "decimals": self.decimals}}
        if method == "getTokenLargestAccounts":
            keys = self.largest
            if keys is None:
                ranked = sorted(self.accounts.items(), key=lambda kv: (-kv[1][1], kv[0]))
                keys = [k for k, _ in ranked[:20]]
            return ctx | {"value": [{"address": k, "amount": str(self.accounts[k][1])}
                                    for k in keys]}  # fmt: skip
        if method == "getMultipleAccounts":
            return ctx | {"value": [self._account(a) for a in params[0]]}
        if method == "getTokenAccounts":
            assert params["limit"] == 1000
            rows: list[Any] = [{"address": k, "owner": o, "amount": a, "mint": MINT}
                               for k, (o, a) in sorted(self.accounts.items()) if a > 0]  # fmt: skip
            rows += self.extra_rows
            start = (params["page"] - 1) * 1000
            page = rows[start : start + 1000]
            return {"token_accounts": page, "total": len(page)}
        raise AssertionError(method)

    def _account(self, address: str) -> Any:
        if address in self.accounts:
            owner = self.accounts[address][0]
            info = {"owner": owner, "mint": MINT, "tokenAmount": {"amount": "0"}}
            return {"owner": TOKEN_PROGRAM, "data": {"parsed": {"type": "account", "info": info}}}
        program = self.owner_programs.get(address)
        if program is None:
            return None
        return {"owner": program, "data": ["", "base64"], "executable": False}


@dataclass
class FakeRpc:
    """address -> what getAccountInfo answers: a dict value, None (no account), or a
    ("http", status) / ("error", message) / ("timeout",) / ("raw", body) instruction.
    With `holders`, the Phase 2 holder methods are answered from it too."""

    accounts: dict[str, Any] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    slot: int = 123
    holders: HolderChain | None = None
    # DEX Screener token-pairs answer: a list of rows, or an instruction as above. None:
    # the DEX endpoint must not be called.
    dex: Any = None

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            assert request.url.path == f"/token-pairs/v1/solana/{MINT}", request.url
            self.calls.append("dex.tokenPairs")
            assert self.dex is not None, "unexpected DEX request"
            if isinstance(self.dex, tuple):
                if self.dex[0] == "http":
                    return httpx2.Response(self.dex[1], json={})
                if self.dex[0] == "timeout":
                    raise httpx2.ConnectTimeout("timed out", request=request)
                if self.dex[0] == "raw":
                    return httpx2.Response(200, json=self.dex[1])
            return httpx2.Response(200, json=self.dex)
        payload = json.loads(request.content)
        method, params = payload["method"], payload["params"]
        self.calls.append(method)
        if method != "getAccountInfo" and self.holders is not None:
            spec = self.holders.fail.get(method)
            if spec is None:
                result = self.holders.result(method, params)
                return httpx2.Response(
                    200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result}
                )
            return self._instruction(spec, payload, request)
        assert method == "getAccountInfo", method
        spec = self.accounts.get(params[0])
        if isinstance(spec, tuple):
            return self._instruction(spec, payload, request)
        result = {"context": {"apiVersion": "2.2.0", "slot": self.slot}, "value": spec}
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    @staticmethod
    def _instruction(spec: Any, payload: Any, request: httpx2.Request) -> httpx2.Response:
        if isinstance(spec, tuple):
            if spec[0] == "http":
                return httpx2.Response(spec[1], json={})
            if spec[0] == "error":
                body = {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {"code": -32000, "message": spec[1]},
                }
                return httpx2.Response(200, json=body)
            if spec[0] == "timeout":
                raise httpx2.ConnectTimeout("timed out", request=request)
            if spec[0] == "raw":
                return httpx2.Response(
                    200, json={"jsonrpc": "2.0", "id": payload["id"], "result": spec[1]}
                )
        raise AssertionError(spec)


def make_service(
    tmp_path: Path,
    rpc: FakeRpc | None = None,
    clock: Clock | None = None,
    provider: bool = True,
    db_name: str = "safety.sqlite3",
    helius: bool = False,
    dex: bool = False,
    **settings: Any,
) -> tuple[SafetyService, Clock, FakeRpc]:
    clock = clock or Clock()
    rpc = rpc if rpc is not None else FakeRpc()
    if dex:
        settings.setdefault("dex_url", DEX_URL)
    cfg = SafetySettings(db_path=str(tmp_path / db_name), **settings)
    repo = SafetyRepository(cfg.db_path)
    guard = RequestGuard(cfg, repo, now=clock.now, monotonic=clock.monotonic, sleep=clock.sleep,
                         rng=lambda: 0.0)  # fmt: skip
    svc = SafetyService(
        cfg,
        repo=repo,
        now=clock.now,
        rpc_url=RPC_URL if provider and not helius else None,
        helius_api_key="SECRETKEY" if provider and helius else None,
        transport=httpx2.MockTransport(rpc.handle),
        guard=guard,
    )
    return svc, clock, rpc


def pair_row(
    pair: str,
    *,
    base: str = MINT,
    quote: str = SOL,
    liquidity: Any = 200_000.0,
    price: Any = "0.5",
    txns: int | None = 50,
    volume: float | None = 10_000.0,
    created: datetime | None = T0 - timedelta(days=30),
    chain: str = "solana",
    dex: str = "raydium",
    base_symbol: str = "TKN",
) -> dict[str, Any]:
    """One DEX Screener pair object."""
    row: dict[str, Any] = {
        "chainId": chain, "dexId": dex, "pairAddress": pair,
        "baseToken": {"address": base, "symbol": base_symbol, "name": "Token"},
        "quoteToken": {"address": quote, "symbol": "SOL" if quote == SOL else "Q"},
        "priceUsd": price, "liquidity": {"usd": liquidity},
    }  # fmt: skip
    if txns is not None:
        row["txns"] = {"h24": {"buys": txns, "sells": 0}}
    if volume is not None:
        row["volume"] = {"h24": volume}
    if created is not None:
        row["pairCreatedAt"] = int(created.timestamp() * 1000)
    return row
