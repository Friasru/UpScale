"""Offline fakes for Radar tests: a scriptable Solana JSON-RPC (MockTransport) and a clock.

No test using these ever reaches a network: every request is answered by `FakeChain`.
"""

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2

from upscale.services.radar.config import RadarSettings
from upscale.services.radar.provider import RequestGuard
from upscale.services.radar.repository import RadarRepository
from upscale.services.radar.service import RadarService
from upscale.services.solana_chain import SYSTEM_PROGRAM, TOKEN_PROGRAM

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def addr(tag: str) -> str:
    """A valid base58 address starting with `tag` (no 0 O I l)."""
    value = (tag + "1" * 44)[:44]
    assert not set(value) & set("0OIl"), tag
    return value


MINT = addr("MintAAA")
POOL = addr("PooLAAA")


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


def tx(
    sig: str,
    block_time: datetime,
    fee_payer: str,
    pre: Sequence[tuple[str, int]] = (),
    post: Sequence[tuple[str, int]] = (),
    mint: str = MINT,
    err: Any = None,
    init_mint: str | None = None,
    signers: Sequence[str] | None = None,
    funds: tuple[str, str] | None = None,  # (source, destination) SOL transfer
    programs: Sequence[str] = (),  # program ids invoked at top level
) -> dict[str, Any]:
    signer_set = set(signers) if signers is not None else {fee_payer}
    keys = [fee_payer] + sorted({o for o, _ in [*pre, *post]} | signer_set - {fee_payer})
    index = {k: i for i, k in enumerate(keys)}

    def balances(rows: Sequence[tuple[str, int]]) -> list[dict[str, Any]]:
        return [
            {"accountIndex": index[o], "mint": mint, "owner": o, "programId": TOKEN_PROGRAM,
             "uiTokenAmount": {"amount": str(a), "decimals": 6}}
            for o, a in rows
        ]  # fmt: skip

    inner: list[dict[str, Any]] = []
    if init_mint:
        inner.append({"index": 0, "instructions": [{"program": "spl-token", "programId": TOKEN_PROGRAM,
                      "parsed": {"type": "initializeMint2", "info": {"mint": init_mint, "decimals": 6}}}]})  # fmt: skip
    top: list[dict[str, Any]] = []
    if funds:
        top.append({"program": "system", "programId": SYSTEM_PROGRAM,
                    "parsed": {"type": "transfer", "info": {"source": funds[0], "destination": funds[1],
                                                            "lamports": 10}}})  # fmt: skip
    top += [{"programId": prog, "accounts": [], "data": ""} for prog in programs]
    return {
        "slot": int(block_time.timestamp()),
        "blockTime": int(block_time.timestamp()),
        "meta": {"err": err, "preTokenBalances": balances(pre), "postTokenBalances": balances(post),
                 "innerInstructions": inner},  # fmt: skip
        "transaction": {
            "signatures": [sig],
            "message": {
                "accountKeys": [{"pubkey": k, "signer": k in signer_set, "writable": True} for k in keys],
                "instructions": top,
            },
        },
    }  # fmt: skip


@dataclass
class FakeChain:
    mint: str = MINT
    decimals: int = 6
    supply: int = 1_000_000
    holders: dict[str, int] = field(default_factory=dict)  # owner -> amount (one account each)
    das_pages_limit: int | None = None  # pretend more pages exist beyond this many accounts
    history: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )  # address -> txs (oldest first)
    fail: Callable[[str], int | str | None] | None = None  # method -> status / "timeout"
    program_owned: set[str] = field(default_factory=set)  # owners whose account a program owns
    calls: Counter[str] = field(default_factory=Counter)
    params: list[tuple[str, Any]] = field(default_factory=list)

    def add(self, address: str, *txs: dict[str, Any]) -> None:
        self.history.setdefault(address, []).extend(txs)

    def _accounts(self) -> list[tuple[str, str, int]]:
        return [(addr("Ac" + o[:12]), o, a) for o, a in sorted(self.holders.items()) if a > 0]

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        method, params = body["method"], body["params"]
        self.calls[method] += 1
        self.params.append((method, params))
        if self.fail is not None:
            how = self.fail(method)
            if how == "timeout":
                raise httpx2.ReadTimeout("timed out", request=request)
            if isinstance(how, int):
                return httpx2.Response(how, json={"error": "nope"})
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": body["id"], "result": self.result(method, params)}
        )

    def result(self, method: str, params: Any) -> Any:
        accounts = self._accounts()
        if method == "getAccountInfo":
            return {"value": {"owner": TOKEN_PROGRAM, "data": {"parsed": {"type": "mint", "info": {
                "decimals": self.decimals, "supply": str(self.supply), "mintAuthority": None,
                "freezeAuthority": None}}}}}  # fmt: skip
        if method == "getTokenLargestAccounts":
            top = sorted(accounts, key=lambda x: -x[2])[:20]
            return {"value": [{"address": a, "amount": str(n)} for a, _, n in top]}
        if method == "getMultipleAccounts":
            owners = {a: o for a, o, _ in accounts}
            values = []
            for address in params[0]:
                if address in owners:
                    values.append({"owner": TOKEN_PROGRAM, "data": {"parsed": {"type": "account",
                                   "info": {"owner": owners[address], "mint": self.mint}}}})  # fmt: skip
                elif address in self.program_owned:
                    values.append({"owner": addr("SomeProgram"), "data": ["", "base64"]})
                else:
                    values.append({"owner": SYSTEM_PROGRAM, "data": ["", "base64"]})
            return {"value": values}
        if method == "getTokenAccounts":
            page, limit = params["page"], params["limit"]
            rows = [
                {"address": a, "owner": o, "amount": n, "mint": self.mint} for a, o, n in accounts
            ]
            if self.das_pages_limit is not None:
                rows = rows + [rows[0]] * (limit * 10)  # endless pages: never complete
            return {"token_accounts": rows[(page - 1) * limit : page * limit]}
        if method == "getSignaturesForAddress":
            address, opts = params
            txs = list(reversed(self.history.get(address, [])))  # newest first
            sigs = [t["transaction"]["signatures"][0] for t in txs]
            start = sigs.index(opts["before"]) + 1 if opts.get("before") in sigs else 0
            end = sigs.index(opts["until"]) if opts.get("until") in sigs else len(sigs)
            rows = [
                {"signature": t["transaction"]["signatures"][0], "slot": t["slot"],
                 "blockTime": t["blockTime"], "err": t["meta"]["err"]}
                for t in txs[start:end]
            ][: opts["limit"]]  # fmt: skip
            return rows
        if method == "getTransaction":
            for txs in self.history.values():
                for t in txs:
                    if t["transaction"]["signatures"][0] == params[0]:
                        return t
            return None
        raise AssertionError(f"unexpected method {method}")


def make_service(
    tmp_path: Any,
    chain: FakeChain | None,
    clock: Clock,
    **overrides: Any,
) -> RadarService:
    settings = RadarSettings(db_path=str(tmp_path / "radar.sqlite3"), **overrides)
    repo = RadarRepository(settings.db_path)
    guard = RequestGuard(settings, repo, now=clock.now, monotonic=clock.monotonic,
                         sleep=clock.sleep, rng=lambda: 0.0)  # fmt: skip
    return RadarService(
        settings, repo, now=clock.now,
        helius_api_key="test-key" if chain else None,
        transport=chain.transport() if chain else None, guard=guard,
    )  # fmt: skip
