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


@dataclass
class FakeRpc:
    """address -> what getAccountInfo answers: a dict value, None (no account), or a
    ("http", status) / ("error", message) / ("timeout",) / ("raw", body) instruction."""

    accounts: dict[str, Any] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    slot: int = 123

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        method, params = payload["method"], payload["params"]
        self.calls.append(method)
        assert method == "getAccountInfo", method
        spec = self.accounts.get(params[0])
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
        result = {"context": {"apiVersion": "2.2.0", "slot": self.slot}, "value": spec}
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})


def make_service(
    tmp_path: Path,
    rpc: FakeRpc | None = None,
    clock: Clock | None = None,
    provider: bool = True,
    db_name: str = "safety.sqlite3",
    **settings: Any,
) -> tuple[SafetyService, Clock, FakeRpc]:
    clock = clock or Clock()
    rpc = rpc if rpc is not None else FakeRpc()
    cfg = SafetySettings(db_path=str(tmp_path / db_name), **settings)
    repo = SafetyRepository(cfg.db_path)
    guard = RequestGuard(cfg, repo, now=clock.now, monotonic=clock.monotonic, sleep=clock.sleep,
                         rng=lambda: 0.0)  # fmt: skip
    svc = SafetyService(
        cfg,
        repo=repo,
        now=clock.now,
        rpc_url=RPC_URL if provider else None,
        transport=httpx2.MockTransport(rpc.handle),
        guard=guard,
    )
    return svc, clock, rpc
