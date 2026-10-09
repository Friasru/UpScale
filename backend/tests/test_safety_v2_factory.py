"""Safety V2's public construction (`service_from_env`) and conservative collection request
bound (`collection_request_bound`): service-level interfaces only, no rule or evidence
change. Offline: services are built, never connected (or answered by `FakeRpc`)."""

import asyncio
from pathlib import Path
from typing import Any

import httpx2
import pytest

from tests.safety_v2_fakes import (
    MINT,
    Clock,
    FakeRpc,
    HolderChain,
    addr,
    make_service,
    mint_value,
    num_addr,
    pair_row,
)
from upscale.services.safety_v2 import cli as safety_cli
from upscale.services.safety_v2.config import (
    DB_SCHEMA_VERSION,
    RULES_VERSION,
    SNAPSHOT_SCHEMA,
    SafetySettings,
)
from upscale.services.safety_v2.service import CollectionRequestBound, service_from_env

CID = f"solana:{MINT}"
SECRET = "TESTSECRET-factory-key"


def settings(tmp_path: Path, **kw: Any) -> SafetySettings:
    return SafetySettings(db_path=str(tmp_path / "s.sqlite3"), **kw)


def test_factory_selects_providers_like_the_cli(tmp_path: Path, capsys: Any) -> None:
    helius = service_from_env(settings(tmp_path), {"UPSCALE_HELIUS_API_KEY": SECRET,
                                                   "UPSCALE_SOLANA_RPC_URL": "https://x/y"})  # fmt: skip
    assert helius.provider is not None and helius.provider.name == "Helius"
    assert helius.provider.supports_scan and helius.dex is None
    rpc = service_from_env(settings(tmp_path), {"UPSCALE_SOLANA_RPC_URL": "https://rpc.invalid"})
    assert rpc.provider is not None and rpc.provider.name == "Solana RPC"
    assert not rpc.provider.supports_scan
    assert service_from_env(settings(tmp_path), {}).provider is None
    dex = service_from_env(settings(tmp_path, dex_url="https://dex.invalid"), {})
    assert dex.dex is not None and dex.provider is None
    env_dex = service_from_env(None, {"UPSCALE_SAFETY_V2_DB": str(tmp_path / "e.sqlite3"),
                                      "UPSCALE_SAFETY_V2_DEX_URL": "https://dex.invalid"})  # fmt: skip
    assert env_dex.dex is not None and env_dex.settings.db_path == str(tmp_path / "e.sqlite3")
    assert SECRET not in repr(helius.provider) + capsys.readouterr().out


def test_the_cli_delegates_to_the_public_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[SafetySettings] = []

    def fake(s: SafetySettings, env: Any = None) -> Any:
        seen.append(s)
        return "service"

    monkeypatch.setattr(safety_cli, "service_from_env", fake)
    s = settings(tmp_path)
    assert safety_cli._service(s) == "service" and seen == [s]


@pytest.mark.parametrize(
    ("helius", "dex", "holders", "market", "pages", "retries", "logical"),
    [
        (False, False, False, False, 2, 2, 1),  # mint only
        (False, False, True, False, 2, 2, 5),  # + 4 holder calls (no scan)
        (True, False, True, False, 2, 2, 7),  # + 2 DAS pages
        (True, False, True, False, 5, 2, 10),
        (False, True, False, True, 2, 2, 3),  # mint + DEX + conditional pool account
        (True, True, True, True, 2, 2, 9),
        (True, True, True, True, 2, 0, 9),
        (True, True, False, False, 2, 1, 1),
    ],
)
def test_collection_request_bound(
    tmp_path: Path, helius: bool, dex: bool, holders: bool, market: bool, pages: int,
    retries: int, logical: int,
) -> None:  # fmt: skip
    svc, _, _ = make_service(tmp_path, helius=helius, dex=dex, holder_max_pages=pages,
                             max_retries=retries)  # fmt: skip
    b = svc.collection_request_bound(holders=holders, market=market)
    assert b == CollectionRequestBound(logical, logical * (retries + 1))


def test_bound_without_providers(tmp_path: Path) -> None:
    svc, _, _ = make_service(tmp_path, provider=False, dex=True)
    assert svc.collection_request_bound() == CollectionRequestBound(1, 3)  # the DEX only
    none, _, _ = make_service(tmp_path, provider=False, db_name="n.sqlite3")
    assert none.collection_request_bound() == CollectionRequestBound(0, 0)


def _chain() -> HolderChain:
    chain = HolderChain()
    for i in range(30):
        chain.add(num_addr("Hdr", i), 1_000_000)
    return chain


@pytest.mark.parametrize("failing", [None, ("timeout",), ("http", 503)])
def test_a_real_collection_never_exceeds_the_bound(tmp_path: Path, failing: Any) -> None:
    pool = addr("PqqLAAA")
    rpc = FakeRpc({MINT: failing or mint_value()}, holders=_chain(), dex=[pair_row(pool)])
    if failing:
        rpc.holders.fail = {m: failing for m in ("getTokenSupply", "getTokenLargestAccounts")}  # type: ignore[union-attr]
        rpc.dex = failing
    svc, clock, _ = make_service(tmp_path, rpc, helius=True, dex=True)
    svc.add_target(MINT)
    bound = svc.collection_request_bound()
    got = asyncio.run(svc.collect(CID, holders=True, market=True))
    assert got.requests <= bound.attempt_max
    assert len(rpc.calls) == got.requests


def test_versions_are_unchanged() -> None:
    assert (DB_SCHEMA_VERSION, RULES_VERSION, SNAPSHOT_SCHEMA) == (4, "4", "safety.snapshot.v2")


def test_clock_and_transport_fakes_are_offline() -> None:
    assert isinstance(httpx2.MockTransport(FakeRpc().handle), httpx2.MockTransport)
    assert Clock().now().tzinfo is not None
