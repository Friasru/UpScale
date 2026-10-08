"""Safety V2's own request guard: budget, failures, retries, rate-limit cooldown, secret
hygiene. Offline (MockTransport fakes only)."""

import asyncio
from pathlib import Path

from tests.safety_v2_fakes import MINT, RPC_URL, FakeRpc, make_service, mint_value

CID = f"solana:{MINT}"


def test_budget_exhaustion_is_not_collected_without_any_request(tmp_path: Path) -> None:
    svc, _, rpc = make_service(tmp_path, FakeRpc({MINT: mint_value()}), daily_request_budget=1)
    svc.add_target(MINT)
    assert asyncio.run(svc.collect(CID)).outcome == "MINT"
    got = asyncio.run(svc.collect(CID))
    assert got.outcome == "NOT_COLLECTED" and got.requests == 0
    assert rpc.calls == ["getAccountInfo"]  # the second collection made no request
    body = svc.build(CID, 4e9)
    assert body["authority"]["mint_authority"]["status"] == "NOT_COLLECTED"
    assert "budget" in body["authority"]["mint_authority"]["reason"]
    assert svc.repo.collection(got.collection_id)[0] == "DONE"


def test_provider_failure_is_provider_unavailable_after_bounded_retries(tmp_path: Path) -> None:
    svc, clock, rpc = make_service(tmp_path, FakeRpc({MINT: ("http", 502)}), max_retries=2)
    svc.add_target(MINT)
    got = asyncio.run(svc.collect(CID))
    assert got.outcome == "PROVIDER_FAILED"
    assert len(rpc.calls) == 3 and got.requests == 3
    assert svc.repo.requests_by_method(svc.guard.day()) == {"getAccountInfo": 3}
    assert svc.repo.collection(got.collection_id)[0] == "DONE"
    body = svc.build(CID, 4e9)
    assert body["authority"]["freeze_authority"]["status"] == "PROVIDER_UNAVAILABLE"


def test_rpc_error_and_timeout_are_provider_failures(tmp_path: Path) -> None:
    for i, spec in enumerate([("error", "node is behind"), ("timeout",)]):
        svc, _, _ = make_service(tmp_path, FakeRpc({MINT: spec}), db_name=f"{i}.sqlite3",
                                 max_retries=1)  # fmt: skip
        svc.add_target(MINT)
        assert asyncio.run(svc.collect(CID)).outcome == "PROVIDER_FAILED"


def test_rate_limit_starts_a_cooldown_and_later_calls_are_not_collected(tmp_path: Path) -> None:
    svc, clock, rpc = make_service(tmp_path, FakeRpc({MINT: ("http", 429)}), cooldown_seconds=60.0)
    svc.add_target(MINT)
    assert asyncio.run(svc.collect(CID)).outcome == "PROVIDER_FAILED"
    assert len(rpc.calls) == 1  # a 429 is never retried
    rpc.accounts[MINT] = mint_value()
    assert asyncio.run(svc.collect(CID)).outcome == "NOT_COLLECTED"
    assert len(rpc.calls) == 1
    clock.advance(61)
    assert asyncio.run(svc.collect(CID)).outcome == "MINT"


def test_api_key_never_leaks(tmp_path: Path) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: ("error", "boom")}), max_retries=0)
    svc.add_target(MINT)
    asyncio.run(svc.collect(CID))
    assert "SECRETKEY" in RPC_URL
    assert "SECRETKEY" not in repr(svc.provider)
    svc.repo.close()
    assert b"SECRETKEY" not in Path(svc.settings.db_path).read_bytes()
