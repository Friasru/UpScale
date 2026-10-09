"""Opportunity orchestration bridge, B3: the real Safety adapter and the guarded one-job live
path, exercised entirely offline: Safety V2's own service over `FakeRpc` /
`httpx2.MockTransport`, temporary databases, sockets blocked by the test runner."""

import asyncio
import io
import os
import socket
import sqlite3
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

from tests.opportunity_model_fakes import Archive, candidate, scout_record
from tests.safety_v2_fakes import (
    DEX_URL,
    MINT,
    OTHER_PROGRAM,
    RPC_URL,
    Clock,
    FakeRpc,
    HolderChain,
    addr,
    mint_value,
    num_addr,
    pair_row,
)
from tests.test_opportunity_orchestrator_b1 import safety_store
from upscale.services.opportunity_model.repository import OpportunityRepository
from upscale.services.opportunity_orchestrator import cli as orch_cli
from upscale.services.opportunity_orchestrator.cli import NO_ORDER, live_lock, main
from upscale.services.opportunity_orchestrator.processing import Processor, enqueue
from upscale.services.opportunity_orchestrator.repository import OrchestratorRepository
from upscale.services.opportunity_orchestrator.safety_adapter import (
    Capabilities,
    RealSafetyPort,
    SafetyAdapterError,
    classify,
    run_async,
)
from upscale.services.safety_v2.config import SafetySettings
from upscale.services.safety_v2.provider import RequestGuard
from upscale.services.safety_v2.repository import SafetyRepository
from upscale.services.safety_v2.service import (
    Collected,
    CollectionRequestBound,
    SafetyService,
)

CID = f"solana:{MINT}"
POOL = addr("PqqLAAA")
EXISTS = {"owner": OTHER_PROGRAM, "lamports": 5_000_000, "data": ["", "base64"]}


def chain(n: int = 120) -> HolderChain:
    c = HolderChain()
    for i in range(n):
        c.add(num_addr("Hdr", i), 1_000_000)
    return c


class Factory:
    """Builds a fresh real `SafetyService` per call, answered offline by `rpc`."""

    def __init__(self, rpc: FakeRpc, clock: Clock, helius: bool = True, dex: bool = True,
                 rpc_provider: bool = True, **overrides: Any):  # fmt: skip
        self.rpc, self.clock, self.helius, self.dex, self.rpc_provider = (
            rpc,
            clock,
            helius,
            dex,
            rpc_provider,
        )
        self.overrides = overrides
        self.built: list[SafetyService] = []

    def __call__(self, settings: SafetySettings) -> SafetyService:
        s = settings.model_copy(update={"dex_url": DEX_URL if self.dex else None, **self.overrides})
        repo = SafetyRepository(s.db_path)
        c = self.clock
        guard = RequestGuard(
            s, repo, now=c.now, monotonic=c.monotonic, sleep=c.sleep, rng=lambda: 0.0
        )
        svc = SafetyService(
            s, repo=repo, now=c.now, guard=guard, transport=httpx2.MockTransport(self.rpc.handle),
            helius_api_key="SECRETKEY" if self.rpc_provider and self.helius else None,
            rpc_url=RPC_URL if self.rpc_provider and not self.helius else None,
        )  # fmt: skip
        self.built.append(svc)
        return svc


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    clock = Clock()
    t = clock.now()
    arch = Archive(tmp_path / "evidence.sqlite3")
    arch.add(scout_record(t, cand=candidate(t, risk_flags=[])))
    arch.close()
    clock.advance(120)  # the Scout run is settled
    return {"tmp": tmp_path, "clock": clock, "evidence": str(arch.path),
            "safety": str(tmp_path / "safety.sqlite3"),
            "opportunity": str(tmp_path / "opportunity.sqlite3"),
            "orch": str(tmp_path / "orchestrator.sqlite3")}  # fmt: skip


def rpc_for(mint: Any = None, dex: Any = None, holders: HolderChain | None = None) -> FakeRpc:
    return FakeRpc({MINT: mint_value() if mint is None else mint}, holders=holders or chain(),
                   dex=[pair_row(POOL)] if dex is None else dex)  # fmt: skip


def port_for(world: dict[str, Any], rpc: FakeRpc, **kw: Any) -> tuple[RealSafetyPort, Factory]:
    reserve = kw.pop("reserve", 50)
    f = Factory(rpc, world["clock"], **kw)
    return RealSafetyPort(world["safety"], reserve=reserve, factory=f, env={}), f


def run_job(world: dict[str, Any], port: RealSafetyPort) -> Any:
    repo = OrchestratorRepository(world["orch"])
    opp = OpportunityRepository(world["opportunity"])
    try:
        clock = world["clock"]
        enqueue(repo, world["evidence"], clock.now())
        proc = Processor(repo, port, opp, world["evidence"], world["safety"], world["opportunity"],
                         clock.now)  # fmt: skip
        job = proc.runnable()[0]
        return proc.run(job), opp
    finally:
        repo.close()


def count(path: str, table: str) -> int:
    if not Path(path).is_file():
        return 0
    c = sqlite3.connect(path)
    try:
        return int(c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        c.close()


# --- classification (pure, typed Safety outcomes) -----------------------------------------------

READY = Capabilities("HELIUS", True)


def _c(mint: str = "MINT", holders: str | None = "COLLECTED", market: str | None = "POOLS",
       pool: str | None = None) -> Collected:  # fmt: skip
    return Collected(1, 1, mint, 3, 1, holders, 1, market, pool)


@pytest.mark.parametrize(
    ("collected", "kind", "outcome"),
    [
        (_c("NOT_A_MINT", "PROVIDER_FAILED", "PROVIDER_FAILED"), "TOKEN_EVIDENCE", "NOT_A_MINT"),
        (_c("ACCOUNT_MISSING", "NOT_COLLECTED"), "TOKEN_EVIDENCE", "ACCOUNT_MISSING"),
        (_c("MALFORMED"), "TOKEN_EVIDENCE", "MALFORMED"),
        (_c("PROVIDER_FAILED"), "INFRASTRUCTURE", "PROVIDER_UNAVAILABLE"),
        (_c(holders="PROVIDER_FAILED"), "INFRASTRUCTURE", "PROVIDER_UNAVAILABLE"),
        (_c(market="PROVIDER_FAILED"), "INFRASTRUCTURE", "PROVIDER_UNAVAILABLE"),
        (_c(pool="PROVIDER_FAILED"), "INFRASTRUCTURE", "PROVIDER_UNAVAILABLE"),
        (_c(holders="NOT_COLLECTED"), "INFRASTRUCTURE", "SAFETY_BUDGET_EXHAUSTED"),
        (_c(pool="NOT_COLLECTED"), "INFRASTRUCTURE", "SAFETY_BUDGET_EXHAUSTED"),
        (_c(market=None), "INFRASTRUCTURE", "SAFETY_BUDGET_EXHAUSTED"),
        (_c(), "TOKEN_EVIDENCE", "COMPLETE"),
        (_c(market="NO_POOLS"), "TOKEN_EVIDENCE", "NO_POOLS"),
        (_c(market="NO_POOLS", pool="EXISTS"), "TOKEN_EVIDENCE", "NO_POOLS"),
        (_c(market="NO_POOLS", pool="ACCOUNT_MISSING"), "TOKEN_EVIDENCE", "MARKET_CLOSED"),
    ],
)
def test_classification_order(collected: Collected, kind: str, outcome: str) -> None:
    r = classify(collected, READY, None, "full_scan")
    assert (r.kind, r.outcome) == (kind, outcome)


def test_classification_of_not_collected_causes_and_partial_holders() -> None:
    t = Clock().now()
    assert classify(_c(market="NOT_COLLECTED"), Capabilities("HELIUS", False), None,
                    "full_scan").outcome == "PROVIDER_NOT_CONFIGURED"  # fmt: skip
    # An active cooldown, whoever started it: rate limited, retry no earlier than its end.
    for part in ({"holders": "NOT_COLLECTED"}, {"market": "PROVIDER_FAILED"}):
        r = classify(_c(**part), READY, t, "full_scan")
        assert (r.outcome, r.retry_at) == ("PROVIDER_RATE_LIMITED", t)
    budget = classify(_c(holders="NOT_COLLECTED"), READY, None, None)
    assert (budget.outcome, budget.retry_at) == ("SAFETY_BUDGET_EXHAUSTED", None)
    assert classify(_c(), READY, None, "largest_accounts").outcome == "HOLDERS_PARTIAL"


# --- real adapter, token evidence -------------------------------------------------------------


@pytest.mark.parametrize(
    ("rpc_kw", "port_kw", "outcome", "decision"),
    [
        ({}, {}, "COMPLETE", None),
        ({}, {"helius": False}, "HOLDERS_PARTIAL", None),
        ({"mint": mint_value(program=OTHER_PROGRAM)}, {}, "NOT_A_MINT", "SKIP"),
        ({"mint": mint_value(supply="12x")}, {}, "MALFORMED", "SKIP"),
        ({"dex": []}, {}, "NO_POOLS", "SKIP"),
    ],
)
def test_token_evidence_snapshots_and_decides(
    world: dict[str, Any], rpc_kw: dict[str, Any], port_kw: dict[str, Any], outcome: str,
    decision: str | None,
) -> None:  # fmt: skip
    rpc = rpc_for(**rpc_kw)
    if outcome == "NOT_A_MINT":
        rpc.holders.fail = {"getTokenSupply": ("error", "not a token mint")}  # type: ignore[union-attr]
    port, factory = port_for(world, rpc, **port_kw)
    job, opp = run_job(world, port)
    try:
        (report,) = port.reports
        assert (report.result.kind, report.result.outcome) == ("TOKEN_EVIDENCE", outcome)
        assert job.state == "DECIDED" and job.safety_snapshot_id is not None
        stored = opp.get(job.opportunity_decision_id)
        assert stored.input.sources.safety.ref.snapshot_id == job.safety_snapshot_id
        if decision:
            assert stored.decision.decision == decision
        assert report.requests == len(rpc.calls) <= report.bound.attempt_max
        assert len({id(s) for s in factory.built}) == len(factory.built) >= 3  # fresh per op
    finally:
        opp.close()


def test_account_missing_is_identity_evidence(world: dict[str, Any]) -> None:
    rpc = FakeRpc({MINT: None}, holders=chain(), dex=[pair_row(POOL)])
    rpc.holders.fail = {"getTokenSupply": ("error", "could not find account")}  # type: ignore[union-attr]
    port, _ = port_for(world, rpc)
    job, opp = run_job(world, port)
    opp.close()
    assert port.reports[0].result.outcome == "ACCOUNT_MISSING" and job.state == "DECIDED"


@pytest.mark.parametrize(
    ("pool_account", "outcome"), [(EXISTS, "NO_POOLS"), (None, "MARKET_CLOSED")]
)
def test_a_previously_reported_pool_that_vanished(
    world: dict[str, Any], pool_account: Any, outcome: str
) -> None:
    rpc = rpc_for()
    port, _ = port_for(world, rpc)
    assert port.collect(CID, world["clock"].now()).outcome == "COMPLETE"
    world["clock"].advance(3600)
    rpc.dex = []
    if pool_account is not None:
        rpc.accounts[POOL] = pool_account
    r = port.collect(CID, world["clock"].now())
    assert (r.kind, r.outcome) == ("TOKEN_EVIDENCE", outcome)
    assert port.reports[-1].outcomes[-1] == (
        "pool_account",
        "EXISTS" if pool_account else "ACCOUNT_MISSING",
    )


# --- real adapter, infrastructure ------------------------------------------------------------


@pytest.mark.parametrize(
    ("rpc_kw", "fail", "category", "state"),
    [
        ({"mint": ("http", 503)}, {}, "PROVIDER_UNAVAILABLE", "RETRY_WAIT"),
        ({"mint": ("timeout",)}, {}, "PROVIDER_UNAVAILABLE", "RETRY_WAIT"),
        ({"mint": ("http", 429)}, {}, "PROVIDER_RATE_LIMITED", "RETRY_WAIT"),
        ({}, {"getTokenSupply": ("http", 503)}, "PROVIDER_UNAVAILABLE", "RETRY_WAIT"),
        ({"dex": ("http", 503)}, {}, "PROVIDER_UNAVAILABLE", "RETRY_WAIT"),
    ],
)
def test_infrastructure_writes_no_snapshot_and_no_decision(
    world: dict[str, Any], rpc_kw: dict[str, Any], fail: dict[str, Any], category: str, state: str
) -> None:
    rpc = rpc_for(**rpc_kw)
    rpc.holders.fail = fail  # type: ignore[union-attr]
    port, _ = port_for(world, rpc)
    job, opp = run_job(world, port)
    opp.close()
    assert (job.state, job.category) == (state, category)
    assert count(world["safety"], "safety_snapshots") == 0
    assert count(world["opportunity"], "opportunity_decisions") == 0


def test_budget_exhausted_mid_collection_is_reported(world: dict[str, Any]) -> None:
    rpc = rpc_for()
    port, _ = port_for(world, rpc, daily_request_budget=2)
    r = port.collect(CID, world["clock"].now())  # mint fits; holders need more than remain
    assert (r.kind, r.outcome) == ("INFRASTRUCTURE", "SAFETY_BUDGET_EXHAUSTED")
    assert dict(port.reports[-1].outcomes)["holders"] == "NOT_COLLECTED"
    assert count(world["safety"], "safety_snapshots") == 0


@pytest.mark.parametrize(("port_kw", "blocked"), [
    ({"rpc_provider": False}, "PROVIDER_NOT_CONFIGURED"),
    ({"dex": False}, "PROVIDER_NOT_CONFIGURED"),
    ({"daily_request_budget": 76}, "SAFETY_BUDGET_EXHAUSTED"),  # 27 + 50 needed
])  # fmt: skip
def test_preflight_blocks_before_any_provider_call(
    world: dict[str, Any], port_kw: dict[str, Any], blocked: str
) -> None:
    rpc = rpc_for()
    port, _ = port_for(world, rpc, **port_kw)
    job, opp = run_job(world, port)
    opp.close()
    assert (job.state, job.category, job.attempt_count) == ("DEFERRED", blocked, 0)
    assert rpc.calls == [] and port.reports == []
    assert count(world["safety"], "safety_targets") == 0  # target added only when collecting


def test_preflight_ok_at_the_exact_budget(world: dict[str, Any]) -> None:
    port, _ = port_for(world, rpc_for(), daily_request_budget=77)
    assert port.preflight(CID, world["clock"].now()).label == "CONSERVATIVE_PREFLIGHT_OK"


def test_an_exception_before_evidence_is_a_retryable_failure(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(self: Any, *a: Any, **k: Any) -> Any:
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(SafetyService, "collect", boom)
    job, opp = run_job(world, port_for(world, rpc_for())[0])
    opp.close()
    assert (job.state, job.category, job.attempt_count) == ("RETRY_WAIT", "COLLECTION_EXCEPTION", 1)
    assert count(world["opportunity"], "opportunity_decisions") == 0


# --- async ----------------------------------------------------------------------------------------


def test_refuses_inside_a_running_event_loop(world: dict[str, Any]) -> None:
    port, _ = port_for(world, rpc_for())

    async def inside() -> Any:
        return port.collect(CID, world["clock"].now())

    with pytest.raises(SafetyAdapterError, match="running event loop"):
        asyncio.run(inside())

    async def coro() -> int:
        return 1

    assert run_async(coro) == 1  # a plain synchronous caller is fine


# --- live CLI -----------------------------------------------------------------------------------


def live(
    world: dict[str, Any], *extra: str, rpc: FakeRpc | None = None, **kw: Any
) -> tuple[int, str, Any]:
    rpc = rpc or rpc_for()
    holder: dict[str, Any] = {}

    def make(safety_db: str) -> RealSafetyPort:
        port, factory = port_for(world, rpc, **kw)
        holder["port"], holder["factory"] = port, factory
        return port

    out = io.StringIO()
    code = main(["--evidence-db", world["evidence"], "--safety-db", world["safety"],
                 "--opportunity-db", world["opportunity"], "--orchestrator-db", world["orch"],
                 "process", "--live", *extra], out=out, clock=world["clock"].now, live_port=make)  # fmt: skip
    holder["rpc"] = rpc
    return code, out.getvalue(), holder


def enqueued(world: dict[str, Any]) -> None:
    repo = OrchestratorRepository(world["orch"])
    enqueue(repo, world["evidence"], world["clock"].now())
    repo.close()


def test_live_processes_exactly_one_job_and_reports(world: dict[str, Any]) -> None:
    enqueued(world)
    code, out, h = live(world, "--max-requests", "27")
    assert code == 0, out
    for needle in ("rpc=HELIUS dex=configured", "logical 9, attempts 27; reserve 50",
                   "preflight: CONSERVATIVE_PREFLIGHT_OK", "collection: TOKEN_EVIDENCE COMPLETE",
                   "safety snapshot 1", "holders=AVAILABLE market=AVAILABLE",
                   "opportunity decision 1:", "1 job processed; stopping", NO_ORDER,
                   "Do not run manual Safety"):  # fmt: skip
        assert needle in out, needle
    assert "SECRETKEY" not in out and RPC_URL not in out and DEX_URL not in out
    assert "safety budget used today after: " + str(h["port"].reports[0].requests) in out
    assert not Path(world["orch"] + ".live.lock").exists() or _lock_free(world)


def _lock_free(world: dict[str, Any]) -> bool:
    with live_lock(world["orch"]):
        return True


@pytest.mark.parametrize(("extra", "needle"), [
    ((), "--max-requests"),
    (("--max-requests", "27", "--limit", "2"), "exactly 1 job"),
    (("--max-requests", "26"), "exceeds --max-requests 26"),
])  # fmt: skip
def test_live_refusals_happen_before_any_provider_call(
    world: dict[str, Any], extra: tuple[str, ...], needle: str
) -> None:
    enqueued(world)
    code, out, h = live(world, *extra)
    assert code == 2 and needle in out
    assert h["rpc"].calls == [] and count(world["safety"], "safety_targets") == 0
    repo = OrchestratorRepository(world["orch"], read_only=True)
    assert [j.state for j in repo.jobs()] == ["QUEUED"]
    repo.close()


def test_live_cooldown_and_missing_stores(world: dict[str, Any]) -> None:
    enqueued(world)
    until = world["clock"].now() + timedelta(minutes=5)
    SafetyRepository(world["safety"]).set_meta("provider.cooldown_until", repr(until.timestamp()))
    code, out, h = live(world, "--max-requests", "27")
    assert code == 0 and "(SAFETY_COOLDOWN)" in out and h["rpc"].calls == []
    os.remove(world["evidence"])
    code, out, _ = live(world, "--max-requests", "27")
    assert code == 2 and "Evidence Archive is missing" in out


def test_a_second_live_run_is_refused_while_the_lock_is_held(world: dict[str, Any]) -> None:
    enqueued(world)
    with live_lock(world["orch"]):
        code, out, h = live(world, "--max-requests", "27")
    assert code == 2 and "holds" in out and h["rpc"].calls == []


def test_live_never_recovers_and_releases_the_lock_on_interrupt(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueued(world)

    def no(*a: Any, **k: Any) -> Any:
        raise AssertionError("live mode must not recover")

    monkeypatch.setattr(Processor, "recover", no)

    async def interrupted(self: Any, *a: Any, **k: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(SafetyService, "collect", interrupted)
    with pytest.raises(KeyboardInterrupt):
        live(world, "--max-requests", "27")
    assert _lock_free(world)  # released
    repo = OrchestratorRepository(world["orch"], read_only=True)
    assert [j.state for j in repo.jobs()] == ["COLLECTING"]  # honest: recover handles it later
    repo.close()
    assert count(world["opportunity"], "opportunity_decisions") == 0


def test_the_adapter_never_configures_radar(world: dict[str, Any]) -> None:
    before = os.environ.get("UPSCALE_SAFETY_V2_RADAR_DB")
    enqueued(world)
    code, _, h = live(world, "--max-requests", "27")
    assert code == 0 and os.environ.get("UPSCALE_SAFETY_V2_RADAR_DB") == before
    assert h["port"].settings.radar_db_path is None
    c = sqlite3.connect(world["safety"])
    assert c.execute("SELECT status FROM safety_radar_captures").fetchall() == [("NOT_CONFIGURED",)]
    c.close()


def test_offline_commands_never_load_safety() -> None:
    code = ("import sys, upscale.services.opportunity_orchestrator.cli as c; "
            "print(any(m.startswith('upscale.services.safety' + '_v2') for m in sys.modules))")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).parents[1], check=True)  # fmt: skip
    assert out.stdout.strip() == "False"
    assert "safety_adapter" not in Path(orch_cli.__file__).read_text().split("def _live_one")[0]


def test_no_socket_is_opened(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    enqueued(world)
    code, out, _ = live(world, "--max-requests", "27")
    assert code == 0 and "opportunity decision 1" in out


def test_an_incompatible_safety_database_is_refused_cleanly(world: dict[str, Any]) -> None:
    enqueued(world)
    safety_store(Path(world["safety"]), [])  # not a real Safety V2 database
    code, out, h = live(world, "--max-requests", "27")
    assert code == 2 and "SafetySchemaError" in out and "Traceback" not in out
    assert h.get("rpc") is None or h["rpc"].calls == []


# --- in-flight 429 vs pre-collection cooldown (hardening) ---------------------------------------


class Runner:
    """Drives one job through the real adapter, waiting out each retry like a later run."""

    def __init__(self, world: dict[str, Any], port: RealSafetyPort):
        self.world, self.port = world, port
        self.repo = OrchestratorRepository(world["orch"])
        self.opp = OpportunityRepository(world["opportunity"])
        clock = world["clock"]
        enqueue(self.repo, world["evidence"], clock.now())
        self.proc = Processor(self.repo, port, self.opp, world["evidence"], world["safety"],
                              world["opportunity"], clock.now)  # fmt: skip
        self.job = self.proc.runnable()[0]

    def again(self) -> Any:
        """Advance to the job's next attempt time, release it and run it."""
        j = self.repo.job(self.job.id)
        if j.next_attempt_at is not None:
            self.world["clock"].t = j.next_attempt_at
        self.proc.release_due()
        self.job = self.proc.run(self.repo.job(self.job.id))
        return self.job

    def close(self) -> None:
        self.repo.close()
        self.opp.close()


def _events(runner: Runner) -> list[str]:
    return [e["to"] for e in runner.repo.events(runner.job.id)]


def test_a_g_a_cooldown_before_collection_defers_without_an_attempt(world: dict[str, Any]) -> None:
    rpc = rpc_for()
    port, _ = port_for(world, rpc)
    until = world["clock"].now() + timedelta(minutes=20)
    SafetyRepository(world["safety"]).set_meta("provider.cooldown_until", repr(until.timestamp()))
    r = Runner(world, port)
    try:
        job = r.again()
        assert (job.state, job.category, job.attempt_count) == ("DEFERRED", "SAFETY_COOLDOWN", 0)
        assert job.next_attempt_at == until and "COLLECTING" not in _events(r)
        assert rpc.calls == []
    finally:
        r.close()


def test_b_c_d_e_repeated_in_flight_429s_count_and_terminate(world: dict[str, Any]) -> None:
    rpc = rpc_for(mint=("http", 429))
    port, _ = port_for(world, rpc)  # Safety cooldown 15 min, doubling per repeat
    r = Runner(world, port)
    try:
        for n, state in ((1, "RETRY_WAIT"), (2, "RETRY_WAIT"), (3, "FAILED")):
            started = world["clock"].now()
            job = r.again()
            assert (job.state, job.category, job.attempt_count) == (
                state,
                "PROVIDER_RATE_LIMITED",
                n,
            )
            assert _events(r).count("COLLECTING") == n  # every one started a collection
            cooldown = SafetyRepository(world["safety"]).get_meta("provider.cooldown_until")
            assert cooldown is not None
            if state == "RETRY_WAIT":
                backoff = started + timedelta(minutes=10 if n == 1 else 30)
                until = datetime_from(cooldown)
                assert job.next_attempt_at == max(until, backoff) >= until
                if n == 1:  # E: a 15 min cooldown outlasts the 10 min backoff
                    assert job.next_attempt_at == until > backoff
        calls = len(rpc.calls)
        r.proc.release_due()
        assert r.proc.runnable() == [] and len(rpc.calls) == calls  # no fourth attempt
        assert count(world["opportunity"], "opportunity_decisions") == 0
        assert count(world["safety"], "safety_snapshots") == 0
    finally:
        r.close()


def datetime_from(raw: str) -> Any:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(float(raw), UTC)


def test_f_a_backoff_longer_than_the_cooldown_wins(world: dict[str, Any]) -> None:
    port, _ = port_for(world, rpc_for(mint=("http", 429)), cooldown_seconds=60.0)
    r = Runner(world, port)
    try:
        started = world["clock"].now()
        job = r.again()
        assert (job.state, job.attempt_count) == ("RETRY_WAIT", 1)
        assert job.next_attempt_at == started + timedelta(minutes=10)
    finally:
        r.close()


# --- request-bound breach is terminal (hardening) -----------------------------------------------


def _actual_requests(world: dict[str, Any]) -> int:
    probe = dict(world, safety=str(world["tmp"] / "probe.sqlite3"))
    port, _ = port_for(probe, rpc_for())
    port.collect(CID, world["clock"].now())
    return port.reports[0].requests


def _with_bound(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    monkeypatch.setattr(SafetyService, "collection_request_bound",
                        lambda self, holders=True, market=True: CollectionRequestBound(n, n))  # fmt: skip


def test_a_b_c_f_a_bound_breach_fails_the_job_terminally(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = _actual_requests(world)
    _with_bound(monkeypatch, actual - 1)  # Safety advertises one less than it spends
    enqueued(world)
    code, out, h = live(world, "--max-requests", "27", reserve=0)
    assert code == 3 and "INVARIANT ERROR" in out
    assert _lock_free(world)
    repo = OrchestratorRepository(world["orch"])
    (job,) = repo.jobs()
    assert (job.state, job.category) == ("FAILED", "SAFETY_REQUEST_BOUND_BREACH")
    assert job.note == f"Safety used {actual} requests, above its advertised bound {actual - 1}"
    assert "SECRETKEY" not in job.note and RPC_URL not in job.note and DEX_URL not in job.note
    assert count(world["opportunity"], "opportunity_decisions") == 0
    assert count(world["safety"], "safety_snapshots") == 0
    day = h["port"].budget().day  # the collection's UTC day, read before the clock moves
    ledger = SafetyRepository(world["safety"]).requests_on(day)
    opp = OpportunityRepository(world["opportunity"])
    proc = Processor(repo, h["port"], opp, world["evidence"], world["safety"], world["opportunity"],
                     world["clock"].now)  # fmt: skip
    world["clock"].advance(24 * 3600)
    assert proc.recover() == [] and proc.process() == []  # B, C: never retried
    assert repo.job(job.id).state == "FAILED"
    assert SafetyRepository(world["safety"]).requests_on(day) == ledger == actual
    opp.close()
    repo.close()


@pytest.mark.parametrize("slack", [0, 5], ids=["exactly-the-bound", "below-the-bound"])
def test_d_e_usage_within_the_bound_is_allowed(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, slack: int
) -> None:
    _with_bound(monkeypatch, _actual_requests(world) + slack)
    enqueued(world)
    code, out, _ = live(world, "--max-requests", "27", reserve=0)
    assert code == 0 and "opportunity decision 1" in out


# --- started collections always count (hardening) -------------------------------------------


class InterferingRpc(FakeRpc):
    """Another process acts on Safety's shared database once this collection has started:
    on the first ``getTokenSupply`` (holders), it starts a cooldown or spends the budget."""

    def __init__(self, safety_db: str, clock: Clock, action: str, **kw: Any):
        super().__init__(**kw)
        self.safety_db, self.clock, self.action = safety_db, clock, action

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST" and b'"getTokenSupply"' in request.content:
            now = self.clock.now()
            c = sqlite3.connect(self.safety_db)
            if self.action == "cooldown":
                c.execute("INSERT OR REPLACE INTO safety_meta (key, value) VALUES "
                          "('provider.cooldown_until', ?)", (repr(now.timestamp() + 3600),))  # fmt: skip
            else:
                c.execute("INSERT INTO safety_requests (day, method, calls, ok) VALUES "
                          "(?, 'other_process', 10000, 10000) ON CONFLICT (day, method) DO "
                          "UPDATE SET calls = calls + 10000",
                          (now.strftime("%Y-%m-%d"),))  # fmt: skip
            c.commit()
            c.close()
        return super().handle(request)


def _interfering(world: dict[str, Any], action: str) -> InterferingRpc:
    return InterferingRpc(world["safety"], world["clock"], action, accounts={MINT: mint_value()},
                          holders=chain(), dex=[pair_row(POOL)])  # fmt: skip


def test_d_a_cooldown_started_elsewhere_after_collecting_counts(world: dict[str, Any]) -> None:
    rpc = _interfering(world, "cooldown")
    r = Runner(world, port_for(world, rpc)[0])
    try:
        started = world["clock"].now()
        job = r.again()
        assert (job.state, job.category, job.attempt_count) == (
            "RETRY_WAIT", "PROVIDER_RATE_LIMITED", 1)  # fmt: skip
        assert job.next_attempt_at == started + timedelta(hours=1)  # the cooldown, not 10 min
        assert "COLLECTING" in _events(r) and "getTokenSupply" in rpc.calls
        assert count(world["safety"], "safety_snapshots") == 0
        assert count(world["opportunity"], "opportunity_decisions") == 0
    finally:
        r.close()


def test_f_g_budget_spent_after_collecting_counts_and_cannot_loop(world: dict[str, Any]) -> None:
    rpc = _interfering(world, "budget")
    r = Runner(world, port_for(world, rpc)[0])
    try:
        for n, state in ((1, "RETRY_WAIT"), (2, "RETRY_WAIT"), (3, "FAILED")):
            started = world["clock"].now()
            job = r.again()
            assert (job.state, job.category, job.attempt_count) == (
                state, "SAFETY_BUDGET_EXHAUSTED", n)  # fmt: skip
            if state == "RETRY_WAIT":  # Safety's budget period: the UTC day after the failure
                day = job.updated_at.replace(hour=0, minute=0, second=0, microsecond=0)
                assert job.next_attempt_at == day + timedelta(days=1) > started
        assert _events(r).count("COLLECTING") == 3  # I: each started collection is counted
        assert len(r.repo.collection_starts(CID)) == 3
        r.proc.release_due()
        assert r.proc.runnable() == []  # G: never a fourth
        assert count(world["safety"], "safety_snapshots") == 0
        assert count(world["opportunity"], "opportunity_decisions") == 0
    finally:
        r.close()


def test_h_a_preflight_budget_shortfall_stays_free(world: dict[str, Any]) -> None:
    rpc = rpc_for()
    r = Runner(world, port_for(world, rpc, daily_request_budget=76)[0])
    try:
        job = r.again()
        assert (job.state, job.category, job.attempt_count) == (
            "DEFERRED",
            "SAFETY_BUDGET_EXHAUSTED",
            0,
        )
        assert "COLLECTING" not in _events(r) and rpc.calls == []
    finally:
        r.close()


@pytest.mark.parametrize("case", ["503", "timeout", "cooldown-elsewhere", "exception"])
def test_every_started_transient_failure_counts_exactly_once(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Each distinct started collection adds exactly one attempt: 1, 2, then 3 (FAILED)."""
    rpc = {"503": lambda: rpc_for(mint=("http", 503)),
           "timeout": lambda: rpc_for(mint=("timeout",)),
           "cooldown-elsewhere": lambda: _interfering(world, "cooldown"),
           "exception": rpc_for}[case]()  # fmt: skip
    if case == "exception":

        async def boom(self: Any, *a: Any, **k: Any) -> Any:
            raise RuntimeError("database is locked")

        monkeypatch.setattr(SafetyService, "collect", boom)
    r = Runner(world, port_for(world, rpc)[0])
    try:
        assert r.repo.job(r.job.id).attempt_count == 0
        seen = []
        for _ in range(3):
            job = r.again()
            seen.append((job.attempt_count, job.state))
        assert seen == [(1, "RETRY_WAIT"), (2, "RETRY_WAIT"), (3, "FAILED")]
        assert _events(r).count("COLLECTING") == 3
        r.proc.release_due()
        assert r.proc.runnable() == []
        assert count(world["opportunity"], "opportunity_decisions") == 0
    finally:
        r.close()
