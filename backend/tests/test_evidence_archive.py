"""Point-in-Time Evidence Archive: append-only production evidence, and Replay Lab reading
it with no lookahead. Temporary databases only; no network."""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import upscale.main
import upscale.services as services
from upscale import log_safety
from upscale.schemas import ChatRequest, ChatResponse
from upscale.services.clock import frozen_now
from upscale.services.evidence_archive import hooks, payloads
from upscale.services.evidence_archive.cli import main as evidence_cli
from upscale.services.evidence_archive.enrichment import (
    EnrichmentSettings,
    SafetyEnrichment,
    load_settings,
)
from upscale.services.evidence_archive.recorder import EvidenceRecorder
from upscale.services.evidence_archive.status import status
from upscale.services.evidence_archive.store import (
    EvidenceStore,
    EvidenceStoreError,
    FutureEvidenceError,
    PendingRecord,
)
from upscale.services.market_data import (
    InvalidRequestError,
    MarketDataError,
    MarketDataUnavailableError,
    ProviderRateLimitedError,
)
from upscale.services.outcomes import OutcomeStore
from upscale.services.replay_lab.analyze import PointInTimeSafetyService
from upscale.services.replay_lab.clock import HistoricalClock, LookaheadError
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.growth.models import GrowthScoutResult
from upscale.services.scout.growth.service import GrowthScoutService
from upscale.services.scout.normalize import Listing, build_candidates
from upscale.services.scout.service import ScoutService
from upscale.services.scout.store import ScoutSnapshotStore
from upscale.services.solana_chain import OnchainSafetySnapshot
from upscale.services.solana_dex import SolanaDexSnapshot

from .conftest import FakeSolanaRpc
from .test_onchain_safety import MINT, POOL_REF, service, setup_token
from .test_outcomes import assessment, chat
from .test_replay_lab import (
    NOW,
    T0,
    Harness,
    build_archive,
    config,
    dex_pool,
    mint,
    pool_address,
)
from .test_scout_growth import momentum

EVM_TOKEN = "0xAbCdEf0123456789aBcDeF0123456789AbCdEf01"


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[tuple[EvidenceRecorder, EvidenceStore]]:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    recorder = EvidenceRecorder(
        store, versions=lambda: {"code": "test", "scout_scoring": "abc"},
        capabilities=lambda: ["candles", "dex", "market_snapshot", "news", "onchain"],
    )  # fmt: skip
    previous = hooks.installed()
    hooks.install(recorder)
    yield recorder, store
    recorder.flush()
    hooks.install(previous)
    store.close()


def rows(store: EvidenceStore, kind: str | None = None) -> list[Any]:
    return store.records(kind)  # type: ignore[arg-type]


def safety(
    token: str, fetched_at: datetime, *, mint_authority: str | None = None, top1: float = 5.0,
    top10: float = 25.0,
) -> OnchainSafetySnapshot:  # fmt: skip
    return OnchainSafetySnapshot(
        canonical_id=f"solana:{token}", mint=token, provider="Helius", fetched_at=fetched_at,
        authorities_available=True, token_program="spl_token", token_program_id="Tokenkeg",
        decimals=6, supply=1e9, mint_authority=mint_authority, freeze_authority=None,
        holders=[], excluded=[], top1_pct=top1, top10_pct=top10, excluded_pct=10.0,
        holder_count=900, meaningful_holder_count=700, holder_count_complete=True,
        concentration_source="full_scan", concentration_lower_bound=False,
        token_accounts_seen=1000, scan_pages_read=1, scan_max_pages=5, largest_accounts_seen=20,
        pool_addresses_used=[], concentration_reliable=True, holder_data_complete=True,
    )  # fmt: skip


def pending(kind: str, asset: str, at: datetime, **kw: Any) -> PendingRecord:
    return PendingRecord(
        kind=kind, asset_id=asset, observed_at=at, payload=kw.pop("payload", {"v": 1}), **kw
    )  # type: ignore[arg-type]


# --- Storage -----------------------------------------------------------------------------------


def test_records_are_append_only_hashed_and_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    assert store.append(pending("safety", "solana:x", T0, provider="Helius", payload={"top1": 5}))
    store.close()
    db = sqlite3.connect(path)
    for sql in (
        "UPDATE evidence_records SET availability = 'NOT_AVAILABLE'",
        "DELETE FROM evidence_records",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            with db:
                db.execute(sql)
    reopened = EvidenceStore(path)  # a restart: the history is still there
    [r] = reopened.records()
    assert (
        r.payload == {"top1": 5}
        and r.availability == "AVAILABLE"
        and r.archived_at >= r.observed_at
    )
    with pytest.raises(sqlite3.DatabaseError):  # observed after it was archived: impossible
        with db:
            db.execute(
                "INSERT INTO evidence_records (record_id, kind, asset_id, component, observed_at, "
                "archived_at, availability, fingerprint, payload, payload_hash, schema_version, "
                "versions_json) VALUES ('x', 'safety', 'a', 'c', 2e9, 1e9, 'AVAILABLE', 'f', x'00', 'h', 1, '{}')"
            )
    ro = EvidenceStore(path, read_only=True)
    with pytest.raises(EvidenceStoreError):
        ro.append(pending("safety", "solana:y", T0))
    with pytest.raises(sqlite3.OperationalError):
        ro._db().execute("DELETE FROM evidence_records")


def test_latest_at_or_before_t_never_after(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "e.sqlite3")
    for minutes, value in ((-30, "old"), (-10, "valid"), (5, "future")):
        store.append(
            pending("social", "solana:a", T0 + timedelta(minutes=minutes), payload={"v": value})
        )
    assert store.latest("social", "solana:a", until=T0).payload == {"v": "valid"}  # type: ignore[union-attr]
    assert store.latest("social", "solana:a", until=T0 - timedelta(hours=1)) is None
    assert store.latest("social", "solana:a", until=T0, since=T0 - timedelta(minutes=5)) is None
    assert store.latest_any("social", T0).payload == {"v": "valid"}  # type: ignore[union-attr]
    assert FutureEvidenceError.__mro__[1] is RuntimeError


def test_duplicates_are_skipped_but_state_changes_are_kept(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "e.sqlite3")
    same = {"snapshot": {"top1": 5, "fetched_at": "x"}}
    assert store.append(pending("safety", "solana:a", T0, provider="Helius", payload=same), 600)
    # The same state re-analyzed from cache minutes later: a duplicate.
    later = {"snapshot": {"top1": 5, "fetched_at": "y"}}
    assert not store.append(
        pending("safety", "solana:a", T0 + timedelta(minutes=2), provider="Helius", payload=later),
        600,
    )
    # A real change is always kept, however soon.
    changed = {"snapshot": {"top1": 40, "fetched_at": "z"}}
    assert store.append(
        pending(
            "safety", "solana:a", T0 + timedelta(minutes=3), provider="Helius", payload=changed
        ),
        600,
    )
    # The unchanged state again after the window: kept, so its persistence stays visible.
    assert store.append(
        pending(
            "safety", "solana:a", T0 + timedelta(minutes=30), provider="Helius", payload=changed
        ),
        600,
    )
    assert [r.payload["snapshot"]["top1"] for r in store.records()] == [5, 40, 40]


def test_canonical_identity_evm_lowercase_solana_exact() -> None:
    at = T0
    evm = SolanaDexSnapshot(
        canonical_id=f"base:{EVM_TOKEN.lower()}", mint=EVM_TOKEN, symbol="PEPE", name=None,
        provider="DEX Screener", dex="uniswap", pair_address="0xPool", pair_url=None,
        quote_symbol="WETH", quote_address="0x4200000000000000000000000000000000000006",
        quote_kind="WETH", price_usd=1.0, price_native=None, liquidity_usd=1e5, market_cap_usd=None,
        fdv_usd=None, pair_created_at=None, pool_age_hours=None, first_pool_created_at=None,
        windows=[], fetched_at=at, candidates=[], primary_clear=True, chain="base",
    )  # fmt: skip
    [r] = payloads.dex_market(evm)
    assert r.asset_id == f"base:{EVM_TOKEN.lower()}"
    [s] = payloads.safety(safety(mint(1), at), [])
    assert s.asset_id == f"solana:{mint(1)}" and s.asset_id != s.asset_id.lower()
    # Same ticker, different mints: never merged.
    a = payloads.safety(safety(mint(2), at), [])[0].asset_id
    b = payloads.safety(safety(mint(3), at), [])[0].asset_id
    assert a != b


def test_no_secret_is_persisted(tmp_path: Path) -> None:
    secret = "sk-evidence-test-secret-123456"
    log_safety.register_secret(secret)
    store = EvidenceStore(tmp_path / "e.sqlite3")
    store.append(pending("decision", "solana:a", T0, payload={
        "request": {"query": f"analyze with key {secret} and https://rpc.x/?api-key=abcdef123456"},
        "headers": "Authorization: Bearer abc.def.ghi"}))  # fmt: skip
    raw = (tmp_path / "e.sqlite3").read_bytes()
    [r] = store.records()
    text = json.dumps(r.payload)
    assert secret not in text and "abcdef123456" not in text and "abc.def.ghi" not in text
    assert secret.encode() not in raw


# --- Production hooks: archived with no extra requests ---------------------------------------------


def test_safety_is_archived_from_the_production_request_without_extra_calls(
    archive: tuple[EvidenceRecorder, EvidenceStore],
) -> None:
    recorder, store = archive
    fake = FakeSolanaRpc()
    setup_token(fake)
    svc = service(fake)
    snap = asyncio.run(svc.get_snapshot(MINT, (POOL_REF,)))
    calls = len(fake.requests)
    asyncio.run(svc.get_snapshot(MINT, (POOL_REF,)))  # cached: no request, same evidence
    assert len(fake.requests) == calls
    hooks.install(None)
    fake_off = FakeSolanaRpc()
    setup_token(fake_off)
    asyncio.run(service(fake_off).get_snapshot(MINT, (POOL_REF,)))
    assert len(fake_off.requests) == calls  # archiving costs no provider request
    hooks.install(recorder)
    recorder.flush()
    [r] = rows(store, "safety")
    assert r.asset_id == f"solana:{MINT}" and r.provider == snap.provider
    assert r.observed_at == snap.fetched_at and r.availability == "AVAILABLE"
    stored = OnchainSafetySnapshot.model_validate(r.payload["snapshot"])
    assert stored == snap  # exactly what production used
    assert r.payload["components"] == {"authorities": "AVAILABLE", "holders": "AVAILABLE"}
    assert r.payload["flags"]["mint_authority_active"] is False
    assert r.payload["pools"][0]["address"] == POOL_REF.address
    assert r.versions == {"code": "test", "scout_scoring": "abc"}


def test_safety_failures_are_archived_with_their_reason(
    archive: tuple[EvidenceRecorder, EvidenceStore],
) -> None:
    recorder, store = archive
    fake = FakeSolanaRpc()
    fake.fail = {"getAccountInfo": 429, "getTokenLargestAccounts": 429}
    with hooks.component("analyze"), pytest.raises(MarketDataError):
        asyncio.run(service(fake).get_snapshot(MINT))
    with pytest.raises(InvalidRequestError):
        asyncio.run(service(FakeSolanaRpc()).get_snapshot("not-a-mint"))
    recorder.flush()
    failed, invalid = rows(store, "safety")
    assert failed.availability in ("RATE_LIMITED", "PROVIDER_FAILED") and failed.reason
    assert failed.component == "analyze"
    assert invalid.availability == "NOT_AVAILABLE"
    assert payloads.failure_state(ProviderRateLimitedError("x")) == "RATE_LIMITED"
    assert (
        payloads.failure_state(
            MarketDataUnavailableError("UpScale's Helius request limit was reached")
        )
        == "RATE_LIMITED"
    )


def test_scout_market_social_and_ranking_are_archived(
    archive: tuple[EvidenceRecorder, EvidenceStore], tmp_path: Path
) -> None:
    recorder, store = archive
    scout = ScoutService([], ScoutSnapshotStore(tmp_path / "scout.sqlite3"), ScoutConfig())
    listing = Listing(provider="DEX Screener", kind="new", name="new_pools", fetched_at=T0)
    candidate = build_candidates([dex_pool(0, T0)], listing, ScoutConfig()).candidates[0]
    full = asyncio.run(scout._persist(candidate))
    growth = GrowthScoutService(scout.store, GrowthConfig(), now=lambda: T0)
    result = asyncio.run(growth.rank([full]))
    hooks.emit("scout", result)
    hooks.emit("social", momentum(mint(0), "ACCELERATING", computed_at=T0))
    recorder.flush()
    [m] = rows(store, "market")
    assert m.pool_address == pool_address(0) and m.provider == "DEX Screener"
    assert m.payload["candidate"]["pool"]["quote_address"]  # immutable pool facts included
    assert m.payload["field_availability"]["market_cap_usd"] == "AVAILABLE"
    [s] = rows(store, "scout")
    assert (
        s.payload["candidate"]["stage"] == result.candidates[0].stage if result.candidates else True
    )
    assert s.payload["capabilities"] and s.observed_at == T0
    assert "scout_momentum" in s.payload["candidate"]
    [soc] = rows(store, "social")
    assert soc.payload["momentum"]["state"] == "ACCELERATING" and "text" not in json.dumps(
        soc.payload
    )


def test_analyze_decision_is_archived_and_linked(
    archive: tuple[EvidenceRecorder, EvidenceStore],
    client: TestClient,
    isolated_outcome_store: OutcomeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder, store = archive
    request, response = chat(assessment("wait"))

    async def respond(_: ChatRequest) -> ChatResponse:
        return response

    monkeypatch.setattr(upscale.main.orchestrator, "respond", respond)
    reply = client.post("/chat", json=request.model_dump(mode="json"))
    assert reply.status_code == 200 and reply.json() == response.model_dump(mode="json")
    recorder.flush()
    [d] = rows(store, "decision")
    [obs] = asyncio.run(isolated_outcome_store.decisions())
    assert d.links["decision_observation_id"] == obs.id and d.component == "analyze"
    assert d.payload["decision"]["action"] == "wait"
    assert set(d.payload["agents"]) == {"opportunity", "technical_analysis", "risk"}
    assert d.asset_id == obs.asset_id


def test_nothing_is_archived_during_replay(archive: tuple[EvidenceRecorder, EvidenceStore]) -> None:
    recorder, store = archive
    with frozen_now(T0):
        hooks.emit("safety", safety(mint(0), T0), pools=[])
    recorder.flush()
    assert rows(store) == [] and recorder.stats.refused_replay == 1


def test_a_full_queue_drops_instead_of_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EvidenceRecorder(EvidenceStore(tmp_path / "e.sqlite3"), max_queue=1)
    monkeypatch.setattr(recorder, "_ensure_thread", lambda: None)
    recorder.submit("safety", safety(mint(0), T0), "test", {"pools": []})
    recorder.submit("safety", safety(mint(1), T0), "test", {"pools": []})
    assert recorder.stats.queued == 1 and recorder.stats.dropped == 1


def test_a_failing_recorder_never_breaks_production(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def submit(self, *args: Any) -> None:
            raise RuntimeError("disk full")

    previous = hooks.installed()
    hooks.install(Broken())
    try:
        fake = FakeSolanaRpc()
        setup_token(fake)
        snap = asyncio.run(service(fake).get_snapshot(MINT, (POOL_REF,)))
        assert snap.mint == MINT
    finally:
        hooks.install(previous)


def test_archiving_changes_no_production_result(
    archive: tuple[EvidenceRecorder, EvidenceStore], tmp_path: Path
) -> None:
    recorder, _ = archive

    def ranking() -> str:
        store = ScoutSnapshotStore(":memory:")
        listing = Listing(provider="DEX Screener", kind="new", name="new_pools", fetched_at=T0)
        c = build_candidates([dex_pool(0, T0)], listing, ScoutConfig()).candidates[0]
        result = asyncio.run(GrowthScoutService(store, GrowthConfig(), now=lambda: T0).rank([c]))
        hooks.emit("scout", result)
        return result.model_dump_json()

    on = ranking()
    hooks.install(None)
    off = ranking()
    hooks.install(recorder)
    assert on == off


# --- Replay Lab consumes archived evidence -------------------------------------------------------


def evidence_db(
    tmp_path: Path, *snapshots: OnchainSafetySnapshot, social_at: datetime | None = None
) -> EvidenceStore:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    for s in snapshots:
        for r in payloads.safety(s, []):
            store.append(r)
    if social_at is not None:
        for r in payloads.social(momentum(mint(0), "ACCELERATING", computed_at=social_at)):
            store.append(r)
    return EvidenceStore(tmp_path / "evidence.sqlite3", read_only=True)


def replay(tmp_path: Path, evidence: EvidenceStore | None, db: str, **kw: Any) -> Any:
    archive_path = tmp_path / "scout.sqlite3"
    if not archive_path.exists():
        build_archive(archive_path)
    h = Harness(tmp_path, archive_path, db=db)
    h.runner.evidence = evidence
    job = h.run(config(**kw))
    stored = h.store.decision(h.store.samples(job)[0].id)
    assert stored is not None
    return stored[0]


def test_replay_uses_archived_safety_at_or_before_t(tmp_path: Path) -> None:
    with_safety = replay(
        tmp_path, evidence_db(tmp_path, safety(mint(0), T0 - timedelta(minutes=5))), "a.sqlite3"
    )
    without = replay(tmp_path, None, "b.sqlite3")
    onchain = with_safety.agents["onchain_safety"]
    assert onchain.status == "ok" and onchain.findings["snapshot"]["mint"] == mint(0)
    assert with_safety.availability["onchain_safety"].startswith(
        "AVAILABLE: archived 5 min before T"
    )
    assert "onchain_safety" not in with_safety.agents_unavailable
    assert with_safety.scout.safety_status == "SAFETY_CHECKS_COMPLETE"
    missing = " ".join(with_safety.agents["opportunity"].findings["missing_evidence"])
    assert "authorit" not in missing and "holder concentration" not in missing
    # Without archived safety: unavailable, exactly as before.
    assert "onchain_safety" not in without.agents
    assert without.availability["onchain_safety"].startswith("NOT_COLLECTED")
    assert without.scout.safety_status == "INSUFFICIENT_SAFETY_DATA"


def test_later_safety_evidence_never_affects_an_earlier_decision(tmp_path: Path) -> None:
    before = safety(mint(0), T0 - timedelta(minutes=5))
    # After T: a mint authority appears and holders concentrate.
    after = safety(
        mint(0), T0 + timedelta(minutes=1), mint_authority="Evil111", top1=80.0, top10=99.0
    )
    only_before = replay(tmp_path / "x", evidence_db(_mk(tmp_path / "x"), before), "a.sqlite3")
    both = replay(tmp_path / "y", evidence_db(_mk(tmp_path / "y"), before, after), "b.sqlite3")
    assert only_before.model_dump(exclude={"versions"}) == both.model_dump(exclude={"versions"})
    only_after = replay(tmp_path / "z", evidence_db(_mk(tmp_path / "z"), after), "c.sqlite3")
    assert "onchain_safety" not in only_after.agents  # invisible at T
    assert only_after.availability["onchain_safety"].startswith("NOT_COLLECTED")


def _mk(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_stale_or_failed_safety_is_not_used_as_current(tmp_path: Path) -> None:
    stale = replay(
        _mk(tmp_path / "s"),
        evidence_db(tmp_path / "s", safety(mint(0), T0 - timedelta(hours=3))),
        "a.sqlite3",
    )
    assert "onchain_safety" not in stale.agents
    store = EvidenceStore(_mk(tmp_path / "f") / "evidence.sqlite3")
    for r in payloads.safety_failed(
        "solana", mint(0), "Helius", ProviderRateLimitedError("429"), T0 - timedelta(minutes=2)
    ):
        store.append(r)
    failed = replay(
        tmp_path / "f",
        EvidenceStore(tmp_path / "f" / "evidence.sqlite3", read_only=True),
        "b.sqlite3",
    )
    assert failed.agents["onchain_safety"].status == "error"  # as production saw it at T
    assert failed.availability["onchain_safety"].startswith("RATE_LIMITED")


def test_later_social_evidence_is_invisible(tmp_path: Path) -> None:
    later = replay(
        _mk(tmp_path / "l"),
        evidence_db(tmp_path / "l", social_at=T0 + timedelta(minutes=1)),
        "a.sqlite3",
        mode="MARKET_PLUS_SOCIAL",
    )
    earlier = replay(
        _mk(tmp_path / "e"),
        evidence_db(tmp_path / "e", social_at=T0 - timedelta(minutes=5)),
        "b.sqlite3",
        mode="MARKET_PLUS_SOCIAL",
    )
    assert later.social["status"] == "SOCIAL_UNAVAILABLE"
    assert (
        earlier.social["status"] == "RECORDED_AT_OR_BEFORE_T"
        and earlier.social["age_minutes"] == 5.0
    )


def test_replay_never_falls_back_to_current_chain_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def forbidden(*args: Any, **kw: Any) -> Any:
        raise AssertionError("current on-chain state must never fill an archive gap")

    monkeypatch.setattr(services.provider_registry, "onchain_safety", forbidden)
    if services.solana_safety_service is not None:
        monkeypatch.setattr(services.solana_safety_service, "get_snapshot", forbidden)
    record = replay(tmp_path, None, "a.sqlite3")
    assert "onchain_safety" not in record.agents
    clock = HistoricalClock(T0, timedelta(hours=26))
    svc = PointInTimeSafetyService(clock, None, None, "Helius")
    with pytest.raises(MarketDataUnavailableError):
        asyncio.run(svc.get_snapshot(mint(0)))
    with pytest.raises(MarketDataUnavailableError):
        asyncio.run(svc.provider.fetch_mint(mint(0)))
    future = PointInTimeSafetyService(
        clock, safety(mint(0), T0 + timedelta(minutes=1)), None, "Helius"
    )
    with pytest.raises(LookaheadError):
        asyncio.run(future.get_snapshot(mint(0)))


# --- Enrichment -----------------------------------------------------------------------------------


class FakeSafety:
    def __init__(self, free: int = 20) -> None:
        self.calls: list[str] = []
        self.free = free

    async def get_snapshot(self, token: str, pools: Any = ()) -> OnchainSafetySnapshot:
        self.calls.append(token)
        snap = safety(token, T0)
        hooks.emit("safety", snap, pools=list(pools))
        return snap

    def available_calls(self) -> int:
        return self.free


def ranked_result(n: int) -> GrowthScoutResult:
    store = ScoutSnapshotStore(":memory:")
    listing = Listing(provider="DEX Screener", kind="new", name="new_pools", fetched_at=T0)
    cands = [
        build_candidates([dex_pool(i, T0)], listing, ScoutConfig()).candidates[0] for i in range(n)
    ]
    result = asyncio.run(GrowthScoutService(store, GrowthConfig(), now=lambda: T0).rank(cands))
    everyone = [*result.candidates, *result.unranked]
    for i, g in enumerate(everyone, start=1):
        g.rank = i
    result.candidates = everyone
    return result


async def _no_sleep(_: float) -> None:
    return None


def test_enrichment_is_bounded_prioritized_and_deduplicated(
    archive: tuple[EvidenceRecorder, EvidenceStore],
) -> None:
    recorder, store = archive
    result = ranked_result(5)
    fake = FakeSafety()
    e = SafetyEnrichment(EnrichmentSettings(enabled=True, max_per_refresh=2), fake, store,
                         sleep=_no_sleep, now=lambda: T0 + timedelta(minutes=1))  # fmt: skip
    report = asyncio.run(e.after_scan(result))
    assert report.fetched == 2 and fake.calls == [g.address for g in result.candidates[:2]]
    recorder.flush()
    assert {r.component for r in rows(store, "safety")} == {"safety_enrichment"}
    # Recently enriched tokens are skipped next time: the next best are looked up.
    again = asyncio.run(e.after_scan(result))
    assert fake.calls[2:] == [g.address for g in result.candidates[2:4]] and again.fetched == 2


def test_enrichment_yields_to_production(archive: tuple[EvidenceRecorder, EvidenceStore]) -> None:
    _, store = archive
    result = ranked_result(3)
    busy = SafetyEnrichment(EnrichmentSettings(enabled=True), FakeSafety(), store,
                            busy=lambda: "Analyze is active", sleep=_no_sleep)  # fmt: skip
    r = asyncio.run(busy.after_scan(result))
    assert r.status == "deferred" and r.reason == "Analyze is active" and r.fetched == 0
    tight = FakeSafety(free=10)
    capped = SafetyEnrichment(EnrichmentSettings(enabled=True), tight, store, sleep=_no_sleep)
    r = asyncio.run(capped.after_scan(result))
    assert r.status == "deferred" and "kept for Analyze" in (r.reason or "") and tight.calls == []
    off = SafetyEnrichment(load_settings(None, None), FakeSafety(), store, sleep=_no_sleep)
    assert asyncio.run(off.after_scan(result)).status == "disabled"
    assert (
        load_settings("1", "999").max_per_refresh == 20
        and load_settings("1", "x").max_per_refresh == 3
    )
    assert load_settings("0", "5").enabled is False


# --- Status, CLI, API, deployment paths ------------------------------------------------------------


def test_status_coverage_and_readiness(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "e.sqlite3")
    old = NOW - timedelta(days=2)
    for asset in ("solana:a", "solana:b"):
        store.append(pending("market", asset, old - timedelta(minutes=1)))
        store.append(pending("scout", asset, old))
    store.append(pending("safety", "solana:a", old - timedelta(minutes=30)))
    store.append(
        pending(
            "safety",
            "solana:b",
            old,
            availability="PROVIDER_FAILED",
            reason="timeout",
            provider="Helius",
        )
    )
    report = status(store, NOW)
    cov, ready = report["coverage"], report["readiness"]
    assert (
        cov["scout_observations"] == 2 and cov["market_pct"] == 100.0 and cov["safety_pct"] == 50.0
    )
    assert ready["full_opportunity_evaluation_possible"] == 1 and ready["replay_usable"] == 1
    assert report["provider_failures"] == {"Helius (PROVIDER_FAILED)": 1}
    assert (
        "not a measure of profitability" in report["label"].lower()
        or "Not a measure" in report["label"]
    )


def test_cli_status_readiness_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = EvidenceStore(tmp_path / "e.sqlite3")
    for r in payloads.safety(safety(mint(0), T0), []):
        store.append(r)
    db = str(tmp_path / "e.sqlite3")
    assert evidence_cli(["--db", db, "status"]) == 0
    assert evidence_cli(["--db", db, "readiness"]) == 0
    assert (
        evidence_cli(
            [
                "--db",
                db,
                "show",
                "--asset",
                f"solana:{mint(0)}",
                "--kind",
                "safety",
                "--at",
                T0.isoformat(),
            ]
        )
        == 0
    )
    assert evidence_cli(["--db", db, "show", "--asset", f"solana:{mint(0)}", "--kind", "safety",
                         "--at", (T0 - timedelta(seconds=1)).isoformat()]) == 1  # fmt: skip
    out = capsys.readouterr().out
    assert "coverage over" in out and '"availability": "AVAILABLE"' in out


def test_evidence_status_endpoint(client: TestClient) -> None:
    body = client.get("/evidence/status").json()
    assert body["enabled"] is True and "coverage" in body and body["enrichment"]["enabled"] is False


def test_railway_path_defaults_next_to_the_scout_database(tmp_path: Path) -> None:
    env = {**os.environ, "UPSCALE_SCOUT_DB": "/data/scout.sqlite3"}
    env.pop("UPSCALE_EVIDENCE_DB", None)
    out = subprocess.run(
        [sys.executable, "-c", "from upscale.config import EVIDENCE_DB_PATH as p; print(p)"],
        capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1], check=True,
    )  # fmt: skip
    assert out.stdout.strip() == "/data/evidence.sqlite3"
    assert Path(services.evidence_store.path).parent == Path(os.environ["UPSCALE_SCOUT_DB"]).parent  # type: ignore[union-attr]
