"""Chain clock vs local clock: Solana block times (whole seconds, chain time) may lead
Radar's local ``fetched_at`` by up to ``CHAIN_CLOCK_TOLERANCE_S``; anti-lookahead
(``fetched_at <= observed_at``) stays strict; an aborted scan never stays RUNNING; an old
Radar database is refused. Offline (MockTransport)."""

import asyncio
import sqlite3
from datetime import timedelta
from typing import Any

import pytest

from tests.radar_fakes import MINT, POOL, T0, Clock, FakeChain, addr, make_service, tx
from upscale.services.radar.config import DB_SCHEMA_VERSION, load_settings
from upscale.services.radar.features import build_snapshot, timing_clusters
from upscale.services.radar.models import (
    CHAIN_CLOCK_TOLERANCE_S,
    ParsedTx,
    RadarCausalityError,
    RadarSchemaError,
    WalletDelta,
    iso,
    ts,
)
from upscale.services.radar.repository import FlowRow, RadarRepository
from upscale.services.radar.service import RadarService

CID = f"solana:{MINT}"
CREATED = T0 - timedelta(hours=1)
F = 1_790_000_000.0  # a fetch time (exactly representable, like +0.5 / +2.0 offsets)
A, B = addr("WaA"), addr("WaB")


def parsed(sig: str, block_time: float, wallet: str = A) -> ParsedTx:
    flow = WalletDelta(wallet, "TOKEN_INFLOW", 1_000, "TRACKED_POOL_COUNTERPARTY",
                       "NORMAL_WALLET", True)  # fmt: skip
    return ParsedTx(sig, 1, block_time, wallet, False, (flow,), False, 0)


def repo_with_target(tmp_path: Any) -> RadarRepository:
    repo = RadarRepository(tmp_path / "radar.sqlite3")
    repo.upsert_target(CID, MINT, POOL, "pumpswap", None, None, "manual", F - 3600)
    return repo


def rows(path: Any, sql: str) -> list[Any]:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


# --- 1-4, 6: storage rule ----------------------------------------------------------------


def test_block_time_equal_to_fetch_time_needs_no_tolerance(tmp_path: Any) -> None:
    repo = repo_with_target(tmp_path)
    got = repo.record_tx(CID, parsed("s", F), F, "x", None, None)
    assert got.inserted and got.flows == 1 and got.chain_clock_ahead_s is None
    db = tmp_path / "radar.sqlite3"
    assert rows(db, "SELECT block_time, chain_clock_ahead_s FROM radar_tx") == [(F, None)]
    assert rows(db, "SELECT chain_clock_ahead_s FROM radar_wallet_flows") == [(None,)]


@pytest.mark.parametrize("lead", [0.5, CHAIN_CLOCK_TOLERANCE_S])
def test_lead_within_tolerance_is_stored_raw_with_the_lead(tmp_path: Any, lead: float) -> None:
    repo = repo_with_target(tmp_path)
    got = repo.record_tx(CID, parsed("s", F + lead), F, "x", None, None)
    assert got.inserted and got.chain_clock_ahead_s == lead
    db = tmp_path / "radar.sqlite3"
    # The raw chain block time is preserved, never clamped in storage.
    assert rows(db, "SELECT block_time, fetched_at, chain_clock_ahead_s FROM radar_tx") == [
        (F + lead, F, lead)
    ]
    assert rows(db, "SELECT block_time, chain_clock_ahead_s FROM radar_wallet_flows") == [
        (F + lead, lead)
    ]


def test_lead_beyond_tolerance_still_raises_and_stores_nothing(tmp_path: Any) -> None:
    repo = repo_with_target(tmp_path)
    with pytest.raises(RadarCausalityError, match="chain-clock tolerance 2.0s"):
        repo.record_tx(CID, parsed("s", F + 2.001), F, "x", None, None)
    db = tmp_path / "radar.sqlite3"
    assert rows(db, "SELECT COUNT(*) FROM radar_tx") == [(0,)]
    assert rows(db, "SELECT COUNT(*) FROM radar_wallet_flows") == [(0,)]
    assert rows(db, "SELECT COUNT(*) FROM radar_wallet_entries") == [(0,)]


def test_sqlite_checks_enforce_the_rule_too(tmp_path: Any) -> None:
    repo = repo_with_target(tmp_path)
    repo.db()
    conn = sqlite3.connect(str(tmp_path / "radar.sqlite3"))
    sql = ("INSERT INTO radar_tx (canonical_id, signature, block_time, fetched_at, failed, "
           "provider, chain_clock_ahead_s) VALUES (?, ?, ?, ?, 0, 'x', ?)")  # fmt: skip
    for sig, bt, lead in (
        ("beyond", F + 2.5, 2.5),  # past the tolerance
        ("unrecorded", F + 0.5, None),  # a lead must be recorded
        ("wrong", F + 0.5, 0.25),  # ...exactly
        ("spurious", F, 0.5),  # no lead, nothing recorded
    ):
        with pytest.raises(sqlite3.IntegrityError), conn:
            conn.execute(sql, (CID, sig, bt, F, lead))
    conn.close()


# --- 5, 7, 8: anti-lookahead and effective event times -----------------------------------


def test_later_fetched_evidence_is_excluded_even_with_an_earlier_block_time(
    tmp_path: Any,
) -> None:
    repo = repo_with_target(tmp_path)
    repo.record_tx(CID, parsed("late", F - 100, B), F + 10, "x", None, None)
    inp = repo.load_inputs(CID, F)
    assert inp.flows == [] and inp.entries == []
    assert [f.signature for f in repo.load_inputs(CID, F + 10).flows] == ["late"]


def test_tolerated_lead_never_puts_a_feature_event_after_its_evidence(tmp_path: Any) -> None:
    repo = repo_with_target(tmp_path)
    repo.record_tx(CID, parsed("calm", F - 10, B), F - 10, "x", None, None)
    # Chain time 1.5 s ahead of the fetch, and of the snapshot taken at the fetch time.
    stored = repo.record_tx(CID, parsed("ahead", F + 1.5, A), F, "x", None, early_cutoff=F)
    assert stored.chain_clock_ahead_s == 1.5
    inp = repo.load_inputs(CID, F)  # as_of == fetched_at: included (strict, not loosened)
    by_sig = {f.signature: f for f in inp.flows}
    ahead = by_sig["ahead"]
    assert ahead.block_time == F + 1.5 > inp.as_of  # raw, for provenance
    assert ahead.event_time == F <= inp.as_of  # what features use
    assert ahead.is_early  # window membership by event time: F <= cutoff F
    entry = next(e for e in inp.entries if e.wallet == A)
    assert entry.first_block_time == F and entry.block_time_known_at == F
    # No negative ages / deltas anywhere.
    assert all(inp.as_of - f.event_time >= 0 for f in inp.flows if f.event_time is not None)
    assert all(inp.as_of - e.first_block_time >= 0 for e in inp.entries if e.first_block_time)
    clusters = timing_clusters(inp.flows, window_seconds=60, min_wallets=2)
    assert clusters == [(F - 10, 2)]
    target = repo.get_target(CID)
    assert target is not None
    body = build_snapshot(target, inp, load_settings({}), "x")
    act = body["activity"]
    assert act["last_block_time"] == iso(F) and act["first_block_time"] == iso(F - 10)
    assert act["chain_clock_lead"]["flows"] == 1
    assert act["chain_clock_lead"]["max_ahead_s"] == 1.5
    assert act["chain_clock_lead"]["tolerance_s"] == CHAIN_CLOCK_TOLERANCE_S


def test_early_entries_are_marked_by_effective_time(tmp_path: Any) -> None:
    repo = repo_with_target(tmp_path)
    repo.record_tx(CID, parsed("ahead", F + 1.0, A), F, "x", None, None)  # cutoff unknown yet
    repo.mark_early_entries(CID, cutoff=F, at=F + 60)
    inp = repo.load_inputs(CID, F + 60)
    assert [e.early for e in inp.entries] == [True]


def timing_flow(sig: str, wallet: str, block_time: float, fetched_at: float) -> FlowRow:
    lead = block_time - fetched_at if block_time > fetched_at else None
    return FlowRow(sig, wallet, "TOKEN_INFLOW", 1, "UNVERIFIED", block_time, fetched_at, None,
                   chain_clock_ahead_s=lead)  # fmt: skip


def test_timing_clusters_use_effective_times() -> None:
    flows = [timing_flow("a", A, F + 2.0, F), timing_flow("b", B, F, F)]
    # Raw times would start the window at F; effective ones are both F, deltas >= 0.
    assert timing_clusters(flows, window_seconds=0, min_wallets=2) == [(F, 2)]


# --- 9, 10: aborted scans ----------------------------------------------------------------


def pool_tx(sig: str, at: Any, wallet: str, err: Any = None) -> dict[str, Any]:
    return tx(sig, at, wallet, err=err, pre=[(POOL, 100_000)],
              post=[(POOL, 99_000), (wallet, 1_000)])  # fmt: skip


def setup(tmp_path: Any) -> tuple[RadarService, Clock, FakeChain]:
    clock, chain = Clock(), FakeChain(holders={addr("HoA"): 600_000, addr("HoB"): 400_000})
    chain.add(POOL, pool_tx("old1", CREATED + timedelta(minutes=1), A),
              pool_tx("old2", CREATED + timedelta(minutes=2), B))  # fmt: skip
    svc = make_service(tmp_path, chain, clock, max_retries=0)
    svc.add_target(MINT, POOL, "pumpswap", CREATED)
    return svc, clock, chain


def target_state(svc: RadarService) -> tuple[str | None, float | None, int]:
    t = svc.repo.get_target(CID)
    assert t is not None
    return t.last_signature, t.last_scan_at, t.snapshots_taken


def scans(svc: RadarService) -> list[tuple[str, str, int, str]]:
    return rows(svc.settings.db_path, "SELECT kind, status, signatures_listed, reasons_json "
                                      "FROM radar_scans ORDER BY id")  # fmt: skip


def test_aborted_causality_scan_is_terminal_keeps_cursor_and_saves_no_snapshot(
    tmp_path: Any,
) -> None:
    svc, clock, chain = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    before = target_state(svc)
    assert before[0] == "old2" and before[2] == 1
    clock.advance(600)
    now = clock.now()
    chain.add(
        POOL,
        pool_tx("failed_ok", now - timedelta(seconds=30), A, err={"x": 1}),  # stored first
        pool_tx("ok", now - timedelta(seconds=20), B),
        pool_tx("future", now + timedelta(seconds=5), A),  # 5 s ahead: beyond tolerance
    )
    with pytest.raises(RadarCausalityError):
        asyncio.run(svc.snapshot(CID))
    kind, status, listed, reasons = scans(svc)[-1]
    assert (kind, status, listed) == ("activity", "ABORTED", 3)
    assert "aborted: RadarCausalityError" in reasons and "future" in reasons
    assert not [s for s in scans(svc) if s[1] == "RUNNING"]
    assert target_state(svc) == before  # cursor, last_scan_at, snapshots_taken unchanged
    assert len(svc.repo.snapshot_times(CID)) == 1

    # Retry once the local clock has passed the block: everything stored exactly once.
    clock.advance(10)
    chain.params.clear()
    r = asyncio.run(svc.snapshot(CID))
    assert r.saved and r.steps["activity"].status == "AVAILABLE"
    got = [p[0] for m, p in chain.params if m == "getTransaction"]
    assert sorted(got) == ["future", "ok"]  # failed_ok was stored by the aborted scan
    db = svc.settings.db_path
    assert rows(db, "SELECT COUNT(*), COUNT(DISTINCT signature) FROM radar_tx") == [(5, 5)]
    assert rows(db, "SELECT COUNT(*), COUNT(DISTINCT signature || wallet) "
                    "FROM radar_wallet_flows") == [(4, 4)]  # fmt: skip
    assert target_state(svc)[0] == "future" and target_state(svc)[2] == 2
    act = r.body["activity"]
    # The aborted scan makes the window PARTIAL, but its listing isn't counted twice.
    assert act["interacting_wallets"]["status"] == "PARTIAL"
    assert act["signatures_listed"] == 3


def test_abort_on_an_unexpected_exception_also_finishes_the_scan(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, clock, chain = setup(tmp_path)

    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(svc.repo, "record_tx", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(svc.snapshot(CID))
    assert [s[:2] for s in scans(svc)] == [("activity", "ABORTED")]
    assert target_state(svc) == (None, None, 0)


def test_tolerated_lead_is_a_scan_note_not_a_failure(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    clock.advance(10.5)  # local clock half a second behind the newest whole-second block
    chain.history[POOL].append(pool_tx("ahead", clock.now() + timedelta(seconds=0.5), B))
    r = asyncio.run(svc.snapshot(CID))
    step = r.steps["activity"]
    assert step.status == "AVAILABLE"
    note = ("chain clock ahead of local observation clock within 2.0s tolerance: "
            "1 transaction(s), 1 flow(s), max 0.500s ahead")  # fmt: skip
    assert any(x.startswith(note) for x in step.reasons)
    assert note in scans(svc)[0][3]
    assert rows(svc.settings.db_path, "SELECT signature, chain_clock_ahead_s FROM radar_tx "
                                      "WHERE chain_clock_ahead_s IS NOT NULL") == [("ahead", 0.5)]  # fmt: skip
    assert r.body["activity"]["chain_clock_lead"]["flows"] == 1


# --- 11: RUNNING scans -------------------------------------------------------------------


def test_running_scans_are_never_snapshot_inputs(tmp_path: Any) -> None:
    repo = repo_with_target(tmp_path)
    repo.record_scan(CID, "activity", F - 5, F - 5, "x", "RUNNING", 1000, 0, 0, False,
                     (None, None), [])  # fmt: skip
    done = repo.record_scan(CID, "activity", F - 4, F - 4, "x", "RUNNING", 7, 0, 0, False,
                            (None, None), [])  # fmt: skip
    repo.finish_scan(done, "AVAILABLE", 0, 0, False, (None, None), [], F - 3)
    inp = repo.load_inputs(CID, F)
    assert [(s.id, s.status) for s in inp.scans] == [(done, "AVAILABLE")]
    target = repo.get_target(CID)
    assert target is not None
    body = build_snapshot(target, inp, load_settings({}), "x")
    assert body["coverage"]["activity_scans"] == 1
    assert body["activity"]["signatures_listed"] == 7


# --- 12: old schema ----------------------------------------------------------------------

V1_TX = """CREATE TABLE radar_tx (canonical_id TEXT NOT NULL, signature TEXT NOT NULL,
    slot INTEGER, block_time REAL, fetched_at REAL NOT NULL, fee_payer TEXT,
    failed INTEGER NOT NULL, scan_id INTEGER, provider TEXT NOT NULL,
    PRIMARY KEY (canonical_id, signature), CHECK (block_time IS NULL OR block_time <= fetched_at))"""


def old_db(path: Any, meta: bool = True) -> list[Any]:
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute(V1_TX)
        if meta:
            conn.execute("CREATE TABLE radar_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO radar_meta VALUES ('schema_version', '1')")
    names = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    conn.close()
    return names


@pytest.mark.parametrize("meta", [True, False])
def test_old_radar_schema_is_refused_clearly_and_left_untouched(tmp_path: Any, meta: bool) -> None:
    path = tmp_path / "radar.sqlite3"
    before = old_db(path, meta)
    repo = RadarRepository(path)
    with pytest.raises(RadarSchemaError, match="start a fresh Radar database") as err:
        repo.get_target(CID)
    assert ("schema version 1" if meta else "schema version unknown") in str(err.value)
    with pytest.raises(RadarSchemaError):  # every use, not just the first
        repo.record_tx(CID, parsed("s", F), F, "x", None, None)
    assert rows(path, "SELECT name, sql FROM sqlite_master ORDER BY name") == before


def test_new_and_current_databases_open(tmp_path: Any) -> None:
    path = tmp_path / "radar.sqlite3"
    sqlite3.connect(str(path)).close()  # an empty file is a new database
    repo = RadarRepository(path)
    assert repo.get_meta("schema_version") == str(DB_SCHEMA_VERSION)
    repo.close()
    assert RadarRepository(path).get_meta("schema_version") == str(DB_SCHEMA_VERSION)  # reopens


def test_service_observed_at_is_never_before_its_inputs(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    r = asyncio.run(svc.snapshot(CID))
    inp = svc.repo.load_inputs(CID, ts(r.observed_at))
    assert inp.flows and all(f.fetched_at <= ts(r.observed_at) for f in inp.flows)
