"""Safety V2 Phase 4A: durable Radar evidence capture. Radar is read read-only during
collection and copied into Safety's append-only tables; snapshots and rebuilds never read
Radar, so Radar replacing, deleting or losing rows can't change them. Offline: Radar
databases are local fixtures built with Radar's own schema."""

import asyncio
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.safety_v2_fakes import MINT, Clock, FakeRpc, HolderChain, addr, make_service, mint_value
from upscale.services.radar.repository import RadarRepository
from upscale.services.safety_v2 import service as service_module
from upscale.services.safety_v2.models import ts
from upscale.services.safety_v2.sources import (
    RADAR_SCHEMA_VERSION,
    CapturedFlow,
    RadarCapture,
    read_radar_capture,
)

CID = f"solana:{MINT}"
T2 = datetime(2026, 10, 9, tzinfo=UTC)
NOW = ts(T2)
SUPPLY = 1_000_000_000
WHALE, STRANGER = addr("WhaLe"), addr("StrangerAA")
DEPLOYER, CREATOR, OTHER = addr("DepLoyer"), addr("CreatorAA"), addr("QtherWa")
CAPTURE_TABLES = ("safety_radar_wallet_proofs", "safety_radar_creator_evidence",
                  "safety_radar_flow_evidence", "safety_radar_activity_coverage")  # fmt: skip


# --- Radar fixtures (real Radar schema, written directly as Radar would) ------------------------


def radar(path: Path) -> Path:
    RadarRepository(path).db()
    RadarRepository(path).close()
    return path


def _exec(path: Path, sql: str, args: tuple[Any, ...] = ()) -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(sql, args)
    conn.close()


def flow(path: Path, wallet: str, at: float, *, sig: str, direction: str = "TOKEN_INFLOW",
         signer: int = 1, participant: str = "NORMAL_WALLET", cid: str = CID,
         amount: str = "5") -> None:  # fmt: skip
    _exec(path, "INSERT INTO radar_wallet_flows (canonical_id, signature, wallet, direction, "
          "amount_raw, counterparty, participant, signer, block_time, fetched_at, provider) "
          "VALUES (?, ?, ?, ?, ?, 'UNVERIFIED', ?, ?, ?, ?, 'fake')",
          (cid, sig, wallet, direction, amount, participant, signer, at - 10, at))  # fmt: skip


def creator(path: Path, role: str, status: str, identity: str | None, at: float) -> None:
    """Radar's own write: INSERT OR REPLACE (one row per target and role)."""
    _exec(path, "INSERT OR REPLACE INTO radar_creators (canonical_id, role, status, identity, "
          "method, signature, block_time, determined_at, provider, provenance_json) "
          "VALUES (?, ?, ?, ?, 'M', 'sigC', ?, ?, 'fake', '{\"note\": \"n\"}')",
          (CID, role, status, identity, at - 100, at))  # fmt: skip


def scan(path: Path, scan_id: int, at: float, status: str = "AVAILABLE") -> None:
    _exec(path, "INSERT INTO radar_scans (id, canonical_id, kind, started_at, fetched_at, "
          "provider, status, signatures_listed, txs_parsed, txs_skipped, reasons_json, "
          "head_listing_complete) VALUES (?, ?, 'activity', ?, ?, 'fake', ?, 10, 9, 1, '[]', 1)",
          (scan_id, CID, at - 5, at, status))  # fmt: skip


def gap(path: Path, gap_id: int, at: float) -> None:
    _exec(path, "INSERT INTO radar_activity_gaps (id, canonical_id, status, opened_at, "
          "opened_scan_id, updated_at, before_signature, until_signature, reason) "
          "VALUES (?, ?, 'OPEN', ?, 1, ?, 'b', 'u', 'page cap')", (gap_id, CID, at, at))  # fmt: skip


def digest(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def setup(tmp_path: Path, radar_path: Path | None) -> tuple[Any, Clock, FakeRpc]:
    chain = HolderChain()
    chain.add(WHALE, 8 * SUPPLY // 100)
    chain.add(STRANGER, 3 * SUPPLY // 100)
    rpc = FakeRpc({MINT: mint_value()}, holders=chain)
    svc, clock, rpc = make_service(
        tmp_path, rpc, helius=True, clock=Clock(T2),
        radar_db_path=str(radar_path) if radar_path else None,
    )  # fmt: skip
    svc.add_target(MINT)
    return svc, clock, rpc


def collect(svc: Any) -> Any:
    return asyncio.run(svc.collect(CID, holders=True))


def classes(body: dict[str, Any]) -> dict[str, str]:
    return {o["owner"]: o["classification"] for o in body["holders"]["top_owners"]}


def count(svc: Any, table: str) -> int:
    return int(svc.repo.db().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


# --- wallet proof (Phase 2 migration) --------------------------------------------------------


def test_signer_proof_is_captured_and_makes_a_normal_wallet(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW - 60, sig="s1")
    before = digest(r)
    svc, _, _ = setup(tmp_path, r)
    res = asyncio.run(svc.snapshot(CID, holders=True))
    assert res.collected.radar_capture_status == "CAPTURED"
    assert classes(res.body) == {WHALE: "NORMAL_WALLET", STRANGER: "UNKNOWN"}
    row = svc.repo.captured_facts("safety_radar_wallet_proofs", CID, NOW)[0]
    assert (row["wallet"], row["kind"], row["signature"], row["source_system"],
            row["radar_schema_version"]) == (WHALE, "SIGNER_FLOW", "s1", "radar", "3")  # fmt: skip
    proof = res.body["holders"]["top_owners"][0]["wallet_proof"]
    assert proof["capture_row_id"] == row["id"] and proof["captured_at"] is not None
    assert digest(r) == before  # Radar byte-identical


def test_normal_wallet_survives_radar_flow_deletion_and_db_removal(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW - 60, sig="s1")
    svc, clock, _ = setup(tmp_path, r)
    res = asyncio.run(svc.snapshot(CID, holders=True))
    _exec(r, "DELETE FROM radar_wallet_flows")  # Radar retention
    assert svc.rebuild(res.snapshot_id).status == "REPRODUCED"
    clock.advance(60)
    assert classes(svc.build(CID, ts(clock.now())))[WHALE] == "NORMAL_WALLET"
    r.unlink()  # the Radar database is gone entirely
    assert svc.rebuild(res.snapshot_id).status == "REPRODUCED"
    assert classes(svc.build(CID, ts(clock.now())))[WHALE] == "NORMAL_WALLET"


def test_a_proof_captured_after_as_of_is_unusable_even_if_radar_knew_earlier(
    tmp_path: Path,
) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    svc, clock, _ = setup(tmp_path, r)
    first = asyncio.run(svc.snapshot(CID, holders=True))
    assert classes(first.body)[WHALE] == "UNKNOWN"
    flow(r, WHALE, NOW - 3600, sig="early")  # Radar knew it an hour before as_of ...
    clock.advance(60)
    collect(svc)  # ... but Safety captured it only now
    assert classes(svc.build(CID, NOW))[WHALE] == "UNKNOWN"  # source < as_of < captured
    assert classes(svc.build(CID, ts(clock.now())))[WHALE] == "NORMAL_WALLET"
    assert svc.rebuild(first.snapshot_id).status == "REPRODUCED"


def test_a_proof_whose_source_time_is_after_as_of_is_unusable(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW + 30, sig="late")  # Radar's clock ahead of Safety's capture
    svc, clock, _ = setup(tmp_path, r)
    collect(svc)
    assert classes(svc.build(CID, NOW))[WHALE] == "UNKNOWN"
    clock.advance(60)
    assert classes(svc.build(CID, ts(clock.now())))[WHALE] == "NORMAL_WALLET"


def test_capture_is_idempotent(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW - 60, sig="s1")
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 50)
    flow(r, DEPLOYER, NOW - 40, sig="o1", direction="TOKEN_OUTFLOW")
    scan(r, 1, NOW - 30)
    svc, clock, _ = setup(tmp_path, r)
    first = collect(svc)
    counts = {t: count(svc, t) for t in CAPTURE_TABLES}
    clock.advance(60)
    second = collect(svc)
    assert {t: count(svc, t) for t in CAPTURE_TABLES} == counts
    assert first.radar_new_facts == sum(counts.values()) and second.radar_new_facts == 0
    rows = (
        svc.repo.db()
        .execute("SELECT new_facts, repeated_facts FROM safety_radar_captures ORDER BY id")
        .fetchall()
    )
    assert rows == [(sum(counts.values()), 0), (0, sum(counts.values()))]


def test_wallet_entry_roll_up_is_durable_proof_too(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    _exec(r, "INSERT INTO radar_wallet_entries (wallet, canonical_id, first_block_time, "
          "first_direction, first_fetched_at, wallet_evidence_at) VALUES (?, 'solana:x', ?, "
          "'TOKEN_INFLOW', ?, ?)", (WHALE, NOW - 900, NOW - 800, NOW - 800))  # fmt: skip
    svc, _, _ = setup(tmp_path, r)
    collect(svc)
    row = svc.repo.captured_facts("safety_radar_wallet_proofs", CID, NOW)[0]
    assert (row["kind"], row["signature"], row["source_time"]) == ("WALLET_ENTRY", None, NOW - 800)
    assert classes(svc.build(CID, NOW))[WHALE] == "NORMAL_WALLET"


def test_without_captured_proof_unknown_stays_unknown(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW - 60, sig="x", signer=0, participant="UNKNOWN")  # not a signer
    svc, _, _ = setup(tmp_path, r)
    collect(svc)
    assert count(svc, "safety_radar_wallet_proofs") == 0
    assert classes(svc.build(CID, NOW)) == {WHALE: "UNKNOWN", STRANGER: "UNKNOWN"}


@pytest.mark.parametrize("kind", ["unconfigured", "missing", "schema_2"])
def test_unusable_radar_is_recorded_honestly(tmp_path: Path, kind: str) -> None:
    path: Path | None = tmp_path / "radar.sqlite3"
    if kind == "unconfigured":
        path = None
    elif kind == "schema_2":
        radar(tmp_path / "radar.sqlite3")
        _exec(tmp_path / "radar.sqlite3",
              "UPDATE radar_meta SET value = '2' WHERE key = 'schema_version'")  # fmt: skip
    svc, _, _ = setup(tmp_path, path)
    got = collect(svc)
    status = {"unconfigured": "NOT_CONFIGURED", "missing": "UNAVAILABLE",
              "schema_2": "INCOMPATIBLE"}[kind]  # fmt: skip
    assert got.radar_capture_status == status
    assert svc.repo.collection(got.collection_id)[0] == "DONE"
    assert all(count(svc, t) == 0 for t in CAPTURE_TABLES)
    if kind == "missing":
        assert not (tmp_path / "radar.sqlite3").exists()


# --- creator / deployer ----------------------------------------------------------------------


def test_creator_roles_are_captured_separately_and_never_upgraded(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    creator(r, "POOL_CREATOR_CANDIDATE", "CANDIDATE", DEPLOYER, NOW - 50)
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 40)  # same address, other role
    svc, _, _ = setup(tmp_path, r)
    collect(svc)
    rows = svc.repo.captured_facts("safety_radar_creator_evidence", CID, NOW)
    assert sorted((x["role"], x["status"], x["address"]) for x in rows) == [
        ("POOL_CREATOR_CANDIDATE", "CANDIDATE", DEPLOYER),
        ("TOKEN_DEPLOYER", "VERIFIED", DEPLOYER),
    ]
    assert json.loads(rows[0]["provenance_json"]) == {"note": "n"}
    conn = svc.repo.db()
    with pytest.raises(sqlite3.IntegrityError):  # a candidate can't be VERIFIED
        conn.execute("UPDATE safety_radar_creator_evidence SET status = 'VERIFIED'")
    cap = conn.execute("SELECT id, collection_id FROM safety_radar_captures").fetchone()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO safety_radar_creator_evidence (canonical_id, capture_id, collection_id, "
            "captured_at, source_system, radar_schema_version, source_key, source_time, role, "
            "status, address, method, radar_provider, provenance_json) VALUES (?, ?, ?, ?, "
            "'radar', '3', 'k', 1, 'POOL_CREATOR_CANDIDATE', 'VERIFIED', ?, 'm', 'p', '{}')",
            (CID, cap[0], cap[1], NOW, CREATOR),
        )


def test_a_replaced_radar_candidate_keeps_the_old_safety_copy(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    creator(r, "POOL_CREATOR_CANDIDATE", "UNAVAILABLE", None, NOW - 50)
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 40)
    svc, clock, _ = setup(tmp_path, r)
    collect(svc)
    early = svc.repo.captured_facts("safety_radar_creator_evidence", CID, NOW)
    clock.advance(3600)
    creator(r, "POOL_CREATOR_CANDIDATE", "CANDIDATE", CREATOR, ts(clock.now()) - 10)  # replaced
    collect(svc)
    assert svc.repo.captured_facts("safety_radar_creator_evidence", CID, NOW) == early
    later = svc.repo.captured_facts("safety_radar_creator_evidence", CID, ts(clock.now()))
    assert sorted((x["role"], x["status"]) for x in later) == [
        ("POOL_CREATOR_CANDIDATE", "CANDIDATE"), ("POOL_CREATOR_CANDIDATE", "UNAVAILABLE"),
        ("TOKEN_DEPLOYER", "VERIFIED"),
    ]  # fmt: skip
    assert sum(x["role"] == "TOKEN_DEPLOYER" for x in later) == 1  # captured once


# --- flows -----------------------------------------------------------------------------------


def test_only_creator_role_token_outflows_of_this_token_are_captured(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 500)
    flow(r, DEPLOYER, NOW - 60, sig="out1", direction="TOKEN_OUTFLOW", amount="1000")
    flow(r, DEPLOYER, NOW - 50, sig="in1", direction="TOKEN_INFLOW")
    flow(r, OTHER, NOW - 40, sig="out2", direction="TOKEN_OUTFLOW")  # another wallet
    flow(r, DEPLOYER, NOW - 30, sig="out3", direction="TOKEN_OUTFLOW", cid="solana:other")
    svc, _, _ = setup(tmp_path, r)
    collect(svc)
    rows = svc.repo.captured_facts("safety_radar_flow_evidence", CID, NOW)
    assert [(x["wallet"], x["signature"], x["direction"], x["amount_raw"]) for x in rows] == [
        (DEPLOYER, "out1", "TOKEN_OUTFLOW", "1000")
    ]
    _exec(r, "DELETE FROM radar_wallet_flows")  # Radar retention
    assert svc.repo.captured_facts("safety_radar_flow_evidence", CID, NOW) == rows
    text = json.dumps(rows).lower()
    assert not any(word in text for word in ("sale", "sell", "dump", "cash-out"))


# --- activity coverage -----------------------------------------------------------------------


def test_activity_coverage_is_captured_and_survives_radar_retention(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    scan(r, 1, NOW - 600)
    scan(r, 2, NOW - 300, status="RUNNING")  # in progress: never captured
    gap(r, 1, NOW - 500)
    svc, clock, _ = setup(tmp_path, r)
    collect(svc)
    first = svc.repo.captured_facts("safety_radar_activity_coverage", CID, NOW)
    assert [(x["kind"], x["radar_id"], x["status"]) for x in first] == [
        ("SCAN", 1, "AVAILABLE"), ("GAP", 1, "OPEN")]  # fmt: skip
    assert json.loads(first[0]["facts_json"])["txs_skipped"] == 1
    _exec(r, "DELETE FROM radar_scans")  # Radar retention
    clock.advance(600)
    _exec(r, "UPDATE radar_activity_gaps SET status = 'CLOSED', updated_at = ?, closed_at = ?, "
          "closed_scan_id = 3 WHERE id = 1", (ts(clock.now()) - 5, ts(clock.now()) - 5))  # fmt: skip
    collect(svc)
    assert svc.repo.captured_facts("safety_radar_activity_coverage", CID, NOW) == first
    later = svc.repo.captured_facts("safety_radar_activity_coverage", CID, ts(clock.now()))
    assert [(x["kind"], x["status"]) for x in later] == [
        ("SCAN", "AVAILABLE"),
        ("GAP", "OPEN"),
        ("GAP", "CLOSED"),
    ]  # every gap state kept


# --- durability / transactions ---------------------------------------------------------------


def test_captured_rows_are_append_only(tmp_path: Path) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW - 60, sig="s1")
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 50)
    flow(r, DEPLOYER, NOW - 40, sig="o1", direction="TOKEN_OUTFLOW")
    scan(r, 1, NOW - 30)
    svc, _, _ = setup(tmp_path, r)
    collect(svc)
    conn = svc.repo.db()
    for table in (*CAPTURE_TABLES, "safety_radar_captures"):
        assert count(svc, table) >= 1
        for sql in (f"UPDATE {table} SET captured_at = captured_at", f"DELETE FROM {table}"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(sql)


def test_a_failing_capture_writes_nothing_and_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW - 60, sig="s1")
    creator(r, "TOKEN_DEPLOYER", "VERIFIED", DEPLOYER, NOW - 50)
    real = read_radar_capture

    def broken(path: str | None, cid: str, wallets: Any) -> RadarCapture:
        good = real(path, cid, wallets)
        bad = CapturedFlow(
            DEPLOYER,
            "x",
            "TOKEN_OUTFLOW",
            "not-a-number",
            None,
            "k",
            1.0,
            True,
            "NORMAL_WALLET",
            "UNVERIFIED",
            None,
        )  # violates a CHECK
        return RadarCapture(good.status, None, RADAR_SCHEMA_VERSION, good.proofs,
                            good.creators, (bad,), ())  # fmt: skip

    monkeypatch.setattr(service_module, "read_radar_capture", broken)
    svc, _, _ = setup(tmp_path, r)
    with pytest.raises(sqlite3.IntegrityError):
        collect(svc)
    assert count(svc, "safety_radar_captures") == 0  # no half batch
    assert all(count(svc, t) == 0 for t in CAPTURE_TABLES)
    statuses = [x[0] for x in svc.repo.db().execute("SELECT status FROM safety_collections")]
    assert statuses == ["ABORTED"]


def test_capture_never_reads_radar_at_build_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = radar(tmp_path / "radar.sqlite3")
    flow(r, WHALE, NOW - 60, sig="s1")
    svc, _, _ = setup(tmp_path, r)
    res = asyncio.run(svc.snapshot(CID, holders=True))

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("Radar read outside collection")

    monkeypatch.setattr(service_module, "read_radar_capture", forbidden)
    assert svc.rebuild(res.snapshot_id).status == "REPRODUCED"
    svc.build(CID, NOW)
