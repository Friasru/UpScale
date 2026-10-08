"""Safety V2 storage, collection lifecycle, anti-lookahead, deterministic hashing and
fingerprinted rebuilds. Offline."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.safety_v2_fakes import AUTH, MINT, FakeRpc, make_service, mint_value
from upscale.services.safety_v2 import service as service_module
from upscale.services.safety_v2.config import DB_SCHEMA_VERSION, RULES_VERSION, SNAPSHOT_SCHEMA
from upscale.services.safety_v2.features import build_body, code_fingerprints
from upscale.services.safety_v2.models import (
    SafetyCausalityError,
    SafetyIdentityError,
    SafetySchemaError,
    SafetyStateError,
    ts,
)
from upscale.services.safety_v2.repository import SafetyRepository, encode_body

CID = f"solana:{MINT}"


def _setup(tmp_path: Path, value: Any = None, **kw: Any) -> Any:
    svc, clock, rpc = make_service(tmp_path, FakeRpc({MINT: value or mint_value()}), **kw)
    svc.add_target(MINT)
    return svc, clock, rpc


# --- schema ---------------------------------------------------------------------------------


def test_new_database_records_schema_versions(tmp_path: Path) -> None:
    repo = SafetyRepository(tmp_path / "s.sqlite3")
    assert repo.get_meta("schema_version") == str(DB_SCHEMA_VERSION)
    assert repo.get_meta("snapshot_schema") == SNAPSHOT_SCHEMA
    with pytest.raises(SafetyStateError):
        repo.set_meta("schema_version", "2")


@pytest.mark.parametrize("version", ["0", "1", "2", "4", None])
def test_unknown_schema_version_is_refused(tmp_path: Path, version: str | None) -> None:
    path = tmp_path / "s.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE safety_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if version is not None:
            conn.execute("INSERT INTO safety_meta VALUES ('schema_version', ?)", (version,))
    conn.close()
    with pytest.raises(SafetySchemaError, match=f"schema version {version or 'unknown'}"):
        SafetyRepository(path).db()


PHASE_2_TABLES = ("safety_holder_balances", "safety_holder_observations", "safety_target_pools")
PHASE_3_TABLES = ("safety_pool_account_observations", "safety_market_pools",
                  "safety_market_observations")  # fmt: skip


@pytest.mark.parametrize(("version", "missing"), [
    ("1", PHASE_3_TABLES + PHASE_2_TABLES),  # a Phase 1 database
    ("2", PHASE_3_TABLES),  # a Phase 2 database
])  # fmt: skip
def test_an_older_phase_database_is_refused_before_any_mutation(
    tmp_path: Path, version: str, missing: tuple[str, ...]
) -> None:
    """A real older layout is never migrated, and newer tables are never created in it."""
    from upscale.services.safety_v2 import repository

    path = tmp_path / f"phase{version}.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(repository._SCHEMA)
        for table in missing:
            conn.execute(f"DROP TABLE {table}")
        conn.execute("INSERT INTO safety_meta VALUES ('schema_version', ?)", (version,))
    conn.close()
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    with pytest.raises(SafetySchemaError, match=f"schema version {version}"):
        SafetyRepository(path).db()
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    with sqlite3.connect(path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert not names & set(missing)
    assert DB_SCHEMA_VERSION == 3


def test_safety_tables_without_meta_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE safety_targets (canonical_id TEXT)")
    conn.close()
    with pytest.raises(SafetySchemaError, match="unknown"):
        SafetyRepository(path).db()


# --- append-only ---------------------------------------------------------------------------


def test_snapshot_update_and_delete_are_rejected(tmp_path: Path) -> None:
    svc, _, _ = _setup(tmp_path)
    res = asyncio.run(svc.snapshot(CID))
    conn = svc.repo.db()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE safety_snapshots SET band = 'NO_TRIGGERED_FLAGS' WHERE id = ?",
                     (res.snapshot_id,))  # fmt: skip
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM safety_snapshots WHERE id = ?", (res.snapshot_id,))
    assert svc.repo.snapshot(res.snapshot_id or 0).body_hash == res.body_hash


def test_observations_are_append_only(tmp_path: Path) -> None:
    svc, _, _ = _setup(tmp_path)
    got = asyncio.run(svc.collect(CID))
    conn = svc.repo.db()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE safety_mint_observations SET mint_authority = NULL WHERE id = ?",
                     (got.observation_id,))  # fmt: skip
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM safety_mint_observations WHERE id = ?", (got.observation_id,))
    with pytest.raises(sqlite3.IntegrityError, match="final"):
        conn.execute("UPDATE safety_collections SET status = 'RUNNING', finished_at = NULL")


def test_a_non_mint_observation_cannot_store_authorities(tmp_path: Path) -> None:
    svc, _, _ = _setup(tmp_path)
    coll = svc.repo.start_collection(CID, 0.0)
    conn = svc.repo.db()
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(
            "INSERT INTO safety_mint_observations (canonical_id, collection_id, fetched_at, "
            "provider, outcome, reason, mint_authority) VALUES (?, ?, 1.0, 'x', "
            "'PROVIDER_FAILED', 'boom', ?)",
            (CID, coll, AUTH),
        )


def test_an_observation_needs_a_running_collection_of_the_same_target(tmp_path: Path) -> None:
    svc, _, _ = _setup(tmp_path)
    got = asyncio.run(svc.collect(CID))  # that collection is now DONE
    conn = svc.repo.db()
    with pytest.raises(sqlite3.IntegrityError, match="RUNNING collection"):
        conn.execute(
            "INSERT INTO safety_mint_observations (canonical_id, collection_id, fetched_at, "
            "provider, outcome, reason) VALUES (?, ?, 1e12, 'x', 'NOT_COLLECTED', 'n')",
            (CID, got.collection_id),
        )


# --- collection lifecycle ------------------------------------------------------------------


def test_collection_finishes_done(tmp_path: Path) -> None:
    svc, _, rpc = _setup(tmp_path)
    got = asyncio.run(svc.collect(CID))
    status, _, finished, requests, _ = svc.repo.collection(got.collection_id)
    assert (status, requests, got.outcome) == ("DONE", 1, "MINT")
    assert finished is not None
    assert rpc.calls == ["getAccountInfo"]


def test_collection_exception_finishes_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, _, _ = _setup(tmp_path)

    def boom(*_: Any) -> Any:
        raise RuntimeError("parser bug")

    monkeypatch.setattr(service_module, "classify_mint_account", boom)
    with pytest.raises(RuntimeError, match="parser bug"):
        asyncio.run(svc.collect(CID))
    rows = (
        svc.repo.db()
        .execute("SELECT status, reasons_json, requests FROM safety_collections")
        .fetchall()
    )
    assert len(rows) == 1
    assert rows[0][0] == "ABORTED"
    assert "parser bug" in json.loads(rows[0][1])[0]
    assert rows[0][2] == 1  # the request that was made is still accounted for
    assert svc.repo.counts()["safety_mint_observations"] == 0
    with pytest.raises(SafetyStateError):
        svc.repo.finish_collection(1, "DONE", 1e12, 0, [])


def test_no_provider_is_not_collected(tmp_path: Path) -> None:
    svc, _, rpc = make_service(tmp_path, FakeRpc({MINT: mint_value()}), provider=False)
    svc.add_target(MINT)
    body = asyncio.run(svc.snapshot(CID)).body
    assert rpc.calls == []
    assert body["provenance"]["inputs"][0]["outcome"] == "NOT_COLLECTED"
    assert body["authority"]["mint_authority"]["status"] == "NOT_COLLECTED"
    assert body["assessment"]["coverage"] == "INSUFFICIENT"


# --- anti-lookahead --------------------------------------------------------------------------


def test_fetched_at_equal_to_as_of_is_accepted(tmp_path: Path) -> None:
    svc, clock, _ = _setup(tmp_path)
    got = asyncio.run(svc.collect(CID))
    row = svc.repo.mint_observation(got.observation_id)
    body = svc.build(CID, row.fetched_at)
    assert body["provenance"]["inputs"][0]["observation_id"] == got.observation_id
    assert body["as_of"] == body["provenance"]["inputs"][0]["fetched_at"]


def test_input_fetched_after_as_of_is_never_selected(tmp_path: Path) -> None:
    svc, clock, _ = _setup(tmp_path)
    asyncio.run(svc.collect(CID))
    body = svc.build(CID, ts(clock.now()) - 0.001)
    assert body["provenance"]["inputs"] == []
    assert body["authority"]["mint_authority"]["status"] == "UNAVAILABLE"
    assert body["identity"]["token_mint"]["status"] == "UNVERIFIED"


def test_build_rejects_an_input_fetched_after_as_of(tmp_path: Path) -> None:
    svc, clock, _ = _setup(tmp_path)
    got = asyncio.run(svc.collect(CID))
    row = svc.repo.mint_observation(got.observation_id)
    with pytest.raises(SafetyCausalityError, match="after as_of"):
        build_body(CID, MINT, row.fetched_at - 1e-6, row, code_fingerprints())


def test_newest_observation_at_or_before_as_of_wins(tmp_path: Path) -> None:
    svc, clock, rpc = _setup(tmp_path, mint_value(mint_authority=AUTH))
    asyncio.run(svc.collect(CID))
    first_as_of = ts(clock.now())
    clock.advance(60)
    rpc.accounts[MINT] = mint_value(mint_authority=None)
    asyncio.run(svc.collect(CID))
    assert svc.build(CID, first_as_of)["authority"]["mint_authority"]["value"]["active"] is True
    assert (
        svc.build(CID, ts(clock.now()))["authority"]["mint_authority"]["value"]["active"] is False
    )


def test_forced_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    svc, clock, _ = _setup(tmp_path)
    got = asyncio.run(svc.collect(CID))
    row = svc.repo.mint_observation(got.observation_id)
    other = MINT.replace("AAA", "aaa")  # a different (case-sensitive) identity
    with pytest.raises(SafetyIdentityError, match="belongs to"):
        build_body(f"solana:{other}", other, ts(clock.now()), row, code_fingerprints())
    with pytest.raises(SafetyIdentityError):
        build_body(CID, other, ts(clock.now()), row, code_fingerprints())
    # The case variant is its own target: the original's evidence is never borrowed.
    cid2, added = svc.add_target(other)
    assert added
    body = svc.build(cid2, ts(clock.now()))
    assert body["provenance"]["inputs"] == []
    assert body["identity"]["token_mint"]["status"] == "UNVERIFIED"


# --- deterministic hashing + fingerprints ----------------------------------------------------


def test_body_and_hash_are_deterministic(tmp_path: Path) -> None:
    hashes = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        svc, _, _ = _setup(tmp_path / name, mint_value(mint_authority=AUTH))
        res = asyncio.run(svc.snapshot(CID))
        assert encode_body(res.body)[2] == res.body_hash
        assert encode_body(svc.build(CID, ts(res.as_of)))[2] == res.body_hash
        hashes.append(res.body_hash)
    assert hashes[0] == hashes[1]
    body = svc.repo.snapshot(res.snapshot_id or 0).body
    assert body["schema_version"] == SNAPSHOT_SCHEMA and body["rules_version"] == RULES_VERSION
    fp = body["provenance"]["fingerprints"]
    assert set(fp) == {"rules_version", "safety_v2_source", "solana_chain_source",
                       "solana_dex_source", "dexscreener_source"}  # fmt: skip
    assert fp == code_fingerprints()


def test_body_has_no_wall_clock_beyond_as_of_and_fetched_at(tmp_path: Path) -> None:
    svc, clock, _ = _setup(tmp_path)
    res = asyncio.run(svc.snapshot(CID))
    text = encode_body(res.body)[0]
    assert text.count("2026-") == 2  # as_of + the input's fetched_at, nothing else


def test_rebuild_reproduces_under_identical_fingerprints(tmp_path: Path) -> None:
    svc, clock, rpc = _setup(tmp_path, mint_value(freeze_authority=AUTH))
    res = asyncio.run(svc.snapshot(CID))
    clock.advance(30)
    rpc.accounts[MINT] = None
    asyncio.run(svc.snapshot(CID))  # later evidence must not leak into the earlier rebuild
    rb = svc.rebuild(res.snapshot_id or 0)
    assert rb.status == "REPRODUCED" and rb.rebuilt_hash == res.body_hash


def test_fingerprint_mismatch_refuses_exact_rebuild(tmp_path: Path) -> None:
    svc, _, _ = _setup(tmp_path)
    res = asyncio.run(svc.snapshot(CID))
    for key in ("rules_version", "safety_v2_source", "solana_chain_source"):
        changed = dict(code_fingerprints(), **{key: "different"})
        svc._fingerprints = lambda c=changed: c  # type: ignore[misc]
        rb = svc.rebuild(res.snapshot_id or 0)
        assert rb.status == "FINGERPRINT_MISMATCH"
        assert rb.rebuilt_hash is None
        assert rb.mismatched == {key: (code_fingerprints()[key], "different")}


def test_snapshot_rows_record_rules_and_fingerprints(tmp_path: Path) -> None:
    svc, _, _ = _setup(tmp_path)
    res = asyncio.run(svc.snapshot(CID))
    row = svc.repo.snapshot(res.snapshot_id or 0)
    assert (row.schema_version, row.rules_version) == (SNAPSHOT_SCHEMA, RULES_VERSION)
    assert row.fingerprints == code_fingerprints()
    assert (row.coverage, row.band) == ("COMPLETE", "NO_TRIGGERED_FLAGS")


# --- wording -------------------------------------------------------------------------------


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


@pytest.mark.parametrize("value", [mint_value(), mint_value(mint_authority=AUTH), None,
                                   ("http", 500), mint_value(supply="x")])  # fmt: skip
def test_safe_is_never_an_assessment_value(tmp_path: Path, value: Any) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: value}), max_retries=0)
    svc.add_target(MINT)
    body = asyncio.run(svc.snapshot(CID)).body
    assert not [s for s in _strings(body["assessment"]) if "safe" in s.lower()]
    assert not [s for s in _strings(body) if s.upper() in ("SAFE", "UNSAFE")]
    assert body["assessment"]["band"] in ("CRITICAL_EVIDENCE", "ELEVATED_EVIDENCE",
                                          "NO_TRIGGERED_FLAGS")  # fmt: skip
    assert body["assessment"]["coverage"] == body["coverage"]["coverage"]
