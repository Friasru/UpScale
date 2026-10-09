"""Opportunity Model V1 (O3): durable decision storage, the offline CLI and exact rebuild /
verification. Offline: local archive / Safety fixtures and temporary databases only."""

import hashlib
import io
import json
import os
import re
import shutil
import socket
import sqlite3
import zlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.opportunity_model_fakes import (
    CID,
    Archive,
    at,
    candidate,
    decision_record,
    safety_db,
    scout_record,
    social_record,
)
from tests.test_opportunity_model_decision import _clean_safety
from upscale.services.opportunity_model import recorder as recorder_module
from upscale.services.opportunity_model import repository as repo_module
from upscale.services.opportunity_model.cli import NO_ORDER, main
from upscale.services.opportunity_model.decision import decide
from upscale.services.opportunity_model.fingerprints import (
    DECISION_FILES,
    SERVICES,
    decision_fingerprints,
)
from upscale.services.opportunity_model.loaders import ArchiveReader, SafetyReader
from upscale.services.opportunity_model.models import OpportunityIdentityError
from upscale.services.opportunity_model.recorder import (
    record_from_paths,
    verify_sources,
)
from upscale.services.opportunity_model.repository import (
    OpportunityConflictError,
    OpportunityRepository,
    OpportunityStorageError,
    reason_rows,
    source_rows,
)

T = at()
SECRET = "TESTSECRET-9f1c2e"


def clock_at(t: Any = None) -> Any:
    return lambda: t or T + timedelta(seconds=5)


@pytest.fixture
def stores(tmp_path: Path) -> tuple[str, str]:
    """A clean ENTER token at T: Scout, social, Analyze and a ready Safety V2 snapshot."""
    arch = Archive(tmp_path / "evidence.sqlite3")
    arch.add(scout_record(T, cand=candidate(T, risk_flags=[])))
    arch.add(social_record(T))
    arch.add(decision_record(T))
    arch.close()
    safety_db(tmp_path / "safety.sqlite3", [(T, "4", _clean_safety(T))])
    return str(tmp_path / "evidence.sqlite3"), str(tmp_path / "safety.sqlite3")


@pytest.fixture
def repo(tmp_path: Path) -> Any:
    r = OpportunityRepository(tmp_path / "opp" / "opportunity.sqlite3")
    yield r
    r.close()


def historical(repo: OpportunityRepository, stores: tuple[str, str], as_of: Any = None) -> Any:
    return record_from_paths(repo, CID, "HISTORICAL_REPLAY", *stores, as_of=as_of or T,
                             clock=clock_at())  # fmt: skip


def rows(repo: OpportunityRepository, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    return repo._conn.execute(sql, args).fetchall()


def snapshot(path: str | Path) -> tuple[bytes, int]:
    p = Path(path)
    return p.read_bytes(), p.stat().st_mtime_ns


# --- schema / opening ----------------------------------------------------------------------------


def test_a_fresh_database_gets_schema_1_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "new" / "opportunity.sqlite3"
    r = OpportunityRepository(path)
    assert dict(rows(r, "SELECT key, value FROM opportunity_meta")) == {
        "component": "opportunity_model", "schema_version": "1"}  # fmt: skip
    r.close()
    OpportunityRepository(path).close()
    OpportunityRepository(path, read_only=True).close()


@pytest.mark.parametrize("version", ["0", "2", "x"])
def test_other_schema_versions_are_refused_unchanged(tmp_path: Path, version: str) -> None:
    path = tmp_path / "o.sqlite3"
    OpportunityRepository(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE opportunity_meta SET value = ? WHERE key = 'schema_version'", (version,))
    conn.commit()
    conn.close()
    before = snapshot(path)
    for ro in (False, True):
        with pytest.raises(OpportunityStorageError, match="schema version"):
            OpportunityRepository(path, read_only=ro)
    assert snapshot(path) == before


def test_missing_or_foreign_metadata_is_refused(tmp_path: Path) -> None:
    no_meta = tmp_path / "a.sqlite3"
    c = sqlite3.connect(no_meta)
    c.execute("CREATE TABLE opportunity_runs (id INTEGER PRIMARY KEY)")
    c.commit()
    c.close()
    other = tmp_path / "b.sqlite3"
    OpportunityRepository(other).close()
    c = sqlite3.connect(other)
    c.execute("UPDATE opportunity_meta SET value = 'radar' WHERE key = 'component'")
    c.commit()
    c.close()
    for path, why in ((no_meta, "no metadata"), (other, "not Opportunity")):
        before = snapshot(path)
        with pytest.raises(OpportunityStorageError, match=why):
            OpportunityRepository(path)
        assert snapshot(path) == before


def test_foreign_databases_are_refused_byte_identically(tmp_path: Path) -> None:
    safety = safety_db(tmp_path / "safety.sqlite3", [(T, "4", _clean_safety(T))])
    radar = tmp_path / "radar.sqlite3"
    c = sqlite3.connect(radar)
    c.execute("CREATE TABLE radar_tokens (canonical_id TEXT)")
    c.commit()
    c.close()
    junk = tmp_path / "junk.sqlite3"
    junk.write_bytes(b"not a database at all" * 50)
    for path in (safety, radar, junk):
        before = snapshot(path)
        with pytest.raises(OpportunityStorageError):
            OpportunityRepository(path)
        assert snapshot(path) == before


def test_read_only_never_creates_a_database(tmp_path: Path) -> None:
    with pytest.raises(OpportunityStorageError):
        OpportunityRepository(tmp_path / "missing.sqlite3", read_only=True)
    assert not (tmp_path / "missing.sqlite3").exists()


# --- storing --------------------------------------------------------------------------------------


def test_the_exact_canonical_input_and_decision_are_stored(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    res = historical(repo, stores)
    assert res.created and res.decision.decision == "ENTER"
    ib, db_, ih, dh = rows(repo, "SELECT input_body_zlib, decision_body_zlib, input_hash, "
                                 "decision_hash FROM opportunity_decisions")[0]  # fmt: skip
    input_text, decision_text = zlib.decompress(ib).decode(), zlib.decompress(db_).decode()
    assert hashlib.sha256(input_text.encode()).hexdigest() == ih
    assert hashlib.sha256(decision_text.encode()).hexdigest() == dh
    assert decision_text == res.decision.canonical_json() and dh == res.decision.decision_hash()
    stored = repo.get(res.decision_id)
    assert stored.input.canonical_json() == input_text and stored.input.input_hash() == ih
    assert stored.decision == res.decision
    assert decide(stored.input).canonical_json() == decision_text  # no upstream needed
    for text in (input_text, decision_text):
        assert "NaN" not in text and "Infinity" not in text
        assert json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")) == text


def test_source_refs_and_reasons_are_derived_from_the_bodies(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    res = historical(repo, stores)
    stored = repo.get(res.decision_id)
    refs = repo.source_refs(res.decision_id)
    assert [r["source_name"] for r in refs] == ["scout", "technical", "technical_analyze",
                                                "social", "news", "safety"]  # fmt: skip
    assert refs == source_rows(stored.input)
    safety = refs[-1]
    assert (safety["safety_snapshot_id"], safety["safety_rules_version"]) == (1, "4")
    assert safety["safety_body_hash"] == stored.input.sources.safety.ref.body_hash
    got = repo.reasons(res.decision_id)
    want = reason_rows(stored.decision)
    assert [{k: r[k] for k in w} for r, w in zip(got, want, strict=True)] == want
    kinds = {r["kind"] for r in got}
    assert "POSITIVE" in kinds and "MISSING" in kinds


def test_the_repository_refuses_a_decision_not_made_from_the_input(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    a, s = ArchiveReader(stores[0]), SafetyReader(stores[1])
    from upscale.services.opportunity_model.service import build_input

    inp = build_input(CID, T, "HISTORICAL_REPLAY", a, s)
    other = build_input(CID, T + timedelta(seconds=1), "HISTORICAL_REPLAY", a, s)
    a.close()
    s.close()
    run = repo.start_run(CID, "HISTORICAL_REPLAY", T, T)
    good = decide(inp)
    for bad in (decide(other), good.model_copy(update={"decision": "WATCH"}),
                good.model_copy(update={"quality": "PARTIAL"})):  # fmt: skip
        with pytest.raises(OpportunityStorageError):
            repo.store(run, inp, bad, T, T)
    assert rows(repo, "SELECT COUNT(*) FROM opportunity_decisions") == [(0,)]


# --- SQL honesty / append-only --------------------------------------------------------------------


def _raw_insert(repo: OpportunityRepository, run_id: int, **over: Any) -> None:
    values = {
        "run_id": run_id, "canonical_id": CID, "chain": "solana", "pool_address": None,
        "decision_at": T.timestamp(), "decided_at": T.timestamp(), "origin": "HISTORICAL_REPLAY",
        "decision": "ENTER", "quality": "COMPLETE", "setup_band": "STRONG",
        "technical_band": "CONFIRMING", "social_band": "SUPPORTING", "risk_tier": "CLEAN",
        "rules_version": "1", "input_schema": "opportunity.input.v1",
        "decision_schema": "opportunity.decision.v1", "input_hash": "a" * 64,
        "decision_hash": "b" * 64, "input_body_zlib": b"x", "decision_body_zlib": b"y",
        "input_fingerprints_json": "{}", "decision_fingerprints_json": "{}", "veto_count": 0,
        "blocker_count": 0, "ineligible_count": 0,
    } | over  # fmt: skip
    cols = ", ".join(values)
    repo._conn.execute(f"INSERT INTO opportunity_decisions ({cols}) VALUES "
                       f"({', '.join('?' for _ in values)})", list(values.values()))  # fmt: skip


@pytest.mark.parametrize(
    "over",
    [{"veto_count": 1}, {"blocker_count": 1}, {"ineligible_count": 1},
     {"quality": "INSUFFICIENT"}, {"decision": "SKIP", "setup_band": "STRONG"},
     {"decision": "BUY"}, {"input_hash": "short"}],
)  # fmt: skip
def test_sql_refuses_dishonest_decisions(repo: OpportunityRepository, over: dict) -> None:
    run = repo.start_run(CID, "HISTORICAL_REPLAY", T, T)
    with pytest.raises(sqlite3.IntegrityError):
        _raw_insert(repo, run, **over)
    _raw_insert(repo, run)  # the same row, honest, is accepted (in a RUNNING run)


def test_stored_rows_are_append_only(repo: OpportunityRepository, stores: tuple[str, str]) -> None:
    res = historical(repo, stores)
    for sql in (
        "UPDATE opportunity_decisions SET decision = 'WATCH'",
        "DELETE FROM opportunity_decisions",
        "UPDATE opportunity_source_refs SET status = 'AVAILABLE'",
        "DELETE FROM opportunity_source_refs",
        "UPDATE opportunity_reasons SET message = 'x'",
        "DELETE FROM opportunity_reasons",
        "DELETE FROM opportunity_runs",
        "UPDATE opportunity_runs SET status = 'RUNNING', finished_at = NULL",
        "UPDATE opportunity_runs SET status = 'ABORTED', reason = 'x'",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            repo._conn.execute(sql)
    with pytest.raises(sqlite3.DatabaseError):  # no rows appended to a finished decision
        repo._conn.execute(
            "INSERT INTO opportunity_reasons (decision_id, ordinal, kind, code, message, "
            "evidence_paths_json) VALUES (?, 999, 'RISK', 'X', 'x', '[]')", (res.decision_id,))  # fmt: skip
    with pytest.raises(sqlite3.DatabaseError):  # no decision outside a RUNNING run
        _raw_insert(repo, res.run_id, decision_at=0.0)


# --- runs and atomicity ---------------------------------------------------------------------------


def _runs(repo: OpportunityRepository) -> list[tuple[str, str | None]]:
    return rows(repo, "SELECT status, reason FROM opportunity_runs ORDER BY id")


def test_runs_finish_done_or_aborted_never_running(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    historical(repo, stores)
    with pytest.raises(OpportunityIdentityError):
        record_from_paths(repo, "solana:not-a-mint", "HISTORICAL_REPLAY", *stores, as_of=T,
                          clock=clock_at())  # fmt: skip
    runs = _runs(repo)
    assert runs[0] == ("DONE", None)
    assert runs[1][0] == "ABORTED" and "OpportunityIdentityError" in str(runs[1][1])
    assert not [r for r in runs if r[0] == "RUNNING"]


@pytest.mark.parametrize("stage", ["refs", "reasons"])
def test_a_failure_mid_write_leaves_no_partial_decision(
    repo: OpportunityRepository, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:  # fmt: skip
    name = "_insert_source_refs" if stage == "refs" else "_insert_reasons"
    real = getattr(repo_module, name)

    def half(db: sqlite3.Connection, decision_id: int, items: list[dict[str, Any]]) -> None:
        real(db, decision_id, items[:1])  # some rows really written...
        raise RuntimeError(f"disk full while writing {stage}")  # ...then a failure

    monkeypatch.setattr(repo_module, name, half)
    with pytest.raises(RuntimeError):
        historical(repo, stores)
    for table in ("opportunity_decisions", "opportunity_source_refs", "opportunity_reasons"):
        assert rows(repo, f"SELECT COUNT(*) FROM {table}") == [(0,)]
    ((status, reason),) = _runs(repo)
    assert status == "ABORTED" and "disk full" in str(reason)


# --- idempotency / conflicts --------------------------------------------------------------------


def test_the_same_decision_twice_is_idempotent(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    first, second = historical(repo, stores), historical(repo, stores)
    assert (first.created, second.created) == (True, False)
    assert first.decision_id == second.decision_id
    assert rows(repo, "SELECT COUNT(*) FROM opportunity_decisions") == [(1,)]
    assert rows(repo, "SELECT COUNT(*) FROM opportunity_source_refs") == [(6,)]
    assert [s for s, _ in _runs(repo)] == ["DONE", "DONE"]


def test_same_key_with_a_different_input_is_a_conflict(
    repo: OpportunityRepository, stores: tuple[str, str], tmp_path: Path
) -> None:
    first = historical(repo, stores)
    arch = Archive(Path(stores[0]))
    arch.add(decision_record(T, action="sell"))  # a later same-time Analyze record: new ref
    arch.close()
    original = repo.get(first.decision_id).decision_hash
    with pytest.raises(OpportunityConflictError, match="different input"):
        historical(repo, stores)
    assert repo.get(first.decision_id).decision_hash == original
    assert rows(repo, "SELECT COUNT(*) FROM opportunity_decisions") == [(1,)]
    assert _runs(repo)[-1][0] == "ABORTED"


def test_same_input_with_a_different_decision_is_a_conflict(
    repo: OpportunityRepository, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    historical(repo, stores)

    def changed_engine(inp: Any) -> Any:
        d = decide(inp)
        return d.model_copy(update={"thresholds": {**d.thresholds, "rising_ratio": 1.25}})

    monkeypatch.setattr(recorder_module, "decide", changed_engine)
    monkeypatch.setattr(repo_module, "decide", changed_engine)
    with pytest.raises(OpportunityConflictError, match="different decision"):
        historical(repo, stores)


# --- verification ------------------------------------------------------------------------------


def test_verify_reproduces_without_any_upstream_store(
    tmp_path: Path, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "v.sqlite3"
    r = OpportunityRepository(path)
    res = historical(r, stores)
    r.close()
    os.remove(stores[0])
    os.remove(stores[1])
    before = snapshot(path)

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    ro = OpportunityRepository(path, read_only=True)
    out = ro.verify_decision(res.decision_id)
    ro.close()
    assert out.status == "REPRODUCED" and out.recomputed_hash == out.stored_hash
    assert snapshot(path) == before  # verify writes nothing


def test_a_changed_decision_source_file_is_a_fingerprint_mismatch(
    repo: OpportunityRepository, stores: tuple[str, str], tmp_path: Path
) -> None:
    res = historical(repo, stores)
    copy = tmp_path / "services"
    for f in DECISION_FILES:
        (copy / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SERVICES / f, copy / f)
    assert repo.verify_decision(res.decision_id, lambda: decision_fingerprints(copy)).status == (
        "REPRODUCED")  # fmt: skip
    target = copy / "opportunity_model" / "decision.py"
    target.write_text(target.read_text().replace("rising_ratio", "rising_ratio", 1) + "\n# edit\n")
    out = repo.verify_decision(res.decision_id, lambda: decision_fingerprints(copy))
    assert out.status == "FINGERPRINT_MISMATCH"
    assert out.differing_fingerprints == ("source:opportunity_model/decision.py",)


def test_forced_divergence_and_tampered_bodies(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    res = historical(repo, stores)

    def other(inp: Any) -> Any:
        return decide(inp).model_copy(update={"decision": "WATCH"})

    assert repo.verify_decision(res.decision_id, engine=other).status == "DIVERGED"
    db = repo._conn
    db.execute("DROP TRIGGER opportunity_decisions_no_update")  # simulate disk tampering
    text = zlib.decompress(rows(repo, "SELECT decision_body_zlib FROM opportunity_decisions")[0][0])
    db.execute("UPDATE opportunity_decisions SET decision_body_zlib = ?",
               (zlib.compress(text.replace(b'"ENTER"', b'"WATCH"')),))  # fmt: skip
    assert repo.verify_decision(res.decision_id).status == "CORRUPT"
    db.execute("UPDATE opportunity_decisions SET decision_body_zlib = ?", (zlib.compress(text),))
    assert repo.verify_decision(res.decision_id).status == "REPRODUCED"
    db.execute("UPDATE opportunity_decisions SET input_body_zlib = ?", (b"garbage",))
    assert repo.verify_decision(res.decision_id).status == "CORRUPT"


# --- live / historical -------------------------------------------------------------------------


def test_origin_rules_and_audit_times_outside_the_hash(
    tmp_path: Path, stores: tuple[str, str]
) -> None:
    a = OpportunityRepository(tmp_path / "a.sqlite3")
    b = OpportunityRepository(tmp_path / "b.sqlite3")
    with pytest.raises(ValueError, match="explicit as_of"):
        record_from_paths(a, CID, "HISTORICAL_REPLAY", *stores, as_of=None)
    with pytest.raises(ValueError, match="no as_of"):
        record_from_paths(a, CID, "LIVE_FORWARD", *stores, as_of=T)
    one = record_from_paths(a, CID, "HISTORICAL_REPLAY", *stores, as_of=T,
                            clock=clock_at(T + timedelta(hours=1)))  # fmt: skip
    two = record_from_paths(b, CID, "HISTORICAL_REPLAY", *stores, as_of=T,
                            clock=clock_at(T + timedelta(days=3)))  # fmt: skip
    assert one.decision.decision_hash() == two.decision.decision_hash()
    assert a.get(one.decision_id).decided_at != b.get(two.decision_id).decided_at
    live = record_from_paths(a, CID, "LIVE_FORWARD", *stores, clock=clock_at(T))
    stored = a.get(live.decision_id)
    assert (stored.origin, stored.decision_at) == ("LIVE_FORWARD", T)
    assert rows(a, "SELECT origin, requested_as_of IS NULL FROM opportunity_runs ORDER BY id")[-1] == (
        "LIVE_FORWARD", 1)  # fmt: skip
    a.close()
    b.close()


# --- source verification ------------------------------------------------------------------------


def test_source_verification_is_separate_and_read_only(
    repo: OpportunityRepository, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    res = historical(repo, stores)

    def check() -> str:
        a, s = ArchiveReader(stores[0]), SafetyReader(stores[1])
        try:
            return verify_sources(repo, res.decision_id, a, s).status
        finally:
            a.close()
            s.close()

    before = [snapshot(p) for p in stores]
    assert check() == "MATCH"
    assert [snapshot(p) for p in stores] == before
    arch = Archive(Path(stores[0]))
    arch.add(decision_record(T, action="sell"))
    arch.close()
    assert check() == "SOURCE_CHANGED"
    os.remove(stores[1])
    assert check() == "SOURCE_MISSING"
    monkeypatch.setattr(recorder_module, "input_fingerprints", lambda: {"source:x": "changed"})
    assert check() == "INCOMPATIBLE"
    assert repo.verify_decision(res.decision_id).status == "REPRODUCED"  # decision unaffected


# --- CLI -------------------------------------------------------------------------------------------


def cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out, clock=clock_at())
    return code, out.getvalue()


def test_cli_end_to_end(tmp_path: Path, stores: tuple[str, str],
                        monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    for key in ("HELIUS_API_KEY", "ANTHROPIC_API_KEY", "UPSCALE_SOLANA_RPC_URL"):
        monkeypatch.setenv(key, SECRET)
    db = str(tmp_path / "cli.sqlite3")
    common = ["--evidence-db", stores[0], "--safety-db", stores[1]]
    code, out = cli("--db", db, "decide", "--asset", CID, "--historical", "--as-of",
                    T.isoformat(), *common)  # fmt: skip
    assert code == 0 and "decision:     ENTER" in out and NO_ORDER in out
    did = re.search(r"decision id:\s+(\d+)", out)
    assert did is not None
    outputs = [out]
    for argv in (["show", "--id", did[1]], ["show", "--asset", CID], ["explain", "--id", did[1]],
                 ["verify", "--id", did[1]], ["verify", "--id", did[1], "--sources", *common],
                 ["status"]):  # fmt: skip
        code, text = cli("--db", db, *argv)
        assert code == 0, (argv, text)
        outputs.append(text)
    show, _, explain, verify, sources, status = outputs[1:]
    assert "safety ready" in show and "safety_snapshots:1" in show
    for title in ("Decision", "Evidence quality", "Positive reasons", "Risks", "Vetoes",
                  "Blockers", "Missing evidence", "Ineligible reasons", "Upgrade path"):  # fmt: skip
        assert f"\n{title}\n" in "\n" + explain
    assert "REPRODUCED" in verify and "sources: MATCH" in sources
    assert "decisions: 1" in status and '"ENTER": 1' in status
    joined = "".join(outputs)
    assert SECRET not in joined
    con = sqlite3.connect(db)
    blobs = con.execute("SELECT input_body_zlib, decision_body_zlib FROM opportunity_decisions")
    for ib, dbody in blobs:
        assert SECRET.encode() not in zlib.decompress(ib) + zlib.decompress(dbody)
    con.close()
    assert SECRET.encode() not in Path(db).read_bytes()


def test_cli_rejects_bad_live_historical_combinations(tmp_path: Path) -> None:
    db = str(tmp_path / "c.sqlite3")
    assert cli("--db", db, "decide", "--asset", CID, "--historical")[0] == 2
    assert cli("--db", db, "decide", "--asset", CID, "--as-of", T.isoformat())[0] == 2
    assert not Path(db).exists()
    code, out = cli("--db", db, "status")
    assert code == 2 and "no Opportunity database" in out and not Path(db).exists()


def test_cli_live_decide_and_a_skip_still_says_no_order(
    tmp_path: Path, stores: tuple[str, str]
) -> None:
    db = str(tmp_path / "l.sqlite3")
    out = io.StringIO()
    code = main(["--db", db, "decide", "--asset", CID, "--evidence-db", stores[0],
                 "--safety-db", str(tmp_path / "none.sqlite3")], out=out, clock=clock_at(T))  # fmt: skip
    assert code == 0 and "LIVE_FORWARD" in out.getvalue()
    assert "decision:     SKIP" in out.getvalue() and NO_ORDER in out.getvalue()


# --- upstream read-only, isolation ------------------------------------------------------------------


def test_upstream_stores_are_only_read(
    tmp_path: Path, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = [snapshot(p) for p in stores]
    opened: list[str] = []
    real = sqlite3.connect

    def spy(database: Any, *a: Any, **k: Any) -> sqlite3.Connection:
        opened.append(str(database))
        return real(database, *a, **k)

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network")

    monkeypatch.setattr(sqlite3, "connect", spy)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    r = OpportunityRepository(tmp_path / "u.sqlite3")
    res = historical(r, stores)
    a, s = ArchiveReader(stores[0]), SafetyReader(stores[1])
    verify_sources(r, res.decision_id, a, s)
    a.close()
    s.close()
    r.close()
    assert [snapshot(p) for p in stores] == before
    upstream = [o for o in opened if "u.sqlite3" not in o]
    assert upstream and all(o.startswith("file:") and o.endswith("?mode=ro") for o in upstream)
    scout_db = os.environ["UPSCALE_SCOUT_DB"]
    assert not [o for o in opened if "radar" in o.lower() or scout_db in o]


def test_o3_modules_import_no_provider_execution_or_protected_code() -> None:
    pkg = SERVICES / "opportunity_model"
    for name in ("repository.py", "recorder.py", "cli.py", "fingerprints.py", "__main__.py"):
        imports = [ln for ln in (pkg / name).read_text().splitlines()
                   if ln.startswith(("import ", "from "))]  # fmt: skip
        tokens = {t for ln in imports for t in re.split(r"[ .,()]+", ln)}
        for word in ("httpx", "anthropic", "radar", "shadow", "scout", "safety_v2", "execution",
                     "socket", "requests", "opportunity"):  # fmt: skip
            assert word not in tokens, (name, word)


# --- semantic fingerprints (pre-commit hardening) ------------------------------------------------


def test_storage_and_runtime_settings_are_not_fingerprinted(
    repo: OpportunityRepository, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from upscale.services.opportunity_model import config as C
    from upscale.services.opportunity_model.fingerprints import input_fingerprints

    res = historical(repo, stores)
    before = (input_fingerprints(), decision_fingerprints())
    assert not [k for k in (*before[0], *before[1])
                if "config.py" in k or "db" in k.lower() or "component" in k]  # fmt: skip
    monkeypatch.setattr(C, "OPPORTUNITY_DB_SCHEMA_VERSION", 2)
    monkeypatch.setattr(C, "OPPORTUNITY_COMPONENT", "renamed")
    monkeypatch.setenv("UPSCALE_OPPORTUNITY_DB", "/elsewhere/opportunity.sqlite3")
    monkeypatch.setenv("UPSCALE_EVIDENCE_DB", "/elsewhere/evidence.sqlite3")
    assert (input_fingerprints(), decision_fingerprints()) == before
    assert repo.verify_decision(res.decision_id).status == "REPRODUCED"


def test_a_real_decision_threshold_change_is_a_named_mismatch(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    from upscale.services.opportunity_model.config import DecisionConfig, Freshness

    res = historical(repo, stores)
    out = repo.verify_decision(
        res.decision_id, lambda: decision_fingerprints(cfg=DecisionConfig(rising_ratio=1.25))
    )
    assert out.status == "FINGERPRINT_MISMATCH"
    assert out.differing_fingerprints == ("config:threshold.rising_ratio",)
    stricter = repo.verify_decision(
        res.decision_id, lambda: decision_fingerprints(freshness=Freshness(scout_s=600.0))
    )
    assert stricter.differing_fingerprints == ("config:freshness.scout_s",)


def test_a_real_input_freshness_change_is_a_named_source_incompatibility(
    repo: OpportunityRepository, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from upscale.services.opportunity_model.config import Freshness
    from upscale.services.opportunity_model.fingerprints import input_fingerprints

    res = historical(repo, stores)
    changed = input_fingerprints(freshness=Freshness(social_s=1800.0))
    assert set(changed) == set(input_fingerprints())
    monkeypatch.setattr(recorder_module, "input_fingerprints", lambda: changed)
    a, s = ArchiveReader(stores[0]), SafetyReader(stores[1])
    out = verify_sources(repo, res.decision_id, a, s)
    a.close()
    s.close()
    assert out.status == "INCOMPATIBLE" and out.detail.endswith("config:freshness.social_s")


# --- interrupts ----------------------------------------------------------------------------------


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["build", "store"])
def test_an_interrupt_aborts_the_run_and_is_re_raised(
    repo: OpportunityRepository, stores: tuple[str, str], monkeypatch: pytest.MonkeyPatch,
    signal: type[BaseException], stage: str,
) -> None:  # fmt: skip
    def interrupt(*a: Any, **k: Any) -> Any:
        raise signal()

    if stage == "build":
        monkeypatch.setattr(recorder_module, "build_input", interrupt)
    else:  # mid-transaction, after the decision row is written
        monkeypatch.setattr(repo_module, "_insert_reasons", interrupt)
    with pytest.raises(signal):
        historical(repo, stores)
    for table in ("opportunity_decisions", "opportunity_source_refs", "opportunity_reasons"):
        assert rows(repo, f"SELECT COUNT(*) FROM {table}") == [(0,)]
    ((status, reason),) = _runs(repo)
    assert status == "ABORTED" and reason.startswith(signal.__name__)


# --- source verification causality ------------------------------------------------------------


def test_source_verification_reports_causality_errors_independently(
    repo: OpportunityRepository, stores: tuple[str, str]
) -> None:
    live = record_from_paths(repo, CID, "LIVE_FORWARD", *stores, clock=clock_at(T))
    # Evidence observed at the decision time but written to the archive only afterwards:
    # it wasn't readable at a LIVE_FORWARD decision made at T.
    arch = Archive(Path(stores[0]))
    arch.add(scout_record(T, cand=candidate(T, risk_flags=[], stage="EARLY")),
             archived=T + timedelta(seconds=10))  # fmt: skip
    arch.close()
    a, s = ArchiveReader(stores[0]), SafetyReader(stores[1])
    out = verify_sources(repo, live.decision_id, a, s)
    a.close()
    s.close()
    assert out.status == "CAUSALITY_ERROR" and "archived" in out.detail
    assert repo.verify_decision(live.decision_id).status == "REPRODUCED"
