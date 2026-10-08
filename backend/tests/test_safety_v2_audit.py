"""Safety V2 Phase 1 pre-commit audit: the full mint-outcome -> feature -> rule matrix, the
identity state machine, fingerprint completeness, coverage with NOT_SUPPORTED rules, neutral
wording, import side effects and storage / rebuild guarantees. Offline."""

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

from tests.safety_v2_fakes import (
    AUTH,
    MINT,
    OTHER_PROGRAM,
    FakeRpc,
    make_service,
    mint_value,
)
from upscale.services.market_data import InvalidRequestError
from upscale.services.safety_v2 import features
from upscale.services.safety_v2.features import build_body, code_fingerprints
from upscale.services.safety_v2.models import RuleResult, SafetySchemaError, SafetyStateError
from upscale.services.safety_v2.repository import MintRow, SafetyRepository, encode_body
from upscale.services.safety_v2.rules import assess
from upscale.services.solana_chain import TOKEN_2022_PROGRAM, TOKEN_PROGRAM

BACKEND = Path(__file__).resolve().parents[1]
CID = f"solana:{MINT}"
RULES = ("NOT_A_TOKEN_MINT", "UNEXPECTED_TOKEN_PROGRAM", "MALFORMED_MINT_ACCOUNT",
         "IDENTITY_MISMATCH", "MINT_AUTHORITY_ACTIVE", "FREEZE_AUTHORITY_ACTIVE")  # fmt: skip
T, N, U = "TRIGGERED", "NOT_TRIGGERED", "UNDETERMINED"


def _row(outcome: str, owner: str | None = None, **kw: Any) -> MintRow:
    base: dict[str, Any] = dict(
        id=1, canonical_id=CID, collection_id=1, fetched_at=0.0, provider="fake",
        outcome=outcome, reason=None if outcome == "MINT" else "why", raw_hash=None,
        context_slot=None, program_owner=owner, token_program=None, decimals=None,
        supply_raw=None, mint_authority=None, freeze_authority=None, extensions=None,
    )  # fmt: skip
    if outcome == "MINT":
        base.update(program_owner=owner or TOKEN_PROGRAM, token_program="spl_token", decimals=6,
                    supply_raw="1", extensions=[])  # fmt: skip
    base.update(kw)
    return MintRow(**base)


# (case, row) -> identity, authority status, coverage, band, rule outcomes in RULES order.
MATRIX: list[tuple[str, MintRow | None, str, str, str, str, tuple[str, ...]]] = [
    ("MINT null authorities", _row("MINT"), "VERIFIED", "AVAILABLE", "COMPLETE",
     "NO_TRIGGERED_FLAGS", (N, N, N, N, N, N)),
    ("MINT active authorities", _row("MINT", mint_authority=AUTH, freeze_authority=AUTH),
     "VERIFIED", "AVAILABLE", "COMPLETE", "ELEVATED_EVIDENCE", (N, N, N, N, T, T)),
    ("NOT_A_MINT foreign owner", _row("NOT_A_MINT", OTHER_PROGRAM), "MISMATCH", "UNAVAILABLE",
     "INSUFFICIENT", "CRITICAL_EVIDENCE", (T, T, N, T, U, U)),
    ("NOT_A_MINT token account", _row("NOT_A_MINT", TOKEN_PROGRAM), "MISMATCH", "UNAVAILABLE",
     "INSUFFICIENT", "CRITICAL_EVIDENCE", (T, N, N, T, U, U)),
    ("ACCOUNT_MISSING", _row("ACCOUNT_MISSING"), "MISMATCH", "UNAVAILABLE", "INSUFFICIENT",
     "CRITICAL_EVIDENCE", (T, U, N, T, U, U)),
    ("MALFORMED token owner", _row("MALFORMED", TOKEN_PROGRAM), "UNVERIFIED", "UNKNOWN",
     "INSUFFICIENT", "CRITICAL_EVIDENCE", (U, N, T, U, U, U)),
    ("MALFORMED no owner", _row("MALFORMED"), "UNVERIFIED", "UNKNOWN", "INSUFFICIENT",
     "CRITICAL_EVIDENCE", (U, U, T, U, U, U)),
    ("PROVIDER_FAILED", _row("PROVIDER_FAILED"), "UNVERIFIED", "PROVIDER_UNAVAILABLE",
     "INSUFFICIENT", "NO_TRIGGERED_FLAGS", (U, U, U, U, U, U)),
    ("NOT_COLLECTED", _row("NOT_COLLECTED"), "UNVERIFIED", "NOT_COLLECTED", "INSUFFICIENT",
     "NO_TRIGGERED_FLAGS", (U, U, U, U, U, U)),
    ("no observation", None, "UNVERIFIED", "UNAVAILABLE", "INSUFFICIENT",
     "NO_TRIGGERED_FLAGS", (U, U, U, U, U, U)),
]  # fmt: skip


@pytest.mark.parametrize("case, row, identity, auth, coverage, band, outcomes", MATRIX,
                         ids=[m[0] for m in MATRIX])  # fmt: skip
def test_mint_outcome_feature_rule_matrix(
    case: str, row: MintRow | None, identity: str, auth: str, coverage: str, band: str,
    outcomes: tuple[str, ...],
) -> None:  # fmt: skip
    body = build_body(CID, MINT, 1.0, row, code_fingerprints())
    assert body["identity"]["token_mint"]["status"] == identity
    assert {f["status"] for f in body["authority"].values()} == {auth}
    assert body["assessment"]["coverage"] == body["coverage"]["coverage"] == coverage
    assert body["assessment"]["band"] == band
    got = {f["id"]: f["outcome"] for f in body["flags"]}
    assert tuple(got[r] for r in RULES) == outcomes
    if auth != "AVAILABLE":  # nothing but a parsed mint ever proves revocation
        assert all(f["value"] is None for f in body["authority"].values())
    undetermined = {u["id"] for u in body["undetermined"]}
    assert undetermined == {r for r, o in zip(RULES, outcomes, strict=True) if o == U}


def test_not_collected_stays_distinct_from_provider_failed(tmp_path: Path) -> None:
    svc, _, rpc = make_service(tmp_path, FakeRpc({MINT: mint_value()}), daily_request_budget=0)
    svc.add_target(MINT)
    body = asyncio.run(svc.snapshot(CID)).body
    assert rpc.calls == []
    assert body["provenance"]["inputs"][0]["outcome"] == "NOT_COLLECTED"
    assert body["coverage"]["components"]["mint_account"] == "NOT_COLLECTED"
    assert {f["status"] for f in body["authority"].values()} == {"NOT_COLLECTED"}


# --- identity state machine -----------------------------------------------------------------


@pytest.mark.parametrize(
    "owner, program",
    [(None, "spl_token"), (OTHER_PROGRAM, "spl_token"), (TOKEN_PROGRAM, "token_2022"),
     (TOKEN_2022_PROGRAM, "spl_token")],
)  # fmt: skip
def test_a_mint_row_not_owned_by_its_token_program_is_rejected(
    tmp_path: Path, owner: str | None, program: str
) -> None:
    repo = SafetyRepository(tmp_path / "s.sqlite3")
    repo.add_target(CID, MINT, "t", 0.0)
    coll = repo.start_collection(CID, 0.0)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        repo.db().execute(
            "INSERT INTO safety_mint_observations (canonical_id, collection_id, fetched_at, "
            "provider, outcome, raw_hash, program_owner, token_program, decimals, supply_raw, "
            "extensions_json) VALUES (?, ?, 1.0, 'x', 'MINT', 'h', ?, ?, 6, '1', '[]')",
            (CID, coll, owner, program),
        )
    # Even outside the database, such a row is never VERIFIED.
    row = _row("MINT", program_owner=owner, token_program=program)
    with pytest.raises(SafetyStateError, match="inconsistent"):
        build_body(CID, MINT, 1.0, row, code_fingerprints())


def test_a_missing_account_cannot_have_an_owner(tmp_path: Path) -> None:
    repo = SafetyRepository(tmp_path / "s.sqlite3")
    repo.add_target(CID, MINT, "t", 0.0)
    coll = repo.start_collection(CID, 0.0)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        repo.db().execute(
            "INSERT INTO safety_mint_observations (canonical_id, collection_id, fetched_at, "
            "provider, outcome, reason, raw_hash, program_owner) VALUES (?, ?, 1.0, 'x', "
            "'ACCOUNT_MISSING', 'gone', 'h', ?)",
            (CID, coll, TOKEN_PROGRAM),
        )


@pytest.mark.parametrize("raw", ["0x" + "a" * 40, f"base:{MINT}", "MintAAA0OIl", f"solana: {MINT}"])
def test_invalid_identities_are_rejected_before_any_provider_call(tmp_path: Path, raw: str) -> None:
    svc, _, rpc = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
    with pytest.raises(InvalidRequestError):
        svc.add_target(raw)
    with pytest.raises(InvalidRequestError):  # never a target: nothing to collect
        asyncio.run(svc.collect(f"solana:{raw.strip()}"))
    assert rpc.calls == []
    assert svc.repo.counts()["safety_collections"] == 0


def test_valid_base58_alone_is_never_verified(tmp_path: Path) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
    svc.add_target(MINT)
    body = svc.build(CID, 4e9)  # a valid target, no evidence yet
    assert body["identity"]["token_mint"]["status"] == "UNVERIFIED"
    assert body["identity"]["pool_match"]["status"] == "UNVERIFIED"
    assert body["identity"]["pool_match"]["evidence"]["status"] == "NOT_COLLECTED"


# --- fingerprints ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["solana_chain", "safety_v2"])
def test_a_changed_body_affecting_source_refuses_exact_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
    svc.add_target(MINT)
    res = asyncio.run(svc.snapshot(CID))
    copy = tmp_path / "src"
    copy.mkdir()
    if target == "solana_chain":
        changed = copy / "solana_chain.py"
        changed.write_bytes(features.SOLANA_CHAIN.read_bytes() + b"\n# changed\n")
        monkeypatch.setattr(features, "SOLANA_CHAIN", changed)
        key = "solana_chain_source"
    else:
        for name in features.FINGERPRINTED:
            (copy / name).write_bytes((features._PACKAGE / name).read_bytes())
        (copy / "rules.py").write_bytes((copy / "rules.py").read_bytes() + b"\n# changed\n")
        monkeypatch.setattr(features, "_PACKAGE", copy)
        key = "safety_v2_source"
    features.code_fingerprints.cache_clear()
    svc._fingerprints = features.code_fingerprints  # type: ignore[method-assign]
    try:
        rb = svc.rebuild(res.snapshot_id or 0)
    finally:
        monkeypatch.undo()
        features.code_fingerprints.cache_clear()
    assert rb.status == "FINGERPRINT_MISMATCH" and rb.rebuilt_hash is None
    assert set(rb.mismatched) == {key}


# --- coverage / NOT_SUPPORTED / wording -----------------------------------------------------


@pytest.mark.parametrize("value", [
    mint_value(),
    mint_value(program=TOKEN_2022_PROGRAM,
               extensions=[{"extension": "permanentDelegate", "state": {"delegate": AUTH}}]),
])  # fmt: skip
def test_not_supported_rules_never_hold_complete_evidence_at_partial(
    tmp_path: Path, value: Any
) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: value}))
    svc.add_target(MINT)
    body = asyncio.run(svc.snapshot(CID)).body
    assert "TOKEN_2022_EXTENSION_RISK" in {n["id"] for n in body["coverage"]["not_supported"]}
    assert body["assessment"]["coverage"] == "COMPLETE"
    assert body["coverage"]["decision_rules"] == sorted(RULES)


def test_no_triggered_flags_with_incomplete_coverage_is_not_a_positive_conclusion() -> None:
    undetermined = RuleResult(id="X", outcome="UNDETERMINED", severity="high",
                              decision_bearing=True, evidence=("a",), reason="r", needs="n")  # fmt: skip
    determined = RuleResult(id="Y", outcome="NOT_TRIGGERED", severity="high",
                            decision_bearing=True, evidence=("a",), reason="r")  # fmt: skip
    partial, coverage, _ = assess([undetermined, determined], "VERIFIED", "ok")
    assert (partial["band"], coverage) == ("NO_TRIGGERED_FLAGS", "PARTIAL")
    insufficient, coverage, _ = assess([determined], "UNVERIFIED", "unread")
    assert (insufficient["band"], coverage) == ("NO_TRIGGERED_FLAGS", "INSUFFICIENT")
    for a in (partial, insufficient):
        text = json.dumps(a).upper()
        assert "SAFE" not in text and "NO_FLAGS_RAISED" not in text


def test_banned_assessment_words_never_appear_in_any_body() -> None:
    for _, row, *_ in MATRIX:
        text = encode_body(build_body(CID, MINT, 1.0, row, code_fingerprints()))[0]
        for word in ('"SAFE"', '"UNSAFE"', "NO_FLAGS_RAISED"):
            assert word not in text


# --- import side effects --------------------------------------------------------------------

_PROBE = r"""
import gc, json, os, socket, sys, threading
connects = []
orig = socket.socket.connect
def blocked(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        connects.append(str(address)); raise RuntimeError("network blocked")
    return orig(self, address)
socket.socket.connect = blocked
import upscale.services  # pre-existing package side effects happen here (baseline)
from upscale.services.evidence_archive import hooks
received = []
class Sink:
    def submit(self, kind, obj, component, extra): received.append(kind)
hooks.install(Sink())
threads = set(threading.enumerate())
import upscale.services.safety_v2, upscale.services.safety_v2.service, upscale.services.safety_v2.cli
from upscale.services.safety_v2.provider import SafetyRpcProvider
print(json.dumps({
    "connects": connects,
    "new_threads": [t.name for t in set(threading.enumerate()) - threads],
    "providers": sum(isinstance(o, SafetyRpcProvider) for o in gc.get_objects()),
    "evidence": received,
    "safety_db_exists": os.path.exists(os.environ["UPSCALE_SAFETY_V2_DB"]),
}))
"""


def test_importing_safety_v2_has_no_side_effects(tmp_path: Path) -> None:
    others = {}
    for name in ("radar", "scout"):
        p = tmp_path / f"{name}.sqlite3"
        with sqlite3.connect(p) as conn:
            conn.execute(f"CREATE TABLE {name}_t (x)")
        conn.close()
        others[p] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    env = {k: v for k, v in os.environ.items()
           if k not in ("UPSCALE_HELIUS_API_KEY", "UPSCALE_SOLANA_RPC_URL")}  # fmt: skip
    env |= {"UPSCALE_SAFETY_V2_DB": str(tmp_path / "safety.sqlite3"),
            "UPSCALE_RADAR_DB": str(tmp_path / "radar.sqlite3"),
            "UPSCALE_SCOUT_DB": str(tmp_path / "scout.sqlite3"),
            "UPSCALE_EVIDENCE_DB": str(tmp_path / "evidence.sqlite3")}  # fmt: skip
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=BACKEND, env=env,
                         capture_output=True, text=True, timeout=120, check=True)  # fmt: skip
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {"connects": [], "new_threads": [], "providers": 0, "evidence": [],
                   "safety_db_exists": False}  # fmt: skip
    for p, (digest, mtime) in others.items():
        assert (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) == (digest, mtime)


# --- storage / rebuild ----------------------------------------------------------------------


def test_body_hash_is_over_exactly_the_stored_canonical_body(tmp_path: Path) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value(freeze_authority=AUTH)}))
    svc.add_target(MINT)
    res = asyncio.run(svc.snapshot(CID))
    blob, digest = (
        svc.repo.db()
        .execute(
            "SELECT body_zlib, body_hash FROM safety_snapshots WHERE id = ?", (res.snapshot_id,)
        )
        .fetchone()
    )
    text = zlib.decompress(blob)
    assert hashlib.sha256(text).hexdigest() == digest == res.body_hash
    assert text.decode() == encode_body(res.body)[0]
    assert json.loads(text) == res.body


def test_rebuild_writes_nothing(tmp_path: Path) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
    svc.add_target(MINT)
    res = asyncio.run(svc.snapshot(CID))
    counts, changes = svc.repo.counts(), svc.repo.db().total_changes
    assert svc.rebuild(res.snapshot_id or 0).status == "REPRODUCED"
    assert svc.repo.counts() == counts and svc.repo.db().total_changes == changes


def test_a_failed_snapshot_save_leaves_no_row(tmp_path: Path) -> None:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
    svc.add_target(MINT)
    asyncio.run(svc.snapshot(CID))
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):  # same as_of: one atomic insert
        asyncio.run(svc.snapshot(CID, fetch=False))
    assert svc.repo.counts()["safety_snapshots"] == 1


@pytest.mark.parametrize("kind", ["foreign", "old_version"])
def test_an_incompatible_database_is_refused_before_any_mutation(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "x.sqlite3"
    with sqlite3.connect(path) as conn:
        if kind == "foreign":
            conn.execute("CREATE TABLE radar_targets (canonical_id TEXT)")
        else:
            conn.execute("CREATE TABLE safety_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO safety_meta VALUES ('schema_version', '0')")
    conn.close()
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    with pytest.raises(SafetySchemaError):
        SafetyRepository(path).db()
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before
    assert not (tmp_path / "x.sqlite3-journal").exists()
