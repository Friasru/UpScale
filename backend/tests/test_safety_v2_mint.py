"""Safety V2 Phase 1: mint-account classification, identity, authority fields and rules.
Offline (MockTransport fakes only)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.safety_v2_fakes import (
    AUTH,
    FREEZER,
    MINT,
    OTHER_PROGRAM,
    FakeRpc,
    make_service,
    mint_value,
)
from upscale.services.safety_v2.features import authority_fields, token_identity
from upscale.services.safety_v2.models import MINT_OUTCOMES
from upscale.services.safety_v2.repository import MintRow
from upscale.services.safety_v2.rules import evaluate
from upscale.services.solana_chain import TOKEN_2022_PROGRAM, TOKEN_PROGRAM


def run(tmp_path: Path, value: Any, **kw: Any) -> dict[str, Any]:
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: value}), **kw)
    cid, _ = svc.add_target(MINT)
    return asyncio.run(svc.snapshot(cid)).body


def flags(body: dict[str, Any]) -> dict[str, str]:
    return {f["id"]: f["outcome"] for f in body["flags"]}


# --- authorities --------------------------------------------------------------------------


def test_active_mint_authority_triggers(tmp_path: Path) -> None:
    body = run(tmp_path, mint_value(mint_authority=AUTH))
    assert body["authority"]["mint_authority"] == {
        "status": "AVAILABLE", "value": {"active": True, "address": AUTH},
        "lower_bound": False, "reason": None,
    }  # fmt: skip
    assert flags(body)["MINT_AUTHORITY_ACTIVE"] == "TRIGGERED"
    assert body["assessment"]["band"] == "ELEVATED_EVIDENCE"
    assert body["assessment"]["triggered"]["high"] == 1


def test_revoked_mint_authority_is_not_triggered(tmp_path: Path) -> None:
    body = run(tmp_path, mint_value(mint_authority=None))
    assert body["authority"]["mint_authority"]["value"] == {"active": False, "address": None}
    assert flags(body)["MINT_AUTHORITY_ACTIVE"] == "NOT_TRIGGERED"


def test_active_freeze_authority_triggers(tmp_path: Path) -> None:
    body = run(tmp_path, mint_value(freeze_authority=FREEZER))
    assert body["authority"]["freeze_authority"]["value"] == {"active": True, "address": FREEZER}
    assert flags(body)["FREEZE_AUTHORITY_ACTIVE"] == "TRIGGERED"


def test_revoked_freeze_authority_is_not_triggered(tmp_path: Path) -> None:
    body = run(tmp_path, mint_value(freeze_authority=None))
    assert flags(body)["FREEZE_AUTHORITY_ACTIVE"] == "NOT_TRIGGERED"


def test_unavailable_mint_read_is_undetermined_never_revoked(tmp_path: Path) -> None:
    body = run(tmp_path, ("http", 503), max_retries=0)
    for key in ("mint_authority", "freeze_authority"):
        field = body["authority"][key]
        assert field["status"] == "PROVIDER_UNAVAILABLE"
        assert field["value"] is None
        assert "revoked" not in (field["reason"] or "")
    f = flags(body)
    assert f["MINT_AUTHORITY_ACTIVE"] == f["FREEZE_AUTHORITY_ACTIVE"] == "UNDETERMINED"
    assert body["assessment"]["band"] == "NO_TRIGGERED_FLAGS"
    assert body["assessment"]["coverage"] == "INSUFFICIENT"
    undetermined = {u["id"] for u in body["undetermined"]}
    assert {"MINT_AUTHORITY_ACTIVE", "FREEZE_AUTHORITY_ACTIVE"} <= undetermined


# --- token programs -----------------------------------------------------------------------


def test_spl_token_mint_is_verified_and_complete(tmp_path: Path) -> None:
    body = run(tmp_path, mint_value())
    a = body["authority"]
    assert a["token_program"]["value"] == {"program": "spl_token", "program_id": TOKEN_PROGRAM}
    assert a["supply_raw"]["value"] == "1000000000"
    assert a["decimals"]["value"] == 6
    assert a["extensions"] == {"status": "AVAILABLE", "value": [], "lower_bound": False,
                               "reason": None}  # fmt: skip
    assert body["identity"]["token_mint"]["status"] == "VERIFIED"
    assert set(flags(body).values()) == {"NOT_TRIGGERED"}
    assert body["assessment"]["band"] == "NO_TRIGGERED_FLAGS"
    assert body["assessment"]["coverage"] == body["coverage"]["coverage"] == "COMPLETE"
    assert body["undetermined"] == []


def test_token_2022_mint_reports_extensions_as_evidence(tmp_path: Path) -> None:
    ext = [{"extension": "transferFeeConfig", "state": {"withheldAmount": 0}},
           {"extension": "metadataPointer", "state": {"authority": None}}]  # fmt: skip
    body = run(tmp_path, mint_value(program=TOKEN_2022_PROGRAM, extensions=ext))
    a = body["authority"]
    assert a["token_program"]["value"]["program"] == "token_2022"
    assert a["extensions"]["value"] == ext
    assert body["identity"]["token_mint"]["status"] == "VERIFIED"
    # Extension risk is deferred (NOT_SUPPORTED), so it can't hold coverage at PARTIAL.
    assert body["assessment"]["coverage"] == "COMPLETE"
    deferred = {n["id"] for n in body["coverage"]["not_supported"]}
    assert "TOKEN_2022_EXTENSION_RISK" in deferred
    assert "TOKEN_2022_EXTENSION_RISK" not in flags(body)


def test_unexpected_program_owner(tmp_path: Path) -> None:
    body = run(tmp_path, mint_value(program=OTHER_PROGRAM))
    f = flags(body)
    assert f["UNEXPECTED_TOKEN_PROGRAM"] == "TRIGGERED"
    assert f["NOT_A_TOKEN_MINT"] == "TRIGGERED"
    assert f["IDENTITY_MISMATCH"] == "TRIGGERED"
    assert f["MALFORMED_MINT_ACCOUNT"] == "NOT_TRIGGERED"
    assert body["identity"]["token_mint"]["status"] == "MISMATCH"
    assert body["authority"]["mint_authority"]["status"] == "UNAVAILABLE"
    assert body["assessment"]["band"] == "CRITICAL_EVIDENCE"
    assert body["assessment"]["coverage"] == "INSUFFICIENT"


def test_account_missing(tmp_path: Path) -> None:
    body = run(tmp_path, None)
    f = flags(body)
    assert f["NOT_A_TOKEN_MINT"] == "TRIGGERED"
    assert f["IDENTITY_MISMATCH"] == "TRIGGERED"
    assert f["UNEXPECTED_TOKEN_PROGRAM"] == "UNDETERMINED"
    assert body["provenance"]["inputs"][0]["outcome"] == "ACCOUNT_MISSING"
    assert body["provenance"]["inputs"][0]["raw_hash"]
    assert body["authority"]["freeze_authority"]["status"] == "UNAVAILABLE"


def test_non_mint_token_account(tmp_path: Path) -> None:
    body = run(tmp_path, mint_value(kind="account"))
    f = flags(body)
    assert f["NOT_A_TOKEN_MINT"] == "TRIGGERED"
    assert f["UNEXPECTED_TOKEN_PROGRAM"] == "NOT_TRIGGERED"
    assert f["IDENTITY_MISMATCH"] == "TRIGGERED"
    assert "'account' account" in body["identity"]["token_mint"]["reason"]


@pytest.mark.parametrize(
    "value, why",
    [
        (mint_value(supply="12x"), "supply"),
        (mint_value(supply=-5), "supply"),
        (mint_value(mint_authority=42), "authority"),
        (mint_value(freeze_authority="not-an-address!"), "authority"),
        (mint_value(initialized=False), "initialized"),
        (mint_value(decimals="6"), "parsed"),
        (mint_value(decimals=300), "out of range"),
        (mint_value(extensions=[{"extension": "permanentDelegate"}]), "SPL Token"),
        (mint_value(program=TOKEN_2022_PROGRAM, extensions="x"), "extension list"),
        ({"owner": TOKEN_PROGRAM, "data": ["AAAA", "base64"]}, "parsed account data"),
        ({"data": {}}, "owner"),
    ],
)
def test_malformed_mint_is_unknown_and_never_revoked(tmp_path: Path, value: Any, why: str) -> None:
    body = run(tmp_path, value)
    f = flags(body)
    assert f["MALFORMED_MINT_ACCOUNT"] == "TRIGGERED"
    assert (
        why in body["provenance"]["inputs"][0]["outcome"] + body["identity"]["token_mint"]["reason"]
    )
    for key in ("mint_authority", "freeze_authority"):
        assert body["authority"][key]["status"] == "UNKNOWN"  # evidence exists, meaning unresolved
        assert body["authority"][key]["value"] is None
        assert f[key.upper() + "_ACTIVE"] == "UNDETERMINED"
    assert body["identity"]["token_mint"]["status"] == "UNVERIFIED"
    assert body["assessment"]["coverage"] == "INSUFFICIENT"


def test_unknown_differs_from_unavailable(tmp_path: Path) -> None:
    malformed = run(tmp_path / "a", mint_value(supply="bad"))
    missing_account = run(tmp_path / "b", None)
    assert malformed["authority"]["mint_authority"]["status"] == "UNKNOWN"
    assert missing_account["authority"]["mint_authority"]["status"] == "UNAVAILABLE"
    assert malformed["coverage"]["components"]["mint_account"] == "UNKNOWN"
    assert missing_account["coverage"]["components"]["mint_account"] == "AVAILABLE"


def test_malformed_rpc_envelope_is_a_provider_failure(tmp_path: Path) -> None:
    body = run(tmp_path, ("raw", {"unexpected": True}))
    assert body["provenance"]["inputs"][0]["outcome"] == "PROVIDER_FAILED"
    assert body["provenance"]["inputs"][0]["raw_hash"] is None
    assert body["authority"]["mint_authority"]["status"] == "PROVIDER_UNAVAILABLE"


# --- missing evidence never decides a rule NOT_TRIGGERED ----------------------------------


def _row(outcome: str, **kw: Any) -> MintRow:
    base: dict[str, Any] = dict(
        id=1, canonical_id=f"solana:{MINT}", collection_id=1, fetched_at=0.0, provider="fake",
        outcome=outcome, reason=None if outcome == "MINT" else "why", raw_hash=None,
        context_slot=None, program_owner=None, token_program=None, decimals=None,
        supply_raw=None, mint_authority=None, freeze_authority=None, extensions=None,
    )  # fmt: skip
    if outcome == "MINT":
        base.update(program_owner=TOKEN_PROGRAM, token_program="spl_token", decimals=6,
                    supply_raw="1", extensions=[])  # fmt: skip
    base.update(kw)
    return MintRow(**base)


@pytest.mark.parametrize("outcome", [o for o in MINT_OUTCOMES if o != "MINT"] + [None])
def test_missing_evidence_never_becomes_not_triggered(outcome: str | None) -> None:
    row = _row(outcome) if outcome else None
    authority = authority_fields(row)
    status, _ = token_identity(row)
    results = {r.id: r for r in evaluate(status, outcome, row.reason if row else None,
                                         row.program_owner if row else None, authority)}  # fmt: skip
    for rid in ("MINT_AUTHORITY_ACTIVE", "FREEZE_AUTHORITY_ACTIVE"):
        assert results[rid].outcome == "UNDETERMINED", (outcome, rid)
    for field in authority.values():
        assert field.status != "AVAILABLE" and field.value is None
    if outcome in (None, "PROVIDER_FAILED", "NOT_COLLECTED"):
        assert {r.outcome for r in results.values()} == {"UNDETERMINED"}
        assert all(r.needs for r in results.values())
