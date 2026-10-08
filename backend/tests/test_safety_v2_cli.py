"""Safety V2 CLI, offline: provider keys are removed, so no command can make a request."""

import json
from pathlib import Path

import pytest

from tests.safety_v2_fakes import MINT
from upscale.services.safety_v2.cli import main


@pytest.fixture(autouse=True)
def _no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPSCALE_HELIUS_API_KEY", raising=False)
    monkeypatch.delenv("UPSCALE_SOLANA_RPC_URL", raising=False)


def test_cli_end_to_end_without_a_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = ["--db", str(tmp_path / "s.sqlite3")]
    assert main([*db, "targets", "add", "--token", MINT]) == 0
    assert main([*db, "collect", "--token", MINT, "--json"]) == 0
    capsys.readouterr()
    assert main([*db, "snapshot", "--token", f"solana:{MINT}", "--no-fetch", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["saved"] and out["body"]["authority"]["mint_authority"]["status"] == "NOT_COLLECTED"
    assert out["body"]["assessment"]["coverage"] == "INSUFFICIENT"
    assert main([*db, "show", "--id", str(out["snapshot_id"]), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == out["body"]
    assert main([*db, "rebuild", "--id", str(out["snapshot_id"])]) == 0
    assert "REPRODUCED" in capsys.readouterr().out
    assert main([*db, "status", "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["provider"] is None and st["requests_today"] == {}


def test_cli_rejects_non_solana_identities(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = ["--db", str(tmp_path / "s.sqlite3")]
    assert main([*db, "targets", "add", "--token", "0x" + "a" * 40]) == 2
    assert "EVM" in capsys.readouterr().err
    assert main([*db, "snapshot", "--token", MINT, "--no-fetch"]) == 2  # not a target
