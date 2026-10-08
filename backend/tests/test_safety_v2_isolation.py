"""Safety V2 isolation: an independent feature family. It imports none of the decision /
learning / execution components or Radar, nothing imports it, it emits no production
evidence, and it never writes another component's database. Offline."""

import ast
import asyncio
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.safety_v2_fakes import MINT, FakeRpc, make_service, mint_value
from upscale.services.evidence_archive import hooks
from upscale.services.safety_v2.models import SafetySchemaError
from upscale.services.safety_v2.repository import SafetyRepository

BACKEND = Path(__file__).resolve().parents[1]
UPSCALE = BACKEND / "upscale"
SAFETY = UPSCALE / "services" / "safety_v2"

FORBIDDEN = (
    "upscale.services.shadow",
    "upscale.services.calibration",
    "upscale.services.audit",
    "upscale.services.outcomes",
    "upscale.services.replay_lab",
    "upscale.services.opportunity",
    "upscale.orchestrator",
    "upscale.background_scout",
    "upscale.services.scout.service",
    "upscale.services.radar",
    "upscale.services.risk",
    "upscale.held_position_watch",
)
FORBIDDEN_WORDS = ("execution", "position")
# Everything Safety V2 may import from UpScale (exact modules or the package itself).
ALLOWED = (
    "upscale.services.chains",
    "upscale.services.market_data",
    "upscale.services.solana_chain",
    "upscale.services.safety_v2",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
            out += [f"{node.module}.{a.name}" for a in node.names]
    return out


def _sources() -> list[Path]:
    files = sorted(SAFETY.glob("*.py"))
    assert len(files) == 12, [f.name for f in files]  # Phase 2: registry.py, sources.py
    return files


def test_safety_v2_imports_no_decision_learning_execution_or_radar_module() -> None:
    for f in _sources():
        for mod in _imports(f):
            assert not any(mod == x or mod.startswith(x + ".") for x in FORBIDDEN), (f, mod)
            if mod.startswith("upscale"):
                assert not any(w in mod for w in FORBIDDEN_WORDS), (f, mod)
                assert any(mod == a or mod.startswith(a + ".") for a in ALLOWED), (f, mod)


def test_safety_v2_never_uses_production_safety_or_evidence_paths() -> None:
    banned = {"SolanaSafetyService", "solana_safety_service", "emit", "evidence_archive",
              "hooks"}  # fmt: skip
    for f in _sources():
        tree = ast.parse(f.read_text(), str(f))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= set(_imports(f))
        assert not names & banned, (f, names & banned)
        assert not [n for n in names if "evidence_archive" in n or "solana_safety" in n], f


def test_nothing_outside_safety_v2_imports_it() -> None:
    for f in UPSCALE.rglob("*.py"):
        if SAFETY in f.parents:
            continue
        assert "safety_v2" not in f.read_text(), f"{f} must not import Safety V2"


class _Sink:
    def __init__(self) -> None:
        self.items: list[str] = []

    def submit(self, kind: str, obj: Any, component: str, extra: dict[str, Any]) -> None:
        self.items.append(kind)


def test_no_production_evidence_is_emitted(tmp_path: Path) -> None:
    sink = _Sink()
    previous = hooks.installed()
    hooks.install(sink)
    try:
        svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
        cid, _ = svc.add_target(MINT)
        asyncio.run(svc.snapshot(cid))
    finally:
        hooks.install(previous)
    assert sink.items == []


def _other_db(path: Path, table: str) -> str:
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute(f"INSERT INTO {table} (body) VALUES ('x')")
    conn.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_safety_run_never_mutates_radar_or_scout_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    radar, scout = tmp_path / "radar.sqlite3", tmp_path / "scout.sqlite3"
    before = {radar: _other_db(radar, "radar_targets"), scout: _other_db(scout, "scout_tokens")}
    mtimes = {p: p.stat().st_mtime_ns for p in before}
    monkeypatch.setenv("UPSCALE_RADAR_DB", str(radar))
    monkeypatch.setenv("UPSCALE_SCOUT_DB", str(scout))
    svc, _, _ = make_service(tmp_path, FakeRpc({MINT: mint_value()}))
    cid, _ = svc.add_target(MINT)
    res = asyncio.run(svc.snapshot(cid))
    svc.repo.close()
    assert res.saved
    for p, digest in before.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        assert p.stat().st_mtime_ns == mtimes[p]


def test_safety_refuses_to_open_another_components_database(tmp_path: Path) -> None:
    radar = tmp_path / "radar.sqlite3"
    digest = _other_db(radar, "radar_targets")
    with pytest.raises(SafetySchemaError, match="other tables"):
        SafetyRepository(radar).db()
    assert hashlib.sha256(radar.read_bytes()).hexdigest() == digest
