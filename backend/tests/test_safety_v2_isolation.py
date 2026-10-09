"""Safety V2 isolation: an independent feature family. It imports none of the decision /
learning / execution components or Radar, nothing imports it except the orchestration
bridge's single Safety adapter file (through allowed modules only), it emits no production
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
    # Phase 3: pure pool models / selection and DEX row parsing only (see below).
    "upscale.services.solana_dex",
    "upscale.services.dexscreener",
)
# The only names Safety V2 may take from solana_dex / dexscreener: never their caching,
# evidence-archiving services or their HTTP provider.
DEX_NAMES = {
    "upscale.services.solana_dex": {"DexPool", "PoolSelection", "PoolSelectionConfig",
                                    "TokenRef", "WindowStats", "select_primary_pool"},
    "upscale.services.dexscreener": {"parse_pair"},
}  # fmt: skip


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
    # Phase 2: registry.py, sources.py; Phase 3: market.py
    assert len(files) == 13, [f.name for f in files]
    return files


def test_safety_v2_imports_no_decision_learning_execution_or_radar_module() -> None:
    for f in _sources():
        for mod in _imports(f):
            assert not any(mod == x or mod.startswith(x + ".") for x in FORBIDDEN), (f, mod)
            if mod.startswith("upscale"):
                assert not any(w in mod for w in FORBIDDEN_WORDS), (f, mod)
                assert any(mod == a or mod.startswith(a + ".") for a in ALLOWED), (f, mod)


def test_safety_v2_takes_only_pure_dex_helpers() -> None:
    for f in _sources():
        tree = ast.parse(f.read_text(), str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name in DEX_NAMES for a in node.names), f
            if isinstance(node, ast.ImportFrom) and node.module in DEX_NAMES:
                names = {a.name for a in node.names}
                assert names <= DEX_NAMES[node.module], (f, names - DEX_NAMES[node.module])
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        banned = {"DexMarketService", "SolanaDexService", "DexScreenerProvider",
                  "build_snapshot", "get_snapshot"}  # fmt: skip
        assert not used & banned, (f, used & banned)


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


# The one production file outside Safety V2 allowed to reference it: the orchestration
# bridge's Safety adapter, and only through these Safety modules.
ADAPTER = UPSCALE / "services" / "opportunity_orchestrator" / "safety_adapter.py"
ADAPTER_MAY_IMPORT = frozenset(
    {"upscale.services.safety_v2.config", "upscale.services.safety_v2.models",
     "upscale.services.safety_v2.service"}
)  # fmt: skip


def outside_references(upscale: Path) -> list[Path]:
    """Files outside Safety V2 that mention it, except the one allowed adapter."""
    safety = upscale / "services" / "safety_v2"
    adapter = upscale / "services" / "opportunity_orchestrator" / "safety_adapter.py"
    return [f for f in sorted(upscale.rglob("*.py"))
            if safety not in f.parents and f != adapter and "safety_v2" in f.read_text()]  # fmt: skip


def adapter_violations(path: Path) -> list[str]:
    """Safety V2 modules the adapter imports beyond `ADAPTER_MAY_IMPORT`."""
    modules = {m for m in _imports(path) if m.startswith("upscale.services.safety_v2")}
    allowed = ADAPTER_MAY_IMPORT | {f"{m}.{n}" for m in ADAPTER_MAY_IMPORT for n in _names(path, m)}
    return sorted(modules - allowed)


def _names(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(), str(path))
    return {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            and node.module == module for a in node.names}  # fmt: skip


def test_nothing_outside_safety_v2_imports_it() -> None:
    """Only the orchestrator's Safety adapter may reference Safety V2, and only through the
    allowed modules."""
    assert outside_references(UPSCALE) == []
    if ADAPTER.exists():
        assert adapter_violations(ADAPTER) == []


def test_a_second_file_referencing_safety_v2_still_fails(tmp_path: Path) -> None:
    root = tmp_path / "upscale"
    (root / "services" / "safety_v2").mkdir(parents=True)
    orch = root / "services" / "opportunity_orchestrator"
    orch.mkdir(parents=True)
    (orch / "safety_adapter.py").write_text("from upscale.services.safety_v2.service import X\n")
    assert outside_references(root) == []
    (orch / "processing.py").write_text("from upscale.services.safety_v2 import service\n")
    (root / "services" / "other.py").write_text("# mentions safety_v2\n")
    assert sorted(f.name for f in outside_references(root)) == ["other.py", "processing.py"]


def test_the_adapter_may_import_only_the_allowed_safety_modules(tmp_path: Path) -> None:
    ok = tmp_path / "ok.py"
    ok.write_text("from upscale.services.safety_v2.service import service_from_env\n"
                  "from upscale.services.safety_v2.models import SafetyError\n")  # fmt: skip
    assert adapter_violations(ok) == []
    bad = tmp_path / "bad.py"
    bad.write_text("from upscale.services.safety_v2.rules import evaluate_changes\n"
                   "import upscale.services.safety_v2.repository\n")  # fmt: skip
    assert adapter_violations(bad) == [
        "upscale.services.safety_v2.repository", "upscale.services.safety_v2.rules",
        "upscale.services.safety_v2.rules.evaluate_changes",
    ]  # fmt: skip


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
