"""Radar V1 isolation: no production evidence, Scout read-only, CLI-only (nothing starts
Radar), its own database, and no use of production's Solana safety service. Offline."""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from tests.radar_fakes import MINT, POOL, T0, Clock, addr, make_service
from tests.test_radar_pipeline import CID, base_chain, setup
from upscale.services.evidence_archive import hooks

BACKEND = Path(__file__).resolve().parents[1]
RADAR = BACKEND / "upscale" / "services" / "radar"


class _Sink:
    def __init__(self) -> None:
        self.items: list[str] = []

    def submit(self, kind: str, obj: Any, component: str, extra: dict[str, Any]) -> None:
        self.items.append(kind)


def test_radar_emits_no_production_evidence(tmp_path: Any) -> None:
    sink = _Sink()
    previous = hooks.installed()
    hooks.install(sink)
    try:
        svc, clock, _ = setup(tmp_path)
        asyncio.run(svc.snapshot(CID))
    finally:
        hooks.install(previous)
    assert sink.items == []


def test_radar_code_never_uses_production_safety_paths() -> None:
    for f in RADAR.glob("*.py"):
        src = f.read_text()
        assert "SolanaSafetyService" not in src.replace(
            "`SolanaSafetyService` is never used", ""
        ), f
        assert "evidence.emit" not in src and "hooks.emit" not in src, f
        assert "solana_safety_service" not in src, f


def test_nothing_starts_radar() -> None:
    for f in (BACKEND / "upscale").rglob("*.py"):
        if RADAR in f.parents:
            continue
        assert "services.radar" not in f.read_text(), f"{f} must not import Radar in V1"


def _scout_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE scout_outcome_observations (id INTEGER PRIMARY KEY, canonical_id TEXT, "
            "chain TEXT, address TEXT, pool_address TEXT, anchored_at REAL)"
        )
        conn.execute("CREATE TABLE scout_latest (canonical_id TEXT PRIMARY KEY, body_json TEXT)")
        rows = [
            (f"solana:{MINT}", "solana", MINT, POOL, T0.timestamp()),
            ("base:0xabc", "base", "0xabc", "0xpool", T0.timestamp()),
            ("solana:bad", "solana", "bad", POOL, T0.timestamp()),
        ]
        conn.executemany(
            "INSERT INTO scout_outcome_observations (canonical_id, chain, address, pool_address, "
            "anchored_at) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.execute(
            "INSERT INTO scout_latest VALUES (?, ?)",
            (f"solana:{MINT}", json.dumps({"pool": {"address": POOL, "dex": "pumpswap",
                                                    "created_at": "2026-09-30T23:00:00+00:00"}})),
        )  # fmt: skip


def test_sync_from_scout_is_read_only_and_keeps_case(tmp_path: Any) -> None:
    scout = tmp_path / "scout.sqlite3"
    _scout_db(scout)
    digest = hashlib.sha256(scout.read_bytes()).hexdigest()
    svc = make_service(tmp_path, base_chain(), Clock())
    r = svc.sync_from_scout(str(scout), None, 10)
    assert r["added"] == [f"solana:{MINT}"]  # EVM and malformed rows ignored
    assert hashlib.sha256(scout.read_bytes()).hexdigest() == digest
    t = svc.repo.get_target(f"solana:{MINT}")
    assert t is not None and t.mint == MINT and t.dex == "pumpswap" and t.source == "scout"
    assert t.pool_created_at == T0.timestamp() - 3600
    assert svc.sync_from_scout(str(scout), None, 10)["existing"] == [f"solana:{MINT}"]
    missing = svc.sync_from_scout(str(tmp_path / "nope.sqlite3"), None, 10)
    assert missing["found"] is False and not (tmp_path / "nope.sqlite3").exists()


def test_target_pool_never_silently_changes(tmp_path: Any) -> None:
    svc = make_service(tmp_path, None, Clock())
    assert svc.add_target(MINT, POOL)[1] is True
    assert svc.add_target(MINT, addr("PooLZZZ"))[1] is False
    assert svc.repo.get_target(f"solana:{MINT}").pool_address == POOL  # type: ignore[union-attr]


def test_radar_uses_its_own_database(tmp_path: Any) -> None:
    svc, _, _ = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"radar.sqlite3"}
