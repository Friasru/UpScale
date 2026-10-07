"""Radar-local retention: dry run changes nothing; cleanup keeps roll-ups, baselines and
active target state; long-lived tables are never pruned. Offline."""

import asyncio
import json
import sqlite3
from datetime import timedelta
from typing import Any

from tests.radar_fakes import MINT, POOL
from tests.test_radar_pipeline import CID, second_round, setup
from upscale.services.radar import retention
from upscale.services.radar.cli import main


def _counts(path: str) -> dict[str, int]:
    with sqlite3.connect(path) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


def _by_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["table"]: r for r in rows}


def test_dry_run_then_cleanup_protects_baselines(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    second_round(chain, clock)
    asyncio.run(svc.snapshot(CID))
    svc.repo.record_request("2026-01-01", "getTransaction", "ok")  # an old ledger row
    path = svc.settings.db_path
    svc.repo.close()
    before = _counts(path)
    now = clock.now() + timedelta(days=40)

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as ro:
        plan = _by_table(retention.plan(ro, svc.settings, now))
    assert _counts(path) == before  # dry run: nothing changed
    assert plan["radar_wallet_flows"]["eligible"] == before["radar_wallet_flows"]
    assert plan["radar_tx"]["protected"]["activity cursor of an active target"] == 1
    assert plan["radar_holder_snapshots"]["eligible"] == 1  # the latest one is the baseline
    assert plan["radar_snapshots"]["eligible"] == 1
    assert plan["radar_requests"]["eligible"] == 1

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        done = _by_table(retention.cleanup(conn, svc.settings, now))
    after = _counts(path)
    assert done["radar_wallet_flows"]["deleted"] == before["radar_wallet_flows"]
    assert after["radar_tx"] == 1 and after["radar_snapshots"] == 1
    assert after["radar_holder_snapshots"] == 1 and after["radar_holder_balances"] > 0
    assert after["radar_scans"] == 1  # the early-history scan of the active target
    for kept in ("radar_targets", "radar_wallet_entries", "radar_creators", "radar_meta"):
        assert after[kept] == before[kept]
    # The next snapshot still has its holder baseline.
    clock.advance(41 * 86400)
    b = asyncio.run(svc.snapshot(CID)).body
    assert b["holders"]["baseline_observed_at"] is not None
    assert b["holders"]["holder_count_change"]["status"] == "AVAILABLE"


def test_flows_without_rollup_and_inactive_targets(tmp_path: Any) -> None:
    svc, clock, _ = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    path = svc.settings.db_path
    with sqlite3.connect(path) as conn:
        wallet = conn.execute("SELECT wallet FROM radar_wallet_flows LIMIT 1").fetchone()[0]
        conn.execute("DELETE FROM radar_wallet_entries WHERE wallet = ?", (wallet,))
    now = clock.now() + timedelta(days=40)
    with sqlite3.connect(path) as conn:
        plan = _by_table(retention.plan(conn, svc.settings, now))
    assert plan["radar_wallet_flows"]["protected"]["wallet-entry roll-up missing"] == 1
    svc.repo.set_target_status(CID, "INACTIVE")
    with sqlite3.connect(path) as conn:
        plan = _by_table(retention.plan(conn, svc.settings, now))
    assert plan["radar_snapshots"]["eligible"] == 1  # no active baseline left to protect
    assert plan["radar_tx"]["protected"]["activity cursor of an active target"] == 0


def test_cli_retention_status_and_cleanup_dry_run(tmp_path: Any, capsys: Any) -> None:
    db = str(tmp_path / "none.sqlite3")
    assert main(["--db", db, "retention-status"]) == 0
    assert "not found" in capsys.readouterr().out
    assert not (tmp_path / "none.sqlite3").exists()  # never created by a read-only command
    svc, clock, _ = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    svc.repo.close()
    before = _counts(svc.settings.db_path)
    assert main(["--db", svc.settings.db_path, "cleanup", "--dry-run", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and _counts(svc.settings.db_path) == before
    assert {t["table"] for t in out["tables"]} >= {"radar_wallet_flows", "radar_snapshots"}


def test_cli_local_commands(tmp_path: Any, capsys: Any) -> None:
    svc, clock, _ = setup(tmp_path)
    asyncio.run(svc.snapshot(CID))
    svc.repo.close()
    db = svc.settings.db_path
    assert main(["--db", db, "status", "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["background_enabled"] is False and st["budget"]["daily"] == 2000
    assert main(["--db", db, "inspect", "--token", MINT]) == 0
    assert "RADAR SNAPSHOT radar.snapshot.v1" in capsys.readouterr().out
    assert main(["--db", db, "targets", "list"]) == 0
    assert POOL in capsys.readouterr().out
    assert main(["--db", db, "inspect", "--token", "ethereum:0xabc"]) == 2
    capsys.readouterr()
