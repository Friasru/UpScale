"""Retention: bounded deletion from the Scout and Evidence Archive databases.

Temporary databases built with the production stores (schemas, triggers, versions); rows
are inserted directly. NOW is a fixed clock; every protection is checked by what survives a
real cleanup, the dry run by proving nothing changed, and every cleanup ends with SQLite's
own foreign-key and integrity checks."""

import asyncio
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from upscale.services.evidence_archive.store import EvidenceStore, PendingRecord
from upscale.services.outcomes.store import OutcomeStore
from upscale.services.retention import engine
from upscale.services.retention.cli import main as cli_main
from upscale.services.retention.config import RetentionSettings, load_settings
from upscale.services.retention.service import BackgroundRetention
from upscale.services.scout.social.store import SocialStore
from upscale.services.scout.store import ScoutSnapshotStore
from upscale.services.shadow.store import ShadowStore

NOW = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
DAY = 86_400.0
OLD = NOW - timedelta(days=12)  # older than every 7-day raw cutoff
RECENT = NOW - timedelta(hours=6)
S = RetentionSettings(mode="on", batch_size=50, batch_pause_seconds=0.0)


def ts(t: datetime) -> float:
    return t.timestamp()


# --- fixtures ------------------------------------------------------------------------------------


class Dbs:
    def __init__(self, tmp: Path) -> None:
        self.scout = tmp / "scout.sqlite3"
        self.evidence = tmp / "evidence.sqlite3"
        self.shadow = tmp / "shadow.sqlite3"
        ScoutSnapshotStore(str(self.scout))._db()
        SocialStore(str(self.scout))._db()
        OutcomeStore(str(self.scout))._db()
        self.ev = EvidenceStore(self.evidence, clock=lambda: ts(NOW))
        self.ev.totals()  # creates the file
        ShadowStore(self.shadow)._db()
        self._seq = 0

    def sc(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.scout)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # Scout
    def token(self, cid: str) -> None:
        with self.sc() as c:
            c.execute(
                "INSERT OR IGNORE INTO scout_tokens VALUES (?, 'solana', ?, 'T', 'T', ?, 'p', "
                "'new', ?, ?)",
                (cid, cid.split(":")[1], ts(OLD) - DAY, ts(NOW), ts(NOW)),
            )

    def snapshot(self, cid: str, at: datetime) -> None:
        self.token(cid)
        with self.sc() as c:
            c.execute(
                "INSERT INTO scout_snapshots (canonical_id, observed_at, provider, pool_address, "
                "dex, price_usd, windows_json) VALUES (?, ?, 'GeckoTerminal', 'pool', 'dex', 1.0, "
                "'[]')",
                (cid, ts(at)),
            )

    def momentum(self, cid: str, at: datetime, state: str) -> None:
        with self.sc() as c:
            c.execute(
                "INSERT INTO scout_social_momentum (canonical_id, computed_at, state, body_json) "
                "VALUES (?, ?, ?, '{}')",
                (cid, ts(at), state),
            )

    def check(
        self, cid: str, at: datetime, status: str, covered_to: datetime | None, provider: str = "x"
    ) -> None:
        with self.sc() as c:
            c.execute(
                "INSERT INTO scout_social_checks (canonical_id, provider, platform, checked_at, "
                "status, covered_from, covered_to) VALUES (?, ?, 'x', ?, ?, ?, ?)",
                (cid, provider, ts(at), status, None, ts(covered_to) if covered_to else None),
            )

    def social_snapshot(self, cid: str, at: datetime, provider: str = "x") -> None:
        with self.sc() as c:
            c.execute(
                "INSERT INTO scout_social_snapshots (canonical_id, provider, observed_at, status, "
                "body_json) VALUES (?, ?, ?, 'OK', '{}')",
                (cid, provider, ts(at)),
            )

    def observation(
        self, cid: str, at: datetime, horizons: list[tuple[str, datetime | None]]
    ) -> int:
        """`horizons`: (status, finalized_at) per horizon."""
        self._seq += 1
        with self.sc() as c:
            cur = c.execute(
                "INSERT INTO scout_outcome_observations (canonical_id, chain, address, "
                "pool_address, observed_at, anchored_at, run_id, anchor_reason, rank, stage, "
                "score, discovery_status, schema_version, body_json) VALUES (?, 'solana', ?, "
                "'pool', ?, ?, 'run', 'NEW', 1, 'EARLY', 50, 'OK', 1, '{}')",
                (cid, cid.split(":")[1], ts(at) + self._seq * 1e-3, ts(at)),
            )
            oid = int(cur.lastrowid or 0)
            for i, (status, fin) in enumerate(horizons):
                c.execute(
                    "INSERT INTO scout_outcome_horizons (observation_id, horizon, "
                    "horizon_minutes, due_at, status, finalized_at) VALUES (?, ?, 60, ?, ?, ?)",
                    (oid, f"h{i}", ts(at) + 3600, status, ts(fin) if fin else None),
                )
        return oid

    def decision(self, scout_observation_id: int) -> None:
        with self.sc() as c:
            c.execute(
                "INSERT INTO decision_observations (asset_id, analyzed_at, action, source, "
                "scout_observation_id, decision_key, schema_version, body_json) VALUES "
                "('solana:X', ?, 'buy', 'chat', ?, ?, 1, '{}')",
                (ts(OLD), scout_observation_id, f"k{scout_observation_id}"),
            )

    # Evidence
    def record(
        self, mint: str, at: datetime, kind: str = "market", pool: str | None = None,
        asset_id: str | None = None,
    ) -> None:  # fmt: skip
        self._seq += 1
        assert self.ev.append(
            PendingRecord(
                kind=kind,  # type: ignore[arg-type]
                asset_id=asset_id or f"solana:{mint}",
                chain="solana",
                address=mint,
                pool_address=pool or f"pool-{mint}",
                provider="GeckoTerminal",
                observed_at=at,
                payload={"seq": self._seq},
            )
        )

    # Shadow
    def run(
        self, run_id: str, cursor: datetime, until: datetime | None = None,
        books: dict[str, Any] | None = None,
    ) -> None:  # fmt: skip
        with sqlite3.connect(self.shadow) as c:
            c.execute(
                "INSERT INTO shadow_runs VALUES (?, ?, ?, ?, 1, 'REALISTIC_V1', '[]', '{}')",
                (run_id, ts(OLD), ts(OLD), ts(until) if until else None),
            )
            c.execute(
                "INSERT INTO shadow_checkpoints VALUES (?, ?, 0, ?, ?, '{}', ?)",
                (run_id, ts(cursor), ts(cursor), json.dumps(books or {}), ts(NOW)),
            )

    def position(
        self, run_id: str, mint: str, entry: datetime, closed: datetime | None,
        pool: str | None = None,
    ) -> None:  # fmt: skip
        self._seq += 1
        status = "CLOSED" if closed else "OPEN"
        with sqlite3.connect(self.shadow) as c:
            c.execute(
                "INSERT INTO shadow_positions VALUES (?, ?, 's', 1, ?, 'solana', ?, 'T', ?, 'dex', "
                "'d', ?, 1.0, ?, 1.0, 100.0, ?, 1.0, 1.0, ?, 1.0, 1.0, 0, NULL, ?, ?)",
                (f"p{self._seq}", run_id, f"solana:{mint}", mint, pool or f"pool-{mint}",
                 ts(entry), ts(entry), status, ts(entry),
                 ts(closed) if closed else None, "TAKE_PROFIT" if closed else None),
            )  # fmt: skip

    def go(
        self, dry_run: bool = False, settings: RetentionSettings = S, **kw: Any
    ) -> dict[str, Any]:
        return engine.run(
            settings, self.scout, self.evidence, self.shadow, dry_run=dry_run, now=NOW,
            sleep=lambda _: None, **kw,
        )  # fmt: skip


@pytest.fixture
def db(tmp_path: Path) -> Dbs:
    d = Dbs(tmp_path)
    d.run("live", NOW - timedelta(minutes=10))  # an unfinished run near the present
    return d


def table(report: dict[str, Any], name: str) -> dict[str, Any]:
    for d in report["databases"].values():
        for t in d.get("tables", []):
            if t["table"] == name:
                return t
    raise KeyError(name)


def rows(path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as c:
        return c.execute(sql, params).fetchall()


def integrity(path: Path) -> None:
    with sqlite3.connect(path) as c:
        assert c.execute("PRAGMA foreign_key_check").fetchall() == []
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def digest(path: Path) -> str:
    """Logical content of every table (independent of WAL / page layout)."""
    with sqlite3.connect(path) as c:
        h = hashlib.sha256()
        for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger') "
                                 "ORDER BY name"):  # fmt: skip
            h.update(name.encode())
            if (
                c.execute("SELECT type FROM sqlite_master WHERE name = ?", (name,)).fetchone()[0]
                == "table"
            ):
                for r in sorted(repr(r) for r in c.execute(f"SELECT * FROM {name}")):
                    h.update(r.encode())
        return h.hexdigest()


# --- settings ------------------------------------------------------------------------------------


def test_settings_defaults_are_off_with_7_7_7_30_days() -> None:
    s = load_settings(enabled="", evidence_days="", snapshot_days="", social_days="",
                      outcome_days="", interval_hours="", batch_size="", ignore_shadow_runs="")  # fmt: skip
    assert s.mode == "off"
    assert (s.evidence_days, s.snapshot_days, s.social_days, s.outcome_days) == (7, 7, 7, 30)
    assert s.interval_hours == 12
    assert load_settings(enabled="1").mode == "on"
    assert load_settings(enabled="dry-run").mode == "dry-run"
    assert load_settings(enabled="0").mode == "off"


def test_bad_values_never_shorten_retention_below_the_floors() -> None:
    s = load_settings(evidence_days="0", snapshot_days="-5", social_days="nan",
                      outcome_days="1", interval_hours="0.01", batch_size="1")  # fmt: skip
    assert s.evidence_days == 7  # invalid: the default
    assert s.snapshot_days == 7
    assert s.social_days == 7
    assert s.outcome_days == 14  # valid but below the floor: the floor
    assert s.interval_hours == 6
    assert s.batch_size == 50
    assert len(s.adjustments) == 5
    assert load_settings(evidence_days="1").evidence_days == 3


# --- dry run -------------------------------------------------------------------------------------


def test_dry_run_reports_and_changes_nothing(db: Dbs) -> None:
    for i in range(5):
        db.snapshot("solana:A", OLD + timedelta(minutes=i))
        db.record("A", OLD + timedelta(minutes=i))
    db.snapshot("solana:A", RECENT)
    db.record("A", RECENT)
    before = {p: digest(p) for p in (db.scout, db.evidence, db.shadow)}
    report = db.go(dry_run=True)
    assert {p: digest(p) for p in (db.scout, db.evidence, db.shadow)} == before
    snaps = table(report, "scout_snapshots")
    assert (snaps["eligible"], snaps["deleted"], snaps["older_than_cutoff"]) == (5, 0, 5)
    assert snaps["eligible_oldest"] == OLD.isoformat()
    assert snaps["approx_mb"] >= 0
    ev = table(report, "evidence_records")
    assert ev["eligible"] == 5
    s = report["databases"]["scout"]["before"]
    assert {"page_count", "freelist_count", "page_size", "freelist_mb"} <= set(s)
    assert report["totals"]["deleted_rows"] == 0


# --- Scout ---------------------------------------------------------------------------------------


def test_snapshots_keep_recent_latest_and_pending_outcome_tokens(db: Dbs) -> None:
    for i in range(3):
        db.snapshot("solana:A", OLD + timedelta(minutes=i))  # A: only old snapshots
        db.snapshot("solana:B", OLD + timedelta(minutes=i))
        db.snapshot("solana:P", OLD + timedelta(minutes=i))
    db.snapshot("solana:B", RECENT)
    db.snapshot("solana:P", RECENT)
    db.observation("solana:P", OLD, [("COMPLETE", OLD), ("PENDING", None)])
    report = db.go()
    t = table(report, "scout_snapshots")
    assert t["deleted"] == 5  # A's two older, B's three old
    left = rows(db.scout, "SELECT canonical_id, observed_at FROM scout_snapshots ORDER BY 1, 2")
    assert ("solana:A", ts(OLD) + 120) in left  # A's latest stays (scout_tokens provider fallback)
    assert len([r for r in left if r[0] == "solana:P"]) == 4  # pending horizon: all kept
    assert [r for r in left if r[0] == "solana:B"] == [("solana:B", ts(RECENT))]
    assert t["protected"] == {"latest snapshot of the token": 1,
                              "token has a pending outcome horizon": 3}  # fmt: skip
    assert rows(db.scout, "SELECT COUNT(*) FROM scout_tokens")[0][0] == 3  # never pruned
    integrity(db.scout)
    again = table(db.go(), "scout_snapshots")  # idempotent
    assert again["deleted"] == 0 and again["eligible"] == 0


def test_snapshot_cutoff_never_inside_production_lookbacks(db: Dbs) -> None:
    short = RetentionSettings(mode="on", snapshot_days=3, social_days=3)
    db.snapshot("solana:A", NOW - timedelta(days=3, hours=12))  # inside 72h lookback + 1 day
    db.snapshot("solana:A", RECENT)
    report = db.go(settings=short)
    t = table(report, "scout_snapshots")
    assert t["basis"]["effective_days"] == pytest.approx(4.0)  # Growth first-seen 72h + 1 day
    assert t["deleted"] == 0
    checks = table(report, "scout_social_checks")
    assert checks["basis"]["effective_days"] == pytest.approx(7.0)  # 144h momentum span + 1 day


def test_social_momentum_keeps_what_growth_scout_reads(db: Dbs) -> None:
    db.momentum("solana:A", OLD, "RISING")  # the latest measured: Growth Scout reads it
    db.momentum("solana:A", OLD - timedelta(hours=1), "QUIET")
    db.momentum("solana:A", OLD + timedelta(hours=1), "UNAVAILABLE")
    db.momentum("solana:A", RECENT, "UNAVAILABLE")  # the latest
    db.momentum("solana:B", OLD, "UNAVAILABLE")  # B's only row
    db.momentum("solana:B", RECENT - timedelta(days=2), "UNAVAILABLE")
    db.momentum("solana:B", RECENT, "UNAVAILABLE")
    t = table(db.go(), "scout_social_momentum")
    left = rows(db.scout, "SELECT canonical_id, state, computed_at FROM scout_social_momentum "
                          "ORDER BY 1, 3")  # fmt: skip
    assert ("solana:A", "RISING", ts(OLD)) in left
    assert ("solana:A", "QUIET", ts(OLD) - 3600) not in left
    assert ("solana:A", "UNAVAILABLE", ts(OLD) + 3600) not in left
    assert t["deleted"] == 3
    # What Growth Scout's `_stored_momentum` returns is unchanged: the latest measured row.
    measured = [r for r in left if r[0] == "solana:A" and r[1] != "UNAVAILABLE"]
    assert measured[-1] == ("solana:A", "RISING", ts(OLD))


def test_social_checks_keep_scheduler_inputs(db: Dbs) -> None:
    db.check("solana:A", OLD - timedelta(days=2), "PROVIDER_UNAVAILABLE", None)  # first unavailable
    db.check("solana:A", OLD - timedelta(days=1), "PROVIDER_UNAVAILABLE", None)
    db.check("solana:A", OLD, "OK", OLD)  # the max covered_to
    db.check("solana:A", OLD + timedelta(hours=1), "OK", OLD - timedelta(hours=5))
    db.check("solana:A", RECENT, "PROVIDER_UNAVAILABLE", None)  # latest, no coverage
    before = (
        rows(db.scout, "SELECT MAX(covered_to) FROM scout_social_checks")[0][0],
        rows(db.scout, "SELECT MIN(checked_at) FROM scout_social_checks "
                       "WHERE status = 'PROVIDER_UNAVAILABLE'")[0][0],
    )  # fmt: skip
    t = table(db.go(), "scout_social_checks")
    assert t["deleted"] == 2
    after = (
        rows(db.scout, "SELECT MAX(covered_to) FROM scout_social_checks")[0][0],
        rows(db.scout, "SELECT MIN(checked_at) FROM scout_social_checks "
                       "WHERE status = 'PROVIDER_UNAVAILABLE'")[0][0],
    )  # fmt: skip
    assert after == before


def test_social_snapshots_keep_latest_per_provider(db: Dbs) -> None:
    for i in range(3):
        db.social_snapshot("solana:A", OLD + timedelta(minutes=i), provider="x")
        db.social_snapshot("solana:A", OLD + timedelta(minutes=i), provider="reddit")
    db.social_snapshot("solana:B", RECENT)
    t = table(db.go(), "scout_social_snapshots")
    assert t["deleted"] == 4
    assert rows(db.scout, "SELECT provider, observed_at FROM scout_social_snapshots "
                          "WHERE canonical_id = 'solana:A' ORDER BY 1") == [
        ("reddit", ts(OLD) + 120), ("x", ts(OLD) + 120)]  # fmt: skip


# --- outcomes ------------------------------------------------------------------------------------


def test_outcomes_prune_only_finalized_history_together(db: Dbs) -> None:
    ancient = NOW - timedelta(days=45)
    db.snapshot("solana:Z", RECENT)  # the writer is alive
    done = db.observation("solana:A", ancient, [("COMPLETE", ancient), ("UNAVAILABLE", ancient),
                                                ("PARTIAL", ancient)])  # fmt: skip
    pending = db.observation("solana:B", ancient, [("COMPLETE", ancient), ("PENDING", None)])
    late = db.observation("solana:C", ancient, [("COMPLETE", NOW - timedelta(days=2))])
    linked = db.observation("solana:D", ancient, [("COMPLETE", ancient)])
    db.decision(linked)
    bare = db.observation("solana:E", ancient, [])
    young = db.observation("solana:F", NOW - timedelta(days=10), [("COMPLETE", OLD)])
    db.observation("solana:G", RECENT, [("PENDING", None)])  # keeps the table "alive"
    report = db.go()
    t = table(report, "scout_outcome_observations")
    assert t["deleted"] == 1
    assert t["children_deleted"] == {"scout_outcome_horizons": 3}
    ids = {r[0] for r in rows(db.scout, "SELECT id FROM scout_outcome_observations")}
    assert done not in ids
    assert {pending, late, linked, bare, young} <= ids
    assert rows(db.scout, "SELECT COUNT(*) FROM scout_outcome_horizons WHERE observation_id = ?",
                (done,))[0][0] == 0  # fmt: skip
    assert t["protected"] == {
        "pending / non-final horizon": 1,
        "horizon finalized within the retention window": 1,
        "linked by an Analyze decision observation": 1,
        "observation without horizons": 1,
    }
    integrity(db.scout)
    # The immutability triggers are back: nothing else can delete outcome history.
    with sqlite3.connect(db.scout) as c, pytest.raises(sqlite3.IntegrityError):
        c.execute("DELETE FROM scout_outcome_observations WHERE id = ?", (pending,))
    with sqlite3.connect(db.scout) as c, pytest.raises(sqlite3.IntegrityError):
        c.execute("DELETE FROM scout_outcome_horizons WHERE observation_id = ?", (late,))


def test_outcomes_under_30_days_are_never_eligible(db: Dbs) -> None:
    db.observation("solana:A", NOW - timedelta(days=29), [("COMPLETE", NOW - timedelta(days=29))])
    db.observation("solana:B", RECENT, [("PENDING", None)])
    t = table(db.go(), "scout_outcome_observations")
    assert t["older_than_cutoff"] == 0 and t["deleted"] == 0


# --- evidence ------------------------------------------------------------------------------------


def test_evidence_protects_open_pending_and_closed_position_windows(db: Dbs) -> None:
    db.run("v2", NOW - timedelta(minutes=5), books={"s:1": {"state": {
        "positions": {"p": {"asset_id": "solana:HELD", "address": "HELD", "pool": "pool-HELD"}},
        "pending_entries": {"d": {"asset_id": "solana:PEND", "address": "PEND",
                                  "pool": "pool-PEND"}},
    }}})  # fmt: skip
    db.position("v2", "OPEN", OLD, None)
    db.position("v2", "SHUT", OLD + timedelta(hours=1), OLD + timedelta(hours=3))
    for mint in ("HELD", "PEND", "OPEN", "SHUT", "FREE"):
        for i in range(3):
            db.record(mint, OLD - timedelta(days=3) + timedelta(minutes=i))  # long before
            db.record(mint, OLD + timedelta(hours=2, minutes=i))  # during SHUT's holding
        db.record(mint, RECENT)
    # A watch variant of the open asset (another asset id, same pool, upper-case pool).
    db.record("variant", OLD, pool="POOL-OPEN", asset_id="solana:Variant")
    db.record("variant", RECENT, pool="other", asset_id="solana:Variant")
    report = db.go()
    t = table(report, "evidence_records")
    left = rows(db.evidence, "SELECT asset_id, observed_at FROM evidence_records")
    by = {m: sorted(at for a, at in left if a == f"solana:{m}") for m in
          ("HELD", "PEND", "OPEN", "SHUT", "FREE")}  # fmt: skip
    assert len(by["HELD"]) == len(by["PEND"]) == len(by["OPEN"]) == 7  # all kept
    assert len(by["SHUT"]) == 4  # the 3 inside its holding window + recent
    assert by["FREE"] == [ts(RECENT)]
    assert ("solana:Variant", ts(OLD)) in left  # matched by pool, case-insensitively
    assert t["deleted"] == 3 + 3 + 3
    assert t["protected"]["Shadow open position / pending intent asset"] == 6 * 3 + 1
    assert t["protected"]["Shadow position holding window"] == 3
    assert report["shadow_guard"]["pending_entry_intents"] == 1
    integrity(db.evidence)
    # The append-only trigger is back.
    with sqlite3.connect(db.evidence) as c, pytest.raises(sqlite3.IntegrityError):
        c.execute("DELETE FROM evidence_records")


def test_evidence_keeps_latest_record_per_kind_and_asset(db: Dbs) -> None:
    db.record("A", OLD - timedelta(hours=1))
    db.record("A", OLD)  # A's latest market record (A not seen since)
    db.record("A", OLD, kind="safety")
    db.record("B", RECENT)
    t = table(db.go(), "evidence_records")
    assert t["deleted"] == 1
    assert t["protected"] == {"latest record of the kind and asset": 2}


def test_evidence_held_back_by_an_unfinished_shadow_runs_cursor(db: Dbs) -> None:
    db.run("lagging", OLD + timedelta(days=1))  # unfinished, cursor 11 days ago
    db.run("done", OLD - timedelta(days=5), until=OLD - timedelta(days=5))  # finished
    db.record("A", OLD - timedelta(days=2))  # before the cursor - 2 days lookback: free
    db.record("A", OLD)  # inside the lookback: kept
    db.record("A", RECENT)
    report = db.go()
    t = table(report, "evidence_records")
    assert t["basis"]["held_back_by_shadow_run"] == "lagging"
    assert t["cutoff"] == (OLD - timedelta(days=1)).isoformat()
    assert t["deleted"] == 1
    # Explicitly ignoring the abandoned run releases its hold (positions stay protected).
    released = RetentionSettings(mode="on", ignored_shadow_runs=("lagging",))
    assert table(db.go(settings=released), "evidence_records")["deleted"] == 1


def test_no_evidence_is_deleted_without_a_readable_shadow_database(db: Dbs) -> None:
    db.shadow.unlink()
    db.record("A", OLD - timedelta(hours=1))
    db.record("A", OLD)
    db.record("A", RECENT)
    db.snapshot("solana:A", OLD)
    db.snapshot("solana:A", OLD + timedelta(minutes=1))
    db.snapshot("solana:A", RECENT)
    report = db.go()
    t = table(report, "evidence_records")
    assert t["deleted"] == 0 and "Shadow protection unavailable" in (t["skipped"] or "")
    assert table(report, "scout_snapshots")["deleted"] == 2  # Scout is unaffected
    assert "error" in report["shadow_guard"]


def test_cleanup_never_writes_the_shadow_database(db: Dbs) -> None:
    db.position("live", "OPEN", OLD, None)
    db.record("OPEN", OLD)
    db.record("OPEN", RECENT)
    before = hashlib.sha256(db.shadow.read_bytes()).hexdigest()
    db.go()
    assert hashlib.sha256(db.shadow.read_bytes()).hexdigest() == before


# --- safeguards, batching, isolation -------------------------------------------------------------


def test_refuses_to_prune_a_table_whose_writer_stopped(db: Dbs) -> None:
    db.snapshot("solana:A", OLD - timedelta(days=1))
    db.snapshot("solana:A", OLD)  # newest is 12 days old: Scout stopped writing
    t = table(db.go(), "scout_snapshots")
    assert t["deleted"] == 0 and "writer may have stopped" in t["skipped"]


def test_refuses_future_data_and_cutoffs_inside_the_floor(db: Dbs) -> None:
    db.snapshot("solana:A", OLD)
    db.snapshot("solana:A", NOW + timedelta(days=3))
    t = table(db.go(), "scout_snapshots")
    assert t["deleted"] == 0 and "in the future" in t["skipped"]
    unsafe = RetentionSettings(mode="on", social_days=0.5)  # bypassing load_settings
    db.snapshot("solana:B", RECENT)
    plans = engine.scout_plans(sqlite3.connect(db.scout), unsafe, ts(NOW))
    social = next(p for p in plans if p.table == "scout_social_snapshots")
    conn = sqlite3.connect(db.scout)
    r = engine.run_plan(conn, social, ts(NOW), dry_run=False, batch_size=50)
    assert "floor" in (r.skipped or "")


def test_unknown_schema_version_is_skipped(db: Dbs) -> None:
    with sqlite3.connect(db.evidence) as c:
        c.execute("PRAGMA user_version = 99")
    t = table(db.go(dry_run=True), "evidence_records")
    assert "schema v99 unknown" in (t["skipped"] or "")


def test_small_batches_and_time_budget_resume_incrementally(db: Dbs) -> None:
    for i in range(230):
        db.snapshot("solana:A", OLD + timedelta(seconds=i))
    db.snapshot("solana:A", RECENT)
    clock = iter(range(0, 10_000))
    first = db.go(budget_seconds=2.5, monotonic=lambda: float(next(clock)))
    t = table(first, "scout_snapshots")
    assert t["incomplete"] and 0 < t["deleted"] < 230
    assert t["deleted"] % 50 == 0  # whole batches of `batch_size`
    rest = table(db.go(), "scout_snapshots")
    assert t["deleted"] + rest["deleted"] == 230 and not rest["incomplete"]
    assert rows(db.scout, "SELECT COUNT(*) FROM scout_snapshots")[0][0] == 1
    integrity(db.scout)


def test_a_failing_database_never_stops_the_other(db: Dbs, tmp_path: Path) -> None:
    db.snapshot("solana:A", OLD)
    db.snapshot("solana:A", OLD + timedelta(minutes=1))
    db.snapshot("solana:A", RECENT)
    report = engine.run(S, db.scout, tmp_path / "missing.sqlite3", db.shadow, dry_run=False,
                        now=NOW)  # fmt: skip
    assert "error" in report["databases"]["evidence"]
    assert table(report, "scout_snapshots")["deleted"] == 2
    assert report["totals"]["errors"]


def test_background_pass_is_isolated_and_defers(db: Dbs) -> None:
    calls: list[str] = []

    def boom(*_: Any, **__: Any) -> dict[str, Any]:
        calls.append("pass")
        raise RuntimeError("disk on fire")

    bg = BackgroundRetention(S, lambda: None, lambda: str(db.scout), lambda: str(db.evidence),
                             lambda: str(db.shadow), now=lambda: NOW)  # fmt: skip
    bg._pass = boom  # type: ignore[method-assign]
    result = asyncio.run(bg.run_once())
    assert result["status"] == "failed" and "disk on fire" in result["error"]
    assert not bg.running
    busy = BackgroundRetention(S, lambda: "a Scout scan is running", lambda: str(db.scout),
                               lambda: str(db.evidence), lambda: str(db.shadow), now=lambda: NOW)  # fmt: skip
    assert asyncio.run(busy.run_once())["status"] == "deferred"
    off = BackgroundRetention(RetentionSettings(), lambda: None, lambda: "", lambda: "", lambda: "")

    async def start_off() -> bool:
        off.start()
        return off.active

    assert asyncio.run(start_off()) is False


def test_background_dry_run_mode_never_deletes(db: Dbs) -> None:
    db.snapshot("solana:A", OLD)
    db.snapshot("solana:A", OLD + timedelta(minutes=1))
    db.snapshot("solana:A", RECENT)
    bg = BackgroundRetention(RetentionSettings(mode="dry-run"), lambda: None, lambda: str(db.scout),
                             lambda: str(db.evidence), lambda: str(db.shadow), now=lambda: NOW)  # fmt: skip
    result = asyncio.run(bg.run_once())
    assert result["status"] == "completed" and result["mode"] == "dry-run"
    assert result["eligible_rows"] == 2 and result["deleted_rows"] == 0
    assert rows(db.scout, "SELECT COUNT(*) FROM scout_snapshots")[0][0] == 3


def test_cli_status_and_dry_run(db: Dbs, capsys: pytest.CaptureFixture[str]) -> None:
    db.snapshot("solana:A", OLD)
    db.snapshot("solana:A", RECENT)
    before = digest(db.scout)
    base = ["--scout-db", str(db.scout), "--evidence-db", str(db.evidence),
            "--shadow-db", str(db.shadow)]  # fmt: skip
    assert cli_main([*base, "status"]) == 0
    out = capsys.readouterr().out
    assert "freelist_count" in out and "scout_snapshots" in out
    assert cli_main([*base, "cleanup", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "NO CHANGES MADE" in out and "evidence_records" in out
    assert cli_main([*base, "cleanup", "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run"
    assert digest(db.scout) == before


def test_open_production_connections_keep_working_across_a_cleanup(db: Dbs) -> None:
    """The archive writer's long-lived connection appends before and after the trigger is
    dropped / recreated inside a cleanup transaction, and still can't delete."""
    db.record("A", OLD - timedelta(hours=1))
    db.record("A", OLD)
    db.record("A", RECENT)
    assert table(db.go(), "evidence_records")["deleted"] == 2
    db.record("A", RECENT + timedelta(minutes=1))  # same writer connection as before
    assert db.ev.totals()["total"] == 2
    conn = db.ev._db()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM evidence_records")
    conn.rollback()
    integrity(db.evidence)
