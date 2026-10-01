"""Shadow MARKET_UNAVAILABLE audit (`audit-unavailable`): read-only, no network, one test
per classification, JSON output and summary counts. Temporary databases only; evidence is
built with the production payload serializers through the Shadow test harness.

Harness strategy timeouts: no exact-pool price for 120 min -> MARKET_UNAVAILABLE; a
pending exit gets 60 min to find a price."""

import hashlib
import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

from upscale.services.evidence_archive.store import EvidenceStore, PendingRecord
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow.audit import CLASSES, AuditSettings, audit_unavailable
from upscale.services.shadow.cli import main as cli_main
from upscale.services.shadow.store import ShadowStore, ShadowStoreError

from .test_shadow import T0, A, B, Harness

F = "MintFFFF9999"  # an ineligible filler: keeps Scout runs going, never entered


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[GrowthScoutResult, GrowthCandidate]:
    from .test_outcomes import ranked

    result, store = ranked(tmp_path_factory.mktemp("template"))
    store.close()
    return result, next(c for c in result.candidates if c.stage == "ACCELERATING")


@pytest.fixture
def h(tmp_path: Path, template: Any) -> Any:
    harness = Harness(tmp_path, template)
    yield harness
    harness.close()


def m(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def runs(h: Harness, *minutes: float, extra: Any = None) -> None:
    """Scout runs that keep collection alive without pricing A's pool."""
    for x in minutes:
        h.scan(m(x), h.cand(F, eligible=False), *(extra(x) if extra else ()))


def record(h: Harness, at: datetime, kind: str = "market", **over: Any) -> None:
    h._observed = at.timestamp()
    fields: dict[str, Any] = {
        "kind": kind, "asset_id": f"solana:{A}", "chain": "solana", "address": A,
        "pool_address": f"pool-{A}", "provider": "GeckoTerminal", "observed_at": at,
        "payload": {"candidate": {"metrics": {"price_usd": 1.0}, "pool": {"address": f"pool-{A}"}}},
    } | over  # fmt: skip
    assert h.writer.append(PendingRecord(**fields))


def audit(h: Harness, **kw: Any) -> dict[str, Any]:
    h.now = h.now or m(10 * 60)  # every timeout has passed
    h.run()
    return audit_unavailable(h.store, h.reader, "t", **kw)


def only(report: dict[str, Any]) -> dict[str, Any]:
    (c,) = report["cases"]
    return c


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- classifications -----------------------------------------------------------------------------


def test_evidence_collection_gap_when_scout_runs_stop(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(60), 1.01)
    h.now = m(8 * 60)  # nothing archived after: Scout stopped running
    c = only(audit(h))
    assert c["classification"] == "EVIDENCE_COLLECTION_GAP" and c["collection_gap"]
    assert c["scout_runs_in_window"] == 0 and c["max_scout_run_gap_minutes"] == 120.0
    assert c["exit_mechanism"] == "MARKET_UNAVAILABLE_TIMEOUT" and c["timeout_minutes"] == 120
    assert c["last_valid_price"] == 1.01 and c["last_valid_price_at"] == m(60).isoformat()
    assert c["exit_at"] == m(180).isoformat() and c["unavailable_minutes"] == 120
    assert c["first_unavailable_at"] == m(60).isoformat()
    assert c["last_valid_observation"]["kind"] == "market"


def test_evidence_collection_gap_when_an_exact_pool_price_was_not_used(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(h, 30, 60, 90)
    # A Solana DEX snapshot of the exact pool: archived, but not a kind Shadow reads.
    record(h, m(45), kind="dex_market", payload={"snapshot": {"price_usd": 1.02,
                                                              "pair_address": f"pool-{A}"}})  # fmt: skip
    c = only(audit(h))
    assert c["classification"] == "EVIDENCE_COLLECTION_GAP" and not c["collection_gap"]
    assert [u["kind"] for u in c["unused_exact_pool_prices"]] == ["dex_market"]


def test_temporary_unavailable_with_later_recovery(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(h, 30, 60, 90)
    h.price(m(150), 1.3)  # the exact pool, after the exit
    c = only(audit(h))
    assert c["classification"] == "TEMPORARY_UNAVAILABLE" and not c["collection_gap"]
    assert c["first_unavailable_at"] == m(30).isoformat()  # the first run without its price
    r = c["retrospective"]
    assert r["later_recovered"] and r["first_recovery_at"] == m(150).isoformat()
    assert r["first_recovery_price"] == 1.3 and "RETROSPECTIVE" in r["note"]


def test_unknown_without_any_later_observation(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(h, 30, 60, 90, 120, 150)
    c = only(audit(h))
    assert c["classification"] == "UNKNOWN" and not c["retrospective"]["later_recovered"]


def test_pool_changed(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(
        h, 30, 60, 90, extra=lambda x: [h.cand(A, price=1.1, pool="other-pool")] if x == 60 else []
    )
    c = only(audit(h))
    assert c["classification"] == "POOL_CHANGED"
    assert c["other_pool_appeared"] and c["other_pools"] == ["other-pool"]


def test_pending_exit_timeout_is_its_own_mechanism(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.scan(m(30), h.cand(A, price=1.1, stage="FADING", pool="other-pool"))
    runs(h, 60, 90, 120)
    c = only(audit(h))
    assert c["exit_mechanism"] == "EXIT_PRICE_TIMEOUT" and c["exit_trigger"] == "PENDING_EXIT"
    assert c["timeout_minutes"] == 60 and c["classification"] == "POOL_CHANGED"


def test_provider_gap_from_stale_carried_scout_records(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(h, 30, 60, 90,
         extra=lambda x: [h.cand(A, price=1.0, data_status="STALE_CARRIED")] if x == 60 else [])  # fmt: skip
    c = only(audit(h))
    assert c["classification"] == "PROVIDER_GAP" and c["stale_carried_scout_records"] == 1


def test_provider_gap_from_rate_limited_records(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(h, 30, 60, 90)
    record(h, m(50), availability="RATE_LIMITED", payload={})
    c = only(audit(h))
    assert c["classification"] == "PROVIDER_GAP" and c["provider_failure_records"] == 1
    assert c["provider_failures_by_kind"] == {"market:RATE_LIMITED": 1}


def test_true_market_disappearance(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(h, 30, 60, 90, 120, 150)
    record(h, m(50), availability="NOT_AVAILABLE", payload={})
    c = only(audit(h))
    assert c["classification"] == "TRUE_MARKET_DISAPPEARANCE" and c["not_found_records"] == 1


def test_not_found_but_recovered_is_temporary(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    runs(h, 30, 60, 90)
    record(h, m(50), availability="NOT_AVAILABLE", payload={})
    h.price(m(150), 1.2)
    assert only(audit(h))["classification"] == "TEMPORARY_UNAVAILABLE"


def test_liquidity_collapse(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0, liquidity=80_000.0))
    h.scan(m(30), h.cand(A, price=1.0, liquidity=5_000.0))  # still priced: the last mark
    runs(h, 60, 90, 120, 150, 180)
    c = only(audit(h))
    assert c["classification"] == "LIQUIDITY_COLLAPSE" and c["liquidity_collapsed"]
    assert c["entry_liquidity_usd"] == 80_000.0 and c["last_liquidity_usd"] == 5_000.0


# --- summary, filters, JSON, read-only, no network -----------------------------------------------


def _two(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, price=2.0))
    runs(h, 30, 60, 90)
    h.price(m(150), 1.3)  # A recovers; B is never seen again
    h.now = m(10 * 60)


def test_summary_counts(h: Harness) -> None:
    _two(h)
    report = audit(h)
    s = report["summary"]
    assert s["total_market_unavailable"] == 2
    assert set(s["by_classification"]) == set(CLASSES)
    assert s["by_classification"]["TEMPORARY_UNAVAILABLE"] == 1
    assert s["by_classification"]["UNKNOWN"] == 1
    assert s["by_strategy"] == {"t": 2} and s["by_chain"] == {"solana": 2}
    assert s["later_recovered"] == 1 and s["permanent_disappearance"] == 1
    assert s["evidence_or_provider_gap"] == 0
    assert s["median_unavailable_minutes"] == 120
    assert s["unresolved_cost_usd"] == pytest.approx(2000.0)
    book = h.store.checkpoint("t")["books"]["t@v1"]["state"]
    assert s["unresolved_cost_usd"] == pytest.approx(book["unresolved_cost"])


def test_filters(h: Harness) -> None:
    _two(h)
    h.run()
    assert audit_unavailable(h.store, h.reader, "t", ["other"])["cases"] == []
    assert len(audit_unavailable(h.store, h.reader, "t", ["t"])["cases"]) == 2
    assert audit_unavailable(h.store, h.reader, "t", since=m(200))["cases"] == []
    assert audit_unavailable(h.store, h.reader, "t", until=m(119))["cases"] == []
    assert len(audit_unavailable(h.store, h.reader, "t", since=m(119), until=m(120))["cases"]) == 2
    tight = audit_unavailable(h.store, h.reader, "t", settings=AuditSettings(recovery_hours=0.5))
    assert tight["summary"]["later_recovered"] == 0  # A's recovery is 31 min after its exit


def _closed(h: Harness) -> tuple[Path, Path]:
    _two(h)
    h.run()
    h.close()
    return h.shadow_path, h.ev_path


def test_cli_json_from_the_environment(
    h: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shadow, evidence = _closed(h)
    monkeypatch.setenv("UPSCALE_SHADOW_DB", str(shadow))
    monkeypatch.setenv("UPSCALE_EVIDENCE_DB", str(evidence))
    assert cli_main(["audit-unavailable", "--run", "t", "--strategy", "t", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["read_only"] and d["run_id"] == "t" and d["summary"]["total_market_unavailable"] == 2
    keys = {"strategy_id", "chain", "address", "pool", "entry_at", "entry_price",
            "last_valid_observation", "last_valid_price", "first_unavailable_at", "exit_at",
            "unavailable_minutes", "exit_mechanism", "other_pool_appeared", "collection_gap",
            "provider_failure_records", "classification", "retrospective"}  # fmt: skip
    assert all(keys <= set(c) for c in d["cases"])
    assert cli_main(["audit-unavailable", "--run", "t"]) == 0
    out = capsys.readouterr().out
    assert "total MARKET_UNAVAILABLE: 2" in out and "RETROSPECTIVE" in out


def test_read_only(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    shadow, evidence = _closed(h)
    before = {p: digest(p) for p in (shadow, evidence)}
    files = set(shadow.parent.iterdir())
    args = ["--db", str(shadow), "--evidence-db", str(evidence), "audit-unavailable", "--run", "t"]
    assert cli_main([*args, "--json"]) == 0
    assert {p: digest(p) for p in (shadow, evidence)} == before
    # SQLite's shared-memory side files of a WAL database (present anyway in production,
    # where the writer has them open) are the only new files, and hold no data.
    new = set(shadow.parent.iterdir()) - files
    assert {p.name.rsplit("-", 1)[1] for p in new} <= {"wal", "shm"}
    assert all(p.stat().st_size == 0 for p in new if p.name.endswith("-wal"))
    store = ShadowStore(shadow, read_only=True)
    with pytest.raises(ShadowStoreError):
        store.register(next(iter(store.strategies())))
    store.close()
    capsys.readouterr()
    # A missing database is an error, never created.
    missing = shadow.parent / "nope.sqlite3"
    assert cli_main(["--db", str(missing), "--evidence-db", str(evidence),
                     "audit-unavailable", "--run", "t"]) == 1  # fmt: skip
    assert not missing.exists()
    evidence_missing = shadow.parent / "nope-evidence.sqlite3"
    assert cli_main(["--db", str(shadow), "--evidence-db", str(evidence_missing),
                     "audit-unavailable", "--run", "t"]) == 1  # fmt: skip
    assert not evidence_missing.exists()


def test_no_provider_or_network_calls(
    h: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shadow, evidence = _closed(h)

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("the audit must not use the network")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert cli_main(["--db", str(shadow), "--evidence-db", str(evidence),
                     "audit-unavailable", "--run", "t", "--json"]) == 0  # fmt: skip
    assert json.loads(capsys.readouterr().out)["summary"]["total_market_unavailable"] == 2


def test_audit_does_not_change_the_book(h: Harness) -> None:
    _two(h)
    h.run()
    rows = (h.store.counts(), h.store.checkpoint("t"), h.trades())
    audit_unavailable(h.store, h.reader, "t")
    h.run()
    assert (h.store.counts(), h.store.checkpoint("t"), h.trades()) == rows
    assert isinstance(h.reader, EvidenceStore) and h.reader.read_only
