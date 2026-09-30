"""Decision-time semantics of the evidence archive: evidence keeps its own observed time,
a decision is final at D >= every piece of evidence it used, and replay at any time sees
only what existed then. Temporary databases only."""

import asyncio
import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from upscale.services.evidence_archive import hooks
from upscale.services.evidence_archive.recorder import EvidenceRecorder
from upscale.services.evidence_archive.status import status
from upscale.services.evidence_archive.store import EvidenceStore, PendingRecord
from upscale.services.replay_lab.sampling import archived_decision_times
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.growth.service import GrowthScoutService
from upscale.services.scout.normalize import Listing, build_candidates
from upscale.services.scout.store import ScoutSnapshotStore

from .test_evidence_archive import safety
from .test_replay_lab import NOW, T0, Harness, build_archive, config, dex_pool, mint
from .test_scout_growth import momentum

MARKET = T0  # 10:00:00 (the recorded Scout snapshot in the replay archive)
SOCIAL = T0 + timedelta(seconds=3)  # 10:00:03
SAFETY = T0 + timedelta(seconds=9)  # 10:00:09
DECISION = T0 + timedelta(seconds=10)  # 10:00:10
CID = f"solana:{mint(0)}"


@pytest.fixture
def recorder(tmp_path: Path) -> Iterator[EvidenceRecorder]:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    rec = EvidenceRecorder(store, versions=lambda: {"code": "test"},
                           capabilities=lambda: ["candles", "dex", "market_snapshot", "news", "onchain"])  # fmt: skip
    previous = hooks.installed()
    hooks.install(rec)
    yield rec
    rec.flush()
    hooks.install(previous)
    store.close()


def scout_run(decision_at: Any, safety_at: Any = SAFETY, extra_safety: Any = None) -> None:
    """One live-like Scout run: market observed, social measured, safety fetched during
    the ranking, decision final at `decision_at` (all inside one evaluation scope)."""
    listing = Listing(provider="DEX Screener", kind="new", name="new_pools", fetched_at=MARKET)
    candidate = build_candidates([dex_pool(0, MARKET)], listing, ScoutConfig()).candidates[0]
    social = momentum(mint(0), "ACCELERATING", computed_at=SOCIAL)
    snap = safety(mint(0), safety_at)
    with hooks.evaluation() as used:
        hooks.emit("market", candidate)
        hooks.emit("social", social)
        hooks.emit("safety", snap, pools=[])
        growth = GrowthScoutService(ScoutSnapshotStore(":memory:"), GrowthConfig(),
                                    now=lambda: T0 + timedelta(seconds=1))  # fmt: skip
        result = asyncio.run(growth.rank([candidate], social={CID: social}, safety={CID: snap}))
    if extra_safety is not None:  # later evidence of the same token, outside the run
        hooks.emit("safety", extra_safety, pools=[])
    hooks.emit("scout", result, decision_at=decision_at, scope=used)


def test_evidence_visibility_follows_observed_time(recorder: EvidenceRecorder) -> None:
    scout_run(DECISION)
    recorder.flush()
    store = recorder.store

    def visible(at: Any) -> dict[str, bool]:
        return {k: store.latest(k, CID, until=at) is not None  # type: ignore[arg-type]
                for k in ("market", "social", "safety", "scout")}  # fmt: skip

    assert visible(MARKET) == {"market": True, "social": False, "safety": False, "scout": False}
    assert visible(T0 + timedelta(seconds=5)) == {
        "market": True, "social": True, "safety": False, "scout": False,
    }  # fmt: skip
    assert visible(DECISION) == {"market": True, "social": True, "safety": True, "scout": True}
    decision = store.latest("scout", CID, until=DECISION)
    assert decision is not None and decision.observed_at == DECISION
    timing = decision.payload["timing"]
    assert timing["market_observed_at"] == MARKET.isoformat()
    assert timing["decision_at"] == DECISION.isoformat()
    kinds = {x["kind"]: x["observed_at"] for x in decision.links["evidence"]}
    assert kinds == {"market": MARKET.isoformat(), "social": SOCIAL.isoformat(),
                     "safety": SAFETY.isoformat()}  # fmt: skip
    # The linked safety is exactly the archived record (not merely "the same asset").
    linked = next(x for x in decision.links["evidence"] if x["kind"] == "safety")
    resolved = store.resolve(linked)
    assert resolved is not None and resolved.record_id == linked["record_id"]
    assert decision.links["causal_valid"] is True
    # Only the decision at 10:00:10 can be reproduced; none exists before it.
    cfg = config(start=T0 - timedelta(hours=1), end=T0 + timedelta(hours=1))
    assert archived_decision_times(store, cfg) == {(CID, round(MARKET.timestamp(), 3)): DECISION}


def test_decision_can_not_precede_the_evidence_it_used(recorder: EvidenceRecorder) -> None:
    scout_run(T0 + timedelta(seconds=5))  # claims to be final before the safety it used
    recorder.flush()
    store = recorder.store
    [decision] = store.records("scout")
    assert decision.links["causal_valid"] is False
    assert any("safety" in v for v in decision.payload["causal"]["violations"])
    assert recorder.stats.causal_violations == 1
    [safety_row] = store.records("safety")
    assert safety_row.observed_at == SAFETY  # evidence time is never "repaired"
    cfg = config(start=T0 - timedelta(hours=1), end=T0 + timedelta(hours=1))
    assert archived_decision_times(store, cfg) == {}  # not replay-capable
    report = status(store, NOW, days=60)
    assert report["readiness"]["causal_violations"] == 1
    assert report["readiness"]["full_opportunity_evaluation_possible"] == 0


def test_archived_at_never_decides_visibility(tmp_path: Path) -> None:
    late_writer = EvidenceStore(
        tmp_path / "e.sqlite3", clock=lambda: (T0 + timedelta(days=1)).timestamp()
    )
    late_writer.append(
        PendingRecord(kind="safety", asset_id=CID, observed_at=SAFETY, payload={"v": 1})
    )
    [r] = late_writer.records()
    assert r.archived_at == T0 + timedelta(days=1)
    assert late_writer.latest("safety", CID, until=SAFETY) is not None  # written a day later
    assert late_writer.latest("safety", CID, until=SAFETY - timedelta(seconds=1)) is None


def test_readiness_counts_linked_safety_at_decision_time(recorder: EvidenceRecorder) -> None:
    scout_run(DECISION)
    # A legacy record (before decision timing): stamped at its run's start, before the
    # safety it used; judged conservatively at that time.
    other = "solana:legacy"
    recorder.store.append(
        PendingRecord(kind="market", asset_id=other, observed_at=MARKET, payload={"m": 1})
    )
    recorder.store.append(
        PendingRecord(kind="safety", asset_id=other, observed_at=SAFETY, payload={"s": 1})
    )
    recorder.store.append(PendingRecord(kind="scout", asset_id=other, observed_at=T0 + timedelta(seconds=1),
                                        payload={"c": 1}, links={"run": "x"}))  # fmt: skip
    recorder.flush()
    ready = status(recorder.store, NOW, days=60)["readiness"]
    assert ready["timestamps"] == 2
    assert ready["with_onchain_safety"] == 1 and ready["with_linked_onchain_safety"] == 1
    assert ready["full_opportunity_evaluation_possible"] == 1 and ready["replay_usable"] == 1
    assert ready["decision_timed_evaluations"] == 1 and ready["legacy_timing_evaluations"] == 1


def _replay(tmp_path: Path, recorder: EvidenceRecorder, db: str) -> Any:
    recorder.flush()
    archive = tmp_path / "scout.sqlite3"
    if not archive.exists():
        build_archive(archive)  # the recorded snapshot at 10:00:00 (live stage stored 10:00:05)
    h = Harness(tmp_path, archive, db=db)
    h.runner.evidence = EvidenceStore(recorder.store.path, read_only=True)
    job = h.runner.create_job(config(mode="MARKET_PLUS_SOCIAL"))
    asyncio.run(h.runner.run(job))
    [sample] = h.store.samples(job)
    stored = h.store.decision(sample.id)
    assert stored is not None
    return sample, stored[0]


def test_replay_reproduces_the_archived_decision_at_d(
    tmp_path: Path, recorder: EvidenceRecorder
) -> None:
    later = safety(mint(0), DECISION + timedelta(minutes=1), mint_authority="Late1111", top1=90.0)
    scout_run(DECISION, extra_safety=later)
    sample, record = _replay(tmp_path, recorder, "r.sqlite3")
    assert sample.plan.decision_at == DECISION and sample.plan.market_observed_at == MARKET
    assert record.decision_at == DECISION and record.market_observed_at == MARKET
    assert record.decision_basis == "archived_scout_decision"
    assert record.evidence_latest_at <= DECISION
    # Market + social + the exact linked safety; the later snapshot is never substituted.
    assert record.availability["onchain_safety"].startswith(
        "AVAILABLE: linked to the archived decision"
    )
    snap = record.agents["onchain_safety"].findings["snapshot"]
    assert snap["fetched_at"].startswith(SAFETY.isoformat()[:19]) and snap["mint_authority"] is None
    assert record.social["status"] == "RECORDED_AT_OR_BEFORE_T"
    assert record.scout.safety_status == "SAFETY_CHECKS_COMPLETE"
    # Full Opportunity evaluation: the required on-chain evidence existed at D.
    missing = " ".join(record.agents["opportunity"].findings["missing_evidence"])
    assert "authorit" not in missing and "holder concentration" not in missing
    # Outcomes start at D; the reference is the market price observed at T.
    [first] = [o for o in record_outcomes(tmp_path, "r.sqlite3", sample.id) if o.horizon == "5m"]
    assert first.window_start == DECISION


def record_outcomes(tmp_path: Path, db: str, sample_id: int) -> list[Any]:
    from upscale.services.replay_lab.store import ReplayStore

    return ReplayStore(tmp_path / db).outcomes(sample_id)


def test_replay_at_market_time_never_sees_later_safety(
    tmp_path: Path, recorder: EvidenceRecorder
) -> None:
    # Without a valid archived decision (here: its linkage is invalid), the sample stays at
    # the market time T = 10:00:00, where the 10:00:09 safety and 10:00:03 social don't exist.
    scout_run(T0 + timedelta(seconds=5))
    sample, record = _replay(tmp_path, recorder, "m.sqlite3")
    assert sample.plan.decision_at == MARKET and sample.plan.market_observed_at is None
    assert "onchain_safety" not in record.agents
    assert record.availability["onchain_safety"].startswith("NOT_COLLECTED")
    assert record.social["status"] == "SOCIAL_UNAVAILABLE"
    assert record.decision_basis == "market_observation"


def test_decision_records_stay_immutable(recorder: EvidenceRecorder) -> None:
    scout_run(DECISION)
    recorder.flush()
    db = sqlite3.connect(recorder.store.path)
    for sql in ("UPDATE evidence_records SET observed_at = observed_at - 9 WHERE kind = 'safety'",
                "UPDATE evidence_records SET links_json = '{}' WHERE kind = 'scout'",
                "DELETE FROM evidence_records WHERE kind = 'scout'"):  # fmt: skip
        with pytest.raises(sqlite3.DatabaseError):
            with db:
                db.execute(sql)
