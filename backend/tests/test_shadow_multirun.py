"""Production blockers for running continuous-v2 and continuous-v2-realistic side by side:
multi-run background Shadow (legacy single-run behavior unchanged, independent runs, one
failure never blocking another, no concurrent advancement of one run), the held-position
watch covering pending REALISTIC_V1 entry intents, and the comparison's matched /
unmatched / divergence semantics. Temporary databases only."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import upscale.main
from upscale.held_position_watch import held_pools, targets_summary
from upscale.services.evidence_archive.recorder import EvidenceRecorder
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow import engine as engine_module
from upscale.services.shadow.config import ExecutionConfig, ShadowSettings, load_settings
from upscale.services.shadow.engine import ShadowError
from upscale.services.shadow.report import compare_runs, compare_text
from upscale.services.shadow.service import BackgroundShadow
from upscale.services.shadow.store import ShadowStoreError

from .test_held_position_watch import SA, FakeDex, cand, cycle, ds_lookup, make, pair, recorder
from .test_shadow import T0, A, Harness, strat

__all__ = ["recorder"]  # the production-recorder fixture, shared with the watch tests

V2 = "EVIDENCE_AWARE_V2"
LONG = {"max_hold_minutes": 100_000.0}


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


def create(h: Harness, run_id: str, execution: ExecutionConfig | None = None, **exit_: Any) -> None:
    """Create a run without advancing it (as the CLI would, before deployment)."""
    h.store.register(strat(exit=exit_ or {}))
    h.engine.ensure_run(run_id, T0 - timedelta(hours=1), None, ["t"], availability_policy=V2,
                        execution=execution)  # fmt: skip


def rows(h: Harness, run_id: str) -> tuple[Any, ...]:
    def strip(xs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in x.items() if k not in ("recorded_at", "id")} for x in xs]

    return (
        strip(h.store.decisions(run_id=run_id, limit=100_000)),
        strip(h.store.trades(run_id=run_id)), strip(h.store.executions(run_id)),
        h.store.positions(run_id=run_id),
        [{k: v for k, v in e.items() if k != "id"} for e in h.store.equity(run_id)],
    )  # fmt: skip


def background(h: Harness, runs: str, at: datetime) -> BackgroundShadow:
    return BackgroundShadow(
        load_settings("1", runs=runs), lambda: None, shadow_db=lambda: str(h.shadow_path),
        evidence_db=lambda: str(h.ev_path), now=lambda: at,
    )  # fmt: skip


def evidence(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(5), 1.02)
    h.price(m(30), 1.25)
    h.price(m(40), 1.30)


# --- multi-run background Shadow ------------------------------------------------------------------


def test_legacy_single_run_settings_and_behavior_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for runs in (None, "", " , "):
        s = load_settings("1", None, "continuous-v2", None, runs)
        assert s.run_ids == () and s.run_id == "continuous-v2"
    assert load_settings("1", None, "x", None, None) == ShadowSettings(
        enabled=True, run_id="x")  # fmt: skip
    calls: list[Any] = []

    def fake_step(settings: Any, *_: Any) -> dict[str, Any]:
        calls.append(settings.run_id)
        return {"events": 1, "actions": {}, "fills": 0, "until": None}

    def no_advance(*_: Any, **__: Any) -> Any:
        raise AssertionError("single-run mode never uses the multi-run path")

    monkeypatch.setattr("upscale.services.shadow.service.step", fake_step)
    monkeypatch.setattr("upscale.services.shadow.service.advance", no_advance)
    svc = BackgroundShadow(load_settings("1", None, "continuous-v2"), lambda: None)
    result = asyncio.run(svc.run_once())
    assert result["status"] == "completed" and calls == ["continuous-v2"]
    st = svc.status()
    assert st["mode"] == "single-run" and st["run_id"] == "continuous-v2"
    assert st["runs_configured"] == ["continuous-v2"]
    assert st["runs"]["continuous-v2"]["advances"] == 1
    assert st["runs"]["continuous-v2"]["last_success_at"] is not None
    assert upscale.main.background_shadow.settings.run_ids == ()  # tests run without it


def test_runs_setting_is_parsed_in_order_without_duplicates() -> None:
    s = load_settings("1", runs="continuous-v2, continuous-v2-realistic,continuous-v2,Bad Id")
    assert s.run_ids == ("continuous-v2", "continuous-v2-realistic")
    assert s.ignored_run_ids == ("Bad Id",)


def test_two_runs_advance_independently(tmp_path: Path, template: Any) -> None:
    (tmp_path / "bg").mkdir()
    (tmp_path / "alone").mkdir()
    bg, alone = Harness(tmp_path / "bg", template), Harness(tmp_path / "alone", template)
    try:
        for hx in (bg, alone):
            evidence(hx)
            create(hx, "continuous-v2")
            create(hx, "continuous-v2-realistic", ExecutionConfig())
        svc = background(bg, "continuous-v2,continuous-v2-realistic", m(500))
        result = asyncio.run(svc.run_once())
        assert result["status"] == "completed"
        assert set(result["runs"]) == {"continuous-v2", "continuous-v2-realistic"}
        st = svc.status()
        assert st["mode"] == "multi-run" and st["run_id"] is None
        assert st["runs_configured"] == ["continuous-v2", "continuous-v2-realistic"]
        assert all(r["advances"] == 1 and r["failures"] == 0 for r in st["runs"].values())
        # Each run is exactly what advancing it alone gives (settle: 10 min before "now").
        alone.now = m(490)
        alone.engine.run("continuous-v2")
        alone.engine.run("continuous-v2-realistic")
        for run_id in ("continuous-v2", "continuous-v2-realistic"):
            assert rows(bg, run_id) == rows(alone, run_id), run_id
        assert bg.store.trades(run_id="continuous-v2")[0]["execution_model"] == "IDEALIZED_NO_FEES"
        assert bg.store.trades(run_id="continuous-v2-realistic")[0]["execution_model"] == (
            "REALISTIC_V1")  # fmt: skip
        # Frozen settings were not touched.
        r = bg.store.run("continuous-v2-realistic")
        assert r is not None and r["execution_model"] == "REALISTIC_V1"
        assert r["args"]["availability_policy"] == V2
    finally:
        bg.close()
        alone.close()


def test_one_failing_run_does_not_block_the_other(h: Harness) -> None:
    evidence(h)
    create(h, "continuous-v2")
    svc = background(h, "missing-run,continuous-v2", m(500))
    result = asyncio.run(svc.run_once())
    assert result["status"] == "partial"
    assert result["runs"]["missing-run"]["status"] == "failed"
    assert "does not exist" in result["runs"]["missing-run"]["error"]
    assert result["runs"]["continuous-v2"]["status"] == "completed"
    assert h.store.run("missing-run") is None  # multi-run mode never creates a run
    assert h.store.trades(run_id="continuous-v2")
    st = svc.status()["runs"]
    assert st["missing-run"]["failures"] == 1 and st["missing-run"]["last_failure_at"]
    assert st["missing-run"]["last_success_at"] is None
    assert st["continuous-v2"]["advances"] == 1 and st["continuous-v2"]["last_error"] is None
    assert svc.status()["runs_failed"] == 0 and svc.counts["partial"] == 1


def test_a_run_is_never_advanced_concurrently(h: Harness) -> None:
    evidence(h)
    create(h, "continuous-v2")
    h.now = m(10)
    # In this process: a run already being advanced is refused, not advanced twice.
    key = (str(Path(h.store.path).expanduser().resolve()), "continuous-v2")
    engine_module._ADVANCING.add(key)
    try:
        with pytest.raises(ShadowError, match="already being advanced"):
            h.engine.run("continuous-v2")
    finally:
        engine_module._ADVANCING.discard(key)
    stale = h.store.checkpoint("continuous-v2")["cursor"]
    h.engine.run("continuous-v2")
    # Across processes: a step that started from an older checkpoint writes nothing.
    before = (h.store.counts(), h.store.checkpoint("continuous-v2"))
    with pytest.raises(ShadowStoreError, match="advanced by another writer"):
        h.store.commit(
            "continuous-v2", decisions=[], opened=[], marked=[], closed=[], trades=[],
            equity=[], cursor=(m(40).timestamp(), 999), processed_until=m(40).timestamp(),
            books={}, stats={}, expected_cursor=(float(stale[0]), int(stale[1])),
        )  # fmt: skip
    assert (h.store.counts(), h.store.checkpoint("continuous-v2")) == before
    # The in-process guard is released after a step (also after a failing one).
    h.now = m(60)
    h.engine.run("continuous-v2")


def test_the_background_never_overlaps_its_own_cycles(h: Harness) -> None:
    create(h, "continuous-v2")
    svc = background(h, "continuous-v2", m(500))
    svc.running = True
    result = asyncio.run(svc.run_once())
    assert result["status"] == "deferred" and "already running" in result["defer_reason"]


# --- the held-position watch and pending entry intents ----------------------------------------------


def pending_entry(h: Harness, run_id: str = "r", **execution: Any) -> None:
    h.scan(T0, cand(h, SA))
    h.now = m(1)
    create(h, run_id, ExecutionConfig(**execution), **LONG)
    h.engine.run(run_id)


def test_pending_entry_is_a_watch_target(h: Harness) -> None:
    pending_entry(h)
    (pool,) = held_pools(str(h.shadow_path), m(2))
    assert (pool.chain, pool.token, pool.pool) == ("solana", SA, f"pool-{SA}")
    assert (pool.positions, pool.pending_entries, pool.runs) == (0, 1, ("r",))
    t = targets_summary([pool])
    assert t["pending_entry_pools"] == [{"chain": "solana", "token": SA, "pool": f"pool-{SA}",
                                         "pending_entries": 1, "open_positions": 0,
                                         "runs": ["r"]}]  # fmt: skip


def test_pending_entry_and_open_position_share_one_lookup(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    h.scan(T0, cand(h, SA))
    h.now = m(1)
    create(h, "t", None, **LONG)  # idealized: fills at once
    create(h, "r", ExecutionConfig(), **LONG)  # realistic: pending
    h.engine.run("t")
    h.engine.run("r")
    (pool,) = held_pools(str(h.shadow_path), m(2))
    assert (pool.positions, pool.pending_entries, pool.holders) == (1, 1, 2)
    assert pool.runs == ("r", "t")
    dex = FakeDex()
    dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", 1.05)]
    w = make(h, [ds_lookup(dex)])
    r = cycle(h, w, recorder, m(30))
    assert r.looked_up == 1 and len(dex.calls) == 1
    watched = [x for x in h.reader.records(kind="market") if x.component == "shadow_watch"]
    assert len(watched) == 1 and watched[0].observed_at == m(30)  # archived when observed
    st = w.status()["targets"]
    assert st["pools"] == 1 and st["open_position_pools"] == 1
    assert st["pending_entry_pools"][0]["pending_entries"] == 1


def test_pending_entry_fills_on_a_later_watch_observation_then_leaves_the_targets(
    h: Harness, recorder: EvidenceRecorder
) -> None:
    pending_entry(h, latency_seconds=3_600)  # eligible from 60 min after the decision
    dex = FakeDex()
    w = make(h, [ds_lookup(dex)])
    dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", 1.05)]
    cycle(h, w, recorder, m(30))  # observed before the minimum latency: never a fill
    h.now = m(31)
    h.engine.run("r")
    assert h.store.executions("r") == []
    (pool,) = held_pools(str(h.shadow_path), m(32))
    assert pool.pending_entries == 1
    dex.pairs[SA] = [pair("solana", SA, f"pool-{SA}", 1.07)]
    cycle(h, w, recorder, m(75))
    h.now = m(76)
    h.engine.run("r")
    (buy,) = h.store.executions("r")
    assert buy["price_source"] == "watch" and buy["observed_price"] == 1.07
    assert buy["price_observed_at"] == m(75).timestamp()
    assert buy["latency_seconds"] == 3_600 and buy["delay_seconds"] == 75 * 60  # observed
    (pool,) = held_pools(str(h.shadow_path), m(77))
    assert (pool.positions, pool.pending_entries) == (1, 0)  # watched as a position now
    assert targets_summary([pool])["pending_entry_pools"] == []


def test_cancelled_entry_stops_being_watched(h: Harness) -> None:
    pending_entry(h, entry_max_wait_minutes=30)
    h.price(m(120), 1.0, mint=A)  # time passes; nothing for SA's pool
    h.now = m(121)
    h.engine.run("r")
    assert held_pools(str(h.shadow_path), m(122)) == []
    st = h.store.checkpoint("r")["books"]["t@v1"]["state"]
    assert st["pending_entries"] == {} and st["cash"] == pytest.approx(10_000.0)


def test_watch_status_endpoint_reports_targets() -> None:
    st = upscale.main.held_position_watch.status()
    assert "targets" in st  # None until a cycle ran


# --- comparison semantics ----------------------------------------------------------------------------


def test_comparison_matched_unmatched_and_first_divergence(h: Harness) -> None:
    evidence(h)  # T0 entry; idealized TP at 30 min, realistic exit fill at 40 min
    h.scan(m(95), h.cand(A, price=1.0))  # idealized: cooldown over; realistic: 55 min in
    h.now = m(96)
    create(h, "t")
    create(h, "r", ExecutionConfig())
    h.engine.run("t")
    h.engine.run("r")
    d = compare_runs(h.store, h.reader, "t", "r")
    (s,) = d["strategies"]
    mt = s["matched"]
    assert (mt["enter_decisions"], mt["filled_in_both"], mt["completed_in_both"]) == (1, 1, 1)
    assert mt["net_pnl_usd"]["base"] == pytest.approx(250.0)
    assert mt["friction_usd"]["other"] > 0 and mt["friction_usd"]["base"] == 0
    assert mt["net_pnl_usd"]["delta"] == pytest.approx(
        mt["gross_pnl_usd"]["delta"] - mt["friction_usd"]["delta"])  # fmt: skip
    ob, oo = s["only_in_base"], s["only_in_other"]
    assert (ob["enter_decisions"], ob["filled"], ob["closed"]) == (1, 1, 0)
    assert ob["first"] == [{"asset_id": f"solana:{A}", "decision_at": m(95).isoformat()}]
    assert oo["enter_decisions"] == 0
    dv = s["divergence"]
    assert dv["first_entry_divergence"] == {"at": m(95).isoformat(), "action": "ENTER",
                                            "asset_id": f"solana:{A}", "only_in": "base"}  # fmt: skip
    assert dv["first_decision_divergence"]["at"] == m(95).isoformat()
    blocked = [x for x in h.store.decisions(run_id="r", action="NO_ACTION")]
    assert [x["reason"] for x in blocked] == ["qualified but blocked: COOLDOWN"]
    out = compare_text(d)
    assert "E first ENTER divergence: " + m(95).isoformat() in out
    assert "portfolio totals (include divergence)" in out


def test_no_divergence_when_decisions_match(h: Harness) -> None:
    evidence(h)
    h.now = m(41)
    create(h, "t")
    create(h, "r", ExecutionConfig())
    h.engine.run("t")
    h.engine.run("r")
    dv = compare_runs(h.store, h.reader, "t", "r")["strategies"][0]["divergence"]
    assert dv["first_entry_divergence"] is None and dv["first_decision_divergence"] is None
