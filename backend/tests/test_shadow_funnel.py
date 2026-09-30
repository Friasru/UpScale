"""Shadow sequential funnel, exact aggregate counters and bounded diagnostics storage:
the funnel attributes every rejected evaluation to ONE gate, counts come from counters
(never from sampled rows), detailed rows are sampled / expired only as configured, and no
strategy behavior changes in any mode."""

import json
import socket
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow.cli import main as cli_main
from upscale.services.shadow.config import DiagnosticsSettings, load_diagnostics_settings
from upscale.services.shadow.engine import ShadowEngine
from upscale.services.shadow.funnel import GATES, funnel, primary_reason, reasons_key
from upscale.services.shadow.store import ShadowStore
from upscale.services.shadow.strategies import BASELINES

from .test_shadow import T0, A, B, Harness, _scenario, strat
from .test_shadow_diagnostics import GOLDEN, technical, tweak

THRESHOLD, TECHNICAL, SAFETY_TECHNICAL, RANDOM = BASELINES
SAMPLED = DiagnosticsSettings(detail="sampled")


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[GrowthScoutResult, GrowthCandidate]:
    from .test_outcomes import ranked

    result, store = ranked(tmp_path_factory.mktemp("template"))
    store.close()
    return result, next(c for c in result.candidates if c.stage == "ACCELERATING")


def make(tmp_path: Path, template: Any, diagnostics: DiagnosticsSettings = SAMPLED) -> Harness:
    h = Harness(tmp_path, template)
    h.diagnostics = diagnostics
    h.reopen()
    return h


@pytest.fixture
def h(tmp_path: Path, template: Any) -> Any:
    harness = make(tmp_path, template)
    yield harness
    harness.close()


def rows_of(f: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    return {r["gate"]: (r["input"], r["passed"], r["failed"]) for r in f["funnel"]}


def by_strategy(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["strategy"].split("@")[0]: s for s in report["strategies"]}


def _gates_scan(h: Harness) -> None:
    """Nine candidates, each failing a known first gate for scout_technical."""
    h.scan(
        T0,
        h.cand(A),  # enters
        h.cand(B, technical=None),  # TECHNICAL_AVAILABLE
        h.cand("MintC", technical=technical(h, trend="down"), score=40.0),  # SCORE first
        h.cand("MintD", score=30.0),  # SCORE
        h.cand("MintE", stage="STEADY", score=30.0, liquidity=10_000.0),  # STAGE first
        h.cand("MintF", eligible=False),  # CURRENT_AND_ELIGIBLE
        h.cand("MintG", technical=technical(h, snapshots=2)),  # SNAPSHOTS
        h.cand("MintH", technical=technical(h, trend="flat")),  # TREND
        tweak(h.cand("MintI"), risk_penalty=25.0),  # RISK
    )


# --- sequential funnel ------------------------------------------------------------------------------


def test_exact_sequential_funnel(h: Harness) -> None:
    _gates_scan(h)
    h.run(*BASELINES)
    tech = by_strategy(h.engine.funnel("t"))["scout_technical"]
    assert (tech["evaluated"], tech["entered"], tech["rejected"], tech["blocked"]) == (9, 1, 8, 0)
    assert rows_of(tech) == {
        "TOTAL_EVALUATED": (9, 9, 0),
        "CURRENT_AND_ELIGIBLE": (9, 8, 1),
        "MARKET_NOT_COLLAPSED": (8, 8, 0),
        "STAGE_ALLOWED": (8, 7, 1),
        "SCORE_PASS": (7, 5, 2),
        "LIQUIDITY_PASS": (5, 5, 0),
        "RISK_PASS": (5, 4, 1),
        "NO_BLOCKING_FLAG": (4, 4, 0),
        "TECHNICAL_AVAILABLE": (4, 3, 1),
        "TECHNICAL_SNAPSHOTS_PASS": (3, 2, 1),
        "TECHNICAL_TREND_PASS": (2, 1, 1),
        "ENTRY_QUALIFIED": (1, 1, 0),
        "ENTERED": (1, 1, 0),
    }
    by_gate = {r["gate"]: r for r in tech["funnel"]}
    assert by_gate["SCORE_PASS"]["pct_of_previous"] == 71.4  # 5 / 7
    assert by_gate["SCORE_PASS"]["pct_of_evaluated"] == 55.6  # 5 / 9
    assert by_gate["CURRENT_AND_ELIGIBLE"]["failed_by_reason"] == {"SCOUT_NOT_ELIGIBLE": 1}
    # Safety evidence is missing for everyone: the stricter strategy's only survivor
    # (after the same Scout and Technical gates) drops out at SAFETY_AVAILABLE.
    safe = rows_of(by_strategy(h.engine.funnel("t"))["scout_safety_technical"])
    assert safe["TECHNICAL_TREND_PASS"] == (2, 1, 1)
    assert safe["SAFETY_AVAILABLE"] == (1, 0, 1) and safe["ENTERED"] == (0, 0, 0)


def test_multiple_reasons_never_double_count(h: Harness) -> None:
    _gates_scan(h)
    h.run(TECHNICAL)
    (tech,) = h.engine.funnel("t")["strategies"]
    failures = sum(r["failed"] for r in tech["funnel"] if r["gate"] != "ENTERED")
    assert failures == tech["rejected"] == 8
    assert sum(tech["reasons"].values()) > tech["rejected"]  # raw reasons overlap
    assert tech["reasons"]["SCORE_BELOW_MIN"] == 3  # C, D, E ...
    assert rows_of(tech)["SCORE_PASS"][2] == 2  # ... but E already failed STAGE_ALLOWED


def test_irrelevant_gates_are_omitted(h: Harness) -> None:
    _gates_scan(h)
    h.run(*BASELINES)
    f = by_strategy(h.engine.funnel("t"))
    gates = {k: [r["gate"] for r in v["funnel"]] for k, v in f.items()}
    order = ["TOTAL_EVALUATED", *[g.name for g in GATES], "ENTRY_QUALIFIED", "ENTERED"]
    for names in gates.values():
        assert names == sorted(names, key=order.index)  # always the documented order
    assert not any(g.startswith(("TECHNICAL", "SAFETY")) for g in gates["scout_threshold"])
    assert "RANDOM_SELECTION_PASS" not in gates["scout_threshold"]
    assert {"SAFETY_AVAILABLE", "SAFETY_LEVEL_PASS", "AUTHORITY_PASS",
            "HOLDER_CONCENTRATION_PASS"} <= set(gates["scout_safety_technical"])  # fmt: skip
    assert "TECHNICAL_OTHER_REQUIREMENTS_PASS" not in gates["scout_technical"]
    assert "RANDOM_SELECTION_PASS" in gates["random_eligible"]
    assert not {"SCORE_PASS", "RISK_PASS", "TECHNICAL_AVAILABLE"} & set(gates["random_eligible"])
    assert not any(g.startswith("SOCIAL") or g == "ANALYZE_PASS" for n in gates.values() for g in n)


def test_conditional_technical_coverage(h: Harness) -> None:
    _gates_scan(h)
    h.run(TECHNICAL)
    (tech,) = h.engine.funnel("t")["strategies"]
    first, cond = tech["conditional"]
    assert first["among"] == "all evaluated" and first["base"] == 9
    assert cond["among"] == "passing Scout" and cond["base"] == 4  # A, B, G, H
    checks = {c["gate"]: (c["passed"], c["of"]) for c in cond["checks"]}
    # Trend of C (down) is not counted here: C failed a Scout rule.
    assert checks == {
        "TECHNICAL_AVAILABLE": (3, 4),
        "TECHNICAL_SNAPSHOTS_PASS": (2, 3),  # given Technical evidence
        "TECHNICAL_TREND_PASS": (2, 3),
    }
    assert cond["all_group_rules"] == {"passed": 1, "of": 4, "pct": 25.0}


def test_conditional_safety_coverage(h: Harness) -> None:
    ok = "SAFETY_CHECKS_COMPLETE"
    h.scan(
        T0,
        h.cand("MintS1", safety=ok),
        tweak(h.cand("MintS2", safety=ok), mint_authority_active=True),
        tweak(h.cand("MintS3", safety=ok), holder_top10_pct=72.0),
        h.cand("MintS4"),  # no safety evidence
        h.cand("MintS5", safety="SAFETY_CHECKS_PARTIAL"),
        h.cand("MintS6", safety=ok, technical=None),  # fails Technical first
        h.cand("MintS7", safety=ok, score=10.0),  # fails Scout first
    )
    h.run(SAFETY_TECHNICAL)
    (s,) = h.engine.funnel("t")["strategies"]
    groups = {c["group"]: c for c in s["conditional"]}
    safety = groups["SAFETY"]
    assert safety["among"] == "passing Scout + Technical" and safety["base"] == 5
    assert {c["gate"]: (c["passed"], c["of"]) for c in safety["checks"]} == {
        "SAFETY_AVAILABLE": (4, 5),
        "SAFETY_LEVEL_PASS": (4, 4),  # partial or complete, given safety evidence
        "AUTHORITY_PASS": (4, 5),
        "HOLDER_CONCENTRATION_PASS": (4, 5),
    }
    assert safety["all_group_rules"]["passed"] == 2 == s["entered"]
    assert {c["gate"]: (c["passed"], c["of"]) for c in groups["TECHNICAL"]["checks"]}[
        "TECHNICAL_AVAILABLE"
    ] == (5, 6)


def test_funnel_math_on_outcome_counts() -> None:
    """Pure counting: blocked entries, the random gate, unknown future codes."""
    combos = [
        ("ENTERED", (), 3),
        ("BLOCKED", ("MAX_OPEN_POSITIONS",), 2),
        ("REJECTED", ("RANDOM_BASELINE_NOT_SELECTED",), 20),
        ("REJECTED", ("LIQUIDITY_BELOW_MIN", "STAGE_NOT_ALLOWED"), 5),
        ("REJECTED", ("SOME_FUTURE_RULE",), 1),
        ("HELD", (), 7),
    ]
    f = funnel(RANDOM, combos)
    assert (f["evaluated"], f["held"], f["blocked_by"]) == (31, 7, {"MAX_OPEN_POSITIONS": 2})
    r = rows_of(f)
    assert r["STAGE_ALLOWED"] == (31, 26, 5) and r["LIQUIDITY_PASS"] == (26, 26, 0)
    assert r["OTHER_RULES_PASS"] == (26, 25, 1)
    assert r["RANDOM_SELECTION_PASS"] == (25, 5, 20)
    assert r["ENTERED"] == (5, 3, 2)
    assert next(x for x in f["funnel"] if x["gate"] == "ENTERED")["failed_by_reason"] == {
        "MAX_OPEN_POSITIONS": 2
    }
    assert primary_reason(["SCORE_BELOW_MIN", "STAGE_NOT_ALLOWED", "X_NEW"]) == "STAGE_NOT_ALLOWED"
    assert reasons_key(["B_CODE", "A_CODE", "B_CODE"]) == "A_CODE,B_CODE"
    assert funnel(THRESHOLD, [])["funnel"][0] == {
        "gate": "TOTAL_EVALUATED", "input": 0, "passed": 0, "failed": 0,
        "pct_of_previous": None, "pct_of_evaluated": None,
    }  # fmt: skip


# --- aggregate counters and detail modes --------------------------------------------------------


def _busy(h: Harness) -> None:
    _scenario(h)
    _gates_scan_later(h, T0 + timedelta(minutes=200))


def _gates_scan_later(h: Harness, at: Any) -> None:
    h.scan(at, *[h.cand(f"Mint{i:03d}", score=30.0 + i) for i in range(40)],
           *[h.cand(f"Tech{i:03d}", technical=None) for i in range(10)])  # fmt: skip


@pytest.mark.parametrize("mode", ["sampled", "aggregate"])
def test_aggregate_counters_are_exact_in_every_mode(
    tmp_path: Path, template: Any, mode: str
) -> None:
    full = make(tmp_path / "full", template, DiagnosticsSettings(detail="full"))
    other = make(tmp_path / mode, template,
                 DiagnosticsSettings(detail=mode, sample_per_reason=1))  # type: ignore[arg-type]  # fmt: skip
    for x in (full, other):
        _busy(x)
        x.run(*BASELINES)
    a, b = full.engine.funnel("t"), other.engine.funnel("t")
    assert a["strategies"] == b["strategies"]
    assert full.engine.diagnostics("t")["strategies"] == other.engine.diagnostics("t")["strategies"]
    # In full mode the counters equal what the stored rows say, outcome by outcome.
    counted = sorted((sid, v, o, k, n) for sid, v, o, k, n in full.store.funnel_counts("t"))
    rebuilt = sorted((sid, v, o, reasons_key(r), n)
                     for sid, v, o, r, n in full.store.row_outcomes("t"))  # fmt: skip
    assert counted == rebuilt
    n_full = full.store.counts()["shadow_rejections"]
    n_other = other.store.counts()["shadow_rejections"]
    assert n_full == sum(s["rejected"] for s in a["strategies"])
    assert (n_other == 0) if mode == "aggregate" else (0 < n_other < n_full)
    for x in (full, other):
        x.close()


def test_sampled_detail_is_bounded_per_hour_and_day(tmp_path: Path, template: Any) -> None:
    h = make(tmp_path, template, DiagnosticsSettings(detail="sampled", sample_per_reason=3))
    assert h.diagnostics.hourly_quota == 1
    for k in range(5):  # five hours of one UTC day, 4 score failures each
        h.scan(T0 + timedelta(hours=k),
               *[h.cand(f"M{k}x{i}", score=10.0) for i in range(4)])  # fmt: skip
    h.scan(T0 + timedelta(hours=25), *[h.cand(f"N{i}", score=10.0) for i in range(4)])
    h.run(THRESHOLD)
    rows = h.store.rejections(run_id="t")
    hours = [(r["decision_at"] - T0.timestamp()) / 3600 for r in rows]
    assert hours == [0, 1, 2, 25]  # 1 per hour, 3 per day, a new day resets the quota
    (f,) = h.engine.funnel("t")["strategies"]
    assert f["rejected"] == 24  # counts stay exact
    q = h.store.checkpoint("t")["stats"]["rejection_sampling"]["scout_threshold@v1"]
    assert q["day"] == (T0 + timedelta(hours=25)).strftime("%Y-%m-%d")
    h.close()


def test_sampling_keys_on_the_primary_reason(tmp_path: Path, template: Any) -> None:
    h = make(tmp_path, template, DiagnosticsSettings(detail="sampled", sample_per_reason=48))
    # 2 per hour per primary reason; STAGE is the primary reason of the stage+score rows.
    h.scan(T0, *[h.cand(f"S{i}", score=10.0) for i in range(5)],
           *[h.cand(f"G{i}", score=10.0, stage="STEADY") for i in range(5)])  # fmt: skip
    h.run(THRESHOLD)
    rows = h.store.rejections(run_id="t")
    assert sorted(primary_reason(r["reasons"]) for r in rows) == [
        "SCORE_BELOW_MIN", "SCORE_BELOW_MIN", "STAGE_NOT_ALLOWED", "STAGE_NOT_ALLOWED",
    ]  # fmt: skip
    h.close()


def test_full_mode_stores_every_rejection(tmp_path: Path, template: Any) -> None:
    h = make(tmp_path, template, DiagnosticsSettings(detail="full", sample_per_reason=1))
    _gates_scan_later(h, T0)
    report = h.run(TECHNICAL)
    assert report["diagnostics_detail"] == "full"
    (f,) = h.engine.funnel("t")["strategies"]
    assert len(h.store.rejections(run_id="t", limit=10_000)) == 40 == report["rejections"]
    assert f["rejected"] == 40 and f["entered"] == 10  # scores 60..69 qualify
    h.close()


def test_sampling_is_deterministic(tmp_path: Path, template: Any) -> None:
    out = []
    for k in range(2):
        x = make(tmp_path / str(k), template, DiagnosticsSettings(sample_per_reason=2))
        _busy(x)
        x.run(*BASELINES)
        out.append([(r["rejection_id"], r["reasons"]) for r in x.store.rejections(run_id="t")])
        x.close()
    assert out[0] == out[1] and out[0]


def test_restart_resume_is_idempotent(tmp_path: Path, template: Any) -> None:
    settings = DiagnosticsSettings(sample_per_reason=2)
    single = make(tmp_path / "single", template, settings)
    staged = make(tmp_path / "staged", template, settings)
    for x in (single, staged):
        _busy(x)
    single.run(*BASELINES)
    staged.now = T0 + timedelta(minutes=50)
    staged.run(*BASELINES)
    for _ in range(2):  # a restart, then a restart with nothing new
        staged.reopen()
        staged.now = None
        staged.engine.run("t")

    def state(x: Harness) -> Any:
        rows = [(r["rejection_id"], r["reasons"], r["observed"])
                for r in x.store.rejections(run_id="t", limit=100_000)]  # fmt: skip
        cp = x.store.checkpoint("t")
        return (rows, sorted(x.store.funnel_counts("t")), x.engine.funnel("t")["strategies"],
                cp["books"], cp["stats"]["rejection_sampling"])  # fmt: skip

    assert state(staged) == state(single)
    for x in (single, staged):
        x.close()


# --- trading results unchanged ------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "sampled", "aggregate"])
def test_trading_results_identical_in_every_mode(tmp_path: Path, template: Any, mode: str) -> None:
    """The golden decisions / trades / positions the engine produced before diagnostics
    existed (commit 01bd2d9), in every diagnostics mode, with retention enabled."""
    result, g = template
    settings = DiagnosticsSettings(detail=mode, sample_per_reason=1, retention_days=0.01)  # type: ignore[arg-type]
    hh = make(tmp_path, template, settings)
    _scenario(hh)
    down = g.momentum.technical.model_copy(update={"trend": "flat"})
    hh.scan(T0 + timedelta(minutes=200), hh.cand("MintD", safety="SAFETY_CHECKS_COMPLETE"),
            hh.cand("MintE", technical=down), hh.cand("MintF", technical=None),
            hh.cand("MintG", score=40.0), *[hh.cand(f"Mint{i:03d}") for i in range(30)])  # fmt: skip
    hh.price(T0 + timedelta(minutes=230), 1.4, mint="MintD")
    hh.run(strat(), *BASELINES)
    got = {
        "decisions": [[x["decision_id"], x["strategy_id"], x["action"], x["asset_id"],
                       x["decision_at"], x["reference_price"], x["reason"]]
                      for x in hh.store.decisions(run_id="t", limit=100_000)],
        "trades": [[x["trade_id"], x["strategy_id"], x["exit_reason"], x["exit_price"],
                    x["exit_at"], x["pnl_usd"]] for x in hh.store.trades(run_id="t")],
        "positions": [[x["position_id"], x["status"], x["entry_price"], x["last_price"]]
                      for x in hh.store.positions(run_id="t")],
    }  # fmt: skip
    assert got == json.loads(GOLDEN.read_text())
    equity = [{k: v for k, v in e.items() if k != "id"} for e in hh.store.equity("t")]
    books = hh.store.checkpoint("t")["books"]
    hh.close()
    ref = make(tmp_path / "ref", template, DiagnosticsSettings(detail="full"))
    _scenario(ref)
    ref.scan(T0 + timedelta(minutes=200), ref.cand("MintD", safety="SAFETY_CHECKS_COMPLETE"),
             ref.cand("MintE", technical=down), ref.cand("MintF", technical=None),
             ref.cand("MintG", score=40.0), *[ref.cand(f"Mint{i:03d}") for i in range(30)])  # fmt: skip
    ref.price(T0 + timedelta(minutes=230), 1.4, mint="MintD")
    ref.run(strat(), *BASELINES)
    assert equity == [{k: v for k, v in e.items() if k != "id"} for e in ref.store.equity("t")]
    assert books == ref.store.checkpoint("t")["books"]
    ref.close()


# --- retention -------------------------------------------------------------------------------------


def _two_days(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, score=10.0), h.cand("MintC", technical=None))
    h.price(T0 + timedelta(minutes=10), 1.5, mint=A)  # take profit: a trade on day 1
    h.scan(T0 + timedelta(hours=50), h.cand("MintD", score=10.0), h.cand("MintE", price=2.0))


def _everything_but_rejections(h: Harness) -> Any:
    raw = sqlite3.connect(h.shadow_path)
    out = {t: raw.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
           for t in ("shadow_strategies", "shadow_runs", "shadow_decisions", "shadow_positions",
                     "shadow_trades", "shadow_equity", "shadow_funnel_counts")}  # fmt: skip
    for t in ("shadow_decisions", "shadow_trades", "shadow_strategies"):  # wall-clock last
        out[t] = [r[:-1] for r in out[t]]
    out["shadow_runs"] = [r[:1] + r[2:] for r in out["shadow_runs"]]
    raw.close()
    return out


def test_retention_only_expires_sampled_rejection_rows(tmp_path: Path, template: Any) -> None:
    kept = make(tmp_path / "kept", template, DiagnosticsSettings())
    pruned = make(tmp_path / "pruned", template, DiagnosticsSettings(retention_days=1))
    for x in (kept, pruned):
        _two_days(x)
        x.run(*BASELINES)
    assert "rejection_rows_expired" not in kept.engine.run("t")  # disabled by default
    assert kept.store.trades(run_id="t") and kept.store.positions(run_id="t")
    assert _everything_but_rejections(kept) == _everything_but_rejections(pruned)
    assert kept.engine.funnel("t")["strategies"] == pruned.engine.funnel("t")["strategies"]
    day2 = (T0 + timedelta(hours=50)).timestamp()
    old = [r for r in kept.store.rejections(run_id="t") if r["decision_at"] < day2]
    new = [r for r in kept.store.rejections(run_id="t") if r["decision_at"] >= day2]
    assert old and new
    assert [r["rejection_id"] for r in pruned.store.rejections(run_id="t")] == [
        r["rejection_id"] for r in new
    ]
    # Everything else is still refused, by the database itself.
    raw = sqlite3.connect(pruned.shadow_path)
    for sql in ("DELETE FROM shadow_rejections", "UPDATE shadow_rejections SET pool = 'x'",
                "DELETE FROM shadow_trades", "DELETE FROM shadow_positions",
                "DELETE FROM shadow_decisions", "DELETE FROM shadow_strategies",
                "DELETE FROM shadow_runs", "DELETE FROM shadow_equity",
                "DELETE FROM shadow_funnel_counts",
                "UPDATE shadow_funnel_counts SET count = count - 1"):  # fmt: skip
        with pytest.raises(sqlite3.DatabaseError):
            raw.execute(sql)
    raw.execute("UPDATE shadow_funnel_counts SET count = count + 0")  # growing is fine
    raw.close()
    for x in (kept, pruned):
        x.close()


def test_full_mode_is_never_pruned(tmp_path: Path, template: Any) -> None:
    h = make(tmp_path, template, DiagnosticsSettings(detail="full", retention_days=1))
    _two_days(h)
    report = h.run(*BASELINES)
    assert "rejection_rows_expired" not in report
    assert min(r["decision_at"] for r in h.store.rejections(run_id="t")) == T0.timestamp()
    h.close()


# --- existing runs (stored before aggregate counters) --------------------------------------------


def _as_v2(path: Path) -> None:
    """Make a file look like one written by the diagnostics version (schema v2)."""
    raw = sqlite3.connect(path)
    raw.executescript("""
        DROP TABLE shadow_funnel_counts;
        DROP TABLE shadow_retention_window;
        DROP TRIGGER shadow_rejections_retention_only;
        CREATE TRIGGER shadow_rejections_no_delete BEFORE DELETE ON shadow_rejections
            BEGIN SELECT RAISE(ABORT, 'shadow_rejections is append-only'); END;
        DELETE FROM shadow_meta WHERE key LIKE 'aggregates_from:%';
        UPDATE shadow_meta SET value = '2' WHERE key = 'schema_version';
        PRAGMA user_version = 2;
    """)
    raw.commit()
    raw.close()


def test_existing_full_rows_continue_exactly(tmp_path: Path, template: Any) -> None:
    ref = make(tmp_path / "ref", template, DiagnosticsSettings(detail="full"))
    old = make(tmp_path / "old", template, DiagnosticsSettings(detail="full"))
    for x in (ref, old):
        _busy(x)
    ref.run(*BASELINES)
    old.now = T0 + timedelta(minutes=100)
    old.run(*BASELINES)
    old.store.close()
    _as_v2(old.shadow_path)
    # Read-only inspection of the v2 file (the API; no migration): counted from its rows.
    ro = ShadowStore(old.shadow_path, read_only=True)
    early = ShadowEngine(ro, None).funnel("t")
    ro.close()
    assert early["aggregates_available_from"] is None
    assert by_strategy(early)["scout_technical"]["evaluated"] > 0
    before = [(r["rejection_id"], r["reasons"], r["observed"])
              for r in old.store.rejections(run_id="t", limit=10_000)]  # fmt: skip
    # The new version continues it (sampled detail, retention on): the pre-counter rows
    # are unchanged and protected, later evaluations are counted by the counters.
    old.diagnostics = DiagnosticsSettings(sample_per_reason=1, retention_days=0.01)
    old.reopen()
    old.now = None
    old.engine.run("t")
    raw = sqlite3.connect(old.shadow_path)
    triggers = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    raw.close()
    assert "shadow_rejections_no_delete" not in triggers
    assert "shadow_rejections_retention_only" in triggers
    after = [(r["rejection_id"], r["reasons"], r["observed"])
             for r in old.store.rejections(run_id="t", limit=10_000)]  # fmt: skip
    assert after[: len(before)] == before
    assert old.engine.funnel("t")["strategies"] == ref.engine.funnel("t")["strategies"]
    storage = {r["run_id"]: r for r in old.engine.storage()["runs"]}["t"]
    assert storage["rejection_rows_protected"] == len(before)
    for x in (ref, old):
        x.close()


# --- time windows and anti-lookahead ----------------------------------------------------------------


def test_funnel_windows_use_whole_hours(h: Harness) -> None:
    h.scan(T0, h.cand(A), h.cand(B, score=10.0))
    h.scan(T0 + timedelta(hours=2), h.cand("MintC", score=10.0))
    h.run(THRESHOLD)
    f = h.engine.funnel("t", since=T0 + timedelta(minutes=10), until=T0 + timedelta(hours=1))
    assert f["window"]["since"] == T0.isoformat() and "widened" in f["note"]
    (s,) = f["strategies"]
    assert (s["evaluated"], s["entered"]) == (2, 1)
    (late,) = h.engine.funnel("t", since=T0 + timedelta(hours=1))["strategies"]
    assert (late["evaluated"], late["rejected"]) == (1, 1)
    assert "widened" not in h.engine.funnel("t", since=T0 + timedelta(hours=1))["note"]


def test_anti_lookahead_counts_are_final(tmp_path: Path, template: Any) -> None:
    cut = T0 + timedelta(hours=1)
    prefix = make(tmp_path / "prefix", template)
    full = make(tmp_path / "full", template)
    for x in (prefix, full):
        _busy(x)
    prefix.run(*BASELINES, until=cut)
    full.run(*BASELINES)
    assert (
        prefix.engine.funnel("t")["strategies"] == full.engine.funnel("t", until=cut)["strategies"]
    )
    assert full.engine.funnel("t")["strategies"] != prefix.engine.funnel("t")["strategies"]
    for x in (prefix, full):
        x.close()


def test_no_provider_or_network_calls(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("funnel and storage must not use the network")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    _busy(h)
    h.run(*BASELINES)
    assert h.engine.funnel("t")["strategies"]
    assert h.engine.storage()["rows"]["shadow_funnel_counts"] > 0
    h.engine.diagnostics("t")


# --- storage, settings, CLI and API -------------------------------------------------------------------


def test_storage_report(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    _busy(h)
    h.run(*BASELINES)
    s = h.engine.storage()
    assert s["rows"] == {k: v for k, v in h.store.counts().items()} | {
        "shadow_checkpoints": 1
    }  # fmt: skip
    assert s["rejection_detail_rows"] == h.store.counts()["shadow_rejections"] > 0
    assert s["aggregate_counter_rows"] == h.store.counts()["shadow_funnel_counts"] > 0
    assert s["size_mb"] > 0 and s["volume"]["free_mb"] > 0
    g = s["estimated_growth"]
    assert g["tables"]["shadow_rejections"]["rows_per_day"] > 0
    assert g["rejection_detail_mb_per_day"] is not None
    assert s["settings"] == {"diagnostics_detail": "sampled", "sample_per_reason_per_day": 20,
                             "sample_per_reason_per_hour": 1, "retention_enabled": False,
                             "retention_days": None}  # fmt: skip
    assert s["runs"][0]["run_id"] == "t" and s["runs"][0]["rejection_rows_protected"] == 0
    base = ["--db", str(h.shadow_path), "--evidence-db", str(h.ev_path)]
    assert cli_main([*base, "storage"]) == 0
    text = capsys.readouterr().out
    assert "retention: disabled" in text and "aggregate counter rows:" in text
    assert cli_main([*base, "storage", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["rejection_detail_rows"] > 0


def test_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("DETAIL", "SAMPLE", "RETENTION"):
        monkeypatch.delenv(f"UPSCALE_SHADOW_{k}", raising=False)
    d = load_diagnostics_settings()
    assert (d.detail, d.sample_per_reason, d.retention_days) == ("sampled", 20, None)
    monkeypatch.setenv("UPSCALE_SHADOW_DIAGNOSTICS_DETAIL", "FULL")
    monkeypatch.setenv("UPSCALE_SHADOW_REJECTION_SAMPLE_PER_REASON", "100")
    monkeypatch.setenv("UPSCALE_SHADOW_REJECTION_RETENTION_DAYS", "7")
    d = load_diagnostics_settings()
    assert (d.detail, d.sample_per_reason, d.retention_days, d.hourly_quota) == (
        "full", 100, 7.0, 5,
    )  # fmt: skip
    assert load_diagnostics_settings("aggregate").detail == "aggregate"  # CLI override
    for bad in ("nonsense", "0", "-3", "nan"):
        monkeypatch.setenv("UPSCALE_SHADOW_DIAGNOSTICS_DETAIL", bad)
        monkeypatch.setenv("UPSCALE_SHADOW_REJECTION_SAMPLE_PER_REASON", bad)
        monkeypatch.setenv("UPSCALE_SHADOW_REJECTION_RETENTION_DAYS", bad)
        d = load_diagnostics_settings()
        assert (d.detail, d.sample_per_reason, d.retention_days) == ("sampled", 20, None)


def test_cli_funnel_and_run_detail(
    h: Harness, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates_scan(h)
    h.run(*BASELINES)
    base = ["--db", str(h.shadow_path), "--evidence-db", str(h.ev_path)]
    args = [*base, "funnel", "--run", "t", "--strategy", "scout_technical",
            "--strategy", "scout_safety_technical"]  # fmt: skip
    assert cli_main(args) == 0
    text = capsys.readouterr().out
    assert "scout_technical@v1" in text and "scout_safety_technical@v1" in text
    assert "scout_threshold" not in text
    assert "  TECHNICAL_TREND_PASS                     2       1       1    50.0%    11.1%" in text
    assert "  among candidates passing Scout (4):" in text
    assert "    TECHNICAL_AVAILABLE: 3/4 (75.0%)" in text
    assert cli_main([*args, "--json", "--since", T0.isoformat()]) == 0
    assert len(json.loads(capsys.readouterr().out)["strategies"]) == 2
    assert cli_main([*base, "funnel", "--run", "t", "--strategy", "nope"]) == 1
    capsys.readouterr()
    # `run --diagnostics-detail` overrides the environment for that step.
    import upscale.services.shadow.cli as cli

    h.scan(T0 + timedelta(hours=3), *[h.cand(f"Z{i}", score=10.0) for i in range(30)])
    h.writer.close()
    later = T0 + timedelta(hours=4)
    monkeypatch.setattr(cli, "ShadowEngine",
                        lambda *a, **kw: ShadowEngine(*a, now=lambda: later, **kw))  # fmt: skip
    before = h.store.counts()["shadow_rejections"]
    monkeypatch.setenv("UPSCALE_SHADOW_DIAGNOSTICS_DETAIL", "aggregate")
    assert cli_main([*base, "run", "--run", "t", "--since", (T0 - timedelta(hours=1)).isoformat(),
                     "--diagnostics-detail", "full"]) == 0  # fmt: skip
    report = json.loads(capsys.readouterr().out)
    assert report["diagnostics_detail"] == "full"
    assert report["rejection_rows_stored"] == report["rejections"] >= 3 * 30  # every one
    assert h.store.counts()["shadow_rejections"] == before + report["rejections"]


def test_api(h: Harness, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _gates_scan(h)
    h.run(*BASELINES, run_id="production")
    monkeypatch.setenv("UPSCALE_SHADOW_DB", str(h.shadow_path))
    f = client.get("/shadow/funnel?strategy=scout_technical").json()
    assert [s["strategy"] for s in f["strategies"]] == ["scout_technical@v1"]
    assert f["strategies"][0]["funnel"][0]["input"] == 9
    s = client.get("/shadow/storage").json()
    assert s["rows"]["shadow_funnel_counts"] > 0
