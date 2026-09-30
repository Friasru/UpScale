"""Shadow rejection diagnostics: every failed entry rule as a stable reason code, stored
append-only in ``shadow_rejections``, without changing any strategy behavior (the golden
snapshot below was produced by the engine before diagnostics existed)."""

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
from upscale.services.shadow.strategies import BASELINES

from .test_shadow import T0, A, B, Harness, _scenario, strat

GOLDEN = Path(__file__).parent / "data" / "shadow_golden_v1.json"
THRESHOLD, TECHNICAL, SAFETY_TECHNICAL, RANDOM = BASELINES


@pytest.fixture
def h(tmp_path: Path, template: tuple[GrowthScoutResult, GrowthCandidate]) -> Any:
    harness = Harness(tmp_path, template)
    yield harness
    harness.close()


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[GrowthScoutResult, GrowthCandidate]:
    from .test_outcomes import ranked

    result, store = ranked(tmp_path_factory.mktemp("template"))
    store.close()
    return result, next(c for c in result.candidates if c.stage == "ACCELERATING")


def tweak(g: GrowthCandidate, **kw: Any) -> GrowthCandidate:
    """Change quality / score fields of a candidate."""
    quality = {k: v for k, v in kw.items() if k in type(g.quality).model_fields}
    momentum = {k: v for k, v in kw.items() if k in type(g.scout_momentum).model_fields}
    return g.model_copy(update={
        "quality": g.quality.model_copy(update=quality),
        "scout_momentum": g.scout_momentum.model_copy(update=momentum),
    })  # fmt: skip


def technical(h: Harness, **kw: Any) -> Any:
    assert h.tpl.momentum.technical is not None
    return h.tpl.momentum.technical.model_copy(update=kw)


def reasons(h: Harness, strategy: str, mint: str = A, run: str = "t") -> list[str]:
    rows = h.store.rejections(run_id=run, strategy_id=strategy, asset_id=f"solana:{mint}")
    assert len(rows) == 1, rows
    return list(rows[0]["reasons"])


# --- reason codes ------------------------------------------------------------------------------


def test_technical_not_available(h: Harness) -> None:
    h.scan(T0, h.cand(technical=None))
    h.run(TECHNICAL)
    assert reasons(h, "scout_technical") == ["TECHNICAL_NOT_AVAILABLE"]
    (row,) = h.store.rejections(run_id="t")
    assert row["observed"]["technical"] is None
    assert "TECHNICAL_NOT_AVAILABLE" in row["observed"]["missing"]


def test_technical_trend_not_allowed(h: Harness) -> None:
    h.scan(
        T0,
        h.cand(technical=technical(h, trend="flat")),
        h.cand(B, technical=technical(h, trend="down")),
    )
    h.run(TECHNICAL)
    assert reasons(h, "scout_technical") == ["TECHNICAL_TREND_NOT_ALLOWED"]
    assert reasons(h, "scout_technical", B) == ["TECHNICAL_TREND_NOT_ALLOWED"]
    rows = h.store.rejections(run_id="t")
    assert [r["observed"]["technical"]["trend"] for r in rows] == ["flat", "down"]


def test_technical_too_few_snapshots(h: Harness) -> None:
    h.scan(T0, h.cand(technical=technical(h, snapshots=2)))
    h.run(TECHNICAL)
    assert reasons(h, "scout_technical") == ["TECHNICAL_TOO_FEW_SNAPSHOTS"]


def test_technical_optional_requirements(h: Harness) -> None:
    s = strat("strict", entry={
        "technical": {"allowed_trends": ["up"], "require_breakout": True,
                      "require_volume_confirmed": True, "require_higher_lows": True},
        "allowed_missing": ["SAFETY_NOT_AVAILABLE", "SOCIAL_NOT_AVAILABLE"],
    })  # fmt: skip
    weak = technical(h, breakout=False, volume_confirmed=None, higher_lows=False)
    h.scan(T0, h.cand(technical=weak))
    h.run(s)
    assert reasons(h, "strict") == [
        "TECHNICAL_BREAKOUT_REQUIRED", "TECHNICAL_VOLUME_NOT_CONFIRMED",
        "TECHNICAL_HIGHER_LOWS_REQUIRED",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"score": 50.0}, "SCORE_BELOW_MIN"),
        ({"stage": "STEADY"}, "STAGE_NOT_ALLOWED"),
        ({"liquidity": 20_000.0}, "LIQUIDITY_BELOW_MIN"),
        ({"eligible": False}, "SCOUT_NOT_ELIGIBLE"),
        ({"data_status": "STALE_CARRIED"}, "STALE_DATA"),
    ],
)
def test_scout_rule_rejections(h: Harness, change: dict[str, Any], code: str) -> None:
    h.scan(T0, h.cand(**change))
    h.run(THRESHOLD)
    assert code in reasons(h, "scout_threshold")


def test_risk_penalty_and_blocking_flag(h: Harness) -> None:
    g = tweak(h.cand(), risk_penalty=25.0)
    h.scan(T0, g)
    h.run(THRESHOLD, strat("flags", entry={"blocking_flag_severities": ["caution"]}))
    assert reasons(h, "scout_threshold") == ["RISK_PENALTY_TOO_HIGH"]
    assert reasons(h, "flags") == ["BLOCKING_RISK_FLAG"]
    (row,) = h.store.rejections(run_id="t", strategy_id="flags")
    assert row["observed"]["blocking_flags"] == ["safety_data_missing"]
    assert row["observed"]["risk_penalty"] == 25.0


def test_safety_not_available_is_not_also_insufficient(h: Harness) -> None:
    h.scan(T0, h.cand(safety="INSUFFICIENT_SAFETY_DATA"))
    h.run(SAFETY_TECHNICAL)
    assert reasons(h, "scout_safety_technical") == ["SAFETY_NOT_AVAILABLE"]


def test_safety_level_insufficient(h: Harness) -> None:
    complete = strat("complete", entry={"required_safety": "COMPLETE",
                                        "allowed_missing": ["SOCIAL_NOT_AVAILABLE"]})  # fmt: skip
    h.scan(T0, h.cand(safety="SAFETY_CHECKS_PARTIAL"))
    h.run(complete)
    assert reasons(h, "complete") == ["SAFETY_LEVEL_INSUFFICIENT"]


@pytest.mark.parametrize(
    ("change", "codes"),
    [
        ({"mint_authority_active": True}, ["MINT_AUTHORITY_ACTIVE"]),
        ({"freeze_authority_active": True}, ["FREEZE_AUTHORITY_ACTIVE"]),
        ({"mint_authority_active": True, "freeze_authority_active": True},
         ["MINT_AUTHORITY_ACTIVE", "FREEZE_AUTHORITY_ACTIVE"]),
        ({"holder_top10_pct": 72.0}, ["HOLDER_CONCENTRATION_TOO_HIGH"]),
    ],
)  # fmt: skip
def test_safety_detail_rejections(h: Harness, change: dict[str, Any], codes: list[str]) -> None:
    h.scan(T0, tweak(h.cand(safety="SAFETY_CHECKS_COMPLETE"), **change))
    h.run(SAFETY_TECHNICAL)
    assert reasons(h, "scout_safety_technical") == codes
    (row,) = h.store.rejections(run_id="t")
    for k, v in change.items():
        assert row["observed"][k] == v


def test_multiple_simultaneous_reasons(h: Harness) -> None:
    g = tweak(h.cand(score=40.0, technical=None, liquidity=10_000.0), risk_penalty=30.0)
    h.scan(T0, g)
    h.run(SAFETY_TECHNICAL)
    assert reasons(h, "scout_safety_technical") == [
        "SCORE_BELOW_MIN", "LIQUIDITY_BELOW_MIN", "RISK_PENALTY_TOO_HIGH",
        "SAFETY_NOT_AVAILABLE", "TECHNICAL_NOT_AVAILABLE",
    ]  # fmt: skip


def test_social_and_analyze_requirements(h: Harness) -> None:
    social = strat("social", entry={"social": {"allowed_statuses": ["SOCIAL_STRONG"]},
                                    "allowed_missing": []})  # fmt: skip
    analyze = strat("analyze", entry={"analyze": {"max_age_minutes": 30}})
    h.analyze(T0 + timedelta(minutes=1))  # after T: invisible at T
    h.scan(T0, h.cand())
    h.run(social, analyze)
    # Social unavailable: the missing label (the social rule itself needs evidence).
    assert "SOCIAL_NOT_AVAILABLE" in reasons(h, "social")
    assert reasons(h, "analyze") == ["ANALYZE_REQUIREMENT_FAILED"]
    (row,) = h.store.rejections(run_id="t", strategy_id="analyze")
    assert row["observed"]["analyze_failed"] == ["NOT_AVAILABLE"]


def test_analyze_failures_are_detailed(h: Harness) -> None:
    analyze = strat("analyze", entry={"analyze": {"max_age_minutes": 30, "min_confidence": "high"}})
    h.analyze(T0 - timedelta(minutes=5), action="wait", confidence="low")
    h.scan(T0, h.cand(score=10.0))  # also fails a Scout rule: both are reported
    h.run(analyze)
    assert reasons(h, "analyze") == ["SCORE_BELOW_MIN", "ANALYZE_REQUIREMENT_FAILED"]
    (row,) = h.store.rejections(run_id="t")
    assert row["observed"]["analyze_failed"] == ["ACTION", "CONFIDENCE"]
    assert row["observed"]["analyze"]["action"] == "wait"


# --- what is (not) a rejection -------------------------------------------------------------------


def test_entries_holds_and_blocked_candidates_are_not_rejections(h: Harness) -> None:
    s = strat(risk={"max_open_positions": 1})
    h.scan(T0, h.cand(A), h.cand(B))  # A enters, B qualifies but is blocked
    h.scan(T0 + timedelta(minutes=10), h.cand(A))  # held
    h.run(s)
    assert h.store.rejections(run_id="t") == []
    assert [d["action"] for d in h.decisions()] == ["ENTER", "NO_ACTION", "HOLD"]
    assert h.store.decisions(run_id="t", action="NO_ACTION")[0]["reason"] == (
        "qualified but blocked: MAX_OPEN_POSITIONS"
    )


def test_rejections_never_appear_as_decisions(h: Harness) -> None:
    h.scan(T0, h.cand(score=1.0), h.cand(B, stage="FADING"))
    h.run(THRESHOLD)
    assert h.decisions() == [] and len(h.store.rejections(run_id="t")) == 2


def test_random_baseline_rejection_is_deterministic(tmp_path: Path, template: Any) -> None:
    out = []
    for k in range(2):
        hh = Harness(tmp_path / str(k), template)
        hh.scan(T0, *[hh.cand(f"Mint{i:04d}") for i in range(40)], hh.cand(B, liquidity=1_000.0))
        hh.run(RANDOM)
        out.append([(r["rejection_id"], r["asset_id"], r["reasons"], r["observed"].get("random"))
                    for r in hh.store.rejections(run_id="t")])  # fmt: skip
        entered = len(hh.store.decisions(run_id="t", action="ENTER"))
        hh.close()
    assert out[0] == out[1]
    by_asset = {a: rs for _, a, rs, _ in out[0]}
    assert by_asset.pop(f"solana:{B}") == ["LIQUIDITY_BELOW_MIN"]  # never "not selected"
    assert all(rs == ["RANDOM_BASELINE_NOT_SELECTED"] for rs in by_asset.values())
    assert entered + len(by_asset) == 40
    draws = [d for *_, d in out[0] if d]
    assert all(d["draw"] >= d["fraction"] == 0.10 for d in draws)


# --- summary --------------------------------------------------------------------------------------


def _mixed(h: Harness) -> None:
    h.scan(
        T0,
        h.cand(A),  # everyone qualifies but C (safety) and maybe D
        h.cand(B, technical=None),
        h.cand("MintC", technical=technical(h, trend="down"), score=40.0),
        h.cand("MintD", score=30.0),
    )


def test_diagnostics_summary_counts(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    _mixed(h)
    h.run(*BASELINES)
    d = h.engine.diagnostics("t", ["scout_technical", "scout_safety_technical"])
    assert d["diagnostics_available_from"] == (T0 - timedelta(hours=1)).isoformat()
    tech, safe = d["strategies"]
    assert tech["strategy"] == "scout_technical@v1"
    assert (tech["evaluated"], tech["entered"], tech["rejected"]) == (4, 1, 3)
    assert tech["reasons"] == {
        "SCORE_BELOW_MIN": {"count": 2, "pct_of_rejected": 66.7},
        "TECHNICAL_NOT_AVAILABLE": {"count": 1, "pct_of_rejected": 33.3},
        "TECHNICAL_TREND_NOT_ALLOWED": {"count": 1, "pct_of_rejected": 33.3},
    }
    assert sum(v["count"] for v in tech["reasons"].values()) > tech["rejected"]
    assert (safe["evaluated"], safe["entered"], safe["rejected"]) == (4, 0, 4)
    assert safe["reasons"]["SAFETY_NOT_AVAILABLE"]["count"] == 4
    only_b = h.engine.diagnostics("t", ["scout_technical"], asset_id=f"solana:{B}")
    assert only_b["strategies"][0]["reasons"] == {
        "TECHNICAL_NOT_AVAILABLE": {"count": 1, "pct_of_rejected": 100.0}
    }
    # The CLI (text and JSON), against a copy of the database.
    base = ["--db", str(h.shadow_path), "--evidence-db", str(h.ev_path)]
    assert cli_main([*base, "diagnostics", "--run", "t", "--strategy", "scout_technical"]) == 0
    text = capsys.readouterr().out
    assert "scout_technical@v1:" in text and "  evaluated: 4" in text
    assert "    SCORE_BELOW_MIN: 2 (66.7%)" in text and "scout_threshold" not in text
    assert cli_main([*base, "diagnostics", "--run", "t", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["strategies"]) == 4
    assert cli_main([*base, "rejections", "--run", "t", "--reason", "TECHNICAL_NOT_AVAILABLE"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["strategy_id"] for r in rows} == {"scout_technical", "scout_safety_technical"}
    assert all(isinstance(r["fingerprints"], list) and r["fingerprints"][0]["kind"] == "scout"
               for r in rows)  # fmt: skip
    assert cli_main([*base, "diagnostics", "--run", "t", "--strategy", "nope"]) == 1


def test_strategy_isolation(h: Harness) -> None:
    _mixed(h)
    h.run(*BASELINES)
    rows = h.store.rejections(run_id="t")
    per = {(r["strategy_id"], r["asset_id"]) for r in rows}
    assert len(per) == len(rows)  # one row per strategy and evaluation
    assert ("scout_threshold", f"solana:{B}") not in per  # A has no Technical rule
    assert ("scout_technical", f"solana:{B}") in per
    assert {r["rejection_id"] for r in rows if r["strategy_id"] == "scout_technical"}.isdisjoint(
        {r["rejection_id"] for r in rows if r["strategy_id"] == "scout_safety_technical"}
    )


def test_api(h: Harness, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _mixed(h)
    h.run(*BASELINES, run_id="production")
    monkeypatch.setenv("UPSCALE_SHADOW_DB", str(h.shadow_path))
    d = client.get("/shadow/diagnostics?strategy=scout_technical").json()
    assert [s["strategy"] for s in d["strategies"]] == ["scout_technical@v1"]
    r = client.get("/shadow/rejections?reason=SCORE_BELOW_MIN").json()
    assert r and all("SCORE_BELOW_MIN" in x["reasons"] for x in r)


# --- guarantees -----------------------------------------------------------------------------------


def test_no_provider_or_network_calls(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("diagnostics must not use the network")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    _mixed(h)
    h.analyze(T0 - timedelta(minutes=5))
    h.run(*BASELINES, strat("an", entry={"analyze": {}}))
    h.engine.diagnostics("t")
    assert h.store.rejections(run_id="t")


def test_anti_lookahead_rejections_use_only_evidence_at_t(h: Harness) -> None:
    h.scan(T0, h.cand(score=40.0))
    h.scan(T0 + timedelta(minutes=30), h.cand(score=90.0, stage="FADING"))  # later evidence
    h.run(THRESHOLD, until=T0 + timedelta(minutes=5))
    (first,) = h.store.rejections(run_id="t")
    assert first["observed"]["score"] == 40.0 and first["reasons"] == ["SCORE_BELOW_MIN"]
    assert first["decision_at"] == T0.timestamp()
    json.dumps(first)  # nothing but plain values


def test_existing_decisions_and_trades_are_unchanged(tmp_path: Path, template: Any) -> None:
    """The exact decisions, trades and positions the engine produced before diagnostics
    existed (commit 01bd2d9), for all baselines and a test strategy."""
    result, g = template
    hh = Harness(tmp_path, template)
    _scenario(hh)
    down = g.momentum.technical.model_copy(update={"trend": "flat"})  # type: ignore[union-attr]
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
    assert hh.store.rejections(run_id="t", limit=100_000)  # diagnostics were added alongside
    hh.close()


def test_restart_resume_is_idempotent(tmp_path: Path, template: Any) -> None:
    def rows(x: Harness) -> list[Any]:
        return [(r["rejection_id"], r["strategy_id"], r["reasons"], r["observed"])
                for r in x.store.rejections(run_id="t", limit=100_000)]  # fmt: skip

    single = Harness(tmp_path / "single", template)
    _scenario(single)
    _mixed_late(single)
    single.run(*BASELINES)
    staged = Harness(tmp_path / "staged", template)
    _scenario(staged)
    _mixed_late(staged)
    staged.now = T0 + timedelta(minutes=50)
    staged.run(*BASELINES)
    staged.reopen()
    staged.now = None
    staged.engine.run("t")
    staged.reopen()
    staged.engine.run("t")  # nothing new
    assert rows(staged) == rows(single) and rows(single)
    assert len({r[0] for r in rows(staged)}) == len(rows(staged))
    raw = sqlite3.connect(staged.shadow_path)
    for sql in (
        "UPDATE shadow_rejections SET reasons_json = '[]'",
        "DELETE FROM shadow_rejections",
    ):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            raw.execute(sql)
    raw.close()
    for x in (single, staged):
        x.close()


def _mixed_late(h: Harness) -> None:
    h.scan(T0 + timedelta(minutes=45), h.cand("MintX", technical=None), h.cand("MintY", score=10.0))


# --- backward compatibility ------------------------------------------------------------------------


def test_runs_from_before_diagnostics_are_not_reconstructed(h: Harness) -> None:
    h.scan(T0, h.cand(score=40.0))
    h.run(THRESHOLD)
    # Make the file look like one written by the previous version (schema v1, no
    # rejection table, no diagnostics marker, the old first-reason counter).
    h.store.close()
    raw = sqlite3.connect(h.shadow_path)
    raw.executescript("""
        DROP TABLE shadow_rejections;
        DELETE FROM shadow_meta WHERE key LIKE 'diagnostics_from:%';
        UPDATE shadow_meta SET value = '1' WHERE key = 'schema_version';
        PRAGMA user_version = 1;
    """)
    cp = json.loads(raw.execute("SELECT books_json FROM shadow_checkpoints").fetchone()[0])
    counts = cp["scout_threshold@v1"]["state"]["counts"]
    counts.pop("rejected:SCORE_BELOW_MIN")
    counts["no_entry:SCORE"] = 1
    raw.execute("UPDATE shadow_checkpoints SET books_json = ?", (json.dumps(cp),))
    raw.commit()
    raw.close()
    # A read-only reader (the API) on the old file: reported as not available.
    from upscale.services.shadow.engine import ShadowEngine
    from upscale.services.shadow.store import ShadowStore

    ro = ShadowStore(h.shadow_path, read_only=True)
    old = ShadowEngine(ro, None).diagnostics("t")
    ro.close()
    assert old["diagnostics_available_from"] is None and "not reconstructed" in old["note"]
    (s,) = old["strategies"]
    assert s["rejected"] == 0 and s["before_diagnostics_first_reason_only"] == {"SCORE": 1}
    # The new version continues the run: diagnostics start at its cursor, the old
    # evaluation is not rebuilt, later ones are recorded.
    h.reopen()
    h.scan(T0 + timedelta(minutes=20), h.cand(B, score=45.0))
    h.engine.run("t")
    d = h.engine.diagnostics("t")
    assert d["diagnostics_available_from"] == (T0 + timedelta(minutes=1)).isoformat()
    assert [r["asset_id"] for r in h.store.rejections(run_id="t")] == [f"solana:{B}"]
    assert d["strategies"][0]["rejected"] == 1
    assert d["strategies"][0]["before_diagnostics_first_reason_only"] == {"SCORE": 1}
    assert h.store.counts()["shadow_rejections"] == 1
