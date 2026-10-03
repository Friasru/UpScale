"""Shadow strategy validation report (`report`): read-only, no network, deterministic,
accounting (open / closed, realized / unrealized), exits, drawdown, empty strategies and
insufficient samples, MARKET_UNAVAILABLE, availability-state aggregation, JSON and text,
run / strategy filters, and anti-lookahead. Temporary databases only; evidence is built
with the production payload serializers through the Shadow test harness.

Harness strategy (`strat()`): TP +20%, SL -10%, max hold 240 min, staleness rule 120 min,
exit delay 60 min, $1,000 per entry, $10,000 capital. Scout prices are observed one
minute before their decision time."""

import hashlib
import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow import report as report_module
from upscale.services.shadow.cli import main as cli_main
from upscale.services.shadow.engine import ShadowError
from upscale.services.shadow.report import COMPARED, shadow_report, text
from upscale.services.shadow.store import ShadowStore, ShadowStoreError

from .test_shadow import T0, A, B, Harness, strat
from .test_shadow_availability import watch

C = "MintCCCC3333"
V1, V2 = "LEGACY_V1", "EVIDENCE_AWARE_V2"


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


def report(h: Harness, run_id: str = "t", **kw: Any) -> dict[str, Any]:
    return shadow_report(h.store, h.reader, run_id, **kw)


def one(d: dict[str, Any], sid: str = "t") -> dict[str, Any]:
    return next(r for r in d["strategies"] if r["strategy_id"] == sid)


def evidence(h: Harness, upto: float = 60) -> None:
    """A: +25% take profit at 30 min; B: -15% stop loss at 40 min; C: entered at 50 min,
    open at +10% (evidence after `upto` minutes is left out)."""
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, price=2.0))
    h.price(m(30), 1.25)
    if upto >= 40:
        h.price(m(40), 1.7, mint=B)
    if upto >= 50:
        h.scan(m(50), h.cand(C, price=1.0))
    if upto >= 60:
        h.price(m(60), 1.1, mint=C)


def basic(h: Harness, *strategies: Any) -> None:
    evidence(h)
    h.now = m(61)
    h.run(*strategies)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- accounting -----------------------------------------------------------------------------------


def test_open_vs_closed_and_realized_vs_unrealized(h: Harness) -> None:
    basic(h)
    r = one(report(h))
    s, p = r["sample"], r["performance"]
    assert s["decisions_by_action"] == {"ENTER": 3, "HOLD": 0, "EXIT": 2, "NO_ACTION": 0}
    assert s["decisions"] == 5
    assert (s["positions_opened"], s["positions_closed"], s["currently_open"]) == (3, 2, 1)
    assert s["resolved_closed"] == 2 and s["unresolved_closed"] == 0
    assert s["unique_assets_entered"] == 3 and s["unique_assets_closed"] == 2
    assert s["effective_sample_size"] == 2.0
    assert p["account"]["source"] == "checkpoint"
    assert p["starting_capital_usd"] == 10_000
    assert p["cash_usd"] == pytest.approx(9_100.0)
    assert p["realized_pnl_usd"] == pytest.approx(100.0)  # +250 (A) - 150 (B)
    assert p["unrealized_pnl_usd"] == pytest.approx(100.0)  # C: 1,000 at +10%
    assert p["current_equity_usd"] == pytest.approx(10_200.0)
    assert p["total_return_pct"] == pytest.approx(2.0)
    assert p["closed_trade_pnl_usd"] == pytest.approx(100.0)
    assert p["median_return_pct"] == pytest.approx(5.0)
    assert p["mean_return_pct"] == pytest.approx(5.0)
    assert p["win_rate"] == 0.5
    assert p["profit_factor"] == pytest.approx(250 / 150)
    assert p["average_winner_pct"] == pytest.approx(25.0)
    assert p["median_loser_pct"] == pytest.approx(-15.0)
    assert p["best_trade"]["asset_id"] == f"solana:{A}" and p["best_trade"]["symbol"] == "TOK"
    assert p["worst_trade"]["return_pct"] == pytest.approx(-15.0)
    # The book's own checkpoint agrees.
    book = h.store.checkpoint("t")["books"]["t@v1"]["state"]
    assert p["cash_usd"] == pytest.approx(book["cash"])
    assert p["realized_pnl_usd"] == pytest.approx(book["realized_pnl"])


def test_exit_counts_holding_and_excursions(h: Harness) -> None:
    basic(h)
    r = one(report(h))
    b, x = r["trade_behavior"], r["excursion"]
    assert b["exit_reasons"] == {"TAKE_PROFIT": 1, "STOP_LOSS": 1, "TRAILING_STOP": 0,
                                 "MAX_HOLD_TIME": 0, "SIGNAL_EXIT": 0,
                                 "MARKET_UNAVAILABLE": 0}  # fmt: skip
    assert (b["take_profit"], b["stop_loss"], b["signal_exit"], b["max_hold"]) == (1, 1, 0, 0)
    assert b["market_unavailable"] == 0 and b["liquidity_collapse_exits"] == 0
    assert b["average_holding_minutes"] == pytest.approx(35.0)
    assert b["median_holding_minutes"] == pytest.approx(35.0)
    assert b["delayed_exit_fills"] == 0
    assert x["mfe_median_pct"] == pytest.approx(12.5)  # A +25, B 0
    assert x["mae_median_pct"] == pytest.approx(-7.5)  # A 0, B -15
    assert x["open_to_date"]["positions"] == 1
    assert x["open_to_date"]["mfe_median_pct"] == pytest.approx(10.0)


def test_liquidity_collapse_signal_exit(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.scan(m(30), h.cand(A, price=1.05, market_status="MARKET_COLLAPSE"))
    h.now = m(31)
    h.run()
    b = one(report(h))["trade_behavior"]
    assert b["signal_exit"] == 1 and b["liquidity_collapse_exits"] == 1


def test_drawdown_peak_and_minimum(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.scan(m(30), h.cand(A, price=0.95))
    h.scan(m(60), h.cand(A, price=1.05))
    h.now = m(61)
    h.run()
    k = one(report(h))["risk"]
    assert k["max_drawdown_pct"] == pytest.approx(-0.5)  # 10,000 -> 9,950
    assert k["last_mark_max_drawdown_pct"] == pytest.approx(-0.5)
    assert k["peak_equity_usd"] == pytest.approx(10_050.0)
    assert k["minimum_equity_usd"] == pytest.approx(9_950.0)
    assert k["max_simultaneous_positions"] == 1
    w = one(report(h, since=m(45)))["risk"]
    assert w["window_max_drawdown_pct"] == pytest.approx(0.0)  # from 9,950 up to 10,050


def test_max_simultaneous_positions(h: Harness) -> None:
    basic(h)
    assert one(report(h))["risk"]["max_simultaneous_positions"] == 2  # A and B; C after


# --- empty strategy / sample size / comparison ------------------------------------------------------


def test_empty_strategy(h: Harness) -> None:
    basic(h, strat(), strat("z", entry={"min_scout_score": 99.0}))
    d = report(h)
    z = one(d, "z")
    assert z["sample"]["positions_opened"] == 0 and z["sample"]["decisions"] == 0
    assert z["sample"]["status"] == "INSUFFICIENT_SAMPLE"
    assert z["performance"]["current_equity_usd"] == 10_000
    assert z["performance"]["total_return_pct"] == 0.0
    assert z["performance"]["win_rate"] is None and z["performance"]["profit_factor"] is None
    assert z["performance"]["best_trade"] is None
    assert z["risk"]["max_drawdown_pct"] == 0.0 and z["risk"]["max_simultaneous_positions"] == 0
    assert z["excursion"]["mfe_median_pct"] is None
    assert z["availability"]["held_position_hours"] == 0
    assert z["availability"]["fresh_price_time_pct"] is None
    assert "z@v1" in text(d)


def test_insufficient_sample_flagged(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    basic(h)
    s = one(report(h))["sample"]
    assert s["status"] == "INSUFFICIENT_SAMPLE"
    assert s["insufficient_reasons"] == ["2 resolved closed trades < 20",
                                         "2 distinct closed assets < 5"]  # fmt: skip
    d = report(h)
    assert d["comparison"]["rows"][0]["insufficient_sample"] is True
    assert "INSUFFICIENT SAMPLE" in text(d)
    monkeypatch.setattr(report_module, "MIN_TRADES", 2)
    monkeypatch.setattr(report_module, "MIN_ASSETS", 2)
    assert one(report(h))["sample"]["status"] == "DESCRIPTIVE"


def test_comparison_is_side_by_side_and_never_ranked(h: Harness) -> None:
    strategies = [strat("random_eligible"), strat("scout_threshold"),
                  strat("aaa", entry={"min_scout_score": 99.0})]  # fmt: skip
    basic(h, *strategies)
    c = report(h)["comparison"]
    assert [r["strategy"] for r in c["rows"]] == [
        "scout_threshold@v1", "random_eligible@v1", "aaa@v1"]  # fmt: skip
    assert c["missing_baselines"] == ["scout_technical", "scout_safety_technical"]
    assert "NOT ranked" in c["note"]
    keys = {"total_return_pct", "closed_trades", "win_rate", "profit_factor",
            "median_return_pct", "max_drawdown_pct", "mfe_median_pct", "mae_median_pct",
            "market_unavailable_exits", "insufficient_sample"}  # fmt: skip
    assert all(keys <= set(r) for r in c["rows"])
    fields = set(c) | {k for r in c["rows"] for k in r}
    assert not [k for k in fields if any(w in k for w in ("rank", "winner", "best", "score"))]
    assert set(COMPARED) == {"scout_threshold", "scout_technical", "scout_safety_technical",
                             "random_eligible"}  # fmt: skip


# --- MARKET_UNAVAILABLE and availability ------------------------------------------------------------


def test_market_unavailable_legacy_timeout(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(10 * 60), 1.0, mint=B)
    h.now = m(601)
    h.run(policy=V1)
    d = report(h)
    r = one(d)
    b, k, p = r["trade_behavior"], r["risk"], r["performance"]
    assert b["market_unavailable"] == 1
    assert b["market_unavailable_by_reason_code"] == {"MARKET_UNAVAILABLE": 1}
    assert r["sample"]["unresolved_closed"] == 1 and r["sample"]["resolved_closed"] == 0
    assert p["median_return_pct"] is None and p["win_rate"] is None  # never a loss or zero
    assert k["unresolved_cost_usd"] == pytest.approx(1_000.0)
    assert k["unresolved_last_mark_value_usd"] == pytest.approx(1_000.0)
    assert k["unresolved_capital_pct"] == pytest.approx(10.0)
    assert p["total_return_pct"] == pytest.approx(-10.0)  # written off
    assert p["last_mark_total_return_pct"] == pytest.approx(0.0)
    assert k["max_drawdown_pct"] == pytest.approx(-10.0)
    assert k["last_mark_max_drawdown_pct"] == pytest.approx(0.0)
    a = r["availability"]
    assert a["policy"] == V1 and a["market_unavailable_exits"] == 1
    assert a["evidence_gap_exits"]["exits_after_price_stale"] == 1
    g = d["data_integrity"]
    assert g["market_unavailable_exits_exist"] and g["market_unavailable_exits"] == 1
    assert g["evidence_gaps_affected_exits"] is True
    assert d["comparison"]["rows"][0]["market_unavailable_exits"] == 1


def test_market_not_found_under_evidence_aware_policy(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    watch(h, m(30), "NOT_FOUND")
    watch(h, m(100), "NOT_FOUND")
    h.price(m(200), 1.0, mint=B)
    h.now = m(201)
    h.run(policy=V2)
    r = one(report(h))
    assert r["trade_behavior"]["market_unavailable_by_reason_code"] == {"MARKET_NOT_FOUND": 1}
    a = r["availability"]
    assert a["watch_records"]["NOT_FOUND"] == 2
    assert a["positions_ever_in_state"]["MARKET_NOT_FOUND"] == 1
    assert a["held_hours_by_state"]["MARKET_AVAILABLE"] == pytest.approx(0.5)
    assert a["held_hours_by_state"]["MARKET_NOT_FOUND"] == pytest.approx(70 / 60)
    assert a["evidence_gap_exits"]["exits_after_evidence_gap"] == 1


def test_availability_state_aggregation(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))  # price observed at -1 min
    watch(h, m(200), "RATE_LIMITED", price=None)
    watch(h, m(400), "PRICED", price=1.0)
    h.price(m(590), 1.0, mint=B)
    h.now = m(600)
    h.run(strat(exit={"max_hold_minutes": 2000.0}), policy=V2)
    d = report(h)
    a = one(d)["availability"]
    hours = a["held_hours_by_state"]
    # -1..59 fresh, 59..119 stale, 119..200 gap, 200..400 provider failing, 400..460 fresh,
    # 460..520 stale, 520..600 gap (from the entry at 0).
    assert hours["MARKET_AVAILABLE"] == pytest.approx(119 / 60)
    assert hours["PRICE_STALE"] == pytest.approx(120 / 60)
    assert hours["EVIDENCE_GAP"] == pytest.approx(161 / 60)
    assert hours["PROVIDER_UNAVAILABLE"] == pytest.approx(200 / 60)
    assert hours["LIQUIDITY_COLLAPSE"] == 0 and hours["MARKET_NOT_FOUND"] == 0
    assert a["held_position_hours"] == pytest.approx(10.0)
    assert a["fresh_price_time_pct"] == pytest.approx(119 / 600 * 100)
    assert a["open_positions_by_state"]["EVIDENCE_GAP"] == 1
    assert a["positions_ever_in_state"] == {"MARKET_AVAILABLE": 1, "LIQUIDITY_COLLAPSE": 0,
                                            "PRICE_STALE": 1, "PROVIDER_UNAVAILABLE": 1,
                                            "EVIDENCE_GAP": 1, "MARKET_NOT_FOUND": 0}  # fmt: skip
    assert a["watch_records"]["RATE_LIMITED"] == 1 and a["watch_records"]["PRICED"] == 1
    assert d["run_watch_records"] == 2
    # The book's own label agrees at the processed time.
    (run,) = [x for x in h.engine.status()["runs"] if x["run_id"] == "t"]
    assert run["books"]["t@v1"]["open_market_states"] == {"EVIDENCE_GAP": 1}
    # The window clips the time accounting, not the rules.
    hw = one(report(h, since=m(300)))["availability"]["held_hours_by_state"]
    assert hw["PROVIDER_UNAVAILABLE"] == pytest.approx(100 / 60)
    assert hw["MARKET_AVAILABLE"] == pytest.approx(1.0)


def test_fresh_pricing_with_continuous_prices(h: Harness) -> None:
    basic(h)
    d = report(h)
    a = one(d)["availability"]
    assert a["fresh_price_time_pct"] == pytest.approx(100.0)
    assert a["open_positions_by_state"]["MARKET_AVAILABLE"] == 1
    g = d["data_integrity"]
    assert g["market_unavailable_exits_exist"] is False
    assert g["evidence_gaps_affected_exits"] is False


# --- filters ------------------------------------------------------------------------------------


def test_window_filters(h: Harness) -> None:
    basic(h)
    r = one(report(h, since=m(35)))
    s = r["sample"]
    assert (s["positions_opened"], s["positions_closed"], s["currently_open"]) == (1, 1, 1)
    assert s["decisions_by_action"] == {"ENTER": 1, "HOLD": 0, "EXIT": 1, "NO_ACTION": 0}
    assert r["performance"]["window_return_pct"] == pytest.approx(2.0)
    past = report(h, until=m(35))
    r = one(past)
    assert past["window"]["as_of"] == m(35).isoformat()
    assert past["window"]["account_source"] == "equity snapshots"
    assert (r["sample"]["positions_closed"], r["sample"]["currently_open"]) == (1, 1)
    assert r["performance"]["account"] == {"source": "equity snapshot", "at": T0.isoformat()}
    assert r["excursion"]["open_to_date"] is None  # current marks are not point in time
    with pytest.raises(ShadowError, match="after --since"):
        report(h, since=m(30), until=m(20))


def test_run_filtering(h: Harness) -> None:
    basic(h)
    h.engine.ensure_run("u", m(45), None, ["t"])
    h.engine.run("u")
    u = one(report(h, "u"))
    assert u["sample"]["positions_opened"] == 1 and u["sample"]["positions_closed"] == 0
    assert report(h, "u")["data_integrity"]["run_since"] == m(45).isoformat()
    assert one(report(h))["sample"]["positions_opened"] == 3
    with pytest.raises(ShadowError, match="unknown run"):
        report(h, "nope")


def test_strategy_filtering(h: Harness) -> None:
    basic(h, strat(), strat("z", entry={"min_scout_score": 99.0}))
    d = report(h, strategies=["z"])
    assert [r["strategy_id"] for r in d["strategies"]] == ["z"]
    assert [r["strategy"] for r in d["comparison"]["rows"]] == ["z@v1"]
    assert d["filters"]["strategies"] == ["z"]
    with pytest.raises(ShadowError, match="no strategy other"):
        report(h, strategies=["other"])


# --- anti-lookahead --------------------------------------------------------------------------------


def test_a_past_until_matches_a_run_that_stopped_there(tmp_path: Path, template: Any) -> None:
    """Reporting the full history up to 35 min equals reporting a run that only ever had
    the evidence up to 35 min: nothing later leaks into it."""
    (tmp_path / "full").mkdir()
    (tmp_path / "cut").mkdir()
    full, cut = Harness(tmp_path / "full", template), Harness(tmp_path / "cut", template)
    try:
        basic(full)
        evidence(cut, upto=35)
        cut.now = m(35)
        cut.run()
        a, b = one(report(full, until=m(35))), one(report(cut))
        for section in ("sample", "trade_behavior", "availability"):
            assert a[section] == b[section], section
        for key in ("median_return_pct", "win_rate", "profit_factor", "best_trade",
                    "worst_trade", "closed_trade_pnl_usd"):  # fmt: skip
            assert a["performance"][key] == b["performance"][key], key
        for key in ("mfe_median_pct", "mae_median_pct", "closed_trades"):
            assert a["excursion"][key] == b["excursion"][key], key
    finally:
        full.close()
        cut.close()


def test_evidence_after_the_processed_time_is_never_read(h: Harness) -> None:
    basic(h)
    before = report(h)
    h.price(m(70), 2.0, mint=C)  # archived, not processed yet: would be a take profit
    watch(h, m(75), "NOT_FOUND", mint=C)
    assert report(h) == before


def test_report_never_changes_the_book(h: Harness) -> None:
    basic(h)
    rows = (h.store.counts(), h.store.checkpoint("t"), h.trades(), h.positions(),
            h.decisions(limit=100_000), h.store.equity("t"))  # fmt: skip
    report(h)
    report(h, since=m(10), until=m(45))
    assert (h.store.counts(), h.store.checkpoint("t"), h.trades(), h.positions(),
            h.decisions(limit=100_000), h.store.equity("t")) == rows  # fmt: skip
    h.run()  # continuing the run is unaffected
    assert h.trades() == rows[2]


# --- CLI: read-only, no network, deterministic, JSON and text --------------------------------------


def _closed(h: Harness) -> tuple[Path, Path]:
    basic(h)
    h.close()
    return h.shadow_path, h.ev_path


def _cli(shadow: Path, evidence: Path, *extra: str) -> list[str]:
    return ["--db", str(shadow), "--evidence-db", str(evidence), "report", "--run", "t", *extra]


def test_read_only(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    shadow, evidence_db = _closed(h)
    before = {p: digest(p) for p in (shadow, evidence_db)}
    files = set(shadow.parent.iterdir())
    assert cli_main(_cli(shadow, evidence_db, "--json")) == 0
    assert cli_main(_cli(shadow, evidence_db)) == 0
    assert {p: digest(p) for p in (shadow, evidence_db)} == before
    new = set(shadow.parent.iterdir()) - files
    assert {p.name.rsplit("-", 1)[1] for p in new} <= {"wal", "shm"}
    assert all(p.stat().st_size == 0 for p in new if p.name.endswith("-wal"))
    store = ShadowStore(shadow, read_only=True)
    with pytest.raises(ShadowStoreError):
        store.register(next(iter(store.strategies())))
    store.close()
    capsys.readouterr()
    missing = shadow.parent / "nope.sqlite3"
    assert cli_main(_cli(missing, evidence_db)) == 1
    assert not missing.exists()
    # Without an Evidence Archive the report still runs; availability is unavailable.
    no_evidence = shadow.parent / "nope-evidence.sqlite3"
    assert cli_main(_cli(shadow, no_evidence, "--json")) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["strategies"][0]["availability"]["available"] is False
    assert d["data_integrity"]["evidence_gaps_affected_exits"] is None
    assert not no_evidence.exists()
    assert cli_main(_cli(shadow, no_evidence)) == 0
    assert "cannot be rebuilt" in capsys.readouterr().out


def test_no_provider_or_network_calls(
    h: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shadow, evidence_db = _closed(h)

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("the report must not use the network")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert cli_main(_cli(shadow, evidence_db, "--json")) == 0
    assert json.loads(capsys.readouterr().out)["provider_requests"] == 0
    assert cli_main(_cli(shadow, evidence_db)) == 0


def test_deterministic_output(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    shadow, evidence_db = _closed(h)
    outs = []
    for args in (["--json"], ["--json"], [], []):
        assert cli_main(_cli(shadow, evidence_db, *args)) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1] and outs[2] == outs[3]


def test_json_and_human_output(
    h: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shadow, evidence_db = _closed(h)
    monkeypatch.setenv("UPSCALE_SHADOW_DB", str(shadow))
    monkeypatch.setenv("UPSCALE_EVIDENCE_DB", str(evidence_db))
    assert cli_main(["report", "--run", "t", "--strategy", "t", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["read_only"] and d["run_id"] == "t"
    (r,) = d["strategies"]
    assert set(r) >= {"sample", "performance", "risk", "trade_behavior", "excursion",
                      "availability"}  # fmt: skip
    g = d["data_integrity"]
    assert g["availability_policy"] == V2 and g["execution_model"] == "IDEALIZED_NO_FEES"
    assert "NOT live-realistic" in g["idealized_warning"]
    assert g["clean_data"] is True and g["late_evidence_ignored"] == 0
    assert g["report_end"] == m(61).isoformat()
    assert cli_main(["report", "--run", "t", "--since", m(35).isoformat(),
                     "--until", m(61).isoformat()]) == 0  # fmt: skip
    out = capsys.readouterr().out
    for marker in ("SHADOW VALIDATION REPORT", "[sample]", "[performance]", "[risk]",
                   "[trade behavior]", "[excursion]", "[availability]",
                   "=== comparison (NOT ranked) ===", "=== data integrity ===",
                   "NOT live-realistic", "INSUFFICIENT SAMPLE"):  # fmt: skip
        assert marker in out, marker
    assert cli_main(["report", "--run", "t", "--strategy", "nope"]) == 1
    assert "no strategy nope" in capsys.readouterr().err
