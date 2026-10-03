"""Shadow REALISTIC_V1 execution: latency, fees, adverse slippage, optional price impact,
pending entry / exit intents (never a fabricated fill), exact pool identity, determinism,
restart / resume, gross vs net P/L, reporting and run comparison, IDEALIZED_NO_FEES
unchanged, anti-lookahead and no network. Temporary databases only; evidence is built with
the production payload serializers through the Shadow test harness.

Harness strategy (`strat()`): TP +20%, SL -10%, max hold 240 min, $1,000 per entry,
$10,000 capital. Default REALISTIC_V1: fees 30 / 30 bps, slippage 50 bps, minimum latency
60 s, no price impact, entry intents wait indefinitely. Scout prices
are observed one minute before their decision time."""

import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

from upscale.services.calibration.config import CalibrationConfig
from upscale.services.calibration.dataset import load_shadow
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow.book import Book, PendingEntry, latency_drift_usd
from upscale.services.shadow.cli import main as cli_main
from upscale.services.shadow.config import ExecutionConfig, run_execution
from upscale.services.shadow.engine import ShadowError
from upscale.services.shadow.evidence import PriceObs
from upscale.services.shadow.report import (
    compare_runs,
    compare_text,
    delay_stats,
    shadow_report,
    text,
)

from .test_shadow import T0, A, B, Harness, strat
from .test_shadow_availability import watch

V2 = "EVIDENCE_AWARE_V2"
EX = ExecutionConfig()


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


def realistic(
    h: Harness, *strategies: Any, run_id: str = "r", execution: ExecutionConfig = EX
) -> dict[str, Any]:
    for s in strategies or (strat(),):
        h.store.register(s)
    ids = [s.strategy_id for s in strategies or (strat(),)]
    h.engine.ensure_run(run_id, T0 - timedelta(hours=1), None, ids, availability_policy=V2,
                        execution=execution)  # fmt: skip
    return h.engine.run(run_id)


def state(h: Harness, run_id: str = "r") -> dict[str, Any]:
    st: dict[str, Any] = h.store.checkpoint(run_id)["books"]["t@v1"]["state"]
    return st


def execs(h: Harness, run_id: str = "r") -> list[dict[str, Any]]:
    return h.store.executions(run_id)


def trades(h: Harness, run_id: str = "r") -> list[dict[str, Any]]:
    return h.store.trades(run_id=run_id)


def roundtrip(h: Harness) -> None:
    """A: ENTER at 0 (Scout price 1.0 at -1 min), entry fill at 5 min (1.02), take profit
    triggered at 30 min (1.25), a price before the exit is eligible (1.4 at 30.5 min), the
    exit fill at 40 min (1.30)."""
    h.scan(T0, h.cand(A, price=1.0))
    h.price(T0 + timedelta(seconds=30), 1.01)  # before decision + latency: never a fill
    h.price(m(5), 1.02)
    h.price(m(30), 1.25)
    h.price(m(30.5), 1.4)
    h.price(m(40), 1.30)


def strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "recorded_at"} for r in rows]


def everything(h: Harness, run_id: str = "r") -> tuple[Any, ...]:
    return (
        strip(h.store.decisions(run_id=run_id, limit=100_000)), strip(trades(h, run_id)),
        strip(execs(h, run_id)), h.store.positions(run_id=run_id), h.store.equity(run_id),
        h.store.checkpoint(run_id)["books"],
    )  # fmt: skip


# --- entries -------------------------------------------------------------------------------------


def test_entry_waits_for_latency_and_never_fills_earlier(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    realistic(h)
    buy = next(x for x in execs(h) if x["side"] == "BUY")
    assert buy["intent_at"] == T0.timestamp()
    assert buy["eligible_at"] == (T0 + timedelta(seconds=60)).timestamp()
    assert buy["price_observed_at"] == m(5).timestamp()  # not the 1.01 at +30 s
    assert buy["observed_price"] == 1.02 and buy["reference_price"] == 1.0
    assert buy["delay_seconds"] == 300 and buy["latency_seconds"] == 60
    # The same spend at the 1.0 reference: quantity * 1.02 tokens, worth quantity * 1.02^2.
    assert buy["latency_cost_usd"] == pytest.approx(buy["quantity"] * 1.02 * 0.02)
    (enter,) = h.store.decisions(run_id="r", action="ENTER")
    assert enter["decision_at"] == T0.timestamp()  # the decision time never moves
    (p,) = h.store.positions(run_id="r")
    assert p["entry_at"] == m(5).timestamp() and p["entry_price_at"] == m(5).timestamp()
    assert p["entry_decision_id"] == enter["decision_id"]


def test_fill_at_exactly_decision_plus_latency(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(T0 + timedelta(seconds=60), 1.0)
    h.now = m(2)
    realistic(h)
    (buy,) = execs(h)
    assert buy["price_observed_at"] == buy["eligible_at"]


def test_entry_fee_and_adverse_buy_slippage(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    realistic(h)
    buy = next(x for x in execs(h) if x["side"] == "BUY")
    assert buy["fee_usd"] == pytest.approx(3.0)  # 30 bps of $1,000
    assert buy["notional_usd"] == pytest.approx(997.0)
    assert buy["execution_price"] == pytest.approx(1.02 * 1.005)  # above the observation
    assert buy["quantity"] == pytest.approx(997.0 / (1.02 * 1.005))
    assert buy["slippage_cost_usd"] == pytest.approx(buy["quantity"] * 1.02 * 0.005)
    assert buy["cash_flow_usd"] == -1000.0
    assert buy["friction_usd"] == pytest.approx(1000.0 - buy["quantity"] * 1.02)
    assert buy["impact_status"] == "DISABLED" and buy["impact_cost_usd"] == 0
    (p,) = h.store.positions(run_id="r")
    assert p["cost_usd"] == 1000.0
    assert p["entry_price"] == pytest.approx(1000.0 / buy["quantity"])  # all-in cost / unit


def test_pending_entry_reserves_cash_then_is_cancelled_without_a_fill(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(30), 1.0, mint=B)
    h.now = m(31)
    realistic(h, execution=ExecutionConfig(entry_max_wait_minutes=120))
    st = state(h)
    assert len(st["pending_entries"]) == 1 and st["positions"] == {}
    assert st["cash"] == pytest.approx(9_000.0)
    (run,) = [r for r in h.engine.status()["runs"] if r["run_id"] == "r"]
    assert run["books"]["t@v1"]["equity"] == pytest.approx(10_000.0)  # reserved, at cost
    # A second Scout pass while the order is pending holds the asset slot.
    h.scan(m(40), h.cand(A, price=1.0, pool="other-pool"))
    h.price(m(200), 1.0, mint=B)
    h.now = m(201)
    h.engine.run("r")
    assert [d["reason"] for d in h.store.decisions(run_id="r", action="HOLD")] == [
        "entry order pending (one per asset)"]  # fmt: skip
    (cancel,) = [d for d in h.store.decisions(run_id="r", action="NO_ACTION")
                 if d["reason"].startswith("entry order cancelled")]  # fmt: skip
    assert cancel["decision_at"] == m(121).timestamp()  # eligible + 120 min
    st = state(h)
    assert st["pending_entries"] == {} and st["cash"] == pytest.approx(10_000.0)
    assert st["realized_pnl"] == 0 and st["unresolved_cost"] == 0  # no P/L
    assert execs(h) == [] and trades(h) == [] and h.store.positions(run_id="r") == []
    (enter,) = h.store.decisions(run_id="r", action="ENTER")
    assert json.loads(cancel["evidence_json"]) == {
        "reason_code": "ENTRY_NOT_FILLED", "entry_decision_id": enter["decision_id"],
        "released_usd": 1000.0, "not_a_trade": True}  # fmt: skip
    d = shadow_report(h.store, h.reader, "r")
    r = d["strategies"][0]
    e = r["execution"]
    assert e["cancelled_entry_intents"] == 1 and e["cancelled_entry_released_usd"] == 1000.0
    assert e["fills"] == {"BUY": 0, "SELL": 0} and e["pending"]["entry_intents"] == 0
    assert r["sample"]["positions_opened"] == 0 and r["performance"]["realized_pnl_usd"] == 0
    assert r["performance"]["current_equity_usd"] == pytest.approx(10_000.0)
    assert "cancelled entry orders 1 (not trades, no P/L; $1,000.00 released)" in text(d)


def test_default_entry_intent_waits_indefinitely(h: Harness) -> None:
    assert EX.entry_max_wait_minutes is None
    assert ExecutionConfig(entry_max_wait_minutes=0) == EX  # 0 means indefinitely
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(60 * 24 * 7), 1.0, mint=B)  # a week without a price of A's pool
    h.now = m(60 * 24 * 7 + 1)
    realistic(h)
    st = state(h)
    assert len(st["pending_entries"]) == 1 and execs(h) == []
    assert st["cash"] == pytest.approx(9_000.0)
    assert not [d for d in h.store.decisions(run_id="r", action="NO_ACTION")]
    h.price(m(60 * 24 * 7 + 5), 1.1)  # the exact pool, finally
    h.now = m(60 * 24 * 7 + 6)
    h.engine.run("r")
    (buy,) = execs(h)
    assert buy["observed_price"] == 1.1 and buy["delay_seconds"] == (60 * 24 * 7 + 5) * 60


# --- exits ---------------------------------------------------------------------------------------


def test_exit_waits_for_latency_with_exit_fee_and_adverse_sell_slippage(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    realistic(h)
    (tp,) = [d for d in h.store.decisions(run_id="r", action="EXIT")]
    assert tp["decision_at"] == m(30).timestamp() and "TAKE_PROFIT" in tp["reason"]
    (t,) = trades(h)
    sell = next(x for x in execs(h) if x["side"] == "SELL")
    assert t["exit_decision_id"] == tp["decision_id"] == sell["decision_id"]
    assert sell["eligible_at"] == m(31).timestamp()
    assert sell["price_observed_at"] == m(40).timestamp()  # not the 1.4 at 30.5 min
    assert t["exit_at"] == m(40).timestamp() and t["exit_reason"] == "TAKE_PROFIT"
    assert sell["execution_price"] == pytest.approx(1.30 * 0.995)  # below the observation
    gross_sale = sell["quantity"] * 1.30 * 0.995
    assert sell["fee_usd"] == pytest.approx(gross_sale * 0.003)
    assert sell["cash_flow_usd"] == pytest.approx(gross_sale * 0.997)
    assert t["proceeds_usd"] == pytest.approx(sell["cash_flow_usd"])
    assert t["exit_price"] == pytest.approx(sell["execution_price"])
    assert t["execution_model"] == "REALISTIC_V1"
    assert sell["delay_seconds"] == 600
    assert sell["latency_cost_usd"] == pytest.approx(sell["quantity"] * (1.25 - 1.30))


def test_gross_vs_net_pnl(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    realistic(h)
    buy, sell = execs(h)
    (t,) = trades(h)
    qty = buy["quantity"]
    assert sell["gross_pnl_usd"] == pytest.approx(qty * (1.30 - 1.02))
    assert t["pnl_usd"] == pytest.approx(sell["cash_flow_usd"] - 1000.0)
    assert sell["net_pnl_usd"] == pytest.approx(t["pnl_usd"])
    assert sell["gross_pnl_usd"] - sell["trade_friction_usd"] == pytest.approx(t["pnl_usd"])
    assert sell["trade_friction_usd"] == pytest.approx(buy["friction_usd"] + sell["friction_usd"])
    assert t["return_pct"] == pytest.approx(t["pnl_usd"] / 10)  # net, on the $1,000 cost
    st = state(h)
    assert st["realized_pnl"] == pytest.approx(t["pnl_usd"])
    assert st["cash"] == pytest.approx(10_000.0 + t["pnl_usd"])


def test_pending_exit_never_fabricates_a_fill(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(5), 1.0)
    h.price(m(30), 0.85)  # stop loss triggered; no later price of the exact pool
    h.price(m(10 * 60), 1.0, mint=B)
    h.now = m(601)
    realistic(h)
    assert trades(h) == []  # never a fill, never MARKET_UNAVAILABLE for missing evidence
    (p,) = h.store.positions(run_id="r")
    assert p["status"] == "OPEN" and p["pending_exit"] == "STOP_LOSS"
    assert [x["side"] for x in execs(h)] == ["BUY"]
    pending = shadow_report(h.store, h.reader, "r")["strategies"][0]["execution"]["pending"]
    assert pending["exit_intents"] == 1 and pending["exit_intents_by_reason"] == {"STOP_LOSS": 1}
    h.price(m(700), 0.8)
    h.now = m(701)
    h.engine.run("r")
    (t,) = trades(h)
    assert t["exit_reason"] == "STOP_LOSS" and t["exit_price_at"] == m(700).timestamp()


def test_signal_exit_is_an_intent(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(5), 1.0)
    h.scan(m(10), h.cand(A, price=1.05, stage="FADING"))  # its own price is 1 min earlier
    h.price(m(10.5), 1.06)  # before the exit is eligible (11 min)
    h.price(m(12), 1.07)
    h.now = m(13)
    realistic(h)
    (t,) = trades(h)
    assert t["exit_reason"] == "SIGNAL_EXIT" and t["exit_price_at"] == m(12).timestamp()


# --- price impact -------------------------------------------------------------------------------


def test_price_impact_from_observed_liquidity(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    watch(h, m(5), "PRICED", price=1.02, liquidity=80_000.0)
    h.now = m(6)
    realistic(h, execution=ExecutionConfig(price_impact=True))
    (buy,) = execs(h)
    impact = 997.0 / 40_000.0  # notional / quote reserve (half the pool liquidity)
    assert buy["impact_status"] == "APPLIED" and buy["liquidity_usd"] == 80_000.0
    assert buy["impact_bps"] == pytest.approx(impact * 1e4)
    assert buy["execution_price"] == pytest.approx(1.02 * (1 + 0.005 + impact))
    assert buy["impact_cost_usd"] == pytest.approx(buy["quantity"] * 1.02 * impact)
    assert buy["price_source"] == "watch"


def test_price_impact_is_capped(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    watch(h, m(5), "PRICED", price=1.02, liquidity=2_000.0)
    h.now = m(6)
    realistic(h, execution=ExecutionConfig(price_impact=True))
    (buy,) = execs(h)
    assert buy["impact_status"] == "CAPPED" and buy["impact_bps"] == pytest.approx(1_000.0)


def test_no_price_impact_without_observed_liquidity(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(5), 1.02)  # a price record without liquidity
    h.now = m(6)
    realistic(h, execution=ExecutionConfig(price_impact=True))
    (buy,) = execs(h)
    assert buy["impact_status"] == "LIQUIDITY_UNAVAILABLE" and buy["liquidity_usd"] is None
    assert buy["impact_bps"] == 0 and buy["impact_cost_usd"] == 0
    assert buy["execution_price"] == pytest.approx(1.02 * 1.005)


# --- identity -----------------------------------------------------------------------------------


def test_exact_pool_identity_and_solana_case_sensitivity(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))
    h.price(m(5), 1.5, pool="other-pool")  # the same token on another pool
    h.price(m(6), 1.5, pool=f"POOL-{A}")  # Solana addresses are case-sensitive
    h.price(m(7), 1.03)
    h.now = m(8)
    realistic(h)
    (buy,) = execs(h)
    assert buy["observed_price"] == 1.03 and buy["pool"] == f"pool-{A}"


def _book() -> Book:
    return Book("u", strat(), None, V2, EX)


def _pending(book: Book, asset: str, pool: str) -> None:
    book.pending_entries["d"] = PendingEntry(
        decision_id="d", asset_id=asset, chain=asset.split(":")[0], address=asset.split(":")[1],
        symbol="TOK", pool=pool, dex=None, decided_at=T0, eligible_at=m(1), expires_at=None,
        budget_usd=1_000.0, reference_price=1.0, reference_price_at=T0,
    )  # fmt: skip
    book.cash -= 1_000.0


def test_evm_pool_identity_is_case_insensitive() -> None:
    book = _book()
    _pending(book, "base:0xtoken", "0xPoolAbCdEf")
    out = book.on_price(PriceObs("base:0xtoken", "0xpoolabcdef", 2.0, m(5), "market", "x"), m(5))
    assert len(out.opened) == 1 and len(out.executions) == 1
    (p,) = book.positions.values()
    out = book.on_price(PriceObs("base:0xtoken", "0XPOOLABCDEF", 2.5, m(6), "market", "y"), m(6))
    assert p.last_price == 2.5 and p.pending_exit == "TAKE_PROFIT"  # marked through the case


def test_solana_pool_identity_is_case_sensitive() -> None:
    book = _book()
    _pending(book, f"solana:{A}", "PoolAbCdEf")
    out = book.on_price(PriceObs(f"solana:{A}", "poolabcdef", 2.0, m(5), "market", "x"), m(5))
    assert out.opened == [] and book.pending_entries
    out = book.on_price(PriceObs(f"solana:{A}", "PoolAbCdEf", 2.0, m(6), "market", "y"), m(6))
    assert len(out.opened) == 1


def test_realistic_requires_the_evidence_aware_policy(h: Harness) -> None:
    with pytest.raises(ValueError, match="EVIDENCE_AWARE_V2"):
        Book("u", strat(), None, "LEGACY_V1", EX)
    h.store.register(strat())
    with pytest.raises(ShadowError, match="EVIDENCE_AWARE_V2"):
        h.engine.ensure_run("r", T0, None, ["t"], availability_policy="LEGACY_V1", execution=EX)


# --- reproducibility --------------------------------------------------------------------------------


def test_deterministic_replay(tmp_path: Path, template: Any) -> None:
    out = []
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        hx = Harness(tmp_path / name, template)
        try:
            roundtrip(hx)
            hx.now = m(41)
            realistic(hx, execution=ExecutionConfig(price_impact=True))
            out.append(everything(hx))
        finally:
            hx.close()
    assert out[0] == out[1]
    assert json.dumps(out[0], sort_keys=True, default=str) == json.dumps(
        out[1], sort_keys=True, default=str)  # fmt: skip


def test_restart_resume_matches_one_pass(tmp_path: Path, template: Any) -> None:
    (tmp_path / "once").mkdir()
    (tmp_path / "steps").mkdir()
    once, steps = Harness(tmp_path / "once", template), Harness(tmp_path / "steps", template)
    try:
        roundtrip(once)
        once.now = m(41)
        realistic(once)
        roundtrip(steps)
        for minute in (0.5, 3, 20, 30.2, 35):  # pending entry, open, pending exit ...
            steps.now = m(minute)
            realistic(steps)
            steps.reopen()
        steps.now = m(41)
        steps.engine.run("r")
        a, b = everything(once), everything(steps)
        assert a[:4] == b[:4] and a[5] == b[5]  # equity rows are written per step
    finally:
        once.close()
        steps.close()


# --- IDEALIZED_NO_FEES unchanged ------------------------------------------------------------------


def test_idealized_runs_are_unchanged_by_a_realistic_run(tmp_path: Path, template: Any) -> None:
    rows = []
    for name, with_realistic in (("alone", False), ("both", True)):
        (tmp_path / name).mkdir()
        hx = Harness(tmp_path / name, template)
        try:
            roundtrip(hx)
            hx.now = m(41)
            hx.run()
            if with_realistic:
                realistic(hx)
            rows.append(everything(hx, "t"))
            st = hx.store.checkpoint("t")["books"]["t@v1"]["state"]
            assert "pending_entries" not in st
            assert hx.store.executions("t") == []
            assert {t["execution_model"] for t in hx.store.trades(run_id="t")} == {
                "IDEALIZED_NO_FEES"}  # fmt: skip
            run = hx.store.run("t")
            assert run is not None and run_execution(run) is None
            assert run["execution_model"] == "IDEALIZED_NO_FEES"
        finally:
            hx.close()
    assert rows[0] == rows[1]
    (t,) = rows[0][1]
    assert t["exit_price"] == 1.25 and t["exit_at"] == m(30).timestamp()  # fills at once


def test_calibration_reads_idealized_trades_only(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    h.run()
    realistic(h)
    assert trades(h) and h.store.trades(run_id="t")
    visible = load_shadow(h.shadow_path, CalibrationConfig().live_split, include_holdout=True)
    assert {o.run_id for o in visible} == {"t"}


# --- anti-lookahead ------------------------------------------------------------------------------


def test_the_decisions_own_price_is_never_a_fill(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0))  # its price was observed a minute before the decision
    h.price(m(5), 1.0, mint=B)
    h.now = m(6)
    realistic(h, execution=ExecutionConfig(latency_seconds=0))
    assert execs(h) == [] and len(state(h)["pending_entries"]) == 1


def test_evidence_after_the_processing_limit_is_never_used(h: Harness) -> None:
    roundtrip(h)
    h.now = m(3)
    realistic(h)
    assert execs(h) == [] and len(state(h)["pending_entries"]) == 1
    h.now = m(41)
    h.engine.run("r")
    for x in execs(h):
        assert x["eligible_at"] <= x["price_observed_at"] <= x["filled_at"]
        assert x["intent_at"] + x["latency_seconds"] == x["eligible_at"]


# --- reporting and comparison -----------------------------------------------------------------------


def test_report_execution_fields(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    realistic(h)
    d = shadow_report(h.store, h.reader, "r")
    e = d["strategies"][0]["execution"]
    buy, sell = execs(h)
    (t,) = trades(h)
    assert e["execution_model"] == "REALISTIC_V1" and e["settings"]["slippage_bps"] == 50
    assert e["fills"] == {"BUY": 1, "SELL": 1}
    assert e["fees_usd"] == pytest.approx(buy["fee_usd"] + sell["fee_usd"])
    assert e["slippage_cost_usd"] == pytest.approx(
        buy["slippage_cost_usd"] + sell["slippage_cost_usd"])  # fmt: skip
    assert e["price_impact_cost_usd"] == 0
    assert e["gross_realized_pnl_usd"] == pytest.approx(sell["gross_pnl_usd"])
    assert e["net_realized_pnl_usd"] == pytest.approx(t["pnl_usd"])
    assert e["latency_cost_usd"] == pytest.approx(
        buy["latency_cost_usd"] + sell["latency_cost_usd"]
    )
    assert e["average_slippage_bps"] == 50
    assert e["average_execution_delay_seconds"] == pytest.approx(450)
    assert e["pending"] == {"entry_intents": 0, "entry_reserved_usd": 0, "exit_intents": 0,
                            "exit_intents_by_reason": {}}  # fmt: skip
    g = d["data_integrity"]
    assert (
        g["execution_model"] == "REALISTIC_V1" and g["execution_settings"]["latency_seconds"] == 60
    )
    assert "REALISTIC_V1" in g["execution_warning"]
    out = text(d)
    assert "[execution] REALISTIC_V1" in out and "execution model: REALISTIC_V1" in out
    # An idealized run reports no friction (gross = net).
    h.run()
    i = shadow_report(h.store, h.reader, "t")["strategies"][0]["execution"]
    assert i["execution_model"] == "IDEALIZED_NO_FEES" and i["fees_usd"] == 0
    assert i["gross_realized_pnl_usd"] == i["net_realized_pnl_usd"] == pytest.approx(250.0)


# --- latency drift (USD) ---------------------------------------------------------------------


def test_latency_drift_tiny_price_huge_quantity() -> None:
    """The continuous-v2-realistic random_eligible fill that reported -$52,762: a $492.62
    BUY of a token whose exact pool fell 0.0003814 -> 0.000003528 between intent and fill.
    The fill quantity (139.6M tokens) was sized at the fill price; multiplying it by the
    price move scaled the drift by reference / observed (x108)."""
    qty, ref, obs = 139_630_463.8675, 0.0003814, 3.528e-06
    drift = latency_drift_usd("BUY", ref, obs, qty)
    assert qty * (obs - ref) == pytest.approx(-52_762.44, abs=0.01)  # the old formula
    assert drift == pytest.approx(-488.06, abs=0.01)
    assert drift > -qty * obs  # never more than the spend, however far the price fell
    # A collapse toward zero tends to minus the spend, never beyond it.
    assert latency_drift_usd("BUY", 1.0, 1e-12, 500.0 / 1e-12) == pytest.approx(-500.0)


@pytest.mark.parametrize(
    ("side", "ref", "obs", "adverse"),
    [("BUY", 1.0, 1.1, True), ("BUY", 1.0, 0.9, False),
     ("SELL", 1.0, 0.9, True), ("SELL", 1.0, 1.1, False)],
)  # fmt: skip
def test_latency_drift_sign(side: str, ref: float, obs: float, adverse: bool) -> None:
    drift = latency_drift_usd(side, ref, obs, 100.0)
    assert (drift > 0) is adverse  # adverse positive: BUY paid more, SELL received less
    assert latency_drift_usd(side, ref, ref, 100.0) == 0


def test_latency_drift_quantity_applied_once() -> None:
    # Linear in quantity (applied once), and in USD: the observed value times the move.
    for side in ("BUY", "SELL"):
        one, two = (latency_drift_usd(side, 2.0, 2.2, q) for q in (10.0, 20.0))
        assert two == pytest.approx(2 * one)
    assert latency_drift_usd("BUY", 2.0, 2.2, 10.0) == pytest.approx(22.0 * 0.1)
    assert latency_drift_usd("SELL", 2.0, 2.2, 10.0) == pytest.approx(-2.0)
    # A SELL's drift is bounded by its reference notional.
    assert latency_drift_usd("SELL", 2.0, 1e-12, 10.0) == pytest.approx(20.0)


def test_latency_drift_end_to_end_exact_pool_collapse(h: Harness) -> None:
    """A tiny-price token collapses before the entry fill. Another pool of the same token
    priced on a different scale is never used, as fill or as reference. The stored row and
    the report agree with the formula on the raw fields, within the spend."""
    h.scan(T0, h.cand(A, price=0.0003814))
    h.price(m(10), 0.5, pool="other-pool")  # same token, another pool: never this fill
    h.price(m(30), 3.528e-06)
    h.now = m(31)
    realistic(h)
    (buy,) = execs(h)
    assert buy["pool"] == f"pool-{A}" and buy["price_observed_at"] == m(30).timestamp()
    assert buy["reference_price"] == 0.0003814 and buy["observed_price"] == 3.528e-06
    expected = latency_drift_usd("BUY", 0.0003814, 3.528e-06, buy["quantity"])
    assert buy["latency_cost_usd"] == pytest.approx(expected)
    assert -buy["observed_value_usd"] < buy["latency_cost_usd"] < 0
    e = shadow_report(h.store, h.reader, "r")["strategies"][0]["execution"]
    assert e["latency_cost_usd"] == pytest.approx(expected)
    assert e["stored_latency_cost_usd"] == pytest.approx(expected)


def test_report_derives_latency_from_raw_fields_not_the_stored_column(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows recorded before the fix keep their stored latency_cost_usd (immutable history);
    the report derives the metric from the raw reference / observed price and quantity."""
    from upscale.services.shadow import book

    def old(side: str, reference: float, observed: float, quantity: float) -> float:
        return quantity * ((observed - reference) if side == "BUY" else (reference - observed))

    roundtrip(h)
    h.now = m(41)
    monkeypatch.setattr(book, "latency_drift_usd", old)  # the book as it was before the fix
    realistic(h)
    monkeypatch.undo()
    buy, sell = execs(h)
    assert buy["latency_cost_usd"] == pytest.approx(buy["quantity"] * 0.02)
    e = shadow_report(h.store, h.reader, "r")["strategies"][0]["execution"]
    derived = sum(latency_drift_usd(x["side"], x["reference_price"], x["observed_price"],
                                    x["quantity"]) for x in (buy, sell))  # fmt: skip
    assert e["latency_cost_usd"] == pytest.approx(derived)
    assert e["stored_latency_cost_usd"] == pytest.approx(
        buy["quantity"] * 0.02 + sell["latency_cost_usd"])  # fmt: skip


# --- execution delay ----------------------------------------------------------------------------


def test_delay_percentiles_and_buckets() -> None:
    d = delay_stats([60, 300, 600, 900, 1_200, 1_800, 2_400, 3_000, 3_600, 7_200])
    assert d["median_execution_delay_seconds"] == pytest.approx(1_500)
    assert d["p90_execution_delay_seconds"] == pytest.approx(3_600 + 0.1 * 3_600)
    assert d["max_execution_delay_seconds"] == 7_200
    assert d["filled_within_pct"] == {"5m": 20.0, "15m": 40.0, "30m": 60.0, "60m": 90.0}
    assert delay_stats([42.0])["p90_execution_delay_seconds"] == 42.0
    empty = delay_stats([])
    assert (
        empty["p90_execution_delay_seconds"] is None
        and empty["max_execution_delay_seconds"] is None
    )
    assert empty["filled_within_pct"] == {"5m": None, "15m": None, "30m": None, "60m": None}


def test_report_delay_fields(h: Harness) -> None:
    roundtrip(h)  # entry fill 300 s after its intent, the exit 600 s after its
    h.now = m(41)
    realistic(h)
    d = shadow_report(h.store, h.reader, "r")
    e = d["strategies"][0]["execution"]
    assert e["configured_min_latency_seconds"] == 60
    assert e["median_execution_delay_seconds"] == pytest.approx(450)
    assert e["p90_execution_delay_seconds"] == pytest.approx(570)
    assert e["max_execution_delay_seconds"] == pytest.approx(600)
    assert e["filled_within_pct"] == {"5m": 50.0, "15m": 100.0, "30m": 100.0, "60m": 100.0}
    out = text(d)
    assert "p90 570 s" in out and "filled within 5m 50%  15m 100%" in out


# --- evidence gaps ------------------------------------------------------------------------------


def test_delayed_exit_fill_alone_is_not_an_evidence_gap(h: Harness) -> None:
    roundtrip(h)  # the take profit decided at 30 min fills at 40 min: delayed, prices fresh
    h.now = m(41)
    realistic(h)
    d = shadow_report(h.store, h.reader, "r")
    a = d["strategies"][0]["availability"]["evidence_gap_exits"]
    assert a["delayed_exit_fills"] == 1
    assert a["exits_after_evidence_gap"] == 0 and a["market_unavailable_exits"] == 0
    g = d["data_integrity"]
    assert g["evidence_gaps_affected_exits"] is False
    assert "delayed exit fills alone never count" in g["evidence_gaps_affected_exits_rule"]
    assert "evidence gaps affected exits: no" in text(d)


def test_report_is_read_only_and_execution_rows_unchanged(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    realistic(h)
    before = everything(h)
    shadow_report(h.store, h.reader, "r")
    text(shadow_report(h.store, h.reader, "r"))
    assert everything(h) == before


def test_compare_idealized_and_realistic(h: Harness) -> None:
    roundtrip(h)
    h.now = m(41)
    h.run()
    realistic(h)
    d = compare_runs(h.store, h.reader, "t", "r")
    (s,) = d["strategies"]
    assert d["runs"]["t"]["execution_model"] == "IDEALIZED_NO_FEES"
    assert d["runs"]["r"]["execution_model"] == "REALISTIC_V1"
    assert s["same_rules"] and s["insufficient_sample"]
    mt = s["matched"]
    assert (mt["enter_decisions"], mt["filled_in_both"], mt["completed_in_both"]) == (1, 1, 1)
    (ti,), (tr,) = h.store.trades(run_id="t"), trades(h)
    buy, sell = execs(h)
    assert mt["net_pnl_usd"] == {"base": 250.0, "other": pytest.approx(tr["pnl_usd"]),
                                 "delta": pytest.approx(tr["pnl_usd"] - 250.0)}  # fmt: skip
    assert mt["gross_pnl_usd"]["base"] == 250.0  # idealized: gross = net
    assert mt["gross_pnl_usd"]["other"] == pytest.approx(sell["gross_pnl_usd"])
    assert mt["friction_usd"]["base"] == 0
    assert mt["friction_usd"]["other"] == pytest.approx(sell["trade_friction_usd"])
    assert mt["return_delta_pct_points_median"] == pytest.approx(tr["return_pct"] - 25.0)
    assert ti["pnl_usd"] == pytest.approx(250.0)
    assert s["only_in_base"]["enter_decisions"] == s["only_in_other"]["enter_decisions"] == 0
    assert s["divergence"]["first_entry_divergence"] is None
    fees = s["portfolio_totals"]["fees_usd"]
    assert fees["base"] == 0 and fees["delta"] == pytest.approx(fees["other"]) and fees["other"] > 0
    keys = set(d) | set(s) | set(s["portfolio_totals"]) | set(mt)
    assert not [k for k in keys if any(w in k for w in ("rank", "winner", "best", "score"))]
    out = compare_text(d)
    assert "NOT a ranking" in out and "NOT caused by" in out
    with pytest.raises(ShadowError, match="unknown run"):
        compare_runs(h.store, h.reader, "t", "nope")


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_creates_a_realistic_run_and_compares(
    h: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    roundtrip(h)
    h.now = m(41)
    h.run()
    h.close()
    db = ["--db", str(h.shadow_path), "--evidence-db", str(h.ev_path)]
    since = (T0 - timedelta(hours=1)).isoformat()

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("Shadow must not use the network")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    base = [*db, "run", "--run", "r", "--since", since, "--strategy", "t"]
    assert cli_main([*base, "--execution-model", "REALISTIC_V1",
                     "--availability-policy", "LEGACY_V1"]) == 1  # fmt: skip
    assert cli_main([*base, "--slippage-bps", "80"]) == 1  # settings need REALISTIC_V1
    assert cli_main([*base, "--execution-model", "REALISTIC_V1", "--slippage-bps", "-1"]) == 1
    capsys.readouterr()
    assert cli_main([*base, "--execution-model", "REALISTIC_V1",
                     "--availability-policy", "EVIDENCE_AWARE_V2", "--slippage-bps", "80",
                     "--entry-max-wait-minutes", "0"]) == 0  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert out["execution_model"] == "REALISTIC_V1" and out["fills"] == 1
    assert cli_main([*base, "--execution-model", "REALISTIC_V1"]) == 1  # other settings
    assert cli_main([*base, "--execution-model", "IDEALIZED_NO_FEES"]) == 1
    assert cli_main([*base]) == 0  # continuing keeps its own settings
    capsys.readouterr()
    assert cli_main([*db, "compare", "--base", "t", "--other", "r", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["runs"]["r"]["execution_settings"]["slippage_bps"] == 80
    assert d["runs"]["r"]["execution_settings"]["entry_max_wait_minutes"] is None
    assert d["read_only"] and d["provider_requests"] == 0
    assert cli_main([*db, "compare", "--base", "t", "--other", "r"]) == 0
    assert "SHADOW RUN COMPARISON" in capsys.readouterr().out
    assert cli_main([*db, "report", "--run", "r"]) == 0
    assert "[execution] REALISTIC_V1" in capsys.readouterr().out
    assert cli_main([*db, "compare", "--base", "t", "--other", "nope"]) == 1
