"""Shadow market availability: EVIDENCE_AWARE_V2 (missing evidence is never a market exit)
against LEGACY_V1 (unchanged), held-position watch evidence, pending exits, confirmed
MARKET_NOT_FOUND, and the reporting-only last-mark equity. Temporary databases only; watch
records are built with the production serializer (`payloads.pool_watch`).

Harness strategy (`strat()`): TP +20%, SL -10%, max hold 240 min, staleness rule 120 min,
exit delay 60 min, $1,000 per entry."""

import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

from upscale.services.evidence_archive import payloads
from upscale.services.evidence_archive.payloads import PoolWatchObservation, WatchResult
from upscale.services.evidence_archive.store import WATCH_COMPONENT
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow.book import Position
from upscale.services.shadow.config import run_policy
from upscale.services.shadow.strategies import BASELINES
from upscale.services.solana_dex import DexPool, TokenRef

from .test_shadow import T0, A, B, Harness, _scenario, strat

V2, V1 = "EVIDENCE_AWARE_V2", "LEGACY_V1"


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


def dex_pool(price: float | None, mint: str = A, pool: str | None = None,
             liquidity: float = 80_000.0) -> DexPool:  # fmt: skip
    return DexPool(
        chain="solana", dex="raydium", pair_address=pool or f"pool-{mint}",
        base=TokenRef(address=mint, symbol="TOK"),
        quote=TokenRef(address="So11111111111111111111111111111111111111112", symbol="SOL"),
        price_usd=price, liquidity_usd=liquidity,
    )  # fmt: skip


def watch(
    h: Harness, at: datetime, status: WatchResult = "PRICED", price: float | None = 1.0,
    mint: str = A, pool: str | None = None, authoritative: bool = True,
    liquidity: float = 80_000.0,
) -> str:  # fmt: skip
    """One held-position watch record, as production archives it."""
    pool = pool or f"pool-{mint}"
    priced = status in ("PRICED", "LIQUIDITY_COLLAPSE")
    (r,) = payloads.pool_watch(PoolWatchObservation(
        chain="solana", token=mint, pool_address=pool, provider="DEX Screener",
        observed_at=at, status=status, authoritative=authoritative,
        pool=dex_pool(price, mint, pool, liquidity) if priced else None,
    ))  # fmt: skip
    assert r.component == WATCH_COMPONENT
    h._observed = at.timestamp()
    assert h.writer.append(r)
    assert r.record_id is not None
    return r.record_id


def rows(h: Harness, run: str = "t") -> tuple[Any, ...]:
    return (
        [(d["action"], d["asset_id"], d["decision_at"], d["reason"])
         for d in h.store.decisions(run_id=run, limit=100_000)],
        [{k: v for k, v in t.items() if k not in ("id", "recorded_at", "trade_id", "run_id",
                                                   "position_id", "entry_decision_id",
                                                   "exit_decision_id")}
         for t in h.store.trades(run_id=run)],
        [(e["at"], e["cash"], e["equity"], e["unresolved_cost"])
         for e in h.store.equity(run, None)],
    )  # fmt: skip


def open_position(h: Harness, run: str = "t") -> Position:
    state = h.store.checkpoint(run)["books"]["t@v1"]["state"]
    (p,) = state["positions"].values()
    return Position.from_json(p)


# --- the policy is a run setting -----------------------------------------------------------------


def test_new_runs_are_evidence_aware_and_old_runs_stay_legacy(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.run()
    assert run_policy(h.store.run("t") or {}) == V2
    h.run(run_id="old", policy=V1)
    assert run_policy(h.store.run("old") or {}) == V1
    # A run stored before the setting existed (no policy in its args) is LEGACY_V1.
    assert run_policy({"args": {"cli": []}}) == V1
    with pytest.raises(Exception, match="never redefined"):
        h.engine.ensure_run("t", T0 - timedelta(hours=1), None, ["t"], availability_policy=V1)


# --- missing evidence is not a market exit -------------------------------------------------------


def test_evidence_gap_does_not_mean_market_unavailable(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(m(10 * 60), 1.0, mint=B)  # time passes: nothing for our pool for 10 hours
    h.run(policy=V2)
    assert h.trades() == []
    p = open_position(h)
    after = strat().exit.market_unavailable_after_minutes
    assert p.market_state(m(30), after) == "MARKET_AVAILABLE"
    assert p.market_state(m(90), after) == "PRICE_STALE"
    assert p.market_state(m(10 * 60), after) == "EVIDENCE_GAP"
    (run,) = [r for r in h.engine.status()["runs"] if r["run_id"] == "t"]
    assert run["availability_policy"] == V2
    assert run["books"]["t@v1"]["open_market_states"] == {"EVIDENCE_GAP": 1}


def test_legacy_policy_still_closes_on_the_timeout(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(m(10 * 60), 1.0, mint=B)
    h.run(policy=V1)
    (t,) = h.trades()
    assert t["exit_reason"] == "MARKET_UNAVAILABLE" and t["exit_price"] is None


def test_provider_outage_does_not_mean_market_disappearance(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    for i in range(1, 20):
        watch(h, m(30 * i), "RATE_LIMITED" if i % 2 else "PROVIDER_FAILED", None)
    h.run(policy=V2)
    assert h.trades() == []
    p = open_position(h)
    assert p.watch_status in ("RATE_LIMITED", "PROVIDER_FAILED") and p.not_found_count == 0
    assert p.market_state(m(600), 120) == "PROVIDER_UNAVAILABLE"


def test_not_listed_by_another_provider_never_closes(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    for i in range(1, 10):
        watch(h, m(30 * i), "NOT_LISTED", None, authoritative=False)
    h.run(policy=V2)
    assert h.trades() == [] and open_position(h).not_found_count == 0


def test_genuine_market_not_found_is_confirmed_then_closed(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    first = watch(h, m(30), "NOT_FOUND", None)
    h.now = m(31)
    h.run(policy=V2)
    assert h.trades() == [] and open_position(h).market_state(m(31), 120) == "MARKET_NOT_FOUND"
    watch(h, m(60), "NOT_FOUND", None)  # confirmed, but only 30 min after the first
    h.now = m(61)
    h.engine.run("t")
    assert h.trades() == []
    last = watch(h, m(95), "NOT_FOUND", None)  # >= the 60 min exit delay since the first
    h.now = m(96)
    h.engine.run("t")
    (t,) = h.trades()
    assert t["exit_reason"] == "MARKET_UNAVAILABLE" and t["exit_price"] is None
    assert t["exit_at"] == m(95).timestamp()
    (d,) = h.decisions(action="EXIT")
    evidence = json.loads(d["evidence_json"])
    assert d["reason"].startswith("MARKET_NOT_FOUND")
    assert evidence["reason_code"] == "MARKET_NOT_FOUND"
    assert evidence["watch_record"] == last and first != last


def test_a_valid_price_resets_not_found(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    watch(h, m(30), "NOT_FOUND", None)
    watch(h, m(60), "PRICED", 1.01)  # the pool answered again
    watch(h, m(100), "NOT_FOUND", None)
    h.now = m(101)
    h.run(policy=V2)
    assert h.trades() == [] and open_position(h).not_found_count == 1


# --- held-position watch prices -------------------------------------------------------------------


def test_held_asset_gone_from_discovery_is_priced_by_the_watch(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))  # Scout never sees the token again
    for i in range(1, 14):
        watch(h, m(30 * i), price=1.0 + i / 1000)
    sl = watch(h, m(420), price=0.85)  # below the -10% stop
    h.now = m(421)
    long_hold = strat(exit={"max_hold_minutes": 1440.0})
    h.run(long_hold, policy=V2)
    (t,) = h.trades()
    assert t["exit_reason"] == "STOP_LOSS" and t["exit_price"] == 0.85
    assert t["exit_price_record"] == sl and t["exit_at"] == m(420).timestamp()
    # The same evidence under LEGACY_V1: watch records ignored, the old timeout applies.
    h.run(long_hold, run_id="legacy", policy=V1)
    (lt,) = h.store.trades(run_id="legacy")
    assert lt["exit_reason"] == "MARKET_UNAVAILABLE"


def test_watch_identity_is_exact_token_and_pool(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    watch(h, m(30), price=0.5, pool="another-pool")  # same token, other pool
    watch(h, m(40), price=0.5, mint=B)  # other token
    watch(h, m(50), price=0.5, pool=f"POOL-{A}")  # Solana: case matters
    h.now = m(60)
    h.run(policy=V2)
    assert h.trades() == [] and open_position(h).last_price == 1.0


def test_liquidity_collapse_is_a_real_price_labelled(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    watch(h, m(30), "LIQUIDITY_COLLAPSE", price=0.95, liquidity=400.0)
    h.now = m(31)
    h.run(policy=V2)
    p = open_position(h)
    assert p.last_price == 0.95 and p.market_state(m(31), 120) == "LIQUIDITY_COLLAPSE"


# --- pending exits wait for a valid price ----------------------------------------------------------


def test_pending_signal_exit_waits_for_the_exact_pool_price(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.scan(m(30), h.cand(price=1.1, stage="FADING", pool="other-pool"))  # no price of ours
    h.price(m(9 * 60), 1.0, mint=B)
    h.now = m(9 * 60)
    h.run(policy=V2)
    assert h.trades() == [] and open_position(h).pending_exit == "SIGNAL_EXIT"
    watch(h, m(10 * 60), price=0.97)
    h.now = m(10 * 60 + 1)
    h.engine.run("t")
    (t,) = h.trades()
    assert t["exit_reason"] == "SIGNAL_EXIT" and t["exit_price"] == 0.97
    assert t["exit_at"] == m(10 * 60).timestamp()


def test_max_hold_waits_for_a_price(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(m(8 * 60), 1.0, mint=B)  # well past the 240 min max hold + 60 min delay
    h.run(policy=V2)
    assert h.trades() == []
    watch(h, m(8 * 60 + 5), price=1.05)
    h.now = m(8 * 60 + 6)
    h.engine.run("t")
    (t,) = h.trades()
    assert t["exit_reason"] == "MAX_HOLD_TIME" and t["exit_price"] == 1.05


def test_take_profit_and_stop_loss_need_an_observed_price(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.scan(T0, h.cand(B, price=1.0))
    h.price(m(10 * 60), 1.0, mint="MintCCCC3333")
    h.run(policy=V2)
    assert h.trades() == []  # nothing priced: nothing fills, nothing is fabricated
    watch(h, m(10 * 60 + 1), price=1.25)
    watch(h, m(10 * 60 + 2), price=0.8, mint=B)
    h.now = m(10 * 60 + 3)
    h.engine.run("t")
    got = {t["asset_id"]: (t["exit_reason"], t["exit_price"]) for t in h.trades()}
    assert got == {f"solana:{A}": ("TAKE_PROFIT", 1.25), f"solana:{B}": ("STOP_LOSS", 0.8)}


# --- legacy and golden behavior ---------------------------------------------------------------------


def test_legacy_run_is_untouched_by_watch_records(tmp_path: Path, template: Any) -> None:
    def scenario(x: Harness, with_watch: bool) -> None:
        _scenario(x)
        if with_watch:
            for i in range(1, 30):
                watch(x, m(20 * i), price=0.5 + i / 10)  # would trigger exits if it were read
                watch(x, m(20 * i + 1), "NOT_FOUND", None, mint=B)
        x.price(m(12 * 60), 1.0, mint="MintCCCC3333")
        x.now = m(12 * 60 + 1)

    plain, watched = Harness(tmp_path / "a", template), Harness(tmp_path / "b", template)
    scenario(plain, False)
    scenario(watched, True)
    for x in (plain, watched):
        x.run(*BASELINES, policy=V1)
    assert rows(plain) == rows(watched) and rows(plain)[1]
    states = [x.store.checkpoint("t")["books"] for x in (plain, watched)]
    assert states[0] == states[1]
    for x in (plain, watched):
        x.close()


def test_decisions_identical_before_the_first_availability_event(
    tmp_path: Path, template: Any
) -> None:
    runs = {}
    for policy in (V1, V2):
        x = Harness(tmp_path / policy, template)
        _scenario(x)
        x.scan(m(160), x.cand("MintH", price=1.0))  # then never priced again
        x.price(m(12 * 60), 1.0, mint="MintCCCC3333")
        x.now = m(12 * 60 + 1)
        x.run(*BASELINES, policy=policy)  # type: ignore[arg-type]
        runs[policy] = rows(x)[0]
        trades = x.store.trades(run_id="t")
        x.close()
        if policy == V1:
            unavailable = [t["exit_at"] for t in trades if t["exit_reason"] == "MARKET_UNAVAILABLE"]
    assert unavailable, "the scenario must contain a LEGACY_V1 evidence-timeout exit"
    first = min(unavailable)
    before = [[d for d in runs[p] if d[2] < first] for p in (V1, V2)]
    assert before[0] == before[1] and before[0]


def test_golden_rows_unchanged_under_both_policies(tmp_path: Path, template: Any) -> None:
    """The golden decisions / trades / positions (commit 01bd2d9) hold under LEGACY_V1 and
    EVIDENCE_AWARE_V2: no availability event happens in that scenario."""
    from .test_shadow_diagnostics import GOLDEN

    _, g = template
    golden = json.loads(GOLDEN.read_text())
    for policy in (V1, V2):
        hh = Harness(tmp_path / policy, template)
        _scenario(hh)
        down = g.momentum.technical.model_copy(update={"trend": "flat"})  # type: ignore[union-attr]
        hh.scan(m(200), hh.cand("MintD", safety="SAFETY_CHECKS_COMPLETE"),
                hh.cand("MintE", technical=down), hh.cand("MintF", technical=None),
                hh.cand("MintG", score=40.0), *[hh.cand(f"Mint{i:03d}") for i in range(30)])  # fmt: skip
        hh.price(m(230), 1.4, mint="MintD")
        hh.run(strat(), *BASELINES, policy=policy)  # type: ignore[arg-type]
        got = {
            "decisions": [[x["decision_id"], x["strategy_id"], x["action"], x["asset_id"],
                           x["decision_at"], x["reference_price"], x["reason"]]
                          for x in hh.store.decisions(run_id="t", limit=100_000)],
            "trades": [[x["trade_id"], x["strategy_id"], x["exit_reason"], x["exit_price"],
                        x["exit_at"], x["pnl_usd"]] for x in hh.store.trades(run_id="t")],
            "positions": [[x["position_id"], x["status"], x["entry_price"], x["last_price"]]
                          for x in hh.store.positions(run_id="t")],
        }  # fmt: skip
        hh.close()
        assert got == golden, policy


# --- anti-lookahead, resume, isolation ---------------------------------------------------------------


def test_watch_evidence_is_used_only_once_the_clock_reaches_it(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    watch(h, m(60), price=0.8)
    h.now = m(59)  # processing stops before the watch observation
    h.run(policy=V2)
    assert h.trades() == []
    h.now = m(61)
    h.engine.run("t")
    (t,) = h.trades()
    assert t["exit_at"] == m(60).timestamp() and t["exit_price"] == 0.8


def test_resume_is_idempotent_with_watch_evidence(tmp_path: Path, template: Any) -> None:
    def build(x: Harness) -> None:
        x.scan(T0, x.cand(price=1.0), x.cand(B, price=2.0))
        for i in range(1, 12):
            watch(x, m(30 * i), price=1.0 + i / 100)
            watch(x, m(30 * i + 1), "NOT_FOUND", None, mint=B)
        x.price(m(8 * 60), 1.0, mint="MintCCCC3333")

    single, staged = Harness(tmp_path / "s", template), Harness(tmp_path / "t", template)
    build(single)
    build(staged)
    single.now = staged.now = m(8 * 60 + 1)
    single.run(policy=V2)
    staged.now = m(100)
    staged.run(policy=V2)
    staged.reopen()
    staged.now = m(200)
    staged.engine.run("t")
    staged.reopen()
    staged.now = m(8 * 60 + 1)
    staged.engine.run("t")
    staged.engine.run("t")  # nothing new
    assert rows(single) == rows(staged) and rows(single)[1]
    for x in (single, staged):
        x.close()


def test_shadow_makes_no_provider_call(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("Shadow must not use the network")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    h.scan(T0, h.cand(price=1.0))
    watch(h, m(30), price=1.3)
    watch(h, m(31), "NOT_FOUND", None, mint=B)
    h.now = m(40)
    h.run(*BASELINES, strat(), policy=V2)
    h.engine.metrics("t")
    h.engine.status()
    assert h.trades()


# --- reporting-only last-mark equity -------------------------------------------------------------------


def test_last_mark_equity_is_reporting_only(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(m(60), 0.95)  # the last mark before the evidence stops
    h.scan(m(400), h.cand(B, price=2.0))  # after A's timeout (60 + 120 min): written off
    h.now = m(401)
    sized = strat(risk={"max_allocation_pct": 10.0})  # 10% of equity per entry binds
    h.run(sized, policy=V1)
    (m_,) = h.engine.metrics("t")["strategies"]
    assert m_["unresolved_cost_usd"] == pytest.approx(1000.0)
    assert m_["written_off_equity_usd"] == m_["equity_usd"]  # the legacy field is kept
    assert m_["unresolved_last_mark_value_usd"] == pytest.approx(950.0)
    assert m_["last_mark_equity_usd"] == pytest.approx(m_["equity_usd"] + 950.0)
    assert m_["unresolved_capital_pct"] == pytest.approx(1000.0 / 10_000 * 100)
    assert m_["written_off_max_drawdown_pct"] == m_["max_drawdown_pct"] < -9.0
    assert -1.0 < m_["last_mark_max_drawdown_pct"] <= 0.0
    assert "REPORTING ONLY" in m_["accounting_note"]
    # Sizing never sees it: B's entry is 10% of the written-off $9,000, not of $9,950.
    (entry,) = [p for p in h.positions() if p["asset_id"] == f"solana:{B}"]
    assert entry["cost_usd"] == pytest.approx(900.0)
    state = h.store.checkpoint("t")["books"]["t@v1"]["state"]
    assert "unresolved_mark" not in state  # nothing new in the book's own accounting


def test_a_run_created_before_the_policy_existed_stays_legacy(h: Harness) -> None:
    """continuous-v1's situation: its stored run has no availability policy."""
    s = strat()
    h.store.register(s)
    h.store.create_run("continuous-v1", T0 - timedelta(hours=1), None, [s], clean_data=True,
                       args={"source": "background"})  # fmt: skip
    h.scan(T0, h.cand(price=1.0))
    for i in range(1, 12):
        watch(h, m(30 * i), price=1.0)  # ignored by LEGACY_V1
    h.price(m(10 * 60), 1.0, mint=B)
    report = h.engine.run("continuous-v1")
    assert report["availability_policy"] == V1
    (t,) = h.store.trades(run_id="continuous-v1")
    assert t["exit_reason"] == "MARKET_UNAVAILABLE"  # exactly the old behavior
    assert h.store.run("continuous-v1")["args"] == {"source": "background"}  # type: ignore[index]
