"""Shadow / Paper Strategy Engine v1: immutable strategies, clean cutoff, entries, exits,
duplicate control, accounting, anti-lookahead, isolation, priority, Calibration SHADOW,
restart / resume. Temporary databases only; archived evidence is built with the
production payload serializers (`evidence_archive.payloads.scout`)."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import upscale.main
from upscale.services.calibration.config import CalibrationConfig
from upscale.services.calibration.dataset import load_shadow
from upscale.services.calibration.engine import CalibrationEngine
from upscale.services.calibration.store import CalibrationStore
from upscale.services.evidence_archive import payloads
from upscale.services.evidence_archive.store import EvidenceStore, PendingRecord
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow.book import Book
from upscale.services.shadow.cli import main as cli_main
from upscale.services.shadow.config import (
    CLEAN_DATA_CUTOFF,
    EntryRules,
    ExitRules,
    RiskRules,
    StrategyConfig,
    TakeProfitLevel,
    default_shadow_db,
    load_settings,
)
from upscale.services.shadow.engine import LookaheadError, ShadowEngine, ShadowError
from upscale.services.shadow.evidence import (
    AnalyzeView,
    Event,
    EvidenceTimeline,
    PriceObs,
    scout_view,
)
from upscale.services.shadow.metrics import effective_sample_size, trimmed_mean
from upscale.services.shadow.service import BackgroundShadow
from upscale.services.shadow.store import ShadowStore, ShadowStoreError
from upscale.services.shadow.strategies import BASELINES

from .test_outcomes import ranked

T0 = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)  # a live CALIBRATION day, after the cutoff
LATE = T0 + timedelta(days=60)
A, B = "MintAAAA1111", "MintBBBB2222"


# --- fixtures ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[GrowthScoutResult, GrowthCandidate]:
    result, store = ranked(tmp_path_factory.mktemp("template"))
    store.close()
    g = next(c for c in result.candidates if c.stage == "ACCELERATING")
    return result, g


def strat(sid: str = "t", version: int = 1, **over: dict[str, Any]) -> StrategyConfig:
    entry = {"min_scout_score": 60.0, "allowed_stages": ["ACCELERATING", "EARLY"],
             "min_liquidity_usd": 10_000.0, "max_risk_penalty": None,
             "blocking_flag_severities": []} | over.get("entry", {})  # fmt: skip
    exit_ = {"take_profit": [{"gain_pct": 20.0}], "stop_loss_pct": 10.0,
             "max_hold_minutes": 240.0, "market_unavailable_after_minutes": 120.0,
             "max_exit_delay_minutes": 60.0} | over.get("exit", {})  # fmt: skip
    risk = {"position_size_usd": 1000.0, "max_allocation_pct": 20.0,
            "max_exposure_per_asset_pct": 20.0, "entry_cooldown_minutes": 60.0,
            "min_entry_spacing_minutes": 30.0,
            "max_entries_per_asset_per_day": 3} | over.get("risk", {})  # fmt: skip
    return StrategyConfig.model_validate(
        {"strategy_id": sid, "version": version, "name": sid, "created_at": T0,
         "entry": entry, "exit": exit_, "risk": risk}
    )  # fmt: skip


class Harness:
    def __init__(self, tmp_path: Path, template: tuple[GrowthScoutResult, GrowthCandidate]) -> None:
        self.result, self.tpl = template
        self.archive_override: float | None = None  # when set: archived at this time
        self._observed = T0.timestamp()
        self.now: datetime | None = None  # None: one minute after the latest evidence
        self.ev_path = tmp_path / "evidence.sqlite3"
        self.shadow_path = tmp_path / "shadow.sqlite3"
        # Archived one second after being observed, unless a test delays it.
        self.writer = EvidenceStore(
            self.ev_path, clock=lambda: self.archive_override or self._observed + 1.0
        )
        self.writer.totals()  # create the file
        self.open()

    def open(self) -> None:
        self.store = ShadowStore(self.shadow_path)
        self.reader = EvidenceStore(self.ev_path, read_only=True)
        self.engine = ShadowEngine(self.store, self.reader, now=self.clock, settle_seconds=0)

    def clock(self) -> datetime:
        if self.now is not None:
            return self.now
        newest = self.writer.totals()["newest"]
        return (newest or T0) + timedelta(minutes=1)

    def reopen(self) -> None:
        self.store.close()
        self.reader.close()
        self.open()

    def cand(
        self, mint: str = A, price: float = 1.0, at: datetime = T0, *, stage: str = "ACCELERATING",
        score: float = 75.0, liquidity: float | None = 80_000.0, pool: str | None = None,
        data_status: str = "CURRENT", eligible: bool = True, safety: str = "INSUFFICIENT_SAFETY_DATA",
        technical: Any = "keep", market_status: str = "OK", symbol: str = "TOK",
    ) -> GrowthCandidate:  # fmt: skip
        g = self.tpl
        market = g.market.model_copy(update={
            "price_usd": price, "liquidity_usd": liquidity,
            "selected_pool": g.market.selected_pool.model_copy(update={"address": pool or f"pool-{mint}"}),
        })  # fmt: skip
        return g.model_copy(update={
            "canonical_id": f"solana:{mint}", "address": mint, "symbol": symbol,
            "observed_at": at, "stage": stage, "unconfirmed_stage": None, "market": market,
            "scout_momentum": g.scout_momentum.model_copy(update={"score": score}),
            "quality": g.quality.model_copy(update={"safety_status": safety,
                                                    "market_status": market_status}),
            "momentum": g.momentum.model_copy(
                update={"technical": g.momentum.technical if technical == "keep" else technical}),
            "data_status": data_status, "eligible": eligible,
        })  # fmt: skip

    def scan(self, at: datetime, *cands: GrowthCandidate, lag: float = 60.0) -> None:
        """One Scout run: market observed `lag` seconds before the decision time `at`."""
        fixed = [c.model_copy(update={"observed_at": at - timedelta(seconds=lag)}) for c in cands]
        result = self.result.model_copy(update={
            "computed_at": at - timedelta(seconds=lag), "candidates": fixed, "unranked": [],
            "evaluated": len(fixed), "eligible": len(fixed)})  # fmt: skip
        self._observed = at.timestamp()
        for r in payloads.scout(result, [], at, None):
            assert self.writer.append(r)

    def price(self, at: datetime, price: float, mint: str = A, pool: str | None = None) -> None:
        pool = pool or f"pool-{mint}"
        self._observed = at.timestamp()
        assert self.writer.append(PendingRecord(
            kind="market", asset_id=f"solana:{mint}", chain="solana", address=mint,
            pool_address=pool, provider="GeckoTerminal", observed_at=at,
            payload={"candidate": {"metrics": {"price_usd": price}, "pool": {"address": pool}}},
        ))  # fmt: skip

    def analyze(
        self, at: datetime, mint: str = A, action: str = "buy", confidence: str = "high"
    ) -> None:
        self._observed = at.timestamp()
        assert self.writer.append(PendingRecord(
            kind="decision", asset_id=f"solana:{mint}", chain="solana", address=mint,
            observed_at=at,
            payload={"decision": {"action": action, "confidence": confidence,
                                  "risk": {"level": "low"}},
                     "agents": {"technical": {"findings": {"trend": {"label": "uptrend"}}}}},
        ))  # fmt: skip

    def run(
        self, *strategies: StrategyConfig, run_id: str = "t", since: datetime = T0 - timedelta(hours=1),
        until: datetime | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        for s in strategies or (strat(),):
            self.store.register(s)
        ids = [s.strategy_id for s in strategies or (strat(),)]
        self.engine.ensure_run(run_id, since, until, ids)
        return self.engine.run(run_id)

    def trades(self, **kw: Any) -> list[dict[str, Any]]:
        return self.store.trades(run_id="t", **kw)

    def positions(self, **kw: Any) -> list[dict[str, Any]]:
        return self.store.positions(run_id="t", **kw)

    def decisions(self, **kw: Any) -> list[dict[str, Any]]:
        return self.store.decisions(run_id="t", **kw)

    def close(self) -> None:
        self.store.close()
        self.reader.close()
        self.writer.close()


@pytest.fixture
def h(tmp_path: Path, template: tuple[GrowthScoutResult, GrowthCandidate]) -> Iterator[Harness]:
    harness = Harness(tmp_path, template)
    yield harness
    harness.close()


def actions(h: Harness) -> list[str]:
    return [d["action"] for d in h.decisions()]


# --- 1. strategies: immutable, versioned ---------------------------------------------------------


def test_strategy_versions_are_immutable(h: Harness) -> None:
    s1 = strat()
    assert h.store.register(s1) == "created"
    assert h.store.register(s1) == "exists"
    changed = strat(entry={"min_scout_score": 70.0})
    assert changed.config_hash != s1.config_hash
    with pytest.raises(ShadowStoreError, match="new version"):
        h.store.register(changed)
    assert h.store.register(strat(version=2, entry={"min_scout_score": 70.0})) == "created"
    # Name / description / creation time are not rules: same hash.
    renamed = s1.model_copy(update={"name": "other", "description": "x"})
    assert renamed.config_hash == s1.config_hash
    raw = sqlite3.connect(h.shadow_path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute("UPDATE shadow_strategies SET config_json = '{}'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute("DELETE FROM shadow_strategies")
    raw.close()


def test_a_run_freezes_strategy_versions(h: Harness) -> None:
    h.store.register(strat())
    h.engine.ensure_run("t", T0, None, ["t"])
    h.store.register(strat(version=2, entry={"min_scout_score": 99.0}))
    run = h.store.run("t")
    assert run is not None and run["strategies"] == [
        {"strategy_id": "t", "version": 1, "config_hash": strat().config_hash}
    ]
    with pytest.raises(ShadowError, match="never redefined"):
        h.engine.ensure_run("t", T0 + timedelta(hours=1), None, ["t"])


def test_strategy_config_validation() -> None:
    with pytest.raises(ValueError, match="SAFETY_NOT_AVAILABLE"):
        EntryRules(required_safety="COMPLETE")
    with pytest.raises(ValueError, match="increasing"):
        ExitRules(take_profit=(TakeProfitLevel(gain_pct=30), TakeProfitLevel(gain_pct=10)))
    with pytest.raises(ValueError, match="allow_scaling"):
        RiskRules(max_positions_per_asset=2)
    with pytest.raises(ValueError, match="slug"):
        strat(sid="Bad Id")


def test_baselines_are_valid_distinct_and_share_exits() -> None:
    assert [b.strategy_id for b in BASELINES] == [
        "scout_threshold", "scout_technical", "scout_safety_technical", "random_eligible",
    ]  # fmt: skip
    assert len({b.config_hash for b in BASELINES}) == 4
    assert len({b.exit.model_dump_json() for b in BASELINES}) == 1
    assert len({b.risk.model_dump_json() for b in BASELINES}) == 1
    assert all(b.execution_model == "IDEALIZED_NO_FEES" for b in BASELINES)
    assert BASELINES[3].entry.random_fraction == 0.10


# --- 2. clean-data cutoff ------------------------------------------------------------------------


def test_clean_cutoff_is_enforced(h: Harness) -> None:
    h.store.register(strat())
    with pytest.raises(ShadowError, match="clean-data cutoff"):
        h.engine.ensure_run("t", CLEAN_DATA_CUTOFF - timedelta(minutes=1), None, ["t"])
    research = h.engine.ensure_run("r", CLEAN_DATA_CUTOFF - timedelta(days=1), None, ["t"],
                                   allow_contaminated=True)  # fmt: skip
    assert research["clean_data"] is False
    assert load_settings("1", since="2026-09-01T00:00:00Z").since == CLEAN_DATA_CUTOFF
    assert load_settings("1", since="2026-10-02T00:00:00Z").since > CLEAN_DATA_CUTOFF


def test_evidence_before_since_is_never_used(h: Harness) -> None:
    h.scan(T0 - timedelta(hours=2), h.cand())  # before the run's start
    h.scan(T0, h.cand(B))
    h.run(since=T0 - timedelta(hours=1))
    assert {p["asset_id"] for p in h.positions()} == {f"solana:{B}"}


# --- 3-4. entries -----------------------------------------------------------------------------------


def test_entry_records_evidence_and_reference_price(h: Harness) -> None:
    h.scan(T0, h.cand(price=2.0))
    report = h.run()
    assert report["actions"] == {"ENTER": 1}
    (d,) = h.decisions()
    assert d["action"] == "ENTER" and d["reference_price"] == 2.0
    assert d["decision_at"] == T0.timestamp()
    assert d["reference_price_at"] == (T0 - timedelta(seconds=60)).timestamp()
    ev, fps = json.loads(d["evidence_json"]), json.loads(d["fingerprints_json"])
    assert ev["scout"]["stage"] == "ACCELERATING" and ev["technical"]["trend"] == "up"
    assert "SAFETY_NOT_AVAILABLE" in ev["missing"] and ev["risk"]["risk_penalty"] is not None
    assert fps[0]["kind"] == "scout" and fps[0]["record_id"]
    (p,) = h.positions()
    assert p["status"] == "OPEN" and p["entry_price"] == 2.0 and p["quantity"] == 500.0
    assert p["pool"] == f"pool-{A}" and p["asset_id"] == f"solana:{A}"
    assert report["books"]["t@v1"]["cash"] == 9000.0


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"score": 50.0}, "SCORE_BELOW_MIN"),
        ({"stage": "STEADY"}, "STAGE_NOT_ALLOWED"),
        ({"liquidity": 5_000.0}, "LIQUIDITY_BELOW_MIN"),
        ({"liquidity": None}, "LIQUIDITY_BELOW_MIN"),
        ({"data_status": "STALE_CARRIED"}, "STALE_DATA"),
        ({"eligible": False}, "SCOUT_NOT_ELIGIBLE"),
        ({"price": float("nan")}, "CURRENT_PRICE_UNAVAILABLE"),
        ({"price": 0.0}, "CURRENT_PRICE_UNAVAILABLE"),
        ({"market_status": "MARKET_COLLAPSE"}, "MARKET_COLLAPSE"),
    ],
)
def test_no_entry(h: Harness, change: dict[str, Any], reason: str) -> None:
    h.scan(T0, h.cand(**change))
    report = h.run()
    assert report["actions"] == {} and not h.positions()
    (r,) = h.store.rejections(run_id="t")
    assert reason in r["reasons"], r["reasons"]


def test_missing_safety_follows_the_explicit_config(h: Harness) -> None:
    strict = strat("strict", entry={"allowed_missing": ["SOCIAL_NOT_AVAILABLE"]})
    lenient = strat("lenient")
    complete = strat("complete", entry={"required_safety": "COMPLETE",
                                        "allowed_missing": ["SOCIAL_NOT_AVAILABLE"]})  # fmt: skip
    h.scan(T0, h.cand())  # safety INSUFFICIENT_SAFETY_DATA
    h.scan(T0 + timedelta(minutes=5), h.cand(B, safety="SAFETY_CHECKS_COMPLETE"))
    h.run(strict, lenient, complete)
    held = {(p["strategy_id"], p["asset_id"]) for p in h.positions()}
    assert held == {("lenient", f"solana:{A}"), ("lenient", f"solana:{B}"),
                    ("strict", f"solana:{B}"), ("complete", f"solana:{B}")}  # fmt: skip


def test_technical_rules_use_scout_technical_context(h: Harness) -> None:
    tech = strat("tech", entry={"technical": {"allowed_trends": ["up"]},
                                "allowed_missing": ["SAFETY_NOT_AVAILABLE", "SOCIAL_NOT_AVAILABLE"]})  # fmt: skip
    down = h.tpl.momentum.technical.model_copy(update={"trend": "down"})  # type: ignore[union-attr]
    h.scan(T0, h.cand(A), h.cand(B, technical=down), h.cand("MintC", technical=None))
    h.run(tech)
    assert [p["asset_id"] for p in h.positions()] == [f"solana:{A}"]


def test_analyze_rules_only_see_decisions_at_or_before_t(h: Harness) -> None:
    s = strat("an", entry={"analyze": {"max_age_minutes": 60, "min_confidence": "medium"}})
    h.analyze(T0 - timedelta(minutes=10), A)  # known at T0
    h.analyze(T0 + timedelta(minutes=1), B)  # only exists after T0
    h.scan(T0, h.cand(A), h.cand(B))
    h.run(s)
    assert [p["asset_id"] for p in h.positions()] == [f"solana:{A}"]
    (d,) = h.decisions(action="ENTER")
    ev = json.loads(d["evidence_json"])
    assert (
        ev["opportunity"]["action"] == "buy" and ev["opportunity"]["technical_trend"] == "uptrend"
    )
    assert any(f["kind"] == "decision" for f in json.loads(d["fingerprints_json"]))


# --- 7-8. position state, duplicate control ---------------------------------------------------------


def test_one_position_per_asset_and_duplicate_scans(h: Harness) -> None:
    for k in range(10):
        h.scan(T0 + timedelta(minutes=10 * k), h.cand(price=1.0 + 0.001 * k))
    h.run()
    assert len(h.positions()) == 1
    assert actions(h).count("ENTER") == 1 and actions(h).count("HOLD") == 9
    m = h.engine.metrics("t")["strategies"][0]
    assert m["positions_opened"] == 1


def test_cooldown_spacing_and_daily_limit(h: Harness) -> None:
    s = strat(risk={"entry_cooldown_minutes": 60.0, "min_entry_spacing_minutes": 0.0,
                    "max_entries_per_asset_per_day": 2})  # fmt: skip
    h.scan(T0, h.cand(price=1.0))
    h.scan(T0 + timedelta(minutes=10), h.cand(price=1.3))  # take profit, then cooldown
    h.scan(T0 + timedelta(minutes=80), h.cand(price=1.3))  # cooldown over: enter again
    h.scan(T0 + timedelta(minutes=90), h.cand(price=1.7))  # take profit
    h.scan(T0 + timedelta(minutes=200), h.cand(price=1.7))  # 2 entries today: blocked
    h.run(s)
    blocked = [d["reason"] for d in h.decisions(action="NO_ACTION")]
    assert blocked == ["qualified but blocked: COOLDOWN", "qualified but blocked: COOLDOWN",
                       "qualified but blocked: MAX_ENTRIES_PER_ASSET_PER_DAY"]  # fmt: skip
    assert actions(h).count("ENTER") == 2


def test_min_entry_spacing(h: Harness) -> None:
    s = strat(risk={"entry_cooldown_minutes": 0.0, "min_entry_spacing_minutes": 120.0})
    h.scan(T0, h.cand(price=1.0))
    h.scan(T0 + timedelta(minutes=10), h.cand(price=1.3))  # exit (TP) 10 min after entry
    h.scan(T0 + timedelta(minutes=20), h.cand(price=1.3))
    h.run(s)
    assert [d["reason"] for d in h.decisions(action="NO_ACTION")] == [
        "qualified but blocked: MIN_ENTRY_SPACING"
    ] * 2


def test_max_open_positions_and_capital(h: Harness) -> None:
    s = strat(risk={"max_open_positions": 2})
    h.scan(T0, h.cand(A), h.cand(B), h.cand("MintC"))
    h.run(s)
    assert len(h.positions()) == 2
    assert "MAX_OPEN_POSITIONS" in h.decisions(action="NO_ACTION")[0]["reason"]


# --- 6. exits -------------------------------------------------------------------------------------


def test_take_profit_fills_at_the_observed_price(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=30), 1.25)  # gaps through the +20% level
    h.price(T0 + timedelta(minutes=40), 5.0)  # after the exit: irrelevant
    h.run()
    (t,) = h.trades()
    assert t["exit_reason"] == "TAKE_PROFIT" and t["exit_price"] == 1.25
    assert t["trigger_level"] == pytest.approx(1.2) and t["return_pct"] == pytest.approx(25.0)
    assert t["pnl_usd"] == pytest.approx(250.0) and t["holding_minutes"] == 30.0
    assert t["mfe_pct"] == pytest.approx(25.0) and t["mae_pct"] == 0.0
    assert t["execution_model"] == "IDEALIZED_NO_FEES"
    assert t["price_basis"] == "OBSERVED_EXACT_POOL_PRICE"
    (p,) = h.positions()
    assert p["status"] == "CLOSED" and p["exit_reason"] == "TAKE_PROFIT"


def test_take_profit_ladder_partial_exits(h: Harness) -> None:
    s = strat(exit={"take_profit": [{"gain_pct": 10.0, "fraction": 0.5}, {"gain_pct": 30.0}]})
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=10), 1.12)
    h.price(T0 + timedelta(minutes=20), 1.35)
    h.run(s)
    t1, t2 = h.trades()
    assert (t1["fraction"], t1["final"], t1["exit_price"]) == (0.5, 0, 1.12)
    assert (t2["fraction"], t2["final"], t2["exit_price"]) == (0.5, 1, 1.35)
    m = h.engine.metrics("t")["strategies"][0]
    assert m["trades"] == 1 and m["median_return_pct"] == pytest.approx(23.5)


def test_stop_loss(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=5), 0.95)
    h.price(T0 + timedelta(minutes=10), 0.8)  # gaps through -10%
    h.run()
    (t,) = h.trades()
    assert t["exit_reason"] == "STOP_LOSS" and t["exit_price"] == 0.8
    assert t["return_pct"] == pytest.approx(-20.0) and t["mae_pct"] == pytest.approx(-20.0)


def test_trailing_stop(h: Harness) -> None:
    s = strat(exit={"take_profit": [{"gain_pct": 500.0}], "trailing_stop_pct": 10.0,
                    "trailing_activation_pct": 5.0})  # fmt: skip
    h.scan(T0, h.cand(price=1.0))
    for k, p in enumerate((1.03, 1.2, 1.5, 1.4, 1.34)):
        h.price(T0 + timedelta(minutes=10 * (k + 1)), p)
    h.run(s)
    (t,) = h.trades()
    assert t["exit_reason"] == "TRAILING_STOP" and t["exit_price"] == 1.34
    assert t["trigger_level"] == pytest.approx(1.35) and t["mfe_pct"] == pytest.approx(50.0)


def test_max_hold_time(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=60), 1.04)
    h.price(T0 + timedelta(minutes=120), 1.05)
    h.price(T0 + timedelta(minutes=180), 1.05)
    h.price(T0 + timedelta(minutes=239), 1.06)
    h.price(T0 + timedelta(minutes=250), 1.07)
    h.run()
    (t,) = h.trades()
    assert t["exit_reason"] == "MAX_HOLD_TIME" and t["exit_price"] == 1.07
    assert t["holding_minutes"] == 250.0


def test_signal_exit_on_fading(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.scan(T0 + timedelta(minutes=30), h.cand(price=1.02, stage="FADING"))
    h.run()
    (t,) = h.trades()
    assert t["exit_reason"] == "SIGNAL_EXIT" and t["exit_price"] == 1.02
    assert t["exit_at"] == (T0 + timedelta(minutes=30)).timestamp()
    assert "SIGNAL_EXIT: Scout stage FADING" in h.decisions(action="EXIT")[0]["reason"]


def test_signal_exit_waits_for_the_exact_pool_price(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    # Scout now prices another pool of the token: that price never fills our position.
    h.scan(T0 + timedelta(minutes=30), h.cand(price=9.0, stage="FADING", pool="other-pool"))
    h.price(T0 + timedelta(minutes=45), 0.97)  # our pool
    h.run()
    (t,) = h.trades()
    assert t["exit_reason"] == "SIGNAL_EXIT" and t["exit_price"] == 0.97
    assert t["exit_at"] == (T0 + timedelta(minutes=45)).timestamp()


def test_market_unavailable_never_fabricates_an_exit(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.scan(T0 + timedelta(minutes=30), h.cand(price=1.1, stage="FADING", pool="other-pool"))
    h.now = T0 + timedelta(hours=3)
    h.run()  # no price of our pool within the 60-minute exit delay
    (t,) = h.trades()
    assert t["exit_reason"] == "MARKET_UNAVAILABLE"
    assert t["exit_price"] is None and t["pnl_usd"] is None and t["return_pct"] is None
    assert t["exit_at"] == (T0 + timedelta(minutes=90)).timestamp()
    book = h.store.checkpoint("t")["books"]["t@v1"]["state"]
    assert book["unresolved_cost"] == 1000.0 and book["cash"] == 9000.0
    m = h.engine.metrics("t")["strategies"][0]
    assert m["unresolved_trades"] == 1 and m["resolved_trades"] == 0 and m["win_rate"] is None
    raw = sqlite3.connect(h.shadow_path)
    with pytest.raises(sqlite3.DatabaseError):  # a price with MARKET_UNAVAILABLE is refused
        raw.execute(
            "INSERT INTO shadow_trades SELECT NULL, 'x', run_id, strategy_id, strategy_version, "
            "position_id, 9, final, asset_id, chain, address, symbol, pool, entry_decision_id, "
            "exit_decision_id, entry_at, entry_price, exit_at, 1.0, NULL, NULL, quantity, "
            "fraction, cost_usd, NULL, NULL, NULL, mfe_pct, mae_pct, holding_minutes, "
            "exit_reason, NULL, price_basis, execution_model, recorded_at FROM shadow_trades"
        )
    raw.close()


def test_stale_pool_becomes_market_unavailable(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=60), 1.01)
    h.price(T0 + timedelta(minutes=300), 1.5, mint=B)  # time passes, nothing for our pool
    h.run()
    (t,) = h.trades()
    assert t["exit_reason"] == "MARKET_UNAVAILABLE" and t["exit_price"] is None
    assert t["exit_at"] == (T0 + timedelta(minutes=180)).timestamp()  # last price + 120 min


def test_exact_token_identity(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0, symbol="SAME"))
    h.price(T0 + timedelta(minutes=5), 0.1, mint=B)  # another token, same ticker
    h.scan(T0 + timedelta(minutes=6), h.cand(B, price=0.1, symbol="SAME", stage="FADING"))
    h.price(T0 + timedelta(minutes=7), 5.0, mint=A, pool="another-pool-of-A")
    h.run()
    assert not h.trades()
    (p,) = h.positions(asset_id=f"solana:{A}")
    assert p["status"] == "OPEN" and p["last_price"] == 1.0


# --- 9. random baseline ------------------------------------------------------------------------------


def test_random_baseline_is_deterministic(tmp_path: Path, template: Any) -> None:
    rnd = strat("rnd", entry={"random_fraction": 0.3, "random_seed": 11})
    picked = []
    for k in range(2):
        hh = Harness(tmp_path / str(k), template)
        mints = [f"Mint{i:04d}" for i in range(40)]
        hh.scan(T0, *[hh.cand(m) for m in mints])
        hh.run(rnd, strat("rnd2", entry={"random_fraction": 0.3, "random_seed": 12}),
               strat("all"))  # fmt: skip
        picked.append({(p["strategy_id"], p["asset_id"]) for p in hh.positions()})
        hh.close()
    assert picked[0] == picked[1]
    rnd_assets = {a for s, a in picked[0] if s == "rnd"}
    assert 4 <= len(rnd_assets) <= 20
    assert rnd_assets != {a for s, a in picked[0] if s == "rnd2"}


# --- accounting ---------------------------------------------------------------------------------------


def test_portfolio_accounting_realized_unrealized_and_drawdown(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, price=2.0))
    h.price(T0 + timedelta(minutes=10), 1.25, mint=A)  # A: take profit +250
    h.scan(T0 + timedelta(minutes=20), h.cand(B, price=1.9))  # B: marked down -50
    h.run()
    eq = h.store.equity("t")
    last = eq[-1]
    assert last["cash"] == pytest.approx(10_000 - 2000 + 1250)
    assert last["open_value"] == pytest.approx(500 * 1.9)
    assert last["equity"] == pytest.approx(last["cash"] + last["open_value"])
    assert last["realized_pnl"] == pytest.approx(250.0)
    assert last["unrealized_pnl"] == pytest.approx(-50.0)
    assert last["exposure_pct"] == pytest.approx(950 / last["equity"] * 100)
    h.scan(T0 + timedelta(minutes=30), h.cand(B, price=1.85))
    h.engine.run("t")
    last = h.store.equity("t")[-1]
    # Peak equity 10,200 (after the take profit, B at 1.9); now 9,250 + 925 = 10,175.
    assert last["equity"] == pytest.approx(10_175.0)
    assert last["drawdown_pct"] == pytest.approx((10_175 / 10_200 - 1) * 100)
    assert last["max_drawdown_pct"] == pytest.approx(last["drawdown_pct"])


def test_metrics_math() -> None:
    assert trimmed_mean([1, 2, 3, 4, 100, 5, 6, 7, 8, 9]) == pytest.approx(5.5)
    assert effective_sample_size(["a", "b", "c"]) == 3
    assert effective_sample_size(["a"] * 4) == 1
    assert effective_sample_size(["a", "a", "b", "b"]) == 2


def test_metrics_report(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, price=1.0))
    h.price(T0 + timedelta(minutes=10), 1.3, mint=A)
    h.price(T0 + timedelta(minutes=10), 0.85, mint=B)
    h.run()
    m = h.engine.metrics("t")
    assert "not real profit" in m["label"].lower()
    s = m["strategies"][0]
    assert s["trades"] == 2 and s["distinct_assets"] == 2 and s["win_rate"] == 0.5
    assert s["profit_factor"] == pytest.approx(300 / 150)
    assert s["sample"] == "INSUFFICIENT_SAMPLE" and s["days_represented"] == 1
    assert s["median_mfe_pct"] == pytest.approx(15.0) and s["effective_sample_size"] == 2
    only_a = h.engine.metrics("t", asset_id=f"solana:{A}")["strategies"][0]
    assert only_a["trades"] == 1 and only_a["profit_factor"] is None


# --- isolation ---------------------------------------------------------------------------------------


def test_multiple_strategies_are_isolated(h: Harness) -> None:
    low, high = strat("low"), strat("high", entry={"min_scout_score": 90.0})
    h.scan(T0, h.cand(A, score=80.0), h.cand(B, score=95.0))
    report = h.run(low, high)
    assert {(p["strategy_id"], p["asset_id"]) for p in h.positions()} == {
        ("low", f"solana:{A}"), ("low", f"solana:{B}"), ("high", f"solana:{B}"),
    }  # fmt: skip
    assert report["books"]["low@v1"]["cash"] == 8000.0
    assert report["books"]["high@v1"]["cash"] == 9000.0


def test_shadow_database_is_separate_and_evidence_read_only(h: Harness) -> None:
    h.scan(T0, h.cand())
    before = h.writer.totals()["total"]
    h.run()
    assert h.writer.totals()["total"] == before  # nothing written to the archive
    assert h.reader.read_only
    tables = {r[0] for r in sqlite3.connect(h.shadow_path).execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}  # fmt: skip
    assert {"shadow_strategies", "shadow_runs", "shadow_decisions", "shadow_positions",
            "shadow_trades", "shadow_equity", "shadow_metrics", "shadow_meta"} <= tables  # fmt: skip
    assert not any(t.startswith(("scout_", "evidence_", "calibration_")) for t in tables)
    assert default_shadow_db().endswith("shadow.sqlite3")


def test_default_db_path_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSCALE_SHADOW_DB", "/data/shadow.sqlite3")
    assert default_shadow_db() == "/data/shadow.sqlite3"
    monkeypatch.delenv("UPSCALE_SHADOW_DB")
    monkeypatch.setenv("UPSCALE_SCOUT_DB", "/somewhere/scout.sqlite3")
    assert default_shadow_db() == "/somewhere/shadow.sqlite3"


# --- immutability -------------------------------------------------------------------------------------


def test_completed_trades_and_positions_are_immutable(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, price=1.0))
    h.price(T0 + timedelta(minutes=10), 1.3, mint=A)
    h.run()
    raw = sqlite3.connect(h.shadow_path)
    for sql in ("UPDATE shadow_trades SET exit_price = 2", "DELETE FROM shadow_trades",
                "UPDATE shadow_decisions SET reason = 'x'", "DELETE FROM shadow_decisions",
                "UPDATE shadow_equity SET equity = 0", "UPDATE shadow_runs SET since = 0",
                f"UPDATE shadow_positions SET last_price = 9 WHERE asset_id = 'solana:{A}'",
                f"UPDATE shadow_positions SET entry_price = 9 WHERE asset_id = 'solana:{B}'",
                "DELETE FROM shadow_positions"):  # fmt: skip
        with pytest.raises(sqlite3.DatabaseError):
            raw.execute(sql)
    # An OPEN position's marks may move; its entry never does.
    raw.execute(f"UPDATE shadow_positions SET last_price = 1.1 WHERE asset_id = 'solana:{B}'")
    raw.close()


# --- 18. anti-lookahead ----------------------------------------------------------------------------


def _scenario(h: Harness) -> None:
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, price=1.0))
    h.price(T0 + timedelta(minutes=20), 1.1, mint=A)
    h.scan(T0 + timedelta(minutes=40), h.cand(A, price=1.3), h.cand(B, price=0.95))
    h.scan(T0 + timedelta(minutes=80), h.cand("MintC", price=1.0), h.cand(B, price=0.85))
    h.price(T0 + timedelta(minutes=100), 1.5, mint="MintC")
    h.scan(T0 + timedelta(minutes=150), h.cand(A, price=1.4), h.cand(B, stage="FADING", price=0.9))


def _rows(h: Harness) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    ds = [(d["decision_id"], d["action"], d["asset_id"], d["decision_at"], d["reference_price"])
          for d in h.decisions()]  # fmt: skip
    ts = [(t["trade_id"], t["exit_reason"], t["exit_price"], t["exit_at"]) for t in h.trades()]
    return ds, ts


def test_future_evidence_never_changes_past_decisions(tmp_path: Path, template: Any) -> None:
    cut = T0 + timedelta(minutes=85)
    prefix = Harness(tmp_path / "prefix", template)
    _scenario(prefix)
    prefix.run(until=cut)
    full = Harness(tmp_path / "full", template)
    _scenario(full)
    full.run()
    pd, pt = _rows(prefix)
    fd, ft = _rows(full)
    early = [d for d in fd if d[3] <= cut.timestamp()]
    assert pd == early and pd  # identical decisions before the cut
    assert pt == [t for t in ft if t[3] <= cut.timestamp()]
    assert len(fd) > len(pd)
    for x in (prefix, full):
        x.close()


def test_future_prices_and_candles_are_never_used(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    import upscale.services.geckoterminal as gt

    async def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("shadow must never request candles")

    monkeypatch.setattr(gt.GeckoTerminalProvider, "fetch_pool_candles", forbidden)
    monkeypatch.setattr(gt.DexCandleService, "get_pool_candles", forbidden, raising=False)
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=10), 0.95)
    h.run(until=T0 + timedelta(minutes=15))
    h.price(T0 + timedelta(minutes=20), 3.0)  # the future, archived later
    (d,) = h.decisions(action="ENTER")
    assert json.loads(d["evidence_json"])["scout"]["price_usd"] == 1.0
    (p,) = h.positions()
    assert p["last_price"] == 0.95 and p["peak_price"] == 1.0  # nothing after the limit


def test_lookahead_guards(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    view = scout_view(h_first_scout(h))
    later = view.decision_at + timedelta(minutes=1)
    # Analyze evidence later than the decision time is refused, never used.
    book = Book("x", strat(entry={"analyze": {"max_age_minutes": 60}}))
    future = AnalyzeView("r", "f", later, "buy", "high", "low", None, None, None)
    with pytest.raises(AssertionError, match="later than the decision time"):
        book.on_event(Event(at=view.decision_at, seq=1, scout=view), lambda *_: future)
    # The ordered archive scan never returns a record after its limit.
    assert h.reader.after(("scout",), (0.0, 0), view.decision_at - timedelta(seconds=1)) == []
    # A price observed no later than the position's last price never rewinds it.
    plain = Book("y", strat())
    plain.on_event(Event(at=view.decision_at, seq=1, scout=view), lambda *_: None)
    (p,) = plain.positions.values()
    plain.on_price(PriceObs(view.asset_id, p.pool, 9.0, p.last_price_at, "market", "r"), later)
    assert p.last_price == 1.0
    # The engine refuses a stream that goes back in time.
    events = [Event(at=later, seq=2, scout=view), Event(at=view.decision_at, seq=3, scout=view)]
    monkeypatch.setattr(EvidenceTimeline, "events", lambda *_: iter(events))
    with pytest.raises(LookaheadError, match="back in time"):
        h.run()


def h_first_scout(h: Harness) -> Any:
    h.scan(T0, h.cand(price=1.0))
    (r,) = h.reader.records(kind="scout")
    return r


def test_causally_invalid_or_legacy_scout_records_are_skipped(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    (r,) = h.reader.records(kind="scout")
    bad = PendingRecord(kind="scout", asset_id=f"solana:{B}", chain="solana", address=B,
                        pool_address=f"pool-{B}", observed_at=T0 + timedelta(minutes=1),
                        payload={**r.payload, "timing": {"version": 1}},
                        links={"causal_valid": True})  # fmt: skip
    worse = PendingRecord(kind="scout", asset_id="solana:MintC", chain="solana", address="MintC",
                          pool_address="pool-MintC", observed_at=T0 + timedelta(minutes=2),
                          payload=r.payload, links={"causal_valid": False})  # fmt: skip
    h._observed = (T0 + timedelta(minutes=2)).timestamp()
    h.writer.append(bad)
    h.writer.append(worse)
    h.run()
    assert [p["asset_id"] for p in h.positions()] == [f"solana:{A}"]
    counts = h.store.checkpoint("t")["books"]["t@v1"]["state"]["counts"]
    assert counts["skipped:legacy_scout_timing"] == 1
    assert counts["skipped:causally_invalid_scout_record"] == 1


# --- restart / resume ----------------------------------------------------------------------------------


def test_restart_resume_matches_a_single_run(tmp_path: Path, template: Any) -> None:
    single = Harness(tmp_path / "single", template)
    _scenario(single)
    single.run()
    staged = Harness(tmp_path / "staged", template)
    _scenario(staged)
    staged.now = T0 + timedelta(minutes=50)  # the archive "now": only part is processable
    staged.run()
    staged.reopen()  # a restart: new connections, state from the checkpoint
    staged.now = None  # the same final clock as the single run
    staged.engine.run("t")
    staged.reopen()
    report = staged.engine.run("t")  # nothing new: a no-op
    assert report["events"] == 0
    assert _rows(staged) == _rows(single)
    assert staged.store.checkpoint("t")["books"] == single.store.checkpoint("t")["books"]
    for x in (single, staged):
        x.close()


def test_resume_refuses_an_inconsistent_checkpoint(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.run()
    raw = sqlite3.connect(h.shadow_path)
    raw.execute("UPDATE shadow_checkpoints SET books_json = '{}'")
    raw.commit()
    raw.close()
    with pytest.raises(ShadowStoreError, match="disagree"):
        h.engine.run("t")


def test_late_archived_evidence_is_counted_not_replayed(h: Harness) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.run()
    h.archive_override = (T0 + timedelta(hours=2)).timestamp()  # archived after the run passed T0
    h.price(T0 + timedelta(seconds=30), 9.0)
    report = h.engine.run("t")
    assert report["late_evidence_ignored"] == 1
    (p,) = h.positions()
    assert p["last_price"] == 1.0


# --- 12-13. background service and priority ----------------------------------------------------------


def test_background_shadow_is_off_by_default() -> None:
    assert load_settings(None).enabled is False
    assert load_settings("0").enabled is False
    assert load_settings("1").enabled is True
    s = load_settings("1", "1", "bad id!", None)
    assert s.interval_minutes == 5.0 and s.run_id == "production"
    assert upscale.main.background_shadow.settings.enabled is False


def test_background_shadow_yields_to_production(monkeypatch: pytest.MonkeyPatch) -> None:
    reasons = iter(["Analyze is active", "a Scout scan is running", None])
    calls: list[str] = []

    def fake_step(*_: Any) -> dict[str, Any]:
        calls.append("step")
        return {"events": 0, "actions": {}, "fills": 0, "until": None}

    monkeypatch.setattr("upscale.services.shadow.service.step", fake_step)
    svc = BackgroundShadow(load_settings("1"), lambda: next(reasons))
    results = [asyncio.run(svc.run_once()) for _ in range(3)]
    assert [r["status"] for r in results] == ["deferred", "deferred", "completed"]
    assert calls == ["step"]
    assert svc.status()["runs_deferred"] == 2


def test_main_defer_reason_priorities(monkeypatch: pytest.MonkeyPatch) -> None:
    m = upscale.main
    monkeypatch.setattr(m, "_interactive", 1)
    assert m._shadow_defer_reason() == "Analyze is active"
    monkeypatch.setattr(m, "_interactive", 0)
    monkeypatch.setattr(m, "_last_analyze", -1e9)
    monkeypatch.setattr(m.background_scout, "running", True)
    assert m._shadow_defer_reason() == "a Scout scan is running"
    monkeypatch.setattr(m.background_scout, "running", False)
    monkeypatch.setattr(m.services.safety_enrichment, "running", True)
    assert m._shadow_defer_reason() == "safety enrichment is running"
    monkeypatch.setattr(m.services.safety_enrichment, "running", False)
    assert m._shadow_defer_reason() is None


def test_a_failing_step_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: Any) -> dict[str, Any]:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("upscale.services.shadow.service.step", broken)
    svc = BackgroundShadow(load_settings("1"), lambda: None)
    result = asyncio.run(svc.run_once())
    assert result["status"] == "failed" and "disk I/O error" in result["error"]


def test_background_step_uses_no_provider_quota(tmp_path: Path, template: Any) -> None:
    import upscale.services as services
    from upscale.services.shadow.service import step

    hh = Harness(tmp_path, template)
    hh.scan(T0, hh.cand(safety="SAFETY_CHECKS_COMPLETE"))
    hh.close()
    gates = services.scout_service.gates()
    before = [(g.name, g.available("default")) for g in gates]
    report = step(load_settings("1", since="2026-09-30T23:00:00Z"),
                  str(tmp_path / "bg-shadow.sqlite3"), str(hh.ev_path),
                  now=lambda: T0 + timedelta(hours=1))  # fmt: skip
    # Baselines A, B and C qualify; D's deterministic draw decides for itself.
    assert 3 <= report["actions"]["ENTER"] <= 4 and report["clean_data"]
    assert [(g.name, g.available("default")) for g in gates] == before


# --- 15. Calibration SHADOW integration ---------------------------------------------------------------


def test_calibration_sees_shadow_separately(h: Harness, tmp_path: Path) -> None:
    h.scan(T0, h.cand(A, price=1.0), h.cand(B, price=1.0))
    h.price(T0 + timedelta(minutes=10), 1.3, mint=A)
    h.price(T0 + timedelta(minutes=10), 0.8, mint=B)
    holdout_day = T0 + timedelta(days=4)  # 2026-10-05: a live HOLDOUT day
    h.scan(holdout_day, h.cand("MintC", price=1.0))
    h.price(holdout_day + timedelta(minutes=5), 1.5, mint="MintC")
    h.run()
    policy = CalibrationConfig().live_split
    visible = load_shadow(h.shadow_path, policy)
    assert {o.asset_id for o in visible} == {f"solana:{A}", f"solana:{B}"}
    assert all(o.origin == "SHADOW" and o.kind == "shadow" for o in visible)
    assert sorted(o.outcomes["trade"].return_pct or 0 for o in visible) == pytest.approx(
        [-20.0, 30.0]
    )
    assert len(load_shadow(h.shadow_path, policy, include_holdout=True)) == 3
    engine = CalibrationEngine(CalibrationStore(tmp_path / "cal.sqlite3"), None, None,
                               shadow_db=str(h.shadow_path))  # fmt: skip
    ready = engine.readiness()
    assert ready["origins"]["SHADOW"]["closed_trades"] == 2
    assert ready["origins"]["LIVE_FORWARD"]["observations"] == 0
    assert ready["calibration_eligible"]["observations"] == 0  # never merged into splits
    comparison = engine.compare_origins("1h")
    assert comparison["SHADOW"]["t@v1"]["measured"] == 2
    assert comparison["LIVE_FORWARD"]["measured"] == 0
    engine.store.close()


def test_calibration_ignores_contaminated_research_runs(h: Harness) -> None:
    h.store.register(strat())
    h.engine.ensure_run("research", CLEAN_DATA_CUTOFF - timedelta(days=1), None, ["t"],
                        allow_contaminated=True)  # fmt: skip
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=10), 1.3)
    h.engine.run("research")
    assert h.store.trades(run_id="research")
    assert load_shadow(h.shadow_path, CalibrationConfig().live_split) == []


# --- 17. CLI and API ---------------------------------------------------------------------------------


def test_cli(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    h.scan(T0, h.cand(price=1.0))
    h.price(T0 + timedelta(minutes=10), 1.3)
    base = [
        "--db",
        str(tmp := h.shadow_path.parent / "cli.sqlite3"),
        "--evidence-db",
        str(h.ev_path),
    ]
    assert cli_main([*base, "init"]) == 0
    assert "scout_threshold@v1: created" in capsys.readouterr().out
    assert cli_main([*base, "run", "--since", "2026-09-01T00:00:00Z"]) == 1  # contaminated
    assert "clean-data cutoff" in capsys.readouterr().err
    assert cli_main([*base, "run"]) == 0
    capsys.readouterr()
    for cmd in (["status"], ["strategies"], ["positions", "--closed"], ["positions", "--open"],
                ["trades", "--strategy", "scout_threshold"], ["decisions", "--action", "ENTER"],
                ["metrics"], ["metrics", "--asset", f"solana:{A}"]):  # fmt: skip
        assert cli_main([*base, *cmd]) == 0, cmd
        json.loads(capsys.readouterr().out)
    assert tmp.exists()
    missing = ["--db", str(tmp), "--evidence-db", str(h.shadow_path.parent / "none.sqlite3")]
    assert cli_main([*missing, "run"]) == 1
    assert "no evidence archive" in capsys.readouterr().err


def test_api_is_read_only(h: Harness, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get("/shadow/status").json()["runs"] == []  # no database yet: empty
    h.scan(T0, h.cand(price=1.0))
    h.run(run_id="production")
    monkeypatch.setenv("UPSCALE_SHADOW_DB", str(h.shadow_path))
    for path in ("status", "strategies", "positions", "trades", "decisions", "metrics"):
        r = client.get(f"/shadow/{path}")
        assert r.status_code == 200, path
    assert client.get("/shadow/positions?status=OPEN").json()[0]["asset_id"] == f"solana:{A}"
    assert client.get("/shadow/metrics").json()["strategies"][0]["positions_opened"] == 1
    assert client.get("/shadow/background/status").json()["enabled"] is False
    paths = {p: set(ops) for p, ops in client.get("/openapi.json").json()["paths"].items()
             if p.startswith("/shadow")}  # fmt: skip
    assert all(ops == {"get"} for ops in paths.values())
