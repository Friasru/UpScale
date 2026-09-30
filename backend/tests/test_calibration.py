"""Calibration Engine v1: origin separation, split and HOLDOUT protection, sample rules,
analyses, bounded candidates, lineage and immutability. Temporary databases only."""

import asyncio
import hashlib
import io
import random
import sqlite3
import tokenize
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import upscale.services as services
from upscale.services.calibration import analysis
from upscale.services.calibration.candidates import evaluate, judge
from upscale.services.calibration.cli import main as cli_main
from upscale.services.calibration.config import (
    CalibrationConfig,
    CorrelationRules,
    LiveSplitPolicy,
    SampleRules,
)
from upscale.services.calibration.dataset import (
    HoldoutSealedError,
    Observation,
    OutcomeView,
    diversify,
    live_purged,
    live_split,
    load_live,
    load_replay,
    with_regimes,
)
from upscale.services.calibration.engine import CalibrationEngine, CalibrationError
from upscale.services.calibration.stats import cohort, strength
from upscale.services.calibration.store import CalibrationStore
from upscale.services.outcomes import record_scout_run
from upscale.services.outcomes.config import OutcomeConfig
from upscale.services.outcomes.models import PricePath
from upscale.services.outcomes.store import HorizonUpdate, OutcomeStore
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.strategy import STRATEGIES

from .test_outcomes import ranked
from .test_replay_lab import Harness, build_archive, config

START = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)  # a CALIBRATION day (cycle day 19? computed below)
W = analysis.production_weights()


def outcome_view(ret: float) -> OutcomeView:
    return OutcomeView(status="COMPLETE", return_pct=ret, mfe_pct=max(ret, 0) + 2, mae_pct=min(ret, 0) - 1,
                       max_drawdown_pct=min(ret, 0) - 2, time_to_mfe_minutes=10, liquidity_change_pct=0.0,
                       market_status="ACTIVE")  # fmt: skip


def make(
    asset: str, at: datetime, ret: float, *, origin: str = "LIVE_FORWARD", split: str = "CALIBRATION",
    kind: str = "scout", **features: Any,
) -> Observation:  # fmt: skip
    fs = {"market_activity": 0.6, "liquidity_quality": 0.5, "social_momentum": 0.5, "earliness": 0.5,
          "cross_confirmation": 0.5}  # fmt: skip
    fs.update(features.pop("family_scores", {}))
    feats: dict[str, Any] = {f"family_{k}": 100 * v * W[k] for k, v in fs.items()}
    feats |= {"stage_adjustment": 0.0, "risk_penalty": 0.0, "safety_status": "SAFETY_CHECKS_COMPLETE",
              "social_status": "SOCIAL_QUIET", "market_cap_usd": 1e6, "liquidity_usd": 50_000.0,
              "technical_trend": "uptrend"}  # fmt: skip
    feats |= features
    feats.setdefault("score", sum(feats[f"family_{k}"] for k in fs))
    from upscale.services.calibration.dataset import _missing

    key = f"{origin}:{asset}:{at.isoformat()}"
    return Observation(
        origin=origin, kind=kind, key=key, asset_id=asset, at=at, split=split, purged=False,  # type: ignore[arg-type]
        features=feats, flags=frozenset(features.get("flags", [])), missing=_missing(feats),
        outcomes={h: outcome_view(ret * m) for h, m in (("5m", 0.2), ("15m", 0.4), ("1h", 1.0), ("4h", 1.5), ("24h", 2.0))},
    )  # fmt: skip


def synthetic(
    n_assets: int = 40, per_asset: int = 3, split: str = "CALIBRATION", origin: str = "LIVE_FORWARD",
    seed: int = 1, start: datetime = START, kind: str = "scout",
) -> list[Observation]:  # fmt: skip
    """Deep pools and ACCELERATING do better; spaced 2h apart per asset."""
    rng = random.Random(seed)
    out = []
    for a in range(n_assets):
        deep = a % 2 == 0
        stage = ("ACCELERATING", "EARLY", "STEADY")[a % 3]
        for k in range(per_asset):
            ret = (
                (3.0 if deep else -3.0)
                + (2.0 if stage == "ACCELERATING" else 0.0)
                + rng.gauss(0, 0.5)
            )
            out.append(make(
                f"solana:asset{a}", start + timedelta(hours=2 * k, minutes=a), ret, origin=origin, split=split,
                stage=stage, liquidity_usd=150_000.0 if deep else 15_000.0,
                family_scores={"liquidity_quality": 0.8 if deep else 0.2},
                price_change_h24_pct=10.0 if deep else 150.0, volume_acceleration=2.5,
                holder_top10_pct=25.0 if deep else 60.0, action="buy" if deep else "wait",
                confidence="high" if deep else "low", kind=kind,
            ))  # fmt: skip
    return out


class SyntheticEngine(CalibrationEngine):
    def __init__(
        self,
        store: CalibrationStore,
        observations: list[Observation],
        cfg: CalibrationConfig | None = None,
    ):
        super().__init__(store, None, None, cfg, now=lambda: datetime(2026, 9, 30, tzinfo=UTC),
                         triggered_by=lambda: "test")  # fmt: skip
        self.observations = observations
        self.loads: list[bool] = []

    def _load(self, include_holdout: bool = False) -> list[Observation]:
        self.loads.append(include_holdout)
        obs = [o for o in self.observations if include_holdout or o.split != "HOLDOUT"]
        return with_regimes(obs)

    def holdout_index(self) -> list[tuple[str, datetime]]:
        return [(o.key, o.at) for o in self.observations if o.split == "HOLDOUT"]


def full_dataset() -> list[Observation]:
    return (
        synthetic(60, 3, "CALIBRATION")
        + synthetic(30, 2, "VALIDATION", seed=2, start=START + timedelta(days=20))
        + synthetic(30, 2, "HOLDOUT", seed=3, start=START + timedelta(days=40))
    )


@pytest.fixture
def engine(tmp_path: Path) -> SyntheticEngine:
    return SyntheticEngine(CalibrationStore(tmp_path / "calibration.sqlite3"), full_dataset())


# --- splits and HOLDOUT --------------------------------------------------------------------------


def test_live_split_is_chronological_sticky_and_purged_at_boundaries() -> None:
    p = LiveSplitPolicy()
    anchor = datetime(2026, 1, 1, 12, tzinfo=UTC)
    splits = [live_split(anchor + timedelta(days=d), p) for d in range(20)]
    assert splits == ["CALIBRATION"] * 14 + ["VALIDATION"] * 3 + ["HOLDOUT"] * 3
    assert live_split(anchor + timedelta(days=20), p) == "CALIBRATION"  # the next cycle
    assert not live_purged(anchor, p) and live_purged(anchor + timedelta(days=13), p)
    with pytest.raises(ValueError):
        LiveSplitPolicy(cycle_days=10, calibration_days=8, validation_days=2)


def test_calibration_never_reads_holdout(engine: SyntheticEngine) -> None:
    engine.analyze("1h")
    engine.create_candidates("1h")
    engine.readiness()
    assert engine.loads and not any(engine.loads)  # every load excluded HOLDOUT

    class Leaky(SyntheticEngine):
        def _load(self, include_holdout: bool = False) -> list[Observation]:
            obs = super()._load(True)
            if not include_holdout and any(o.split == "HOLDOUT" for o in obs):
                raise HoldoutSealedError("HOLDOUT observations reached a calibration step")
            return obs

    leaky = Leaky(engine.store, engine.observations)
    with pytest.raises(HoldoutSealedError):
        leaky.analyze("1h")


def test_final_evaluation_gate_logging_and_freeze(engine: SyntheticEngine) -> None:
    _, ids = engine.create_candidates("1h")
    assert ids
    cid = ids[0]
    with pytest.raises(CalibrationError):
        engine.final_evaluate(cid, confirm=True)  # not validated yet
    engine.validate(cid)
    assert engine.store.candidate(cid)["status"] == "VALIDATED"  # type: ignore[index]
    engine.loads.clear()
    with pytest.raises(HoldoutSealedError):
        engine.final_evaluate(cid, confirm=False)
    assert engine.loads == [] and engine.store.holdout_accesses() == []  # nothing read or logged
    params = engine.store.candidate(cid)["changes"]  # type: ignore[index]
    result = engine.final_evaluate(cid, confirm=True)
    [log] = engine.store.holdout_accesses()
    assert (
        log["candidate_id"] == cid
        and log["triggered_by"] == "test"
        and log["window"]["observations"] == 60
    )
    assert log["data_version"] and log["purpose"] == "final evaluation"
    c = engine.store.candidate(cid)
    assert c is not None and c["status"] == "FROZEN_FOR_FINAL_TEST" and c["changes"] == params
    assert result["candidate"]["selected"]["measured"] > 0
    with pytest.raises(CalibrationError):
        engine.validate(cid)  # results never feed back into the candidate
    with pytest.raises(CalibrationError):
        engine.record_manual_promotion(cid, confirm=False)
    assert engine.store.candidate(cid)["status"] == "FROZEN_FOR_FINAL_TEST"  # type: ignore[index]


# --- origins, samples, correlation, missing evidence -------------------------------------------------------


def test_live_and_replay_are_reported_separately(tmp_path: Path) -> None:
    obs = synthetic(30, 2) + synthetic(30, 2, origin="HISTORICAL_REPLAY", seed=5)
    rows = analysis.feature_rows(obs, "1h", CalibrationConfig(), "CALIBRATION")
    stage = {(r.origin, r.bucket): r.cohort for r in rows if r.feature == "stage"}
    assert stage[("LIVE_FORWARD", "ACCELERATING")].origins == {"LIVE_FORWARD": 20}
    assert stage[("HISTORICAL_REPLAY", "ACCELERATING")].origins == {"HISTORICAL_REPLAY": 20}
    assert stage[("COMBINED", "ACCELERATING")].origins == {
        "LIVE_FORWARD": 20,
        "HISTORICAL_REPLAY": 20,
    }
    replay_only = synthetic(40, 3, origin="HISTORICAL_REPLAY")
    f = analysis.compare_buckets(replay_only, [], "liquidity_usd", "1h", CalibrationConfig())
    assert f is not None and any("replay-only" in n for n in f.notes)


def test_minimum_sample_and_asset_diversity_rules() -> None:
    cfg = CalibrationConfig()
    few_assets = [make(f"solana:a{i % 2}", START + timedelta(hours=i), 1.0) for i in range(30)]
    c = cohort("x", few_assets, "1h", cfg)
    assert (
        c.status == "INSUFFICIENT_SAMPLE"
        and c.measured == 30
        and c.assets == 2
        and c.median_return is None
    )
    enough = [make(f"solana:a{i}", START, 1.0 + i / 10) for i in range(25)]
    ok = cohort("y", enough, "1h", cfg)
    assert ok.status == "OK" and ok.p10 is None and ok.median_ci is not None  # p10 needs 50
    label, rule = strength(
        ok, cfg.samples, horizons_agree=5, validation_agrees=True, origins_agree=None
    )
    assert label == "WEAK_EVIDENCE" and "candidate sample" in rule
    many = cohort("z", [make(f"solana:a{i}", START, 1.0) for i in range(120)], "1h", cfg)
    assert strength(many, cfg.samples, 3, True, True)[0] == "STRONGER_EVIDENCE"
    assert strength(many, cfg.samples, 3, None, True)[0] == "WEAK_EVIDENCE"
    assert strength(many, cfg.samples, 1, True, True)[0] == "WEAK_EVIDENCE"
    with pytest.raises(ValueError):
        SampleRules(descriptive_n=60, candidate_n=50)


def test_correlated_observations_are_thinned_and_weighted() -> None:
    burst = [make("solana:busy", START + timedelta(minutes=10 * i), 1.0) for i in range(30)]
    kept = diversify(burst, CorrelationRules(max_per_asset=3, min_spacing_minutes=60))
    assert len(kept) == 3
    assert all(b.at - a.at >= timedelta(minutes=60) for a, b in zip(kept, kept[1:], strict=False))
    mixed = [make("solana:busy", START + timedelta(hours=i), 1.0) for i in range(10)] + [
        make(f"solana:{i}", START, 1.0) for i in range(10)
    ]
    c = cohort("m", mixed, "1h", CalibrationConfig())
    assert c.measured == 20 and c.assets == 11 and c.effective_n < 20 and c.max_per_asset == 10


def test_missing_evidence_is_a_label_not_zero() -> None:
    o = make("solana:m", START, 1.0, safety_status=None, liquidity_usd=None, market_cap_usd=None)
    assert {
        "SAFETY_NOT_AVAILABLE",
        "LIQUIDITY_NOT_AVAILABLE",
        "MARKET_CAP_NOT_AVAILABLE",
    } <= o.missing
    rows = analysis.feature_rows([o], "1h", CalibrationConfig(), "CALIBRATION")
    liq = [r for r in rows if r.feature == "liquidity_usd" and r.origin == "COMBINED"]
    assert [r.bucket for r in liq] == ["unavailable"]  # never bucketed as $0
    obs = [make(f"solana:{i}", START, 2.0, safety_status=None) for i in range(25)] + [
        make(f"solana:x{i}", START, -2.0) for i in range(25)
    ]
    gap = analysis.missing_evidence(obs, "1h", CalibrationConfig())["SAFETY_NOT_AVAILABLE"]
    assert gap["return_gap_pp"] == pytest.approx(4.0)


# --- analyses ------------------------------------------------------------------------------------------------


def test_stage_and_confidence_calibration() -> None:
    cal = synthetic(60, 3)
    stages = analysis.stage_calibration(cal, [], "1h", CalibrationConfig())
    assert stages["questions"]["ACCELERATING stronger continuation than EARLY"].startswith("yes")
    assert stages["stages"]["CROWDED"]["status"] == "INSUFFICIENT_SAMPLE"
    decisions = synthetic(60, 3, kind="decision")
    conf = analysis.confidence_calibration(cal + decisions, "1h", CalibrationConfig())
    assert conf["opportunity_confidence_verdict"].startswith("SEPARATES")
    assert conf["scout_score_rank_correlation"] > 0.5
    inverted = [make(o.asset_id, o.at, -(o.outcomes["1h"].return_pct or 0), confidence=o.s("confidence"),
                     action=o.s("action"), kind="decision") for o in decisions]  # fmt: skip
    assert analysis.confidence_calibration(inverted, "1h", CalibrationConfig())[
        "opportunity_confidence_verdict"].startswith("INVERTED")  # fmt: skip


def test_feature_findings_use_association_language_and_validation() -> None:
    cal, val = synthetic(60, 3), synthetic(30, 2, "VALIDATION", seed=2)
    f = analysis.compare_buckets(cal, val, "liquidity_usd", "1h", CalibrationConfig())
    assert f is not None and f.subject.startswith("liquidity_usd: 100000..250000")
    assert "associated with" in f.statement and "cause" not in f.statement.lower()
    assert f.validation is not None and f.validation["agrees"] is True
    # 90 measured from 30 assets: candidate-size (not strong), all horizons and validation agree.
    assert f.strength == "MODERATE_EVIDENCE" and f.origins["horizons_agreeing"] == 5
    patterns, tested = analysis.interactions(cal, "1h", CalibrationConfig(max_patterns=10))
    assert tested <= 10 and all(p["cohort"]["measured"] >= 20 for p in patterns)


def test_threshold_sensitivity_reports_retention_and_diversity() -> None:
    rows = analysis.threshold_sensitivity(synthetic(60, 3), "1h", CalibrationConfig())[
        "min_liquidity_usd"
    ]
    assert [r["value"] for r in rows] == [10_000.0, 20_000.0, 50_000.0, 100_000.0]
    assert rows[0]["retention"] == 1.0 and rows[-1]["retention"] == pytest.approx(0.5, abs=0.05)
    assert rows[-1]["diversity_loss"] == pytest.approx(0.5, abs=0.05)
    assert rows[-1]["return_vs_base_pp"] > 0 and rows[-1]["false_filter_risk"] is not None


def test_weight_perturbations_are_bounded_and_normalized() -> None:
    variants = analysis.weight_variants(CalibrationConfig())
    assert len(variants) == 1 + 5 * 4 and variants[0] == ("production", W)
    for _, w in variants:
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)
        assert all(abs(w[k] / W[k] - 1) <= 0.11 for k in W)
    o = make("solana:s", START, 1.0)
    assert analysis.rescore(o, W) == pytest.approx(o.f("score"))


# --- candidates -----------------------------------------------------------------------------------------------


def test_candidates_are_immutable_with_lineage(engine: SyntheticEngine) -> None:
    run_id, ids = engine.create_candidates("1h")
    assert ids
    c = engine.store.candidate(ids[0])
    assert c is not None and c["status"] == "CALIBRATED" and c["source_run_id"] == run_id
    assert c["complexity"]["level"] == "low" and c["complexity"]["changes"] == 1
    assert (
        c["parent_version"]["evaluation_horizon"] == "1h" and "scout_scoring" in c["parent_version"]
    )
    assert "variants tested" in c["reason"]
    again_run, again = engine.create_candidates("1h")
    assert again == ids and len(engine.store.candidates()) == len(ids)  # same change: same id
    db = sqlite3.connect(engine.store.path)
    for sql in ("UPDATE calibration_candidates SET changes_json = '[]'",
                "UPDATE calibration_candidates SET status = 'DRAFT'",
                "UPDATE calibration_candidates SET status = 'PROMOTED_MANUALLY'",
                "DELETE FROM calibration_candidates",
                "UPDATE candidate_metrics SET metrics_json = '{}'", "DELETE FROM candidate_lineage"):  # fmt: skip
        with pytest.raises(sqlite3.DatabaseError):
            with db:
                db.execute(sql)
    lineage = db.execute(
        "SELECT source_run_id FROM candidate_lineage WHERE candidate_id = ?", (ids[0],)
    ).fetchall()
    assert lineage == [(run_id,)]


def test_validation_is_reproducible_and_never_retunes(engine: SyntheticEngine) -> None:
    _, ids = engine.create_candidates("1h")
    first = engine.validate(ids[0])
    second = engine.validate(ids[0])
    assert first["outcome"] in ("VALIDATED", "REJECTED")
    assert second["outcome"].startswith("REPRODUCTION")
    assert first["judgement"] == second["judgement"] and first["candidate"] == second["candidate"]
    runs = engine.store.validations(ids[0])
    assert len(runs) == 2 and runs[0]["dataset"] == runs[1]["dataset"]
    comparison = engine.compare(ids[0])
    assert set(comparison["VALIDATION"]["baselines"]) == {
        "production_scout", "production_opportunity_buy", "simple_momentum", "liquidity_and_volume",
        "buy_and_hold_horizon", "random_eligible_half",
    }  # fmt: skip
    assert "not a profitability claim" in comparison["label"].lower()


def test_a_candidate_not_supported_by_validation_is_rejected(tmp_path: Path) -> None:
    # Validation data where the calibration pattern is reversed.
    flipped = [make(o.asset_id, o.at, -(o.outcomes["1h"].return_pct or 0), split="VALIDATION",
                    liquidity_usd=o.f("liquidity_usd"), stage=o.s("stage"),
                    family_scores={"liquidity_quality": 0.8 if (o.f("liquidity_usd") or 0) > 1e5 else 0.2})
               for o in synthetic(30, 2, "VALIDATION", seed=2, start=START + timedelta(days=20))]  # fmt: skip
    e = SyntheticEngine(CalibrationStore(tmp_path / "c.sqlite3"), synthetic(60, 3) + flipped)
    _, ids = e.create_candidates("1h")
    outcomes = {e.validate(cid)["outcome"] for cid in ids}
    assert "REJECTED" in outcomes
    assert all(e.store.candidate(cid)["status"] in ("REJECTED", "VALIDATED") for cid in ids)  # type: ignore[index]


def test_judge_requires_material_improvement_without_regression() -> None:
    cfg = CalibrationConfig()
    base = evaluate([], synthetic(60, 3), "1h", cfg)
    same = judge(base, base, cfg)
    assert same["improves"] is False and same["materially_worse"] is False
    deeper = evaluate(
        [{"parameter": "filter.min_liquidity_usd", "from": None, "to": 100_000}],
        synthetic(60, 3),
        "1h",
        cfg,
    )
    assert judge(deeper, base, cfg)["improves"] is True
    with pytest.raises(ValueError):
        evaluate([{"parameter": "scout.weights.secret", "to": 1}], synthetic(10, 1), "1h", cfg)


# --- determinism, versions, production safety ----------------------------------------------------------------


def test_runs_are_deterministic_and_versioned(tmp_path: Path) -> None:
    a = SyntheticEngine(CalibrationStore(tmp_path / "a.sqlite3"), full_dataset())
    b = SyntheticEngine(CalibrationStore(tmp_path / "b.sqlite3"), full_dataset())
    _, ra = a.analyze("1h")
    _, rb = b.analyze("1h")
    assert ra == rb
    [run] = a.store.runs("analyze")
    versions = (
        sqlite3.connect(a.store.path)
        .execute("SELECT versions_json FROM calibration_runs")
        .fetchone()[0]
    )
    for key in ("code", "scout_scoring", "risk", "opportunity", "technical", "feature_schema", "replay_record",
                "evidence_archive_schema"):  # fmt: skip
        assert key in versions
    assert run["dataset"]["fingerprint"] == ra["dataset"]["fingerprint"]
    assert "comparisons were evaluated" in ra["multiple_comparisons"]["warning"]


def test_production_configuration_and_data_never_change(tmp_path: Path) -> None:
    snapshot = (GrowthConfig().model_dump(), ScoutConfig().model_dump(), OutcomeConfig().model_dump(),
                repr(dict(STRATEGIES)), services.growth_scout_service.config.model_dump())  # fmt: skip
    e = SyntheticEngine(CalibrationStore(tmp_path / "c.sqlite3"), full_dataset())
    e.analyze("1h")
    for cid in e.create_candidates("1h")[1]:
        e.validate(cid)
    after = (GrowthConfig().model_dump(), ScoutConfig().model_dump(), OutcomeConfig().model_dump(),
             repr(dict(STRATEGIES)), services.growth_scout_service.config.model_dump())  # fmt: skip
    assert snapshot == after


def test_calibration_has_no_execution_or_key_path() -> None:
    root = Path(__file__).resolve().parents[1] / "upscale"
    forbidden = {"private_key", "privatekey", "keypair", "mnemonic", "seed_phrase", "wallet",
                 "sign_transaction", "send_transaction", "swap", "place_order", "submit_order", "post", "put"}  # fmt: skip
    for path in [*(root / "services" / "calibration").glob("*.py"), root / "calibration_api.py"]:
        names = {t.string.lower() for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
                 if t.type == tokenize.NAME}  # fmt: skip
        assert not names & forbidden, (path.name, names & forbidden)


# --- real live and replay data --------------------------------------------------------------------------------


def test_live_outcomes_are_loaded_read_only(tmp_path: Path) -> None:
    result, _ = ranked(tmp_path)
    store = OutcomeStore(tmp_path / "live.sqlite3")
    observations = asyncio.run(record_scout_run(store, result, OutcomeConfig()))
    for o in observations:
        path = PricePath(source="candles", points=10, window_start=o.observed_at,
                         window_end=o.observed_at + timedelta(hours=1), reference_price=1.0, return_pct=4.0,
                         mfe_pct=6.0, mae_pct=-1.0, max_drawdown_pct=-2.0)  # fmt: skip
        asyncio.run(store.update_horizon("scout", o.id or 0, "1h", HorizonUpdate(price=path, finalize="COMPLETE"),
                                         o.observed_at + timedelta(hours=2)))  # fmt: skip
    before = hashlib.sha256((tmp_path / "live.sqlite3").read_bytes()).hexdigest()
    loaded = load_live(tmp_path / "live.sqlite3", CalibrationConfig(), include_holdout=True)
    assert len(loaded) == len(observations) and {o.origin for o in loaded} == {"LIVE_FORWARD"}
    o = loaded[0]
    assert (
        o.outcomes["1h"].return_pct == 4.0
        and o.features["stage"]
        and o.features["score"] is not None
    )
    assert o.split == live_split(o.at, CalibrationConfig().live_split)
    assert hashlib.sha256((tmp_path / "live.sqlite3").read_bytes()).hexdigest() == before
    policy = LiveSplitPolicy(
        anchor=(o.at - timedelta(days=18)).date().isoformat()
    )  # o's day is HOLDOUT
    sealed = load_live(tmp_path / "live.sqlite3", CalibrationConfig(live_split=policy))
    assert (
        sealed == []
        and len(load_live(tmp_path / "live.sqlite3", CalibrationConfig(live_split=policy), True))
        > 0
    )


def test_replay_samples_are_loaded_with_their_splits(tmp_path: Path) -> None:
    archive = build_archive(
        tmp_path / "scout.sqlite3", tokens=5, snapshots=4, spacing=timedelta(hours=6)
    )
    h = Harness(tmp_path, archive)
    h.run(config(end=config().start + timedelta(days=2)))
    visible = load_replay(tmp_path / "replay.sqlite3")
    everything = load_replay(tmp_path / "replay.sqlite3", include_holdout=True)
    assert visible and {o.origin for o in visible} == {"HISTORICAL_REPLAY"}
    assert all(o.split != "HOLDOUT" for o in visible) and any(
        o.split == "HOLDOUT" for o in everything
    )
    assert visible[0].features["stage"] and visible[0].versions and visible[0].outcomes


def test_cli_and_api(tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "cli.sqlite3")
    live, replay = str(tmp_path / "none-live.sqlite3"), str(tmp_path / "none-replay.sqlite3")
    base = ["--db", db, "--live-db", live, "--replay-db", replay]
    for args in (["status"], ["readiness"], ["analyze", "--horizon", "1h"], ["findings"],
                 ["create-candidates"], ["candidates"]):  # fmt: skip
        assert cli_main([*base, *args]) == 0, args
    assert cli_main([*base, "final-evaluate", "cand-x"]) == 1
    assert cli_main([*base, "validate", "cand-missing"]) == 1
    monkeypatch.setenv("UPSCALE_CALIBRATION_DB", db)
    for path in ("status", "readiness", "findings", "candidates"):
        assert client.get(f"/calibration/{path}").status_code == 200, path
    paths = {p: set(ops) for p, ops in client.get("/openapi.json").json()["paths"].items()
             if p.startswith("/calibration")}  # fmt: skip
    assert set(paths) == {
        f"/calibration/{x}" for x in ("status", "readiness", "findings", "candidates")
    }
    assert all(ops == {"get"} for ops in paths.values())  # read-only: nothing can change state
