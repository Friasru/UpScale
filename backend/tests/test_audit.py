"""Audit / Learning V1: read-only (no row, file or checkpoint changes; databases opened
``mode=ro``), no network, anti-lookahead (features never later than the decision; future
evidence and labels never leak into features), canonical identity, missing evidence kept
unavailable, asset-aware samples, filters, deterministic JSON / text, REALISTIC_V1
execution metrics, hypothesis gating and phrasing. Temporary databases only: Scout
evidence and Shadow runs are built with the production serializers through the Shadow
test harness; anchors and horizons through the production OutcomeStore."""

import asyncio
import hashlib
import json
import os
import socket
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

from upscale.services.audit import analytics, features, sources
from upscale.services.audit.cli import main as cli_main
from upscale.services.audit.config import AuditConfig, Minimums
from upscale.services.audit.dataset import (
    AuditObservation,
    HorizonLabel,
    Paths,
    TradeLabel,
    assert_causal,
    load,
)
from upscale.services.audit.features import FeatureGroup, LookaheadError, Snapshot
from upscale.services.audit.report import build, feature_audit, text, to_json
from upscale.services.evidence_archive.store import PendingRecord
from upscale.services.outcomes import OutcomeStore
from upscale.services.outcomes.audit import AuditRow, write_audits
from upscale.services.outcomes.config import DEFAULT_HORIZONS
from upscale.services.outcomes.models import PricePath
from upscale.services.outcomes.observe import scout_observation
from upscale.services.outcomes.store import HorizonUpdate
from upscale.services.scout.growth.models import GrowthCandidate, GrowthScoutResult
from upscale.services.shadow.book import latency_drift_usd
from upscale.services.shadow.config import ExecutionConfig
from upscale.services.shadow.strategies import BASELINES

from .test_shadow import T0, A, B, Harness, strat

C = "MintCCCC3333"
CFG = AuditConfig()


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[GrowthScoutResult, GrowthCandidate]:
    from .test_outcomes import ranked

    result, store = ranked(tmp_path_factory.mktemp("template"))
    store.close()
    return result, next(c for c in result.candidates if c.stage == "ACCELERATING")


class AH(Harness):
    """The Shadow harness plus the Scout database's outcome store (anchors, horizons)."""

    def __init__(self, tmp_path: Path, template: Any) -> None:
        super().__init__(tmp_path, template)
        self.scout_path = tmp_path / "scout.sqlite3"
        self.outcomes = OutcomeStore(self.scout_path)

    @property
    def paths(self) -> Paths:
        return Paths(str(self.scout_path), str(self.ev_path), str(self.shadow_path))

    def anchor(
        self, at: datetime, *cands: GrowthCandidate, labels: dict[str, Any] | None = None,
        lag: float = 60.0,
    ) -> dict[str, int]:  # fmt: skip
        """One Scout run (archived evidence, decision at `at`) and its anchors, with
        finalized horizon labels {mint: {horizon: (return, mfe, mae)}}."""
        self.scan(at, *cands, lag=lag)
        start = at - timedelta(seconds=lag)
        result = self.result.model_copy(update={"computed_at": start})
        obs = [
            scout_observation(c.model_copy(update={"observed_at": start, "rank": i + 1}),
                              result, "FIRST_RANKED", None)
            for i, c in enumerate(cands)
        ]  # fmt: skip
        stored = asyncio.run(self.outcomes.add_scout_observations(obs, DEFAULT_HORIZONS))
        for o in stored:
            assert o.id is not None
            for hz, (ret, mfe, mae) in (labels or {}).get(o.address, {}).items():
                minutes = next(x.minutes for x in DEFAULT_HORIZONS if x.label == hz)
                path = PricePath(
                    source="candles", points=10, window_start=start,
                    window_end=start + timedelta(minutes=minutes), reference_price=1.0,
                    return_pct=ret, mfe_pct=mfe, mae_pct=mae,
                )  # fmt: skip
                update = HorizonUpdate(price=path, finalize="COMPLETE", market_status="ACTIVE")
                done = start + timedelta(minutes=minutes + 5)
                assert asyncio.run(self.outcomes.update_horizon("scout", o.id, hz, update, done))
        return {o.address: o.id for o in stored if o.id is not None}

    def close(self) -> None:
        super().close()
        self.outcomes.close()


@pytest.fixture
def h(tmp_path: Path, template: Any) -> Any:
    harness = AH(tmp_path, template)
    yield harness
    harness.close()


def m(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def scenario(h: AH) -> dict[str, int]:
    """A (technical up) and B (no technical, social / safety missing) at T0; A takes profit
    (+25% at 30 min), B stops out (-15% at 40 min). An idealized run with two strategies and
    a REALISTIC_V1 run of one; anchors with 4h / 1h labels."""
    ids = h.anchor(T0, h.cand(A, price=1.0), h.cand(B, price=2.0, technical=None),
                   labels={A: {"4h": (12.0, 30.0, -4.0), "1h": (5.0, 8.0, -1.0)},
                           B: {"4h": (-45.0, 1.0, -50.0)}})  # fmt: skip
    h.price(m(5), 1.02)
    h.price(m(30), 1.25)
    h.price(m(40), 1.7, mint=B)
    h.price(m(45), 1.3)
    h.now = m(61)
    h.run(strat("scout_technical"), strat("random_eligible"), run_id="ideal")
    h.store.register(strat("scout_threshold"))
    h.engine.ensure_run("real", T0 - timedelta(hours=1), None, ["scout_threshold"],
                        availability_policy="EVIDENCE_AWARE_V2", execution=ExecutionConfig())  # fmt: skip
    h.engine.run("real")
    return ids


def report(h: AH, **kw: Any) -> dict[str, Any]:
    ds = load(h.paths, CFG, **kw)
    return build(ds, CFG, {k: str(v) if v is not None else None for k, v in kw.items()})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in sorted(names)}
    finally:
        conn.close()


def obs(h: AH, key: str, **kw: Any) -> AuditObservation:
    return next(o for o in load(h.paths, CFG, **kw).observations if o.key == key)


# --- read-only guarantees -----------------------------------------------------------------------


def test_audit_changes_no_rows_files_or_checkpoints(
    h: AH, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = scenario(h)
    checkpoints = (h.store.checkpoint("ideal"), h.store.checkpoint("real"))
    h.close()  # every writer closed: the files are at rest
    files = [h.scout_path, h.ev_path, h.shadow_path]
    before = ([digest(p) for p in files], [counts(p) for p in files])
    base = ["--scout-db", str(h.scout_path), "--evidence-db", str(h.ev_path),
            "--shadow-db", str(h.shadow_path)]  # fmt: skip
    entry = next(o.key for o in load(h.paths, CFG).observations if o.population == "shadow")
    for args in (["status"], ["status", "--json"], ["report"], ["report", "--json"],
                 ["feature-audit", "--decision-id", f"scout:{ids[A]}"],
                 ["feature-audit", "--decision-id", entry, "--json"]):  # fmt: skip
        assert cli_main([*base, *args]) == 0
    capsys.readouterr()
    assert ([digest(p) for p in files], [counts(p) for p in files]) == before
    h.open()
    assert (h.store.checkpoint("ideal"), h.store.checkpoint("real")) == checkpoints


def test_connections_are_read_only(h: AH) -> None:
    scenario(h)
    for path in (h.scout_path, h.ev_path, h.shadow_path):
        conn = sources.connect(path)
        assert conn is not None
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE audit_cache (x)")
        conn.close()


def test_missing_databases_are_never_created(tmp_path: Path) -> None:
    paths = Paths(*(str(tmp_path / f"{n}.sqlite3") for n in ("scout", "evidence", "shadow")))
    ds = load(paths, CFG)
    assert ds.observations == [] and len(ds.notes) == 3
    r = build(ds, CFG, {})
    assert r["scout"] is None and r["shadow"] is None and r["hypotheses"] == []
    assert "No difference met the rules" in text(r)
    assert list(tmp_path.iterdir()) == []


def test_zero_provider_or_network_requests(
    h: AH, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = scenario(h)

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("Audit must not use the network")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    base = ["--scout-db", str(h.scout_path), "--evidence-db", str(h.ev_path),
            "--shadow-db", str(h.shadow_path)]  # fmt: skip
    assert cli_main([*base, "report", "--json"]) == 0
    assert cli_main([*base, "status"]) == 0
    assert cli_main([*base, "feature-audit", "--decision-id", f"scout:{ids[B]}"]) == 0
    assert json.loads(capsys.readouterr().out.split("\n=== AUDIT STATUS")[0])["read_only"]


def test_no_production_configuration_is_mutated(h: AH) -> None:
    scenario(h)
    hashes = [s.config_hash for s in BASELINES]
    registered = [s.config_hash for s in h.store.strategies()]
    runs = h.store.runs()
    env = dict(os.environ)
    cfg = AuditConfig()
    build(load(h.paths, cfg), cfg, {})
    assert [s.config_hash for s in BASELINES] == hashes
    assert [s.config_hash for s in h.store.strategies()] == registered
    assert dict(os.environ) == env and cfg == AuditConfig()
    assert h.store.runs() == runs


# --- anti-lookahead ---------------------------------------------------------------------------------


def test_feature_timestamps_never_exceed_the_decision(h: AH) -> None:
    scenario(h)
    ds = load(h.paths, CFG)
    assert {o.population for o in ds.observations} == {"scout", "shadow"}
    for o in ds.observations:
        for g in o.snapshot.groups.values():
            assert g.observed_at is None or g.observed_at <= o.decision_at, (o.key, g.name)
    a = next(o for o in ds.observations if o.population == "scout" and o.asset_id.endswith(A))
    # The decision is when the ranking was final (the archived Scout record), the market
    # observation a minute earlier.
    assert a.decision_at == T0 and a.decision_basis == "SCOUT_EVIDENCE_DECISION_TIME"
    assert a.snapshot.groups["market"].observed_at == T0 - timedelta(seconds=60)
    assert a.snapshot.groups["technical"].observed_at == T0 - timedelta(seconds=60)


def test_future_evidence_cannot_leak_into_features(h: AH) -> None:
    ids = h.anchor(T0, h.cand(A, price=1.0, score=75.0), labels={A: {"4h": (3.0, 5.0, -1.0)}})
    # Later evidence of the same token: a new ranking (other score / stage / no technical),
    # a later Analyze decision and later prices. None of it is known at T0.
    h.scan(m(30), h.cand(A, price=9.0, score=10.0, stage="FADING", technical=None,
                         safety="SAFETY_CHECKS_COMPLETE"))  # fmt: skip
    h.analyze(m(10))
    h.price(m(20), 5.0)
    o = obs(h, f"scout:{ids[A]}")
    s = o.snapshot
    assert s.num("score") == 75.0 and s.text("stage") == "ACCELERATING"
    assert s.num("price_usd") == 1.0 and s.text("technical_trend") == "up"
    assert s.text("safety_status") == "INSUFFICIENT_SAFETY_DATA"
    assert s.groups["analyze_technical"].availability == "NOT_AVAILABLE"
    # An Analyze decision before T0 (within the max age) is used; its time is kept.
    h.analyze(T0 - timedelta(minutes=30))
    o = obs(h, f"scout:{ids[A]}")
    g = o.snapshot.groups["analyze_technical"]
    assert g.available and g.observed_at == T0 - timedelta(minutes=30)
    assert g.values["analyze_trend"] == "uptrend"


def test_analyze_older_than_the_max_age_is_not_used(h: AH) -> None:
    ids = h.anchor(T0, h.cand(A))
    h.analyze(T0 - timedelta(minutes=CFG.analyze_max_age_minutes + 1))
    assert (
        obs(h, f"scout:{ids[A]}").snapshot.groups["analyze_technical"].availability
        == "NOT_AVAILABLE"
    )


def test_later_linked_evidence_is_excluded_not_used(template: Any) -> None:
    """A ranking that linked social / safety evidence observed after its decision (an
    archive causal violation): those groups are EXCLUDED_FUTURE, with no values."""
    _, g = template
    c = g.model_copy(update={"quality": g.quality.model_copy(update={
        "safety_status": "SAFETY_CHECKS_COMPLETE"})}).model_dump(mode="json")  # fmt: skip
    c["momentum"]["social_status"] = "SOCIAL_EMERGING"
    later = (T0 + timedelta(minutes=5)).isoformat()
    links = [{"kind": "social", "observed_at": later, "record_id": "s1"},
             {"kind": "safety", "observed_at": later, "record_id": "q1"}]  # fmt: skip
    snap = features.build_snapshot(T0, features.candidate_groups(c, T0, T0, "test", links))
    assert set(snap.excluded_future) == {"social", "safety"}
    for name in ("social", "safety"):
        assert snap.groups[name].availability == "EXCLUDED_FUTURE"
        assert snap.groups[name].values == {}
    assert snap.value("safety_status") is None and snap.value("social_status") is None


def test_lookahead_assertions_fire() -> None:
    late = FeatureGroup("market", "AVAILABLE", T0 + timedelta(seconds=1), "x", {"price_usd": 1.0})
    with pytest.raises(LookaheadError):
        features.assert_snapshot(Snapshot(T0, {"market": late}))
    with pytest.raises(LookaheadError):
        features.assert_snapshot(
            Snapshot(T0, {"market": FeatureGroup("market", "AVAILABLE", None, "x", {"p": 1})})
        )
    ok = features.build_snapshot(T0, [late])
    assert ok.excluded_future == ("market",) and ok.value("price_usd") is None
    bad = AuditObservation("scout", "k", "solana:x", "solana", T0, "t",
                           Snapshot(T0, {"market": late}), {})  # fmt: skip
    with pytest.raises(LookaheadError):
        assert_causal([bad])


def test_outcome_labels_are_future_but_never_features(h: AH) -> None:
    ids = scenario(h)
    o = obs(h, f"scout:{ids[A]}")
    label = o.horizons["4h"]
    assert (label.return_pct, label.mfe_pct, label.mae_pct) == (12.0, 30.0, -4.0)
    names = {k for g in o.snapshot.groups.values() for k in g.values}
    assert not names & {"return_pct", "mfe_pct", "mae_pct", "future_stage", "liquidity_change_pct"}
    fa = feature_audit(o)
    assert fa["future_labels"]["horizons"]["4h"]["return_pct"] == 12.0
    assert all(g["observed_before_decision"] in (True, None)
               for g in fa["decision_time_features"].values())  # fmt: skip
    assert "return_pct" not in json.dumps(fa["decision_time_features"])


def test_pending_and_integrity_invalid_horizons_are_not_labels(h: AH) -> None:
    ids = h.anchor(T0, h.cand(A), labels={A: {"1h": (1.0, 2.0, -1.0), "4h": (2.0, 3.0, -1.0)}})
    h.outcomes.close()
    row = AuditRow(kind="scout", observation_id=ids[A], horizon="4h", horizon_minutes=240,
                   status="COMPLETE", return_pct=2.0, mfe_pct=3.0, mae_pct=-1.0,
                   market_status="ACTIVE", attempts=1, missing=[], path=None, body={},
                   integrity="INVALID_REFERENCE_PRICE", reason="test")  # fmt: skip
    write_audits(h.scout_path, [row])
    ds = load(h.paths, CFG)
    o = next(x for x in ds.observations if x.key == f"scout:{ids[A]}")
    assert set(o.horizons) == {"1h"}  # 4h invalid; 5m / 15m / 24h still PENDING
    assert ds.quality["horizon_integrity_excluded_4h"] == 1
    assert ds.quality["horizon_pending_24h"] == 1


# --- identity --------------------------------------------------------------------------------------


def test_evm_canonical_identity_is_lowercase_and_solana_case_sensitive(h: AH) -> None:
    upper = "0xAbCdEf0123456789abcdef0123456789ABCDEF01"
    assert features.canonical("base", upper) == f"base:{upper.lower()}"
    assert features.canonical("ethereum", upper) == features.canonical("ethereum", upper.lower())
    assert features.canonical("solana", "MintAbc") == "solana:MintAbc"
    assert features.canonical("solana", "MintAbc") != features.canonical("solana", "MINTABC")
    # Two Solana mints differing only in case are two assets (never merged).
    h.anchor(T0, h.cand("MintAbc"), h.cand("MINTABC"),
             labels={"MintAbc": {"4h": (1.0, 2.0, 0.0)}, "MINTABC": {"4h": (1.0, 2.0, 0.0)}})  # fmt: skip
    r = report(h)
    assert r["scout"]["unique_assets"] == 2
    assert r["scout"]["by_horizon"]["4h"]["unique_assets"] == 2


# --- missing data, samples, filters --------------------------------------------------------------------


def test_missing_technical_social_safety_stay_unavailable(h: AH) -> None:
    ids = scenario(h)
    b = obs(h, f"scout:{ids[B]}").snapshot
    assert (
        b.groups["technical"].availability == "NOT_AVAILABLE" and b.groups["technical"].values == {}
    )
    assert b.groups["social"].availability == "NOT_AVAILABLE"
    assert b.text("social_status") == "SOCIAL_UNAVAILABLE"
    assert b.groups["safety"].availability == "NOT_AVAILABLE"
    assert b.value("mint_authority_active") is None and b.value("holder_top10_pct") is None
    r = report(h)
    g = r["scout"]["groupings"]["4h"]
    assert "UNAVAILABLE (NOT_AVAILABLE)" in g["technical_confirmation"]["buckets"]
    assert "CONFIRMED" in g["technical_confirmation"]["buckets"]
    assert set(g["authority"]["buckets"]) == {"UNAVAILABLE"}
    # An anchor without its Scout evidence record: technical NOT_LINKED, never "down".
    conn = sqlite3.connect(h.ev_path)
    conn.execute("DROP TRIGGER evidence_no_delete")
    conn.execute("DELETE FROM evidence_records WHERE kind = 'scout'")
    conn.commit()
    conn.close()
    o = obs(h, f"scout:{ids[A]}")
    assert o.snapshot.groups["technical"].availability == "NOT_LINKED"
    assert o.decision_basis.startswith("RANKING_RUN_START") and o.decision_at == T0 - timedelta(
        seconds=60
    )
    assert o.snapshot.num("score") is not None  # the anchor's own stored fields remain


def _synthetic(
    assets: list[str], ret: float, liq: float, mae: float | None = None
) -> list[AuditObservation]:
    out = []
    for i, a in enumerate(assets):
        groups = {
            "scout": FeatureGroup("scout", "AVAILABLE", T0, "x", {"score": 70.0, "stage": "EARLY"}),
            "market": FeatureGroup("market", "AVAILABLE", T0, "x", {"liquidity_usd": liq}),
        }
        label = HorizonLabel("COMPLETE", ret, max(ret, 0.0), mae if mae is not None else min(ret, 0.0),
                             "ACTIVE", None, None)  # fmt: skip
        out.append(AuditObservation("scout", f"scout:{a}:{i}:{liq}", f"solana:{a}", "solana",
                                    T0 + timedelta(minutes=i), "t", Snapshot(T0 + timedelta(minutes=i), groups),
                                    {"4h": label}))  # fmt: skip
    return out


def test_repeated_token_observations_do_not_inflate_the_sample() -> None:
    repeated = _synthetic(["Same"] * 60, -40.0, 5_000.0)
    others = _synthetic([f"T{i}" for i in range(60)], 5.0, 500_000.0)
    s = analytics.summarize([analytics.horizon_item(o, "4h") for o in repeated], CFG)
    assert (s["measured"], s["unique_assets"], s["effective_sample_size"]) == (60, 1, 1.0)
    assert s["status"] == "INSUFFICIENT_SAMPLE"
    assert analytics.effective_sample_size(["a", "a", "b", "b"]) == 2.0
    assert analytics.effective_sample_size([f"x{i}" for i in range(7)]) == 7.0
    # One token repeated 60 times can't produce a hypothesis however extreme it looks.
    found, stats = analytics.scout_hypotheses(repeated + others, CFG)
    assert found == [] and stats["notable_but_insufficient"] > 0
    # 60 distinct catastrophic tokens can.
    many = _synthetic([f"L{i}" for i in range(60)], -40.0, 5_000.0)
    found, _ = analytics.scout_hypotheses(many + others, CFG)
    liq = next(x for x in found if x["dimension"] == "liquidity_bucket")
    # Two buckets: one comparison, phrased from the first bucket's side (no mirror twin).
    assert liq["bucket"] == "250k-1M"
    cat = next(c for c in liq["evidence"] if c["metric"] == "catastrophic")
    assert cat["group"] == 0.0 and cat["other"] == 100.0
    assert "a lower catastrophic-loss frequency" in liq["hypothesis"]
    metrics = {c["metric"] for c in liq["evidence"]}
    assert {"catastrophic", "win", "return"} <= metrics
    assert liq["hypothesis"].startswith("Candidate hypothesis for Calibration: test whether")


def test_cluster_standard_error_counts_assets_not_rows() -> None:
    one = analytics.ratio([("a", 100.0)] * 50 + [("b", 0.0)] * 50)
    many = analytics.ratio(
        [(f"a{i}", 100.0) for i in range(50)] + [(f"b{i}", 0.0) for i in range(50)]
    )
    assert one is not None and many is not None
    assert one.value == many.value == 50.0
    assert one.se > 5 * many.se  # two clusters: almost no information
    single = analytics.ratio([("a", 1.0)] * 10)
    assert single is not None and single.se == float("inf")


def test_strategy_filter_keeps_only_that_strategy(h: AH) -> None:
    scenario(h)
    r = report(h, strategy="scout_technical")
    assert r["scout"] is None
    runs = r["shadow"]["runs"]
    assert set(runs) == {"ideal"}
    assert set(runs["ideal"]["strategies"]) == {"scout_technical@v1"}
    assert any("Scout anchors left out" in n for n in r["notes"])
    assert set(report(h, run="real")["shadow"]["runs"]) == {"real"}
    assert report(h, population="scout")["shadow"] is None


def test_since_and_until_filter_by_decision_time(h: AH) -> None:
    h.anchor(T0, h.cand(A), labels={A: {"4h": (1.0, 2.0, 0.0)}})
    h.anchor(m(120), h.cand(B), labels={B: {"4h": (1.0, 2.0, 0.0)}})
    assert report(h)["scout"]["observations"] == 2
    later = report(h, since=m(60))
    assert later["scout"]["observations"] == 1 and later["data_from"] == m(120).isoformat()
    earlier = report(h, until=m(60))
    assert earlier["scout"]["observations"] == 1 and earlier["data_through"] == T0.isoformat()
    assert report(h, since=m(121))["scout"] is None


def test_since_works_from_the_cli(h: AH, capsys: pytest.CaptureFixture[str]) -> None:
    h.anchor(T0, h.cand(A))
    h.anchor(m(120), h.cand(B))
    base = ["--scout-db", str(h.scout_path), "--evidence-db", str(h.ev_path),
            "--shadow-db", str(h.shadow_path)]  # fmt: skip
    assert cli_main([*base, "report", "--json", "--since", m(60).isoformat()]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scout"]["observations"] == 1 and out["filters"]["since"] == m(60).isoformat()
    with pytest.raises(SystemExit):
        cli_main([*base, "report", "--since", "2026-10-01T00:00:00"])  # no timezone


# --- determinism ------------------------------------------------------------------------------------


def test_json_and_text_reports_are_deterministic(h: AH, capsys: pytest.CaptureFixture[str]) -> None:
    scenario(h)
    base = ["--scout-db", str(h.scout_path), "--evidence-db", str(h.ev_path),
            "--shadow-db", str(h.shadow_path), "report"]  # fmt: skip
    outs = []
    for args in (["--json"], ["--json"], [], []):
        assert cli_main([*base, *args]) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1] and outs[2] == outs[3]
    parsed = json.loads(outs[0])
    assert to_json(parsed) == outs[0].rstrip("\n")
    assert "=== HYPOTHESES FOR CALIBRATION ===" in outs[2]
    assert outs[2].rstrip().endswith("before anything changes.")
    assert "Change production" not in outs[2] and "change production" not in outs[
        2
    ].lower().replace("never changes production", "")


# --- Shadow / REALISTIC_V1 -------------------------------------------------------------------------------


def test_realistic_execution_metrics_are_read_correctly(h: AH) -> None:
    scenario(h)
    ds = load(h.paths, CFG, run="real")
    (o,) = [x for x in ds.observations if x.asset_id.endswith(A)]
    buy = next(x for x in h.store.executions("real") if x["side"] == "BUY")
    sells = [x for x in h.store.executions("real") if x["side"] == "SELL"]
    x = o.execution
    assert x is not None and o.execution_model == "REALISTIC_V1"
    v = x.values
    assert v["execution_delay_seconds"] == buy["delay_seconds"]
    assert v["observed_fill_price"] == buy["observed_price"]
    assert v["execution_price"] == buy["execution_price"]
    assert v["intent_price"] == buy["reference_price"]
    assert (v["fee_usd"], v["slippage_bps"], v["price_impact_bps"]) == (
        buy["fee_usd"],
        buy["slippage_bps"],
        buy["impact_bps"],
    )
    assert v["latency_drift_usd"] == pytest.approx(
        latency_drift_usd("BUY", buy["reference_price"], buy["observed_price"], buy["quantity"])
    )
    assert x.observed_at is not None and x.observed_at.timestamp() == buy["filled_at"]
    assert x.observed_at > o.decision_at  # post-decision: never a decision-time feature
    assert "execution" not in o.snapshot.groups
    t = o.trade
    assert t is not None and t.status == "CLOSED" and t.exit_reason == "TAKE_PROFIT"
    assert t.gross_pnl_usd == pytest.approx(sum(s["gross_pnl_usd"] for s in sells))
    assert t.friction_usd == pytest.approx(sum(s["trade_friction_usd"] for s in sells))
    assert t.pnl_usd == pytest.approx(sum(s["net_pnl_usd"] for s in sells))
    assert t.gross_pnl_usd > t.pnl_usd  # friction costs
    r = report(h, run="real")
    e = r["shadow"]["runs"]["real"]["strategies"]["scout_threshold@v1"]["execution"]
    assert e["execution_model"] == "REALISTIC_V1" and e["entries_filled"] >= 1
    assert e["net_pnl_usd"] == pytest.approx(round(sum(
        q.trade.pnl_usd for q in ds.observations if q.trade and q.trade.resolved and q.trade.pnl_usd is not None), 2))  # fmt: skip
    assert "entry execution delay vs trade outcome" in text(r)


def test_idealized_trades_have_no_friction(h: AH) -> None:
    scenario(h)
    ds = load(h.paths, CFG, run="ideal", strategy="scout_technical")
    a = next(o for o in ds.observations if o.asset_id.endswith(A))
    assert a.trade is not None and a.trade.resolved and a.trade.return_pct == pytest.approx(25.0)
    assert a.trade.gross_pnl_usd == a.trade.pnl_usd and a.trade.friction_usd == 0.0
    assert a.execution is not None and a.execution.values["execution_delay_seconds"] == 0.0
    b = next(o for o in ds.observations if o.asset_id.endswith(B))
    assert b.trade is not None and b.trade.exit_reason == "STOP_LOSS"
    # The same Scout evaluation's anchor gives the entry its horizon labels.
    assert a.anchor_id is not None and a.horizons["4h"].return_pct == 12.0


def test_shadow_features_come_from_the_entry_decision_time(h: AH) -> None:
    scenario(h)
    h.scan(m(50), h.cand(A, price=1.3, score=20.0, stage="FADING"))  # later evidence of A
    for o in load(h.paths, CFG, run="ideal").observations:
        assert o.snapshot.decision_at == o.decision_at == T0
        assert o.snapshot.num("score") == 75.0 and o.snapshot.text("stage") == "ACCELERATING"


def test_feature_audit_cli(h: AH, capsys: pytest.CaptureFixture[str]) -> None:
    ids = scenario(h)
    base = ["--scout-db", str(h.scout_path), "--evidence-db", str(h.ev_path),
            "--shadow-db", str(h.shadow_path), "feature-audit", "--decision-id"]  # fmt: skip
    assert cli_main([*base, f"scout:{ids[A]}"]) == 0
    out = capsys.readouterr().out
    assert (
        "DECISION-TIME FEATURES" in out and "FUTURE LABELS" in out and "technical_trend = up" in out
    )
    d = h.store.decisions(run_id="real", action="ENTER")[0]["decision_id"]
    assert cli_main([*base, d, "--json"]) == 0
    a = json.loads(capsys.readouterr().out)
    assert a["key"] == f"shadow:{d}" and a["execution_model"] == "REALISTIC_V1"
    assert a["post_decision_execution"]["values"]["execution_model"] == "REALISTIC_V1"
    assert a["future_labels"]["trade"]["status"] == "CLOSED"
    assert cli_main([*base, "scout:999999"]) == 1
    assert "no Scout anchor" in capsys.readouterr().err


# --- hypotheses ------------------------------------------------------------------------------------------


def test_hypotheses_are_phrased_as_tests_and_gated_by_samples() -> None:
    cfg = AuditConfig(minimums=Minimums(measured=30, unique_assets=10, effective_sample=10))
    low = _synthetic([f"L{i}" for i in range(40)], -35.0, 5_000.0)
    high = _synthetic([f"H{i}" for i in range(40)], 4.0, 500_000.0)
    r = build_from(low + high, cfg)
    assert r["hypotheses"], "a large, well-sampled difference must be listed"
    for hyp in r["hypotheses"]:
        assert hyp["hypothesis"].startswith("Candidate hypothesis for Calibration: test whether")
        assert all(c["sufficient_sample"] for c in hyp["evidence"])
    out = text(r)
    assert "Change production" not in out
    # Too few: nothing surfaced, but the large difference is counted as not surfaced.
    r2 = build_from(low[:8] + high[:8], cfg)
    assert r2["hypotheses"] == [] and r2["hypothesis_stats"]["notable_but_insufficient_sample"] > 0


def build_from(observations: list[AuditObservation], cfg: AuditConfig) -> dict[str, Any]:
    from collections import Counter

    from upscale.services.audit.dataset import Dataset

    ds = Dataset(
        observations, Counter(), [], {"scout": "x", "evidence": None, "shadow": None}, {}, []
    )
    return build(ds, cfg, {})


def test_shadow_strategy_vs_random_baseline_hypothesis() -> None:
    def entry(sid: str, i: int, ret: float) -> AuditObservation:
        g = {"scout": FeatureGroup("scout", "AVAILABLE", T0, "x", {"score": 70.0})}
        return AuditObservation(
            "shadow", f"shadow:{sid}:{i}", f"solana:{sid}{i}", "solana", T0, "t", Snapshot(T0, g), {},
            run_id="r", strategy_id=sid, strategy_version=1, execution_model="IDEALIZED_NO_FEES",
            trade=TradeLabel("CLOSED", return_pct=ret, pnl_usd=ret * 5, mfe_pct=max(ret, 0),
                             mae_pct=min(ret, 0), exit_at=T0 + timedelta(hours=1)),
        )  # fmt: skip

    rows = [entry("scout_technical", i, 20.0 if i % 4 else -10.0) for i in range(40)]
    rows += [entry("random_eligible", i, -35.0 if i % 2 else 5.0) for i in range(40)]
    found, stats = analytics.shadow_hypotheses(rows, CFG)
    assert stats["tested"] == len(analytics.METRICS)
    (hyp,) = found
    assert hyp["bucket"] == "scout_technical" and "random_eligible trades" in hyp["hypothesis"]
    r = build_from(rows, CFG)
    s = r["shadow"]["runs"]["r"]["strategies"]
    assert s["scout_technical@v1"]["trade_outcome"]["profit_factor_basis"] == "USD P/L"
    assert s["random_eligible@v1"]["trade_outcome"]["catastrophic_rate"] == 0.5


def test_bucket_edges() -> None:
    edges = CFG.buckets.liquidity_usd
    assert analytics.bucket(None, edges) == "UNAVAILABLE"
    assert analytics.bucket(9_999.0, edges) == "<10k"
    assert analytics.bucket(10_000.0, edges) == "10k-50k"
    assert analytics.bucket(2_000_000.0, edges) == ">=1M"
    assert analytics._risk(0.0, CFG.buckets.risk_penalty) == "0"
    assert analytics._risk(3.0, CFG.buckets.risk_penalty) == "0-5"
    assert analytics._risk(25.0, CFG.buckets.risk_penalty) == ">=20"


def test_status_lists_readable_tables(h: AH, capsys: pytest.CaptureFixture[str]) -> None:
    scenario(h)
    from upscale.services.audit.cli import status

    s = status(h.paths)
    assert s["databases"]["scout"]["rows"]["scout_outcome_observations"] == 2
    assert s["databases"]["evidence"]["by_kind"]["scout"] >= 2
    runs = {r["run_id"]: r["execution_model"] for r in s["databases"]["shadow"]["runs"]}
    assert runs == {"ideal": "IDEALIZED_NO_FEES", "real": "REALISTIC_V1"}
    assert set(s["databases"]["shadow"]["rows"]) <= set(sources.SHADOW_TABLES)


def test_datetime_parsing_of_feature_times() -> None:
    assert features.when("2026-10-01T00:00:00+00:00") == T0
    assert features.when(T0.timestamp()) == T0
    assert features.when("nonsense") is None and features.when(None) is None
    assert features.when("2026-10-01T00:00:00") == datetime(2026, 10, 1, tzinfo=UTC)


def test_harness_records_use_the_production_serializer(h: AH) -> None:
    """Sanity: the fixtures' Scout records carry the timing links Audit matches on."""
    h.anchor(T0, h.cand(A))
    rec = h.reader.records(kind="scout")[0]
    assert rec.links["evaluation_started_at"] == (T0 - timedelta(seconds=60)).isoformat()
    assert rec.provider_at == T0 - timedelta(seconds=60) and rec.observed_at == T0
    assert isinstance(PendingRecord, type)


def test_groupings_default_to_the_primary_horizon(h: AH) -> None:
    scenario(h)
    ds = load(h.paths, CFG)
    assert set(build(ds, CFG, {})["scout"]["groupings"]) == {"4h"}
    assert set(build(ds, CFG, {}, all_horizons=True)["scout"]["groupings"]) == set(
        ("5m", "15m", "1h", "4h", "24h")
    )
    one_h = AuditConfig(primary_horizon="1h")
    r = build(load(h.paths, one_h), one_h, {})
    assert set(r["scout"]["groupings"]) == {"1h"} and r["primary_horizon"] == "1h"
    with pytest.raises(ValueError):
        AuditConfig(primary_horizon="2h")


def test_free_text_features_only_in_the_feature_audit(h: AH) -> None:
    ids = scenario(h)
    report_obs = obs(h, f"scout:{ids[A]}")
    assert report_obs.snapshot.value("stage_reasons") is None
    detailed = next(o for o in load(h.paths, AuditConfig(text_features=True),
                                    only=f"scout:{ids[A]}").observations)  # fmt: skip
    assert isinstance(detailed.snapshot.value("stage_reasons"), str)
    # Everything else is identical: the feature audit shows exactly what the report used.
    strip = {k: {n: v for n, v in g.values.items() if n not in features.TEXT_FEATURES}
             for k, g in detailed.snapshot.groups.items()}  # fmt: skip
    assert strip == {k: g.values for k, g in report_obs.snapshot.groups.items()}
    assert (
        detailed.decision_at == report_obs.decision_at and detailed.horizons == report_obs.horizons
    )
