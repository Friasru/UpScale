"""Opportunity orchestration bridge, B1: pure admission, deterministic priority and the
read-only dry-run. Offline: local fixture stores only; nothing is collected or written."""

import io
import json
import re
import socket
import sqlite3
import zlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.opportunity_model_fakes import (
    Archive,
    at,
    candidate,
    scout_record,
    social_record,
)
from tests.safety_v2_fakes import addr
from tests.test_opportunity_model_decision import _clean_safety
from upscale.services.evidence_archive.store import EvidenceRecord, prepare
from upscale.services.opportunity_model.recorder import record_from_paths
from upscale.services.opportunity_model.repository import OpportunityRepository
from upscale.services.opportunity_orchestrator import policy
from upscale.services.opportunity_orchestrator.admission import (
    evaluate_candidate,
    group_runs,
    prioritized,
    select_run,
)
from upscale.services.opportunity_orchestrator.cli import LIVE_NOT_IMPLEMENTED, main
from upscale.services.opportunity_orchestrator.config import POLICY, SAFETY_BUDGET_ENV
from upscale.services.opportunity_orchestrator.dry_run import classify_safety, dry_run
from upscale.services.opportunity_orchestrator.readers import SafetySnapshotRow
from upscale.services.safety_v2.repository import encode_body

T = at()  # the Scout run's decision time
NOW = T + timedelta(minutes=2)
MINTS = {tag: addr(tag) for tag in ("TokA", "TokB", "TokC", "TokD")}
_ID = iter(range(1, 10_000))


def cand(tag: str = "TokA", market_at: Any = None, **over: Any) -> dict[str, Any]:
    mint = MINTS[tag]
    c = candidate(market_at or T, canonical_id=f"solana:{mint}", address=mint, risk_flags=[])
    for key, value in over.items():
        if key in ("identity_status", "market_status", "flow_quality"):
            c["quality"] = {**c["quality"], key: value}
        else:
            c[key] = value
    return c


def record(tag: str = "TokA", *, observed: Any = None, market_at: Any = None,
           archived: Any = None, version: int = 2, causal: bool = True, kind: str = "scout",
           c: dict[str, Any] | None = None, **over: Any) -> EvidenceRecord:  # fmt: skip
    """A stored-like Scout record (pure tests: no database)."""
    observed = observed or T
    market_at = market_at or observed
    c = c if c is not None else cand(tag, market_at, **over)
    p = prepare(scout_record(observed, market_at=market_at, cand=c, causal_valid=causal,
                             version=version, asset_id=c["canonical_id"]))  # fmt: skip
    assert p.record_id and p.fingerprint and p.payload_hash
    return EvidenceRecord(
        id=next(_ID), record_id=p.record_id, kind=kind, asset_id=p.asset_id, chain=p.chain,  # type: ignore[arg-type]
        address=p.address, pool_address=p.pool_address, dex=None, provider=p.provider,
        component=p.component, observed_at=observed, provider_at=market_at,
        archived_at=archived or observed, availability="AVAILABLE", reason=None,
        fingerprint=p.fingerprint, payload=json.loads(p.text or "{}"),
        payload_hash=p.payload_hash, versions={}, links=p.links,
    )  # fmt: skip


def reasons(r: EvidenceRecord, now: Any = NOW, **kw: Any) -> tuple[str, ...]:
    return evaluate_candidate(r, now=now, **kw).reasons


# --- admission ---------------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["ACCELERATING", "EARLY"])
def test_fresh_eligible_candidates_are_admitted(stage: str) -> None:
    a = evaluate_candidate(record(stage=stage), now=NOW)
    assert a.admitted and a.reasons == ("ADMITTED",) and a.stage == stage
    assert a.priority is not None and a.age_seconds == 120.0


@pytest.mark.parametrize(
    ("over", "code"),
    [
        ({"eligible": False}, "SCOUT_INELIGIBLE"),
        ({"rank": None}, "UNRANKED"),
        ({"rank": 11}, "RANK_TOO_LOW"),
        ({"data_status": "STALE_CARRIED"}, "STALE_CARRIED"),
        ({"identity_status": "UNVERIFIED"}, "IDENTITY_NOT_VERIFIED"),
        ({"stage": "NEW"}, "STAGE_NOT_ADMISSIBLE"),
        ({"stage": "STEADY"}, "STAGE_NOT_ADMISSIBLE"),
        ({"stage": "CROWDED"}, "STAGE_NOT_ADMISSIBLE"),
        ({"stage": "FADING"}, "STAGE_NOT_ADMISSIBLE"),
        ({"stage": "INSUFFICIENT_DATA"}, "STAGE_NOT_ADMISSIBLE"),
        ({"unconfirmed_stage": "FADING"}, "STAGE_UNCONFIRMED"),
        ({"market_status": "MARKET_COLLAPSE"}, "SCOUT_MARKET_COLLAPSE"),
        ({"flow_quality": "divergent"}, "DIVERGENT_FLOW"),
    ],
)
def test_each_rejection_reason(over: dict[str, Any], code: str) -> None:
    a = evaluate_candidate(record(**over), now=NOW)
    assert not a.admitted and code in a.reasons and a.priority is None


def test_rank_ten_is_admitted() -> None:
    assert evaluate_candidate(record(rank=10), now=NOW).admitted


def test_identity_timing_and_causality_rejections() -> None:
    evm = cand()
    evm |= {"chain": "base", "canonical_id": "base:0x" + "ab" * 20, "address": "0x" + "ab" * 20}
    assert "UNSUPPORTED_CHAIN" in reasons(record(c=evm))
    bad = cand() | {"address": "not-a-mint", "canonical_id": "solana:not-a-mint"}
    assert "INVALID_IDENTITY" in reasons(record(c=bad))
    variant = cand()
    variant["address"] = "m" + variant["address"][1:]  # case variant: another mint
    assert "INVALID_IDENTITY" in reasons(record(c=variant))
    assert "INVALID_SCOUT_TIMING" in reasons(record(version=1))
    assert "INVALID_SCOUT_TIMING" in reasons(record(market_at=T + timedelta(seconds=1)))
    assert "CAUSAL_INVALID" in reasons(record(causal=False))
    assert "NOT_A_SCOUT_RECORD" in reasons(record(kind="decision"))


def test_market_age_boundary_is_exact() -> None:
    r = record()
    assert evaluate_candidate(r, now=T + timedelta(minutes=15)).admitted
    late = evaluate_candidate(r, now=T + timedelta(minutes=15, seconds=1))
    assert late.reasons == ("SCOUT_TOO_OLD",)


def test_several_reasons_are_all_reported() -> None:
    got = reasons(record(eligible=False, rank=None, stage="FADING", flow_quality="divergent"))
    assert {"SCOUT_INELIGIBLE", "UNRANKED", "STAGE_NOT_ADMISSIBLE", "DIVERGENT_FLOW"} <= set(got)


def test_social_never_changes_admission() -> None:
    rejected = cand(stage="STEADY")
    hyped = json.loads(json.dumps(rejected))
    hyped["momentum"] |= {"social_status": "SOCIAL_STRONG", "social_state": "STRONG",
                          "mention_acceleration": 50.0}  # fmt: skip
    hyped["quality"] |= {"social_attribution": "exact", "spam_risk": "low"}
    for c in (rejected, hyped):
        assert reasons(record(c=c)) == ("STAGE_NOT_ADMISSIBLE",)
    ok = cand()
    quiet = json.loads(json.dumps(ok))
    quiet["momentum"] |= {"social_status": "SOCIAL_UNAVAILABLE", "social_state": None}
    assert evaluate_candidate(record(c=quiet), now=NOW).admitted


def test_admission_never_reads_social_records_or_opportunity_output() -> None:
    src = (Path(__file__).parents[1] / "upscale" / "services" / "opportunity_orchestrator"
           / "admission.py").read_text()  # fmt: skip
    assert "social" not in src.lower().replace("social evidence is never read", "")
    assert "scout_momentum" not in src and "opportunity_model" not in src


# --- priority ----------------------------------------------------------------------------------


def test_priority_is_deterministic_and_order_independent() -> None:
    rs = [
        record("TokA", stage="EARLY", rank=1),
        record("TokB", stage="ACCELERATING", rank=3),
        record("TokC", stage="ACCELERATING", rank=2, market_at=T - timedelta(minutes=5)),
        record("TokD", stage="ACCELERATING", rank=2),
    ]
    results = [evaluate_candidate(r, now=NOW) for r in rs]
    expected = [MINTS[t] for t in ("TokD", "TokC", "TokB", "TokA")]
    for order in (results, list(reversed(results)), results[2:] + results[:2]):
        assert [a.canonical_id.split(":")[1] for a in prioritized(order)] == expected


def test_canonical_id_breaks_full_ties() -> None:
    rs = [record(tag, stage="ACCELERATING", rank=1) for tag in ("TokC", "TokA", "TokB")]
    got = [a.canonical_id for a in prioritized(evaluate_candidate(r, now=NOW) for r in rs)]
    assert got == sorted(got)


# --- Scout run settle ------------------------------------------------------------------------


def test_the_newest_settled_run_is_used_and_newer_unsettled_runs_reported() -> None:
    old = [record("TokA", observed=T - timedelta(minutes=30), archived=T - timedelta(minutes=29)),
           record("TokB", observed=T - timedelta(minutes=30), archived=T - timedelta(minutes=29))]  # fmt: skip
    new = [record("TokC", observed=T, archived=T + timedelta(seconds=30)),
           record("TokD", observed=T, archived=T + timedelta(seconds=45))]  # fmt: skip
    now = T + timedelta(seconds=104)  # 59 s after the newest run's last archive write
    runs = group_runs([*new, *old], now)
    assert [(r.decision_time, r.record_count) for r, _ in runs] == [
        (T, 2),
        (T - timedelta(minutes=30), 2),
    ]
    selected, newer = select_run(runs)
    assert selected is not None and selected[0].decision_time == T - timedelta(minutes=30)
    assert [n.decision_time for n in newer] == [T]
    settled, newer2 = select_run(
        group_runs([*new, *old], now + timedelta(seconds=1))
    )  # exactly 60 s
    assert settled is not None and settled[0].decision_time == T and newer2 == ()


# --- Safety reuse preview ----------------------------------------------------------------------


def _row(
    body: dict[str, Any], as_of: Any = T, version: str = "4", sid: int = 1
) -> SafetySnapshotRow:
    _, blob, digest = encode_body(body)
    return SafetySnapshotRow(sid, as_of, version, blob, digest)


def test_safety_reuse_classification() -> None:
    body = _clean_safety(T)
    assert classify_safety("READABLE", _row(body), NOW).reuse == "FRESH_REUSABLE"
    partial = json.loads(json.dumps(body))
    partial["holders"]["status"] = "PARTIAL"
    assert classify_safety("READABLE", _row(partial), NOW).reuse == "FRESH_REUSABLE"
    for section in ("holders", "market"):
        missing = json.loads(json.dumps(body))
        missing[section]["status"] = "UNAVAILABLE"
        assert missing["coverage"]["coverage"] == "COMPLETE"  # coverage never substitutes
        out = classify_safety("READABLE", _row(missing), NOW)
        assert (out.reuse, out.collection_required) == ("NOT_READY", "YES")
    no_auth = json.loads(json.dumps(body))
    no_auth["authority"]["freeze_authority"]["status"] = "PROVIDER_UNAVAILABLE"
    assert classify_safety("READABLE", _row(no_auth), NOW).reuse == "NOT_READY"
    old = classify_safety("READABLE", _row(body, as_of=NOW - timedelta(minutes=45, seconds=1)), NOW)
    assert old.reuse == "REFRESH_RECOMMENDED"
    edge = classify_safety("READABLE", _row(body, as_of=NOW - timedelta(minutes=45)), NOW)
    assert edge.reuse == "FRESH_REUSABLE"
    assert classify_safety("READABLE", None, NOW).reuse == "MISSING"
    assert classify_safety("MISSING", None, NOW).reuse == "MISSING"
    assert classify_safety("INCOMPATIBLE", None, NOW).collection_required == "UNKNOWN"
    assert classify_safety("READABLE", _row(body, version="3"), NOW).reuse == "INCOMPATIBLE"
    tampered = _row(body)
    tampered = SafetySnapshotRow(1, T, "4", zlib.compress(b'{"x":1}'), tampered.body_hash)
    assert classify_safety("READABLE", tampered, NOW).reuse == "INCOMPATIBLE"


# --- dry-run integration -----------------------------------------------------------------------


def safety_store(path: Path, snapshots: list[tuple[str, Any, dict[str, Any]]],
                 used: int = 0, cooldown: Any = None) -> Path:  # fmt: skip
    """The Safety V2 tables the bridge reads, as Safety stores them."""
    c = sqlite3.connect(path)
    c.executescript(
        "CREATE TABLE safety_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE safety_requests (day TEXT, method TEXT, calls INTEGER, ok INTEGER, "
        "rate_limited INTEGER, timeouts INTEGER, failures INTEGER, PRIMARY KEY (day, method));"
        "CREATE TABLE safety_snapshots (id INTEGER PRIMARY KEY, canonical_id TEXT, as_of REAL, "
        "schema_version TEXT, rules_version TEXT, fingerprints_json TEXT, coverage TEXT, "
        "band TEXT, body_zlib BLOB, body_hash TEXT);"
    )
    for cid, as_of, body in snapshots:
        body = json.loads(json.dumps(body))
        mint = cid.split(":")[1]
        body["identity"] |= {"canonical_id": cid, "mint": mint}
        _, blob, digest = encode_body(body)
        c.execute("INSERT INTO safety_snapshots (canonical_id, as_of, schema_version, "
                  "rules_version, fingerprints_json, coverage, band, body_zlib, body_hash) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (cid, as_of.timestamp(), "4", "4", "{}", "COMPLETE", "NO_TRIGGERED_FLAGS",
                   blob, digest))  # fmt: skip
    if used:
        c.execute("INSERT INTO safety_requests VALUES (?, 'getAccountInfo', ?, ?, 0, 0, 0)",
                  (NOW.strftime("%Y-%m-%d"), used, used))  # fmt: skip
    if cooldown is not None:
        c.execute("INSERT INTO safety_meta VALUES ('provider.cooldown_until', ?)",
                  (repr(cooldown.timestamp()),))  # fmt: skip
    c.commit()
    c.close()
    return path


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    """One settled Scout run (TokA..TokD) at T; Safety for TokA (fresh, ready) and TokB
    (holders never collected); an Opportunity decision for TokA."""
    arch = Archive(tmp_path / "evidence.sqlite3")
    arch.add(scout_record(T, cand=cand("TokA", rank=2), asset_id=f"solana:{MINTS['TokA']}"))
    arch.add(scout_record(T, cand=cand("TokB", rank=1, stage="EARLY"),
                          asset_id=f"solana:{MINTS['TokB']}"))  # fmt: skip
    arch.add(scout_record(T, cand=cand("TokC", rank=3), asset_id=f"solana:{MINTS['TokC']}"))
    arch.add(scout_record(T, cand=cand("TokD", stage="FADING", rank=None, eligible=False),
                          asset_id=f"solana:{MINTS['TokD']}"))  # fmt: skip
    arch.add(social_record(T))  # social never enters admission
    arch.close()
    no_holders = json.loads(json.dumps(_clean_safety(T)))
    no_holders["holders"]["status"] = "UNAVAILABLE"
    safety = safety_store(tmp_path / "safety.sqlite3", [
        (f"solana:{MINTS['TokA']}", T, _clean_safety(T)),
        (f"solana:{MINTS['TokB']}", T, no_holders),
    ], used=37)  # fmt: skip
    return {"evidence": str(arch.path), "safety": str(safety),
            "opportunity": str(tmp_path / "opportunity.sqlite3"), "tmp": tmp_path}  # fmt: skip


def _plans(report: Any) -> dict[str, Any]:
    return {p.admission.canonical_id.split(":")[1]: p for p in report.candidates}


def test_dry_run_projects_actions_in_priority_order(world: dict[str, Any]) -> None:
    rep = dry_run(world["evidence"], world["safety"], world["opportunity"], NOW, env={})
    assert rep.selected_run is not None and rep.selected_run.settled
    assert rep.selected_run.record_count == 4  # the social record isn't a Scout record
    plans = _plans(rep)
    a, b, c, d = (plans[MINTS[t]] for t in ("TokA", "TokB", "TokC", "TokD"))
    assert [a.order, c.order, b.order] == [1, 2, 3]  # ACCELERATING rank 2, 3, then EARLY
    assert a.action == "REUSE_SAFETY_AND_WOULD_DECIDE" and a.safety.reuse == "FRESH_REUSABLE"
    assert b.action == "WOULD_REQUIRE_SAFETY_COLLECTION" and b.safety.reuse == "NOT_READY"
    assert c.action == "WOULD_REQUIRE_SAFETY_COLLECTION" and c.safety.reuse == "MISSING"
    assert d.action == "REJECT" and d.safety is None and d.order is None
    assert rep.budget.requests_used_today == 37 and rep.budget.limit_source == "NOT_KNOWN"
    assert rep.budget.remaining is None and rep.budget.cost_bound == "COST_BOUND_UNKNOWN"
    assert rep.opportunity_db_status == "MISSING"
    assert not Path(world["opportunity"]).exists()


def test_known_budget_cooldown_and_exhaustion_defer(world: dict[str, Any]) -> None:
    rep = dry_run(world["evidence"], world["safety"], world["opportunity"], NOW,
                  env={SAFETY_BUDGET_ENV: "500"})  # fmt: skip
    assert (rep.budget.daily_limit, rep.budget.remaining, rep.budget.limit_source) == (
        500,
        463,
        "ENV",
    )
    spent = dry_run(world["evidence"], world["safety"], world["opportunity"], NOW,
                    env={SAFETY_BUDGET_ENV: "37"})  # fmt: skip
    assert _plans(spent)[MINTS["TokC"]].action == "DEFER_BUDGET"
    assert _plans(spent)[MINTS["TokA"]].action == "REUSE_SAFETY_AND_WOULD_DECIDE"
    cool = safety_store(world["tmp"] / "cool.sqlite3", [], cooldown=NOW + timedelta(minutes=5))
    rep2 = dry_run(world["evidence"], str(cool), world["opportunity"], NOW, env={})
    assert rep2.budget.cooldown_until == NOW + timedelta(minutes=5)
    assert {p.action for p in rep2.candidates if p.admission.admitted} == {"DEFER_BUDGET"}


def test_an_unsettled_newest_run_waits(world: dict[str, Any]) -> None:
    early = T + timedelta(seconds=30)  # less than 60 s after the run was archived
    rep = dry_run(world["evidence"], world["safety"], world["opportunity"], early, env={})
    assert rep.selected_run is not None and not rep.selected_run.settled
    plans = _plans(rep)
    for tag in ("TokA", "TokB", "TokC"):  # otherwise admissible: only waiting
        p = plans[MINTS[tag]]
        assert p.admission.reasons == ("RUN_NOT_SETTLED",) and p.action == "WAITING_FOR_DATA"
    d = plans[MINTS["TokD"]]  # rejected for its own reasons too
    assert "RUN_NOT_SETTLED" in d.admission.reasons and d.action == "REJECT"


def test_opportunity_decisions_are_shown_but_never_change_admission(world: dict[str, Any]) -> None:
    before = dry_run(world["evidence"], world["safety"], world["opportunity"], NOW, env={})
    repo = OpportunityRepository(world["opportunity"])
    record_from_paths(repo, f"solana:{MINTS['TokA']}", "HISTORICAL_REPLAY", world["evidence"],
                      world["safety"], as_of=T, clock=lambda: T + timedelta(seconds=5))  # fmt: skip
    repo.close()
    after = dry_run(world["evidence"], world["safety"], world["opportunity"], NOW, env={})
    a = _plans(after)[MINTS["TokA"]]
    assert a.opportunity.decision_id == 1 and a.opportunity.decision_at == T
    assert a.opportunity.decision in ("SKIP", "WATCH", "ENTER")
    strip = [(p.admission, p.action, p.order) for p in before.candidates]
    assert strip == [(p.admission, p.action, p.order) for p in after.candidates]


def test_missing_and_foreign_stores_are_reported_not_created(tmp_path: Path) -> None:
    rep = dry_run(str(tmp_path / "e.sqlite3"), str(tmp_path / "s.sqlite3"),
                  str(tmp_path / "o.sqlite3"), NOW, env={})  # fmt: skip
    assert (rep.archive_status, rep.safety_db_status, rep.opportunity_db_status) == (
        "MISSING", "MISSING", "MISSING")  # fmt: skip
    assert rep.selected_run is None and not list(tmp_path.iterdir())
    foreign = tmp_path / "radar.sqlite3"
    c = sqlite3.connect(foreign)
    c.execute("CREATE TABLE radar_tokens (x TEXT)")
    c.commit()
    c.close()
    rep2 = dry_run(str(foreign), str(foreign), str(foreign), NOW, env={})
    assert (rep2.safety_db_status, rep2.opportunity_db_status) == ("INCOMPATIBLE", "INCOMPATIBLE")


# --- zero writes, CLI, isolation --------------------------------------------------------------


def _snapshot(paths: list[str]) -> list[tuple[bytes, int]]:
    return [(Path(p).read_bytes(), Path(p).stat().st_mtime_ns) for p in paths]


def test_the_dry_run_writes_nothing_and_opens_stores_read_only(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = OpportunityRepository(world["opportunity"])
    record_from_paths(repo, f"solana:{MINTS['TokA']}", "HISTORICAL_REPLAY", world["evidence"],
                      world["safety"], as_of=T, clock=lambda: T + timedelta(seconds=5))  # fmt: skip
    repo.close()
    paths = [world["evidence"], world["safety"], world["opportunity"]]
    files_before = sorted(p.name for p in world["tmp"].iterdir())
    before = _snapshot(paths)
    opened: list[str] = []
    real = sqlite3.connect

    def spy(database: Any, *a: Any, **k: Any) -> sqlite3.Connection:
        opened.append(str(database))
        return real(database, *a, **k)

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network")

    monkeypatch.setattr(sqlite3, "connect", spy)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    for cmd in (["scan", "--dry-run"], ["status"]):
        assert cli(world, *cmd)[0] == 0
    assert _snapshot(paths) == before
    assert sorted(p.name for p in world["tmp"].iterdir()) == files_before
    assert opened and all(o.startswith("file:") and o.endswith("?mode=ro") for o in opened)
    counts = real(world["opportunity"]).execute("SELECT COUNT(*) FROM opportunity_decisions")
    assert counts.fetchone() == (1,)  # the one made above; the dry-run decided nothing


def cli(world: dict[str, Any], *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["--evidence-db", world["evidence"], "--safety-db", world["safety"],
                 "--opportunity-db", world["opportunity"], *argv], out=out,
                clock=lambda: NOW)  # fmt: skip
    return code, out.getvalue()


def test_cli_scan_requires_dry_run_and_reports(world: dict[str, Any],
                                               monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    for key in ("UPSCALE_HELIUS_API_KEY", "UPSCALE_SOLANA_RPC_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "TESTSECRET-b1")
    code, out = cli(world, "scan")
    assert code == 2 and LIVE_NOT_IMPLEMENTED in out
    code, out = cli(world, "scan", "--dry-run", "--limit", "1")
    assert code == 0 and "admitted (priority order): 3" in out
    assert out.count("-> REUSE_SAFETY_AND_WOULD_DECIDE") == 1  # only the first is shown
    assert "-> REJECT" in out and LIVE_NOT_IMPLEMENTED in out and "DRY RUN" in out
    code, status = cli(world, "status")
    assert code == 0 and "candidates: 4   admitted: 3   rejected: 1" in status
    assert "orchestrator db: none (B1)" in status and "COST_BOUND_UNKNOWN" in status
    assert "TESTSECRET" not in out + status


def test_the_package_imports_no_safety_radar_provider_scout_service_or_execution() -> None:
    pkg = Path(__file__).parents[1] / "upscale" / "services" / "opportunity_orchestrator"
    for path in pkg.glob("*.py"):
        text = path.read_text()
        assert "safety_v2" not in text, path.name  # Safety V2's own isolation rule
        imports = [ln for ln in text.splitlines() if ln.startswith(("import ", "from "))]
        tokens = {t for ln in imports for t in re.split(r"[ .,()]+", ln)}
        for word in ("radar", "httpx", "httpx2", "anthropic", "background_scout", "scout",
                     "shadow", "execution", "socket", "requests", "solana_chain", "solana_dex",
                     "dexscreener"):  # fmt: skip
            assert word not in tokens, (path.name, word)
    assert not list(pkg.glob("repository.py")) + list(pkg.glob("queue.py")) + list(
        pkg.glob("worker.py")) + list(pkg.glob("safety_adapter.py"))  # fmt: skip


def test_frozen_b3_policy_is_recorded() -> None:
    assert not set(policy.TOKEN_EVIDENCE_OUTCOMES) & set(policy.INFRASTRUCTURE_FAILURES)
    assert {"PROVIDER_UNAVAILABLE", "SAFETY_BUDGET_EXHAUSTED", "SAFETY_COOLDOWN",
            "PROVIDER_NOT_CONFIGURED"} <= set(policy.INFRASTRUCTURE_FAILURES)  # fmt: skip
    assert {"NOT_A_MINT", "NO_POOLS", "HOLDERS_PARTIAL"} <= set(policy.TOKEN_EVIDENCE_OUTCOMES)
    assert len(policy.DEFERRED_B3_ITEMS) == 6
    assert POLICY.max_rank == 10 and POLICY.admission_market_age_s == 900
    assert POLICY.settle_s == 60 and POLICY.safety_reuse_max_age_s == 2700
