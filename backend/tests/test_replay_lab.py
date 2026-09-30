"""Replay Lab: historical replay through production logic, with no lookahead.

Every test uses temporary databases (a synthetic Scout archive written by the production
Scout store, and a throwaway replay store) and a fake GeckoTerminal provider: no network,
never ~/.upscale.
"""

import asyncio
import hashlib
import io
import os
import sqlite3
import tokenize
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest

import upscale.services as services
from upscale.orchestrator import Orchestrator
from upscale.schemas import AgentResult, AssetRef, ChatMessage, ChatRequest
from upscale.services.clock import frozen_now, frozen_time, utcnow
from upscale.services.market_data import (
    TIMEFRAME_SECONDS,
    AssetNotFoundError,
    Candle,
    CandleSeries,
    ProviderRateLimitedError,
    Timeframe,
)
from upscale.services.outcomes.config import AnalyticsConfig, OutcomeConfig
from upscale.services.outcomes.metrics import path_from_candles
from upscale.services.quota import LaneLimiter
from upscale.services.replay_lab import analytics, findings
from upscale.services.replay_lab.analyze import analyze_at
from upscale.services.replay_lab.archive import ScoutArchive
from upscale.services.replay_lab.candles import (
    HistoricalCandleFetcher,
    PointInTimeCandles,
    chunk_bounds,
)
from upscale.services.replay_lab.cli import main as cli_main
from upscale.services.replay_lab.clock import (
    DecisionReceipt,
    HistoricalClock,
    LookaheadError,
    utc,
)
from upscale.services.replay_lab.config import (
    AssetSpec,
    ReplayJobConfig,
    SplitConfig,
    default_replay_db,
)
from upscale.services.replay_lab.engine import JobLockedError, ReplayRunner
from upscale.services.replay_lab.quota import (
    LocalBackendProbe,
    ReplayDeferred,
    ReplayGate,
    production_limiter,
)
from upscale.services.replay_lab.sampling import plan
from upscale.services.replay_lab.shadow import evaluate_baselines
from upscale.services.replay_lab.store import (
    DecisionIntegrityError,
    ReplayIsolationError,
    ReplayStore,
    ReplayStoreError,
)
from upscale.services.scout.config import ScoutConfig, ScoutProviderLimits
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.models import ScoutSnapshot
from upscale.services.scout.normalize import Listing, build_candidates
from upscale.services.scout.social.store import SocialStore
from upscale.services.scout.store import ScoutSnapshotStore
from upscale.services.solana_dex import DexPool, TokenRef, WindowStats
from upscale.services.strategy import STRATEGIES

from .test_scout_growth import momentum

SOL = "So11111111111111111111111111111111111111112"
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
T0 = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
NOW = datetime(2026, 2, 1, tzinfo=UTC)


def addr(seed: str) -> str:
    n = int.from_bytes(hashlib.sha256(seed.encode()).digest(), "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return out[:44]


def mint(i: int) -> str:
    return addr(f"mint-{i}")


def pool_address(i: int, variant: str = "") -> str:
    return addr(f"pool-{i}{variant}")


def dex_pool(
    i: int,
    at: datetime,
    *,
    price: float = 1.0,
    liquidity: float = 100_000.0,
    volume_scale: float = 1.0,
    pool: str | None = None,
) -> DexPool:
    v = volume_scale
    return DexPool(
        chain="solana",
        dex="raydium",
        pair_address=pool or pool_address(i),
        base=TokenRef(address=mint(i), symbol=f"TK{i}", name=f"Token {i}"),
        quote=TokenRef(address=SOL, symbol="SOL"),
        price_usd=price,
        liquidity_usd=liquidity,
        market_cap_usd=500_000.0,
        fdv_usd=600_000.0,
        pair_created_at=T0 - timedelta(days=3),
        windows=[
            WindowStats(
                window="m5", buys=int(8 * v), sells=4, volume_usd=900 * v, price_change_pct=0.5
            ),
            WindowStats(
                window="h1", buys=int(60 * v), sells=40, volume_usd=9_000 * v, price_change_pct=2
            ),
            WindowStats(window="h6", buys=300, sells=250, volume_usd=50_000, price_change_pct=5),
            WindowStats(
                window="h24", buys=1000, sells=900, volume_usd=150_000, price_change_pct=10
            ),
        ],
    )


async def record(
    store: ScoutSnapshotStore, pool: DexPool, at: datetime, stage: str = "EARLY"
) -> str:
    """What a live Scout scan stores for one observation."""
    listing = Listing(provider="DEX Screener", kind="new", name="new_pools", fetched_at=at)
    c = build_candidates([pool], listing, ScoutConfig()).candidates[0]
    first = await store.record_seen(c)
    await store.save_snapshot(
        ScoutSnapshot(
            canonical_id=c.canonical_id, observed_at=at, provider="DEX Screener",
            pool_address=c.pool.address, dex=c.pool.dex, metrics=c.metrics,
        ),
        0.0,
    )  # fmt: skip
    await store.save_latest(c.model_copy(update={"first_seen_at": first}))
    # The live ranking run that used this snapshot finishes seconds later (after T).
    await store.record_stages(
        at + timedelta(seconds=5), {c.canonical_id: stage}, {c.canonical_id: (1, 55.0)}
    )
    return c.canonical_id


def build_archive(
    path: Path,
    tokens: int = 1,
    snapshots: int = 1,
    spacing: timedelta = timedelta(hours=2),
    extra: Callable[[ScoutSnapshotStore], Any] | None = None,
) -> Path:
    async def go() -> None:
        store = ScoutSnapshotStore(path)
        for i in range(tokens):
            for k in range(snapshots):
                at = T0 + k * spacing + timedelta(minutes=i)
                await record(store, dex_pool(i, at, price=1.0 + 0.01 * k), at)
        if extra is not None:
            await extra(store)
        store.close()

    asyncio.run(go())
    return path


def flat(ts: float) -> float:
    return 1.0


class FakeGeckoTerminal:
    """Deterministic pool candles from a price function; records every request."""

    name = "GeckoTerminal"

    def __init__(self, price: Callable[[float], float] = flat, fail: Exception | None = None):
        self.price = price
        self.fail = fail
        self.calls: list[tuple[str, str, datetime | None]] = []

    async def fetch_pool_candles(
        self, chain: str, pool: str, timeframe: Timeframe, limit: int, *, symbol: str,
        canonical_id: str | None, now: datetime, before: datetime | None = None,
        contiguous: bool = True, token: str | None = None,
    ) -> CandleSeries:  # fmt: skip
        self.calls.append((pool, timeframe, before))
        if self.fail is not None:
            raise self.fail
        assert before is not None
        sec = TIMEFRAME_SECONDS[timeframe]
        end = int(before.timestamp())
        candles = []
        for k in range(1, limit + 1):
            ts = end - k * sec
            if ts + sec > now.timestamp():
                continue
            p = self.price(ts)
            candles.append(
                Candle(timestamp=utc(ts), open=p, high=p * 1.01, low=p * 0.99, close=p, volume=10.0)
            )
        candles.reverse()
        return CandleSeries(
            symbol=symbol, provider="GeckoTerminal", provider_id=pool, timeframe=timeframe,
            candles=candles, volume_available=True, fetched_at=now,
        )  # fmt: skip


class Harness:
    def __init__(
        self,
        tmp: Path,
        archive: Path,
        provider: FakeGeckoTerminal | None = None,
        gate: ReplayGate | None = None,
        db: str = "replay.sqlite3",
    ):
        self.store = ReplayStore(tmp / db, forbidden=[str(archive)])
        self.archive = ScoutArchive(archive)
        self.provider = provider or FakeGeckoTerminal()
        self.gate = gate or ReplayGate(LaneLimiter(10_000, 60.0))
        self.sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.fetcher = HistoricalCandleFetcher(
            self.store, self.gate, provider=self.provider,  # type: ignore[arg-type]
            wall_clock=lambda: NOW.timestamp(),
        )  # fmt: skip
        self.runner = ReplayRunner(
            self.store, self.archive, self.fetcher, ScoutConfig(), GrowthConfig(),
            wall_clock=lambda: NOW, sleep=sleep,
        )  # fmt: skip

    def run(self, config: ReplayJobConfig, exit_on_pause: bool = False) -> str:
        job_id = self.runner.create_job(config)
        asyncio.run(self.runner.run(job_id, exit_on_pause=exit_on_pause))
        return job_id

    def decisions(self, job_id: str) -> list[str]:
        out = []
        for s in self.store.samples(job_id):
            stored = self.store.decision(s.id)
            if stored is not None:
                out.append(stored[0].model_dump_json())
        return out


def config(**kw: Any) -> ReplayJobConfig:
    base: dict[str, Any] = {"start": T0 - timedelta(hours=1), "end": T0 + timedelta(days=1)}
    return ReplayJobConfig(**(base | kw))


def archive_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Historical clock ------------------------------------------------------------------------


def test_clock_hides_the_future_until_the_decision_is_stored() -> None:
    clock = HistoricalClock(T0, timedelta(hours=26))
    assert clock.phase == "DECIDING" and clock.visible_until == T0
    with pytest.raises(LookaheadError):
        clock.check_request(T0 + timedelta(seconds=1), "candles")
    with pytest.raises(LookaheadError):
        clock.check_time(T0 + timedelta(minutes=1), "a snapshot")
    with pytest.raises(LookaheadError):
        clock.reveal(DecisionReceipt(1, T0, ""))  # no stored decision: no reveal
    with pytest.raises(LookaheadError):
        clock.reveal(DecisionReceipt(1, T0 + timedelta(minutes=1), "abc"))  # another decision
    clock.reveal(DecisionReceipt(1, T0, "abc"))
    assert clock.phase == "REVEALED"
    clock.check_request(T0 + timedelta(hours=24), "outcome candles")
    with pytest.raises(LookaheadError):
        clock.check_request(T0 + timedelta(hours=27), "beyond the longest horizon")
    with pytest.raises(LookaheadError):
        clock.check_decision_phase("Analyze")


def test_future_candles_never_reach_the_decision() -> None:
    t = T0 + timedelta(seconds=30)
    clock = HistoricalClock(t, timedelta(hours=26))
    candles = PointInTimeCandles(clock, "solana", mint(0), pool_address(0))
    ts = [T0 - timedelta(minutes=m) for m in (3, 2, 1)] + [T0, T0 + timedelta(minutes=5)]
    candles.load("1m", [Candle(timestamp=x, open=1, high=1, low=1, close=1, volume=1) for x in ts])
    closed = candles.closed("1m")
    # The candle opened 30 s before T closes after T: it holds post-T trades.
    assert [c.timestamp for c in closed] == ts[:3]
    series = candles.series_at("1m", 10, "TK0", None)
    assert series.candles[-1].timestamp == ts[2] and series.as_of == t
    with pytest.raises(LookaheadError):
        candles.window("1m", t, t + timedelta(minutes=10))  # outcome window before reveal
    with pytest.raises(LookaheadError):
        clock.check_candles(
            [Candle(timestamp=T0, open=1, high=1, low=1, close=1)], timedelta(minutes=1), "x"
        )


def test_frozen_clock_is_scoped_and_utc() -> None:
    assert frozen_time() is None
    with frozen_now(T0):
        assert utcnow() == T0

        async def inner() -> datetime:
            return utcnow()

        assert asyncio.run(inner()) == T0  # tasks started inside see T too
    assert frozen_time() is None and utcnow() > T0
    with pytest.raises(ValueError):
        with frozen_now(datetime(2026, 1, 1)):
            pass


# --- Strict no-lookahead, end to end ---------------------------------------------------------


def test_decision_is_identical_whatever_happens_after_t(tmp_path: Path) -> None:
    archive = build_archive(tmp_path / "scout.sqlite3")
    t = T0.timestamp()

    def calm(ts: float) -> float:
        return 1.0

    def moon(ts: float) -> float:  # identical up to T, 50x afterwards
        return 1.0 if ts < t else 50.0

    a = Harness(tmp_path, archive, FakeGeckoTerminal(calm), db="a.sqlite3")
    b = Harness(tmp_path, archive, FakeGeckoTerminal(moon), db="b.sqlite3")
    ja, jb = a.run(config()), b.run(config())
    assert a.decisions(ja) and a.decisions(ja) == b.decisions(jb)
    # The future data was really there, and only the outcomes saw it.
    sa = a.store.samples(ja)[0]
    sb = b.store.samples(jb)[0]
    ra = {o.horizon: o.price.return_pct for o in a.store.outcomes(sa.id) if o.price}
    rb = {o.horizon: o.price.return_pct for o in b.store.outcomes(sb.id) if o.price}
    assert ra["1h"] != rb["1h"] and rb["1h"] > 1000


def test_future_liquidity_volume_and_stage_are_ignored(tmp_path: Path) -> None:
    async def later(store: ScoutSnapshotStore) -> None:
        at = T0 + timedelta(minutes=10)
        huge = dex_pool(0, at, price=5.0, liquidity=10_000_000.0, volume_scale=50.0)
        await record(store, huge, at, stage="FADING")

    plain = build_archive(tmp_path / "plain.sqlite3")
    future = build_archive(tmp_path / "future.sqlite3", extra=later)
    a = Harness(tmp_path, plain, db="a.sqlite3")
    b = Harness(tmp_path, future, db="b.sqlite3")
    ja, jb = a.run(config()), b.run(config())
    da, db = a.decisions(ja), b.decisions(jb)
    assert len(da) == 1 and da == db  # the T + 10 min snapshot / stage changed nothing
    record_b = b.store.decision(b.store.samples(jb)[0].id)
    assert record_b is not None
    features = record_b[0].features
    assert features["liquidity_usd"] == 100_000.0 and features["volume_h1_usd"] == 9_000.0


def test_future_social_is_ignored_and_past_social_is_labeled(tmp_path: Path) -> None:
    cid = f"solana:{mint(0)}"

    async def social(computed_at: datetime) -> Callable[[ScoutSnapshotStore], Any]:
        async def write(store: ScoutSnapshotStore) -> None:
            s = SocialStore(store.path)
            await s.add_momentum(momentum(mint(0), "ACCELERATING", computed_at=computed_at))
            s.close()

        return write

    after = build_archive(
        tmp_path / "after.sqlite3", extra=asyncio.run(social(T0 + timedelta(minutes=1)))
    )
    before = build_archive(
        tmp_path / "before.sqlite3", extra=asyncio.run(social(T0 - timedelta(minutes=5)))
    )
    ha = Harness(tmp_path, after, db="a.sqlite3")
    hb = Harness(tmp_path, before, db="b.sqlite3")
    ja = ha.run(config(mode="MARKET_PLUS_SOCIAL"))
    jb = hb.run(config(mode="MARKET_PLUS_SOCIAL"))
    ra = ha.store.decision(ha.store.samples(ja)[0].id)
    rb = hb.store.decision(hb.store.samples(jb)[0].id)
    assert ra is not None and rb is not None
    assert ra[0].social["status"] == "SOCIAL_UNAVAILABLE"
    assert ra[0].scout.social_status == "SOCIAL_UNAVAILABLE"
    assert rb[0].social["status"] == "RECORDED_AT_OR_BEFORE_T"
    assert rb[0].social["state"] == "ACCELERATING" and rb[0].social["age_minutes"] == 5.0
    assert rb[0].scout.social_status == "SOCIAL_ACCELERATING"
    assert cid == ra[0].asset_id


def test_market_only_mode_marks_social_unavailable(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"))
    job = h.run(config())
    rec = h.store.decision(h.store.samples(job)[0].id)
    assert rec is not None
    assert rec[0].social == {
        "mode": "MARKET_ONLY",
        "status": "SOCIAL_UNAVAILABLE",
        "reason": "MARKET_ONLY mode",
    }


def test_lookahead_fails_the_job_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"))
    real = ScoutArchive.snapshots

    def leaky(self: ScoutArchive, *args: Any, **kw: Any) -> list[ScoutSnapshot]:
        out = real(self, *args, **kw)
        if out:
            out.append(
                out[-1].model_copy(update={"observed_at": out[-1].observed_at + timedelta(hours=1)})
            )
        return out

    monkeypatch.setattr(ScoutArchive, "snapshots", leaky)
    job = h.runner.create_job(config())
    with pytest.raises(LookaheadError):
        asyncio.run(h.runner.run(job))
    state = h.store.job(job)
    assert (
        state is not None
        and state.status == "FAILED"
        and "Lookahead" in (state.status_reason or "")
    )
    assert h.decisions(job) == []  # nothing contaminated was stored


def test_pool_created_after_t_is_refused(tmp_path: Path) -> None:
    archive = ScoutArchive(build_archive(tmp_path / "scout.sqlite3"))
    cid = f"solana:{mint(0)}"
    assert archive.pool_metadata(cid, pool_address(0), T0) is not None
    with pytest.raises(LookaheadError):
        archive.pool_metadata(cid, pool_address(0), T0 - timedelta(days=4))


def test_replay_uses_decision_time_not_wall_clock_or_live_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kw: Any) -> Any:
        raise AssertionError("replay must never use live provider state")

    for name in ("prefetch", "pool_candles", "dex_market", "onchain_safety", "market_snapshot"):
        monkeypatch.setattr(services.provider_registry, name, forbidden)
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"))
    job = h.run(config())
    rec = h.store.decision(h.store.samples(job)[0].id)
    assert rec is not None
    opportunity = rec[0].agents["opportunity"].findings
    # The pool was created 3 days before T: the profile ages it at T, not today.
    assert opportunity["asset_profile"]["pool_age_days"] == 3
    dex = rec[0].agents["dex_market"].findings["snapshot"]
    assert dex["fetched_at"].startswith("2026-01-10T12:00") and round(dex["pool_age_hours"]) == 72
    tech = rec[0].agents["technical_analysis"].findings
    assert tech["as_of"].startswith("2026-01-10T12:00")
    assert datetime.fromisoformat(tech["last_candle_at"]) + timedelta(hours=4) <= T0
    assert rec[0].evidence_latest_at <= rec[0].decision_at


# --- Production logic reuse -------------------------------------------------------------------


def test_replay_builds_the_same_agent_context_as_live_analyze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    async def capture(
        self: Orchestrator, names: list[str], context: Any, timings: Any = None
    ) -> list[AgentResult]:
        captured.append((list(names), context))
        return []

    async def no_prefetch(*args: Any) -> None:
        return None

    monkeypatch.setattr(Orchestrator, "run_agents", capture)
    monkeypatch.setattr(services.provider_registry, "prefetch", no_prefetch)
    token, symbol = mint(0), "TK0"
    live = ChatRequest(
        messages=[ChatMessage(role="user", content=f"Analyze {symbol}: {token} on Solana")],
        asset=AssetRef(
            chain="solana",
            address=token,
            symbol=symbol,
            name="Token 0",
            pool_address=pool_address(0),
        ),
    )
    asyncio.run(Orchestrator().respond(live))
    clock = HistoricalClock(T0, timedelta(hours=26))
    candles = PointInTimeCandles(clock, "solana", token, pool_address(0))
    asyncio.run(
        analyze_at(clock, chain="solana", token=token, symbol=symbol, name="Token 0",
                   pool=dex_pool(0, T0), market_provider="DEX Screener", candles=candles)
    )  # fmt: skip
    (live_agents, live_ctx), (replay_agents, replay_ctx) = captured
    assert live_agents == replay_agents
    for field in (
        "query",
        "assets",
        "assets_source",
        "asset_identity",
        "timeframe",
        "timeframe_source",
    ):
        assert getattr(live_ctx, field) == getattr(replay_ctx, field), field
    assert live_ctx.trade.model_dump() == replay_ctx.trade.model_dump()


def test_replay_scout_matches_the_live_recorded_stage(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3", tokens=2, snapshots=3))
    job = h.run(config())
    samples = h.store.samples(job, ("COMPLETE",))
    assert samples
    for s in samples:
        rec = h.store.decision(s.id)
        assert rec is not None and rec[0].scout.status == "RECONSTRUCTED"
        assert rec[0].scout.stage and rec[0].scout.score is not None
        fid = h.store.fidelity(s.id)
        assert fid is not None and fid.live_stage == "EARLY"  # what the archive recorded


# --- Isolation --------------------------------------------------------------------------------


def test_replay_store_refuses_live_databases(tmp_path: Path) -> None:
    archive = build_archive(tmp_path / "scout.sqlite3")
    with pytest.raises(ReplayIsolationError):
        ReplayStore(archive, forbidden=[str(archive)])  # by path
    with pytest.raises(ReplayIsolationError):
        ReplayStore(archive)  # by content: live Scout tables
    assert Path(default_replay_db()) != Path.home() / ".upscale" / "replay.sqlite3"
    assert os.environ["UPSCALE_SCOUT_DB"] != str(Path.home() / ".upscale" / "scout.sqlite3")


def test_replay_never_modifies_the_live_database(tmp_path: Path) -> None:
    archive = build_archive(tmp_path / "scout.sqlite3", tokens=2, snapshots=2)
    before = archive_sha(archive)
    h = Harness(tmp_path, archive)
    job = h.run(config())
    findings.generate(h.store, "1h")
    assert archive_sha(archive) == before
    with pytest.raises(sqlite3.OperationalError):
        h.archive._db().execute("DELETE FROM scout_snapshots")  # read-only connection
    db = sqlite3.connect(tmp_path / "replay.sqlite3")
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not names & {"scout_snapshots", "scout_outcome_observations", "decision_observations"}
    for table in ("replay_jobs", "replay_samples", "replay_decisions", "replay_outcomes"):
        origins = {r[0] for r in db.execute(f"SELECT DISTINCT origin FROM {table}")}
        assert origins == {"HISTORICAL_REPLAY"}, table
    assert h.store.job(job) is not None


def test_decisions_are_immutable_after_reveal(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"))
    job = h.run(config())
    sample = h.store.samples(job)[0]
    assert sample.status == "COMPLETE" and h.store.outcomes(sample.id)
    db = sqlite3.connect(tmp_path / "replay.sqlite3")
    for sql in (
        "UPDATE replay_decisions SET record_json = '{}'",
        "DELETE FROM replay_decisions",
        "UPDATE replay_outcomes SET status = 'COMPLETE'",
        "DELETE FROM replay_outcomes",
        "UPDATE replay_samples SET split = 'CALIBRATION' WHERE split != 'CALIBRATION'",
        "UPDATE replay_samples SET decision_at = decision_at + 1",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            with db:
                db.execute(sql)
    record, receipt = h.store.decision(sample.id) or (None, None)
    assert record is not None and receipt is not None
    with pytest.raises(ReplayStoreError):
        h.store.add_decision(sample.id, record)  # never decided twice
    # Tampering (with the guard removed) is caught by the hash check.
    with db:
        db.execute("DROP TRIGGER replay_decisions_no_update")
        db.execute("UPDATE replay_decisions SET record_json = replace(record_json, 'wait', 'buy')")
    with pytest.raises(DecisionIntegrityError):
        h.store.decision(sample.id)


def test_outcomes_need_the_stored_decision(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3", snapshots=2))
    job = h.runner.create_job(config())
    first = h.store.samples(job)[0]
    db = sqlite3.connect(tmp_path / "replay.sqlite3")
    with pytest.raises(sqlite3.DatabaseError):
        with db:
            db.execute(
                "INSERT INTO replay_outcomes VALUES (?, '1h', 60, 'HISTORICAL_REPLAY', 'COMPLETE', '{}', 0)",
                (first.id,),
            )
    with pytest.raises(ReplayStoreError):
        h.store.add_outcomes(DecisionReceipt(first.id, first.plan.decision_at, "forged"), [], None)  # type: ignore[arg-type]


# --- Sampling and splits ----------------------------------------------------------------------


def test_spacing_per_asset_cap_and_diversity(tmp_path: Path) -> None:
    archive = ScoutArchive(
        build_archive(
            tmp_path / "scout.sqlite3", tokens=4, snapshots=8, spacing=timedelta(minutes=20)
        )
    )
    samples, report = plan(config(min_spacing_minutes=60, max_per_asset=2), archive, NOW)
    by_asset: dict[str, list[datetime]] = {}
    for s in samples:
        by_asset.setdefault(s.asset_id, []).append(s.decision_at)
    assert len(by_asset) == 4 and all(len(v) <= 2 for v in by_asset.values())
    for times in by_asset.values():
        times.sort()
        assert all(b - a >= timedelta(minutes=60) for a, b in zip(times, times[1:], strict=False))
    assert report.spacing_dropped > 0 and report.per_asset_dropped > 0
    # max_samples round-robins: 4 samples = one per asset, not 4 of the busiest token.
    few, _ = plan(config(max_samples=4, min_spacing_minutes=60), archive, NOW)
    assert len({s.asset_id for s in few}) == 4


def test_time_split_is_deterministic_and_purges_boundary_overlap(tmp_path: Path) -> None:
    archive = ScoutArchive(
        build_archive(tmp_path / "scout.sqlite3", tokens=5, snapshots=4, spacing=timedelta(hours=6))
    )
    cfg = config(end=T0 + timedelta(days=2), max_per_asset=10)
    a, _ = plan(cfg, archive, NOW)
    b, _ = plan(cfg, archive, NOW)
    assert a == b and len(a) == 20
    splits = [s.split for s in sorted(a, key=lambda s: s.decision_at)]
    assert splits.count("CALIBRATION") == 14 and splits.count("VALIDATION") == 3
    assert splits.count("HOLDOUT") == 3
    assert splits == sorted(splits, key=["CALIBRATION", "VALIDATION", "HOLDOUT"].index)
    val_start = min(s.decision_at for s in a if s.split == "VALIDATION")
    for s in a:
        if s.split == "CALIBRATION":
            assert s.purged == (s.decision_at + timedelta(hours=24) > val_start)
    assert any(s.purged for s in a)


def test_holdout_windows_are_sticky_across_jobs(tmp_path: Path) -> None:
    archive_path = build_archive(
        tmp_path / "scout.sqlite3", tokens=5, snapshots=4, spacing=timedelta(hours=6)
    )
    h = Harness(tmp_path, archive_path)
    first = h.runner.create_job(config(end=T0 + timedelta(hours=13)))
    holdout = [s.plan.decision_at for s in h.store.samples(first) if s.plan.split == "HOLDOUT"]
    assert holdout
    second = h.runner.create_job(
        config(
            end=T0 + timedelta(days=2),
            split=SplitConfig(calibration_pct=90, validation_pct=5, holdout_pct=5),
        )
    )
    again = {s.plan.decision_at: s.plan.split for s in h.store.samples(second)}
    assert all(again[t] == "HOLDOUT" for t in holdout)


def test_holdout_is_never_used_without_an_explicit_final_evaluation(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3", tokens=4, snapshots=2))
    h.run(config())
    with pytest.raises(PermissionError):
        analytics.summarize(h.store, "1h", "stage", ("HOLDOUT",))
    with pytest.raises(PermissionError):
        evaluate_baselines(h.store, "1h", ("CALIBRATION", "HOLDOUT"))
    assert h.store.holdout_accesses() == 0
    default = analytics.summarize(h.store, "1h", "split")
    assert {c.group for c in default.cohorts} <= {"CALIBRATION", "VALIDATION"}
    final = analytics.summarize(h.store, "1h", "split", ("HOLDOUT",), final_evaluation=True)
    assert final.decisions > 0 and h.store.holdout_accesses() == 1
    with pytest.raises(sqlite3.DatabaseError):
        h.store.add_findings([{"split_used": "HOLDOUT", "job_ids": [], "kind": "x", "subject": "y",
                               "horizon": "1h", "statement": "s", "validation": "v", "evidence": {}}])  # fmt: skip


def test_findings_use_calibration_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3", tokens=3, snapshots=2))
    h.run(config())
    seen: list[tuple[str, ...]] = []
    real = ReplayStore.analysis_rows

    def spy(self: ReplayStore, horizon: str, splits: Any, *a: Any, **kw: Any) -> Any:
        seen.append(tuple(splits))
        return real(self, horizon, splits, *a, **kw)

    monkeypatch.setattr(ReplayStore, "analysis_rows", spy)
    out = findings.generate(h.store, "1h", cfg=AnalyticsConfig(min_sample=1))
    assert seen == [("CALIBRATION",), ("VALIDATION",)]
    assert all(
        f["split_used"] == "CALIBRATION" and f["statement"].startswith("EXPERIMENTAL") for f in out
    )


def test_summary_requires_minimum_samples(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3", tokens=2, snapshots=2))
    h.run(config())
    s = analytics.summarize(h.store, "1h", "stage")
    assert s.cohorts and all(
        c.status == "INSUFFICIENT_SAMPLE" and c.median_return_pct is None for c in s.cohorts
    )
    assert any("no cohort reaches" in w for w in s.warnings)
    loose = analytics.summarize(
        h.store,
        "1h",
        "chain",
        cfg=AnalyticsConfig(min_sample=1, min_sample_outer_percentiles=100),
        min_assets=1,
    )
    ok = [c for c in loose.cohorts if c.status == "OK"]
    assert ok and ok[0].median_return_pct is not None and ok[0].p10_return_pct is None


# --- Jobs: resume, duplicates, interruption ----------------------------------------------------


def test_resume_continues_without_duplicates(tmp_path: Path) -> None:
    busy = {"on": True}

    async def production() -> str | None:
        return "Analyze is active" if busy["on"] else None

    gate = ReplayGate(LaneLimiter(10_000, 60.0), production)
    h = Harness(
        tmp_path, build_archive(tmp_path / "scout.sqlite3", tokens=2, snapshots=2), gate=gate
    )
    job = h.runner.create_job(config())
    result = asyncio.run(h.runner.run(job, exit_on_pause=True))
    assert (
        result.paused
        and result.job.status == "PAUSED"
        and result.pause_reason == "Analyze is active"
    )
    assert h.provider.calls == [] and h.decisions(job) == []
    busy["on"] = False
    done = asyncio.run(h.runner.run(job))
    assert done.job.status == "COMPLETE" and done.job.counts == {"COMPLETE": done.job.planned}
    first = h.decisions(job)
    again = asyncio.run(h.runner.run(job))
    assert again.notes == ["job already complete"] and h.decisions(job) == first
    count = (
        sqlite3.connect(tmp_path / "replay.sqlite3")
        .execute("SELECT COUNT(*) FROM replay_decisions")
        .fetchone()[0]
    )
    assert count == done.job.planned


def test_paused_job_sleeps_instead_of_busy_looping(tmp_path: Path) -> None:
    checks = {"n": 0}

    async def production() -> str | None:
        checks["n"] += 1
        return "background Scout is running" if checks["n"] <= 2 else None

    gate = ReplayGate(LaneLimiter(10_000, 60.0), production, busy_retry_seconds=300)
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"), gate=gate)
    job = h.run(config())
    assert h.store.job(job).status == "COMPLETE"  # type: ignore[union-attr]
    assert sum(h.sleeps) >= 600 and len(h.sleeps) <= 12 and min(h.sleeps) >= 5


def test_interrupted_run_resumes_from_the_stored_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"))
    job = h.runner.create_job(config())
    real = ReplayRunner._acquire_outcome_candles

    async def crash(*args: Any, **kw: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ReplayRunner, "_acquire_outcome_candles", crash)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(h.runner.run(job))
    state = h.store.job(job)
    assert state is not None and state.status == "PAUSED"
    sample = h.store.samples(job)[0]
    assert sample.status == "DECIDED"
    decided = h.decisions(job)
    monkeypatch.setattr(ReplayRunner, "_acquire_outcome_candles", real)

    async def never(*args: Any, **kw: Any) -> None:
        raise AssertionError("a stored decision is never re-decided")

    monkeypatch.setattr(ReplayRunner, "_decide", never)
    done = asyncio.run(h.runner.run(job))
    assert done.job.status == "COMPLETE" and h.decisions(job) == decided
    assert len(h.store.outcomes(sample.id)) == len(OutcomeConfig().horizons)


def test_a_live_runner_holds_the_job(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"))
    job = h.runner.create_job(config())
    assert h.store.claim_job(job, os.getppid(), 180.0)  # another live process holds it
    h.store.set_job_status(job, "RUNNING")
    with pytest.raises(JobLockedError):
        asyncio.run(h.runner.run(job))
    h.store.claim_job(job, 2**22 + 12345, 180.0, force=True)  # a process that doesn't exist
    assert asyncio.run(h.runner.run(job)).job.status == "COMPLETE"


# --- Cache and quota ---------------------------------------------------------------------------


def test_historical_candles_are_cached_and_reused(tmp_path: Path) -> None:
    archive = build_archive(tmp_path / "scout.sqlite3", snapshots=3)
    h = Harness(tmp_path, archive)
    h.run(config())
    first_calls = len(h.provider.calls)
    assert first_calls < 3 * 4  # three samples of one pool share aligned chunks
    assert h.fetcher.usage.cache_hits > 0
    h.run(config())  # a new job on the same history: everything comes from the cache
    assert len(h.provider.calls) == first_calls
    rows = (
        sqlite3.connect(tmp_path / "replay.sqlite3")
        .execute("SELECT DISTINCT provider, chain, token_address, pool_address FROM candle_cache")
        .fetchall()
    )
    # Technical input (pool base, as live Analyze) and outcomes (the exact token) are cached
    # apart, never mixed.
    assert sorted(rows) == [("GeckoTerminal", "solana", mint(0), pool_address(0)),
                            ("GeckoTerminal:token-priced", "solana", mint(0), pool_address(0))]  # fmt: skip
    assert chunk_bounds("1m", T0, T0 + timedelta(minutes=1))[0][1] - chunk_bounds(
        "1m", T0, T0 + timedelta(minutes=1)
    )[0][0] == timedelta(minutes=1000)


def test_replay_lane_never_uses_reserved_capacity() -> None:
    now = {"t": 0.0}
    limiter = production_limiter(
        ScoutProviderLimits(calls_per_minute=6, reservations={"refresh": 2, "interactive": 2}),
        clock=lambda: now["t"],
    )
    gate = ReplayGate(limiter)
    asyncio.run(gate.acquire())
    asyncio.run(gate.acquire())
    with pytest.raises(ReplayDeferred) as exc:
        asyncio.run(gate.acquire())
    assert not exc.value.production  # replay's own pacing
    assert limiter.available("interactive") == 2 and limiter.available("refresh") == 2
    now["t"] = 61.0
    assert limiter.try_acquire("interactive")
    with pytest.raises(ReplayDeferred) as exc:
        asyncio.run(gate.acquire())  # someone else is using the provider
    assert exc.value.production
    now["t"] = 200.0
    limiter.note_rate_limited()
    with pytest.raises(ReplayDeferred):
        asyncio.run(gate.acquire())  # the provider pushed back recently
    with pytest.raises(ValueError):
        ReplayGate(limiter, lane="interactive")


def test_provider_rate_limit_pauses_the_job(tmp_path: Path) -> None:
    provider = FakeGeckoTerminal(fail=ProviderRateLimitedError("429"))
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"), provider=provider)
    job = h.runner.create_job(config())
    result = asyncio.run(h.runner.run(job, exit_on_pause=True))
    assert result.paused and result.job.status == "PAUSED"
    assert h.gate.limiter.rate_limited_within(60) and h.fetcher.usage.rate_limited == 1


def test_local_backend_probe() -> None:
    up = LocalBackendProbe(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json={"status": "ok"}))
    )
    assert "local UpScale backend is running" in (asyncio.run(up()) or "")

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    down = LocalBackendProbe(transport=httpx2.MockTransport(refuse))
    assert asyncio.run(down()) is None


# --- Outcomes ------------------------------------------------------------------------------------


def test_outcomes_use_the_production_definitions(tmp_path: Path) -> None:
    t = T0.timestamp()

    def path(ts: float) -> float:
        if ts < t:
            return 1.0
        return 1.1 if ts < t + 1800 else 0.95

    h = Harness(
        tmp_path, build_archive(tmp_path / "scout.sqlite3"), provider=FakeGeckoTerminal(path)
    )
    job = h.run(config())
    sample = h.store.samples(job)[0]
    out = {o.horizon: o for o in h.store.outcomes(sample.id)}
    assert set(out) == {"5m", "15m", "1h", "4h", "24h"}
    one = out["1h"]
    assert one.price is not None and one.price.reference_price == 1.0
    assert one.price.return_pct == pytest.approx(-5.0)
    assert one.price.mfe_pct == pytest.approx(11.1)
    assert one.price.mae_pct == pytest.approx(-5.95)
    assert one.price.time_to_mfe_minutes == 0.0
    candles = [
        Candle(timestamp=utc(t + 60 * k), open=path(t + 60 * k), high=path(t + 60 * k) * 1.01,
               low=path(t + 60 * k) * 0.99, close=path(t + 60 * k), volume=10.0)
        for k in range(60)
    ]  # fmt: skip
    expected = path_from_candles(
        candles, timedelta(minutes=1), 1.0, T0, T0 + timedelta(hours=1),
        provider="GeckoTerminal", timeframe="1m", price_drop_pct=90.0,
    )  # fmt: skip
    assert one.price.model_dump() == expected.model_dump()
    assert one.market_status == "ACTIVE" and one.status == "PARTIAL"
    assert any("horizon-end market state not observed" in m for m in one.missing)
    rec = h.store.decision(sample.id)
    assert rec is not None and rec[0].decision is not None
    for key in one.triggers:
        assert key in ("buy_trigger", "sell_trigger", "invalidation")


def test_horizon_end_market_state_from_later_snapshots(tmp_path: Path) -> None:
    async def later(store: ScoutSnapshotStore) -> None:
        at = T0 + timedelta(hours=1, minutes=1)
        await record(store, dex_pool(0, at, liquidity=50_000.0), at, stage="STEADY")

    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3", extra=later))
    job = h.run(config(min_spacing_minutes=120))
    sample = next(s for s in h.store.samples(job) if s.plan.decision_at == T0)
    one = {o.horizon: o for o in h.store.outcomes(sample.id)}["1h"]
    assert one.status == "COMPLETE" and one.market is not None
    assert one.market.liquidity_change_pct == pytest.approx(-50.0)
    assert one.future_stage == "STEADY"


# --- Missing data ---------------------------------------------------------------------------------


def test_candle_only_samples_mark_what_is_unavailable(tmp_path: Path) -> None:
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"))
    spec = AssetSpec(chain="solana", token=mint(7), pool=pool_address(7))
    job = h.run(config(evidence="CANDLES", assets=(spec,), min_spacing_minutes=360, max_samples=2))
    samples = h.store.samples(job)
    assert samples and all(s.plan.universe_basis.startswith("USER_SELECTED") for s in samples)
    rec = h.store.decision(samples[0].id)
    assert rec is not None
    r = rec[0]
    assert r.scout.status == "NOT_RECONSTRUCTIBLE" and "no historical source" in (
        r.scout.reason or ""
    )
    assert r.availability["liquidity_usd"].startswith("UNAVAILABLE")
    assert r.reference_basis == "last_closed_candle"
    assert r.agents["technical_candles_only"].status == "ok"
    outcome = h.store.outcomes(samples[0].id)[0]
    assert outcome.status == "PARTIAL" and any("no historical source" in m for m in outcome.missing)


def test_missing_history_is_skipped_with_a_reason(tmp_path: Path) -> None:
    dead = FakeGeckoTerminal(fail=AssetNotFoundError("gone"))
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3"), provider=dead)
    job = h.run(config())
    s = h.store.samples(job)[0]
    assert s.status == "SKIPPED" and "doesn't know pool" in (s.skip_reason or "")

    async def moved(store: ScoutSnapshotStore) -> None:  # the latest observation is another pool
        at = T0 + timedelta(days=3)
        await record(store, dex_pool(0, at, pool=pool_address(0, "b")), at)

    h2 = Harness(tmp_path, build_archive(tmp_path / "moved.sqlite3", extra=moved), db="b.sqlite3")
    job2 = h2.run(config())
    s2 = h2.store.samples(job2)[0]
    assert s2.status == "SKIPPED" and "immutable pool metadata" in (s2.skip_reason or "")

    quiet = FakeGeckoTerminal(lambda ts: 1.0)
    h3 = Harness(
        tmp_path, build_archive(tmp_path / "quiet.sqlite3"), provider=quiet, db="c.sqlite3"
    )
    far = AssetSpec(chain="solana", token=mint(9), pool=pool_address(9))
    quiet.fail = None
    job3 = h3.runner.create_job(config(evidence="CANDLES", assets=(far,), max_samples=1))

    async def nothing(*a: Any, **kw: Any) -> list[Candle]:
        return []

    h3.fetcher.range = nothing  # type: ignore[method-assign]
    asyncio.run(h3.runner.run(job3))
    s3 = h3.store.samples(job3)[0]
    assert s3.status == "SKIPPED" and "no reference price" in (s3.skip_reason or "")


# --- Safety: configuration and execution --------------------------------------------------------


def test_no_production_configuration_is_changed(tmp_path: Path) -> None:
    snapshot = (
        GrowthConfig().model_dump(), ScoutConfig().model_dump(), OutcomeConfig().model_dump(),
        repr(dict(STRATEGIES)), services.growth_scout_service.config.model_dump(),
        services.scout_config.model_dump(), services.outcome_config.model_dump(),
        {k: v for k, v in os.environ.items() if k.startswith("UPSCALE_")},
    )  # fmt: skip
    h = Harness(tmp_path, build_archive(tmp_path / "scout.sqlite3", tokens=3, snapshots=2))
    h.run(config())
    h.store.add_findings(findings.generate(h.store, "1h", cfg=AnalyticsConfig(min_sample=1)))
    evaluate_baselines(h.store, "1h")
    after = (
        GrowthConfig().model_dump(), ScoutConfig().model_dump(), OutcomeConfig().model_dump(),
        repr(dict(STRATEGIES)), services.growth_scout_service.config.model_dump(),
        services.scout_config.model_dump(), services.outcome_config.model_dump(),
        {k: v for k, v in os.environ.items() if k.startswith("UPSCALE_")},
    )  # fmt: skip
    assert snapshot == after


def test_replay_lab_has_no_execution_or_key_path() -> None:
    root = Path(__file__).resolve().parents[1] / "upscale" / "services" / "replay_lab"
    forbidden = {"private_key", "privatekey", "keypair", "mnemonic", "seed_phrase", "wallet",
                 "sign_transaction", "send_transaction", "sendtransaction", "swap", "place_order",
                 "submit_order", "create_order", "post", "put", "patch"}  # fmt: skip
    for path in root.glob("*.py"):
        names = {
            tok.string.lower()
            for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
            if tok.type == tokenize.NAME
        }
        assert not names & forbidden, (path.name, names & forbidden)


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_plan_status_and_summary(tmp_path: Path) -> None:
    archive = build_archive(tmp_path / "scout.sqlite3", tokens=2, snapshots=2)
    db = tmp_path / "cli.sqlite3"
    base = ["--db", str(db), "--archive", str(archive)]
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main([*base, "run", "--start", "2026-01-10T11:00", "--end", "2026-01-11",
                         "--chains", "solana", "--max-samples", "10", "--min-spacing-minutes", "60",
                         "--mode", "MARKET_ONLY", "--plan-only"]) == 0  # fmt: skip
        assert cli_main([*base, "status"]) == 0
        assert cli_main([*base, "summary", "--horizon", "1h", "--group-by", "stage"]) == 0
        assert cli_main([*base, "summary", "--include-holdout"]) == 1
    text = out.getvalue()
    assert "planned replay-" in text and "PENDING" in text and "not realized trading profit" in text
    with pytest.raises(ReplayIsolationError):
        cli_main(["--db", str(archive), "--archive", str(archive), "status"])
