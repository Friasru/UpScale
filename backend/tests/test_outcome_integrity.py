"""Outcome integrity: exact-token candle pricing, deterministic path checks, the append-only
audit sidecar, and Calibration Engine exclusion of confirmed-invalid outcomes only.
Temporary databases only; the provider is faked."""

import asyncio
import hashlib
import math
import random
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from upscale.services.calibration.config import CalibrationConfig
from upscale.services.calibration.dataset import load_live
from upscale.services.geckoterminal import DexCandleService, GeckoTerminalProvider
from upscale.services.market_data import Candle
from upscale.services.outcomes import OUTCOME_LANE
from upscale.services.outcomes.audit import audit, latest_audits, load_rows, main, write_audits
from upscale.services.outcomes.integrity import (
    TOKEN_ORIENTED,
    Verification,
    candle_path_problem,
    classify,
    path_problems,
)
from upscale.services.outcomes.metrics import path_from_candles
from upscale.services.outcomes.models import PricePath
from upscale.services.outcomes.store import HorizonUpdate

from .test_outcomes import (
    CFG,
    MINTS,
    MINUTE,
    NOW,
    Clock,
    FakeCandles,
    FakePools,
    anchor,
    collector,
    dex_pool,
    gt_rows,
    horizon,
    minute_candles,
    run,
)

TOKEN = MINTS["A"]
OTHER = "DoGEV7LASBkQbibMc5k5vKnTZoMg423GpJ5QtJEGfm7R"


def path(ref: float, prices: list[float], token_note: bool = False) -> PricePath:
    candles = [Candle(timestamp=NOW + i * MINUTE, open=p, high=p * 1.01, low=p * 0.99, close=p, volume=1.0)
               for i, p in enumerate(prices)]  # fmt: skip
    p = path_from_candles(candles, MINUTE, ref, NOW, NOW + len(prices) * MINUTE, provider="GeckoTerminal",
                          timeframe="1m", price_drop_pct=90.0)  # fmt: skip
    if token_note:
        p.notes.append(f"{TOKEN_ORIENTED}{TOKEN}")
    return p


# --- deterministic classification -----------------------------------------------------------------


def test_valid_extreme_move_is_preserved() -> None:
    p = path(1.0, [1.0, 5.0, 20.0, 60.0])  # +5,900%: a real path starting at the reference
    assert p.return_pct is not None and p.return_pct > 1000
    assert path_problems(p) == []
    assert classify(p, p.return_pct, TOKEN)[0] == "VALID_EXTREME_MOVE"
    recon = path(1.0, [1.0, 5.0, 20.0, 60.0])
    v = Verification(pool_base=TOKEN, pool_quote=OTHER, reconstructed=recon)
    status, reason, evidence = classify(p, p.return_pct, TOKEN, v)
    assert status == "VALID_EXTREME_MOVE" and evidence["return_difference_pp"] == 0


@pytest.mark.parametrize("ref", [0.0, -1.0, math.nan, math.inf])
def test_invalid_reference_price(ref: float) -> None:
    p = path(1.0, [1.0, 1.1]).model_copy(update={"reference_price": ref})
    assert classify(p, 10.0, TOKEN)[0] == "INVALID_REFERENCE_PRICE"


def test_near_zero_reference_is_never_valid_without_proof() -> None:
    p = path(1e-12, [1.0, 1.02])  # a malformed, near-zero denominator
    status, reason, _ = classify(p, p.return_pct, TOKEN)
    assert status == "UNKNOWN_INTEGRITY" and "never comes within" in reason


def test_invalid_horizon_price() -> None:
    p = path(1.0, [1.0, 1.1]).model_copy(update={"end_price": 0.0})
    assert classify(p, 10.0, TOKEN)[0] == "INVALID_HORIZON_PRICE"
    p2 = path(1.0, [1.0, 1.1]).model_copy(update={"end_price": math.nan})
    assert classify(p2, 10.0, TOKEN)[0] == "INVALID_HORIZON_PRICE"


def test_quote_base_inversion_from_provider_orientation() -> None:
    # The live bug: the pool's other token (priced ~$0.09) recorded as the token's path.
    p = path(3.283e-05, [0.0947, 0.0943, 0.0932])
    v = Verification(
        pool_base=OTHER, pool_quote=TOKEN, reconstructed=path(3.283e-05, [3.2e-05, 2e-05, 1.3e-05])
    )
    status, reason, evidence = classify(p, p.return_pct, TOKEN, v)
    assert status == "QUOTE_BASE_INVERSION" and OTHER in reason
    assert evidence["reconstructed_return_pct"] < 0 < (p.return_pct or 0)
    # A token-priced path from the same pool is not an inversion.
    ok = path(3.283e-05, [3.3e-05, 3.2e-05], token_note=True)
    assert (
        classify(ok, ok.return_pct, TOKEN, Verification(pool_base=OTHER, pool_quote=TOKEN))[0]
        == "VALID"
    )


def test_reciprocal_price_is_an_inversion() -> None:
    p = path(1e-5, [1e5, 1.01e5])  # 1 / 0.00001
    v = Verification(reconstructed=path(1e-5, [1e-5, 0.99e-5]))
    assert classify(p, p.return_pct, TOKEN, v)[0] == "QUOTE_BASE_INVERSION"


def test_decimal_or_unit_error() -> None:
    p = path(0.001, [1000.0, 1001.0])  # 10^6 x the true price
    v = Verification(reconstructed=path(0.001, [0.001, 0.001001]))
    status, reason, _ = classify(p, p.return_pct, TOKEN, v)
    assert status == "DECIMAL_OR_UNIT_ERROR" and "10^6" in reason


def test_pool_identity_mismatch() -> None:
    p = path(1.0, [1.0, 1.2])
    v = Verification(pool_base="SomeOtherToken111111111111111111111111111", pool_quote=OTHER)
    assert classify(p, p.return_pct, TOKEN, v)[0] == "POOL_IDENTITY_MISMATCH"


def test_unavailable_market_used_as_price() -> None:
    empty = PricePath(
        source="candles", points=0, window_start=NOW, window_end=NOW + MINUTE, reference_price=1.0
    )
    assert classify(empty, 5.0, TOKEN)[0] == "MARKET_UNAVAILABLE_AS_PRICE"
    assert classify(None, 5.0, TOKEN)[0] == "MARKET_UNAVAILABLE_AS_PRICE"


def test_mfe_mae_and_path_consistency() -> None:
    p = path(1.0, [1.0, 1.5, 1.2])
    assert path_problems(p) == []
    assert any("MFE" in w for _, w in path_problems(p.model_copy(update={"mfe_pct": 1.0})))
    assert any("MAE" in w for _, w in path_problems(p.model_copy(update={"mae_pct": -50.0})))
    assert any("above MFE" in w for _, w in path_problems(p, return_pct=1e6))
    assert any(
        "range" in w
        for _, w in path_problems(p.model_copy(update={"end_price": 9.0, "return_pct": 800.0}))
    )
    assert (
        classify(p.model_copy(update={"mfe_pct": 1.0}), p.return_pct, TOKEN)[0]
        == "PROVIDER_DATA_ANOMALY"
    )
    rng = random.Random(3)
    for _ in range(200):  # every path the production code builds passes its own checks
        prices = [math.exp(rng.gauss(0, 1)) for _ in range(rng.randint(1, 30))]
        assert path_problems(path(math.exp(rng.gauss(0, 1)), prices)) == []


def test_candle_path_discontinuity_guard() -> None:
    real = [Candle(timestamp=NOW, open=1.2, high=3.0, low=1.1, close=2.9, volume=1.0)]
    assert candle_path_problem(real, 1.0) is None  # a +190% first minute is kept
    other_token = [
        Candle(timestamp=NOW, open=1248.0, high=1255.0, low=1244.0, close=1250.0, volume=1.0)
    ]
    assert "not the same market" in (candle_path_problem(other_token, 0.0001392) or "")
    assert candle_path_problem(real, 0.0) is not None
    assert candle_path_problem([], 1.0) is None


# --- the collector fix ------------------------------------------------------------------------------


def test_outcome_candles_are_requested_for_the_exact_token(fake_geckoterminal: Any) -> None:
    fake_geckoterminal.candles["pool-x"] = gt_rows(NOW, 3)
    dex = DexCandleService(GeckoTerminalProvider(transport=fake_geckoterminal.transport()),
                           now=lambda: NOW + timedelta(hours=1))  # fmt: skip
    run(dex.get_window("solana", "pool-x", "1m", 20, before=NOW + 15 * MINUTE, lane=OUTCOME_LANE,
                       canonical_id=f"solana:{TOKEN}", token=TOKEN))  # fmt: skip
    run(dex.get_window("solana", "pool-x", "1m", 20, before=NOW + 15 * MINUTE, lane=OUTCOME_LANE,
                       canonical_id=f"solana:{TOKEN}"))  # fmt: skip
    tokens = [r.url.params["token"] for r in fake_geckoterminal.requests]
    assert tokens == [TOKEN, "base"]  # separate cache entries; Analyze's default unchanged


def _collect(tmp_path: Path, prices: list[tuple[float, float, float, float]]) -> Any:
    store, obs, scout = anchor(tmp_path)
    candles = FakeCandles()
    candles.by_pool[obs.pool_address] = minute_candles(obs.observed_at, prices)
    pools = FakePools()
    pools.by_token[obs.canonical_id] = [dex_pool("A", obs.pool_address, liquidity=90_000)]
    clock = Clock(
        obs.observed_at + 5 * MINUTE + CFG.collector.settle_seconds * timedelta(seconds=1)
    )
    run(collector(store, clock, scout, candles, pools).collect())
    return store, obs, horizon(store, obs.id or 0, "5m")


def test_collector_withholds_a_number_that_measures_another_price(tmp_path: Path) -> None:
    store, obs, h = _collect(
        tmp_path, [(1248.0, 1255.0, 1244.0, 1250.0)] * 5
    )  # the pool's other token
    assert h.status in ("PARTIAL", "UNAVAILABLE") and h.price is None
    assert any(m.startswith("outcome integrity:") for m in h.missing)
    assert obs.market.price_usd is not None


def test_collector_keeps_a_real_extreme_move_and_records_orientation(tmp_path: Path) -> None:
    store, obs, _ = anchor(tmp_path)
    p0 = obs.market.price_usd or 0.0
    _, _, h = _collect(tmp_path / "x", [(p0, p0 * 5, p0, p0 * 4), (p0 * 4, p0 * 30, p0 * 4, p0 * 25)] + [
        (p0 * 25, p0 * 26, p0 * 24, p0 * 25)] * 3)  # fmt: skip
    assert (
        h.status == "COMPLETE"
        and h.price is not None
        and h.price.return_pct == pytest.approx(2400.0)
    )
    assert any(n.startswith(TOKEN_ORIENTED) for n in h.price.notes)
    assert classify(h.price, h.price.return_pct, obs.address)[0] == "VALID_EXTREME_MOVE"


# --- audit sidecar and calibration ---------------------------------------------------------------------


def _live_db(tmp_path: Path) -> tuple[Path, list[Any]]:
    """Real outcome rows: one bad (another token priced), one valid extreme, others normal."""
    from upscale.services.outcomes import record_scout_run

    from .test_outcomes import outcome_store, ranked  # noqa: PLC0415

    result, _ = ranked(tmp_path)
    store = outcome_store(tmp_path)
    obs = run(record_scout_run(store, result, CFG))
    for i, o in enumerate(obs):
        ref = o.market.price_usd or 1.0
        prices = [[ref * 1e7] * 3, [ref, ref * 20, ref * 60]][i] if i < 2 else [ref, ref * 1.01]
        p = path(ref, prices).model_copy(update={"window_start": o.observed_at,
                                                 "window_end": o.observed_at + timedelta(hours=1)})  # fmt: skip
        run(store.update_horizon("scout", o.id or 0, "1h", HorizonUpdate(price=p, finalize="COMPLETE"),
                                 o.observed_at + timedelta(hours=2)))  # fmt: skip
    return Path(store.path), obs


def _outcome_tables(db: Path) -> list[Any]:
    c = sqlite3.connect(db)
    try:
        return [c.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
                for t in ("scout_outcome_observations", "scout_outcome_horizons")]  # fmt: skip
    finally:
        c.close()


def test_audit_is_read_only_by_default_and_append_only_when_written(tmp_path: Path) -> None:
    db, obs = _live_db(tmp_path)
    before_file = hashlib.sha256(db.read_bytes()).hexdigest()
    assert main(["--db", str(db), "--threshold", "1000"]) == 0
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before_file  # READ_ONLY
    tables = _outcome_tables(db)
    assert main(["--db", str(db), "--threshold", "1000", "--write"]) == 0
    assert _outcome_tables(db) == tables  # outcome rows are never modified
    c = sqlite3.connect(db)
    rows = c.execute(
        "SELECT observation_id, integrity_status FROM outcome_integrity_audits"
    ).fetchall()
    assert {s for _, s in rows} == {
        "UNKNOWN_INTEGRITY",
        "VALID_EXTREME_MOVE",
    }  # no provider proof yet
    for sql in ("UPDATE outcome_integrity_audits SET integrity_status = 'VALID'",
                "DELETE FROM outcome_integrity_audits"):  # fmt: skip
        with pytest.raises(sqlite3.DatabaseError):
            with c:
                c.execute(sql)
    for sql in (
        "UPDATE scout_outcome_observations SET score = 0",
        "DELETE FROM scout_outcome_observations",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            with c:
                c.execute(sql)


def test_calibration_excludes_only_confirmed_invalid_outcomes(tmp_path: Path) -> None:
    db, obs = _live_db(tmp_path)
    cfg = CalibrationConfig()
    bad, extreme = obs[0].id, obs[1].id

    def returns() -> dict[int, float | None]:
        out = {}
        for o in load_live(db, cfg, include_holdout=True):
            v = o.outcomes.get("1h")
            out[int(o.key.rsplit(":", 1)[1])] = v.return_pct if v else None
        return out

    before = returns()
    assert before[bad] is not None and before[bad] > 1e8 and before[extreme] > 1000
    rows = load_rows(db, 1000.0)
    audit(rows)  # offline: unaudited / unproven rows are not invalid
    write_audits(db, rows)
    assert returns() == before  # UNKNOWN_INTEGRITY and VALID_EXTREME_MOVE are both kept
    proven = load_rows(db, 1000.0, observation=bad)
    audit(proven, {("scout", bad, "1h"): Verification(pool_base=OTHER, pool_quote=obs[0].address)})  # type: ignore[dict-item]
    assert proven[0].integrity == "QUOTE_BASE_INVERSION"
    write_audits(db, proven)
    after = load_live(db, cfg, include_holdout=True)
    excluded = {o.key: o.integrity_excluded for o in after if o.integrity_excluded}
    assert excluded == {f"live:scout:{bad}": {"1h": "QUOTE_BASE_INVERSION"}}
    got = returns()
    assert got[bad] is None and got[extreme] == before[extreme]  # the valid extreme move stays
    assert len(after) == len(obs)  # raw observations are kept and counted
    with sqlite3.connect(db) as c:
        assert latest_audits(c)[("scout", bad, "1h")][0] == "QUOTE_BASE_INVERSION"


def test_verifier_uses_token_priced_candles_and_pool_orientation() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def pool_tokens(self, chain: str, pool: str) -> tuple[str, str]:
            self.calls.append(("pool", pool))
            return OTHER, TOKEN

        async def fetch_pool_candles(
            self, chain: str, pool: str, timeframe: str, limit: int, **kw: Any
        ) -> Any:
            self.calls.append(("candles", pool, kw.get("token")))
            from upscale.services.market_data import CandleSeries

            c = [Candle(timestamp=NOW + i * MINUTE, open=1e-5, high=1e-5, low=1e-5, close=1e-5, volume=1.0)
                 for i in range(3)]  # fmt: skip
            return CandleSeries(symbol=TOKEN, provider="GeckoTerminal", provider_id=pool, timeframe="1m",
                                candles=c, volume_available=True, fetched_at=NOW)  # fmt: skip

    from upscale.services.outcomes.audit import AuditRow, Verifier

    provider = Provider()
    bad = path(1e-5, [0.09, 0.091, 0.092])
    row = AuditRow(kind="scout", observation_id=1, horizon="5m", horizon_minutes=5, status="COMPLETE",
                   return_pct=bad.return_pct or 0.0, mfe_pct=bad.mfe_pct, mae_pct=bad.mae_pct,
                   market_status="ACTIVE", attempts=1, missing=[], path=bad, body={},
                   identity={"chain": "solana", "pool": "pool-g", "token": TOKEN, "symbol": "GOAT"})  # fmt: skip
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    verifier = Verifier(provider=provider, sleep=sleep)  # type: ignore[arg-type]
    verifications = asyncio.run(verifier.verify([row], reconstruct_above=1000.0))
    assert provider.calls == [("pool", "pool-g"), ("candles", "pool-g", TOKEN)] and sleeps == [15.0]
    audit([row], verifications)
    assert row.integrity == "QUOTE_BASE_INVERSION"
    assert row.evidence["reconstructed_return_pct"] == pytest.approx(0.0)


def test_replay_outcome_candles_are_token_priced_and_cached_apart(tmp_path: Path) -> None:
    from upscale.services.quota import LaneLimiter
    from upscale.services.replay_lab.candles import TOKEN_PRICED_KEY, HistoricalCandleFetcher
    from upscale.services.replay_lab.quota import ReplayGate
    from upscale.services.replay_lab.store import ReplayStore

    from .test_replay_lab import T0, FakeGeckoTerminal

    calls: list[Any] = []

    class Recording(FakeGeckoTerminal):
        async def fetch_pool_candles(self, *args: Any, token: str | None = None, **kw: Any) -> Any:
            calls.append(token)
            return await super().fetch_pool_candles(*args, token=token, **kw)

    store = ReplayStore(tmp_path / "r.sqlite3")
    fetcher = HistoricalCandleFetcher(store, ReplayGate(LaneLimiter(1000, 60.0)), provider=Recording(),  # type: ignore[arg-type]
                                      wall_clock=lambda: (T0 + timedelta(days=30)).timestamp())  # fmt: skip
    run(
        fetcher.range(
            "solana", TOKEN, "pool-r", "1m", T0, T0 + timedelta(hours=1), token_priced=True
        )
    )
    run(fetcher.range("solana", TOKEN, "pool-r", "1m", T0, T0 + timedelta(hours=1)))
    assert calls == [TOKEN, None]  # never reuses the other orientation's cache
    keys = {
        r[0]
        for r in sqlite3.connect(store.path).execute("SELECT DISTINCT provider FROM candle_cache")
    }
    assert keys == {TOKEN_PRICED_KEY, "GeckoTerminal"}
