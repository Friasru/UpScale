"""Growth Scout: stage classification, ScoutMomentumScore, risk penalties, missing-social
handling, safety statuses, ranking, and fair X budget scheduling.

All offline and deterministic: market evidence is synthesized from piecewise-constant
trading rates (so every rolling window is consistent with the others), growth features
come from Scout's real `window_acceleration` / `compare_with`, and social evidence is
either a SocialMomentum built from the real models or produced by the real social
pipeline over fixture posts.
"""

import asyncio
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from upscale.services.market_data import MarketDataUnavailableError
from upscale.services.scout.config import ScoutFeatureConfig, ScoutProviderLimits
from upscale.services.scout.features import compare_with, window_acceleration
from upscale.services.scout.growth import (
    GrowthConfig,
    GrowthConfigError,
    GrowthScoutResult,
    GrowthScoutService,
    load_growth_config,
)
from upscale.services.scout.growth.models import GrowthCandidate
from upscale.services.scout.growth.signals import (
    clamp,
    log_band_score,
    market_evidence,
    pct_score,
    ratio_score,
    social_evidence,
)
from upscale.services.scout.models import (
    WINDOW_MINUTES,
    ScoutCandidate,
    ScoutGrowthFeatures,
    ScoutMarketMetrics,
    ScoutPool,
    ScoutRiskFlags,
    ScoutRun,
    ScoutSnapshot,
    ScoutWindow,
)
from upscale.services.scout.social import (
    SocialConfig,
    SocialScoutService,
    SocialStore,
    StaticSocialProvider,
    TokenIdentity,
    XRecentSearchProvider,
)
from upscale.services.scout.social.config import SocialProviderConfig
from upscale.services.scout.social.models import (
    CrossPlatformConfirmation,
    MarketCrossCheck,
    MetricTrend,
    SocialMomentum,
    SocialQuality,
    SocialRun,
    SocialWindowStats,
    WindowTrend,
)
from upscale.services.scout.social.service import fair_order
from upscale.services.scout.store import ScoutSnapshotStore
from upscale.services.solana_chain import OnchainSafetySnapshot

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
SOL = "So11111111111111111111111111111111111111112"
PROVIDER = "GeckoTerminal"
FEATURES = ScoutFeatureConfig()

MINTS = {
    "A": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
    "B": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "C": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "D": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "E": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "F": "HhJpBhRRn4g56VsyLuT8DL5Bv31HkXqsrahTTUCZeZg4",
    "G": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "H": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
}
SEGMENTS = list[
    tuple[int, float, float, float]
]  # (until minutes ago, $/min, trades/min, buy share)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- Synthetic market -----------------------------------------------------------------------


def windows(segments: SEGMENTS, price: dict[str, float] | None = None) -> list[ScoutWindow]:
    """Nested rolling windows from piecewise-constant rates (minute 0 = now)."""
    out = []
    for name, minutes in WINDOW_MINUTES.items():
        vol = txns = buys = 0.0
        start = 0
        for end, usd, trades, share in segments:
            span = max(0, min(end, minutes) - start)
            vol += usd * span
            txns += trades * span
            buys += trades * span * share
            start = end
            if end >= minutes:
                break
        b = round(buys)
        s = round(txns) - b
        out.append(
            ScoutWindow(
                window=name,  # type: ignore[arg-type]
                volume_usd=vol,
                buys=b,
                sells=s,
                buyers=round(b * 0.6),
                sellers=round(s * 0.6),
                price_change_pct=(price or {}).get(name, 0.0),
            )
        )
    return out


def metrics(
    segments: SEGMENTS,
    *,
    price: float = 0.001,
    liquidity: float = 80_000,
    mcap: float | None = 400_000,
    fdv: float | None = None,
    changes: dict[str, float] | None = None,
) -> ScoutMarketMetrics:
    return ScoutMarketMetrics(
        price_usd=price,
        market_cap_usd=mcap,
        fdv_usd=fdv if fdv is not None else mcap,
        liquidity_usd=liquidity,
        windows=windows(segments, changes),
    )


class Past:
    """An earlier stored observation of the same pool."""

    def __init__(
        self, minutes_ago: int, segments: SEGMENTS, price: float, liquidity: float, **kw: Any
    ):
        self.minutes_ago = minutes_ago
        self.metrics = metrics(segments, price=price, liquidity=liquidity, **kw)


def candidate(
    key: str,
    segments: SEGMENTS,
    *,
    symbol: str | None = None,
    age_hours: float = 30.0,
    past: list[Past] | None = None,
    flags: ScoutRiskFlags | None = None,
    first_seen_hours: float = 2.0,
    **metric_kw: Any,
) -> tuple[ScoutCandidate, list[ScoutSnapshot]]:
    mint = MINTS.get(key, key)
    cid = f"solana:{mint}"
    pool = ScoutPool(
        address=f"pool-{key}",
        dex="raydium",
        quote_address=SOL,
        quote_symbol="SOL",
        quote_kind="SOL",
        created_at=NOW - timedelta(hours=age_hours),
        age_hours=age_hours,
    )
    c = ScoutCandidate(
        canonical_id=cid,
        chain="solana",
        address=mint,
        symbol=symbol or key,
        name=f"Token {key}",
        observed_at=NOW,
        first_seen_at=NOW - timedelta(hours=first_seen_hours),
        oldest_pool_created_at=NOW - timedelta(hours=age_hours),
        market_provider=PROVIDER,
        pool=pool,
        metrics=metrics(segments, **metric_kw),
        pool_count=1,
        risk_flags=flags or ScoutRiskFlags(),
    )
    snapshots = [
        ScoutSnapshot(
            canonical_id=cid,
            observed_at=NOW - timedelta(minutes=p.minutes_ago),
            provider=PROVIDER,
            pool_address=pool.address,
            dex=pool.dex,
            metrics=p.metrics,
        )
        for p in past or []
    ]
    history = [
        compare_with(c, s, p.minutes_ago, FEATURES)
        for p, s in zip(past or [], snapshots, strict=True)
        if p.minutes_ago in FEATURES.lookback_minutes
    ]
    c.features = ScoutGrowthFeatures(
        computed_at=NOW,
        window_acceleration=window_acceleration(c.metrics, FEATURES),
        history=history,
        missing_lookbacks=[
            m for m in FEATURES.lookback_minutes if m not in {h.lookback_minutes for h in history}
        ],
    )
    return c, snapshots


# Scenario markets --------------------------------------------------------------------------

ACCEL: SEGMENTS = [
    (15, 900, 6, 0.72),
    (60, 450, 3, 0.60),
    (360, 220, 1.5, 0.56),
    (1440, 150, 1, 0.52),
]
ACCEL_CHANGES = {"m5": 1, "m15": 3, "m30": 5, "h1": 9, "h6": 18, "h24": 35}
FLAT: SEGMENTS = [(1440, 150, 1, 0.6)]


def accelerating(key: str, **kw: Any) -> tuple[ScoutCandidate, list[ScoutSnapshot]]:
    """Real early acceleration: volume, trades and buyers rising on every time scale,
    liquidity growing, price starting to move (+35% in 24h)."""
    past = [
        Past(60, [(60, 300, 2, 0.55), (360, 200, 1.4, 0.54), (1440, 150, 1, 0.52)], 0.00092, 62_000),
        Past(45, [(60, 340, 2.2, 0.56), (360, 200, 1.4, 0.54), (1440, 150, 1, 0.52)], 0.00094, 66_000),
        Past(30, [(60, 380, 2.5, 0.58), (360, 210, 1.4, 0.55), (1440, 150, 1, 0.52)], 0.00096, 70_000),
        Past(15, [(60, 420, 2.8, 0.60), (360, 215, 1.5, 0.55), (1440, 150, 1, 0.52)], 0.00098, 75_000),
    ]  # fmt: skip
    return candidate(key, ACCEL, past=past, changes=ACCEL_CHANGES, **kw)


def flat(key: str, **kw: Any) -> tuple[ScoutCandidate, list[ScoutSnapshot]]:
    past = [Past(60, FLAT, 0.001, 80_000), Past(30, FLAT, 0.001, 80_000)]
    return candidate(key, FLAT, past=past, **kw)


def thin_pump(key: str) -> tuple[ScoutCandidate, list[ScoutSnapshot]]:
    segments = [(60, 200, 0.5, 0.7), (600, 50, 0.08, 0.6), (1440, 0, 0, 0.5)]
    return candidate(
        key, segments, age_hours=10, liquidity=8_000, mcap=60_000,
        changes={"m5": 10, "m15": 30, "h1": 120, "h6": 300, "h24": 400},
    )  # fmt: skip


def crowded(key: str) -> tuple[ScoutCandidate, list[ScoutSnapshot]]:
    segments = [(60, 1000, 5, 0.5), (360, 1000, 5, 0.52), (1440, 400, 2, 0.55)]
    return candidate(
        key, segments, age_hours=20, liquidity=300_000, mcap=4_000_000,
        changes={"m5": 0, "m15": 1, "h1": 2, "h6": 40, "h24": 700},
    )  # fmt: skip


def fading(key: str) -> tuple[ScoutCandidate, list[ScoutSnapshot]]:
    segments = [(15, 50, 0.8, 0.35), (60, 100, 1.6, 0.45), (360, 200, 3, 0.5), (1440, 200, 3, 0.5)]
    return candidate(
        key, segments, age_hours=48, changes={"m5": -2, "m15": -4, "h1": -8, "h6": -12, "h24": -5}
    )


# --- Synthetic social -----------------------------------------------------------------------


def trend(prev: float, recent: float) -> MetricTrend:
    return MetricTrend(
        recent=recent,
        previous=prev,
        velocity_per_hour=recent,
        previous_velocity_per_hour=prev,
        acceleration_per_hour=recent - prev,
        acceleration_ratio=(recent + 1) / (prev + 1),
    )


def momentum(
    key: str,
    state: str,
    *,
    mentions: tuple[float, float] = (1, 8),
    authors: tuple[float, float] = (1, 7),
    exact: int = 6,
    attributable: int = 8,
    weighted: float = 7.0,
    spam: str = "low",
    organic: str = "medium",
    corroborated: bool = False,
    computed_at: datetime = NOW,
) -> SocialMomentum:
    cid = f"solana:{MINTS.get(key, key)}"
    return SocialMomentum(
        canonical_id=cid,
        computed_at=computed_at,
        state=state,  # type: ignore[arg-type]
        reasons=[f"{state.lower()} (synthetic)"],
        window="h1",
        trend=WindowTrend(
            window="h1",
            mentions=trend(*mentions),
            unique_authors=trend(*authors),
            engagement=trend(mentions[0] * 3, mentions[1] * 3),
        ),
        windows=[
            SocialWindowStats(
                window="h1",
                mentions=weighted,
                unique_authors=int(authors[1]),
                engagement=10,
                exact_mentions=exact,
                attributable_posts=attributable,
                promoted_posts=0,
                ambiguous_posts=0,
                sources=1,
            )
        ],  # fmt: skip
        quality=SocialQuality(spam_risk=spam, organic_signal_strength=organic),  # type: ignore[arg-type]
        cross_platform=CrossPlatformConfirmation(
            providers_configured=2,
            providers_checked=2,
            platforms_with_activity=2 if corroborated else 1,
            platforms_accelerating=2 if corroborated else 1,
            source_diversity=1.0,
            corroborated=corroborated,
            single_source_available=False,
            only_one_active_of_several=not corroborated,
        ),  # fmt: skip
        market=MarketCrossCheck(state="INSUFFICIENT_DATA"),
    )


def safety_snapshot(
    key: str, *, top1: float = 5.0, top10: float = 25.0, **kw: Any
) -> OnchainSafetySnapshot:
    mint = MINTS[key]
    fields: dict[str, Any] = dict(
        canonical_id=f"solana:{mint}", mint=mint, provider="fixture", fetched_at=NOW,
        authorities_available=True, token_program="spl_token", token_program_id="x",
        decimals=6, supply=1e9, mint_authority=None, freeze_authority=None, holders=[],
        excluded=[], top1_pct=top1, top10_pct=top10, excluded_pct=10.0, holder_count=900,
        meaningful_holder_count=700, holder_count_complete=True,
        concentration_source="full_scan", concentration_lower_bound=False,
        token_accounts_seen=1000, scan_pages_read=1, scan_max_pages=5, largest_accounts_seen=20,
        pool_addresses_used=[], concentration_reliable=True, holder_data_complete=True,
    )  # fmt: skip
    fields.update(kw)
    return OnchainSafetySnapshot(**fields)


# --- Ranking harness ------------------------------------------------------------------------


def service(tmp_path: Path, config: GrowthConfig | None = None, **kw: Any) -> GrowthScoutService:
    return GrowthScoutService(
        ScoutSnapshotStore(tmp_path / "scout.sqlite3"), config, now=lambda: NOW, **kw
    )


def rank(
    tmp_path: Path,
    scenarios: list[tuple[ScoutCandidate, list[ScoutSnapshot]]],
    social: list[SocialMomentum] = (),  # type: ignore[assignment]
    safety: list[OnchainSafetySnapshot] = (),  # type: ignore[assignment]
    config: GrowthConfig | None = None,
    limit: int | None = 50,
    **kw: Any,
) -> GrowthScoutResult:
    svc = service(tmp_path, config, **kw)

    async def go() -> GrowthScoutResult:
        for c, snaps in scenarios:
            await svc.store.record_seen(c)
            for s in snaps:
                await svc.store.save_snapshot(s, 0)
        return await svc.rank(
            [c for c, _ in scenarios],
            {m.canonical_id: m for m in social},
            {s.canonical_id: s for s in safety},
            limit=limit,
        )

    return run(go())


def by_symbol(result: GrowthScoutResult) -> dict[str, GrowthCandidate]:
    return {g.symbol or "": g for g in [*result.candidates, *result.unranked]}


def family(g: GrowthCandidate, name: str) -> Any:
    return next(f for f in g.scout_momentum.families if f.family == name)


def flag_codes(g: GrowthCandidate) -> set[str]:
    return {f.code for f in g.risk_flags}


# --- Normalization --------------------------------------------------------------------------


def test_normalization_is_bounded_and_log_scaled() -> None:
    assert ratio_score(1.0, 3) == 0.5
    assert ratio_score(3.0, 3) == ratio_score(300.0, 3) == 1.0  # capped: 300x can't dominate
    assert ratio_score(1 / 3, 3) == ratio_score(0.001, 3) == 0.0
    assert ratio_score(None, 3) is None and ratio_score(0, 3) is None
    assert pct_score(0, 30) == 0.5 and pct_score(30, 30) == 1.0 and pct_score(-100.5, 30) is None
    assert pct_score(-100, 30) == pct_score(-99, 30) == 0.0  # a total loss is the worst score
    assert pct_score(1000, 30) == 1.0


def test_a_collapse_never_scores_better_than_a_smaller_drop(tmp_path: Path) -> None:
    """Live regression: DEX Screener reports a pool that collapsed within the hour as
    exactly -100%. That used to drop the price signal (as if unmeasured), so a rug ranked
    above the same pool down 99%, and its reasons called it "still early"."""
    dump: SEGMENTS = [(60, 2000, 12, 0.9), (1440, 200, 1.5, 0.9)]

    def at(key: str, pct: float) -> tuple[ScoutCandidate, list[ScoutSnapshot]]:
        return candidate(key, dump, changes={"m5": 0, "h1": pct, "h6": pct, "h24": pct})

    ranked = rank(tmp_path, [at("A", -100.0), at("B", -99.0)]).candidates
    by = {g.symbol: g for g in ranked}
    assert by["A"].score == by["B"].score
    prices = [
        next(s for f in g.scout_momentum.families for s in f.signals if s.name == "price_momentum")
        for g in ranked
    ]
    assert [s.score for s in prices] == [0.0, 0.0]
    assert not any("still early" in r for g in ranked for r in g.reasons_surfaced)
    assert log_band_score(1_000, 5_000, 250_000) == 0.0
    assert log_band_score(250_000, 5_000, 250_000) == 1.0
    assert log_band_score(100_000_000, 5_000, 250_000) == 1.0
    assert clamp(-4) == 0.0 and clamp(9) == 1.0


def test_size_alone_never_decides(tmp_path: Path) -> None:
    """$100M-a-day volume doesn't beat $1M-a-day volume with the same acceleration."""
    big_segments = [(end, usd * 100, t, s) for end, usd, t, s in ACCEL]
    small = accelerating("A")
    big = candidate("B", big_segments, past=[], changes=ACCEL_CHANGES)
    small_no_history = candidate("A", ACCEL, past=[], changes=ACCEL_CHANGES)
    result = rank(tmp_path, [small_no_history, big])
    s = by_symbol(result)
    assert family(s["A"], "market_activity").score == family(s["B"], "market_activity").score
    assert (
        small[0].metrics.window("h24").volume_usd * 100 == big[0].metrics.window("h24").volume_usd
    )  # type: ignore[union-attr]


# --- Scenarios A-K --------------------------------------------------------------------------


def all_scenarios() -> list[tuple[ScoutCandidate, list[ScoutSnapshot]]]:
    return [
        accelerating("A"),  # A: true early acceleration (+ social emerging)
        accelerating("B"),  # B: market only
        flat("C"),  # C: social-only hype
        thin_pump("D"),  # D: low-liquidity pump
        crowded("E"),  # E: crowded
        fading("F"),  # F: fading
        accelerating("G"),  # G: spam
        accelerating("H"),  # H: holder concentration
    ]


def all_social() -> list[SocialMomentum]:
    return [
        momentum("A", "EMERGING"),
        momentum("C", "STRONG", mentions=(20, 80), authors=(10, 40), corroborated=True),
        momentum("E", "SATURATED", mentions=(300, 290), authors=(120, 118)),
        momentum(
            "G",
            "ACCELERATING",
            mentions=(10, 60),
            authors=(2, 3),
            spam="high",
            organic="low",
            exact=40,
            attributable=60,
            weighted=55,
        ),  # fmt: skip
        momentum("H", "EMERGING"),
    ]


@pytest.fixture
def ranked(tmp_path: Path) -> GrowthScoutResult:
    return rank(
        tmp_path,
        all_scenarios(),
        all_social(),
        [safety_snapshot("H", top1=28.0, top10=72.0), safety_snapshot("A")],
    )


def test_a_true_early_acceleration_ranks_first(ranked: GrowthScoutResult) -> None:
    a = by_symbol(ranked)["A"]
    assert a.rank == 1
    assert a.stage == "ACCELERATING"
    assert a.score >= 75
    assert a.momentum.social_status == "SOCIAL_EMERGING"
    assert "SAFETY_CHECKS_COMPLETE" in a.quality.verification
    assert {"VERIFIED_IDENTITY", "MARKET_CONFIRMED"} <= set(a.quality.verification)
    text = " | ".join(a.reasons_surfaced)
    for expected in ("volume accelerating", "trade count accelerating",
                     "buyer activity increasing", "liquidity growing", "social emerging"):  # fmt: skip
        assert expected in text
    assert set(a.momentum.timescales_rising) == {"short", "medium", "history"}
    assert a.momentum.technical is not None and a.momentum.technical.trend == "up"


def test_b_market_only_signal_stays_competitive(ranked: GrowthScoutResult) -> None:
    s = by_symbol(ranked)
    a, b = s["A"], s["B"]
    assert b.eligible and b.stage == "ACCELERATING"
    assert b.momentum.social_status == "SOCIAL_UNAVAILABLE"
    assert b.rank is not None and b.rank <= 3
    # Social is support, not a requirement: it is worth under 10 points here (the rest of
    # the gap is A's completed safety checks).
    assert a.scout_momentum.base - b.scout_momentum.base < 10
    for other in ("C", "D", "E", "F", "G"):
        assert b.score > s[other].score, other


def test_c_social_only_hype_is_penalized(ranked: GrowthScoutResult) -> None:
    s = by_symbol(ranked)
    c = s["C"]
    assert "social_only_hype" in flag_codes(c)
    assert c.momentum.social_status == "SOCIAL_STRONG"
    assert family(c, "social_momentum").score > 0.8  # attention really is strong ...
    assert c.score < s["B"].score - 20  # ... but it can't carry the candidate
    assert c.rank is not None and c.rank > 3
    assert c.stage == "STEADY"
    assert not any("social" in r for r in c.reasons_surfaced)
    assert family(c, "cross_confirmation").score < 0.25


def test_d_thin_market_pump_is_heavily_penalized(ranked: GrowthScoutResult) -> None:
    d = by_symbol(ranked)["D"]
    [pump] = [f for f in d.risk_flags if f.code == "thin_market_pump"]
    assert pump.severity == "high" and pump.penalty == 30
    assert d.scout_momentum.risk_penalty >= 30
    assert d.score < 35
    assert d.quality.liquidity_quality == "thin"
    assert d.stage == "CROWDED"  # +400% on a thin market is not "early"


def test_e_crowded_move(ranked: GrowthScoutResult) -> None:
    e = by_symbol(ranked)["E"]
    assert e.stage == "CROWDED"
    assert any("already up 700%" in r for r in e.stage_reasons)
    assert any("SATURATED" in r for r in e.stage_reasons)
    assert family(e, "earliness").score < 0.5
    assert e.score < by_symbol(ranked)["B"].score - 20


def test_f_fading(ranked: GrowthScoutResult) -> None:
    f = by_symbol(ranked)["F"]
    assert f.stage == "FADING"
    assert {"volume declining", "buy pressure weakening", "price momentum weakening"} <= set(
        f.stage_reasons
    )
    assert family(f, "market_activity").score < 0.35


def test_g_spam_social_is_penalized(ranked: GrowthScoutResult) -> None:
    s = by_symbol(ranked)
    g = s["G"]
    assert family(g, "social_momentum").score <= 0.2  # capped, below neutral
    assert "social_spam" in flag_codes(g)
    assert g.score < s["B"].score  # spammy attention is worse than none
    assert not any("social" in r for r in g.reasons_surfaced)


def test_h_holder_concentration_is_a_risk_penalty(ranked: GrowthScoutResult) -> None:
    s = by_symbol(ranked)
    h, a = s["H"], s["A"]
    assert {"holder_concentration_top1", "holder_concentration_top10"} <= flag_codes(h)
    assert h.quality.holder_top10_pct == 72.0
    assert h.score <= a.score - 20
    assert h.rank is not None and a.rank is not None and h.rank > a.rank


def test_i_missing_social_is_not_zero(tmp_path: Path) -> None:
    quiet = momentum(
        "B", "QUIET", mentions=(0, 0), authors=(0, 0), exact=0, attributable=0, weighted=0
    )
    unavailable = rank(tmp_path / "u", [accelerating("B")])
    checked = rank(tmp_path / "q", [accelerating("B")], [quiet])
    [u], [q] = unavailable.candidates, checked.candidates
    fam_u, fam_q = family(u, "social_momentum"), family(q, "social_momentum")
    assert u.momentum.social_status == "SOCIAL_UNAVAILABLE"
    assert q.momentum.social_status == "SOCIAL_QUIET"
    assert not fam_u.available and fam_u.score == 0.4  # neutral stand-in, never 0
    assert fam_q.available and fam_q.score == 0.4
    assert u.score == q.score  # unavailable is treated like quiet, not as zero activity
    [info] = [f for f in u.risk_flags if f.code == "social_unavailable"]
    assert info.severity == "info" and info.penalty == 0
    # Stale momentum is unavailable, not evidence.
    stale = momentum("B", "STRONG", computed_at=NOW - timedelta(hours=5))
    [old] = rank(tmp_path / "s", [accelerating("B")], [stale]).candidates
    assert old.momentum.social_status == "SOCIAL_UNAVAILABLE"
    assert "minutes old" in (old.momentum.social_reason or "")


def test_j_duplicate_tickers_stay_separate(tmp_path: Path) -> None:
    one = accelerating("A", symbol="NEWT")
    two = fading("B")
    two[0].symbol = "NEWT"
    result = rank(tmp_path, [one, two, accelerating("A", symbol="NEWT")])  # A observed twice
    ids = [g.canonical_id for g in [*result.candidates, *result.unranked]]
    assert ids.count(f"solana:{MINTS['A']}") == 1 and ids.count(f"solana:{MINTS['B']}") == 1
    assert result.evaluated == 2
    assert any("duplicate observation" in n for n in result.notes)
    newts = [g for g in result.candidates if g.symbol == "NEWT"]
    assert len(newts) == 2 and newts[0].stage != newts[1].stage


# --- K: X budget fairness -------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.at = NOW

    def __call__(self) -> datetime:
        return self.at


def x_tokens(n: int) -> list[TokenIdentity]:
    return [
        TokenIdentity(canonical_id=f"solana:{m}", chain="solana", address=m, symbol=None)
        for m in list(MINTS.values())[:n]
    ]


def x_service(
    tmp_path: Path, clock: Clock, reads: int = 25
) -> tuple[SocialScoutService, list[str]]:
    searched: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        # A full page of posts that attribute to nobody: each search really costs 10 reads.
        searched.append(request.url.params["query"])
        rows = [
            {"id": f"{len(searched)}-{i}", "author_id": str(i), "text": "gm",
             "created_at": (clock() - timedelta(minutes=i + 1)).isoformat()}
            for i in range(int(request.url.params["max_results"]))
        ]  # fmt: skip
        return httpx2.Response(200, json={"data": rows, "meta": {"result_count": len(rows)}})

    provider = XRecentSearchProvider(
        "token", True, reads,
        SocialProviderConfig(limits=ScoutProviderLimits(calls_per_minute=100),
                             max_results_per_page=10, max_terms_per_query=1),
        transport=httpx2.MockTransport(handle), now=clock,
    )  # fmt: skip
    return SocialScoutService([provider], SocialStore(tmp_path / "s.sqlite3"), now=clock), searched


def test_k_fair_order_prefers_never_sampled_then_stalest_with_rotation() -> None:
    t = x_tokens(4)
    last = {
        t[0].canonical_id: NOW - timedelta(minutes=5),
        t[1].canonical_id: NOW - timedelta(hours=1),
        t[2].canonical_id: None,
        t[3].canonical_id: None,
    }
    order = [x.canonical_id for x in fair_order(t, last, 0, now=NOW)]
    assert set(order[:2]) == {t[2].canonical_id, t[3].canonical_id}  # never sampled first
    assert order[2:] == [t[1].canonical_id, t[0].canonical_id]  # then stalest
    # List order never matters; ties rotate between runs.
    assert fair_order(list(reversed(t)), last, 0, now=NOW) == fair_order(t, last, 0, now=NOW)
    ties = {x.canonical_id: None for x in t}
    firsts = {fair_order(t, ties, r, now=NOW)[0].canonical_id for r in range(4)}
    assert len(firsts) == 4
    # The never-searched bonus is bounded: a refresh overdue by hours goes first.
    overdue = last | {t[1].canonical_id: NOW - timedelta(hours=3)}
    assert fair_order(t, overdue, 0, now=NOW)[0].canonical_id == t[1].canonical_id


def test_k_x_budget_is_shared_fairly_without_starvation(tmp_path: Path) -> None:
    clock = Clock()
    svc, searched = x_service(tmp_path, clock, reads=25)  # 2 searches of 10 per run
    tokens = x_tokens(6)
    seen: Counter[str] = Counter()
    for i in range(3):
        result = run(svc.observe(tokens if i % 2 == 0 else list(reversed(tokens))))
        statuses = {m.canonical_id: m.sources[0].status for m in result.momentum}
        assert Counter(statuses.values()) == {
            "PROVIDER_CHECKED_ZERO_MATCHES": 2,
            "PROVIDER_UNAVAILABLE": 4,  # deferred: unavailable, never a zero
        }
        for m in result.momentum:
            if statuses[m.canonical_id] == "PROVIDER_UNAVAILABLE":
                assert "result budget" in (m.sources[0].error or "")
                assert m.state == "UNAVAILABLE"
            else:
                seen[m.canonical_id] += 1
        assert result.providers[0].results == 20  # hard cap (25) never exceeded
        clock.at += timedelta(minutes=15)
    assert len(seen) == 6 and set(seen.values()) == {1}  # every token once, nobody starved
    assert len(searched) == 6
    # Fourth run: everybody sampled once; the stalest (run 1's) go first again.
    run(svc.observe(tokens))
    assert searched[6:] == searched[:2]


def test_k_deferred_token_is_not_starved_by_new_discoveries(tmp_path: Path) -> None:
    """Live regression: discovery adds dozens of never-searched tokens every run. They tied
    with tokens deferred earlier and won by canonical id, so a deferred `solana:` token lost
    to every new `base:` token, run after run. Waiting longest now goes first."""
    clock = Clock()
    svc, _ = x_service(tmp_path, clock, reads=25)  # 2 searches of 10 per run

    def ident(address: str) -> TokenIdentity:
        return TokenIdentity(canonical_id=f"x:{address}", chain="solana", address=address)

    late = ident("zz-sorts-last")

    def statuses(tracked: list[TokenIdentity]) -> dict[str, str]:
        result = run(svc.observe(tracked))
        clock.at += timedelta(minutes=10)
        return {m.canonical_id: m.sources[0].status for m in result.momentum}

    first = statuses([ident("a1"), ident("a2"), late])
    assert first[late.canonical_id] == "PROVIDER_UNAVAILABLE"  # deferred, never a zero
    second = statuses([late, *(ident(f"b{i}") for i in range(6))])
    assert second[late.canonical_id] == "PROVIDER_CHECKED_ZERO_MATCHES"
    order = fair_order(
        [late, ident("c1")], {late.canonical_id: None, "x:c1": None}, 0,
        {late.canonical_id: NOW - timedelta(minutes=10)}, now=NOW,
    )  # fmt: skip
    assert order[0] == late  # waiting beats brand new, whatever the id


def test_k_deferral_does_not_erase_a_recent_measurement(tmp_path: Path) -> None:
    """Live regression: a token X searched 10 minutes earlier was deferred by the budget
    next run; that run's "no provider could search it" momentum replaced the real reading,
    so social evidence only ever counted in the run a token was searched."""
    store = SocialStore(tmp_path / "s.sqlite3")
    earlier = momentum("A", "EMERGING", computed_at=NOW - timedelta(minutes=10))
    stale = momentum("B", "EMERGING", computed_at=NOW - timedelta(hours=3))
    deferred = [momentum(k, "UNAVAILABLE") for k in ("A", "B")]
    for m in (earlier, stale, *deferred):
        run(store.add_momentum(m))
    result = rank(
        tmp_path, [accelerating("A"), accelerating("B")], social=deferred, social_store=store
    )
    by = {g.symbol: g for g in result.candidates}
    assert by["A"].momentum.social_status == "SOCIAL_EMERGING"
    social = next(f for f in by["A"].scout_momentum.families if f.family == "social_momentum")
    assert any("10 minutes ago" in n for n in social.notes)
    assert by["B"].momentum.social_status == "SOCIAL_UNAVAILABLE"  # too old: never reused


def test_k_budget_never_overspends_with_real_results(tmp_path: Path) -> None:
    clock = Clock()
    reads: list[int] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        n = int(request.url.params["max_results"])
        reads.append(n)
        rows = [
            {"id": f"{len(reads)}{i}", "author_id": str(i), "text": "gm",
             "created_at": (NOW - timedelta(minutes=i + 1)).isoformat()}
            for i in range(3)
        ]  # fmt: skip
        return httpx2.Response(200, json={"data": rows, "meta": {"result_count": 3}})

    provider = XRecentSearchProvider(
        "token", True, 45,
        SocialProviderConfig(limits=ScoutProviderLimits(calls_per_minute=100),
                             max_results_per_page=20, max_terms_per_query=1),
        transport=httpx2.MockTransport(handle), now=clock,
    )  # fmt: skip
    svc = SocialScoutService([provider], SocialStore(tmp_path / "s.sqlite3"), now=clock)
    result = run(svc.observe(x_tokens(8)))
    # Small incremental searches use little of the budget, so more tokens get searched
    # than a first-come full-page reservation would allow (2 of 8).
    assert provider.results_this_run <= 45
    searched = sum(m.sources[0].status != "PROVIDER_UNAVAILABLE" for m in result.momentum)
    assert searched > 2
    assert all(n >= 10 for n in reads)


def test_k_missing_x_does_not_lower_the_rank(tmp_path: Path) -> None:
    """A token whose X search was deferred ranks exactly as if social were quiet."""
    deferred = SocialMomentum.model_validate(
        momentum("B", "UNAVAILABLE").model_dump()
        | {"reasons": ["no social provider could be searched"]}
    )
    quiet = momentum(
        "A", "QUIET", mentions=(0, 0), authors=(0, 0), exact=0, attributable=0, weighted=0
    )
    result = rank(tmp_path, [accelerating("A"), accelerating("B")], [deferred, quiet])
    a, b = by_symbol(result)["A"], by_symbol(result)["B"]
    assert a.score == b.score
    assert b.momentum.social_status == "SOCIAL_UNAVAILABLE"


# --- Stage rules ----------------------------------------------------------------------------


def test_one_isolated_price_spike_is_not_acceleration(tmp_path: Path) -> None:
    spike = candidate("A", FLAT, past=[Past(60, FLAT, 0.0007, 80_000)],
                      changes={"m5": 20, "m15": 30, "h1": 40, "h24": 40})  # fmt: skip
    [g] = rank(tmp_path, [spike]).candidates
    assert g.stage != "ACCELERATING"
    assert g.stage == "STEADY"


def test_young_pool_partial_windows_are_not_trusted(tmp_path: Path) -> None:
    # 40 minutes old: its "h1"/"h6" windows only hold 40 minutes, which would fake a
    # large h1/h6 acceleration. They are ignored; the token is NEW.
    young = candidate("A", [(40, 500, 5, 0.6), (1440, 0, 0, 0.5)], age_hours=0.67)
    c, _ = young
    ev = market_evidence(c, [], GrowthConfig())
    assert ev.timescales == []
    assert any("partial" in note for note in ev.ignored)
    [g] = rank(tmp_path, [young]).candidates
    assert g.stage == "NEW"
    assert "very_new" not in flag_codes(g)  # only flagged when Scout flags it


def test_young_pool_history_comparison_is_not_trusted(tmp_path: Path) -> None:
    """Live regression: an 18-minute-old pool compared with itself 15 minutes earlier (when
    it was 3 minutes old) showed "volume 6x, h1 now vs 15m ago" and ranked first. Both h1
    windows only held the minutes since launch: that is the pool filling up."""
    live: SEGMENTS = [(18, 500, 5, 0.6), (1440, 0, 0, 0.5)]
    then: SEGMENTS = [(3, 500, 5, 0.6), (1440, 0, 0, 0.5)]
    young = candidate("A", live, age_hours=0.3, past=[Past(15, then, 0.001, 80_000)])
    ev = market_evidence(young[0], [], GrowthConfig())
    assert ev.timescales == []
    assert any("3 minutes old then" in note for note in ev.ignored)
    [g] = rank(tmp_path, [young]).candidates
    assert g.stage == "NEW" and "history" not in g.momentum.timescales_rising
    assert not next(f for f in g.scout_momentum.families if f.family == "market_activity").available
    # The same comparison for a mature pool is real acceleration and is kept.
    old = candidate("B", [(15, 500, 5, 0.6), (1440, 100, 1, 0.5)], age_hours=48,
                    past=[Past(15, [(1440, 100, 1, 0.5)], 0.001, 80_000)])  # fmt: skip
    assert [t.name for t in market_evidence(old[0], [], GrowthConfig()).timescales][-1] == "history"


def test_thin_short_windows_are_not_trusted(tmp_path: Path) -> None:
    thin = candidate("A", [(15, 50, 0.2, 0.9), (1440, 10, 0.1, 0.5)], age_hours=48)
    ev = market_evidence(thin[0], [], GrowthConfig())
    assert not any(t.name == "short" for t in ev.timescales)
    assert any("trades in" in note for note in ev.ignored)


def test_early_stage(tmp_path: Path) -> None:
    # Short-term volume and trades pick up, but the medium scale and history don't agree.
    segments = [(15, 400, 3, 0.55), (1440, 150, 1, 0.55)]
    [g] = rank(tmp_path, [candidate("A", segments, age_hours=4)]).candidates
    assert g.stage == "EARLY"
    assert g.momentum.timescales_rising == ["short"]


def test_insufficient_data(tmp_path: Path) -> None:
    few = candidate("A", [(1440, 5, 0.005, 0.5)], age_hours=48)
    result = rank(tmp_path, [few])
    assert result.candidates == [] and result.eligible == 0
    [g] = result.unranked
    assert g.stage == "INSUFFICIENT_DATA" and not g.eligible
    assert "insufficient data" in g.ineligible_reasons[0]


def test_distribution_is_flagged(tmp_path: Path) -> None:
    segments = [
        (15, 1500, 12, 0.3),
        (60, 500, 4, 0.55),
        (360, 300, 2, 0.55),
        (1440, 200, 1.5, 0.55),
    ]
    [g] = rank(tmp_path, [candidate("A", segments)]).candidates
    assert "distribution" in flag_codes(g)
    assert g.stage != "ACCELERATING"
    assert any("contradiction" in n for n in family(g, "cross_confirmation").notes)


# --- Safety ---------------------------------------------------------------------------------


def test_critical_authority_strongly_reduces_rank(tmp_path: Path) -> None:
    risky = safety_snapshot("B", freeze_authority="Frz111", mint_authority="Mnt111")
    result = rank(tmp_path, [accelerating("A"), accelerating("B")], safety=[risky])
    s = by_symbol(result)
    b = s["B"]
    assert {"freeze_authority_active", "mint_authority_active"} <= flag_codes(b)
    assert (
        next(f for f in b.risk_flags if f.code == "freeze_authority_active").severity == "critical"
    )
    assert b.score <= s["A"].score - 30
    assert b.quality.freeze_authority_active is True


def test_safety_statuses_are_honest(tmp_path: Path) -> None:
    partial = safety_snapshot("B", concentration_source="largest_accounts",
                              concentration_lower_bound=True, holder_count_complete=False)  # fmt: skip
    evm, _ = accelerating("C")
    evm.chain, evm.address = "base", "0x" + "ab" * 20
    evm.canonical_id = f"base:{evm.address}"
    result = rank(tmp_path, [accelerating("A"), accelerating("B"), (evm, [])], safety=[partial])
    s = by_symbol(result)
    assert s["A"].quality.safety_status == "INSUFFICIENT_SAFETY_DATA"
    assert "safety_data_missing" in flag_codes(s["A"])
    assert s["B"].quality.safety_status == "SAFETY_CHECKS_PARTIAL"
    assert any("lower bounds" in m for m in s["B"].quality.safety_missing)
    assert s["C"].quality.safety_status == "INSUFFICIENT_SAFETY_DATA"
    assert "no on-chain safety source for base" in s["C"].quality.safety_missing[0]
    assert s["C"].quality.mint_authority_active is None  # unknown, never "revoked"


class FakeSafety:
    def __init__(self, fail: set[str]) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def get_snapshot(self, mint: str, pools: Any = ()) -> OnchainSafetySnapshot:
        self.calls.append(mint)
        if mint in self.fail:
            raise MarketDataUnavailableError("rpc down")
        key = next(k for k, v in MINTS.items() if v == mint)
        return safety_snapshot(key)


def test_safety_is_looked_up_for_top_candidates_only(tmp_path: Path) -> None:
    fake = FakeSafety(fail={MINTS["B"]})
    config = GrowthConfig(safety={"top_k": 2})
    svc = service(tmp_path, config, safety=fake)
    scenarios = [accelerating("A"), accelerating("B"), fading("F")]
    result = run(svc.rank([c for c, _ in scenarios]))
    assert sorted(fake.calls) == sorted([MINTS["A"], MINTS["B"]])  # top 2 only
    s = by_symbol(result)
    assert s["A"].quality.safety_status == "SAFETY_CHECKS_COMPLETE"
    assert s["B"].quality.safety_status == "INSUFFICIENT_SAFETY_DATA"  # failure: never safe
    assert any("safety lookup failed for 1" in n for n in result.notes)


# --- Score, config, output ------------------------------------------------------------------


def test_score_is_explainable(ranked: GrowthScoutResult) -> None:
    for g in [*ranked.candidates, *ranked.unranked]:
        sm = g.scout_momentum
        assert {f.family for f in sm.families} == {
            "market_activity", "liquidity_quality", "social_momentum", "earliness",
            "cross_confirmation",
        }  # fmt: skip
        assert abs(sum(f.weight for f in sm.families) - 1) < 1e-9
        assert abs(sum(f.contribution for f in sm.families) - sm.base) < 0.1
        assert sm.risk_penalty == round(min(70, sum(f.penalty for f in g.risk_flags)), 1)
        assert sm.score == round(
            max(0, min(100, sm.base + sm.stage_adjustment - sm.risk_penalty)), 1
        )
        assert sm.components["final_score"] == sm.score
        assert all(0 <= f.score <= 1 for f in sm.families)


def test_ranking_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    first = rank(tmp_path / "1", all_scenarios(), all_social())
    second = rank(tmp_path / "2", list(reversed(all_scenarios())), list(reversed(all_social())))
    assert [(g.canonical_id, g.score, g.stage) for g in first.candidates] == [
        (g.canonical_id, g.score, g.stage) for g in second.candidates
    ]


def test_top_n_keeps_everything_else_evaluated(tmp_path: Path) -> None:
    result = rank(tmp_path, all_scenarios(), all_social(), limit=3)
    assert len(result.candidates) == 3 and result.top(2) == result.candidates[:2]
    assert result.evaluated == 8 and result.eligible == 8
    assert [g.rank for g in result.candidates] == [1, 2, 3]
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    assert len(run(store.tokens())) == 8  # history for all of them is still stored


def test_discovery_modes(tmp_path: Path) -> None:
    big = accelerating("A", mcap=900_000_000, liquidity=5_000_000)
    old = accelerating("B", age_hours=24 * 400)
    early = rank(tmp_path / "e", [big, old, accelerating("C")])
    assert [g.symbol for g in early.candidates] == ["C"]
    assert all("established market" in g.ineligible_reasons[0] for g in early.unranked)
    trending = rank(
        tmp_path / "t", [big, old, accelerating("C")], config=GrowthConfig(mode="ALL_TRENDING")
    )
    assert trending.eligible == 3
    assert any(
        "ALL_TRENDING" in n for g in trending.candidates for n in family(g, "earliness").notes
    )


def test_scan_spends_social_searches_on_rankable_tokens_only(tmp_path: Path) -> None:
    """Live regression: 14 of 37 paid X searches went to tokens NEW_AND_EARLY can never
    rank (WBTC, ETH, 600-day-old pools, pools with a handful of trades)."""
    big = accelerating("A", mcap=900_000_000, liquidity=5_000_000)
    thin = candidate("B", [(1440, 5, 0.005, 0.5)], age_hours=48)
    good = accelerating("C")
    svc = service(tmp_path)
    scenarios = [big, thin, good]

    class Scout:
        store = svc.store

        async def discover(self) -> ScoutRun:
            for c, snaps in scenarios:
                await svc.store.record_seen(c)
                for s in snaps:
                    await svc.store.save_snapshot(s, 0)
            return ScoutRun(started_at=NOW, candidates=[c for c, _ in scenarios])

        async def lookup_exact_tokens(self, chain: str, addresses: list[str]) -> ScoutRun:
            return ScoutRun(started_at=NOW, candidates=[])

    searched: list[str] = []

    class Social:
        async def observe(self, tracked: list[TokenIdentity], *_: Any, **__: Any) -> SocialRun:
            searched.extend(t.canonical_id for t in tracked)
            return SocialRun(started_at=NOW, momentum=[momentum("C", "EMERGING")], providers=[])

    result = run(svc.scan(Scout(), Social()))  # type: ignore[arg-type]
    assert searched == [f"solana:{MINTS['C']}"]
    assert [g.symbol for g in result.candidates] == ["C"]
    assert result.candidates[0].momentum.social_status == "SOCIAL_EMERGING"
    assert {g.symbol for g in result.unranked} == {"A", "B"}  # still evaluated and reported


def test_market_cap_is_only_reported_when_trustworthy(tmp_path: Path) -> None:
    fdv_only = accelerating("A", mcap=None, fdv=2_000_000)
    odd_quote = accelerating("B", flags=ScoutRiskFlags(unrecognized_quote=True))
    s = by_symbol(rank(tmp_path, [fdv_only, odd_quote, accelerating("C")]))
    assert s["A"].market.market_cap_usd is None and s["A"].market.fdv_usd == 2_000_000
    assert "FDV only" in (s["A"].market.market_cap_note or "")
    assert s["B"].market.market_cap_usd is None and "unrecognized_quote" in flag_codes(s["B"])
    assert "MARKET_CONFIRMED" not in s["B"].quality.verification
    assert s["C"].market.market_cap_usd == 400_000


def test_weak_attribution_scales_social_down() -> None:
    cfg = GrowthConfig()
    exact = social_evidence(
        momentum("A", "ACCELERATING", exact=8, attributable=8, weighted=8), NOW, cfg
    )
    probable = social_evidence(
        momentum("A", "ACCELERATING", exact=0, attributable=8, weighted=4), NOW, cfg
    )
    assert exact.attribution == "exact" and probable.attribution == "probable"
    from upscale.services.scout.growth.scoring import social_family

    assert social_family(exact, cfg, 0.12).score > social_family(probable, cfg, 0.12).score > 0.4


def test_config_is_validated() -> None:
    with pytest.raises(ValidationError, match="dominate"):
        GrowthConfig(weights={"market_activity": 5.0})
    with pytest.raises(ValidationError, match="social_momentum"):
        GrowthConfig(weights={"social_momentum": 0.3, "market_activity": 0.3})
    with pytest.raises(ValidationError):
        GrowthConfig(mode="MOON")  # type: ignore[arg-type]
    with pytest.raises(GrowthConfigError):
        load_growth_config('{"stage": {"crowded_move_pct": 5000}}')
    assert load_growth_config(None) == GrowthConfig()
    assert load_growth_config('{"mode": "ALL_TRENDING"}').mode == "ALL_TRENDING"


def test_output_is_never_a_trade_decision(ranked: GrowthScoutResult) -> None:
    body = ranked.model_dump(mode="json")

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            assert not {"action", "decision", "recommendation", "signal"} & set(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            assert x.upper() not in ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL")

    walk(body)
    assert "not a probability of profit" in ranked.disclaimer
    json.dumps(body)


# --- End to end with the real social pipeline -----------------------------------------------


def test_real_spam_pipeline_lowers_social(tmp_path: Path) -> None:
    """Many mentions from one author with near-identical text: the social layer calls it
    high spam risk; Growth Scout caps social below neutral and flags it."""
    mint = MINTS["G"]
    posts = [
        {"id": f"s{i}", "author": "bot", "created_at": NOW - timedelta(minutes=2 * i + 1),
         "text": f"$GEM {mint} to the moon, buy now!!"}
        for i in range(25)
    ]  # fmt: skip
    clock = Clock()
    provider = StaticSocialProvider("X fixture", "x", posts, now=clock)
    social_svc = SocialScoutService(
        [provider], SocialStore(tmp_path / "s.sqlite3"), SocialConfig(), now=clock
    )
    identity = TokenIdentity(
        canonical_id=f"solana:{mint}", chain="solana", address=mint, symbol="GEM"
    )
    [m] = run(social_svc.observe([identity])).momentum
    assert m.quality.spam_risk == "high"
    result = rank(tmp_path, [accelerating("G"), accelerating("B")], [m])
    g, b = by_symbol(result)["G"], by_symbol(result)["B"]
    assert family(g, "social_momentum").score <= 0.2
    assert "social_spam" in flag_codes(g) and g.score < b.score


def test_growth_scout_is_wired_without_touching_the_disk_at_import() -> None:
    from upscale.services import growth_scout_service, scout_service, social_scout_service

    assert growth_scout_service.store is scout_service.store
    assert growth_scout_service.social_store is social_scout_service.store
    assert growth_scout_service.config.mode == "NEW_AND_EARLY"
    assert scout_service.store._conn is None
