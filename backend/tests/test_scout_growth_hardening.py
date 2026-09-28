"""Growth Scout hardening: regressions for every issue live validation exposed.

Stage adjustment, MARKET_COLLAPSE, earliness that isn't "smallest cap wins", cautious
buy / sell flow, tracked-token continuity, two-pass social priority, the X 402 short
circuit, request budgets within rate limits, NEW_AND_EARLY maturity and stage stability.
All offline and deterministic (same synthetic markets as test_scout_growth).
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from tests.test_scout_growth import (
    ACCEL,
    FLAT,
    MINTS,
    NOW,
    SEGMENTS,
    Clock,
    Past,
    accelerating,
    by_symbol,
    candidate,
    family,
    flag_codes,
    momentum,
    rank,
    run,
    service,
    x_service,
)
from upscale.services.scout.config import ScoutProviderLimits
from upscale.services.scout.growth import GrowthConfig, GrowthScoutService
from upscale.services.scout.growth.config import StageAdjustmentConfig
from upscale.services.scout.models import (
    ScoutCandidate,
    ScoutRejection,
    ScoutRun,
    ScoutSnapshot,
    ScoutSourceError,
    ScoutSourceEvidence,
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
from upscale.services.scout.social.models import SocialRun
from upscale.services.scout.store import ScoutSnapshotStore

Scenario = tuple[ScoutCandidate, list[ScoutSnapshot]]


# --- 1. Stage matters to ranking --------------------------------------------------------------


def test_fading_does_not_outrank_a_similar_accelerating_candidate(tmp_path: Path) -> None:
    """Live: pPOLY (FADING, volume 0.05x) ranked #3 on deep liquidity and earliness. Here
    the fading token has the *higher* base score (deep liquidity, social support) and
    still ranks below the accelerating one: the stage adjustment decides."""
    fading_market: SEGMENTS = [(15, 300, 2.5, 0.6), (60, 400, 3, 0.6), (360, 900, 7, 0.55),
                               (1440, 900, 7, 0.55)]  # fmt: skip
    rich_fading = candidate("F", fading_market, age_hours=40, liquidity=900_000,
                            mcap=3_000_000, changes={"m5": 0, "h1": -1, "h6": -3, "h24": 4})  # fmt: skip
    rising = candidate("A", [(15, 300, 2.5, 0.56), (60, 200, 1.7, 0.55), (360, 150, 1.2, 0.55),
                             (1440, 140, 1.1, 0.55)], age_hours=40, liquidity=9_000,
                       mcap=3_000_000, changes={"m5": 0, "h1": 3, "h6": 5, "h24": 8})  # fmt: skip
    result = rank(tmp_path, [rich_fading, rising], social=[momentum("F", "EMERGING")])
    s = by_symbol(result)
    f, a = s["F"].scout_momentum, s["A"].scout_momentum
    assert (s["F"].stage, s["A"].stage) == ("FADING", "ACCELERATING")
    assert f.base > a.base  # on signals alone the fading token would lead
    assert f.base + f.stage_adjustment < a.base + a.stage_adjustment
    assert s["A"].rank == 1 and s["F"].rank == 2  # still ranked and visible
    assert (a.stage_adjustment, f.stage_adjustment) == (8.0, -12.0)
    # Every term is exposed and adds up.
    for sm in (a, f):
        parts = sm.components
        assert {"base_score", "market_activity", "liquidity_quality", "social_momentum",
                "earliness", "cross_confirmation", "stage_adjustment", "risk_penalty",
                "final_score"} <= set(parts)  # fmt: skip
        total = parts["base_score"] + parts["stage_adjustment"] + parts["risk_penalty"]
        assert parts["final_score"] == pytest.approx(max(0.0, min(100.0, total)), abs=0.11)


def test_stage_adjustments_are_validated_in_preference_order() -> None:
    with pytest.raises(ValidationError):
        StageAdjustmentConfig(fading=5.0)  # FADING may never beat STEADY
    with pytest.raises(ValidationError):
        StageAdjustmentConfig(new=3.0)  # plain NEW above promising NEW


# --- 2. MARKET_COLLAPSE -----------------------------------------------------------------------


def _dump(key: str, pct: float) -> Scenario:
    segments: SEGMENTS = [(60, 2000, 12, 0.9), (1440, 200, 1.5, 0.9)]
    return candidate(key, segments, changes={"m5": 0, "h1": pct, "h6": pct, "h24": pct})


@pytest.mark.parametrize("pct", [-100.0, -99.0])
def test_price_collapse_gets_no_earliness(tmp_path: Path, pct: float) -> None:
    [g] = rank(tmp_path, [_dump("A", pct)]).candidates
    assert g.quality.market_status == "MARKET_COLLAPSE"
    assert "market_collapse" in flag_codes(g)
    assert family(g, "earliness").score == 0.0 and family(g, "cross_confirmation").score == 0.0
    assert g.stage == "FADING" and g.stage_reasons[0] == "MARKET_COLLAPSE"
    assert not any("still early" in r for r in g.reasons_surfaced)
    assert g.score < 20


def test_severe_liquidity_collapse(tmp_path: Path) -> None:
    drained = candidate(
        "A", ACCEL, liquidity=3_000,
        past=[Past(60, ACCEL, 0.001, 90_000), Past(30, ACCEL, 0.001, 60_000)],
    )  # fmt: skip
    [g] = rank(tmp_path, [drained]).candidates
    evidence = " ".join(g.quality.collapse_evidence)
    assert "liquidity collapsed" in evidence and "no longer usable" in evidence
    assert family(g, "earliness").score == 0.0


def test_dying_activity_is_a_collapse(tmp_path: Path) -> None:
    dying = candidate("A", [(60, 5, 0.02, 0.5), (1440, 400, 3, 0.5)], age_hours=48)
    [g] = rank(tmp_path, [dying]).candidates
    assert any("activity dying" in e for e in g.quality.collapse_evidence)


def test_a_legitimately_new_token_is_not_a_collapse(tmp_path: Path) -> None:
    young = candidate("A", [(40, 500, 5, 0.6), (1440, 0, 0, 0.5)], age_hours=0.67,
                      changes={"m5": 2, "h1": 15, "h24": 15})  # fmt: skip
    [g] = rank(tmp_path, [young]).candidates
    assert g.quality.market_status == "OK" and g.stage == "NEW"
    assert family(g, "earliness").score > 0.5


# --- 3. Earliness is not "smallest cap wins" -------------------------------------------------


def test_healthy_500k_outranks_a_dead_10k(tmp_path: Path) -> None:
    healthy = candidate("A", [(60, 300, 3, 0.58), (1440, 220, 2, 0.55)], age_hours=30,
                        liquidity=90_000, mcap=500_000, changes={"h1": 3, "h24": 12})  # fmt: skip
    dead = candidate("B", [(60, 3, 0.1, 0.5), (1440, 30, 0.05, 0.5)], age_hours=30,
                     liquidity=4_000, mcap=10_000)  # fmt: skip
    s = by_symbol(rank(tmp_path, [healthy, dead]))
    assert s["A"].score > s["B"].score
    assert family(s["A"], "earliness").score >= family(s["B"], "earliness").score
    tiny_size = next(
        x for x in family(s["B"], "earliness").signals if x.name == "market_size_context"
    )
    assert tiny_size.score <= 0.5  # thin liquidity: size earns no extra earliness


def test_a_move_largely_made_caps_earliness(tmp_path: Path) -> None:
    ran = candidate("A", FLAT, age_hours=30, changes={"h1": 1, "h24": 700})
    [g] = rank(tmp_path, [ran]).candidates
    assert family(g, "earliness").score < 0.5


# --- 4. Buy / sell flow quality ---------------------------------------------------------------


def _flow(key: str, share: float, buyer_share: float | None = None, h1: float = 12.0) -> Scenario:
    c = candidate(key, [(60, 900, 6, share), (1440, 200, 1.5, 0.5)],
                  changes={"m5": 1, "h1": h1, "h24": 20})  # fmt: skip
    if buyer_share is not None or buyer_share is None:
        w = c[0].metrics.window("h1")
        assert w is not None and w.buys is not None and w.sells is not None
        if buyer_share is None:
            for win in c[0].metrics.windows:
                win.buyers = win.sellers = None
        else:
            w.buyers, w.sellers = round(100 * buyer_share), round(100 * (1 - buyer_share))
    return c


def test_strong_buyer_participation_is_consistent_flow(tmp_path: Path) -> None:
    [g] = rank(tmp_path, [_flow("A", 0.65, buyer_share=0.6)]).candidates
    assert g.quality.flow_quality == "consistent" and not g.quality.flow_notes
    assert "flow_divergence" not in flag_codes(g)


def test_high_volume_on_a_9_percent_buy_count_is_divergence(tmp_path: Path) -> None:
    """Live: Holdoween ranked #1 with 9% of trades buys while price rose 46%."""
    s = by_symbol(rank(tmp_path, [_flow("A", 0.09, buyer_share=0.2), _flow("B", 0.6, 0.6)]))
    weak, fine = s["A"], s["B"]
    assert weak.quality.flow_quality == "divergent"
    assert "flow_divergence" in flag_codes(weak)
    assert family(weak, "cross_confirmation").score < family(fine, "cross_confirmation").score
    pressure = next(x for x in family(weak, "market_activity").signals if x.name == "buy_pressure")
    assert pressure.score <= 0.3
    assert "buyer activity increasing" not in " ".join(weak.reasons_surfaced)
    assert weak.score < fine.score


def test_bot_like_many_small_buys_are_not_buyer_strength(tmp_path: Path) -> None:
    [g] = rank(tmp_path, [_flow("A", 0.85, buyer_share=0.3)]).candidates
    assert g.quality.flow_quality == "divergent"
    assert any("few wallets" in n for n in g.quality.flow_notes)
    assert "flow_in_doubt" in flag_codes(g)
    assert "buyer activity increasing" not in " ".join(g.reasons_surfaced)


def test_buys_without_a_price_response_are_in_doubt(tmp_path: Path) -> None:
    [g] = rank(tmp_path, [_flow("A", 0.85, buyer_share=0.8, h1=-2.0)]).candidates
    assert any("without any price response" in n for n in g.quality.flow_notes)


def test_missing_distinct_wallet_data_is_count_only_not_zero(tmp_path: Path) -> None:
    [g] = rank(tmp_path, [_flow("A", 0.65, buyer_share=None)]).candidates
    assert g.quality.flow_quality == "count_only"
    assert g.momentum.buyer_share is None
    pressure = next(x for x in family(g, "market_activity").signals if x.name == "buy_pressure")
    assert pressure.score > 0.5  # counts still count; the missing wallets aren't a zero


# --- 5. Tracked-token continuity --------------------------------------------------------------


def _listed(sc: Scenario, at: datetime = NOW) -> ScoutCandidate:
    c, _ = sc
    return c.model_copy(
        update={
            "observed_at": at,
            "sources": [ScoutSourceEvidence(provider="GeckoTerminal", kind="new",
                                            listing="new_pools", fetched_at=at)],
        }
    )  # fmt: skip


def _looked_up(c: ScoutCandidate, at: datetime) -> ScoutCandidate:
    return c.model_copy(
        update={
            "observed_at": at,
            "sources": [ScoutSourceEvidence(provider="DEX Screener", kind="lookup",
                                            listing="tokens/v1", fetched_at=at)],
        }
    )  # fmt: skip


class FakeScout:
    """Discovery lists `listed` (per run); exact lookups answer from `market`."""

    def __init__(self, store: ScoutSnapshotStore, clock: Clock, market: list[ScoutCandidate]):
        self.store = store
        self.clock = clock
        self.market = {c.canonical_id: c for c in market}
        self.listed: list[str] = []
        self.fail_lookup = False
        self.lookups: list[str] = []

    async def discover(self) -> ScoutRun:
        found = [_listed((self.market[cid], []), self.clock()) for cid in self.listed]
        for c in found:
            await self.store.record_seen(c)
        return ScoutRun(started_at=self.clock(), candidates=found)

    async def lookup_exact_tokens(self, chain: str, addresses: list[str]) -> ScoutRun:
        if self.fail_lookup:
            # The first provider rejects every token; the fallback provider's lookup fails.
            return ScoutRun(
                started_at=self.clock(),
                candidates=[],
                rejected=[ScoutRejection(canonical_id=f"{chain}:{a}", chain=chain, address=a,
                                         symbol=None, reasons=["no usable pool"])
                          for a in addresses],
                errors=[ScoutSourceError(provider="GeckoTerminal", kind="lookup", chain=chain,
                                         error="GeckoTerminal rate limit reached")],
            )  # fmt: skip
        out = []
        for a in addresses:
            self.lookups.append(a)
            c = self.market.get(f"{chain}:{a}")
            if c is not None:
                c = _looked_up(c, self.clock())
                await self.store.record_seen(c)
                out.append(c)
        return ScoutRun(started_at=self.clock(), candidates=out)


def _scan_setup(tmp_path: Path, keys: str = "ABC") -> tuple[GrowthScoutService, FakeScout, Clock]:
    clock = Clock()
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    svc = GrowthScoutService(store, now=clock)
    scout = FakeScout(store, clock, [accelerating(k)[0] for k in keys])
    return svc, scout, clock


def test_token_missed_by_discovery_stays_rankable_from_refresh(tmp_path: Path) -> None:
    svc, scout, clock = _scan_setup(tmp_path)
    scout.listed = [f"solana:{MINTS[k]}" for k in "ABC"]
    first = run(svc.scan(scout))  # type: ignore[arg-type]
    assert first.universe is not None and first.universe.discovered == 3
    clock.at += timedelta(minutes=10)
    scout.listed = [f"solana:{MINTS['C']}"]  # A and B dropped off the listings
    second = run(svc.scan(scout))  # type: ignore[arg-type]
    assert {g.symbol for g in second.candidates} == {"A", "B", "C"}
    assert second.universe is not None
    assert (second.universe.discovered, second.universe.refreshed) == (1, 2)
    assert sorted(scout.lookups) == sorted([MINTS["A"], MINTS["B"]])  # C isn't fetched twice


def test_provider_failure_does_not_churn_the_top_list(tmp_path: Path) -> None:
    svc, scout, clock = _scan_setup(tmp_path)
    scout.listed = [f"solana:{MINTS[k]}" for k in "ABC"]
    before = [g.canonical_id for g in run(svc.scan(scout)).candidates]  # type: ignore[arg-type]
    clock.at += timedelta(minutes=10)
    scout.listed = []  # every discovery listing failed this run
    after = run(svc.scan(scout))  # type: ignore[arg-type]
    assert sorted(g.canonical_id for g in after.candidates) == sorted(before)


def test_a_genuinely_stale_token_expires(tmp_path: Path) -> None:
    svc, scout, clock = _scan_setup(tmp_path)
    scout.listed = [f"solana:{MINTS[k]}" for k in "AB"]
    run(svc.scan(scout))  # type: ignore[arg-type]
    scout.listed = [f"solana:{MINTS['B']}"]  # B keeps being rediscovered, A never again
    for _ in range(4):
        clock.at += timedelta(hours=2)
        result = run(svc.scan(scout))  # type: ignore[arg-type]
    assert {g.symbol for g in result.candidates} == {"B"}
    assert result.universe is not None and result.universe.expired == 1


def test_refresh_failure_is_reported_not_counted_as_gone(tmp_path: Path) -> None:
    """Live: DEX Screener rejected tokens that only GeckoTerminal knows, and the
    GeckoTerminal fallback was rate-limited: those are unresolved, not dead."""
    svc, scout, clock = _scan_setup(tmp_path)
    scout.listed = [f"solana:{MINTS[k]}" for k in "AB"]
    run(svc.scan(scout))  # type: ignore[arg-type]
    clock.at += timedelta(minutes=10)
    scout.listed, scout.fail_lookup = [], True
    result = run(svc.scan(scout))  # type: ignore[arg-type]
    assert result.universe is not None
    assert (result.universe.unusable, result.universe.unresolved) == (0, 2)
    assert any("refresh source(s) failed" in n for n in result.notes)


def test_a_tracked_token_that_is_gone_or_unusable_is_dropped(tmp_path: Path) -> None:
    svc, scout, clock = _scan_setup(tmp_path)
    scout.listed = [f"solana:{MINTS[k]}" for k in "AB"]
    run(svc.scan(scout))  # type: ignore[arg-type]
    clock.at += timedelta(minutes=10)
    scout.listed = []
    del scout.market[f"solana:{MINTS['A']}"]  # the lookup no longer knows A
    result = run(svc.scan(scout))  # type: ignore[arg-type]
    assert {g.symbol for g in result.candidates} == {"B"}
    assert result.universe is not None
    assert (result.universe.refreshed, result.universe.unusable) == (1, 1)


def test_canonical_identity_is_never_duplicated(tmp_path: Path) -> None:
    """The same token found by discovery and by refresh is one candidate; two tokens that
    share a ticker stay two."""
    clock = Clock()
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    twin = accelerating("B", symbol="SAME")[0]
    scout = FakeScout(store, clock, [accelerating("A", symbol="SAME")[0], twin])
    svc = GrowthScoutService(store, now=clock)
    scout.listed = [f"solana:{MINTS['A']}", f"solana:{MINTS['B']}", f"solana:{MINTS['A']}"]
    result = run(svc.scan(scout))  # type: ignore[arg-type]
    ids = [g.canonical_id for g in [*result.candidates, *result.unranked]]
    assert sorted(ids) == sorted({f"solana:{MINTS['A']}", f"solana:{MINTS['B']}"})
    assert [g.symbol for g in result.candidates] == ["SAME", "SAME"]


# --- 6. Two-pass social priority -------------------------------------------------------------


def test_limited_budget_goes_to_leading_candidates_without_starving_the_rest(
    tmp_path: Path,
) -> None:
    clock = Clock()
    svc, _ = x_service(tmp_path, clock, reads=25)  # 2 searches per run
    tokens = [TokenIdentity(canonical_id=f"x:t{i}", chain="solana", address=f"t{i}")
              for i in range(6)]  # fmt: skip
    priority = {t.canonical_id: i for i, t in enumerate(tokens)}  # t0 leads

    def searched() -> set[str]:
        result = run(svc.observe(list(reversed(tokens)), priority=priority))
        clock.at += timedelta(minutes=10)
        return {m.canonical_id for m in result.momentum
                if m.sources[0].status == "PROVIDER_CHECKED_ZERO_MATCHES"}  # fmt: skip

    assert searched() == {"x:t0", "x:t1"}  # pass 2 spends the budget on the leaders first
    covered = searched() | searched()
    assert covered == {f"x:t{i}" for i in range(2, 6)}  # then everybody else, in turn
    assert searched() == {"x:t0", "x:t1"}  # and the leaders' refresh comes back round


# --- 7. Request budgets within the real rate limit --------------------------------------------


def test_request_budget_cannot_exceed_the_rate_limit() -> None:
    cfg = SocialConfig()
    for p in (cfg.farcaster, cfg.x):
        assert p.max_requests_per_run is not None
        assert p.max_requests_per_run <= p.limits.calls_per_minute
    with pytest.raises(ValidationError):
        SocialProviderConfig(limits=ScoutProviderLimits(calls_per_minute=30),
                             max_requests_per_run=60)  # fmt: skip


def test_scheduling_never_sends_calls_the_rate_limit_would_refuse(tmp_path: Path) -> None:
    sent: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.url.params["query"])
        return httpx2.Response(200, json={"data": [], "meta": {"result_count": 0}})

    provider = XRecentSearchProvider(
        "token", True, 1000,
        SocialProviderConfig(limits=ScoutProviderLimits(calls_per_minute=3),
                             max_requests_per_run=3, max_terms_per_query=1),
        transport=httpx2.MockTransport(handle), now=lambda: NOW,
    )  # fmt: skip
    svc = SocialScoutService([provider], SocialStore(tmp_path / "s.sqlite3"), now=lambda: NOW)
    tokens = [TokenIdentity(canonical_id=f"x:t{i}", chain="solana", address=f"t{i}")
              for i in range(6)]  # fmt: skip
    result = run(svc.observe(tokens))
    assert len(sent) == 3
    errors = [m.sources[0].error or "" for m in result.momentum
              if m.sources[0].status == "PROVIDER_UNAVAILABLE"]  # fmt: skip
    assert len(errors) == 3 and all("deferred" in e for e in errors)
    assert not any("request limit was reached" in e for e in errors)  # never refused


# --- 8. X credits exhausted (HTTP 402) --------------------------------------------------------


def test_x_402_stops_every_further_x_request_this_run(tmp_path: Path) -> None:
    sent: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.url.params["query"])
        return httpx2.Response(402, json={"title": "CreditsDepleted"})

    x = XRecentSearchProvider(
        "token", True, 1000,
        SocialProviderConfig(limits=ScoutProviderLimits(calls_per_minute=10, max_concurrency=1),
                             max_requests_per_run=10, max_terms_per_query=1),
        transport=httpx2.MockTransport(handle), now=lambda: NOW,
    )  # fmt: skip
    posts = [{"id": "1", "author": "a", "text": "gm", "created_at": NOW - timedelta(minutes=5)}]
    other = StaticSocialProvider("Forum", "discourse", posts, now=lambda: NOW)
    svc = SocialScoutService([x, other], SocialStore(tmp_path / "s.sqlite3"), now=lambda: NOW)
    tokens = [TokenIdentity(canonical_id=f"x:t{i}", chain="solana", address=f"t{i}")
              for i in range(6)]  # fmt: skip
    result = run(svc.observe(tokens))
    assert len(sent) == 1  # the 402, then nothing more
    statuses = {m.canonical_id: {s.provider: s for s in m.sources} for m in result.momentum}
    for cid, sources in statuses.items():
        assert sources["X"].status == "PROVIDER_UNAVAILABLE"  # never a checked zero
        assert "credits are exhausted" in (sources["X"].error or ""), cid
        assert sources["Forum"].status in ("PROVIDER_OK", "PROVIDER_CHECKED_ZERO_MATCHES")
    [check] = [p for p in result.providers if p.provider == "X"]
    assert check.requests == 1 and check.results == 0
    # A new run tries again (credits may have been added).
    run(svc.observe(tokens[:1]))
    assert len(sent) == 2


# --- 9. NEW_AND_EARLY maturity ----------------------------------------------------------------


def test_mature_market_needs_a_new_acceleration_regime(tmp_path: Path) -> None:
    """Live: QQQB (75 days, $60M cap) and CTR (124 days) stayed in NEW_AND_EARLY."""
    # Short-term pick-up only (EARLY): the medium windows don't agree.
    early_segments = [(15, 300, 2, 0.55), (60, 150, 1, 0.55), (1440, 150, 1, 0.55)]
    mature = candidate("A", early_segments, age_hours=24 * 90, mcap=60_000_000,
                       liquidity=700_000)  # fmt: skip
    regime = accelerating("B", age_hours=24 * 90, mcap=60_000_000, liquidity=700_000)
    young = candidate("C", early_segments, age_hours=30)
    result = rank(tmp_path, [mature, regime, young])
    s = by_symbol(result)
    assert not s["A"].eligible and "mature market" in s["A"].ineligible_reasons[0]
    assert s["A"].stage == "EARLY"
    assert s["A"].quality.maturity is not None and s["A"].quality.maturity >= 0.5
    assert s["B"].eligible and s["B"].stage == "ACCELERATING"  # a genuinely new regime
    assert s["C"].eligible and (s["C"].quality.maturity or 0) < 0.5
    trending = rank(tmp_path / "t", [mature], config=GrowthConfig(mode="ALL_TRENDING"))
    assert trending.eligible == 1  # ALL_TRENDING stays broad


# --- 10. Stage stability ---------------------------------------------------------------------


def _stages(tmp_path: Path, cid: str, *stages: tuple[int, str]) -> ScoutSnapshotStore:
    store = ScoutSnapshotStore(tmp_path / "scout.sqlite3")
    for minutes_ago, stage in stages:
        run(store.record_stages(NOW - timedelta(minutes=minutes_ago), {cid: stage}))
    return store


def _one_scale_fade(key: str) -> Scenario:
    # Only the short windows deteriorate (one noisy window): FADING on current evidence.
    return candidate(key, [(15, 60, 0.9, 0.3), (1440, 150, 1.5, 0.5)], age_hours=48,
                     changes={"m5": -2, "m15": -3, "h1": -6, "h24": -2})  # fmt: skip


def test_one_noisy_window_does_not_flip_early_to_fading(tmp_path: Path) -> None:
    cid = f"solana:{MINTS['A']}"
    fresh = rank(tmp_path / "fresh", [_one_scale_fade("A")]).candidates
    assert fresh[0].stage == "FADING"  # without a stored stage, current evidence decides
    _stages(tmp_path, cid, (10, "EARLY"))
    [g] = rank(tmp_path, [_one_scale_fade("A")]).candidates
    assert g.stage == "STEADY" and g.unconfirmed_stage == "FADING"
    assert "awaiting confirmation" in g.stage_reasons[0]


def test_a_confirmed_reversal_is_not_held_back(tmp_path: Path) -> None:
    cid = f"solana:{MINTS['A']}"
    _stages(tmp_path, cid, (10, "EARLY"))
    broad = candidate("A", [(15, 160, 2, 0.3), (60, 320, 3.2, 0.4), (360, 800, 8, 0.5),
                            (1440, 1200, 12, 0.5)], age_hours=48,
                      changes={"m5": -2, "h1": -8, "h24": -5})  # fmt: skip
    [g] = rank(tmp_path, [broad]).candidates
    assert g.stage == "FADING" and g.unconfirmed_stage is None


def test_fading_to_accelerating_needs_broader_evidence(tmp_path: Path) -> None:
    cid = f"solana:{MINTS['A']}"
    _stages(tmp_path, cid, (10, "FADING"))
    two_scales = candidate("A", ACCEL, age_hours=30, changes={"h1": 9, "h24": 20})
    [g] = rank(tmp_path, [two_scales]).candidates
    assert g.stage == "EARLY" and g.unconfirmed_stage == "ACCELERATING"
    # The same stored stage an hour ago is outside the memory: current evidence rules.
    _stages(tmp_path / "old", cid, (90, "FADING"))
    [h] = rank(tmp_path / "old", [two_scales]).candidates
    assert h.stage == "ACCELERATING"


def test_a_collapse_is_never_held_back(tmp_path: Path) -> None:
    cid = f"solana:{MINTS['A']}"
    _stages(tmp_path, cid, (10, "ACCELERATING"))
    [g] = rank(tmp_path, [_dump("A", -100.0)]).candidates
    assert g.stage == "FADING" and g.quality.market_status == "MARKET_COLLAPSE"


def test_rising_on_one_scale_and_falling_on_another_is_not_early(tmp_path: Path) -> None:
    """Live: pPOLY was EARLY on a 6h-vs-24h burst while its last hour had two trades and
    h1 volume had dropped to 0.2x of 30 minutes earlier."""
    burst: SEGMENTS = [(60, 40, 0.05, 0.5), (360, 400, 2.2, 0.5), (1440, 20, 0.15, 0.5)]
    before: SEGMENTS = [(60, 100, 0.12, 0.5), (360, 400, 2.2, 0.5), (1440, 20, 0.15, 0.5)]
    mixed = candidate("A", burst, age_hours=85, past=[Past(30, before, 0.001, 80_000)])
    [g] = rank(tmp_path, [mixed]).candidates
    assert "medium" in g.momentum.timescales_rising
    assert g.stage == "STEADY" and g.stage_reasons[0].startswith("mixed")


# --- Scan wiring --------------------------------------------------------------------------------


def test_scan_records_stages_and_passes_provisional_priority(tmp_path: Path) -> None:
    svc, scout, clock = _scan_setup(tmp_path, "AB")
    scout.listed = [f"solana:{MINTS[k]}" for k in "AB"]
    seen: dict[str, Any] = {}

    class Social:
        async def observe(self, tracked: list[TokenIdentity], *_: Any, **kw: Any) -> SocialRun:
            seen["priority"] = kw["priority"]
            return SocialRun(started_at=clock(), momentum=[momentum("A", "EMERGING")],
                             providers=[])  # fmt: skip

    result = run(svc.scan(scout, Social()))  # type: ignore[arg-type]
    assert sorted(seen["priority"].values()) == [0, 1]
    stored = run(svc.store.recent_stages([g.canonical_id for g in result.candidates],
                                         NOW - timedelta(minutes=1)))  # fmt: skip
    assert {cid: s[-1][1] for cid, s in stored.items()} == {
        g.canonical_id: g.stage for g in result.candidates
    }


def test_v1_store_is_upgraded_in_place(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "v1.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE scout_tokens (canonical_id TEXT PRIMARY KEY, chain TEXT NOT NULL,
            address TEXT NOT NULL, symbol TEXT, name TEXT, first_seen_at REAL NOT NULL,
            first_seen_provider TEXT NOT NULL, first_seen_kind TEXT NOT NULL,
            last_seen_at REAL NOT NULL);
        INSERT INTO scout_tokens VALUES ('solana:x', 'solana', 'x', 'X', 'X', 1, 'p', 'new', 5);
        PRAGMA user_version = 1;
        """
    )
    db.close()
    store = ScoutSnapshotStore(path)
    assert run(store.discovered_tokens(datetime.fromtimestamp(4, NOW.tzinfo))) == [
        ("solana:x", "solana", "x")
    ]
    assert run(store.expired_count(NOW, NOW)) == 0
    _ = (FLAT, service)  # shared helpers kept importable
