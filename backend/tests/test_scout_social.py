"""Scout social / attention intelligence: attribution, windows, acceleration, quality,
cross-platform and market cross-checks, provider adapters, storage and privacy.

All offline: providers are fixture-backed (`StaticSocialProvider`) or MockTransport fakes.
"""

import asyncio
import json
import random
from collections import Counter
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from upscale.services.market_data import MarketDataUnavailableError
from upscale.services.scout.config import ScoutProviderLimits
from upscale.services.scout.gate import RequestGate
from upscale.services.scout.models import (
    HistoryComparison,
    ScoutGrowthFeatures,
    WindowAcceleration,
)
from upscale.services.scout.social import (
    AttributionIndex,
    DiscourseForumProvider,
    NeynarFarcasterProvider,
    RedditProvider,
    SocialConfig,
    SocialConfigError,
    SocialScoutService,
    SocialStore,
    StaticSocialProvider,
    TokenIdentity,
    XRecentSearchProvider,
    load_social_config,
)
from upscale.services.scout.social.analysis import momentum_state
from upscale.services.scout.social.models import (
    CrossPlatformConfirmation,
    MetricTrend,
    SocialPost,
    SocialQuality,
    WindowTrend,
)
from upscale.services.scout.social.providers import plan_queries
from upscale.services.scout.social.text import (
    author_key,
    extract_addresses,
    extract_cashtags,
    fingerprint,
    hamming,
    simhash,
)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
MINT = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
MINT_2 = "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"
MINT_3 = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
EVM = "0xabcdef0123456789abcdef0123456789abcdef01"
EVM_2 = "0x1111111111111111111111111111111111111111"

NEWT = TokenIdentity(
    canonical_id=f"solana:{MINT}", chain="solana", address=MINT, symbol="NEWT", name="Newt Protocol"
)
NEWT_BASE = TokenIdentity(
    canonical_id=f"base:{EVM}", chain="base", address=EVM, symbol="NEWT", name="Newt Base"
)
OTHER = TokenIdentity(
    canonical_id=f"solana:{MINT_2}", chain="solana", address=MINT_2, symbol="OTH", name="Otherland"
)


class Now:
    def __init__(self) -> None:
        self.at = NOW

    def __call__(self) -> datetime:
        return self.at

    def mono(self) -> float:
        return self.at.timestamp()


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def post(text: str, *, author: str = "a", platform: str = "x", handle: str | None = None,
         urls: list[str] | None = None) -> SocialPost:  # fmt: skip
    return SocialPost(
        provider="P",
        platform=platform,
        post_id="1",
        created_at=NOW,
        text=text,
        author_key=author,
        author_handle=handle,
        urls=urls or [],
    )


def levels(index: AttributionIndex, p: SocialPost) -> dict[str | None, str]:
    return {a.canonical_id: a.level for a in index.attribute(p)}


# --- Text utilities -------------------------------------------------------------------------


def test_reference_extraction() -> None:
    text = f"Buy $NEWT now! CA {MINT} or {EVM.upper().replace('0X', '0x')} - $100 gain"
    assert extract_cashtags(text) == {"newt"}  # "$100" is a price, not a ticker
    assert ("solana", MINT) in extract_addresses(text)
    assert ("evm", EVM) in extract_addresses(text)


def test_fingerprints_ignore_links_and_contracts_and_catch_near_duplicates() -> None:
    a = f"Huge news for the community, join now {MINT} https://a.example/x"
    b = f"Huge news for the community, join now {MINT_2} https://b.example/y"
    assert fingerprint(a) == fingerprint(b)
    near = simhash("huge news for the whole community today join us now")
    assert hamming(near, simhash("huge news for the whole community today join us now!!")) == 0
    assert hamming(near, simhash("completely unrelated sentence about the weather tomorrow")) > 3


def test_author_keys_are_opaque_and_salted() -> None:
    k1 = author_key(b"salt-1", "x", "12345")
    assert "12345" not in k1 and len(k1) == 20
    assert k1 == author_key(b"salt-1", "x", "12345")
    assert k1 != author_key(b"salt-2", "x", "12345")
    assert k1 != author_key(b"salt-1", "reddit", "12345")


# --- Attribution ----------------------------------------------------------------------------


def test_exact_contract_attribution() -> None:
    index = AttributionIndex([NEWT, OTHER])
    [a] = index.attribute(post(f"new gem {MINT} just launched"))
    assert (a.level, a.canonical_id, a.token_reference) == ("EXACT", NEWT.canonical_id, MINT)


def test_contract_in_a_link_is_exact() -> None:
    index = AttributionIndex([NEWT])
    p = post("look at this", urls=[f"https://dexscreener.com/solana/{MINT}"])
    assert levels(index, p) == {NEWT.canonical_id: "EXACT"}


def test_strong_from_official_account_naming_the_token() -> None:
    newt = NEWT.model_copy(update={"official_accounts": {"x": {"newtprotocol"}}})
    index = AttributionIndex([newt])
    p = post("Newt Protocol v2 is live on Solana", handle="NewtProtocol")
    [a] = index.attribute(p)
    assert a.level == "STRONG" and "official account" in a.reason


def test_strong_from_official_domain_link() -> None:
    newt = NEWT.model_copy(update={"official_domains": {"newt.example"}})
    index = AttributionIndex([newt])
    p = post("read the docs", urls=["https://www.newt.example/docs"])
    assert levels(index, p) == {NEWT.canonical_id: "STRONG"}


def test_strong_from_ticker_name_and_chain() -> None:
    index = AttributionIndex([NEWT], universe=[NEWT, OTHER])
    p = post("$NEWT (Newt Protocol) on Solana is moving")
    [a] = index.attribute(p)
    assert a.level == "STRONG" and "ticker, name and chain" in a.reason


def test_probable_ticker_with_chain_and_no_competitor() -> None:
    index = AttributionIndex([NEWT])
    [a] = index.attribute(post("$NEWT on solana looks alive"))
    assert (a.level, a.canonical_id) == ("PROBABLE", NEWT.canonical_id)


def test_bare_ticker_is_ambiguous_without_a_directory() -> None:
    index = AttributionIndex([NEWT])  # no universe: competing tokens can't be ruled out
    [a] = index.attribute(post("$NEWT is exploding"))
    assert a.level == "AMBIGUOUS" and a.canonical_id is None


def test_bare_ticker_stays_ambiguous_even_when_the_directory_knows_no_competitor() -> None:
    # Scout's directory only holds what Scout has seen: "no competitor known" isn't evidence
    index = AttributionIndex([NEWT], universe=[NEWT, OTHER])
    [a] = index.attribute(post("$NEWT is exploding"))
    assert a.level == "AMBIGUOUS" and a.canonical_id is None and a.candidates == [NEWT.canonical_id]


def test_ticker_shared_by_many_tokens_is_ambiguous() -> None:
    lookalikes = [
        TokenIdentity(
            canonical_id=f"solana:{m}", chain="solana", address=m, symbol="NEWT", name=f"Clone {i}"
        )
        for i, m in enumerate(["A" * 32 + str(i) for i in range(1, 5)])
    ]
    index = AttributionIndex([NEWT], universe=lookalikes)
    [a] = index.attribute(post("$NEWT is exploding"))
    assert a.level == "AMBIGUOUS" and a.canonical_id is None
    assert len(a.candidates) == 5 and NEWT.canonical_id in a.candidates


def test_duplicate_ticker_across_chains_is_resolved_by_the_chain() -> None:
    index = AttributionIndex([NEWT, NEWT_BASE])
    assert levels(index, post("$NEWT on base is running")) == {
        NEWT_BASE.canonical_id: "PROBABLE",
        NEWT.canonical_id: "REJECTED",
    }
    [a] = index.attribute(post("$NEWT is running"))  # no chain: both are plausible
    assert a.level == "AMBIGUOUS" and set(a.candidates) == {
        NEWT.canonical_id,
        NEWT_BASE.canonical_id,
    }


def test_same_evm_address_on_two_chains_needs_the_chain() -> None:
    eth = NEWT_BASE.model_copy(update={"canonical_id": f"ethereum:{EVM}", "chain": "ethereum"})
    index = AttributionIndex([NEWT_BASE, eth])
    [a] = index.attribute(post(f"contract {EVM}"))
    assert a.level == "AMBIGUOUS"
    assert levels(index, post(f"contract {EVM} on base")) == {NEWT_BASE.canonical_id: "EXACT"}


def test_wrong_contract_is_rejected() -> None:
    index = AttributionIndex([NEWT], universe=[NEWT])
    [a] = index.attribute(post(f"$NEWT real contract: {MINT_3}"))
    assert (a.level, a.canonical_id) == ("REJECTED", NEWT.canonical_id)
    assert "different contract" in a.reason


def test_a_post_can_reference_several_tokens() -> None:
    index = AttributionIndex([NEWT, OTHER])
    got = levels(index, post(f"rotating from {MINT_2} into {MINT}"))
    assert got == {NEWT.canonical_id: "EXACT", OTHER.canonical_id: "EXACT"}


def test_untracked_tickers_are_ignored() -> None:
    assert AttributionIndex([NEWT]).attribute(post("$BTC and $ETH today")) == []


# --- Momentum state (pure) ------------------------------------------------------------------


def trend(recent: float, previous: float, baseline: float | None, authors: tuple[float, float] | None = None) -> WindowTrend:  # fmt: skip
    s = 1.0

    def m(r: float, p: float, b: float | None) -> MetricTrend:
        return MetricTrend(
            recent=r,
            previous=p,
            baseline=b,
            velocity_per_hour=r,
            previous_velocity_per_hour=p,
            acceleration_per_hour=r - p,
            acceleration_ratio=(r + s) / (p + s),
            baseline_ratio=(p + s) / (b + s) if b is not None else None,
        )

    ar, ap = authors or (recent, previous)
    return WindowTrend(
        window="h1",
        mentions=m(recent, previous, baseline),
        unique_authors=m(ar, ap, None),
        engagement=m(0, 0, None),
    )


GOOD_QUALITY = SocialQuality(spam_risk="low", organic_signal_strength="high")
ONE_SOURCE = CrossPlatformConfirmation(
    providers_configured=1,
    providers_checked=1,
    platforms_with_activity=1,
    platforms_accelerating=1,
    source_diversity=1.0,
    corroborated=False,
    single_source_available=True,
    only_one_active_of_several=False,
)


def state(t: WindowTrend | None, **kw: Any) -> str:
    return momentum_state(
        t,
        kw.get("checked", True),
        kw.get("quality", GOOD_QUALITY),
        kw.get("cross", ONE_SOURCE),
        SocialConfig(),
    )[0]


def test_growing_attention_is_strong_but_large_fading_attention_is_not() -> None:
    assert state(trend(400, 120, 50)) == "STRONG"  # 50 -> 120 -> 400
    assert state(trend(9_400, 9_800, 10_000)) == "FADING"  # 10,000 -> 9,800 -> 9,400


def test_absolute_popularity_is_not_momentum() -> None:
    assert state(trend(5_000, 4_900, 5_000)) == "SATURATED"
    assert state(trend(40, 38, 41)) == "STABLE"


def test_other_states() -> None:
    assert state(trend(1, 0, 0)) == "QUIET"
    assert state(trend(12, 0, 0)) == "EMERGING"
    assert state(trend(10, 30, None)) == "FADING"
    assert state(None) == "INSUFFICIENT_DATA"
    assert state(None, checked=False) == "UNAVAILABLE"


def test_unique_author_acceleration_is_required() -> None:
    assert state(trend(60, 20, 10, authors=(4, 4))) == "STABLE"  # same few people louder
    assert state(trend(60, 20, 10, authors=(40, 15))) == "STRONG"


def test_strong_needs_sustained_broad_clean_attention() -> None:
    assert state(trend(60, 20, 19)) == "ACCELERATING"  # not sustained over baseline
    assert state(trend(9, 3, 1)) == "ACCELERATING"  # too few authors
    spam = SocialQuality(spam_risk="high", organic_signal_strength="low")
    assert state(trend(400, 120, 50), quality=spam) == "ACCELERATING"
    lonely = ONE_SOURCE.model_copy(
        update={"single_source_available": False, "only_one_active_of_several": True}
    )
    assert state(trend(400, 120, 50), cross=lonely) == "ACCELERATING"


# --- Service fixtures -----------------------------------------------------------------------

WORDS = [f"w{i}" for i in range(400)]


def spread(n: int, start: datetime, end: datetime, **kw: Any) -> list[dict[str, Any]]:
    """`n` posts evenly inside (start, end], with distinct texts and authors by default."""
    out = []
    step = (end - start) / (n + 1)
    for i in range(n):
        rng = random.Random(f"{start.isoformat()}|{i}|{kw.get('seed', 0)}|{kw.get('prefix', 'p')}")
        filler = " ".join(rng.sample(WORDS, 8))
        text = kw.get("text", f"$NEWT {MINT} {filler}")
        out.append(
            {
                "id": f"{kw.get('prefix', 'p')}-{start.timestamp():.0f}-{i}",
                "created_at": start + step * (i + 1),
                "text": text,
                "author": kw.get("author") or f"u{kw.get('prefix', 'p')}{start.timestamp():.0f}{i}",
                "likes": kw.get("likes", 2),
                "replies": 1,
                "promoted": kw.get("promoted"),
            }
        )
    return out


def growth_posts(counts: tuple[int, int, int], **kw: Any) -> list[dict[str, Any]]:
    """baseline (per hour, 4 hours), previous hour, recent hour."""
    base, prev, recent = counts
    h = timedelta(hours=1)
    posts: list[dict[str, Any]] = []
    for k in range(4):
        start = NOW - (6 - k) * h
        posts += spread(base, start, start + h, **kw)
    posts += spread(prev, NOW - 2 * h, NOW - h, **kw)
    posts += spread(recent, NOW - h, NOW, **kw)
    return posts


def static(
    name: str, platform: str, posts: list[dict[str, Any]], now: Now, **kw: Any
) -> StaticSocialProvider:
    return StaticSocialProvider(name, platform, posts, now=now, **kw)


def social(
    tmp_path: Path, providers: list[Any], now: Now, config: SocialConfig | None = None
) -> SocialScoutService:
    return SocialScoutService(providers, SocialStore(tmp_path / "scout.sqlite3"), config, now=now)


def market(
    volume_ratio: float | None, liquidity_change: float | None = None
) -> ScoutGrowthFeatures:
    return ScoutGrowthFeatures(
        computed_at=NOW,
        window_acceleration=[
            WindowAcceleration(
                short="m15", long="h1", volume_rate_ratio=volume_ratio, txn_rate_ratio=volume_ratio
            )
        ],
        history=[
            HistoryComparison(
                lookback_minutes=60,
                compared_at=NOW - timedelta(hours=1),
                elapsed_minutes=60,
                same_pool=True,
                liquidity_change_pct=liquidity_change,
            )
        ],
    )


# --- Service: acceleration and states -------------------------------------------------------


def test_accelerating_mentions_become_strong(tmp_path: Path) -> None:
    now = Now()
    svc = social(tmp_path, [static("X fixture", "x", growth_posts((5, 12, 40)), now)], now)
    [m] = run(svc.observe([NEWT])).momentum
    assert m.state == "STRONG", m.reasons
    assert m.trend is not None
    assert (m.trend.mentions.baseline, m.trend.mentions.previous, m.trend.mentions.recent) == (
        5,
        12,
        40,
    )
    assert m.trend.unique_authors.acceleration_ratio > 3
    assert m.trend.mentions.velocity_per_hour == 40
    h1 = next(w for w in m.windows if w.window == "h1")
    assert h1.exact_mentions == 40 and h1.unique_authors == 40 and h1.engagement == 120
    assert [w.window for w in m.windows] == ["m5", "m15", "m30", "h1", "h6", "h24"]
    assert "never a BUY / SELL" in m.disclaimer


def test_high_but_fading_mentions(tmp_path: Path) -> None:
    now = Now()
    svc = social(tmp_path, [static("X fixture", "x", growth_posts((60, 55, 50)), now)], now)
    [m] = run(svc.observe([NEWT])).momentum
    assert m.state == "FADING", m.reasons


def test_uncovered_windows_are_missing_not_zero(tmp_path: Path) -> None:
    now = Now()
    # a search that hit its result cap only reaches back 30 minutes
    provider = static("X fixture", "x", growth_posts((5, 12, 40)), now, max_results=20)
    [m] = run(social(tmp_path, [provider], now).observe([NEWT])).momentum
    assert m.state == "INSUFFICIENT_DATA"
    assert {w.window for w in m.windows} <= {"m5", "m15"}
    assert m.trend is None


def test_ambiguous_mentions_never_count(tmp_path: Path) -> None:
    now = Now()
    posts = growth_posts((5, 12, 40), text="$NEWT is exploding")
    svc = social(tmp_path, [static("X fixture", "x", posts, now)], now)
    lookalike = NEWT.model_copy(update={"canonical_id": f"solana:{MINT_3}", "address": MINT_3})
    result = run(svc.observe([NEWT], universe=[lookalike]))
    [m] = result.momentum
    assert m.state == "QUIET"
    h1 = next(w for w in m.windows if w.window == "h1")
    assert h1.mentions == 0 and h1.ambiguous_posts == 40
    assert result.ambiguous_mentions > 0


def test_probable_mentions_are_down_weighted(tmp_path: Path) -> None:
    now = Now()
    posts = spread(10, NOW - timedelta(hours=1), NOW, text="$NEWT on solana today")
    svc = social(tmp_path, [static("X fixture", "x", posts, now)], now)
    [m] = run(svc.observe([NEWT])).momentum
    h1 = next(w for w in m.windows if w.window == "h1")
    assert h1.mentions == 5.0 and h1.attributable_posts == 10


# --- Quality --------------------------------------------------------------------------------


def test_one_author_spam(tmp_path: Path) -> None:
    now = Now()
    posts = growth_posts((1, 3, 30), author="shill")
    [m] = run(
        social(tmp_path, [static("X fixture", "x", posts, now)], now).observe([NEWT])
    ).momentum
    assert m.quality.spam_risk == "high"
    assert m.quality.top_author_share == 1.0
    assert m.state != "STRONG"
    assert any("one author" in r for r in m.quality.reasons)


def test_repeated_content_spam(tmp_path: Path) -> None:
    now = Now()
    posts = growth_posts((2, 5, 30), text=f"$NEWT {MINT} to the moon, buy before it is too late")
    [m] = run(
        social(tmp_path, [static("X fixture", "x", posts, now)], now).observe([NEWT])
    ).momentum
    assert m.quality.spam_risk == "high"
    assert m.quality.duplicate_share == 1.0
    assert m.state == "ACCELERATING" and "spam risk is high" in " ".join(m.reasons)


def test_repeated_contract_promotion_and_conflicting_contracts(tmp_path: Path) -> None:
    now = Now()
    h = timedelta(hours=1)
    posts = spread(12, NOW - h, NOW, seed=1)
    posts += spread(6, NOW - h, NOW, author="promoter", prefix="r", seed=2)
    posts += spread(3, NOW - h, NOW, text=f"$NEWT real one is {MINT_3}", prefix="c")
    svc = social(tmp_path, [static("X fixture", "x", posts, now)], now)
    [m] = run(svc.observe([NEWT], universe=[NEWT])).momentum
    assert m.quality.repeated_contract_share == pytest.approx(6 / 18)
    assert m.quality.conflicting_contracts == 3
    assert m.quality.spam_risk in ("medium", "high")


def test_paid_promotion_is_separated(tmp_path: Path) -> None:
    now = Now()
    posts = spread(8, NOW - timedelta(hours=1), NOW) + spread(
        8, NOW - timedelta(hours=1), NOW, promoted=True, prefix="ad"
    )
    [m] = run(
        social(tmp_path, [static("X fixture", "x", posts, now)], now).observe([NEWT])
    ).momentum
    h1 = next(w for w in m.windows if w.window == "h1")
    assert h1.mentions == 8 and h1.promoted_posts == 8
    assert m.quality.promoted_share == 0.5
    assert m.quality.spam_risk == "high"


def test_too_few_posts_is_unknown_quality(tmp_path: Path) -> None:
    now = Now()
    posts = spread(2, NOW - timedelta(hours=1), NOW)
    [m] = run(
        social(tmp_path, [static("X fixture", "x", posts, now)], now).observe([NEWT])
    ).momentum
    assert (m.quality.spam_risk, m.quality.organic_signal_strength) == ("unknown", "unknown")


# --- Cross-platform -------------------------------------------------------------------------


def test_cross_platform_confirmation(tmp_path: Path) -> None:
    now = Now()
    providers = [
        static("X fixture", "x", growth_posts((5, 12, 40), prefix="x"), now),
        static("Reddit fixture", "reddit", growth_posts((3, 8, 25), prefix="r"), now),
        static("Forum fixture", "forum:f", growth_posts((0, 1, 6), prefix="f"), now),
    ]
    [m] = run(social(tmp_path, providers, now).observe([NEWT])).momentum
    c = m.cross_platform
    assert c.providers_configured == 3 and c.providers_checked == 3
    assert c.platforms_with_activity == 3 and c.platforms_accelerating == 3
    assert c.corroborated and c.source_diversity == 1.0
    assert m.state == "STRONG"
    assert m.trend is not None and m.trend.mentions.recent == 71  # summed across platforms


def test_one_provider_configured_is_not_a_penalty(tmp_path: Path) -> None:
    now = Now()
    providers = [
        static("X fixture", "x", growth_posts((5, 12, 40)), now),
        static("Reddit fixture", "reddit", [], now, configured=False),
    ]
    [m] = run(social(tmp_path, providers, now).observe([NEWT])).momentum
    assert m.cross_platform.single_source_available
    assert not m.cross_platform.only_one_active_of_several
    assert m.state == "STRONG"


def test_several_providers_checked_only_one_active(tmp_path: Path) -> None:
    now = Now()
    providers = [
        static("X fixture", "x", growth_posts((5, 12, 40)), now),
        static("Reddit fixture", "reddit", [], now),
        static("Farcaster fixture", "farcaster", [], now),
    ]
    [m] = run(social(tmp_path, providers, now).observe([NEWT])).momentum
    c = m.cross_platform
    assert c.providers_checked == 3 and c.platforms_with_activity == 1
    assert c.only_one_active_of_several and not c.single_source_available
    assert m.state == "ACCELERATING"
    assert "activity on only one" in " ".join(m.reasons)


# --- Provider availability ------------------------------------------------------------------


def test_unavailable_vs_checked_zero_vs_not_configured(tmp_path: Path) -> None:
    now = Now()
    providers = [
        static("Down", "x", [], now, fail=MarketDataUnavailableError("Down returned HTTP 503")),
        static("Quiet", "reddit", [], now),
        static("Locked", "farcaster", [], now, configured=False),
    ]
    result = run(social(tmp_path, providers, now).observe([NEWT]))
    statuses = {s.provider: s.status for s in result.momentum[0].sources}
    assert statuses == {
        "Down": "PROVIDER_UNAVAILABLE",
        "Quiet": "PROVIDER_CHECKED_ZERO_MATCHES",
        "Locked": "PROVIDER_NOT_CONFIGURED",
    }
    by_provider = {c.provider: c for c in result.providers}
    assert by_provider["Down"].error == "Down returned HTTP 503"
    assert by_provider["Locked"].requirement
    [m] = result.momentum
    assert m.state == "QUIET"  # the quiet provider really checked: zero is a real zero
    quiet = next(s for s in m.sources if s.provider == "Quiet")
    assert quiet.windows and all(w.mentions == 0 for w in quiet.windows)
    down = next(s for s in m.sources if s.provider == "Down")
    assert down.windows == []  # never counted as zero activity


def test_no_provider_available_is_unavailable(tmp_path: Path) -> None:
    now = Now()
    providers = [
        static("Down", "x", [], now, fail=MarketDataUnavailableError("boom")),
        static("Locked", "reddit", [], now, configured=False),
    ]
    [m] = run(social(tmp_path, providers, now).observe([NEWT])).momentum
    assert m.state == "UNAVAILABLE"
    assert social(tmp_path, [], now).providers == []
    [m2] = run(social(tmp_path, [], now).observe([NEWT])).momentum
    assert m2.state == "UNAVAILABLE" and m2.cross_platform.providers_configured == 0


def test_partial_provider_failure_keeps_the_rest(tmp_path: Path) -> None:
    now = Now()
    providers = [
        static("X fixture", "x", growth_posts((5, 12, 40)), now),
        static("Down", "reddit", [], now, fail=MarketDataUnavailableError("timeout")),
    ]
    result = run(social(tmp_path, providers, now).observe([NEWT]))
    [m] = result.momentum
    assert m.state == "STRONG"
    assert m.cross_platform.providers_checked == 1


def test_rate_limited_batches_are_reported_unavailable(tmp_path: Path) -> None:
    now = Now()
    gate = RequestGate("X fixture", ScoutProviderLimits(calls_per_minute=1), now.mono)
    provider = static(
        "X fixture", "x", growth_posts((5, 12, 40)), now, gate=gate, max_terms_per_query=2
    )
    result = run(social(tmp_path, [provider], now).observe([NEWT, OTHER]))
    statuses = {m.canonical_id: m.sources[0].status for m in result.momentum}
    assert statuses == {
        NEWT.canonical_id: "PROVIDER_OK",
        OTHER.canonical_id: "PROVIDER_UNAVAILABLE",  # its batch was over budget
    }
    errors = [s.error for m in result.momentum for s in m.sources if s.error]
    assert errors and "request limit" in errors[0]


# --- Market cross-check ---------------------------------------------------------------------


def test_social_spike_without_market_confirmation(tmp_path: Path) -> None:
    now = Now()
    svc = social(tmp_path, [static("X fixture", "x", growth_posts((5, 12, 40)), now)], now)
    [m] = run(svc.observe([NEWT], market={NEWT.canonical_id: market(1.0)})).momentum
    assert m.market.state == "UNCONFIRMED_SOCIAL_SPIKE"
    assert "(flat)" in m.market.reasons[0]
    assert any("without matching market activity" in r for r in m.quality.reasons)


def test_market_and_social_confirmation(tmp_path: Path) -> None:
    now = Now()
    svc = social(tmp_path, [static("X fixture", "x", growth_posts((5, 12, 40)), now)], now)
    [m] = run(svc.observe([NEWT], market={NEWT.canonical_id: market(2.4, 5.0)})).momentum
    assert m.market.state == "CORROBORATED"
    assert m.market.market_window == "m15/h1" and m.market.volume_rate_ratio == 2.4
    assert m.state == "STRONG"


def test_attention_rising_while_liquidity_drains(tmp_path: Path) -> None:
    now = Now()
    svc = social(tmp_path, [static("X fixture", "x", growth_posts((5, 12, 40)), now)], now)
    [m] = run(svc.observe([NEWT], market={NEWT.canonical_id: market(2.0, -40.0)})).momentum
    assert m.market.state == "CAUTION_LIQUIDITY_FALLING"


def test_market_without_social_and_missing_market(tmp_path: Path) -> None:
    now = Now()
    svc = social(tmp_path, [static("X fixture", "x", [], now)], now)
    [m] = run(svc.observe([NEWT], market={NEWT.canonical_id: market(3.0)})).momentum
    assert m.market.state == "MARKET_WITHOUT_SOCIAL"
    [m2] = run(svc.observe([NEWT])).momentum
    assert m2.market.state == "INSUFFICIENT_DATA"


# --- Storage, history, privacy --------------------------------------------------------------


def test_snapshots_persist_and_history_is_never_rewritten(tmp_path: Path) -> None:
    now = Now()
    posts = growth_posts((5, 12, 40))
    provider = static("X fixture", "x", posts, now)
    svc = social(tmp_path, [provider], now)
    run(svc.observe([NEWT]))

    # the same posts come back later with more engagement: stored values are kept
    for p in posts:
        p["likes"] = 999
    now.at += timedelta(minutes=10)
    run(svc.observe([NEWT]))
    svc.store.close()

    store = SocialStore(tmp_path / "scout.sqlite3")
    events = run(store.events(NEWT.canonical_id, NOW - timedelta(days=2), NOW))
    assert len(events) == len(posts)  # stored once each
    assert {e.likes for e in events} == {2}
    assert len(run(store.snapshots(NEWT.canonical_id))) == 2  # appended, not replaced
    history = run(store.momentum_history(NEWT.canonical_id))
    assert [h.computed_at for h in history] == [NOW, NOW + timedelta(minutes=10)]


def test_incremental_searches_start_where_the_last_one_ended(tmp_path: Path) -> None:
    now = Now()
    seen: list[datetime] = []
    provider = static("X fixture", "x", [], now)
    original = provider.search

    async def spy(terms: Any, since: datetime, keyer: Any) -> Any:
        seen.append(since)
        return await original(terms, since, keyer)

    provider.search = spy  # type: ignore[method-assign]
    svc = social(tmp_path, [provider], now)
    run(svc.observe([NEWT]))
    now.at += timedelta(minutes=30)
    run(svc.observe([NEWT]))
    assert seen == [NOW - timedelta(hours=48), NOW - timedelta(minutes=5)]


def test_no_text_or_handles_are_stored(tmp_path: Path) -> None:
    now = Now()
    posts = spread(6, NOW - timedelta(hours=1), NOW, text=f"secret words $NEWT {MINT}")
    for p in posts:
        p["handle"] = "RealPerson"
        p["author"] = "real-person-id"
    run(social(tmp_path, [static("X fixture", "x", posts, now)], now).observe([NEWT]))
    import sqlite3

    dump = "\n".join(sqlite3.connect(tmp_path / "scout.sqlite3").iterdump())
    assert "secret words" not in dump
    assert "realperson" not in dump.lower() and "real-person-id" not in dump


def test_old_events_are_pruned(tmp_path: Path) -> None:
    now = Now()
    posts = spread(3, NOW - timedelta(hours=1), NOW)
    svc = social(tmp_path, [static("X fixture", "x", posts, now)], now)
    run(svc.observe([NEWT]))
    now.at += timedelta(hours=100)
    run(svc.observe([NEWT]))
    assert run(svc.store.events(NEWT.canonical_id, NOW - timedelta(days=30), now.at)) == []


# --- Performance: batching, caching, dedup --------------------------------------------------


def test_terms_are_batched_into_few_queries() -> None:
    provider = StaticSocialProvider("P", "x", max_terms_per_query=4, max_query_chars=1000)
    tokens = {f"t{i}": [f"addr{i}", f"$T{i}"] for i in range(5)}
    plans = plan_queries(provider, tokens)
    assert [ids for ids, _ in plans] == [["t0", "t1"], ["t2", "t3"], ["t4"]]
    long_one = {"t": ["x" * 995, "$T"]}
    assert plan_queries(provider, long_one) == [(["t"], ["x" * 995]), (["t"], ["$T"])]


def test_one_query_serves_many_tokens(tmp_path: Path) -> None:
    now = Now()
    provider = static("X fixture", "x", growth_posts((1, 2, 5)), now)
    run(social(tmp_path, [provider], now).observe([NEWT, OTHER]))
    assert len(provider.queries) == 1


def test_searches_are_cached_and_concurrent_ones_deduplicated() -> None:
    now = Now()
    gate = RequestGate(
        "P", ScoutProviderLimits(calls_per_minute=100, cache_ttl_seconds=60), now.mono
    )
    provider = static("P", "x", spread(3, NOW - timedelta(hours=1), NOW), now, gate=gate)

    async def main() -> None:
        since = NOW - timedelta(hours=2)
        await asyncio.gather(*(provider.search([MINT], since, lambda p, a: a) for _ in range(4)))
        await provider.search([MINT], since, lambda p, a: a)

    run(main())
    assert len(provider.queries) == 1


# --- HTTP provider adapters (MockTransport) -------------------------------------------------


def settings(**kw: Any) -> Any:
    from upscale.services.scout.social.config import SocialProviderConfig

    return SocialProviderConfig(limits=ScoutProviderLimits(calls_per_minute=100), **kw)


def keyer(platform: str, author: str) -> str:
    return f"k:{platform}:{author}"


def test_reddit_adapter_authenticates_and_normalizes() -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.path == "/api/v1/access_token":
            return httpx2.Response(200, json={"access_token": "tok", "expires_in": 3600})
        return httpx2.Response(
            200,
            json={
                "data": {
                    "after": None,
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "name": "t3_abc",
                                "title": f"$NEWT {MINT}",
                                "selftext": "",
                                "author": "someone",
                                "created_utc": (NOW - timedelta(minutes=5)).timestamp(),
                                "score": 12,
                                "num_comments": 3,
                                "permalink": "/r/x/comments/abc/",
                                "url": "https://example.org",
                            },
                        },
                        {"kind": "t3", "data": {"name": "t3_del", "author": "[deleted]"}},
                    ],
                }
            },
        )

    provider = RedditProvider(
        "id",
        "secret",
        "UpScale/1.0",
        settings(),
        transport=httpx2.MockTransport(handle),
        now=lambda: NOW,
    )
    assert provider.configured
    result = run(provider.search([MINT, "$NEWT"], NOW - timedelta(hours=1), keyer))
    [p] = result.posts
    assert (p.post_id, p.likes, p.replies, p.author_key) == ("t3_abc", 12, 3, "k:reddit:someone")
    assert p.source_url == "https://www.reddit.com/r/x/comments/abc/"
    search = requests[-1]
    assert search.headers["authorization"] == "Bearer tok"
    assert search.url.params["q"] == f'"{MINT}" OR "NEWT"'
    assert result.complete_since == NOW - timedelta(hours=1)
    assert not RedditProvider(None, None, None, settings()).configured


def test_neynar_adapter_normalizes_casts() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["x-api-key"] == "key"
        return httpx2.Response(
            200,
            json={
                "result": {
                    "casts": [
                        {
                            "hash": "0xcast",
                            "author": {"fid": 42, "username": "Dev"},
                            "text": "$NEWT",
                            "timestamp": (NOW - timedelta(minutes=1)).isoformat(),
                            "reactions": {"likes_count": 5, "recasts_count": 2},
                            "replies": {"count": 1},
                            "embeds": [{"url": "https://newt.example"}],
                        },
                        {"hash": "broken"},
                    ],
                    "next": {"cursor": None},
                }
            },
        )

    provider = NeynarFarcasterProvider(
        "key", settings(), transport=httpx2.MockTransport(handle), now=lambda: NOW
    )
    [p] = run(provider.search(["$NEWT"], NOW - timedelta(hours=1), keyer)).posts
    assert (p.author_key, p.author_handle, p.engagement) == ("k:farcaster:42", "dev", 8)
    assert p.urls == ["https://newt.example"]
    assert not NeynarFarcasterProvider(None, settings()).configured


def test_x_adapter_requires_paid_opt_in_and_respects_the_read_budget() -> None:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        rows = [
            {
                "id": str(i),
                "text": "$NEWT",
                "author_id": "7",
                "created_at": (NOW - timedelta(minutes=i + 1)).isoformat().replace("+00:00", "Z"),
                "public_metrics": {
                    "like_count": 1,
                    "reply_count": 0,
                    "retweet_count": 2,
                    "quote_count": 0,
                    "impression_count": 50,
                },
            }
            for i in range(int(request.url.params["max_results"]))
        ]
        return httpx2.Response(
            200,
            json={
                "data": rows,
                "includes": {"users": [{"id": "7", "username": "Poster"}]},
                "meta": {"next_token": "n"},
            },
        )

    assert not XRecentSearchProvider("token", False, 500, settings()).configured
    provider = XRecentSearchProvider(
        "token",
        True,
        150,
        settings(max_pages=5),
        transport=httpx2.MockTransport(handle),
        now=lambda: NOW,
        fetch_usernames=True,
    )
    result = run(provider.search(["$NEWT"], NOW - timedelta(hours=1), keyer))
    assert provider.reads_this_run == 150 and len(calls) == 2
    assert result.complete_since > NOW - timedelta(hours=1)  # truncated: never assumed complete
    assert calls[0].url.params["query"] == "($NEWT) -is:retweet"
    assert result.posts[0].views == 50 and result.posts[0].author_handle == "poster"


def test_discourse_adapter_reads_public_search() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/search.json"
        assert "order:latest" in request.url.params["q"]
        return httpx2.Response(
            200,
            json={
                "posts": [
                    {"id": 9, "username": "mod", "created_at": (NOW - timedelta(minutes=3)).isoformat(),
                     "blurb": f"contract {MINT}", "like_count": 4, "topic_id": 5, "post_number": 2},
                ],
                "topics": [{"id": 5, "title": "New listings"}],
            },
        )  # fmt: skip

    provider = DiscourseForumProvider(
        "https://forum.example.org",
        settings(),
        transport=httpx2.MockTransport(handle),
        now=lambda: NOW,
    )
    assert provider.platform == "forum:forum.example.org"
    [p] = run(provider.search([MINT], NOW - timedelta(hours=1), keyer)).posts
    assert p.source_url == "https://forum.example.org/t/5/2" and MINT in p.text


@pytest.mark.parametrize("status", [429, 401, 500])
def test_adapter_http_failures_are_provider_errors(status: int) -> None:
    provider = NeynarFarcasterProvider(
        "key",
        settings(),
        transport=httpx2.MockTransport(lambda r: httpx2.Response(status)),
        now=lambda: NOW,
    )
    with pytest.raises(MarketDataUnavailableError):
        run(provider.search(["$NEWT"], NOW - timedelta(hours=1), keyer))


def test_malformed_adapter_response_is_an_error() -> None:
    provider = NeynarFarcasterProvider(
        "key",
        settings(),
        transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json={"x": 1})),
        now=lambda: NOW,
    )
    with pytest.raises(MarketDataUnavailableError, match="unexpected"):
        run(provider.search(["$NEWT"], NOW - timedelta(hours=1), keyer))


# --- Config and wiring ----------------------------------------------------------------------


def test_social_config_is_validated() -> None:
    with pytest.raises(ValidationError):
        SocialConfig(momentum={"accelerating_ratio": 0.5})
    with pytest.raises(ValidationError):
        SocialConfig(momentum={"state_window": "h6", "windows": ["h1"]})
    with pytest.raises(ValidationError):
        SocialConfig(search_span_hours=1.0, momentum={"state_window": "h6"})
    with pytest.raises(SocialConfigError):
        load_social_config("[1, 2")
    cfg = load_social_config(json.dumps({"momentum": {"min_mentions": 5}}))
    assert cfg.momentum.min_mentions == 5


def test_social_is_wired_but_not_configured_without_credentials() -> None:
    from upscale.config import NEYNAR_API_KEY
    from upscale.services import scout_service, social_scout_service

    names = {p.name: p.configured for p in social_scout_service.providers}
    # Farcaster follows the local environment (a developer may have a Neynar key in .env).
    assert names == {"Reddit": False, "Farcaster (Neynar)": bool(NEYNAR_API_KEY), "X": False}
    assert social_scout_service.store._conn is None
    assert all("social" not in type(p).__name__.lower() for p in scout_service.providers)


# --- Neynar: live-validated behavior (sanitized real response shape) -------------------------

# A real Neynar cast-search response (2026-09), structure unchanged; every identity, text
# and handle replaced by synthetic values reproducing the patterns seen live.
NEYNAR_FIXTURE = Path(__file__).parent / "fixtures" / "neynar_cast_search.json"
DEGEN_ADDRESS = "0x4ed4e862860bed51a9570b96d89af5e1b0efefed"
DEGEN = TokenIdentity(
    canonical_id=f"base:{DEGEN_ADDRESS}",
    chain="base",
    address=DEGEN_ADDRESS,
    symbol="DEGEN",
    name="Degen",
)
FIXTURE_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def neynar_fixture() -> dict[str, Any]:
    body: dict[str, Any] = json.loads(NEYNAR_FIXTURE.read_text())
    return body


def neynar(handle: Any, now: Any = lambda: FIXTURE_NOW, **kw: Any) -> NeynarFarcasterProvider:
    return NeynarFarcasterProvider(
        "key", settings(**kw), transport=httpx2.MockTransport(handle), now=now
    )


def test_neynar_parses_the_real_response_shape() -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=neynar_fixture())

    since = FIXTURE_NOW - timedelta(hours=2)
    result = run(neynar(handle).search(["$DEGEN"], since, keyer))
    assert len(result.posts) == 7
    params = requests[0].url.params
    # bare cashtag (quoting drops the `$`), server-side bound in UTC without a `Z`
    assert params["q"] == "$DEGEN after:2026-09-26T10:00:00"
    assert (params["mode"], params["sort_type"], params["limit"]) == (
        "literal",
        "desc_chron",
        "100",
    )
    by_id = {p.post_id[-1]: p for p in result.posts}
    p3 = by_id["3"]
    assert (p3.likes, p3.reposts, p3.replies, p3.engagement) == (4, 1, 2, 7)
    assert p3.author_key == "k:farcaster:103" and p3.author_handle == "user103"
    assert p3.author_quality == 0.83 and p3.source_url is None
    assert by_id["4"].urls == [f"https://dexscreener.com/base/{DEGEN_ADDRESS}"]
    assert by_id["7"].author_quality is None  # neither score nor experimental score given
    # one page with a next cursor: complete only back to the oldest cast, never assumed
    assert result.complete_since == min(p.created_at for p in result.posts)


def test_neynar_author_score_falls_back_to_the_experimental_field() -> None:
    body = neynar_fixture()
    cast = body["result"]["casts"][0]
    del cast["author"]["score"]
    cast["author"]["experimental"]["neynar_user_score"] = 0.7
    body["result"]["casts"] = [cast, {**cast, "hash": "0xbad", "author": {"fid": True}}]
    result = run(
        neynar(lambda r: httpx2.Response(200, json=body)).search(
            ["$DEGEN"], FIXTURE_NOW - timedelta(hours=2), keyer
        )
    )
    [p] = result.posts  # a boolean fid is not an author id
    assert p.author_quality == 0.7


def test_neynar_real_shape_attribution() -> None:
    result = run(
        neynar(lambda r: httpx2.Response(200, json=neynar_fixture())).search(
            ["$DEGEN"], FIXTURE_NOW - timedelta(hours=2), keyer
        )
    )
    by_id = {p.post_id[-1]: p for p in result.posts}
    directory = AttributionIndex([DEGEN], universe=[DEGEN, OTHER])
    assert levels(directory, by_id["3"]) == {DEGEN.canonical_id: "EXACT"}  # checksummed
    assert levels(directory, by_id["4"]) == {DEGEN.canonical_id: "EXACT"}  # in a link only
    assert levels(directory, by_id["5"]) == {DEGEN.canonical_id: "REJECTED"}
    assert levels(directory, by_id["6"]) == {}  # "degens" is not the cashtag
    for index in (directory, AttributionIndex([DEGEN])):  # ticker only: never attached
        [a] = index.attribute(by_id["1"])
        assert a.level == "AMBIGUOUS" and a.canonical_id is None
    [c] = directory.attribute(by_id["3"])
    assert c.level == "EXACT"


def test_neynar_queries_hold_one_bare_term() -> None:
    provider = NeynarFarcasterProvider("key", settings(max_terms_per_query=10))
    assert provider.max_terms_per_query == 1  # combined terms lose mentions on Neynar
    plans = plan_queries(provider, {"a": [MINT, "$NEWT"], "b": [EVM, "$OTH"]})
    assert [terms for _, terms in plans] == [[MINT], ["$NEWT"], [EVM], ["$OTH"]]
    assert provider.render_query(["$NEWT"]) == "$NEWT"
    assert provider.render_query(['$a"b | c']) == '"a b c"'  # no operators smuggled in


def test_neynar_after_bound_is_utc_without_z() -> None:
    seen: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.url.params["q"])
        return httpx2.Response(200, json={"result": {"casts": [], "next": {"cursor": None}}})

    eastern = timezone(timedelta(hours=-4))
    since = datetime(2026, 9, 26, 1, 30, 15, 999, tzinfo=eastern)
    result = run(neynar(handle).search([MINT], since, keyer))
    assert seen == [f"{MINT} after:2026-09-26T05:30:15"]
    assert result.posts == [] and result.complete_since == since  # a real zero


def test_neynar_out_of_credits_is_reported() -> None:
    provider = neynar(lambda r: httpx2.Response(402))
    with pytest.raises(MarketDataUnavailableError, match="credits are exhausted"):
        run(provider.search(["$NEWT"], NOW - timedelta(hours=1), keyer))


def test_request_budget_per_run(tmp_path: Path) -> None:
    now = Now()
    empty = {"result": {"casts": [], "next": {"cursor": None}}}
    provider = neynar(lambda r: httpx2.Response(200, json=empty), now=now, max_requests_per_run=2)
    svc = social(tmp_path, [provider], now)
    result = run(svc.observe([NEWT, OTHER]))
    statuses = {m.canonical_id: m.sources[0].status for m in result.momentum}
    assert statuses == {
        NEWT.canonical_id: "PROVIDER_CHECKED_ZERO_MATCHES",
        OTHER.canonical_id: "PROVIDER_UNAVAILABLE",
    }
    assert "request budget" in (result.providers[0].error or "")
    assert provider.requests_this_run == 2
    now.at += timedelta(minutes=10)
    run(svc.observe([NEWT, OTHER]))
    assert provider.requests_this_run == 2  # a new run gets a new budget


# --- Service: live-found regressions --------------------------------------------------------


class MovingNow(Now):
    """A live-like clock: every reading is a little later than the previous one."""

    def __call__(self) -> datetime:
        self.at += timedelta(milliseconds=3)
        return self.at


def test_windows_exist_with_a_moving_clock(tmp_path: Path) -> None:
    now = MovingNow()
    posts = spread(4, NOW - timedelta(minutes=50), NOW - timedelta(minutes=1))
    [m] = run(
        social(tmp_path, [static("X fixture", "x", posts, now)], now).observe([NEWT])
    ).momentum
    windows = {w.window: w for w in m.windows}
    assert windows["h1"].mentions == 4  # coverage ends after the run's start: covered
    assert m.state != "UNAVAILABLE"


class SplitProvider(StaticSocialProvider):
    """One term per query; queries for `failing` terms raise."""

    def __init__(self, posts: list[dict[str, Any]], now: Now, failing: set[str]):
        super().__init__("Split", "farcaster", posts, now=now, max_terms_per_query=1)
        self.failing = failing

    async def search(self, terms: Any, since: datetime, keyer: Any) -> Any:
        if set(terms) & self.failing:
            raise MarketDataUnavailableError("Split returned HTTP 503")
        return await super().search(terms, since, keyer)


def test_a_token_split_over_queries_is_judged_on_all_of_them(tmp_path: Path) -> None:
    now = Now()
    posts = spread(3, NOW - timedelta(minutes=50), NOW, text=f"contract {MINT}")
    posts += [{**posts[0], "text": f"$NEWT {MINT}"}]  # also found by the ticker query
    [m] = run(social(tmp_path, [SplitProvider(posts, now, set())], now).observe([NEWT])).momentum
    assert m.sources[0].status == "PROVIDER_OK"  # the ticker query's zero doesn't overwrite
    store = SocialStore(tmp_path / "scout.sqlite3")
    [check] = run(store.checks(NEWT.canonical_id, NOW - timedelta(days=1)))
    assert check.status == "PROVIDER_OK"  # one check per token, not one per query


def test_a_partially_failed_token_is_unavailable_not_zero(tmp_path: Path) -> None:
    now = Now()
    posts = spread(3, NOW - timedelta(minutes=50), NOW, text=f"contract {MINT}")
    provider = SplitProvider(posts, now, {"$NEWT"})
    svc = social(tmp_path, [provider], now)
    [m] = run(svc.observe([NEWT])).momentum
    assert m.sources[0].status == "PROVIDER_UNAVAILABLE"
    assert m.sources[0].windows == []
    assert run(svc.store.last_covered_to(NEWT.canonical_id, "Split")) is None


def test_terms_only_include_real_cashtags() -> None:
    index = AttributionIndex([])
    odd = [NEWT.model_copy(update={"symbol": s}) for s in ("GTA 7VII", "d/acc", "哭哭牛", "1INCH")]
    assert all(index.terms_for(t) == [MINT] for t in odd)
    assert index.terms_for(NEWT) == [MINT, "$NEWT"]


def test_a_name_equal_to_the_ticker_is_not_extra_evidence() -> None:
    index = AttributionIndex([DEGEN])  # no directory
    [a] = index.attribute(post("$DEGEN is up today"))
    assert a.level == "AMBIGUOUS"
    [b] = index.attribute(post("$DEGEN on base is up today"))
    assert b.level == "PROBABLE"  # the chain rules others out; not STRONG (no real name)


def test_provider_author_scores_are_separate_evidence(tmp_path: Path) -> None:
    now = Now()

    def momentum(score: float, folder: str) -> Any:
        posts = spread(8, NOW - timedelta(minutes=50), NOW)
        for p in posts:
            p["author_quality"] = score
        provider = static("X fixture", "x", posts, now)
        (tmp_path / folder).mkdir()
        [m] = run(social(tmp_path / folder, [provider], now).observe([NEWT])).momentum
        return m

    low, high = momentum(0.1, "low"), momentum(0.95, "high")
    assert low.quality.provider_low_quality_share == 1.0
    assert high.quality.provider_low_quality_share == 0.0
    assert low.quality.provider_median_author_quality == 0.1
    # UpScale's own verdict never follows the provider's score
    assert (low.quality.spam_risk, low.quality.organic_signal_strength, low.state) == (
        high.quality.spam_risk,
        high.quality.organic_signal_strength,
        high.state,
    )


def test_store_migrates_schema_v1(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "scout.sqlite3"
    run(SocialStore(path).author_salt())
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE scout_social_events DROP COLUMN author_quality")
    db.execute("UPDATE scout_meta SET value = '1' WHERE key = 'social_schema_version'")
    db.commit()
    db.close()
    now = Now()
    posts = spread(2, NOW - timedelta(minutes=30), NOW)
    posts[0]["author_quality"] = 0.4
    run(social(tmp_path, [static("X fixture", "x", posts, now)], now).observe([NEWT]))
    store = SocialStore(path)
    events = run(store.events(NEWT.canonical_id, NOW - timedelta(hours=1), NOW))
    assert sorted(e.author_quality or 0 for e in events) == [0, 0.4]
    version = (
        sqlite3.connect(path)
        .execute("SELECT value FROM scout_meta WHERE key = 'social_schema_version'")
        .fetchone()
    )
    assert version == ("2",)


# --- X: live-validated behavior (sanitized real response shape) ------------------------------

# A real X recent-search response (2026-09), keys unchanged (incl. a `note_tweet` long
# post); every id, handle and text replaced by synthetic values reproducing live patterns.
X_FIXTURE = Path(__file__).parent / "fixtures" / "x_recent_search.json"


def x_fixture() -> dict[str, Any]:
    body: dict[str, Any] = json.loads(X_FIXTURE.read_text())
    return body


def x_provider(
    handle: Any, now: Any = lambda: FIXTURE_NOW, reads: int = 500, **kw: Any
) -> XRecentSearchProvider:
    fetch_usernames = kw.pop("fetch_usernames", False)
    return XRecentSearchProvider(
        "token",
        True,
        reads,
        settings(**kw),
        transport=httpx2.MockTransport(handle),
        now=now,
        fetch_usernames=fetch_usernames,
    )


def test_x_parses_the_real_response_shape() -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=x_fixture())

    since = FIXTURE_NOW - timedelta(hours=2)
    result = run(x_provider(handle).search([DEGEN_ADDRESS], since, keyer))
    params = requests[0].url.params
    assert params["query"] == f"({DEGEN_ADDRESS}) -is:retweet"
    assert params["start_time"] == "2026-09-26T10:00:00Z"
    assert "note_tweet" in params["tweet.fields"]
    assert "expansions" not in params  # author usernames are a billed user read
    assert len(result.posts) == 7  # the post without an author id is dropped
    by_id = {p.post_id[-1]: p for p in result.posts}
    p1 = by_id["1"]
    assert (p1.likes, p1.replies, p1.reposts, p1.quotes, p1.views) == (12, 3, 4, 1, 900)
    assert p1.engagement == 20 and p1.author_key == "k:x:501" and p1.author_handle is None
    assert p1.source_url == f"https://x.com/i/web/status/{p1.post_id}"
    assert by_id["2"].urls == [f"https://dexscreener.com/base/{DEGEN_ADDRESS}"]
    assert DEGEN_ADDRESS in by_id["7"].text.lower()  # only in the long post's note
    assert result.complete_since == min(p.created_at for p in result.posts)  # next_token


def test_x_real_shape_attribution() -> None:
    result = run(
        x_provider(lambda r: httpx2.Response(200, json=x_fixture())).search(
            [DEGEN_ADDRESS], FIXTURE_NOW - timedelta(hours=2), keyer
        )
    )
    by_id = {p.post_id[-1]: p for p in result.posts}
    index = AttributionIndex([DEGEN], universe=[DEGEN, OTHER])
    got = {i: levels(index, by_id[i]) for i in "1234567"}
    assert got["1"] == got["2"] == got["7"] == {DEGEN.canonical_id: "EXACT"}
    assert got["3"] == got["4"] == {DEGEN.canonical_id: "PROBABLE"}  # ticker + "on base"
    assert got["5"] == {None: "AMBIGUOUS"}  # ticker alone
    assert got["6"] == {DEGEN.canonical_id: "REJECTED"}  # another contract


def test_x_search_terms_only_buy_countable_posts() -> None:
    x = XRecentSearchProvider("t", True, 10, settings())
    index = AttributionIndex([])

    def query(symbol: str, name: str, chain: str = "base") -> str:
        t = TokenIdentity(
            canonical_id=f"{chain}:a", chain=chain, address=EVM, symbol=symbol, name=name
        )
        return x.render_query(x.search_terms(index.references(t)))

    assert query("AERO", "Aerodrome") == (
        f'({EVM} OR ($AERO (Aerodrome OR "base chain" OR "on base"))) -is:retweet'
    )
    assert query("EUNICE", "Eunice") == (  # the name is the ticker again: chain only
        f'({EVM} OR ($EUNICE ("base chain" OR "on base"))) -is:retweet'
    )
    assert query("X", "X100") == f"({EVM} OR ($X (X100))) -is:retweet"  # short: name only
    assert query("GO", "go") == f"({EVM}) -is:retweet"  # nothing could corroborate it
    assert query("NEWT", 'Newt "Pro" | x', "solana").startswith(
        f'({EVM} OR ($NEWT ("Newt Pro x" OR '
    )


def test_x_fetches_usernames_only_when_asked() -> None:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"meta": {"result_count": 0}})

    run(x_provider(handle, fetch_usernames=True).search([EVM], FIXTURE_NOW, keyer))
    assert seen[0].url.params["expansions"] == "author_id"


def test_x_zero_results_are_a_real_zero() -> None:
    since = FIXTURE_NOW - timedelta(hours=1)
    result = run(
        x_provider(lambda r: httpx2.Response(200, json={"meta": {"result_count": 0}})).search(
            [EVM], since, keyer
        )
    )
    assert result.posts == [] and result.complete_since == since


def test_x_budget_too_small_for_a_page_is_unavailable_not_zero() -> None:
    provider = x_provider(lambda r: httpx2.Response(200, json=x_fixture()), reads=5)
    with pytest.raises(MarketDataUnavailableError, match="read budget"):
        run(provider.search([EVM], FIXTURE_NOW - timedelta(hours=1), keyer))


def test_x_result_budget_is_reserved_before_concurrent_requests(tmp_path: Path) -> None:
    now = Now()

    def handle(request: httpx2.Request) -> httpx2.Response:
        n = int(request.url.params["max_results"])
        rows = [
            {"id": str(10_000 + i), "author_id": str(i), "text": "hello",
             "created_at": (NOW - timedelta(minutes=i + 1)).isoformat()}
            for i in range(n)
        ]  # fmt: skip
        return httpx2.Response(200, json={"data": rows, "meta": {"result_count": n}})

    provider = x_provider(handle, now=now, reads=45, max_results_per_page=20, max_terms_per_query=1)
    tokens = [
        TokenIdentity(canonical_id=f"solana:{m}", chain="solana", address=m, symbol=None)
        for m in [MINT, MINT_2, MINT_3, "5" * 40]
    ]
    result = run(social(tmp_path, [provider], now).observe(tokens))
    assert provider.results_this_run <= 45  # 4 concurrent searches never overspend
    statuses = Counter(m.sources[0].status for m in result.momentum)
    assert statuses["PROVIDER_UNAVAILABLE"] >= 1  # over budget: never reported as zero
    [check] = result.providers
    assert check.results == provider.results_this_run
    assert check.estimated_cost_usd is None  # this test config sets no price


def test_x_daily_result_budget_holds_across_runs(tmp_path: Path) -> None:
    now = Now()
    body = {
        "data": [
            {"id": str(i), "author_id": str(i), "text": "x",
             "created_at": (NOW - timedelta(minutes=i + 1)).isoformat()}
            for i in range(20)
        ],
        "meta": {"result_count": 20},
    }  # fmt: skip
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, json=body)

    provider = x_provider(handle, now=now, max_results_per_day=20, cost_per_result_usd=0.005)
    svc = social(tmp_path, [provider], now)
    first = run(svc.observe([NEWT]))
    assert (first.providers[0].results, first.providers[0].estimated_cost_usd) == (20, 0.1)
    now.at += timedelta(minutes=10)
    second = run(svc.observe([NEWT]))
    assert len(calls) == 1  # daily budget spent: nothing sent
    assert second.providers[0].status == "PROVIDER_UNAVAILABLE"
    assert "daily result budget" in (second.providers[0].error or "")
    assert second.momentum[0].sources[0].status == "PROVIDER_UNAVAILABLE"
    now.at += timedelta(days=1)
    run(svc.observe([NEWT]))
    assert len(calls) == 2  # a new UTC day, a new budget


def test_x_usage_counts_cache_hits() -> None:
    provider = x_provider(lambda r: httpx2.Response(200, json=x_fixture()))
    provider.start_run()
    since = FIXTURE_NOW - timedelta(hours=2)
    run(provider.search([EVM], since, keyer))
    run(provider.search([EVM], since, keyer))
    usage = provider.usage()
    assert (usage["requests"], usage["results"], usage["cache_hits"]) == (1, 8, 1)


# --- X + Farcaster together (real adapters over mock transports) ----------------------------


def x_rows(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": str(abs(hash(p["id"])) % 10**18), "author_id": p["author"], "text": p["text"],
         "created_at": p["created_at"].isoformat(),
         "public_metrics": {"like_count": p["likes"], "reply_count": 1, "retweet_count": 0,
                            "quote_count": 0, "impression_count": 10}}
        for p in posts
    ]  # fmt: skip


def cast_rows(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"hash": "0x" + p["id"], "author": {"fid": abs(hash(p["author"])) % 10**6, "score": 0.7},
         "text": p["text"], "timestamp": p["created_at"].isoformat(),
         "reactions": {"likes_count": p["likes"], "recasts_count": 0}, "replies": {"count": 1},
         "embeds": []}
        for p in posts
    ]  # fmt: skip


def pair(
    now: Now,
    x_posts: list[dict[str, Any]],
    f_posts: list[dict[str, Any]],
    x_fail: int | None = None,
) -> list[Any]:
    def newer(rows: list[dict[str, Any]], key: str, since: datetime) -> list[dict[str, Any]]:
        return [r for r in rows if datetime.fromisoformat(r[key]) > since]

    def x_handle(request: httpx2.Request) -> httpx2.Response:
        if x_fail:
            return httpx2.Response(x_fail)
        since = datetime.fromisoformat(request.url.params["start_time"].replace("Z", "+00:00"))
        rows = sorted(newer(x_rows(x_posts), "created_at", since), key=lambda r: r["created_at"])
        return httpx2.Response(200, json={"data": rows[::-1], "meta": {"result_count": len(rows)}})

    def f_handle(request: httpx2.Request) -> httpx2.Response:
        after = request.url.params["q"].split("after:")[1]
        since = datetime.fromisoformat(after).replace(tzinfo=UTC)
        rows = sorted(newer(cast_rows(f_posts), "timestamp", since), key=lambda r: r["timestamp"])
        return httpx2.Response(200, json={"result": {"casts": rows[::-1], "next": {}}})

    return [
        neynar(f_handle, now=now),
        x_provider(x_handle, now=now, max_pages=3),
    ]


RISING = (1, 3, 14)  # per hour: baseline, previous hour, recent hour
FLAT = (4, 4, 4)


@pytest.mark.parametrize(
    ("x_counts", "f_counts", "accelerating", "active", "corroborated"),
    [
        (RISING, None, {"X"}, 1, False),  # X-only acceleration
        (None, RISING, {"Farcaster (Neynar)"}, 1, False),  # Farcaster-only acceleration
        (RISING, RISING, {"X", "Farcaster (Neynar)"}, 2, True),  # both accelerating
        (RISING, FLAT, {"X"}, 2, False),  # one accelerating, one steady
    ],
)
def test_x_and_farcaster_cross_platform(
    tmp_path: Path,
    x_counts: tuple[int, int, int] | None,
    f_counts: tuple[int, int, int] | None,
    accelerating: set[str],
    active: int,
    corroborated: bool,
) -> None:
    now = Now()
    x_posts = growth_posts(x_counts, prefix="x") if x_counts else []
    f_posts = growth_posts(f_counts, prefix="f") if f_counts else []
    [m] = run(social(tmp_path, pair(now, x_posts, f_posts), now).observe([NEWT])).momentum
    assert {s.provider for s in m.sources if s.accelerating} == accelerating
    c = m.cross_platform
    assert (c.platforms_with_activity, c.corroborated) == (active, corroborated)
    assert c.only_one_active_of_several == (active == 1)
    x_source = next(s for s in m.sources if s.provider == "X")
    x_h1 = next(w for w in x_source.windows if w.window == "h1")
    assert x_h1.unique_authors == (x_counts[2] if x_counts else 0)


def test_x_failure_leaves_farcaster_and_history_untouched(tmp_path: Path) -> None:
    import sqlite3

    now = Now()
    x_posts = growth_posts(RISING, prefix="x")
    f_posts = growth_posts(RISING, prefix="f")
    run(social(tmp_path, pair(now, x_posts, f_posts), now).observe([NEWT]))
    db = tmp_path / "scout.sqlite3"
    tables = ("scout_social_events", "scout_social_snapshots", "scout_social_momentum")

    def dump() -> dict[str, list[Any]]:
        conn = sqlite3.connect(db)
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY id").fetchall() for t in tables}

    before = dump()
    now.at += timedelta(minutes=10)
    result = run(social(tmp_path, pair(now, x_posts, f_posts, x_fail=429), now).observe([NEWT]))
    statuses = {c.provider: c.status for c in result.providers}
    assert statuses == {"X": "PROVIDER_UNAVAILABLE", "Farcaster (Neynar)": "PROVIDER_OK"}
    assert "rate limit" in (next(c for c in result.providers if c.provider == "X").error or "")
    [m] = result.momentum
    assert m.cross_platform.providers_checked == 1 and m.state != "UNAVAILABLE"
    after = dump()
    for t in tables:  # append-only: earlier rows are byte-for-byte unchanged
        assert after[t][: len(before[t])] == before[t]
    assert len(after["scout_social_events"]) == len(before["scout_social_events"])


def test_x_overlap_and_near_duplicates(tmp_path: Path) -> None:
    now = MovingNow()
    now.at = FIXTURE_NOW
    body = x_fixture()
    del body["meta"]["next_token"]  # a complete page: the whole span is covered
    provider = x_provider(lambda r: httpx2.Response(200, json=body), now=now)
    svc = social(tmp_path, [provider], now)
    [m] = run(svc.observe([DEGEN], universe=[DEGEN, OTHER])).momentum
    h1 = next(w for w in m.windows if w.window == "h1")
    # EXACT x3 + PROBABLE x2 (half weight); the bare-ticker post never counts
    assert (h1.mentions, h1.exact_mentions, h1.unique_authors, h1.ambiguous_posts) == (4.0, 3, 5, 1)
    assert h1.engagement == 20 + 0 + 7 + 1 + 48  # likes + replies + reposts + quotes
    now.at += timedelta(minutes=10)
    run(svc.observe([DEGEN], universe=[DEGEN, OTHER]))  # overlapping search, same posts
    events = run(svc.store.events(DEGEN.canonical_id, FIXTURE_NOW - timedelta(days=1), now.at))
    assert len(events) == 6  # 3 EXACT + 2 PROBABLE + 1 REJECTED, each stored once
    near = social_quality_of(events)
    assert near.posts == 5 and near.duplicate_share == 0.4  # posts 3 and 4: one text


def social_quality_of(events: list[Any]) -> Any:
    from upscale.services.scout.social.analysis import social_quality

    return social_quality(events, None, None, None, SocialConfig(quality={"min_posts": 2}))


def test_no_social_output_carries_a_trade_signal(tmp_path: Path) -> None:
    now = Now()
    run_ = run(
        social(tmp_path, pair(now, growth_posts(RISING), growth_posts(RISING)), now).observe([NEWT])
    )
    body = run_.model_dump(mode="json")
    body.pop("disclaimer")
    for m in body["momentum"]:
        m.pop("disclaimer")
    dumped = json.dumps(body).upper()
    assert "BUY" not in dumped and "SELL" not in dumped


# --- Attribution hardening: ticker-only stays conservative ----------------------------------


@pytest.mark.parametrize(
    ("symbol", "name", "text", "level"),
    [
        ("NEWT", "Newt Protocol", "$NEWT is exploding", "AMBIGUOUS"),  # ticker only
        ("NEWT", "Newt Protocol", "$NEWT on solana", "PROBABLE"),  # ticker + chain
        ("NEWT", "Newt Protocol", "$NEWT (Newt Protocol)", "PROBABLE"),  # ticker + name
        ("NEWT", "Newt Protocol", "Newt Protocol ships v2", "AMBIGUOUS"),  # name only
        ("NEWT", "Newt Protocol", "Newt Protocol on solana", "PROBABLE"),  # name + chain
        ("NEWT", "Newt Protocol", "$NEWT Newt Protocol on solana", "STRONG"),
        ("AI", "Artifact Intel", "$AI on solana", "AMBIGUOUS"),  # short: chain isn't enough
        ("AI", "Artifact Intel", "$AI Artifact Intel", "PROBABLE"),  # short + name
        ("AI", "Artifact Intel", f"$AI {MINT}", "EXACT"),  # a contract always decides
    ],
)
def test_ticker_only_evidence_stays_conservative(
    symbol: str, name: str, text: str, level: str
) -> None:
    token = NEWT.model_copy(update={"symbol": symbol, "name": name})
    # a directory that knows no competitor must not make a lone reference count
    index = AttributionIndex([token], universe=[token, OTHER])
    [a] = index.attribute(post(text))
    assert a.level == level
    assert (a.canonical_id is None) == (level == "AMBIGUOUS")
