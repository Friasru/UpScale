"""Scout in the app: the `/scout` view (top N, identity, freshness, safety, score
components, filters, warnings, refresh behavior) and the exact Analyze handoff through
`/chat` (`asset`: chain + contract / mint, never a ticker search).

Offline: Growth Scout results are built from the synthetic markets of test_scout_growth;
the network is blocked by conftest.
"""

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import upscale.main
from tests.test_scout_continuity import mint
from tests.test_scout_growth import (
    MINTS,
    NOW,
    accelerating,
    candidate,
    fading,
    flat,
    momentum,
    run,
    safety_snapshot,
    service,
    thin_pump,
)
from upscale.agents import Agent, AgentContext
from upscale.orchestrator import Orchestrator
from upscale.schemas import AgentName, AgentResult, AssetRef, ChatMessage, ChatRequest
from upscale.scout_api import ScoutFeed, ScoutFilters, build_view
from upscale.services.asset_resolver import AssetResolver
from upscale.services.scout.growth.models import GrowthScoutResult, UniverseReport
from upscale.services.scout.models import ScoutFeedReport
from upscale.services.scout.social.models import ProviderCheck, SocialSourceSnapshot

X_402 = "X credits are exhausted (HTTP 402)"


def with_x_unavailable(key: str, state: str) -> Any:
    m = momentum(key, state)
    m.sources = [
        SocialSourceSnapshot(canonical_id=m.canonical_id, provider="X", platform="x",
                             observed_at=NOW, status="PROVIDER_UNAVAILABLE", error=X_402),
        SocialSourceSnapshot(canonical_id=m.canonical_id, provider="Farcaster (Neynar)",
                             platform="farcaster", observed_at=NOW,
                             status="PROVIDER_CHECKED_ZERO_MATCHES"),
    ]  # fmt: skip
    return m


def ranking(tmp_path: Path, extra: int = 0) -> GrowthScoutResult:
    """A realistic ranking: an accelerating token with complete safety and emerging social
    (X unavailable), a stale carried one, a thin pump, fading / flat ones, and `extra`
    more accelerating tokens (for top-N)."""
    svc = service(tmp_path)
    scenarios = [accelerating("A"), accelerating("B", mcap=None, fdv=900_000), thin_pump("C"),
                 fading("D"), flat("E")]  # fmt: skip
    scenarios += [accelerating(mint(i)) for i in range(extra)]

    async def go() -> GrowthScoutResult:
        for c, snaps in scenarios:
            await svc.store.record_seen(c)
            for s in snaps:
                await svc.store.save_snapshot(s, 0)
        return await svc.rank(
            [c for c, _ in scenarios],
            {m.canonical_id: m for m in [with_x_unavailable("A", "EMERGING")]},
            {s.canonical_id: s for s in [safety_snapshot("A")]},
            limit=500,
            carried={f"solana:{MINTS['B']}": 8.0},
        )

    result = run(go())
    result.social_checks = [
        ProviderCheck(provider="X", platform="x", status="PROVIDER_UNAVAILABLE",
                      checked_at=NOW, error=X_402),
        ProviderCheck(provider="Farcaster (Neynar)", platform="farcaster",
                      status="PROVIDER_OK", checked_at=NOW),
    ]  # fmt: skip
    result.universe = UniverseReport(
        discovered=4, refreshed=0, carried_stale=1, expired=0, unusable=0, horizon_hours=6,
        feeds=[ScoutFeedReport(provider="GeckoTerminal", available=["new:solana"],
                               executed=["new:solana"], deferred=[], failed=["new:solana"],
                               requests=1)],
    )  # fmt: skip
    return result


@pytest.fixture
def scout_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> Any:
    result = ranking(tmp_path, extra=18)
    calls = {"n": 0}

    async def scan() -> GrowthScoutResult:
        calls["n"] += 1
        return result

    feed = ScoutFeed(scan, now=lambda: NOW + timedelta(minutes=2))
    monkeypatch.setattr(upscale.main, "scout_feed", feed)
    return client, calls, result


def cards_by_symbol(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["symbol"]: c for c in body["candidates"]}


# --- /scout -------------------------------------------------------------------------------


def test_scout_returns_top_n_with_exact_identity(scout_client: Any) -> None:
    client, calls, result = scout_client
    body = client.get("/scout").json()
    assert body["status"] == "ok" and body["limit"] == 10 and len(body["candidates"]) == 10
    assert body["mode"] == "NEW_AND_EARLY" and body["ranked"] == result.eligible
    top20 = client.get("/scout", params={"limit": 20}).json()
    assert len(top20["candidates"]) == 20
    assert client.get("/scout", params={"limit": 500}).status_code == 422  # never hundreds
    assert calls["n"] == 1  # the ranking is reused, not rescanned per request
    a = cards_by_symbol(body)["A"]
    assert a["canonical_id"] == f"solana:{MINTS['A']}"
    assert (a["chain"], a["chain_label"], a["address"]) == ("solana", "Solana", MINTS["A"])
    assert a["analyze"] == {
        "chain": "solana", "address": MINTS["A"], "symbol": "A", "name": "Token A",
        "pool_address": "pool-A", "source": "scout",
    }  # fmt: skip
    assert a["stage"] == "ACCELERATING"  # the backend stage, unchanged
    assert isinstance(a["scout_momentum"], int)
    assert a["reasons"] and all(isinstance(r, str) for r in a["reasons"])
    assert body["disclaimer"].startswith("Growth Scout ranks discovery candidates")


def test_stale_status_is_included(scout_client: Any) -> None:
    client, _, _ = scout_client
    b = cards_by_symbol(client.get("/scout", params={"limit": 20}).json())["B"]
    assert b["freshness"]["status"] == "STALE_CARRIED"
    assert b["freshness"]["snapshot_age_minutes"] == pytest.approx(2.0)
    assert any(f["label"] == "Stale market data" for f in b["safety"]["flags"])
    freshness = next(s for s in b["details"] if s["title"] == "Freshness")
    assert any("carried forward" in r["value"] for r in freshness["rows"])
    a = cards_by_symbol(client.get("/scout").json())["A"]
    assert a["freshness"]["status"] == "CURRENT"


def test_safety_status_and_serious_flags(scout_client: Any) -> None:
    client, _, _ = scout_client
    cards = cards_by_symbol(client.get("/scout", params={"limit": 20}).json())
    assert cards["A"]["safety"]["label"] == "Safety checks complete"
    b = cards["B"]["safety"]
    assert b["status"] == "INSUFFICIENT_SAFETY_DATA" and b["label"] == "Insufficient safety data"
    labels = {f["label"] for c in cards.values() for f in c["safety"]["flags"]}
    assert "Safe" not in " ".join(labels) and "Legit" not in " ".join(labels)


def test_score_components_serialize(scout_client: Any) -> None:
    client, _, result = scout_client
    a = cards_by_symbol(client.get("/scout").json())["A"]
    scoring = {r["label"]: r["value"] for r in next(
        s for s in a["details"] if s["title"] == "Scoring")["rows"]}  # fmt: skip
    assert set(scoring) >= {"Market activity", "Liquidity quality", "Social momentum",
                            "Earliness", "Cross-confirmation", "Base score",
                            "Stage adjustment", "Risk penalty", "Scout Momentum"}  # fmt: skip
    g = next(g for g in result.candidates if g.symbol == "A")
    assert scoring["Stage adjustment"] == f"{g.scout_momentum.stage_adjustment:+.1f}"
    assert a["scout_momentum"] == round(g.score)
    text = str(a).lower()
    assert "confidence" not in text and "probability" not in text


def test_see_more_sections_and_social_unavailable_is_not_zero(scout_client: Any) -> None:
    client, _, _ = scout_client
    a = cards_by_symbol(client.get("/scout").json())["A"]
    titles = [s["title"] for s in a["details"]]
    assert titles == ["Identity", "Market", "Social", "Scoring", "Safety", "Freshness"]
    social = {r["label"]: r["value"] for r in a["details"][2]["rows"]}
    assert social["X"] == f"unavailable ({X_402})"
    assert social["Farcaster (Neynar)"] == "searched, no mentions"
    identity = {r["label"]: r["value"] for r in a["details"][0]["rows"]}
    assert identity["Mint"] == MINTS["A"] and "pool-A" in identity["Selected market"]


def test_market_cap_is_only_shown_when_trustworthy(scout_client: Any) -> None:
    client, _, _ = scout_client
    b = cards_by_symbol(client.get("/scout", params={"limit": 20}).json())["B"]
    assert b["market"]["market_cap_usd"] is None and b["market"]["fdv_usd"] == 900_000


def test_partial_provider_failures_still_return_rankings(scout_client: Any) -> None:
    client, _, _ = scout_client
    body = client.get("/scout").json()
    assert body["status"] == "ok" and body["candidates"]
    warnings = " ".join(body["warnings"])
    assert "Some social sources are unavailable: X (credits exhausted)" in warnings
    assert "never as zero attention" in warnings
    assert "partially unavailable" in warnings
    assert "stale market data" in warnings


def test_filters_select_without_rescoring(scout_client: Any) -> None:
    client, _, _ = scout_client
    everything = client.get("/scout", params={"limit": 20}).json()
    early = client.get("/scout", params={"limit": 20, "stage": "ACCELERATING"}).json()
    assert early["candidates"] and all(c["stage"] == "ACCELERATING" for c in early["candidates"])
    scores = {c["canonical_id"]: c["scout_momentum"] for c in everything["candidates"]}
    for c in early["candidates"]:
        if c["canonical_id"] in scores:
            assert c["scout_momentum"] == scores[c["canonical_id"]]  # never re-scored
    ranks = [c["rank"] for c in early["candidates"]]
    assert ranks == sorted(ranks)  # Growth Scout's order is kept
    rich = client.get("/scout", params={"min_liquidity": 50_000}).json()
    assert all(c["market"]["liquidity_usd"] >= 50_000 for c in rich["candidates"])
    assert client.get("/scout", params={"chain": "base"}).json()["status"] == "empty"
    assert client.get("/scout", params={"chain": "dogechain"}).status_code == 422


def test_no_candidates_is_empty_not_an_error(tmp_path: Path) -> None:
    view = build_view(GrowthScoutResult(computed_at=NOW, mode="NEW_AND_EARLY", evaluated=0,
                                        eligible=0, limit=10, candidates=[]),
                      10, ScoutFilters(), NOW)  # fmt: skip
    assert (view.status, view.candidates, view.error) == ("empty", [], None)


# --- Refresh ------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_refresh_is_single_flight_and_rate_limited(tmp_path: Path) -> None:
    result = ranking(tmp_path)
    calls = {"n": 0}
    gate = asyncio.Event()

    async def scan() -> GrowthScoutResult:
        calls["n"] += 1
        await gate.wait()
        return result

    clock = Clock()
    feed = ScoutFeed(scan, min_refresh_seconds=60, clock=clock, now=lambda: NOW)

    async def go() -> None:
        first = asyncio.create_task(feed.refresh())
        second = asyncio.create_task(feed.refresh())  # a double click
        await asyncio.sleep(0)
        assert feed.refreshing
        gate.set()
        await asyncio.gather(first, second)
        assert calls["n"] == 1
        await feed.refresh()  # right after: kept, providers aren't hammered
        assert calls["n"] == 1
        clock.t = 61
        await feed.refresh()
        assert calls["n"] == 2

    asyncio.run(go())


def test_refresh_failure_keeps_the_previous_results(tmp_path: Path) -> None:
    result = ranking(tmp_path)
    outcomes: list[Any] = [result, RuntimeError("provider down")]

    async def scan() -> GrowthScoutResult:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[no-any-return]

    feed = ScoutFeed(scan, min_refresh_seconds=0, now=lambda: NOW)
    first = asyncio.run(feed.view(10, ScoutFilters()))
    asyncio.run(feed.refresh())
    after = asyncio.run(feed.view(10, ScoutFilters()))
    assert after.status == "ok" and len(after.candidates) == len(first.candidates)
    assert after.error == "Scout refresh failed (RuntimeError); showing the last results."


def test_first_refresh_failure_is_unavailable(tmp_path: Path) -> None:
    async def scan() -> GrowthScoutResult:
        raise RuntimeError("everything down")

    view = asyncio.run(ScoutFeed(scan).view(10, ScoutFilters()))
    assert view.status == "unavailable" and view.candidates == []
    assert view.error == "Scout refresh failed (RuntimeError)."


def test_refresh_endpoint(scout_client: Any) -> None:
    client, calls, _ = scout_client
    body = client.post("/scout/refresh", params={"limit": 20}).json()
    assert body["status"] == "ok" and len(body["candidates"]) == 20
    client.post("/scout/refresh")
    assert calls["n"] == 1  # the second one came within the refresh interval


# --- Analyze handoff ------------------------------------------------------------------------


class Capture(Agent):
    """Records which token each agent was asked about."""

    def __init__(self, name: AgentName, seen: list[tuple[str, Any]]):
        self.name = name
        self.depends_on = ()
        self.description = "test"
        self.seen = seen

    async def run(self, context: AgentContext) -> AgentResult:
        identity = context.asset_identity
        self.seen.append((self.name, (identity.chain, identity.address) if identity else None))
        return AgentResult(agent=self.name, mock=False, summary="ok")


class SameTickerSearch:
    """A DEX search that would resolve "PEPE" to a *different* token."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search_pools(self, query: str) -> list[Any]:
        self.queries.append(query)
        raise AssertionError("Analyze must not search by ticker when Scout hands over a token")


AGENTS: list[AgentName] = ["dex_market", "technical_analysis", "market", "news_sentiment",
                           "onchain_safety", "risk", "opportunity"]  # fmt: skip


def analyze(asset: AssetRef, text: str) -> tuple[list[tuple[str, Any]], SameTickerSearch, Any]:
    seen: list[tuple[str, Any]] = []
    search = SameTickerSearch()
    orchestrator = Orchestrator(
        agents=[Capture(a, seen) for a in AGENTS], resolver=AssetResolver(search=search)
    )
    request = ChatRequest(messages=[ChatMessage(role="user", content=text)], asset=asset)
    return seen, search, asyncio.run(orchestrator.respond(request))


def test_analyze_receives_the_exact_canonical_identity() -> None:
    mint = MINTS["A"]
    seen, search, response = analyze(
        AssetRef(chain="solana", address=mint, symbol="PEPE", pool_address="pool-A"),
        f"Analyze PEPE (Solana {mint})",
    )
    assert search.queries == []  # no discovery, no ticker search
    assert seen and {identity for _, identity in seen} == {("solana", mint)}
    assert "dex_market" in response.analysis.agents_used  # the token path, by address


def test_a_duplicate_ticker_cannot_redirect_analyze() -> None:
    """The text names only the ticker (another PEPE would win a ticker search); the handed
    over identity decides."""
    evm = "0x6982508145454ce325ddbe47a25d4ec3d2311933"
    seen, search, _ = analyze(AssetRef(chain="base", address=evm, symbol="PEPE"), "Analyze PEPE")
    assert search.queries == []
    assert {identity for _, identity in seen} == {("base", evm)}


def test_chat_accepts_the_scout_asset_and_rejects_a_malformed_one(client: TestClient) -> None:
    message = {"role": "user", "content": "Analyze A", "attachments": []}
    bad = client.post("/chat", json={"messages": [message],
                                     "asset": {"chain": "solana", "address": "not-a-mint"}})  # fmt: skip
    assert bad.status_code == 422
    ok = client.post("/chat", json={"messages": [message],
                                    "asset": {"chain": "solana", "address": MINTS["A"]}})  # fmt: skip
    assert ok.status_code == 200
    analysis = ok.json()["analysis"]
    assert "dex_market" in analysis["agents_used"]
    assert analysis["routing"]["dex_market"].startswith(f"Exact token solana:{MINTS['A']}")
    _ = candidate  # shared helper kept importable
