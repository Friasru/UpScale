"""Production objects -> evidence records. Serialization only: every value is what the
production object already holds (nothing fetched, estimated or filled in), plus per-group
availability so "not reported" is never confused with "safe" or "zero"."""

from datetime import UTC, datetime
from typing import Any

from upscale.schemas import ChatRequest, ChatResponse
from upscale.services.evidence_archive.store import Availability, PendingRecord
from upscale.services.market_data import (
    AssetNotFoundError,
    InvalidRequestError,
    ProviderRateLimitedError,
)
from upscale.services.outcomes.models import DecisionObservation
from upscale.services.scout.growth.models import GrowthScoutResult
from upscale.services.scout.models import ScoutCandidate, ScoutMarketMetrics
from upscale.services.scout.normalize import canonical_id
from upscale.services.scout.social.models import SocialMomentum
from upscale.services.solana_chain import KnownPool, OnchainSafetySnapshot
from upscale.services.solana_dex import SolanaDexSnapshot

# Free text that may be copyrighted (news) is referenced (URL, source, time), not copied.
TEXT_KEYS = frozenset({"title", "summary", "description", "content", "text", "headline", "body"})
MAX_QUERY_CHARS = 2000


def _av(value: Any) -> Availability:
    return "AVAILABLE" if value is not None else "NOT_AVAILABLE"


def failure_state(error: BaseException | str | None) -> Availability:
    if isinstance(error, ProviderRateLimitedError):
        return "RATE_LIMITED"
    if isinstance(error, AssetNotFoundError | InvalidRequestError):
        return "NOT_AVAILABLE"
    text = str(error or "").lower()
    limited = "rate limit" in text or "429" in text or "limit was reached" in text
    return "RATE_LIMITED" if limited else "PROVIDER_FAILED"


def market_fields(m: ScoutMarketMetrics, pool_created: datetime | None) -> dict[str, Availability]:
    h1, h24 = m.window("h1"), m.window("h24")
    return {
        "price_usd": _av(m.price_usd),
        "liquidity_usd": _av(m.liquidity_usd),
        "market_cap_usd": _av(m.market_cap_usd),
        "fdv_usd": _av(m.fdv_usd),
        "volume_h24_usd": _av(h24.volume_usd if h24 else None),
        "txns_h24": _av(h24.txns if h24 else None),
        "buyers_h1": _av(h1.buyers if h1 else None),
        "pool_created_at": _av(pool_created),
    }


def market(c: ScoutCandidate) -> list[PendingRecord]:
    return [
        PendingRecord(
            kind="market",
            asset_id=c.canonical_id,
            chain=c.chain,
            address=c.address,
            pool_address=c.pool.address,
            dex=c.pool.dex,
            provider=c.market_provider,
            observed_at=c.observed_at,
            payload={
                "candidate": c.model_dump(mode="json"),
                "field_availability": market_fields(c.metrics, c.pool.created_at),
                "normalized_by": "scout.normalize.build_candidates + scout.features",
            },
        )
    ]


def dex_market(s: SolanaDexSnapshot) -> list[PendingRecord]:
    day = s.window("h24")
    return [
        PendingRecord(
            kind="dex_market",
            asset_id=canonical_id(s.chain, s.mint),
            chain=s.chain,
            address=s.mint,
            pool_address=s.pair_address,
            dex=s.dex,
            provider=s.provider,
            observed_at=s.fetched_at,
            payload={
                "snapshot": s.model_dump(mode="json"),
                "field_availability": {
                    "market_cap_usd": _av(s.market_cap_usd),
                    "fdv_usd": _av(s.fdv_usd),
                    "pair_created_at": _av(s.pair_created_at),
                    "txns_h24": _av(day.txns if day else None),
                },
                "normalized_by": "solana_dex.build_snapshot",
            },
        )
    ]


def safety(s: OnchainSafetySnapshot, pools: list[KnownPool]) -> list[PendingRecord]:
    components: dict[str, Availability] = {
        "authorities": "AVAILABLE"
        if s.authorities_available
        else failure_state(s.authorities_error),
        "holders": "AVAILABLE" if s.holders_available else failure_state(s.holders_error),
    }
    usable = s.authorities_available or s.holders_available
    return [
        PendingRecord(
            kind="safety",
            asset_id=s.canonical_id,
            chain="solana",
            address=s.mint,
            provider=s.provider,
            observed_at=s.fetched_at,
            availability="AVAILABLE" if usable else "PROVIDER_FAILED",
            reason=None if usable else f"{s.authorities_error}; {s.holders_error}",
            payload={
                "snapshot": s.model_dump(mode="json"),
                "pools": [p.model_dump(mode="json") for p in pools],
                "components": components,
                "flags": {
                    "mint_authority_active": s.mint_authority_active,
                    "freeze_authority_active": s.freeze_authority_active,
                    "concentration_authoritative": s.concentration_authoritative,
                },
                "normalized_by": "solana_chain.SolanaSafetyService.get_snapshot (analyze)",
            },
        )
    ]


def safety_failed(
    chain: str, address: str, provider: str | None, error: BaseException, at: datetime
) -> list[PendingRecord]:
    state = failure_state(error)
    return [
        PendingRecord(
            kind="safety",
            asset_id=canonical_id(chain, address),
            chain=chain,
            address=address,
            provider=provider,
            observed_at=at,
            availability=state,
            reason=f"{type(error).__name__}: {error}",
            payload={"error": str(error), "components": {"authorities": state, "holders": state}},
        )
    ]


def social(m: SocialMomentum) -> list[PendingRecord]:
    chain, _, address = m.canonical_id.partition(":")
    measured = m.state != "UNAVAILABLE"
    return [
        PendingRecord(
            kind="social",
            asset_id=m.canonical_id,
            chain=chain,
            address=address,
            provider=",".join(sorted({s.provider for s in m.sources})) or None,
            observed_at=m.computed_at,
            availability="AVAILABLE" if measured else "NOT_AVAILABLE",
            reason=None if measured else ("; ".join(m.reasons) or "no provider could search"),
            payload={
                "momentum": m.model_dump(mode="json"),  # metrics only: posts are never stored
                "providers": {s.provider: s.status for s in m.sources},
                "normalized_by": "scout.social.SocialScoutService",
            },
        )
    ]


def scout(result: GrowthScoutResult, capabilities: list[str]) -> list[PendingRecord]:
    run = {
        "computed_at": result.computed_at.isoformat(),
        "mode": result.mode,
        "evaluated": result.evaluated,
        "eligible": result.eligible,
        "notes": result.notes,
    }
    out = []
    for g in [*result.candidates, *result.unranked]:
        out.append(
            PendingRecord(
                kind="scout",
                asset_id=g.canonical_id,
                chain=g.chain,
                address=g.address,
                pool_address=g.market.selected_pool.address,
                dex=g.market.selected_pool.dex,
                provider=g.market.market_provider,
                observed_at=result.computed_at,
                provider_at=g.observed_at,  # when its market evidence was fetched
                payload={
                    "candidate": g.model_dump(mode="json"),
                    "run": run,
                    "capabilities": capabilities,
                    "availability": {
                        "social": g.momentum.social_status,
                        "safety": g.quality.safety_status,
                        "data_status": g.data_status,
                    },
                },
                links={"run": run["computed_at"]},
            )
        )
    return out


def _strip_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_text(v) for k, v in value.items() if k not in TEXT_KEYS}
    if isinstance(value, list):
        return [_strip_text(v) for v in value]
    return value


def decision(
    request: ChatRequest,
    response: ChatResponse,
    observation: DecisionObservation | None,
    at: datetime,
    capabilities: list[str],
) -> list[PendingRecord]:
    analysis = response.analysis
    if analysis is None or not analysis.agent_results:
        return []  # no analysis ran (a clarification or a concept question)
    results = {r.agent: r for r in analysis.agent_results}
    chain = address = None
    if request.asset is not None:
        chain, address = request.asset.chain, request.asset.address.strip()
    elif observation is not None and observation.chain and observation.address:
        chain, address = observation.chain, observation.address
    else:
        dex = results.get("dex_market")
        if dex is not None and isinstance(dex.findings.get("mint"), str):
            chain, address = dex.findings.get("chain") or "solana", dex.findings["mint"]
    if chain and address:
        asset_id = canonical_id(chain, address)
    elif observation is not None:
        asset_id = observation.asset_id
    else:
        asset_id = "unresolved:" + ",".join(analysis.assets or ["none"])
    agents = {
        name: {
            "status": r.status,
            "mock": r.mock,
            "summary": r.summary if name != "news_sentiment" else None,
            "error": r.error,
            "findings": _strip_text(r.findings) if name == "news_sentiment" else r.findings,
        }
        for name, r in results.items()
    }
    latest = request.messages[-1]
    return [
        PendingRecord(
            kind="decision",
            asset_id=asset_id,
            chain=chain,
            address=address,
            pool_address=observation.price_ref.pool_address if observation else None,
            provider=None,
            observed_at=at,
            availability="AVAILABLE" if observation is not None else "NOT_AVAILABLE",
            reason=None
            if observation is not None
            else "no new measurable decision recorded (none made, market not measurable, or a repeat)",
            payload={
                "request": {
                    "query": latest.content[:MAX_QUERY_CHARS],
                    "attachments": len(latest.attachments),
                    "earlier_messages": len(request.messages) - 1,
                    "asset": request.asset.model_dump(mode="json") if request.asset else None,
                },
                "agents": agents,
                "routing": analysis.routing,
                "uncertainty": analysis.uncertainty.model_dump(mode="json"),
                "decision": observation.model_dump(mode="json") if observation else None,
                "capabilities": capabilities,
            },
            links={"decision_observation_id": observation.id if observation else None},
        )
    ]


def utcnow() -> datetime:
    return datetime.now(UTC)
