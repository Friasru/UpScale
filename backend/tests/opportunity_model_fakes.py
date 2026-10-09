"""Offline fixtures for Opportunity Model tests: a local Evidence Archive written with the
production `EvidenceStore`, and a local Safety V2 database (either a real one written by
Safety V2 with offline fakes, or a bare ``safety_snapshots`` table for version tests).
Nothing here reaches a network."""

import copy
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tests.safety_v2_fakes import MINT, addr
from upscale.services.evidence_archive.store import EvidenceStore, PendingRecord
from upscale.services.opportunity_model.loaders import ArchiveReader, SafetyReader
from upscale.services.opportunity_model.models import OpportunityInput, OpportunityOrigin
from upscale.services.opportunity_model.service import build_input
from upscale.services.safety_v2.repository import encode_body

CID = f"solana:{MINT}"
VARIANT = "m" + MINT[1:]  # a case variant: another (valid) Solana mint
OTHER = addr("AnotherMint")
T = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def at(minutes: float = 0.0, seconds: float = 0.0) -> datetime:
    return T + timedelta(minutes=minutes, seconds=seconds)


def iso(t: datetime) -> str:
    return t.isoformat()


# --- Evidence Archive -------------------------------------------------------------------------


class Archive:
    """A writable archive for setting up records; `archived` sets their archived_at."""

    def __init__(self, path: Path):
        self.path = path
        self.archived: datetime | None = None
        self.store = EvidenceStore(path, clock=self._clock)

    def _clock(self) -> float:
        return (self.archived or T + timedelta(days=1)).timestamp()

    def add(self, r: PendingRecord, archived: datetime | None = None) -> None:
        self.archived = archived or r.observed_at
        assert self.store.append(r)

    def close(self) -> None:
        self.store.close()


def _sig(name: str, score: float, weight: float, raw: float | None = None) -> dict[str, Any]:
    return {"name": name, "score": score, "weight": weight, "raw": raw, "detail": name}


def candidate(observed: datetime, **over: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "rank": 1, "canonical_id": CID, "symbol": "TEST", "name": "Test", "chain": "solana",
        "address": MINT, "observed_at": iso(observed), "data_status": "CURRENT",
        "snapshot_age_minutes": None, "stage": "ACCELERATING", "stage_reasons": ["volume up"],
        "unconfirmed_stage": None, "eligible": True, "ineligible_reasons": [],
        "market": {"price_usd": 0.01, "liquidity_usd": 120_000.0, "pool_age_hours": 30.0,
                   "oldest_pool_age_hours": 30.0, "market_cap_usd": 900_000.0},
        "momentum": {
            "volume_acceleration": 2.4, "volume_acceleration_basis": "m15/h1",
            "txn_acceleration": 1.8, "txn_acceleration_basis": "m15/h1",
            "buy_share": 0.61, "buy_pressure_change": 0.05, "buyer_share": 0.58,
            "liquidity_change_pct": 12.0, "liquidity_change_basis": "60m",
            "move_since_first_seen_pct": 40.0,
            "technical": {"snapshots": 6, "span_hours": 2.5, "trend": "up", "change_pct": 18.0,
                          "breakout": True, "volume_confirmed": True, "higher_lows": True},
            "social_status": "SOCIAL_ACCELERATING", "social_state": "ACCELERATING",
            "social_reason": "authors up", "mention_acceleration": 2.0,
            "cross_platform_corroborated": True, "platforms_active": 2,
        },
        "quality": {
            "verification": ["VERIFIED_IDENTITY"], "identity_status": "VERIFIED_IDENTITY",
            "market_confirmed": True, "safety_status": "SAFETY_CHECKS_PARTIAL",
            "safety_missing": ["holder scan incomplete"], "social_attribution": "exact",
            "exact_mention_share": 0.8, "spam_risk": "low", "organic_signal": "medium",
            "holder_top1_pct": 9.0, "holder_top10_pct": 55.0, "holder_data_lower_bound": True,
            "mint_authority_active": True, "freeze_authority_active": False,
            "liquidity_quality": "healthy", "market_status": "OK", "collapse_evidence": [],
            "flow_quality": "consistent", "flow_notes": [],
        },
        "scout_momentum": {
            "score": 72.0, "base": 70.0, "stage_adjustment": 5.0, "risk_penalty": 3.0,
            "families": [
                {"family": "market_activity", "score": 0.7, "weight": 0.35, "contribution": 24.5,
                 "available": True, "notes": [], "signals": [
                     _sig("volume_acceleration", 0.8, 0.30, 2.4),
                     _sig("trades_acceleration", 0.7, 0.25, 1.8),
                     _sig("buy_pressure", 0.6, 0.20, 0.61),
                     _sig("price_momentum", 0.9, 0.15, 25.0),
                     _sig("technical", 0.9, 0.10, 18.0)]},
                {"family": "liquidity_quality", "score": 0.6, "weight": 0.15, "contribution": 9,
                 "available": True, "notes": [], "signals": [
                     _sig("depth", 0.5, 0.4, 120_000.0), _sig("growth", 0.7, 0.3, 12.0),
                     _sig("stability", 0.9, 0.2, -2.0), _sig("liquidity_to_cap", 0.5, 0.1, 0.13)]},
                {"family": "social_momentum", "score": 0.8, "weight": 0.12, "contribution": 9.6,
                 "available": True, "notes": [], "signals": [_sig("state", 0.8, 1.0)]},
                {"family": "earliness", "score": 0.6, "weight": 0.18, "contribution": 10.8,
                 "available": True, "notes": [], "signals": [
                     _sig("move_already_made", 0.7, 0.4, 40.0), _sig("age", 1.0, 0.2, 30.0),
                     _sig("activity_regime", 0.6, 0.3, 1.4),
                     _sig("market_size_context", 0.5, 0.1, 900_000.0)]},
                {"family": "cross_confirmation", "score": 0.85, "weight": 0.2,
                 "contribution": 17.0, "available": True, "notes": ["3 confirmations"],
                 "signals": [_sig("volume accelerating", 1.0, 0.0),
                             _sig("price rising (not vertical)", 1.0, 0.0),
                             _sig("social agrees", 1.0, 0.0)]},
            ],
        },
        "risk_flags": [
            {"code": "mint_authority_active", "severity": "high", "detail": "mint", "penalty": 15},
            {"code": "holder_concentration_top10", "severity": "high", "detail": "55%",
             "penalty": 15},
            {"code": "liquidity_draining", "severity": "high", "detail": "-30%", "penalty": 8},
            {"code": "thin_market_pump", "severity": "high", "detail": "pump", "penalty": 8},
            {"code": "social_only_hype", "severity": "high", "detail": "hype", "penalty": 8},
            {"code": "flow_in_doubt", "severity": "caution", "detail": "few wallets",
             "penalty": 0},
        ],
    }  # fmt: skip
    c.update(over)
    return c


def scout_record(
    observed: datetime,
    market_at: datetime | None = None,
    cand: dict[str, Any] | None = None,
    causal_valid: bool = True,
    version: int = 2,
    asset_id: str = CID,
) -> PendingRecord:
    market_at = market_at or observed
    c = cand if cand is not None else candidate(market_at)
    timing = {"version": version, "market_observed_at": iso(market_at),
              "evaluation_started_at": iso(market_at), "decision_at": iso(observed)}  # fmt: skip
    return PendingRecord(
        kind="scout", asset_id=asset_id, chain="solana", address=asset_id.partition(":")[2],
        pool_address=addr("PooLAAA"), provider="geckoterminal", component="scout",
        observed_at=observed, provider_at=market_at,
        payload={"candidate": c, "timing": timing, "causal": {"valid": causal_valid}},
        links={**timing, "causal_valid": causal_valid, "evidence": []},
    )  # fmt: skip


def momentum(computed: datetime, state: str = "ACCELERATING", cid: str = CID) -> dict[str, Any]:
    return {
        "canonical_id": cid, "computed_at": iso(computed), "state": state,
        "reasons": ["authors up"], "window": "h1",
        "trend": {"window": "h1",
                  "mentions": {"acceleration_ratio": 2.0},
                  "unique_authors": {"acceleration_ratio": 1.5},
                  "engagement": {"acceleration_ratio": 1.2}},
        "windows": [{"window": "h1", "mentions": 12.0, "unique_authors": 9, "engagement": 40,
                     "exact_mentions": 8, "attributable_posts": 10, "promoted_posts": 1,
                     "ambiguous_posts": 2, "sources": 2}],
        "quality": {"spam_risk": "medium", "organic_signal_strength": "medium",
                    "reasons": ["one author dominates"], "top_author_share": 0.4,
                    "duplicate_share": 0.1, "promoted_share": 0.05, "conflicting_contracts": 0},
        "cross_platform": {"providers_configured": 2, "providers_checked": 2,
                           "platforms_with_activity": 2, "platforms_accelerating": 1,
                           "corroborated": False, "single_source_available": False,
                           "only_one_active_of_several": False},
        "market": {"state": "CORROBORATED", "reasons": ["volume up too"], "market_window": "m15/h1"},
        "sources": [{"canonical_id": cid, "provider": "X", "platform": "x", "status": "PROVIDER_OK",
                     "observed_at": iso(computed)}],
    }  # fmt: skip


def social_record(
    observed: datetime, state: str = "ACCELERATING", m: dict[str, Any] | None = None
) -> PendingRecord:
    m = m if m is not None else momentum(observed, state)
    measured = m["state"] != "UNAVAILABLE"
    return PendingRecord(
        kind="social", asset_id=CID, chain="solana", address=MINT, provider="X",
        component="scout_social", observed_at=observed,
        availability="AVAILABLE" if measured else "NOT_AVAILABLE",
        reason=None if measured else "credits exhausted",
        payload={"momentum": m, "providers": {"X": "PROVIDER_OK" if measured else
                                              "PROVIDER_UNAVAILABLE"}},
    )  # fmt: skip


def technical_findings(**over: Any) -> dict[str, Any]:
    f = {"symbol": "TEST", "timeframe": "15m", "provider": "geckoterminal", "candle_count": 120,
         "canonical_id": CID, "last_candle_at": iso(T - timedelta(hours=2)), "as_of": None,
         "trend": {"method": "ema", "available": True, "label": "uptrend",
                   "reasons": ["EMA20 > EMA50"], "unavailable_reason": None}}  # fmt: skip
    f.update(over)
    return f


def news_findings(model: str | None = "claude-x") -> dict[str, Any]:
    story = {"source": "CoinDesk", "published_at": iso(T - timedelta(hours=3)),
             "sentiment": "bullish", "impact": "high", "scope": "asset", "stale": False}  # fmt: skip
    return {"provider": "rss", "reports": [{
        "asset": "TEST", "subject": "TEST", "overall_news_sentiment": "bullish",
        "market_news_sentiment": "neutral", "sentiment_counts": {"bullish": 1},
        "asset_story_count": 1, "stale_story_count": 0, "sentiment_model": model,
        "provider": "rss", "retrieved_at": iso(T - timedelta(hours=1)),
        "analyzed_at": iso(T - timedelta(hours=1)), "stories": [story]}]}  # fmt: skip


def decision_record(
    observed: datetime,
    action: str = "buy",
    confidence: str = "high",
    technical: dict[str, Any] | None = None,
    news: dict[str, Any] | None = None,
    technical_status: str = "ok",
) -> PendingRecord:
    agents = {
        "technical_analysis": {"status": technical_status, "mock": False, "summary": "",
                               "error": None if technical_status == "ok" else "no candles",
                               "findings": technical or technical_findings()},
        "news_sentiment": {"status": "ok", "mock": False, "summary": None, "error": None,
                           "findings": news or news_findings()},
    }  # fmt: skip
    return PendingRecord(
        kind="decision", asset_id=CID, chain="solana", address=MINT, component="analyze",
        observed_at=observed,
        payload={"agents": agents, "decision": {"action": action, "confidence": confidence,
                                                "risk": {"level": "medium"}}},
    )  # fmt: skip


# --- Safety V2 ---------------------------------------------------------------------------------


def rule(id: str, outcome: str, severity: str = "high") -> dict[str, Any]:
    return {"id": id, "outcome": outcome, "severity": severity, "decision_bearing": True,
            "evidence": [f"path.{id}"], "reason": f"{id} {outcome}"}  # fmt: skip


SAFETY_FLAGS = [
    rule("NOT_A_TOKEN_MINT", "NOT_TRIGGERED", "critical"),
    rule("UNEXPECTED_TOKEN_PROGRAM", "NOT_TRIGGERED", "critical"),
    rule("MALFORMED_MINT_ACCOUNT", "NOT_TRIGGERED", "critical"),
    rule("IDENTITY_MISMATCH", "NOT_TRIGGERED", "critical"),
    rule("MINT_AUTHORITY_ACTIVE", "TRIGGERED"),
    rule("FREEZE_AUTHORITY_ACTIVE", "NOT_TRIGGERED"),
    rule("TOP1_CONCENTRATION", "NOT_TRIGGERED", "medium"),
    rule("TOP10_CONCENTRATION", "TRIGGERED", "medium"),
    rule("FEW_HOLDERS", "UNDETERMINED", "medium"),
    rule("LIQUIDITY_COLLAPSE", "TRIGGERED"),
    rule("LOW_LIQUIDITY", "NOT_TRIGGERED", "medium"),
    rule("VERIFIED_DEPLOYER_HOLDS_SUPPLY", "NOT_TRIGGERED", "medium"),
]


def safety_body(
    as_of: datetime,
    rules_version: str = "4",
    flags: list[dict[str, Any]] | None = None,
    mint: str = MINT,
) -> dict[str, Any]:
    flags = copy.deepcopy(flags if flags is not None else SAFETY_FLAGS)
    return {
        "schema_version": "safety.snapshot.v2",
        "rules_version": rules_version,
        "as_of": iso(as_of),
        "identity": {"canonical_id": f"solana:{mint}", "chain": "solana", "mint": mint,
                     "token_mint": {"status": "VERIFIED", "reason": "parsed mint"},
                     "pool_match": None},
        "authority": {"mint": {"status": "AVAILABLE", "value": "Auth"}},
        "holders": {"status": "COLLECTED"}, "market": {"status": "POOLS"},
        "creator": {"source_status": "CAPTURED"}, "changes": {"status": "UNAVAILABLE"},
        "flags": flags,
        "undetermined": [{"id": f["id"], "needs": "a complete holder scan", "reason": f["reason"]}
                         for f in flags if f["outcome"] == "UNDETERMINED"],
        "assessment": {"band": "ELEVATED_EVIDENCE", "coverage": "PARTIAL"},
        "coverage": {
            "coverage": "PARTIAL", "reasons": ["FEW_HOLDERS is undetermined"],
            "components": {"holders": "COLLECTED"},
            "not_supported": [{"id": "TOKEN_2022_EXTENSION_RISK", "reason": "deferred"}],
            "out_of_scope": [{"id": "CONCENTRATION_RISING", "reason": "no earlier scan"},
                             {"id": "LARGE_HOLDER_EXIT", "reason": "no earlier scan"},
                             {"id": "RAPID_HOLDER_LOSS", "reason": "no earlier scan"}],
        },
    }  # fmt: skip


def safety_db(path: Path, rows: list[tuple[datetime, str, dict[str, Any]]], cid: str = CID) -> Path:
    """A bare ``safety_snapshots`` table (only what Opportunity reads)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS safety_snapshots (id INTEGER PRIMARY KEY, canonical_id TEXT, "
        "as_of REAL, schema_version TEXT, rules_version TEXT, fingerprints_json TEXT, "
        "coverage TEXT, band TEXT, body_zlib BLOB, body_hash TEXT)"
    )
    for as_of, version, body in rows:
        _, blob, digest = encode_body(body)
        conn.execute(
            "INSERT INTO safety_snapshots (canonical_id, as_of, schema_version, rules_version, "
            "fingerprints_json, coverage, band, body_zlib, body_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (cid, as_of.timestamp(), "4", version, json.dumps({}), "PARTIAL",
             "ELEVATED_EVIDENCE", blob, digest),
        )  # fmt: skip
    conn.commit()
    conn.close()
    return path


def build(
    archive: Path,
    safety: Path,
    decision_at: datetime = T,
    origin: OpportunityOrigin = "HISTORICAL_REPLAY",
    cid: str = CID,
) -> OpportunityInput:
    a, s = ArchiveReader(archive), SafetyReader(safety)
    try:
        return build_input(cid, decision_at, origin, a, s)
    finally:
        a.close()
        s.close()
