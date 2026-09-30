"""What a shadow strategy may see: archived production evidence, point in time.

The only inputs are Evidence Archive records (read-only), never a provider request:

* ``scout`` records: Growth Scout's evaluation of one exact token at its decision time D
  (score, stage, liquidity, risk flags, safety, social, snapshot Technical context, the
  selected pool and its price, observed at T <= D);
* ``market`` records: the exact pool's observed price at its observation time;
* ``decision`` records (Analyze: Opportunity action / confidence, Technical trend, Risk
  level), looked up only at or before a Scout decision time.

Events are replayed strictly in time order (`observed_at`, then archive id). An event is
converted into a small immutable view before a strategy sees it; nothing later than the
event's own time is ever attached to it.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from upscale.services.evidence_archive.store import EvidenceRecord, EvidenceStore, Kind

PRICE_KINDS: tuple[Kind, ...] = ("market", "scout")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def valid_price(value: Any) -> float | None:
    """A finite, strictly positive USD price, else None (never a price)."""
    v = _num(value)
    return v if v is not None and v > 0 else None


def _dig(d: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _time(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class PriceObs:
    """One observed price of an exact pool (the token's USD price in that pool)."""

    asset_id: str
    pool: str
    price: float
    at: datetime  # when the price was observed
    source: Literal["market", "scout"]
    record_id: str


@dataclass(frozen=True)
class ScoutView:
    """Growth Scout's evaluation of one exact token, final at `decision_at`."""

    record_id: str
    fingerprint: str
    asset_id: str
    chain: str | None
    address: str | None
    symbol: str | None
    pool: str | None
    dex: str | None
    decision_at: datetime
    market_observed_at: datetime | None
    price_usd: float | None
    liquidity_usd: float | None
    market_cap_usd: float | None
    score: float | None
    risk_penalty: float | None
    stage: str | None
    eligible: bool
    rank: int | None
    data_status: str | None
    safety_status: str | None
    market_status: str | None
    mint_authority_active: bool | None
    freeze_authority_active: bool | None
    holder_top10_pct: float | None
    spam_risk: str | None
    social_status: str | None
    technical: dict[str, Any] | None
    risk_flags: tuple[tuple[str, str, float], ...]  # (code, severity, penalty)
    causal_valid: bool | None
    timing_version: int | None
    evidence_links: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def price(self) -> PriceObs | None:
        """The embedded price: only a CURRENT observation of the selected pool."""
        if (
            self.data_status != "CURRENT"
            or self.pool is None
            or self.price_usd is None
            or self.market_observed_at is None
            or self.market_observed_at > self.decision_at
        ):
            return None
        return PriceObs(
            self.asset_id,
            self.pool,
            self.price_usd,
            self.market_observed_at,
            "scout",
            self.record_id,
        )

    def missing(self) -> frozenset[str]:
        """The same labels as Calibration: missing evidence is a label, never a zero."""
        out = set()
        if self.safety_status in (None, "INSUFFICIENT_SAFETY_DATA"):
            out.add("SAFETY_NOT_AVAILABLE")
        if self.social_status in (None, "SOCIAL_UNAVAILABLE"):
            out.add("SOCIAL_NOT_AVAILABLE")
        if self.market_cap_usd is None:
            out.add("MARKET_CAP_NOT_AVAILABLE")
        if self.liquidity_usd is None:
            out.add("LIQUIDITY_NOT_AVAILABLE")
        if self.technical is None:
            out.add("TECHNICAL_NOT_AVAILABLE")
        return frozenset(out)

    def summary(self) -> dict[str, Any]:
        return {
            "scout": {
                "score": self.score, "stage": self.stage, "eligible": self.eligible,
                "rank": self.rank, "data_status": self.data_status,
                "decision_at": self.decision_at.isoformat(),
                "market_observed_at": self.market_observed_at.isoformat()
                if self.market_observed_at else None,
                "price_usd": self.price_usd, "liquidity_usd": self.liquidity_usd,
                "market_cap_usd": self.market_cap_usd, "social_status": self.social_status,
                "spam_risk": self.spam_risk,
            },
            "technical": self.technical,
            "risk": {
                "risk_penalty": self.risk_penalty,
                "flags": [{"code": c, "severity": s, "penalty": p} for c, s, p in self.risk_flags],
                "safety_status": self.safety_status, "market_status": self.market_status,
                "mint_authority_active": self.mint_authority_active,
                "freeze_authority_active": self.freeze_authority_active,
                "holder_top10_pct": self.holder_top10_pct,
            },
            "missing": sorted(self.missing()),
        }  # fmt: skip

    def fingerprints(self) -> list[dict[str, Any]]:
        own = {"kind": "scout", "record_id": self.record_id, "fingerprint": self.fingerprint,
               "observed_at": self.decision_at.isoformat()}  # fmt: skip
        linked = [
            {k: x.get(k) for k in ("kind", "record_id", "fingerprint", "observed_at")}
            for x in self.evidence_links
        ]
        return [own, *linked]


@dataclass(frozen=True)
class AnalyzeView:
    """The latest archived Analyze decision for the asset, observed at or before a time."""

    record_id: str
    fingerprint: str
    observed_at: datetime
    action: str | None
    confidence: str | None
    risk_level: str | None
    technical_trend: str | None
    bullish_score: int | None
    bearish_score: int | None

    def summary(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at.isoformat(), "action": self.action,
            "confidence": self.confidence, "risk_level": self.risk_level,
            "technical_trend": self.technical_trend, "bullish_score": self.bullish_score,
            "bearish_score": self.bearish_score,
        }  # fmt: skip

    def fingerprint_link(self) -> dict[str, Any]:
        return {"kind": "decision", "record_id": self.record_id, "fingerprint": self.fingerprint,
                "observed_at": self.observed_at.isoformat()}  # fmt: skip


@dataclass(frozen=True)
class Event:
    at: datetime  # when this evidence became known (a Scout decision / a price observation)
    seq: int  # archive row id: the tie-breaker, and the resume cursor
    price: PriceObs | None = None
    scout: ScoutView | None = None


def scout_view(r: EvidenceRecord) -> ScoutView:
    c = r.payload.get("candidate") or {}
    timing = r.payload.get("timing") or {}
    flags = tuple(
        (str(f.get("code")), str(f.get("severity")), _num(f.get("penalty")) or 0.0)
        for f in c.get("risk_flags") or []
        if isinstance(f, dict)
    )
    technical = _dig(c, "momentum", "technical")
    rank = c.get("rank")
    return ScoutView(
        record_id=r.record_id,
        fingerprint=r.fingerprint,
        asset_id=r.asset_id,
        chain=r.chain,
        address=r.address,
        symbol=c.get("symbol"),
        pool=_dig(c, "market", "selected_pool", "address") or r.pool_address,
        dex=_dig(c, "market", "selected_pool", "dex") or r.dex,
        decision_at=r.observed_at,
        market_observed_at=_time(timing.get("market_observed_at")) or _time(c.get("observed_at")),
        price_usd=valid_price(_dig(c, "market", "price_usd")),
        liquidity_usd=_num(_dig(c, "market", "liquidity_usd")),
        market_cap_usd=_num(_dig(c, "market", "market_cap_usd")),
        score=_num(_dig(c, "scout_momentum", "score")),
        risk_penalty=_num(_dig(c, "scout_momentum", "risk_penalty")),
        stage=c.get("stage"),
        eligible=bool(c.get("eligible", False)),
        rank=rank if isinstance(rank, int) else None,
        data_status=c.get("data_status"),
        safety_status=_dig(c, "quality", "safety_status"),
        market_status=_dig(c, "quality", "market_status"),
        mint_authority_active=_dig(c, "quality", "mint_authority_active"),
        freeze_authority_active=_dig(c, "quality", "freeze_authority_active"),
        holder_top10_pct=_num(_dig(c, "quality", "holder_top10_pct")),
        spam_risk=_dig(c, "quality", "spam_risk"),
        social_status=_dig(c, "momentum", "social_status"),
        technical=technical if isinstance(technical, dict) else None,
        risk_flags=flags,
        causal_valid=r.links.get("causal_valid"),
        timing_version=timing.get("version") if isinstance(timing.get("version"), int) else None,
        evidence_links=tuple(x for x in r.links.get("evidence") or [] if isinstance(x, dict)),
    )


def market_price(r: EvidenceRecord) -> PriceObs | None:
    c = r.payload.get("candidate") or {}
    price = valid_price(_dig(c, "metrics", "price_usd"))
    pool = _dig(c, "pool", "address") or r.pool_address
    if price is None or not pool or r.availability != "AVAILABLE":
        return None
    return PriceObs(r.asset_id, pool, price, r.observed_at, "market", r.record_id)


def analyze_view(r: EvidenceRecord) -> AnalyzeView:
    decision = r.payload.get("decision") or {}
    agents = r.payload.get("agents") or {}
    trend = _dig(agents, "technical", "findings", "trend", "label")
    return AnalyzeView(
        record_id=r.record_id,
        fingerprint=r.fingerprint,
        observed_at=r.observed_at,
        action=decision.get("action"),
        confidence=decision.get("confidence"),
        risk_level=_dig(decision, "risk", "level"),
        technical_trend=trend if isinstance(trend, str) else None,
        bullish_score=decision.get("bullish_score"),
        bearish_score=decision.get("bearish_score"),
    )


class EvidenceTimeline:
    """An ordered, resumable scan over the archive (read-only)."""

    def __init__(self, store: EvidenceStore, batch: int = 2000):
        self.store = store
        self.batch = batch
        self.skipped: dict[str, int] = {}

    def _skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def events(self, cursor: tuple[float, int], until: datetime) -> Iterator[Event]:
        """Every usable event after `cursor`, observed at or before `until`, in order."""
        while True:
            records = self.store.after(PRICE_KINDS, cursor, until, self.batch)
            for r in records:
                cursor = (r.observed_at.timestamp(), r.id)
                if r.kind == "market":
                    price = market_price(r)
                    if price is None:
                        self._skip("market record without a valid exact-pool price")
                        continue
                    yield Event(at=r.observed_at, seq=r.id, price=price)
                else:
                    yield Event(at=r.observed_at, seq=r.id, scout=scout_view(r))
            if len(records) < self.batch:
                return

    def analyze(self, asset_id: str, at: datetime, max_age: timedelta) -> AnalyzeView | None:
        """The latest Analyze decision for the exact asset observed in [at - max_age, at]."""
        r = self.store.latest("decision", asset_id, until=at, since=at - max_age)
        if r is None:
            return None
        if r.observed_at > at:  # the store guards this too; never trust a single layer
            raise AssertionError("an Analyze decision later than the decision time was returned")
        return analyze_view(r)
