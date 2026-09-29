"""Scout for the UpScale app: Growth Scout's ranking as a compact, normalized view.

The frontend renders this and never re-derives Growth Scout logic. Scout Momentum is a
discovery ranking, never a trade confidence: BUY / SELL / WAIT comes only from Analyze,
which receives the exact token (`analyze`: chain + contract / mint) through `/chat`.

* `ScoutFeed` keeps the latest full ranking. Refreshes are single-flight (concurrent
  requests share one scan) and rate-limited (`min_refresh_seconds`), so the UI can't
  hammer providers; Scout's own provider limits apply underneath.
* `build_view` applies the UI's simple filters (chain, stage, minimum liquidity) to that
  ranking (never re-scoring anything), takes the top N, and turns each candidate into a
  card plus "See more" sections.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from upscale.schemas import AssetRef
from upscale.services.chains import chain_label
from upscale.services.outcomes.models import SurfacingHistory
from upscale.services.scout.growth.models import (
    NOT_A_TRADE_SIGNAL,
    GrowthCandidate,
    GrowthScoutResult,
    GrowthStage,
)

logger = logging.getLogger("upscale.scout")

# The UI offers the top 10 or top 20 (never hundreds of cards).
SCOUT_LIMITS = (10, 20)
StageFilter = Literal["ACCELERATING", "EARLY", "NEW"]
ChainFilter = Literal["solana", "ethereum", "base", "bsc"]

# Stage order the view groups by (the backend's own stage names, unchanged).
STAGE_ORDER: tuple[GrowthStage, ...] = (
    "ACCELERATING",
    "EARLY",
    "NEW",
    "STEADY",
    "CROWDED",
    "FADING",
    "INSUFFICIENT_DATA",
)
SAFETY_LABELS = {
    "SAFETY_CHECKS_COMPLETE": "Safety checks complete",
    "SAFETY_CHECKS_PARTIAL": "Safety checks partial",
    "INSUFFICIENT_SAFETY_DATA": "Insufficient safety data",
}
# Risk flags serious enough to show on the card itself (others are under "See more").
SERIOUS_FLAGS = {
    "freeze_authority_active": "Freeze authority active",
    "mint_authority_active": "Mint authority active",
    "holder_concentration_top1": "High holder concentration",
    "holder_concentration_top10": "High holder concentration",
    "market_collapse": "Market collapse",
    "thin_market_pump": "Thin-market pump",
    "flow_divergence": "Flow divergence",
    "social_only_hype": "Social-only hype",
    "stale_market_data": "Stale market data",
    "distribution": "Distribution",
    "liquidity_draining": "Liquidity draining",
    "social_spam": "Social spam",
}
PROVIDER_STATES = {
    "PROVIDER_OK": "searched",
    "PROVIDER_CHECKED_ZERO_MATCHES": "searched, no mentions",
    "PROVIDER_UNAVAILABLE": "unavailable",
    "PROVIDER_NOT_CONFIGURED": "not configured",
}
SOCIAL_LABELS = {
    "SOCIAL_UNAVAILABLE": "Unavailable",
    "SOCIAL_QUIET": "Quiet",
    "SOCIAL_EMERGING": "Emerging",
    "SOCIAL_ACCELERATING": "Accelerating",
    "SOCIAL_STRONG": "Strong",
    "SOCIAL_STEADY": "Steady",
    "SOCIAL_SATURATED": "Saturated",
    "SOCIAL_FADING": "Fading",
}
SCORE_LABELS = {
    "market_activity": "Market activity",
    "liquidity_quality": "Liquidity quality",
    "social_momentum": "Social momentum",
    "earliness": "Earliness",
    "cross_confirmation": "Cross-confirmation",
}


# --- View models ----------------------------------------------------------------------------


class ScoutFlag(BaseModel):
    code: str
    label: str
    severity: str


class ScoutSafety(BaseModel):
    status: str  # the backend status, unchanged
    label: str  # "Safety checks complete" / "Safety checks partial" / "Insufficient ..."
    flags: list[ScoutFlag] = Field(default_factory=list)  # serious flags, card-visible


class ScoutFreshness(BaseModel):
    status: Literal["CURRENT", "STALE_CARRIED"]
    observed_at: datetime  # when this market evidence was observed
    snapshot_age_minutes: float  # at the time this view was built


class ScoutMarket(BaseModel):
    market_cap_usd: float | None  # only when trustworthy
    fdv_usd: float | None
    liquidity_usd: float | None
    age_hours: float | None  # oldest pool


class ScoutRow(BaseModel):
    label: str
    value: str


class ScoutSection(BaseModel):
    title: str
    rows: list[ScoutRow]


class ScoutCard(BaseModel):
    rank: int
    canonical_id: str
    chain: str
    chain_label: str
    address: str
    symbol: str | None
    name: str | None
    stage: GrowthStage  # the backend stage, never renamed
    stage_reasons: list[str]
    scout_momentum: int  # discovery ranking 0..100 (not a trade confidence)
    market: ScoutMarket
    safety: ScoutSafety
    freshness: ScoutFreshness
    reasons: list[str]  # Growth Scout's own surfaced reasons
    details: list[ScoutSection]  # "See more"
    analyze: AssetRef  # the exact identity Analyze receives


class ScoutFilters(BaseModel):
    chain: ChainFilter | None = None
    stage: StageFilter | None = None
    min_liquidity_usd: float | None = Field(default=None, ge=0)


class ScoutView(BaseModel):
    status: Literal["ok", "empty", "unavailable"]
    computed_at: datetime | None
    mode: str | None
    limit: int
    filters: ScoutFilters
    ranked: int  # ranked candidates before the UI's filters
    matching: int  # after the filters (the view shows the top `limit` of these)
    candidates: list[ScoutCard]
    warnings: list[str] = Field(default_factory=list)  # friendly, non-fatal issues
    refreshing: bool = False
    error: str | None = None
    disclaimer: str = NOT_A_TRADE_SIGNAL


# --- Building the view ----------------------------------------------------------------------


def _money(x: float | None) -> str:
    if x is None:
        return "n/a"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"${x / div:.1f}{unit}"
    return f"${x:,.0f}"


def _price(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:.4g}" if x < 1 else f"${x:,.2f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.0f}%"


def _ratio(x: float | None, basis: str | None) -> str:
    if x is None:
        return "not measurable yet"
    return f"{x:.2f}x" + (f" ({basis})" if basis else "")


def _share(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _minutes(m: float) -> str:
    if m < 1:
        return "under a minute"
    if m < 120:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


def _age(hours: float | None) -> str:
    if hours is None:
        return "unknown"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.0f} days"


def _safety(g: GrowthCandidate) -> ScoutSafety:
    flags: dict[str, ScoutFlag] = {}
    for f in g.risk_flags:
        label = SERIOUS_FLAGS.get(f.code)
        if label and label not in flags:
            flags[label] = ScoutFlag(code=f.code, label=label, severity=f.severity)
    if g.quality.liquidity_quality == "thin" and "Thin liquidity" not in flags:
        flags["Thin liquidity"] = ScoutFlag(code="thin_liquidity", label="Thin liquidity",
                                            severity="caution")  # fmt: skip
    status = g.quality.safety_status
    return ScoutSafety(status=status, label=SAFETY_LABELS[status], flags=list(flags.values()))


def _details(
    g: GrowthCandidate,
    age_minutes: float,
    history: SurfacingHistory | None = None,
    computed_at: datetime | None = None,
) -> list[ScoutSection]:
    m, mo, q, sm = g.market, g.momentum, g.quality, g.scout_momentum
    pool = m.selected_pool
    identity = [
        ScoutRow(label="Canonical id", value=g.canonical_id),
        ScoutRow(label="Chain", value=chain_label(g.chain)),
        ScoutRow(label="Mint" if g.chain == "solana" else "Contract", value=g.address),
        ScoutRow(
            label="Selected market",
            value=f"{pool.dex} pool {pool.address} ({pool.quote_symbol or pool.quote_kind} quote)",
        ),
        ScoutRow(label="Market data from", value=m.market_provider),
    ]
    cap = (
        _money(m.market_cap_usd)
        if m.market_cap_usd is not None
        else f"n/a ({m.market_cap_note})"
        if m.market_cap_note
        else "n/a"
    )
    buy = (
        f"{_share(mo.buy_share)} of trades are buys"
        + (f", {_share(mo.buyer_share)} of wallets buying" if mo.buyer_share is not None else "")
        + (f", change {mo.buy_pressure_change:+.0%}" if mo.buy_pressure_change is not None else "")
        + f" (trade counts; flow {q.flow_quality.replace('_', '-')})"
    )
    market = [
        ScoutRow(label="Price", value=_price(m.price_usd)),
        ScoutRow(label="Market cap", value=cap),
        ScoutRow(label="FDV", value=_money(m.fdv_usd)),
        ScoutRow(label="Liquidity", value=_money(m.liquidity_usd)),
        ScoutRow(
            label="Volume acceleration",
            value=_ratio(mo.volume_acceleration, mo.volume_acceleration_basis),
        ),
        ScoutRow(
            label="Trade acceleration",
            value=_ratio(mo.txn_acceleration, mo.txn_acceleration_basis),
        ),
        ScoutRow(label="Buy pressure", value=buy if mo.buy_share is not None else "n/a"),
        ScoutRow(
            label="Liquidity change",
            value=f"{_pct(mo.liquidity_change_pct)} {mo.liquidity_change_basis or ''}".strip()
            if mo.liquidity_change_pct is not None
            else "no stored history yet",
        ),
        ScoutRow(
            label="Price change",
            value=f"{_pct(mo.price_change_h1_pct)} 1h, {_pct(mo.price_change_h24_pct)} 24h",
        ),
        ScoutRow(label="Pool age", value=_age(m.oldest_pool_age_hours or m.pool_age_hours)),
    ]
    for note in q.flow_notes:
        market.append(ScoutRow(label="Flow caution", value=note))
    social = [
        ScoutRow(label="Social momentum", value=SOCIAL_LABELS[mo.social_status]),
    ]
    for p in mo.social_providers:
        state = PROVIDER_STATES.get(p.status, p.status)
        detail = f" ({p.detail})" if p.detail and p.status == "PROVIDER_UNAVAILABLE" else ""
        social.append(ScoutRow(label=p.provider, value=state + detail))
    if mo.social_status == "SOCIAL_UNAVAILABLE":
        social.append(ScoutRow(label="Note", value="Unavailable is not zero attention"))
    social += [
        ScoutRow(label="Attribution quality", value=q.social_attribution),
        ScoutRow(label="Spam risk", value=q.spam_risk),
        ScoutRow(
            label="Cross-platform confirmation",
            value="n/a"
            if mo.cross_platform_corroborated is None
            else "confirmed on several platforms"
            if mo.cross_platform_corroborated
            else "not confirmed",
        ),
    ]
    scoring = [
        ScoutRow(label=SCORE_LABELS[f.family], value=f"{f.contribution:.1f}") for f in sm.families
    ] + [
        ScoutRow(label="Base score", value=f"{sm.base:.1f}"),
        ScoutRow(label="Stage adjustment", value=f"{sm.stage_adjustment:+.1f}"),
        ScoutRow(label="Risk penalty", value=f"{-sm.risk_penalty:.1f}"),
        ScoutRow(label="Scout Momentum", value=f"{sm.score:.0f}"),
    ]
    safety = [ScoutRow(label="Status", value=SAFETY_LABELS[q.safety_status])]
    safety += [
        ScoutRow(label=f"Flag ({f.severity})", value=f.detail)
        for f in g.risk_flags
        if f.code != "social_unavailable"
    ]
    lower = " (at least; holder scan incomplete)" if q.holder_data_lower_bound else ""
    if q.holder_top1_pct is not None:
        safety.append(ScoutRow(label="Largest holder", value=f"{q.holder_top1_pct:.1f}%{lower}"))
    if q.holder_top10_pct is not None:
        safety.append(ScoutRow(label="Top 10 holders", value=f"{q.holder_top10_pct:.1f}%{lower}"))
    for label, active in (
        ("Mint authority", q.mint_authority_active),
        ("Freeze authority", q.freeze_authority_active),
    ):
        if active is not None:
            safety.append(ScoutRow(label=label, value="active" if active else "revoked"))
    safety += [ScoutRow(label="Unknown", value=x) for x in q.safety_missing]
    freshness = [
        ScoutRow(label="Observed", value=g.observed_at.astimezone(UTC).strftime("%H:%M UTC")),
        ScoutRow(label="Snapshot age", value=_minutes(age_minutes)),
        ScoutRow(
            label="This run",
            value="refreshed this run"
            if g.data_status == "CURRENT"
            else "carried forward on the last good observation (provider unavailable)",
        ),
    ]
    if history is not None and computed_at is not None:
        # Ranking runs before this one (the current run is already counted).
        earlier = history.times_ranked - (1 if history.first_ranked_at <= computed_at else 0)
        freshness += [
            ScoutRow(
                label="Previously surfaced",
                value=f"{earlier} time{'s' if earlier != 1 else ''}"
                if earlier > 0
                else "first time",
            ),
            ScoutRow(
                label="First surfaced",
                value=f"{_minutes((computed_at - history.first_ranked_at).total_seconds() / 60)} ago"
                if earlier > 0
                else "this run",
            ),
        ]
    return [
        ScoutSection(title="Identity", rows=identity),
        ScoutSection(title="Market", rows=market),
        ScoutSection(title="Social", rows=social),
        ScoutSection(title="Scoring", rows=scoring),
        ScoutSection(title="Safety", rows=safety),
        ScoutSection(title="Freshness", rows=freshness),
    ]


def card(
    g: GrowthCandidate,
    rank: int,
    now: datetime,
    history: SurfacingHistory | None = None,
    computed_at: datetime | None = None,
) -> ScoutCard:
    age = max(0.0, (now - g.observed_at).total_seconds() / 60)
    m = g.market
    return ScoutCard(
        rank=rank,
        canonical_id=g.canonical_id,
        chain=g.chain,
        chain_label=chain_label(g.chain),
        address=g.address,
        symbol=g.symbol,
        name=g.name,
        stage=g.stage,
        stage_reasons=g.stage_reasons,
        scout_momentum=round(g.score),
        market=ScoutMarket(
            market_cap_usd=m.market_cap_usd,
            fdv_usd=m.fdv_usd,
            liquidity_usd=m.liquidity_usd,
            age_hours=m.oldest_pool_age_hours or m.pool_age_hours,
        ),
        safety=_safety(g),
        freshness=ScoutFreshness(
            status=g.data_status, observed_at=g.observed_at, snapshot_age_minutes=round(age, 1)
        ),
        reasons=g.reasons_surfaced,
        details=_details(g, age, history, computed_at),
        analyze=AssetRef(
            chain=g.chain,
            address=g.address,
            symbol=g.symbol,
            name=g.name,
            pool_address=m.selected_pool.address,
        ),
    )


def warnings(result: GrowthScoutResult) -> list[str]:
    """Friendly, non-fatal issues: the screen still shows what could be ranked."""
    out = []
    unavailable = [c for c in result.social_checks if c.status == "PROVIDER_UNAVAILABLE"]
    if unavailable:
        parts = []
        for c in unavailable:
            why = (c.error or "").lower()
            reason = (
                "credits exhausted"
                if "credit" in why
                else "rate-limited"
                if "limit" in why or "budget" in why
                else "unavailable"
            )
            parts.append(f"{c.provider} ({reason})")
        out.append(
            "Some social sources are unavailable: " + ", ".join(parts) + ". Missing social "
            "data shows as unavailable, never as zero attention."
        )
    u = result.universe
    failed_feeds = sum(len(r.failed) for r in u.feeds) if u else 0
    if failed_feeds or any("source(s) failed" in n for n in result.notes):
        out.append("Some market data sources were partially unavailable this run.")
    if u and u.carried_stale:
        out.append(
            f"{u.carried_stale} candidate(s) use stale market data (their provider was "
            "unavailable); they are labeled and penalized."
        )
    if any("safety lookup failed" in n for n in result.notes):
        out.append("On-chain safety lookups were unavailable for some candidates.")
    return out


def build_view(
    result: GrowthScoutResult | None,
    limit: int,
    filters: ScoutFilters,
    now: datetime,
    refreshing: bool = False,
    error: str | None = None,
    history: Mapping[str, SurfacingHistory] | None = None,
) -> ScoutView:
    if result is None:
        return ScoutView(
            status="unavailable" if error else "empty",
            computed_at=None,
            mode=None,
            limit=limit,
            filters=filters,
            ranked=0,
            matching=0,
            candidates=[],
            refreshing=refreshing,
            error=error,
        )
    ranked = result.candidates  # already in Growth Scout's order
    matching = [
        g
        for g in ranked
        if (filters.chain is None or g.chain == filters.chain)
        and (filters.stage is None or g.stage == filters.stage)
        and (
            filters.min_liquidity_usd is None
            or (g.market.liquidity_usd or 0) >= filters.min_liquidity_usd
        )
    ]
    known = history or {}
    cards = [
        card(g, g.rank or i + 1, now, known.get(g.canonical_id), result.computed_at)
        for i, g in enumerate(matching[:limit])
    ]
    return ScoutView(
        status="ok" if cards else "empty",
        computed_at=result.computed_at,
        mode=result.mode,
        limit=limit,
        filters=filters,
        ranked=len(ranked),
        matching=len(matching),
        candidates=cards,
        warnings=warnings(result),
        refreshing=refreshing,
        error=error,
    )


# --- The feed ---------------------------------------------------------------------------------

Scan = Callable[[], Awaitable[GrowthScoutResult]]
History = Callable[[list[str]], Awaitable[dict[str, SurfacingHistory]]]


class ScoutFeed:
    """The latest Growth Scout ranking, refreshed on request (single-flight, rate-limited)."""

    def __init__(
        self,
        scan: Scan,
        min_refresh_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        history: History | None = None,
    ):
        self._scan = scan
        self._history = history
        self.min_refresh_seconds = min_refresh_seconds
        self._clock = clock
        self.now = now
        self.result: GrowthScoutResult | None = None
        self.error: str | None = None
        self._last_scan: float | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def refreshing(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        try:
            self.result = await self._scan()
            self.error = None
        except Exception as exc:  # the previous ranking stays usable
            logger.exception("Scout scan failed")
            self.error = f"Scout refresh failed ({type(exc).__name__}); showing the last results."
            if self.result is None:
                self.error = f"Scout refresh failed ({type(exc).__name__})."
        finally:
            self._last_scan = self._clock()

    async def refresh(self, force: bool = False) -> bool:
        """Run a scan unless one is running (then wait for it) or one finished less than
        `min_refresh_seconds` ago (then keep that result), so providers aren't hammered.
        Returns whether a scan ran (started here or joined), False if it was skipped."""
        if self.refreshing:
            assert self._task is not None
            await asyncio.shield(self._task)
            return True
        recent = (
            self._last_scan is not None
            and self._clock() - self._last_scan < self.min_refresh_seconds
        )
        if recent and not force:
            return False
        self._task = asyncio.create_task(self._run())
        await asyncio.shield(self._task)
        return True

    async def view(self, limit: int, filters: ScoutFilters) -> ScoutView:
        if self.result is None and self.error is None:
            await self.refresh()  # first request: one scan, shared by concurrent callers
        history = None
        if self._history is not None and self.result is not None:
            history = await self._history([g.canonical_id for g in self.result.candidates])
        return build_view(
            self.result, limit, filters, self.now(), self.refreshing, self.error, history
        )
