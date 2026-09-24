import asyncio
from datetime import datetime
from typing import Any

from upscale.agents.base import Agent, AgentContext
from upscale.schemas import AgentResult, Risk
from upscale.services import news_service
from upscale.services.news import NewsError, NewsReport, NewsService, Story

MAX_ASSETS = 3
MAX_EVIDENCE_STORIES = 6
TONE_NOTE = "This describes the tone of news coverage, not a price forecast."
NO_CAUSATION = (
    "News timing alone does not show that any story caused a price move; "
    "news sentiment is not a buy or sell signal."
)


class NewsSentimentAgent(Agent):
    """Recent real crypto news for the requested assets, with evidence-based sentiment.

    All I/O goes through `NewsService`: articles come from a news provider, and sentiment
    labels are attached only to those articles. Price movement (from the market agent, when
    it ran) is reported separately and never used to label news sentiment.
    """

    name = "news_sentiment"
    description = (
        "Recent crypto news from real sources: sentiment tied to the articles, potential "
        "impact, publication times and staleness."
    )
    # Vision can supply the asset; market supplies price movement to report *separately*.
    depends_on = ("vision", "market")
    timeout = 90.0  # feed fetches plus a model call to label the articles

    def __init__(self, service: NewsService | None = None):
        self.service = service or news_service

    async def run(self, context: AgentContext) -> AgentResult:
        targets: list[str | None] = list(context.assets[:MAX_ASSETS]) or [None]
        outcomes = await asyncio.gather(
            *(self.service.get_report(t) for t in targets), return_exceptions=True
        )
        reports: list[NewsReport] = []
        unavailable: list[dict[str, str]] = []
        for target, outcome in zip(targets, outcomes, strict=True):
            if isinstance(outcome, NewsReport):
                reports.append(outcome)
            elif isinstance(outcome, NewsError):
                unavailable.append({"asset": target or "crypto market", "reason": str(outcome)})
            else:
                raise outcome  # unexpected bug: let the orchestrator record the failure

        price_context = _price_context(context)
        findings: dict[str, Any] = {
            "provider": self.service.provider_name,
            "reports": [_report_findings(r) for r in reports],
            "unavailable": unavailable,
            "price_context": price_context,
            "price_note": (
                "Price movement is shown separately and is not used to label news sentiment."
            ),
        }
        if not reports:
            reasons = "; ".join(f"{u['asset']}: {u['reason']}" for u in unavailable)
            return AgentResult(
                agent=self.name,
                status="error",
                mock=False,
                summary="Recent news could not be retrieved, so no news sentiment is available.",
                findings=findings,
                error=reasons,
            )

        evidence = [line for r in reports for line in _evidence(r)]
        evidence += [
            f"Price context (separate from news sentiment): {p['statement']}" for p in price_context
        ]
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=" ".join(_summary(r) for r in reports),
            findings=findings,
            evidence=evidence,
            risks=_risks(reports, unavailable, bool(price_context)),
        )


def _subject(report: NewsReport) -> str:
    return report.asset or "the crypto market"


def _stamp(when: datetime) -> str:
    return when.strftime("%Y-%m-%d %H:%M UTC")


def _age(hours: float) -> str:
    if hours < 1:
        return f"{max(1, round(hours * 60))}m ago"
    if hours < 48:
        return f"{round(hours)}h ago"
    return f"{round(hours / 24)}d ago"


def _story_line(s: Story) -> str:
    label: str = s.sentiment or "unclassified"
    if s.impact:
        label += f", {s.impact} impact"
    when = f"{_stamp(s.published_at)}, {_age(s.age_hours)}{', stale' if s.stale else ''}"
    also = f"; also reported by {', '.join(s.also_reported_by)}" if s.also_reported_by else ""
    scope = " [market-wide]" if s.scope == "market" else ""
    line = f'[{label}]{scope} {s.source} ({when}{also}): "{s.title}"'
    return line + (f" — {s.impact_reason}" if s.impact_reason else "")


def _story_findings(s: Story) -> dict[str, Any]:
    return s.model_dump(mode="json") | {"age": _age(s.age_hours)}


def _report_findings(r: NewsReport) -> dict[str, Any]:
    def lines(*sentiments: str) -> list[str]:
        return [_story_line(s) for s in r.stories if s.sentiment in sentiments]

    impact_order = {"high": 0, "medium": 1}
    impactful = sorted(
        (s for s in r.stories if s.impact in impact_order),
        key=lambda s: impact_order[s.impact or ""],
    )
    return {
        "asset": r.asset,
        "subject": _subject(r),
        "overall_news_sentiment": r.overall_sentiment,
        "market_news_sentiment": r.market_news_sentiment,
        "sentiment_basis": r.sentiment_basis,
        "market_sentiment_basis": r.market_sentiment_basis,
        "sentiment_counts": r.sentiment_counts,
        "asset_story_count": r.asset_story_count,
        "retrieved_at": r.retrieved_at.isoformat(),
        "analyzed_at": r.analyzed_at.isoformat(),
        "provider": r.provider,
        "sources": r.sources,
        "failed_sources": r.failed_sources,
        "sentiment_model": r.model,
        "stories": [_story_findings(s) for s in r.stories],
        "bullish_evidence": lines("bullish"),
        "bearish_evidence": lines("bearish"),
        "neutral_mixed_evidence": lines("neutral", "mixed"),
        "impact_events": [_story_line(s) for s in impactful],
        "conflicting_evidence": r.conflicts,
        "uncertainty": r.uncertainties,
        "stale_story_count": sum(s.stale for s in r.stories),
        "excluded": r.excluded,
        "discarded": r.discarded,
    }


def _summary(r: NewsReport) -> str:
    subject = _subject(r)
    if not r.stories:
        return f"No recent relevant news about {subject} was found (insufficient data)."
    sources = sorted({s.source for s in r.stories})
    text = (
        f"News for {subject}: {len(r.stories)} recent article(s) from {', '.join(sources)}"
        f" (retrieved {_stamp(r.retrieved_at)})"
    )
    if r.asset is not None and r.asset_story_count == 0:
        return text + f"; none specifically about {r.asset}, market-wide stories only."
    text += f"; overall news sentiment: {r.overall_sentiment.replace('_', ' ')}."
    return text + (f" {r.sentiment_basis}" if r.sentiment_basis else "")


def _evidence(r: NewsReport) -> list[str]:
    subject = _subject(r)
    if not r.stories:
        return [
            f"No recent relevant news about {subject} in {', '.join(r.sources)} "
            f"(retrieved {_stamp(r.retrieved_at)}); no headlines are shown rather than invented."
        ]
    counts = ", ".join(f"{n} {k}" for k, n in r.sentiment_counts.items() if n)
    focus = "asset-specific" if r.asset is not None else "market"
    lines = [
        f"News sentiment for {subject}: {r.overall_sentiment.replace('_', ' ')} "
        f"({counts or 'no'} {focus} article(s); retrieved {_stamp(r.retrieved_at)}). {TONE_NOTE}"
    ]
    if r.sentiment_basis:
        lines.append(r.sentiment_basis)
    if r.asset is not None and r.market_news_sentiment != "insufficient_data":
        market = f"Market-wide news sentiment: {r.market_news_sentiment.replace('_', ' ')}."
        if r.market_sentiment_basis:
            market += f" {r.market_sentiment_basis}"
        lines.append(market)
    lines += [_story_line(s) for s in r.stories[:MAX_EVIDENCE_STORIES]]
    lines += [f"Conflicting news: {c}" for c in r.conflicts]
    lines += [f"News uncertainty: {u}" for u in r.uncertainties]
    return lines


def _price_context(context: AgentContext) -> list[dict[str, Any]]:
    market = context.prior_results.get("market")
    if not market or market.status != "ok":
        return []
    context_rows: list[dict[str, Any]] = []
    for snap in market.findings.get("snapshots", []):
        change = snap.get("change_24h_pct")
        if not isinstance(change, int | float):
            continue
        context_rows.append(
            {
                "symbol": snap.get("symbol"),
                "change_24h_pct": change,
                "provider": snap.get("provider"),
                "statement": (
                    f"{snap.get('symbol')} moved {change:+.2f}% over 24h ({snap.get('provider')})."
                ),
            }
        )
    return context_rows


def _risks(
    reports: list[NewsReport], unavailable: list[dict[str, str]], has_price: bool
) -> list[Risk]:
    risks: list[Risk] = []
    if unavailable:
        missing = ", ".join(f"{u['asset']} ({u['reason']})" for u in unavailable)
        risks.append(Risk(description=f"News unavailable for {missing}.", severity="medium"))
    failed = {f"{f['source']} ({f['reason']})" for r in reports for f in r.failed_sources}
    if failed:
        risks.append(
            Risk(
                description=f"Some news sources could not be read: {', '.join(sorted(failed))}.",
                severity="low",
            )
        )
    if any(s.stale for r in reports for s in r.stories):
        risks.append(
            Risk(description="Some news items are stale and may be outdated.", severity="low")
        )
    if any(s.sentiment is None for r in reports for s in r.stories):
        risks.append(
            Risk(
                description="Some articles could not be classified, so news sentiment is partial.",
                severity="medium",
            )
        )
    if any(not r.stories for r in reports):
        risks.append(
            Risk(
                description="Little or no recent news was found; news coverage is insufficient.",
                severity="low",
            )
        )
    risks.append(
        Risk(
            description=NO_CAUSATION
            if has_price
            else "News sentiment is not a buy or sell signal.",
            severity="low",
        )
    )
    return risks
