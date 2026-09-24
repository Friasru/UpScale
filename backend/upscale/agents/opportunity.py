from upscale.agents.base import Agent, AgentContext
from upscale.formatting import usd, usd_zone
from upscale.schemas import AgentResult, Risk
from upscale.services.opportunity import (
    OpportunityAssessment,
    OpportunityConfig,
    Trigger,
    assess,
)
from upscale.services.risk import profile_from_results

ACTION_LABELS = {"buy": "BUY", "sell": "SELL", "wait": "WAIT"}


class OpportunityAgent(Agent):
    """One primary action for the asset: BUY, SELL or WAIT.

    Reviews the evidence the Technical Analysis, Market, News & Sentiment, Risk and Vision
    agents already produced; it makes no network or model calls. The action, triggers,
    invalidation and confidence are decided by the deterministic rules in
    `upscale.services.opportunity`. SELL means reduce or exit a long position, never open a
    short. WAIT is returned whenever the evidence doesn't clearly support acting.
    """

    name = "opportunity"
    description = (
        "BUY / SELL / WAIT decision support from the other agents' evidence, with trigger, "
        "invalidation, risk and confidence."
    )
    depends_on = ("vision", "technical_analysis", "market", "news_sentiment", "risk")

    def __init__(self, config: OpportunityConfig | None = None):
        self.config = config

    async def run(self, context: AgentContext) -> AgentResult:
        profile = profile_from_results(
            context.prior_results, context.primary_asset, context.asset_identity
        )
        a = assess(context.prior_results, context.primary_asset, self.config, profile)
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=render_decision(a),
            findings=a.model_dump(mode="json"),
            evidence=_evidence(a),
            risks=_risks(a),
        )


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _trigger_line(label: str, trigger: Trigger | None) -> str | None:
    if trigger is None:
        return None
    line = f"{label}: {trigger.condition}"
    if trigger.live_price_beyond:
        line += " (live price is already past it; the close decides)"
    return line


def render_decision(a: OpportunityAssessment) -> str:
    """The short decision block shown first in the chat reply."""
    action = ACTION_LABELS[a.action]
    lines = [action, a.summary]
    if a.action == "wait":
        for line in (
            _trigger_line("Buy trigger", a.bullish_trigger),
            _trigger_line("Sell trigger", a.bearish_trigger),
        ):
            if line:
                lines.append(line)
    else:
        trigger = a.bullish_trigger if a.action == "buy" else a.bearish_trigger
        if trigger is not None:
            lines.append(f"Trigger: {trigger.condition}")
        if a.entry_zone is not None:
            lines.append(
                f"{'Entry' if a.action == 'buy' else 'Exit'} zone: {usd_zone(a.entry_zone.lower, a.entry_zone.upper)} ({a.entry_basis})"
            )
        if a.invalidation is not None:
            lines.append(f"Invalidation: {a.invalidation.condition}")
        if a.action == "sell":
            lines.append(f"Note: {a.sell_meaning}")
    risk = "unavailable" if a.risk_level == "unavailable" else a.risk_level
    lines.append(f"Risk: {_cap(risk)} · Confidence: {_cap(a.confidence)}")
    return "\n".join(lines)


def _evidence(a: OpportunityAssessment) -> list[str]:
    lines = [
        f"Decision: {ACTION_LABELS[a.action]} on {a.asset or 'no asset'}"
        f"{f' {a.timeframe}' if a.timeframe else ''} (bullish {a.bullish_score} vs bearish "
        f"{a.bearish_score} points; confidence {a.confidence}). Rule-based, not a forecast."
    ]
    lines += [f"Bullish: {_cap(s.clause)}." for s in a.bullish_evidence]
    lines += [f"Bearish: {_cap(s.clause)}." for s in a.bearish_evidence]
    lines += [f"Blocking: {_cap(f.reason)}." for f in a.blocking_factors]
    lines += [f"Caution: {_cap(f.reason)}." for f in a.cautions]
    lines += [f"Also invalidated by: {i.condition}." for i in a.other_invalidations]
    if a.last_close is not None and a.timeframe:
        lines.append(f"Last {a.timeframe} close: {usd(a.last_close)}.")
    lines += a.context
    lines += [f"Missing evidence: {m}." for m in a.missing_evidence]
    return lines


def _risks(a: OpportunityAssessment) -> list[Risk]:
    risks: list[Risk] = []
    if a.action != "wait":
        risks.append(
            Risk(
                description=(
                    f"A confirmed {a.timeframe} setup can still fail; the invalidation level "
                    "is where this read stops applying."
                ),
                severity="medium",
            )
        )
    if a.risk_level == "unavailable":
        risks.append(
            Risk(
                description="No risk review was available, so the decision used stricter rules.",
                severity="medium",
            )
        )
    return risks
