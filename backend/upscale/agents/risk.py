from upscale.agents.base import Agent, AgentContext
from upscale.schemas import AgentResult, Risk
from upscale.services.asset_profile import upper_first
from upscale.services.risk import (
    AGENT_LABELS,
    DexRiskConfig,
    RiskAssessment,
    RiskConfig,
    assess,
    category_label,
    profile_from_results,
)

MAX_SUMMARY_REASONS = 3


class RiskAgent(Agent):
    """What could make the current analysis unreliable or fail.

    Reviews the evidence the Vision, Technical Analysis, Market and News & Sentiment agents
    already produced; it makes no network or model calls. Every factor, severity, the
    overall risk level and the uncertainty level are decided by the deterministic rules in
    `upscale.services.risk`. It never recommends buying or selling.
    """

    name = "risk"
    description = (
        "Concrete risks, overall risk level, uncertainty and invalidation conditions derived "
        "from the other agents' evidence."
    )
    depends_on = ("vision", "technical_analysis", "market", "dex_market", "news_sentiment")

    def __init__(self, config: RiskConfig | None = None, dex_config: DexRiskConfig | None = None):
        self.config = config
        self.dex_config = dex_config

    async def run(self, context: AgentContext) -> AgentResult:
        profile = profile_from_results(
            context.prior_results, context.primary_asset, context.asset_identity
        )
        a = assess(
            context.prior_results, context.primary_asset, self.config, profile, self.dex_config
        )
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=_summary(a),
            findings={
                "reviewed_agents": sorted(context.prior_results),
                **a.model_dump(mode="json"),
            },
            evidence=_evidence(a),
            risks=[
                Risk(
                    description=f"{upper_first(category_label(f.category))}: {f.explanation}",
                    severity=f.severity,
                    source=f.source,
                )
                for f in a.factors
            ],
        )


def _clauses(items: list[str], capitalize: bool = True) -> str:
    shown = items[:MAX_SUMMARY_REASONS]
    text = "; ".join(shown)
    if len(items) > len(shown):
        text += f"; plus {len(items) - len(shown)} more"
    return text[:1].upper() + text[1:] if capitalize else text


def _summary(a: RiskAssessment) -> str:
    subject = f" for {a.asset}" if a.asset else ""
    text = f"Overall risk{subject}: {a.overall_risk}. {_clauses(a.overall_reasons)}."
    if a.overall_risk == "unknown" and a.missing_evidence:
        text += f" Missing: {'; '.join(a.missing_evidence)}."
    text += f" Uncertainty: {a.uncertainty_level}"
    if a.uncertainty_reasons:
        text += f" because {_clauses(a.uncertainty_reasons, capitalize=False)}"
    return text + "."


def _evidence(a: RiskAssessment) -> list[str]:
    reviewed = [AGENT_LABELS[i.agent].lower() for i in a.inputs if i.status == "ok"]
    risk_factors = [f for f in a.factors if f.affects == "risk"]
    counts = {lv: sum(f.severity == lv for f in risk_factors) for lv in ("high", "medium", "low")}
    lines = [
        f"Overall risk {a.overall_risk} from {sum(counts.values())} risk factor(s) "
        f"({counts['high']} high, {counts['medium']} medium, {counts['low']} low); "
        f"evidence reviewed: {', '.join(reviewed) or 'none'}. Rule-based, not a forecast."
    ]
    lines += [
        f"[{f.severity} · {category_label(f.category)}"
        f"{' · reliability' if f.affects == 'uncertainty' else ''}] {f.explanation}"
        for f in a.factors
    ]
    lines += [
        f"Would invalidate or weaken the analysis: {c.condition}" for c in a.invalidation_conditions
    ]
    lines += [f"Missing evidence: {m}." for m in a.missing_evidence]
    lines += [f"Asset profile: {note}" for note in a.profile_notes]
    return lines
