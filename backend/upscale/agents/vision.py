import asyncio

from upscale.agents.base import Agent, AgentContext
from upscale.formatting import usd
from upscale.schemas import AgentResult, Risk
from upscale.services import vision_service
from upscale.services.vision import ChartVision, VisionError, VisionService

VISUAL_NOTE = "read from the screenshot, not live data"


class VisionAgent(Agent):
    """Reads uploaded chart screenshots into structured visual evidence.

    It only reports what the screenshot shows (via `VisionService`); it never computes
    indicators or fetches market data. Its detected asset and timeframe are handed to
    downstream agents by the orchestrator when the user didn't specify them.
    """

    name = "vision"
    description = "Reads chart screenshots: asset, timeframe, indicators, levels, lines, patterns."
    timeout = 120.0  # a vision model call takes longer than a market-data lookup

    def __init__(self, service: VisionService | None = None):
        self.service = service or vision_service

    async def run(self, context: AgentContext) -> AgentResult:
        if not context.attachments:
            return AgentResult(
                agent=self.name,
                mock=False,
                summary="No screenshot was attached, so there was nothing to read.",
            )

        outcomes = await asyncio.gather(
            *(self.service.analyze(image) for image in context.attachments), return_exceptions=True
        )
        charts: list[ChartVision] = []
        failed: list[dict[str, str]] = []
        for image, outcome in zip(context.attachments, outcomes, strict=True):
            if isinstance(outcome, ChartVision):
                charts.append(outcome)
            elif isinstance(outcome, VisionError):
                failed.append({"image": image.name, "reason": str(outcome)})
            else:
                raise outcome  # unexpected bug: let the orchestrator record the failure

        primary = next((c for c in charts if c.reading.is_price_chart), None)
        findings = {
            "charts": [c.model_dump(mode="json") for c in charts],
            "failed": failed,
            "detected_asset": primary.reading.asset.symbol if primary else None,
            "detected_timeframe": primary.normalized_timeframe if primary else None,
            "detected_timeframe_label": primary.reading.timeframe.label if primary else None,
        }
        if not charts:
            reasons = "; ".join(f"{f['image']}: {f['reason']}" for f in failed)
            return AgentResult(
                agent=self.name,
                status="error",
                mock=False,
                summary="The screenshot could not be analyzed.",
                findings=findings,
                error=reasons,
            )

        evidence = [line for chart in charts for line in _describe(chart)]
        risks = [
            Risk(
                description=(
                    "Screenshot values are visual readings that may be outdated or misread; "
                    "live data takes precedence."
                ),
                severity="medium",
            )
        ]
        if failed:
            names = ", ".join(f"{f['image']} ({f['reason']})" for f in failed)
            risks.append(
                Risk(description=f"Some screenshots could not be read: {names}.", severity="medium")
            )
        mismatch = _asset_mismatch(context, primary)
        if mismatch:
            evidence.append(mismatch)
            risks.append(Risk(description=mismatch, severity="medium"))

        return AgentResult(
            agent=self.name,
            mock=False,
            summary=_summary(charts, failed),
            findings=findings,
            evidence=evidence,
            risks=risks,
        )


def _summary(charts: list[ChartVision], failed: list[dict[str, str]]) -> str:
    readable = [c for c in charts if c.reading.is_price_chart]
    if not readable:
        return "The screenshot does not appear to be a price chart."
    r = readable[0].reading
    asset = r.asset.pair or r.asset.symbol or "unknown asset"
    timeframe = r.timeframe.label or "unknown timeframe"
    text = f"Screenshot shows a {r.chart_type.replace('_', ' ')} chart of {asset} ({timeframe})."
    if len(charts) + len(failed) > 1:
        text += f" Read {len(charts)} of {len(charts) + len(failed)} screenshots."
    return text


def _describe(chart: ChartVision) -> list[str]:
    r = chart.reading
    name = chart.image_name
    if not r.is_price_chart:
        return [f"{name}: not recognized as a price chart."]

    asset = r.asset.pair or r.asset.symbol
    lines = [
        f"{name}: asset {asset} ({r.asset.basis})" if asset else f"{name}: asset unknown",
    ]
    if r.timeframe.label:
        mapped = f" = {chart.normalized_timeframe}" if chart.normalized_timeframe else ""
        lines[0] += f", timeframe {r.timeframe.label}{mapped} ({r.timeframe.basis})"
    else:
        lines[0] += ", timeframe unknown"
    lines[0] += f", {r.chart_type.replace('_', ' ')} chart."

    if r.displayed_price.value is not None:
        lines.append(
            f"Price shown on the screenshot: {usd(r.displayed_price.value)} ({VISUAL_NOTE})."
        )
    for ind in r.indicators:
        settings = f" {ind.settings}" if ind.settings else ""
        values = ", ".join(f"{v.label + ' ' if v.label else ''}{v.value:g}" for v in ind.values)
        shown = f": {values}" if values else " (no value printed)"
        lines.append(f"Indicator on screenshot: {ind.name}{settings}{shown} ({VISUAL_NOTE}).")
    for kind, levels in (("Support", r.support_levels), ("Resistance", r.resistance_levels)):
        for level in levels:
            if level.price is not None:
                lines.append(
                    f"{kind} marked on screenshot near {usd(level.price)} "
                    f"({level.basis}; {level.source.replace('_', ' ')})."
                )
    for d in r.drawn_levels:
        if d.price is not None:
            label = f" '{d.label}'" if d.label else ""
            lines.append(f"User-drawn horizontal level{label} at {usd(d.price)}.")
    for line in r.trend_lines:
        lines.append(
            f"{line.kind.capitalize()} ({line.direction}, {line.basis}): {line.description}."
        )
    for p in r.patterns:
        lines.append(f"Pattern visible: {p.name} ({p.evidence}).")
    lines += [f"Visual observation: {o}" for o in r.observations]
    lines += [f"Could not determine: {u}" for u in r.uncertainties]
    lines += [f"Discarded from vision output: {d}" for d in chart.discarded]
    return lines


def _asset_mismatch(context: AgentContext, primary: ChartVision | None) -> str | None:
    if primary is None or not context.assets or context.assets_source != "user":
        return None
    shown = primary.reading.asset.symbol
    asked = context.assets[0]
    if shown and shown != asked:
        return (
            f"The screenshot shows {shown}, but you asked about {asked}; "
            f"live analysis uses {asked}."
        )
    return None
