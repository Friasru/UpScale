"""Optional, bounded on-chain safety enrichment of Scout candidates (for Replay Lab).

Growth Scout already fetches on-chain safety for its leading Solana candidates
(`GrowthConfig.safety.top_k`) and those snapshots are archived for free. Enrichment adds
NEW requests through the existing `SolanaSafetyService` (its cache, rate limit and
de-duplication) for the next-best ranked Solana candidates that have no recent archived
safety evidence. It is the lowest-priority production work (below Analyze, due outcomes,
manual and background Scout), and it:

* is off unless ``UPSCALE_EVIDENCE_SAFETY_ENRICHMENT=1``;
* looks up at most ``UPSCALE_EVIDENCE_SAFETY_MAX_PER_REFRESH`` tokens per scan (default 3);
* waits `quiet_seconds` after the scan (a user often opens Analyze right after one), then
  stops at the first sign of higher-priority work (`busy`);
* leaves at least `min_free_calls` of the safety service's per-minute budget free for
  Analyze, and never retries within a scan.

Provider impact: one lookup is one chain-data fetch (mint account, largest accounts, owner
lookups, and with Helius a holder scan of up to ``UPSCALE_HELIUS_MAX_HOLDER_PAGES`` pages),
so 3 per scan x 48 background scans / day is about 144 fetches a day on top of production.
"""

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from upscale.services.evidence_archive import hooks as evidence
from upscale.services.evidence_archive.store import EvidenceStore
from upscale.services.market_data import MarketDataError
from upscale.services.scout.growth.models import GrowthScoutResult
from upscale.services.solana_chain import KnownPool, OnchainSafetySnapshot

logger = logging.getLogger("upscale.evidence")
_OFF = ("0", "false", "off", "no")
DEFAULT_MAX_PER_REFRESH = 3
MAX_PER_REFRESH_CAP = 20


class SafetySource(Protocol):
    async def get_snapshot(
        self, mint: str, pools: Sequence[KnownPool] = ()
    ) -> OnchainSafetySnapshot: ...

    def available_calls(self) -> int: ...


@dataclass(frozen=True)
class EnrichmentSettings:
    enabled: bool = False
    max_per_refresh: int = DEFAULT_MAX_PER_REFRESH
    refresh_hours: float = 6.0  # archived safety this recent needs no new lookup
    min_free_calls: int = 10  # of the safety service's per-minute calls, kept for Analyze
    quiet_seconds: float = 60.0


def load_settings(enabled: str | None, max_per_refresh: str | None) -> EnrichmentSettings:
    on = (enabled or "0").strip().lower() not in _OFF
    n = DEFAULT_MAX_PER_REFRESH
    if max_per_refresh is not None and max_per_refresh.strip():
        try:
            n = int(max_per_refresh)
        except ValueError:
            logger.warning("invalid UPSCALE_EVIDENCE_SAFETY_MAX_PER_REFRESH %r", max_per_refresh)
            n = DEFAULT_MAX_PER_REFRESH
    return EnrichmentSettings(enabled=on, max_per_refresh=max(0, min(n, MAX_PER_REFRESH_CAP)))


@dataclass
class EnrichmentReport:
    at: datetime
    status: str = "completed"  # completed / disabled / deferred / skipped
    reason: str | None = None
    selected: list[str] = field(default_factory=list)
    fetched: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "at": self.at.isoformat(),
            "status": self.status,
            "reason": self.reason,
            "selected": self.selected,
            "fetched": self.fetched,
            "failed": self.failed,
        }


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class SafetyEnrichment:
    def __init__(
        self,
        settings: EnrichmentSettings,
        service: SafetySource | None,
        store: EvidenceStore | None,
        busy: Callable[[], str | None] = lambda: None,
        sleep: Callable[[float], Awaitable[None]] = _sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.settings = settings
        self.service = service
        self.store = store
        self.busy = busy
        self.sleep = sleep
        self.now = now
        self.running = False
        self.last_report: EnrichmentReport | None = None

    def select(self, result: GrowthScoutResult) -> list[tuple[str, KnownPool]]:
        """Ranked Solana candidates production didn't safety-check this scan and that have
        no recent archived safety evidence, best rank first (at most the budget)."""
        if self.store is None or self.settings.max_per_refresh <= 0:
            return []
        now = self.now()
        since = now - timedelta(hours=self.settings.refresh_hours)
        picks: list[tuple[str, KnownPool]] = []
        for g in sorted(result.candidates, key=lambda g: g.rank or math.inf):
            if len(picks) >= self.settings.max_per_refresh:
                break
            if g.chain != "solana" or g.quality.safety_status != "INSUFFICIENT_SAFETY_DATA":
                continue
            recent = self.store.latest("safety", g.canonical_id, until=now, since=since)
            if recent is not None and recent.availability == "AVAILABLE":
                continue
            pool = g.market.selected_pool
            picks.append((g.address, KnownPool(address=pool.address, dex=pool.dex, eligible=True)))
        return picks

    async def after_scan(self, result: GrowthScoutResult) -> EnrichmentReport:
        report = EnrichmentReport(at=self.now())
        self.last_report = report
        s = self.settings
        if not s.enabled:
            report.status, report.reason = "disabled", "UPSCALE_EVIDENCE_SAFETY_ENRICHMENT is off"
            return report
        if self.service is None or self.store is None:
            report.status, report.reason = (
                "disabled",
                "no Solana RPC or evidence archive configured",
            )
            return report
        if self.running:
            report.status, report.reason = "skipped", "an enrichment pass is already running"
            return report
        self.running = True
        try:
            await self.sleep(s.quiet_seconds)
            picks = await asyncio.to_thread(self.select, result)
            report.selected = [mint for mint, _ in picks]
            for mint, pool in picks:
                reason = self.busy()
                if reason is None and self.service.available_calls() <= s.min_free_calls:
                    reason = "on-chain safety capacity is kept for Analyze"
                if reason is not None:
                    report.status, report.reason = "deferred", reason
                    break
                with evidence.component("safety_enrichment"):
                    try:
                        await self.service.get_snapshot(mint, [pool])  # archived by the hook
                        report.fetched += 1
                    except MarketDataError:
                        report.failed += 1  # the failure itself is archived by the hook
            return report
        finally:
            self.running = False
