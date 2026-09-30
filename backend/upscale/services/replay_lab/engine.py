"""The replay job runner: plan, then for each sample, strictly in this order:

1. **acquire** the evidence available at T (recorded snapshot, pool candles closed by T;
   provider requests go through the lowest-priority gate and may defer the job);
2. **decide** with production logic (Scout, Technical, Risk, Opportunity) on that evidence
   only, no network;
3. **commit** the frozen decision (immutable; the store issues a receipt);
4. **reveal**: only with the receipt, read what happened after T and measure outcomes from
   the decision *as stored*; commit them and complete the sample.

Resumable: a job's samples are persisted when planned; a rerun continues with the samples
not yet complete. A sample interrupted after step 3 resumes at step 4 with its stored
decision (never re-decided). Completed samples are never repeated (unique per job).

Pausing: when higher-priority work needs the provider, the job becomes PAUSED and the
runner sleeps (never polls in a tight loop) or exits (`exit_on_pause`); `resume` goes on.
Lookahead or a decision failing its integrity check fails the job loudly.
"""

import asyncio
import hashlib
import json
import math
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from upscale.services.asset_profile import integrated_capabilities
from upscale.services.evidence_archive.store import EvidenceStore, FutureEvidenceError
from upscale.services.market_data import MarketDataError, Timeframe
from upscale.services.outcomes.config import CollapseConfig, CollectorConfig
from upscale.services.replay_lab.analyze import analyze_at
from upscale.services.replay_lab.archive import ArchiveUnavailableError, PoolMetadata, ScoutArchive
from upscale.services.replay_lab.candles import (
    HistoricalCandleFetcher,
    PointInTimeCandles,
    PoolHistoryUnavailableError,
    interval_of,
)
from upscale.services.replay_lab.clock import HistoricalClock, LookaheadError
from upscale.services.replay_lab.config import ReplayJobConfig
from upscale.services.replay_lab.models import (
    AgentOutput,
    ReplayDecisionRecord,
    ScoutReplay,
)
from upscale.services.replay_lab.outcomes import (
    DEFAULT_COLLAPSE,
    DEFAULT_COLLECTOR,
    fidelity,
    measure,
)
from upscale.services.replay_lab.quota import ReplayDeferred
from upscale.services.replay_lab.reconstruct import evaluate_scout, pool_at
from upscale.services.replay_lab.sampling import plan
from upscale.services.replay_lab.store import (
    DecisionIntegrityError,
    JobRow,
    ReplayStore,
    StoredSample,
    pid_alive,
)
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.growth.config import GrowthConfig
from upscale.services.scout.growth.models import GrowthCandidate
from upscale.services.scout.models import ScoutSnapshot
from upscale.services.scout.social.models import SocialMomentum
from upscale.services.solana_chain import OnchainSafetySnapshot
from upscale.services.strategy import DEFAULT_STRATEGY
from upscale.services.technical_analysis import InvalidCandleDataError, analyze_series
from upscale.services.versions import fingerprints

MAX_ATTEMPTS = 3
ReferenceBasis = Literal["recorded_snapshot_price", "last_closed_candle"]
LEASE_STALE_SECONDS = 180.0
# CANDLES samples: the last 1m close is the reference price only if this recent.
MAX_REFERENCE_AGE = timedelta(minutes=15)


class JobLockedError(RuntimeError):
    pass


class SampleSkipped(Exception):
    pass


@dataclass
class RunResult:
    job: JobRow
    paused: bool = False
    pause_reason: str | None = None
    processed: int = 0
    notes: list[str] = field(default_factory=list)


def _hash_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ReplayRunner:
    def __init__(
        self,
        store: ReplayStore,
        archive: ScoutArchive | None,
        fetcher: HistoricalCandleFetcher,
        scout_config: ScoutConfig,
        growth_config: GrowthConfig,
        collapse: CollapseConfig = DEFAULT_COLLAPSE,
        collector: CollectorConfig = DEFAULT_COLLECTOR,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        log: Callable[[str], None] = lambda _: None,
        max_pause_seconds: float = 900.0,
        evidence: EvidenceStore | None = None,
        safety_max_age: timedelta = timedelta(minutes=60),
    ):
        self.store = store
        self.archive = archive
        self.fetcher = fetcher
        self.scout_config = scout_config
        self.growth_config = growth_config
        self.collapse = collapse
        self.collector = collector
        self.wall_clock = wall_clock
        self.sleep = sleep
        self.log = log
        self.max_pause_seconds = max_pause_seconds
        self.evidence = evidence
        self.safety_max_age = safety_max_age
        self.versions = {
            "record": "1",
            **fingerprints(),
            "replay_scout_config": _hash_json(scout_config.model_dump(mode="json")),
            "replay_growth_config": _hash_json(growth_config.model_dump(mode="json")),
        }

    # --- jobs -------------------------------------------------------------------------------

    def create_job(self, config: ReplayJobConfig, job_id: str | None = None) -> str:
        job_id = job_id or f"replay-{self.wall_clock():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        samples, report = plan(
            config, self.archive, self.wall_clock(), self.store.holdout_windows()
        )
        planning = report.as_dict() | {"versions": self.versions}
        self.store.create_job(job_id, config, samples, planning)
        return job_id

    async def run(self, job_id: str, exit_on_pause: bool = False) -> RunResult:
        job = self.store.job(job_id)
        if job is None:
            raise KeyError(f"unknown replay job {job_id}")
        if job.status == "COMPLETE":
            return RunResult(job=job, notes=["job already complete"])
        holder_gone = job.runner_pid is not None and not pid_alive(job.runner_pid)
        if not self.store.claim_job(job_id, os.getpid(), LEASE_STALE_SECONDS, force=holder_gone):
            raise JobLockedError(f"job {job_id} is being run by another process")
        self.store.set_job_status(job_id, "RUNNING")
        processed = 0
        try:
            while True:
                pending = [
                    s
                    for s in self.store.samples(job_id, ("PLANNED", "DECIDED"))
                    if s.attempts < MAX_ATTEMPTS
                ]
                if not pending:
                    break
                progressed = False
                for sample in pending:
                    try:
                        await self._process(job.config, sample)
                        progressed = True
                        processed += 1
                    except ReplayDeferred as exc:
                        paused = await self._defer(job_id, exc, exit_on_pause)
                        if paused:
                            return RunResult(
                                job=self._job(job_id), paused=True, pause_reason=exc.reason,
                                processed=processed,
                            )  # fmt: skip
                        progressed = True  # slept; try this sample's successors afterwards
                        break
                    finally:
                        self.store.heartbeat(job_id, sample.id)
                        self.store.save_usage(job_id, self.fetcher.usage.as_dict())
                if not progressed:
                    break
            for s in self.store.samples(job_id, ("PLANNED", "DECIDED")):
                self.store.mark_sample(
                    s.id, "FAILED", error=s.error or "gave up after repeated failures"
                )
            self.store.set_job_status(job_id, "COMPLETE")
        except (
            LookaheadError,
            FutureEvidenceError,
            DecisionIntegrityError,
            ArchiveUnavailableError,
        ) as exc:
            self.store.set_job_status(job_id, "FAILED", f"{type(exc).__name__}: {exc}")
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.store.set_job_status(job_id, "PAUSED", "interrupted; resume to continue")
            raise
        finally:
            self.store.save_usage(job_id, self.fetcher.usage.as_dict())
            self.store.release_job(job_id)
        return RunResult(job=self._job(job_id), processed=processed)

    def _job(self, job_id: str) -> JobRow:
        job = self.store.job(job_id)
        assert job is not None
        return job

    async def _defer(self, job_id: str, exc: ReplayDeferred, exit_on_pause: bool) -> bool:
        """True when the runner stops (job left PAUSED)."""
        wait = min(max(exc.retry_after, 5.0), self.max_pause_seconds)
        if exc.production:
            self.store.set_job_status(job_id, "PAUSED", exc.reason)
            self.log(f"paused: {exc.reason}")
            if exit_on_pause:
                return True
        else:
            self.log(f"throttled: {exc.reason}; next request in {wait:.0f}s")
        remaining = wait
        while remaining > 0:  # keep the lease while waiting (never a busy loop)
            step = min(remaining, 60.0)
            await self.sleep(step)
            remaining -= step
            self.store.heartbeat(job_id)
        self.store.set_job_status(job_id, "RUNNING")
        return False

    # --- one sample ---------------------------------------------------------------------------

    async def _process(self, config: ReplayJobConfig, sample: StoredSample) -> None:
        plan_ = sample.plan
        reveal = timedelta(
            minutes=config.max_horizon_minutes * (1 + self.collector.market_tolerance)
        )
        clock = HistoricalClock(plan_.decision_at, reveal)
        candles = PointInTimeCandles(clock, plan_.chain, plan_.token_address, plan_.pool_address)
        try:
            if sample.status == "PLANNED":
                self.store.add_decision(
                    sample.id, await self._decide(config, sample, clock, candles)
                )
            # Outcomes are measured from the decision as stored (hash-checked), never from
            # the objects that produced it.
            record, receipt = self._stored(sample.id)
            clock.reveal(receipt)
            await self._acquire_outcome_candles(config, clock, candles, record)
            outcomes = measure(
                clock, record, candles, self.archive, config.horizons, self.collapse, self.collector
            )
            self.store.add_outcomes(receipt, outcomes, fidelity(clock, record, self.archive))
        except SampleSkipped as exc:
            self.store.mark_sample(sample.id, "SKIPPED", skip_reason=str(exc), attempt=True)
        except PoolHistoryUnavailableError as exc:
            if sample.status == "PLANNED" and self.store.decision(sample.id) is None:
                self.store.mark_sample(sample.id, "SKIPPED", skip_reason=str(exc), attempt=True)
            else:
                self.store.mark_sample(sample.id, "DECIDED", error=str(exc), attempt=True)
        except (MarketDataError, InvalidCandleDataError) as exc:
            self.store.note_attempt(sample.id, f"{type(exc).__name__}: {exc}")

    def _stored(self, sample_id: int) -> tuple[ReplayDecisionRecord, Any]:
        stored = self.store.decision(sample_id)
        if stored is None:
            raise DecisionIntegrityError(f"no stored decision for sample {sample_id}")
        return stored

    async def _load(
        self,
        candles: PointInTimeCandles,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> None:
        loaded = await self.fetcher.range(
            candles.chain, candles.token, candles.pool, timeframe, start, end
        )
        candles.load(timeframe, loaded)

    async def _decide(
        self,
        config: ReplayJobConfig,
        sample: StoredSample,
        clock: HistoricalClock,
        candles: PointInTimeCandles,
    ) -> ReplayDecisionRecord:
        p = sample.plan
        t = p.decision_at
        technical = DEFAULT_STRATEGY.technical
        tf = technical.default_timeframe
        # The live request: the latest `candles_to_fetch` (+1 in progress) before T.
        await self._load(candles, tf, t - interval_of(tf) * (technical.candles_to_fetch + 2), t)
        if p.evidence == "CANDLES":
            await self._load(candles, "1m", t - timedelta(hours=2), t)
        clock.check_decision_phase("the decision")

        warnings: list[str] = []
        availability: dict[str, str] = {}
        features: dict[str, Any] = {}
        social_info: dict[str, Any] = {"mode": config.mode}
        snapshot: ScoutSnapshot | None = None
        growth: GrowthCandidate | None = None
        symbol, name = p.symbol, None
        dex_pool = None
        market_provider = p.snapshot_provider or "historical"
        evidence_times: list[datetime] = []
        safety_snap: OnchainSafetySnapshot | None = None
        safety_failure: str | None = None
        safety_provider = "archive"
        capabilities: list[str] | None = None

        if p.evidence == "RECORDED":
            if self.archive is None:
                raise ArchiveUnavailableError("RECORDED samples need the Scout archive")
            snapshot = self.archive.snapshot_at(p.asset_id, p.pool_address, t)
            if snapshot is None or snapshot.metrics.price_usd is None:
                raise SampleSkipped("the recorded snapshot at T is missing or has no price")
            evidence_times.append(snapshot.observed_at)
            token = self.archive.token(p.asset_id)
            symbol = token.symbol if token else p.symbol
            name = token.name if token else None
            first_seen = token.first_seen_at if token and token.first_seen_at <= t else None
            meta = self.archive.pool_metadata(p.asset_id, p.pool_address, t)
            if meta is None:
                meta = self._pool_metadata(p.asset_id, p.pool_address, t)
            if meta is None:
                raise SampleSkipped(
                    "the pool's quote token / DEX (immutable pool metadata) was not recorded"
                )
            availability["pool_metadata"] = "IMMUTABLE_METADATA"
            momentum = None
            if config.mode == "MARKET_PLUS_SOCIAL":
                momentum = self._social_at(p.asset_id, t) or self.archive.social_momentum(
                    p.asset_id, t
                )
                if momentum is not None:
                    clock.check_time(momentum.computed_at, "social momentum")
                    evidence_times.append(momentum.computed_at)
                    social_info |= {
                        "status": "RECORDED_AT_OR_BEFORE_T",
                        "state": momentum.state,
                        "computed_at": momentum.computed_at.isoformat(),
                        "age_minutes": round((t - momentum.computed_at).total_seconds() / 60, 1),
                    }
                else:
                    social_info |= {
                        "status": "SOCIAL_UNAVAILABLE",
                        "reason": "no social measurement stored at or before T",
                    }
            else:
                social_info |= {"status": "SOCIAL_UNAVAILABLE", "reason": "MARKET_ONLY mode"}
            dex_pool = pool_at(snapshot, meta, p.chain, p.token_address, symbol, name)
            safety_snap, safety_failure, safety_provider = self._safety_at(
                clock, p.asset_id, t, availability, evidence_times
            )
            capabilities = self._capabilities_at(
                t, safety_snap is not None or safety_failure is not None, warnings
            )
            scout, growth, notes = await evaluate_scout(
                clock, self.archive, dex_pool, snapshot, first_seen, momentum,
                self.scout_config, self.growth_config, safety=safety_snap,
            )  # fmt: skip
            warnings += notes
            reference: float | None = snapshot.metrics.price_usd
            basis: ReferenceBasis = "recorded_snapshot_price"
            availability |= {
                "other_pools": "UNAVAILABLE: other pools of the token at T were not recorded",
                "discovery_listing": "UNAVAILABLE: which listing surfaced the token at T",
            }
        else:
            scout = ScoutReplay(
                status="NOT_RECONSTRUCTIBLE",
                reason=(
                    "candle-only evidence: production pool selection and Growth Scout need "
                    "trade counts, liquidity and rolling windows, which have no historical source"
                ),
            )
            social_info |= {
                "status": "SOCIAL_UNAVAILABLE",
                "reason": "no recorded social evidence for candle-only samples",
            }
            one_minute = candles.closed("1m")
            if not one_minute:
                raise SampleSkipped("no trade in the 2 hours before T: no reference price")
            last = one_minute[-1]
            close_at = last.timestamp + interval_of("1m")
            if t - close_at > MAX_REFERENCE_AGE:
                raise SampleSkipped(
                    f"last trade closed {(t - close_at).total_seconds() / 60:.0f} min before T: "
                    "no current reference price"
                )
            evidence_times.append(close_at)
            reference = last.close
            basis = "last_closed_candle"
            for feature in ("liquidity_usd", "market_cap_usd", "fdv_usd", "txns", "buyers"):
                availability[feature] = "UNAVAILABLE: no historical source (candles only)"

        analysis = await analyze_at(
            clock, chain=p.chain, token=p.token_address, symbol=symbol, name=name,
            pool=dex_pool, market_provider=market_provider, candles=candles,
            safety=safety_snap, safety_failure=safety_failure, safety_provider=safety_provider,
            capabilities=capabilities,
        )  # fmt: skip
        missing_tf = {tf_ for _, tf_, _ in analysis.candle_requests if not candles.loaded(tf_)}
        if missing_tf:
            raise MarketDataError(f"technical analysis asked for unloaded timeframes {missing_tf}")
        agents: dict[str, AgentOutput] = dict(analysis.agents)
        closed_tf = candles.closed(tf)
        if closed_tf:
            evidence_times.append(closed_tf[-1].timestamp + interval_of(tf))
        if p.evidence == "CANDLES":
            try:
                series = candles.series_at(
                    tf, technical.candles_to_fetch, symbol or p.token_address, p.asset_id
                )
                ta = analyze_series(series, technical)
                agents["technical_candles_only"] = AgentOutput(
                    agent="technical_candles_only", status="ok",
                    summary="production technical analysis on the pool's candles closed by T "
                            "(evidence only: not an input to the decision above)",
                    findings=ta.model_dump(mode="json"),
                )  # fmt: skip
            except (InvalidCandleDataError, ValueError) as exc:
                agents["technical_candles_only"] = AgentOutput(
                    agent="technical_candles_only", status="error", summary=str(exc), error=str(exc)
                )
        if "onchain_safety" in analysis.unavailable:
            warnings.append(
                "on-chain safety (authorities, holder concentration) has no historical source: "
                "production Risk / Opportunity treat it as missing evidence, as they do live "
                "without an RPC"
            )
        latest = max(evidence_times) if evidence_times else t
        if latest > t:
            raise LookaheadError(
                f"evidence from {latest.isoformat()} used for a decision at {t.isoformat()}"
            )

        features |= _features(snapshot, growth, candles, tf, reference, agents, availability)
        opportunity = agents.get("opportunity")
        risk = agents.get("risk")
        decision = analysis.decision
        return ReplayDecisionRecord(
            sample_key=p.sample_key,
            asset_id=p.asset_id,
            chain=p.chain,
            token_address=p.token_address,
            pool_address=p.pool_address,
            symbol=symbol,
            decision_at=t,
            evidence=p.evidence,
            mode=config.mode,
            evidence_latest_at=latest,
            reference_price=reference,
            reference_basis=basis,
            features=features,
            availability=availability,
            social=social_info,
            scout=scout,
            agents=agents,
            agents_unavailable=analysis.unavailable,
            decision=decision,
            action=decision.action if decision else _finding(opportunity, "action"),
            confidence=decision.confidence if decision else _finding(opportunity, "confidence"),
            risk_level=_finding(risk, "overall_risk"),
            uncertainty_level=_finding(risk, "uncertainty_level"),
            warnings=warnings,
            versions=self.versions,
        )

    # --- the evidence archive (point-in-time: observed_at <= T only) ------------------------

    def _safety_at(
        self,
        clock: HistoricalClock,
        asset_id: str,
        t: datetime,
        availability: dict[str, str],
        evidence_times: list[datetime],
    ) -> tuple[OnchainSafetySnapshot | None, str | None, str]:
        """The latest archived on-chain safety at or before T (within `safety_max_age`),
        or a failure production archived then. Never current chain state."""
        minutes = self.safety_max_age.total_seconds() / 60
        if self.evidence is None:
            availability["onchain_safety"] = "NOT_COLLECTED: no evidence archive configured"
            return None, None, "archive"
        rec = self.evidence.latest("safety", asset_id, until=t, since=t - self.safety_max_age)
        if rec is None:
            availability["onchain_safety"] = (
                f"NOT_COLLECTED: no archived on-chain safety within {minutes:g} min before T"
            )
            return None, None, "archive"
        clock.check_time(rec.observed_at, "archived on-chain safety")
        evidence_times.append(rec.observed_at)
        age = (t - rec.observed_at).total_seconds() / 60
        provider = rec.provider or "archive"
        if rec.availability != "AVAILABLE":
            availability["onchain_safety"] = (
                f"{rec.availability}: {rec.reason} (archived {age:.0f} min before T)"
            )
            return None, rec.reason or rec.availability, provider
        snap = OnchainSafetySnapshot.model_validate(rec.payload["snapshot"])
        clock.check_time(snap.fetched_at, "archived on-chain safety")
        availability["onchain_safety"] = f"AVAILABLE: archived {age:.0f} min before T ({provider})"
        return snap, None, provider

    def _capabilities_at(self, t: datetime, safety: bool, warnings: list[str]) -> list[str] | None:
        """The data capabilities production had at T (archived with each Scout run)."""
        rec = self.evidence.latest_any("scout", t) if self.evidence is not None else None
        caps = rec.payload.get("capabilities") if rec is not None else None
        if isinstance(caps, list):
            return sorted({str(c) for c in caps} | ({"onchain"} if safety else set()))
        if safety:
            warnings.append(
                "production capabilities at T were not archived: on-chain data is known to have "
                "been available (archived safety evidence), the rest from this process"
            )
            return sorted(set(integrated_capabilities()) | {"onchain"})
        return None

    def _social_at(self, asset_id: str, t: datetime) -> SocialMomentum | None:
        if self.evidence is None:
            return None
        rec = self.evidence.latest("social", asset_id, until=t)
        if rec is None or rec.availability != "AVAILABLE":
            return None
        return SocialMomentum.model_validate(rec.payload["momentum"])

    def _pool_metadata(self, asset_id: str, pool: str, t: datetime) -> PoolMetadata | None:
        """Immutable pool facts from archived market evidence of this pool at or before T."""
        if self.evidence is None:
            return None
        rec = self.evidence.latest("market", asset_id, until=t, pool=pool)
        if rec is None:
            return None
        p = (rec.payload.get("candidate") or {}).get("pool") or {}
        if p.get("address") != pool or not p.get("quote_address"):
            return None
        created = datetime.fromisoformat(p["created_at"]) if p.get("created_at") else None
        if created is not None and created > t:
            raise LookaheadError(f"pool {pool} was created after the decision time")
        return PoolMetadata(
            pool_address=pool, dex=p.get("dex") or "unknown", quote_address=p["quote_address"],
            quote_symbol=p.get("quote_symbol"), created_at=created, url=p.get("url"),
        )  # fmt: skip

    async def _acquire_outcome_candles(
        self,
        config: ReplayJobConfig,
        clock: HistoricalClock,
        candles: PointInTimeCandles,
        record: ReplayDecisionRecord,
    ) -> None:
        if clock.phase != "REVEALED":
            raise LookaheadError("outcome candles are fetched only after the decision is stored")
        t = record.decision_at
        spans: dict[Timeframe, int] = {}
        for h in config.horizons:
            spans[h.candles] = max(spans.get(h.candles, 0), h.minutes)
        for timeframe, minutes in spans.items():
            end = t + timedelta(minutes=minutes) + interval_of(timeframe)
            clock.check_request(end, f"{timeframe} outcome candles")
            await self._load(candles, timeframe, t, end)


def _finding(output: AgentOutput | None, key: str) -> str | None:
    if output is None or output.status != "ok":
        return None
    value = output.findings.get(key)
    return value if isinstance(value, str) else None


def _features(
    snapshot: ScoutSnapshot | None,
    growth: GrowthCandidate | None,
    candles: PointInTimeCandles,
    timeframe: Timeframe,
    reference: float | None,
    agents: dict[str, AgentOutput],
    availability: dict[str, str],
) -> dict[str, Any]:
    """Flat, analysis-friendly features, all from evidence available at T."""
    f: dict[str, Any] = {"reference_price": reference}
    if snapshot is not None:
        m = snapshot.metrics
        h1, h24 = m.window("h1"), m.window("h24")
        f |= {
            "price_usd": m.price_usd,
            "liquidity_usd": m.liquidity_usd,
            "fdv_usd": m.fdv_usd,
            "volume_h1_usd": h1.volume_usd if h1 else None,
            "volume_h24_usd": h24.volume_usd if h24 else None,
            "txns_h1": h1.txns if h1 else None,
            "txns_h24": h24.txns if h24 else None,
            "buy_share_h1": h1.buy_share if h1 else None,
            "price_change_h1_pct": h1.price_change_pct if h1 else None,
            "price_change_h24_pct": h24.price_change_pct if h24 else None,
            "activity_h1_vs_h24": (
                (h1.txns / 60) / (h24.txns / 1440)
                if h1 and h24 and h1.txns is not None and h24.txns
                else None
            ),
        }
    if growth is not None:
        mo, mk = growth.momentum, growth.market
        f |= {
            "market_cap_usd": mk.market_cap_usd,  # only when Growth Scout trusts it
            "market_cap_note": mk.market_cap_note,
            "pool_age_hours": mk.pool_age_hours,
            "volume_acceleration": mo.volume_acceleration,
            "txn_acceleration": mo.txn_acceleration,
            "buy_pressure_change": mo.buy_pressure_change,
            "move_since_first_seen_pct": mo.move_since_first_seen_pct,
            "liquidity_change_pct": mo.liquidity_change_pct,
            "tracked_hours": mk.tracked_hours,
            "maturity": growth.quality.maturity,
            "liquidity_quality": growth.quality.liquidity_quality,
            "flow_quality": growth.quality.flow_quality,
        }
    closed = candles.closed(timeframe)
    closes = [c.close for c in closed[-43:]]
    if len(closes) >= 8:
        returns = [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
        mu = sum(returns) / len(returns)
        sd = math.sqrt(sum((r - mu) ** 2 for r in returns) / (len(returns) - 1))
        f["volatility_4h_pct"] = round(100 * sd, 4)  # stdev of 4h log returns, up to 7 days
        availability["volatility_4h_pct"] = "AVAILABLE"
    else:
        f["volatility_4h_pct"] = None
        availability["volatility_4h_pct"] = "UNAVAILABLE: fewer than 8 closed 4h candles before T"
    if len(closed) >= 7 and f.get("price_change_h24_pct") is None:
        f["price_change_h24_pct"] = round(100 * (closed[-1].close / closed[-7].close - 1), 4)
        availability["price_change_h24_pct"] = "AVAILABLE: from 4h candles closed by T"
    tech = agents.get("technical_analysis")
    if tech is None or tech.status != "ok":
        tech = agents.get("technical_candles_only")
    if tech is not None and tech.status == "ok":
        trend = tech.findings.get("trend") or {}
        f["technical_trend"] = trend.get("label") if isinstance(trend, dict) else None
        for ind in tech.findings.get("indicators") or []:
            if isinstance(ind, dict) and ind.get("name") == "RSI 14":
                f["rsi_14"] = ind.get("value")
    for key, value in f.items():
        if key not in availability:
            availability[key] = (
                "AVAILABLE" if value is not None else "UNAVAILABLE: not reported at T"
            )
    return f
