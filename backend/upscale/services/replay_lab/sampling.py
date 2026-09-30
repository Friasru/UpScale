"""Which historical decision times to replay, and which split each belongs to.

**Universe (no lookahead in who is sampled).** An asset may only be sampled at T if it
was in UpScale's universe at T:

* RECORDED: T is a moment Scout itself stored a snapshot of the token (it was being
  tracked then), so the evidence and the universe are both genuinely from T.
* CANDLES from the Scout archive: only at T >= the time Scout first saw the token.
* CANDLES for `--asset chain:token:pool`: the user chose the asset, possibly knowing its
  later history; every such sample is labeled `USER_SELECTED` (a selection-bias warning).

**Independence.** Samples of one asset are at least `min_spacing_minutes` apart and at
most `max_per_asset`; selection round-robins across assets (then days), deterministically
(seeded hash order), so one busy token can't pose as many independent examples. Samples at
the same moment share a `cohort_at` bucket (for future "which of these candidates?"
evaluation).

**Splits.** Time-based and persisted: the oldest `calibration_pct` of the selected
samples are CALIBRATION, the next `validation_pct` VALIDATION, the latest HOLDOUT. A
sample falling inside any earlier job's HOLDOUT time range is HOLDOUT too (holdout
windows are sticky across jobs). A sample whose longest outcome window reaches into a
later split's time range is `purged`: it keeps its split, but findings exclude it.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from upscale.services.replay_lab.archive import ScoutArchive
from upscale.services.replay_lab.config import Evidence, ReplayJobConfig, Split
from upscale.services.replay_lab.models import PlannedSample
from upscale.services.scout.normalize import canonical_id

COHORT_MINUTES = 15


@dataclass(frozen=True)
class Candidate:
    asset_id: str
    chain: str
    token: str
    pool: str
    symbol: str | None
    at: datetime
    evidence: Evidence
    universe_basis: str
    snapshot_provider: str | None = None


@dataclass
class PlanReport:
    considered: int = 0
    not_elapsed: int = 0
    spacing_dropped: int = 0
    per_asset_dropped: int = 0
    over_max_samples: int = 0
    selected: int = 0
    assets: int = 0
    sticky_holdout: int = 0
    purged: int = 0
    user_selected: bool = False
    notes: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def sample_key(c: Candidate, mode: str) -> str:
    raw = f"{c.evidence}|{mode}|{c.asset_id}|{c.pool}|{c.at.timestamp():.3f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _hash(seed: int, *parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in (seed, *parts)).encode()).hexdigest()


def candidates(
    config: ReplayJobConfig, archive: ScoutArchive | None
) -> tuple[list[Candidate], list[str]]:
    notes: list[str] = []
    out: list[Candidate] = []
    wanted = {a.canonical_id: a for a in config.assets}
    if config.evidence == "RECORDED":
        if archive is None:
            return [], ["RECORDED samples need the Scout archive"]
        for s in archive.snapshot_times(config.start, config.end, config.chains):
            spec = wanted.get(s.canonical_id)
            if wanted and (spec is None or (spec.pool and spec.pool != s.pool_address)):
                continue
            out.append(
                Candidate(
                    asset_id=s.canonical_id, chain=s.chain, token=s.address, pool=s.pool_address,
                    symbol=s.symbol, at=s.observed_at,
                    evidence="RECORDED",
                    universe_basis="scout_archive: Scout tracked the token at T",
                    snapshot_provider=s.provider,
                )
            )  # fmt: skip
        return out, notes
    # CANDLES: grid times.
    step = timedelta(minutes=max(config.min_spacing_minutes, 1.0))
    explicit = [a for a in config.assets if a.pool]
    if explicit:
        notes.append(
            "USER_SELECTED universe: assets chosen by the user may carry selection bias "
            "(chosen knowing their later history)"
        )
        for a in explicit:
            if a.chain not in config.chains:
                continue
            for at in _grid(config.start, config.end, step):
                out.append(
                    Candidate(
                        canonical_id(a.chain, a.token),
                        a.chain,
                        a.token,
                        a.pool or "",
                        None,
                        at,
                        "CANDLES",
                        "USER_SELECTED: --asset",
                    )  # fmt: skip
                )
        return out, notes
    if archive is None:
        return [], ["CANDLES samples need --asset chain:token:pool or the Scout archive"]
    for token in archive.tokens(config.chains):
        if wanted and token.canonical_id not in wanted:
            continue
        begin = max(config.start, token.first_seen_at)
        for at in _grid(begin, config.end, step):
            pool = archive.pool_at(token.canonical_id, at)
            if pool is None:
                continue
            out.append(
                Candidate(
                    token.canonical_id,
                    token.chain,
                    token.address,
                    pool,
                    token.symbol,
                    at,
                    "CANDLES",
                    "scout_archive: first seen by Scout at or before T",
                )  # fmt: skip
            )
    return out, notes


def _grid(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    first = datetime.fromtimestamp(
        math.ceil(start.timestamp() / step.total_seconds()) * step.total_seconds(), UTC
    )
    out, at = [], first
    while at <= end:
        out.append(at)
        at += step
    return out


def select(
    pool: Sequence[Candidate], config: ReplayJobConfig, now: datetime, report: PlanReport
) -> list[Candidate]:
    report.considered = len(pool)
    latest = now - timedelta(minutes=config.max_horizon_minutes + config.settle_minutes)
    elapsed = [c for c in pool if c.at <= latest]
    report.not_elapsed = len(pool) - len(elapsed)
    by_asset: dict[str, list[Candidate]] = {}
    seen: set[tuple[str, float]] = set()
    for c in sorted(elapsed, key=lambda c: (c.asset_id, c.at, c.pool)):
        key = (c.asset_id, c.at.timestamp())
        if key in seen:
            continue
        seen.add(key)
        kept = by_asset.setdefault(c.asset_id, [])
        if kept and c.at - kept[-1].at < timedelta(minutes=config.min_spacing_minutes):
            report.spacing_dropped += 1
            continue
        kept.append(c)
    per_asset: dict[str, list[Candidate]] = {}
    for asset, items in by_asset.items():
        if len(items) > config.max_per_asset:
            # Spread across the asset's history (deterministic), not just its first hours.
            ranked = sorted(items, key=lambda c: _hash(config.seed, asset, c.at.timestamp()))
            chosen = sorted(ranked[: config.max_per_asset], key=lambda c: c.at)
            report.per_asset_dropped += len(items) - len(chosen)
            items = chosen
        per_asset[asset] = items
    # Round-robin: every asset's k-th sample before anyone's (k+1)-th; within a round,
    # spread across days, then a seeded hash (no systematic bias toward any token).
    order = sorted(
        ((i, c) for items in per_asset.values() for i, c in enumerate(items)),
        key=lambda x: (x[0], _hash(config.seed, x[1].at.date(), x[1].asset_id)),
    )
    chosen = [c for _, c in order[: config.max_samples]]
    report.over_max_samples = max(0, len(order) - config.max_samples)
    report.selected = len(chosen)
    report.assets = len({c.asset_id for c in chosen})
    return sorted(chosen, key=lambda c: (c.at, c.asset_id))


def assign_splits(
    chosen: Sequence[Candidate],
    config: ReplayJobConfig,
    holdout_windows: Sequence[tuple[datetime, datetime]],
    report: PlanReport,
) -> list[PlannedSample]:
    n = len(chosen)
    s = config.split
    cal_end = math.floor(n * s.calibration_pct / 100)
    val_end = cal_end + math.floor(n * s.validation_pct / 100)
    ordered = sorted(chosen, key=lambda c: (c.at, c.asset_id, c.pool))
    splits: list[Split] = [
        "CALIBRATION" if i < cal_end else "VALIDATION" if i < val_end else "HOLDOUT"
        for i in range(n)
    ]
    for i, c in enumerate(ordered):
        if splits[i] != "HOLDOUT" and any(a <= c.at <= b for a, b in holdout_windows):
            splits[i] = "HOLDOUT"
            report.sticky_holdout += 1
    # Earliest time of each later split: an outcome window reaching it is purged.
    starts: dict[Split, datetime] = {}
    for sp, c in zip(splits, ordered, strict=True):
        starts.setdefault(sp, c.at)
    horizon = timedelta(minutes=config.max_horizon_minutes)
    later: dict[Split, tuple[Split, ...]] = {
        "CALIBRATION": ("VALIDATION", "HOLDOUT"),
        "VALIDATION": ("HOLDOUT",),
        "HOLDOUT": (),
    }
    planned: list[PlannedSample] = []
    for i, (sp, c) in enumerate(zip(splits, ordered, strict=True)):
        purged = any(nxt in starts and c.at + horizon > starts[nxt] for nxt in later[sp])
        report.purged += purged
        cohort = datetime.fromtimestamp(
            math.floor(c.at.timestamp() / (COHORT_MINUTES * 60)) * COHORT_MINUTES * 60, UTC
        )
        planned.append(
            PlannedSample(
                sample_key=sample_key(c, config.mode),
                asset_id=c.asset_id,
                chain=c.chain,
                token_address=c.token,
                pool_address=c.pool,
                symbol=c.symbol,
                decision_at=c.at,
                evidence=c.evidence,
                universe_basis=c.universe_basis,
                split=sp,
                purged=purged,
                cohort_at=cohort,
                plan_order=i,
                snapshot_provider=c.snapshot_provider,
            )
        )
    return planned


def plan(
    config: ReplayJobConfig,
    archive: ScoutArchive | None,
    now: datetime,
    holdout_windows: Sequence[tuple[datetime, datetime]] = (),
) -> tuple[list[PlannedSample], PlanReport]:
    report = PlanReport()
    pool, notes = candidates(config, archive)
    report.notes = notes
    report.user_selected = any(c.universe_basis.startswith("USER_SELECTED") for c in pool)
    chosen = select(pool, config, now, report)
    return assign_splits(chosen, config, holdout_windows, report), report
