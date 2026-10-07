"""Radar V1 orchestration: targets, one bounded collection + snapshot per target.

CLI-only in V1: nothing here is started by ``main.py`` or a background loop.

Targets come from Scout (read-only: the Scout database is opened ``mode=ro``; Scout's
ranking, scores and thresholds are never touched) or are added by hand. A target's pool
is fixed when it's added, so stored history always refers to one pool.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx2

from upscale.services.chains import SOLANA, is_solana_address
from upscale.services.market_data import InvalidRequestError
from upscale.services.radar.collector import Collector, StepResult
from upscale.services.radar.config import RadarSettings
from upscale.services.radar.features import build_snapshot, overall_coverage
from upscale.services.radar.models import Target, solana_identity, ts
from upscale.services.radar.provider import RadarProvider, RequestGuard, build_provider
from upscale.services.radar.readonly import connect as connect_read_only
from upscale.services.radar.readonly import tables
from upscale.services.radar.repository import RadarRepository


@dataclass
class SnapshotResult:
    canonical_id: str
    observed_at: datetime
    body: dict[str, Any]
    steps: dict[str, StepResult] = field(default_factory=dict)
    requests: int = 0
    saved: bool = False
    snapshot_id: int | None = None
    body_hash: str | None = None


class RadarService:
    def __init__(
        self,
        settings: RadarSettings,
        repo: RadarRepository | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        helius_api_key: str | None = None,
        rpc_url: str | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        guard: RequestGuard | None = None,
    ):
        self.settings = settings
        self.repo = repo or RadarRepository(settings.db_path)
        self._now = now
        self.guard = guard or RequestGuard(settings, self.repo, now=now)
        self.provider: RadarProvider | None = build_provider(
            self.guard, helius_api_key, rpc_url, transport
        )

    # --- targets ------------------------------------------------------------------------

    def add_target(
        self,
        mint: str,
        pool_address: str,
        dex: str | None = None,
        pool_created_at: datetime | None = None,
        source: str = "manual",
    ) -> tuple[str, bool]:
        cid, mint = solana_identity(mint)
        pool = pool_address.strip()
        if not is_solana_address(pool):
            raise InvalidRequestError(f"{pool_address!r} is not a valid Solana pool address")
        created = ts(pool_created_at) if pool_created_at else None
        added = self.repo.upsert_target(
            cid, mint, pool, dex, created,
            "provider_reported_pair_created_at" if created is not None else None,
            source, ts(self._now()),
        )  # fmt: skip
        return cid, added

    def sync_from_scout(self, scout_db: str, since: datetime | None, limit: int) -> dict[str, Any]:
        """Add Solana tokens Scout anchored (ranked) since `since`, newest first, up to
        `limit`. The Scout database is only read."""
        conn = connect_read_only(scout_db)
        if conn is None:
            return {"found": False, "path": scout_db, "added": [], "existing": []}
        added: list[str] = []
        existing: list[str] = []
        try:
            if "scout_outcome_observations" not in tables(conn):
                return {"found": True, "path": scout_db, "added": [], "existing": [],
                        "note": "no Scout anchors yet"}  # fmt: skip
            rows = conn.execute(
                "SELECT canonical_id, address, pool_address, MAX(anchored_at) FROM "
                "scout_outcome_observations WHERE chain = ? AND anchored_at >= ? "
                "GROUP BY canonical_id ORDER BY MAX(anchored_at) DESC, canonical_id LIMIT ?",
                (SOLANA, ts(since) if since else -1.0, limit),
            ).fetchall()
            latest = "scout_latest" in tables(conn)
            for cid, address, pool, _ in rows:
                dex = created = None
                if latest:
                    got = conn.execute(
                        "SELECT body_json FROM scout_latest WHERE canonical_id = ?", (cid,)
                    ).fetchone()
                    body = json.loads(got[0]) if got else {}
                    p = body.get("pool") or {}
                    if p.get("address") == pool:
                        dex = p.get("dex")
                        created = p.get("created_at")
                when = datetime.fromisoformat(created) if isinstance(created, str) else None
                try:
                    tid, new = self.add_target(address, pool, dex, when, source="scout")
                except InvalidRequestError:
                    continue
                (added if new else existing).append(tid)
        finally:
            conn.close()
        return {"found": True, "path": scout_db, "added": added, "existing": existing}

    # --- collection + snapshot ----------------------------------------------------------

    async def collect(self, target: Target) -> dict[str, StepResult]:
        """Run every step for one target. A failing step never stops the others, and a
        provider failure never raises."""
        if self.provider is None:
            why = "no Solana provider configured (UPSCALE_HELIUS_API_KEY / UPSCALE_SOLANA_RPC_URL)"
            return {k: StepResult("PROVIDER_UNAVAILABLE", [why])
                    for k in ("holders", "activity", "early", "deployer")}  # fmt: skip
        c = Collector(self.settings, self.repo, self.provider, self._now)
        steps: dict[str, StepResult] = {}
        steps["holders"] = await c.collect_holders(target)
        steps["activity"] = await c.collect_activity(target)
        fresh = self.repo.get_target(target.canonical_id) or target
        steps["early"] = await c.collect_early(fresh)
        steps["deployer"] = await c.collect_deployer(fresh)
        if self.settings.wallet_age or self.settings.first_funder:
            steps["wallet_profiles"] = await c.collect_wallet_profiles(fresh)
        return steps

    def build(
        self, canonical_id: str, as_of: datetime, run: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        target = self.repo.get_target(canonical_id)
        if target is None:
            raise InvalidRequestError(f"{canonical_id} is not a Radar target")
        inp = self.repo.load_inputs(canonical_id, ts(as_of))
        name = self.provider.name if self.provider else None
        return build_snapshot(target, inp, self.settings, name, run)

    async def snapshot(
        self, canonical_id: str, fetch: bool = True, save: bool = True
    ) -> SnapshotResult:
        target = self.repo.get_target(canonical_id)
        if target is None:
            raise InvalidRequestError(f"{canonical_id} is not a Radar target (add it first)")
        before = self.guard.used_this_run
        steps = await self.collect(target) if fetch else {}
        observed = self._now()
        run = {k: v.as_dict() for k, v in steps.items()}
        body = self.build(canonical_id, observed, run)
        result = SnapshotResult(
            canonical_id, observed, body, steps, self.guard.used_this_run - before
        )
        if save and fetch:
            sid, digest = self.repo.save_snapshot(
                canonical_id, ts(observed), overall_coverage(body), body
            )
            result.saved, result.snapshot_id, result.body_hash = True, sid, digest
        return result

    async def snapshot_active(self, limit: int) -> list[SnapshotResult | tuple[str, str]]:
        """Snapshot up to `limit` active targets (least recently scanned first). One
        target failing is reported and never stops the batch."""
        out: list[SnapshotResult | tuple[str, str]] = []
        # Never scanned first, then the oldest finished activity scan (a quiet scan counts);
        # ties keep selection order.
        queue = sorted(
            self.repo.targets("ACTIVE"),
            key=lambda t: (t.last_scan_at is not None, t.last_scan_at or 0.0),
        )
        for target in queue[:limit]:
            if self.guard.remaining_today() <= 0:
                out.append((target.canonical_id, "NOT_COLLECTED: daily request budget spent"))
                continue
            try:
                out.append(await self.snapshot(target.canonical_id))
            except Exception as exc:  # isolation: one target never breaks the batch
                out.append((target.canonical_id, f"failed: {type(exc).__name__}: {exc}"))
        return out
