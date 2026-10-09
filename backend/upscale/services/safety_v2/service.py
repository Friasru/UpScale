"""Safety V2 orchestration: targets (and pinned pools), one bounded collection (the mint
account, and optionally the holders), and deterministic ``safety.snapshot.v2`` snapshots.

CLI-only: nothing in the app starts it. A collection always ends DONE or ABORTED: a
provider failure is stored as evidence (PROVIDER_FAILED / NOT_COLLECTED) and the collection
is DONE; any other exception finishes it ABORTED and is re-raised. The mint and holder
components fail independently: a holder failure never discards the mint observation.
Holders are collected only when asked (``holders=True``; the CLI's default), so a
mint-only refresh never shadows earlier holder evidence. The market likewise
(``market=True``): one DEX request, plus at most one ``getAccountInfo(pool)`` and only when
a previously reported pool is missing from a successful DEX response.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx2

from upscale.services.chains import is_solana_address
from upscale.services.market_data import InvalidRequestError
from upscale.services.safety_v2.config import (
    RULES_VERSION,
    SNAPSHOT_SCHEMA,
    SafetySettings,
    load_settings,
)
from upscale.services.safety_v2.features import (
    ChangeInputs,
    CreatorInputs,
    HolderInputs,
    MarketInputs,
    build_body,
    code_fingerprints,
    is_complete,
)
from upscale.services.safety_v2.market import market_view
from upscale.services.safety_v2.models import (
    SafetyBudgetExhaustedError,
    SafetyProviderError,
    solana_identity,
    ts,
)
from upscale.services.safety_v2.provider import (
    HolderObservation,
    MarketObservation,
    MintObservation,
    PoolAccountObservation,
    RequestGuard,
    SafetyDexProvider,
    SafetyRpcProvider,
    build_dex_provider,
    build_provider,
    classify_market,
    classify_mint_account,
    classify_pool_account,
    collect_holders,
    holder_not_observed,
    holder_request_bound,
    market_not_observed,
    not_observed,
    pool_account_not_observed,
)
from upscale.services.safety_v2.registry import RAYDIUM_AMM_V4
from upscale.services.safety_v2.repository import SafetyRepository, encode_body
from upscale.services.safety_v2.sources import (
    NOT_CONSULTED,
    ProofStatus,
    WalletProofs,
    read_radar_capture,
)

NO_PROVIDER = "no Solana provider configured (UPSCALE_HELIUS_API_KEY / UPSCALE_SOLANA_RPC_URL)"
NO_DEX = "no market provider configured (UPSCALE_SAFETY_V2_DEX_URL)"


KNOWN_DEXES = (RAYDIUM_AMM_V4,)


@dataclass(frozen=True)
class Collected:
    collection_id: int
    observation_id: int
    outcome: str
    requests: int
    holder_observation_id: int | None = None
    holder_outcome: str | None = None
    market_observation_id: int | None = None
    market_outcome: str | None = None
    pool_account_outcome: str | None = None
    radar_capture_status: str | None = None
    radar_new_facts: int = 0


@dataclass
class SnapshotResult:
    canonical_id: str
    as_of: datetime
    body: dict[str, Any]
    body_hash: str
    collected: Collected | None = None
    saved: bool = False
    snapshot_id: int | None = None


RebuildStatus = Literal["REPRODUCED", "FINGERPRINT_MISMATCH", "DIVERGED"]


@dataclass(frozen=True)
class RebuildResult:
    snapshot_id: int
    status: RebuildStatus
    stored_hash: str
    rebuilt_hash: str | None
    mismatched: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectionRequestBound:
    """A conservative upper bound on what one `SafetyService.collect` can spend: the most
    logical provider calls it can make, and the most attempts (every retry counts against
    the daily budget). An upper bound, never a promise of actual usage."""

    logical_max: int
    attempt_max: int


class SafetyService:
    def __init__(
        self,
        settings: SafetySettings,
        repo: SafetyRepository | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        helius_api_key: str | None = None,
        rpc_url: str | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        guard: RequestGuard | None = None,
        fingerprints: Callable[[], Mapping[str, str]] = code_fingerprints,
    ):
        self.settings = settings
        self.repo = repo or SafetyRepository(settings.db_path)
        self._now = now
        self._fingerprints = fingerprints
        self.guard = guard or RequestGuard(settings, self.repo, now=now)
        self.provider: SafetyRpcProvider | None = build_provider(
            self.guard, helius_api_key, rpc_url, transport
        )
        self.dex: SafetyDexProvider | None = build_dex_provider(
            self.guard, settings.dex_url, transport
        )

    def collection_request_bound(
        self, holders: bool = True, market: bool = True
    ) -> CollectionRequestBound:
        """The most one ``collect(canonical_id, holders, market)`` can spend with this
        service's providers and settings (read-only: nothing is requested or recorded).

        Logical calls, as `collect` makes them: ``getAccountInfo(mint)`` (1, with a Solana
        provider); with `holders`, `holder_request_bound` (4, plus ``holder_max_pages`` with a
        full-scan provider); with `market`, the DEX request (1, with a market provider) and
        the conditional ``getAccountInfo(pool)`` (1, with both providers: counted always,
        since whether a tracked pool went missing is only known during the collection).
        Each logical call takes at most ``max_retries + 1`` attempts."""
        rpc = 0 if self.provider is None else 1
        if holders and self.provider is not None:
            rpc += holder_request_bound(self.provider, self.settings.holder_max_pages)
        dex = 0
        if market and self.dex is not None:
            dex = 1
            if self.provider is not None:
                rpc += 1
        logical = rpc + dex
        return CollectionRequestBound(logical, logical * (self.settings.max_retries + 1))

    def add_target(self, raw: str, source: str = "manual") -> tuple[str, bool]:
        cid, mint = solana_identity(raw)
        return cid, self.repo.add_target(cid, mint, source, ts(self._now()))

    def pin_pool(
        self, raw: str, pool_address: str, dex: str | None = None, source: str = "manual"
    ) -> bool:
        """Pin an exact pool on a target: its owner is excluded as POOL_OR_VAULT in
        snapshots with ``as_of`` at or after now (never retroactively)."""
        cid, mint = solana_identity(raw)
        self._mint(cid)
        pool = pool_address.strip()
        if pool != pool_address or not is_solana_address(pool):
            raise InvalidRequestError(f"{pool_address!r} is not a valid Solana address")
        if pool == mint:
            raise InvalidRequestError("the pool address can't be the mint itself")
        if dex is not None and dex not in KNOWN_DEXES:
            raise InvalidRequestError(f"unknown dex {dex!r} (known: {', '.join(KNOWN_DEXES)})")
        return self.repo.pin_pool(cid, pool, dex, source, ts(self._now()))

    def _mint(self, canonical_id: str) -> str:
        mint = self.repo.target_mint(canonical_id)
        if mint is None:
            raise InvalidRequestError(f"{canonical_id} is not a Safety V2 target (add it first)")
        return mint

    async def _holders(self, mint: str) -> tuple[HolderObservation, str]:
        if self.provider is None:
            return HolderObservation("NOT_COLLECTED", NO_PROVIDER), "none"
        name = self.provider.name
        needed = holder_request_bound(self.provider, self.settings.holder_max_pages)
        try:
            self.guard.check()
            if self.guard.remaining_today() < needed:
                raise SafetyBudgetExhaustedError(
                    f"a bounded holder collection needs up to {needed} requests and only "
                    f"{self.guard.remaining_today()} remain in today's budget"
                )
            obs = await collect_holders(self.provider, mint, self.settings.holder_max_pages,
                                        self.settings.holder_owner_lookups)  # fmt: skip
        except SafetyProviderError as exc:
            obs = holder_not_observed(exc)
        return obs, name

    async def _market(
        self, canonical_id: str, mint: str, collection: int
    ) -> tuple[int, MarketObservation, PoolAccountObservation | None]:
        """One DEX market observation, then (only when the tracked pool was previously
        reported and is now missing) one getAccountInfo(pool). At most 2 requests."""
        obs: MarketObservation
        if self.dex is None:
            obs, name = MarketObservation("NOT_COLLECTED", NO_DEX), "none"
        else:
            name = self.dex.name
            try:
                obs = classify_market(mint, await self.dex.token_pairs(mint))
            except SafetyProviderError as exc:
                obs = market_not_observed(exc)
        now = ts(self._now())
        mid = self.repo.record_market_observation(canonical_id, collection, now, name, obs)
        if obs.outcome not in ("POOLS", "NO_POOLS"):
            return mid, obs, None
        pins = self.repo.pool_pins(canonical_id, now)
        view = market_view(
            mint, self.repo.market_observations(canonical_id, now),
            [(p.pool_address, p.dex, p.pinned_at) for p in pins], (),
        )  # fmt: skip
        if not view.closure_applicable or view.anchor is None:
            return mid, obs, None  # no RPC unless the closure rule is applicable
        account: PoolAccountObservation
        if self.provider is None:
            account, rpc = PoolAccountObservation("NOT_COLLECTED", NO_PROVIDER), "none"
        else:
            rpc = self.provider.name
            try:
                result = await self.provider.get_account_info(view.anchor)
            except SafetyProviderError as exc:
                account = pool_account_not_observed(exc)
            else:
                account = classify_pool_account(view.anchor, result, rpc)
        self.repo.record_pool_account(canonical_id, collection, view.anchor,
                                      ts(self._now()), rpc, account)  # fmt: skip
        return mid, obs, account

    async def collect(
        self, canonical_id: str, holders: bool = False, market: bool = False
    ) -> Collected:
        """One getAccountInfo(mint) observation, plus, with `holders`, one bounded holder
        observation (at most ``4 + holder_max_pages`` more logical provider calls) and, with
        `market`, one market observation (at most 2 more)."""
        mint = self._mint(canonical_id)
        started = ts(self._now())
        collection = self.repo.start_collection(canonical_id, started)
        before = self.guard.used_this_run
        try:
            obs: MintObservation
            if self.provider is None:
                obs, name = MintObservation("NOT_COLLECTED", NO_PROVIDER), "none"
            else:
                name = self.provider.name
                try:
                    result = await self.provider.get_account_info(mint)
                except SafetyProviderError as exc:
                    obs = not_observed(exc)
                else:
                    obs = classify_mint_account(mint, result, name)
            fetched = ts(self._now())
            oid = self.repo.record_mint_observation(canonical_id, collection, fetched, name, obs)
            reasons = [obs.reason] if obs.reason else []
            hid = hobs = None
            if holders:
                hobs, hname = await self._holders(mint)
                hid = self.repo.record_holder_observation(
                    canonical_id, collection, ts(self._now()), hname, hobs
                )
                if hobs.reason:
                    reasons.append(f"holders: {hobs.reason}")
            mid = mobs = account = None
            if market:
                mid, mobs, account = await self._market(canonical_id, mint, collection)
                if mobs.reason:
                    reasons.append(f"market: {mobs.reason}")
                if account is not None and account.reason:
                    reasons.append(f"pool account: {account.reason}")
            # Durable Radar capture (read-only, no network): one transaction, after the
            # holder observation so this collection's owners can get wallet proof.
            captured_at = ts(self._now())
            latest = self.repo.latest_holder_observation(canonical_id, captured_at)
            owners = (
                [b.owner for b in self.repo.holder_balances(latest.id) if b.owner is not None]
                if latest is not None and latest.outcome == "COLLECTED"
                else []
            )
            cap = read_radar_capture(self.settings.radar_db_path, canonical_id, owners)
            _, radar_new, _ = self.repo.record_radar_capture(
                canonical_id, collection, captured_at, cap
            )
            if cap.reason:
                reasons.append(f"radar: {cap.reason}")
            used = self.guard.used_this_run - before
            self.repo.finish_collection(collection, "DONE", ts(self._now()), used, reasons)
        except BaseException as exc:
            self.repo.finish_collection(
                collection, "ABORTED", max(ts(self._now()), started),
                self.guard.used_this_run - before, [f"aborted: {type(exc).__name__}: {exc}"],
            )  # fmt: skip
            raise
        return Collected(collection, oid, obs.outcome, used, hid,
                         hobs.outcome if hobs else None, mid,
                         mobs.outcome if mobs else None,
                         account.outcome if account else None, cap.status,
                         radar_new)  # fmt: skip

    def wallet_proofs(self, canonical_id: str, wallets: list[str], as_of: float) -> WalletProofs:
        """Positive wallet proof from Safety's captured Radar evidence only (never live
        Radar): usable when ``source_time <= as_of`` and ``captured_at <= as_of``. The
        status is the latest capture attempt at or before `as_of`."""
        if not wallets:
            return NOT_CONSULTED
        run = self.repo.latest_radar_capture(canonical_id, as_of)
        proofs = self.repo.captured_wallet_proofs(canonical_id, wallets, as_of)
        if run is None:
            return WalletProofs("NOT_CONFIGURED", "no Radar capture at or before as_of", proofs)
        rid, status, reason, _ = run
        mapped: ProofStatus = (
            "AVAILABLE" if status == "CAPTURED"
            else "NOT_CONFIGURED" if status == "NOT_CONFIGURED"
            else "INCOMPATIBLE" if status == "INCOMPATIBLE"
            else "UNAVAILABLE"
        )  # fmt: skip
        return WalletProofs(mapped, reason, proofs, rid)

    def holder_inputs(self, canonical_id: str, as_of: float) -> HolderInputs:
        """Stored holder evidence, pool pins and Radar wallet proofs known at `as_of`."""
        obs = self.repo.latest_holder_observation(canonical_id, as_of)
        if obs is None:
            return HolderInputs(None)
        balances = tuple(self.repo.holder_balances(obs.id))
        wallets = sorted({b.owner for b in balances if b.owner is not None})
        proofs = self.wallet_proofs(canonical_id, wallets, as_of)
        pools = tuple(self.repo.pool_pins(canonical_id, as_of))
        return HolderInputs(obs, balances, pools, proofs)

    def market_inputs(self, canonical_id: str, as_of: float) -> MarketInputs:
        """Stored market observations, pool pins and pool-account checks known at `as_of`."""
        observations = tuple(self.repo.market_observations(canonical_id, as_of))
        if not observations:
            return MarketInputs()
        return MarketInputs(observations, tuple(self.repo.pool_pins(canonical_id, as_of)),
                            tuple(self.repo.pool_accounts(canonical_id, as_of)))  # fmt: skip

    def creator_inputs(self, canonical_id: str, as_of: float) -> CreatorInputs:
        """Safety-captured Radar evidence usable at `as_of` (never live Radar)."""
        capture = self.repo.latest_radar_capture(canonical_id, as_of)
        if capture is None:
            return CreatorInputs()
        facts = self.repo.captured_facts
        return CreatorInputs(
            capture,
            tuple(facts("safety_radar_creator_evidence", canonical_id, as_of)),
            tuple(facts("safety_radar_flow_evidence", canonical_id, as_of)),
            tuple(facts("safety_radar_activity_coverage", canonical_id, as_of)),
        )

    def change_inputs(self, canonical_id: str, as_of: float) -> ChangeInputs:
        """The latest strictly earlier complete holder observation (in (fetched_at, id)
        order), with pins and wallet proofs as knowable at its own fetched_at."""
        observations = self.repo.holder_observations(canonical_id, as_of)
        priors = observations[:-1]
        for prev in reversed(priors):
            balances = tuple(self.repo.holder_balances(prev.id))
            if not is_complete(prev, balances):
                continue
            at = prev.fetched_at
            wallets = sorted({b.owner for b in balances if b.owner is not None})
            inputs = HolderInputs(prev, balances, tuple(self.repo.pool_pins(canonical_id, at)),
                                  self.wallet_proofs(canonical_id, wallets, at))  # fmt: skip
            return ChangeInputs(True, inputs)
        return ChangeInputs(bool(priors), None)

    def build(self, canonical_id: str, as_of: float) -> dict[str, Any]:
        """The body from stored Safety inputs known at `as_of` (no request, no Radar read)."""
        mint = self._mint(canonical_id)
        row = self.repo.latest_mint_observation(canonical_id, as_of)
        holders = self.holder_inputs(canonical_id, as_of)
        market = self.market_inputs(canonical_id, as_of)
        creator = self.creator_inputs(canonical_id, as_of)
        changes = self.change_inputs(canonical_id, as_of)
        return build_body(canonical_id, mint, as_of, row, self._fingerprints(), holders, market,
                          creator, changes)  # fmt: skip

    async def snapshot(
        self,
        canonical_id: str,
        fetch: bool = True,
        save: bool = True,
        holders: bool = False,
        market: bool = False,
    ) -> SnapshotResult:
        collected = await self.collect(canonical_id, holders, market) if fetch else None
        as_of = self._now()
        body = self.build(canonical_id, ts(as_of))
        _, _, digest = encode_body(body)
        result = SnapshotResult(canonical_id, as_of, body, digest, collected)
        if save:
            assessment = body["assessment"]
            sid, stored = self.repo.save_snapshot(
                canonical_id, ts(as_of), assessment["coverage"], assessment["band"],
                body["provenance"]["fingerprints"], body,
            )  # fmt: skip
            assert stored == digest
            result.saved, result.snapshot_id = True, sid
        return result

    def rebuild(self, snapshot_id: int) -> RebuildResult:
        """Rebuild a stored snapshot from stored inputs at its own ``as_of``. Exact
        reproduction is claimed only under the same schema, rules version and code
        fingerprints; otherwise the mismatch is reported and nothing is rebuilt."""
        row = self.repo.snapshot(snapshot_id)
        current = dict(self._fingerprints())
        stored = dict(row.fingerprints)
        stored.setdefault("rules_version", row.rules_version)
        current_meta = {"schema_version": SNAPSHOT_SCHEMA, "rules_version": RULES_VERSION}
        mismatched = {
            k: (stored.get(k), current.get(k))
            for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)
        }
        for k, v in current_meta.items():
            have = row.schema_version if k == "schema_version" else row.rules_version
            if have != v:
                mismatched[k] = (have, v)
        if mismatched:
            return RebuildResult(
                snapshot_id, "FINGERPRINT_MISMATCH", row.body_hash, None, mismatched
            )
        body = self.build(row.canonical_id, row.as_of)
        _, _, digest = encode_body(body)
        status: RebuildStatus = "REPRODUCED" if digest == row.body_hash else "DIVERGED"
        return RebuildResult(snapshot_id, status, row.body_hash, digest)


def service_from_env(
    settings: SafetySettings | None = None, env: Mapping[str, str] | None = None
) -> SafetyService:
    """The `SafetyService` the Safety CLI uses: `settings` (else ``UPSCALE_SAFETY_V2_*``),
    Helius when ``UPSCALE_HELIUS_API_KEY`` is set, else the plain RPC at
    ``UPSCALE_SOLANA_RPC_URL``, else no Solana provider; the DEX provider from
    ``settings.dex_url``. Credentials are passed through only (never logged or stored)."""
    source = os.environ if env is None else env
    return SafetyService(
        settings or load_settings(source),
        helius_api_key=source.get("UPSCALE_HELIUS_API_KEY") or None,
        rpc_url=source.get("UPSCALE_SOLANA_RPC_URL") or None,
    )
