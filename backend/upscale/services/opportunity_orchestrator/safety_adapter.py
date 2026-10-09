"""The real Safety port: Safety V2's own service behind B2's synchronous `SafetyPort`.

The ONLY orchestrator file allowed to import Safety V2 (and only its ``service``,
``config`` and ``models`` modules: Safety's isolation test enforces it). Every other part
of the bridge depends on `SafetyPort` alone.

* Construction: `service_from_env` (exactly the Safety CLI's provider / DEX selection);
  credentials are never read, printed or stored here.
* Async: a FRESH `SafetyService` per async operation, run by one ``asyncio.run`` (its
  RequestGuard holds an ``asyncio.Lock`` that must never cross event loops). Called from a
  thread that already runs an event loop, it refuses (`SafetyAdapterError`).
* Preflight: Safety's own `collection_request_bound` (never recomputed here), today's
  remaining budget and cooldown; OK only when the Solana provider and the DEX provider are
  configured, no cooldown is active and ``remaining >= attempt_max + reserve``. A
  conservative estimate, never a guarantee: Safety's RequestGuard decides.
* Collection: ``add_target`` (idempotent; only here, on the live path) then
  ``collect(holders=True, market=True)``; the typed outcomes are classified (`classify`)
  into observed token evidence (snapshot with ``fetch=False``, then Opportunity) or
  infrastructure (no snapshot, no decision). Radar is never run or configured: Safety reads
  its captured Radar evidence only if ``UPSCALE_SAFETY_V2_RADAR_DB`` is already set.
"""

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeVar, get_args

from upscale.services.opportunity_orchestrator.config import LIVE_BUDGET_RESERVE
from upscale.services.opportunity_orchestrator.safety_port import (
    CollectionResult,
    InfraCategory,
    PortInvariantError,
    Preflight,
    ReadOnlySafetyPort,
    RequestBoundBreach,
    SafetyInspection,
    SnapshotRef,
    TokenOutcome,
)
from upscale.services.safety_v2.config import SafetySettings, load_settings
from upscale.services.safety_v2.models import MintOutcome
from upscale.services.safety_v2.service import (
    Collected,
    CollectionRequestBound,
    SafetyService,
    service_from_env,
)

TARGET_SOURCE = "opportunity_orchestrator"
# A positively observed mint result: genuine identity evidence on its own.
IDENTITY_EVIDENCE: tuple[MintOutcome, ...] = ("NOT_A_MINT", "ACCOUNT_MISSING", "MALFORMED")
FAILED, NOT_COLLECTED = "PROVIDER_FAILED", "NOT_COLLECTED"
RpcKind = Literal["HELIUS", "RPC", "NONE"]
T = TypeVar("T")


class SafetyAdapterError(PortInvariantError):
    """The adapter refused to run (e.g. inside a running event loop)."""


def run_async(make: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run one coroutine with ``asyncio.run``; refuse inside a running event loop (a nested
    loop would fail, and a background thread would hide it)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    raise SafetyAdapterError("refusing a Safety call inside a running event loop")


@dataclass(frozen=True)
class Capabilities:
    rpc: RpcKind
    dex_configured: bool

    @property
    def ready(self) -> bool:
        return self.rpc != "NONE" and self.dex_configured


@dataclass(frozen=True)
class BudgetView:
    day: str
    used_today: int
    remaining: int
    cooldown_until: datetime | None
    bound: CollectionRequestBound
    reserve: int


@dataclass(frozen=True)
class CollectionReport:
    canonical_id: str
    requests: int  # Safety's own count of attempts this collection made
    bound: CollectionRequestBound
    result: CollectionResult
    outcomes: tuple[tuple[str, str | None], ...]  # (component, Safety outcome)


def classify(
    c: Collected, caps: Capabilities, cooldown_until: datetime | None, holder_source: str | None
) -> CollectionResult:
    """Safety's typed outcomes -> token evidence or infrastructure, in this order:
    A. a positively observed non-mint / missing / malformed account is identity evidence
       (later holder / market failures don't erase it);
    B. otherwise any required component PROVIDER_FAILED or NOT_COLLECTED is
       infrastructure: PROVIDER_RATE_LIMITED while Safety's cooldown is active (whoever
       started it; retry no earlier than its end), else PROVIDER_UNAVAILABLE for a failure,
       else PROVIDER_NOT_CONFIGURED / SAFETY_BUDGET_EXHAUSTED for a part never attempted;
    C. otherwise observed token evidence (complete / partial holders, pools or none, a pool
       account present or missing).
    The caller decides what infrastructure costs: inside a started collection, one attempt."""
    parts = (("mint", c.outcome), ("holders", c.holder_outcome), ("market", c.market_outcome),
             ("pool_account", c.pool_account_outcome))  # fmt: skip
    if c.outcome in IDENTITY_EVIDENCE:
        outcome: TokenOutcome = c.outcome  # type: ignore[assignment]
        return CollectionResult("TOKEN_EVIDENCE", outcome, detail=f"mint {c.outcome}")
    failed = [k for k, v in parts if v == FAILED]
    missing = [k for k, v in parts[:3] if v in (NOT_COLLECTED, None)]
    missing += [k for k, v in parts[3:] if v == NOT_COLLECTED]
    if failed or missing:
        what = "; ".join(x for x in (f"failed: {', '.join(failed)}" if failed else "",
                                     f"not collected: {', '.join(missing)}" if missing else "") if x)  # fmt: skip
        if cooldown_until is not None:
            return CollectionResult("INFRASTRUCTURE", "PROVIDER_RATE_LIMITED", cooldown_until,
                                    f"Safety cooling down; {what}")  # fmt: skip
        category: InfraCategory = (
            "PROVIDER_UNAVAILABLE" if failed
            else "PROVIDER_NOT_CONFIGURED" if not caps.ready else "SAFETY_BUDGET_EXHAUSTED"
        )  # fmt: skip
        return CollectionResult("INFRASTRUCTURE", category, None, what)
    if c.pool_account_outcome == "ACCOUNT_MISSING":
        outcome = "MARKET_CLOSED"
    elif c.market_outcome == "NO_POOLS":
        outcome = "NO_POOLS"
    elif holder_source != "full_scan":
        outcome = "HOLDERS_PARTIAL"
    else:
        outcome = "COMPLETE"
    return CollectionResult("TOKEN_EVIDENCE", outcome, detail=f"holders {holder_source}")


assert set(IDENTITY_EVIDENCE) <= set(get_args(MintOutcome))


class RealSafetyPort:
    """`SafetyPort` over Safety V2's service. `factory` builds a fresh service from the
    settings (default: `service_from_env`, the Safety CLI's construction)."""

    def __init__(
        self,
        safety_db: str,
        reserve: int = LIVE_BUDGET_RESERVE,
        factory: Callable[[SafetySettings], SafetyService] | None = None,
        env: Mapping[str, str] | None = None,
    ):
        self.settings = load_settings(env).model_copy(update={"db_path": safety_db})
        self.reserve = reserve
        self._factory = factory or (lambda s: service_from_env(s, env))
        self.reports: list[CollectionReport] = []

    def _service(self) -> SafetyService:
        return self._factory(self.settings)

    # --- read-only views -----------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        svc = self._service()
        rpc: RpcKind = ("NONE" if svc.provider is None
                        else "HELIUS" if svc.provider.supports_scan else "RPC")  # fmt: skip
        return Capabilities(rpc, svc.dex is not None)

    def budget(self) -> BudgetView:
        svc = self._service()
        day = svc.guard.day()
        return BudgetView(day, svc.repo.requests_on(day), svc.guard.remaining_today(),
                          svc.guard.cooldown_until(),
                          svc.collection_request_bound(holders=True, market=True), self.reserve)  # fmt: skip

    # --- SafetyPort --------------------------------------------------------------------------

    def inspect(self, canonical_id: str, now: datetime) -> SafetyInspection:
        return ReadOnlySafetyPort(self.settings.db_path).inspect(canonical_id, now)

    def preflight(self, canonical_id: str, now: datetime) -> Preflight:
        caps = self.capabilities()
        if not caps.ready:
            return Preflight("CONSERVATIVE_PREFLIGHT_INSUFFICIENT", "PROVIDER_NOT_CONFIGURED",
                             detail=f"rpc={caps.rpc} dex={'yes' if caps.dex_configured else 'no'}")  # fmt: skip
        b = self.budget()
        if b.cooldown_until is not None:
            return Preflight("CONSERVATIVE_PREFLIGHT_INSUFFICIENT", "SAFETY_COOLDOWN",
                             b.cooldown_until, "Safety is cooling down after a rate limit")  # fmt: skip
        need = b.bound.attempt_max + self.reserve
        if b.remaining < need:
            return Preflight("CONSERVATIVE_PREFLIGHT_INSUFFICIENT", "SAFETY_BUDGET_EXHAUSTED",
                             detail=f"remaining {b.remaining} < bound {b.bound.attempt_max} + "
                                    f"reserve {self.reserve}")  # fmt: skip
        return Preflight("CONSERVATIVE_PREFLIGHT_OK",
                         detail=f"remaining {b.remaining} >= bound {b.bound.attempt_max} + "
                                f"reserve {self.reserve}")  # fmt: skip

    def collect(self, canonical_id: str, now: datetime) -> CollectionResult:
        svc = self._service()
        caps = Capabilities(
            "NONE" if svc.provider is None else "HELIUS" if svc.provider.supports_scan else "RPC",
            svc.dex is not None,
        )  # fmt: skip
        bound = svc.collection_request_bound(holders=True, market=True)
        svc.add_target(canonical_id, source=TARGET_SOURCE)  # idempotent

        async def go() -> Collected:
            return await svc.collect(canonical_id, holders=True, market=True)

        collected = run_async(go)
        holder = (svc.repo.holder_observation(collected.holder_observation_id)
                  if collected.holder_observation_id is not None else None)  # fmt: skip
        result = classify(collected, caps, svc.guard.cooldown_until(),
                          holder.source if holder is not None else None)  # fmt: skip
        outcomes = (("mint", collected.outcome), ("holders", collected.holder_outcome),
                    ("market", collected.market_outcome),
                    ("pool_account", collected.pool_account_outcome))  # fmt: skip
        self.reports.append(CollectionReport(canonical_id, collected.requests, bound, result,
                                             outcomes))  # fmt: skip
        if collected.requests > bound.attempt_max:
            raise RequestBoundBreach(collected.requests, bound.attempt_max)
        return result

    def snapshot(self, canonical_id: str, now: datetime) -> SnapshotRef:
        svc = self._service()

        async def go() -> Any:
            return await svc.snapshot(canonical_id, fetch=False, save=True)

        res = run_async(go)
        if res.snapshot_id is None:
            raise PortInvariantError("Safety didn't store the snapshot it built")
        return SnapshotRef(res.snapshot_id, res.as_of)
