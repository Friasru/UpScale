"""One strategy's paper book: its decisions, positions, simulated fills and accounting.

Pure and deterministic: it is driven by `Event`s in time order and holds no connection,
clock or provider. What it may know at event time t is exactly the events up to t (and an
Analyze lookup bounded to t, passed in by the engine); every output row is final when
emitted.

Accounting (simulated money only, `IDEALIZED_NO_FEES`):

* entry: ``cost = quantity x entry price`` leaves cash;
* each exit fill: ``proceeds = quantity x observed exit price`` returns to cash; realized
  P/L is proceeds minus that quantity's cost;
* equity = cash + open positions marked at their exact pool's last observed price;
  unrealized P/L = that mark minus the open cost basis;
* MARKET_UNAVAILABLE closes a position without a price: its cost basis is reported as
  ``unresolved_cost_usd`` and the position is valued at zero in equity (written off:
  conservative, never a fabricated exit); it is left out of return statistics. Metrics
  also report a last-mark view (reporting only, see `metrics`).

Market availability (`config.AvailabilityPolicy`, per run):

* LEGACY_V1: a position with no exact-pool price for `market_unavailable_after_minutes`,
  or a triggered exit (signal / max hold) that found no price within
  `max_exit_delay_minutes`, is closed as MARKET_UNAVAILABLE at that deadline (`sweep`).
  Held-position watch evidence is ignored entirely.
* EVIDENCE_AWARE_V2: missing evidence is never a market exit. `sweep` closes nothing; a
  position without a recent price stays open, and `market_state` labels it (reporting
  only): MARKET_AVAILABLE (a price within PRICE_STALE_AFTER_MINUTES), LIQUIDITY_COLLAPSE
  (that price's pool liquidity below LIQUIDITY_COLLAPSE_USD), PRICE_STALE (older, up to
  `market_unavailable_after_minutes`), PROVIDER_UNAVAILABLE (the latest watch lookup
  failed / was rate limited), EVIDENCE_GAP (older still, nothing explains it),
  MARKET_NOT_FOUND (the provider that priced the pool answered without it). A triggered
  exit (TP / SL / trailing / max hold / signal) still fills only at an observed
  exact-pool price: it waits as long as it takes. The one priceless close is a confirmed
  MARKET_NOT_FOUND: NOT_FOUND_CONFIRMATIONS authoritative not-found observations of the
  exact pool spanning at least `max_exit_delay_minutes` with no valid price in between
  (closed at the confirming observation, as MARKET_UNAVAILABLE with reason code
  MARKET_NOT_FOUND).

Execution model (`config.ExecutionConfig`, per run; None: IDEALIZED_NO_FEES, unchanged):

* REALISTIC_V1 (EVIDENCE_AWARE_V2 runs only): the strategy's decisions are made exactly as
  above, but an ENTER or a triggered exit is an *intent*. It fills only at the first valid
  exact-pool price observed at or after intent time + latency (pool identity compared
  chain-aware: EVM lowercased, Solana case-sensitive), with adverse slippage, fees and
  optional price impact (`_execute`). Until then it is pending: an entry intent reserves its
  cash (it counts toward equity at cost and holds the asset slot) and is cancelled without
  a fill after `entry_max_wait_minutes`; an exit intent waits for a price as long as it
  takes. Take profit / stop loss / trailing levels are measured on observed market prices
  against the observed entry price (`Position.trigger_price`), so the rules read the same
  market moves as IDEALIZED_NO_FEES; the position's `entry_price` is its all-in cost per
  unit (fees, slippage, impact included), so cost, proceeds and realized P/L are net.
"""

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from upscale.services.chains import same_address
from upscale.services.shadow.config import (
    CONFIDENCE_ORDER,
    LEGACY_POLICY,
    RISK_ORDER,
    SPAM_ORDER,
    Action,
    AvailabilityPolicy,
    ExecutionConfig,
    ExitReason,
    StrategyConfig,
)
from upscale.services.shadow.evidence import (
    AnalyzeView,
    Event,
    PriceObs,
    ScoutView,
    WatchStatus,
)

AnalyzeLookup = Callable[[str, datetime, timedelta], AnalyzeView | None]
PRICE_BASIS = "OBSERVED_EXACT_POOL_PRICE"
REALISTIC_BASIS = "REALISTIC_V1_EXECUTION_PRICE"  # observed price, adverse slippage / impact
MIN_TIMING_VERSION = 2  # Scout records before it carry the legacy (run start) timing
EPS = 1e-12
# EVIDENCE_AWARE_V2 market availability (see the module docstring).
MarketState = Literal[
    "MARKET_AVAILABLE", "PRICE_STALE", "EVIDENCE_GAP", "PROVIDER_UNAVAILABLE",
    "MARKET_NOT_FOUND", "LIQUIDITY_COLLAPSE",
]  # fmt: skip
PRICE_STALE_AFTER_MINUTES = 60.0
LIQUIDITY_COLLAPSE_USD = 1_000.0  # Scout's own floor for a usable pool
NOT_FOUND_CONFIRMATIONS = 2
_DATES = ("entry_at", "entry_price_at", "last_price_at", "pending_since", "closed_at",
          "watch_status_at", "not_found_since", "pending_eligible_at")  # fmt: skip
# REALISTIC_V1-only position facts: left out of an IDEALIZED_NO_FEES book's state.
_REALISTIC = ("trigger_price", "pending_quantity", "pending_level", "pending_eligible_at",
              "pending_reference_price")  # fmt: skip


def _id(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


def _iso(t: datetime | None) -> str | None:
    return t.isoformat() if t is not None else None


@dataclass
class Position:
    position_id: str
    asset_id: str
    chain: str | None
    address: str | None
    symbol: str | None
    pool: str
    dex: str | None
    entry_decision_id: str
    entry_at: datetime
    entry_price: float
    entry_price_at: datetime
    quantity: float
    cost_usd: float
    remaining_quantity: float
    last_price: float
    last_price_at: datetime
    peak_price: float
    trough_price: float
    tp_hit: int = 0
    fills: int = 0
    pending_exit: str | None = None  # a SIGNAL_EXIT / MAX_HOLD_TIME awaiting a price
    pending_since: datetime | None = None
    pending_decision_id: str | None = None
    closed_at: datetime | None = None
    exit_reason: str | None = None
    # EVIDENCE_AWARE_V2 availability facts (event-derived; reset by every valid price).
    last_liquidity_usd: float | None = None  # of the last price's pool (reporting only)
    watch_status: str | None = None  # latest watch lookup without a price, after it
    watch_status_at: datetime | None = None
    not_found_count: int = 0  # authoritative MARKET_NOT_FOUND observations since it
    not_found_since: datetime | None = None
    # REALISTIC_V1: the observed exact-pool price at the entry fill (the rules' reference;
    # `entry_price` is the all-in cost per unit) and the pending exit intent's order.
    trigger_price: float | None = None
    pending_quantity: float | None = None  # None: everything remaining at fill time
    pending_level: float | None = None
    pending_eligible_at: datetime | None = None
    pending_reference_price: float | None = None

    @property
    def basis(self) -> float:
        """The price entry-relative rules and excursions are measured against."""
        return self.trigger_price if self.trigger_price is not None else self.entry_price

    @property
    def remaining_cost(self) -> float:
        return self.remaining_quantity * self.entry_price

    @property
    def value(self) -> float:
        return self.remaining_quantity * self.last_price

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k in _DATES:
            d[k] = _iso(d[k])
        for k in _REALISTIC:
            if d[k] is None:
                del d[k]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Position":
        d = dict(d)
        for k in _DATES:
            if k in d:
                d[k] = datetime.fromisoformat(d[k]) if d.get(k) else None
        return cls(**d)

    def market_state(self, now: datetime, unavailable_after_minutes: float) -> MarketState:
        """EVIDENCE_AWARE_V2's label for this open position at `now` (reporting only)."""
        if self.not_found_count > 0:
            return "MARKET_NOT_FOUND"
        age = now - self.last_price_at
        if age <= timedelta(minutes=PRICE_STALE_AFTER_MINUTES):
            low = (
                self.last_liquidity_usd is not None
                and self.last_liquidity_usd < LIQUIDITY_COLLAPSE_USD
            )
            return "LIQUIDITY_COLLAPSE" if low else "MARKET_AVAILABLE"
        if self.watch_status in ("PROVIDER_FAILED", "RATE_LIMITED"):
            return "PROVIDER_UNAVAILABLE"
        if age <= timedelta(minutes=unavailable_after_minutes):
            return "PRICE_STALE"
        return "EVIDENCE_GAP"


@dataclass
class PendingEntry:
    """A REALISTIC_V1 entry intent: its ENTER decision is final, its fill is not yet."""

    decision_id: str
    asset_id: str
    chain: str | None
    address: str | None
    symbol: str | None
    pool: str
    dex: str | None
    decided_at: datetime
    eligible_at: datetime  # the first observation time that may fill it
    expires_at: datetime | None
    budget_usd: float  # reserved cash: fee + notional
    reference_price: float  # the decision's observed price (never a fill price)
    reference_price_at: datetime

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("decided_at", "eligible_at", "expires_at", "reference_price_at"):
            d[k] = _iso(d[k])
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PendingEntry":
        d = dict(d)
        for k in ("decided_at", "eligible_at", "expires_at", "reference_price_at"):
            d[k] = datetime.fromisoformat(d[k]) if d.get(k) else None
        return cls(**d)


@dataclass
class Output:
    """Rows produced by one step, written together with the next checkpoint."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    opened: list[Position] = field(default_factory=list)
    closed: list[Position] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    # One (decision time, outcome, reasons) per Scout evaluation, for the exact aggregate
    # counters: ENTERED, BLOCKED (reasons: the control), REJECTED (reasons: every failed
    # rule) or HELD (asset already held: not evaluated).
    evaluations: list[tuple[datetime, str, tuple[str, ...]]] = field(default_factory=list)
    executions: list[dict[str, Any]] = field(default_factory=list)  # REALISTIC_V1 fills

    def extend(self, other: "Output") -> None:
        self.executions += other.executions
        self.rejections += other.rejections
        self.evaluations += other.evaluations
        self.decisions += other.decisions
        self.opened += other.opened
        self.closed += other.closed
        self.trades += other.trades
        self.equity += other.equity


class Book:
    def __init__(
        self,
        run_id: str,
        cfg: StrategyConfig,
        state: dict[str, Any] | None = None,
        policy: AvailabilityPolicy = LEGACY_POLICY,
        execution: ExecutionConfig | None = None,
    ):
        self.run_id = run_id
        self.cfg = cfg
        self.policy = policy
        self.execution = execution
        if execution is not None and policy != "EVIDENCE_AWARE_V2":
            raise ValueError("REALISTIC_V1 execution requires the EVIDENCE_AWARE_V2 policy")
        s = state or {}
        self.pending_entries: dict[str, PendingEntry] = {
            k: PendingEntry.from_json(v) for k, v in (s.get("pending_entries") or {}).items()
        }
        self.cash: float = s.get("cash", cfg.risk.initial_capital_usd)
        self.positions: dict[str, Position] = {
            k: Position.from_json(v) for k, v in (s.get("positions") or {}).items()
        }
        self.realized_pnl: float = s.get("realized_pnl", 0.0)
        self.unresolved_cost: float = s.get("unresolved_cost", 0.0)
        self.peak_equity: float = s.get("peak_equity", cfg.risk.initial_capital_usd)
        self.max_drawdown_pct: float = s.get("max_drawdown_pct", 0.0)
        self.entries: dict[str, list[str]] = s.get("entries", {})  # asset -> entry times
        self.last_exit: dict[str, str] = s.get("last_exit", {})
        self.counts: dict[str, int] = s.get("counts", {})
        self.closed_positions: int = s.get("closed_positions", 0)
        self.sequence: int = s.get("sequence", 0)
        self.touched: set[str] = set()  # positions whose marks changed since the last flush

    # --- state ------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        out = self._state()
        if self.execution is not None:
            out["pending_entries"] = {k: e.to_json() for k, e in self.pending_entries.items()}
        return out

    def _state(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": {k: p.to_json() for k, p in self.positions.items()},
            "realized_pnl": self.realized_pnl,
            "unresolved_cost": self.unresolved_cost,
            "peak_equity": self.peak_equity,
            "max_drawdown_pct": self.max_drawdown_pct,
            "entries": self.entries,
            "last_exit": self.last_exit,
            "counts": self.counts,
            "closed_positions": self.closed_positions,
            "sequence": self.sequence,
        }

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def open_value(self) -> float:
        return sum(p.value for p in self.positions.values())

    @property
    def pending_cost(self) -> float:
        """Cash reserved by pending REALISTIC_V1 entry intents (valued at cost)."""
        return sum(e.budget_usd for e in self.pending_entries.values())

    @property
    def equity(self) -> float:
        if self.pending_entries:
            return self.cash + self.open_value + self.pending_cost
        return self.cash + self.open_value

    @property
    def model(self) -> str:
        return self.execution.name if self.execution is not None else self.cfg.execution_model

    def _same_pool(self, asset_id: str, a: str, b: str) -> bool:
        """Exact pool identity. REALISTIC_V1 compares chain-aware (EVM addresses are
        case-insensitive, Solana's are not); IDEALIZED_NO_FEES keeps its exact comparison."""
        if self.execution is None:
            return a == b
        return same_address(asset_id.split(":", 1)[0], a, b)

    @property
    def evidence_aware(self) -> bool:
        return self.policy == "EVIDENCE_AWARE_V2"

    def market_states(self, now: datetime) -> dict[str, int]:
        """Open positions by EVIDENCE_AWARE_V2 market state at `now` (reporting only)."""
        out: dict[str, int] = {}
        after = self.cfg.exit.market_unavailable_after_minutes
        for p in self.positions.values():
            state = p.market_state(now, after)
            out[state] = out.get(state, 0) + 1
        return dict(sorted(out.items()))

    def snapshot(self, at: datetime) -> dict[str, Any]:
        equity = self.equity
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (equity / self.peak_equity - 1) * 100 if self.peak_equity > 0 else 0.0
        self.max_drawdown_pct = min(self.max_drawdown_pct, drawdown)
        open_cost = sum(p.remaining_cost for p in self.positions.values())
        return {
            "run_id": self.run_id,
            "strategy_id": self.cfg.strategy_id,
            "strategy_version": self.cfg.version,
            "at": at,
            "cash": self.cash,
            "open_value": self.open_value,
            "equity": equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.open_value - open_cost,
            "unresolved_cost": self.unresolved_cost,
            "exposure_pct": self.open_value / equity * 100 if equity > 0 else 0.0,
            "open_positions": len(self.positions),
            "drawdown_pct": drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
        }

    # --- events -----------------------------------------------------------------------------

    def sweep(self, now: datetime) -> Output:
        """Time-based closes known by `now` (every event up to it has been applied):
        a triggered exit that found no price in time, or a pool with no observed price for
        too long. Closed without a price at the rule's own deadline (deterministic).
        LEGACY_V1 only: under EVIDENCE_AWARE_V2 missing evidence never closes a position."""
        out = Output()
        if self.execution is not None:
            self._expire_entries(out, now)
        if self.evidence_aware:
            return out
        x = self.cfg.exit
        delay = timedelta(minutes=x.max_exit_delay_minutes)
        for p in sorted(self.positions.values(), key=lambda p: p.position_id):
            limits: list[tuple[datetime, str]] = []
            if p.pending_since is not None:
                limits.append((p.pending_since + delay, f"{p.pending_exit} triggered, no observed "
                               f"price of the exact pool within {x.max_exit_delay_minutes:g} min"))  # fmt: skip
            deadline = p.entry_at + timedelta(minutes=x.max_hold_minutes)
            limits.append((deadline + delay, "maximum hold time reached, no observed price of "
                           f"the exact pool within {x.max_exit_delay_minutes:g} min"))  # fmt: skip
            stale = p.last_price_at + timedelta(minutes=x.market_unavailable_after_minutes)
            limits.append((stale, "no observed price of the exact pool for "
                           f"{x.market_unavailable_after_minutes:g} min"))  # fmt: skip
            at, why = min(limits, key=lambda t: t[0])
            if now > at:
                decision = self._decision("EXIT", at, p.asset_id, why, position=p,
                                          evidence={"reason_code": "MARKET_UNAVAILABLE"})  # fmt: skip
                out.decisions.append(decision)
                self._fill(out, p, at, None, p.remaining_quantity, "MARKET_UNAVAILABLE",
                           decision["decision_id"], None)  # fmt: skip
        return out

    def on_event(self, event: Event, analyze: AnalyzeLookup) -> Output:
        if event.from_watch and not self.evidence_aware:
            return Output()  # LEGACY_V1 reads Scout-derived evidence only, as it always did
        out = self.sweep(event.at)
        if event.price is not None:
            out.extend(self.on_price(event.price, event.at))
        if event.status is not None:
            out.extend(self.on_status(event.status))
        if event.scout is not None:
            out.extend(self.on_scout(event.scout, analyze))
        return out

    def on_price(self, obs: PriceObs, at: datetime) -> Output:
        """A price of an exact pool, observed at `obs.at` and learned at event time `at`
        (the same for a market record; a Scout decision's time for its embedded price):
        any exit it triggers is decided and filled at `at`, at that observed price."""
        if self.execution is not None:
            return self._on_price_realistic(obs, at)
        out = Output()
        for p in sorted(self.positions.values(), key=lambda p: p.position_id):
            # Exact-token identity: the same token AND the same pool; never another market.
            if p.asset_id != obs.asset_id or p.pool != obs.pool or obs.at <= p.last_price_at:
                continue
            p.last_price, p.last_price_at = obs.price, obs.at
            if self.evidence_aware:  # a valid price ends every "no price" state
                p.last_liquidity_usd = obs.liquidity_usd
                p.watch_status = p.watch_status_at = p.not_found_since = None
                p.not_found_count = 0
            p.peak_price = max(p.peak_price, obs.price)
            p.trough_price = min(p.trough_price, obs.price)
            self.touched.add(p.position_id)
            self._price_exits(out, p, obs, at)
        return out

    def _on_price_realistic(self, obs: PriceObs, at: datetime) -> Output:
        """REALISTIC_V1: fill pending intents eligible at this observation (its own
        observation time at or after intent time + latency), then mark the open positions
        and turn any rule it triggers into an exit intent."""
        out = Output()
        for e in sorted(self.pending_entries.values(), key=lambda e: e.decision_id):
            if (e.asset_id == obs.asset_id and self._same_pool(e.asset_id, e.pool, obs.pool)
                    and obs.at >= e.eligible_at):  # fmt: skip
                self._fill_entry(out, e, obs, at)
        for p in sorted(self.positions.values(), key=lambda p: p.position_id):
            if (p.asset_id != obs.asset_id or not self._same_pool(p.asset_id, p.pool, obs.pool)
                    or obs.at <= p.last_price_at):  # fmt: skip
                continue
            p.last_price, p.last_price_at = obs.price, obs.at
            p.last_liquidity_usd = obs.liquidity_usd
            p.watch_status = p.watch_status_at = p.not_found_since = None
            p.not_found_count = 0
            p.peak_price = max(p.peak_price, obs.price)
            p.trough_price = min(p.trough_price, obs.price)
            self.touched.add(p.position_id)
            if p.pending_exit is not None:
                assert p.pending_eligible_at is not None
                if obs.at >= p.pending_eligible_at:
                    qty = p.remaining_quantity if p.pending_quantity is None else p.pending_quantity
                    assert p.pending_decision_id is not None
                    self._fill(out, p, at, obs, qty, p.pending_exit,  # type: ignore[arg-type]
                               p.pending_decision_id, p.pending_level)  # fmt: skip
                continue  # one intent at a time: rules resume once it has filled
            self._price_exits(out, p, obs, at)
        return out

    def on_status(self, st: WatchStatus) -> Output:
        """EVIDENCE_AWARE_V2: a watch lookup of an exact pool that found no price. Only an
        authoritative not-found, confirmed (NOT_FOUND_CONFIRMATIONS over at least
        `max_exit_delay_minutes`, no valid price in between), closes the position."""
        out = Output()
        delay = timedelta(minutes=self.cfg.exit.max_exit_delay_minutes)
        for p in sorted(self.positions.values(), key=lambda p: p.position_id):
            if (p.asset_id != st.asset_id or not self._same_pool(p.asset_id, p.pool, st.pool)
                    or st.at <= p.last_price_at):  # fmt: skip
                continue
            p.watch_status, p.watch_status_at = st.status, st.at
            if st.status != "NOT_FOUND" or not st.authoritative:
                continue
            p.not_found_count += 1
            p.not_found_since = p.not_found_since or st.at
            if p.not_found_count < NOT_FOUND_CONFIRMATIONS or st.at - p.not_found_since < delay:
                continue
            link = {"kind": "market", "record_id": st.record_id, "fingerprint": None,
                    "observed_at": st.at.isoformat()}  # fmt: skip
            d = self._decision(
                "EXIT", st.at, p.asset_id,
                f"MARKET_NOT_FOUND: {st.provider or 'the provider'} that priced the exact pool "
                f"reported it not found {p.not_found_count} times since "
                f"{p.not_found_since.isoformat()}, no observed price since "
                f"{p.last_price_at.isoformat()}",
                position=p, links=[link],
                evidence={"reason_code": "MARKET_NOT_FOUND", "watch_record": st.record_id,
                          "not_found_count": p.not_found_count},
            )  # fmt: skip
            out.decisions.append(d)
            self._fill(out, p, st.at, None, p.remaining_quantity, "MARKET_UNAVAILABLE",
                       d["decision_id"], None)  # fmt: skip
        return out

    def _price_exits(self, out: Output, p: Position, obs: PriceObs, at: datetime) -> None:
        x = self.cfg.exit
        price, entry = obs.price, p.basis
        reason: ExitReason | None = None
        level: float | None = None
        if p.pending_exit is not None:
            reason = p.pending_exit  # type: ignore[assignment]
        elif x.stop_loss_pct is not None and price <= entry * (1 - x.stop_loss_pct / 100):
            reason, level = "STOP_LOSS", entry * (1 - x.stop_loss_pct / 100)
        elif (
            x.trailing_stop_pct is not None
            and p.peak_price >= entry * (1 + x.trailing_activation_pct / 100)
            and price <= p.peak_price * (1 - x.trailing_stop_pct / 100)
        ):
            reason, level = "TRAILING_STOP", p.peak_price * (1 - x.trailing_stop_pct / 100)
        if reason is not None:
            decision_id = p.pending_decision_id
            if decision_id is None:
                d = self._decision("EXIT", at, p.asset_id,
                                   f"{reason}: observed {price:.10g} (level {level:.10g})",
                                   position=p, evidence={"price": _price_json(obs)}, reference=obs,
                                   links=[_price_link(obs)])  # fmt: skip
                out.decisions.append(d)
                decision_id = d["decision_id"]
            self._exit(out, p, at, obs, p.remaining_quantity, reason, decision_id, level, True)
            return
        # Take-profit ladder: fractions of the original quantity, the last level sells the rest.
        levels = x.take_profit
        while p.tp_hit < len(levels) and price >= entry * (1 + levels[p.tp_hit].gain_pct / 100):
            lv = levels[p.tp_hit]
            last = p.tp_hit == len(levels) - 1
            qty = (
                p.remaining_quantity
                if last
                else min(p.quantity * lv.fraction, p.remaining_quantity)
            )
            p.tp_hit += 1
            d = self._decision("EXIT", at, p.asset_id,
                               f"TAKE_PROFIT level {p.tp_hit} (+{lv.gain_pct:g}%): observed {price:.10g}",
                               position=p, evidence={"price": _price_json(obs)}, reference=obs,
                               links=[_price_link(obs)])  # fmt: skip
            out.decisions.append(d)
            self._exit(out, p, at, obs, qty, "TAKE_PROFIT", d["decision_id"],
                       entry * (1 + lv.gain_pct / 100), last)  # fmt: skip
            if p.position_id not in self.positions or p.pending_exit is not None:
                return
        if p.pending_exit is not None:
            return
        if at >= p.entry_at + timedelta(minutes=x.max_hold_minutes):
            d = self._decision("EXIT", at, p.asset_id,
                               f"MAX_HOLD_TIME ({x.max_hold_minutes:g} min): observed {price:.10g}",
                               position=p, evidence={"price": _price_json(obs)}, reference=obs,
                               links=[_price_link(obs)])  # fmt: skip
            out.decisions.append(d)
            self._exit(out, p, at, obs, p.remaining_quantity, "MAX_HOLD_TIME",
                       d["decision_id"], None, True)  # fmt: skip

    def _exit(
        self,
        out: Output,
        p: Position,
        at: datetime,
        obs: PriceObs,
        qty: float,
        reason: ExitReason,
        decision_id: str,
        level: float | None,
        everything: bool,
    ) -> None:
        """A triggered exit: filled now at the triggering observation (IDEALIZED_NO_FEES),
        or an intent for the first eligible later observation (REALISTIC_V1)."""
        if self.execution is None:
            self._fill(out, p, at, obs, qty, reason, decision_id, level)
        else:
            self._intend_exit(p, at, obs.price, None if everything else qty, reason,
                              decision_id, level)  # fmt: skip

    def _intend_exit(
        self,
        p: Position,
        at: datetime,
        reference: float,
        qty: float | None,
        reason: ExitReason,
        decision_id: str,
        level: float | None,
    ) -> None:
        assert self.execution is not None
        p.pending_exit, p.pending_since, p.pending_decision_id = reason, at, decision_id
        p.pending_quantity, p.pending_level = qty, level
        p.pending_eligible_at = at + timedelta(seconds=self.execution.latency_seconds)
        p.pending_reference_price = reference
        self._count(f"intent:{reason}")
        self.touched.add(p.position_id)

    def on_scout(self, s: ScoutView, analyze: AnalyzeLookup) -> Output:
        out = Output()
        if s.causal_valid is False:
            self._count("skipped:causally_invalid_scout_record")
            return out
        if s.timing_version is None or s.timing_version < MIN_TIMING_VERSION:
            self._count("skipped:legacy_scout_timing")
            return out
        price = s.price
        if price is not None:
            out.extend(self.on_price(price, s.decision_at))
        held = [p for p in self.positions.values() if p.asset_id == s.asset_id]
        for p in sorted(held, key=lambda p: p.position_id):
            self._signal_exit(out, p, s)
        held = [p for p in self.positions.values() if p.asset_id == s.asset_id]
        ordered = sorted((e for e in self.pending_entries.values() if e.asset_id == s.asset_id),
                         key=lambda e: e.decision_id)  # fmt: skip
        risk = self.cfg.risk
        if (held or ordered) and not risk.allow_scaling:
            out.evaluations.append((s.decision_at, "HELD", ()))
            for p in held:
                self._count("decision:HOLD")
                out.decisions.append(self._decision(
                    "HOLD", s.decision_at, s.asset_id, "position already open (one per asset)",
                    position=p, scout=s,
                ))  # fmt: skip
            for _ in ordered:
                self._count("decision:HOLD")
                out.decisions.append(self._decision(
                    "HOLD", s.decision_at, s.asset_id,
                    "entry order pending (one per asset)", scout=s,
                ))  # fmt: skip
            return out
        reasons, detail = self._evaluate(s, analyze)
        if reasons:
            # Diagnostics only: a rejection row never becomes a decision and never changes
            # the book (no sequence number, no state besides the counter).
            self._count(f"rejected:{reasons[0]}")
            out.rejections.append(self._rejection(s, reasons, detail))
            out.evaluations.append((s.decision_at, "REJECTED", tuple(reasons)))
            return out
        assert price is not None  # _evaluate requires the exact-pool price
        blocked = self._risk_block(s, len(held) + len(ordered))
        size = 0.0
        if blocked is None:
            size, blocked = self._size(s.asset_id)
        if blocked is not None:
            # Qualified by the signal rules but stopped by position / duplicate / risk
            # control: recorded, so suppressed repeats stay visible.
            self._count(f"blocked:{blocked}")
            out.evaluations.append((s.decision_at, "BLOCKED", (blocked,)))
            out.decisions.append(self._decision(
                "NO_ACTION", s.decision_at, s.asset_id, f"qualified but blocked: {blocked}",
                scout=s, analyze=self._analyze_for(s, analyze),
            ))  # fmt: skip
            return out
        if self.execution is None:
            self._enter(out, s, price, size, analyze)
        else:
            self._intend_entry(out, s, price, size, analyze)
        out.evaluations.append((s.decision_at, "ENTERED", ()))
        return out

    # --- entries ----------------------------------------------------------------------------

    def _analyze_for(self, s: ScoutView, analyze: AnalyzeLookup) -> AnalyzeView | None:
        rules = self.cfg.entry.analyze
        if rules is None:
            return None
        return analyze(s.asset_id, s.decision_at, timedelta(minutes=rules.max_age_minutes))

    def _evaluate(self, s: ScoutView, analyze: AnalyzeLookup) -> tuple[list[str], dict[str, Any]]:
        """Every entry rule this Scout evaluation fails, as stable reason codes (empty: it
        qualifies), plus details. Only evidence already at hand is inspected: the Scout
        record and, for strategies with Analyze rules, the archived Analyze decision at or
        before the decision time. The random draw only selects among otherwise qualifying
        candidates, so RANDOM_BASELINE_NOT_SELECTED is only ever the sole reason."""
        e = self.cfg.entry
        r: list[str] = []
        detail: dict[str, Any] = {}
        if e.require_eligible and not s.eligible:
            r.append("SCOUT_NOT_ELIGIBLE")
        if e.require_current_data and s.data_status != "CURRENT":
            r.append("STALE_DATA")
        if s.price is None:
            r.append("CURRENT_PRICE_UNAVAILABLE")  # never trade without the exact pool's price
        if s.market_status == "MARKET_COLLAPSE":
            r.append("MARKET_COLLAPSE")
        if s.stage not in e.allowed_stages:
            r.append("STAGE_NOT_ALLOWED")
        if s.score is None or s.score < e.min_scout_score:
            r.append("SCORE_BELOW_MIN")
        if s.liquidity_usd is None or s.liquidity_usd < e.min_liquidity_usd:
            r.append("LIQUIDITY_BELOW_MIN")
        if e.max_risk_penalty is not None and (
            s.risk_penalty is None or s.risk_penalty > e.max_risk_penalty
        ):
            r.append("RISK_PENALTY_TOO_HIGH")
        blocking = [c for c, sev, _ in s.risk_flags if sev in e.blocking_flag_severities]
        if blocking:
            r.append("BLOCKING_RISK_FLAG")
            detail["blocking_flags"] = blocking
        missing = s.missing()
        # Missing evidence the strategy does not allow: the label is the reason code
        # (SAFETY_NOT_AVAILABLE, TECHNICAL_NOT_AVAILABLE, SOCIAL_NOT_AVAILABLE, ...).
        r += sorted(missing - set(e.allowed_missing))
        safety_known = "SAFETY_NOT_AVAILABLE" not in missing
        if e.required_safety == "COMPLETE" and s.safety_status != "SAFETY_CHECKS_COMPLETE":
            r.append("SAFETY_LEVEL_INSUFFICIENT" if safety_known else "SAFETY_NOT_AVAILABLE")
        if e.required_safety == "PARTIAL_OR_COMPLETE" and s.safety_status not in (
            "SAFETY_CHECKS_COMPLETE", "SAFETY_CHECKS_PARTIAL",
        ):  # fmt: skip
            r.append("SAFETY_LEVEL_INSUFFICIENT" if safety_known else "SAFETY_NOT_AVAILABLE")
        if e.block_active_authorities and s.mint_authority_active:
            r.append("MINT_AUTHORITY_ACTIVE")
        if e.block_active_authorities and s.freeze_authority_active:
            r.append("FREEZE_AUTHORITY_ACTIVE")
        if e.max_holder_top10_pct is not None and (
            s.holder_top10_pct is not None and s.holder_top10_pct > e.max_holder_top10_pct
        ):
            r.append("HOLDER_CONCENTRATION_TOO_HIGH")
        if e.technical is not None and s.technical is not None:
            t, tr = s.technical, e.technical
            snapshots = t.get("snapshots")
            if t.get("trend") not in tr.allowed_trends:
                r.append("TECHNICAL_TREND_NOT_ALLOWED")
            if not isinstance(snapshots, int) or snapshots < tr.min_snapshots:
                r.append("TECHNICAL_TOO_FEW_SNAPSHOTS")
            if tr.require_breakout and t.get("breakout") is not True:
                r.append("TECHNICAL_BREAKOUT_REQUIRED")
            if tr.require_volume_confirmed and t.get("volume_confirmed") is not True:
                r.append("TECHNICAL_VOLUME_NOT_CONFIRMED")
            if tr.require_higher_lows and t.get("higher_lows") is not True:
                r.append("TECHNICAL_HIGHER_LOWS_REQUIRED")
        if e.social is not None and "SOCIAL_NOT_AVAILABLE" not in missing:
            so = e.social
            failed = []
            if so.allowed_statuses is not None and s.social_status not in so.allowed_statuses:
                failed.append("STATUS")
            if (
                so.max_spam_risk is not None
                and SPAM_ORDER.get(s.spam_risk or "", 3) > SPAM_ORDER[so.max_spam_risk]
            ):
                failed.append("SPAM_RISK")
            if failed:
                r.append("SOCIAL_REQUIREMENT_FAILED")
                detail["social_failed"] = failed
        # The Analyze lookup reads the local archive only, bounded to the decision time.
        if e.analyze is not None:
            a = self._analyze_for(s, analyze)
            ar = e.analyze
            failed = []
            if a is None:
                failed.append("NOT_AVAILABLE")
            else:
                if a.observed_at > s.decision_at:
                    raise AssertionError("Analyze evidence later than the decision time")
                detail["analyze"] = a.summary()
                if a.action not in ar.allowed_actions:
                    failed.append("ACTION")
                if (
                    CONFIDENCE_ORDER.get(a.confidence or "", -1)
                    < CONFIDENCE_ORDER[ar.min_confidence]
                ):
                    failed.append("CONFIDENCE")
                if (
                    ar.max_risk_level is not None
                    and RISK_ORDER.get(a.risk_level or "", 9) > RISK_ORDER[ar.max_risk_level]
                ):
                    failed.append("RISK")
                if ar.allowed_technical_trends is not None and (
                    a.technical_trend not in ar.allowed_technical_trends
                ):
                    failed.append("TECHNICAL")
            if failed:
                r.append("ANALYZE_REQUIREMENT_FAILED")
                detail["analyze_failed"] = failed
        if r:
            return list(dict.fromkeys(r)), detail
        if e.random_fraction is not None:
            key = f"{e.random_seed}|{self.cfg.strategy_id}|{s.asset_id}|{s.decision_at.isoformat()}"
            draw = int(hashlib.sha256(key.encode()).hexdigest()[:15], 16) / float(16**15)
            if draw >= e.random_fraction:
                detail["random"] = {"draw": draw, "fraction": e.random_fraction,
                                    "seed": e.random_seed}  # fmt: skip
                return ["RANDOM_BASELINE_NOT_SELECTED"], detail
        return [], detail

    def _rejection(
        self, s: ScoutView, reasons: list[str], detail: dict[str, Any]
    ) -> dict[str, Any]:
        """An immutable diagnostic row: why this strategy did not enter on this Scout
        evaluation, with the observed values the rules looked at."""
        t = s.technical or {}
        observed = {
            "score": s.score, "stage": s.stage, "eligible": s.eligible,
            "data_status": s.data_status, "price_usd": s.price_usd,
            "current_price_available": s.price is not None,
            "market_status": s.market_status, "liquidity_usd": s.liquidity_usd,
            "risk_penalty": s.risk_penalty, "safety_status": s.safety_status,
            "mint_authority_active": s.mint_authority_active,
            "freeze_authority_active": s.freeze_authority_active,
            "holder_top10_pct": s.holder_top10_pct, "social_status": s.social_status,
            "spam_risk": s.spam_risk,
            "technical": {k: t.get(k) for k in ("trend", "snapshots", "breakout",
                                                "volume_confirmed", "higher_lows")}
            if s.technical is not None else None,
            "missing": sorted(s.missing()),
            **detail,
        }  # fmt: skip
        return {
            "rejection_id": _id(self.run_id, self.cfg.key, "rejection", s.record_id),
            "run_id": self.run_id,
            "strategy_id": self.cfg.strategy_id,
            "strategy_version": self.cfg.version,
            "asset_id": s.asset_id,
            "pool": s.pool,
            "decision_at": s.decision_at,
            "scout_record_id": s.record_id,
            "reasons": reasons,
            "observed": observed,
            "fingerprints": s.fingerprints(),
        }

    def _risk_block(self, s: ScoutView, held: int) -> str | None:
        risk = self.cfg.risk
        at = s.decision_at
        if held >= risk.max_positions_per_asset:
            return "MAX_POSITIONS_PER_ASSET"
        last_exit = self.last_exit.get(s.asset_id)
        if last_exit is not None and at - datetime.fromisoformat(last_exit) < timedelta(
            minutes=risk.entry_cooldown_minutes
        ):
            return "COOLDOWN"
        times = [datetime.fromisoformat(t) for t in self.entries.get(s.asset_id, [])]
        if times and at - max(times) < timedelta(minutes=risk.min_entry_spacing_minutes):
            return "MIN_ENTRY_SPACING"
        if sum(1 for t in times if t.date() == at.date()) >= risk.max_entries_per_asset_per_day:
            return "MAX_ENTRIES_PER_ASSET_PER_DAY"
        if len(self.positions) + len(self.pending_entries) >= risk.max_open_positions:
            return "MAX_OPEN_POSITIONS"
        return None

    def _size(self, asset_id: str) -> tuple[float, str | None]:
        risk = self.cfg.risk
        equity = self.equity
        asset_value = sum(p.value for p in self.positions.values() if p.asset_id == asset_id)
        caps = {
            "POSITION_SIZE": risk.position_size_usd,
            "MAX_ALLOCATION": equity * risk.max_allocation_pct / 100,
            "INSUFFICIENT_CASH": self.cash,
            "MAX_ASSET_EXPOSURE": equity * risk.max_exposure_per_asset_pct / 100 - asset_value,
            "MAX_GROSS_EXPOSURE": equity * risk.max_gross_exposure_pct / 100 - self.open_value,
        }
        if self.pending_entries:  # reserved cash is committed exposure
            caps["MAX_GROSS_EXPOSURE"] -= self.pending_cost
        binding = min(caps, key=lambda k: caps[k])
        size = caps[binding]
        if size < risk.min_position_usd:
            return 0.0, binding if binding != "POSITION_SIZE" else "INSUFFICIENT_CAPITAL"
        return size, None

    def _enter(
        self, out: Output, s: ScoutView, price: PriceObs, size: float, analyze: AnalyzeLookup
    ) -> None:
        a = self._analyze_for(s, analyze)
        d = self._decision(
            "ENTER", s.decision_at, s.asset_id,
            f"{self.cfg.name}: Scout {s.stage} score {s.score:.1f}, liquidity "
            f"${(s.liquidity_usd or 0):,.0f}; entry at the observed exact-pool price",
            scout=s, analyze=a, reference=price,
        )  # fmt: skip
        out.decisions.append(d)
        qty = size / price.price
        p = Position(
            position_id=_id(self.run_id, self.cfg.key, "position", d["decision_id"]),
            asset_id=s.asset_id, chain=s.chain, address=s.address, symbol=s.symbol,
            pool=price.pool, dex=s.dex, entry_decision_id=d["decision_id"],
            entry_at=s.decision_at, entry_price=price.price, entry_price_at=price.at,
            quantity=qty, cost_usd=size, remaining_quantity=qty, last_price=price.price,
            last_price_at=price.at, peak_price=price.price, trough_price=price.price,
        )  # fmt: skip
        d["position_id"] = p.position_id
        self.cash -= size
        self.positions[p.position_id] = p
        self.entries.setdefault(s.asset_id, []).append(s.decision_at.isoformat())
        # Only what duplicate control can still need (spacing, the UTC day's count).
        keep = s.decision_at - timedelta(days=2)
        self.entries[s.asset_id] = [
            t for t in self.entries[s.asset_id] if datetime.fromisoformat(t) >= keep
        ]
        self._count("decision:ENTER")
        out.opened.append(p)

    def _intend_entry(
        self, out: Output, s: ScoutView, price: PriceObs, size: float, analyze: AnalyzeLookup
    ) -> None:
        """REALISTIC_V1: the ENTER decision (at the Scout decision time, as always) and an
        entry intent reserving `size`; the position exists once the intent fills."""
        assert self.execution is not None
        a = self._analyze_for(s, analyze)
        latency = timedelta(seconds=self.execution.latency_seconds)
        wait = self.execution.entry_max_wait_minutes
        d = self._decision(
            "ENTER", s.decision_at, s.asset_id,
            f"{self.cfg.name}: Scout {s.stage} score {s.score:.1f}, liquidity "
            f"${(s.liquidity_usd or 0):,.0f}; entry order for the first exact-pool price "
            f"observed {self.execution.latency_seconds:g}s after the decision",
            scout=s, analyze=a, reference=price,
        )  # fmt: skip
        out.decisions.append(d)
        e = PendingEntry(
            decision_id=d["decision_id"], asset_id=s.asset_id, chain=s.chain, address=s.address,
            symbol=s.symbol, pool=price.pool, dex=s.dex, decided_at=s.decision_at,
            eligible_at=s.decision_at + latency,
            expires_at=s.decision_at + latency + timedelta(minutes=wait) if wait else None,
            budget_usd=size, reference_price=price.price, reference_price_at=price.at,
        )  # fmt: skip
        self.cash -= size
        self.pending_entries[e.decision_id] = e
        self.entries.setdefault(s.asset_id, []).append(s.decision_at.isoformat())
        keep = s.decision_at - timedelta(days=2)
        self.entries[s.asset_id] = [
            t for t in self.entries[s.asset_id] if datetime.fromisoformat(t) >= keep
        ]
        self._count("decision:ENTER")

    def _fill_entry(self, out: Output, e: PendingEntry, obs: PriceObs, at: datetime) -> None:
        x = self._execute("BUY", obs, e.budget_usd)
        qty = x["quantity"]
        p = Position(
            position_id=_id(self.run_id, self.cfg.key, "position", e.decision_id),
            asset_id=e.asset_id, chain=e.chain, address=e.address, symbol=e.symbol,
            pool=e.pool, dex=e.dex, entry_decision_id=e.decision_id, entry_at=at,
            entry_price=e.budget_usd / qty, entry_price_at=obs.at, quantity=qty,
            cost_usd=e.budget_usd, remaining_quantity=qty, last_price=obs.price,
            last_price_at=obs.at, peak_price=obs.price, trough_price=obs.price,
            last_liquidity_usd=obs.liquidity_usd, trigger_price=obs.price,
        )  # fmt: skip
        del self.pending_entries[e.decision_id]
        self.positions[p.position_id] = p
        self.touched.add(p.position_id)
        out.opened.append(p)
        out.executions.append(self._execution_row(
            p, "BUY", "ENTRY", 1, e.decision_id, e.decided_at, e.eligible_at, at, obs,
            e.reference_price, x,
            gross_pnl=None, friction=x["friction_usd"], net_pnl=None,
        ))  # fmt: skip
        self._count("fill:ENTER")

    def _expire_entries(self, out: Output, now: datetime) -> None:
        """Entry intents with no eligible observation by their deadline (only with a
        finite `entry_max_wait_minutes`): cancelled at the deadline as an explicit NO_ACTION
        decision, never filled, not a trade; the reserved cash returns (no P/L)."""
        for e in sorted(self.pending_entries.values(), key=lambda e: e.decision_id):
            if e.expires_at is None or now <= e.expires_at:
                continue
            assert self.execution is not None
            out.decisions.append(self._decision(
                "NO_ACTION", e.expires_at, e.asset_id,
                f"entry order cancelled: no exact-pool price observed within "
                f"{self.execution.entry_max_wait_minutes:g} min of its eligible time",
                evidence={"reason_code": "ENTRY_NOT_FILLED", "entry_decision_id": e.decision_id,
                          "released_usd": e.budget_usd, "not_a_trade": True},
            ))  # fmt: skip
            self.cash += e.budget_usd
            del self.pending_entries[e.decision_id]
            self._count("cancelled:ENTER")

    def _execute(self, side: str, obs: PriceObs, amount: float) -> dict[str, Any]:
        """One REALISTIC_V1 execution at observation `obs`. BUY: `amount` is the cash spent
        (fee included); SELL: `amount` is the quantity sold. Adverse slippage and impact
        are applied to the observed price; fees on the traded notional."""
        assert self.execution is not None
        ex = self.execution
        P = obs.price
        liquidity = obs.liquidity_usd
        impact_status = "DISABLED"
        impact = 0.0  # as a fraction of the observed price
        if side == "BUY":
            fee = amount * ex.entry_fee_bps / 1e4
            notional = amount - fee
            trade_value = notional
        else:
            fee = 0.0
            notional = 0.0
            trade_value = amount * P
        if ex.price_impact:
            if liquidity is None or liquidity <= 0:
                impact_status = "LIQUIDITY_UNAVAILABLE"
            else:
                y = liquidity / 2  # the quote reserve of a constant-product pool
                raw = trade_value / y if side == "BUY" else trade_value / (y + trade_value)
                impact = min(raw, ex.max_price_impact_bps / 1e4)
                impact_status = "APPLIED" if raw <= ex.max_price_impact_bps / 1e4 else "CAPPED"
        slip = ex.slippage_bps / 1e4
        if side == "BUY":
            price = P * (1 + slip + impact)
            qty = notional / price
            observed_value = qty * P
            slip_cost, impact_cost = observed_value * slip, observed_value * impact
            cash = -amount
        else:
            qty = amount
            price = P * (1 - slip - impact)
            gross = qty * price
            fee = gross * ex.exit_fee_bps / 1e4
            observed_value = qty * P
            slip_cost, impact_cost = observed_value * slip, observed_value * impact
            notional = gross
            cash = gross - fee
        return {
            "observed_price": P, "execution_price": price, "quantity": qty,
            "notional_usd": notional, "observed_value_usd": observed_value,
            "fee_bps": ex.entry_fee_bps if side == "BUY" else ex.exit_fee_bps, "fee_usd": fee,
            "slippage_bps": ex.slippage_bps, "slippage_cost_usd": slip_cost,
            "impact_bps": impact * 1e4, "impact_cost_usd": impact_cost,
            "impact_status": impact_status, "liquidity_usd": liquidity,
            "friction_usd": fee + slip_cost + impact_cost, "cash_flow_usd": cash,
        }  # fmt: skip

    def _execution_row(
        self, p: Position, side: str, reason: str, fill_no: int, decision_id: str,
        intent_at: datetime, eligible_at: datetime, at: datetime, obs: PriceObs,
        reference: float, x: dict[str, Any], *, gross_pnl: float | None, friction: float,
        net_pnl: float | None,
    ) -> dict[str, Any]:  # fmt: skip
        assert self.execution is not None
        qty = x["quantity"]
        # Price drift between the intent's reference observation and the fill observation
        # (adverse positive): measurable, part of gross P/L, not of trading friction.
        drift = qty * (obs.price - reference) if side == "BUY" else qty * (reference - obs.price)
        return {
            "execution_id": _id(self.run_id, self.cfg.key, p.position_id, side, fill_no),
            "run_id": self.run_id, "strategy_id": self.cfg.strategy_id,
            "strategy_version": self.cfg.version, "position_id": p.position_id,
            "side": side, "reason": reason, "fill_no": fill_no, "decision_id": decision_id,
            "asset_id": p.asset_id, "pool": p.pool, "intent_at": intent_at,
            "eligible_at": eligible_at, "filled_at": at, "price_observed_at": obs.at,
            "price_record": obs.record_id, "price_source": obs.source,
            "reference_price": reference, **x,
            "latency_seconds": self.execution.latency_seconds,
            "delay_seconds": (obs.at - intent_at).total_seconds(),
            "latency_cost_usd": drift,
            "gross_pnl_usd": gross_pnl, "trade_friction_usd": friction, "net_pnl_usd": net_pnl,
            "execution_model": self.execution.name, "config_hash": self.execution.config_hash,
        }  # fmt: skip

    # --- exits ------------------------------------------------------------------------------

    def _signal_exit(self, out: Output, p: Position, s: ScoutView) -> None:
        x = self.cfg.exit
        why: str | None = None
        if s.stage in x.signal_exit_stages:
            why = f"Scout stage {s.stage}"
        elif x.signal_exit_below_score is not None and (
            s.score is not None and s.score < x.signal_exit_below_score
        ):
            why = f"Scout score {s.score:.1f} below {x.signal_exit_below_score:g}"
        elif x.signal_exit_on_market_collapse and s.market_status == "MARKET_COLLAPSE":
            why = "Scout reports MARKET_COLLAPSE"
        if why is None or p.pending_exit is not None:
            return
        d = self._decision("EXIT", s.decision_at, s.asset_id, f"SIGNAL_EXIT: {why}",
                           position=p, scout=s)  # fmt: skip
        out.decisions.append(d)
        price = s.price
        if self.execution is not None:
            ref = price.price if price is not None and self._same_pool(
                p.asset_id, price.pool, p.pool) else p.last_price  # fmt: skip
            self._intend_exit(p, s.decision_at, ref, None, "SIGNAL_EXIT", d["decision_id"], None)
            return
        if price is not None and price.pool == p.pool and price.at >= p.last_price_at:
            self._fill(out, p, s.decision_at, price, p.remaining_quantity, "SIGNAL_EXIT",
                       d["decision_id"], None)  # fmt: skip
        else:
            # No current price of the exact pool in this evaluation: the exit fills at the
            # next observed price of that pool (or becomes MARKET_UNAVAILABLE in time).
            p.pending_exit, p.pending_since = "SIGNAL_EXIT", s.decision_at
            p.pending_decision_id = d["decision_id"]
            self.touched.add(p.position_id)

    def _fill(
        self,
        out: Output,
        p: Position,
        at: datetime,
        obs: PriceObs | None,
        qty: float,
        reason: ExitReason,
        decision_id: str,
        level: float | None,
    ) -> None:
        p.fills += 1
        qty = min(qty, p.remaining_quantity)
        cost = qty * p.entry_price
        final = p.remaining_quantity - qty <= p.quantity * 1e-9
        if final:
            qty, cost = p.remaining_quantity, p.remaining_cost
        exit_price = obs.price if obs is not None else None
        proceeds = qty * exit_price if exit_price is not None else None
        x: dict[str, Any] | None = None
        if self.execution is not None and obs is not None:
            x = self._execute("SELL", obs, qty)
            exit_price, proceeds = x["execution_price"], x["cash_flow_usd"]
        pnl = proceeds - cost if proceeds is not None else None
        if proceeds is not None and pnl is not None:
            self.cash += proceeds
            self.realized_pnl += pnl
        else:
            self.unresolved_cost += cost
        p.remaining_quantity = 0.0 if final else p.remaining_quantity - qty
        mfe = max(0.0, p.peak_price / p.basis - 1) * 100
        mae = min(0.0, p.trough_price / p.basis - 1) * 100
        if x is not None and obs is not None and pnl is not None:
            assert p.trigger_price is not None and p.pending_since is not None
            assert p.pending_eligible_at is not None and p.pending_reference_price is not None
            # Gross: the observed-price P/L of this quantity; friction: this fill's exit
            # costs plus its share of the entry costs; gross - friction = net, exactly.
            entry_friction = qty * (p.entry_price - p.trigger_price)
            gross = qty * (obs.price - p.trigger_price)
            out.executions.append(self._execution_row(
                p, "SELL", reason, p.fills, decision_id, p.pending_since,
                p.pending_eligible_at, at, obs, p.pending_reference_price, x,
                gross_pnl=gross, friction=x["friction_usd"] + entry_friction, net_pnl=pnl,
            ))  # fmt: skip
            p.pending_exit = p.pending_since = p.pending_decision_id = None
            p.pending_quantity = p.pending_level = p.pending_eligible_at = None
            p.pending_reference_price = None
        out.trades.append({
            "trade_id": _id(self.run_id, self.cfg.key, p.position_id, "fill", p.fills),
            "run_id": self.run_id, "strategy_id": self.cfg.strategy_id,
            "strategy_version": self.cfg.version, "position_id": p.position_id,
            "fill_no": p.fills, "final": final, "asset_id": p.asset_id, "chain": p.chain,
            "address": p.address, "symbol": p.symbol, "pool": p.pool,
            "entry_decision_id": p.entry_decision_id, "exit_decision_id": decision_id,
            "entry_at": p.entry_at, "entry_price": p.entry_price, "exit_at": at,
            "exit_price": exit_price, "exit_price_at": obs.at if obs is not None else None,
            "exit_price_record": obs.record_id if obs is not None else None,
            "quantity": qty, "fraction": qty / p.quantity if p.quantity else None,
            "cost_usd": cost, "proceeds_usd": proceeds, "pnl_usd": pnl,
            "return_pct": (proceeds / cost - 1) * 100 if x is not None and proceeds is not None
            else (exit_price / p.entry_price - 1) * 100 if exit_price is not None else None,
            "mfe_pct": mfe, "mae_pct": mae,
            "holding_minutes": (at - p.entry_at).total_seconds() / 60,
            "exit_reason": reason, "trigger_level": level,
            "price_basis": (REALISTIC_BASIS if x is not None else PRICE_BASIS)
            if obs is not None else "NO_PRICE",
            "execution_model": self.model,
        })  # fmt: skip
        self._count(f"exit:{reason}")
        self.touched.add(p.position_id)
        if final:
            del self.positions[p.position_id]
            p.pending_exit = None
            p.closed_at, p.exit_reason = at, reason
            self.closed_positions += 1
            self.last_exit[p.asset_id] = at.isoformat()
            out.closed.append(p)

    # --- decisions --------------------------------------------------------------------------

    def _decision(
        self,
        action: Action,
        at: datetime,
        asset_id: str,
        reason: str,
        *,
        position: Position | None = None,
        scout: ScoutView | None = None,
        analyze: AnalyzeView | None = None,
        reference: PriceObs | None = None,
        evidence: dict[str, Any] | None = None,
        links: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.sequence += 1
        body: dict[str, Any] = dict(evidence or {})
        fps: list[dict[str, Any]] = list(links or [])
        if scout is not None:
            body |= scout.summary()
            fps += scout.fingerprints()
        if analyze is not None:
            body["opportunity"] = analyze.summary()
            fps.append(analyze.fingerprint_link())
        ref = reference or (scout.price if scout is not None else None)
        return {
            "decision_id": _id(
                self.run_id, self.cfg.key, action, asset_id, at.isoformat(), self.sequence
            ),  # fmt: skip
            "run_id": self.run_id,
            "strategy_id": self.cfg.strategy_id,
            "strategy_version": self.cfg.version,
            "action": action,
            "asset_id": asset_id,
            "chain": scout.chain if scout else position.chain if position else None,
            "address": scout.address if scout else position.address if position else None,
            "pool": position.pool if position else scout.pool if scout else None,
            "decision_at": at,
            "reference_price": ref.price if ref is not None else None,
            "reference_price_at": ref.at if ref is not None else None,
            "reason": reason,
            "evidence": body,
            "fingerprints": fps,
            "position_id": position.position_id if position else None,
        }


def _price_json(obs: PriceObs) -> dict[str, Any]:
    return {"price": obs.price, "observed_at": obs.at.isoformat(), "pool": obs.pool,
            "source": obs.source, "record_id": obs.record_id}  # fmt: skip


def _price_link(obs: PriceObs) -> dict[str, Any]:
    return {"kind": obs.source, "record_id": obs.record_id, "fingerprint": None,
            "observed_at": obs.at.isoformat()}  # fmt: skip
