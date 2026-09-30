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
  ``unresolved_cost_usd``, excluded from equity (conservative, never a fabricated exit)
  and from return statistics.
"""

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from upscale.services.shadow.config import (
    CONFIDENCE_ORDER,
    RISK_ORDER,
    SPAM_ORDER,
    Action,
    ExitReason,
    StrategyConfig,
)
from upscale.services.shadow.evidence import AnalyzeView, Event, PriceObs, ScoutView

AnalyzeLookup = Callable[[str, datetime, timedelta], AnalyzeView | None]
PRICE_BASIS = "OBSERVED_EXACT_POOL_PRICE"
MIN_TIMING_VERSION = 2  # Scout records before it carry the legacy (run start) timing
EPS = 1e-12


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

    @property
    def remaining_cost(self) -> float:
        return self.remaining_quantity * self.entry_price

    @property
    def value(self) -> float:
        return self.remaining_quantity * self.last_price

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("entry_at", "entry_price_at", "last_price_at", "pending_since", "closed_at"):
            d[k] = _iso(d[k])
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Position":
        d = dict(d)
        for k in ("entry_at", "entry_price_at", "last_price_at", "pending_since", "closed_at"):
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

    def extend(self, other: "Output") -> None:
        self.rejections += other.rejections
        self.evaluations += other.evaluations
        self.decisions += other.decisions
        self.opened += other.opened
        self.closed += other.closed
        self.trades += other.trades
        self.equity += other.equity


class Book:
    def __init__(self, run_id: str, cfg: StrategyConfig, state: dict[str, Any] | None = None):
        self.run_id = run_id
        self.cfg = cfg
        s = state or {}
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
    def equity(self) -> float:
        return self.cash + self.open_value

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
        too long. Closed without a price at the rule's own deadline (deterministic)."""
        out = Output()
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
        out = self.sweep(event.at)
        if event.price is not None:
            out.extend(self.on_price(event.price, event.at))
        if event.scout is not None:
            out.extend(self.on_scout(event.scout, analyze))
        return out

    def on_price(self, obs: PriceObs, at: datetime) -> Output:
        """A price of an exact pool, observed at `obs.at` and learned at event time `at`
        (the same for a market record; a Scout decision's time for its embedded price):
        any exit it triggers is decided and filled at `at`, at that observed price."""
        out = Output()
        for p in sorted(self.positions.values(), key=lambda p: p.position_id):
            # Exact-token identity: the same token AND the same pool; never another market.
            if p.asset_id != obs.asset_id or p.pool != obs.pool or obs.at <= p.last_price_at:
                continue
            p.last_price, p.last_price_at = obs.price, obs.at
            p.peak_price = max(p.peak_price, obs.price)
            p.trough_price = min(p.trough_price, obs.price)
            self.touched.add(p.position_id)
            self._price_exits(out, p, obs, at)
        return out

    def _price_exits(self, out: Output, p: Position, obs: PriceObs, at: datetime) -> None:
        x = self.cfg.exit
        price, entry = obs.price, p.entry_price
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
            self._fill(out, p, at, obs, p.remaining_quantity, reason, decision_id, level)  # fmt: skip
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
            self._fill(out, p, at, obs, qty, "TAKE_PROFIT", d["decision_id"],
                       entry * (1 + lv.gain_pct / 100))  # fmt: skip
            if p.position_id not in self.positions:
                return
        if at >= p.entry_at + timedelta(minutes=x.max_hold_minutes):
            d = self._decision("EXIT", at, p.asset_id,
                               f"MAX_HOLD_TIME ({x.max_hold_minutes:g} min): observed {price:.10g}",
                               position=p, evidence={"price": _price_json(obs)}, reference=obs,
                               links=[_price_link(obs)])  # fmt: skip
            out.decisions.append(d)
            self._fill(out, p, at, obs, p.remaining_quantity, "MAX_HOLD_TIME",
                       d["decision_id"], None)  # fmt: skip

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
        risk = self.cfg.risk
        if held and not risk.allow_scaling:
            out.evaluations.append((s.decision_at, "HELD", ()))
            for p in held:
                self._count("decision:HOLD")
                out.decisions.append(self._decision(
                    "HOLD", s.decision_at, s.asset_id, "position already open (one per asset)",
                    position=p, scout=s,
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
        blocked = self._risk_block(s, len(held))
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
        self._enter(out, s, price, size, analyze)
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
        if len(self.positions) >= risk.max_open_positions:
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
        pnl = proceeds - cost if proceeds is not None else None
        if proceeds is not None and pnl is not None:
            self.cash += proceeds
            self.realized_pnl += pnl
        else:
            self.unresolved_cost += cost
        p.remaining_quantity = 0.0 if final else p.remaining_quantity - qty
        mfe = max(0.0, p.peak_price / p.entry_price - 1) * 100
        mae = min(0.0, p.trough_price / p.entry_price - 1) * 100
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
            "return_pct": (exit_price / p.entry_price - 1) * 100 if exit_price is not None else None,
            "mfe_pct": mfe, "mae_pct": mae,
            "holding_minutes": (at - p.entry_at).total_seconds() / 60,
            "exit_reason": reason, "trigger_level": level,
            "price_basis": PRICE_BASIS if obs is not None else "NO_PRICE",
            "execution_model": self.cfg.execution_model,
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
