"""Pure, deterministic ``radar.snapshot.v1`` construction from stored inputs.

`build_snapshot` reads nothing but its arguments: the same inputs, target, settings, run
status and ``as_of`` always give the same body (JSON with sorted keys). Every input was
fetched at or before ``as_of`` (`RadarRepository.load_inputs` guarantees it, and it is
re-checked here), so a snapshot never contains information from after its own time.

Rules that keep it honest:

* A metric without data has a status and no value; it is never zero-filled.
* ``PARTIAL`` values are lower bounds (bounded scans, sampled transactions).
* A *difference* between two incomplete measurements isn't a bound of anything, so
  changes are only ``AVAILABLE`` between two complete holder scans and otherwise
  ``UNAVAILABLE``; net flows are only reported for a complete activity window.
* Wallet statistics (activity, early, repeated, timing, funding, history) use
  ``NORMAL_WALLET`` identities only: a participant typed ``NORMAL_WALLET`` in a flow, or
  an ``UNKNOWN`` one proven to be a wallet (it signed a transaction Radar read) at or
  before ``as_of``. Programs, pools / vaults and routers never count; remaining
  ``UNKNOWN`` participants are excluded and make the affected counts ``PARTIAL``.
* Flows are ``TOKEN_INFLOW`` / ``TOKEN_OUTFLOW``. Buys, sells, deployer sells and
  liquidity actions are ``NOT_SUPPORTED`` in V1.
* Cross-token comparisons (repeated wallets, groups) only cover tokens Radar tracked, so
  they are lower bounds with that selection bias stated.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from upscale.services.radar.config import SNAPSHOT_SCHEMA, RadarSettings
from upscale.services.radar.models import (
    VALUED,
    Metric,
    RadarCausalityError,
    Status,
    Target,
    available,
    iso,
    missing,
    partial,
)
from upscale.services.radar.repository import FlowRow, HolderRow, Inputs, ScanRow

NEW_WALLET_MAX_AGE_HOURS = 24.0
LIMITATIONS = (
    "Solana only; activity is read from the tracked pool's transactions only",
    "flows are wallet token balance changes (TOKEN_INFLOW / TOKEN_OUTFLOW), not buys or sells",
    "wallet statistics count NORMAL_WALLET identities only (proven by signing a "
    "transaction Radar read); programs, pools, vault authorities and known routers are "
    "excluded, and UNKNOWN participants (e.g. transfer recipients, unrecognized router "
    "or pool accounts) are excluded too, so wallet counts are lower bounds",
    "cross-token comparisons cover Radar targets only (Scout-selected tokens: selection bias)",
    "no holder history exists before Radar's first observation of a token",
)


def as_status(value: str, valued_as: Status) -> Status:
    """A stored scan status as a metric status; valued ones map to `valued_as`."""
    return _UNVALUED.get(value, valued_as)


_UNVALUED: dict[str, Status] = {
    "UNAVAILABLE": "UNAVAILABLE",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
    "PROVIDER_UNAVAILABLE": "PROVIDER_UNAVAILABLE",
    "NOT_COLLECTED": "NOT_COLLECTED",
}


def _m(metric: Metric) -> dict[str, Any]:
    return metric.model_dump()


def _round(value: float) -> float:
    return round(value, 6)


def _scan_window(inp: Inputs, kind: str) -> list[ScanRow]:
    lo = inp.previous_snapshot_at
    return [s for s in inp.scans if s.kind == kind and (lo is None or s.fetched_at > lo)]


def _window_flows(inp: Inputs) -> list[FlowRow]:
    lo = inp.previous_snapshot_at
    return [f for f in inp.flows if lo is None or f.fetched_at > lo]


NON_WALLET_TYPES = frozenset({"PROGRAM", "POOL_OR_VAULT", "ROUTER_OR_INTERMEDIARY"})


class Roles:
    """Which participants count as wallets at ``as_of`` (positive evidence only)."""

    def __init__(self, inp: Inputs):
        self.proven = frozenset(
            {f.wallet for f in inp.flows if f.participant == "NORMAL_WALLET"}
            | {e.wallet for e in inp.entries + inp.other_entries if e.proven_wallet}
        )

    def kind(self, wallet: str, participant: str) -> str:
        if participant == "UNKNOWN" and wallet in self.proven:
            return "NORMAL_WALLET"
        return participant

    def eligible(self, flows: Sequence[FlowRow]) -> list[FlowRow]:
        return [f for f in flows if self.kind(f.wallet, f.participant) == "NORMAL_WALLET"]

    def unknown(self, flows: Sequence[FlowRow]) -> set[str]:
        return {f.wallet for f in flows if self.kind(f.wallet, f.participant) == "UNKNOWN"}

    def summary(self, flows: Sequence[FlowRow]) -> dict[str, int]:
        kinds: dict[str, set[str]] = defaultdict(set)
        for f in flows:
            kinds[self.kind(f.wallet, f.participant)].add(f.wallet)
        return {k: len(v) for k, v in sorted(kinds.items())}


def _with_unknown(status: Status, why: str | None, unknown: int) -> tuple[Status, str | None]:
    """Unclassified participants make wallet counts lower bounds."""
    if not unknown or status not in VALUED:
        return status, why
    note = (
        f"{unknown} participants couldn't be proven to be wallets and are excluded "
        "(wallet counts are lower bounds)"
    )
    return "PARTIAL", "; ".join(x for x in (why, note) if x)


def _holders(
    inp: Inputs, settings: RadarSettings, run: Mapping[str, Any], roles: Roles
) -> tuple[dict[str, Any], dict[str, Any]]:
    step = run.get("holders")
    if step is not None and step.get("status") not in VALUED:
        status: Status = step["status"]
        why = "; ".join(step.get("reasons") or []) or "holder fetch failed"
        none = {k: _m(missing(status, why)) for k in (
            "holder_count", "top1_pct", "top10_pct", "top1_change_pp", "top10_change_pp",
            "holder_count_change")}  # fmt: skip
        large = {k: _m(missing(status, why)) for k in LARGE_METRICS}
        return {**none, "observed_at": None, "source": None, "baseline_observed_at": None}, {
            **large, "changes": []}  # fmt: skip
    if not inp.holders:
        why = "no holder snapshot collected yet"
        none = {k: _m(missing("NOT_COLLECTED", why)) for k in (
            "holder_count", "top1_pct", "top10_pct", "top1_change_pp", "top10_change_pp",
            "holder_count_change")}  # fmt: skip
        large = {k: _m(missing("NOT_COLLECTED", why)) for k in LARGE_METRICS}
        return {**none, "observed_at": None, "source": None, "baseline_observed_at": None}, {
            **large, "changes": []}  # fmt: skip
    cur = inp.holders[0]
    prev = inp.holders[1] if len(inp.holders) > 1 else None
    notes = "; ".join(cur.reasons) or None

    if cur.holder_count is None:
        count = missing("UNAVAILABLE", notes or "no holder listing (largest accounts only)")
    elif cur.holder_count_complete:
        count = available(cur.holder_count)
    else:
        count = partial(cur.holder_count, notes or "holder scan incomplete")

    def share(value: float | None) -> Metric:
        if value is None:
            return missing("UNAVAILABLE", notes or "concentration can't be attributed")
        if cur.source == "full_scan" and cur.reliable:
            return available(_round(value))
        return partial(_round(value), notes or "incomplete holder data")

    complete = prev is not None and all(h.source == "full_scan" and h.reliable for h in (cur, prev))

    def change(a: float | int | None, b: float | int | None) -> Metric:
        if prev is None:
            return missing("NOT_COLLECTED", "no earlier holder snapshot to compare with")
        if not complete or a is None or b is None:
            return missing(
                "UNAVAILABLE",
                "a change needs two complete holder scans (differences of lower bounds "
                "aren't bounds)",
            )
        return available(_round(a - b))

    count_change = (
        change(cur.holder_count, prev.holder_count)
        if prev is not None and cur.holder_count_complete and prev.holder_count_complete
        else change(None, None)
    )
    holders = {
        "observed_at": iso(cur.observed_at),
        "source": cur.source,
        "baseline_observed_at": iso(prev.observed_at) if prev else None,
        "holder_count": _m(count),
        "top1_pct": _m(share(cur.top1_pct)),
        "top10_pct": _m(share(cur.top10_pct)),
        "top1_change_pp": _m(change(cur.top1_pct, prev.top1_pct if prev else None)),
        "top10_change_pp": _m(change(cur.top10_pct, prev.top10_pct if prev else None)),
        "holder_count_change": _m(count_change),
    }
    return holders, _large(cur, prev, settings, roles)


# Large-holder metrics (token holders that aren't known programs / pools / routers:
# NORMAL_WALLET + UNKNOWN) and large-WALLET metrics (proven NORMAL_WALLET as of the snapshot
# time only). UNKNOWN never enters a metric named "wallet".
HOLDER_METRICS = ("large_holder_accumulation_count", "large_holder_reduction_count")
WALLET_METRICS = (
    "large_wallet_accumulation_count", "large_wallet_reduction_count", "large_wallet_exit_count",
)  # fmt: skip
LARGE_METRICS = HOLDER_METRICS + WALLET_METRICS


def _large(
    cur: HolderRow, prev: HolderRow | None, settings: RadarSettings, roles: Roles
) -> dict[str, Any]:
    if prev is None:
        why = "no earlier holder snapshot to compare with"
        return {k: _m(missing("NOT_COLLECTED", why)) for k in LARGE_METRICS} | {"changes": []}
    threshold, step = settings.large_holder_min_pct, settings.large_change_min_pp
    owners = sorted(
        {o for o, b in prev.balances.items() if b[2] == "LARGE"}
        | {o for o, b in cur.balances.items() if b[2] == "LARGE"}
    )
    changes: list[dict[str, Any]] = []
    unmeasured = 0
    types: dict[str, str] = {}
    for owner in owners:
        before = prev.balances.get(owner)
        after = cur.balances.get(owner)
        # Positive evidence from either snapshot (both fetched at or before as_of) wins over
        # UNKNOWN: an owner that left the scan can't be looked up again.
        seen = [b[3] for b in (after, before) if b is not None]
        stored = next((t for t in seen if t != "UNKNOWN"), "UNKNOWN")
        # As of this snapshot: proven wallets (signed a transaction Radar read) are wallets.
        types[owner] = roles.kind(owner, stored)
        after_pct = after[1] if after else None
        if after_pct is None:
            unmeasured += 1
            continue
        was_large = before is not None and before[2] == "LARGE"
        if before is not None and before[1] is not None:
            delta, bound = after_pct - before[1], False
        else:  # wasn't large before: it held below the threshold, so this is a minimum
            delta, bound = after_pct - threshold, True
        if abs(delta) >= step:
            changes.append({
                "owner": owner,
                "owner_type": types[owner],
                "before_pct": _round(before[1]) if before and before[1] is not None else None,
                "after_pct": _round(after_pct),
                "change_pp": _round(delta),
                "change_is_lower_bound": bound,
                "exit": was_large and after[1] == 0 if after else False,
                "kind": "LARGE_HOLDER_ACCUMULATION" if delta > 0 else "LARGE_HOLDER_REDUCTION",
            })  # fmt: skip
    holders = [c for c in changes if c["owner_type"] not in NON_WALLET_TYPES]
    wallets = [c for c in holders if c["owner_type"] == "NORMAL_WALLET"]
    unknown = sorted(o for o, t in types.items() if t == "UNKNOWN")
    known_non_wallets = sum(t in NON_WALLET_TYPES for t in types.values())
    full = cur.source == "full_scan" and prev.source == "full_scan" and not unmeasured
    scan_why = None if full else (
        (f"{unmeasured} prior large owners couldn't be re-measured; " if unmeasured else "")
        + "per-owner balances from incomplete scans are lower bounds"
    )  # fmt: skip

    def count(rows: list[dict[str, Any]], pick: str, wallet: bool) -> Metric:
        n = sum(
            c["exit"] if pick == "exit" else (c["change_pp"] > 0) == (pick == "acc") for c in rows
        )
        if pick == "exit" and not full:
            return missing("UNAVAILABLE", "an exit (balance zero) needs two complete holder scans")
        notes = [x for x in (scan_why,) if x]
        if wallet and unknown:
            notes.append(
                f"{len(unknown)} large owners aren't proven wallets and are excluded "
                "(wallet counts are lower bounds)"
            )
        return partial(n, "; ".join(notes)) if notes else available(n)

    changes.sort(key=lambda c: (-abs(c["change_pp"]), c["owner"]))
    return {
        "large_holder_accumulation_count": _m(count(holders, "acc", False)),
        "large_holder_reduction_count": _m(count(holders, "red", False)),
        "large_wallet_accumulation_count": _m(count(wallets, "acc", True)),
        "large_wallet_reduction_count": _m(count(wallets, "red", True)),
        "large_wallet_exit_count": _m(count(wallets, "exit", True)),
        "coverage": {
            "holder_metrics_include": ["NORMAL_WALLET", "UNKNOWN"],
            "wallet_metrics_include": ["NORMAL_WALLET"],
            "excluded_known_non_wallet_owners": known_non_wallets,
            "unknown_owners": len(unknown),
            "owners_by_type": {
                t: sum(v == t for v in types.values()) for t in sorted(set(types.values()))
            },
            "note": "NORMAL_WALLET = signed a transaction Radar read at or before this snapshot",
        },
        "changes": changes[:10],
    }


def _activity_status(scans: Sequence[ScanRow], run: Mapping[str, Any]) -> tuple[Status, str | None]:
    step = run.get("activity")
    if not scans:
        if step is not None and step.get("status") not in VALUED:
            return step["status"], "; ".join(step.get("reasons") or []) or None
        return "NOT_COLLECTED", "no activity scan in this snapshot's window"
    reasons = [r for s in scans for r in s.reasons]
    if any(s.status != "AVAILABLE" for s in scans):
        return "PARTIAL", "; ".join(reasons) or "an activity scan was incomplete"
    return "AVAILABLE", None


def _count(status: Status, value: int, why: str | None) -> Metric:
    if status == "AVAILABLE":
        return available(value)
    if status == "PARTIAL":
        return partial(value, why or "bounded scan")
    return missing(status, why or "not collected")


def _counterparty(flows: Sequence[FlowRow], status: Status, why: str | None, n: int) -> Metric:
    unsupported = sum(f.counterparty == "NOT_SUPPORTED" for f in flows)
    if flows and unsupported == len(flows):
        return missing(
            "NOT_SUPPORTED",
            "the tracked pool's vaults sit under a shared vault authority (Raydium AMM v4): "
            "its side of a transaction can't be attributed",
        )
    if unsupported and status in VALUED:
        return partial(n, "; ".join(x for x in (why, f"{unsupported} flows couldn't be "
                                                "attributed to the tracked pool") if x))  # fmt: skip
    return _count(status, n, why)


def _activity(inp: Inputs, run: Mapping[str, Any], roles: Roles) -> dict[str, Any]:
    scans = _scan_window(inp, "activity")
    status, why = _activity_status(scans, run)
    window = _window_flows(inp)
    status, why = _with_unknown(status, why, len(roles.unknown(window)))
    flows = roles.eligible(window)
    inflow = {f.wallet for f in flows if f.direction == "TOKEN_INFLOW"}
    outflow = {f.wallet for f in flows if f.direction == "TOKEN_OUTFLOW"}
    matched_in = {f.wallet for f in flows if f.direction == "TOKEN_INFLOW"
                  and f.counterparty == "TRACKED_POOL_COUNTERPARTY"}  # fmt: skip
    matched_out = {f.wallet for f in flows if f.direction == "TOKEN_OUTFLOW"
                   and f.counterparty == "TRACKED_POOL_COUNTERPARTY"}  # fmt: skip
    times = [f.block_time for f in window if f.block_time is not None]
    net = (
        available(len(inflow) - len(outflow))
        if status == "AVAILABLE"
        else missing(
            "UNAVAILABLE" if status == "PARTIAL" else status,
            "a net of two lower bounds isn't a bound" if status == "PARTIAL" else why or "",
        )
    )
    return {
        "window_since": iso(inp.previous_snapshot_at),
        "first_block_time": iso(min(times)) if times else None,
        "last_block_time": iso(max(times)) if times else None,
        "signatures_listed": sum(s.signatures_listed for s in scans),
        "transactions_parsed": sum(s.txs_parsed for s in scans),
        "transactions_skipped": sum(s.txs_skipped for s in scans),
        "participants": roles.summary(window),
        "interacting_wallets": _m(_count(status, len(inflow | outflow), why)),
        "token_inflow_wallets": _m(_count(status, len(inflow), why)),
        "token_outflow_wallets": _m(_count(status, len(outflow), why)),
        "net_inflow_wallets": _m(net),
        "tracked_pool_counterparty_inflow_wallets": _m(
            _counterparty(flows, status, why, len(matched_in))
        ),
        "tracked_pool_counterparty_outflow_wallets": _m(
            _counterparty(flows, status, why, len(matched_out))
        ),
    }


def _since_tracking(inp: Inputs) -> tuple[Status, str]:
    if not inp.scans:
        return "NOT_COLLECTED", "no activity collected yet"
    return "PARTIAL", "bounded scans: activity before tracking or beyond caps is missing"


def _early_cutoff(target: Target, inp: Inputs, settings: RadarSettings) -> float | None:
    start = target.pool_created_at
    if start is None:
        cand = inp.creators.get("POOL_CREATOR_CANDIDATE")
        start = cand.block_time if cand and cand.identity else None
    return start + settings.early_window_minutes * 60 if start is not None else None


def _early(target: Target, inp: Inputs, settings: RadarSettings, roles: Roles) -> dict[str, Any]:
    cutoff = _early_cutoff(target, inp, settings)
    keys = ("early_wallet_count", "early_inflow_wallets", "early_outflow_wallets",
            "repeated_early_wallet_count")  # fmt: skip
    why: str | None
    if cutoff is None:
        why = "the pool's creation time is unknown"
        return {"window_end": None, **{k: _m(missing("UNAVAILABLE", why)) for k in keys}}
    in_window = [f for f in inp.flows if f.block_time is not None and f.block_time <= cutoff]
    early = roles.eligible(in_window)
    early_scans = [s for s in inp.scans if s.kind == "early"]
    covered = [s for s in inp.scans if s.reached_oldest]
    status: Status
    if covered and all(s.txs_skipped == 0 and s.status == "AVAILABLE" for s in covered):
        status, why = "AVAILABLE", None
    elif in_window:
        status, why = "PARTIAL", "the early window was only partly scanned (history caps)"
    else:
        last = early_scans[-1] if early_scans else None
        status = "UNAVAILABLE" if last is None else as_status(last.status, "UNAVAILABLE")
        why = (
            "; ".join(last.reasons) if last and last.reasons else "the early window wasn't observed"
        )
        return {"window_end": iso(cutoff), **{k: _m(missing(status, why)) for k in keys}}
    status, why = _with_unknown(status, why, len(roles.unknown(in_window)))
    wallets = {f.wallet for f in early}
    others_early = {e.wallet for e in inp.other_entries if e.early and e.wallet in roles.proven}
    repeated = wallets & others_early
    bias = "only Radar targets are compared, and their early coverage may be partial"
    return {
        "window_end": iso(cutoff),
        "early_wallet_count": _m(_count(status, len(wallets), why)),
        "early_inflow_wallets": _m(
            _count(status, len({f.wallet for f in early if f.direction == "TOKEN_INFLOW"}), why)
        ),  # fmt: skip
        "early_outflow_wallets": _m(
            _count(status, len({f.wallet for f in early if f.direction == "TOKEN_OUTFLOW"}), why)
        ),  # fmt: skip
        "repeated_early_wallet_count": _m(partial(len(repeated), bias)),
    }


def _repeated(inp: Inputs, settings: RadarSettings, roles: Roles) -> dict[str, Any]:
    mine = {e.wallet for e in inp.entries if e.wallet in roles.proven}
    if not mine:
        why = "no proven wallet activity collected for this token yet"
        return {
            "repeated_wallet_count": _m(missing("NOT_COLLECTED", why)),
            "repeated_wallet_group_count": _m(missing("NOT_COLLECTED", why)),
            "groups": [],
        }
    per_token: dict[str, set[str]] = defaultdict(set)
    for e in inp.other_entries:
        if e.wallet in roles.proven:
            per_token[e.canonical_id].add(e.wallet)
    repeated = set().union(*per_token.values()) & mine if per_token else set()
    shared = sorted(
        ((cid, len(ws & mine)) for cid, ws in per_token.items()),
        key=lambda x: (-x[1], x[0]),
    )
    groups = [
        {"label": "REPEATED_WALLET_GROUP", "other_token": cid, "shared_wallets": n}
        for cid, n in shared
        if n >= settings.repeated_group_min_wallets
    ]
    why = "only Radar targets and proven wallets are compared (selection bias); bounded scans"
    return {
        "repeated_wallet_count": _m(partial(len(repeated), why)),
        "repeated_wallet_group_count": _m(partial(len(groups), why)),
        "groups": groups[:5],
    }


def timing_clusters(
    flows: Sequence[FlowRow], window_seconds: float, min_wallets: int
) -> list[tuple[float, int]]:
    """COORDINATED_TIMING_PATTERN: fixed windows starting at each unclaimed TOKEN_INFLOW,
    kept when at least `min_wallets` distinct wallets fall in [start, start + window].
    Returns (start block time, distinct wallets), oldest first. Deterministic."""
    events = sorted(
        (f.block_time, f.wallet) for f in flows
        if f.direction == "TOKEN_INFLOW" and f.block_time is not None
    )  # fmt: skip
    out: list[tuple[float, int]] = []
    i = 0
    while i < len(events):
        start = events[i][0]
        j = i
        while j < len(events) and events[j][0] - start <= window_seconds:
            j += 1
        wallets = {w for _, w in events[i:j]}
        if len(wallets) >= min_wallets:
            out.append((start, len(wallets)))
            i = j
        else:
            i += 1
    return out


def funding_clusters(
    wallets: set[str], profiles: Mapping[str, Any], min_wallets: int
) -> list[tuple[str, int]]:
    by_funder: dict[str, set[str]] = defaultdict(set)
    for w in wallets:
        p = profiles.get(w)
        if p is not None and p.funder and p.funder_status == "VERIFIED_FIRST_TX_FUNDER":
            by_funder[p.funder].add(w)
    return sorted(
        ((f, len(ws)) for f, ws in by_funder.items() if len(ws) >= min_wallets),
        key=lambda x: (-x[1], x[0]),
    )


def _clusters(inp: Inputs, settings: RadarSettings, roles: Roles) -> dict[str, Any]:
    status, why = _since_tracking(inp)
    if status == "NOT_COLLECTED":
        timing: dict[str, Any] = {"count": _m(missing(status, why)), "largest_wallets": None}
    else:
        found = timing_clusters(
            roles.eligible(inp.flows), settings.timing_window_seconds, settings.timing_min_wallets
        )
        timing = {
            "label": "COORDINATED_TIMING_PATTERN",
            "count": _m(partial(len(found), why)),
            "largest_wallets": max((n for _, n in found), default=None),
            "window_seconds": settings.timing_window_seconds,
            "min_wallets": settings.timing_min_wallets,
        }
    if not settings.funding_clusters_enabled:
        funding: dict[str, Any] = {
            "count": _m(missing("NOT_COLLECTED", "first-funder discovery is off "
                                "(UPSCALE_RADAR_FIRST_FUNDER=0)")),
            "clusters": [],
        }  # fmt: skip
    else:
        mine = {e.wallet for e in inp.entries if e.wallet in roles.proven}
        found_f = funding_clusters(mine, inp.profiles, settings.funding_cluster_min_wallets)
        funding = {
            "label": "FUNDING_CLUSTER",
            "count": _m(partial(len(found_f), "only profiled wallets with a verified first funder; "
                                "exchange hot wallets create false clusters")),
            "clusters": [{"funder": f, "wallets": n} for f, n in found_f[:5]],
        }  # fmt: skip
    return {"coordinated_timing": timing, "funding": funding}


def _creators(inp: Inputs, settings: RadarSettings, holders: list[HolderRow]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role, off in (("POOL_CREATOR_CANDIDATE", False),
                      ("TOKEN_DEPLOYER", not settings.verify_deployer)):  # fmt: skip
        row = inp.creators.get(role)
        key = role.lower()
        if row is None:
            reason = "deployer verification is off" if off else "not determined yet"
            out[key] = {"status": "NOT_COLLECTED", "identity": None, "reason": reason}
            continue
        entry: dict[str, Any] = {
            "status": row.status,
            "identity": row.identity,
            "method": row.method,
            "signature": row.signature,
            "block_time": iso(row.block_time),
            "determined_at": iso(row.determined_at),
            "provider": row.provider,
            "provenance": row.provenance,
        }
        if row.identity:
            entry["token_flows"] = _creator_flows(row.identity, inp, holders)
        out[key] = entry
    cand = out["pool_creator_candidate"].get("identity")
    dep = out["token_deployer"]
    out["candidate_matches_verified_deployer"] = (
        cand == dep["identity"] if cand and dep.get("status") == "VERIFIED" else None
    )
    unsupported = "V1 doesn't classify flows as sales or liquidity actions"
    out["sell_detected"] = _m(missing("NOT_SUPPORTED", unsupported))
    out["liquidity_action_detected"] = _m(missing("NOT_SUPPORTED", unsupported))
    return out


def _creator_flows(wallet: str, inp: Inputs, holders: list[HolderRow]) -> dict[str, Any]:
    status, why = _since_tracking(inp)
    flows = [f for f in inp.flows if f.wallet == wallet]
    out: dict[str, Any] = {
        "token_inflow_events": _m(
            _count(status, sum(f.direction == "TOKEN_INFLOW" for f in flows), why)
        ),  # fmt: skip
        "token_outflow_events": _m(
            _count(status, sum(f.direction == "TOKEN_OUTFLOW" for f in flows), why)
        ),  # fmt: skip
    }
    if len(holders) > 1 and wallet in holders[0].balances and wallet in holders[1].balances:
        a, b = holders[0].balances[wallet][1], holders[1].balances[wallet][1]
        full = all(h.source == "full_scan" for h in holders[:2])
        out["holder_balance_change_pp"] = _m(
            available(_round(a - b)) if a is not None and b is not None and full
            else missing("UNAVAILABLE", "needs the wallet in two complete holder scans")
        )  # fmt: skip
    else:
        out["holder_balance_change_pp"] = _m(
            missing("UNAVAILABLE", "the wallet isn't a tracked large holder in two snapshots")
        )
    return out


def _wallet_age(inp: Inputs, settings: RadarSettings, roles: Roles) -> dict[str, Any]:
    if not settings.wallet_age:
        why = "wallet-age history scan is off (UPSCALE_RADAR_WALLET_AGE=0)"
        return {"profiled_wallets": _m(missing("NOT_COLLECTED", why)),
                "new_wallet_count": _m(missing("NOT_COLLECTED", why))}  # fmt: skip
    mine = {e.wallet for e in inp.entries if e.wallet in roles.proven}
    prof = [inp.profiles[w] for w in sorted(mine) if w in inp.profiles]
    limit = inp.as_of - NEW_WALLET_MAX_AGE_HOURS * 3600
    new = sum(p.age_status == "EXACT" and p.oldest_block_time is not None
              and p.oldest_block_time >= limit for p in prof)  # fmt: skip
    why = "profiled wallets only; ages beyond the history cap are lower bounds"
    return {
        "profiled_wallets": _m(partial(len(prof), why)),
        "new_wallet_count": _m(partial(new, why)),
        "new_wallet_max_age_hours": NEW_WALLET_MAX_AGE_HOURS,
    }


def _wallet_history(inp: Inputs, settings: RadarSettings, roles: Roles) -> dict[str, Any]:
    mine = {e.wallet for e in inp.entries if e.wallet in roles.proven}
    if not mine:
        why = "no proven wallet activity collected for this token yet"
        return {"wallets_with_history": _m(missing("NOT_COLLECTED", why)),
                "wallets_meeting_minimum_unique_tokens": _m(missing("NOT_COLLECTED", why))}  # fmt: skip
    tokens: dict[str, set[str]] = defaultdict(set)
    for e in inp.entries + inp.other_entries:
        if e.wallet in mine:
            tokens[e.wallet].add(e.canonical_id)
    with_history = sum(len(tokens[w]) > 1 for w in mine)
    meeting = sum(len(tokens[w]) >= settings.history_min_unique_tokens for w in mine)
    why = "Radar targets only; descriptive (no wallet-quality score exists in V1)"
    return {
        "wallets_with_history": _m(partial(with_history, why)),
        "wallets_meeting_minimum_unique_tokens": _m(partial(meeting, why)),
        "minimum_unique_tokens": settings.history_min_unique_tokens,
        "note": "outcome-based minimums (observations) apply to the `wallet` command only",
    }


def overall_coverage(body: Mapping[str, Any]) -> str:
    """COMPLETE (every headline metric AVAILABLE), PARTIAL, or EMPTY (none valued)."""
    heads = [
        body["holders"]["holder_count"], body["holders"]["top10_pct"],
        body["activity"]["interacting_wallets"],
    ]  # fmt: skip
    statuses = [h["status"] for h in heads]
    if all(s == "AVAILABLE" for s in statuses):
        return "COMPLETE"
    return "PARTIAL" if any(s in VALUED for s in statuses) else "EMPTY"


def build_snapshot(
    target: Target,
    inp: Inputs,
    settings: RadarSettings,
    provider: str | None,
    run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``radar.snapshot.v1`` body describing `target` at ``inp.as_of``."""
    as_of = inp.as_of
    stamps = (
        [f.fetched_at for f in inp.flows]
        + [h.observed_at for h in inp.holders]
        + [s.fetched_at for s in inp.scans]
        + [c.determined_at for c in inp.creators.values()]
    )
    if any(t > as_of for t in stamps) or target.selected_at > as_of:
        raise RadarCausalityError("snapshot inputs must be fetched at or before observed_at")
    run = run or {}
    roles = Roles(inp)
    holders, large = _holders(inp, settings, run, roles)
    tracked, why = _with_unknown(*_since_tracking(inp), len(roles.unknown(inp.flows)))
    by_type = roles.summary(inp.flows)
    body: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA,
        "identity": {
            "chain": target.chain,
            "canonical_id": target.canonical_id,
            "mint": target.mint,
            "pool_address": target.pool_address,
            "dex": target.dex,
            "pool_created_at": iso(target.pool_created_at),
            "pool_created_source": target.pool_created_source,
            "target_source": target.source,
        },
        "observed_at": iso(as_of),
        "coverage": {
            "provider": provider,
            "run": {k: run[k] for k in sorted(run)},
            "holder_source": holders["source"],
            "activity_scans": len(_scan_window(inp, "activity")),
            # From scans known at as_of, never the target's current (mutable) state.
            "early_status": next(
                (x.status for x in reversed(inp.scans) if x.kind == "early"), None
            ),
            "participant_classification": {
                "status": "INCOMPLETE" if by_type.get("UNKNOWN") else "COMPLETE",
                "participants_by_type": by_type,
                "note": "only NORMAL_WALLET participants feed wallet statistics",
            },
        },
        "holders": holders,
        "large_holders": large,
        "activity": _activity(inp, run, roles),
        "since_tracking": {
            "interacting_wallets": _m(
                _count(tracked, len({f.wallet for f in roles.eligible(inp.flows)}), why)
            ),
        },
        "early_activity": _early(target, inp, settings, roles),
        "repeated": _repeated(inp, settings, roles),
        "clusters": _clusters(inp, settings, roles),
        "creators": _creators(inp, settings, inp.holders),
        "wallet_age": _wallet_age(inp, settings, roles),
        "wallet_history": _wallet_history(inp, settings, roles),
        "limitations": list(LIMITATIONS),
    }
    body["coverage"]["overall"] = overall_coverage(body)
    return body
