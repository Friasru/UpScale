"""Outcome integrity: deterministic checks that a numeric outcome measures the right price.

Never judged by size: a +5,000% move whose candle path starts at the reference price and
agrees with its own MFE / MAE is a valid extreme move. What makes a row invalid is a
provenance or consistency failure:

* the reference or horizon price isn't a finite positive number;
* a numeric return without any price point (an unavailable market used as a price);
* the stored path contradicts itself (return outside [MAE, MFE], extremes not matching
  the highest / lowest prices, end price outside the observed range);
* the candles priced another token of the pool (quote / base orientation: GeckoTerminal
  orients some pools the other way round from DEX Screener, e.g. "DOGE / GOAT"), a pool
  that doesn't contain the token, or a unit / decimal factor, as shown by provider
  evidence (the pool's orientation, an independent reconstruction).

Without provider evidence, a path that never comes within `DISCONTINUITY_FACTOR` of its
reference is UNKNOWN_INTEGRITY ("needs verification"), never silently invalid.

Pre-existing rows are never changed: classifications go to the append-only sidecar table
``outcome_integrity_audits`` (written only on request).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from upscale.services.market_data import Candle
from upscale.services.outcomes.models import PricePath

IntegrityStatus = Literal[
    "VALID",
    "VALID_EXTREME_MOVE",
    "INVALID_REFERENCE_PRICE",
    "INVALID_HORIZON_PRICE",
    "POOL_IDENTITY_MISMATCH",
    "QUOTE_BASE_INVERSION",
    "DECIMAL_OR_UNIT_ERROR",
    "INVALID_FALLBACK",
    "MARKET_UNAVAILABLE_AS_PRICE",
    "PROVIDER_DATA_ANOMALY",
    "UNKNOWN_INTEGRITY",
]
VALID_STATUSES: frozenset[str] = frozenset({"VALID", "VALID_EXTREME_MOVE"})
INVALID_STATUSES: frozenset[str] = frozenset(
    {"INVALID_REFERENCE_PRICE", "INVALID_HORIZON_PRICE", "POOL_IDENTITY_MISMATCH",
     "QUOTE_BASE_INVERSION", "DECIMAL_OR_UNIT_ERROR", "INVALID_FALLBACK",
     "MARKET_UNAVAILABLE_AS_PRICE", "PROVIDER_DATA_ANOMALY"}
)  # fmt: skip
EXTREME_RETURN_PCT = 1000.0
# A path none of whose prices comes within this factor of the reference can't be the same
# market moments later (a 1,000x jump before the first candle): only checked, never
# assumed, for stored rows; blocks the numeric outcome for new ones.
DISCONTINUITY_FACTOR = 1000.0
TOLERANCE_PP = 0.01  # percentage points, for recomputing MFE / MAE / return
TOKEN_ORIENTED = "candles priced in USD for token "  # the collector's note on new paths


def _positive(x: float | None) -> bool:
    return x is not None and math.isfinite(x) and x > 0


def _pct(new: float, old: float) -> float:
    return (new / old - 1) * 100


def path_problems(
    p: PricePath, return_pct: float | None = None
) -> list[tuple[IntegrityStatus, str]]:
    """Internal consistency of one stored price path (no provider needed)."""
    ret = p.return_pct if return_pct is None else return_pct
    out: list[tuple[IntegrityStatus, str]] = []
    if not _positive(p.reference_price):
        return [
            (
                "INVALID_REFERENCE_PRICE",
                f"reference price {p.reference_price!r} is not finite and positive",
            )
        ]
    if ret is not None and p.points == 0 and p.end_price is None:
        return [("MARKET_UNAVAILABLE_AS_PRICE", "a numeric return without any price point")]
    if ret is None:
        return out
    if not _positive(p.end_price):
        return [
            ("INVALID_HORIZON_PRICE", f"horizon price {p.end_price!r} is not finite and positive")
        ]
    assert p.end_price is not None
    if abs(_pct(p.end_price, p.reference_price) - ret) > TOLERANCE_PP:
        out.append(("PROVIDER_DATA_ANOMALY", "return doesn't match reference and horizon prices"))
    if p.source != "candles":
        return out  # snapshot paths: extremes are lower bounds, only the return is exact
    if not (_positive(p.highest_price) and _positive(p.lowest_price)):
        out.append(("INVALID_HORIZON_PRICE", "path extremes missing or not positive"))
        return out
    assert p.highest_price is not None and p.lowest_price is not None
    if not p.lowest_price - 1e-12 <= p.end_price <= p.highest_price + 1e-12:
        out.append(("PROVIDER_DATA_ANOMALY", "horizon price outside the path's own range"))
    if (
        p.mfe_pct is not None
        and abs(p.mfe_pct - max(0.0, _pct(p.highest_price, p.reference_price))) > TOLERANCE_PP
    ):
        out.append(("PROVIDER_DATA_ANOMALY", "MFE doesn't match the highest price"))
    if (
        p.mae_pct is not None
        and abs(p.mae_pct - min(0.0, _pct(p.lowest_price, p.reference_price))) > TOLERANCE_PP
    ):
        out.append(("PROVIDER_DATA_ANOMALY", "MAE doesn't match the lowest price"))
    if p.mfe_pct is not None and ret > p.mfe_pct + TOLERANCE_PP:
        out.append(("PROVIDER_DATA_ANOMALY", "return above MFE"))
    if p.mae_pct is not None and ret < p.mae_pct - TOLERANCE_PP:
        out.append(("PROVIDER_DATA_ANOMALY", "return below MAE"))
    if p.max_drawdown_pct is not None and p.max_drawdown_pct > TOLERANCE_PP:
        out.append(("PROVIDER_DATA_ANOMALY", "positive max drawdown"))
    return out


def discontinuity(p: PricePath) -> float | None:
    """How far (x) the whole path stays from the reference, when it never approaches it."""
    ref, lo, hi = p.reference_price, p.lowest_price, p.highest_price
    if not (_positive(ref) and _positive(lo) and _positive(hi)):
        return None
    assert lo is not None and hi is not None
    if lo > ref:
        return lo / ref
    if hi < ref:
        return ref / hi
    return 1.0


def candle_path_problem(candles: Sequence[Candle], reference: float | None) -> str | None:
    """Collector check before a numeric outcome is stored: the first in-window candle must
    open within DISCONTINUITY_FACTOR of the reference (a real move starts from it)."""
    if not _positive(reference):
        return "reference price is not finite and positive"
    if not candles:
        return None
    assert reference is not None
    first = candles[0]
    factor = max(first.open / reference, reference / first.open)
    if factor >= DISCONTINUITY_FACTOR:
        return (
            f"the first candle opens {factor:,.0f}x away from the reference price: not the same "
            "market (e.g. the pool's other token was priced); numeric outcome withheld"
        )
    return None


@dataclass
class Verification:
    """Provider evidence for one row (optional)."""

    pool_base: str | None = None
    pool_quote: str | None = None
    reconstructed: PricePath | None = None
    notes: list[str] = field(default_factory=list)


def token_oriented(p: PricePath) -> bool:
    return any(n.startswith(TOKEN_ORIENTED) for n in p.notes)


def classify(
    p: PricePath | None,
    return_pct: float | None,
    token: str | None,
    verification: Verification | None = None,
) -> tuple[IntegrityStatus, str, dict[str, Any]]:
    """(status, reason, evidence) for one numeric outcome."""
    evidence: dict[str, Any] = {}
    if p is None:
        return "MARKET_UNAVAILABLE_AS_PRICE", "numeric return without a stored price path", evidence
    problems = path_problems(p, return_pct)
    if problems:
        status, why = problems[0]
        evidence["problems"] = [w for _, w in problems]
        return status, why, evidence
    ret = return_pct if return_pct is not None else p.return_pct
    extreme = ret is not None and abs(ret) > EXTREME_RETURN_PCT
    gap = discontinuity(p) if p.source == "candles" else 1.0
    evidence["discontinuity_factor"] = gap
    v = verification
    if v is not None and p.source == "candles":
        evidence |= {"pool_base": v.pool_base, "pool_quote": v.pool_quote, "notes": v.notes}
        r = v.reconstructed
        ratio: float | None = None
        if r is not None and _positive(r.end_price) and p.end_price is not None:
            assert r.end_price is not None
            ratio = p.end_price / r.end_price
            evidence |= {
                "reconstructed_end_price": r.end_price,
                "reconstructed_return_pct": r.return_pct,
                "stored_vs_reconstructed_ratio": ratio,
                "return_difference_pp": (ret - r.return_pct)
                if ret is not None and r.return_pct is not None
                else None,
            }
        if token and v.pool_base and v.pool_quote and token not in (v.pool_base, v.pool_quote):
            return "POOL_IDENTITY_MISMATCH", "the pool doesn't contain this token", evidence
        if token and v.pool_base and v.pool_base != token and not token_oriented(p):
            return (
                "QUOTE_BASE_INVERSION",
                f"candles were requested for the pool's base token, which the provider "
                f"orients as {v.pool_base}, not this token: they priced the pool's other token",
                evidence,
            )
        if r is not None and ratio is not None:
            if abs(ratio - 1) <= 0.05 or (
                r.return_pct is not None and ret is not None and abs(r.return_pct - ret) <= 1
            ):
                return (
                    ("VALID_EXTREME_MOVE" if extreme else "VALID"),
                    "matches an independent reconstruction",
                    evidence,
                )
            k = math.log10(ratio)
            if any(abs(abs(k) - e) < 0.02 for e in (3, 6, 9, 12, 18)):
                return (
                    "DECIMAL_OR_UNIT_ERROR",
                    f"stored price is 10^{round(k)} x the reconstruction",
                    evidence,
                )
            assert p.end_price is not None and r.end_price is not None
            if abs(math.log10(p.end_price * r.end_price)) < 0.3:
                return (
                    "QUOTE_BASE_INVERSION",
                    "stored price is ~ the reciprocal of the reconstruction",
                    evidence,
                )
            return (
                "PROVIDER_DATA_ANOMALY",
                f"stored price differs {ratio:.4g}x from the reconstruction",
                evidence,
            )
    if gap is not None and gap >= DISCONTINUITY_FACTOR:
        return (
            "UNKNOWN_INTEGRITY",
            f"the path never comes within {gap:,.0f}x of its reference: needs provider verification",
            evidence,
        )
    return (
        ("VALID_EXTREME_MOVE" if extreme else "VALID"),
        "consistent path starting from the reference",
        evidence,
    )
