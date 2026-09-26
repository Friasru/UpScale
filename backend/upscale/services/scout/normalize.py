"""Provider pools → ScoutCandidates: exact identity, one market per token, risk flags.

Every provider hands over `DexPool`s (the same normalized pool model the DEX Market agent
uses). Pools are grouped by `<chain>:<base token address>`: a ticker is never used, so two
tokens called XYZ stay separate, and several pools of one token become one candidate whose
market is the pool chosen by UpScale's standard pool selection (`select_primary_pool`).
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from upscale.services.chains import (
    EVM_CHAINS,
    KNOWN_QUOTES,
    SOLANA,
    chain_label,
    is_valid_address,
    normalize_address,
)
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.models import (
    SCOUT_WINDOWS,
    DiscoveryKind,
    ScoutCandidate,
    ScoutMarketMetrics,
    ScoutPool,
    ScoutRejection,
    ScoutRiskFlags,
    ScoutSourceEvidence,
    ScoutWindow,
)
from upscale.services.solana_dex import DexPool, PoolSelectionConfig, select_primary_pool

# Chains whose token address format UpScale can check.
VERIFIABLE_CHAINS = EVM_CHAINS | {SOLANA}


@dataclass
class DiscoveryResult:
    """What one provider call produced: normalized candidates and what was left out."""

    candidates: list[ScoutCandidate] = field(default_factory=list)
    rejected: list[ScoutRejection] = field(default_factory=list)


@dataclass(frozen=True)
class Listing:
    """Where pools came from, for the candidates' source evidence."""

    provider: str
    kind: DiscoveryKind
    name: str
    fetched_at: datetime
    note: str | None = None


def canonical_id(chain: str, address: str) -> str:
    return f"{chain}:{normalize_address(chain, address)}"


def identity_problem(chain: str, address: str, config: ScoutConfig) -> str | None:
    """Why `chain` + `address` is not a reliable canonical identity, or None if it is."""
    if not chain or not address or address != address.strip() or " " in address:
        return "missing or malformed chain / token address"
    if chain in VERIFIABLE_CHAINS:
        if not is_valid_address(chain, address):
            return f"{address!r} is not a valid {chain_label(chain)} token address"
        return None
    if config.filters.require_verifiable_address:
        return f"UpScale can't verify token addresses on {chain_label(chain)} yet"
    return None


def build_candidates(
    pools: Sequence[DexPool],
    listing: Listing,
    config: ScoutConfig,
    positions: Mapping[str, int] | None = None,
    malformed_rows: int = 0,
    restrict_to: Collection[str] | None = None,
) -> DiscoveryResult:
    """Group pools by exact token and build one candidate per token.

    `positions` maps a canonical id to its position in the provider's listing (evidence
    only; it is never used to rank). `malformed_rows` counts provider rows that couldn't be
    parsed at all; they are reported as one rejection. With `restrict_to` (canonical ids),
    pools of any other base token are ignored (an exact lookup also returns pools where
    the requested token is only the quote side).
    """
    result = DiscoveryResult()
    if malformed_rows:
        result.rejected.append(
            ScoutRejection(
                canonical_id=None,
                chain=None,
                address=None,
                symbol=None,
                reasons=[f"{malformed_rows} malformed row(s) in {listing.provider} {listing.name}"],
                provider=listing.provider,
            )
        )
    groups: dict[str, list[DexPool]] = {}
    for pool in pools:
        chain, address = pool.chain, pool.base.address
        if restrict_to is not None and canonical_id(chain, address) not in restrict_to:
            continue
        problem = identity_problem(chain, address, config)
        if problem is None and normalize_address(chain, address) in _quotes(chain):
            problem = "the pool's base token is a quote asset, not a discoverable token"
        if problem is not None:
            result.rejected.append(_rejection(pool, [problem], listing.provider))
            continue
        groups.setdefault(canonical_id(chain, address), []).append(pool)

    selection_cfg = PoolSelectionConfig(
        min_liquidity_usd=config.filters.min_liquidity_usd,
        min_txns_24h=config.filters.min_txns_h24,
    )
    for cid, group in groups.items():
        chain = group[0].chain
        address = normalize_address(chain, group[0].base.address) or group[0].base.address
        normalized = [
            p.model_copy(update={"base": p.base.model_copy(update={"address": address})})
            for p in group
        ]
        selection = select_primary_pool(normalized, address, selection_cfg, chain)
        primary = selection.primary
        if primary is None:
            reasons = sorted({r for c in selection.candidates for r in c.rejected_because})
            result.rejected.append(
                _rejection(
                    group[0],
                    ["no pool with enough liquidity, trades and a USD price", *reasons],
                    listing.provider,
                )
            )
            continue
        source = ScoutSourceEvidence(
            provider=listing.provider,
            kind=listing.kind,
            listing=listing.name,
            fetched_at=listing.fetched_at,
            position=(positions or {}).get(cid),
            note=listing.note,
        )
        created = [p.pair_created_at for p in normalized if p.pair_created_at]
        recognized = [
            p.liquidity_usd
            for p in normalized
            if p.quote_kind != "other" and p.liquidity_usd is not None
        ]
        age = (
            max(0.0, (listing.fetched_at - primary.pair_created_at).total_seconds() / 3600)
            if primary.pair_created_at
            else None
        )
        metrics = metrics_from_pool(primary)
        market = ScoutPool(
            address=primary.pair_address,
            dex=primary.dex,
            url=primary.url,
            quote_address=primary.quote.address,
            quote_symbol=primary.quote.symbol,
            quote_kind=primary.quote_kind,
            created_at=primary.pair_created_at,
            age_hours=age,
        )
        result.candidates.append(
            ScoutCandidate(
                canonical_id=cid,
                chain=chain,
                address=address,
                symbol=primary.base.symbol,
                name=primary.base.name,
                observed_at=listing.fetched_at,
                oldest_pool_created_at=min(created) if created else None,
                sources=[source],
                market_provider=listing.provider,
                pool=market,
                metrics=metrics,
                pool_count=len(normalized),
                recognized_quote_liquidity_usd=sum(recognized) if recognized else None,
                primary_clear=selection.clear,
                ambiguity=selection.ambiguity,
                risk_flags=risk_flags(metrics, market, selection.clear, config),
            )
        )
    return result


def metrics_from_pool(pool: DexPool) -> ScoutMarketMetrics:
    windows = []
    for name in SCOUT_WINDOWS:
        w = pool.window(name)
        if w is None:
            continue
        windows.append(
            ScoutWindow(
                window=name,
                volume_usd=w.volume_usd,
                buys=w.buys,
                sells=w.sells,
                buyers=w.buyers,
                sellers=w.sellers,
                price_change_pct=w.price_change_pct,
            )
        )
    return ScoutMarketMetrics(
        price_usd=pool.price_usd,
        market_cap_usd=pool.market_cap_usd,
        fdv_usd=pool.fdv_usd,
        liquidity_usd=pool.liquidity_usd,
        windows=windows,
    )


def risk_flags(
    metrics: ScoutMarketMetrics, pool: ScoutPool, clear: bool, config: ScoutConfig
) -> ScoutRiskFlags:
    cfg = config.flags
    flags = ScoutRiskFlags(
        primary_pool_unclear=not clear,
        unrecognized_quote=pool.quote_kind == "other",
        market_cap_missing=metrics.market_cap_usd is None,
        fdv_only=metrics.market_cap_usd is None and metrics.fdv_usd is not None,
        pool_age_unknown=pool.age_hours is None,
        very_new_pool=pool.age_hours is not None and pool.age_hours < cfg.very_new_pool_hours,
    )
    if metrics.market_cap_usd and metrics.fdv_usd:
        ratio = metrics.fdv_usd / metrics.market_cap_usd
        if ratio >= cfg.fdv_to_market_cap_ratio:
            flags.fdv_far_above_market_cap = True
            flags.notes.append(f"FDV is {ratio:.1f}x the reported market cap")
    day = metrics.window("h24")
    if day and day.volume_usd is not None and metrics.liquidity_usd:
        turnover = day.volume_usd / metrics.liquidity_usd
        if turnover >= cfg.high_volume_to_liquidity_h24:
            flags.high_volume_to_liquidity = True
            flags.notes.append(f"24h volume is {turnover:.1f}x the pool's liquidity")
    if flags.unrecognized_quote:
        flags.notes.append("quoted in an unrecognized token: USD liquidity may be overstated")
    return flags


def merge_candidates(candidates: Sequence[ScoutCandidate]) -> list[ScoutCandidate]:
    """One candidate per canonical id. The market with the most USD liquidity in a
    recognized quote wins; every source is kept."""
    merged: dict[str, ScoutCandidate] = {}
    for cand in candidates:
        current = merged.get(cand.canonical_id)
        if current is None:
            merged[cand.canonical_id] = cand
            continue
        best, other = (
            (cand, current) if _market_key(cand) < _market_key(current) else (current, cand)
        )
        oldest = [t for t in (best.oldest_pool_created_at, other.oldest_pool_created_at) if t]
        merged[cand.canonical_id] = best.model_copy(
            update={
                "sources": _unique_sources([*current.sources, *cand.sources]),
                "oldest_pool_created_at": min(oldest) if oldest else None,
                "symbol": best.symbol or other.symbol,
                "name": best.name or other.name,
            }
        )
    return list(merged.values())


def _market_key(c: ScoutCandidate) -> tuple[int, float, float]:
    return (
        c.pool.quote_kind == "other",
        -(c.metrics.liquidity_usd or 0.0),
        -c.observed_at.timestamp(),  # fresher evidence breaks ties
    )


def _unique_sources(sources: Sequence[ScoutSourceEvidence]) -> list[ScoutSourceEvidence]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for s in sources:
        key = (s.provider, s.kind, s.listing)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _quotes(chain: str) -> set[str]:
    return {normalize_address(chain, a) or a for a in KNOWN_QUOTES.get(chain, {})}


def _rejection(pool: DexPool, reasons: list[str], provider: str) -> ScoutRejection:
    return ScoutRejection(
        canonical_id=(
            canonical_id(pool.chain, pool.base.address)
            if pool.chain and pool.base.address
            else None
        ),
        chain=pool.chain,
        address=pool.base.address,
        symbol=pool.base.symbol,
        reasons=reasons,
        provider=provider,
    )
