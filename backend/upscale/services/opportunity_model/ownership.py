"""The fact-ownership map: every aspect of the evidence has exactly one scoring owner.

A layer's record often repeats evidence another layer owns (Scout's archived candidate
carries legacy on-chain safety, social momentum and its own technical reading). Those
copies are routed here to the owning aspect and kept as DIAGNOSTIC facts, so a later
decision can never count one fact twice:

* IDENTITY, AUTHORITY, HOLDERS, MARKET_STRUCTURE and liquidity level / risk -> Safety V2
  (captured Radar facts surfaced through Safety stay Safety's; Radar is never read);
* liquidity trend (growth, stability), FLOW and earliness -> Scout (per sub-signal; never
  its composite score or family scores);
* PRICE structure -> Technical (Scout's TechnicalContext, archived Analyze technical);
* ATTENTION -> Social; NEWS -> News (context only).

An aspect owned by no layer in V1 (``None``) can only ever be a diagnostic.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from upscale.services.opportunity_model.models import (
    CorrelationGroup,
    Fact,
    FactRole,
    Layer,
    OwnershipError,
)


@dataclass(frozen=True)
class Owner:
    group: CorrelationGroup
    layer: Layer | None  # None: no V1 scoring owner (diagnostic only)
    role: FactRole  # the role of the owner's own fact: SCORING or CONTEXT


def _o(group: CorrelationGroup, layer: Layer | None, role: FactRole = "SCORING") -> Owner:
    return Owner(group, layer, role if layer is not None else "DIAGNOSTIC")


OWNERSHIP: Mapping[str, Owner] = MappingProxyType(
    {
        # IDENTITY: Safety V2
        "identity.token_mint": _o("IDENTITY", "safety", "CONTEXT"),
        "identity.mint_account": _o("IDENTITY", "safety"),
        "identity.token_program": _o("IDENTITY", "safety"),
        "identity.mint_data": _o("IDENTITY", "safety"),
        "identity.mismatch": _o("IDENTITY", "safety"),
        "safety.coverage": _o("IDENTITY", "safety", "CONTEXT"),
        # AUTHORITY: Safety V2
        "authority.mint": _o("AUTHORITY", "safety"),
        "authority.freeze": _o("AUTHORITY", "safety"),
        "authority.token_2022_extensions": _o("AUTHORITY", "safety"),
        # HOLDERS: Safety V2 (including captured creator / deployer evidence)
        "holders.top1": _o("HOLDERS", "safety"),
        "holders.top10": _o("HOLDERS", "safety"),
        "holders.few": _o("HOLDERS", "safety"),
        "holders.unknown_large_owner": _o("HOLDERS", "safety"),
        "holders.unclassified_program_owner": _o("HOLDERS", "safety"),
        "holders.deployer_supply": _o("HOLDERS", "safety"),
        "holders.concentration_rising": _o("HOLDERS", "safety"),
        "holders.rapid_loss": _o("HOLDERS", "safety"),
        "holders.large_exit": _o("HOLDERS", "safety"),
        # LIQUIDITY: level / risk -> Safety V2; trend -> Scout
        "liquidity.level": _o("LIQUIDITY", "safety"),
        "liquidity.collapse": _o("LIQUIDITY", "safety"),
        "liquidity.growth": _o("LIQUIDITY", "scout"),
        "liquidity.stability": _o("LIQUIDITY", "scout"),
        # MARKET_STRUCTURE: Safety V2
        "market.closed_on_chain": _o("MARKET_STRUCTURE", "safety"),
        "market.not_reported": _o("MARKET_STRUCTURE", "safety"),
        "market.no_eligible": _o("MARKET_STRUCTURE", "safety"),
        "market.primary_unclear": _o("MARKET_STRUCTURE", "safety"),
        "market.pool_age": _o("MARKET_STRUCTURE", "safety"),
        "market.fdv_gap": _o("MARKET_STRUCTURE", None),
        "market.cap_missing": _o("MARKET_STRUCTURE", None),
        # FLOW: Scout
        "flow.volume_acceleration": _o("FLOW", "scout"),
        "flow.trade_acceleration": _o("FLOW", "scout"),
        "flow.buy_pressure": _o("FLOW", "scout"),
        "flow.quality": _o("FLOW", "scout"),
        "flow.thin_pump": _o("FLOW", "scout"),
        "flow.distribution": _o("FLOW", "scout"),
        "flow.volume_to_liquidity": _o("FLOW", "scout"),
        "earliness.activity_regime": _o("FLOW", "scout"),
        "earliness.market_size": _o("FLOW", None),
        "scout.stage": _o("FLOW", "scout", "CONTEXT"),
        "scout.eligibility": _o("FLOW", "scout", "CONTEXT"),
        "scout.data_status": _o("FLOW", "scout", "CONTEXT"),
        "scout.market_collapse": _o("FLOW", "scout", "CONTEXT"),
        "scout.composite": _o("FLOW", None),
        "scout.cross_confirmation": _o("FLOW", None),
        "scout.unrouted": _o("FLOW", None),
        # PRICE: Technical (earliness' "move already made" is Scout's, in the PRICE group)
        "price.snapshot_trend": _o("PRICE", "technical"),
        "price.breakout": _o("PRICE", "technical"),
        "price.volume_confirmed": _o("PRICE", "technical"),
        "price.higher_lows": _o("PRICE", "technical"),
        "price.analyze_trend": _o("PRICE", "technical"),
        "price.h1_change": _o("PRICE", "technical"),
        "earliness.move_already_made": _o("PRICE", "scout"),
        # ATTENTION: Social
        "attention.momentum": _o("ATTENTION", "social"),
        "attention.attribution": _o("ATTENTION", "social"),
        "attention.spam": _o("ATTENTION", "social"),
        "attention.market_cross": _o("ATTENTION", "social"),
        "attention.cross_platform": _o("ATTENTION", "social"),
        "attention.providers": _o("ATTENTION", "social", "CONTEXT"),
        # NEWS: context only (labels may come from a language model)
        "news.sentiment": _o("NEWS", "news", "CONTEXT"),
        "news.impact": _o("NEWS", "news", "CONTEXT"),
    }
)

# Scout sub-signals, by (family, signal name) -> aspect. Only the ones whose aspect Scout
# owns become scoring facts; cross_confirmation is dropped entirely (see `normalize`).
SCOUT_SIGNALS: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        ("market_activity", "volume_acceleration"): "flow.volume_acceleration",
        ("market_activity", "trades_acceleration"): "flow.trade_acceleration",
        ("market_activity", "buy_pressure"): "flow.buy_pressure",
        ("market_activity", "price_momentum"): "price.h1_change",
        ("market_activity", "technical"): "price.snapshot_trend",
        ("liquidity_quality", "depth"): "liquidity.level",
        ("liquidity_quality", "growth"): "liquidity.growth",
        ("liquidity_quality", "stability"): "liquidity.stability",
        ("liquidity_quality", "liquidity_to_cap"): "liquidity.level",
        ("social_momentum", "state"): "attention.momentum",
        ("social_momentum", "unique_author_growth"): "attention.momentum",
        ("earliness", "move_already_made"): "earliness.move_already_made",
        ("earliness", "age"): "market.pool_age",
        ("earliness", "activity_regime"): "earliness.activity_regime",
        ("earliness", "market_size_context"): "earliness.market_size",
        ("earliness", "market_collapse"): "scout.market_collapse",
    }
)
SCOUT_RETAINED_SIGNALS: tuple[tuple[str, str], ...] = tuple(
    key
    for key, aspect in SCOUT_SIGNALS.items()
    if OWNERSHIP[aspect].layer == "scout" and OWNERSHIP[aspect].role == "SCORING"
)

# Scout risk flags, by code -> aspect. Legacy safety, social and Safety-owned market /
# liquidity flags are copies (diagnostic); Scout's own flow risks are scoring.
SCOUT_FLAGS: Mapping[str, str] = MappingProxyType(
    {
        "freeze_authority_active": "authority.freeze",
        "mint_authority_active": "authority.mint",
        "holder_concentration_top1": "holders.top1",
        "holder_concentration_top10": "holders.top10",
        "safety_data_missing": "safety.coverage",
        "safety_data_partial": "safety.coverage",
        "market_collapse": "scout.market_collapse",
        "flow_divergence": "flow.quality",
        "flow_in_doubt": "flow.quality",
        "thin_market_pump": "flow.thin_pump",
        "distribution": "flow.distribution",
        "high_volume_to_liquidity": "flow.volume_to_liquidity",
        "social_only_hype": "attention.market_cross",
        "social_without_market_evidence": "attention.market_cross",
        "social_spam": "attention.spam",
        "social_unavailable": "attention.momentum",
        # A draining pool is a weaker form of Safety's LIQUIDITY_COLLAPSE (and Scout's
        # liquidity stability already measures the trend): never independent evidence.
        "liquidity_draining": "liquidity.collapse",
        "unrecognized_quote": "liquidity.level",
        "primary_pool_unclear": "market.primary_unclear",
        "fdv_far_above_market_cap": "market.fdv_gap",
        "very_new": "market.pool_age",
        "pool_age_unknown": "market.pool_age",
        "market_cap_missing": "market.cap_missing",
        "stale_market_data": "scout.data_status",
    }
)
# Scout flags that are Scout's own scoring evidence (one fact per aspect).
SCOUT_SCORING_FLAGS: tuple[str, ...] = tuple(
    code
    for code, aspect in SCOUT_FLAGS.items()
    if OWNERSHIP[aspect].layer == "scout" and OWNERSHIP[aspect].role == "SCORING"
    and aspect != "flow.quality"  # flow quality is read from quality.flow_quality
)  # fmt: skip

# Safety V2 RULES_VERSION "4" rule ids -> aspect.
SAFETY_RULES: Mapping[str, str] = MappingProxyType(
    {
        "NOT_A_TOKEN_MINT": "identity.mint_account",
        "UNEXPECTED_TOKEN_PROGRAM": "identity.token_program",
        "MALFORMED_MINT_ACCOUNT": "identity.mint_data",
        "IDENTITY_MISMATCH": "identity.mismatch",
        "MINT_AUTHORITY_ACTIVE": "authority.mint",
        "FREEZE_AUTHORITY_ACTIVE": "authority.freeze",
        "TOKEN_2022_EXTENSION_RISK": "authority.token_2022_extensions",
        "TOP1_CONCENTRATION": "holders.top1",
        "TOP10_CONCENTRATION": "holders.top10",
        "FEW_HOLDERS": "holders.few",
        "LARGE_UNKNOWN_OWNER": "holders.unknown_large_owner",
        "LARGE_UNCLASSIFIED_PROGRAM_OWNER": "holders.unclassified_program_owner",
        "VERIFIED_DEPLOYER_HOLDS_SUPPLY": "holders.deployer_supply",
        "CONCENTRATION_RISING": "holders.concentration_rising",
        "RAPID_HOLDER_LOSS": "holders.rapid_loss",
        "LARGE_HOLDER_EXIT": "holders.large_exit",
        "LOW_LIQUIDITY": "liquidity.level",
        "LIQUIDITY_COLLAPSE": "liquidity.collapse",
        "MARKET_CLOSED_ON_CHAIN": "market.closed_on_chain",
        "MARKET_NOT_REPORTED": "market.not_reported",
        "NO_ELIGIBLE_MARKET": "market.no_eligible",
        "PRIMARY_MARKET_UNCLEAR": "market.primary_unclear",
        "VERY_NEW_POOL": "market.pool_age",
    }
)


def owner(aspect: str) -> Owner:
    try:
        return OWNERSHIP[aspect]
    except KeyError:
        raise OwnershipError(f"aspect {aspect!r} has no entry in the ownership map") from None


def check(facts: Iterable[Fact]) -> None:
    """Every fact agrees with the map, and each aspect has at most one non-diagnostic fact
    (from its owner, in the owner's role)."""
    owned: Counter[str] = Counter()
    for f in facts:
        o = owner(f.aspect)
        if f.group != o.group:
            raise OwnershipError(f"{f.aspect} belongs to {o.group}, not {f.group}")
        if f.scoring_owner != o.layer:
            raise OwnershipError(f"{f.aspect} is owned by {o.layer}, not {f.scoring_owner}")
        if f.role == "DIAGNOSTIC":
            continue
        if f.layer != o.layer or f.role != o.role:
            raise OwnershipError(
                f"{f.aspect}: a {f.role} fact from {f.layer}; only {o.layer} may hold it ({o.role})"
            )
        owned[f.aspect] += 1
    twice = sorted(a for a, n in owned.items() if n > 1)
    if twice:
        raise OwnershipError(f"aspects with more than one owned fact: {', '.join(twice)}")


def fact(
    aspect: str,
    layer: Layer,
    path: str,
    value: object = None,
    reason: str | None = None,
    diagnostic: bool = False,
) -> Fact:
    """A fact for `aspect` read from `layer`: the owner's role when `layer` owns it,
    otherwise (or when `diagnostic`) a DIAGNOSTIC copy. ``value=None`` with a reason is
    NOT_AVAILABLE."""
    o = owner(aspect)
    role: FactRole = "DIAGNOSTIC" if diagnostic or o.layer != layer else o.role
    return Fact.model_validate(
        {
            "aspect": aspect,
            "group": o.group,
            "layer": layer,
            "role": role,
            "status": "NOT_AVAILABLE" if value is None else "AVAILABLE",
            "value": value,
            "reason": reason,
            "path": path,
            "scoring_owner": o.layer,
        }
    )
