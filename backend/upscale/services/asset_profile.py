"""Crypto asset profiles: what kind of asset this is, and which evidence matters for it.

No network I/O and no model calls. A profile is built from:

* the asset's **identity**: a ticker, a centralized-exchange pair, or a chain + contract /
  mint address. Two tokens with the same ticker get different canonical ids and never
  share evidence; a ticker that isn't in the reviewed registry is flagged as ambiguous.
* its **characteristics**: reviewed registry facts (`upscale.services.asset_registry`) or
  facts supplied by a provider, plus live market data when the Market agent fetched it.
* the **data capabilities** UpScale actually has for it. Anything without an integrated
  provider is reported as unavailable, never assumed.

Categories (first matching rule wins; every threshold lives in `ProfileConfig`):

1. ``stablecoin``: tagged as a stablecoin or pegged to a currency.
2. ``new_dex_token``: traded on a DEX, no known centralized-exchange listing, and younger
   than `new_token_max_age_days` (or of unknown age).
3. ``established_memecoin``: tagged meme, at least `memecoin_min_age_days` old, and either
   a mid-or-larger market cap or a centralized-exchange listing. A younger or smaller
   memecoin is ``unknown_crypto`` (it isn't established, and it isn't DEX-only).
4. ``major_crypto``: mega market cap.
5. ``large_cap_alt``: large market cap, or a mid market cap with a centralized-exchange
   listing and at least `established_min_age_days` of history.
6. ``unknown_crypto``: anything else, including every ticker-only identity that isn't in
   the registry (its data may belong to another token with the same ticker).

Each category has a `CategoryProfile`: which evidence is required, optional or planned,
how much weight each carries, which evidence is decision-critical, and whether Risk's
thresholds are calibrated for it. Those tables, not scattered if-statements, drive the
adaptive behavior in Risk and Opportunity.
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from types import MappingProxyType
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from upscale.schemas import AgentName
from upscale.services.asset_registry import (
    DEFAULT_REGISTRY,
    AssetMetadata,
    AssetRegistry,
)
from upscale.services.chains import (
    DEX_CHAINS,
    EVM_CHAINS,
    GECKOTERMINAL_NETWORKS,
    chain_label,
    normalize_address,
    same_address,
)
from upscale.services.market_data import MarketSnapshot
from upscale.services.solana_chain import OnchainSafetySnapshot
from upscale.services.solana_dex import SolanaDexSnapshot
from upscale.services.technical_analysis import TechnicalAnalysis, TechnicalAnalysisConfig

Category = Literal[
    "major_crypto",
    "large_cap_alt",
    "established_memecoin",
    "new_dex_token",
    "stablecoin",
    "unknown_crypto",
]
MarketCapClass = Literal["mega", "large", "mid", "small", "micro"]
LiquidityClass = Literal["deep", "high", "medium", "low", "very_low"]
VolatilityClass = Literal["low", "moderate", "high", "extreme"]
MarketType = Literal["cex_spot", "dex_spot", "cex_and_dex", "unknown"]
IdentityBasis = Literal["registry_symbol", "cex_pair", "contract", "symbol_only"]
DataCapability = Literal[
    "candles", "market_snapshot", "news", "dex", "onchain", "social", "derivatives"
]
CapabilityState = Literal["available", "unavailable", "unknown"]
EvidenceType = Literal[
    "technical_structure",
    "market_snapshot",
    "news",
    "dex_liquidity",
    "buy_sell_flow",
    "pool_age",
    "holder_concentration",
    "token_authorities",
    "onchain_activity",
    "ecosystem",
    "social",
    "peg_stability",
    "issuer_risk",
    "derivatives",
]
# How much an evidence type should count for this kind of asset.
Weight = Literal["primary", "supporting", "context", "not_used"]
Requirement = Literal["required", "optional", "planned"]
# available: observed this turn. expected: a provider exists but hasn't reported yet.
EvidenceState = Literal[
    "available", "expected", "unavailable", "insufficient", "not_analyzed", "unknown"
]
StepStatus = Literal["run", "conditional", "unavailable", "skip"]
T = TypeVar("T")

CAPABILITIES: tuple[DataCapability, ...] = (
    "candles",
    "market_snapshot",
    "news",
    "dex",
    "onchain",
    "social",
    "derivatives",
)
# Providers UpScale has today. Anything else is reported as unavailable.
INTEGRATED_CAPABILITIES: frozenset[DataCapability] = frozenset(
    {"candles", "market_snapshot", "news", "dex"}
)
# Capabilities that exist only when configured (e.g. on-chain data needs a Solana RPC).
_ENABLED_OPTIONAL: set[DataCapability] = set()


def set_capability_enabled(capability: DataCapability, enabled: bool) -> None:
    """Called at startup by `upscale.services` once it knows which providers exist."""
    if enabled:
        _ENABLED_OPTIONAL.add(capability)
    else:
        _ENABLED_OPTIONAL.discard(capability)


def integrated_capabilities() -> frozenset[DataCapability]:
    return INTEGRATED_CAPABILITIES | frozenset(_ENABLED_OPTIONAL)


CAPABILITY_LABELS: dict[DataCapability, str] = {
    "candles": "OHLCV candles",
    "market_snapshot": "market snapshots",
    "news": "news from publisher RSS feeds",
    "dex": "DEX pool data",
    "onchain": "on-chain data",
    "social": "social data",
    "derivatives": "derivatives data",
}
EVIDENCE_LABELS: dict[EvidenceType, str] = {
    "technical_structure": "technical structure (trend, levels, momentum)",
    "market_snapshot": "live market snapshot (price, 24h change, volume)",
    "news": "recent news",
    "dex_liquidity": "DEX pool liquidity",
    "buy_sell_flow": "DEX buy/sell flow",
    "pool_age": "pool age",
    "holder_concentration": "holder concentration",
    "token_authorities": "token mint/freeze authorities",
    "onchain_activity": "on-chain activity",
    "ecosystem": "ecosystem activity",
    "social": "social activity",
    "peg_stability": "peg stability",
    "issuer_risk": "issuer / protocol risk",
    "derivatives": "derivatives (funding, open interest)",
}
CATEGORY_LABELS: dict[Category, str] = {
    "major_crypto": "major crypto asset",
    "large_cap_alt": "large-cap altcoin",
    "established_memecoin": "established memecoin",
    "new_dex_token": "new DEX token",
    "stablecoin": "stablecoin",
    "unknown_crypto": "crypto asset of unknown type",
}
# Where each evidence type comes from, and which agent analyzes it (None: no rule yet).
EVIDENCE_SOURCE: dict[EvidenceType, DataCapability | None] = {
    "technical_structure": "candles",
    "market_snapshot": "market_snapshot",
    "news": "news",
    "dex_liquidity": "dex",
    "buy_sell_flow": "dex",
    "pool_age": "dex",
    "holder_concentration": "onchain",
    "token_authorities": "onchain",
    "onchain_activity": "onchain",
    "ecosystem": "onchain",
    "social": "social",
    "peg_stability": "market_snapshot",
    "issuer_risk": None,
    "derivatives": "derivatives",
}
EVIDENCE_AGENT: dict[EvidenceType, AgentName] = {
    "technical_structure": "technical_analysis",
    "market_snapshot": "market",
    "news": "news_sentiment",
    "dex_liquidity": "dex_market",
    "buy_sell_flow": "dex_market",
    "pool_age": "dex_market",
    "token_authorities": "onchain_safety",
    "holder_concentration": "onchain_safety",
}


def _min_technical_candles(cfg: TechnicalAnalysisConfig) -> int:
    """Fewest candles for the trend rule and MACD to exist at all."""
    return max(cfg.trend_slow_sma, cfg.macd_slow + cfg.macd_signal)


MIN_TECHNICAL_CANDLES = _min_technical_candles(TechnicalAnalysisConfig())


# --- Configuration --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileConfig:
    """Classification bands. They describe the asset; they are not decision thresholds."""

    mega_cap_usd: float = 150e9
    large_cap_usd: float = 10e9
    mid_cap_usd: float = 1e9
    small_cap_usd: float = 100e6
    established_min_age_days: int = 365  # a mid-cap alt counts as established after this
    memecoin_min_age_days: int = 180
    new_token_max_age_days: int = 90
    # Primary-pool liquidity that counts as a substantial market for a DEX-only memecoin
    # (liquidity, not the provider's self-reported market cap, is the size evidence).
    established_dex_liquidity_usd: float = 1e6
    # 24h traded volume (centralized markets) and pool liquidity (DEX tokens), USD.
    volume_bands: tuple[float, float, float, float] = (1e9, 100e6, 10e6, 1e6)
    pool_liquidity_bands: tuple[float, float, float, float] = (50e6, 5e6, 1e6, 100e3)
    # 24h high-low range as % of the low (same measure Risk uses for its range factor).
    volatility_bands: tuple[float, float, float] = (3.0, 8.0, 15.0)


@dataclass(frozen=True)
class EvidenceRule:
    evidence: EvidenceType
    requirement: Requirement
    weight: Weight
    # Without it, Opportunity must not act (a WAIT blocker) and Risk's uncertainty is high.
    decision_critical: bool = False


@dataclass(frozen=True)
class CategoryProfile:
    category: Category
    description: str
    evidence: tuple[EvidenceRule, ...]
    risk_characteristics: tuple[str, ...]
    # True when Risk's current thresholds were designed for this kind of asset.
    risk_thresholds_calibrated: bool
    # RiskConfig field overrides for this category. Empty until each value is tested.
    risk_overrides: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    # Technical evidence only counts with at least this many candles (None: no minimum
    # beyond what Technical Analysis itself reports as unavailable).
    technical_min_candles: int | None = None


def _r(
    evidence: EvidenceType, req: Requirement, weight: Weight, critical: bool = False
) -> EvidenceRule:
    return EvidenceRule(evidence, req, weight, critical)


CATEGORY_PROFILES: Mapping[Category, CategoryProfile] = MappingProxyType(
    {
        "major_crypto": CategoryProfile(
            category="major_crypto",
            description="Largest, most liquid crypto assets; they set the tone for the market.",
            evidence=(
                _r("technical_structure", "required", "primary", critical=True),
                _r("market_snapshot", "required", "supporting"),
                _r("news", "required", "supporting"),
                _r("derivatives", "planned", "supporting"),
            ),
            risk_characteristics=(
                "Deep spot liquidity on major centralized exchanges.",
                "A 5-10% daily move is large for this asset; Risk's move thresholds were "
                "designed for it.",
                "Macro and regulatory news move it and the wider market.",
            ),
            risk_thresholds_calibrated=True,
        ),
        "large_cap_alt": CategoryProfile(
            category="large_cap_alt",
            description="Established, liquid altcoins with a long centralized-exchange history.",
            evidence=(
                _r("technical_structure", "required", "primary", critical=True),
                _r("market_snapshot", "required", "supporting"),
                _r("news", "required", "supporting"),
                _r("ecosystem", "optional", "supporting"),
                _r("onchain_activity", "optional", "context"),
                _r("derivatives", "planned", "supporting"),
            ),
            risk_characteristics=(
                "Liquid on centralized exchanges.",
                "Usually moves more than BTC and ETH in the same market.",
                "Ecosystem-specific news and network activity matter.",
            ),
            risk_thresholds_calibrated=True,
        ),
        "established_memecoin": CategoryProfile(
            category="established_memecoin",
            description="Memecoins with long trading history and substantial liquidity.",
            evidence=(
                _r("technical_structure", "required", "supporting", critical=True),
                _r("market_snapshot", "required", "supporting"),
                _r("dex_liquidity", "optional", "supporting"),
                _r("buy_sell_flow", "optional", "supporting"),
                _r("holder_concentration", "optional", "supporting"),
                _r("social", "optional", "supporting"),
                _r("news", "optional", "context"),
                _r("onchain_activity", "optional", "context"),
            ),
            risk_characteristics=(
                "Price is driven largely by attention and sentiment.",
                "Large daily swings are common; move thresholds designed for BTC can "
                "overstate how unusual a move is.",
                "Holder concentration and liquidity depth matter.",
            ),
            risk_thresholds_calibrated=False,
        ),
        "new_dex_token": CategoryProfile(
            category="new_dex_token",
            description="Young tokens traded mainly in DEX pools.",
            evidence=(
                _r("dex_liquidity", "required", "primary", critical=True),
                _r("token_authorities", "required", "primary", critical=True),
                _r("holder_concentration", "required", "primary", critical=True),
                _r("buy_sell_flow", "required", "primary"),
                _r("pool_age", "required", "primary"),
                _r("onchain_activity", "optional", "supporting"),
                _r("social", "optional", "supporting"),
                _r("technical_structure", "optional", "context"),
                _r("market_snapshot", "optional", "context"),
                _r("news", "optional", "context"),
            ),
            risk_characteristics=(
                "Pool liquidity can be withdrawn, and thin pools move sharply on small trades.",
                "Mint or freeze authorities can create supply or block transfers.",
                "A few wallets may hold most of the supply.",
                "Short price history makes indicators unreliable.",
            ),
            risk_thresholds_calibrated=False,
            technical_min_candles=MIN_TECHNICAL_CANDLES,
        ),
        "stablecoin": CategoryProfile(
            category="stablecoin",
            description="Tokens designed to hold a peg to a currency.",
            evidence=(
                _r("peg_stability", "required", "primary", critical=True),
                _r("issuer_risk", "required", "primary", critical=True),
                _r("market_snapshot", "required", "primary"),
                _r("news", "optional", "supporting"),
                _r("technical_structure", "optional", "not_used"),
            ),
            risk_characteristics=(
                "The main risk is losing the peg.",
                "Issuer reserves, redemptions and regulation drive that risk.",
                "RSI, MACD and trend are not meaningful decision sources.",
            ),
            risk_thresholds_calibrated=False,
        ),
        "unknown_crypto": CategoryProfile(
            category="unknown_crypto",
            description="Assets whose type couldn't be established from the evidence.",
            evidence=(
                _r("technical_structure", "required", "supporting", critical=True),
                _r("market_snapshot", "required", "supporting"),
                _r("news", "optional", "context"),
            ),
            risk_characteristics=(
                "The asset type is unknown, so generic evidence weighting applies.",
                "A ticker-only identity can collide with other tokens using the same ticker.",
            ),
            risk_thresholds_calibrated=False,
        ),
    }
)


# --- Identity --------------------------------------------------------------------------------


class AssetIdentity(BaseModel):
    """How the user or a provider identified the asset. At least one field must be set.

    Prefer the most exact form available: chain + address, then a CEX pair, then a ticker.
    """

    symbol: str | None = None
    name: str | None = None
    chain: str | None = None
    address: str | None = None
    kraken_pair: str | None = None
    # Facts a provider already knows about this exact token (e.g. from a DEX pool).
    metadata: AssetMetadata | None = None

    @classmethod
    def of(cls, value: "AssetIdentity | str") -> "AssetIdentity":
        return value if isinstance(value, AssetIdentity) else cls(symbol=value.upper())


@dataclass(frozen=True)
class ResolvedIdentity:
    canonical_id: str
    basis: IdentityBasis
    metadata: AssetMetadata
    registered: bool
    # A ticker-only match that isn't in the registry: could be any token with that ticker.
    ambiguous: bool
    # UpScale's current providers (Kraken, CoinGecko, news) look assets up by ticker.
    # Their data only belongs to this asset when the ticker resolves to it.
    ticker_data_attributable: bool


def resolve_identity(
    identity: AssetIdentity, registry: AssetRegistry = DEFAULT_REGISTRY
) -> ResolvedIdentity | None:
    if identity.address:
        if not identity.chain:
            return None  # an address without its chain is not an identity
        address = normalize_address(identity.chain, identity.address)
        entry = registry.by_address(identity.chain, identity.address)
        canonical = f"{identity.chain}:{address}"
        if entry is not None:
            return _resolved(_registry_id(entry), "contract", entry, registered=True)
        meta = _caller_metadata(identity, chain=identity.chain, address=address)
        return _resolved(canonical, "contract", meta, attributable=False)
    if identity.kraken_pair:
        entry = registry.by_kraken_pair(identity.kraken_pair)
        if entry is not None:
            return _resolved(_registry_id(entry), "cex_pair", entry, registered=True)
        meta = _caller_metadata(identity, kraken_pair=identity.kraken_pair.upper())
        return _resolved(f"kraken:{identity.kraken_pair.upper()}", "cex_pair", meta)
    if identity.symbol:
        entry = registry.by_symbol(identity.symbol)
        if entry is not None:
            return _resolved(_registry_id(entry), "registry_symbol", entry, registered=True)
        meta = _caller_metadata(identity)
        return _resolved(f"symbol:{identity.symbol.upper()}", "symbol_only", meta, ambiguous=True)
    return None


def _resolved(
    canonical_id: str,
    basis: IdentityBasis,
    metadata: AssetMetadata,
    *,
    registered: bool = False,
    ambiguous: bool = False,
    attributable: bool = True,
) -> ResolvedIdentity:
    return ResolvedIdentity(
        canonical_id=canonical_id,
        basis=basis,
        metadata=metadata,
        registered=registered,
        ambiguous=ambiguous,
        ticker_data_attributable=attributable,
    )


def _registry_id(entry: AssetMetadata) -> str:
    if entry.coingecko_id:
        return f"coingecko:{entry.coingecko_id}"
    return f"{entry.chain}:{entry.address}" if entry.address else f"symbol:{entry.symbol}"


def _caller_metadata(identity: AssetIdentity, **extra: str | None) -> AssetMetadata:
    base = identity.metadata or AssetMetadata(symbol=(identity.symbol or "?").upper())
    updates: dict[str, object] = {k: v for k, v in extra.items() if v is not None}
    if identity.name and base.name is None:
        updates["name"] = identity.name
    if identity.symbol and base.symbol == "?":
        updates["symbol"] = identity.symbol.upper()
    return base.model_copy(update=updates)


# --- Profile model ---------------------------------------------------------------------------


class CapabilityStatus(BaseModel):
    capability: DataCapability
    status: CapabilityState
    verified: bool  # True when this turn's results confirmed it, False when inferred
    reason: str


Sufficiency = Literal["required", "important", "optional"]


class EvidenceStatus(BaseModel):
    evidence: EvidenceType
    requirement: Requirement
    weight: Weight
    decision_critical: bool
    status: EvidenceState
    reason: str
    # How much its absence matters: required (safety-critical, can block BUY/SELL),
    # important (lowers confidence / adds a caution), optional (lowers confidence at most).
    sufficiency: Sufficiency = "optional"


def sufficiency(rule: "EvidenceRule") -> Sufficiency:
    if rule.decision_critical:
        return "required"
    return "important" if rule.requirement == "required" else "optional"


class PlannedStep(BaseModel):
    """One piece of the analysis this profile calls for, and whether it can run."""

    evidence: EvidenceType
    agent: AgentName | None  # the agent that produces it, when one exists
    weight: Weight
    status: StepStatus
    reason: str


class CryptoAssetProfile(BaseModel):
    canonical_id: str  # e.g. "coingecko:bitcoin" or "solana:<mint address>"
    identity_basis: IdentityBasis
    identity_ambiguous: bool
    ticker_data_attributable: bool
    symbol: str
    name: str | None
    chain: str | None
    address: str | None
    category: Category
    category_label: str
    category_reasons: list[str]
    market_type: MarketType
    # Static size tier used for classification (registry metadata, or the live class when
    # the asset isn't in the registry). Not a statement about the current market cap.
    classification_cap_tier: MarketCapClass | None
    classification_cap_tier_basis: str | None
    # Live market cap, only from a provider this turn (None when none reported it).
    market_cap_usd: float | None
    market_cap_class: MarketCapClass | None
    market_cap_basis: str | None
    liquidity_class: LiquidityClass | None
    liquidity_basis: str | None
    volatility_class: VolatilityClass | None
    volatility_basis: str | None
    asset_age_days: int | None
    pool_age_days: int | None
    cex_available: bool | None
    dex_available: bool | None
    capabilities: list[CapabilityStatus]
    required_evidence: list[EvidenceType]
    optional_evidence: list[EvidenceType]
    planned_evidence: list[EvidenceType]
    decision_critical_evidence: list[EvidenceType]
    evidence_weights: dict[EvidenceType, Weight]
    evidence: list[EvidenceStatus]
    analysis_plan: list[PlannedStep]
    risk_characteristics: list[str]
    risk_thresholds_calibrated: bool
    risk_overrides: dict[str, float] = Field(default_factory=dict)
    missing_metadata: list[str]
    metadata_source: str
    # Evidence the market being traded demands, independent of the asset's category.
    market_policy: "MarketPolicy | None" = None

    def capability(self, name: DataCapability) -> CapabilityStatus:
        return next(c for c in self.capabilities if c.capability == name)

    def evidence_status(self, name: EvidenceType) -> EvidenceStatus | None:
        return next((e for e in self.evidence if e.evidence == name), None)

    def unavailable_critical(self) -> list[EvidenceStatus]:
        """Decision-critical evidence that wasn't actually reported. At decision time,
        "expected" (a provider exists but nothing arrived) is as missing as "unavailable"."""
        return [e for e in self.evidence if e.decision_critical and e.status != "available"]

    @property
    def technical_usable(self) -> bool:
        status = self.evidence_status("technical_structure")
        return status is None or status.status != "insufficient"


def usable(e: EvidenceStatus) -> bool:
    return e.status in ("available", "expected", "unknown")


def planned_agents(profile: CryptoAssetProfile) -> list[AgentName]:
    """Evidence agents this profile asks for that can actually run, in plan order."""
    seen: list[AgentName] = []
    for step in profile.analysis_plan:
        if step.agent and step.status in ("run", "conditional") and step.agent not in seen:
            seen.append(step.agent)
    return [*seen, "risk", "opportunity"]


# --- Building a profile ----------------------------------------------------------------------


@dataclass(frozen=True)
class Observations:
    """What this turn's agents reported for the asset (from `risk.collect_inputs`)."""

    snapshot: MarketSnapshot | None = None
    technical: TechnicalAnalysis | None = None
    dex: SolanaDexSnapshot | None = None  # keyed by mint, so never ticker-ambiguous
    onchain: OnchainSafetySnapshot | None = None  # keyed by mint too
    # Input agent -> (status, detail), with the statuses of `risk.InputState`.
    states: Mapping[AgentName, tuple[str, str | None]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Facts:
    market_cap_usd: float | None
    live_cap_class: MarketCapClass | None
    live_cap_basis: str | None
    cap_class: MarketCapClass | None  # classification tier: static if known, else live
    cap_basis: str | None
    age_days: int | None  # from the static launch date
    pool_age_days: int | None  # primary pool
    first_pool_age_days: int | None  # oldest pool: a lower bound on the token's age
    dex_liquidity_usd: float | None
    dex_liquidity_basis: str | None
    cex: bool | None
    dex: bool | None

    @property
    def token_age_days(self) -> int | None:
        """Best available age: launch date, else the oldest pool, else the primary pool."""
        for age in (self.age_days, self.first_pool_age_days, self.pool_age_days):
            if age is not None:
                return age
        return None


def build_profile(
    identity: AssetIdentity | str | None,
    observed: Observations | None = None,
    *,
    registry: AssetRegistry = DEFAULT_REGISTRY,
    config: ProfileConfig | None = None,
    categories: Mapping[Category, CategoryProfile] = CATEGORY_PROFILES,
    integrated: frozenset[DataCapability] | None = None,
    now: datetime | None = None,
    venue: str | None = None,
) -> CryptoAssetProfile | None:
    """The asset's profile, or None when there is no usable identity. `integrated`
    defaults to the capabilities configured at startup. `venue` ("cex" / "dex") is the
    market being traded when known; it sets the market policy (see `market_policy_for`)."""
    if integrated is None:
        integrated = integrated_capabilities()
    if identity is None:
        return None
    resolved = resolve_identity(AssetIdentity.of(identity), registry)
    if resolved is None:
        return None
    cfg = config or ProfileConfig()
    now = now or datetime.now(UTC)
    meta = resolved.metadata
    obs = _attributable_observations(observed or Observations(), resolved)
    if obs.dex is not None and meta.symbol == "?" and obs.dex.symbol:
        # A mint-only identity takes its label from DEX data for that exact mint. The
        # label is display only: identity stays the mint.
        meta = meta.model_copy(update={"symbol": obs.dex.symbol, "name": meta.name or obs.dex.name})
    facts = _facts(meta, resolved, obs, cfg, now)
    category, reasons = classify(meta, facts, resolved, cfg)
    policy = market_policy_for(meta, resolved, obs, venue)
    spec = apply_market_policy(categories[category], policy)
    caps = _capabilities(resolved, obs, integrated)
    evidence = _evidence(spec, caps, obs, meta)
    liquidity, liquidity_basis = _liquidity(obs.snapshot, facts, cfg)
    volatility, volatility_basis = _volatility(obs.snapshot, cfg)
    return CryptoAssetProfile(
        canonical_id=resolved.canonical_id,
        identity_basis=resolved.basis,
        identity_ambiguous=resolved.ambiguous,
        ticker_data_attributable=resolved.ticker_data_attributable,
        symbol=meta.symbol,
        name=meta.name or (obs.snapshot.name if obs.snapshot else None),
        chain=meta.chain,
        address=meta.address,
        category=category,
        category_label=CATEGORY_LABELS[category],
        category_reasons=reasons,
        market_type=_market_type(facts.cex, facts.dex),
        classification_cap_tier=facts.cap_class,
        classification_cap_tier_basis=facts.cap_basis,
        market_cap_usd=facts.market_cap_usd,
        market_cap_class=facts.live_cap_class,
        market_cap_basis=facts.live_cap_basis,
        liquidity_class=liquidity,
        liquidity_basis=liquidity_basis,
        volatility_class=volatility,
        volatility_basis=volatility_basis,
        asset_age_days=facts.age_days,
        pool_age_days=facts.pool_age_days,
        cex_available=facts.cex,
        dex_available=facts.dex,
        capabilities=caps,
        required_evidence=[r.evidence for r in spec.evidence if r.requirement == "required"],
        optional_evidence=[r.evidence for r in spec.evidence if r.requirement == "optional"],
        planned_evidence=[r.evidence for r in spec.evidence if r.requirement == "planned"],
        decision_critical_evidence=[r.evidence for r in spec.evidence if r.decision_critical],
        evidence_weights={r.evidence: r.weight for r in spec.evidence},
        evidence=evidence,
        analysis_plan=_plan(
            evidence, conditional_technical=_conditional_technical(spec, obs, meta)
        ),
        risk_characteristics=list(spec.risk_characteristics),
        risk_thresholds_calibrated=spec.risk_thresholds_calibrated,
        risk_overrides=dict(spec.risk_overrides),
        missing_metadata=_missing_metadata(meta, facts),
        metadata_source=meta.source,
        market_policy=policy,
    )


MarketPolicyKind = Literal["dex_contract", "exchange", "unspecified"]


class MarketPolicy(BaseModel):
    """What the market being traded requires, separately from what kind of asset it is.

    A **DEX contract trade** (a token identified by its Solana mint / EVM contract, traded
    in a DEX pool) must show, before any BUY: usable DEX liquidity, enough candles from the
    token's own pool, and chain-specific token safety (authorities, holder concentration).
    That holds for any such token: meme-tagged or not, new or years old. On a centralized
    exchange the same asset follows its category's evidence rules instead.
    """

    kind: MarketPolicyKind
    reason: str
    required: list[EvidenceType] = Field(default_factory=list)


# Decision-critical evidence for every DEX contract trade, whatever the category.
DEX_CONTRACT_REQUIREMENTS: tuple[EvidenceRule, ...] = (
    EvidenceRule("dex_liquidity", "required", "primary", True),
    EvidenceRule("technical_structure", "required", "primary", True),
    EvidenceRule("token_authorities", "required", "primary", True),
    EvidenceRule("holder_concentration", "required", "primary", True),
)


def market_policy_for(
    meta: AssetMetadata, resolved: ResolvedIdentity, obs: Observations, venue: str | None
) -> MarketPolicy:
    """DEX contract trade when the token has a chain + address and the analyzed market is a
    DEX: stated by the trader, implied by identifying the token by its address, or shown
    by the candles actually coming from the token's own pool."""
    on_chain = meta.chain in DEX_CHAINS and bool(meta.address)
    if on_chain and venue != "cex":
        pool_candles = technical_matches(obs.technical, resolved.canonical_id)
        if venue == "dex" or resolved.basis == "contract" or pool_candles:
            why = (
                "the trader asked about a DEX market"
                if venue == "dex"
                else "the token was identified by its contract/mint address"
                if resolved.basis == "contract"
                else "the analyzed candles come from the token's DEX pool"
            )
            return MarketPolicy(
                kind="dex_contract",
                reason=(
                    f"DEX contract trade ({why}): liquidity, the token's own pool candles, "
                    "token authorities and holder concentration are required before any BUY."
                ),
                required=[r.evidence for r in DEX_CONTRACT_REQUIREMENTS],
            )
    if venue == "cex" or meta.cex_listed or meta.kraken_pair:
        return MarketPolicy(
            kind="exchange",
            reason="Centralized-exchange trade: the asset's category sets the evidence rules.",
        )
    return MarketPolicy(kind="unspecified", reason="No specific market; category rules apply.")


def apply_market_policy(spec: CategoryProfile, policy: MarketPolicy) -> CategoryProfile:
    """The category's evidence rules, with the market policy's requirements made
    decision-critical (added when missing, upgraded when present)."""
    if policy.kind != "dex_contract":
        return spec
    by_evidence = {r.evidence: r for r in spec.evidence}
    for rule in DEX_CONTRACT_REQUIREMENTS:
        by_evidence[rule.evidence] = rule
    ordered = [by_evidence[r.evidence] for r in spec.evidence] + [
        r
        for r in DEX_CONTRACT_REQUIREMENTS
        if r.evidence not in {x.evidence for x in spec.evidence}
    ]
    return dataclasses.replace(
        spec,
        evidence=tuple(ordered),
        technical_min_candles=spec.technical_min_candles or MIN_TECHNICAL_CANDLES,
    )


def dex_matches(dex: SolanaDexSnapshot | None, meta: AssetMetadata) -> bool:
    """DEX data belongs to this asset only when it was fetched for this exact token."""
    return (
        dex is not None
        and meta.chain is not None
        and dex.chain == meta.chain
        and same_address(meta.chain, dex.mint, meta.address)
    )


def technical_matches(technical: TechnicalAnalysis | None, canonical_id: str) -> bool:
    """Candles keyed by this exact asset (e.g. its DEX pool), not by a shared ticker."""
    return technical is not None and technical.canonical_id == canonical_id


def onchain_matches(onchain: OnchainSafetySnapshot | None, meta: AssetMetadata) -> bool:
    return onchain is not None and meta.chain == "solana" and onchain.mint == meta.address


def _attributable_observations(obs: Observations, resolved: ResolvedIdentity) -> Observations:
    """Keep only evidence that belongs to this exact asset: mint-keyed data (DEX, on-chain)
    for its mint, and ticker-keyed data only when the ticker is proven to be its own."""
    meta = resolved.metadata
    mint_keyed: dict[AgentName, bool] = {
        "dex_market": obs.dex is None or dex_matches(obs.dex, meta),
        "onchain_safety": obs.onchain is None or onchain_matches(obs.onchain, meta),
    }
    # Candles keyed by this exact asset (its DEX pool) belong to it whatever the ticker.
    mint_keyed["technical_analysis"] = technical_matches(obs.technical, resolved.canonical_id)
    ticker_ok = resolved.ticker_data_attributable

    def keep(agent: AgentName) -> bool:
        if agent == "technical_analysis":
            return ticker_ok or mint_keyed[agent]
        return mint_keyed[agent] if agent in mint_keyed else ticker_ok

    states = {k: v for k, v in obs.states.items() if keep(k)}
    return Observations(
        snapshot=obs.snapshot if ticker_ok else None,
        technical=obs.technical if ticker_ok or mint_keyed["technical_analysis"] else None,
        dex=obs.dex if mint_keyed["dex_market"] else None,
        onchain=obs.onchain if mint_keyed["onchain_safety"] else None,
        states=states,
    )


def _age_days(start: date | datetime | None, now: datetime) -> int | None:
    if start is None:
        return None
    if isinstance(start, datetime):
        start_dt = start if start.tzinfo else start.replace(tzinfo=UTC)
    else:
        start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    return max(0, (now - start_dt).days)


_CAP_CLASSES: dict[str, MarketCapClass] = {
    "mega": "mega",
    "large": "large",
    "mid": "mid",
    "small": "small",
    "micro": "micro",
}
_LIQUIDITY_LABELS: tuple[LiquidityClass, ...] = ("deep", "high", "medium", "low", "very_low")
_VOLATILITY_LABELS: tuple[VolatilityClass, ...] = ("extreme", "high", "moderate", "low")


def cap_class(market_cap_usd: float, cfg: ProfileConfig) -> MarketCapClass:
    if market_cap_usd >= cfg.mega_cap_usd:
        return "mega"
    if market_cap_usd >= cfg.large_cap_usd:
        return "large"
    if market_cap_usd >= cfg.mid_cap_usd:
        return "mid"
    return "small" if market_cap_usd >= cfg.small_cap_usd else "micro"


def _facts(
    meta: AssetMetadata,
    resolved: ResolvedIdentity,
    obs: Observations,
    cfg: ProfileConfig,
    now: datetime,
) -> _Facts:
    live_cap = obs.snapshot.market_cap_usd if obs.snapshot else None
    market_cap = live_cap if live_cap is not None else meta.market_cap_usd
    live_cls: MarketCapClass | None = None
    live_basis: str | None = None
    dex = obs.dex
    dex_reported_cap = False
    if market_cap is None and dex is not None and dex.market_cap_usd is not None:
        # Shown for context only: a DEX-reported market cap is supply x pool price, so it
        # never decides the category (a thin pool can make it arbitrarily large).
        market_cap, dex_reported_cap = dex.market_cap_usd, True
        live_cls = cap_class(market_cap, cfg)
        live_basis = f"reported by {dex.provider} (supply x pool price); not used to classify"
    elif market_cap is not None:
        live_cls = cap_class(market_cap, cfg)
        source = (
            f"live {obs.snapshot.provider} market cap"
            if live_cap is not None and obs.snapshot
            else meta.source
        )
        live_basis = source + (" (ticker-only match)" if resolved.ambiguous else "")
    # Static tier first, so classification doesn't flip with daily price moves.
    cls: MarketCapClass | None = _CAP_CLASSES.get(meta.classification_cap_tier or "")
    basis = f"static classification metadata, {meta.source}; not live market data" if cls else None
    if cls is None and not dex_reported_cap:
        cls, basis = live_cls, live_basis
    cex = meta.cex_listed
    if cex is None and (meta.kraken_pair or _served_by(obs, "Kraken")):
        cex = True
    return _Facts(
        market_cap_usd=market_cap,
        live_cap_class=live_cls,
        live_cap_basis=live_basis,
        cap_class=cls,
        cap_basis=basis,
        age_days=_age_days(meta.launched, now),
        pool_age_days=_age_days(dex.pair_created_at if dex else meta.pool_created_at, now),
        first_pool_age_days=_age_days(dex.first_pool_created_at, now) if dex else None,
        dex_liquidity_usd=dex.liquidity_usd if dex else meta.liquidity_usd,
        dex_liquidity_basis=(
            f"{dex.provider} primary pool, {dex.dex} {dex.pair_address}" if dex else meta.source
        ),
        cex=cex,
        dex=True if dex is not None else meta.dex_listed,
    )


def _served_by(obs: Observations, provider: str) -> bool:
    return obs.technical is not None and obs.technical.provider == provider


def classify(
    meta: AssetMetadata, facts: _Facts, resolved: ResolvedIdentity, cfg: ProfileConfig
) -> tuple[Category, list[str]]:
    """Category and the characteristics that decided it (see the module docstring)."""
    if resolved.ambiguous:
        return "unknown_crypto", [
            f"{meta.symbol} was identified by ticker only and isn't in UpScale's asset "
            "registry; a category needs an exact identity (registry entry, exchange pair, "
            "or chain + contract address)."
        ]
    if "stablecoin" in meta.tags or meta.peg:
        return "stablecoin", [f"Stablecoin{f' pegged to {meta.peg}' if meta.peg else ''}."]

    age = facts.token_age_days
    if facts.dex is True and facts.cex is not True:
        if age is None or age < cfg.new_token_max_age_days:
            when = (
                "age unknown"
                if age is None
                else f"{age} days old (under {cfg.new_token_max_age_days})"
            )
            return "new_dex_token", [
                "Traded on a DEX with no known centralized-exchange listing.",
                f"Young token: {when}.",
            ]

    cap = facts.cap_class
    if "meme" in meta.tags:
        old = age is not None and age >= cfg.memecoin_min_age_days
        deep_pool = (facts.dex_liquidity_usd or 0) >= cfg.established_dex_liquidity_usd
        liquid = cap in ("mega", "large", "mid") or facts.cex is True or deep_pool
        if old and liquid:
            if cap in ("mega", "large", "mid"):
                depth = f"{cap} market-cap tier"
            elif facts.cex is True:
                depth = "listed on a CEX"
            else:
                depth = f"${facts.dex_liquidity_usd:,.0f} DEX pool liquidity"
            return "established_memecoin", [
                "Memecoin.",
                f"{age} days of history (at least {cfg.memecoin_min_age_days}).",
                f"Substantial market: {depth}.",
            ]
        missing = []
        if not old:
            missing.append(
                f"at least {cfg.memecoin_min_age_days} days of history"
                + ("" if age is None else f" (has {age})")
            )
        if not liquid:
            missing.append(
                "a mid-or-larger market-cap tier, a centralized-exchange listing, or "
                f"${cfg.established_dex_liquidity_usd:,.0f}+ DEX pool liquidity"
            )
        return "unknown_crypto", [f"Memecoin without {' and '.join(missing)}."]

    if cap == "mega":
        return "major_crypto", [f"Mega market-cap tier ({facts.cap_basis})."]
    if cap == "large":
        return "large_cap_alt", [f"Large market-cap tier ({facts.cap_basis})."]
    if cap == "mid" and facts.cex is True and (age or 0) >= cfg.established_min_age_days:
        return "large_cap_alt", [
            f"Mid market-cap tier ({facts.cap_basis}), listed on centralized exchanges with "
            f"{age} days of history: treated as an established large-cap altcoin.",
        ]

    unknowns = [
        label
        for label, value in (
            ("market cap", cap),
            ("age", age),
            ("centralized-exchange listing", facts.cex),
            ("DEX listing", facts.dex),
        )
        if value is None
    ]
    reason = "Characteristics don't match a known category"
    if facts.dex is True and facts.cex is not True and age is not None:
        reason = (
            f"DEX-traded for {age} days, so not a new DEX token, but not tagged as a "
            "memecoin or another known kind of asset"
        )
    if unknowns:
        reason += f" (unknown: {', '.join(unknowns)})"
    return "unknown_crypto", [reason + "."]


def _market_type(cex: bool | None, dex: bool | None) -> MarketType:
    if cex and dex:
        return "cex_and_dex"
    if cex:
        return "cex_spot"
    if dex:
        return "dex_spot"
    return "unknown"


def _band(value: float, bands: tuple[float, ...], labels: tuple[T, ...]) -> T:
    """The label of the first band (highest first) that `value` reaches, else the last."""
    for threshold, label in zip(bands, labels, strict=False):
        if value >= threshold:
            return label
    return labels[-1]


def _liquidity(
    snapshot: MarketSnapshot | None, facts: _Facts, cfg: ProfileConfig
) -> tuple[LiquidityClass | None, str | None]:
    liquidity = facts.dex_liquidity_usd
    if facts.dex is True and facts.cex is not True and liquidity is not None:
        cls = _band(liquidity, cfg.pool_liquidity_bands, _LIQUIDITY_LABELS)
        return cls, f"DEX pool liquidity ${liquidity:,.0f} ({facts.dex_liquidity_basis})"
    if snapshot is not None and snapshot.volume_24h_usd is not None:
        cls = _band(snapshot.volume_24h_usd, cfg.volume_bands, _LIQUIDITY_LABELS)
        return cls, f"24h volume ${snapshot.volume_24h_usd:,.0f} ({snapshot.provider})"
    return None, None


def _volatility(
    snapshot: MarketSnapshot | None, cfg: ProfileConfig
) -> tuple[VolatilityClass | None, str | None]:
    if snapshot is None or snapshot.high_24h_usd is None or snapshot.low_24h_usd is None:
        return None, None
    if snapshot.low_24h_usd <= 0:
        return None, None
    span = 100 * (snapshot.high_24h_usd - snapshot.low_24h_usd) / snapshot.low_24h_usd
    cls = _band(span, tuple(reversed(cfg.volatility_bands)), _VOLATILITY_LABELS)
    return cls, f"24h high-low range {span:.1f}% ({snapshot.provider}); one day only"


def _capabilities(
    resolved: ResolvedIdentity, obs: Observations, integrated: frozenset[DataCapability]
) -> list[CapabilityStatus]:
    meta = resolved.metadata
    out: list[CapabilityStatus] = []
    for cap in CAPABILITIES:
        label = CAPABILITY_LABELS[cap]
        if cap == "onchain" and meta.chain in EVM_CHAINS and meta.address:
            out.append(_onchain_status(meta, obs, _result_for(cap)))
            continue
        if cap not in integrated:
            reason = (
                "No provider for on-chain data is configured (set UPSCALE_HELIUS_API_KEY or "
                "UPSCALE_SOLANA_RPC_URL)."
                if cap == "onchain"
                else f"No provider for {label} is integrated in UpScale yet."
            )
            out.append(
                CapabilityStatus(
                    capability=cap, status="unavailable", verified=False, reason=reason
                )
            )
            continue
        if cap == "candles" and not resolved.ticker_data_attributable and _has_pools(meta):
            out.append(_pool_candle_status(resolved, obs))
            continue
        if not resolved.ticker_data_attributable and cap in _TICKER_KEYED:
            out.append(
                CapabilityStatus(
                    capability=cap,
                    status="unavailable",
                    verified=False,
                    reason=(
                        f"UpScale looks up {label} by ticker, so they can't be tied to "
                        f"{resolved.canonical_id}; another token may share the ticker "
                        f"{meta.symbol}."
                    ),
                )
            )
            continue
        out.append(_integrated_status(cap, resolved, obs))
    return out


_TICKER_KEYED: frozenset[DataCapability] = frozenset({"candles", "market_snapshot", "news"})
_CAPABILITY_AGENT: dict[DataCapability, AgentName] = {
    "candles": "technical_analysis",
    "market_snapshot": "market",
    "news": "news_sentiment",
}


def _integrated_status(
    cap: DataCapability, resolved: ResolvedIdentity, obs: Observations
) -> CapabilityStatus:
    meta = resolved.metadata
    agent = _CAPABILITY_AGENT.get(cap)
    status, detail = obs.states.get(agent, ("not_run", None)) if agent else ("not_run", None)

    def result(state: CapabilityState, verified: bool, reason: str) -> CapabilityStatus:
        return CapabilityStatus(capability=cap, status=state, verified=verified, reason=reason)

    if cap == "candles":
        if obs.technical is not None:
            if obs.technical.provider == "Kraken":
                return result("available", True, f"Kraken served {obs.technical.pair} candles.")
            why = "; ".join(obs.technical.fallback_notes) or "another provider served candles"
            return result("unavailable", True, f"Kraken wasn't used this turn: {why}.")
        if status in ("failed", "no_data"):
            return result("unavailable", True, f"Candles couldn't be retrieved: {detail}.")
        if meta.kraken_pair:
            return result("available", False, f"Registry lists Kraken pair {meta.kraken_pair}.")
        return result(
            "unknown",
            False,
            f"Kraken pairs are found by ticker; whether {meta.symbol}/USD exists is only "
            "known after a request.",
        )
    if cap == "market_snapshot":
        if obs.snapshot is not None:
            return result("available", True, f"{obs.snapshot.provider} reported a snapshot.")
        if status in ("failed", "no_data"):
            return result("unavailable", True, f"No snapshot this turn: {detail}.")
        if meta.coingecko_id:
            return result("available", False, f"Registry lists CoinGecko id {meta.coingecko_id}.")
        return result(
            "unknown",
            False,
            "CoinGecko would pick the largest coin with this ticker, which may be a "
            "different token.",
        )
    if cap == "dex":
        return _dex_status(meta, obs, status, detail, result)
    if cap == "onchain":
        return _onchain_status(meta, obs, result)
    if cap != "news":
        return _future_provider_status(cap, meta, result)
    if status == "ok":
        return result("available", True, "Classified news was found this turn.")
    if status in ("failed", "no_data"):
        return result("unavailable", True, f"No classified news this turn: {detail}.")
    if resolved.registered:
        return result("available", False, "Publisher RSS feeds, matched by ticker and name.")
    return result(
        "unknown", False, "Headlines are matched by ticker only and may be about another token."
    )


def _has_pools(meta: AssetMetadata) -> bool:
    return meta.chain in GECKOTERMINAL_NETWORKS and bool(meta.address)


def _result_for(cap: DataCapability) -> Callable[[CapabilityState, bool, str], CapabilityStatus]:
    def result(state: CapabilityState, verified: bool, reason: str) -> CapabilityStatus:
        return CapabilityStatus(capability=cap, status=state, verified=verified, reason=reason)

    return result


def _pool_candle_status(resolved: ResolvedIdentity, obs: Observations) -> CapabilityStatus:
    """Candles for a contract token come from its own DEX pool, never from a ticker."""
    status, detail = obs.states.get("technical_analysis", ("not_run", None))
    if technical_matches(obs.technical, resolved.canonical_id) and obs.technical is not None:
        reason = f"{obs.technical.provider} served candles for this token's pool."
        return CapabilityStatus(
            capability="candles", status="available", verified=True, reason=reason
        )
    if status in ("failed", "no_data"):
        return CapabilityStatus(
            capability="candles",
            status="unavailable",
            verified=True,
            reason=f"No pool candles this turn: {detail}.",
        )
    return CapabilityStatus(
        capability="candles",
        status="available",
        verified=False,
        reason="Candles are read from this token's selected DEX pool.",
    )


def _dex_status(
    meta: AssetMetadata,
    obs: Observations,
    status: str,
    detail: str | None,
    result: Callable[[CapabilityState, bool, str], CapabilityStatus],
) -> CapabilityStatus:
    if meta.chain not in DEX_CHAINS or not meta.address:
        return result(
            "unavailable",
            False,
            f"UpScale's DEX data covers tokens identified by chain + contract/mint address "
            f"(Solana and EVM chains); {meta.symbol} has none.",
        )
    if obs.dex is not None:
        return result(
            "available",
            True,
            f"{obs.dex.provider} reported {len(obs.dex.candidates)} pool(s) for this mint.",
        )
    if status in ("failed", "no_data"):
        return result("unavailable", True, f"No usable DEX pool this turn: {detail}.")
    return result("available", False, f"DEX pools are looked up by mint {meta.address}.")


def _onchain_status(
    meta: AssetMetadata,
    obs: Observations,
    result: Callable[[CapabilityState, bool, str], CapabilityStatus],
) -> CapabilityStatus:
    if meta.chain in EVM_CHAINS and meta.address:
        return result(
            "unavailable",
            False,
            f"No on-chain safety provider is integrated for {chain_label(meta.chain)} yet "
            "(Solana only for now).",
        )
    if meta.chain != "solana" or not meta.address:
        return result(
            "unavailable",
            False,
            f"UpScale's on-chain data covers Solana tokens identified by mint; {meta.symbol} "
            "has no Solana mint.",
        )
    status, detail = obs.states.get("onchain_safety", ("not_run", None))
    if obs.onchain is not None:
        return result("available", True, f"{obs.onchain.provider} read the mint account.")
    if status in ("failed", "no_data"):
        return result("unavailable", True, f"No on-chain data this turn: {detail}.")
    return result("available", False, f"Mint {meta.address} is read from the chain.")


def _future_provider_status(
    cap: DataCapability,
    meta: AssetMetadata,
    result: Callable[[CapabilityState, bool, str], CapabilityStatus],
) -> CapabilityStatus:
    """Status for a provider that is integrated but has no agent reporting on it yet."""
    label = upper_first(CAPABILITY_LABELS[cap])
    if cap == "dex" and meta.dex_listed:
        return result("available", False, f"{label}: the asset is listed in a DEX pool.")
    if cap == "onchain" and meta.chain and meta.address:
        return result("available", False, f"{label}: token {meta.chain}:{meta.address}.")
    return result("unknown", False, f"{label}: coverage of {meta.symbol} isn't known yet.")


def _evidence(
    spec: CategoryProfile,
    caps: list[CapabilityStatus],
    obs: Observations,
    meta: AssetMetadata,
) -> list[EvidenceStatus]:
    by_cap = {c.capability: c for c in caps}
    candles = _candles(obs, meta)
    out: list[EvidenceStatus] = []
    for rule in spec.evidence:
        state, reason = _evidence_state(rule.evidence, spec, by_cap, obs, candles)
        out.append(
            EvidenceStatus(
                evidence=rule.evidence,
                requirement=rule.requirement,
                weight=rule.weight,
                decision_critical=rule.decision_critical,
                status=state,
                reason=reason,
                sufficiency=sufficiency(rule),
            )
        )
    return out


def _evidence_state(
    e: EvidenceType,
    spec: CategoryProfile,
    caps: Mapping[DataCapability, CapabilityStatus],
    obs: Observations,
    candles: int | None,
) -> tuple[EvidenceState, str]:
    label = EVIDENCE_LABELS[e]
    source = EVIDENCE_SOURCE[e]
    agent = EVIDENCE_AGENT.get(e)
    if source is None:
        return "unavailable", f"UpScale has no data source for {label}."
    cap = caps[source]
    if e == "technical_structure" and spec.technical_min_candles is not None:
        if candles is not None and candles < spec.technical_min_candles:
            return "insufficient", (
                f"only {candles} candles exist; {label} needs at least "
                f"{spec.technical_min_candles}."
            )
    if cap.status == "unavailable" and not (
        e == "technical_structure" and obs.technical is not None
    ):
        return "unavailable", cap.reason
    if agent is None:
        return "not_analyzed", (
            f"{upper_first(CAPABILITY_LABELS[source])} are available, but UpScale has no rule for "
            f"{label} yet."
        )
    status, detail = obs.states.get(agent, ("not_run", None))
    if status == "ok" and agent == "dex_market" and obs.dex is not None:
        return _dex_evidence(e, obs.dex)
    if status == "ok" and agent == "onchain_safety" and obs.onchain is not None:
        return _onchain_evidence(e, obs.onchain)
    if status == "ok" and (e != "technical_structure" or obs.technical is not None):
        return "available", "Reported this turn."
    if status in ("failed", "no_data", "unreadable"):
        return "unavailable", f"{agent} reported {status}: {detail}."
    if cap.status == "unknown":
        return "unknown", cap.reason
    return "expected", f"{upper_first(CAPABILITY_LABELS[source])} are available; not reported yet."


def upper_first(text: str) -> str:
    """Capitalize the first letter only ("DEX pool data" stays "DEX pool data")."""
    return text[:1].upper() + text[1:]


def _candles(obs: Observations, meta: AssetMetadata) -> int | None:
    return obs.technical.candle_count if obs.technical is not None else meta.candle_history


def _conditional_technical(spec: CategoryProfile, obs: Observations, meta: AssetMetadata) -> bool:
    """Technical has a minimum history for this category that hasn't been checked yet."""
    return spec.technical_min_candles is not None and _candles(obs, meta) is None


def _dex_evidence(e: EvidenceType, dex: SolanaDexSnapshot) -> tuple[EvidenceState, str]:
    """DEX evidence is available only for the fields the provider actually reported."""
    where = f"{dex.provider}, {dex.dex} pool {dex.pair_address}"
    if e == "pool_age":
        if dex.pair_created_at is None:
            return "unavailable", f"{dex.provider} didn't report when the pool was created."
        return "available", f"Pool created {dex.pair_created_at:%Y-%m-%d %H:%M} UTC ({where})."
    if e == "buy_sell_flow":
        if not any(w.buys is not None and w.sells is not None for w in dex.windows):
            return "unavailable", f"{dex.provider} didn't report buy/sell counts."
        return "available", f"Buy/sell counts reported ({where})."
    return "available", f"${dex.liquidity_usd:,.0f} liquidity ({where})."


def _onchain_evidence(e: EvidenceType, s: OnchainSafetySnapshot) -> tuple[EvidenceState, str]:
    """Authorities come straight from the mint account. Holder concentration only counts
    when the largest accounts' owners could be resolved; otherwise it is insufficient
    (never assumed safe)."""
    if e == "token_authorities":
        if not s.authorities_available:
            return "unavailable", f"The mint account couldn't be read: {s.authorities_error}."
        mint = "active" if s.mint_authority_active else "revoked"
        freeze = "active" if s.freeze_authority_active else "revoked"
        return "available", f"Mint authority {mint}, freeze authority {freeze} ({s.provider})."
    if not s.holders_available:
        return "unavailable", f"Holder data couldn't be read: {s.holders_error}."
    if not s.concentration_authoritative:
        # Lower bounds (largest accounts only, or a scan cut short) can flag concentration,
        # but can't show it is low enough to rely on.
        return "insufficient", "Holder data is incomplete: " + "; ".join(s.incomplete_reasons) + "."
    top10 = f"{s.top10_pct:.1f}%" if s.top10_pct is not None else "unknown"
    return "available", (
        f"Top 10 non-pool holders hold {top10} of supply (full scan by owner, {s.provider})."
    )


def wants_onchain_data(profile: CryptoAssetProfile) -> bool:
    """True when this kind of asset needs token-safety evidence and UpScale can read it."""
    safety = {"token_authorities", "holder_concentration"}
    return profile.capability("onchain").status != "unavailable" and any(
        e.evidence in safety for e in profile.evidence
    )


def wants_dex_data(profile: CryptoAssetProfile) -> bool:
    """True when this kind of asset calls for DEX evidence and UpScale can fetch it
    (a Solana token identified by mint)."""
    dex_evidence = {"dex_liquidity", "buy_sell_flow", "pool_age"}
    return profile.capability("dex").status != "unavailable" and any(
        e.evidence in dex_evidence and e.weight != "not_used" for e in profile.evidence
    )


def _plan(evidence: list[EvidenceStatus], conditional_technical: bool) -> list[PlannedStep]:
    order: dict[Weight, int] = {"primary": 0, "supporting": 1, "context": 2, "not_used": 3}
    steps: list[PlannedStep] = []
    for e in sorted(evidence, key=lambda e: order[e.weight]):
        if e.requirement == "planned":
            status: StepStatus = "unavailable"
            reason = f"Planned; {e.reason[:1].lower()}{e.reason[1:]}"
        elif e.weight == "not_used":
            status, reason = "skip", "Not a decision source for this kind of asset."
        elif e.status == "insufficient":
            status, reason = "skip", f"Skipped: {e.reason}"
        elif e.status in ("unavailable", "not_analyzed"):
            status, reason = "unavailable", e.reason
        elif e.evidence == "technical_structure" and conditional_technical:
            status = "conditional"
            reason = f"Runs only with at least {MIN_TECHNICAL_CANDLES} candles of history."
        else:
            status, reason = "run", e.reason
        steps.append(
            PlannedStep(
                evidence=e.evidence,
                agent=EVIDENCE_AGENT.get(e.evidence),
                weight=e.weight,
                status=status,
                reason=reason,
            )
        )
    return steps


def _missing_metadata(meta: AssetMetadata, facts: _Facts) -> list[str]:
    checks = (
        ("name", meta.name),
        ("chain", meta.chain),
        ("market cap", facts.cap_class),
        ("launch / pool creation date", facts.token_age_days),
        ("centralized-exchange listing", facts.cex),
        ("DEX listing", facts.dex),
    )
    return [label for label, value in checks if value is None]
