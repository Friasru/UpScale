"""The normalized context every trading request is analyzed in: which exact asset, on which
market, for which trader situation, what kind of asset it is, and what data exists for it.

Built once per request by the orchestrator (after the asset resolver) and handed to every
agent. Deterministic: position and intent come from the wording of the request and the
recent conversation, never from a model.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from upscale.services.asset_profile import CryptoAssetProfile
from upscale.services.chains import DEX_CHAINS

Position = Literal["none", "long", "short", "unknown"]
RequestedAction = Literal["buy", "sell", "hold", "buy_or_wait", "buy_or_sell", "analyze"]
Horizon = Literal["scalp", "intraday", "swing", "unknown"]
IdentityConfidence = Literal["exact", "registry", "discovered", "ticker_only"]
MarketType = Literal["spot", "perpetual", "dex", "unknown"]
VenueKind = Literal["cex", "dex", "unknown"]


class TradeIdentity(BaseModel):
    canonical_id: str | None
    symbol: str | None
    name: str | None = None
    chain: str | None = None
    address: str | None = None  # contract / mint, when applicable
    exchange_pair: str | None = None
    category: str | None = None
    confidence: IdentityConfidence


class MarketContext(BaseModel):
    market_type: MarketType
    venue_kind: VenueKind
    venue: str | None = None  # e.g. "Kraken" or "raydium", once known
    base_asset: str | None = None
    quote_asset: str | None = None
    pool: str | None = None  # selected DEX pool / pair address, once known
    timeframe: str | None = None
    # What the trader explicitly asked for: a DEX (e.g. "orca") or a pool address. An
    # explicit venue is never switched, not even for candles.
    requested_dex: str | None = None
    requested_pool: str | None = None
    venue_explicit: bool = False


class TraderContext(BaseModel):
    position: Position
    requested_action: RequestedAction
    horizon: Horizon
    explicit_timeframe: str | None = None
    leverage: float | None = None
    # Where the position came from: this message, or an earlier one in the conversation.
    position_source: Literal["message", "history", "default"] = "default"


class DataAvailability(BaseModel):
    capability: str
    status: str  # available / unavailable / unknown
    reason: str


class CryptoTradeContext(BaseModel):
    identity: TradeIdentity
    market: MarketContext
    trader: TraderContext
    profile: CryptoAssetProfile | None = None
    data: list[DataAvailability] = Field(default_factory=list)


# --- Trader context -------------------------------------------------------------------------

# fmt: off
_LONG = (
    r"\bmy (?:bag|bags|position|coins?|tokens?|holdings?)\b", r"\bsell my\b", r"\bi (?:own|hold|have|bought)\b",
    r"\bi'?m (?:holding|long|in profit|in loss|up|down)\b", r"\bim (?:holding|long)\b",
    r"\bholding\b", r"\bshould i (?:sell|hold|keep|exit|take profits?)\b", r"\bkeep (?:it|holding|my)\b",
    r"\b(?:hold|keep) (?:or sell|my)\b", r"\btake profits?\b", r"\bexit\b", r"\bmy [A-Z$][A-Za-z0-9]*\b",
)
_SHORT = (r"\bi'?m short\b", r"\bim short\b", r"\bmy short\b", r"\bcover (?:my )?short\b")
_NONE = (
    r"\bshould i (?:buy|enter|get in|ape)\b", r"\bbuy or wait\b", r"\bgood entry\b", r"\bentry\b",
    r"\bnot in\b", r"\bno position\b", r"\bworth buying\b",
)
# fmt: on
_LEVERAGE = re.compile(
    r"\b(\d{1,3}(?:\.\d+)?)\s*x\b(?=.*\b(?:leverage|long|short|lev)\b)|\b(?:leverage|lev)\s*(?:of\s*)?(\d{1,3}(?:\.\d+)?)\s*x?\b"
)


def _any(patterns: Sequence[str], text: str, flags: int = re.IGNORECASE) -> bool:
    return any(re.search(p, text, flags) for p in patterns)


def requested_action(text: str) -> RequestedAction:
    t = text.lower()
    if re.search(r"\bbuy or wait\b", t):
        return "buy_or_wait"
    if re.search(r"\bbuy or sell\b|\bsell or buy\b", t):
        return "buy_or_sell"
    if re.search(
        r"\b(?:should|do|can) i (?:hold|keep)\b|\bhold or sell\b|\bkeep (?:it|holding)\b|\bhold (?:it|on)\b",
        t,
    ):
        return "hold"
    if re.search(r"\bcover\b", t):
        return "buy"  # buying back to close a short
    if re.search(r"\b(?:sell|exit|take profits?|dump)\b", t):
        return "sell"
    if re.search(r"\b(?:buy|enter|entry|ape|long)\b", t):
        return "buy"
    return "analyze"


def position_from_text(text: str) -> Position:
    if _any(_SHORT, text):
        return "short"
    # "my SOL" is ownership only in the original casing; check case-sensitively for tickers.
    if _any(_LONG[:-1], text) or re.search(_LONG[-1], text):
        return "long"
    if _any(_NONE, text):
        return "none"
    return "unknown"


def horizon_for(text: str, timeframe: str | None) -> Horizon:
    t = text.lower()
    if re.search(r"\bscalp", t) or timeframe in ("1m", "5m"):
        return "scalp"
    if re.search(r"\b(?:swing|long[- ]term|weeks?|months?)\b", t) or timeframe == "1d":
        return "swing"
    if timeframe in ("15m", "30m", "1h", "4h") or re.search(r"\b(?:today|intraday)\b", t):
        return "intraday"
    return "unknown"


def leverage_from_text(text: str) -> float | None:
    match = _LEVERAGE.search(text.lower())
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return float(value) if value else None


def trader_context(text: str, history: Sequence[str], timeframe: str | None) -> TraderContext:
    """Position and intent from this message; a position stated earlier in the conversation
    (e.g. "I hold SOL" then "should I sell?") carries over when this message states none."""
    position = position_from_text(text)
    source: Literal["message", "history", "default"] = "message"
    if position == "unknown":
        source = "default"
        for earlier in reversed(history):
            found = position_from_text(earlier)
            if found in ("long", "short"):
                position, source = found, "history"
                break
    return TraderContext(
        position=position,
        requested_action=requested_action(text),
        horizon=horizon_for(text, timeframe),
        explicit_timeframe=timeframe,
        leverage=leverage_from_text(text),
        position_source=source,
    )


# DEX names as DEX Screener's `dexId`s, and exchange names, as a request may mention them.
DEX_NAMES: dict[str, str] = {
    "raydium": "raydium",
    "orca": "orca",
    "meteora": "meteora",
    "uniswap": "uniswap",
    "pancakeswap": "pancakeswap",
    "pumpswap": "pumpswap",
    "pump.fun": "pumpfun",
    "pumpfun": "pumpfun",
    "aerodrome": "aerodrome",
    "sushiswap": "sushiswap",
}
CEX_NAMES = ("kraken", "coinbase", "binance", "bybit", "okx", "bitstamp", "gemini")
_GENERIC_DEX = re.compile(
    r"\b(?:on (?:a |the )?dex|dex (?:pool|price|market)|on-chain pool|liquidity pool)\b"
)
_CEX_PAIR = re.compile(r"\b[a-z0-9]{2,10}/usd[tc]?\b|\bon (?:an |the )?exchange\b|\bcex\b")
_ADDRESS = re.compile(
    r"(?<![0-9A-Za-z])(?:0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44})(?![0-9A-Za-z])"
)


@dataclass(frozen=True)
class VenueRequest:
    kind: VenueKind  # "cex", "dex" or "unknown" (nothing said)
    dex: str | None = None
    pool: str | None = None
    name: str | None = None  # the venue as named, e.g. "Kraken"

    @property
    def explicit(self) -> bool:
        return self.kind != "unknown"


def requested_venue(text: str, token_address: str | None = None) -> VenueRequest:
    """The market the trader asked about, if the request says: a named exchange, a named
    DEX, a generic "on the DEX", or a pool address next to the token's own address."""
    lowered = text.lower()
    pools = [
        a
        for a in _ADDRESS.findall(text)
        if token_address is None or a.lower() != token_address.lower()
    ]
    if token_address is not None and pools:
        return VenueRequest(kind="dex", pool=pools[0])
    for phrase, dex_id in DEX_NAMES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return VenueRequest(kind="dex", dex=dex_id, name=phrase)
    for name in CEX_NAMES:
        if re.search(rf"\b{name}\b", lowered):
            return VenueRequest(kind="cex", name=name.capitalize())
    if _GENERIC_DEX.search(lowered):
        return VenueRequest(kind="dex")
    if _CEX_PAIR.search(lowered):
        return VenueRequest(kind="cex")
    return VenueRequest(kind="unknown")


def market_context(
    profile: CryptoAssetProfile | None,
    timeframe: str | None,
    venue: VenueRequest | None = None,
) -> MarketContext:
    venue = venue or VenueRequest(kind="unknown")
    if profile is None:
        return MarketContext(
            market_type="unknown",
            venue_kind=venue.kind,
            timeframe=timeframe,
            requested_dex=venue.dex,
            requested_pool=venue.pool,
            venue_explicit=venue.explicit,
        )
    policy = profile.market_policy
    dex = (policy is not None and policy.kind == "dex_contract") or (
        venue.kind != "cex"
        and (
            profile.market_type == "dex_spot"
            or (
                profile.identity_basis == "contract"
                and profile.chain in DEX_CHAINS
                and not profile.cex_available
            )
        )
    )
    return MarketContext(
        market_type="dex" if dex else "spot" if profile.market_type != "unknown" else "unknown",
        venue_kind="dex"
        if dex
        else "cex"
        if venue.kind == "cex" or profile.cex_available
        else "unknown",
        venue=venue.name if venue.kind == "cex" else None,
        base_asset=profile.symbol,
        timeframe=timeframe,
        requested_dex=venue.dex,
        requested_pool=venue.pool,
        venue_explicit=venue.explicit,
    )


def identity_from_profile(
    profile: CryptoAssetProfile, confidence: IdentityConfidence, exchange_pair: str | None = None
) -> TradeIdentity:
    return TradeIdentity(
        canonical_id=profile.canonical_id,
        symbol=profile.symbol,
        name=profile.name,
        chain=profile.chain,
        address=profile.address,
        exchange_pair=exchange_pair,
        category=profile.category,
        confidence=confidence,
    )


def build_trade_context(
    profile: CryptoAssetProfile | None,
    confidence: IdentityConfidence,
    trader: TraderContext,
    timeframe: str | None,
    exchange_pair: str | None = None,
    venue: VenueRequest | None = None,
) -> CryptoTradeContext:
    identity = (
        identity_from_profile(profile, confidence, exchange_pair)
        if profile is not None
        else TradeIdentity(canonical_id=None, symbol=None, confidence="ticker_only")
    )
    data = (
        [
            DataAvailability(capability=c.capability, status=c.status, reason=c.reason)
            for c in profile.capabilities
        ]
        if profile
        else []
    )
    return CryptoTradeContext(
        identity=identity,
        market=market_context(profile, timeframe, venue),
        trader=trader,
        profile=profile,
        data=data,
    )
