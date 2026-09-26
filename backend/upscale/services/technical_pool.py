"""Which DEX pool's candles the technical analysis uses.

Two roles, deliberately separate:

* **market pool**: the token's primary market (see `solana_dex`), used for price,
  liquidity, buy/sell flow and pool age.
* **technical pool**: the pool whose OHLCV feeds Technical Analysis. It is the market pool
  unless that pool's candle history is too short (pools only have candles for intervals
  with trades), in which case another already-qualified pool of the SAME token may be used.

Fallback rules (thresholds in `TechnicalPoolConfig`):

1. Never when the trader named a DEX or pool: that market's candles or none.
2. Only pools of the same exact token, accepted as usable markets, with meaningful
   liquidity and real trading activity.
3. Only when the pool's current price agrees with the market pool's within
   `max_price_divergence_pct`; a pool pricing the token differently isn't the same market.
4. Deterministic order: most trades in 24h, then volume, then pair address.
5. Candles are never stitched across pools and gaps are never filled.
"""

from dataclasses import dataclass

from upscale.services.asset_profile import MIN_TECHNICAL_CANDLES
from upscale.services.solana_dex import PoolCandidate, SolanaDexSnapshot


@dataclass(frozen=True)
class TechnicalPoolConfig:
    min_candles: int = MIN_TECHNICAL_CANDLES  # consecutive closed candles for indicators
    min_liquidity_usd: float = 50_000.0  # an alternate pool's liquidity
    min_txns_24h: int = 100  # an alternate pool's trades over 24h
    max_price_divergence_pct: float = 2.0  # alternate vs market pool price
    max_alternates: int = 2  # alternate pools tried (each costs one candle request)

    def __post_init__(self) -> None:
        if self.min_candles < 1 or self.max_alternates < 0:
            raise ValueError("min_candles must be positive and max_alternates non-negative")
        if self.min_liquidity_usd < 0 or self.min_txns_24h < 0:
            raise ValueError("liquidity and activity minimums can't be negative")
        if not 0 < self.max_price_divergence_pct < 100:
            raise ValueError("max_price_divergence_pct must be between 0 and 100")


@dataclass(frozen=True)
class Alternates:
    pools: list[PoolCandidate]  # in the order they should be tried
    rejected: list[str]  # qualified-looking pools that were excluded, and why
    price_rejected: list[str]  # the subset excluded because they price the token differently


def price_divergence_pct(price: float, reference: float) -> float:
    return 100 * abs(price - reference) / reference


def alternate_pools(snapshot: SolanaDexSnapshot, cfg: TechnicalPoolConfig) -> Alternates:
    """Other pools of the same token that may supply candles, best first."""
    pools: list[PoolCandidate] = []
    rejected: list[str] = []
    price_rejected: list[str] = []
    for c in snapshot.candidates:  # every candidate has this token as its base
        if c.pair_address == snapshot.pair_address or not c.eligible:
            continue
        where = f"{c.dex} pool {c.pair_address}"
        if (c.liquidity_usd or 0) < cfg.min_liquidity_usd:
            continue  # too small to stand in for the market; not worth listing
        if (c.txns_24h or 0) < cfg.min_txns_24h:
            rejected.append(f"{where}: {c.txns_24h or 0} trades in 24h (needs {cfg.min_txns_24h})")
            continue
        if c.price_usd is None:
            rejected.append(f"{where}: no USD price to compare")
            continue
        gap = price_divergence_pct(c.price_usd, snapshot.price_usd)
        if gap > cfg.max_price_divergence_pct:
            note = (
                f"{where}: price ${c.price_usd:.6g} differs {gap:.1f}% from the market pool "
                f"(limit {cfg.max_price_divergence_pct:g}%)"
            )
            rejected.append(note)
            price_rejected.append(note)
            continue
        pools.append(c)
    pools.sort(key=lambda c: (-(c.txns_24h or 0), -(c.volume_24h_usd or 0), c.pair_address))
    return Alternates(pools=pools, rejected=rejected, price_rejected=price_rejected)
