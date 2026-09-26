"""Basic filters: remove only what is clearly unusable. They are not trading thresholds.

Pools are already screened when candidates are built (a token's market must be a priced
pool with liquidity and trades); this final check applies to every candidate, whatever
provider produced it.
"""

from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.models import ScoutCandidate
from upscale.services.scout.normalize import canonical_id, identity_problem


def candidate_problems(candidate: ScoutCandidate, config: ScoutConfig) -> list[str]:
    cfg = config.filters
    problems: list[str] = []
    identity = identity_problem(candidate.chain, candidate.address, config)
    if identity:
        problems.append(identity)
    elif candidate.canonical_id != canonical_id(candidate.chain, candidate.address):
        problems.append("canonical id doesn't match the chain and token address")
    m = candidate.metrics
    if m.price_usd is None:
        problems.append("no USD price reported")
    if m.liquidity_usd is None:
        problems.append("no USD liquidity reported")
    elif m.liquidity_usd < cfg.min_liquidity_usd:
        problems.append(f"liquidity ${m.liquidity_usd:,.0f} is below ${cfg.min_liquidity_usd:,.0f}")
    day = m.window("h24")
    txns = day.txns if day else None
    if txns is None:
        problems.append("no 24h trade counts reported")
    elif txns < cfg.min_txns_h24:
        problems.append(f"{txns} trades in 24h (no real trading activity)")
    return problems
