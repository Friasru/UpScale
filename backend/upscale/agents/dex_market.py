import upscale.services as services
from upscale.agents.base import Agent, AgentContext
from upscale.formatting import usd, usd_compact
from upscale.schemas import AgentResult
from upscale.services.asset_registry import DEFAULT_REGISTRY
from upscale.services.chains import DEX_CHAINS, chain_label, normalize_address
from upscale.services.market_data import AssetNotFoundError, InvalidRequestError, MarketDataError
from upscale.services.solana_dex import (
    NoUsablePoolError,
    PoolCandidate,
    SolanaDexService,
    SolanaDexSnapshot,
    WindowStats,
)

WINDOW_LABELS = {"m5": "5m", "h1": "1h", "h6": "6h", "h24": "24h"}
NOT_CHECKED = (
    "This is market data only: token authorities and holder concentration come from the "
    "on-chain safety review when it is available."
)


def target_token(context: AgentContext) -> tuple[str, str] | None:
    """(chain, address) of the token to look up: the exact identity if given, else the
    registry's address for the named asset. Never a ticker search."""
    identity = context.asset_identity
    if identity is not None and identity.chain in DEX_CHAINS and identity.address:
        return identity.chain, normalize_address(
            identity.chain, identity.address
        ) or identity.address
    if context.primary_asset:
        entry = DEFAULT_REGISTRY.by_symbol(context.primary_asset)
        if entry is not None and entry.chain in DEX_CHAINS and entry.address:
            return entry.chain, entry.address
    return None


def target_mint(context: AgentContext) -> str | None:
    """The Solana mint to look up (on-chain data is Solana-only for now)."""
    token = target_token(context)
    return token[1] if token is not None and token[0] == "solana" else None


class DexMarketAgent(Agent):
    """Live DEX market data for a token identified by chain + exact contract/mint address.

    Discovers the token's pools, picks the primary market deterministically, and reports
    price, liquidity, trades, volume and price changes exactly as the provider returned
    them. Data comes from the `dex_market` capability; it never looks a token up by ticker.
    """

    name = "dex_market"
    description = (
        "DEX pools for an exact token address (Solana and EVM chains): primary pool, price, "
        "liquidity, buys/sells, volume, price changes and pool age."
    )

    def __init__(self, service: SolanaDexService | None = None):
        self._service = service

    @property
    def service(self) -> SolanaDexService:
        return self._service or services.provider_registry.dex

    async def run(self, context: AgentContext) -> AgentResult:
        provider = self.service.provider_name
        token = target_token(context)
        base: dict[str, object] = {
            "provider": provider,
            "mint": token[1] if token else None,
            "chain": token[0] if token else None,
            "snapshot": None,
            "candidates": [],
        }
        if token is None:
            return AgentResult(
                agent=self.name,
                mock=False,
                summary="No token address was identified, so no DEX data was requested.",
                findings=base | {"unavailable": "no token address"},
            )
        chain, mint = token
        canonical = f"{chain}:{mint}"
        base["canonical_id"] = canonical
        try:
            market = context.trade.market if context.trade else None
            snapshot = await self.service.get_snapshot(
                mint,
                chain,
                dex=market.requested_dex if market else None,
                pool=market.requested_pool if market else None,
            )
        except NoUsablePoolError as exc:
            return AgentResult(
                agent=self.name,
                mock=False,
                summary=f"No usable {chain_label(chain)} DEX market for {canonical}: {exc}.",
                findings=base
                | {
                    "unavailable": str(exc),
                    "candidates": [c.model_dump(mode="json") for c in exc.candidates],
                },
                evidence=[_candidate_line(c) for c in exc.candidates],
            )
        except (AssetNotFoundError, InvalidRequestError) as exc:
            return AgentResult(
                agent=self.name,
                mock=False,
                summary=f"No {chain_label(chain)} DEX market data for {canonical}: {exc}.",
                findings=base | {"unavailable": str(exc)},
            )
        except MarketDataError as exc:
            return AgentResult(
                agent=self.name,
                status="error",
                mock=False,
                summary=f"DEX data is unavailable right now ({exc}).",
                findings=base,
                error=str(exc),
            )
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=_summary(snapshot),
            findings=base
            | {
                "snapshot": snapshot.model_dump(mode="json"),
                "candidates": [c.model_dump(mode="json") for c in snapshot.candidates],
                "unavailable": None,
            },
            evidence=_evidence(snapshot),
        )


def _age(hours: float) -> str:
    if hours < 48:
        return f"{hours:.0f} hours"
    return f"{hours / 24:.0f} days"


def _label(s: SolanaDexSnapshot) -> str:
    return s.symbol or s.canonical_id


def _summary(s: SolanaDexSnapshot) -> str:
    text = (
        f"{_label(s)} on {s.dex} ({s.quote_symbol or 'unknown quote'} pool): "
        f"{usd(s.price_usd)}, liquidity {usd_compact(s.liquidity_usd)}"
    )
    day = s.window("h24")
    if day is not None and day.volume_usd is not None:
        text += f", 24h volume {usd_compact(day.volume_usd)}"
    if s.pool_age_hours is not None:
        text += f", pool age {_age(s.pool_age_hours)}"
    text += f" ({s.provider})."
    if not s.primary_clear:
        text += " The primary market is unclear: several pools compete."
    return text


def _window_line(w: WindowStats) -> str | None:
    parts = []
    if w.buys is not None and w.sells is not None:
        parts.append(f"{w.buys:,} buys / {w.sells:,} sells")
    if w.volume_usd is not None:
        parts.append(f"volume {usd_compact(w.volume_usd)}")
    if w.price_change_pct is not None:
        parts.append(f"price {w.price_change_pct:+.2f}%")
    return f"{WINDOW_LABELS[w.window]}: {', '.join(parts)}." if parts else None


def _candidate_line(c: PoolCandidate) -> str:
    liquidity = usd_compact(c.liquidity_usd) if c.liquidity_usd is not None else "unreported"
    status = "primary" if c.primary else "eligible" if c.eligible else "rejected"
    line = f"Pool {c.pair_address} ({c.dex}, {c.quote_symbol or 'unknown quote'}): liquidity {liquidity}, {status}"
    if c.rejected_because:
        line += f" ({'; '.join(c.rejected_because)})"
    return line + "."


def _evidence(s: SolanaDexSnapshot) -> list[str]:
    native = (
        f" ({s.price_native:.6g} {s.quote_symbol or 'quote'})" if s.price_native is not None else ""
    )
    lines = [
        f"{_label(s)}{f' ({s.name})' if s.name else ''}, mint {s.mint}: {usd(s.price_usd)}"
        f"{native} on {s.dex}, pool {s.pair_address}; liquidity {usd_compact(s.liquidity_usd)} "
        f"({s.provider}, retrieved {s.fetched_at:%Y-%m-%d %H:%M} UTC)."
    ]
    reported = [
        f"{label} {usd_compact(value)}"
        for label, value in (("market cap", s.market_cap_usd), ("FDV", s.fdv_usd))
        if value is not None
    ]
    if reported:
        lines.append(
            f"Reported {' and '.join(reported)} ({s.provider}; supply x pool price, not "
            "used as identity or size proof)."
        )
    if s.pair_created_at is not None and s.pool_age_hours is not None:
        lines.append(
            f"Pool created {s.pair_created_at:%Y-%m-%d %H:%M} UTC ({_age(s.pool_age_hours)} ago)."
        )
    else:
        lines.append("Pool creation time was not reported.")
    lines += [line for w in s.windows if (line := _window_line(w))]
    eligible = sum(c.eligible for c in s.candidates)
    lines.append(
        f"Pools considered: {len(s.candidates)} ({eligible} eligible). The primary pool has "
        "the most USD liquidity among active pools, preferring SOL/USDC/USDT quotes."
    )
    lines += [f"Competing market: {a}." for a in s.ambiguity]
    lines.append(NOT_CHECKED)
    return lines
