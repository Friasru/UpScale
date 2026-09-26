from typing import Any

import upscale.services as services
from upscale.agents.base import Agent, AgentContext
from upscale.agents.dex_market import target_mint, target_token
from upscale.schemas import AgentName, AgentResult
from upscale.services.chains import chain_label
from upscale.services.market_data import AssetNotFoundError, InvalidRequestError, MarketDataError
from upscale.services.solana_chain import (
    HolderEntry,
    KnownPool,
    OnchainSafetySnapshot,
    SolanaSafetyService,
)

MAX_HOLDER_LINES = 5
NOT_A_VERDICT = (
    "These are on-chain facts, not a verdict: no single field makes a token safe or a scam."
)


def known_pools(prior: dict[AgentName, AgentResult], mint: str) -> list[KnownPool]:
    """This mint's DEX pools as the DEX agent reported them (only pools it accepted as a
    usable market have their vaults excluded from holder concentration)."""
    dex = prior.get("dex_market")
    if dex is None or dex.status != "ok" or dex.mock or dex.findings.get("mint") != mint:
        return []
    return [
        KnownPool(
            address=c["pair_address"], dex=str(c.get("dex", "")), eligible=bool(c.get("eligible"))
        )
        for c in dex.findings.get("candidates") or []
        if isinstance(c, dict) and isinstance(c.get("pair_address"), str)
    ]


class OnchainSafetyAgent(Agent):
    """Token safety facts for a Solana token identified by its exact mint address.

    Reads the mint account (mint/freeze authorities, token program, Token-2022 extensions,
    supply, decimals) and the largest holders, excluding pool vaults and burned tokens only
    when they can be identified with confidence. All I/O goes through `SolanaSafetyService`;
    it never looks a token up by ticker and never labels a token a scam.
    """

    name = "onchain_safety"
    description = (
        "Solana mint/freeze authorities, token program and extensions, supply, and holder "
        "concentration for an exact mint."
    )
    # Pool addresses from the DEX agent let pool vaults be excluded from holder counts.
    depends_on = ("dex_market",)

    def __init__(self, service: SolanaSafetyService | None = None):
        self._service = service

    @property
    def service(self) -> SolanaSafetyService | None:
        return self._service or services.solana_safety_service

    async def run(self, context: AgentContext) -> AgentResult:
        mint = target_mint(context)
        service = self.service
        base: dict[str, Any] = {
            "provider": service.provider_name if service else None,
            "mint": mint,
            "snapshot": None,
        }
        token = target_token(context)
        if token is not None and token[0] != "solana":
            return self._no_data(
                base, f"no on-chain safety provider is integrated for {chain_label(token[0])} yet"
            )
        if service is None:
            return self._no_data(base, "no Solana RPC provider is configured")
        if mint is None:
            return self._no_data(base, "no Solana mint address was identified")
        base["canonical_id"] = f"solana:{mint}"
        pools = known_pools(context.prior_results, mint)
        try:
            snapshot = await service.get_snapshot(mint, pools)
        except (AssetNotFoundError, InvalidRequestError) as exc:
            return self._no_data(base, str(exc))
        except MarketDataError as exc:
            return AgentResult(
                agent=self.name,
                status="error",
                mock=False,
                summary=f"On-chain safety data is unavailable right now ({exc}).",
                findings=base,
                error=str(exc),
            )
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=_summary(snapshot),
            findings=base | {"snapshot": snapshot.model_dump(mode="json"), "unavailable": None},
            evidence=_evidence(snapshot),
        )

    def _no_data(self, base: dict[str, Any], reason: str) -> AgentResult:
        return AgentResult(
            agent=self.name,
            mock=False,
            summary=f"No on-chain safety data: {reason}.",
            findings=base | {"unavailable": reason},
        )


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1f}%"


def _summary(s: OnchainSafetySnapshot) -> str:
    if s.authorities_available:
        mint = "still enabled" if s.mint_authority_active else "revoked"
        freeze = "enabled" if s.freeze_authority_active else "revoked"
        authorities = f"mint authority {mint}, freeze authority {freeze}"
    else:
        authorities = f"token authorities unavailable ({s.authorities_error})"
    if s.holders_available:
        holders = f"top 10 non-pool holders hold {_pct(s.top10_pct)} of supply"
        if s.top1_pct is not None:
            holders += f", the largest {_pct(s.top1_pct)}"
    else:
        holders = f"holder data unavailable ({s.holders_error})"
    text = f"On-chain ({s.provider}): {authorities}; {holders}."
    if s.holders_available and not s.holder_data_complete:
        text += " Holder data is incomplete."
    return text


def _holder_line(e: HolderEntry) -> str:
    who = e.owner or f"unresolved owner of {e.token_accounts[0]}"
    line = f"{who}: {e.pct_of_supply:.2f}% of supply"
    if e.label:
        line += f" ({e.label})"
    return line


def _evidence(s: OnchainSafetySnapshot) -> list[str]:
    read = f"{s.provider}, read {s.fetched_at:%Y-%m-%d %H:%M} UTC"
    if not s.authorities_available:
        lines = [
            f"Mint {s.mint}: token authorities unavailable ({s.authorities_error}); whether "
            f"supply can be minted or accounts frozen is unknown ({read})."
        ]
    else:
        programs = {"spl_token": "SPL Token", "token_2022": "Token-2022"}
        program = programs.get(s.token_program or "", s.token_program_id or "unknown program")
        lines = [
            f"Mint {s.mint} ({program}): supply {s.supply or 0:,.0f}, {s.decimals} decimals "
            f"({read}).",
            (
                f"Mint authority still enabled ({s.mint_authority}): more supply can be created."
                if s.mint_authority_active
                else "Mint authority revoked: no more supply can be created."
            ),
            (
                f"Freeze authority enabled ({s.freeze_authority}): holders' accounts can be frozen."
                if s.freeze_authority_active
                else "Freeze authority revoked: holders' accounts can't be frozen."
            ),
        ]
    if not s.holders_available:
        lines.append(f"Holder concentration unavailable ({s.holders_error}).")
        lines += [f"Incomplete: {r}." for r in s.incomplete_reasons]
        lines.append(NOT_A_VERDICT)
        return lines
    if s.extensions:
        lines.append(f"Token-2022 extensions: {', '.join(s.extensions)}.")
    bound = "at least " if s.concentration_lower_bound else ""
    source = {
        "full_scan": f"full scan of {s.token_accounts_seen:,} token accounts, by owner wallet",
        "partial_scan": f"partial scan ({s.scan_pages_read} pages, {s.token_accounts_seen:,} "
        "token accounts), by owner wallet",
        "largest_accounts": f"the {s.largest_accounts_seen} largest token accounts only",
    }[s.concentration_source]
    lines.append(
        f"Top 10 non-pool holders control {bound}{_pct(s.top10_pct)}; one non-pool wallet "
        f"controls {bound}{_pct(s.top1_pct)} ({source})."
    )
    lines += [f"Holder: {_holder_line(e)}." for e in s.holders[:MAX_HOLDER_LINES]]
    lines += [f"Excluded: {_holder_line(e)}." for e in s.excluded]
    if s.holder_count is not None:
        bound = "" if s.holder_count_complete else "at least "
        lines.append(
            f"Holders: {bound}{s.holder_count:,} ({bound}{s.meaningful_holder_count or 0:,} "
            "meaningful)."
        )
    lines += [f"Incomplete: {r}." for r in s.incomplete_reasons]
    lines.append(NOT_A_VERDICT)
    return lines
