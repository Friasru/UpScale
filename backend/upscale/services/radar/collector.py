"""Bounded collection for one Radar target. Every step is capped, failure-isolated and
records what it covered; nothing here interprets the data (see `features`).

Steps (all through Radar's own guarded provider):

1. **Holders**: mint + largest accounts + their owners + a DAS holder scan (page cap
   ``holder_max_pages``), analyzed by `solana_chain.analyze` (pure; no evidence emitted).
   Owners at or above ``large_holder_min_pct`` of supply are stored, plus the current
   balance of every owner that was large in the previous holder snapshot.
2. **Activity**: the pool's new signatures since the stored cursor (one page by default),
   then at most ``max_tx_per_snapshot`` successful transactions, newest first. Anything
   beyond the caps makes the scan ``PARTIAL``.
3. **Early history** (once per target): page the pool's signatures back to its first
   transaction within ``early_max_sig_pages``. If reached, the first ``early_max_tx``
   successful transactions are parsed and the oldest one's fee payer is stored as the
   ``POOL_CREATOR_CANDIDATE`` (never as the token deployer). If not, early activity and
   the pool creator candidate are ``UNAVAILABLE``.
4. **Token deployer** (once per target, ``verify_deployer``): page the mint's signatures
   back to its first transaction within ``deployer_max_sig_pages``; only if that
   transaction initializes this exact mint is its fee payer stored as a ``VERIFIED``
   ``TOKEN_DEPLOYER``.
5. **Wallet profiles** (OFF by default: ``wallet_age`` / ``first_funder``).
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from upscale.services.market_data import AssetNotFoundError, MarketDataError
from upscale.services.radar.config import RadarSettings
from upscale.services.radar.models import (
    ParsedTx,
    RadarUnavailableError,
    SignatureInfo,
    Status,
    Target,
    ts,
)
from upscale.services.radar.parsing import classify, first_funder, parse_transaction
from upscale.services.radar.provider import RadarProvider
from upscale.services.radar.repository import RadarRepository
from upscale.services.solana_chain import (
    SYSTEM_PROGRAM,
    AccountRecord,
    ChainData,
    HolderAnalysisConfig,
    KnownPool,
    MintInfo,
    aggregate_owners,
    analyze,
    merge_accounts,
)


@dataclass
class StepResult:
    status: Status
    reasons: list[str] = field(default_factory=list)
    requests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons), "requests": self.requests}


def failure_status(exc: MarketDataError) -> Status:
    return exc.status if isinstance(exc, RadarUnavailableError) else "PROVIDER_UNAVAILABLE"


@dataclass
class _Listing:
    signatures: list[SignatureInfo]
    pages: int
    complete: bool  # a short page ended the listing (reached `until` or the oldest)


class Collector:
    def __init__(
        self,
        settings: RadarSettings,
        repo: RadarRepository,
        provider: RadarProvider,
        now: Callable[[], datetime],
    ):
        self.settings = settings
        self.repo = repo
        self.provider = provider
        self._now = now
        # Every Radar target's pool: their vault owners are pools, never wallets.
        self.known_pools = frozenset(t.pool_address for t in repo.targets())

    def _t(self) -> float:
        return ts(self._now())

    def _used(self) -> int:
        return self.provider.guard.used_this_run

    async def _list(self, address: str, max_pages: int, until: str | None = None) -> _Listing:
        size = self.settings.signature_page_size
        out: list[SignatureInfo] = []
        before: str | None = None
        for page in range(1, max_pages + 1):
            got = await self.provider.get_signatures(
                address, limit=size, before=before, until=until
            )
            out += got
            if len(got) < size:
                return _Listing(out, page, True)
            before = got[-1].signature
        return _Listing(out, max_pages, False)

    async def _fetch_parse(self, target: Target, sig: SignatureInfo) -> ParsedTx | None:
        raw = await self.provider.get_transaction(sig.signature)
        if raw is None:
            return None
        return parse_transaction(
            raw, target.mint, target.pool_address, self.known_pools,
            self.settings.known_intermediaries,
        )  # fmt: skip

    def early_cutoff(self, target: Target) -> float | None:
        """End of the early window, when the pool's creation time is known."""
        start = target.pool_created_at
        if start is None:
            cand = self.repo.creator(target.canonical_id, "POOL_CREATOR_CANDIDATE")
            start = cand.block_time if cand and cand.identity else None
        return start + self.settings.early_window_minutes * 60 if start is not None else None

    # --- 1. holders ---------------------------------------------------------------------

    async def collect_holders(self, target: Target) -> StepResult:
        p, mint, start = self.provider, target.mint, self._used()
        reasons: list[str] = []
        try:
            info: MintInfo | None = None
            mint_error: str | None = None
            try:
                info = await p.fetch_mint(mint)
            except AssetNotFoundError as exc:
                return StepResult("UNAVAILABLE", [f"not a token mint: {exc}"], self._used() - start)
            except (RadarUnavailableError, MarketDataError) as exc:
                if isinstance(exc, RadarUnavailableError) and exc.status == "NOT_COLLECTED":
                    raise
                mint_error = str(exc)
            largest = await p.fetch_largest_accounts(mint)
            token_accounts = await p.fetch_accounts([a.address for a in largest])
            try:
                scan = await p.scan_token_accounts(mint)
            except MarketDataError as exc:
                if isinstance(exc, RadarUnavailableError) and exc.status == "NOT_COLLECTED":
                    raise
                scan = None
                reasons.append(f"holder scan failed ({exc}); largest accounts only")
            ranked = aggregate_owners(merge_accounts(mint, largest, token_accounts, scan))
            lookups = [o.owner for o in ranked if o.owner][: self.settings.owner_lookups]
            owners = await p.fetch_accounts(lookups) if lookups else {}
            supply_raw = decimals = None
            if info is None:
                supply_raw, decimals = await p.fetch_supply(mint)
        except MarketDataError as exc:
            return StepResult(failure_status(exc), [str(exc)], self._used() - start)

        observed = self._now()
        data = ChainData(
            address=mint, mint=info, mint_error=mint_error, largest=largest,
            token_accounts=token_accounts, scan=scan, owners=owners, supply_raw=supply_raw,
            decimals=decimals, provider=p.name, fetched_at=observed,
        )  # fmt: skip
        pool = KnownPool(address=target.pool_address, dex=target.dex or "unknown", eligible=True)
        snap = analyze(
            data, [pool], HolderAnalysisConfig(owner_lookups=self.settings.owner_lookups)
        )
        total = info.supply_raw if info else supply_raw
        excluded = {e.owner for e in snap.excluded if e.owner}
        amounts = {o.owner: o.amount_raw for o in ranked if o.owner}
        full = snap.concentration_source == "full_scan"

        def pct(amount: int) -> float | None:
            return 100 * amount / total if total else None

        def kind(owner: str) -> str:
            return holder_owner_type(owner, owners.get(owner), self.known_pools | {target.pool_address},
                                     self.settings.known_intermediaries)  # fmt: skip

        balances: list[tuple[str, int | None, float | None, str, str]] = []
        large: set[str] = set()
        if total:
            for owner, amount in sorted(amounts.items()):
                share = pct(amount)
                if owner not in excluded and share is not None:
                    if share >= self.settings.large_holder_min_pct:
                        balances.append((owner, amount, share, "LARGE", kind(owner)))
                        large.add(owner)
            for owner in sorted(self.repo.latest_large_owners(target.canonical_id) - large):
                if owner in amounts:
                    balances.append(
                        (owner, amounts[owner], pct(amounts[owner]), "PRIOR_LARGE", kind(owner))
                    )
                elif full:  # a complete scan without the owner: its balance is zero
                    balances.append((owner, 0, 0.0, "PRIOR_LARGE", kind(owner)))
                else:
                    balances.append((owner, None, None, "PRIOR_LARGE_UNMEASURED", kind(owner)))
        self.repo.record_holder_snapshot(
            target.canonical_id, ts(observed), p.name, snap.concentration_source,
            snap.holder_count, snap.holder_count_complete, snap.top1_pct, snap.top10_pct,
            snap.concentration_reliable, snap.concentration_lower_bound, total,
            info.decimals if info else decimals, reasons + snap.incomplete_reasons, balances,
        )  # fmt: skip
        status: Status = "AVAILABLE" if snap.concentration_authoritative else "PARTIAL"
        return StepResult(status, reasons + snap.incomplete_reasons, self._used() - start)

    # --- 2. activity --------------------------------------------------------------------

    async def collect_activity(self, target: Target) -> StepResult:
        s, start, started_at = self.settings, self._used(), self._t()
        reasons: list[str] = []
        try:
            listing = await self._list(
                target.pool_address, s.activity_max_sig_pages, until=target.last_signature
            )
        except MarketDataError as exc:
            return StepResult(failure_status(exc), [str(exc)], self._used() - start)
        sigs = listing.signatures
        if not listing.complete:
            reasons.append(
                f"more than {len(sigs)} new pool signatures since the last scan "
                f"(listing cap {s.activity_max_sig_pages} page(s)); older ones are skipped"
            )
        if target.last_signature is None and not listing.complete:
            reasons.append("first scan: only the most recent pool activity is covered")
        known = self.repo.known_signatures(target.canonical_id, [x.signature for x in sigs])
        fresh = [x for x in sigs if x.signature not in known]
        ok = [x for x in fresh if not x.failed]
        chosen = ok[: s.max_tx_per_snapshot]
        skipped = len(ok) - len(chosen)
        if skipped:
            reasons.append(
                f"{skipped} successful transactions beyond the per-snapshot cap "
                f"({s.max_tx_per_snapshot}) were not parsed"
            )
        cutoff = self.early_cutoff(target)
        scan_id = self.repo.record_scan(
            target.canonical_id, "activity", started_at, started_at, self.provider.name,
            "RUNNING", len(sigs), 0, 0, False, (None, None), [],
        )  # fmt: skip
        for x in fresh:
            if x.failed:  # no balance change; recorded without a request
                self._store(target, ParsedTx(x.signature, x.slot, x.block_time, None, True, (),
                                             False, 0), scan_id, cutoff)  # fmt: skip
        parsed, failure = await self._parse_many(target, chosen, scan_id, cutoff)
        if failure is not None:
            reasons.append(f"stopped early: {failure[1]}")
            skipped += len(chosen) - parsed
        status: Status = "AVAILABLE" if not reasons else "PARTIAL"
        if failure is not None and parsed == 0 and chosen:
            status = failure[0]
        times = [x.block_time for x in sigs if x.block_time is not None]
        self._finish_scan(scan_id, status, parsed, skipped, listing.complete and target.last_signature is None,
                          (min(times) if times else None, max(times) if times else None), reasons)  # fmt: skip
        if sigs and (failure is None or parsed > 0 or not chosen):
            self.repo.advance_cursor(target.canonical_id, sigs[0].signature, self._t())
        return StepResult(status, reasons, self._used() - start)

    def _store(self, target: Target, tx: ParsedTx, scan_id: int, cutoff: float | None) -> int:
        return self.repo.record_tx(
            target.canonical_id, tx, self._t(), self.provider.name, scan_id, cutoff
        )

    async def _parse_many(
        self, target: Target, sigs: list[SignatureInfo], scan_id: int, cutoff: float | None
    ) -> tuple[int, tuple[Status, str] | None]:
        parsed = 0
        for x in sigs:
            try:
                tx = await self._fetch_parse(target, x)
            except MarketDataError as exc:
                return parsed, (failure_status(exc), str(exc))
            if tx is None:
                continue
            self._store(target, tx, scan_id, cutoff)
            parsed += 1
        return parsed, None

    def _finish_scan(
        self,
        scan_id: int,
        status: Status,
        parsed: int,
        skipped: int,
        reached_oldest: bool,
        window: tuple[float | None, float | None],
        reasons: list[str],
    ) -> None:
        self.repo.finish_scan(
            scan_id, status, parsed, skipped, reached_oldest, window, reasons, self._t()
        )

    # --- 3. early history ----------------------------------------------------------------

    async def collect_early(self, target: Target) -> StepResult:
        s, start, started_at = self.settings, self._used(), self._t()
        pages = s.effective_early_sig_pages
        if target.early_status is not None:
            return StepResult("AVAILABLE", ["early history already collected"], 0)
        if pages == 0:
            return StepResult("NOT_COLLECTED", ["early backfill is off"], 0)
        try:
            listing = await self._list(target.pool_address, pages)
        except MarketDataError as exc:
            return StepResult(failure_status(exc), [str(exc)], self._used() - start)
        scan_id = self.repo.record_scan(
            target.canonical_id, "early", started_at, started_at, self.provider.name, "RUNNING",
            len(listing.signatures), 0, 0, listing.complete, (None, None), [],
        )  # fmt: skip
        if not listing.complete:
            why = (
                f"the pool's first transaction is beyond the history cap "
                f"({pages} page(s) of {s.signature_page_size} signatures)"
            )
            self.repo.record_creator(
                target.canonical_id, "POOL_CREATOR_CANDIDATE", "UNAVAILABLE", None,
                "FEE_PAYER_OF_OLDEST_SUCCESSFUL_POOL_TX", None, None, self._t(),
                self.provider.name, {"reason": "HISTORY_CAP_REACHED", "signature_pages": pages},
            )  # fmt: skip
            self._finish_scan(scan_id, "UNAVAILABLE", 0, 0, False, (None, None), [why])
            self.repo.set_early_status(target.canonical_id, "UNAVAILABLE", self._t())
            return StepResult("UNAVAILABLE", [why], self._used() - start)

        oldest_first = [x for x in reversed(listing.signatures) if not x.failed]
        chosen = oldest_first[: s.effective_early_max_tx]
        known = self.repo.known_signatures(target.canonical_id, [x.signature for x in chosen])
        reasons: list[str] = []
        creator: ParsedTx | None = None
        parsed = 0
        failure: tuple[Status, str] | None = None
        cutoff = self.early_cutoff(target)
        for x in chosen:
            if x.signature in known and creator is not None:
                continue
            try:
                tx = await self._fetch_parse(target, x)
            except MarketDataError as exc:
                failure = (failure_status(exc), str(exc))
                break
            if tx is None:
                continue
            if creator is None:
                creator = tx
                if cutoff is None and tx.block_time is not None:
                    cutoff = tx.block_time + s.early_window_minutes * 60
            self._store(target, tx, scan_id, cutoff)
            parsed += 1
        if creator is None:
            status: Status = failure[0] if failure else "UNAVAILABLE"
            why = failure[1] if failure else "no successful pool transaction could be read"
            self._finish_scan(scan_id, status, 0, len(chosen), True, (None, None), [why])
            if failure is None:
                self.repo.set_early_status(target.canonical_id, "UNAVAILABLE", self._t())
            return StepResult(status, [why], self._used() - start)

        self.repo.record_creator(
            target.canonical_id, "POOL_CREATOR_CANDIDATE", "CANDIDATE", creator.fee_payer,
            "FEE_PAYER_OF_OLDEST_SUCCESSFUL_POOL_TX", creator.signature, creator.block_time,
            self._t(), self.provider.name,
            {"pool_address": target.pool_address, "slot": creator.slot,
             "signature_pages": listing.pages,
             "note": "the pool's first successful transaction's fee payer; not proven to be "
                     "the token deployer"},
        )  # fmt: skip
        if cutoff is not None:
            self.repo.mark_early_entries(target.canonical_id, cutoff, self._t())
        skipped = len(oldest_first) - len(chosen)
        if skipped:
            reasons.append(f"{skipped} early transactions beyond the cap ({len(chosen)}) skipped")
        if failure is not None:
            reasons.append(f"stopped early: {failure[1]}")
        status = "AVAILABLE" if not reasons else "PARTIAL"
        times = [x.block_time for x in chosen if x.block_time is not None]
        self._finish_scan(scan_id, status, parsed, skipped, True,
                          (min(times) if times else None, max(times) if times else None), reasons)  # fmt: skip
        if failure is None:
            self.repo.set_early_status(target.canonical_id, status, self._t())
        return StepResult(status, reasons, self._used() - start)

    # --- 4. verified token deployer ------------------------------------------------------

    async def collect_deployer(self, target: Target) -> StepResult:
        s, start = self.settings, self._used()
        if not s.verify_deployer:
            return StepResult("NOT_COLLECTED", ["deployer verification is off"], 0)
        existing = self.repo.creator(target.canonical_id, "TOKEN_DEPLOYER")
        if existing is not None:
            return StepResult("AVAILABLE", [f"already determined ({existing.status})"], 0)
        method = "FEE_PAYER_OF_MINT_INITIALIZATION_TX"
        try:
            listing = await self._list(target.mint, s.deployer_max_sig_pages)
            if not listing.complete:
                why = "the mint's first transaction is beyond the history cap"
                self.repo.record_creator(
                    target.canonical_id, "TOKEN_DEPLOYER", "UNAVAILABLE", None, method, None,
                    None, self._t(), self.provider.name,
                    {"reason": "HISTORY_CAP_REACHED", "signature_pages": listing.pages},
                )  # fmt: skip
                return StepResult("UNAVAILABLE", [why], self._used() - start)
            first = next((x for x in reversed(listing.signatures) if not x.failed), None)
            tx = await self._fetch_parse(target, first) if first else None
        except MarketDataError as exc:
            return StepResult(failure_status(exc), [str(exc)], self._used() - start)
        if first is None or tx is None:
            return StepResult("PROVIDER_UNAVAILABLE", ["the mint's first transaction couldn't be read"],
                              self._used() - start)  # fmt: skip
        if not tx.initializes_mint or tx.failed or tx.fee_payer is None:
            why = "the mint's oldest successful transaction doesn't initialize this mint"
            self.repo.record_creator(
                target.canonical_id, "TOKEN_DEPLOYER", "UNAVAILABLE", None, method, tx.signature,
                tx.block_time, self._t(), self.provider.name,
                {"reason": "OLDEST_MINT_TX_DOES_NOT_INITIALIZE_MINT"},
            )  # fmt: skip
            return StepResult("UNAVAILABLE", [why], self._used() - start)
        self.repo.record_creator(
            target.canonical_id, "TOKEN_DEPLOYER", "VERIFIED", tx.fee_payer, method, tx.signature,
            tx.block_time, self._t(), self.provider.name,
            {"slot": tx.slot, "instruction": "initializeMint", "mint": target.mint,
             "signature_pages": listing.pages},
        )  # fmt: skip
        return StepResult("AVAILABLE", [], self._used() - start)

    # --- 5. optional wallet profiles ----------------------------------------------------

    async def collect_wallet_profiles(self, target: Target) -> StepResult:
        s, start = self.settings, self._used()
        if not (s.wallet_age or s.first_funder):
            return StepResult("NOT_COLLECTED", ["wallet age and first-funder discovery are off"], 0)
        wallets = self._profile_candidates(target)
        done = self.repo.profiled_wallets(wallets)
        todo = [w for w in wallets if w not in done][: s.wallet_profile_max]
        reasons: list[str] = []
        for wallet in todo:
            try:
                listing = await self._list(wallet, s.wallet_profile_max_sig_pages)
                oldest = listing.signatures[-1] if listing.signatures else None
                funder: str | None = None
                funder_status = "NOT_COLLECTED"
                if s.first_funder:
                    funder_status = "UNAVAILABLE"
                    if listing.complete and oldest is not None:
                        raw = await self.provider.get_transaction(oldest.signature)
                        funder = first_funder(raw, wallet) if raw else None
                        funder_status = "VERIFIED_FIRST_TX_FUNDER" if funder else "NOT_FOUND"
            except MarketDataError as exc:
                reasons.append(f"stopped early: {exc}")
                break
            self.repo.record_wallet_profile(
                wallet, self._t(), self.provider.name,
                "EXACT" if listing.complete and oldest else "LOWER_BOUND" if oldest else "UNAVAILABLE",
                oldest.block_time if oldest else None, oldest.signature if oldest else None,
                funder, funder_status,
            )  # fmt: skip
        if len(wallets) - len(done) > len(todo):
            reasons.append(f"profile cap ({s.wallet_profile_max}) reached")
        return StepResult("PARTIAL" if reasons else "AVAILABLE", reasons, self._used() - start)

    def _profile_candidates(self, target: Target) -> list[str]:
        """Large holders not typed as programs / pools / routers, and early proven
        wallets, deterministically ordered."""
        inp = self.repo.load_inputs(target.canonical_id, self._t())
        large = (
            sorted(o for o, b in inp.holders[0].balances.items() if b[3] == "UNKNOWN")
            if inp.holders
            else []
        )
        cutoff = self.early_cutoff(target)
        early = sorted(
            {f.wallet for f in inp.flows if f.participant == "NORMAL_WALLET"
             and cutoff is not None and f.block_time is not None and f.block_time <= cutoff}
        )  # fmt: skip
        return list(dict.fromkeys(large + early))


def holder_owner_type(
    owner: str,
    record: AccountRecord | None,
    known_pools: Collection[str],
    known_intermediaries: Collection[str],
) -> str:
    """A holder's type from positive evidence: a program id, a pool, a router, or an
    account owned by a program (not a keypair wallet). A system-owned account isn't proof
    of a wallet (PDAs can be too), so everything else is UNKNOWN."""
    kind = classify(owner, signers=(), invoked_programs=(), pool_address="",
                    known_pools=known_pools, known_intermediaries=known_intermediaries)  # fmt: skip
    if kind != "UNKNOWN":
        return kind
    if record is not None and record.program_owner != SYSTEM_PROGRAM:
        return "PROGRAM"
    return "UNKNOWN"
