"""Bounded collection for one Radar target. Every step is capped, failure-isolated and
records what it covered; nothing here interprets the data (see `features`).

Steps (all through Radar's own guarded provider):

1. **Holders**: mint + largest accounts + their owners + a DAS holder scan (page cap
   ``holder_max_pages``), analyzed by `solana_chain.analyze` (pure; no evidence emitted).
   Owners at or above ``large_holder_min_pct`` of supply are stored, plus the current
   balance of every owner that was large in the previous holder snapshot.
2. **Activity**: the pool's new signatures since the stored head cursor (one page by
   default), then at most ``max_tx_per_snapshot`` successful transactions, newest first.
   Anything beyond the caps makes the scan ``PARTIAL``. A transaction that can't be read on
   its own (unsupported version, malformed, unknown to the provider) is skipped with its
   reason and the batch goes on; a provider-wide failure (rate limit, auth, timeout,
   outage, budget) aborts it. The cursor only advances when the batch finished: after an
   abort it stays put, so the next scan lists the unhandled signatures again
   (already-stored ones are filtered out, and storage is idempotent). An exception inside a
   scan (e.g. a `RadarCausalityError`) finishes it ``ABORTED`` with the reason, leaves the
   cursor and ``last_scan_at`` alone and propagates, so no snapshot is saved from that run.

   **Listing continuity.** When an incremental head listing (a prior cursor exists) fills
   its page cap without reaching that cursor, the untraversed range becomes an ``OPEN``
   activity gap (``before`` = the oldest listed signature, ``until`` = the prior cursor),
   stored in the same transaction that advances the cursor, so the cursor never leaves a
   range behind silently. A first scan's bounded history never opens a gap. After the head
   batch, at most ``activity_catchup_max_pages`` pages are spent on open gaps, oldest
   opened first; a page is accounted for like a head batch (sharing its parse budget, so
   the transaction cap is never exceeded) before its gap's ``before`` moves (full page). A
   short page closes the gap only when one confirmation listing (``limit`` 1, no ``until``)
   right after it returns exactly the gap's ``until``: providers answer an unknown
   ``before`` with an empty page, so a short page alone is no proof. A failed catch-up or
   confirmation request never closes a gap. Listing continuity is reported apart from parse coverage: a closed gap
   doesn't make capped transactions parsed.
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
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from upscale.services.market_data import AssetNotFoundError, MarketDataError
from upscale.services.radar.config import RadarSettings
from upscale.services.radar.models import (
    CHAIN_CLOCK_TOLERANCE_S,
    ParsedTx,
    RadarTxUnavailableError,
    RadarUnavailableError,
    SignatureInfo,
    Status,
    Target,
    effective_time,
    ts,
)
from upscale.services.radar.parsing import classify, first_funder, parse_transaction
from upscale.services.radar.provider import RadarProvider
from upscale.services.radar.repository import ListingCoverage, NewGap, RadarRepository, StoredTx
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
    """How one collection step went in this run (``coverage.run``): an execution status,
    not feature availability. AVAILABLE means the step ran or was rightly skipped without
    error (e.g. "already determined (UNAVAILABLE)"); each metric carries its own status."""

    status: Status
    reasons: list[str] = field(default_factory=list)
    requests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons), "requests": self.requests}


def failure_status(exc: MarketDataError) -> Status:
    return exc.status if isinstance(exc, RadarUnavailableError) else "PROVIDER_UNAVAILABLE"


@dataclass
class _ScanTally:
    """What one scan stored: parsed transactions and chain-clock tolerance use."""

    parsed: int = 0
    selected: int = 0  # successful transactions chosen for parsing (head + catch-up)
    lead_txs: int = 0
    lead_flows: int = 0
    lead_max_s: float = 0.0

    def add(self, tx: ParsedTx, stored: StoredTx) -> None:
        if not stored.inserted:
            return
        self.parsed += not tx.failed
        if stored.chain_clock_ahead_s is not None:
            self.lead_txs += 1
            self.lead_flows += stored.flows
            self.lead_max_s = max(self.lead_max_s, stored.chain_clock_ahead_s)

    def note(self) -> list[str]:
        """A note, not a failure: the scan's status doesn't change because of it."""
        if not self.lead_txs:
            return []
        return [
            f"chain clock ahead of local observation clock within {CHAIN_CLOCK_TOLERANCE_S}s "
            f"tolerance: {self.lead_txs} transaction(s), {self.lead_flows} flow(s), max "
            f"{self.lead_max_s:.3f}s ahead (raw block times kept; features use "
            "min(block_time, fetched_at))"
        ]


@dataclass
class _CatchUp:
    """What one scan's catch-up added (its listing coverage goes to `ListingCoverage`)."""

    parsed: int = 0
    skipped: int = 0
    still_open: int = 0  # open gaps after catch-up (not counting one the head opens)
    reasons: list[str] = field(default_factory=list)  # parse-coverage reasons


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
        self._tally = _ScanTally()  # the running scan's

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
        try:
            tx = parse_transaction(
                raw, target.mint, target.pool_address, self.known_pools,
                self.settings.known_intermediaries,
            )  # fmt: skip
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise RadarTxUnavailableError(f"malformed transaction ({exc!r})") from None
        if tx is not None and tx.signature != sig.signature:
            raise RadarTxUnavailableError("the provider returned a different transaction")
        return tx

    def early_cutoff(self, target: Target) -> float | None:
        """End of the early window, when the pool's creation time is known."""
        start = target.pool_created_at
        if start is None:
            cand = self.repo.creator(target.canonical_id, "POOL_CREATOR_CANDIDATE")
            start = (
                effective_time(cand.block_time, cand.determined_at)
                if cand and cand.identity
                else None
            )
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
        prior = target.last_signature
        cov = ListingCoverage(
            head_listing_complete=listing.complete,
            head_reached_cursor=None if prior is None else listing.complete,
        )
        gap: NewGap | None = None
        if prior is not None and not listing.complete:
            why = (
                f"more than {len(sigs)} new pool signatures since the last scan (listing cap "
                f"{s.activity_max_sig_pages} page(s)): the older ones weren't traversed and are "
                "kept as an open activity gap for catch-up"
            )
            gap = NewGap(sigs[-1], prior, why)
            cov.listing_reasons.append(why)
        elif not listing.complete:  # a first scan: bounded history, never a gap
            reasons.append(
                f"more than {len(sigs)} new pool signatures since the last scan "
                f"(listing cap {s.activity_max_sig_pages} page(s)); older ones are skipped"
            )
            reasons.append("first scan: only the most recent pool activity is covered")
        known = self.repo.known_signatures(target.canonical_id, [x.signature for x in sigs])
        fresh = [x for x in sigs if x.signature not in known]
        ok = [x for x in fresh if not x.failed]
        chosen = ok[: s.max_tx_per_snapshot]
        skipped = cov.txs_beyond_cap = len(ok) - len(chosen)
        if skipped:
            reasons.append(
                f"{skipped} successful transactions beyond the per-snapshot cap "
                f"({s.max_tx_per_snapshot}) were not parsed"
            )
        cutoff = self.early_cutoff(target)
        times = [x.block_time for x in sigs if x.block_time is not None]
        window = (min(times) if times else None, max(times) if times else None)
        scan_id = self.repo.record_scan(
            target.canonical_id, "activity", started_at, started_at, self.provider.name,
            "RUNNING", len(sigs), 0, 0, False, (None, None), [],
        )  # fmt: skip
        self._tally = _ScanTally(selected=len(chosen))
        try:
            return await self._activity_batch(target, sigs, listing.complete, fresh, chosen,
                                              skipped, reasons, cov, gap, cutoff, scan_id,
                                              window, start)  # fmt: skip
        except BaseException as exc:  # never leave the scan RUNNING; the cursor stays put
            self._abort_scan(scan_id, exc, window, cov)
            raise

    async def _activity_batch(
        self,
        target: Target,
        sigs: list[SignatureInfo],
        complete: bool,
        fresh: list[SignatureInfo],
        chosen: list[SignatureInfo],
        skipped: int,
        reasons: list[str],
        cov: ListingCoverage,
        gap: NewGap | None,
        cutoff: float | None,
        scan_id: int,
        window: tuple[float | None, float | None],
        start: int,
    ) -> StepResult:
        s = self.settings
        for x in fresh:
            if x.failed:  # no balance change; recorded without a request
                self._store(target, ParsedTx(x.signature, x.slot, x.block_time, None, True, (),
                                             False, 0), scan_id, cutoff)  # fmt: skip
        parsed, unreadable, failure = await self._parse_many(target, chosen, scan_id, cutoff)
        if unreadable:
            skipped += len(unreadable)
            reasons.append(
                f"{len(unreadable)} transaction(s) couldn't be read and were skipped: "
                + "; ".join(f"{sig} ({why})" for sig, why in unreadable)
            )
        reached_oldest = complete and target.last_signature is None
        if failure is not None:
            unhandled = len(chosen) - parsed - len(unreadable)
            skipped += unhandled
            reasons.append(
                f"stopped early: {failure[1]}; {unhandled} selected transaction(s) were left "
                "unprocessed, and the cursor wasn't advanced so the next scan retries them"
            )
            # The head isn't accounted for: no cursor move, no new gap, no catch-up.
            cov.listing_reasons = []
            if still_open := len(self.repo.open_gaps(target.canonical_id)):
                cov.listing_reasons.append(
                    f"{still_open} activity signature gap(s) still open: catch-up didn't run "
                    "because the head batch stopped early"
                )
            status: Status = failure[0] if parsed == 0 and chosen else "PARTIAL"
            reasons += self._tally.note()
            self._finish_scan(scan_id, status, parsed, skipped, reached_oldest, window,
                              reasons, cov)  # fmt: skip
            return StepResult(status, cov.listing_reasons + reasons, self._used() - start)
        catch = await self._catch_up(target, scan_id, cutoff, s.max_tx_per_snapshot - len(chosen), cov)  # fmt: skip
        parsed += catch.parsed
        skipped += catch.skipped
        reasons += catch.reasons
        if still_open := catch.still_open + (gap is not None):
            pages = s.activity_catchup_max_pages
            cov.listing_reasons.append(
                f"{still_open} activity signature gap(s) open: older pool signatures between "
                "scans aren't traversed yet; "
                + (f"catch-up lists at most {pages} page(s) per scan" if pages else
                   "catch-up is off (UPSCALE_RADAR_ACTIVITY_CATCHUP_MAX_PAGES=0)")
            )  # fmt: skip
        status = "AVAILABLE" if not reasons and not cov.listing_reasons else "PARTIAL"
        reasons += self._tally.note()
        # The head batch (and every catch-up page recorded above) is accounted for: the cursor
        # moves to the newest listed signature together with the gap it leaves behind, so
        # nothing older becomes unreachable. A scan that listed nothing new keeps the cursor
        # but still counts as a recent scan (last_scan_at); an aborted one is neither.
        self.repo.finish_activity_scan(
            target.canonical_id, scan_id, status, parsed, skipped, reached_oldest, window,
            reasons, self._t(), cov, sigs[0].signature if sigs else None, gap,
        )  # fmt: skip
        return StepResult(status, cov.listing_reasons + reasons, self._used() - start)

    async def _catch_up(
        self,
        target: Target,
        scan_id: int,
        cutoff: float | None,
        budget: int,
        cov: ListingCoverage,
    ) -> "_CatchUp":
        """Spend at most ``activity_catchup_max_pages`` signature pages on the target's open
        gaps, oldest opened first. Each page is accounted for exactly like a head batch
        (failed transactions stored, at most `budget` successful ones parsed in total, the
        rest counted beyond the cap) before its gap moves or closes. A provider failure
        stops catch-up and leaves that gap where it was.

        A full page moves the gap. A short page (possibly empty) only closes it when one
        confirmation request, ``before`` = the page's oldest signature (or the gap's
        ``before`` for an empty page), ``limit`` 1 and no ``until``, returns exactly the gap's
        ``until``; otherwise a non-empty page still moves the gap and it stays open. So a scan
        makes at most ``activity_catchup_max_pages`` traversal requests plus as many
        confirmation requests (at most one per short page)."""
        s, out = self.settings, _CatchUp()
        gaps = self.repo.open_gaps(target.canonical_id)
        out.still_open = len(gaps)
        pages, size = s.activity_catchup_max_pages, s.signature_page_size
        beyond = 0
        unreadable: list[tuple[str, str]] = []
        failure: str | None = None
        for gap in gaps:
            while pages > 0 and failure is None:
                pages -= 1
                cov.catchup_pages_attempted += 1
                try:
                    page = await self.provider.get_signatures(
                        target.pool_address, limit=size, before=gap.before_signature,
                        until=gap.until_signature,
                    )  # fmt: skip
                except MarketDataError as exc:
                    cov.listing_reasons.append(
                        f"catch-up stopped: {exc}; activity gap {gap.id} wasn't moved, so the "
                        "next scan retries it"
                    )
                    pages = 0
                    break
                known = self.repo.known_signatures(target.canonical_id, [x.signature for x in page])
                fresh = [x for x in page if x.signature not in known]
                ok = [x for x in fresh if not x.failed]
                chosen = ok[: max(budget, 0)]
                budget -= len(chosen)
                beyond += len(ok) - len(chosen)
                cov.txs_beyond_cap += len(ok) - len(chosen)
                out.skipped += len(ok) - len(chosen)
                self._tally.selected += len(chosen)
                for x in fresh:
                    if x.failed:
                        self._store(target, ParsedTx(x.signature, x.slot, x.block_time, None,
                                                     True, (), False, 0), scan_id, cutoff)  # fmt: skip
                parsed, bad, failed = await self._parse_many(target, chosen, scan_id, cutoff)
                out.parsed += parsed
                out.skipped += len(bad)
                unreadable += bad
                if failed is not None:
                    unhandled = len(chosen) - parsed - len(bad)
                    out.skipped += unhandled
                    failure = (
                        f"catch-up stopped early: {failed[1]}; {unhandled} selected "
                        f"transaction(s) were left unprocessed, and activity gap {gap.id} wasn't "
                        "moved so the next scan retries its page"
                    )
                    break
                if len(page) >= size:  # full page: the gap moves and stays open
                    self.repo.record_gap_page(gap, len(page), page[-1], self._t(), scan_id)
                    cov.catchup_pages_completed += 1
                    cov.catchup_signatures_listed += len(page)
                    gap = replace(gap, before_signature=page[-1].signature)
                    continue
                # A short page alone proves nothing (an unknown `before` is answered with an
                # empty page): close only once plain backward paging from the last accounted
                # signature returns exactly the gap's `until`. One confirmation per short page.
                anchor = page[-1].signature if page else gap.before_signature
                cov.boundary_confirmations_attempted += 1
                try:
                    probe = await self.provider.get_signatures(
                        target.pool_address, limit=1, before=anchor
                    )
                except MarketDataError as exc:
                    probe, why = None, f"the confirmation request failed ({exc})"
                    pages = 0  # the provider is failing: no more catch-up this scan
                else:
                    why = "backward paging after it returned " + (
                        probe[0].signature if probe else "no signature"
                    )
                if probe is not None and [x.signature for x in probe] == [gap.until_signature]:
                    self.repo.record_gap_page(gap, len(page), None, self._t(), scan_id)
                    cov.boundary_confirmations_succeeded += 1
                    cov.catchup_pages_completed += 1
                    cov.catchup_signatures_listed += len(page)
                    cov.gaps_closed += 1
                    out.still_open -= 1
                    break
                if page:  # keep the accounted progress; the boundary is retried from there
                    self.repo.record_gap_page(gap, len(page), page[-1], self._t(), scan_id)
                    cov.catchup_pages_completed += 1
                    cov.catchup_signatures_listed += len(page)
                cov.listing_reasons.append(
                    f"activity gap {gap.id} boundary not confirmed: a short catch-up page "
                    f"({len(page)} signature(s)) wasn't followed by {gap.until_signature} in "
                    f"{why}; the gap stays open "
                    + ("from that page's oldest signature" if page else "and unmoved")
                )
                break
            if pages == 0 or failure is not None:
                break
        if beyond:
            out.reasons.append(
                f"{beyond} successful catch-up transactions beyond the per-snapshot cap "
                f"({s.max_tx_per_snapshot}) were not parsed (their signatures were traversed)"
            )
        if unreadable:
            out.reasons.append(
                f"{len(unreadable)} catch-up transaction(s) couldn't be read and were skipped: "
                + "; ".join(f"{sig} ({why})" for sig, why in unreadable)
            )
        if failure is not None:
            out.reasons.append(failure)
        return out

    def _store(self, target: Target, tx: ParsedTx, scan_id: int, cutoff: float | None) -> StoredTx:
        stored = self.repo.record_tx(
            target.canonical_id, tx, self._t(), self.provider.name, scan_id, cutoff
        )
        self._tally.add(tx, stored)
        return stored

    def _abort_scan(
        self,
        scan_id: int,
        exc: BaseException,
        window: tuple[float | None, float | None],
        listing: ListingCoverage | None = None,
    ) -> None:
        """Finish an interrupted scan ``ABORTED`` (terminal) with its reason. The cursor,
        ``last_scan_at``, early status and the gap the head would have opened are left alone,
        so the next run retries; what was stored before the abort stays (storage is
        idempotent), and so does every catch-up page already recorded."""
        parsed = self._tally.parsed
        why = (
            f"aborted: {type(exc).__name__}: {exc}; the cursor wasn't advanced, so the next "
            "scan retries"
        )
        if listing is not None:
            listing.listing_reasons = []  # the scan's gap state never took effect
        self.repo.finish_scan(scan_id, "ABORTED", parsed, max(self._tally.selected - parsed, 0),
                              False, window, [why, *self._tally.note()], self._t(), listing)  # fmt: skip

    async def _parse_many(
        self, target: Target, sigs: list[SignatureInfo], scan_id: int, cutoff: float | None
    ) -> tuple[int, list[tuple[str, str]], tuple[Status, str] | None]:
        """(stored, [(signature, reason) skipped on their own], provider-wide failure).
        One unreadable transaction is skipped; a provider-wide failure stops the batch."""
        parsed = 0
        unreadable: list[tuple[str, str]] = []
        for x in sigs:
            try:
                tx = await self._fetch_parse(target, x)
            except RadarTxUnavailableError as exc:
                unreadable.append((x.signature, str(exc)))
                continue
            except MarketDataError as exc:
                return parsed, unreadable, (failure_status(exc), str(exc))
            if tx is None:
                unreadable.append((x.signature, "the provider has no usable record of it"))
                continue
            self._store(target, tx, scan_id, cutoff)
            parsed += 1
        return parsed, unreadable, None

    def _finish_scan(
        self,
        scan_id: int,
        status: Status,
        parsed: int,
        skipped: int,
        reached_oldest: bool,
        window: tuple[float | None, float | None],
        reasons: list[str],
        listing: ListingCoverage | None = None,
    ) -> None:
        self.repo.finish_scan(
            scan_id, status, parsed, skipped, reached_oldest, window, reasons, self._t(), listing
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
        self._tally = _ScanTally(selected=min(len(listing.signatures), s.effective_early_max_tx))
        try:
            return await self._early_batch(target, listing, scan_id, start)
        except BaseException as exc:  # never leave the scan RUNNING; early status stays unset
            self._abort_scan(scan_id, exc, (None, None))
            raise

    async def _early_batch(
        self, target: Target, listing: _Listing, scan_id: int, start: int
    ) -> StepResult:
        s, pages = self.settings, self.settings.effective_early_sig_pages
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
                first = effective_time(tx.block_time, self._t())
                if cutoff is None and first is not None:
                    cutoff = first + s.early_window_minutes * 60
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
        reasons += self._tally.note()
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
             and cutoff is not None and f.event_time is not None and f.event_time <= cutoff}
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
