"""Radar activity listing continuity: a capped incremental head listing opens a persistent
gap instead of silently skipping older signatures, and bounded catch-up traverses open gaps
(oldest first) without moving any boundary before its signatures are accounted for.
Listing continuity stays separate from parse coverage. Offline (MockTransport); pages are
3 signatures so "a full page" is cheap to build."""

import asyncio
import sqlite3
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest

from tests.radar_fakes import MINT, POOL, T0, Clock, FakeChain, addr, make_service, tx
from upscale.services.radar.collector import Collector, StepResult
from upscale.services.radar.config import DB_SCHEMA_VERSION, load_settings
from upscale.services.radar.features import build_snapshot
from upscale.services.radar.models import (
    RadarCausalityError,
    RadarSchemaError,
    RadarStateError,
    SignatureInfo,
)
from upscale.services.radar.repository import ListingCoverage, NewGap, RadarRepository
from upscale.services.radar.service import RadarService

CID = f"solana:{MINT}"
CREATED = T0 - timedelta(hours=1)
_B58 = "ABCDEFGHJKMNPQRSTUVWXYZ"
GAP_COLS = ("status", "before_signature", "until_signature", "pages_processed",
            "signatures_accounted")  # fmt: skip


def wallet(i: int) -> str:
    return addr("Wa" + _B58[i // len(_B58)] + _B58[i % len(_B58)])


def pool_tx(i: int, err: Any = None, at: Any = None) -> dict[str, Any]:
    """Pool transaction ``sig<i>``: wallet i receives 1,000 tokens from the pool."""
    return tx(f"sig{i}", at or CREATED + timedelta(minutes=i), wallet(i), err=err,
              pre=[(POOL, 100_000)], post=[(POOL, 99_000), (wallet(i), 1_000)])  # fmt: skip


def setup(tmp_path: Any, *idx: int, **kw: Any) -> tuple[RadarService, Clock, FakeChain]:
    clock, chain = Clock(), FakeChain()
    add(chain, *idx)
    opts = {"max_retries": 0, "signature_page_size": 3, "early_max_sig_pages": 0,
            "verify_deployer": False, **kw}  # fmt: skip
    svc = make_service(tmp_path, chain, clock, **opts)
    svc.add_target(MINT, POOL, "pumpswap", CREATED)
    return svc, clock, chain


def add(chain: FakeChain, *idx: int) -> None:
    chain.add(POOL, *(pool_tx(i) for i in idx))


def activity(svc: RadarService, clock: Clock, **overrides: Any) -> StepResult:
    """One activity step (optionally with other settings), a minute after the last one."""
    assert svc.provider is not None
    clock.advance(60)
    target = svc.repo.get_target(CID)
    assert target is not None
    settings = svc.settings.model_copy(update=overrides)
    return asyncio.run(Collector(settings, svc.repo, svc.provider, clock.now).collect_activity(target))  # fmt: skip


def snapshot(svc: RadarService, clock: Clock) -> dict[str, Any]:
    clock.advance(60)
    return asyncio.run(svc.snapshot(CID)).body


def rows(svc: RadarService, sql: str) -> list[Any]:
    conn = sqlite3.connect(svc.settings.db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def gaps(svc: RadarService) -> list[tuple[Any, ...]]:
    return rows(svc, f"SELECT {', '.join(GAP_COLS)} FROM radar_activity_gaps ORDER BY id")


def cursor(svc: RadarService) -> str | None:
    target = svc.repo.get_target(CID)
    assert target is not None
    return target.last_signature


def stored(svc: RadarService) -> list[str]:
    return sorted(r[0] for r in rows(svc, "SELECT signature FROM radar_tx"))


def sigs(*idx: int) -> list[str]:
    return sorted(f"sig{i}" for i in idx)


def listing_calls(chain: FakeChain) -> list[tuple[str | None, str | None]]:
    """(before, until) of every getSignaturesForAddress call since the last clear."""
    return [(p[1].get("before"), p[1].get("until"))
            for m, p in chain.params if m == "getSignaturesForAddress"]  # fmt: skip


def tx_calls(chain: FakeChain) -> list[str]:
    return [p[0] for m, p in chain.params if m == "getTransaction"]


def last_scan(svc: RadarService) -> dict[str, Any]:
    cols = ("status", "signatures_listed", "txs_parsed", "txs_skipped", "txs_beyond_cap",
            "head_listing_complete", "head_reached_cursor", "gap_opened_id",
            "catchup_pages_attempted", "catchup_pages_completed", "catchup_signatures_listed",
            "gaps_closed", "gaps_open")  # fmt: skip
    got = rows(svc, f"SELECT {', '.join(cols)} FROM radar_scans WHERE kind = 'activity' "
                    "ORDER BY id DESC LIMIT 1")  # fmt: skip
    return dict(zip(cols, got[0], strict=True))


def fail_nth(method: str, n: int, how: int | str | BaseException) -> Callable[[str], Any]:
    """Fail the n-th `method` call (1-based) with an HTTP status, "timeout", or by raising."""
    seen: Counter[str] = Counter()

    def fail(m: str) -> int | str | None:
        if m != method:
            return None
        seen[m] += 1
        if seen[m] != n:
            return None
        if isinstance(how, BaseException):
            raise how
        return how

    return fail


def counts(svc: RadarService) -> tuple[int, ...]:
    txs = rows(svc, "SELECT COUNT(*), COUNT(DISTINCT signature) FROM radar_tx")[0]
    flows = rows(svc, "SELECT COUNT(*), COUNT(DISTINCT signature || wallet) "
                      "FROM radar_wallet_flows")[0]  # fmt: skip
    return (*txs, *flows)


def gap_after_head(tmp_path: Any, new: int = 5, **kw: Any) -> tuple[RadarService, Clock, FakeChain]:
    """Cursor sig1, then `new` signatures (>= a page) arrive: a gap opens under sig1."""
    svc, clock, chain = setup(tmp_path, 1, **kw)
    activity(svc, clock)
    add(chain, *range(2, 2 + new))
    activity(svc, clock, activity_catchup_max_pages=0)
    return svc, clock, chain


# --- head listing: gap or no gap ----------------------------------------------------------


def test_short_incremental_page_reaches_the_prior_cursor_and_opens_no_gap(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2)
    activity(svc, clock)
    add(chain, 3, 4)
    r = activity(svc, clock)
    assert r.status == "AVAILABLE" and r.reasons == []
    assert gaps(svc) == [] and cursor(svc) == "sig4"
    scan = last_scan(svc)
    assert (scan["head_listing_complete"], scan["head_reached_cursor"], scan["gap_opened_id"],
            scan["gaps_open"], scan["catchup_pages_attempted"]) == (1, 1, None, 0, 0)  # fmt: skip


@pytest.mark.parametrize("new", [3, 5])  # exactly one full page, and more than a page
def test_full_page_before_the_prior_cursor_opens_a_gap(tmp_path: Any, new: int) -> None:
    svc, clock, chain = gap_after_head(tmp_path, new)
    newest, oldest_listed = f"sig{1 + new}", f"sig{new - 1}"
    assert cursor(svc) == newest  # the head cursor still tracks the newest signature
    assert gaps(svc) == [("OPEN", oldest_listed, "sig1", 0, 0)]
    scan = last_scan(svc)
    assert scan["status"] == "PARTIAL"
    assert (scan["head_listing_complete"], scan["head_reached_cursor"], scan["gaps_open"]) == (
        0,
        0,
        1,
    )
    assert scan["gap_opened_id"] == rows(svc, "SELECT id FROM radar_activity_gaps")[0][0]
    assert rows(svc, "SELECT opened_scan_id, until_block_time FROM radar_activity_gaps") == [
        (rows(svc, "SELECT MAX(id) FROM radar_scans")[0][0],
         (CREATED + timedelta(minutes=1)).timestamp())  # the old cursor's stored block time
    ]  # fmt: skip
    # Catch-up later traverses the untraversed range and closes the gap.
    chain.params.clear()
    activity(svc, clock)
    assert listing_calls(chain) == [(None, newest), (oldest_listed, "sig1"), ("sig2", None)]
    assert gaps(svc) == [("CLOSED", oldest_listed, "sig1", 1, new - 3)]
    assert stored(svc) == sigs(*range(1, 2 + new))


def test_first_scan_bounded_history_never_opens_a_gap(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3, 4, 5)
    r = activity(svc, clock)
    assert "first scan: only the most recent pool activity is covered" in r.reasons
    assert gaps(svc) == [] and cursor(svc) == "sig5" and stored(svc) == sigs(3, 4, 5)
    scan = last_scan(svc)
    assert (scan["head_listing_complete"], scan["head_reached_cursor"], scan["gaps_open"]) == (
        0,
        None,
        0,
    )
    activity(svc, clock)  # nothing new: still no gap, and sig1 / sig2 stay out of scope
    assert gaps(svc) == [] and stored(svc) == sigs(3, 4, 5)


# --- catch-up -----------------------------------------------------------------------------


def test_catch_up_full_page_moves_the_gap_and_a_short_page_closes_it(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 8)  # head sig9..sig7; gap sig6..sig2
    assert gaps(svc) == [("OPEN", "sig7", "sig1", 0, 0)]
    chain.params.clear()
    r = activity(svc, clock, activity_catchup_max_pages=1)
    assert listing_calls(chain) == [(None, "sig9"), ("sig7", "sig1")]
    assert gaps(svc) == [("OPEN", "sig4", "sig1", 1, 3)]  # full page: moved, still open
    assert r.status == "PARTIAL" and any("1 activity signature gap(s) open" in x for x in r.reasons)
    scan = last_scan(svc)
    assert (scan["catchup_pages_attempted"], scan["catchup_pages_completed"],
            scan["catchup_signatures_listed"], scan["gaps_closed"], scan["gaps_open"]) == (1, 1, 3, 0, 1)  # fmt: skip
    chain.params.clear()
    r = activity(svc, clock, activity_catchup_max_pages=1)
    assert listing_calls(chain) == [(None, "sig9"), ("sig4", "sig1"), ("sig2", None)]
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 2, 5)]
    assert r.status == "AVAILABLE" and r.reasons == []
    assert last_scan(svc)["gaps_closed"] == 1 and last_scan(svc)["gaps_open"] == 0
    assert stored(svc) == sigs(*range(1, 10))


def test_multiple_catch_up_pages_in_one_scan_stay_within_bounds(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 11)  # head sig12..sig10; gap sig9..sig2
    chain.params.clear()
    activity(svc, clock, activity_catchup_max_pages=2)
    # One head page plus exactly the two catch-up pages allowed, never more.
    assert listing_calls(chain) == [(None, "sig12"), ("sig10", "sig1"), ("sig7", "sig1")]
    assert gaps(svc) == [("OPEN", "sig4", "sig1", 2, 6)]
    assert len(tx_calls(chain)) == 6 <= svc.settings.max_tx_per_snapshot
    chain.params.clear()
    activity(svc, clock, activity_catchup_max_pages=2)
    assert listing_calls(chain) == [(None, "sig12"), ("sig4", "sig1"), ("sig2", None)]
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 3, 8)]
    assert stored(svc) == sigs(*range(1, 13))


def test_catch_up_shares_the_parse_cap_and_counts_the_rest_beyond_it(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 8, max_tx_per_snapshot=2)
    assert stored(svc) == sigs(1, 8, 9)  # head: sig9 sig8 parsed, sig7 beyond the cap
    chain.params.clear()
    r = activity(svc, clock)
    assert tx_calls(chain) == ["sig6", "sig5"]  # the whole scan reads at most 2 transactions
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 2, 5)]  # every signature traversed...
    scan = last_scan(svc)
    assert (scan["txs_parsed"], scan["txs_skipped"], scan["txs_beyond_cap"]) == (2, 3, 3)
    assert r.status == "PARTIAL"  # ...but not every transaction parsed
    assert r.reasons == ["3 successful catch-up transactions beyond the per-snapshot cap (2) "
                         "were not parsed (their signatures were traversed)"]  # fmt: skip
    assert stored(svc) == sigs(1, 5, 6, 8, 9)


def test_failed_catch_up_transactions_are_stored_without_requests(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1)
    activity(svc, clock)
    chain.add(POOL, pool_tx(2, err={"x": 1}), pool_tx(3, err={"x": 1}))
    add(chain, 4, 5, 6)
    activity(svc, clock, activity_catchup_max_pages=0)
    chain.params.clear()
    assert activity(svc, clock).status == "AVAILABLE"
    assert tx_calls(chain) == []
    assert rows(svc, "SELECT signature, failed FROM radar_tx WHERE signature IN ('sig2', 'sig3') "
                     "ORDER BY signature") == [("sig2", 1), ("sig3", 1)]  # fmt: skip
    assert gaps(svc)[0][0] == "CLOSED"


# --- multiple simultaneous gaps -----------------------------------------------------------


def test_multiple_gaps_coexist_close_oldest_first_and_never_overwrite(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1)
    activity(svc, clock)
    assert cursor(svc) == "sig1"  # A
    add(chain, *range(2, 7))
    activity(svc, clock, activity_catchup_max_pages=0)
    assert cursor(svc) == "sig6"  # B
    g1 = ("OPEN", "sig4", "sig1", 0, 0)
    assert gaps(svc) == [g1]
    add(chain, *range(7, 12))  # more than a page again before G1 closed
    activity(svc, clock, activity_catchup_max_pages=0)
    assert cursor(svc) == "sig11"  # C
    g2 = ("OPEN", "sig9", "sig6", 0, 0)
    assert gaps(svc) == [g1, g2]  # both kept; G1 untouched by G2
    assert last_scan(svc)["gaps_open"] == 2

    add(chain, 12, 13)  # the head keeps collecting while gaps are open
    chain.params.clear()
    r = activity(svc, clock, activity_catchup_max_pages=1)
    assert listing_calls(chain) == [(None, "sig11"), ("sig4", "sig1"), ("sig2", None)]
    assert cursor(svc) == "sig13"
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 1, 2), g2]
    assert any("1 activity signature gap(s) open" in x for x in r.reasons)

    chain.params.clear()
    r = activity(svc, clock, activity_catchup_max_pages=1)
    assert listing_calls(chain) == [(None, "sig13"), ("sig9", "sig6"), ("sig7", None)]
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 1, 2), ("CLOSED", "sig9", "sig6", 1, 2)]
    assert r.status == "AVAILABLE" and cursor(svc) == "sig13"
    assert stored(svc) == sigs(*range(1, 14))  # nothing between A and the head was lost
    assert counts(svc) == (13, 13, 13, 13)


def test_an_old_gap_with_many_pages_is_not_starved_by_a_newer_one(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 11)  # G1: sig9..sig2 (3 pages)
    add(chain, *range(13, 18))
    activity(svc, clock, activity_catchup_max_pages=0)  # G2: sig14..sig13
    chain.params.clear()
    activity(svc, clock, activity_catchup_max_pages=2)
    # Both pages go to G1 (oldest opened), even though G2 would close in one.
    assert listing_calls(chain)[1:] == [("sig10", "sig1"), ("sig7", "sig1")]
    assert [g[:2] for g in gaps(svc)] == [("OPEN", "sig4"), ("OPEN", "sig15")]
    chain.params.clear()
    activity(svc, clock, activity_catchup_max_pages=2)
    # G1 closes on its last page; the remaining page moves on to G2 and closes it too.
    assert listing_calls(chain)[1:] == [("sig4", "sig1"), ("sig2", None), ("sig15", "sig12"),
                                        ("sig13", None)]  # fmt: skip
    assert [g[0] for g in gaps(svc)] == ["CLOSED", "CLOSED"]
    assert stored(svc) == sigs(*range(1, 18))


def test_gap_rows_are_final_once_closed(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    [gap] = svc.repo.open_gaps(CID)
    activity(svc, clock)
    assert gaps(svc)[0][0] == "CLOSED"
    with pytest.raises(RadarStateError):  # a page is never applied twice
        svc.repo.record_gap_page(gap, 2, None, clock.now().timestamp(), 1)
    conn = sqlite3.connect(svc.settings.db_path)
    with pytest.raises(sqlite3.DatabaseError, match="final"), conn:
        conn.execute("UPDATE radar_activity_gaps SET status = 'OPEN', closed_at = NULL, "
                     "closed_scan_id = NULL")  # fmt: skip
    with pytest.raises(sqlite3.IntegrityError), conn:  # one gap per former cursor
        conn.execute("INSERT INTO radar_activity_gaps (canonical_id, status, opened_at, "
                     "opened_scan_id, updated_at, before_signature, until_signature, reason) "
                     f"VALUES ('{CID}', 'OPEN', 1, 1, 1, 'x', 'sig1', 'dup')")  # fmt: skip
    conn.close()


# --- catch-up off -------------------------------------------------------------------------


def test_disabled_catch_up_keeps_the_gap_visible(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, activity_catchup_max_pages=0)
    snapshot(svc, clock)
    add(chain, *range(2, 7))
    snapshot(svc, clock)
    chain.params.clear()
    body = snapshot(svc, clock)  # a quiet scan: catch-up is off, so the gap stays open
    assert listing_calls(chain) == [(None, "sig6")]
    sc = body["activity"]["signature_coverage"]
    assert (sc["incremental_complete"], sc["open_gap_count"], sc["catchup_pages_attempted"],
            sc["catchup_max_pages_per_scan"]) == (False, 1, 0, 0)  # fmt: skip
    assert sc["oldest_open_gap_opened_at"] is not None and len(sc["open_gaps"]) == 1
    iw = body["activity"]["interacting_wallets"]
    assert iw["status"] == "PARTIAL" and "1 activity signature gap(s) open" in iw["reason"]
    assert "activity signature gap" in body["since_tracking"]["interacting_wallets"]["reason"]
    assert body["coverage"]["overall"] != "COMPLETE"
    run = body["coverage"]["run"]["activity"]
    assert run["status"] == "PARTIAL"
    assert any("catch-up is off (UPSCALE_RADAR_ACTIVITY_CATCHUP_MAX_PAGES=0)" in x
               for x in run["reasons"])  # fmt: skip


def test_catch_up_pages_setting_is_bounded() -> None:
    assert load_settings({}).activity_catchup_max_pages == 2
    for raw, want in (("0", 0), ("10", 10)):
        env = {"UPSCALE_RADAR_ACTIVITY_CATCHUP_MAX_PAGES": raw}
        assert load_settings(env).activity_catchup_max_pages == want
    for raw in ("-1", "11"):
        with pytest.raises(ValueError):
            load_settings({"UPSCALE_RADAR_ACTIVITY_CATCHUP_MAX_PAGES": raw})


# --- failures and retries -----------------------------------------------------------------


@pytest.mark.parametrize("how", [429, 503, "timeout"])
def test_catch_up_listing_failure_leaves_the_gap_boundary_unchanged(
    tmp_path: Any, how: int | str
) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    add(chain, 7)
    chain.fail = fail_nth("getSignaturesForAddress", 2, how)  # the catch-up request
    r = activity(svc, clock)
    assert gaps(svc) == [("OPEN", "sig4", "sig1", 0, 0)]  # not moved, not closed
    assert cursor(svc) == "sig7"  # the head batch was accounted for, so it advanced
    assert r.status == "PARTIAL" and any(x.startswith("catch-up stopped: ") for x in r.reasons)
    scan = last_scan(svc)
    assert (
        scan["catchup_pages_attempted"],
        scan["catchup_pages_completed"],
        scan["gaps_open"],
    ) == (1, 0, 1)
    if how == 429:  # Radar's own cooldown: the next run can't even list the head
        clock.advance(60)
        assert activity(svc, clock).status == "PROVIDER_UNAVAILABLE"
        assert gaps(svc) == [("OPEN", "sig4", "sig1", 0, 0)] and cursor(svc) == "sig7"
    chain.fail = None
    clock.advance(svc.settings.max_cooldown_seconds)
    assert activity(svc, clock).status == "AVAILABLE"
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 1, 2)]
    assert stored(svc) == sigs(*range(1, 8)) and counts(svc) == (7, 7, 7, 7)


def test_catch_up_parse_failure_retries_the_page_without_duplicates(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    chain.fail = fail_nth("getTransaction", 2, 503)  # sig3 read, sig2 fails
    r = activity(svc, clock)
    assert gaps(svc) == [("OPEN", "sig4", "sig1", 0, 0)]
    assert r.reasons[-1].startswith("catch-up stopped early: ")
    assert stored(svc) == sigs(1, 3, 4, 5, 6)
    chain.fail = None
    chain.params.clear()
    activity(svc, clock)
    assert listing_calls(chain)[1:] == [("sig4", "sig1"), ("sig2", None)]  # same page again
    assert tx_calls(chain) == ["sig2"]  # already-stored sig3 isn't read or stored twice
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 1, 2)]
    assert counts(svc) == (6, 6, 6, 6)
    assert rows(svc, "SELECT COUNT(*) FROM radar_wallet_entries")[0][0] == 6


def test_head_failure_moves_nothing_and_skips_catch_up(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    add(chain, 7, 8)
    chain.fail = fail_nth("getTransaction", 2, 401)  # sig8 read, sig7 fails
    chain.params.clear()
    r = activity(svc, clock)
    assert listing_calls(chain) == [(None, "sig6")]  # no catch-up request after the failure
    assert cursor(svc) == "sig6" and gaps(svc) == [("OPEN", "sig4", "sig1", 0, 0)]
    assert any("catch-up didn't run because the head batch stopped early" in x for x in r.reasons)
    chain.fail = None
    activity(svc, clock)
    assert cursor(svc) == "sig8" and gaps(svc)[0][0] == "CLOSED"
    assert counts(svc) == (8, 8, 8, 8)


def test_head_listing_failure_records_nothing(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    before = rows(svc, "SELECT COUNT(*) FROM radar_scans")
    chain.fail = fail_nth("getSignaturesForAddress", 1, 503)
    assert activity(svc, clock).status == "PROVIDER_UNAVAILABLE"
    assert rows(svc, "SELECT COUNT(*) FROM radar_scans") == before
    assert cursor(svc) == "sig6" and gaps(svc) == [("OPEN", "sig4", "sig1", 0, 0)]


def test_causality_error_in_catch_up_aborts_and_stays_retryable(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1)
    activity(svc, clock)
    add(chain, 2)
    chain.add(POOL, pool_tx(3, at=T0 + timedelta(hours=1)))  # far ahead of the local clock
    add(chain, 4, 5, 6)
    activity(svc, clock, activity_catchup_max_pages=0)
    before = svc.repo.get_target(CID)
    add(chain, 7)
    with pytest.raises(RadarCausalityError):
        activity(svc, clock)
    assert last_scan(svc)["status"] == "ABORTED"
    after = svc.repo.get_target(CID)
    assert after is not None and before is not None
    # Aborted: the head cursor, last_scan_at and the gap all stay where they were.
    assert (after.last_signature, after.last_scan_at) == (
        before.last_signature,
        before.last_scan_at,
    )
    assert gaps(svc) == [("OPEN", "sig4", "sig1", 0, 0)]
    chain.tx_override = {"sig3": pool_tx(3)}  # the provider now reports a sane block time
    assert activity(svc, clock).status == "AVAILABLE"
    assert cursor(svc) == "sig7" and gaps(svc)[0][0] == "CLOSED"
    assert stored(svc) == sigs(*range(1, 8)) and counts(svc) == (7, 7, 7, 7)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, asyncio.CancelledError])
def test_interrupted_catch_up_aborts_and_stays_retryable(
    tmp_path: Any, interrupt: type[BaseException]
) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    add(chain, 7)
    chain.fail = fail_nth("getSignaturesForAddress", 2, interrupt())
    with pytest.raises(interrupt):
        activity(svc, clock)
    assert last_scan(svc)["status"] == "ABORTED"
    assert cursor(svc) == "sig6" and gaps(svc) == [("OPEN", "sig4", "sig1", 0, 0)]
    assert rows(svc, "SELECT COUNT(*) FROM radar_scans WHERE status = 'RUNNING'") == [(0,)]
    chain.fail = None
    assert activity(svc, clock).status == "AVAILABLE"
    assert cursor(svc) == "sig7" and gaps(svc)[0][0] == "CLOSED"
    assert counts(svc) == (7, 7, 7, 7)


# --- listing vs parse coverage in snapshots -----------------------------------------------


def test_parse_cap_alone_is_partial_with_complete_listing(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, signature_page_size=1000, max_tx_per_snapshot=2)
    snapshot(svc, clock)
    add(chain, *range(2, 8))  # previous cursor reached; 6 successful, only 2 parsed
    a = snapshot(svc, clock)["activity"]
    assert gaps(svc) == []
    sc, pc = a["signature_coverage"], a["parse_coverage"]
    assert (sc["incremental_complete"], sc["head_reached_prior_cursor"], sc["open_gap_count"]) == (
        True,
        True,
        0,
    )
    assert (pc["complete"], pc["transactions_beyond_cap"], pc["max_tx_per_snapshot"]) == (
        False,
        4,
        2,
    )
    iw = a["interacting_wallets"]
    assert iw["status"] == "PARTIAL" and "beyond the per-snapshot cap" in iw["reason"]
    assert "gap" not in iw["reason"]


def test_closing_a_gap_drops_its_reason_but_keeps_the_parse_cap_reason(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, max_tx_per_snapshot=2)
    snapshot(svc, clock)
    add(chain, *range(2, 10))  # head sig9..sig7; gap sig6..sig2
    second = snapshot(svc, clock)
    a2 = second["activity"]
    assert a2["signature_coverage"]["open_gap_count"] == 1
    assert a2["signature_coverage"]["head_reached_prior_cursor"] is False
    assert a2["signature_coverage"]["gaps_opened"] == 1
    assert "1 activity signature gap(s) open" in a2["interacting_wallets"]["reason"]
    third = snapshot(svc, clock)  # catch-up (2 pages) closes the gap within one scan
    a3 = third["activity"]
    sc, pc = a3["signature_coverage"], a3["parse_coverage"]
    assert (sc["incremental_complete"], sc["open_gap_count"], sc["gaps_opened"], sc["gaps_closed"],
            sc["catchup_pages_attempted"], sc["catchup_pages_processed"],
            sc["catchup_signatures_listed"], sc["reasons"]) == (True, 0, 0, 1, 2, 2, 5, [])  # fmt: skip
    assert a3["signatures_listed"] == 5
    assert (pc["complete"], pc["transactions_beyond_cap"]) == (False, 3)
    iw = a3["interacting_wallets"]
    assert iw["status"] == "PARTIAL"
    assert "gap" not in iw["reason"].replace("catch-up", "")  # the gap reason is gone...
    assert "beyond the per-snapshot cap" in iw["reason"]  # ...the parse cap reason is not
    assert third["coverage"]["overall"] != "COMPLETE"

    # Anti-lookahead: re-built at the second snapshot's time, the gap is still open.
    again = svc.build(CID, _at(second))
    assert again["activity"]["signature_coverage"] == a2["signature_coverage"]
    inp = svc.repo.load_inputs(CID, _ts(second))
    assert [g.closed_at for g in inp.gaps] == [None]
    assert svc.repo.load_inputs(CID, _ts(second) - 120).gaps == []  # before it opened


def _at(body: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(body["observed_at"])


def _ts(body: dict[str, Any]) -> float:
    return _at(body).timestamp()


F = 1_790_000_000.0


def test_an_open_gap_forces_activity_partial_from_gap_state_alone(tmp_path: Any) -> None:
    repo = RadarRepository(tmp_path / "radar.sqlite3")
    repo.upsert_target(CID, MINT, POOL, "pumpswap", None, None, "manual", F - 100)
    scan = repo.record_scan(CID, "activity", F - 10, F - 10, "x", "RUNNING", 3, 0, 0, False,
                            (None, None), [])  # fmt: skip
    gap = NewGap(SignatureInfo("old", 1, F - 50, False), "cursor", "why")
    repo.finish_activity_scan(CID, scan, "AVAILABLE", 0, 0, False, (None, None), [], F - 5,
                              ListingCoverage(False, False), "newest", gap)  # fmt: skip
    target = repo.get_target(CID)
    assert target is not None and target.last_signature == "newest"
    settings = load_settings({})

    def body(as_of: float) -> dict[str, Any]:
        return build_snapshot(target, repo.load_inputs(CID, as_of), settings, "x")

    open_ = body(F)["activity"]
    assert open_["interacting_wallets"]["status"] == "PARTIAL"
    assert open_["signature_coverage"]["incremental_complete"] is False
    [row] = repo.open_gaps(CID)
    repo.record_gap_page(row, 0, None, F + 10, scan)
    assert body(F)["activity"] == open_  # closed later: unchanged as of F
    closed = body(F + 20)["activity"]
    assert closed["interacting_wallets"] == {"status": "AVAILABLE", "value": 0,
                                             "lower_bound": False, "reason": None}  # fmt: skip
    assert closed["signature_coverage"]["incremental_complete"] is True


# --- schema -------------------------------------------------------------------------------


def test_a_version_2_radar_database_is_refused_and_left_untouched(tmp_path: Any) -> None:
    path = tmp_path / "radar.sqlite3"
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute("CREATE TABLE radar_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO radar_meta VALUES ('schema_version', '2')")
        conn.execute("CREATE TABLE radar_targets (canonical_id TEXT PRIMARY KEY)")
    before = conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    conn.close()
    assert DB_SCHEMA_VERSION == 3
    with pytest.raises(RadarSchemaError, match="schema version 2.*start a fresh Radar database"):
        RadarRepository(path).open_gaps(CID)
    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall() == before
    conn.close()


# --- gap closure proof (boundary confirmation) ----------------------------------------------
# Live Helius answers a syntactically valid but unknown `before` with HTTP 200 and `[]`, so
# a short / empty catch-up page alone never closes a gap: one confirmation listing (before =
# the last accounted signature, limit 1, no until) must return exactly the gap's `until`.


def confirm_with(chain: FakeChain, answer: Callable[[str], list[str]]) -> None:
    """Answer confirmation listings (limit 1, no until) with `answer(before)`."""
    real = chain.result

    def result(method: str, params: Any) -> Any:
        opts = params[1] if method == "getSignaturesForAddress" else {}
        if method == "getSignaturesForAddress" and "until" not in opts and opts["limit"] == 1:
            return [{"signature": x, "slot": 1, "blockTime": 1, "err": None}
                    for x in answer(opts["before"])]  # fmt: skip
        return real(method, params)

    chain.result = result  # type: ignore[method-assign]


def proofs(svc: RadarService) -> tuple[int, int]:
    scan = rows(svc, "SELECT boundary_confirmations_attempted, boundary_confirmations_succeeded "
                     "FROM radar_scans WHERE kind = 'activity' ORDER BY id DESC LIMIT 1")  # fmt: skip
    return scan[0]  # type: ignore[no-any-return]


UNKNOWN = "5" * 88  # a syntactically valid signature no provider knows


def test_short_page_closes_only_once_until_is_confirmed(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)  # gap sig3..sig2 under sig1
    chain.params.clear()
    r = activity(svc, clock)
    # Traversal page [sig3, sig2] is short; plain backward paging after sig2 gives sig1.
    assert listing_calls(chain) == [(None, "sig6"), ("sig4", "sig1"), ("sig2", None)]
    assert [p[1]["limit"] for m, p in chain.params if m == "getSignaturesForAddress"][-1] == 1
    assert gaps(svc) == [("CLOSED", "sig4", "sig1", 1, 2)]
    assert proofs(svc) == (1, 1) and r.status == "AVAILABLE"


def test_helius_unknown_before_empty_page_never_closes_a_gap(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    conn = sqlite3.connect(svc.settings.db_path)
    with conn:  # the boundary is now unknown to the provider (pruned, lagging, other backend)
        conn.execute("UPDATE radar_activity_gaps SET before_signature = ?", (UNKNOWN,))
    conn.close()
    assert asyncio.run(svc.provider.get_signatures(POOL, limit=3, before=UNKNOWN,  # type: ignore[union-attr]
                                                   until="sig1")) == []  # fmt: skip
    for _ in range(3):  # however often it's retried, an empty page alone proves nothing
        chain.params.clear()
        r = activity(svc, clock)
        assert listing_calls(chain) == [(None, "sig6"), (UNKNOWN, "sig1"), (UNKNOWN, None)]
        assert gaps(svc) == [("OPEN", UNKNOWN, "sig1", 0, 0)]  # unmoved, still open
        assert proofs(svc) == (1, 0) and r.status == "PARTIAL"
        assert any(x.startswith("activity gap 1 boundary not confirmed: a short catch-up page "
                                "(0 signature(s))") and x.endswith("and unmoved")
                   for x in r.reasons)  # fmt: skip
    scan = last_scan(svc)
    assert (scan["catchup_pages_completed"], scan["gaps_closed"], scan["gaps_open"]) == (0, 0, 1)
    body = snapshot(svc, clock)
    sc = body["activity"]["signature_coverage"]
    assert (sc["incremental_complete"], sc["open_gap_count"]) == (False, 1)
    # No earlier snapshot: the window holds all four scans, each one unproven attempt.
    assert (sc["boundary_confirmations_attempted"], sc["boundary_confirmations_succeeded"]) == (
        4,
        0,
    )
    assert body["activity"]["interacting_wallets"]["status"] == "PARTIAL"
    assert body["coverage"]["overall"] != "COMPLETE"


@pytest.mark.parametrize("answer", [["sig99"], []], ids=["another-signature", "empty"])
def test_unconfirmed_non_empty_short_page_keeps_progress_and_the_gap_open(
    tmp_path: Any, answer: list[str]
) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    real = chain.result
    confirm_with(chain, lambda before: answer)
    r = activity(svc, clock)
    assert gaps(svc) == [("OPEN", "sig2", "sig1", 1, 2)]  # moved to the page's oldest, open
    assert stored(svc) == sigs(*range(1, 7))  # the page was accounted for
    assert proofs(svc) == (1, 0) and r.status == "PARTIAL"
    why = next(x for x in r.reasons if "boundary not confirmed" in x)
    assert "(2 signature(s))" in why and why.endswith("from that page's oldest signature")
    assert ("returned sig99" if answer else "returned no signature") in why
    # Retried from the new boundary: an empty traversal page, then a real proof.
    chain.result = real  # type: ignore[method-assign]
    chain.params.clear()
    assert activity(svc, clock).status == "AVAILABLE"
    assert listing_calls(chain) == [(None, "sig6"), ("sig2", "sig1"), ("sig2", None)]
    assert gaps(svc) == [("CLOSED", "sig2", "sig1", 2, 2)] and proofs(svc) == (1, 1)
    assert counts(svc) == (6, 6, 6, 6)


@pytest.mark.parametrize("how", [503, 429, "timeout"])
def test_failed_confirmation_never_closes_and_keeps_safe_progress(
    tmp_path: Any, how: int | str
) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 5)
    chain.fail = fail_nth("getSignaturesForAddress", 3, how)  # head, traversal, confirmation
    r = activity(svc, clock)
    assert gaps(svc) == [("OPEN", "sig2", "sig1", 1, 2)]  # page progress kept, never closed
    assert proofs(svc) == (1, 0) and r.status == "PARTIAL"
    assert any("boundary not confirmed" in x and "the confirmation request failed" in x
               for x in r.reasons)  # fmt: skip
    assert cursor(svc) == "sig6"
    if how == 429:  # Radar's cooldown: the next run lists nothing at all
        assert activity(svc, clock).status == "PROVIDER_UNAVAILABLE"
        assert gaps(svc) == [("OPEN", "sig2", "sig1", 1, 2)]
    chain.fail = None
    clock.advance(svc.settings.max_cooldown_seconds)
    assert activity(svc, clock).status == "AVAILABLE"
    assert gaps(svc) == [("CLOSED", "sig2", "sig1", 2, 2)]
    assert counts(svc) == (6, 6, 6, 6)


def test_failed_confirmation_after_an_empty_page_leaves_the_gap_untouched(tmp_path: Any) -> None:
    svc, clock, chain = gap_after_head(tmp_path, 3)  # exactly one head page: empty gap
    chain.fail = fail_nth("getSignaturesForAddress", 3, 503)
    activity(svc, clock)
    assert gaps(svc) == [("OPEN", "sig2", "sig1", 0, 0)] and proofs(svc) == (1, 0)
    chain.fail = None
    activity(svc, clock)
    assert gaps(svc) == [("CLOSED", "sig2", "sig1", 1, 0)] and proofs(svc) == (1, 1)


def test_confirmations_are_bounded_by_catch_up_pages(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1)
    activity(svc, clock)
    for start in (2, 7, 12):  # three gaps
        add(chain, *range(start, start + 5))
        activity(svc, clock, activity_catchup_max_pages=0)
    conn = sqlite3.connect(svc.settings.db_path)
    with conn:  # pathological: every boundary unknown, every page short and unprovable
        conn.execute(
            "UPDATE radar_activity_gaps SET before_signature = 'x' || id || ?", (UNKNOWN[3:],)
        )
    conn.close()
    for n in (2, 3):
        chain.params.clear()
        activity(svc, clock, activity_catchup_max_pages=n)
        calls = listing_calls(chain)
        assert len(calls) == 1 + n + n  # head + N traversal + N confirmation, never more
        assert sum(until is None for _, until in calls[1:]) == n == proofs(svc)[0]
        assert [g[0] for g in gaps(svc)] == ["OPEN"] * 3
