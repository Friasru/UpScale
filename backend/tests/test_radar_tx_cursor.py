"""Radar activity scans after the first live run: version-1 transactions, one unreadable
transaction vs a provider-wide failure, and an activity cursor that never moves past
signatures an aborted batch didn't handle. Offline (MockTransport)."""

import asyncio
import sqlite3
from collections import Counter
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from tests.radar_fakes import MINT, POOL, T0, Clock, FakeChain, addr, make_service, tx
from upscale.services.radar.collector import Collector, StepResult
from upscale.services.radar.models import RadarTxUnavailableError, RadarUnavailableError
from upscale.services.radar.parsing import parse_transaction
from upscale.services.radar.provider import RadarHeliusProvider
from upscale.services.radar.service import RadarService

CID = f"solana:{MINT}"
CREATED = T0 - timedelta(hours=1)
WALLETS = [addr("Wa" + c) for c in "ABCDEFGHJK"]
VERSION_ERROR = {"error": {"code": -32015, "message": "Transaction version (2) is not supported "
                           "by the requesting client. Please try the request again with the "
                           "following configuration parameter: \"maxSupportedTransactionVersion\": 2"}}  # fmt: skip


def pool_tx(i: int, version: int | str = 0, err: Any = None) -> dict[str, Any]:
    """Pool transaction ``sig<i>``: wallet i receives 1,000 tokens from the pool."""
    t = tx(f"sig{i}", CREATED + timedelta(minutes=i), WALLETS[i], err=err,
           pre=[(POOL, 100_000)], post=[(POOL, 99_000), (WALLETS[i], 1_000)])  # fmt: skip
    t["version"] = version
    return t


def setup(tmp_path: Any, *idx: int, **kw: Any) -> tuple[RadarService, Clock, FakeChain]:
    clock, chain = Clock(), FakeChain()
    chain.add(POOL, *(pool_tx(i) for i in idx))
    svc = make_service(tmp_path, chain, clock, max_retries=0, **kw)
    svc.add_target(MINT, POOL, "pumpswap", CREATED)
    return svc, clock, chain


def activity(svc: RadarService, clock: Clock) -> StepResult:
    assert svc.provider is not None
    target = svc.repo.get_target(CID)
    assert target is not None
    return asyncio.run(Collector(svc.settings, svc.repo, svc.provider, clock.now).collect_activity(target))  # fmt: skip


def fail_on_call(n: int, how: int | str) -> Callable[[str], int | str | None]:
    """Fail the n-th getTransaction call (1-based) with `how`; every other call succeeds."""
    seen: Counter[str] = Counter()

    def fail(method: str) -> int | str | None:
        if method != "getTransaction":
            return None
        seen[method] += 1
        return how if seen[method] == n else None

    return fail


def cursor(svc: RadarService) -> str | None:
    target = svc.repo.get_target(CID)
    assert target is not None
    return target.last_signature


def rows(svc: RadarService, sql: str) -> list[Any]:
    conn = sqlite3.connect(svc.settings.db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def stored(svc: RadarService) -> list[str]:
    return [r[0] for r in rows(svc, "SELECT signature FROM radar_tx ORDER BY signature")]


def counted(activity: dict[str, Any]) -> tuple[int, int, int]:
    return (activity["signatures_listed"], activity["transactions_parsed"],
            activity["transactions_skipped"])  # fmt: skip


def last_scan(svc: RadarService) -> tuple[str, int, int, int]:
    got = rows(svc, "SELECT status, signatures_listed, txs_parsed, txs_skipped FROM radar_scans "
                    "WHERE kind = 'activity' ORDER BY id DESC LIMIT 1")  # fmt: skip
    return got[0]  # type: ignore[no-any-return]


# --- 1-3: transaction versions --------------------------------------------------------


@pytest.mark.parametrize("version", ["legacy", 0, 1])
def test_legacy_version_0_and_version_1_transactions_parse(version: int | str) -> None:
    parsed = parse_transaction(pool_tx(1, version), MINT, POOL)
    assert parsed is not None and parsed.signature == "sig1"
    assert [(f.wallet, f.direction, f.amount_raw, f.participant, f.counterparty)
            for f in parsed.flows] == [
        (WALLETS[1], "TOKEN_INFLOW", 1_000, "NORMAL_WALLET", "TRACKED_POOL_COUNTERPARTY")
    ]  # fmt: skip


def test_radar_requests_max_supported_transaction_version_1(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    chain.add(POOL, pool_tx(1, version=0), pool_tx(2, version=1))
    r = activity(svc, clock)
    assert r.status == "AVAILABLE" and r.reasons == []
    assert stored(svc) == ["sig1", "sig2"]  # the version-1 transaction is stored too
    sent = [p for m, p in chain.params if m == "getTransaction"]
    assert sent == [
        [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 1,
               "commitment": "confirmed"}]
        for sig in ("sig2", "sig1")
    ]  # fmt: skip


def test_per_transaction_errors_are_told_apart_from_provider_errors(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path)
    assert isinstance(svc.provider, RadarHeliusProvider)
    p = svc.provider
    chain.tx_override = {
        "version": VERSION_ERROR,
        "skipped": {"error": {"code": -32009, "message": "Slot 5 was skipped"}},
        "badsig": {"error": {"code": -32602, "message": "Invalid param: Invalid"}},
        "malformed": ["not", "a", "transaction"],
        "lagging": {"error": {"code": -32004, "message": "Block not available for slot 5"}},
        "internal": {"error": {"code": -32603, "message": "Internal error"}},
    }
    for sig in ("version", "skipped", "badsig", "malformed"):
        with pytest.raises(RadarTxUnavailableError) as err:
            asyncio.run(p.get_transaction(sig))
        assert err.value.status == "UNAVAILABLE"
    for sig in ("lagging", "internal"):
        with pytest.raises(RadarUnavailableError) as other:
            asyncio.run(p.get_transaction(sig))
        assert not isinstance(other.value, RadarTxUnavailableError)
        assert other.value.status == "PROVIDER_UNAVAILABLE"
    assert asyncio.run(p.get_transaction("unknown")) is None
    assert p.guard.cooldown_until() is None


# --- 4: one unreadable transaction ----------------------------------------------------


def test_one_unreadable_transaction_does_not_stop_the_batch(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3, 4, 5, 6)
    garbled = pool_tx(3)
    garbled["transaction"] = "garbage"
    chain.tx_override = {"sig5": VERSION_ERROR, "sig3": garbled, "sig2": None}
    r = activity(svc, clock)
    assert chain.calls["getTransaction"] == 6  # every bounded signature was requested
    assert stored(svc) == ["sig1", "sig4", "sig6"]
    assert r.status == "PARTIAL"
    [why] = r.reasons
    assert why.startswith("3 transaction(s) couldn't be read and were skipped: ")
    assert "sig5 (" in why and "Transaction version (2)" in why
    assert "sig3 (malformed transaction" in why
    assert "sig2 (the provider has no usable record of it)" in why
    assert last_scan(svc) == ("PARTIAL", 6, 3, 3)
    assert cursor(svc) == "sig6"  # the batch finished: skipped ones are recorded, not lost


# --- 5-6: provider-wide failure -------------------------------------------------------


@pytest.mark.parametrize("how", [401, 429, 503, "timeout", "rpc-internal"])
def test_provider_wide_failure_aborts_the_batch_and_keeps_the_cursor(
    tmp_path: Any, how: int | str
) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3, 4, 5, 6)
    if how == "rpc-internal":
        chain.tx_override = {"sig5": {"error": {"code": -32603, "message": "Internal error"}}}
    else:
        chain.fail = fail_on_call(2, how)
    r = activity(svc, clock)
    assert chain.calls["getTransaction"] == 2  # stopped at the failure, nothing after it
    assert stored(svc) == ["sig6"]
    assert r.status == "PARTIAL"
    assert r.reasons[-1].startswith("stopped early: ")
    assert "5 selected transaction(s) were left unprocessed" in r.reasons[-1]
    assert last_scan(svc) == ("PARTIAL", 6, 1, 5)
    assert cursor(svc) is None  # first scan: the next one lists everything again


def test_aborted_batch_never_moves_the_cursor_past_unprocessed_signatures(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3)
    assert activity(svc, clock).status == "AVAILABLE" and cursor(svc) == "sig3"
    clock.advance(600)
    chain.add(POOL, *(pool_tx(i) for i in (4, 5, 6, 7, 8)))
    chain.fail = fail_on_call(3, 401)  # sig8, sig7 read; sig6 fails; sig5, sig4 never asked
    r = activity(svc, clock)
    assert r.status == "PARTIAL"
    assert cursor(svc) == "sig3"  # not sig8: sig6..sig4 must stay reachable
    assert stored(svc) == ["sig1", "sig2", "sig3", "sig7", "sig8"]
    assert last_scan(svc) == ("PARTIAL", 5, 2, 3)


# --- 7-8: retry recovers, without duplicates ------------------------------------------


def test_retry_recovers_unprocessed_signatures_without_duplicates(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3)
    activity(svc, clock)
    clock.advance(600)
    chain.add(POOL, *(pool_tx(i) for i in (4, 5, 6, 7, 8)))
    chain.fail = fail_on_call(3, 401)
    activity(svc, clock)
    chain.fail = None
    chain.params.clear()
    clock.advance(600)
    r = activity(svc, clock)
    assert r.status == "AVAILABLE"
    sig_calls = [p for m, p in chain.params if m == "getSignaturesForAddress"]
    assert sig_calls[0][1]["until"] == "sig3"  # listed from the unmoved cursor
    # Already-stored sig8 / sig7 are not fetched again; the unhandled ones are recovered.
    assert [p[0] for m, p in chain.params if m == "getTransaction"] == ["sig6", "sig5", "sig4"]
    assert last_scan(svc) == ("AVAILABLE", 5, 3, 0)
    assert cursor(svc) == "sig8"
    every = [f"sig{i}" for i in range(1, 9)]
    assert stored(svc) == every

    def counts() -> tuple[int, ...]:
        txs = rows(svc, "SELECT COUNT(*), COUNT(DISTINCT signature) FROM radar_tx")[0]
        flows = rows(svc, "SELECT COUNT(*), COUNT(DISTINCT signature || wallet) "
                          "FROM radar_wallet_flows")[0]  # fmt: skip
        return (*txs, *flows)

    assert counts() == (8, 8, 8, 8)
    assert rows(svc, "SELECT COUNT(*) FROM radar_wallet_entries")[0][0] == 8
    # A retry that re-reads already-stored transactions (cursor lost) stores nothing new.
    parsed = parse_transaction(pool_tx(7), MINT, POOL)
    assert parsed is not None
    again = svc.repo.record_tx(CID, parsed, clock.now().timestamp(), "Helius", None, None)
    assert not again.inserted and again.flows == 0
    conn = sqlite3.connect(svc.settings.db_path)
    with conn:
        conn.execute("UPDATE radar_targets SET last_signature = NULL")
    conn.close()
    chain.params.clear()
    activity(svc, clock)
    assert [m for m, _ in chain.params].count("getTransaction") == 0  # all known
    assert counts() == (8, 8, 8, 8)
    assert rows(svc, "SELECT COUNT(*) FROM radar_wallet_entries")[0][0] == 8


# --- 9: snapshot counts ----------------------------------------------------------------


def test_snapshot_parsed_and_skipped_counts_stay_truthful(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, max_tx_per_snapshot=4, early_max_sig_pages=0,
                              verify_deployer=False)  # fmt: skip
    chain.add(POOL, pool_tx(1), pool_tx(2), pool_tx(9, err={"InstructionError": [0, "x"]}),
              pool_tx(3), pool_tx(4), pool_tx(5), pool_tx(6))  # fmt: skip
    chain.tx_override = {"sig5": VERSION_ERROR}
    # Newest first: sig6 sig5 sig4 sig3 | cap | sig2 sig1, plus failed sig9 (no request).
    chain.fail = fail_on_call(3, 401)  # sig6 read, sig5 unreadable, sig4 fails: sig3 unasked
    a = asyncio.run(svc.snapshot(CID)).body["activity"]
    assert counted(a) == (7, 1, 5)
    # 1 parsed + 5 skipped (2 beyond the cap, 1 unreadable, 2 unprocessed) + 1 failed.
    assert cursor(svc) is None
    assert a["interacting_wallets"]["status"] == "PARTIAL"
    assert "the cursor wasn't advanced" in a["interacting_wallets"]["reason"]

    chain.fail = None
    clock.advance(3600)
    b = asyncio.run(svc.snapshot(CID)).body["activity"]
    # Listed again (7); sig6 and failed sig9 are known. Fresh 5: sig5 unreadable, sig4 sig3
    # sig2 parsed, sig1 beyond the cap.
    assert counted(b) == (7, 3, 2)
    assert cursor(svc) == "sig6"
    assert stored(svc) == ["sig2", "sig3", "sig4", "sig6", "sig9"]


# --- 10-13: last_scan_at marks finished scans, quiet ones included ----------------------


def scanned_at(svc: RadarService, cid: str = CID) -> float | None:
    target = svc.repo.get_target(cid)
    assert target is not None
    return target.last_scan_at


def test_quiet_scan_updates_last_scan_at_and_keeps_the_cursor(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3)
    activity(svc, clock)
    first = scanned_at(svc)
    assert cursor(svc) == "sig3" and first == clock.now().timestamp()
    clock.advance(600)
    chain.params.clear()
    r = activity(svc, clock)
    assert r.status == "AVAILABLE" and r.reasons == []
    assert [p[1]["until"] for m, p in chain.params if m == "getSignaturesForAddress"] == ["sig3"]
    assert last_scan(svc) == ("AVAILABLE", 0, 0, 0)
    assert cursor(svc) == "sig3"  # nothing new: the cursor stays
    assert scanned_at(svc) == clock.now().timestamp() == first + 600  # type: ignore[operator]


def test_scan_with_new_signatures_updates_last_scan_at_and_cursor(tmp_path: Any) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3)
    activity(svc, clock)
    clock.advance(600)
    chain.add(POOL, pool_tx(4), pool_tx(5))
    assert activity(svc, clock).status == "AVAILABLE"
    assert cursor(svc) == "sig5"
    assert scanned_at(svc) == clock.now().timestamp()


@pytest.mark.parametrize("method", ["getTransaction", "getSignaturesForAddress"])
def test_aborted_scan_is_not_a_recent_scan_and_keeps_the_cursor(tmp_path: Any, method: str) -> None:
    svc, clock, chain = setup(tmp_path, 1, 2, 3)
    activity(svc, clock)
    before = scanned_at(svc)
    clock.advance(600)
    chain.add(POOL, pool_tx(4), pool_tx(5))
    chain.fail = lambda m: 503 if m == method else None
    r = activity(svc, clock)
    assert r.status == "PROVIDER_UNAVAILABLE"
    assert cursor(svc) == "sig3"
    assert scanned_at(svc) == before  # not marked as successfully scanned
    # A first scan that aborts leaves the target never scanned.
    svc2, clock2, chain2 = setup(tmp_path / "fresh", 1, 2, 3)
    chain2.fail = lambda m: 503 if m == method else None
    activity(svc2, clock2)
    assert cursor(svc2) is None and scanned_at(svc2) is None


def test_snapshot_active_rotates_past_a_quiet_target(tmp_path: Any) -> None:
    mint2, pool2 = addr("MintBBB"), addr("PooLBBB")
    cid2 = f"solana:{mint2}"
    svc, clock, chain = setup(tmp_path, 1, 2, verify_deployer=False, early_max_sig_pages=0)
    clock.advance(1)
    svc.add_target(mint2, pool2, "pumpswap", CREATED)  # selected after the quiet target

    def batch() -> str:
        clock.advance(60)
        [r] = asyncio.run(svc.snapshot_active(1))
        assert not isinstance(r, tuple), r
        return r.canonical_id

    assert batch() == CID  # neither scanned yet: selection order
    assert batch() == cid2  # never scanned beats scanned
    # CID's pool stays quiet (zero new signatures) yet its scan still counts, so the two
    # targets take turns instead of the earliest-selected one winning every batch.
    assert [batch() for _ in range(4)] == [CID, cid2, CID, cid2]
    assert last_scan(svc)[1] == 0  # the latest scans listed nothing new
    assert cursor(svc) == "sig2"
