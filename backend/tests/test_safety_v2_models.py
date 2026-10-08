"""Safety V2 field-status honesty, exact Solana identity and settings. Offline."""

from typing import Any

import pytest
from pydantic import ValidationError

from tests.safety_v2_fakes import MINT
from upscale.services.market_data import InvalidRequestError
from upscale.services.safety_v2.config import SafetySettings, load_settings
from upscale.services.safety_v2.models import Evidence, RuleResult, missing, solana_identity

NON_VALUED = ("UNKNOWN", "UNAVAILABLE", "PROVIDER_UNAVAILABLE", "NOT_COLLECTED", "NOT_SUPPORTED")


@pytest.mark.parametrize("status", NON_VALUED)
@pytest.mark.parametrize("value", [0, 0.0, False, "", [], "revoked"])
def test_no_value_with_a_non_valued_status(status: str, value: Any) -> None:
    with pytest.raises(ValidationError, match="can't carry a value"):
        Evidence(status=status, value=value, reason="why")  # type: ignore[arg-type]


@pytest.mark.parametrize("status", NON_VALUED)
def test_non_valued_statuses_need_a_reason(status: str) -> None:
    with pytest.raises(ValidationError, match="needs a reason"):
        Evidence(status=status)  # type: ignore[arg-type]
    assert missing(status, "why").value is None  # type: ignore[arg-type]


def test_valued_statuses_need_a_value() -> None:
    for status in ("AVAILABLE", "PARTIAL"):
        with pytest.raises(ValidationError, match="needs a value"):
            Evidence(status=status, reason="why")  # type: ignore[arg-type]


def test_partial_numeric_field_is_a_lower_bound() -> None:
    with pytest.raises(ValidationError, match="lower bound"):
        Evidence(status="PARTIAL", value=12.5, reason="page cap")
    with pytest.raises(ValidationError, match="needs a reason"):
        Evidence(status="PARTIAL", value=12.5, lower_bound=True)
    ok = Evidence(status="PARTIAL", value=12.5, lower_bound=True, reason="page cap")
    assert ok.lower_bound
    with pytest.raises(ValidationError, match="only a PARTIAL"):
        Evidence(status="AVAILABLE", value=12.5, lower_bound=True)


def test_unknown_and_unavailable_are_distinct_statuses() -> None:
    a, b = missing("UNKNOWN", "unresolved"), missing("UNAVAILABLE", "absent")
    assert a.status != b.status and a.value is None and b.value is None


def test_rule_result_needs_iff_undetermined() -> None:
    kw: dict[str, Any] = dict(id="X", severity="high", decision_bearing=True, evidence=("a",),
                              reason="r")  # fmt: skip
    with pytest.raises(ValidationError):
        RuleResult(outcome="UNDETERMINED", **kw)
    with pytest.raises(ValidationError):
        RuleResult(outcome="TRIGGERED", needs="x", **kw)
    assert RuleResult(outcome="NOT_TRIGGERED", **kw).needs is None


def test_solana_identity_is_exact_and_case_sensitive() -> None:
    assert solana_identity(MINT) == (f"solana:{MINT}", MINT)
    assert solana_identity(f"solana:{MINT}") == (f"solana:{MINT}", MINT)
    swapped = MINT.replace("AAA", "aaa")
    assert solana_identity(swapped)[1] == swapped != MINT


@pytest.mark.parametrize(
    "raw",
    [
        "0x" + "a" * 40,  # EVM
        "solana:0x" + "a" * 40,
        f"ethereum:{MINT}",
        f"base:{MINT}",
        "MintAAA0OIl",  # not base58
        "short",
        f"solana: {MINT}",
        "",
    ],
)
def test_invalid_solana_identity_is_rejected(raw: str) -> None:
    with pytest.raises(InvalidRequestError):
        solana_identity(raw)


def test_settings_are_frozen_strict_and_namespaced() -> None:
    s = SafetySettings(db_path="x")
    with pytest.raises(ValidationError):
        s.daily_request_budget = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SafetySettings(db_path="x", radar_budget=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SafetySettings(db_path="x", daily_request_budget="10")  # type: ignore[arg-type]
    env = {"UPSCALE_SAFETY_V2_DB": "/tmp/s.sqlite3", "UPSCALE_SAFETY_V2_DAILY_REQUEST_BUDGET": "7",
           "UPSCALE_RADAR_DAILY_REQUEST_BUDGET": "999", "UPSCALE_SAFETY_V2_MAX_RPS": "2.5"}  # fmt: skip
    loaded = load_settings(env)
    assert (loaded.db_path, loaded.daily_request_budget, loaded.max_rps) == (
        "/tmp/s.sqlite3",
        7,
        2.5,
    )
    with pytest.raises(ValueError, match="isn't a number"):
        load_settings({"UPSCALE_SAFETY_V2_MAX_RETRIES": "two"})
