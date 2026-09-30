"""Calibration Engine post-fix time filter: only observations decided in [since, until)
enter the dataset, before correlation controls, counts, findings, candidates and
validation. LIVE_FORWARD only by default. Temporary databases only."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from upscale.services.calibration.cli import main as cli_main
from upscale.services.calibration.config import CalibrationConfig, TimeFilter, parse_timestamp
from upscale.services.calibration.dataset import Observation
from upscale.services.calibration.store import CalibrationStore

from .test_calibration import START, SyntheticEngine, make, synthetic

CUTOFF = START + timedelta(days=10)


def cohorts() -> list[Observation]:
    """Pre-fix data (with corrupted extreme returns) and a clean post-fix cohort."""
    old = synthetic(60, 3, "CALIBRATION", seed=1, start=START)
    old += [make(f"solana:bad{i}", START + timedelta(hours=i), 9e8) for i in range(10)]  # corrupted
    new = synthetic(60, 3, "CALIBRATION", seed=2, start=CUTOFF)
    new += synthetic(30, 2, "VALIDATION", seed=3, start=CUTOFF + timedelta(days=20))
    return old + new


def engine(
    tmp_path: Path, name: str, observations: list[Observation], **kw: object
) -> SyntheticEngine:
    e = SyntheticEngine(CalibrationStore(tmp_path / f"{name}.sqlite3"), observations)
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def test_no_filter_is_unchanged(tmp_path: Path) -> None:
    plain = engine(tmp_path, "a", cohorts())
    explicit = engine(tmp_path, "b", cohorts(), time_filter=TimeFilter())
    _, a = plain.analyze("1h")
    _, b = explicit.analyze("1h")
    assert a == b
    report = a["dataset"]
    assert report["time_filter"]["active"] is False
    assert report["raw_before_time_filter"] == report["raw_after_time_filter"]


def test_since_excludes_older_observations_before_everything(tmp_path: Path) -> None:
    e = engine(tmp_path, "c", cohorts(), time_filter=TimeFilter(since=CUTOFF))
    _, results = e.analyze("1h")
    report = results["dataset"]
    assert report["time_filter"]["since"] == CUTOFF.isoformat() and report["time_filter"]["active"]
    assert report["raw_before_time_filter"] == {"LIVE_FORWARD": 60 * 3 + 10 + 60 * 3 + 30 * 2}
    assert report["raw_after_time_filter"] == {"LIVE_FORWARD": 60 * 3 + 30 * 2}
    live = results["by_origin"]["LIVE_FORWARD"]
    assert live["extreme_returns"] == 0  # the corrupted pre-fix rows never entered
    cal, val, _ = e.dataset()
    assert cal and val and all(o.at >= CUTOFF for o in [*cal, *val])


def test_exact_boundaries_since_inclusive_until_exclusive() -> None:
    f = TimeFilter(since=CUTOFF, until=CUTOFF + timedelta(hours=1))
    assert f.keeps("LIVE_FORWARD", CUTOFF)
    assert not f.keeps("LIVE_FORWARD", CUTOFF - timedelta(microseconds=1))
    assert f.keeps("LIVE_FORWARD", CUTOFF + timedelta(minutes=59, seconds=59))
    assert not f.keeps("LIVE_FORWARD", CUTOFF + timedelta(hours=1))
    with pytest.raises(ValueError):
        TimeFilter(since=CUTOFF, until=CUTOFF)
    with pytest.raises(ValueError):
        TimeFilter(since=datetime(2026, 9, 30, 5, 50))  # naive


def test_timestamps_with_z_and_offsets() -> None:
    z = parse_timestamp("2026-09-30T05:50:00Z")
    offset = parse_timestamp("2026-09-30T07:50:00+02:00")
    assert z == offset == datetime(2026, 9, 30, 5, 50, tzinfo=UTC)
    for bad in ("2026-09-30T05:50:00", "yesterday", "2026-13-01T00:00:00Z"):
        with pytest.raises(ValueError):
            parse_timestamp(bad)


def test_invalid_timestamps_are_rejected_by_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = [
        "--db",
        str(tmp_path / "c.sqlite3"),
        "--live-db",
        str(tmp_path / "x"),
        "--replay-db",
        str(tmp_path / "y"),
    ]
    with pytest.raises(SystemExit) as exit_:
        cli_main([*base, "analyze", "--since", "2026-09-30T05:50:00"])
    assert exit_.value.code == 2 and "no timezone" in capsys.readouterr().err
    assert (
        cli_main(
            [*base, "analyze", "--since", "2026-09-30T06:00:00Z", "--until", "2026-09-30T05:00:00Z"]
        )
        == 2
    )
    for command in (["readiness"], ["analyze", "--horizon", "1h"], ["create-candidates"]):
        assert (
            cli_main(
                [*base, *command, "--since", "2026-09-30T05:50:00Z", "--origin", "LIVE_FORWARD"]
            )
            == 0
        )
    for command in ("validate", "compare"):
        assert (
            cli_main([*base, command, "cand-missing", "--since", "2026-09-30T05:50:00Z"]) == 1
        )  # unknown id


def test_readiness_counts_respect_the_filter(tmp_path: Path) -> None:
    all_ = engine(tmp_path, "d", cohorts()).readiness()
    clean = engine(tmp_path, "e", cohorts(), time_filter=TimeFilter(since=CUTOFF)).readiness()
    assert all_["origins"]["LIVE_FORWARD"]["observations"] == 60 * 3 + 10 + 60 * 3 + 30 * 2
    assert clean["origins"]["LIVE_FORWARD"]["observations"] == 60 * 3 + 30 * 2
    assert clean["time_filter"]["since"] == CUTOFF.isoformat()
    assert clean["raw_after_time_filter"] == {"LIVE_FORWARD": 60 * 3 + 30 * 2}


def test_candidates_and_validation_use_the_filtered_cohort(tmp_path: Path) -> None:
    e = engine(tmp_path, "f", cohorts(), time_filter=TimeFilter(since=CUTOFF))
    _, ids = e.create_candidates("1h")
    assert ids
    c = e.store.candidate(ids[0])
    assert (
        c is not None
        and c["parent_version"]["cohort"]["time_filter"]["since"] == CUTOFF.isoformat()
    )
    calibration = e.store.metrics(ids[0])[0]["metrics"]["candidate"]["selected"]
    assert calibration["measured"] <= 60 * 3  # post-fix calibration only
    # A fresh engine without a filter validates on the candidate's recorded cohort.
    fresh = engine(tmp_path, "f", cohorts())
    result = fresh.validate(ids[0])
    assert result["dataset"]["time_filter"]["since"] == CUTOFF.isoformat()
    assert result["dataset"]["raw_after_time_filter"] == {"LIVE_FORWARD": 60 * 3 + 30 * 2}
    compared = engine(tmp_path, "f", cohorts()).compare(ids[0])
    assert compared["dataset"]["time_filter"]["since"] == CUTOFF.isoformat()
    unfiltered_ids = engine(tmp_path, "g", cohorts()).create_candidates("1h")[1]
    assert not set(unfiltered_ids) & set(ids)  # a different cohort is a different candidate


def test_correlation_controls_run_after_the_time_filter(tmp_path: Path) -> None:
    # One busy asset around the cutoff; with 60-minute spacing, filtering after thinning
    # would keep the pre-cutoff row and drop the post-cutoff one 40 minutes later.
    busy = [make("solana:busy", CUTOFF + timedelta(minutes=m), 1.0) for m in (-30, 10, 20)]
    e = engine(tmp_path, "h", busy, time_filter=TimeFilter(since=CUTOFF))
    cal, _, _ = e.dataset()
    assert [o.at for o in cal] == [CUTOFF + timedelta(minutes=10)]


def test_replay_is_only_filtered_when_asked(tmp_path: Path) -> None:
    obs = cohorts() + synthetic(
        20, 2, "CALIBRATION", origin="HISTORICAL_REPLAY", seed=4, start=START
    )
    live_only = engine(tmp_path, "i", obs, time_filter=TimeFilter(since=CUTOFF))
    live_only.dataset()
    assert live_only.filter_counts["raw_after_time_filter"]["HISTORICAL_REPLAY"] == 40  # untouched
    everything = engine(tmp_path, "j", obs,
                        time_filter=TimeFilter(since=CUTOFF, origins=("LIVE_FORWARD", "HISTORICAL_REPLAY")))  # fmt: skip
    everything.dataset()
    assert "HISTORICAL_REPLAY" not in everything.filter_counts["raw_after_time_filter"]
    only_live = engine(tmp_path, "k", obs, origin="LIVE_FORWARD")
    cal, _, _ = only_live.dataset()
    assert {o.origin for o in cal} == {"LIVE_FORWARD"}


def test_newer_valid_extreme_moves_remain(tmp_path: Path) -> None:
    obs = cohorts() + [
        make(f"solana:moon{i}", CUTOFF + timedelta(hours=i), 2500.0) for i in range(3)
    ]
    e = engine(tmp_path, "l", obs, time_filter=TimeFilter(since=CUTOFF))
    cal, _, _ = e.dataset()
    kept = [o for o in cal if o.asset_id.startswith("solana:moon")]
    assert len(kept) == 3 and all(o.outcomes["1h"].return_pct == 2500.0 for o in kept)
    assert not any(o.asset_id.startswith("solana:bad") for o in cal)
    assert CalibrationConfig().samples.descriptive_n == 20  # thresholds unchanged
