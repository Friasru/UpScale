"""Retention settings: how long high-frequency history is kept on the /data volume.

Environment (each optional; an invalid value falls back to the default, a value below its
floor is raised to the floor, and both are reported in `adjustments`):

* ``UPSCALE_RETENTION_ENABLED``: background retention. OFF unless set: ``1`` / ``true`` /
  ``on`` / ``yes`` deletes, ``dry-run`` only logs what would be deleted. The CLI works
  either way (``cleanup --dry-run`` never writes).
* ``UPSCALE_EVIDENCE_RETENTION_DAYS`` (default 7), ``UPSCALE_SCOUT_SNAPSHOT_RETENTION_DAYS``
  (7), ``UPSCALE_SOCIAL_RETENTION_DAYS`` (7): raw history, floor 3 days.
* ``UPSCALE_OUTCOME_RETENTION_DAYS`` (default 30): finalized outcome history, floor 14 days.
* ``UPSCALE_RETENTION_INTERVAL_HOURS`` (default 12, at least 6).
* ``UPSCALE_RETENTION_BATCH_SIZE`` (default 500 rows per transaction, 50..5000).
* ``UPSCALE_RETENTION_IGNORE_SHADOW_RUNS``: comma-separated Shadow run ids whose cursor no
  longer holds back evidence retention (an abandoned run). Their positions stay protected.

The dependency floors (how far back production code itself reads) are applied on top, in
`upscale.services.retention.engine`: a short retention never removes rows a live reader
still needs.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Literal

import upscale.config  # noqa: F401  (loads the repo-root .env before reading variables)

logger = logging.getLogger("upscale.retention")

Mode = Literal["off", "dry-run", "on"]
RAW_FLOOR_DAYS = 3.0
OUTCOME_FLOOR_DAYS = 14.0
DEFAULT_EVIDENCE_DAYS = 7.0
DEFAULT_SNAPSHOT_DAYS = 7.0
DEFAULT_SOCIAL_DAYS = 7.0
DEFAULT_OUTCOME_DAYS = 30.0
DEFAULT_INTERVAL_HOURS = 12.0
MIN_INTERVAL_HOURS = 6.0
DEFAULT_BATCH_SIZE = 500
MIN_BATCH_SIZE, MAX_BATCH_SIZE = 50, 5000
_ON = ("1", "true", "on", "yes")


@dataclass(frozen=True)
class RetentionSettings:
    mode: Mode = "off"
    evidence_days: float = DEFAULT_EVIDENCE_DAYS
    snapshot_days: float = DEFAULT_SNAPSHOT_DAYS
    social_days: float = DEFAULT_SOCIAL_DAYS
    outcome_days: float = DEFAULT_OUTCOME_DAYS
    interval_hours: float = DEFAULT_INTERVAL_HOURS
    startup_delay_minutes: float = 60.0  # never right after a deploy / restart
    retry_minutes: float = 30.0  # after a deferral
    batch_size: int = DEFAULT_BATCH_SIZE
    batch_pause_seconds: float = 0.05  # between transactions: other writers get the lock
    max_seconds: float = 120.0  # per database and run; the rest is left for the next run
    ignored_shadow_runs: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = field(default_factory=tuple)

    def policy(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "evidence_days": self.evidence_days,
            "scout_snapshot_days": self.snapshot_days,
            "social_days": self.social_days,
            "outcome_days": self.outcome_days,
            "interval_hours": self.interval_hours,
            "batch_size": self.batch_size,
            "ignored_shadow_runs": list(self.ignored_shadow_runs),
            "adjustments": list(self.adjustments),
        }


def _days(name: str, raw: str | None, default: float, floor: float, notes: list[str]) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        notes.append(f"{name}={raw!r} is invalid; using {default:g}")
        return default
    if value < floor:
        notes.append(f"{name}={value:g} is below the {floor:g}-day floor; using {floor:g}")
        return floor
    return value


def load_settings(
    enabled: str | None = None,
    evidence_days: str | None = None,
    snapshot_days: str | None = None,
    social_days: str | None = None,
    outcome_days: str | None = None,
    interval_hours: str | None = None,
    batch_size: str | None = None,
    ignore_shadow_runs: str | None = None,
) -> RetentionSettings:
    """Arguments override the environment (None: read it)."""

    def env(value: str | None, name: str) -> str | None:
        return value if value is not None else os.getenv(name)

    notes: list[str] = []
    raw_mode = (env(enabled, "UPSCALE_RETENTION_ENABLED") or "").strip().lower()
    mode: Mode = "on" if raw_mode in _ON else "dry-run" if raw_mode == "dry-run" else "off"
    interval = _days(
        "UPSCALE_RETENTION_INTERVAL_HOURS",
        env(interval_hours, "UPSCALE_RETENTION_INTERVAL_HOURS"),
        DEFAULT_INTERVAL_HOURS,
        MIN_INTERVAL_HOURS,
        notes,
    )
    batch = DEFAULT_BATCH_SIZE
    raw_batch = env(batch_size, "UPSCALE_RETENTION_BATCH_SIZE")
    if raw_batch and raw_batch.strip():
        try:
            batch = min(max(int(raw_batch), MIN_BATCH_SIZE), MAX_BATCH_SIZE)
        except ValueError:
            notes.append(f"UPSCALE_RETENTION_BATCH_SIZE={raw_batch!r} is invalid")
    ignored = tuple(
        r.strip()
        for r in (env(ignore_shadow_runs, "UPSCALE_RETENTION_IGNORE_SHADOW_RUNS") or "").split(",")
        if r.strip()
    )
    settings = RetentionSettings(
        mode=mode,
        evidence_days=_days(
            "UPSCALE_EVIDENCE_RETENTION_DAYS",
            env(evidence_days, "UPSCALE_EVIDENCE_RETENTION_DAYS"),
            DEFAULT_EVIDENCE_DAYS,
            RAW_FLOOR_DAYS,
            notes,
        ),
        snapshot_days=_days(
            "UPSCALE_SCOUT_SNAPSHOT_RETENTION_DAYS",
            env(snapshot_days, "UPSCALE_SCOUT_SNAPSHOT_RETENTION_DAYS"),
            DEFAULT_SNAPSHOT_DAYS,
            RAW_FLOOR_DAYS,
            notes,
        ),
        social_days=_days(
            "UPSCALE_SOCIAL_RETENTION_DAYS",
            env(social_days, "UPSCALE_SOCIAL_RETENTION_DAYS"),
            DEFAULT_SOCIAL_DAYS,
            RAW_FLOOR_DAYS,
            notes,
        ),
        outcome_days=_days(
            "UPSCALE_OUTCOME_RETENTION_DAYS",
            env(outcome_days, "UPSCALE_OUTCOME_RETENTION_DAYS"),
            DEFAULT_OUTCOME_DAYS,
            OUTCOME_FLOOR_DAYS,
            notes,
        ),
        interval_hours=interval,
        batch_size=batch,
        ignored_shadow_runs=ignored,
        adjustments=tuple(notes),
    )
    for note in notes:
        logger.warning("retention: %s", note)
    return settings
