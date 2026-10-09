"""Code fingerprints for stored Opportunity decisions: what can change a stored body.

* ``source:<path>``: SHA-256 of each implementation module that shapes a body.
* ``config:<name>``: one canonical value per *semantic* constant of ``config.py``. The file
  itself isn't hashed: it also holds storage / runtime settings (database paths, the
  Opportunity DB schema version, component metadata) that never change a body, and an
  old decision must not become a FINGERPRINT_MISMATCH because of them.

INPUT fingerprints: the code and constants that turn stored upstream evidence into a
canonical `OpportunityInput` (selection, interpretation, ownership, models, freshness,
required Safety version, store / agent names written into refs), including the Evidence
Archive record parser / point-in-time query and the address helper.

DECISION fingerprints: the code and constants that map a canonical `OpportunityInput` to an
`OpportunityDecision` (the engine and its types, the input model and ownership map it
re-validates, every O2 threshold, the freshness limits and Safety version its integrity
check enforces, the versions it writes).

Transport code (CLI, repository, recorder, this module) never shapes a body: excluded.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from upscale.services.opportunity_model import config as C
from upscale.services.opportunity_model.config import DecisionConfig, Freshness

SERVICES = Path(__file__).resolve().parent.parent  # upscale/services
INPUT_FILES = (
    "chains.py",
    "evidence_archive/store.py",
    "opportunity_model/loaders.py",
    "opportunity_model/models.py",
    "opportunity_model/normalize.py",
    "opportunity_model/ownership.py",
    "opportunity_model/service.py",
)
DECISION_FILES = (
    "chains.py",
    "opportunity_model/decision.py",
    "opportunity_model/decision_models.py",
    "opportunity_model/models.py",
    "opportunity_model/ownership.py",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _freshness(f: Freshness) -> dict[str, str]:
    return {f"config:freshness.{k}": _value(v) for k, v in sorted(vars(f).items())}


def input_semantics(freshness: Freshness | None = None) -> dict[str, str]:
    """The ``config.py`` values normalization and input building read."""
    return {
        "config:input_schema": _value(C.INPUT_SCHEMA),
        "config:supported_chains": _value(list(C.SUPPORTED_CHAINS)),
        "config:safety_rules_version": _value(C.SAFETY_RULES_VERSION),
        "config:safety_snapshot_schema": _value(C.SAFETY_SNAPSHOT_SCHEMA),
        "config:min_scout_timing_version": _value(C.MIN_SCOUT_TIMING_VERSION),
        "config:technical_agent": _value(C.TECHNICAL_AGENT),
        "config:news_agent": _value(C.NEWS_AGENT),
        "config:archive_store": _value(C.ARCHIVE_STORE),
        "config:safety_store": _value(C.SAFETY_STORE),
        **_freshness(freshness or C.FRESHNESS),
    }


def decision_semantics(
    cfg: DecisionConfig | None = None, freshness: Freshness | None = None
) -> dict[str, str]:
    """The ``config.py`` values `decide()` reads (thresholds, versions, integrity limits)."""
    thresholds = (cfg or C.DECISION_CONFIG).as_dict()
    return {
        "config:rules_version": _value(C.OPPORTUNITY_RULES_VERSION),
        "config:decision_schema": _value(C.DECISION_SCHEMA),
        "config:input_schema": _value(C.INPUT_SCHEMA),
        "config:safety_rules_version": _value(C.SAFETY_RULES_VERSION),
        **{f"config:threshold.{k}": _value(v) for k, v in thresholds.items()},
        **_freshness(freshness or C.FRESHNESS),
    }


def input_fingerprints(root: Path = SERVICES, freshness: Freshness | None = None) -> dict[str, str]:
    out = {f"source:{f}": _digest(root / f) for f in INPUT_FILES} | input_semantics(freshness)
    return dict(sorted(out.items()))


def decision_fingerprints(
    root: Path = SERVICES, cfg: DecisionConfig | None = None, freshness: Freshness | None = None
) -> dict[str, str]:
    out = {f"source:{f}": _digest(root / f) for f in DECISION_FILES}
    out |= decision_semantics(cfg, freshness)
    return dict(sorted(out.items()))


def differing(stored: dict[str, str], current: dict[str, str]) -> list[str]:
    """Keys whose value differs or that exist on one side only, sorted."""
    return sorted(k for k in stored.keys() | current.keys() if stored.get(k) != current.get(k))
