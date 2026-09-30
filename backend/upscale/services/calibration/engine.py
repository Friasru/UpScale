"""Calibration Engine: analyses, candidates, validation and the sealed final evaluation.

Workflow (chronological, never looking at HOLDOUT until the end):

    CALIBRATION data -> analyze -> findings -> create-candidates (DRAFT -> CALIBRATED)
    VALIDATION data  -> validate <candidate> (VALIDATED / REJECTED; never re-tuned: an
                        altered candidate is a new candidate id)
    HOLDOUT data     -> final-evaluate <candidate> --confirm-final-evaluation
                        (FROZEN_FOR_FINAL_TEST first; logged; results never feed back)

Nothing here changes production configuration, and no candidate is ever promoted
automatically (PROMOTED_MANUALLY is a human's explicit record).
"""

import getpass
import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from upscale.services.calibration import analysis
from upscale.services.calibration.candidates import (
    baselines,
    candidate_id,
    complexity,
    evaluate,
    generate,
    judge,
)
from upscale.services.calibration.config import (
    HORIZONS,
    CalibrationConfig,
    Origin,
    Split,
    TimeFilter,
)
from upscale.services.calibration.dataset import (
    HoldoutSealedError,
    Observation,
    diversify,
    fingerprint,
    live_split,
    load_live,
    load_replay,
    load_shadow,
    with_regimes,
)
from upscale.services.calibration.stats import cohort, meets
from upscale.services.calibration.store import SCHEMA_VERSION, CalibrationStore
from upscale.services.evidence_archive.store import SCHEMA_VERSION as EVIDENCE_SCHEMA
from upscale.services.replay_lab.models import RECORD_VERSION
from upscale.services.versions import fingerprints

NOT_A_PROFIT_CLAIM = (
    "Experimental calibration research: associations between UpScale's evidence and later "
    "market outcomes. Not a profitability claim, not realized trading results, and nothing "
    "here changes production configuration."
)


class CalibrationError(Exception):
    pass


def _hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _default_trigger() -> str:
    try:
        return f"cli:{getpass.getuser()}"
    except (OSError, KeyError):
        return "cli:unknown"


class CalibrationEngine:
    def __init__(
        self,
        store: CalibrationStore,
        live_db: str | None,
        replay_db: str | None,
        cfg: CalibrationConfig | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        triggered_by: Callable[[], str] = _default_trigger,
        time_filter: TimeFilter | None = None,
        origin: Origin | None = None,
        shadow_db: str | None = None,
    ):
        self.store = store
        self.live_db = live_db
        self.replay_db = replay_db
        self.shadow_db = shadow_db  # read-only; SHADOW is never merged with the other origins
        self.cfg = cfg or CalibrationConfig()
        self.now = now
        self.triggered_by = triggered_by
        self.time_filter = time_filter or TimeFilter()
        self.origin = origin  # only this origin (None: every origin, kept apart)
        self.filter_counts: dict[str, Any] = {}

    # --- data -------------------------------------------------------------------------------------

    def _raw(self, include_holdout: bool = False) -> list[Observation]:
        """Every observation from the sources, before any filtering."""
        obs: list[Observation] = []
        if self.live_db:
            obs += load_live(self.live_db, self.cfg, include_holdout=include_holdout)
        if self.replay_db:
            obs += load_replay(self.replay_db, include_holdout=include_holdout)
        if self.shadow_db:
            obs += load_shadow(self.shadow_db, self.cfg.live_split, include_holdout=include_holdout)
        return obs

    def _load(self, include_holdout: bool = False) -> list[Observation]:
        obs = self._raw(include_holdout)
        if not include_holdout and any(o.split == "HOLDOUT" for o in obs):
            raise HoldoutSealedError("HOLDOUT observations reached a calibration step")
        # The time / origin filter comes first: regimes, splits, correlation controls,
        # counts, findings, candidates and validation only ever see the filtered cohort.
        kept = [o for o in obs if self._keeps(o.origin, o.at)]
        self.filter_counts = {
            "raw_before_time_filter": dict(Counter(o.origin for o in obs)),
            "raw_after_time_filter": dict(Counter(o.origin for o in kept)),
        }
        # Regimes describe the observed market universe: simulated trades never shape them.
        shadow = [o for o in kept if o.origin == "SHADOW"]
        return with_regimes([o for o in kept if o.origin != "SHADOW"]) + shadow

    def _keeps(self, origin: str, at: datetime) -> bool:
        if self.origin is not None and origin != self.origin:
            return False
        return self.time_filter.keeps(origin, at)

    def _filter_report(self) -> dict[str, Any]:
        return {
            "time_filter": self.time_filter.describe() | {"active": self.time_filter.active},
            "origin": self.origin,
            **self.filter_counts,
        }

    def split(self, split: Split, observations: Sequence[Observation]) -> list[Observation]:
        chosen = [
            o for o in observations if o.split == split and not o.purged and o.origin != "SHADOW"
        ]
        return diversify(chosen, self.cfg.correlation)

    def dataset(self) -> tuple[list[Observation], list[Observation], dict[str, Any]]:
        raw = self._load()
        cal, val = self.split("CALIBRATION", raw), self.split("VALIDATION", raw)
        report = {
            **self._filter_report(),
            "fingerprint": fingerprint([*cal, *val]),
            "raw": {
                f"{o}/{s}": n
                for (o, s), n in sorted(Counter((o.origin, o.split) for o in raw).items())
            },
            "purged": sum(1 for o in raw if o.purged),
            # Outcomes left out because an integrity audit confirmed them invalid (raw
            # observations are kept and counted above; unaudited rows are never excluded).
            "integrity_excluded": dict(
                Counter(s for o in raw for s in o.integrity_excluded.values())
            ),
            "integrity_excluded_by_horizon": {
                h: dict(Counter(o.integrity_excluded[h] for o in raw if h in o.integrity_excluded))
                for h in HORIZONS
                if any(h in o.integrity_excluded for o in raw)
            },
            "calibration": dict(Counter(o.origin for o in cal)),
            "validation": dict(Counter(o.origin for o in val)),
            "dropped_by_correlation_controls": {
                "calibration": sum(1 for o in raw if o.split == "CALIBRATION" and not o.purged)
                - len(cal),
                "validation": sum(1 for o in raw if o.split == "VALIDATION" and not o.purged)
                - len(val),
            },
            "assets": {
                "calibration": len({o.asset_id for o in cal}),
                "validation": len({o.asset_id for o in val}),
            },
            "days": {
                "calibration": len({o.at.date() for o in cal}),
                "validation": len({o.at.date() for o in val}),
            },
            "latest_observation": max((o.at for o in raw), default=None),
        }
        return cal, val, report

    def versions(self) -> dict[str, str]:
        return {
            **fingerprints(),
            "replay_record": str(RECORD_VERSION),
            "evidence_archive_schema": str(EVIDENCE_SCHEMA),
            "calibration_schema": str(SCHEMA_VERSION),
            "live_split_policy": _hash(self.cfg.live_split.model_dump()),
        }

    def _run_id(self, kind: str, args: dict[str, Any]) -> str:
        return f"{kind}-{self.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"

    # --- analyze ------------------------------------------------------------------------------------

    def analyze(
        self, horizon: str = "1h", args: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        if horizon not in HORIZONS:
            raise CalibrationError(f"horizon must be one of {HORIZONS}")
        args = dict(args or {}) | {"horizon": horizon}
        cal, val, report = self.dataset()
        cfg = self.cfg
        rows = analysis.feature_rows(cal, horizon, cfg, "CALIBRATION") + analysis.feature_rows(
            val, horizon, cfg, "VALIDATION"
        )
        findings = [
            f
            for name in analysis.FEATURES
            if (f := analysis.compare_buckets(cal, val, name, horizon, cfg))
        ]
        patterns, tested = analysis.interactions(cal, horizon, cfg)
        comparisons = (
            len(analysis.FEATURES)
            + tested
            + sum(len(getattr(cfg, a)) for a, _ in analysis.FILTERS.values())
        )
        comparisons += len(analysis.weight_variants(cfg))
        results: dict[str, Any] = {
            "label": NOT_A_PROFIT_CLAIM,
            "language": analysis.LANGUAGE,
            "horizon": horizon,
            "dataset": report,
            "by_origin": {
                view: analysis._summary(cohort(view, analysis._origin(cal, view), horizon, cfg))
                for view in analysis.ORIGIN_VIEWS
            },
            "stages": analysis.stage_calibration(cal, val, horizon, cfg),
            "confidence": analysis.confidence_calibration(cal, horizon, cfg),
            "interactions": {"patterns": patterns[:25], "tested": tested},
            "thresholds": analysis.threshold_sensitivity(cal, horizon, cfg),
            "weights": analysis.weight_sensitivity(cal, horizon, cfg),
            "missing_evidence": analysis.missing_evidence(cal, horizon, cfg),
            "profiles": analysis.profiles(cal, horizon, cfg),
            "findings": [f.model_dump() for f in findings],
            "multiple_comparisons": {
                "comparisons": comparisons,
                "warning": (
                    f"{comparisons} comparisons were evaluated: some differences are expected by "
                    "chance alone; only findings reproduced on VALIDATION (and ideally in both "
                    "origins) deserve attention."
                ),
            },
        }
        run_id = self._run_id("analyze", args)
        self.store.add_run(
            run_id, "analyze", args, cfg.model_dump(), self.versions(), report, results, comparisons
        )
        self.store.add_findings(run_id, results["findings"])
        self.store.add_feature_rows(run_id, [r.model_dump() for r in rows])
        return run_id, results

    # --- candidates -----------------------------------------------------------------------------------

    def create_candidates(self, horizon: str = "1h") -> tuple[str, list[str]]:
        cal, _, report = self.dataset()
        made, tested = generate(cal, horizon, self.cfg)
        run_id = self._run_id("create_candidates", {"horizon": horizon})
        parent: dict[str, Any] = {**self.versions(), "evaluation_horizon": horizon}
        if self.time_filter.active or self.origin is not None:  # unfiltered ids stay as before
            parent["cohort"] = {"time_filter": self.time_filter.describe(), "origin": self.origin}
        findings = [f["finding_id"] for f in self.store.findings()]
        self.store.add_run(run_id, "create_candidates", {"horizon": horizon}, self.cfg.model_dump(),
                           self.versions(), report, {"candidates": made, "variants_tested": tested}, tested)  # fmt: skip
        ids = []
        for m in made:
            cid = candidate_id(parent, m["changes"])
            created = self.store.add_candidate({
                "candidate_id": cid, "parent_version": parent, "parent_candidate_id": None,
                "source_run_id": run_id,
                "reason": f"{m['reason']} ({tested} variants tested on CALIBRATION)",
                "source_findings": findings, "changes": m["changes"],
                "complexity": complexity(m["changes"]), "origin_mix": m["metrics"]["origins"],
            })  # fmt: skip
            if created:
                self.store.add_metrics(cid, "CALIBRATION", report["fingerprint"],
                                       {"candidate": m["metrics"], "judgement": m["judgement"]})  # fmt: skip
                self.store.set_status(cid, "CALIBRATED")
            ids.append(cid)
        return run_id, ids

    def _adopt_cohort(self, c: dict[str, Any]) -> None:
        """Without an explicit filter, a candidate is evaluated on the cohort it was created
        from (its recorded time filter / origin), so validation matches calibration."""
        cohort_ = c["parent_version"].get("cohort")
        if cohort_ is None or self.time_filter.active or self.origin is not None:
            return
        self.time_filter = TimeFilter.model_validate(cohort_["time_filter"])
        self.origin = cohort_.get("origin")

    def _candidate(self, cid: str) -> dict[str, Any]:
        c = self.store.candidate(cid)
        if c is None:
            raise CalibrationError(f"unknown candidate {cid}")
        return c

    def validate(self, cid: str) -> dict[str, Any]:
        c = self._candidate(cid)
        if c["status"] not in ("CALIBRATED", "VALIDATED", "REJECTED"):
            raise CalibrationError(
                f"{cid} is {c['status']}: only CALIBRATED candidates are validated"
            )
        horizon = c["parent_version"]["evaluation_horizon"]
        self._adopt_cohort(c)
        _, val, report = self.dataset()
        m = evaluate(c["changes"], val, horizon, self.cfg)
        base = evaluate([], val, horizon, self.cfg)
        j = judge(m, base, self.cfg)
        if not m["meets_descriptive_sample"]:
            outcome = "INSUFFICIENT_SAMPLE"
        else:
            outcome = "VALIDATED" if j["improves"] else "REJECTED"
        result = {"candidate": m, "production": base, "judgement": j, "baselines": baselines(val, horizon, self.cfg),
                  "dataset": report}  # fmt: skip
        first = c["status"] == "CALIBRATED"
        if first and outcome in ("VALIDATED", "REJECTED"):
            self.store.set_status(cid, outcome)
        recorded = outcome if first else f"REPRODUCTION ({outcome}; status stays {c['status']})"
        self.store.add_validation(cid, report["fingerprint"], recorded, result)
        self.store.add_metrics(
            cid, "VALIDATION", report["fingerprint"], {"candidate": m, "judgement": j}
        )
        return {"candidate_id": cid, "outcome": recorded, **result}

    def compare(self, cid: str) -> dict[str, Any]:
        c = self._candidate(cid)
        horizon = c["parent_version"]["evaluation_horizon"]
        self._adopt_cohort(c)
        cal, val, report = self.dataset()
        out: dict[str, Any] = {"candidate": c, "label": NOT_A_PROFIT_CLAIM, "dataset": report}
        for name, obs in (("CALIBRATION", cal), ("VALIDATION", val)):
            m = evaluate(c["changes"], obs, horizon, self.cfg)
            base = evaluate([], obs, horizon, self.cfg)
            per_origin = {
                origin: evaluate(
                    c["changes"], [o for o in obs if o.origin == origin], horizon, self.cfg
                )["selected"]
                for origin in ("LIVE_FORWARD", "HISTORICAL_REPLAY")
            }
            out[name] = {"candidate": m, "production": base, "judgement": judge(m, base, self.cfg),
                         "baselines": baselines(obs, horizon, self.cfg), "by_origin": per_origin}  # fmt: skip
        out["superiority"] = (
            "supported on VALIDATION" if out["VALIDATION"]["judgement"]["improves"]
            and out["VALIDATION"]["candidate"]["meets_descriptive_sample"]
            else "not established: VALIDATION does not support it"
        )  # fmt: skip
        return out

    # --- HOLDOUT ----------------------------------------------------------------------------------------

    def holdout_index(self) -> list[tuple[str, datetime]]:
        """Keys and times of HOLDOUT observations only (no features or outcomes)."""
        import sqlite3

        out: list[tuple[str, datetime]] = []
        if self.live_db and Path(self.live_db).expanduser().exists():
            conn = sqlite3.connect(
                f"file:{Path(self.live_db).expanduser().resolve()}?mode=ro", uri=True
            )
            try:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
                for table, col, prefix in (("scout_outcome_observations", "observed_at", "live:scout"),
                                           ("decision_observations", "analyzed_at", "live:decision")):  # fmt: skip
                    if table in tables:
                        for oid, at in conn.execute(f"SELECT id, {col} FROM {table}"):
                            t = datetime.fromtimestamp(at, UTC)
                            if live_split(t, self.cfg.live_split) == "HOLDOUT":
                                out.append((f"{prefix}:{oid}", t))
            finally:
                conn.close()
        if self.replay_db and Path(self.replay_db).expanduser().exists():
            conn = sqlite3.connect(
                f"file:{Path(self.replay_db).expanduser().resolve()}?mode=ro", uri=True
            )
            try:
                if "replay_samples" in {
                    r[0] for r in conn.execute("SELECT name FROM sqlite_master")
                }:
                    for key, at in conn.execute(
                        "SELECT s.sample_key, s.decision_at FROM replay_samples s "
                        "JOIN replay_decisions d ON d.sample_id = s.id WHERE s.split = 'HOLDOUT'"
                    ):
                        out.append((f"replay:{key}", datetime.fromtimestamp(at, UTC)))
            finally:
                conn.close()
        origin_of = {"live": "LIVE_FORWARD", "replay": "HISTORICAL_REPLAY"}
        kept = [(k, t) for k, t in out if self._keeps(origin_of.get(k.split(":", 1)[0], ""), t)]
        return sorted(set(kept), key=lambda x: (x[1], x[0]))

    def final_evaluate(self, cid: str, confirm: bool) -> dict[str, Any]:
        if not confirm:
            raise HoldoutSealedError(
                "HOLDOUT is sealed: final evaluation needs --confirm-final-evaluation (nothing was read)"
            )
        c = self._candidate(cid)
        if c["status"] not in ("VALIDATED", "FROZEN_FOR_FINAL_TEST"):
            raise CalibrationError(
                f"{cid} is {c['status']}: only VALIDATED candidates are finally evaluated"
            )
        self._adopt_cohort(c)
        if c["status"] == "VALIDATED":
            self.store.set_status(cid, "FROZEN_FOR_FINAL_TEST")  # frozen before HOLDOUT is read
        index = self.holdout_index()
        window = {"from": index[0][1] if index else None, "to": index[-1][1] if index else None,
                  "observations": len(index)}  # fmt: skip
        version = _hash([k for k, _ in index])
        self.store.log_holdout_access(cid, "final evaluation", self.triggered_by(), window, version)
        horizon = c["parent_version"]["evaluation_horizon"]
        holdout = diversify([o for o in self._load(include_holdout=True)
                             if o.split == "HOLDOUT" and o.origin != "SHADOW"],
                            self.cfg.correlation)  # fmt: skip
        m = evaluate(c["changes"], holdout, horizon, self.cfg)
        base = evaluate([], holdout, horizon, self.cfg)
        result = {"candidate": m, "production": base, "judgement": judge(m, base, self.cfg),
                  "baselines": baselines(holdout, horizon, self.cfg), "holdout_window": window,
                  "note": "final evaluation: these results never feed back into this candidate"}  # fmt: skip
        self.store.add_final_evaluation(cid, version, result)
        self.store.add_metrics(
            cid, "HOLDOUT", version, {"candidate": m, "judgement": result["judgement"]}
        )
        return {"candidate_id": cid, **result}

    def record_manual_promotion(self, cid: str, confirm: bool) -> None:
        """Records that a human promoted the candidate by hand. Changes no configuration."""
        if not confirm:
            raise CalibrationError("recording a manual promotion needs --confirm-manual-promotion")
        if self._candidate(cid)["status"] != "FROZEN_FOR_FINAL_TEST":
            raise CalibrationError(
                "only a candidate that passed the final evaluation can be recorded"
            )
        self.store.set_status(cid, "PROMOTED_MANUALLY")

    # --- readiness / status -------------------------------------------------------------------------------

    def readiness(self) -> dict[str, Any]:
        raw = self._load()
        cal, val = self.split("CALIBRATION", raw), self.split("VALIDATION", raw)
        rules = self.cfg.samples

        def completed(obs: Sequence[Observation]) -> dict[str, int]:
            return {
                h: sum(1 for o in obs if (v := o.outcomes.get(h)) is not None and v.measured)
                for h in HORIZONS
            }

        def level(obs: Sequence[Observation], h: str) -> str:
            c = cohort("x", obs, h, self.cfg)
            if meets(c, rules, "strong"):
                return "SUFFICIENT_FOR_STRONGER_EVIDENCE"
            if meets(c, rules, "candidate"):
                return "SUFFICIENT_FOR_CANDIDATES"
            if meets(c, rules, "descriptive"):
                return "SUFFICIENT_FOR_DESCRIPTIVE"
            return "INSUFFICIENT_SAMPLE"

        scout_cal = analysis._for(cal, analysis.SCOUT_KINDS)
        decision_cal = analysis._for(cal, analysis.DECISION_KINDS)
        holdout = self.holdout_index()
        return {
            "label": "Factual data coverage for calibration. Not a statement of readiness to trade.",
            **self._filter_report(),
            "origins": {
                origin: {
                    "observations": sum(1 for o in raw if o.origin == origin),
                    "distinct_assets": len({o.asset_id for o in raw if o.origin == origin}),
                    "completed": completed([o for o in raw if o.origin == origin]),
                }
                | (
                    {
                        "closed_trades": sum(
                            1 for o in raw if o.origin == origin and "trade" in o.outcomes
                        ),
                        "resolved_trades": sum(
                            1
                            for o in raw
                            if o.origin == origin
                            and (v := o.outcomes.get("trade")) is not None
                            and v.measured
                        ),
                    }
                    if origin == "SHADOW"
                    else {}
                )
                for origin in ("LIVE_FORWARD", "HISTORICAL_REPLAY", "SHADOW")
            },
            "calibration_eligible": {
                "observations": len(cal),
                "assets": len({o.asset_id for o in cal}),
                "completed": completed(cal),
            },  # fmt: skip
            "validation_eligible": {
                "observations": len(val),
                "assets": len({o.asset_id for o in val}),
                "completed": completed(val),
            },  # fmt: skip
            "holdout_reserved": len(holdout),
            "complete_safety": sum(
                1 for o in raw if o.s("safety_status") == "SAFETY_CHECKS_COMPLETE"
            ),
            "decision_grade": sum(
                1
                for o in raw
                if o.s("safety_status") == "SAFETY_CHECKS_COMPLETE"
                and o.f("liquidity_usd") is not None
            ),
            "analyses": {
                h: {
                    "scout_signals (CALIBRATION)": level(scout_cal, h),
                    "opportunity_decisions (CALIBRATION)": level(decision_cal, h),
                    "validation (VALIDATION)": level(val, h),
                    "final_evaluation (HOLDOUT, counts only)": (
                        "RESERVED" if len(holdout) >= rules.descriptive_n else "INSUFFICIENT_SAMPLE"
                    ),
                }
                for h in HORIZONS
            },
            "rules": rules.model_dump(),
        }

    def compare_origins(self, horizon: str) -> dict[str, Any]:
        """LIVE_FORWARD, HISTORICAL_REPLAY and SHADOW side by side, never merged. Live and
        replay are fixed-horizon market outcomes on CALIBRATION + VALIDATION (diversified);
        SHADOW is each strategy's simulated entry-to-exit trades (IDEALIZED_NO_FEES), per
        strategy version. HOLDOUT is excluded everywhere."""
        raw = self._load()
        base = [o for o in raw if o.split in ("CALIBRATION", "VALIDATION") and not o.purged]
        out: dict[str, Any] = {
            "label": NOT_A_PROFIT_CLAIM,
            **self._filter_report(),
            "horizon": horizon,
            "note": (
                "Origins are reported separately and never merged. SHADOW rows are simulated "
                "paper trades (entry to exit, no fees or slippage), not fixed-horizon outcomes "
                "and not real profit."
            ),
        }
        for origin in ("LIVE_FORWARD", "HISTORICAL_REPLAY"):
            obs = diversify([o for o in base if o.origin == origin], self.cfg.correlation)
            out[origin] = analysis._summary(cohort(origin, obs, horizon, self.cfg))
        shadow = [o for o in base if o.origin == "SHADOW"]
        out["SHADOW"] = {
            key: analysis._summary(cohort(key, [o for o in shadow if o.s("strategy") == key],
                                          "trade", self.cfg))
            for key in sorted({o.s("strategy") or "unknown" for o in shadow})
        }  # fmt: skip
        return out

    def status(self) -> dict[str, Any]:
        cands = self.store.candidates()
        return {
            "label": NOT_A_PROFIT_CLAIM,
            "calibration_db": self.store.path,
            "sources": {
                "live_db": self.live_db
                if self.live_db and Path(self.live_db).expanduser().exists()
                else None,
                "replay_db": self.replay_db
                if self.replay_db and Path(self.replay_db).expanduser().exists()
                else None,
                "shadow_db": self.shadow_db
                if self.shadow_db and Path(self.shadow_db).expanduser().exists()
                else None,
            },
            "runs": self.store.runs(),
            "candidates_by_status": dict(Counter(c["status"] for c in cands)),
            "holdout_accesses": len(self.store.holdout_accesses()),
            "versions": self.versions(),
        }
