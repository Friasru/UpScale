"""Builds one canonical `OpportunityInput` per token and decision time from stored evidence.

`build_input` is the whole contract: read-only selection (`loaders`) followed by pure
interpretation (`normalize`). It never reads a clock. `OpportunityInputService` opens the
stores read-only and supplies ``decision_at``: a HISTORICAL_REPLAY input takes it as given;
a LIVE_FORWARD input takes it from the clock at the moment it reads, so it can only use
evidence actually present when it was built.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from upscale.services.evidence_archive.store import EvidenceRecord, Kind
from upscale.services.opportunity_model import normalize
from upscale.services.opportunity_model.config import (
    ARCHIVE_STORE,
    FRESHNESS,
    SAFETY_STORE,
    Freshness,
    evidence_db_path,
    safety_db_path,
)
from upscale.services.opportunity_model.loaders import (
    ArchiveReader,
    ArchiveSource,
    SafetyReader,
    SafetySource,
    StoreUnavailable,
)
from upscale.services.opportunity_model.models import (
    NewsFacts,
    OpportunityInput,
    OpportunityOrigin,
    SafetyFacts,
    SafetySourceRef,
    ScoutFacts,
    ScoutMarketView,
    SocialFacts,
    SourceRef,
    Sources,
    TechnicalFacts,
)

NOT_SUPPORTED = "Opportunity V1 is Solana-only"


def build_input(
    canonical_id: str,
    decision_at: datetime,
    origin: OpportunityOrigin,
    archive: ArchiveSource,
    safety: SafetySource,
    freshness: Freshness = FRESHNESS,
) -> OpportunityInput:
    """The canonical input at ``decision_at``. Raises `OpportunityCausalityError` for any
    selected source observed after ``decision_at`` and `OpportunityIdentityError` for one
    describing another token; a store that can't be read makes its sources UNAVAILABLE."""
    if decision_at.tzinfo is None:
        raise ValueError("decision_at must be timezone-aware")
    decision_at = decision_at.astimezone(UTC)
    chain, address = normalize.parse_canonical(canonical_id)
    if not normalize.supported(chain):
        return _unsupported(canonical_id, chain, address, decision_at, origin)

    def latest(kind: Kind) -> tuple[EvidenceRecord | None, SourceRef | None]:
        try:
            return archive.latest(kind, canonical_id, decision_at), None
        except StoreUnavailable as exc:
            return None, normalize.missing_ref("UNAVAILABLE", str(exc), ARCHIVE_STORE)

    args = (canonical_id, address, decision_at, origin, freshness)
    scout_record, scout_missing = latest("scout")
    if scout_missing is not None:
        scout = _scout_unavailable(scout_missing)
        context = None
    else:
        scout, context = normalize.scout(scout_record, *args)
    analyze, analyze_missing = latest("decision")
    social_record, social_missing = latest("social")
    social = (
        SocialFacts(
            ref=social_missing,
            facts=normalize.fill_missing("social", (), str(social_missing.reason)),
        )
        if social_missing is not None
        else normalize.social(social_record, *args)
    )
    try:
        row = safety.latest_snapshot(canonical_id, decision_at)
    except StoreUnavailable as exc:
        safety_facts = normalize.safety_missing("UNAVAILABLE", str(exc))
    else:
        safety_facts = normalize.safety(row, canonical_id, address, decision_at, freshness)
    return OpportunityInput(
        canonical_id=canonical_id,
        chain=chain,
        address=address,
        decision_at=decision_at,
        origin=origin,
        sources=Sources(
            scout=scout,
            technical=normalize.technical(scout.ref, context, analyze, analyze_missing, *args),
            social=social,
            news=normalize.news(analyze, analyze_missing, *args),
            safety=safety_facts,
        ),
    )


def _scout_unavailable(ref: SourceRef) -> ScoutFacts:
    facts = normalize.fill_missing("scout", (), str(ref.reason))
    return ScoutFacts(
        ref=ref,
        market_view=ScoutMarketView(facts=tuple(f for f in facts if f.role == "SCORING")),
        context=tuple(f for f in facts if f.role == "CONTEXT"),
    )


def _unsupported(
    cid: str, chain: str, address: str, decision_at: datetime, origin: OpportunityOrigin
) -> OpportunityInput:
    """A non-Solana token: every source NOT_SUPPORTED, nothing read."""

    def ref(store: str) -> SourceRef:
        return normalize.missing_ref("NOT_SUPPORTED", NOT_SUPPORTED, store)

    return OpportunityInput(
        canonical_id=cid,
        chain=chain,
        address=address,
        decision_at=decision_at,
        origin=origin,
        sources=Sources(
            scout=_scout_unavailable(ref(ARCHIVE_STORE)),
            technical=TechnicalFacts(
                ref=ref(ARCHIVE_STORE),
                analyze_ref=ref(ARCHIVE_STORE),
                facts=normalize.fill_missing("technical", (), NOT_SUPPORTED),
            ),
            social=SocialFacts(
                ref=ref(ARCHIVE_STORE), facts=normalize.fill_missing("social", (), NOT_SUPPORTED)
            ),
            news=NewsFacts(
                ref=ref(ARCHIVE_STORE), facts=normalize.fill_missing("news", (), NOT_SUPPORTED)
            ),
            safety=SafetyFacts(
                ref=SafetySourceRef(
                    status="NOT_SUPPORTED", store=SAFETY_STORE, reason=NOT_SUPPORTED
                ),
                facts=normalize.fill_missing("safety", (), NOT_SUPPORTED),
            ),
        ),
    )


class OpportunityInputService:
    """Opens the Evidence Archive and Safety V2 read-only for each build."""

    def __init__(
        self,
        evidence_db: str | None = None,
        safety_db: str | None = None,
        freshness: Freshness = FRESHNESS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.evidence_db = evidence_db or evidence_db_path()
        self.safety_db = safety_db or safety_db_path()  # None: Safety UNAVAILABLE
        self.freshness = freshness
        self._clock = clock

    def _build(
        self, canonical_id: str, decision_at: datetime, origin: OpportunityOrigin
    ) -> OpportunityInput:
        archive, safety = ArchiveReader(self.evidence_db), SafetyReader(self.safety_db)
        try:
            return build_input(canonical_id, decision_at, origin, archive, safety, self.freshness)
        finally:
            archive.close()
            safety.close()

    def build_replay(self, canonical_id: str, decision_at: datetime) -> OpportunityInput:
        """HISTORICAL_REPLAY: evidence observed by ``decision_at`` (a Safety snapshot with
        ``as_of <= decision_at`` may have been materialized later)."""
        return self._build(canonical_id, decision_at, "HISTORICAL_REPLAY")

    def build_live(self, canonical_id: str) -> OpportunityInput:
        """LIVE_FORWARD: ``decision_at`` is now; only evidence present now can be read."""
        return self._build(canonical_id, self._clock(), "LIVE_FORWARD")
