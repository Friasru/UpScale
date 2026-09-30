"""The one call production code makes to archive evidence it already has.

`emit(kind, obj)` hands an object production just produced (a Scout candidate, a DEX
snapshot, an on-chain safety snapshot, social momentum, a Growth Scout ranking, an
Analyze reply) to the installed recorder, which serializes it and queues it for a
background writer. It never makes a network request, never blocks on disk, and never
raises into the caller: archiving can't change or break production behavior. With no
recorder installed (disabled, or a replay process) it does nothing.

`component(name)` labels which production path produced the evidence (analyze, scout,
enrichment...), for provenance.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

logger = logging.getLogger("upscale.evidence")

_COMPONENT: ContextVar[str | None] = ContextVar("upscale_evidence_component", default=None)


class EvidenceSink(Protocol):
    def submit(self, kind: str, obj: Any, component: str, extra: dict[str, Any]) -> None: ...


_sink: EvidenceSink | None = None


def install(sink: EvidenceSink | None) -> None:
    global _sink
    _sink = sink


def installed() -> EvidenceSink | None:
    return _sink


@contextmanager
def component(name: str) -> Iterator[None]:
    token = _COMPONENT.set(name)
    try:
        yield
    finally:
        _COMPONENT.reset(token)


def current_component(default: str = "unknown") -> str:
    return _COMPONENT.get() or default


# Evidence emitted while one Scout run / Analyze is in progress: the decision record it
# ends with links exactly this evidence (never later evidence of the same asset).
_SCOPE: ContextVar[list[dict[str, Any]] | None] = ContextVar("upscale_evidence_scope", default=None)
DECISION_KINDS = frozenset({"scout", "decision"})


@contextmanager
def evaluation() -> Iterator[list[dict[str, Any]]]:
    """Collect the evidence emitted inside this block (and tasks it starts) for the decision
    it produces. Yields the collection, to pass to the decision's `emit(..., scope=...)`."""
    scope: list[dict[str, Any]] = []
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def emit(kind: str, obj: Any, **extra: Any) -> None:
    sink = _sink
    if sink is None:
        return
    try:
        scope = _SCOPE.get()
        if kind in DECISION_KINDS:
            extra.setdefault("scope", list(scope) if scope is not None else None)
        elif scope is not None:
            entry: dict[str, Any] = {"kind": kind, "links": []}
            scope.append(entry)
            extra["scope_entry"] = entry  # the recorder fills in the archived identities
        sink.submit(kind, obj, current_component(), extra)
    except Exception:  # archiving must never affect production
        logger.exception("evidence archive: could not queue %s evidence", kind)
