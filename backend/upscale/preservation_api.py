"""Read-only export API for the preservation service (private network, signed requests).

Disabled by default: `upscale.main` mounts the router ONLY when `configured()` is true, i.e. when
``UPSCALE_PRESERVATION_TOKEN`` (>= 32 characters), ``UPSCALE_PRESERVATION_ALLOWED_HOSTS`` and
``UPSCALE_PRESERVATION_ENVIRONMENT`` (which must equal this process's ``RAILWAY_ENVIRONMENT_NAME``)
are all set. Otherwise the routes do not exist.

Every request is authenticated by an HMAC-SHA256 signature made with the shared token over
``timestamp \\n METHOD \\n path \\n sha256(body)`` (the token itself never travels): a request with a
wrong signature, a timestamp more than 60 s from the server's clock, a repeated signature (replay),
a body over 1 KiB, or a ``Host`` outside the allow-list gets the same empty 404 as a route that does
not exist. No endpoint writes to a research store, runs a command or reads a path chosen by the caller.
"""

import hmac
import os
import re
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from upscale.services.corpus_cycle.export_service import (
    ExportBusy,
    ExportManager,
    ExportUnknown,
    sign,
)

MIN_TOKEN = 32
MAX_BODY = 1024
NONCE = re.compile(r"[0-9a-f]{16,64}")
SKEW_S = 60
PREFIX = "/internal/preservation/v1"
router = APIRouter(prefix=PREFIX, tags=["preservation"])
_manager: ExportManager | None = None
_seen: dict[str, float] = {}
_seen_lock = threading.Lock()


def configured(env: dict[str, str] | None = None) -> bool:
    e = os.environ if env is None else env
    expected = e.get("UPSCALE_PRESERVATION_ENVIRONMENT", "")
    return (
        len(e.get("UPSCALE_PRESERVATION_TOKEN", "")) >= MIN_TOKEN
        and bool(e.get("UPSCALE_PRESERVATION_ALLOWED_HOSTS", "").strip())
        and bool(expected)
        and e.get("RAILWAY_ENVIRONMENT_NAME", "") == expected
    )


def manager() -> ExportManager:
    global _manager
    if _manager is None:
        import upscale.config as config

        scout = config.SCOUT_DB_PATH
        beside = os.path.dirname(scout)
        _manager = ExportManager.from_env(
            os.environ,
            {
                "scout": scout,
                "evidence": config.EVIDENCE_DB_PATH,
                "calibration": os.path.join(beside, "calibration.sqlite3"),
                "shadow": os.path.join(beside, "shadow.sqlite3"),
            },
        )
    return _manager


async def guard(request: Request) -> None:
    token = os.environ.get("UPSCALE_PRESERVATION_TOKEN", "")
    hosts = {
        h.strip().lower()
        for h in os.environ.get("UPSCALE_PRESERVATION_ALLOWED_HOSTS", "").split(",")
        if h.strip()
    }
    h = request.headers
    try:
        if int(h.get("content-length", "0") or 0) > MAX_BODY:
            raise ValueError
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY:
                raise ValueError
        request._body = body  # let FastAPI parse the (already read) body
        ts = int(h.get("x-preserver-timestamp", ""))
        nonce = h.get("x-preserver-nonce", "")
        if not NONCE.fullmatch(nonce):
            raise ValueError
        sig = h.get("x-preserver-signature", "")
        host_ok = h.get("host", "").lower() in hosts
        fresh = abs(time.time() - ts) <= SKEW_S
        good = hmac.compare_digest(
            sig.encode(), sign(token, ts, nonce, request.method, request.url.path, body).encode()
        )
    except (ValueError, TypeError):
        raise HTTPException(404) from None
    if not (len(token) >= MIN_TOKEN and hosts and host_ok and fresh and good):
        raise HTTPException(404)
    now = time.time()
    with _seen_lock:  # a signature is good for one use: a replayed request is refused
        for k in [k for k, exp in _seen.items() if exp < now]:
            del _seen[k]
        if nonce in _seen:
            raise HTTPException(404)
        _seen[nonce] = now + 2 * SKEW_S


class StartExport(BaseModel):
    export_id: str = Field(min_length=3, max_length=80, pattern=r"^[a-z0-9][a-z0-9._-]{2,79}$")


@router.get("/ping", dependencies=[Depends(guard)])
def ping() -> dict[str, Any]:
    """Authenticated liveness + source identity. Exports nothing and touches no store."""
    return {"ok": True, "identity": manager().identity, "stores": sorted(manager().stores)}


@router.post("/exports", status_code=202, dependencies=[Depends(guard)])
def start_export(body: StartExport) -> dict[str, Any]:
    try:
        return manager().start(body.export_id)
    except ExportBusy as exc:
        raise HTTPException(409, str(exc)) from None
    except ExportUnknown:
        raise HTTPException(404) from None


@router.get("/exports/{export_id}", dependencies=[Depends(guard)])
def export_status(export_id: str) -> dict[str, Any]:
    try:
        return manager().status(export_id)
    except ExportUnknown:
        raise HTTPException(404) from None


@router.get("/exports/{export_id}/files/{name}", dependencies=[Depends(guard)])
def export_file(export_id: str, name: str) -> FileResponse:
    try:
        path, size, sha = manager().open_file(export_id, name)
    except ExportUnknown:
        raise HTTPException(404) from None
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={
            "X-Content-SHA256": sha,
            "X-Content-Length": str(size),
            "Cache-Control": "no-store",
        },
    )


@router.delete("/exports/{export_id}", dependencies=[Depends(guard)])
def delete_export(export_id: str) -> dict[str, Any]:
    try:
        return {"deleted": manager().delete(export_id)}
    except ExportBusy as exc:
        raise HTTPException(409, str(exc)) from None
    except ExportUnknown:
        raise HTTPException(404) from None
