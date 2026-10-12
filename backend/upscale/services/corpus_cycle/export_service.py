"""Source-side read-only export service logic (no web framework; see `upscale.preservation_api`).

Replaces the unrestricted-shell transfer (`railway ssh`) with a narrow interface: a caller with the
bearer token can ask for ONE named export of the approved research stores, read the resulting
files (byte-exact, with the SHA-256 measured here) and delete that export. It cannot run commands,
read other paths or write to any store.

* The export is the unchanged P1 `export_stores`: every source opened ``mode=ro`` with
  ``query_only``, one online-backup copy per store into a private directory on another filesystem.
* One export job at a time; the same id is idempotent (a READY export is returned, never redone);
  exports older than `max_age_s` are removed when the next one starts.
* File names are taken from the export's own receipt, never from the request.
* The source identity (Railway environment / service / deployment / commit) is written into the
  receipt by `export_stores`, so the caller can verify it is talking to production.
"""

import hashlib
import hmac
import json
import re
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from upscale.services.corpus_cycle.export import RECEIPT, export_stores

EXPORT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
STORE_ENV = {
    "evidence": "UPSCALE_EVIDENCE_DB", "scout": "UPSCALE_SCOUT_DB", "safety": "UPSCALE_SAFETY_V2_DB",
    "opportunity": "UPSCALE_OPPORTUNITY_DB", "orchestrator": "UPSCALE_ORCHESTRATOR_DB",
    "calibration": "UPSCALE_CALIBRATION_DB", "shadow": "UPSCALE_SHADOW_DB",
}  # fmt: skip
IDENTITY_ENV = {"environment": "RAILWAY_ENVIRONMENT_NAME", "service": "RAILWAY_SERVICE_NAME",
                "deployment": "RAILWAY_DEPLOYMENT_ID", "commit": "RAILWAY_GIT_COMMIT_SHA"}  # fmt: skip


MAX_EXPORT_BYTES = 8 * 1024**3  # refuse to export stores whose total size exceeds this
MARKER = {"interface": "export_service"}  # written into the receipt: what this service created


class ExportBusy(Exception):
    """Another export is running."""


class ExportUnknown(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sign(token: str, ts: int, nonce: str, method: str, path: str, body: bytes) -> str:
    """HMAC-SHA256 request signature shared by the export API and its client."""
    msg = "\n".join([str(ts), nonce, method.upper(), path, hashlib.sha256(body).hexdigest()])
    return hmac.new(token.encode(), msg.encode(), hashlib.sha256).hexdigest()


def resolve_stores(env: Mapping[str, str], defaults: Mapping[str, str]) -> dict[str, Path]:
    """Store name -> path: the environment variable the app itself uses, else the app's default
    location; only files that exist."""
    out: dict[str, Path] = {}
    for name, var in STORE_ENV.items():
        raw = env.get(var) or defaults.get(name)
        if raw and Path(raw).expanduser().is_file():
            out[name] = Path(raw).expanduser()
    return out


class ExportManager:
    def __init__(self, root: Path, stores: Mapping[str, Path], identity: Mapping[str, str], *,
                 max_age_s: float = 6 * 3600, max_bytes: int = MAX_EXPORT_BYTES, clock: Callable[[], float] = time.time,
                 synchronous: bool = False, allow_same_filesystem: bool = False):  # fmt: skip
        self.root = root
        self.stores = dict(stores)
        self.identity = {k: v for k, v in identity.items() if v}
        self.max_age_s, self.clock, self.synchronous = max_age_s, clock, synchronous
        self.same_fs = allow_same_filesystem  # local rehearsals only
        self.max_bytes = max_bytes
        self._started_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._running: str | None = None
        self._failed: dict[str, str] = {}
        self._claims: dict[str, dict[str, Any]] = {}
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)

    @classmethod
    def from_env(cls, env: Mapping[str, str], defaults: Mapping[str, str]) -> "ExportManager":
        root = Path(env.get("UPSCALE_PRESERVATION_TMP") or "/tmp/upscale-preservation")  # noqa: S108
        return cls(
            root,
            resolve_stores(env, defaults),
            {k: env.get(v, "") for k, v in IDENTITY_ENV.items()},
        )

    # -- jobs ----------------------------------------------------------------------------------

    def _dir(self, export_id: str) -> Path:
        if not EXPORT_ID.fullmatch(export_id):
            raise ExportUnknown("invalid export id")
        return self.root / export_id

    def start(self, export_id: str) -> dict[str, Any]:
        d = self._dir(export_id)
        with self._lock:
            if self._running == export_id:
                return {"status": "RUNNING", "export_id": export_id}
            if d.exists():
                return self.status(export_id)
            if self._running is not None:
                raise ExportBusy(f"export {self._running} is running")
            self._purge_stale()
            self._failed.pop(export_id, None)
            try:
                total = sum(p.stat().st_size for p in self.stores.values())
            except OSError as exc:  # a store file is missing: report FAILED, never a server error
                self._failed[export_id] = f"a source store is unreadable: {type(exc).__name__}"
                return {
                    "status": "FAILED",
                    "export_id": export_id,
                    "error": self._failed[export_id],
                }
            if total > self.max_bytes:
                self._failed[export_id] = (
                    f"the stores total {total} bytes, over the {self.max_bytes} limit"
                )
                return {
                    "status": "FAILED",
                    "export_id": export_id,
                    "error": self._failed[export_id],
                }
            self._running = export_id
            self._started_at[export_id] = self.clock()
        if self.synchronous:
            self._run(export_id)
        else:
            threading.Thread(
                target=self._run, args=(export_id,), daemon=True, name=f"export-{export_id}"
            ).start()
        return {"status": "RUNNING", "export_id": export_id}

    def _run(self, export_id: str) -> None:
        try:
            export_stores(
                {k: v for k, v in self.stores.items()},  # noqa: C416 - widen the value type
                self._dir(export_id),
                export_id,
                {**self.identity, **MARKER},
                allow_same_filesystem=self.same_fs,
            )
        except Exception as exc:  # noqa: BLE001 - reported as FAILED (message only, no traceback to the caller)
            self._failed[export_id] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            with self._lock:
                self._running = None

    def _owned(self, d: Path) -> bool:
        """A real directory directly under the root whose receipt says this service made it."""
        try:
            if d.is_symlink() or not d.is_dir() or d.resolve().parent != self.root.resolve():
                return False
            receipt = json.loads((d / RECEIPT).read_bytes())
            return all(receipt.get("source", {}).get(k) == v for k, v in MARKER.items())
        except (OSError, ValueError):
            return False

    def _purge_stale(self) -> None:
        """Remove only exports this service created, once they are older than `max_age_s`."""
        for p in self.root.iterdir():
            if self._owned(p) and self.clock() - p.stat().st_mtime > self.max_age_s:
                shutil.rmtree(p, ignore_errors=True)

    def status(self, export_id: str) -> dict[str, Any]:
        d = self._dir(export_id)
        if self._running == export_id:
            return {"status": "RUNNING", "export_id": export_id}
        if export_id in self._failed and not d.exists():
            return {"status": "FAILED", "export_id": export_id, "error": self._failed[export_id]}
        if not self._owned(d):
            raise ExportUnknown("no such export")
        claims = self._claims.get(export_id)
        if claims is None:
            claims = {
                p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
                for p in sorted(d.iterdir())
                if p.is_file() and not p.is_symlink()
            }
            self._claims[export_id] = claims
        return {
            "status": "READY",
            "export_id": export_id,
            "files": claims,
            "identity": self.identity,
        }

    def open_file(self, export_id: str, name: str) -> tuple[Path, int, str]:
        st = self.status(export_id)
        if st["status"] != "READY" or name not in st["files"]:
            raise ExportUnknown("no such file")
        path = self._dir(export_id) / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != self._dir(export_id).resolve()
        ):
            raise ExportUnknown("no such file")
        return path, st["files"][name]["bytes"], st["files"][name]["sha256"]

    def delete(self, export_id: str) -> bool:
        d = self._dir(export_id)
        with self._lock:
            if self._running == export_id:
                raise ExportBusy("the export is still running")
            self._claims.pop(export_id, None)
            self._failed.pop(export_id, None)
            if self._owned(d):
                shutil.rmtree(d)
                return True
            if d.exists() or d.is_symlink():
                raise ExportUnknown(
                    "not an export of this service"
                )  # never delete what we didn't create
        return False
