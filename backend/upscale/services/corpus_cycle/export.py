"""Read-only export of UpScale's SQLite stores in the Replay V1 P1 export format.

Standard library only and self-contained (it imports nothing from UpScale), so the exact file
can be piped into the production container without deploying anything:

    railway ssh -- python3 - --dest /tmp/upscale-export-<id> --export-id <id> \\
        evidence=/data/evidence.sqlite3 scout=/data/scout.sqlite3 ... < export.py

(see ``docs/dream-roadmap/phase-1.6-export-runbook.md``; NOT run against production without
explicit approval).

For each source: open it ``mode=ro`` with ``PRAGMA query_only = ON`` (the source file is never
opened writable), take ONE single-step online backup (`sqlite3.Connection.backup`, pages=-1:
one read transaction, a consistent point-in-time copy even while production writes) into a
private staging directory, switch the COPY to ``journal_mode=DELETE``, run ``PRAGMA
integrity_check`` on the copy, and record its size, SHA-256 and backup start / finish times.
Then ``export_receipt.json`` (receipt_version 1, exactly what `replay_v1.freeze` checks) is
written and the staging directory is renamed into place: a failure at any point removes the
staging directory and leaves no partial export. The destination must not exist and must not
be inside a source's directory tree; nothing is ever written next to a source.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECEIPT = "export_receipt.json"
RECEIPT_VERSION = 1
# Tables each known store must hold (checked on the COPY, before anything is published): an
# export missing research data fails in the container, not after the download.
REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "evidence": ("evidence_records",),
    "scout": (
        "scout_tokens",
        "scout_snapshots",
        "scout_outcome_observations",
        "scout_outcome_horizons",
    ),
    "safety": (
        "safety_snapshots",
    ),  # store names are arbitrary: the freeze finds stores by their tables
    "opportunity": ("opportunity_decisions", "opportunity_source_refs"),
    "orchestrator": ("orch_jobs",),
    "shadow": ("shadow_checkpoints",),
}
SPACE_FACTOR = 1.2  # free space needed: sources x this + SPACE_MARGIN
SPACE_MARGIN = 64 * 1024 * 1024


class ExportError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _tool_sha256() -> str | None:
    """This file's SHA-256 when it runs from a file (None when piped through stdin)."""
    try:
        return _sha256(Path(__file__).resolve())
    except (NameError, OSError):
        return None


def export_stores(sources: dict[str, str | Path], dest: str | Path, export_id: str,
                  source_info: dict[str, str] | None = None, *,
                  allow_same_filesystem: bool = False) -> dict[str, Any]:  # fmt: skip
    """Export every named source (name -> path) into the new directory `dest`.

    `dest` must be on another filesystem than every source (in production: the container's
    ``/tmp``, never the ``/data`` volume), unless `allow_same_filesystem` (local rehearsals
    only)."""
    if not sources:
        raise ExportError("name at least one source store")
    if not export_id.strip():
        raise ExportError("an export id is required")
    target = Path(dest).expanduser().absolute()
    if target.exists():
        raise ExportError(f"{target} already exists: an export is never overwritten")
    resolved: dict[str, Path] = {}
    for name, raw in sorted(sources.items()):
        path = Path(raw).expanduser().resolve()
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ExportError(f"store name {name!r}: letters, digits, '-' and '_' only")
        if not path.is_file():
            raise ExportError(f"no source store at {path}")
        if target == path.parent or path.parent in target.parents:
            raise ExportError(f"{target} is inside the directory of source {path}")
        resolved[name] = path
    target.parent.mkdir(parents=True, exist_ok=True)
    devices = {p.stat().st_dev for p in resolved.values()}
    if not allow_same_filesystem and target.parent.stat().st_dev in devices:
        raise ExportError(f"{target.parent} is on the same filesystem as a source: export to "
                          "another filesystem (e.g. /tmp), or pass allow_same_filesystem for a "
                          "local rehearsal")  # fmt: skip
    needed = int(sum(p.stat().st_size for p in resolved.values()) * SPACE_FACTOR) + SPACE_MARGIN
    free = shutil.disk_usage(target.parent).free
    if free < needed:
        raise ExportError(f"{target.parent} has {free} bytes free; the export needs {needed}")
    staging = target.parent / f".{target.name}.staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(mode=0o700)
    files = []
    try:
        for name, path in resolved.items():
            copy = staging / f"{name}.sqlite3"
            src = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
            src.execute("PRAGMA query_only = ON")
            dst = sqlite3.connect(copy)
            try:
                started = _now()
                src.backup(dst, pages=-1)
                finished = _now()
                dst.execute("PRAGMA journal_mode = DELETE")
                check = dst.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                dst.close()
                src.close()
            if check != "ok":
                raise ExportError(f"{name}: integrity_check of the copy is {check!r}")
            verify = sqlite3.connect(f"{copy.as_uri()}?mode=ro", uri=True)
            try:
                tables = {r[0] for r in verify.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'")}  # fmt: skip
            finally:
                verify.close()
            absent = [t for t in REQUIRED_TABLES.get(name, ()) if t not in tables]
            if absent:
                raise ExportError(f"{name}: the store lacks {', '.join(absent)}: wrong file or "
                                  "an incomplete store")  # fmt: skip
            for extra in (
                copy.with_name(copy.name + "-wal"),
                copy.with_name(copy.name + "-journal"),
            ):
                if extra.exists():
                    raise ExportError(f"{name}: the copy left {extra.name} behind")
            os.chmod(copy, 0o400)
            files.append({"filename": copy.name, "bytes": copy.stat().st_size,
                          "sha256": _sha256(copy), "backup_started_at": started.isoformat(),
                          "backup_finished_at": finished.isoformat()})  # fmt: skip
        receipt = {"receipt_version": RECEIPT_VERSION, "export_id": export_id,
                   "source": {"tool": "upscale.services.corpus_cycle.export",
                              "tool_sha256": _tool_sha256() or "unknown (run from stdin)",
                              "copy_method": "sqlite3 online backup (pages=-1) of each source "
                              "opened mode=ro + query_only; copy switched to journal_mode=DELETE",
                              **(source_info or {})},
                   "files": files}  # fmt: skip
        (staging / RECEIPT).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="read-only export of UpScale stores (P1 format)")
    p.add_argument("--dest", required=True, help="NEW directory for the export")
    p.add_argument("--export-id", required=True)
    p.add_argument("--source-info", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument(
        "--allow-same-filesystem",
        action="store_true",
        help="local rehearsals only: never on production",
    )
    p.add_argument("stores", nargs="+", metavar="NAME=PATH")
    args = p.parse_args(argv)
    try:
        sources = dict(s.split("=", 1) for s in args.stores)
        info = dict(s.split("=", 1) for s in args.source_info)
        t0 = time.monotonic()
        receipt = export_stores(
            sources,
            args.dest,
            args.export_id,
            info,
            allow_same_filesystem=args.allow_same_filesystem,
        )
    except (ExportError, ValueError, sqlite3.Error, OSError) as exc:
        print(f"export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"dest": args.dest, "seconds": round(time.monotonic() - t0, 1),
                      "files": receipt["files"]}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
