"""Radar's private read-only access to other UpScale databases (Scout outcomes / anchors).

Opened ``mode=ro`` with ``PRAGMA query_only``: Radar can't write, create a table or change a
checkpoint in another database even by mistake. A missing file is reported as None, never
created. Radar depends on those databases' stored schema only, never on another
component's code.
"""

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def default_scout_db() -> str:
    return os.getenv("UPSCALE_SCOUT_DB") or str(Path.home() / ".upscale" / "scout.sqlite3")


def connect(path: str | Path | None) -> sqlite3.Connection | None:
    """A read-only connection, or None when the file doesn't exist (never created)."""
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    conn = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def parse_timestamp(raw: str) -> datetime:
    """An ISO-8601 timestamp with a timezone (``Z`` or an offset), as UTC."""
    try:
        value = datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {raw!r} (e.g. 2026-09-30T05:50:00Z)") from exc
    if value.tzinfo is None:
        raise ValueError(f"timestamp {raw!r} has no timezone: add Z or an offset")
    return value.astimezone(UTC)
