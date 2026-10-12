"""The read-only export API (standalone: only needs the production app's own packages).

Runs the real router on loopback with minimal SQLite stores that satisfy the export tool's required
tables, and attacks it: forged / replayed / stale signatures, wrong host, oversized bodies, hostile
ids and file names, symlinks, foreign directories, races and size limits."""

import hashlib
import json
import os
import secrets
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI

from upscale import preservation_api as api
from upscale.services.corpus_cycle import export as export_mod
from upscale.services.corpus_cycle.export import REQUIRED_TABLES
from upscale.services.corpus_cycle.export_service import ExportManager, resolve_stores, sign

TOKEN = "t" * 40
EID = "prod-20261001t0400z"
EXPORT = json.dumps({"export_id": EID}).encode()
BASE = "/internal/preservation/v1"


def make_stores(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True)
    out = {}
    for name, tables in REQUIRED_TABLES.items():
        p = base / f"{name}.sqlite3"
        c = sqlite3.connect(p)
        for t in tables:
            c.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, v TEXT)")
            c.execute(f"INSERT INTO {t} (v) VALUES ('x')")
        c.commit()
        c.close()
        out[name] = p
    return out


class Served:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp = tmp
        self.stores = make_stores(tmp / "stores")
        self.root = tmp / "exports"
        self.mgr = ExportManager(
            self.root,
            self.stores,
            {"environment": "production", "service": "UpScale"},
            allow_same_filesystem=True,
        )
        monkeypatch.setattr(api, "_manager", self.mgr)
        monkeypatch.setattr(api, "_seen", {})
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        for k, v in (
            ("UPSCALE_PRESERVATION_TOKEN", TOKEN),
            ("UPSCALE_PRESERVATION_ALLOWED_HOSTS", f"127.0.0.1:{self.port}"),
            ("UPSCALE_PRESERVATION_ENVIRONMENT", "production"),
            ("RAILWAY_ENVIRONMENT_NAME", "production"),
        ):
            monkeypatch.setenv(k, v)
        app = FastAPI()
        app.include_router(api.router)
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level=None, log_config=None)
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(5)

    def req(self, method: str, path: str, body: bytes | None = None, *, token: str | None = TOKEN, host: str | None = None,
            ts: int | None = None, nonce: str | None = None, sign_path: str | None = None, sign_body: bytes | None = None,
            headers: dict[str, str] | None = None) -> tuple[int, bytes]:  # fmt: skip
        full = f"{BASE}/{path}"
        h: dict[str, str] = {"Content-Type": "application/json"}
        if token is not None:
            t = int(time.time()) if ts is None else ts
            n = nonce or secrets.token_hex(16)
            h.update({"X-Preserver-Timestamp": str(t), "X-Preserver-Nonce": n,
                      "X-Preserver-Signature": sign(token, t, n, method, sign_path or full, (body or b"") if sign_body is None else sign_body)})  # fmt: skip
        if host:
            h["Host"] = host
        h.update(headers or {})
        r = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{full}", data=body, method=method, headers=h
        )
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def ready(self, eid: str = EID) -> None:
        self.req("POST", "exports", json.dumps({"export_id": eid}).encode())
        for _ in range(200):
            if self.mgr.status(eid)["status"] in ("READY", "FAILED"):
                return
            time.sleep(0.05)


@pytest.fixture
def svc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(export_mod, "_now", export_mod._now)
    s = Served(tmp_path, monkeypatch)
    yield s
    s.stop()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_a_signed_export_round_trip_is_byte_exact_and_leaves_the_sources_untouched(
    svc: Served,
) -> None:
    before = {n: sha(p) for n, p in svc.stores.items()}
    svc.ready()
    code, body = svc.req("GET", f"exports/{EID}")
    st = json.loads(body)
    assert code == 200 and st["status"] == "READY" and st["identity"]["environment"] == "production"
    assert set(st["files"]) == {f"{n}.sqlite3" for n in svc.stores} | {"export_receipt.json"}
    for name, f in st["files"].items():
        code, data = svc.req("GET", f"exports/{EID}/files/{name}")
        assert (
            code == 200
            and hashlib.sha256(data).hexdigest() == f["sha256"]
            and len(data) == f["bytes"]
        )
    assert svc.req("DELETE", f"exports/{EID}")[0] == 200 and not list(svc.root.iterdir())
    assert before == {n: sha(p) for n, p in svc.stores.items()}
    assert not [
        p for p in svc.stores["evidence"].parent.iterdir() if p.suffix in ("-wal", "-journal")
    ]


@pytest.mark.parametrize("kw", [
    {"token": None}, {"token": "x" * 40}, {"token": TOKEN[:-1]}, {"host": "example.com"},
    {"ts": 1}, {"ts": int(time.time()) + 3600}, {"sign_path": f"{BASE}/exports/other"},
    {"sign_body": b'{"export_id": "prod-29999999t0400z"}'}, {"nonce": "zz"}, {"headers": {"X-Preserver-Nonce": ""}},
    {"headers": {"X-Preserver-Signature": ""}}, {"headers": {"X-Preserver-Timestamp": "1e9"}},
])  # fmt: skip
def test_every_unauthenticated_or_forged_request_is_an_identical_empty_404(
    svc: Served, kw: dict[str, Any]
) -> None:
    refs = [
        svc.req("POST", "exports", EXPORT, **kw),
        svc.req("GET", f"exports/{EID}", **kw),
        svc.req("DELETE", f"exports/{EID}", **kw),
    ]
    nothing = svc.req("GET", "exports/never-existed-1")
    assert all(code == 404 for code, _ in refs)
    assert (
        len({body for _, body in refs} | {nothing[1]}) == 1
    )  # no oracle: same body as a route that doesn't exist
    assert not list(svc.root.iterdir()) and svc.mgr._running is None


def test_ping_proves_authentication_and_identity_without_exporting(svc: Served) -> None:
    code, body = svc.req("GET", "ping")
    doc = json.loads(body)
    assert (
        code == 200
        and doc["ok"]
        and doc["identity"]["environment"] == "production"
        and "evidence" in doc["stores"]
    )
    assert not list(svc.root.iterdir())  # nothing was exported
    assert svc.req("GET", "ping", token="x" * 40)[0] == 404


def test_a_signature_made_for_one_method_or_path_cannot_authorize_another(svc: Served) -> None:
    t, n = int(time.time()), "ab" * 8
    sig = sign(TOKEN, t, n, "GET", f"{BASE}/exports", EXPORT)
    hdr = {"X-Preserver-Signature": sig, "X-Preserver-Timestamp": str(t), "X-Preserver-Nonce": n}
    assert svc.req("POST", "exports", EXPORT, token=None, headers=hdr)[0] == 404


def test_a_replayed_request_is_refused(svc: Served) -> None:
    svc.ready()
    t, n = int(time.time()), "cd" * 8
    hdr = {"X-Preserver-Signature": sign(TOKEN, t, n, "GET", f"{BASE}/exports/{EID}", b""),
           "X-Preserver-Timestamp": str(t), "X-Preserver-Nonce": n}  # fmt: skip
    assert svc.req("GET", f"exports/{EID}", token=None, headers=hdr)[0] == 200
    assert svc.req("GET", f"exports/{EID}", token=None, headers=hdr)[0] == 404


def test_identical_concurrent_requests_with_fresh_nonces_are_all_served(svc: Served) -> None:
    out: list[int] = []
    ts = [
        threading.Thread(target=lambda: out.append(svc.req("POST", "exports", EXPORT)[0]))
        for _ in range(6)
    ]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert out == [202] * 6
    svc.ready()
    assert [p.name for p in svc.root.iterdir()] == [EID]  # one export, not six


def test_oversized_bodies_are_refused_unread(svc: Served) -> None:
    big = b'{"export_id": "' + b"a" * 5000 + b'"}'
    assert svc.req("POST", "exports", big)[0] == 404 and not list(svc.root.iterdir())


@pytest.mark.parametrize(
    "bad",
    ["../etc", "..", "a/b", "A_UPPER", "x", "a" * 81, "%2e%2e%2fetc", ".hidden", "-lead", "a\\b"],
)
def test_hostile_ids_never_reach_the_filesystem(svc: Served, bad: str) -> None:
    body = json.dumps({"export_id": bad}).encode()
    assert svc.req("POST", "exports", body)[0] in (404, 422)
    enc = bad.replace("/", "%2F").replace("\\", "%5C")
    for m in ("GET", "DELETE"):
        assert svc.req(m, f"exports/{enc}")[0] in (404, 422)
    assert not list(svc.root.iterdir())


def test_file_names_are_taken_from_the_exports_own_listing(svc: Served) -> None:
    svc.ready()
    for name in (
        "..%2F..%2Fetc%2Fpasswd",
        "evil.sqlite3",
        "%2e%2e",
        ".hidden",
        "export_receipt.json%00",
    ):
        assert svc.req("GET", f"exports/{EID}/files/{name}")[0] == 404
    assert svc.req("GET", f"exports/{EID}/files/export_receipt.json")[0] == 200


def test_delete_removes_only_what_the_service_created(svc: Served) -> None:
    foreign = svc.root / "prod-20261002t0400z"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep")
    outside = svc.tmp / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep too")
    (svc.root / "prod-20261003t0400z").symlink_to(outside)
    for eid in ("prod-20261002t0400z", "prod-20261003t0400z"):
        assert svc.req("DELETE", f"exports/{eid}")[0] == 404
    assert (foreign / "precious.txt").exists() and (outside / "precious.txt").exists()
    svc.ready()
    assert svc.req("DELETE", f"exports/{EID}")[0] == 200 and (foreign / "precious.txt").exists()


def test_stale_purge_spares_foreign_directories(svc: Served) -> None:
    foreign = svc.root / "prod-20261002t0400z"
    foreign.mkdir()
    old = time.time() - 10 * 3600
    os.utime(foreign, (old, old))
    svc.ready()
    assert foreign.exists()


def test_a_planted_symlink_is_neither_listed_nor_served(svc: Served) -> None:
    svc.ready()
    secret = svc.tmp / "secret.txt"
    secret.write_text("TOP SECRET")
    (svc.root / EID / "evil.sqlite3").symlink_to(secret)
    svc.mgr._claims.clear()
    assert "evil.sqlite3" not in svc.mgr.status(EID)["files"]
    code, body = svc.req("GET", f"exports/{EID}/files/evil.sqlite3")
    assert code == 404 and b"TOP SECRET" not in body


def test_a_second_export_is_refused_while_one_runs_and_the_size_limit_holds(svc: Served) -> None:
    svc.mgr._running = EID
    assert svc.req("POST", "exports", b'{"export_id": "prod-20261002t0400z"}')[0] == 409
    svc.mgr._running = None
    svc.mgr.max_bytes = 10
    code, body = svc.req("POST", "exports", EXPORT)
    assert code == 202 and json.loads(body)["status"] == "FAILED" and not list(svc.root.iterdir())


def test_a_missing_store_is_a_reported_failure_not_a_server_error(svc: Served) -> None:
    svc.stores["evidence"].unlink()
    code, body = svc.req("POST", "exports", EXPORT)
    assert code == 202 and json.loads(body)["status"] == "FAILED" and TOKEN.encode() not in body
    assert str(svc.tmp).encode() not in body


def test_the_export_root_is_private_and_other_methods_are_refused(svc: Served) -> None:
    assert (svc.root.stat().st_mode & 0o777) == 0o700
    assert svc.req("PUT", f"exports/{EID}")[0] in (404, 405) and svc.req("PATCH", f"exports/{EID}")[
        0
    ] in (404, 405)


def test_the_api_is_disabled_unless_fully_configured() -> None:
    full = {"UPSCALE_PRESERVATION_TOKEN": "a" * 32, "UPSCALE_PRESERVATION_ALLOWED_HOSTS": "h",
            "UPSCALE_PRESERVATION_ENVIRONMENT": "production", "RAILWAY_ENVIRONMENT_NAME": "production"}  # fmt: skip
    assert api.configured({}) is False and api.configured(full) is True
    assert api.configured({**full, "UPSCALE_PRESERVATION_TOKEN": "a" * 31}) is False
    assert api.configured({**full, "RAILWAY_ENVIRONMENT_NAME": "staging"}) is False
    for drop in (
        "UPSCALE_PRESERVATION_ALLOWED_HOSTS",
        "UPSCALE_PRESERVATION_ENVIRONMENT",
        "RAILWAY_ENVIRONMENT_NAME",
    ):
        assert api.configured({k: v for k, v in full.items() if k != drop}) is False


def test_the_production_app_does_not_expose_the_routes_by_default() -> None:
    from upscale.main import app

    # (newer FastAPI wraps included routers, so the real route table is the OpenAPI document)
    assert not [p for p in app.openapi()["paths"] if "/internal/preservation" in p]


def test_store_paths_come_only_from_the_apps_own_variables(tmp_path: Path) -> None:
    a, b = tmp_path / "a.sqlite3", tmp_path / "b.sqlite3"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    got = resolve_stores(
        {"UPSCALE_EVIDENCE_DB": str(a), "SOMETHING_ELSE": str(b)},
        {"scout": str(b), "shadow": str(tmp_path / "none")},
    )
    assert got == {"evidence": a, "scout": b}


def test_replay_protection_survives_an_app_restart(
    svc: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, n = int(time.time()), "9a" * 8
    hdr = {
        "X-Preserver-Signature": sign(TOKEN, t, n, "GET", f"{BASE}/ping", b""),
        "X-Preserver-Timestamp": str(t),
        "X-Preserver-Nonce": n,
    }
    assert svc.req("GET", "ping", token=None, headers=hdr)[0] == 200
    monkeypatch.setattr(api, "_seen", {})  # a restart forgets every nonce...
    monkeypatch.setattr(
        api, "_started_at", time.time() + 5
    )  # ...but the new process started later than the signature
    assert (
        svc.req("GET", "ping", token=None, headers=hdr)[0] == 404
    )  # so the replay is still refused
    assert (
        svc.req("GET", "ping", ts=int(time.time()) + 5)[0] == 200
    )  # a request signed after the restart works


def test_the_real_app_mounts_the_routes_only_with_the_full_configuration() -> None:
    """Startup behaviour of `upscale.main` itself, in fresh interpreters."""
    import subprocess
    import sys

    if (
        "preservation_router"
        not in (Path(__file__).resolve().parent.parent / "upscale" / "main.py").read_text()
    ):
        pytest.skip("this tree does not mount the API (only the production-app branch does)")
    code = "import upscale.main as m; print(sorted(p for p in m.app.openapi()['paths'] if 'preservation' in p))"
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("UPSCALE_PRESERVATION", "RAILWAY_"))
    }
    cwd = Path(__file__).resolve().parent.parent

    def routes(extra: dict[str, str]) -> str:
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={**base, **extra},
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        )  # noqa: S603
        return out.stdout.strip()

    full = {"UPSCALE_PRESERVATION_TOKEN": "t" * 40, "UPSCALE_PRESERVATION_ALLOWED_HOSTS": "upscale.railway.internal:8080",
            "UPSCALE_PRESERVATION_ENVIRONMENT": "production", "RAILWAY_ENVIRONMENT_NAME": "production"}  # fmt: skip
    assert routes({}) == "[]"
    assert routes({**full, "RAILWAY_ENVIRONMENT_NAME": "staging"}) == "[]"
    assert (
        routes({k: v for k, v in full.items() if k != "UPSCALE_PRESERVATION_ALLOWED_HOSTS"}) == "[]"
    )
    mounted = routes(full)
    assert (
        "/internal/preservation/v1/ping" in mounted
        and "/internal/preservation/v1/exports/{export_id}" in mounted
    )
