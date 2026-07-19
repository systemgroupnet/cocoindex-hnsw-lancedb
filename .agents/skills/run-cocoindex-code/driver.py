#!/usr/bin/env python
"""End-to-end smoke driver for the `ccc` CLI + in-daemon MCP server.

Drives the REAL app (not the test suite) through a full lifecycle against a
throwaway sample repo and an isolated daemon:

    index -> search -> mcp-http (initialize/tools/list/tools/call over /mcp)
          -> status -> doctor -> compact -> daemon status -> daemon stop

The daemon is launched with COCOINDEX_CODE_MCP_PORT set, so it serves the
streamable-HTTP MCP server in-process; the driver hits the real `/mcp`
endpoint over HTTP (no proxy, no `ccc mcp` subprocess) to confirm the `search`
tool reaches the project registry.

Everything runs under a private COCOINDEX_CODE_DIR so it never touches a real
daemon, index, or your global settings. Settings are written directly (the
`ccc init` wizard is interactive and can't run headless).

Run it:

    uv run python .claude/skills/run-cocoindex-code/driver.py

Exits 0 on success, non-zero (with the failing step) otherwise. Use
--keep to leave the temp dir in place for inspection.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

KEEP = "--keep" in sys.argv[1:]


def _free_port() -> int:
    """Grab an OS-assigned free TCP port for the daemon's MCP server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _mcp_post(port: int, body: dict, session_id: str | None = None) -> tuple[dict, str | None]:
    """POST one JSON-RPC message to /mcp; unwrap the SSE frame it replies with."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    if session_id:
        req.add_header("Mcp-Session-Id", session_id)
    with urllib.request.urlopen(req, timeout=30) as resp:
        sid = resp.headers.get("Mcp-Session-Id")
        raw = resp.read().decode()
    if not raw.strip():  # notifications get a 202 with an empty body
        return {}, sid
    payload = raw
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            break
    return json.loads(payload), sid


def mcp_http_search(port: int, query: str) -> dict:
    """Full MCP handshake over HTTP, then call the `search` tool. Returns its result.

    Polls initialize until the daemon's HTTP server is accepting connections
    (the daemon binds the port a beat after the socket comes up).
    """
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "driver", "version": "0"},
        },
    }
    sid = None
    deadline = time.monotonic() + 30
    while True:
        try:
            resp, sid = _mcp_post(port, init)
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)
    _mcp_post(port, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    call, _ = _mcp_post(
        port,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": query, "limit": 3}},
        },
        sid,
    )
    result = call["result"]
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])

# Small, fast local model (same one the test suite uses). Cached after the
# first run, so repeated invocations don't re-download.
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

SAMPLE_FILES = {
    "auth.py": (
        '"""User authentication helpers."""\n\n'
        "def authenticate_user(username: str, password: str) -> bool:\n"
        '    """Check a username/password pair against the credential store."""\n'
        "    stored = _lookup_password_hash(username)\n"
        "    return stored is not None and _verify(password, stored)\n"
    ),
    "math_utils.py": (
        '"""Numeric helpers."""\n\n'
        "def calculate_fibonacci(n: int) -> int:\n"
        '    """Return the nth Fibonacci number."""\n'
        "    if n <= 1:\n"
        "        return n\n"
        "    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)\n"
    ),
}


def _ccc_path() -> str:
    """Locate the `ccc` console script next to the running interpreter."""
    name = "ccc.exe" if os.name == "nt" else "ccc"
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ccc")
    if found:
        return found
    raise SystemExit("could not find `ccc` console script — run via `uv run python ...`")


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="ccc_smoke_"))
    ccc_home = base / "home"
    project = base / "sample_repo"
    ccc_home.mkdir()
    project.mkdir()
    for name, content in SAMPLE_FILES.items():
        (project / name).write_text(content)

    # Isolate everything under a private home dir -> unique daemon socket/pipe.
    mcp_port = _free_port()
    env = dict(os.environ)
    env["COCOINDEX_CODE_DIR"] = str(ccc_home)
    # `ccc doctor` (and other rich output) prints U+2500 box-drawing chars that
    # crash on a Windows cp1252 console; force UTF-8 so the driver is portable.
    env["PYTHONUTF8"] = "1"
    # Make the auto-started daemon serve the in-process HTTP MCP server, pinned
    # to our sample repo (HTTP clients carry no cwd, so the root is explicit).
    env["COCOINDEX_CODE_MCP_HOST"] = "127.0.0.1"
    env["COCOINDEX_CODE_MCP_PORT"] = str(mcp_port)
    env["COCOINDEX_CODE_ROOT_PATH"] = str(project)
    env.pop("COCOINDEX_CODE_MCP_DISABLE", None)  # ignore an inherited kill switch

    # Write settings directly (the `ccc init` wizard is interactive).
    from cocoindex_code.settings import (
        EmbeddingSettings,
        UserSettings,
        default_project_settings,
        save_project_settings,
        save_user_settings,
    )

    os.environ["COCOINDEX_CODE_DIR"] = str(ccc_home)  # for the save_* calls below
    save_user_settings(
        UserSettings(embedding=EmbeddingSettings(provider="sentence-transformers", model=EMBED_MODEL))
    )
    save_project_settings(project, default_project_settings())

    ccc = _ccc_path()
    failures: list[str] = []

    def run(step: str, args: list[str], *, expect: list[str] | None = None) -> None:
        print(f"\n=== {step}: ccc {' '.join(args)} ===", flush=True)
        proc = subprocess.run(
            [ccc, *args], cwd=project, env=env, capture_output=True, text=True, timeout=600
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        print(out.rstrip(), flush=True)
        if proc.returncode != 0:
            failures.append(f"{step}: exit {proc.returncode}")
            return
        for needle in expect or []:
            if needle not in out:
                failures.append(f"{step}: missing expected output {needle!r}")

    def run_mcp(step: str, query: str, *, expect_file: str) -> None:
        print(f"\n=== {step}: POST /mcp tools/call search {query!r} (port {mcp_port}) ===", flush=True)
        try:
            result = mcp_http_search(mcp_port, query)
        except Exception as e:
            failures.append(f"{step}: {type(e).__name__}: {e}")
            return
        hits = [r["file_path"] for r in result.get("results", [])]
        print(f"success={result.get('success')} total={result.get('total_returned')} hits={hits}", flush=True)
        if not result.get("success"):
            failures.append(f"{step}: search returned success=False ({result.get('message')})")
        elif not any(expect_file in h for h in hits):
            failures.append(f"{step}: no hit in {expect_file!r} (got {hits})")

    try:
        run("index", ["index"], expect=["Chunks:", "Files:"])
        run("search", ["search", "authentication"], expect=["auth.py"])
        run("search-fib", ["search", "fibonacci number"], expect=["math_utils.py"])
        run_mcp("mcp-http", "user authentication", expect_file="auth.py")
        run("status", ["status"], expect=["Chunks:"])
        run("doctor", ["doctor"], expect=["Model Check (indexing)", "Embedding dimension"])
        run("compact", ["compact"], expect=["Compaction complete.", "Reclaimed:"])
        run("daemon-status", ["daemon", "status"], expect=["Daemon version:"])
    finally:
        # Always stop the isolated daemon, even on failure.
        subprocess.run(
            [ccc, "daemon", "stop"], cwd=project, env=env, capture_output=True, text=True, timeout=60
        )
        if KEEP:
            print(f"\n[kept] {base}")
        else:
            shutil.rmtree(base, ignore_errors=True)

    print("\n" + "=" * 50)
    if failures:
        print("SMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
