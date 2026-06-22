"""End-to-end tests for the CLI → daemon subprocess flow.

These tests start a real daemon subprocess via ``start_daemon()`` and interact
with it through the per-request client functions, mirroring how ``ccc index`` /
``ccc search`` actually work.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import make_test_user_settings

from cocoindex_code import client
from cocoindex_code._version import __version__
from cocoindex_code.client import start_daemon, stop_daemon
from cocoindex_code.daemon import daemon_socket_path
from cocoindex_code.settings import (
    default_project_settings,
    save_project_settings,
    save_user_settings,
)

SAMPLE_PY = '''\
"""Sample module."""

def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
'''

# Port for the daemon-embedded streamable-HTTP MCP server under test.
MCP_PORT = 8799


@pytest.fixture(scope="module")
def e2e_daemon() -> Iterator[tuple[str, Path]]:
    """Start a real daemon subprocess and return (sock_path, project_dir).

    Uses COCOINDEX_CODE_DIR env var so the subprocess uses the temp directory.
    The embedded HTTP MCP server is enabled (COCOINDEX_CODE_MCP_PORT) and pinned
    to ``project_dir`` so ``test_mcp_http_search`` can exercise the real
    ``/mcp`` endpoint.
    """
    # Use a short temp dir to stay within AF_UNIX path limit
    base_dir = Path(tempfile.mkdtemp(prefix="ccc_e2e_"))
    project_dir = base_dir / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text(SAMPLE_PY)

    # Set env vars BEFORE calling any daemon/settings functions
    saved_env = {
        k: os.environ.get(k)
        for k in (
            "COCOINDEX_CODE_DIR",
            "COCOINDEX_CODE_MCP_PORT",
            "COCOINDEX_CODE_MCP_HOST",
            "COCOINDEX_CODE_ROOT_PATH",
        )
    }
    os.environ["COCOINDEX_CODE_DIR"] = str(base_dir)
    os.environ["COCOINDEX_CODE_MCP_PORT"] = str(MCP_PORT)
    os.environ["COCOINDEX_CODE_MCP_HOST"] = "127.0.0.1"
    os.environ["COCOINDEX_CODE_ROOT_PATH"] = str(project_dir)

    try:
        save_user_settings(make_test_user_settings())
        save_project_settings(project_dir, default_project_settings())

        proc = start_daemon()

        sock_path = daemon_socket_path()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log = base_dir / "daemon.log"
                log_content = log.read_text() if log.exists() else "(no log)"
                raise RuntimeError(f"Daemon process exited early.\nLog:\n{log_content}")
            if os.path.exists(sock_path):
                break
            time.sleep(0.2)
        else:
            log = base_dir / "daemon.log"
            log_content = log.read_text() if log.exists() else "(no log)"
            raise TimeoutError(f"Daemon did not start.\nLog:\n{log_content}")

        yield sock_path, project_dir
    finally:
        stop_daemon()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_daemon_subprocess_starts(e2e_daemon: tuple[str, Path]) -> None:
    """The daemon should be reachable via a fresh connection after start_daemon()."""
    resp = client.daemon_status()
    assert resp.version == __version__


def test_index_and_search_via_client(e2e_daemon: tuple[str, Path]) -> None:
    """Index a project and search via the client, same as ccc index / ccc search."""
    _, project_dir = e2e_daemon

    resp = client.index(str(project_dir))
    assert resp.success

    status = client.project_status(str(project_dir))
    assert status.total_chunks > 0
    assert status.total_files > 0

    search_resp = client.search(str(project_dir), query="fibonacci")
    assert search_resp.success
    assert len(search_resp.results) > 0
    assert "main.py" in search_resp.results[0].file_path


def test_compact_via_client(e2e_daemon: tuple[str, Path]) -> None:
    """`ccc compact` reclaims space and leaves the index searchable.

    The store must already exist (a prior test in this module-scoped daemon
    indexed it). Compaction should succeed, not grow the store, and the table
    must still answer queries afterwards (i.e. the prune didn't corrupt it).
    """
    _, project_dir = e2e_daemon

    # Ensure there is an index to compact (idempotent if a prior test ran it).
    client.index(str(project_dir))

    resp = client.compact(str(project_dir))
    assert resp.ok
    assert resp.bytes_after <= resp.bytes_before

    # Still searchable after the aggressive prune.
    search_resp = client.search(str(project_dir), query="fibonacci")
    assert search_resp.success
    assert len(search_resp.results) > 0


def _mcp_post(
    body: dict[str, Any], session_id: str | None = None
) -> tuple[dict[str, Any], str | None]:
    """POST a JSON-RPC message to the embedded ``/mcp`` endpoint.

    The streamable-HTTP transport replies with an SSE frame (``data: {...}``);
    this unwraps it and returns the parsed JSON plus the ``Mcp-Session-Id``.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:{MCP_PORT}/mcp",
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
    # Notifications get a 202 with an empty body — nothing to parse.
    if not raw.strip():
        return {}, sid
    payload = raw
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            break
    return json.loads(payload), sid


def test_mcp_http_search(e2e_daemon: tuple[str, Path]) -> None:
    """The daemon-embedded streamable-HTTP MCP server answers a real search.

    Exercises the full transport — initialize handshake, tools/list, tools/call
    — against the in-process server, confirming the ``search`` tool reaches the
    project registry without any external proxy or socket round-trip.
    """
    _, project_dir = e2e_daemon
    client.index(str(project_dir))  # ensure there is something to find

    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    }
    resp, sid = _mcp_post(init)
    assert resp["result"]["serverInfo"]["name"] == "cocoindex-code"
    assert sid is not None

    # Required initialized notification before issuing requests.
    _mcp_post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)

    tools, _ = _mcp_post({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, sid)
    assert "search" in [t["name"] for t in tools["result"]["tools"]]

    call, _ = _mcp_post(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "fibonacci recursion", "limit": 3},
            },
        },
        sid,
    )
    result = call["result"]
    structured = result.get("structuredContent") or json.loads(result["content"][0]["text"])
    assert structured["success"] is True
    assert structured["total_returned"] >= 1
    assert "main.py" in structured["results"][0]["file_path"]


def test_compact_without_index_is_safe(e2e_daemon: tuple[str, Path]) -> None:
    """Compacting a project that was never indexed returns ok with a message."""
    _, project_dir = e2e_daemon
    fresh = project_dir.parent / "never_indexed"
    fresh.mkdir(exist_ok=True)
    save_project_settings(fresh, default_project_settings())

    resp = client.compact(str(fresh))
    assert resp.ok
    assert resp.message is not None


# ---------------------------------------------------------------------------
# No-settings mode + host_path_mappings wiring
# ---------------------------------------------------------------------------


def test_daemon_starts_in_no_settings_mode_without_global_settings() -> None:
    """Daemon started against an empty COCOINDEX_CODE_DIR should come up without
    creating ``global_settings.yml``. The file stays absent; the handshake reports
    ``mtime=None``. Project requests are rejected with a clear "run `ccc init`" error.
    """
    from cocoindex_code.client import stop_daemon as _stop
    from cocoindex_code.protocol import ProjectStatusRequest, encode_request
    from cocoindex_code.settings import user_settings_path

    base_dir = Path(tempfile.mkdtemp(prefix="ccc_nosettings_"))
    old_env = os.environ.get("COCOINDEX_CODE_DIR")
    os.environ["COCOINDEX_CODE_DIR"] = str(base_dir)

    try:
        assert not user_settings_path().is_file()

        proc = start_daemon()
        sock = daemon_socket_path()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log = base_dir / "daemon.log"
                raise RuntimeError(
                    f"Daemon exited early. Log:\n{log.read_text() if log.exists() else '(none)'}"
                )
            if os.path.exists(sock):
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("Daemon did not start in time")

        # Daemon is up, but global_settings.yml stays absent — no auto-create.
        assert not user_settings_path().is_file()

        # Handshake works and reports mtime=None (no settings yet).
        # Use the lower-level raw handshake so we can inspect the response
        # directly; the high-level client would loop on mtime mismatch.
        from cocoindex_code.client import _raw_connect_and_handshake
        from cocoindex_code.protocol import decode_response

        # _raw_connect_and_handshake does its own handshake read — but it also
        # raises DaemonVersionError when the client-side mtime disagrees. With
        # the file absent on both sides, mtime=None matches, so handshake OK.
        conn = _raw_connect_and_handshake()
        try:
            # Send a project request — should get an ErrorResponse pointing at
            # `ccc init`, not a crash.
            conn.send_bytes(encode_request(ProjectStatusRequest(project_root=str(base_dir))))
            resp = decode_response(conn.recv_bytes())
        finally:
            conn.close()

        from cocoindex_code.protocol import ErrorResponse

        assert isinstance(resp, ErrorResponse)
        assert "ccc init" in resp.message
    finally:
        _stop()
        if old_env is None:
            os.environ.pop("COCOINDEX_CODE_DIR", None)
        else:
            os.environ["COCOINDEX_CODE_DIR"] = old_env


def test_daemon_env_response_includes_host_path_mappings(
    e2e_daemon: tuple[str, Path],
) -> None:
    """``client.daemon_env`` surfaces the parsed COCOINDEX_CODE_HOST_PATH_MAPPING."""
    _, _project_dir = e2e_daemon

    # The session daemon was started without COCOINDEX_CODE_HOST_PATH_MAPPING,
    # so this just verifies the field is exposed on the wire and defaults to empty.
    resp = client.daemon_env()
    assert hasattr(resp, "host_path_mappings")
    assert resp.host_path_mappings == []
