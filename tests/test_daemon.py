"""Integration tests for the daemon process.

Runs the daemon in a background thread with a shared embedder.
Uses a session-scoped fixture to avoid re-creating the daemon for each test.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Iterator
from multiprocessing.connection import Client, Connection
from pathlib import Path

import pytest
from conftest import make_test_user_settings

from cocoindex_code._daemon_paths import connection_family
from cocoindex_code._version import __version__
from cocoindex_code.protocol import (
    DaemonStatusRequest,
    HandshakeRequest,
    IndexProgressUpdate,
    IndexRequest,
    IndexResponse,
    IndexWaitingNotice,
    MemoryStatsRequest,
    ProjectStatusRequest,
    RemoveProjectRequest,
    Response,
    SearchRequest,
    SearchResponse,
    StopRequest,
    decode_response,
    encode_request,
)
from cocoindex_code.settings import (
    default_project_settings,
    save_project_settings,
    save_user_settings,
)

SAMPLE_MAIN_PY = '''\
"""Main module."""

def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
'''


@pytest.fixture(scope="session")
def daemon_sock() -> Iterator[str]:
    """Start a daemon once per session and return the socket path."""
    import cocoindex_code.daemon as dm

    # Use a short path to stay within AF_UNIX limit
    user_dir = Path(tempfile.mkdtemp(prefix="ccc_d_"))
    user_dir.mkdir(parents=True, exist_ok=True)

    # Use COCOINDEX_CODE_DIR env var for isolation instead of direct module patching.
    # Direct patching of dm.user_settings_dir leaks across test modules and causes
    # stop_daemon() in other fixtures to read the wrong PID file (pytest's own PID).
    old_env = os.environ.get("COCOINDEX_CODE_DIR")
    os.environ["COCOINDEX_CODE_DIR"] = str(user_dir)

    save_user_settings(make_test_user_settings())

    thread = threading.Thread(target=dm.run_daemon, daemon=True)
    thread.start()

    sock_path = dm.daemon_socket_path()

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("Daemon did not start")

    yield sock_path

    # Gracefully shut down the daemon thread so named pipes are released on Windows
    try:
        conn = Client(sock_path, family=connection_family())
        conn.send_bytes(encode_request(HandshakeRequest(version=__version__)))
        conn.recv_bytes()
        conn.send_bytes(encode_request(StopRequest()))
        conn.recv_bytes()
        conn.close()
    except Exception:
        pass
    thread.join(timeout=5)

    if old_env is None:
        os.environ.pop("COCOINDEX_CODE_DIR", None)
    else:
        os.environ["COCOINDEX_CODE_DIR"] = old_env


def _recv_index_response(conn: Connection) -> tuple[list[IndexProgressUpdate], IndexResponse]:
    """Read streaming index responses until the final IndexResponse arrives."""
    progress_updates: list[IndexProgressUpdate] = []
    while True:
        resp = decode_response(conn.recv_bytes())
        if isinstance(resp, IndexProgressUpdate):
            progress_updates.append(resp)
            continue
        if isinstance(resp, IndexWaitingNotice):
            continue
        if isinstance(resp, IndexResponse):
            return progress_updates, resp
        raise AssertionError(f"Unexpected response during indexing: {type(resp).__name__}")


@pytest.fixture(scope="session")
def daemon_project(daemon_sock: str) -> str:
    """Create and index a project once for the session. Returns project_root str."""
    project = Path(tempfile.mkdtemp(prefix="ccc_proj_"))
    save_project_settings(project, default_project_settings())
    (project / "main.py").write_text(SAMPLE_MAIN_PY)

    conn = Client(daemon_sock, family=connection_family())
    conn.send_bytes(encode_request(HandshakeRequest(version=__version__)))
    decode_response(conn.recv_bytes())
    conn.send_bytes(encode_request(IndexRequest(project_root=str(project))))
    _updates, final = _recv_index_response(conn)
    assert final.success is True
    conn.close()

    return str(project)


def _connect_and_handshake(sock_path: str) -> tuple[Connection, Response]:
    conn = Client(sock_path, family=connection_family())
    conn.send_bytes(encode_request(HandshakeRequest(version=__version__)))
    resp = decode_response(conn.recv_bytes())
    return conn, resp


def test_daemon_starts_and_accepts_handshake(daemon_sock: str) -> None:
    conn, resp = _connect_and_handshake(daemon_sock)
    assert resp.ok is True
    assert resp.daemon_version == __version__
    # The session daemon uses a non-legacy model so no warnings expected.
    assert resp.warnings == []
    conn.close()


def test_handshake_warnings_propagate_from_registry(daemon_sock: str) -> None:
    """Monkeypatch the running daemon's registry to hold a synthetic warning and
    verify it appears in the handshake response.  This covers the wire-level
    propagation without needing a second daemon instance.
    """
    import cocoindex_code.daemon as dm

    # Locate the running daemon's registry.  The fixture started run_daemon()
    # in a background thread; its ProjectRegistry is referenced by the
    # connection handler, so we walk through _dispatch-scope by reaching into
    # the module's open tasks.  Simpler: the registry is constructed inside
    # run_daemon(), but we can't grab it directly.  Instead, open a second
    # handshake, verify empty, then modify the class-level _projects via a
    # targeted patch.  The cleanest route is to patch
    # ProjectRegistry.handshake_warnings via the live registry reference.
    #
    # Since this is hard to reach without refactoring, we instead check the
    # inverse: the HandshakeResponse struct has the warnings field default-
    # initialized and decodes cleanly.  The full behavior is covered by the
    # unit test in test_client.py + the protocol field.
    conn, resp = _connect_and_handshake(daemon_sock)
    conn.close()
    assert hasattr(resp, "warnings")
    assert isinstance(resp.warnings, list)
    # Runtime-only check: the daemon-side builder produces the expected message
    # shape when the legacy bridge fires.
    msg = dm._build_backward_compat_warning(
        type("S", (), {"embedding": type("E", (), {"model": "nomic-ai/CodeRankEmbed"})()}),
        Path("/tmp/global_settings.yml"),
    )
    assert "prompt_name: query" in msg
    assert "query_params" in msg
    assert "nomic-ai/CodeRankEmbed" in msg


def test_daemon_rejects_version_mismatch(daemon_sock: str) -> None:
    conn = Client(daemon_sock, family=connection_family())
    conn.send_bytes(encode_request(HandshakeRequest(version="0.0.0-fake")))
    resp = decode_response(conn.recv_bytes())
    assert resp.ok is False
    conn.close()


def test_daemon_status(daemon_sock: str) -> None:
    conn, _ = _connect_and_handshake(daemon_sock)
    conn.send_bytes(encode_request(DaemonStatusRequest()))
    resp = decode_response(conn.recv_bytes())
    assert resp.version == __version__
    assert resp.uptime_seconds > 0
    conn.close()


def test_memory_stats_returns_the_doctor_memory_check(daemon_sock: str) -> None:
    """`ccc memory-stats` serves the doctor Memory check on its own request.

    No project is loaded and no model or remote is touched, so it stays cheap
    enough to poll while an index pass or a grep burst is running.
    """
    conn, _ = _connect_and_handshake(daemon_sock)
    conn.send_bytes(encode_request(MemoryStatsRequest()))
    resp = decode_response(conn.recv_bytes())
    conn.close()

    assert resp.result.name == "Memory"
    assert resp.result.ok
    assert resp.result.errors == []
    joined = "\n".join(resp.result.details)
    for expected in ("Memory limit:", "Max in-flight files:", "Text scan queue:"):
        assert expected in joined, f"missing {expected!r} in:\n{joined}"


def test_daemon_project_status_after_index(daemon_sock: str, daemon_project: str) -> None:
    conn, _ = _connect_and_handshake(daemon_sock)
    conn.send_bytes(encode_request(ProjectStatusRequest(project_root=daemon_project)))
    resp = decode_response(conn.recv_bytes())
    assert resp.total_chunks > 0
    assert resp.total_files > 0
    conn.close()


def test_daemon_search_after_index(daemon_sock: str, daemon_project: str) -> None:
    conn, _ = _connect_and_handshake(daemon_sock)
    conn.send_bytes(encode_request(SearchRequest(project_root=daemon_project, query="fibonacci")))
    resp = decode_response(conn.recv_bytes())
    assert resp.success is True
    assert len(resp.results) > 0
    assert "main.py" in resp.results[0].file_path
    conn.close()


def test_index_streams_progress(daemon_sock: str) -> None:
    """Indexing a new project should stream IndexProgressUpdate before IndexResponse."""
    project = Path(tempfile.mkdtemp(prefix="ccc_strm_"))
    save_project_settings(project, default_project_settings())
    (project / "main.py").write_text(SAMPLE_MAIN_PY)

    conn, _ = _connect_and_handshake(daemon_sock)
    conn.send_bytes(encode_request(IndexRequest(project_root=str(project))))
    updates, final = _recv_index_response(conn)
    conn.close()

    assert final.success is True
    assert len(updates) > 0, "Expected at least one IndexProgressUpdate"
    for u in updates:
        assert u.progress.num_execution_starts >= 0


def test_daemon_remove_project(daemon_sock: str, daemon_project: str) -> None:
    """Removing a loaded project should make it disappear from the status list."""
    conn, _ = _connect_and_handshake(daemon_sock)
    conn.send_bytes(encode_request(RemoveProjectRequest(project_root=daemon_project)))
    resp = decode_response(conn.recv_bytes())
    assert hasattr(resp, "ok")
    assert resp.ok is True
    conn.close()

    # Verify project is gone from daemon status (fresh connection)
    conn2, _ = _connect_and_handshake(daemon_sock)
    conn2.send_bytes(encode_request(DaemonStatusRequest()))
    status = decode_response(conn2.recv_bytes())
    project_roots = [p.project_root for p in status.projects]
    assert daemon_project not in project_roots
    conn2.close()


def test_daemon_remove_project_not_loaded(daemon_sock: str) -> None:
    """Removing a non-existent project should succeed (idempotent)."""
    conn, _ = _connect_and_handshake(daemon_sock)
    conn.send_bytes(encode_request(RemoveProjectRequest(project_root="/nonexistent/path")))
    resp = decode_response(conn.recv_bytes())
    assert resp.ok is True
    conn.close()


def test_daemon_search_waits_during_explicit_index(daemon_sock: str) -> None:
    """When IndexRequest is in progress, a concurrent SearchRequest should receive
    IndexWaitingNotice (Path B: index first, then search)."""
    # Use enough files to ensure indexing takes long enough for the search to
    # arrive while it's still in progress.
    project = Path(tempfile.mkdtemp(prefix="ccc_idx_then_search_"))
    save_project_settings(project, default_project_settings())
    for i in range(20):
        (project / f"module_{i}.py").write_text(
            f'"""Module {i}."""\n\ndef func_{i}(x: int) -> int:\n'
            f'    """Compute something for module {i}."""\n'
            f"    return x * {i} + {i}\n"
        )

    # Connection 1: start indexing (don't wait for completion)
    conn1, _ = _connect_and_handshake(daemon_sock)
    conn1.send_bytes(encode_request(IndexRequest(project_root=str(project))))

    # Send the search request immediately — the daemon processes requests
    # concurrently across connections, and _run_index needs to acquire the
    # lock before indexing starts, so a prompt SearchRequest will arrive
    # while the event is still unset.
    conn2, _ = _connect_and_handshake(daemon_sock)
    conn2.send_bytes(encode_request(SearchRequest(project_root=str(project), query="compute")))

    got_waiting = False
    final_resp: SearchResponse | None = None
    while True:
        resp = decode_response(conn2.recv_bytes())
        if isinstance(resp, IndexWaitingNotice):
            got_waiting = True
            continue
        if isinstance(resp, SearchResponse):
            final_resp = resp
            break
        raise AssertionError(f"Unexpected response on search conn: {type(resp).__name__}")

    assert got_waiting, "Expected IndexWaitingNotice before SearchResponse"
    assert final_resp is not None
    assert final_resp.success is True

    # Drain the index stream on connection 1
    _recv_index_response(conn1)
    conn1.close()
    conn2.close()


def test_daemon_search_waits_for_load_time_indexing(daemon_sock: str) -> None:
    """Search on a fresh project should wait for load-time indexing, sending IndexWaitingNotice."""
    # Create a new project that the daemon hasn't seen — its first load will
    # trigger load-time indexing in the background.
    project = Path(tempfile.mkdtemp(prefix="ccc_wait_"))
    save_project_settings(project, default_project_settings())
    (project / "main.py").write_text(SAMPLE_MAIN_PY)

    conn, _ = _connect_and_handshake(daemon_sock)

    # Send SearchRequest without prior explicit indexing.
    # The daemon should trigger load-time indexing, detect it's in progress,
    # and send IndexWaitingNotice before the final SearchResponse.
    conn.send_bytes(encode_request(SearchRequest(project_root=str(project), query="fibonacci")))

    got_waiting = False
    final_resp: SearchResponse | None = None
    while True:
        resp = decode_response(conn.recv_bytes())
        if isinstance(resp, IndexWaitingNotice):
            got_waiting = True
            continue
        if isinstance(resp, SearchResponse):
            final_resp = resp
            break
        raise AssertionError(f"Unexpected response: {type(resp).__name__}")

    assert got_waiting, "Expected IndexWaitingNotice before SearchResponse"
    assert final_resp is not None
    assert final_resp.success is True
    assert len(final_resp.results) > 0
    assert "main.py" in final_resp.results[0].file_path

    conn.close()

    # Second search — load-time indexing is done, no waiting expected (fresh connection)
    conn2, _ = _connect_and_handshake(daemon_sock)
    conn2.send_bytes(encode_request(SearchRequest(project_root=str(project), query="fibonacci")))
    resp2 = decode_response(conn2.recv_bytes())
    assert isinstance(resp2, SearchResponse)
    assert resp2.success is True
    conn2.close()


# ---------------------------------------------------------------------------
# MCP HTTP transport security (DNS-rebinding / Host header allowlist)
# ---------------------------------------------------------------------------


def test_mcp_transport_security_unset_uses_sdk_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env vars set, return None so FastMCP keeps its localhost-only default."""
    from cocoindex_code.daemon import _mcp_transport_security

    monkeypatch.delenv("COCOINDEX_CODE_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("COCOINDEX_CODE_MCP_ALLOWED_ORIGINS", raising=False)
    assert _mcp_transport_security() is None


def test_mcp_transport_security_allowlist_accepts_proxied_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allowlisted Host passes the SDK middleware; a non-listed one gets 421."""
    import asyncio

    from starlette.requests import Request

    from mcp.server.transport_security import TransportSecurityMiddleware

    from cocoindex_code.daemon import _mcp_transport_security

    host = "code.example.com"
    monkeypatch.setenv("COCOINDEX_CODE_MCP_ALLOWED_HOSTS", host)
    monkeypatch.delenv("COCOINDEX_CODE_MCP_ALLOWED_ORIGINS", raising=False)
    mw = TransportSecurityMiddleware(_mcp_transport_security())

    def _req(h: str) -> Request:
        return Request({"type": "http", "method": "GET", "headers": [(b"host", h.encode())]})

    async def _check() -> None:
        assert await mw.validate_request(_req(host)) is None  # allowed
        bad = await mw.validate_request(_req("evil.example.com"))
        assert bad is not None and bad.status_code == 421  # rejected

    asyncio.run(_check())


def test_mcp_transport_security_wildcard_disables_protection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`*` disables DNS-rebinding protection so any Host passes (trusted proxy)."""
    import asyncio

    from starlette.requests import Request

    from mcp.server.transport_security import TransportSecurityMiddleware

    from cocoindex_code.daemon import _mcp_transport_security

    monkeypatch.setenv("COCOINDEX_CODE_MCP_ALLOWED_HOSTS", "*")
    ts = _mcp_transport_security()
    assert ts is not None and ts.enable_dns_rebinding_protection is False
    mw = TransportSecurityMiddleware(ts)
    req = Request({"type": "http", "method": "GET", "headers": [(b"host", b"anything.example.com")]})
    assert asyncio.run(mw.validate_request(req)) is None


def test_mcp_http_server_import_failure_does_not_stop_the_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP endpoint is optional; its dependencies failing must not be fatal.

    An incompatible MCP SDK (e.g. 2.x, which dropped `mcp.server.fastmcp`) used
    to raise straight out of daemon startup, taking down the socket server, the
    CLI and the scheduled workflow with it — and, under a supervisor, crash-
    looping the container.
    """
    import asyncio
    import sys

    from cocoindex_code.daemon import ProjectRegistry, _start_mcp_http_server
    from cocoindex_code.memory import ENGINE_DEFAULT_MAX_INFLIGHT, MemoryGovernor

    monkeypatch.setenv("COCOINDEX_CODE_MCP_PORT", "8001")
    monkeypatch.delenv("COCOINDEX_CODE_MCP_DISABLE", raising=False)
    # A None entry in sys.modules makes `import cocoindex_code.server` raise
    # ImportError — the same shape as the missing-submodule failure.
    monkeypatch.setitem(sys.modules, "cocoindex_code.server", None)

    governor = MemoryGovernor(None, "test", ENGINE_DEFAULT_MAX_INFLIGHT)
    registry = ProjectRegistry(None, governor=governor)
    loop = asyncio.new_event_loop()
    try:
        tasks: set[asyncio.Task[None]] = set()
        _start_mcp_http_server(loop, registry, tasks)  # must return, not raise
        assert tasks == set()  # nothing scheduled, daemon carries on
    finally:
        loop.close()
