"""Tests for the DevLake metrics push (config resolution + write behavior).

The write tests inject a fake DB-API connection so they exercise the real SQL
and transaction logic without needing a live MySQL (or the optional PyMySQL
driver).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from cocoindex_code import metrics
from cocoindex_code.metrics import MetricsConfig
from cocoindex_code.protocol import LanguageStats, ProjectStatusResponse

_METRICS_ENV = [
    metrics.ENV_ENABLED,
    metrics.ENV_HOST,
    metrics.ENV_PORT,
    metrics.ENV_USER,
    metrics.ENV_PASSWORD,
    metrics.ENV_DATABASE,
    metrics.ENV_REPO,
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient metrics env leaks into a test."""
    for name in _METRICS_ENV:
        monkeypatch.delenv(name, raising=False)
    # Reset the one-shot "unconfigured" warning latch between tests.
    monkeypatch.setattr(metrics, "_warned_unconfigured", False)


def _status(
    *,
    chunks: int = 100,
    files: int = 10,
    loc: int = 500,
    languages: dict[str, LanguageStats] | None = None,
    index_exists: bool = True,
) -> ProjectStatusResponse:
    return ProjectStatusResponse(
        indexing=False,
        total_chunks=chunks,
        total_files=files,
        total_loc=loc,
        languages=languages if languages is not None else {},
        index_exists=index_exists,
    )


class _FakeCursor:
    def __init__(self, log: list[tuple[str, str, Any]]) -> None:
        self._log = log
        self.closed = False

    def execute(self, sql: str, params: Any) -> None:
        self._log.append(("execute", sql, params))

    def executemany(self, sql: str, seq: Any) -> None:
        self._log.append(("executemany", sql, list(seq)))

    def close(self) -> None:
        self.closed = True


class _FakeConn:
    def __init__(self) -> None:
        self.log: list[tuple[str, str, Any]] = []
        self.committed = False
        self.closed = False
        self.cursors: list[_FakeCursor] = []

    def cursor(self) -> _FakeCursor:
        cur = _FakeCursor(self.log)
        self.cursors.append(cur)
        return cur

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


def _config(repo_override: str | None = None) -> MetricsConfig:
    return MetricsConfig(
        host="db",
        port=3306,
        user="root",
        password="pw",
        database="devlake",
        repo_override=repo_override,
    )


# --- load_config -----------------------------------------------------------


def test_load_config_disabled_when_unconfigured() -> None:
    assert metrics.load_config() is None


def test_load_config_requires_host_and_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(metrics.ENV_HOST, "db")  # no database
    assert metrics.load_config() is None


def test_load_config_reads_all_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(metrics.ENV_HOST, "db.example")
    monkeypatch.setenv(metrics.ENV_DATABASE, "devlake")
    monkeypatch.setenv(metrics.ENV_PORT, "3307")
    monkeypatch.setenv(metrics.ENV_USER, "coco")
    monkeypatch.setenv(metrics.ENV_PASSWORD, "secret")
    monkeypatch.setenv(metrics.ENV_REPO, "my-repo")

    config = metrics.load_config()
    assert config == MetricsConfig(
        host="db.example",
        port=3307,
        user="coco",
        password="secret",
        database="devlake",
        repo_override="my-repo",
    )


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(metrics.ENV_HOST, "db")
    monkeypatch.setenv(metrics.ENV_DATABASE, "devlake")
    config = metrics.load_config()
    assert config is not None
    assert config.port == 3306
    assert config.user == "root"
    assert config.password == ""
    assert config.repo_override is None


def test_load_config_bad_port_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(metrics.ENV_HOST, "db")
    monkeypatch.setenv(metrics.ENV_DATABASE, "devlake")
    monkeypatch.setenv(metrics.ENV_PORT, "not-a-number")
    config = metrics.load_config()
    assert config is not None
    assert config.port == 3306


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
def test_load_config_explicit_disable(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(metrics.ENV_HOST, "db")
    monkeypatch.setenv(metrics.ENV_DATABASE, "devlake")
    monkeypatch.setenv(metrics.ENV_ENABLED, value)
    assert metrics.load_config() is None


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_load_config_explicit_enable_still_needs_target(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(metrics.ENV_ENABLED, value)
    monkeypatch.setenv(metrics.ENV_HOST, "db")
    monkeypatch.setenv(metrics.ENV_DATABASE, "devlake")
    assert metrics.load_config() is not None


# --- resolve_repo ----------------------------------------------------------


def test_resolve_repo_prefers_override() -> None:
    assert metrics.resolve_repo(_config(repo_override="pinned"), "/path/to/repo") == "pinned"


def test_resolve_repo_falls_back_to_default() -> None:
    assert metrics.resolve_repo(_config(), "/path/to/repo") == "/path/to/repo"


# --- push_status_sync ------------------------------------------------------


def test_push_writes_repo_and_language_rows_with_shared_snapshot() -> None:
    conn = _FakeConn()
    status = _status(
        chunks=100,
        files=10,
        loc=500,
        languages={
            "python": LanguageStats(chunks=80, loc=400),
            "rust": LanguageStats(chunks=20, loc=100),
        },
    )

    ok = metrics.push_status_sync("/repo", status, config=_config(), connect=lambda _c: conn)
    assert ok is True
    assert conn.committed is True
    assert conn.closed is True

    execs = [e for e in conn.log if e[0] == "execute"]
    manys = [e for e in conn.log if e[0] == "executemany"]
    assert len(execs) == 1
    assert len(manys) == 1

    # Repo row.
    _, repo_sql, repo_params = execs[0]
    assert metrics.REPO_STATS_TABLE in repo_sql
    snapshot_id, repo, collected_at, tot_chunks, tot_files, tot_loc = repo_params
    assert repo == "/repo"
    assert (tot_chunks, tot_files, tot_loc) == (100, 10, 500)
    assert isinstance(collected_at, datetime)
    assert isinstance(snapshot_id, str) and len(snapshot_id) == 32

    # Language rows: same snapshot_id and timestamp, one per language.
    _, lang_sql, lang_rows = manys[0]
    assert metrics.LANGUAGE_STATS_TABLE in lang_sql
    assert len(lang_rows) == 2
    for row in lang_rows:
        assert row[0] == snapshot_id
        assert row[1] == "/repo"
        assert row[2] == collected_at
    by_lang = {row[3]: (row[4], row[5]) for row in lang_rows}
    assert by_lang == {"python": (80, 400), "rust": (20, 100)}


def test_push_with_no_languages_skips_language_insert() -> None:
    conn = _FakeConn()
    ok = metrics.push_status_sync("/repo", _status(languages={}), config=_config(), connect=lambda _c: conn)
    assert ok is True
    assert any(e[0] == "execute" for e in conn.log)
    assert not any(e[0] == "executemany" for e in conn.log)


def test_push_uses_repo_override() -> None:
    conn = _FakeConn()
    metrics.push_status_sync(
        "/default", _status(), config=_config(repo_override="pinned"), connect=lambda _c: conn
    )
    repo_params = next(e[2] for e in conn.log if e[0] == "execute")
    assert repo_params[1] == "pinned"


def test_push_noop_when_disabled() -> None:
    # config=None and no env → load_config() returns None → no connection made.
    def _boom(_c: MetricsConfig) -> Any:
        raise AssertionError("connect must not be called when disabled")

    assert metrics.push_status_sync("/repo", _status(), connect=_boom) is False


def test_push_noop_when_index_missing() -> None:
    def _boom(_c: MetricsConfig) -> Any:
        raise AssertionError("connect must not be called when the index is absent")

    ok = metrics.push_status_sync(
        "/repo", _status(index_exists=False), config=_config(), connect=_boom
    )
    assert ok is False


def test_push_swallows_connect_error() -> None:
    def _fail(_c: MetricsConfig) -> Any:
        raise ConnectionError("refused")

    assert metrics.push_status_sync("/repo", _status(), config=_config(), connect=_fail) is False


def test_push_swallows_write_error_without_commit() -> None:
    class _BadCursor(_FakeCursor):
        def execute(self, sql: str, params: Any) -> None:
            raise RuntimeError("write failed")

    class _BadConn(_FakeConn):
        def cursor(self) -> _FakeCursor:
            cur = _BadCursor(self.log)
            self.cursors.append(cur)
            return cur

    conn = _BadConn()
    ok = metrics.push_status_sync("/repo", _status(), config=_config(), connect=lambda _c: conn)
    assert ok is False
    assert conn.committed is False
    assert conn.closed is True  # connection still cleaned up
    assert conn.cursors[0].closed is True  # cursor closed via finally


# --- check_connection_sync -------------------------------------------------


def test_check_connection_ok() -> None:
    conn = _FakeConn()
    assert metrics.check_connection_sync(_config(), connect=lambda _c: conn) is None
    assert conn.closed is True


def test_check_connection_reports_error() -> None:
    def _fail(_c: MetricsConfig) -> Any:
        raise ConnectionError("host down")

    assert metrics.check_connection_sync(_config(), connect=_fail) == "host down"


# --- async wrapper ---------------------------------------------------------


async def test_push_status_async_delegates() -> None:
    conn = _FakeConn()
    ok = await metrics.push_status("/repo", _status(), config=_config(), connect=lambda _c: conn)
    assert ok is True
    assert conn.committed is True
