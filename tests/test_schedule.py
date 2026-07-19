"""Tests for the daily maintenance workflow config, timing, and git step."""

from __future__ import annotations

import subprocess
from datetime import datetime, time
from pathlib import Path

import pytest

from cocoindex_code import schedule

_SCHEDULE_ENV = [
    schedule.ENV_ENABLED,
    schedule.ENV_TIME,
    schedule.ENV_WORKSPACES,
    schedule.ENV_GIT_PULL_ENABLED,
    schedule.ENV_GIT_USERNAME,
    schedule.ENV_GIT_PASSWORD,
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient schedule env leaks into a test."""
    for name in _SCHEDULE_ENV:
        monkeypatch.delenv(name, raising=False)


# --- config -----------------------------------------------------------------


def test_load_config_defaults() -> None:
    config = schedule.load_config()
    assert config.enabled is True
    assert config.run_time == time(3, 0)
    assert config.workspaces == ()
    assert config.git_pull_enabled is False
    assert config.git_credentials is None


def test_enabled_falsy_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(schedule.ENV_ENABLED, "off")
    assert schedule.load_config().enabled is False


def test_git_pull_truthy_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(schedule.ENV_GIT_PULL_ENABLED, "1")
    assert schedule.load_config().git_pull_enabled is True


def test_git_pull_default_and_garbage_stay_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(schedule.ENV_GIT_PULL_ENABLED, "maybe")
    assert schedule.load_config().git_pull_enabled is False


def test_time_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(schedule.ENV_TIME, "23:45")
    assert schedule.load_config().run_time == time(23, 45)


@pytest.mark.parametrize("bad", ["", "nope", "25:00", "12:60", "12", "12:00:00"])
def test_time_invalid_falls_back(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv(schedule.ENV_TIME, bad)
    assert schedule.load_config().run_time == time(3, 0)


def test_workspaces_parsed_and_deduped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv(schedule.ENV_WORKSPACES, f"{a}, {b} ,{a}")
    workspaces = schedule.load_config().workspaces
    assert set(workspaces) == {a.resolve(), b.resolve()}
    assert len(workspaces) == 2  # deduped


def test_workspaces_skips_nonexistent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    missing = tmp_path / "missing"
    monkeypatch.setenv(schedule.ENV_WORKSPACES, f"{real},{missing}")
    assert schedule.load_config().workspaces == (real.resolve(),)


# --- credentials ------------------------------------------------------------


def test_credentials_require_password(monkeypatch: pytest.MonkeyPatch) -> None:
    # Username alone (no password/token) does not activate credential injection.
    monkeypatch.setenv(schedule.ENV_GIT_USERNAME, "someone")
    assert schedule.load_config().git_credentials is None


def test_credentials_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(schedule.ENV_GIT_USERNAME, "x-access-token")
    monkeypatch.setenv(schedule.ENV_GIT_PASSWORD, "ghp_secret")
    creds = schedule.load_config().git_credentials
    assert creds is not None
    assert creds.username == "x-access-token"
    assert creds.password == "ghp_secret"


def test_credentials_password_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(schedule.ENV_GIT_PASSWORD, "ghp_secret")
    creds = schedule.load_config().git_credentials
    assert creds is not None
    assert creds.username == ""
    assert creds.password == "ghp_secret"


def test_describe_config_masks_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(schedule.ENV_GIT_PASSWORD, "ghp_supersecret")
    summary = schedule.describe_config(schedule.load_config())
    assert "ghp_supersecret" not in summary
    assert "git_creds=set" in summary


def test_credential_git_args_present_only_with_creds() -> None:
    assert schedule._credential_git_args(None) == []
    args = schedule._credential_git_args(
        schedule.GitCredentials("distinct-user-xyz", "distinct-secret-xyz")
    )
    assert args[0] == "-c"
    assert args[1].startswith("credential.helper=")
    # The helper string carries no secret — it reads them from the environment,
    # so nothing sensitive lands in the process argv.
    assert "distinct-user-xyz" not in args[1]
    assert "distinct-secret-xyz" not in args[1]


def test_git_env_injects_credentials() -> None:
    env = schedule._git_env(schedule.GitCredentials("user", "tok"))
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env[schedule._CRED_ENV_USERNAME] == "user"
    assert env[schedule._CRED_ENV_PASSWORD] == "tok"


def test_git_env_no_credentials() -> None:
    env = schedule._git_env(None)
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert schedule._CRED_ENV_PASSWORD not in env


# --- timing -----------------------------------------------------------------


def test_seconds_until_next_run_later_today() -> None:
    now = datetime(2026, 7, 18, 1, 0, 0)
    assert schedule.seconds_until_next_run(now, time(3, 0)) == 2 * 60 * 60


def test_seconds_until_next_run_wraps_to_tomorrow() -> None:
    now = datetime(2026, 7, 18, 4, 0, 0)
    # 03:00 already passed → next is 03:00 tomorrow, 23h away.
    assert schedule.seconds_until_next_run(now, time(3, 0)) == 23 * 60 * 60


def test_seconds_until_next_run_at_target_is_full_day() -> None:
    now = datetime(2026, 7, 18, 3, 0, 0)
    assert schedule.seconds_until_next_run(now, time(3, 0)) == 24 * 60 * 60


def test_seconds_until_next_run_never_below_one() -> None:
    now = datetime(2026, 7, 18, 2, 59, 59, 999000)
    assert schedule.seconds_until_next_run(now, time(3, 0)) >= 1.0


# --- git update -------------------------------------------------------------


def test_git_hard_reset_skips_non_git_dir(tmp_path: Path) -> None:
    result = schedule.git_hard_reset_sync(tmp_path)
    assert result.status == "skipped"
    assert "not a git repository" in result.message


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_git_hard_reset_error_without_upstream(tmp_path: Path) -> None:
    # A repo with a commit but no upstream/remote → the reset step reports an
    # actionable error rather than raising.
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("hi")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    result = schedule.git_hard_reset_sync(repo)
    assert result.status == "error"
    assert "upstream" in result.message


def _repo_with_remote(tmp_path: Path) -> Path:
    """Create a bare remote + a clone tracking it; return the clone path."""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare", "-b", "main")

    author = tmp_path / "author"
    author.mkdir()
    _git(author, "init", "-b", "main")
    _git(author, "config", "user.email", "t@example.com")
    _git(author, "config", "user.name", "t")
    _git(author, "remote", "add", "origin", str(remote))
    (author / "f.txt").write_text("v1")
    _git(author, "add", ".")
    _git(author, "commit", "-m", "v1")
    _git(author, "push", "-u", "origin", "main")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(remote), str(clone))
    return clone


# --- connectivity check -----------------------------------------------------


def test_check_connection_skips_non_git_dir(tmp_path: Path) -> None:
    result = schedule.check_connection_sync(tmp_path)
    assert result.error is not None
    assert "not a git repository" in result.error


def test_check_connection_error_without_upstream(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("hi")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    result = schedule.check_connection_sync(repo)
    assert result.error is not None
    assert "upstream" in result.error


def test_check_connection_reachable(tmp_path: Path) -> None:
    clone = _repo_with_remote(tmp_path)
    result = schedule.check_connection_sync(clone)
    assert result.error is None
    assert any("Upstream:" in d for d in result.details)
    assert any("Remote:" in d for d in result.details)


def test_sanitize_masks_url_credentials() -> None:
    assert schedule._sanitize("https://user:tok@github.com/o/r.git") == (
        "https://***@github.com/o/r.git"
    )
    assert schedule._sanitize("https://x-access-token@host/o/r") == "https://***@host/o/r"
    # A URL with no embedded credentials is left untouched.
    assert schedule._sanitize("https://github.com/o/r.git") == "https://github.com/o/r.git"


def test_git_hard_reset_updates_to_upstream(tmp_path: Path) -> None:
    # Clone tracking a remote, then advance the remote and confirm the clone's
    # working tree is hard-reset forward.
    clone = _repo_with_remote(tmp_path)
    assert (clone / "f.txt").read_text() == "v1"

    # Advance the remote via the author checkout created by the helper.
    author = tmp_path / "author"
    (author / "f.txt").write_text("v2")
    _git(author, "commit", "-am", "v2")
    _git(author, "push", "origin", "main")

    result = schedule.git_hard_reset_sync(clone)
    assert result.status == "updated"
    assert (clone / "f.txt").read_text() == "v2"
