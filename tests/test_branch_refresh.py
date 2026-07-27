"""Tests for the pre-search clone refresh that runs before every branch search.

Two layers: the throttle/never-raises contract of
``BranchOverlayManager._refresh_clone`` (no git, no env needed), and
``_refresh_clone_sync`` driven against a real temp clone to pin down what the
pull gate actually does to the checkout.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from cocoindex_code import branch_overlay, git_ops, schedule
from cocoindex_code.branch_overlay import BranchOverlayManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _out(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        schedule.ENV_GIT_PULL_ENABLED,
        branch_overlay.ENV_REFRESH_SECONDS,
        git_ops.ENV_GIT_USERNAME,
        git_ops.ENV_GIT_PASSWORD,
    ):
        monkeypatch.delenv(name, raising=False)


def _manager(root: Path) -> BranchOverlayManager:
    # _refresh_clone never touches the CocoIndex environment, so a placeholder
    # keeps these tests free of embedder/LanceDB setup.
    return BranchOverlayManager(cast(Any, None), root)


# --- throttle + failure containment ------------------------------------------


def _recording_refresh(calls: list[Path]) -> Any:
    def _refresh(root: Path) -> str:
        calls.append(root)
        return "fetched"

    return _refresh


async def test_refresh_is_throttled_within_the_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(branch_overlay, "_refresh_clone_sync", _recording_refresh(calls))
    manager = _manager(tmp_path)

    for _ in range(3):
        await manager._refresh_clone()

    # A burst of searches costs one network round-trip, not three.
    assert calls == [tmp_path]


async def test_refresh_runs_every_time_when_interval_is_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(branch_overlay.ENV_REFRESH_SECONDS, "0")
    calls: list[Path] = []
    monkeypatch.setattr(branch_overlay, "_refresh_clone_sync", _recording_refresh(calls))
    manager = _manager(tmp_path)

    for _ in range(3):
        await manager._refresh_clone()

    assert len(calls) == 3


async def test_refresh_failure_is_swallowed_and_still_throttles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken remote must not fail the search, nor cost a timeout per search."""
    calls = 0

    def _boom(root: Path) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("remote is down")

    monkeypatch.setattr(branch_overlay, "_refresh_clone_sync", _boom)
    manager = _manager(tmp_path)

    await manager._refresh_clone()  # must not raise
    await manager._refresh_clone()

    assert calls == 1


# --- what the refresh actually does to the clone -----------------------------


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """A clone of a remote, plus an 'author' checkout to advance that remote."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "-b", "main")

    author = tmp_path / "author"
    author.mkdir()
    _git(author, "init", "-b", "main")
    _git(author, "remote", "add", "origin", str(origin))
    (author / "keep.py").write_text("v1\n")
    _git(author, "add", "-A")
    _git(author, "commit", "-m", "v1")
    _git(author, "push", "-u", "origin", "main")

    clone_path = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone_path))
    return clone_path


def _push_new_branch(tmp_path: Path) -> None:
    author = tmp_path / "author"
    _git(author, "checkout", "-b", "late")
    (author / "late.py").write_text("late\n")
    _git(author, "add", "-A")
    _git(author, "commit", "-m", "late")
    _git(author, "push", "origin", "late")
    _git(author, "checkout", "main")


def _push_to_main(tmp_path: Path) -> None:
    author = tmp_path / "author"
    (author / "keep.py").write_text("v2\n")
    _git(author, "commit", "-am", "v2")
    _git(author, "push", "origin", "main")


def test_refresh_without_pull_enabled_fetches_only(clone: Path, tmp_path: Path) -> None:
    """Gate off: refs advance, the checkout does not move at all."""
    _push_new_branch(tmp_path)
    _push_to_main(tmp_path)
    head_before = _out(clone, "rev-parse", "HEAD")
    index_before = (clone / ".git" / "index").read_bytes()

    assert branch_overlay._refresh_clone_sync(clone) == "fetched"

    # The branch pushed after the clone is now resolvable...
    assert git_ops.resolve_commit(clone, "late") is not None
    # ...and so is the newer origin/main...
    assert _out(clone, "rev-parse", "origin/main") != head_before
    # ...while HEAD, the index, and the working tree are exactly as they were.
    assert _out(clone, "rev-parse", "HEAD") == head_before
    assert (clone / ".git" / "index").read_bytes() == index_before
    assert (clone / "keep.py").read_text() == "v1\n"
    assert _out(clone, "status", "--porcelain") == ""


def test_refresh_with_pull_enabled_advances_the_base(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate on: the base ref and working tree move forward, as `ccc pull` does."""
    monkeypatch.setenv(schedule.ENV_GIT_PULL_ENABLED, "1")
    _push_new_branch(tmp_path)
    _push_to_main(tmp_path)

    assert branch_overlay._refresh_clone_sync(clone).startswith("updated")

    assert (clone / "keep.py").read_text() == "v2\n"
    assert _out(clone, "rev-parse", "HEAD") == _out(clone, "rev-parse", "origin/main")
    # Still on the base branch — a pull advances it, it never switches away.
    assert _out(clone, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git_ops.resolve_commit(clone, "late") is not None


def test_refresh_reports_failure_instead_of_raising(tmp_path: Path) -> None:
    """An unreachable remote yields a log line, not an exception."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "remote", "add", "origin", str(tmp_path / "does-not-exist.git"))
    (repo / "f.txt").write_text("hi\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")

    assert branch_overlay._refresh_clone_sync(repo).startswith("fetch failed:")
