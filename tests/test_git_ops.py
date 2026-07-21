"""Tests for the read-only git helpers, driven against a real temp repo.

Skipped entirely when git is not installed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cocoindex_code import git_ops

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo on 'main' with a 'feature' branch that adds/modifies/deletes files.

    Working tree is left on 'main' (the base), mirroring the daemon's setup.
    """
    _git(tmp_path, "init")
    _git(tmp_path, "symbolic-ref", "HEAD", "refs/heads/main")

    (tmp_path / "keep.py").write_text("unchanged base content\n")
    (tmp_path / "mod.py").write_text("original modified content\n")
    (tmp_path / "gone.py").write_text("to be deleted\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "base")

    _git(tmp_path, "checkout", "-b", "feature")
    (tmp_path / "mod.py").write_text("BRANCH modified content\n")
    (tmp_path / "added.py").write_text("brand new on the branch\n")
    (tmp_path / "gone.py").unlink()
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "feature changes")

    _git(tmp_path, "checkout", "main")
    return tmp_path


def test_is_git_repo(repo: Path, tmp_path: Path) -> None:
    assert git_ops.is_git_repo(repo)
    assert not git_ops.is_git_repo(tmp_path.parent / "definitely-not-a-repo")


def test_detect_base_ref_reads_head(repo: Path) -> None:
    assert git_ops.detect_base_ref(repo) == "main"


def test_detect_base_ref_env_override(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(git_ops.ENV_BASE_REF, "develop")
    assert git_ops.detect_base_ref(repo) == "develop"


def test_resolve_commit(repo: Path) -> None:
    sha = git_ops.resolve_commit(repo, "feature")
    assert sha is not None and len(sha) == 40
    assert git_ops.resolve_commit(repo, "no-such-branch") is None


def test_branch_diff_classifies_changes(repo: Path) -> None:
    diff = git_ops.branch_diff(repo, "main", "feature")
    assert diff is not None
    assert diff.added == ("added.py",)
    assert diff.modified == ("mod.py",)
    assert diff.deleted == ("gone.py",)
    assert set(diff.to_embed) == {"added.py", "mod.py"}
    assert set(diff.shadow) == {"mod.py", "gone.py"}
    assert diff.total_changed == 3


def test_branch_diff_bad_ref_returns_none(repo: Path) -> None:
    assert git_ops.branch_diff(repo, "main", "bogus-ref") is None


def test_read_blob_reads_branch_version(repo: Path) -> None:
    # Branch version of a modified file, without touching the working tree.
    assert git_ops.read_blob(repo, "feature", "mod.py") == "BRANCH modified content\n"
    # Base version is still the original.
    assert git_ops.read_blob(repo, "main", "mod.py") == "original modified content\n"
    # A file deleted on the branch is absent there.
    assert git_ops.read_blob(repo, "feature", "gone.py") is None


def test_working_tree_untouched_by_reads(repo: Path) -> None:
    # The base working tree must be exactly the base after all the reads above.
    assert (repo / "mod.py").read_text() == "original modified content\n"
    assert (repo / "gone.py").exists()
    assert not (repo / "added.py").exists()
