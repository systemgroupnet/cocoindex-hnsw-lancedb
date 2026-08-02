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


def _local_branches(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "branch", "--format=%(refname:short)"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


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


def test_resolve_commit_rejects_option_like_ref(repo: Path) -> None:
    # A ref that git would read as an option never reaches the command line.
    assert git_ops.resolve_commit(repo, "--upload-pack=touch pwned") is None
    assert git_ops.resolve_commit(repo, "") is None


# --- remote-tracking refs + on-demand fetch ---------------------------------


@pytest.fixture
def clone_with_remote(tmp_path: Path) -> Path:
    """A clone of a remote that has a 'feature' branch never checked out locally.

    Mirrors a deployment: the daemon's clone sits on 'main', and the branch to
    search exists only as ``refs/remotes/origin/feature`` (or not yet at all).
    """
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "-b", "main")

    author = tmp_path / "author"
    author.mkdir()
    _git(author, "init", "-b", "main")
    _git(author, "remote", "add", "origin", str(origin))
    (author / "keep.py").write_text("base content\n")
    _git(author, "add", "-A")
    _git(author, "commit", "-m", "base")
    _git(author, "push", "-u", "origin", "main")

    _git(author, "checkout", "-b", "feature")
    (author / "added.py").write_text("branch content\n")
    _git(author, "add", "-A")
    _git(author, "commit", "-m", "feature")
    _git(author, "push", "origin", "feature")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))
    return clone


def test_list_remotes_puts_origin_first(clone_with_remote: Path) -> None:
    _git(clone_with_remote, "remote", "add", "backup", "https://example.invalid/r.git")
    assert git_ops.list_remotes(clone_with_remote) == ["origin", "backup"]
    assert git_ops.list_remotes(clone_with_remote.parent / "not-a-repo") == []


def test_resolve_commit_finds_remote_tracking_branch(clone_with_remote: Path) -> None:
    """A branch present only as origin/<name> resolves under its bare name.

    ``git rev-parse feature`` alone misses it — the remote-tracking fallback is
    what makes a fetched-but-never-checked-out branch searchable.
    """
    assert "feature" not in _local_branches(clone_with_remote)
    sha = git_ops.resolve_commit(clone_with_remote, "feature")
    assert sha is not None and len(sha) == 40
    # Same commit the explicit remote-tracking spelling gives.
    assert sha == git_ops.resolve_commit(clone_with_remote, "origin/feature")


def test_resolve_commit_fetches_branch_pushed_after_clone(tmp_path: Path) -> None:
    """A branch created after the last pull is fetched on demand, then resolves."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "-b", "main")

    author = tmp_path / "author"
    author.mkdir()
    _git(author, "init", "-b", "main")
    _git(author, "remote", "add", "origin", str(origin))
    (author / "keep.py").write_text("base\n")
    _git(author, "add", "-A")
    _git(author, "commit", "-m", "base")
    _git(author, "push", "-u", "origin", "main")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))

    # The branch appears on the remote only *after* the clone.
    _git(author, "checkout", "-b", "late")
    (author / "late.py").write_text("late content\n")
    _git(author, "add", "-A")
    _git(author, "commit", "-m", "late")
    _git(author, "push", "origin", "late")

    assert git_ops.resolve_commit(clone, "late") is None  # not fetched yet

    sha = git_ops.resolve_commit(clone, "late", allow_fetch=True)
    assert sha is not None
    # The fetch left a normal remote-tracking ref, so later lookups need no network...
    assert git_ops.resolve_commit(clone, "late") == sha
    # ...and the branch's blobs are readable without touching the working tree.
    assert git_ops.read_blob(clone, sha, "late.py") == "late content\n"
    assert not (clone / "late.py").exists()
    assert _local_branches(clone) == ["main"]


def test_branch_search_git_calls_leave_the_checkout_alone(clone_with_remote: Path) -> None:
    """Every git call branch search makes must leave the checkout exactly as found.

    The daemon serves parallel searches against one clone while the working tree
    stays on the base branch, so a search that moved HEAD (or dirtied the index)
    would corrupt every concurrent request. The only permitted mutation is the
    remote-tracking ref the fetch writes.
    """
    # Force the fetch path: drop the ref the clone already has for the branch.
    _git(clone_with_remote, "update-ref", "-d", "refs/remotes/origin/feature")
    index_before = (clone_with_remote / ".git" / "index").read_bytes()
    head_before = (clone_with_remote / ".git" / "HEAD").read_text()
    sha_before = git_ops.resolve_commit(clone_with_remote, "main")

    branch_sha = git_ops.resolve_commit(clone_with_remote, "feature", allow_fetch=True)
    assert branch_sha is not None
    diff = git_ops.branch_diff(clone_with_remote, "main", branch_sha)
    assert diff is not None and diff.added == ("added.py",)
    assert git_ops.read_blob(clone_with_remote, branch_sha, "added.py") == "branch content\n"

    assert (clone_with_remote / ".git" / "HEAD").read_text() == head_before
    assert git_ops.resolve_commit(clone_with_remote, "main") == sha_before
    assert (clone_with_remote / ".git" / "index").read_bytes() == index_before
    assert _local_branches(clone_with_remote) == ["main"]
    assert not (clone_with_remote / "added.py").exists()
    status = subprocess.run(
        ["git", "-C", str(clone_with_remote), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""


def test_fetch_ref_refuses_non_branch_refs(clone_with_remote: Path) -> None:
    # Anything that isn't a plain branch name never reaches `git fetch` — a ref
    # read as an option there would be remote code execution.
    assert git_ops.fetch_ref(clone_with_remote, "--upload-pack=touch pwned") is None
    assert git_ops.fetch_ref(clone_with_remote, "main..feature") is None


def test_fetch_ref_returns_none_when_remote_lacks_branch(clone_with_remote: Path) -> None:
    assert git_ops.fetch_ref(clone_with_remote, "no-such-branch") is None


def test_fetch_enabled_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(git_ops.ENV_FETCH_ENABLED, raising=False)
    assert git_ops.fetch_enabled() is True
    monkeypatch.setenv(git_ops.ENV_FETCH_ENABLED, "off")
    assert git_ops.fetch_enabled() is False
    monkeypatch.setenv(git_ops.ENV_FETCH_ENABLED, "1")
    assert git_ops.fetch_enabled() is True


# --- credentials -------------------------------------------------------------


def test_load_credentials_requires_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(git_ops.ENV_GIT_PASSWORD, raising=False)
    monkeypatch.setenv(git_ops.ENV_GIT_USERNAME, "someone")
    assert git_ops.load_credentials() is None

    monkeypatch.setenv(git_ops.ENV_GIT_PASSWORD, "ghp_secret")
    creds = git_ops.load_credentials()
    assert creds is not None
    assert creds.username == "someone"
    assert creds.password == "ghp_secret"


def test_credential_git_args_present_only_with_creds() -> None:
    assert git_ops.credential_git_args(None) == []
    args = git_ops.credential_git_args(
        git_ops.GitCredentials("distinct-user-xyz", "distinct-secret-xyz")
    )
    assert args[0] == "-c"
    assert args[1].startswith("credential.helper=")
    # The helper string carries no secret — it reads them from the environment,
    # so nothing sensitive lands in the process argv.
    assert "distinct-user-xyz" not in args[1]
    assert "distinct-secret-xyz" not in args[1]


def test_git_env_injects_credentials() -> None:
    env = git_ops.git_env(git_ops.GitCredentials("user", "tok"))
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env[git_ops._CRED_ENV_USERNAME] == "user"
    assert env[git_ops._CRED_ENV_PASSWORD] == "tok"


def test_git_env_no_credentials() -> None:
    env = git_ops.git_env(None)
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert git_ops._CRED_ENV_PASSWORD not in env


def test_branch_diff_classifies_changes(repo: Path) -> None:
    diff = git_ops.branch_diff(repo, "main", "feature")
    assert diff is not None
    assert diff.added == ("added.py",)
    assert diff.modified == ("mod.py",)
    assert diff.deleted == ("gone.py",)
    assert set(diff.to_scan) == {"added.py", "mod.py"}
    assert set(diff.shadow) == {"mod.py", "gone.py"}


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
