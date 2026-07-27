"""Tests for the ripgrep-backed text search that powers the MCP `ripgrep` tool.

Skipped when the `rg` binary is absent, except for the tests that pin the
behavior when it *is* absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cocoindex_code import ripgrep
from cocoindex_code.settings import SETTINGS_DIR_NAME

_needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
_needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _query(pattern: str, **kwargs: Any) -> ripgrep.RipgrepQuery:
    return ripgrep.RipgrepQuery(patterns=(pattern,), limit=50, **kwargs)


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _paths(outcome: ripgrep.RipgrepOutcome | None) -> set[str]:
    assert outcome is not None
    return {m.file_path for m in outcome.matches}


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(root),
            "-c", "user.email=t@example.com",
            "-c", "user.name=T",
            "-c", "commit.gpgsign=false",
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


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    _write(tmp_path, "src/app.py", "def login(user):\n    return TOKEN\n")
    _write(tmp_path, "src/util.py", "TOKEN = 'abc'\n")
    _write(tmp_path, "docs/readme.md", "the token is documented here\n")
    return tmp_path


# --- working-tree search ------------------------------------------------------


@_needs_rg
def test_search_tree_returns_relative_paths_and_line_numbers(tree: Path) -> None:
    outcome = ripgrep.search_tree(tree, _query("TOKEN ="))
    assert outcome is not None
    assert len(outcome.matches) == 1
    match = outcome.matches[0]
    # Repo-relative POSIX path, regardless of platform separator.
    assert match.file_path == "src/util.py"
    assert match.line_number == 1
    assert match.content == "TOKEN = 'abc'"
    assert (match.start_line, match.end_line) == (1, 1)
    assert outcome.truncated is False


@_needs_rg
def test_search_tree_is_case_insensitive_by_default(tree: Path) -> None:
    assert _paths(ripgrep.search_tree(tree, _query("token"))) == {
        "src/app.py",
        "src/util.py",
        "docs/readme.md",
    }
    assert _paths(ripgrep.search_tree(tree, _query("token", case_sensitive=True))) == {
        "docs/readme.md"
    }


@_needs_rg
def test_search_tree_applies_globs(tree: Path) -> None:
    assert _paths(ripgrep.search_tree(tree, _query("token", globs=("*.md",)))) == {
        "docs/readme.md"
    }
    # A '!' glob excludes.
    assert _paths(ripgrep.search_tree(tree, _query("token", globs=("!docs/**",)))) == {
        "src/app.py",
        "src/util.py",
    }


@_needs_rg
def test_search_tree_fixed_strings_treats_metacharacters_literally(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "value = get(cfg)[0]\n")
    _write(tmp_path, "b.py", "value = getXcfgY_0_\n")

    # As a regex, `get(cfg)[0]` matches "getcfg0"-ish text, not the literal.
    literal = ripgrep.search_tree(tmp_path, _query("get(cfg)[0]", fixed_strings=True))
    assert _paths(literal) == {"a.py"}


@_needs_rg
def test_search_tree_context_lines_widen_the_snippet(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "one\ntwo\nNEEDLE\nfour\nfive\n")
    outcome = ripgrep.search_tree(tmp_path, _query("NEEDLE", context_lines=1))
    assert outcome is not None
    match = outcome.matches[0]
    assert match.content == "two\nNEEDLE\nfour"
    assert (match.start_line, match.end_line) == (2, 4)
    assert match.line_number == 3


@_needs_rg
def test_search_tree_limit_marks_truncated(tmp_path: Path) -> None:
    for i in range(10):
        _write(tmp_path, f"f{i}.py", "needle here\n")
    outcome = ripgrep.search_tree(
        tmp_path, ripgrep.RipgrepQuery(patterns=("needle",), limit=3)
    )
    assert outcome is not None
    assert len(outcome.matches) == 3
    assert outcome.truncated is True

    everything = ripgrep.search_tree(
        tmp_path, ripgrep.RipgrepQuery(patterns=("needle",), limit=50)
    )
    assert everything is not None
    assert len(everything.matches) == 10
    assert everything.truncated is False


@_needs_rg
def test_search_tree_skips_the_index_dir_but_searches_hidden_files(tmp_path: Path) -> None:
    _write(tmp_path, f"{SETTINGS_DIR_NAME}/overlays.json", '{"needle": 1}\n')
    _write(tmp_path, ".github/workflows/ci.yml", "run: needle\n")
    _write(tmp_path, "src/a.py", "# needle\n")

    # The index directory is our own data, never a search result; hidden dotted
    # paths like .github are real source and must be searchable.
    assert _paths(ripgrep.search_tree(tmp_path, _query("needle"))) == {
        ".github/workflows/ci.yml",
        "src/a.py",
    }


@_needs_rg
@_needs_git
def test_search_tree_respects_gitignore(tmp_path: Path) -> None:
    # rg only reads .gitignore inside a real repo — which the project root
    # always is when branch search is in play.
    _git(tmp_path, "init")
    _write(tmp_path, ".gitignore", "build/\n")
    _write(tmp_path, "build/generated.py", "needle\n")
    _write(tmp_path, "src/a.py", "needle\n")
    assert _paths(ripgrep.search_tree(tmp_path, _query("needle"))) == {"src/a.py"}


@_needs_rg
def test_search_tree_pattern_starting_with_dash_is_not_an_option(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "parser.add_argument('--help-me')\n")
    outcome = ripgrep.search_tree(tmp_path, _query("--help-me", fixed_strings=True))
    assert _paths(outcome) == {"a.py"}


@_needs_rg
def test_search_tree_orders_matches_by_path_and_line(tmp_path: Path) -> None:
    """rg walks in parallel; the same query must still read the same way twice."""
    _write(tmp_path, "z.py", "needle\nneedle\n")
    _write(tmp_path, "a.py", "x\nneedle\n")
    _write(tmp_path, "m/b.py", "needle\n")
    outcome = ripgrep.search_tree(tmp_path, _query("needle"))
    assert outcome is not None
    assert [(m.file_path, m.line_number) for m in outcome.matches] == [
        ("a.py", 2),
        ("m/b.py", 1),
        ("z.py", 1),
        ("z.py", 2),
    ]


@_needs_rg
def test_search_tree_exclude_paths_hides_results(tree: Path) -> None:
    outcome = ripgrep.search_tree(tree, _query("token"), exclude_paths=["src/app.py"])
    assert _paths(outcome) == {"src/util.py", "docs/readme.md"}


# --- in-memory blobs ----------------------------------------------------------


@_needs_rg
def test_search_blobs_maps_back_to_repo_paths() -> None:
    outcome = ripgrep.search_blobs(
        {"pkg/mod.py": "line one\nfind me\n", "other.py": "nothing\n"},
        _query("find me"),
    )
    assert outcome is not None
    assert len(outcome.matches) == 1
    assert outcome.matches[0].file_path == "pkg/mod.py"
    assert outcome.matches[0].line_number == 2


def test_search_blobs_empty_input_is_an_empty_outcome() -> None:
    outcome = ripgrep.search_blobs({}, _query("anything"))
    assert outcome is not None
    assert outcome.is_empty
    assert outcome.truncated is False


# --- branch view --------------------------------------------------------------


@pytest.fixture
def branch_repo(tmp_path: Path) -> Path:
    """main: keep/mod/gone all say NEEDLE. feature: rewrites mod, adds added, deletes gone."""
    _git(tmp_path, "init")
    _git(tmp_path, "symbolic-ref", "HEAD", "refs/heads/main")
    _write(tmp_path, "keep.py", "NEEDLE in keep\n")
    _write(tmp_path, "mod.py", "NEEDLE base version\n")
    _write(tmp_path, "gone.py", "NEEDLE in gone\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "base")

    _git(tmp_path, "checkout", "-b", "feature")
    _write(tmp_path, "mod.py", "NEEDLE branch version\n")
    _write(tmp_path, "added.py", "NEEDLE in added\n")
    (tmp_path / "gone.py").unlink()
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "feature")

    _git(tmp_path, "checkout", "main")  # working tree stays on the base
    return tmp_path


@_needs_rg
@_needs_git
def test_search_branch_sees_the_branch_not_the_checkout(branch_repo: Path) -> None:
    """The branch's version wins, the base's stale copy is hidden, deletes vanish."""
    sha = _out(branch_repo, "rev-parse", "feature")
    outcome = ripgrep.search_branch(
        branch_repo,
        _query("NEEDLE"),
        branch_sha=sha,
        branch_paths=["added.py", "mod.py"],
        shadow_paths=["mod.py", "gone.py"],
    )
    assert outcome is not None
    by_path = {m.file_path: m.content for m in outcome.matches}

    assert by_path["keep.py"] == "NEEDLE in keep"  # untouched -> from the checkout
    assert by_path["mod.py"] == "NEEDLE branch version"  # branch's version, not base's
    assert by_path["added.py"] == "NEEDLE in added"  # exists only on the branch
    assert "gone.py" not in by_path  # deleted on the branch


@_needs_rg
@_needs_git
def test_search_branch_leaves_the_checkout_alone(branch_repo: Path) -> None:
    """Grepping a branch must not move HEAD or touch the tree.

    The daemon greps arbitrary branches against one shared clone whose working
    tree stays on the base; a scan that switched branches would corrupt every
    concurrent request and the base index's source tree.
    """
    sha = _out(branch_repo, "rev-parse", "feature")
    head_before = (branch_repo / ".git" / "HEAD").read_text()
    head_sha_before = _out(branch_repo, "rev-parse", "HEAD")
    index_before = (branch_repo / ".git" / "index").read_bytes()

    ripgrep.search_branch(
        branch_repo,
        _query("NEEDLE"),
        branch_sha=sha,
        branch_paths=["added.py", "mod.py"],
        shadow_paths=["mod.py", "gone.py"],
    )

    assert (branch_repo / ".git" / "HEAD").read_text() == head_before
    assert _out(branch_repo, "rev-parse", "HEAD") == head_sha_before
    assert _out(branch_repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert (branch_repo / ".git" / "index").read_bytes() == index_before
    assert _out(branch_repo, "status", "--porcelain") == ""
    # The tree still holds the base's content, not the branch's.
    assert (branch_repo / "mod.py").read_text() == "NEEDLE base version\n"
    assert (branch_repo / "gone.py").exists()
    assert not (branch_repo / "added.py").exists()


@_needs_rg
@_needs_git
def test_search_branch_limit_applies_to_the_merged_result(branch_repo: Path) -> None:
    sha = _out(branch_repo, "rev-parse", "feature")
    outcome = ripgrep.search_branch(
        branch_repo,
        ripgrep.RipgrepQuery(patterns=("NEEDLE",), limit=2),
        branch_sha=sha,
        branch_paths=["added.py", "mod.py"],
        shadow_paths=["mod.py", "gone.py"],
    )
    assert outcome is not None
    assert len(outcome.matches) == 2
    assert outcome.truncated is True


# --- rg missing ---------------------------------------------------------------


def test_search_returns_none_without_rg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """None (not an empty result) is what tells callers rg itself is unusable."""
    monkeypatch.setattr(ripgrep, "available", lambda: False)
    _write(tmp_path, "a.py", "needle\n")
    assert ripgrep.search_tree(tmp_path, _query("needle")) is None
    assert ripgrep.search_blobs({"a.py": "needle\n"}, _query("needle")) is None
    assert (
        ripgrep.search_branch(
            tmp_path, _query("needle"), branch_sha="deadbeef",
            branch_paths=[], shadow_paths=[],
        )
        is None
    )
