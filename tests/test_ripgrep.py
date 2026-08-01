"""Tests for the ripgrep-backed text search that powers the MCP `ripgrep` tool.

Skipped when the `rg` binary is absent, except for the tests that pin the
behavior when it *is* absent.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cocoindex_code import git_ops, ripgrep
from cocoindex_code.memory import DEFAULT_SCAN_BUDGET, ScanBudget
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


# --- resource bounds ----------------------------------------------------------
#
# These assert the *bounds*, not the byte counts behind them: a branch scan
# holds one batch at a time, an oversized blob is never read, and a match's
# retained text is capped. All hold whatever the governor sized the budget to.


def _budget(**kwargs: Any) -> ScanBudget:
    fields: dict[str, Any] = {
        "max_concurrent": 1,
        "blob_batch_bytes": DEFAULT_SCAN_BUDGET.blob_batch_bytes,
        "max_filesize_bytes": DEFAULT_SCAN_BUDGET.max_filesize_bytes,
    }
    return ScanBudget(**{**fields, **kwargs})


@_needs_git
def test_blob_batches_never_exceed_the_batch_budget(branch_repo: Path) -> None:
    """Peak resident branch text is one batch, however many files changed."""
    sha = _out(branch_repo, "rev-parse", "feature")
    paths = ["added.py", "mod.py"]
    # Each file is ~16-18 bytes; a 20-byte budget forces one file per batch.
    batches = list(ripgrep._blob_batches(branch_repo, sha, paths, _budget(blob_batch_bytes=20)))

    assert len(batches) == 2  # not one dict holding both files
    for batch in batches:
        assert sum(len(c) for c in batch.values()) <= 20 or len(batch) == 1
    # Batching changes when files are read, never which ones.
    assert {p for b in batches for p in b} == set(paths)


@_needs_git
def test_blob_batches_skip_oversized_files_without_reading_them(
    branch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file cut is applied from git's size record, before any read."""
    sha = _out(branch_repo, "rev-parse", "feature")
    reads: list[str] = []
    # Bound before patching: git_ops.read_blob *is* the attribute being replaced,
    # so calling it by name inside the stub would recurse.
    real_read: Callable[[Path, str, str], str | None] = git_ops.read_blob

    def tracked_read(root: Path, ref: str, path: str) -> str | None:
        reads.append(path)
        return real_read(root, ref, path)

    monkeypatch.setattr(ripgrep.git_ops, "read_blob", tracked_read)

    batches = list(
        ripgrep._blob_batches(
            branch_repo, sha, ["added.py", "mod.py"], _budget(max_filesize_bytes=1)
        )
    )
    assert batches == []
    assert reads == []  # skipped on size alone — never pulled into memory


@_needs_git
def test_blob_sizes_match_the_content_git_returns(branch_repo: Path) -> None:
    """The pre-read size check has to agree with what read_blob later produces."""
    sha = _out(branch_repo, "rev-parse", "feature")
    paths = ["added.py", "mod.py"]
    sizes = git_ops.blob_sizes(branch_repo, sha, paths)
    assert set(sizes) == set(paths)
    for path in paths:
        content = git_ops.read_blob(branch_repo, sha, path)
        assert content is not None
        assert sizes[path] == len(content.encode("utf-8"))


@_needs_git
def test_blob_sizes_of_a_missing_path_is_omitted(branch_repo: Path) -> None:
    sha = _out(branch_repo, "rev-parse", "feature")
    assert git_ops.blob_sizes(branch_repo, sha, ["nope.py"]) == {}


def test_context_expansion_reads_a_file_once_per_scan() -> None:
    """Many matches in one file must not mean many full reads of it."""
    reads: list[str] = []

    def counting_read(rel: str) -> list[str]:
        reads.append(rel)
        return ["needle"] * 50

    cached = ripgrep._one_file_cache(counting_read)
    query = _query("needle", context_lines=2)
    for line_number in range(1, 21):
        ripgrep._build_match("big.py", line_number, "needle", query, cached)

    assert reads == ["big.py"]  # 20 matches, one read


def test_long_lines_are_capped_and_marked() -> None:
    """`limit` bounds how many matches come back, not how big each one is."""
    long_line = "x" * (ripgrep._MAX_LINE_CHARS + 500)
    match = ripgrep._build_match("a.js", 1, long_line, _query("x"), lambda rel: [])
    assert len(match.content) == ripgrep._MAX_LINE_CHARS + len(ripgrep._TRUNCATION_MARKER)
    assert match.content.endswith(ripgrep._TRUNCATION_MARKER)


def test_context_lines_are_capped_individually() -> None:
    """Capping per line keeps start_line/end_line honest about the window."""
    lines = ["short", "y" * (ripgrep._MAX_LINE_CHARS + 10), "also short"]
    match = ripgrep._build_match(
        "a.js", 2, lines[1], _query("y", context_lines=1), lambda rel: lines
    )
    got = match.content.split("\n")
    assert len(got) == 3
    assert (got[0], got[2]) == ("short", "also short")
    assert got[1].endswith(ripgrep._TRUNCATION_MARKER)
    assert (match.start_line, match.end_line) == (1, 3)


@_needs_rg
@_needs_git
def test_batched_branch_search_finds_what_an_unbatched_one_does(branch_repo: Path) -> None:
    """A batch budget changes cost, not results."""
    sha = _out(branch_repo, "rev-parse", "feature")
    kwargs: dict[str, Any] = {
        "branch_sha": sha,
        "branch_paths": ["added.py", "mod.py"],
        "shadow_paths": ["mod.py", "gone.py"],
    }
    roomy = ripgrep.search_branch(branch_repo, _query("NEEDLE"), **kwargs)
    one_file_at_a_time = ripgrep.search_branch(
        branch_repo, _query("NEEDLE"), budget=_budget(blob_batch_bytes=20), **kwargs
    )
    assert roomy is not None and one_file_at_a_time is not None
    assert roomy.matches == one_file_at_a_time.matches
    assert one_file_at_a_time.truncated is False


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
