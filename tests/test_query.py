"""Tests for the LanceDB query path.

Two layers:

* Pure unit tests for the GLOB->LIKE filter translation (no model, no I/O).
* Integration tests driving ``Project`` directly with a deterministic
  keyword embedder, so KNN ranking, language/path filters, and incremental
  re-embedding are exercised end-to-end against a real LanceDB table without
  needing a heavyweight embedding model.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cocoindex.resources.schema import VectorSchema

from cocoindex_code.project import Project
from cocoindex_code.protocol import IndexingProgress
from cocoindex_code.query import _build_filter, _glob_to_like
from cocoindex_code.settings import ProjectSettings, save_project_settings

# ---------------------------------------------------------------------------
# Unit tests: GLOB -> LIKE translation
# ---------------------------------------------------------------------------


def test_glob_to_like_star_maps_to_percent() -> None:
    assert _glob_to_like("lib/*") == "lib/%"
    assert _glob_to_like("*.py") == "%.py"
    assert _glob_to_like("src/utils/*") == "src/utils/%"


def test_glob_to_like_question_maps_to_underscore() -> None:
    assert _glob_to_like("file?.py") == "file_.py"


def test_glob_to_like_escapes_sql_wildcards() -> None:
    # Literal % and _ in the path must be escaped so LIKE treats them literally.
    assert _glob_to_like("a_b%c") == "a\\_b\\%c"
    assert _glob_to_like("a_b*") == "a\\_b%"


def test_build_filter_language_only() -> None:
    pred = _build_filter(["python", "go"], None)
    assert pred == "language IN ('python', 'go')"


def test_build_filter_path_only() -> None:
    pred = _build_filter(None, ["lib/*"])
    assert pred is not None
    assert "file_path LIKE 'lib/%' ESCAPE '\\'" in pred


def test_build_filter_combines_language_and_path() -> None:
    pred = _build_filter(["python"], ["src/*", "*.py"])
    assert pred is not None
    assert pred.startswith("language IN ('python') AND (")
    assert "file_path LIKE 'src/%'" in pred
    assert "file_path LIKE '%.py'" in pred
    assert " OR " in pred


def test_build_filter_none_when_empty() -> None:
    assert _build_filter(None, None) is None
    assert _build_filter([], []) is None


def test_build_filter_exclude_paths_only() -> None:
    pred = _build_filter(None, None, ["a.py", "b.py"])
    assert pred == "file_path NOT IN ('a.py', 'b.py')"


def test_build_filter_combines_exclude_with_language() -> None:
    pred = _build_filter(["python"], None, ["gone.py"])
    assert pred == "language IN ('python') AND file_path NOT IN ('gone.py')"


def test_build_filter_escapes_quotes_in_language() -> None:
    # Defensive: a language value with a quote must not break out of the literal.
    pred = _build_filter(["o'brien"], None)
    assert pred == "language IN ('o''brien')"


# ---------------------------------------------------------------------------
# Integration: deterministic keyword embedder
# ---------------------------------------------------------------------------

# Each vocabulary word owns one dimension; an embedding is the (normalized) sum
# of the basis vectors for the vocab words present in the text. A query for a
# single word therefore ranks the chunk dominated by that word first.
_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
_DIM = len(_VOCAB)


class _KeywordEmbedder:
    """Deterministic embedder: text -> normalized bag-of-known-words vector."""

    def __coco_memo_key__(self) -> str:
        return "keyword-embedder-v1"

    async def __coco_vector_schema__(self) -> VectorSchema:
        return VectorSchema(dtype=np.dtype("float32"), size=_DIM)

    async def embed(self, text: str, **_kwargs: Any) -> np.ndarray:
        vec = np.zeros(_DIM, dtype=np.float32)
        tokens = re.findall(r"[a-z]+", text.lower())
        for i, word in enumerate(_VOCAB):
            vec[i] = float(sum(1 for t in tokens if t == word))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        else:
            # Avoid an all-zero vector (cosine distance is undefined); park it
            # on a reserved-ish direction so it never wins a ranked query.
            vec[-1] = 1.0
        return vec


async def _make_project(project_root: Path) -> Project:
    settings = ProjectSettings(
        include_patterns=["**/*.*"],
        exclude_patterns=["**/.cocoindex_code"],
    )
    save_project_settings(project_root, settings)
    embedder = _KeywordEmbedder()
    return await Project.create(
        project_root,
        embedder,
        embedder,
        indexing_params={},
        query_params={},
    )


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


async def test_knn_ranks_most_relevant_first(tmp_path: Path) -> None:
    """A single-keyword query ranks the matching file's chunk first."""
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "a.py", "alpha alpha alpha keyword content here\n")
    _write(tmp_path, "b.py", "beta beta beta different content here\n")
    _write(tmp_path, "c.py", "gamma gamma gamma other content here\n")

    project = await _make_project(tmp_path)
    await project.run_index()

    results = await project.search("alpha", limit=3)
    assert results, "expected at least one result"
    assert results[0].file_path == "a.py"
    # Cosine similarity score in the documented 0..1 range; top hit is strong.
    assert 0.0 <= results[0].score <= 1.0001
    assert results[0].score > 0.5


async def test_language_filter_restricts_results(tmp_path: Path) -> None:
    """The language filter only returns chunks in the requested language(s)."""
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "a.py", "alpha alpha alpha shared token\n")
    _write(tmp_path, "b.go", "alpha alpha alpha shared token\n")

    project = await _make_project(tmp_path)
    await project.run_index()

    py_only = await project.search("alpha", limit=10, languages=["python"])
    assert py_only
    assert {r.language for r in py_only} == {"python"}
    assert all(r.file_path.endswith(".py") for r in py_only)


async def test_path_filter_restricts_results(tmp_path: Path) -> None:
    """The path glob filter (translated to LIKE) restricts by file path."""
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "lib/db.py", "alpha alpha alpha database token\n")
    _write(tmp_path, "app/main.py", "alpha alpha alpha application token\n")

    project = await _make_project(tmp_path)
    await project.run_index()

    lib_only = await project.search("alpha", limit=10, paths=["lib/*"])
    assert lib_only
    assert all(r.file_path.startswith("lib/") for r in lib_only)


async def test_incremental_reindex_reembeds_only_changed(tmp_path: Path) -> None:
    """A second index pass re-embeds only the file whose content changed."""
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "a.py", "alpha alpha alpha\n")
    _write(tmp_path, "b.py", "beta beta beta\n")
    _write(tmp_path, "c.py", "gamma gamma gamma\n")

    project = await _make_project(tmp_path)
    await project.run_index()

    # Second pass, nothing changed: every file is unchanged, none reprocessed.
    snapshots: list[IndexingProgress] = []
    await project.run_index(on_progress=snapshots.append)
    assert snapshots, "expected progress snapshots"
    final = snapshots[-1]
    assert final.num_reprocesses == 0
    assert final.num_adds == 0
    assert final.num_unchanged >= 3

    # Edit one file, re-index: exactly one file reprocesses.
    _write(tmp_path, "b.py", "beta beta beta delta delta\n")
    snapshots = []
    await project.run_index(on_progress=snapshots.append)
    final = snapshots[-1]
    assert final.num_reprocesses == 1

    # The edited content is searchable under its new keyword.
    results = await project.search("delta", limit=3)
    assert results
    assert results[0].file_path == "b.py"
    project.close()


@pytest.mark.asyncio
async def test_pagination_offset_skips_results(tmp_path: Path) -> None:
    """offset skips leading ranked results without dropping the tail."""
    (tmp_path / ".git").mkdir()
    # Three files with decreasing alpha weight -> deterministic rank order.
    _write(tmp_path, "a.py", "alpha alpha alpha alpha\n")
    _write(tmp_path, "b.py", "alpha alpha beta\n")
    _write(tmp_path, "c.py", "alpha gamma gamma\n")

    project = await _make_project(tmp_path)
    await project.run_index()

    page1 = await project.search("alpha", limit=1, offset=0)
    page2 = await project.search("alpha", limit=1, offset=1)
    assert page1 and page2
    assert page1[0].file_path != page2[0].file_path
    project.close()


# ---------------------------------------------------------------------------
# Branch search: overlay on a real git repo (deterministic keyword embedder)
# ---------------------------------------------------------------------------

_needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@e.com", "-c", "user.name=T",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    )


def _make_branch_repo(root: Path) -> None:
    """main has a.py(alpha)+b.py(beta); feature changes a.py->gamma, adds c.py(delta)."""
    _git(root, "init")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    _write(root, "a.py", "alpha alpha alpha\n")
    _write(root, "b.py", "beta beta beta\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "base")

    _git(root, "checkout", "-b", "feature")
    _write(root, "a.py", "gamma gamma gamma\n")
    _write(root, "c.py", "delta delta delta\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "feature")
    _git(root, "checkout", "main")  # leave the working tree on the base


@_needs_git
async def test_branch_overlay_returns_branch_version(tmp_path: Path) -> None:
    """A branch search surfaces the branch's version of a modified file."""
    _make_branch_repo(tmp_path)
    project = await _make_project(tmp_path)
    await project.run_index()  # indexes 'main' (the base)

    # On the base, 'gamma' matches nothing strongly; on 'feature', a.py is gamma.
    results = await project.search("gamma", limit=5, branch="feature")
    assert results
    assert results[0].file_path == "a.py"
    assert results[0].source == "semantic"
    assert "gamma" in results[0].content
    project.close()


@_needs_git
async def test_branch_search_shadows_stale_base_file(tmp_path: Path) -> None:
    """The base version of a file the branch modified must not leak through."""
    _make_branch_repo(tmp_path)
    project = await _make_project(tmp_path)
    await project.run_index()

    # 'alpha' was a.py on the base, but the branch replaced it with 'gamma'. The
    # branch search must never return the stale base (alpha) a.py.
    results = await project.search("alpha", limit=5, branch="feature")
    for r in results:
        if r.file_path == "a.py":
            assert "alpha" not in r.content, "stale base version of a.py leaked into branch search"
    project.close()


@_needs_git
async def test_branch_search_unknown_ref_raises(tmp_path: Path) -> None:
    _make_branch_repo(tmp_path)
    project = await _make_project(tmp_path)
    await project.run_index()

    with pytest.raises(RuntimeError, match="not found in the local clone"):
        await project.search("gamma", limit=5, branch="no-such-branch")
    project.close()


@_needs_git
async def test_branch_equal_to_base_is_plain_search(tmp_path: Path) -> None:
    """Passing the base ref as the branch is a normal base search (no overlay)."""
    _make_branch_repo(tmp_path)
    project = await _make_project(tmp_path)
    await project.run_index()

    results = await project.search("alpha", limit=5, branch="main")
    assert results and results[0].file_path == "a.py"
    assert "alpha" in results[0].content
    project.close()
