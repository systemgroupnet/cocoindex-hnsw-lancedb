"""Read-only git helpers for branch search.

Everything here uses only read-only git plumbing (``rev-parse``,
``merge-base``, ``diff``, ``show``) against the workspace's existing local
clone. Nothing writes the working tree, the index, or the object database.

.. note::

    This module is deliberately **not** the hardened read-only guarantee layer
    described in ``docs/branch-search.md`` (Future work). It does not yet run an
    allowlist wrapper, does not set ``GIT_INDEX_FILE`` to a temp path, and does
    not fetch: the requested branch must already exist in the local clone. The
    commands invoked are inherently read-only, but the *enforcement* that makes
    that safe against arbitrary/untrusted branch input is still to come. Keep new
    code here to read-only plumbing only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Hard ceiling on a single git invocation so a pathological repo can't wedge a
# search request. Branch reads are local (no network), so this is generous.
_GIT_TIMEOUT_SECONDS = 60

ENV_BASE_REF = "COCOINDEX_CODE_BASE_REF"


@dataclass(frozen=True)
class BranchDiff:
    """The files a branch changed relative to its merge-base with the base ref.

    ``added``/``modified``/``deleted`` are repo-root-relative POSIX paths, the
    same form stored as ``file_path`` in the index (so ``shadow`` lines up with
    ``code_chunks.file_path`` directly).
    """

    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]

    @property
    def to_embed(self) -> tuple[str, ...]:
        """Files whose branch content should be chunked into the overlay."""
        return self.added + self.modified

    @property
    def shadow(self) -> tuple[str, ...]:
        """Base-index paths to hide (branch modified or deleted them)."""
        return self.modified + self.deleted

    @property
    def total_changed(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)


def git_available() -> bool:
    return shutil.which("git") is not None


def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a read-only git command in *root*, capturing text output.

    Mirrors :func:`schedule._run_git` (``-c safe.directory`` for bind-mounted
    repos owned by another UID, ``GIT_TERMINAL_PROMPT=0`` to fail fast instead of
    blocking on a prompt), minus the credential injection — branch reads are
    local and never touch a remote.
    """
    return subprocess.run(
        ["git", "-C", str(root), "-c", f"safe.directory={root}", *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )


def detect_base_ref(root: Path) -> str | None:
    """The ref the base index represents: ``COCOINDEX_CODE_BASE_REF`` or ``HEAD``.

    Returns ``None`` only when git can't report the current branch (e.g. a
    detached HEAD with the env override unset).
    """
    override = os.environ.get(ENV_BASE_REF, "").strip()
    if override:
        return override
    try:
        proc = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    except (OSError, subprocess.TimeoutExpired):
        return None
    ref = proc.stdout.strip()
    if proc.returncode != 0 or not ref or ref == "HEAD":
        return None
    return ref


def resolve_commit(root: Path, ref: str) -> str | None:
    """Resolve *ref* (branch, tag, or SHA) to a full commit SHA, or ``None``.

    Uses ``^{commit}`` so a tag object resolves to the commit it points at. A
    ``None`` return means the ref does not exist locally (no fetch is performed
    yet — see the module note).
    """
    try:
        proc = _run_git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def branch_diff(root: Path, base_ref: str, branch_ref: str) -> BranchDiff | None:
    """Files *branch_ref* changed since it diverged from *base_ref*.

    Uses ``git diff --name-status -z --no-renames base...branch`` (three-dot:
    diff from the merge-base to the branch tip), so changes made on the base
    *after* divergence are excluded. ``-z`` makes parsing robust to paths with
    spaces/tabs/newlines; ``--no-renames`` expands a rename into delete-old +
    add-new, which is exactly the shadow/embed split we want.

    Returns ``None`` on any git failure (bad refs, not a repo).
    """
    try:
        proc = _run_git(
            root, "diff", "--name-status", "-z", "--no-renames", f"{base_ref}...{branch_ref}"
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None

    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []

    # -z record layout (no renames): status NUL path NUL status NUL path NUL ...
    tokens = [t for t in proc.stdout.split("\0") if t != ""]
    for i in range(0, len(tokens) - 1, 2):
        status = tokens[i]
        path = tokens[i + 1]
        code = status[:1]
        if code == "A":
            added.append(path)
        elif code == "D":
            deleted.append(path)
        else:
            # M (modified), T (type change), and any unexpected status: treat as
            # modified — embed the branch version and shadow the base version.
            modified.append(path)

    return BranchDiff(added=tuple(added), modified=tuple(modified), deleted=tuple(deleted))


def read_blob(root: Path, ref: str, path: str) -> str | None:
    """Return the UTF-8 text of ``ref:path``, or ``None``.

    ``None`` means the blob is missing or not decodable as UTF-8 (binary) — the
    same "skip it" outcome the on-disk indexer reaches on ``UnicodeDecodeError``.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "-c", f"safe.directory={root}", "show", f"{ref}:{path}"],
            capture_output=True,
            text=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None
