"""Git helpers for branch search — reads, plus one narrowly scoped fetch.

Everything here uses read-only git plumbing (``rev-parse``, ``diff``, ``show``,
``remote``) against the workspace's existing local clone, with one exception:
:func:`fetch_ref` fetches a branch that isn't in the clone yet. That fetch
writes only objects and ``refs/remotes/<remote>/<branch>`` — never the working
tree, the index, or any local branch — so the base index's source tree is
untouched either way.

.. note::

    This module is deliberately **not** the hardened read-only guarantee layer
    described in ``docs/branch-search.md`` (Future work). It does not yet run an
    allowlist wrapper and does not set ``GIT_INDEX_FILE`` to a temp path. The
    commands invoked are inherently non-mutating (of the tree), but the
    *enforcement* that makes that safe against arbitrary/untrusted branch input
    is still to come — for now :func:`_is_safe_ref` is the guard that keeps a
    caller-supplied ref from being read as a git option. Keep new code here to
    reads plus the existing fetch.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Hard ceiling on a single local git invocation so a pathological repo can't
# wedge a search request. Local reads only, so this is generous.
_GIT_TIMEOUT_SECONDS = 60

# Fetches talk to a remote, so they get their own (larger) ceiling.
_FETCH_TIMEOUT_SECONDS = 180

ENV_BASE_REF = "COCOINDEX_CODE_BASE_REF"
ENV_FETCH_ENABLED = "COCOINDEX_CODE_BRANCH_FETCH_ENABLED"
ENV_GIT_USERNAME = "COCOINDEX_CODE_GIT_USERNAME"
ENV_GIT_PASSWORD = "COCOINDEX_CODE_GIT_PASSWORD"

_FALSY = {"0", "false", "no", "off"}

# Environment names the inline credential helper reads. Distinct from the
# user-facing ENV_GIT_* knobs: we set these on the git subprocess env explicitly
# so the helper's value can stay a fixed, secret-free string.
_CRED_ENV_USERNAME = "CCC_GIT_CRED_USERNAME"
_CRED_ENV_PASSWORD = "CCC_GIT_CRED_PASSWORD"

# Inline git credential helper: on a `get`, echo the credentials from the
# environment in git's credential-protocol format. The password never appears in
# argv (only this fixed string does) nor on disk. A leading `!` makes git run it
# through a shell; the helper ignores git's `get`/`store`/`erase` argument.
_CREDENTIAL_HELPER = (
    f'!f() {{ echo "username=${_CRED_ENV_USERNAME}"; '
    f'echo "password=${_CRED_ENV_PASSWORD}"; }}; f'
)


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
    def to_scan(self) -> tuple[str, ...]:
        """Files whose branch content should be read and searched."""
        return self.added + self.modified

    @property
    def shadow(self) -> tuple[str, ...]:
        """Base-index paths to hide (branch modified or deleted them)."""
        return self.modified + self.deleted


@dataclass(frozen=True)
class GitCredentials:
    """HTTPS username + password/token for git operations that reach a remote.

    Shared by the scheduled pull (:mod:`cocoindex_code.schedule`) and branch
    search's on-demand fetch, so both authenticate the same way.
    """

    username: str
    password: str


def load_credentials() -> GitCredentials | None:
    """Read HTTPS credentials from the environment, or ``None`` when unset.

    Credentials are active only when a password/token is present; the username
    is optional (some hosts accept any value for token auth). The password is
    not stripped — a token is used verbatim.
    """
    password = os.environ.get(ENV_GIT_PASSWORD) or ""
    if not password:
        return None
    return GitCredentials(
        username=(os.environ.get(ENV_GIT_USERNAME) or "").strip(), password=password
    )


def credential_git_args(credentials: GitCredentials | None) -> list[str]:
    """The ``-c credential.helper=...`` args that inject *credentials*, if any."""
    if credentials is None:
        return []
    return ["-c", f"credential.helper={_CREDENTIAL_HELPER}"]


def git_env(credentials: GitCredentials | None) -> dict[str, str]:
    """Subprocess environment for a git call: no prompts, credentials injected."""
    # GIT_TERMINAL_PROMPT=0 makes git fail fast instead of blocking on a
    # credential prompt in the non-interactive daemon. When credentials are
    # configured, expose them under the names the inline helper reads.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if credentials is not None:
        env[_CRED_ENV_USERNAME] = credentials.username
        env[_CRED_ENV_PASSWORD] = credentials.password
    return env


def git_available() -> bool:
    return shutil.which("git") is not None


def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def fetch_enabled() -> bool:
    """Whether branch search may fetch a ref missing from the local clone.

    On by default (``COCOINDEX_CODE_BRANCH_FETCH_ENABLED``): a branch opened
    after the last scheduled pull is the common case, and refusing to search it
    is the surprising behavior. Set the variable to a falsy value in deployments
    where the query path must never touch the network.
    """
    return (os.environ.get(ENV_FETCH_ENABLED) or "").strip().lower() not in _FALSY


def _is_safe_ref(ref: str) -> bool:
    """Whether *ref* is safe to put on a git command line.

    A caller-supplied ref reaches argv, where a leading ``-`` would be read as an
    option (``--upload-pack=...`` on a fetch is remote code execution) and
    whitespace breaks parsing. Reject both outright rather than escape. Refs
    valid for local resolution stay permissive — ``HEAD~2``, ``v1.2^{}``, and raw
    SHAs all pass.
    """
    return bool(ref) and not ref.startswith("-") and not any(c.isspace() for c in ref)


# A plain branch name: letters/digits/._/+- , no leading dash. Applied to refs
# that will be *fetched*, which are additionally embedded in a refspec and sent
# to the remote.
_BRANCH_NAME_RE = re.compile(r"^\w[\w./+-]*$")


def _fetchable(ref: str) -> bool:
    return bool(_BRANCH_NAME_RE.match(ref)) and ".." not in ref


def _run_git(
    root: Path,
    *args: str,
    credentials: GitCredentials | None = None,
    timeout: int = _GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run a git command in *root*, capturing text output.

    ``-c safe.directory=<root>`` keeps git from refusing a bind-mounted repo
    owned by another UID; the environment blocks credential prompts and carries
    the injected credentials for the one command that needs them (the fetch).
    """
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            f"safe.directory={root}",
            *credential_git_args(credentials),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=git_env(credentials),
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


def list_remotes(root: Path) -> list[str]:
    """Configured remote names, ``origin`` first when present."""
    try:
        proc = _run_git(root, "remote")
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    names.sort(key=lambda name: name != "origin")  # stable: origin first, rest in order
    return names


def _rev_parse(root: Path, ref: str) -> str | None:
    """``git rev-parse`` *ref* to a commit SHA. ``^{commit}`` derefs tag objects."""
    try:
        proc = _run_git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def _resolve_local(root: Path, ref: str) -> str | None:
    """Resolve *ref* against the local clone, including remote-tracking copies.

    ``git rev-parse`` does not do the remote-tracking DWIM that ``git checkout``
    does: in a clone that only ever checked out the base branch, a feature branch
    exists solely as ``refs/remotes/origin/<name>``, and a bare ``rev-parse
    <name>`` misses it. Trying each remote's spelling is what makes "search
    branch X" work off a plain ``git fetch``, with no network round-trip.
    """
    if (sha := _rev_parse(root, ref)) is not None:
        return sha
    for remote in list_remotes(root):
        if (sha := _rev_parse(root, f"refs/remotes/{remote}/{ref}")) is not None:
            return sha
    return None


def fetch_ref(root: Path, ref: str, *, credentials: GitCredentials | None = None) -> str | None:
    """Fetch branch *ref* from a remote; return its commit SHA, or ``None``.

    Writes only objects and ``refs/remotes/<remote>/<ref>`` — never the working
    tree, the index, or a local branch. The explicit refspec (rather than a bare
    ``git fetch <remote> <ref>``, which only leaves ``FETCH_HEAD``) makes the
    branch a normal remote-tracking ref, so later searches resolve it locally.

    Only plain branch names are fetchable; tags and SHAs must already be in the
    clone. Each remote is tried in turn (``origin`` first).
    """
    if not _fetchable(ref):
        logger.warning("Refusing to fetch %r: not a plain branch name", ref)
        return None
    for remote in list_remotes(root):
        dest = f"refs/remotes/{remote}/{ref}"
        try:
            proc = _run_git(
                root,
                "fetch",
                "--no-tags",
                "--quiet",
                remote,
                f"+refs/heads/{ref}:{dest}",
                credentials=credentials,
                timeout=_FETCH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Fetch of %r from %s timed out", ref, remote)
            continue
        except OSError as e:
            logger.warning("Fetch of %r from %s could not be run: %s", ref, remote, e)
            continue
        if proc.returncode != 0:
            logger.info(
                "Fetch of %r from %s failed: %s", ref, remote, proc.stderr.strip() or "(no output)"
            )
            continue
        if (sha := _rev_parse(root, dest)) is not None:
            logger.info("Fetched %s from %s for branch search", ref, remote)
            return sha
    return None


def fetch_all(root: Path, *, credentials: GitCredentials | None = None) -> str | None:
    """``git fetch --prune`` every remote. Returns ``None`` on success, else why not.

    Refreshes all remote-tracking refs — new branches, new commits on existing
    ones, and pruning branches deleted upstream. Writes only objects and
    ``refs/remotes/*``: HEAD, the index, and the working tree are untouched, so
    this is safe to run while the base index's tree is being walked.
    """
    try:
        proc = _run_git(
            root,
            "fetch",
            "--all",
            "--prune",
            "--quiet",
            credentials=credentials,
            timeout=_FETCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {_FETCH_TIMEOUT_SECONDS}s"
    except OSError as e:
        return f"could not be run: {e}"
    if proc.returncode != 0:
        return proc.stderr.strip() or "git fetch failed"
    return None


def resolve_commit(
    root: Path,
    ref: str,
    *,
    allow_fetch: bool = False,
    credentials: GitCredentials | None = None,
) -> str | None:
    """Resolve *ref* to a full commit SHA, or ``None`` if it can't be found.

    *ref* may be a local branch, a remote-tracking branch (bare name or
    ``origin/name``), a tag, or a SHA. When *allow_fetch* is set and nothing
    matches locally, the branch is fetched from the remote and re-resolved —
    that network call can take up to ``_FETCH_TIMEOUT_SECONDS``, so callers on
    the event loop should run this in a thread.
    """
    if not _is_safe_ref(ref):
        logger.warning("Rejecting unsafe ref %r", ref)
        return None
    sha = _resolve_local(root, ref)
    if sha is not None or not allow_fetch:
        return sha
    if (fetched := fetch_ref(root, ref, credentials=credentials)) is not None:
        return fetched
    # A concurrent search for the same new branch may hold git's ref lock, which
    # fails our fetch even though the ref it was writing is now there. Re-check
    # locally before giving up, so parallel requests don't fail spuriously.
    return _resolve_local(root, ref)


def branch_diff(root: Path, base_ref: str, branch_ref: str) -> BranchDiff | None:
    """Files *branch_ref* changed since it diverged from *base_ref*.

    Uses ``git diff --name-status -z --no-renames base...branch`` (three-dot:
    diff from the merge-base to the branch tip), so changes made on the base
    *after* divergence are excluded. ``-z`` makes parsing robust to paths with
    spaces/tabs/newlines; ``--no-renames`` expands a rename into delete-old +
    add-new, which is exactly the shadow/embed split we want.

    Returns ``None`` on any git failure (bad refs, not a repo).
    """
    if not _is_safe_ref(base_ref) or not _is_safe_ref(branch_ref):
        return None
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


def blob_sizes(root: Path, ref: str, paths: Sequence[str]) -> dict[str, int]:
    """Byte sizes of ``ref:path`` for each of *paths*, in one ``git cat-file`` pass.

    Paths whose blob is missing (or whose size git doesn't report) are omitted
    — the caller falls back to measuring after the read. Exists so an oversized
    blob can be skipped *before* :func:`read_blob` pulls it into memory, which
    is the only point at which the cost is still avoidable.
    """
    if not paths or not _is_safe_ref(ref):
        return {}
    try:
        proc = subprocess.run(
            [
                "git", "-C", str(root), "-c", f"safe.directory={root}",
                # -z: NUL-delimited input, so a path containing a newline can't
                # desynchronize the request list from the reply lines.
                "cat-file", "--batch-check=%(objectsize)", "-z",
            ],
            input="".join(f"{ref}:{p}\0" for p in paths),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=git_env(None),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    # One reply line per request, in order: the size, or "<input> missing".
    sizes: dict[str, int] = {}
    for path, line in zip(paths, proc.stdout.splitlines()):
        text = line.strip()
        if text.isdigit():
            sizes[path] = int(text)
    return sizes


def read_blob(root: Path, ref: str, path: str) -> str | None:
    """Return the UTF-8 text of ``ref:path``, or ``None``.

    ``None`` means the blob is missing or not decodable as UTF-8 (binary) — the
    same "skip it" outcome the on-disk indexer reaches on ``UnicodeDecodeError``.
    """
    if not _is_safe_ref(ref):
        return None
    try:
        # Not _run_git: blobs are read as bytes so binary files can be detected
        # by a failed decode rather than mangled by text mode.
        proc = subprocess.run(
            ["git", "-C", str(root), "-c", f"safe.directory={root}", "show", f"{ref}:{path}"],
            capture_output=True,
            text=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=git_env(None),
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
