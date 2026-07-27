"""Ripgrep-backed text search over a working tree, a branch, or in-memory blobs.

Backs the MCP ``ripgrep`` tool and supplies candidate lines to
:mod:`cocoindex_code.lexical`, so both go through one rg argument builder and
one ``--json`` parser.

Every entry point returns ``None`` when rg is unusable — absent from ``PATH``,
unlaunchable, or killed by the timeout. That distinction (rather than "no
matches") is what lets ``lexical`` fall back to its in-process scan and the MCP
tool report an actionable message instead of a silently empty result.

Searching a *branch* mirrors the branch-overlay model: the base working tree
minus the files the branch touched, plus the branch's own version of the files
it added or modified — read out of the object database, never checked out.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import git_ops
from .settings import SETTINGS_DIR_NAME

# Hard ceiling on a single rg invocation. A pathological pattern (catastrophic
# backtracking is off the table with rg's regex engine, but a huge tree isn't)
# must not wedge a request; on timeout the process is killed and whatever was
# collected so far is returned as a truncated result.
_RG_TIMEOUT_SECONDS = 30

# Lines of context included on each side when the caller asks for context.
_MAX_CONTEXT_LINES = 20


@dataclass(frozen=True)
class RipgrepQuery:
    """What to search for. ``patterns`` are OR'd, exactly as repeated ``-e``."""

    patterns: tuple[str, ...]
    # Maximum matches to return; None means "every match" (used by lexical,
    # which scores the full candidate set itself).
    limit: int | None = None
    globs: tuple[str, ...] = ()
    case_sensitive: bool = False
    fixed_strings: bool = False
    context_lines: int = 0


@dataclass(frozen=True)
class RipgrepMatch:
    """One matching line, with its context window when one was requested."""

    file_path: str
    line_number: int
    content: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class RipgrepOutcome:
    matches: tuple[RipgrepMatch, ...] = ()
    # True when the limit (or the timeout) cut the scan short, so the caller can
    # say "there are more" rather than implying these are all of them.
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.matches


@dataclass
class _Collector:
    """Accumulates matches from one rg run, stopping at the query's limit."""

    query: RipgrepQuery
    matches: list[RipgrepMatch] = field(default_factory=list)
    truncated: bool = False

    @property
    def full(self) -> bool:
        return self.query.limit is not None and len(self.matches) >= self.query.limit


def available() -> bool:
    """Whether the ``rg`` binary is on ``PATH``."""
    return shutil.which("rg") is not None


def search_tree(
    root: Path,
    query: RipgrepQuery,
    *,
    exclude_paths: Collection[str] = (),
) -> RipgrepOutcome | None:
    """Search the working tree at *root*.

    Honors the repo's ignore rules (rg reads ``.gitignore``) and searches hidden
    files, minus ``.git`` and the index directory. Paths in *exclude_paths* are
    dropped from the results rather than excluded via rg globs: a repo path can
    contain gitignore metacharacters, and post-filtering a known-small set is
    exact where escaping them would be fragile.
    """
    excluded = set(exclude_paths)

    def to_rel(reported: str) -> str | None:
        rel = _to_posix_rel(reported)
        return None if rel is None or rel in excluded else rel

    return _run(
        _rg_args(query, extra_globs=(f"!{SETTINGS_DIR_NAME}/**",)),
        cwd=root,
        to_rel=to_rel,
        query=query,
        read_lines=lambda rel: _read_file(root, rel),
    )


def search_blobs(blobs: Mapping[str, str], query: RipgrepQuery) -> RipgrepOutcome | None:
    """Search in-memory file contents, keyed by repo-relative path.

    The contents are materialized into a temp tree so rg can do the scanning;
    results come back keyed by the original repo-relative path.
    """
    if not blobs:
        return RipgrepOutcome()
    try:
        with tempfile.TemporaryDirectory(prefix="ccc-rg-") as tmp:
            tmp_root = Path(tmp)
            written = _materialize(blobs, tmp_root)
            if not written:
                return RipgrepOutcome()
            return _run(
                _rg_args(query, extra_globs=()),
                cwd=tmp_root,
                to_rel=_to_posix_rel,
                query=query,
                read_lines=lambda rel: blobs.get(rel, "").splitlines(),
            )
    except OSError:
        return None


def search_branch(
    root: Path,
    query: RipgrepQuery,
    *,
    branch_sha: str,
    branch_paths: Sequence[str],
    shadow_paths: Sequence[str],
) -> RipgrepOutcome | None:
    """Search *branch_sha*'s view of the codebase without checking it out.

    The base working tree supplies every file the branch left alone (files it
    modified or deleted are hidden), and the branch's own version of the files
    it added or modified is read from the object database. Same decomposition as
    a semantic branch search, so the two agree on what "the branch" contains.
    """
    tree = search_tree(root, query, exclude_paths=shadow_paths)
    if tree is None:
        return None

    blobs: dict[str, str] = {}
    for path in branch_paths:
        content = git_ops.read_blob(root, branch_sha, path)
        if content is not None:
            blobs[path] = content
    branch = search_blobs(blobs, query)
    if branch is None:
        return None

    merged = sorted(
        tree.matches + branch.matches, key=lambda m: (m.file_path, m.line_number)
    )
    truncated = tree.truncated or branch.truncated
    if query.limit is not None and len(merged) > query.limit:
        merged = merged[: query.limit]
        truncated = True
    return RipgrepOutcome(tuple(merged), truncated)


# --- internals ---------------------------------------------------------------


def _rg_args(query: RipgrepQuery, *, extra_globs: Sequence[str]) -> list[str]:
    """Build the rg command line, always searching ``.`` from the run's cwd.

    Searching ``.`` (rather than an absolute root) is what makes ``--glob``
    patterns anchor at the search root — rg matches them against the path as
    reported, so an absolute target would anchor them at the filesystem root and
    ``src/**`` would never match. Patterns go through ``-e`` and the target
    after ``--``, so neither can be read as an option.
    """
    args = ["rg", "--json", "--no-messages", "--hidden", "--glob", "!.git/**"]
    for glob in (*extra_globs, *query.globs):
        args += ["--glob", glob]
    args.append("--case-sensitive" if query.case_sensitive else "--ignore-case")
    if query.fixed_strings:
        args.append("--fixed-strings")
    for pattern in query.patterns:
        args += ["-e", pattern]
    args += ["--", "."]
    return args


def _run(
    args: list[str],
    *,
    cwd: Path,
    to_rel: Callable[[str], str | None],
    query: RipgrepQuery,
    read_lines: Callable[[str], list[str]],
) -> RipgrepOutcome | None:
    """Run rg, streaming its JSON so a broad pattern stops at the limit.

    Returns ``None`` if rg can't be run at all; a killed-by-timeout run yields
    whatever was collected, flagged as truncated.
    """
    if not query.patterns:
        return RipgrepOutcome()
    if not available():
        return None
    try:
        proc = subprocess.Popen(  # noqa: S603 - args are built here, never shell
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    collector = _Collector(query)
    # rg streams, so subprocess.run's timeout doesn't apply — a watchdog thread
    # is what bounds the run. Its flag distinguishes "rg finished" from "we
    # killed it", which decides whether the result is truncated.
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(_RG_TIMEOUT_SECONDS, _on_timeout)
    watchdog.start()
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if not _consume(raw, collector, to_rel=to_rel, read_lines=read_lines):
                break
    finally:
        watchdog.cancel()
        _terminate(proc)

    # rg walks in parallel, so its output order varies run to run. Sort so the
    # same query reads the same way twice. (Which matches survive a truncated
    # run still depends on the walk — that's what `truncated` warns about.)
    ordered = sorted(collector.matches, key=lambda m: (m.file_path, m.line_number))
    return RipgrepOutcome(tuple(ordered), collector.truncated or timed_out.is_set())


def _consume(
    raw: str,
    collector: _Collector,
    *,
    to_rel: Callable[[str], str | None],
    read_lines: Callable[[str], list[str]],
) -> bool:
    """Fold one rg JSON event into *collector*. Returns False to stop reading."""
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return True
    if event.get("type") != "match":
        return True

    data = event.get("data") or {}
    abs_path = (data.get("path") or {}).get("text")
    line_number = data.get("line_number")
    # `lines.text` is absent when the line isn't valid UTF-8 (rg sends base64
    # `bytes` instead). Such a line has no useful textual result, so skip it.
    line_text = (data.get("lines") or {}).get("text")
    if abs_path is None or line_number is None or line_text is None:
        return True

    rel = to_rel(abs_path)
    if rel is None:
        return True
    if collector.full:
        collector.truncated = True
        return False

    collector.matches.append(
        _build_match(rel, int(line_number), line_text, collector.query, read_lines)
    )
    return True


def _build_match(
    rel: str,
    line_number: int,
    line_text: str,
    query: RipgrepQuery,
    read_lines: Callable[[str], list[str]],
) -> RipgrepMatch:
    """Turn a raw rg hit into a match, expanding context if the query asked."""
    context = min(max(query.context_lines, 0), _MAX_CONTEXT_LINES)
    if context == 0:
        return RipgrepMatch(
            file_path=rel,
            line_number=line_number,
            content=line_text.rstrip("\r\n"),
            start_line=line_number,
            end_line=line_number,
        )

    lines = read_lines(rel)
    if not lines:
        return RipgrepMatch(
            file_path=rel,
            line_number=line_number,
            content=line_text.rstrip("\r\n"),
            start_line=line_number,
            end_line=line_number,
        )
    start = max(1, line_number - context)
    end = min(len(lines), line_number + context)
    return RipgrepMatch(
        file_path=rel,
        line_number=line_number,
        content="\n".join(lines[start - 1 : end]),
        start_line=start,
        end_line=end,
    )


def _materialize(blobs: Mapping[str, str], tmp_root: Path) -> dict[str, str]:
    """Write *blobs* into *tmp_root*, returning the paths actually written."""
    written: dict[str, str] = {}
    resolved_root = str(tmp_root.resolve())
    for rel, content in blobs.items():
        # git paths are repo-relative and never contain ".."; guard anyway so a
        # crafted path can't escape the temp tree.
        dest = (tmp_root / rel).resolve()
        if not str(dest).startswith(resolved_root):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written[rel] = content
    return written


def _to_posix_rel(reported: str) -> str | None:
    """Normalize a path as rg reports it (``./src/a.py``, ``.\\src\\a.py``).

    rg echoes the search target, so with a ``.`` target every path is already
    relative to the run's cwd — only the ``./`` prefix and the separator need
    fixing. Anything that still climbs out is rejected.
    """
    rel = reported.replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    if not rel or rel == ".." or rel.startswith("../"):
        return None
    return rel


def _read_file(root: Path, rel: str) -> list[str]:
    try:
        return (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Close the pipe and make sure rg is gone, even if we stopped reading early."""
    if proc.stdout is not None:
        proc.stdout.close()
    if proc.poll() is None:
        proc.kill()
    proc.wait()
