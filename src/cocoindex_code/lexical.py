"""Lexical (keyword) search over a set of in-memory files.

Used by the branch-search high-divergence path: when a branch changed too many
files to embed into a semantic overlay, its changed files are searched
lexically instead and returned as a separate ``source="lexical"`` section.

Prefers the ``rg`` (ripgrep) binary to locate candidate lines when it is on
``PATH``; otherwise falls back to an equivalent in-process Python scan. Scoring
and snippet extraction always run in Python over the content held in memory, so
the two paths return identical results — ripgrep is only a candidate-finding
accelerator, never a source of truth.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Query tokenization: alphanumeric/underscore runs, lowercased. Very short and
# ultra-common tokens carry no signal for code search and would match nearly
# every line, so we drop them.
_TERM_RE = re.compile(r"\w+")
_MIN_TERM_LEN = 3
_MAX_TERMS = 12
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "how", "does", "with", "that", "this", "are", "was",
        "you", "your", "from", "into", "use", "used", "using", "get", "set",
        "where", "what", "when", "which", "why", "who", "can", "will",
    }
)

# Lines around a match returned as context in the snippet.
_CONTEXT_LINES = 2
_RG_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class LexicalHit:
    """One lexical match, shaped like a search result row."""

    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    score: float


@dataclass(frozen=True)
class LexicalFile:
    """A file to search: repo-relative path, UTF-8 content, detected language."""

    path: str
    content: str
    language: str


def extract_terms(query: str) -> list[str]:
    """Tokenize *query* into distinct lowercased search terms (order preserved)."""
    seen: set[str] = set()
    terms: list[str] = []
    for match in _TERM_RE.finditer(query.lower()):
        term = match.group(0)
        if len(term) < _MIN_TERM_LEN or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= _MAX_TERMS:
            break
    return terms


def lexical_search(
    files: Iterable[LexicalFile],
    query: str,
    *,
    limit: int,
) -> list[LexicalHit]:
    """Return up to *limit* lexical matches for *query* across *files*.

    A line matches if it contains any query term (case-insensitive). Each match
    is scored by the fraction of distinct query terms present in its context
    window, so a line hitting more of the query ranks higher; ties break by file
    path then line number for stable output.
    """
    file_list = list(files)
    terms = extract_terms(query)
    if not terms or not file_list:
        return []

    by_path = {f.path: f for f in file_list}
    candidates = _rg_candidate_lines(file_list, terms)
    if candidates is None:
        candidates = _py_candidate_lines(file_list, terms)

    hits: list[LexicalHit] = []
    for path in sorted(candidates):
        lf = by_path.get(path)
        if lf is None:
            continue
        lines = lf.content.splitlines()
        for lineno in sorted(candidates[path]):
            start = max(1, lineno - _CONTEXT_LINES)
            end = min(len(lines), lineno + _CONTEXT_LINES)
            snippet = "\n".join(lines[start - 1 : end])
            snippet_lower = snippet.lower()
            matched = sum(1 for t in terms if t in snippet_lower)
            hits.append(
                LexicalHit(
                    file_path=path,
                    language=lf.language,
                    content=snippet,
                    start_line=start,
                    end_line=end,
                    score=matched / len(terms),
                )
            )

    hits.sort(key=lambda h: (-h.score, h.file_path, h.start_line))
    return hits[:limit]


def _py_candidate_lines(
    files: Iterable[LexicalFile], terms: list[str]
) -> dict[str, set[int]]:
    """Find 1-indexed line numbers containing any term, purely in Python."""
    result: dict[str, set[int]] = {}
    for lf in files:
        matched: set[int] = set()
        for idx, line in enumerate(lf.content.splitlines(), start=1):
            low = line.lower()
            if any(t in low for t in terms):
                matched.add(idx)
        if matched:
            result[lf.path] = matched
    return result


def _rg_candidate_lines(
    files: list[LexicalFile], terms: list[str]
) -> dict[str, set[int]] | None:
    """Find candidate lines with ripgrep, or ``None`` if rg is unusable.

    Writes each file's content to a temp tree and runs a single case-insensitive
    fixed-string ``rg --json`` pass. Returns ``None`` (so the caller falls back
    to Python) when rg is absent or errors; an empty dict means rg ran and found
    nothing.
    """
    if shutil.which("rg") is None:
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="ccc-lexical-") as tmp:
            tmp_root = Path(tmp)
            rel_by_abs: dict[str, str] = {}
            for lf in files:
                # git paths are repo-relative and never contain ".."; guard anyway
                # so a crafted path can't escape the temp tree.
                dest = (tmp_root / lf.path).resolve()
                if not str(dest).startswith(str(tmp_root.resolve())):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(lf.content, encoding="utf-8")
                rel_by_abs[str(dest)] = lf.path

            args = ["rg", "--json", "--ignore-case", "--fixed-strings"]
            for term in terms:
                args += ["-e", term]
            args.append(str(tmp_root))

            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=_RG_TIMEOUT_SECONDS, check=False
            )
            # 0 = matches, 1 = no matches (both fine); >=2 = real error -> fallback.
            if proc.returncode >= 2:
                return None

            result: dict[str, set[int]] = {}
            for raw in proc.stdout.splitlines():
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "match":
                    continue
                data = event["data"]
                abs_path = data.get("path", {}).get("text")
                lineno = data.get("line_number")
                rel = rel_by_abs.get(os.path.abspath(abs_path)) if abs_path else None
                if rel is None or lineno is None:
                    continue
                result.setdefault(rel, set()).add(int(lineno))
            return result
    except (OSError, subprocess.TimeoutExpired):
        return None
