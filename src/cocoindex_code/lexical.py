"""Lexical (keyword) search over a set of in-memory files.

Used by branch search for the diff side: the branch's own version of the files
it added or modified is searched here and returned as a distinct
``source="lexical"`` section alongside semantic results from the base index.

Prefers the ``rg`` (ripgrep) binary to locate candidate lines when it is on
``PATH`` (via :mod:`cocoindex_code.ripgrep`); otherwise falls back to an
equivalent in-process Python scan. Scoring and snippet extraction always run in
Python over the content held in memory, so the two paths return identical
results — ripgrep is only a candidate-finding accelerator, never a source of
truth.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from . import ripgrep
from .memory import DEFAULT_SCAN_BUDGET, ScanBudget

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
    budget: ScanBudget = DEFAULT_SCAN_BUDGET,
) -> list[LexicalHit]:
    """Return up to *limit* lexical matches for *query* across *files*.

    A line matches if it contains any query term (case-insensitive). Each match
    is scored by the fraction of distinct query terms present in its context
    window, so a line hitting more of the query ranks higher; ties break by file
    path then line number for stable output.

    Scores depend only on the hit's own snippet, never on the rest of the
    corpus, so callers may split a large file set into batches and merge the
    per-batch winners without changing the ranking.
    """
    file_list = list(files)
    terms = extract_terms(query)
    if not terms or not file_list:
        return []

    by_path = {f.path: f for f in file_list}
    candidates = _rg_candidate_lines(file_list, terms, budget)
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
    files: list[LexicalFile], terms: list[str], budget: ScanBudget
) -> dict[str, set[int]] | None:
    """Find candidate lines with ripgrep, or ``None`` if rg is unusable.

    One case-insensitive fixed-string pass over the files' content. No limit:
    scoring below ranks the full candidate set, so truncating here would change
    which hits win. ``None`` (rg absent or unrunnable) sends the caller to the
    Python scan; an empty dict means rg ran and found nothing.
    """
    outcome = ripgrep.search_blobs(
        {f.path: f.content for f in files},
        ripgrep.RipgrepQuery(patterns=tuple(terms), fixed_strings=True),
        budget=budget,
    )
    if outcome is None:
        return None
    result: dict[str, set[int]] = {}
    for match in outcome.matches:
        result.setdefault(match.file_path, set()).add(match.line_number)
    return result
