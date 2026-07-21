"""Unit tests for the lexical (keyword) search used by the branch-search
high-divergence fallback. Pure in-process path — no ripgrep, no I/O."""

from __future__ import annotations

from cocoindex_code.lexical import LexicalFile, extract_terms, lexical_search


def test_extract_terms_drops_stopwords_and_short_tokens() -> None:
    terms = extract_terms("How are users authenticated in the DB?")
    assert "how" not in terms  # stopword
    assert "are" not in terms  # stopword
    assert "db" not in terms  # too short (< 3)
    assert "users" in terms
    assert "authenticated" in terms


def test_extract_terms_dedupes_preserving_order() -> None:
    assert extract_terms("cache cache CACHE lookup") == ["cache", "lookup"]


def test_extract_terms_empty_for_no_signal() -> None:
    assert extract_terms("is a to the of") == []


def test_lexical_search_finds_matching_lines() -> None:
    files = [
        LexicalFile(
            path="auth.py",
            content="def login(user):\n    token = make_token(user)\n    return token\n",
            language="python",
        ),
        LexicalFile(
            path="math.py",
            content="def add(a, b):\n    return a + b\n",
            language="python",
        ),
    ]
    hits = lexical_search(files, "token", limit=10)
    assert hits, "expected a lexical hit for 'token'"
    assert hits[0].file_path == "auth.py"
    assert "token" in hits[0].content.lower()
    assert hits[0].start_line >= 1
    assert hits[0].end_line >= hits[0].start_line


def test_lexical_search_ranks_more_terms_higher() -> None:
    files = [
        LexicalFile(path="a.py", content="database connection pool here\n", language="python"),
        LexicalFile(path="b.py", content="database only\n", language="python"),
    ]
    hits = lexical_search(files, "database connection pool", limit=10)
    assert hits[0].file_path == "a.py"  # matches all three terms
    assert hits[0].score > hits[-1].score


def test_lexical_search_empty_when_no_terms() -> None:
    files = [LexicalFile(path="a.py", content="whatever\n", language="python")]
    assert lexical_search(files, "a to the of", limit=10) == []


def test_lexical_search_respects_limit() -> None:
    files = [
        LexicalFile(path=f"f{i}.py", content="token here\n", language="python") for i in range(10)
    ]
    hits = lexical_search(files, "token", limit=3)
    assert len(hits) == 3
