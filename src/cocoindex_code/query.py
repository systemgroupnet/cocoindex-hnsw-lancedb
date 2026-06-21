"""Query implementation for codebase search (LanceDB / HNSW backend)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .lancedb_store import (
    DEFAULT_EF_SEARCH,
    DISTANCE_TYPE,
    TABLE_NAME,
    score_from_distance,
)
from .schema import QueryResult
from .shared import EMBEDDER, LANCE_DB, QUERY_EMBED_PARAMS

if TYPE_CHECKING:
    from lancedb.table import AsyncTable


def _sql_str_literal(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _glob_to_like(pattern: str) -> str:
    """Translate a shell GLOB path pattern into a SQL ``LIKE`` pattern.

    The sqlite-vec path filtered with ``file_path GLOB ?``; LanceDB's filter is
    SQL (DataFusion), which has no ``GLOB`` but does have ``LIKE``. We map:

    * ``*`` -> ``%`` (match any run of characters, including ``/`` — same as
      sqlite GLOB, so ``lib/*`` still matches ``lib/sub/x.py``)
    * ``?`` -> ``_`` (match a single character)

    Literal ``%`` / ``_`` / ``\\`` in the pattern are escaped so they aren't
    treated as wildcards (``LIKE ... ESCAPE '\\'``). GLOB character classes
    (``[...]``) are not supported by ``LIKE`` and are left as literal text;
    this is the one documented filtering delta from the sqlite-vec backend.
    """
    out: list[str] = []
    for ch in pattern:
        if ch in ("\\", "%", "_"):
            out.append("\\" + ch)
        elif ch == "*":
            out.append("%")
        elif ch == "?":
            out.append("_")
        else:
            out.append(ch)
    return "".join(out)


def _build_filter(
    languages: list[str] | None,
    paths: list[str] | None,
) -> str | None:
    """Build a LanceDB SQL ``WHERE`` predicate from language/path filters.

    Languages are matched exactly (``language IN (...)``); paths are matched
    via translated ``LIKE`` patterns OR'd together. Returns ``None`` when no
    filter applies.
    """
    clauses: list[str] = []

    if languages:
        in_list = ", ".join(_sql_str_literal(lang) for lang in languages)
        clauses.append(f"language IN ({in_list})")

    if paths:
        like_clauses = " OR ".join(
            f"file_path LIKE {_sql_str_literal(_glob_to_like(p))} ESCAPE '\\'" for p in paths
        )
        clauses.append(f"({like_clauses})")

    if not clauses:
        return None
    return " AND ".join(clauses)


async def query_codebase(
    query: str,
    table: AsyncTable,
    env: Any,
    limit: int = 10,
    offset: int = 0,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
    ef_search: int = DEFAULT_EF_SEARCH,
) -> list[QueryResult]:
    """Perform vector similarity search over the LanceDB ``code_chunks`` table.

    Uses LanceDB's vector search, which transparently uses the HNSW index when
    present and falls back to an exact flat scan otherwise (small codebases).
    HNSW is approximate: ``ef_search`` widens the graph traversal to trade a
    little latency for higher recall. Language and path filters are applied as
    a pre-filter (the predicate is evaluated before the KNN step) so the top-k
    is taken from the filtered set rather than truncated after the fact.

    Scores are cosine similarity (``1 - cosine_distance``), the same scale as
    the legacy sqlite-vec path.
    """
    embedder = env.get_context(EMBEDDER)
    query_params = env.get_context(QUERY_EMBED_PARAMS)

    query_embedding = await embedder.embed(query, **query_params)

    search = await table.search(query_embedding.astype("float32"))
    search = search.distance_type(DISTANCE_TYPE).ef(ef_search)

    predicate = _build_filter(languages, paths)
    if predicate is not None:
        search = search.where(predicate)

    # Fetch limit+offset then slice, since pagination here is "skip N then take
    # M" over a single ranked result set (mirrors the legacy sqlite-vec path).
    rows = await search.limit(limit + offset).to_list()
    rows = rows[offset : offset + limit]

    return [
        QueryResult(
            file_path=row["file_path"],
            language=row["language"],
            content=row["content"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            score=score_from_distance(row["_distance"]),
        )
        for row in rows
    ]


async def open_table(env: Any) -> AsyncTable:
    """Open the LanceDB ``code_chunks`` table from the project's connection.

    Raises ``RuntimeError`` with a user-facing hint when the table does not
    exist yet (i.e. the project has not been indexed).
    """
    conn = env.get_context(LANCE_DB)
    try:
        return await conn.open_table(TABLE_NAME)
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(
            "Index not found. Please run `ccc index` (or query with refresh) first."
        ) from e
