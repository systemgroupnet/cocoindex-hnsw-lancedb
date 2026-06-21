"""Shared LanceDB store configuration: table layout, HNSW tuning, helpers.

Single source of truth for the LanceDB vector store used by both the write
path (:mod:`indexer`) and the read path (:mod:`query`).  Keeping the table
name, vector column, distance metric, and HNSW parameters here avoids drift
between the indexer that writes the table and the query path that reads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lancedb.table import AsyncTable

# Table layout ---------------------------------------------------------------

TABLE_NAME = "code_chunks"
VECTOR_COLUMN = "embedding"

# Vector search configuration ------------------------------------------------

# Cosine distance matches the unit-norm embeddings produced by the embedders
# (the params layer deliberately keeps `normalize_embeddings` on). With cosine
# distance, ``score = 1 - distance`` is exactly cosine similarity, which keeps
# the returned score on the same scale as the old sqlite-vec L2->cosine path.
DISTANCE_TYPE = "cosine"

# HNSW (graph) index build parameters. ``HnswFlat`` keeps full-precision
# vectors (no quantization), so recall stays high — important for code search
# where a near-miss chunk is a real miss. These are conservative, recall-
# favoring defaults; ``ef_construction`` trades build time for graph quality.
HNSW_M = 20
HNSW_EF_CONSTRUCTION = 300

# Query-time HNSW search width. Higher ``ef`` widens the graph traversal,
# raising recall at some latency cost. 256 favors recall (the chosen default
# posture for code search, where a missed chunk is a real miss) while still
# searching far faster than an exact scan. Tunable per query via
# :func:`cocoindex_code.query.query_codebase`.
DEFAULT_EF_SEARCH = 256

# Below this row count an ANN index is pointless: LanceDB's automatic flat
# (brute-force) fallback is already exact and fast, and training an index on a
# handful of vectors only adds overhead. At/above it we build the HNSW graph so
# query latency stays sublinear as the codebase grows.
INDEX_MIN_ROWS = 256


def score_from_distance(distance: float) -> float:
    """Convert a LanceDB cosine ``_distance`` into a similarity score.

    Cosine distance is ``1 - cosine_similarity``, so the similarity score is
    ``1 - distance``. This matches the 0..1-ish scale (1.0 = identical) the
    sqlite-vec path produced for unit vectors, keeping ``QueryResult.score``
    backward compatible.
    """
    return 1.0 - distance


async def ensure_vector_index(table: AsyncTable) -> bool:
    """Ensure an HNSW vector index exists on the embedding column.

    Idempotent and cheap to call after each index run:

    * Returns ``False`` without touching the table when the row count is below
      :data:`INDEX_MIN_ROWS` (flat search is exact and fast there) or when a
      vector index already exists (LanceDB's periodic ``optimize()`` — invoked
      by the CocoIndex target after mutation batches — folds newly upserted
      rows into the existing index, so we don't rebuild on every run).
    * Otherwise builds an :class:`~lancedb.index.HnswFlat` cosine index and
      returns ``True``.
    """
    from lancedb.index import HnswFlat

    if await table.count_rows() < INDEX_MIN_ROWS:
        return False

    for existing in await table.list_indices():
        if VECTOR_COLUMN in existing.columns:
            return False

    await table.create_index(
        VECTOR_COLUMN,
        config=HnswFlat(
            distance_type=DISTANCE_TYPE,
            m=HNSW_M,
            ef_construction=HNSW_EF_CONSTRUCTION,
        ),
        replace=True,
    )
    return True
