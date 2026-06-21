"""Benchmark LanceDB HNSW (approximate) vs flat brute-force search latency.

The previous sqlite-vec backend did exact brute-force KNN (a linear scan over
every chunk vector per query) — algorithmically identical to LanceDB's flat
search (``bypass_vector_index``). This script therefore compares:

    * **flat**  — exact, O(n) scan  (the old sqlite-vec behavior)
    * **HNSW**  — approximate graph index  (the new default)

across a few index sizes, reporting mean/p95 query latency and HNSW recall@k
against the flat ground truth. It uses synthetic unit vectors so it runs in
seconds without an embedding model.

Run:

    uv run python scripts/benchmark_search.py
    uv run python scripts/benchmark_search.py --sizes 1000 10000 100000 --dim 384
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import HnswFlat

# Keep these in sync with cocoindex_code.lancedb_store.
VECTOR_COLUMN = "embedding"
DISTANCE_TYPE = "cosine"
DEFAULT_EF_SEARCH = 256
HNSW_M = 20
HNSW_EF_CONSTRUCTION = 300


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _clustered_corpus(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate clustered unit vectors.

    Uniform-random high-dimensional vectors are the adversarial worst case for
    any ANN index (every pair is near-equidistant), so they badly understate
    real recall. Real code embeddings cluster by topic/language, so we sample
    around a set of cluster centers — a far more representative corpus.
    """
    n_clusters = max(8, n // 500)
    centers = _normalize(rng.standard_normal((n_clusters, dim)).astype(np.float32))
    assignment = rng.integers(0, n_clusters, size=n)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * 0.15
    return _normalize(centers[assignment] + noise)


def _queries_near_corpus(
    corpus: np.ndarray, n_queries: int, rng: np.random.Generator
) -> np.ndarray:
    """Queries are perturbations of random corpus points.

    This mirrors the real "find code similar to X" pattern and gives each
    query a well-defined nearest neighbour, so recall@k is meaningful.
    """
    idx = rng.integers(0, len(corpus), size=n_queries)
    noise = rng.standard_normal((n_queries, corpus.shape[1])).astype(np.float32) * 0.05
    return _normalize(corpus[idx] + noise)


async def _build_table(
    conn: lancedb.AsyncConnection, vectors: np.ndarray, dim: int
) -> lancedb.table.AsyncTable:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(VECTOR_COLUMN, pa.list_(pa.float32(), dim)),
        ]
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(list(range(len(vectors))), pa.int64()),
            pa.array(list(vectors), pa.list_(pa.float32(), dim)),
        ],
        schema=schema,
    )
    return await conn.create_table("bench", batch, mode="overwrite")


async def _search(
    table: lancedb.table.AsyncTable, query: np.ndarray, k: int, *, flat: bool, ef: int
) -> list[int]:
    q = await table.search(query)
    q = q.distance_type(DISTANCE_TYPE)
    if flat:
        q = q.bypass_vector_index()
    else:
        q = q.ef(ef)
    rows = await q.select(["id"]).limit(k).to_list()
    return [r["id"] for r in rows]


async def _time_queries(
    table: lancedb.table.AsyncTable,
    queries: np.ndarray,
    k: int,
    *,
    flat: bool,
    ef: int,
) -> tuple[list[float], list[list[int]]]:
    latencies: list[float] = []
    results: list[list[int]] = []
    for query in queries:
        start = time.perf_counter()
        ids = await _search(table, query, k, flat=flat, ef=ef)
        latencies.append((time.perf_counter() - start) * 1000.0)
        results.append(ids)
    return latencies, results


def _recall(approx: list[list[int]], exact: list[list[int]], k: int) -> float:
    hits = 0
    total = 0
    for a, e in zip(approx, exact):
        e_set = set(e[:k])
        hits += len(set(a[:k]) & e_set)
        total += len(e_set)
    return hits / total if total else 1.0


async def _bench_size(
    size: int, dim: int, k: int, n_queries: int, ef: int, rng: np.random.Generator
) -> None:
    workdir = Path(tempfile.mkdtemp(prefix=f"ccc_bench_{size}_"))
    try:
        conn = await lancedb.connect_async(str(workdir))
        vectors = _clustered_corpus(size, dim, rng)
        table = await _build_table(conn, vectors, dim)
        queries = _queries_near_corpus(vectors, n_queries, rng)

        # Flat (exact) — ground truth + the "old sqlite-vec" latency baseline.
        flat_lat, flat_ids = await _time_queries(table, queries, k, flat=True, ef=ef)

        # Build HNSW and search.
        await table.create_index(
            VECTOR_COLUMN,
            config=HnswFlat(
                distance_type=DISTANCE_TYPE, m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION
            ),
            replace=True,
        )
        hnsw_lat, hnsw_ids = await _time_queries(table, queries, k, flat=False, ef=ef)

        recall = _recall(hnsw_ids, flat_ids, k)
        speedup = statistics.mean(flat_lat) / statistics.mean(hnsw_lat)
        print(
            f"{size:>9,} | "
            f"flat {statistics.mean(flat_lat):7.2f}ms (p95 {_p95(flat_lat):7.2f}) | "
            f"hnsw {statistics.mean(hnsw_lat):6.2f}ms (p95 {_p95(hnsw_lat):6.2f}) | "
            f"speedup {speedup:5.1f}x | recall@{k} {recall:5.3f}"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _p95(xs: list[float]) -> float:
    return sorted(xs)[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 50_000, 200_000])
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--k", type=int, default=10, help="Top-k")
    parser.add_argument("--queries", type=int, default=50, help="Queries per size")
    parser.add_argument("--ef", type=int, default=DEFAULT_EF_SEARCH, help="HNSW search ef")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    print(
        f"dim={args.dim} k={args.k} queries={args.queries} ef={args.ef}  "
        f"(flat = exact brute-force ~ old sqlite-vec)\n"
        f"{'rows':>9} | {'exact (flat)':^24} | {'approx (HNSW)':^22} | "
        f"{'':^11} | recall"
    )
    for size in args.sizes:
        await _bench_size(size, args.dim, args.k, args.queries, args.ef, rng)


if __name__ == "__main__":
    asyncio.run(_main())
