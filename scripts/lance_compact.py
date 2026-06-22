"""Standalone LanceDB diagnostic + compaction — reclaims disk without the daemon.

Usage:
    python scripts/lance_compact.py <project-root | .cocoindex_code | lancedb dir>
    python scripts/lance_compact.py <path> --inspect   # report only, no changes

This talks to the LanceDB store directly, so it does NOT require restarting the
``ccc`` daemon. It first reports what's actually consuming space (row count,
number of retained versions, on-disk size), then — unless ``--inspect`` is
passed — runs a full ``optimize`` (compact fragments + prune every version but
the latest) and reports the LanceDB-side stats plus how much disk was reclaimed.

IMPORTANT: stop indexing first (``ccc daemon stop``). The prune uses
``delete_unverified=True``, which must not race a concurrent writer.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import lancedb

TABLE = "code_chunks"


def _resolve_db_dir(arg: Path) -> Path:
    """Accept a project root, a .cocoindex_code dir, or the lancedb dir itself."""
    candidates = [
        arg / ".cocoindex_code" / "lancedb",
        arg / "lancedb",
        arg,
    ]
    for cand in candidates:
        if (cand / f"{TABLE}.lance").exists():
            return cand
    raise SystemExit(f"Could not find {TABLE}.lance under any of: {[str(c) for c in candidates]}")


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


async def main(arg: str, inspect_only: bool) -> None:
    db_dir = _resolve_db_dir(Path(arg).resolve())
    print(f"LanceDB dir: {db_dir}")

    conn = await lancedb.connect_async(str(db_dir))
    table = await conn.open_table(TABLE)

    rows = await table.count_rows()
    versions = len(await table.list_versions())
    before = _dir_size(db_dir)
    print(f"  rows:     {rows:,}")
    print(f"  versions: {versions:,}")
    print(f"  size:     {_gb(before)}")
    if rows:
        print(f"  ~bytes/row (live):  {before / rows:,.0f}  (high => unpruned/uncompacted)")

    if inspect_only:
        return

    print("\nOptimizing: compacting fragments and pruning all versions but the latest...")
    stats = await table.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)
    after = _dir_size(db_dir)
    versions_after = len(await table.list_versions())

    print(f"\nLanceDB stats: {stats}")
    print(f"  versions: {versions:,} -> {versions_after:,}")
    print(f"  size:     {_gb(before)} -> {_gb(after)}")
    print(f"  reclaimed:{_gb(max(0, before - after))}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    inspect_only = "--inspect" in sys.argv[1:]
    if len(args) != 1:
        raise SystemExit("usage: lance_compact.py <project-root | lancedb dir> [--inspect]")
    asyncio.run(main(args[0], inspect_only))
