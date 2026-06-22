"""Recover a LanceDB table corrupted by an interrupted write.

Symptom (from `ccc compact`, search, or index):

    lance error: Encountered corrupt file .../_deletions/<frag>-<ver>-<id>.bin:
    failed to fill whole buffer

A truncated file under ``_deletions/`` (or ``data/``) means a write was killed
partway — daemon OOM, ``docker stop`` mid-merge, disk full, etc. The bad file
belongs to a specific table *version*; every later version that references it is
unreadable, but earlier versions are intact.

Strategy: walk backwards from the latest version to the most recent one that
reads cleanly, ``restore()`` it as the new latest, then prune the orphaned
(including corrupt) files. Changes after that version are dropped — re-run
``ccc index`` afterward to re-apply them incrementally (cheap, memoized).

Usage:
    python scripts/lance_recover.py <project-root | lancedb dir>            # dry run
    python scripts/lance_recover.py <project-root | lancedb dir> --apply    # fix it

Stop the daemon first: ``ccc daemon stop``.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import lancedb

TABLE = "code_chunks"


def _resolve_db_dir(arg: Path) -> Path:
    for cand in (arg / ".cocoindex_code" / "lancedb", arg / "lancedb", arg):
        if (cand / f"{TABLE}.lance").exists():
            return cand
    raise SystemExit(f"Could not find {TABLE}.lance under {arg}")


async def _readable(table: "lancedb.table.AsyncTable") -> bool:
    """True if the checked-out version scans cleanly.

    Does a FULL single-column scan (not ``limit(1)``): applying every fragment's
    deletion vector is what forces Lance to actually open each ``_deletions``
    file, which is where a truncated one surfaces. ``count_rows`` alone uses
    manifest metadata and won't catch it, and a tiny ``limit`` may only touch the
    first fragment. The ``id`` column is the cheapest to read.
    """
    try:
        await table.query().select(["id"]).to_list()
        return True
    except Exception as e:
        print(f"    unreadable: {str(e).splitlines()[0]}")
        return False


async def main(arg: str, apply: bool) -> int:
    db_dir = _resolve_db_dir(Path(arg).resolve())
    print(f"LanceDB dir: {db_dir}")
    conn = await lancedb.connect_async(str(db_dir))
    table = await conn.open_table(TABLE)

    versions = sorted({v["version"] for v in await table.list_versions()})
    print(f"versions: {versions[0]}..{versions[-1]} ({len(versions)} total)")

    good: int | None = None
    for ver in reversed(versions):
        await table.checkout(ver)
        print(f"  checking version {ver}...")
        if await _readable(table):
            good = ver
            print(f"  -> version {ver} reads cleanly")
            break
    await table.checkout_latest()

    if good is None:
        print("No readable version found — the table cannot be recovered by rollback.")
        print("Rebuild instead: delete the lancedb dir + cocoindex.db, then `ccc index`.")
        return 1

    if good == versions[-1]:
        print("Latest version is already readable — no corruption detected by rollback.")
        return 0

    dropped = [v for v in versions if v > good]
    print(f"\nLatest readable version: {good}. Restoring it drops versions: {dropped}")

    if not apply:
        print("Dry run — re-run with --apply to restore and prune.")
        return 0

    await table.checkout(good)
    await table.restore()
    await table.checkout_latest()
    print(f"Restored to version {good}.")

    print("Pruning orphaned/corrupt files...")
    stats = await table.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)
    print(f"  {stats}")
    rows = await table.count_rows()
    print(f"Done. rows={rows:,}. Now run `ccc index` to re-apply any dropped changes.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv[1:]
    if len(args) != 1:
        raise SystemExit("usage: lance_recover.py <project-root | lancedb dir> [--apply]")
    raise SystemExit(asyncio.run(main(args[0], apply)))
