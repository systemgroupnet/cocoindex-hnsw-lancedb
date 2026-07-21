"""Branch-search overlays: build, search, and evict per-branch indexes.

See ``docs/branch-search.md`` for the full design. In short: the persistent
``code_chunks`` table is the *base*; each searched branch commit gets a small,
ephemeral ``overlay_<sha>`` table holding just the chunks of the files it
added/modified. A branch search merges the overlay with the base (minus the
files the branch touched — the "shadow set"). Branches that changed too many
files skip the semantic overlay and fall back to a lexical scan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import numpy as np
from cocoindex.ops.text import detect_code_language
from cocoindex.resources.file import PatternFilePathMatcher

from . import git_ops
from .chunking import CHUNKER_REGISTRY
from .indexer import GitignoreAwareMatcher, chunk_file_content
from .lexical import LexicalFile, lexical_search
from .protocol import SearchResult
from .query import open_table, query_codebase
from .schema import QueryResult
from .settings import lancedb_dir_path, load_gitignore_spec, load_project_settings
from .shared import (
    EMBEDDER,
    INDEXING_EMBED_PARAMS,
    LANCE_DB,
    MEMORY_GOVERNOR,
)

if TYPE_CHECKING:
    import cocoindex as coco
    from lancedb.table import AsyncTable

logger = logging.getLogger(__name__)

# --- Config knobs (single source of truth) ----------------------------------

ENV_MAX_CHANGED_FILES = "COCOINDEX_CODE_BRANCH_MAX_CHANGED_FILES"
ENV_OVERLAY_TTL_DAYS = "COCOINDEX_CODE_BRANCH_OVERLAY_TTL_DAYS"

_DEFAULT_MAX_CHANGED_FILES = 50
_DEFAULT_TTL_DAYS = 7.0

_SIDECAR_NAME = "overlays.json"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return value if value > 0 else default


def _overlay_table_name(sha: str) -> str:
    return f"overlay_{sha[:12]}"


class BranchOverlayManager:
    """Owns overlay tables for one project: build/reuse, search-merge, evict.

    One instance per :class:`~cocoindex_code.project.Project`. All table
    creation, sidecar mutation, and eviction serialize on ``_lock`` (the daemon
    is single-event-loop, so this only guards the build/evict critical sections;
    plain reads/searches run concurrently once an overlay exists).
    """

    def __init__(self, env: coco.Environment, project_root: Path) -> None:
        self._env = env
        self._root = project_root
        self._lock = asyncio.Lock()

    # -- public API ----------------------------------------------------------

    async def search(
        self,
        *,
        query: str,
        base_ref: str,
        branch_ref: str,
        languages: list[str] | None,
        paths: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[SearchResult]:
        """Search *branch_ref* overlaid on the base index.

        Raises ``RuntimeError`` when the branch can't be resolved in the local
        clone (no fetch yet) or the diff can't be computed.
        """
        branch_sha = git_ops.resolve_commit(self._root, branch_ref)
        if branch_sha is None:
            raise RuntimeError(
                f"ref {branch_ref!r} not found in the local clone "
                f"(on-demand fetch is not enabled yet — fetch it first)"
            )

        diff = git_ops.branch_diff(self._root, base_ref, branch_ref)
        if diff is None:
            raise RuntimeError(
                f"could not diff {branch_ref!r} against base {base_ref!r} "
                f"(is {base_ref!r} a valid ref?)"
            )

        await self.evict_stale()

        main_table = await open_table(self._env)

        # Branch identical to base (or only touches ignored files): plain search.
        embed_paths, shadow_paths = self._filter(diff)
        if not embed_paths and not shadow_paths:
            rows = await query_codebase(
                query, main_table, self._env,
                limit=limit, offset=offset, languages=languages, paths=paths,
            )
            return [_to_result(r, "semantic") for r in rows]

        if diff.total_changed <= _int_env(ENV_MAX_CHANGED_FILES, _DEFAULT_MAX_CHANGED_FILES):
            return await self._search_overlay(
                query=query, branch_ref=branch_ref, branch_sha=branch_sha,
                main_table=main_table, embed_paths=embed_paths, shadow_paths=shadow_paths,
                languages=languages, paths=paths, limit=limit, offset=offset,
            )

        logger.info(
            "Branch %s changed %d files (> %d); using lexical fallback",
            branch_ref, diff.total_changed,
            _int_env(ENV_MAX_CHANGED_FILES, _DEFAULT_MAX_CHANGED_FILES),
        )
        return await self._search_lexical(
            query=query, branch_ref=branch_ref, main_table=main_table,
            embed_paths=embed_paths, shadow_paths=shadow_paths,
            languages=languages, paths=paths, limit=limit, offset=offset,
        )

    # -- semantic overlay path -----------------------------------------------

    async def _search_overlay(
        self,
        *,
        query: str,
        branch_ref: str,
        branch_sha: str,
        main_table: AsyncTable,
        embed_paths: list[str],
        shadow_paths: list[str],
        languages: list[str] | None,
        paths: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[SearchResult]:
        overlay_table = await self._ensure_overlay(branch_ref, branch_sha, embed_paths)

        # Fetch limit+offset from each side, merge by score, then paginate — the
        # merged ranking is a single ordered set so offset/limit apply to it.
        take = limit + offset
        main_rows = await query_codebase(
            query, main_table, self._env,
            limit=take, offset=0, languages=languages, paths=paths,
            exclude_paths=shadow_paths or None,
        )
        overlay_rows: list[QueryResult] = []
        if overlay_table is not None:
            overlay_rows = await query_codebase(
                query, overlay_table, self._env,
                limit=take, offset=0, languages=languages, paths=paths,
            )

        merged = sorted(main_rows + overlay_rows, key=lambda r: r.score, reverse=True)
        return [_to_result(r, "semantic") for r in merged[offset : offset + limit]]

    async def _ensure_overlay(
        self, branch_ref: str, branch_sha: str, embed_paths: list[str]
    ) -> AsyncTable | None:
        """Build (or reuse) the overlay table for *branch_sha*.

        Returns ``None`` when the branch's changed files produced no indexable
        chunks (all binary/empty) — the caller then searches the base alone.
        """
        conn: Any = self._env.get_context(LANCE_DB)
        name = _overlay_table_name(branch_sha)

        async with self._lock:
            if name in await conn.table_names():
                self._touch(name)
                return await conn.open_table(name)

            rows = await self._build_rows(branch_ref, embed_paths)
            if not rows:
                return None

            dim = len(rows[0]["embedding"])
            table = await conn.create_table(name, data=_rows_to_arrow(rows, dim), mode="overwrite")
            self._record(name, branch_ref, branch_sha)
            logger.info(
                "Built branch overlay %s for %s (%d chunks, %d files)",
                name, branch_ref, len(rows), len(embed_paths),
            )
            return table

    async def _build_rows(self, branch_ref: str, embed_paths: list[str]) -> list[dict[str, Any]]:
        """Chunk + embed each branch file into LanceDB row dicts.

        Uses the *indexing* embedder + params (not the query embedder) so overlay
        vectors are passage-style and directly comparable to base rows. Each
        file's work runs inside the memory governor's gate, exactly like the
        on-disk indexer's ``process_file``.
        """
        embedder = self._env.get_context(EMBEDDER)
        params = self._env.get_context(INDEXING_EMBED_PARAMS)
        chunker_registry = self._env.get_context(CHUNKER_REGISTRY)
        governor = self._env.get_context(MEMORY_GOVERNOR)
        ext_lang_map = self._language_overrides()

        rows: list[dict[str, Any]] = []
        rid = 0
        for path in embed_paths:
            content = git_ops.read_blob(self._root, branch_ref, path)
            if content is None or not content.strip():
                continue
            async with governor.slot():
                language, chunks = chunk_file_content(
                    Path(path), content,
                    chunker_registry=chunker_registry, language_overrides=ext_lang_map,
                )
                for chunk in chunks:
                    vec = await embedder.embed(chunk.text, **params)
                    rid += 1
                    rows.append(
                        {
                            "id": rid,
                            "file_path": path,
                            "language": language,
                            "content": chunk.text,
                            "start_line": chunk.start.line,
                            "end_line": chunk.end.line,
                            "embedding": np.asarray(vec, dtype=np.float32),
                        }
                    )
        return rows

    # -- lexical fallback path -----------------------------------------------

    async def _search_lexical(
        self,
        *,
        query: str,
        branch_ref: str,
        main_table: AsyncTable,
        embed_paths: list[str],
        shadow_paths: list[str],
        languages: list[str] | None,
        paths: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[SearchResult]:
        # Section 1: semantic over the base, with the branch's touched files hidden.
        main_rows = await query_codebase(
            query, main_table, self._env,
            limit=limit, offset=offset, languages=languages, paths=paths,
            exclude_paths=shadow_paths or None,
        )
        results = [_to_result(r, "semantic") for r in main_rows]

        # Section 2: lexical over the branch's version of the changed files.
        ext_lang_map = self._language_overrides()
        lex_files: list[LexicalFile] = []
        for path in embed_paths:
            content = git_ops.read_blob(self._root, branch_ref, path)
            if content is None or not content.strip():
                continue
            lex_files.append(LexicalFile(path, content, _detect_language(path, ext_lang_map)))

        for hit in lexical_search(lex_files, query, limit=limit):
            results.append(
                SearchResult(
                    file_path=hit.file_path, language=hit.language, content=hit.content,
                    start_line=hit.start_line, end_line=hit.end_line, score=hit.score,
                    source="lexical",
                )
            )
        return results

    # -- filtering -----------------------------------------------------------

    def _filter(self, diff: git_ops.BranchDiff) -> tuple[list[str], list[str]]:
        """Apply the base indexer's include/exclude/gitignore rules to the diff.

        Returns ``(embed_paths, shadow_paths)`` — files to embed (added+modified)
        and base paths to hide (modified+deleted), each restricted to what the
        base index would actually contain.
        """
        matcher = self._matcher()
        embed = [p for p in diff.to_embed if matcher.is_file_included(PurePath(p))]
        shadow = [p for p in diff.shadow if matcher.is_file_included(PurePath(p))]
        return embed, shadow

    def _matcher(self) -> GitignoreAwareMatcher:
        ps = load_project_settings(self._root)
        base = PatternFilePathMatcher(
            included_patterns=ps.include_patterns, excluded_patterns=ps.exclude_patterns
        )
        return GitignoreAwareMatcher(base, load_gitignore_spec(self._root), self._root)

    def _language_overrides(self) -> dict[str, str]:
        ps = load_project_settings(self._root)
        return {f".{lo.ext}": lo.lang for lo in ps.language_overrides}

    # -- eviction / sidecar metadata -----------------------------------------

    def _sidecar_path(self) -> Path:
        return lancedb_dir_path(self._root) / _SIDECAR_NAME

    def _load_meta(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._sidecar_path().read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        overlays = data.get("overlays")
        return overlays if isinstance(overlays, dict) else {}

    def _save_meta(self, overlays: dict[str, dict[str, Any]]) -> None:
        try:
            self._sidecar_path().write_text(json.dumps({"overlays": overlays}, indent=2))
        except OSError:
            logger.warning("Could not write overlay sidecar %s", self._sidecar_path())

    def _record(self, name: str, branch_ref: str, sha: str) -> None:
        now = time.time()
        meta = self._load_meta()
        meta[name] = {"branch": branch_ref, "sha": sha, "created": now, "last_access": now}
        self._save_meta(meta)

    def _touch(self, name: str) -> None:
        meta = self._load_meta()
        if name in meta:
            meta[name]["last_access"] = time.time()
            self._save_meta(meta)

    async def evict_stale(self) -> None:
        """Drop overlays not searched within the TTL. Best-effort.

        Called lazily before each branch search and again from the daily
        maintenance workflow, so overlays are reclaimed even if branch searches
        stop arriving.
        """
        ttl_seconds = _float_env(ENV_OVERLAY_TTL_DAYS, _DEFAULT_TTL_DAYS) * 86400.0
        async with self._lock:
            meta = self._load_meta()
            if not meta:
                return
            now = time.time()
            stale = [
                name
                for name, m in meta.items()
                if now - float(m.get("last_access", 0)) > ttl_seconds
            ]
            if not stale:
                return
            conn: Any = self._env.get_context(LANCE_DB)
            for name in stale:
                try:
                    await conn.drop_table(name, ignore_missing=True)
                except Exception:
                    logger.exception("Failed to drop stale overlay %s", name)
                meta.pop(name, None)
            self._save_meta(meta)
            logger.info("Evicted %d stale branch overlay(s): %s", len(stale), ", ".join(stale))


def _detect_language(path: str, ext_lang_map: dict[str, str]) -> str:
    suffix = PurePath(path).suffix
    return ext_lang_map.get(suffix) or detect_code_language(filename=PurePath(path).name) or "text"


def _to_result(row: QueryResult, source: str) -> SearchResult:
    return SearchResult(
        file_path=row.file_path, language=row.language, content=row.content,
        start_line=row.start_line, end_line=row.end_line, score=row.score, source=source,
    )


def _rows_to_arrow(rows: list[dict[str, Any]], dim: int) -> Any:
    """Build a pyarrow Table matching the base ``code_chunks`` physical schema.

    The ``embedding`` column is a fixed-size float32 list (LanceDB's vector
    column type), so ``AsyncTable.search`` works on the overlay identically to
    the base table.
    """
    import pyarrow as pa

    return pa.table(
        {
            "id": pa.array([r["id"] for r in rows], pa.int64()),
            "file_path": pa.array([r["file_path"] for r in rows], pa.string()),
            "language": pa.array([r["language"] for r in rows], pa.string()),
            "content": pa.array([r["content"] for r in rows], pa.string()),
            "start_line": pa.array([r["start_line"] for r in rows], pa.int64()),
            "end_line": pa.array([r["end_line"] for r in rows], pa.int64()),
            "embedding": pa.array([r["embedding"].tolist() for r in rows], pa.list_(pa.float32(), dim)),
        }
    )
