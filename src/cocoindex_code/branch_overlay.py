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
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from cocoindex.ops.text import detect_code_language
from cocoindex.resources.file import PatternFilePathMatcher

from . import git_ops
from .chunking import CHUNKER_REGISTRY
from .indexer import GitignoreAwareMatcher, chunk_file_content
from .lexical import LexicalFile, lexical_search
from .protocol import SearchResult
from .query import open_table, query_codebase
from .schedule import git_hard_reset_sync, git_pull_enabled
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
ENV_REFRESH_SECONDS = "COCOINDEX_CODE_BRANCH_REFRESH_SECONDS"

_DEFAULT_MAX_CHANGED_FILES = 50
_DEFAULT_TTL_DAYS = 7.0
_DEFAULT_REFRESH_SECONDS = 60.0

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


def _float_env(name: str, default: float, *, allow_zero: bool = False) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if value > 0 or (allow_zero and value == 0):
        return value
    return default


def _overlay_table_name(sha: str) -> str:
    return f"overlay_{sha[:12]}"


class BranchView(NamedTuple):
    """A branch resolved against the base: what to read from it, what to hide.

    Shared by every branch-aware search path (semantic overlay, lexical
    fallback, ripgrep) so they all agree on what "the branch" contains.
    """

    sha: str
    # Files to read from the branch — added + modified, filtered to what the
    # base indexer would include.
    branch_paths: list[str]
    # Base paths to hide — modified + deleted, same filtering.
    shadow_paths: list[str]
    # Unfiltered count of changed files, which is what the divergence gate
    # measures (a branch deleting 500 ignored files is still divergent).
    total_changed: int


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
        # Separate from _lock so a slow refresh doesn't block overlay builds for
        # branches that are already resolved.
        self._refresh_lock = asyncio.Lock()
        self._last_refresh: float | None = None

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

        Raises ``RuntimeError`` when the branch can't be resolved (locally or by
        fetching it) or the diff can't be computed.
        """
        view = await self.resolve_branch(base_ref=base_ref, branch_ref=branch_ref)

        await self.evict_stale()

        main_table = await open_table(self._env)

        # Branch identical to base (or only touches ignored files): plain search.
        if not view.branch_paths and not view.shadow_paths:
            rows = await query_codebase(
                query, main_table, self._env,
                limit=limit, offset=offset, languages=languages, paths=paths,
            )
            return [_to_result(r, "semantic") for r in rows]

        if view.total_changed <= _int_env(ENV_MAX_CHANGED_FILES, _DEFAULT_MAX_CHANGED_FILES):
            return await self._search_overlay(
                query=query, branch_ref=branch_ref, branch_sha=view.sha,
                main_table=main_table, embed_paths=view.branch_paths,
                shadow_paths=view.shadow_paths,
                languages=languages, paths=paths, limit=limit, offset=offset,
            )

        logger.info(
            "Branch %s changed %d files (> %d); using lexical fallback",
            branch_ref, view.total_changed,
            _int_env(ENV_MAX_CHANGED_FILES, _DEFAULT_MAX_CHANGED_FILES),
        )
        return await self._search_lexical(
            query=query, branch_sha=view.sha, main_table=main_table,
            embed_paths=view.branch_paths, shadow_paths=view.shadow_paths,
            languages=languages, paths=paths, limit=limit, offset=offset,
        )

    async def resolve_branch(self, *, base_ref: str, branch_ref: str) -> BranchView:
        """Refresh the clone, resolve *branch_ref*, and diff it against the base.

        The shared front half of every branch-aware search. Raises
        ``RuntimeError`` when the ref can't be resolved (locally or by fetching
        it) or the diff can't be computed.
        """
        await self._refresh_clone()

        # Resolution can fetch, so it runs off the event loop. Everything after
        # this point addresses the branch by SHA: the ref may live only as
        # `refs/remotes/<remote>/<branch>`, which `git diff`/`git show` would not
        # find under its bare name.
        branch_sha = await asyncio.to_thread(
            git_ops.resolve_commit,
            self._root,
            branch_ref,
            allow_fetch=git_ops.fetch_enabled(),
            credentials=git_ops.load_credentials(),
        )
        if branch_sha is None:
            raise RuntimeError(_unresolved_message(branch_ref))

        diff = git_ops.branch_diff(self._root, base_ref, branch_sha)
        if diff is None:
            raise RuntimeError(
                f"could not diff {branch_ref!r} against base {base_ref!r} "
                f"(is {base_ref!r} a valid ref?)"
            )

        branch_paths, shadow_paths = self._filter(diff)
        return BranchView(
            sha=branch_sha,
            branch_paths=branch_paths,
            shadow_paths=shadow_paths,
            total_changed=diff.total_changed,
        )

    # -- pre-search clone refresh ---------------------------------------------

    async def _refresh_clone(self) -> None:
        """Bring the clone up to date before searching. Best-effort, never raises.

        A branch is usually searched moments after it is pushed, so a search that
        only reads whatever the last scheduled pull left behind is routinely
        stale. Refreshing first means both the branch *and* the base it is diffed
        against reflect the remote.

        Throttled to one refresh per ``COCOINDEX_CODE_BRANCH_REFRESH_SECONDS``
        (default 60, ``0`` to refresh on every search): an agent typically fires a
        burst of searches, and each one paying a network round-trip buys nothing.
        The timestamp is stamped on failure too, so an unreachable remote costs
        one timeout per interval rather than one per search.
        """
        interval = _float_env(ENV_REFRESH_SECONDS, _DEFAULT_REFRESH_SECONDS, allow_zero=True)
        async with self._refresh_lock:
            now = time.monotonic()
            if self._last_refresh is not None and now - self._last_refresh < interval:
                return
            try:
                outcome = await asyncio.to_thread(_refresh_clone_sync, self._root)
            except Exception:
                # _refresh_clone_sync is already non-raising; this is the last
                # guard that a refresh problem can never fail a search.
                logger.exception("Pre-search refresh of %s failed", self._root)
            else:
                logger.info("Pre-search refresh of %s: %s", self._root, outcome)
            self._last_refresh = time.monotonic()

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

            rows = await self._build_rows(branch_sha, embed_paths)
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

    async def _build_rows(self, branch_sha: str, embed_paths: list[str]) -> list[dict[str, Any]]:
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
            content = git_ops.read_blob(self._root, branch_sha, path)
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
        branch_sha: str,
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
            content = git_ops.read_blob(self._root, branch_sha, path)
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


def _refresh_clone_sync(root: Path) -> str:
    """Pull (or fetch) *root*; return a one-line outcome for the log. Never raises.

    Which one depends on ``COCOINDEX_CODE_GIT_PULL_ENABLED``, the same gate the
    daily maintenance workflow uses:

    * **on** — a full pull (``fetch --prune`` + ``reset --hard @{u}``), so the
      base ref and working tree advance and the diff base is current. This
      rewrites the working tree, which is why it stays behind the operator's
      opt-in; note the base *index* only catches up on the next index pass, so a
      refresh here widens the staleness window documented in
      ``docs/branch-search.md``.
    * **off** — ``fetch --prune`` only. Every remote branch and its newest
      commits become searchable with no working-tree impact; the diff base stays
      wherever the last real pull left it.
    """
    if not git_pull_enabled():
        error = git_ops.fetch_all(root, credentials=git_ops.load_credentials())
        return f"fetch failed: {error}" if error else "fetched"
    result = git_hard_reset_sync(root, git_ops.load_credentials())
    return f"{result.status}: {result.message}"


def _unresolved_message(branch_ref: str) -> str:
    """Why *branch_ref* couldn't be resolved — the fix differs by fetch setting."""
    if git_ops.fetch_enabled():
        return (
            f"ref {branch_ref!r} could not be resolved: no local branch, tag, or "
            f"remote-tracking ref matches it, and fetching it from the remote failed "
            f"(check the branch name, the remote, and the git credentials)"
        )
    return (
        f"ref {branch_ref!r} not found in the local clone and on-demand fetch is "
        f"disabled ({git_ops.ENV_FETCH_ENABLED}); fetch it first or re-enable fetching"
    )


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
            "embedding": pa.array(
                [r["embedding"].tolist() for r in rows], pa.list_(pa.float32(), dim)
            ),
        }
    )
