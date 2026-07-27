"""Project management: wraps a CocoIndex Environment + App."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectors import lancedb as coco_lancedb

from . import git_ops, ripgrep
from .branch_overlay import BranchOverlayManager, BranchView
from .chunking import CHUNKER_REGISTRY, ChunkerFn
from .indexer import indexer_main
from . import metrics
from .lancedb_store import TABLE_NAME, ensure_vector_index, prune_old_versions
from .memory import MemoryGovernor, resolve_ceiling
from .protocol import (
    CompactResponse,
    IndexingProgress,
    IndexProgressUpdate,
    IndexResponse,
    IndexStreamResponse,
    IndexWaitingNotice,
    LanguageStats,
    ProjectStatusResponse,
    PushMetricsResponse,
    SearchResult,
)
from .query import open_table, query_codebase
from .settings import (
    cocoindex_db_path as _cocoindex_db_path,
)
from .settings import (
    format_path_for_display,
)
from .settings import (
    lancedb_dir_path as _lancedb_dir_path,
)
from .settings import (
    resolve_db_dir,
)
from .shared import (
    CODEBASE_DIR,
    EMBEDDER,
    INDEXING_EMBED_PARAMS,
    LANCE_DB,
    MEMORY_GOVERNOR,
    QUERY_EMBED_PARAMS,
    QUERY_EMBEDDER,
    Embedder,
)

logger = logging.getLogger(__name__)


def _dir_size(path: Path) -> int:
    """Total size in bytes of all files under *path* (0 if it doesn't exist)."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            # Files can disappear mid-walk (e.g. a concurrent prune); skip them.
            pass
    return total


class Project:
    _env: coco.Environment
    _app: coco.App[[], None]
    _project_root: Path
    _overlays: BranchOverlayManager
    _index_lock: asyncio.Lock
    _initial_index_done: asyncio.Event
    # Set synchronously the moment an index task is created (before it acquires
    # _index_lock) and cleared when it finishes. Lets a concurrent search detect
    # an already-scheduled index and wait for it, instead of racing to spawn a
    # second, redundant index pass against the same LanceDB table.
    _indexing_scheduled: bool = False
    _indexing_stats: IndexingProgress | None = None

    def close(self) -> None:
        """Close project resources to release file handles (LMDB, LanceDB)."""
        try:
            db = self._env.get_context(LANCE_DB)
            db.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def run_index(
        self,
        on_progress: Callable[[IndexingProgress], None] | None = None,
        on_started: asyncio.Event | None = None,
        push_metrics: bool = True,
    ) -> None:
        """Acquire the index lock, run indexing, and release.

        If *on_started* is provided, it is set once the lock is acquired
        (i.e. indexing has truly begun).  On completion (success or failure)
        ``_initial_index_done`` is set.

        *push_metrics* controls the opportunistic post-index metrics push in
        :meth:`_finalize_index`. The scheduled maintenance workflow passes
        ``False`` because it pushes explicitly as its own final step (avoiding a
        duplicate snapshot); all other callers leave it ``True``.
        """
        async with self._index_lock:
            self._indexing_stats = IndexingProgress(
                num_execution_starts=0,
                num_unchanged=0,
                num_adds=0,
                num_deletes=0,
                num_reprocesses=0,
                num_errors=0,
            )
            if on_started is not None:
                on_started.set()
            await self._run_index_inner(on_progress=on_progress, push_metrics=push_metrics)

    async def _run_index_inner(
        self,
        on_progress: Callable[[IndexingProgress], None] | None = None,
        push_metrics: bool = True,
    ) -> None:
        """Run indexing (lock must already be held)."""
        try:
            handle = self._app.update()
            async for snapshot in handle.watch():
                file_stats = snapshot.stats.by_component.get("process_file")
                if file_stats is not None:
                    progress = IndexingProgress(
                        num_execution_starts=file_stats.num_execution_starts,
                        num_unchanged=file_stats.num_unchanged,
                        num_adds=file_stats.num_adds,
                        num_deletes=file_stats.num_deletes,
                        num_reprocesses=file_stats.num_reprocesses,
                        num_errors=file_stats.num_errors,
                    )
                    self._indexing_stats = progress
                    if on_progress is not None:
                        on_progress(progress)
                    await asyncio.sleep(0.1)
            await self._finalize_index(push_metrics=push_metrics)
        finally:
            self._initial_index_done.set()
            self._indexing_stats = None

    async def _finalize_index(self, push_metrics: bool = True) -> None:
        """Post-index housekeeping: build the HNSW index and reclaim disk.

        Builds the HNSW vector index once the table is large enough to benefit,
        then prunes superseded LanceDB versions/fragments accumulated during the
        run (LanceDB's own prune only reclaims versions >7 days old, so without
        this the store grows without bound under churn). Failures are logged, not
        raised: both steps are optimizations — search still works via LanceDB's
        flat fallback, and an un-pruned table is correct, just larger.
        """
        try:
            conn = self._env.get_context(LANCE_DB)
            table = await conn.open_table(TABLE_NAME)
            if await ensure_vector_index(table):
                logger.info("Built HNSW vector index for %s", self._project_root)
            stats = await prune_old_versions(table)
            logger.info("Pruned LanceDB versions for %s: %s", self._project_root, stats)
        except (FileNotFoundError, ValueError):
            # No table yet (nothing indexed) — nothing to finalize.
            pass
        except Exception:
            logger.exception("Failed to finalize index (vector index / prune)")

        if push_metrics:
            await self._push_metrics()

    async def _push_metrics(self) -> None:
        """Push the current index stats to MySQL for DevLake (best-effort).

        No-op unless a MySQL target is configured (see :mod:`.metrics`). Any
        failure is swallowed — metrics must never break an index pass.
        """
        if metrics.load_config() is None:
            return
        try:
            status = await self.get_status()
        except Exception:
            logger.exception("Failed to gather stats for metrics push")
            return
        repo = format_path_for_display(str(self._project_root))
        try:
            await metrics.push_status(repo, status)
        except Exception:
            logger.exception("Metrics push raised unexpectedly")

    async def push_metrics_now(self) -> PushMetricsResponse:
        """Push the current stats snapshot on demand (for ``ccc push-metrics``).

        Unlike :meth:`_push_metrics`, this reports the outcome instead of
        swallowing it: whether a row was written, or why not (metrics disabled,
        no index yet, driver missing, DB unreachable).
        """
        config = metrics.load_config()
        if config is None:
            return PushMetricsResponse(
                ok=True,
                pushed=False,
                message=(
                    "Metrics is not configured. Set COCOINDEX_CODE_METRICS_MYSQL_HOST "
                    "and COCOINDEX_CODE_METRICS_MYSQL_DATABASE (and ensure "
                    "COCOINDEX_CODE_METRICS_ENABLED is not disabled)."
                ),
            )
        status = await self.get_status()
        if not status.index_exists:
            return PushMetricsResponse(
                ok=True, pushed=False, message="No index yet — nothing to push. Run `ccc index` first."
            )
        repo = format_path_for_display(str(self._project_root))
        try:
            snapshot_id = await metrics.push_snapshot(repo, status, config=config)
        except metrics.MetricsDriverMissing as e:
            return PushMetricsResponse(ok=False, pushed=False, message=str(e))
        except Exception as e:
            return PushMetricsResponse(
                ok=False,
                pushed=False,
                message=f"Push to {metrics.describe_config(config)} failed: {e}",
            )
        return PushMetricsResponse(
            ok=True,
            pushed=True,
            message=(
                f"Pushed snapshot {snapshot_id} to {metrics.describe_config(config)} "
                f"(chunks={status.total_chunks}, files={status.total_files}, "
                f"loc={status.total_loc}, languages={len(status.languages)})."
            ),
        )

    def _spawn_index(
        self,
        on_progress: Callable[[IndexingProgress], None] | None = None,
        on_started: asyncio.Event | None = None,
    ) -> asyncio.Task[None]:
        """Create a background index task, marking indexing as scheduled.

        ``_indexing_scheduled`` is flipped synchronously here (before the task
        runs) so concurrent callers observe a pending index immediately and do
        not start a competing pass. The done-callback clears it.
        """
        self._indexing_scheduled = True
        task = asyncio.create_task(self.run_index(on_progress=on_progress, on_started=on_started))

        def _clear(_task: asyncio.Task[None]) -> None:
            self._indexing_scheduled = False

        task.add_done_callback(_clear)
        return task

    async def has_indexed_rows(self) -> bool:
        """True if the LanceDB table exists and currently holds ≥1 row.

        Lets a search read partial results committed by an in-flight index —
        including the *first* index pass — instead of waiting for the whole run
        to finish. LanceDB commits chunks incrementally, so rows become
        queryable as indexing progresses; only a genuinely empty/not-yet-created
        table should make a search wait.
        """
        try:
            conn = self._env.get_context(LANCE_DB)
            table = await conn.open_table(TABLE_NAME)
            count: int = await table.count_rows()
            return count > 0
        except (FileNotFoundError, ValueError):
            return False

    @property
    def is_indexing(self) -> bool:
        """True if an index pass is running or scheduled to run imminently.

        Covers both the window after a task is created but before it acquires
        the lock (``_indexing_scheduled``) and the window while it holds the
        lock. Used to decide whether a search should refresh the index or read
        the current table concurrently.
        """
        return self._index_lock.locked() or self._indexing_scheduled

    async def refresh_index(self) -> None:
        """Run an incremental index pass and wait for it to finish.

        Spawned as an independent task (rather than awaited inline) so a client
        disconnect cancelling the request handler does not abort the indexing
        mid-write; the handle is then awaited for completion.

        Callers must check :attr:`is_indexing` first and skip this when an index
        is already in flight — otherwise this queues a redundant second pass
        behind the lock instead of reading concurrently.
        """
        await self._spawn_index()

    async def ensure_indexing_started(self) -> None:
        """Kick off background indexing and wait until it has actually started.

        Returns once the indexing task holds the lock.  Safe to call multiple
        times — only the first call spawns a task; subsequent calls (including a
        search arriving while an explicit index is already scheduled) return
        immediately without starting a redundant pass.
        """
        if (
            self._initial_index_done.is_set()
            or self._index_lock.locked()
            or self._indexing_scheduled
        ):
            return
        started = asyncio.Event()
        self._spawn_index(on_started=started)
        await started.wait()

    async def stream_index(self) -> AsyncIterator[IndexStreamResponse]:
        """Run indexing, streaming progress updates and a final IndexResponse.

        If the lock is already held, yields ``IndexWaitingNotice`` first.
        The actual indexing runs in a separate task so that client disconnects
        (``GeneratorExit``) do not abort the indexing.
        """
        if self._index_lock.locked():
            yield IndexWaitingNotice()

        progress_queue: asyncio.Queue[IndexingProgress] = asyncio.Queue()
        index_task = self._spawn_index(on_progress=lambda p: progress_queue.put_nowait(p))

        try:
            while not index_task.done():
                try:
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    yield IndexProgressUpdate(progress=progress)
                except TimeoutError:
                    continue

            while not progress_queue.empty():
                yield IndexProgressUpdate(progress=progress_queue.get_nowait())

            index_task.result()
            yield IndexResponse(success=True)
        except GeneratorExit:
            return
        except Exception as e:
            yield IndexResponse(success=False, message=str(e))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @property
    def should_wait_for_indexing(self) -> bool:
        """True if indexing has been started but not yet completed."""
        return not self._initial_index_done.is_set()

    async def wait_for_indexing_done(self) -> None:
        """Wait until initial indexing is complete and no indexing is running."""
        await self._initial_index_done.wait()
        if self._index_lock.locked():
            async with self._index_lock:
                pass

    async def search(
        self,
        query: str,
        languages: list[str] | None = None,
        paths: list[str] | None = None,
        limit: int = 5,
        offset: int = 0,
        branch: str | None = None,
    ) -> list[SearchResult]:
        """Search within this project.

        When *branch* is set and differs from the base ref, the search runs
        against a branch overlay (base index minus the branch's touched files,
        plus that ref's version of them) instead of the base index. Raises
        ``RuntimeError`` if the base ref can't be determined or the branch can't
        be resolved locally.
        """
        branch = branch.strip() if branch else None
        if branch:
            base_ref = git_ops.detect_base_ref(self._project_root)
            if base_ref is None:
                raise RuntimeError(
                    "could not determine the base ref for branch search "
                    "(not a git repo or detached HEAD; set COCOINDEX_CODE_BASE_REF)"
                )
            if branch != base_ref:
                return await self._overlays.search(
                    query=query, base_ref=base_ref, branch_ref=branch,
                    languages=languages, paths=paths, limit=limit, offset=offset,
                )

        table = await open_table(self._env)
        results = await query_codebase(
            query=query,
            table=table,
            env=self._env,
            limit=limit,
            offset=offset,
            languages=languages,
            paths=paths,
        )
        return [
            SearchResult(
                file_path=r.file_path,
                language=r.language,
                content=r.content,
                start_line=r.start_line,
                end_line=r.end_line,
                score=r.score,
            )
            for r in results
        ]

    async def ripgrep(
        self,
        pattern: str,
        *,
        limit: int = 50,
        globs: list[str] | None = None,
        case_sensitive: bool = False,
        fixed_strings: bool = False,
        context_lines: int = 0,
        branch: str | None = None,
    ) -> ripgrep.RipgrepOutcome:
        """Literal/regex search of the codebase via ``rg``. Does not touch the index.

        With *branch* set (and different from the base ref), searches that ref's
        view of the tree — the base minus the files it touched, plus its own
        version of what it added or modified — reusing the same branch
        resolution the semantic branch search uses.

        Raises ``RuntimeError`` if ``rg`` isn't installed or the branch can't be
        resolved.
        """
        query = ripgrep.RipgrepQuery(
            patterns=(pattern,),
            limit=limit,
            globs=tuple(globs or ()),
            case_sensitive=case_sensitive,
            fixed_strings=fixed_strings,
            context_lines=context_lines,
        )

        branch = branch.strip() if branch else None
        view: BranchView | None = None
        if branch:
            base_ref = git_ops.detect_base_ref(self._project_root)
            if base_ref is None:
                raise RuntimeError(
                    "could not determine the base ref for branch search "
                    "(not a git repo or detached HEAD; set COCOINDEX_CODE_BASE_REF)"
                )
            if branch != base_ref:
                view = await self._overlays.resolve_branch(
                    base_ref=base_ref, branch_ref=branch
                )

        # rg and the blob reads are blocking, so the whole scan runs off the loop.
        if view is None:
            outcome = await asyncio.to_thread(
                ripgrep.search_tree, self._project_root, query
            )
        else:
            outcome = await asyncio.to_thread(
                ripgrep.search_branch,
                self._project_root,
                query,
                branch_sha=view.sha,
                branch_paths=view.branch_paths,
                shadow_paths=view.shadow_paths,
            )
        if outcome is None:
            raise RuntimeError(
                "ripgrep (rg) is not available on the server — install it "
                "(e.g. `apt-get install ripgrep`) to use this tool"
            )
        return outcome

    async def evict_stale_overlays(self) -> None:
        """Drop branch overlays past their TTL (delegates to the overlay manager)."""
        await self._overlays.evict_stale()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> ProjectStatusResponse:
        """Get index stats by querying the LanceDB table."""
        index_exists = True
        total_chunks = 0
        total_files = 0
        total_loc = 0
        languages: dict[str, LanguageStats] = {}
        try:
            conn = self._env.get_context(LANCE_DB)
            table = await conn.open_table(TABLE_NAME)
            total_chunks = await table.count_rows()
            # Stream the (file_path, language, end_line) projection in Arrow
            # batches rather than materializing every chunk row as a Python dict
            # via to_list(). On a large index that list was an O(rows) transient
            # spike on every status call (and status is hit concurrently by many
            # MCP clients). Both aggregates below are bounded by file count, not
            # chunk count.
            #
            # LoC = sum over files of the file's highest end_line. Chunks can
            # overlap or leave gaps, so summing per-chunk spans would over/under-
            # count; the per-file max is the file's line count. Per-language LoC
            # attributes each file's line count to its language.
            chunk_counts: dict[str, int] = {}
            file_lang: dict[str, str] = {}
            file_max_line: dict[str, int] = {}
            reader = (
                await table.query()
                .select(["file_path", "language", "end_line"])
                .to_batches()
            )
            async for batch in reader:
                paths = batch.column("file_path").to_pylist()
                langs = batch.column("language").to_pylist()
                ends = batch.column("end_line").to_pylist()
                for path, lang, end in zip(paths, langs, ends):
                    chunk_counts[lang] = chunk_counts.get(lang, 0) + 1
                    file_lang[path] = lang
                    if end > file_max_line.get(path, 0):
                        file_max_line[path] = end
            total_files = len(file_max_line)
            total_loc = sum(file_max_line.values())

            loc_by_lang: dict[str, int] = {}
            for path, max_line in file_max_line.items():
                lang = file_lang[path]
                loc_by_lang[lang] = loc_by_lang.get(lang, 0) + max_line
            languages = {
                lang: LanguageStats(chunks=count, loc=loc_by_lang.get(lang, 0))
                for lang, count in chunk_counts.items()
            }
        except (FileNotFoundError, ValueError):
            index_exists = False

        is_indexing = self._index_lock.locked()
        progress = self._indexing_stats if is_indexing else None
        return ProjectStatusResponse(
            indexing=is_indexing,
            total_chunks=total_chunks,
            total_files=total_files,
            total_loc=total_loc,
            languages=languages,
            progress=progress,
            index_exists=index_exists,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def compact(self) -> CompactResponse:
        """Aggressively reclaim disk: compact files and prune all old versions.

        Holds the index lock for the duration so no index pass writes
        concurrently — required because ``delete_unverified=True`` also removes
        files that could belong to an in-progress write. Concurrent reads stay
        safe (the latest version is never pruned). Reports the on-disk size of
        the store before and after.
        """
        db_dir = _lancedb_dir_path(self._project_root)
        loop = asyncio.get_event_loop()
        before = await loop.run_in_executor(None, _dir_size, db_dir)
        async with self._index_lock:
            try:
                conn = self._env.get_context(LANCE_DB)
                table = await conn.open_table(TABLE_NAME)
                stats = await prune_old_versions(
                    table,
                    cleanup_older_than=timedelta(0),
                    delete_unverified=True,
                )
                logger.info("Compacted LanceDB for %s: %s", self._project_root, stats)
            except (FileNotFoundError, ValueError):
                return CompactResponse(
                    ok=True,
                    bytes_before=before,
                    bytes_after=before,
                    message="No index to compact yet.",
                )
            except Exception as e:
                logger.exception("Compaction failed for %s", self._project_root)
                return CompactResponse(ok=False, bytes_before=before, message=str(e))
        after = await loop.run_in_executor(None, _dir_size, db_dir)
        return CompactResponse(ok=True, bytes_before=before, bytes_after=after)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def indexing_stats(self) -> IndexingProgress | None:
        return self._indexing_stats

    @property
    def root(self) -> Path:
        return self._project_root

    @property
    def env(self) -> coco.Environment:
        return self._env

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    async def create(
        project_root: Path,
        embedder: Embedder,
        query_embedder: Embedder,
        indexing_params: dict[str, Any],
        query_params: dict[str, Any],
        governor: MemoryGovernor | None = None,
        chunker_registry: dict[str, ChunkerFn] | None = None,
    ) -> Project:
        """Create a project with explicit embedder and per-call params.

        Project-level settings and .gitignore are NOT cached here — the
        indexer loads them fresh from disk on every run so that user edits
        take effect without restarting the daemon.

        Args:
            project_root: Root directory of the codebase to index.
            embedder: Embedding model instance used by the indexer.
            query_embedder: Separate embedding model instance used by the query
                path, so a search's embedding never serializes behind indexing
                (LiteLLM gates requests through a per-instance lock; ST batches
                through a per-instance runner). See :data:`QUERY_EMBEDDER`.
            indexing_params: Extra kwargs spread into ``embedder.embed()`` during
                indexing (e.g. ``{"prompt_name": "passage"}``).  Pass ``{}`` for
                no extras.
            query_params: Extra kwargs spread into ``embedder.embed()`` for the
                query side.
            governor: Process-global memory governor. Its computed
                ``max_inflight`` sizes the CocoIndex engine's fan-out to the
                memory budget, and ``process_file`` acquires its gate so the
                in-flight file count can be throttled live under RAM pressure.
                Defaults to an unconstrained governor (no limit / no throttling)
                for standalone and test use; the daemon always passes a
                calibrated one.
            chunker_registry: Optional mapping of file suffix (e.g. ``".toml"``)
                to a ``ChunkerFn``. When a suffix matches, the registered
                chunker is called instead of the built-in splitter.
        """
        if governor is None:
            governor = MemoryGovernor(None, "undetected", resolve_ceiling())
            governor.calibrate()

        settings_dir = project_root / ".cocoindex_code"
        settings_dir.mkdir(parents=True, exist_ok=True)

        db_dir = resolve_db_dir(project_root)
        db_dir.mkdir(parents=True, exist_ok=True)

        cocoindex_db = _cocoindex_db_path(project_root)
        lancedb_dir = _lancedb_dir_path(project_root)
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        settings = coco.Settings.from_env(cocoindex_db)

        lance_conn = await coco_lancedb.connect_async(str(lancedb_dir))

        context = coco.ContextProvider()
        context.provide(CODEBASE_DIR, project_root)
        context.provide(LANCE_DB, lance_conn)
        context.provide(EMBEDDER, embedder)
        context.provide(QUERY_EMBEDDER, query_embedder)
        context.provide(INDEXING_EMBED_PARAMS, dict(indexing_params))
        context.provide(QUERY_EMBED_PARAMS, dict(query_params))
        context.provide(CHUNKER_REGISTRY, dict(chunker_registry) if chunker_registry else {})
        context.provide(MEMORY_GOVERNOR, governor)

        env = coco.Environment(settings, context_provider=context)
        app = coco.App(
            coco.AppConfig(
                name="CocoIndexCode",
                environment=env,
                # Cap the engine's fan-out at the memory-budget-derived value
                # instead of the library default (1024). This is the primary
                # guard against OOM: it bounds how many files are resident at
                # once. The governor's live gate throttles further under pressure.
                max_inflight_components=governor.max_inflight,
            ),
            indexer_main,
        )

        result = Project.__new__(Project)
        result._env = env
        result._app = app
        result._project_root = project_root
        result._overlays = BranchOverlayManager(env, project_root)
        result._index_lock = asyncio.Lock()
        result._initial_index_done = asyncio.Event()
        result._indexing_scheduled = False
        return result
