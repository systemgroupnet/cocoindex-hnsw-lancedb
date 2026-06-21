"""Project management: wraps a CocoIndex Environment + App."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectors import lancedb as coco_lancedb

from .chunking import CHUNKER_REGISTRY, ChunkerFn
from .indexer import indexer_main
from .lancedb_store import TABLE_NAME, ensure_vector_index
from .protocol import (
    IndexingProgress,
    IndexProgressUpdate,
    IndexResponse,
    IndexStreamResponse,
    IndexWaitingNotice,
    ProjectStatusResponse,
    SearchResult,
)
from .query import open_table, query_codebase
from .settings import (
    cocoindex_db_path as _cocoindex_db_path,
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
    QUERY_EMBED_PARAMS,
    Embedder,
)

logger = logging.getLogger(__name__)


class Project:
    _env: coco.Environment
    _app: coco.App[[], None]
    _project_root: Path
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
    ) -> None:
        """Acquire the index lock, run indexing, and release.

        If *on_started* is provided, it is set once the lock is acquired
        (i.e. indexing has truly begun).  On completion (success or failure)
        ``_initial_index_done`` is set.
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
            await self._run_index_inner(on_progress=on_progress)

    async def _run_index_inner(
        self,
        on_progress: Callable[[IndexingProgress], None] | None = None,
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
            await self._ensure_vector_index()
        finally:
            self._initial_index_done.set()
            self._indexing_stats = None

    async def _ensure_vector_index(self) -> None:
        """Build the HNSW vector index after indexing, once the table is large
        enough to benefit. Failures are logged, not raised: an index is a query
        optimization, and search still works via LanceDB's flat fallback.
        """
        try:
            conn = self._env.get_context(LANCE_DB)
            table = await conn.open_table(TABLE_NAME)
            if await ensure_vector_index(table):
                logger.info("Built HNSW vector index for %s", self._project_root)
        except (FileNotFoundError, ValueError):
            # No table yet (nothing indexed) — nothing to index.
            pass
        except Exception:
            logger.exception("Failed to build HNSW vector index")

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
    ) -> list[SearchResult]:
        """Search within this project."""
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

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> ProjectStatusResponse:
        """Get index stats by querying the LanceDB table."""
        index_exists = True
        total_chunks = 0
        total_files = 0
        languages: dict[str, int] = {}
        try:
            conn = self._env.get_context(LANCE_DB)
            table = await conn.open_table(TABLE_NAME)
            total_chunks = await table.count_rows()
            rows = await table.query().select(["file_path", "language"]).to_list()
            files: set[str] = set()
            for row in rows:
                files.add(row["file_path"])
                languages[row["language"]] = languages.get(row["language"], 0) + 1
            total_files = len(files)
        except (FileNotFoundError, ValueError):
            index_exists = False

        is_indexing = self._index_lock.locked()
        progress = self._indexing_stats if is_indexing else None
        return ProjectStatusResponse(
            indexing=is_indexing,
            total_chunks=total_chunks,
            total_files=total_files,
            languages=languages,
            progress=progress,
            index_exists=index_exists,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def indexing_stats(self) -> IndexingProgress | None:
        return self._indexing_stats

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
        indexing_params: dict[str, Any],
        query_params: dict[str, Any],
        chunker_registry: dict[str, ChunkerFn] | None = None,
    ) -> Project:
        """Create a project with explicit embedder and per-call params.

        Project-level settings and .gitignore are NOT cached here — the
        indexer loads them fresh from disk on every run so that user edits
        take effect without restarting the daemon.

        Args:
            project_root: Root directory of the codebase to index.
            embedder: Embedding model instance.
            indexing_params: Extra kwargs spread into ``embedder.embed()`` during
                indexing (e.g. ``{"prompt_name": "passage"}``).  Pass ``{}`` for
                no extras.
            query_params: Extra kwargs spread into ``embedder.embed()`` for the
                query side.
            chunker_registry: Optional mapping of file suffix (e.g. ``".toml"``)
                to a ``ChunkerFn``. When a suffix matches, the registered
                chunker is called instead of the built-in splitter.
        """
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
        context.provide(INDEXING_EMBED_PARAMS, dict(indexing_params))
        context.provide(QUERY_EMBED_PARAMS, dict(query_params))
        context.provide(CHUNKER_REGISTRY, dict(chunker_registry) if chunker_registry else {})

        env = coco.Environment(settings, context_provider=context)
        app = coco.App(
            coco.AppConfig(
                name="CocoIndexCode",
                environment=env,
            ),
            indexer_main,
        )

        result = Project.__new__(Project)
        result._env = env
        result._app = app
        result._project_root = project_root
        result._index_lock = asyncio.Lock()
        result._initial_index_done = asyncio.Event()
        result._indexing_scheduled = False
        return result
