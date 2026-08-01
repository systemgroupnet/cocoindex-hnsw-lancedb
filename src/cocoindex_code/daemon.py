"""Daemon process: listener loop, project registry, request dispatch."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .project import Project

from ._daemon_paths import (
    connection_family,
    daemon_log_path,
    daemon_pid_path,
    daemon_runtime_dir,
    daemon_socket_path,
)
from . import metrics
from . import schedule
from ._version import __version__
from .chunking import ChunkerFn as _ChunkerFn
from .embedder_params import resolve_embedder_params
from .memory import (
    ENV_MAX_CONCURRENT_SCANS,
    ENV_MEMORY_LIMIT_MB,
    SAFETY_MARGIN_FRACTION,
    SCAN_QUEUE_WARN_SECONDS,
    MemoryGovernor,
    current_usage_bytes,
    detect_memory_limit_bytes,
    format_bytes,
    resolve_ceiling,
)
from .protocol import (
    CompactRequest,
    DaemonEnvRequest,
    DaemonEnvResponse,
    DaemonProjectInfo,
    DaemonStatusRequest,
    DaemonStatusResponse,
    DoctorCheckResult,
    DoctorRequest,
    DoctorResponse,
    DoctorStreamResponse,
    ErrorResponse,
    HandshakeRequest,
    HandshakeResponse,
    IndexRequest,
    IndexStreamResponse,
    IndexWaitingNotice,
    ProjectStatusRequest,
    PullRequest,
    PullResponse,
    PushMetricsRequest,
    RemoveProjectRequest,
    RemoveProjectResponse,
    Request,
    Response,
    RipgrepMatch,
    RipgrepRequest,
    RipgrepResponse,
    SearchRequest,
    SearchResponse,
    SearchStreamResponse,
    StopRequest,
    StopResponse,
    decode_request,
    encode_response,
)
from .settings import (
    ChunkerMapping,
    UserSettings,
    format_path_for_display,
    get_host_path_mappings,
    global_settings_mtime_us,
    lancedb_dir_path,
    load_project_settings,
    load_user_settings,
    user_settings_path,
)
from .shared import Embedder, check_embedding, create_embedder

logger = logging.getLogger(__name__)


def _build_backward_compat_warning(
    user_settings: UserSettings,
    settings_path: Path,
) -> str:
    """Compose the one-time handshake warning for legacy-bridge models.

    Fired when a user's settings omit ``indexing_params`` / ``query_params`` for
    a model that was previously hardcoded to use ``prompt_name="query"`` for
    queries.  See embedder_defaults.LEGACY_QUERY_PROMPT_MODELS.
    """
    return (
        f"Your embedding model ({user_settings.embedding.model}) was previously "
        f'hardcoded to use prompt_name="query" for queries. Add the following to '
        f"{settings_path} to keep this behavior and silence this warning:\n"
        f"\n"
        f"  embedding:\n"
        f"    query_params:\n"
        f"      prompt_name: query\n"
    )


def _resolve_chunker_registry(mappings: list[ChunkerMapping]) -> dict[str, _ChunkerFn]:
    """Resolve ``ChunkerMapping`` settings entries to a ``{suffix: fn}`` dict.

    Each ``mapping.module`` must be a ``"module.path:callable"`` string importable
    from the current environment.
    """
    registry: dict[str, _ChunkerFn] = {}
    for cm in mappings:
        module_path, _, attr = cm.module.partition(":")
        if not attr:
            raise ValueError(f"chunker module {cm.module!r} must use 'module.path:callable' format")
        mod = importlib.import_module(module_path)
        fn = getattr(mod, attr)
        if not callable(fn):
            raise ValueError(f"chunker {cm.module!r}: {attr!r} is not callable")
        registry[f".{cm.ext}"] = fn
    return registry


def _second_embedder_fits(
    limit_bytes: int | None,
    rss_before: int | None,
    rss_after_first: int | None,
) -> bool:
    """Whether a second embedder instance fits the memory budget.

    The daemon normally loads a *separate* query-side embedder so a search's
    embedding never serializes behind indexing (see :data:`shared.QUERY_EMBEDDER`).
    For sentence-transformers that doubles the resident model. When RAM is
    known-tight we skip the second copy and share the indexing embedder instead,
    trading query/index isolation for not being OOM-killed.

    Returns ``True`` (keep the separate embedder — the preferred default) unless
    we can measure that a second copy would breach the budget. For LiteLLM the
    measured model cost is ~0, so this always returns ``True``.
    """
    if limit_bytes is None or rss_before is None or rss_after_first is None:
        return True
    model_cost = max(0, rss_after_first - rss_before)
    projected = rss_after_first + model_cost
    return projected <= limit_bytes * (1.0 - SAFETY_MARGIN_FRACTION)


# ---------------------------------------------------------------------------
# Project Registry
# ---------------------------------------------------------------------------


class ProjectRegistry:
    """Cache of loaded projects, keyed by project root path.

    ``_embedder`` is ``None`` when the daemon is running in "no-settings mode"
    (started before ``global_settings.yml`` existed). In that state
    ``get_project`` raises an error pointing the user at ``ccc init``; the
    daemon still serves handshakes so the client can detect the mtime
    mismatch once the file is created and trigger a supervisor respawn.
    """

    _projects: dict[str, Project]
    _embedder: Embedder | None
    _query_embedder: Embedder | None
    _create_lock: asyncio.Lock
    indexing_params: dict[str, Any]
    query_params: dict[str, Any]
    governor: MemoryGovernor

    def __init__(
        self,
        embedder: Embedder | None,
        governor: MemoryGovernor,
        query_embedder: Embedder | None = None,
        indexing_params: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> None:
        self._projects = {}
        self._embedder = embedder
        # Shared across every project — memory is a process-global resource.
        self.governor = governor
        # Dedicated query-side embedder so searches don't serialize behind
        # indexing on the embedder's request lock/runner. Falls back to the
        # indexing embedder only if a caller omits it (keeps old behavior).
        self._query_embedder = query_embedder if query_embedder is not None else embedder
        # Serializes project creation. `Project.create` now awaits (opening the
        # async LanceDB connection) before opening the process-global
        # `coco.Environment`, so without this lock two concurrent first-time
        # requests for the same root could both miss the cache and race to open
        # the environment (which permits only one open instance per process).
        self._create_lock = asyncio.Lock()
        self.indexing_params = dict(indexing_params) if indexing_params else {}
        self.query_params = dict(query_params) if query_params else {}

    async def get_project(self, project_root: str) -> Project:
        """Get or create a Project for the given root. Lazy initialization."""
        if self._embedder is None:
            raise RuntimeError(
                "Daemon has no global settings loaded. Run `ccc init` to set up cocoindex-code."
            )
        cached = self._projects.get(project_root)
        if cached is not None:
            return cached
        async with self._create_lock:
            # Double-checked: another coroutine may have created it while we
            # awaited the lock.
            cached = self._projects.get(project_root)
            if cached is not None:
                return cached
            # Imported lazily: pulling in Project eagerly would import the
            # LanceDB connector (pyarrow + lance native, ~13s) at daemon
            # startup, delaying socket creation. Deferring it to the first
            # project request keeps daemon launch fast.
            from .project import Project

            root = Path(project_root)
            project_settings = load_project_settings(root)
            chunker_registry = _resolve_chunker_registry(project_settings.chunkers)
            assert self._query_embedder is not None  # set whenever _embedder is
            project = await Project.create(
                root,
                self._embedder,
                self._query_embedder,
                indexing_params=self.indexing_params,
                query_params=self.query_params,
                governor=self.governor,
                chunker_registry=chunker_registry,
            )
            self._projects[project_root] = project
            return project

    def remove_project(self, project_root: str) -> bool:
        """Remove a project from the registry. Returns True if it was loaded."""
        import gc

        project = self._projects.pop(project_root, None)
        if project is not None:
            project.close()
            del project
            gc.collect()
            return True
        return False

    def close_all(self) -> None:
        """Close all loaded projects and release resources."""
        import gc

        for project in self._projects.values():
            project.close()
        self._projects.clear()
        gc.collect()

    def list_projects(self) -> list[DaemonProjectInfo]:
        """List all loaded projects with their indexing state."""
        return [
            DaemonProjectInfo(
                project_root=root,
                indexing=project._index_lock.locked(),
            )
            for root, project in self._projects.items()
        ]

    def active_projects(self) -> list[Project]:
        """Snapshot of the currently-loaded projects (registered since startup)."""
        return list(self._projects.values())


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_connection(
    conn: Connection,
    registry: ProjectRegistry,
    start_time: float,
    on_shutdown: Callable[[], None],
    settings_mtime_us: int | None,
    settings_env_names: list[str],
    handshake_warnings: list[str],
) -> None:
    """Handle a single client connection (per-request model).

    Reads exactly two messages: a ``HandshakeRequest`` followed by one
    ``Request``.  Sends the response(s) and closes the connection.
    """
    loop = asyncio.get_event_loop()
    try:
        # 1. Handshake
        data: bytes = await loop.run_in_executor(None, conn.recv_bytes)
        req = decode_request(data)

        if not isinstance(req, HandshakeRequest):
            conn.send_bytes(
                encode_response(ErrorResponse(message="First message must be a handshake"))
            )
            return

        ok = req.version == __version__
        conn.send_bytes(
            encode_response(
                HandshakeResponse(
                    ok=ok,
                    daemon_version=__version__,
                    global_settings_mtime_us=settings_mtime_us,
                    warnings=list(handshake_warnings),
                )
            )
        )
        if not ok:
            return

        # 2. Single request
        data = await loop.run_in_executor(None, conn.recv_bytes)
        req = decode_request(data)

        result = await _dispatch(req, registry, start_time, on_shutdown, settings_env_names)
        if isinstance(result, AsyncIterator):
            try:
                async for resp in result:
                    conn.send_bytes(encode_response(resp))
            except Exception as exc:
                logger.exception("Error during streaming response")
                conn.send_bytes(
                    encode_response(
                        ErrorResponse(message=str(exc), traceback=traceback.format_exc())
                    )
                )
        else:
            conn.send_bytes(encode_response(result))
    except (EOFError, OSError, asyncio.CancelledError):
        pass
    except Exception:
        logger.exception("Error handling connection")
    finally:
        try:
            conn.close()
        except Exception:
            pass


async def _run_search(project: Project, req: SearchRequest) -> SearchResponse:
    """Execute the query against an already-prepared project."""
    results = await project.search(
        query=req.query,
        languages=req.languages,
        paths=req.paths,
        limit=req.limit,
        offset=req.offset,
        branch=req.branch,
    )
    return SearchResponse(
        success=True,
        results=results,
        total_returned=len(results),
        offset=req.offset,
    )


async def _run_ripgrep(project: Project, req: RipgrepRequest) -> RipgrepResponse:
    """Execute a ripgrep request. No index involved, so nothing is waited on."""
    outcome = await project.ripgrep(
        req.pattern,
        limit=req.limit,
        globs=req.globs,
        case_sensitive=req.case_sensitive,
        fixed_strings=req.fixed_strings,
        context_lines=req.context_lines,
        branch=req.branch,
    )
    matches = [
        RipgrepMatch(
            file_path=m.file_path,
            line_number=m.line_number,
            content=m.content,
            start_line=m.start_line,
            end_line=m.end_line,
        )
        for m in outcome.matches
    ]
    return RipgrepResponse(
        success=True,
        matches=matches,
        total_returned=len(matches),
        truncated=outcome.truncated,
    )


async def ripgrep_project(registry: ProjectRegistry, req: RipgrepRequest) -> RipgrepResponse:
    """Resolve a ripgrep request end-to-end for the in-process MCP server."""
    project = await registry.get_project(req.project_root)
    return await _run_ripgrep(project, req)


async def _search_with_wait(
    project: Project, req: SearchRequest
) -> AsyncIterator[SearchStreamResponse]:
    """Stream search response, waiting for ongoing indexing first."""
    yield IndexWaitingNotice()
    await project.wait_for_indexing_done()
    try:
        yield await _run_search(project, req)
    except Exception as e:
        yield ErrorResponse(message=str(e))


async def search_project(registry: ProjectRegistry, req: SearchRequest) -> SearchResponse:
    """Resolve a search request end-to-end for the in-process MCP server.

    Mirrors the ``SearchRequest`` dispatch path, but *blocks* while the first
    index pass completes instead of streaming an ``IndexWaitingNotice`` — the
    HTTP MCP transport has no incremental-notice channel.
    """
    project = await registry.get_project(req.project_root)
    await project.ensure_indexing_started()
    if req.refresh and not project.is_indexing:
        await project.refresh_index()
    if project.should_wait_for_indexing and not await project.has_indexed_rows():
        await project.wait_for_indexing_done()
    return await _run_search(project, req)


async def _handle_doctor(
    req: DoctorRequest,
    registry: ProjectRegistry,
) -> AsyncIterator[DoctorStreamResponse]:
    """Run doctor checks sequentially, yielding results as they complete.

    When ``project_root`` is None, only the model check runs (global scope).
    When ``project_root`` is set, only project-specific checks run (file walk + index status).
    The CLI calls this twice — once without project, once with — so that global checks
    appear before project settings in the output.
    """
    if req.project_root is None:
        # Global-scope checks — two separate embed calls because indexing and
        # query may pass different kwargs (asymmetric embedding models), and
        # either side can fail independently (e.g. a malformed input_type).
        yield DoctorResponse(
            result=await _check_model(
                registry._embedder, label="indexing", params=registry.indexing_params
            )
        )
        yield DoctorResponse(
            result=await _check_model(
                registry._embedder, label="query", params=registry.query_params
            )
        )
        yield DoctorResponse(result=_check_memory(registry.governor))
        yield DoctorResponse(result=await _check_metrics())
    else:
        # Project-scope checks
        yield DoctorResponse(result=await _check_file_walk(req.project_root))
        yield DoctorResponse(result=await _check_index_status(req.project_root))
        # Only when the scheduled git-pull step is enabled — the probe needs a
        # repo + remote, so it belongs with the project-scope checks.
        if schedule.load_config().git_pull_enabled:
            yield DoctorResponse(result=await _check_git(req.project_root))

    # Final marker
    yield DoctorResponse(
        result=DoctorCheckResult(name="done", ok=True, details=[], errors=[]),
        final=True,
    )


def _check_memory(governor: MemoryGovernor) -> DoctorCheckResult:
    """Report the detected memory limit, budget, and current usage.

    Always ``ok`` — it's informational. Flags an ``undetected`` limit as a
    detail (not an error) so an operator running an unconstrained container
    knows the OOM guard is inactive and can set ``COCOINDEX_CODE_MEMORY_LIMIT_MB``.
    """
    s = governor.snapshot()
    details = [
        f"Memory limit: {format_bytes(s.limit_bytes)} (source: {s.source})",
        f"Idle baseline: {format_bytes(s.baseline_bytes)}",
        f"Current usage: {format_bytes(s.current_bytes)}",
        f"Max in-flight files: {s.max_inflight} (current gate: {s.current_capacity})",
        (
            f"Max concurrent text scans: {s.scan_budget.max_concurrent} "
            f"(current gate: {s.current_scan_capacity}); per scan: "
            f"branch-blob batch {format_bytes(s.scan_budget.blob_batch_bytes)}, "
            f"file size cap {format_bytes(s.scan_budget.max_filesize_bytes)}"
        ),
        (
            f"Text scan queue: {s.scans_running} running, {s.scans_queued} queued "
            f"(peak queued: {s.peak_scans_queued})"
        ),
    ]
    if s.delayed_scans:
        # Queued scans are served, never rejected — so a growing count here is a
        # latency signal (and a hint to raise the pool), not an error.
        details.append(
            f"Scans delayed over {SCAN_QUEUE_WARN_SECONDS:.0f}s: {s.delayed_scans} "
            f"(longest wait: {s.max_scan_wait_seconds:.1f}s). Raise "
            f"{ENV_MAX_CONCURRENT_SCANS} if there's memory headroom."
        )
    frac = s.usage_fraction
    if frac is not None:
        details.append(f"Usage: {frac * 100:.0f}% of limit")
    if s.throttle_events:
        details.append(f"Throttle events this session: {s.throttle_events}")
    if s.limit_bytes is None:
        details.append(
            f"No memory limit detected — indexing uses the default cap and no live "
            f"throttling, and text scans use fixed defaults rather than a sized "
            f"budget. Set {ENV_MEMORY_LIMIT_MB} to enable the OOM guard."
        )
    return DoctorCheckResult(name="Memory", ok=True, details=details, errors=[])


async def _check_metrics() -> DoctorCheckResult:
    """Report whether DevLake metrics push is configured and reachable.

    Disabled/unconfigured is reported as ``ok`` (it's opt-in). When a target is
    configured, a short connection probe runs; an unreachable target is an
    error so an operator notices a broken pipeline.
    """
    config = metrics.load_config()
    if config is None:
        return DoctorCheckResult(
            name="Metrics",
            ok=True,
            details=["Disabled (no MySQL target configured)"],
            errors=[],
        )
    details = [f"Target: {metrics.describe_config(config)}"]
    error = await asyncio.to_thread(metrics.check_connection_sync, config)
    if error is not None:
        return DoctorCheckResult(name="Metrics", ok=False, details=details, errors=[error])
    details.append("Connection OK")
    return DoctorCheckResult(name="Metrics", ok=True, details=details, errors=[])


async def _check_git(project_root: str) -> DoctorCheckResult:
    """Report whether the scheduled git-pull step can reach the workspace's remote.

    Only invoked when git pull is enabled. Runs a read-only ``git ls-remote``
    probe (auth + connectivity) so a broken remote or bad credentials surfaces in
    ``ccc doctor`` instead of failing silently at the next scheduled pull.
    """
    config = schedule.load_config()
    result = await asyncio.to_thread(
        schedule.check_connection_sync, Path(project_root), config.git_credentials
    )
    if result.error is not None:
        return DoctorCheckResult(
            name="Git pull", ok=False, details=result.details, errors=[result.error]
        )
    return DoctorCheckResult(
        name="Git pull", ok=True, details=[*result.details, "Remote reachable"], errors=[]
    )


async def _check_model(
    embedder: Embedder | None,
    label: str,
    params: dict[str, Any],
) -> DoctorCheckResult:
    """Test the embedding model by embedding a short string using *params*.

    *label* appears in the check's name (e.g. ``"indexing"`` / ``"query"``) so
    users see which side of the config the result corresponds to.  Returns a
    failed result when the embedder is ``None`` (daemon running in no-settings
    mode).
    """
    name = f"Model Check ({label})"
    if embedder is None:
        return DoctorCheckResult(
            name=name,
            ok=False,
            details=[],
            errors=["Daemon has no global settings loaded. Run `ccc init` to set up."],
        )
    result = await check_embedding(embedder, params)
    params_detail = f"params: {params}" if params else "params: {} (no extra kwargs)"
    if result.error is None:
        return DoctorCheckResult(
            name=name,
            ok=True,
            details=[params_detail, f"Embedding dimension: {result.dim}"],
            errors=[],
        )
    return DoctorCheckResult(
        name=name,
        ok=False,
        details=[params_detail],
        errors=[result.error],
        traceback=result.traceback,
    )


async def _check_file_walk(project_root_str: str) -> DoctorCheckResult:
    """Walk project files and report counts + gitignore paths."""
    from pathlib import PurePath

    from cocoindex.resources.file import PatternFilePathMatcher

    from .indexer import GitignoreAwareMatcher
    from .settings import load_gitignore_spec, load_project_settings

    project_root = Path(project_root_str)
    try:
        ps = load_project_settings(project_root)
    except FileNotFoundError as e:
        return DoctorCheckResult(name="File Walk", ok=False, details=[], errors=[str(e)])

    gitignore_spec = load_gitignore_spec(project_root)
    base_matcher = PatternFilePathMatcher(
        included_patterns=ps.include_patterns,
        excluded_patterns=ps.exclude_patterns,
    )
    matcher = GitignoreAwareMatcher(base_matcher, gitignore_spec, project_root)

    counts_by_ext: dict[str, int] = {}
    gitignore_dirs: list[str] = []
    total = 0

    def _walk() -> None:
        nonlocal total
        for dirpath_str, dirnames, filenames in os.walk(project_root):
            dirpath = Path(dirpath_str)
            rel_dir = PurePath(dirpath.relative_to(project_root))
            if rel_dir != PurePath(".") and not matcher.is_dir_included(rel_dir):
                dirnames.clear()
                continue

            if (dirpath / ".gitignore").is_file():
                gitignore_dirs.append(str(rel_dir))

            for fname in filenames:
                rel_path = rel_dir / fname if rel_dir != PurePath(".") else PurePath(fname)
                if matcher.is_file_included(rel_path):
                    total += 1
                    ext = PurePath(fname).suffix or "(no ext)"
                    counts_by_ext[ext] = counts_by_ext.get(ext, 0) + 1

    await asyncio.get_event_loop().run_in_executor(None, _walk)

    details = [f"Total matched files: {total}"]
    for ext, count in sorted(counts_by_ext.items(), key=lambda x: -x[1]):
        details.append(f"  {ext}: {count}")
    if gitignore_dirs:
        details.append(f"Loaded .gitignore from: {', '.join(gitignore_dirs)}")

    return DoctorCheckResult(name="File Walk", ok=True, details=details, errors=[])


async def _check_index_status(project_root_str: str) -> DoctorCheckResult:
    """Check index status by reading the LanceDB store directly."""
    from cocoindex.connectors import lancedb as coco_lancedb

    from .lancedb_store import TABLE_NAME, VECTOR_COLUMN

    project_root = Path(project_root_str)
    db_dir = lancedb_dir_path(project_root)
    details = [f"Index: {format_path_for_display(db_dir)}"]

    if not db_dir.exists():
        details.append("Index not created yet.")
        return DoctorCheckResult(name="Index Status", ok=True, details=details, errors=[])

    try:
        conn = await coco_lancedb.connect_async(str(db_dir))
        try:
            table = await conn.open_table(TABLE_NAME)
        except (FileNotFoundError, ValueError):
            details.append("Index not created yet.")
            return DoctorCheckResult(name="Index Status", ok=True, details=details, errors=[])

        total_chunks = await table.count_rows()
        rows = await table.query().select(["file_path", "language", "end_line"]).to_list()
        chunk_counts: dict[str, int] = {}
        file_lang: dict[str, str] = {}
        file_max_line: dict[str, int] = {}
        for row in rows:
            path = row["file_path"]
            lang = row["language"]
            chunk_counts[lang] = chunk_counts.get(lang, 0) + 1
            file_lang[path] = lang
            if row["end_line"] > file_max_line.get(path, 0):
                file_max_line[path] = row["end_line"]
        total_loc = sum(file_max_line.values())
        loc_by_lang: dict[str, int] = {}
        for path, max_line in file_max_line.items():
            lang = file_lang[path]
            loc_by_lang[lang] = loc_by_lang.get(lang, 0) + max_line
        has_hnsw = any(VECTOR_COLUMN in idx.columns for idx in await table.list_indices())
        conn.close()

        details.append(f"Chunks: {total_chunks}")
        details.append(f"Files: {len(file_max_line)}")
        details.append(f"LoC: {total_loc}")
        details.append(f"Vector index: {'HNSW' if has_hnsw else 'flat (exact, small index)'}")
        if chunk_counts:
            details.append("Languages:")
            for lang, count in sorted(chunk_counts.items(), key=lambda x: -loc_by_lang[x[0]]):
                details.append(f"  {lang}: {count} chunks, {loc_by_lang[lang]} LoC")
        return DoctorCheckResult(name="Index Status", ok=True, details=details, errors=[])
    except Exception as e:
        return DoctorCheckResult(name="Index Status", ok=False, details=details, errors=[str(e)])


async def _dispatch(
    req: Request,
    registry: ProjectRegistry,
    start_time: float,
    on_shutdown: Callable[[], None],
    settings_env_names: list[str],
) -> (
    Response
    | AsyncIterator[IndexStreamResponse]
    | AsyncIterator[SearchStreamResponse]
    | AsyncIterator[DoctorStreamResponse]
):
    """Dispatch a request to the appropriate handler.

    Returns a single Response for most requests, or an AsyncIterator for
    streaming requests (IndexRequest, SearchRequest when waiting, DoctorRequest).
    """
    try:
        if isinstance(req, IndexRequest):
            project = await registry.get_project(req.project_root)
            return project.stream_index()

        if isinstance(req, SearchRequest):
            project = await registry.get_project(req.project_root)
            await project.ensure_indexing_started()

            # Refresh before searching only when the index is idle. If a pass is
            # already in flight (e.g. an explicit `ccc index` or the initial
            # background index), skip the refresh and read the current table
            # directly — LanceDB serves reads concurrently with in-flight
            # writes, so a search must not block behind the index lock.
            if req.refresh and not project.is_indexing:
                await project.refresh_index()

            # Read the current table whenever it already has rows — even mid
            # index, including during the first index pass (LanceDB commits
            # chunks incrementally, so they're queryable before the run
            # finishes). Only wait when there is genuinely nothing to search yet.
            if project.should_wait_for_indexing and not await project.has_indexed_rows():
                return _search_with_wait(project, req)

            return await _run_search(project, req)

        if isinstance(req, RipgrepRequest):
            # No ensure_indexing_started(): ripgrep reads the tree directly, so
            # it works before (and without) an index, and a grep must never kick
            # off a full index pass.
            project = await registry.get_project(req.project_root)
            return await _run_ripgrep(project, req)

        if isinstance(req, ProjectStatusRequest):
            project = await registry.get_project(req.project_root)
            await project.ensure_indexing_started()
            return await project.get_status()

        if isinstance(req, DaemonStatusRequest):
            return DaemonStatusResponse(
                version=__version__,
                uptime_seconds=time.monotonic() - start_time,
                projects=registry.list_projects(),
            )

        if isinstance(req, RemoveProjectRequest):
            registry.remove_project(req.project_root)
            return RemoveProjectResponse(ok=True)

        if isinstance(req, StopRequest):
            on_shutdown()
            return StopResponse(ok=True)

        if isinstance(req, DaemonEnvRequest):
            from .protocol import DbPathMappingEntry
            from .settings import get_db_path_mappings

            return DaemonEnvResponse(
                env_names=sorted(os.environ.keys()),
                settings_env_names=settings_env_names,
                db_path_mappings=[
                    DbPathMappingEntry(source=str(m.source), target=str(m.target))
                    for m in get_db_path_mappings()
                ],
                host_path_mappings=[
                    DbPathMappingEntry(source=str(m.source), target=str(m.target))
                    for m in get_host_path_mappings()
                ],
            )

        if isinstance(req, DoctorRequest):
            return _handle_doctor(req, registry)

        if isinstance(req, CompactRequest):
            project = await registry.get_project(req.project_root)
            return await project.compact()

        if isinstance(req, PushMetricsRequest):
            project = await registry.get_project(req.project_root)
            return await project.push_metrics_now()

        if isinstance(req, PullRequest):
            return await _handle_pull(req)

        return ErrorResponse(message=f"Unknown request type: {type(req).__name__}")
    except Exception as e:
        logger.exception("Error dispatching request")
        return ErrorResponse(message=str(e))


# ---------------------------------------------------------------------------
# Embedded MCP HTTP server
# ---------------------------------------------------------------------------


def _resolve_mcp_project_root() -> str:
    """Resolve the single project the HTTP MCP server serves.

    HTTP clients carry no working directory, so the endpoint is pinned to one
    codebase. Prefers ``COCOINDEX_CODE_ROOT_PATH``, then a project marker found
    from the daemon's cwd, then cwd itself — so the endpoint still comes up
    before ``ccc init`` (the search tool then returns a settings-missing error).
    """
    from .settings import find_project_root

    env_root = os.environ.get("COCOINDEX_CODE_ROOT_PATH", "").strip()
    if env_root:
        return str(Path(env_root).resolve())
    root = find_project_root(Path.cwd())
    return str(root) if root is not None else str(Path.cwd().resolve())


def _mcp_transport_security() -> Any:
    """Build the streamable-HTTP transport security settings from env.

    FastMCP auto-enables DNS-rebinding protection allowing only localhost
    (because we don't set a ``host``), so behind a reverse proxy every request
    is rejected with ``421 Invalid Host header``. These env vars let the
    operator allow the proxied host:

    * ``COCOINDEX_CODE_MCP_ALLOWED_HOSTS`` — comma-separated allowed ``Host``
      values (e.g. ``code.example.com`` or ``code.example.com:*``). The literal
      ``*`` disables DNS-rebinding protection entirely (use when a trusted proxy
      already controls access).
    * ``COCOINDEX_CODE_MCP_ALLOWED_ORIGINS`` — comma-separated allowed ``Origin``
      values, only needed for browser-based clients.

    Returns ``None`` when neither is set, preserving FastMCP's secure
    localhost-only default for direct (non-proxied) use.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    raw_hosts = os.environ.get("COCOINDEX_CODE_MCP_ALLOWED_HOSTS", "").strip()
    raw_origins = os.environ.get("COCOINDEX_CODE_MCP_ALLOWED_ORIGINS", "").strip()
    if not raw_hosts and not raw_origins:
        return None

    hosts = [h.strip() for h in raw_hosts.split(",") if h.strip()]
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    if "*" in hosts:
        logger.info("MCP HTTP server: DNS-rebinding protection disabled (allowed hosts = *)")
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    logger.info("MCP HTTP server: allowed hosts=%s origins=%s", hosts, origins)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


def _start_mcp_http_server(
    loop: asyncio.AbstractEventLoop,
    registry: ProjectRegistry,
    tasks: set[asyncio.Task[Any]],
) -> None:
    """Start the streamable-HTTP MCP server on *loop* when configured.

    Enabled by ``COCOINDEX_CODE_MCP_PORT`` (host overridable via
    ``COCOINDEX_CODE_MCP_HOST``, default 127.0.0.1), unless
    ``COCOINDEX_CODE_MCP_DISABLE`` is set truthy — a kill switch that wins over
    the port being set (e.g. to turn the HTTP server off for one container
    without unsetting the image's baked-in ``COCOINDEX_CODE_MCP_PORT``). The
    ``search`` tool queries the registry in-process, so there is no socket
    round-trip and no external proxy. Runs as a task on the daemon's own event
    loop so it shares the project registry; uvicorn's signal handlers are
    disabled so the daemon keeps ownership of SIGTERM/SIGINT.
    """
    if os.environ.get("COCOINDEX_CODE_MCP_DISABLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        logger.info("MCP HTTP server disabled by COCOINDEX_CODE_MCP_DISABLE")
        return
    port_str = os.environ.get("COCOINDEX_CODE_MCP_PORT", "").strip()
    if not port_str:
        return
    try:
        port = int(port_str)
    except ValueError:
        logger.error("Invalid COCOINDEX_CODE_MCP_PORT=%r; MCP server not started", port_str)
        return
    host = os.environ.get("COCOINDEX_CODE_MCP_HOST", "").strip() or "127.0.0.1"
    project_root = _resolve_mcp_project_root()

    import contextlib
    from collections.abc import Iterator

    try:
        import uvicorn

        from .server import create_mcp_server
    except ImportError:
        # The HTTP endpoint is optional; the socket server is already listening
        # and serves the CLI, stdio MCP, indexing and the scheduled workflow.
        # Letting an import error escape here takes all of that down and puts a
        # supervised container into a crash loop — degrade loudly instead.
        logger.exception(
            "MCP HTTP server unavailable: its dependencies failed to import. "
            "The daemon continues without it; the socket server and CLI are "
            "unaffected. Set COCOINDEX_CODE_MCP_DISABLE=1 to silence this."
        )
        return

    class _NoSignalServer(uvicorn.Server):
        """uvicorn server that leaves SIGTERM/SIGINT to the daemon.

        The daemon (on the main thread) installs its own signal handlers;
        uvicorn would otherwise replace them and the daemon would never shut
        down on ``docker stop``.
        """

        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

    async def _backend(
        *,
        query: str,
        languages: list[str] | None,
        paths: list[str] | None,
        limit: int,
        offset: int,
        refresh: bool,
        branch: str | None,
    ) -> SearchResponse:
        return await search_project(
            registry,
            SearchRequest(
                project_root=project_root,
                query=query,
                languages=languages,
                paths=paths,
                limit=limit,
                offset=offset,
                refresh=refresh,
                branch=branch,
            ),
        )

    async def _rg_backend(
        *,
        pattern: str,
        limit: int,
        globs: list[str] | None,
        case_sensitive: bool,
        fixed_strings: bool,
        context_lines: int,
        branch: str | None,
    ) -> RipgrepResponse:
        return await ripgrep_project(
            registry,
            RipgrepRequest(
                project_root=project_root,
                pattern=pattern,
                limit=limit,
                globs=globs,
                case_sensitive=case_sensitive,
                fixed_strings=fixed_strings,
                context_lines=context_lines,
                branch=branch,
            ),
        )

    try:
        mcp = create_mcp_server(
            project_root,
            search_backend=_backend,
            ripgrep_backend=_rg_backend,
            transport_security=_mcp_transport_security(),
        )
        config = uvicorn.Config(
            mcp.streamable_http_app(),
            host=host,
            port=port,
            log_level="info",
            access_log=False,
        )
        task = loop.create_task(_NoSignalServer(config).serve())
    except Exception:
        # Same reasoning as the import guard above: an incompatible MCP SDK
        # version fails here rather than at import, and must not be fatal.
        logger.exception("MCP HTTP server failed to start; the daemon continues without it")
        return

    tasks.add(task)
    task.add_done_callback(tasks.discard)
    logger.info(
        "MCP HTTP server listening on http://%s:%d%s (project: %s)",
        host,
        port,
        mcp.settings.streamable_http_path,
        project_root,
    )


# ---------------------------------------------------------------------------
# Scheduled maintenance workflow (git pull -> index -> push metrics)
# ---------------------------------------------------------------------------

# Upper bound on a single sleep in the scheduler. asyncio sleeps on the loop's
# *monotonic* clock, so one ~24h sleep computed from the wall clock would drift
# across NTP steps / DST / host suspend and miss the target time. Capping the
# sleep and re-checking the wall clock each wake tracks it robustly; hourly
# wakeups are negligible.
_SCHEDULE_POLL_CAP_SECONDS = 3600.0


async def _scheduled_target_projects(
    registry: ProjectRegistry, config: schedule.ScheduleConfig
) -> list[Project]:
    """Projects to run the workflow against: configured workspaces ∪ loaded projects.

    Configured workspaces are loaded on demand so the workflow bootstraps a repo
    that nothing has queried yet (the common single-repo Docker case); projects
    already in the registry are picked up automatically. De-duplicated by root.
    """
    projects: dict[str, Project] = {}
    for root in config.workspaces:
        try:
            project = await registry.get_project(str(root))
        except Exception:
            logger.exception("Scheduled workflow: could not load workspace %s", root)
            continue
        projects[str(project.root)] = project
    for project in registry.active_projects():
        projects.setdefault(str(project.root), project)
    return list(projects.values())


async def _run_scheduled_workflow_for(
    project: Project, config: schedule.ScheduleConfig
) -> None:
    """Run git pull -> index -> push metrics -> evict stale overlays for one project.

    Every step is best-effort and independently guarded: a failure is logged and
    the next step still runs, so a broken remote never blocks indexing and a
    failed index never blocks the metrics snapshot.
    """
    root = project.root

    # Step 1: refresh the working tree from git (opt-in; skips non-git dirs).
    if config.git_pull_enabled:
        try:
            result = await asyncio.to_thread(
                schedule.git_hard_reset_sync, root, config.git_credentials
            )
            logger.info("Scheduled git pull for %s: %s (%s)", root, result.status, result.message)
        except Exception:
            logger.exception("Scheduled git pull crashed for %s", root)

    # Step 2: incremental index pass over the refreshed tree. Skip when a pass is
    # already in flight rather than queueing a redundant one. push_metrics=False:
    # the explicit push below is the single snapshot for this run.
    try:
        if project.is_indexing:
            logger.info("Scheduled index skipped for %s: already indexing", root)
        else:
            await project.run_index(push_metrics=False)
            logger.info("Scheduled index complete for %s", root)
    except Exception:
        logger.exception("Scheduled index failed for %s", root)

    # Step 3: push the current stats snapshot to MySQL (no-op unless configured).
    try:
        resp = await project.push_metrics_now()
        logger.info("Scheduled metrics push for %s: %s", root, resp.message)
    except Exception:
        logger.exception("Scheduled metrics push failed for %s", root)

    # Step 4: reclaim branch overlays past their TTL (no-op when none are stale).
    try:
        await project.evict_stale_overlays()
    except Exception:
        logger.exception("Scheduled overlay eviction failed for %s", root)


async def _run_scheduled_workflow(
    registry: ProjectRegistry, config: schedule.ScheduleConfig
) -> None:
    """Run the workflow once for every target project (best-effort, logged)."""
    projects = await _scheduled_target_projects(registry, config)
    if not projects:
        logger.info("Scheduled workflow: no projects to process this run")
        return
    for project in projects:
        await _run_scheduled_workflow_for(project, config)


async def _scheduled_workflow_loop(
    registry: ProjectRegistry, config: schedule.ScheduleConfig
) -> None:
    """Run the maintenance workflow once per local day at the configured time.

    Wakes at most hourly (see :data:`_SCHEDULE_POLL_CAP_SECONDS`) and fires once
    the wall clock has passed the target time on a day it hasn't run yet, rather
    than issuing one long sleep — so it tracks the wall clock even if the clock
    is stepped or the host is suspended. Runs until the daemon shuts down.
    """
    last_run_date = datetime.now().date()  # don't fire immediately on startup
    while True:
        delay = min(
            schedule.seconds_until_next_run(datetime.now(), config.run_time),
            _SCHEDULE_POLL_CAP_SECONDS,
        )
        await asyncio.sleep(delay)
        now = datetime.now()
        if now.date() == last_run_date:
            continue
        # A cap-length wake can land after the date rolls over but before the
        # target time; wait for the target time before running.
        if (now.hour, now.minute) < (config.run_time.hour, config.run_time.minute):
            continue
        await _run_scheduled_workflow(registry, config)
        last_run_date = now.date()


async def _handle_pull(req: PullRequest) -> PullResponse:
    """On-demand git update for ``ccc pull``: same fetch + hard-reset as the
    scheduled workflow, using the configured HTTPS credentials if any.

    ``ok`` is True only when the working tree was actually updated, so the CLI
    exits non-zero for a non-git workspace ("skipped") or a git failure.
    """
    config = schedule.load_config()
    result = await asyncio.to_thread(
        schedule.git_hard_reset_sync, Path(req.project_root), config.git_credentials
    )
    return PullResponse(
        ok=result.status == "updated",
        status=result.status,
        message=result.message,
    )


# ---------------------------------------------------------------------------
# Daemon main
# ---------------------------------------------------------------------------


def run_daemon() -> None:
    """Main entry point for the daemon process (blocking).

    Sets up the listener, runs the asyncio event loop (``loop.run_forever``)
    to serve connections, and performs cleanup when shutdown is requested via
    ``StopRequest`` or a signal (SIGTERM / SIGINT).
    """
    daemon_runtime_dir().mkdir(parents=True, exist_ok=True)

    # No-settings mode: start even when global_settings.yml is missing so the
    # client can complete its handshake, detect the mtime mismatch once
    # `ccc init` writes the file, and trigger a supervisor respawn. The
    # alternative (auto-creating defaults) would skip the interactive
    # provider/model picker in `ccc init`.
    # Learn the real memory ceiling (cgroup limit inside a container, not the
    # host total) before loading any model, so the query-embedder decision and
    # the indexing fan-out can both be sized to it.
    limit_bytes, limit_source = detect_memory_limit_bytes()

    settings_mtime_us = global_settings_mtime_us()  # None when file is missing
    embedder: Embedder | None
    query_embedder: Embedder | None = None
    indexing_params: dict[str, Any] = {}
    query_params: dict[str, Any] = {}
    handshake_warnings: list[str] = []
    if user_settings_path().is_file():
        user_settings = load_user_settings()
        settings_env_keys = list(user_settings.envs.keys())
        for key, value in user_settings.envs.items():
            os.environ[key] = value
        # Resolve params BEFORE constructing the embedder so invalid configs
        # fail fast without paying the model-load cost.
        try:
            embedder_params = resolve_embedder_params(user_settings.embedding)
        except ValueError:
            logger.exception("Invalid embedder params in global_settings.yml")
            sys.exit(1)
        indexing_params = embedder_params.indexing
        query_params = embedder_params.query
        if embedder_params.used_backward_compat:
            handshake_warnings.append(
                _build_backward_compat_warning(user_settings, user_settings_path())
            )
        rss_before_model = current_usage_bytes()
        embedder = create_embedder(user_settings.embedding, indexing_params=indexing_params)
        rss_after_model = current_usage_bytes()
        # Separate instance for the query path (its own request lock / batcher) so
        # searches don't block behind indexing. Its constructor defaults to the
        # query params; query_codebase also spreads them per call. Under a tight
        # memory budget, loading a second sentence-transformers model would risk
        # OOM, so we share the indexing embedder instead (and warn).
        if _second_embedder_fits(limit_bytes, rss_before_model, rss_after_model):
            query_embedder = create_embedder(user_settings.embedding, indexing_params=query_params)
        else:
            query_embedder = embedder
            handshake_warnings.append(
                "Low memory budget: sharing one embedding model between indexing "
                "and search (searches may briefly block behind indexing). Raise the "
                f"container memory limit or set {ENV_MEMORY_LIMIT_MB} to restore the "
                "dedicated query embedder."
            )
            logger.warning(
                "Sharing query embedder with indexing embedder to fit memory budget "
                "(limit=%s)",
                format_bytes(limit_bytes),
            )
    else:
        settings_env_keys = []
        embedder = None

    # Write PID file
    pid_path = daemon_pid_path()
    pid_path.write_text(str(os.getpid()))

    # Set up logging to file
    log_path = daemon_log_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(str(log_path), mode="w"), logging.StreamHandler()],
        force=True,
    )

    logger.info("Daemon starting (PID %d, version %s)", os.getpid(), __version__)

    start_time = time.monotonic()

    # Now that the embedding model(s) are resident, calibrate the governor: it
    # records the idle footprint as its baseline and derives the static
    # max-inflight cap from (limit - baseline). The monitor starts once the
    # event loop is running (below).
    governor = MemoryGovernor(limit_bytes, limit_source, resolve_ceiling())
    governor.calibrate()

    registry = ProjectRegistry(
        embedder,
        governor=governor,
        query_embedder=query_embedder,
        indexing_params=indexing_params,
        query_params=query_params,
    )

    sock_path = daemon_socket_path()
    if sys.platform != "win32":
        try:
            Path(sock_path).unlink(missing_ok=True)
        except Exception:
            pass

    listener = Listener(sock_path, family=connection_family())
    logger.info("Listening on %s", sock_path)

    loop = asyncio.new_event_loop()
    tasks: set[asyncio.Task[Any]] = set()

    def _request_shutdown() -> None:
        """Trigger daemon shutdown — called by StopRequest or signal handler."""
        loop.stop()

    def _spawn_handler(conn: Connection) -> None:
        task = loop.create_task(
            handle_connection(
                conn,
                registry,
                start_time,
                _request_shutdown,
                settings_mtime_us,
                settings_env_keys,
                handshake_warnings,
            )
        )
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    # Handle signals for graceful shutdown
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _request_shutdown)
    except (RuntimeError, NotImplementedError):
        pass  # Not in main thread, or not supported on this platform (e.g. Windows)

    # Accept loop runs in a background thread; new connections are dispatched
    # to the event loop via call_soon_threadsafe.  The loop exits when
    # listener.close() (called during shutdown) causes accept() to raise.
    def _accept_loop() -> None:
        while True:
            try:
                conn = listener.accept()
                loop.call_soon_threadsafe(_spawn_handler, conn)
            except OSError:
                break

    accept_thread = threading.Thread(target=_accept_loop, daemon=True)
    accept_thread.start()

    # Start the memory-pressure monitor on the serving loop so indexing
    # concurrency is throttled live as RAM usage approaches the limit.
    governor.start_monitor(loop)

    # Run the daily maintenance workflow (git pull -> index -> push metrics) at
    # the configured local time. Added to the handler-task set so shutdown
    # cancels it cleanly.
    schedule_config = schedule.load_config()
    if schedule_config.enabled:
        tasks.add(loop.create_task(_scheduled_workflow_loop(registry, schedule_config)))
        logger.info("Scheduled workflow enabled: %s", schedule.describe_config(schedule_config))

    # Optionally expose an in-process streamable-HTTP MCP server on the same
    # event loop (enabled by COCOINDEX_CODE_MCP_PORT).
    _start_mcp_http_server(loop, registry, tasks)

    # --- Serve until shutdown ---
    try:
        loop.run_forever()
    finally:
        # 1. Stop accepting new connections.
        listener.close()
        governor.stop_monitor()

        # 2. Cancel handler tasks (they may be blocked in run_in_executor).
        for task in tasks:
            task.cancel()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        # 3. Release project resources.
        registry.close_all()
        loop.close()

        # 4. Remove socket and PID file.
        if sys.platform != "win32":
            try:
                Path(sock_path).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            stored = pid_path.read_text().strip()
            if stored == str(os.getpid()):
                pid_path.unlink(missing_ok=True)
        except Exception:
            pass

        logger.info("Daemon stopped")

        # 5. Hard-exit to avoid slow Python teardown (torch, threadpool, etc.).
        #    All resources are already cleaned up above.  Only do this when
        #    running as the main entry point (not when the daemon is started
        #    in-process for testing).
        if threading.current_thread() is threading.main_thread():
            os._exit(0)
