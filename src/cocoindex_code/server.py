"""MCP server for codebase indexing and querying.

Supports two modes:
1. Daemon-backed: ``create_mcp_server(client, project_root)`` — lightweight MCP
   server that delegates to the daemon via per-request client functions.
2. Legacy entry point: ``main()`` — backward-compatible ``cocoindex-code`` CLI that
   auto-creates settings from env vars and delegates to the daemon.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mcp.server.transport_security import TransportSecuritySettings

    from .protocol import SearchResponse

# Resolves a search query to a SearchResponse. The stdio server (`ccc mcp`)
# uses a backend that round-trips through the daemon over the client socket;
# the in-daemon HTTP server passes one that queries the project registry
# in-process. Keyword-only args mirror the `search` tool's parameters.
SearchBackend = Callable[..., Awaitable["SearchResponse"]]

_MCP_INSTRUCTIONS = (
    "Code search and codebase understanding tools."
    "\n"
    "Use when you need to find code, understand how something works,"
    " locate implementations, or explore an unfamiliar codebase."
    "\n"
    "Provides semantic search that understands meaning --"
    " unlike grep or text matching,"
    " it finds relevant code even when exact keywords are unknown."
)


# === Pydantic Models for Tool Inputs/Outputs ===


class CodeChunkResult(BaseModel):
    """A single code chunk result."""

    file_path: str = Field(description="Relative path to the file")
    language: str = Field(description="Programming language")
    content: str = Field(description="The code content")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (1-indexed)")
    score: float = Field(description="Similarity score (0-1, higher is better)")


class SearchResultModel(BaseModel):
    """Result from search tool."""

    success: bool
    results: list[CodeChunkResult] = Field(default_factory=list)
    total_returned: int = Field(default=0)
    offset: int = Field(default=0)
    message: str | None = None


# === Daemon-backed MCP server factory ===


def _make_client_search_backend(project_root: str) -> SearchBackend:
    """Search backend that round-trips through the daemon over the client socket.

    Used by the stdio MCP server (``ccc mcp``), which runs in a separate process
    from the daemon. The blocking client call is offloaded to a thread so the
    asyncio event loop stays responsive.
    """

    async def backend(
        *,
        query: str,
        languages: list[str] | None,
        paths: list[str] | None,
        limit: int,
        offset: int,
        refresh: bool,
    ) -> SearchResponse:
        from . import client as _client

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: _client.search(
                project_root=project_root,
                query=query,
                languages=languages,
                paths=paths,
                limit=limit,
                offset=offset,
                refresh=refresh,
            ),
        )

    return backend


def create_mcp_server(
    project_root: str,
    search_backend: SearchBackend | None = None,
    transport_security: TransportSecuritySettings | None = None,
) -> FastMCP:
    """Create an MCP server exposing the codebase ``search`` tool.

    *search_backend* resolves each query to a ``SearchResponse``. It defaults to
    a daemon-client backend (socket round-trip) for the stdio server; the daemon
    passes a backend that queries its in-process project registry directly.

    *transport_security* configures the streamable-HTTP transport's DNS-rebinding
    protection (Host/Origin validation). When ``None``, FastMCP applies its
    default — which, because we don't set a ``host``, auto-enables protection
    allowing only localhost. Behind a reverse proxy (a public ``Host``), pass a
    settings object that allows that host or disables the check; otherwise every
    proxied request gets ``421 Invalid Host header``.
    """
    backend = search_backend or _make_client_search_backend(project_root)
    mcp = FastMCP(
        "cocoindex-code",
        instructions=_MCP_INSTRUCTIONS,
        transport_security=transport_security,
    )

    @mcp.tool(
        name="search",
        description=(
            "Semantic code search across the entire codebase"
            " -- finds code by meaning, not just text matching."
            " Use this instead of grep/glob when you need to find implementations,"
            " understand how features work,"
            " or locate related code without knowing exact names or keywords."
            " Accepts natural language queries"
            " (e.g., 'authentication logic', 'database connection handling')"
            " or code snippets."
            " Returns matching code chunks with file paths,"
            " line numbers, and relevance scores."
            " Start with a small limit (e.g., 5);"
            " if most results look relevant, use offset to paginate for more."
        ),
    )
    async def search(
        query: str = Field(
            description=(
                "Natural language query or code snippet to search for."
                " Examples: 'error handling middleware',"
                " 'how are users authenticated',"
                " 'database connection pool',"
                " or paste a code snippet to find similar code."
            )
        ),
        limit: int = Field(
            default=5,
            ge=1,
            le=100,
            description="Maximum number of results to return (1-100)",
        ),
        offset: int = Field(
            default=0,
            ge=0,
            description="Number of results to skip for pagination",
        ),
        refresh_index: bool = Field(
            default=False,
            description=(
                "Whether to incrementally update the index before searching."
                " Defaults to False: the index is expected to be refreshed out"
                " of band (e.g. a scheduled `ccc index`), so searches read the"
                " current table directly. Set to True to force an incremental"
                " update before this query."
            ),
        ),
        languages: list[str] | None = Field(
            default=None,
            description="Filter by programming language(s). Example: ['python', 'typescript']",
        ),
        paths: list[str] | None = Field(
            default=None,
            description=(
                "Filter by file path pattern(s) using GLOB wildcards (* and ?)."
                " Example: ['src/utils/*', '*.py']"
            ),
        ),
    ) -> SearchResultModel:
        """Query the codebase index via the configured backend."""
        try:
            # By default searches read the current table directly and do not
            # refresh the index (the index is refreshed out of band). When
            # refresh_index is True the daemon refreshes before searching only
            # while idle; if an index pass is already running it reads the
            # current table concurrently rather than blocking behind the lock.
            resp = await backend(
                query=query,
                languages=languages,
                paths=paths,
                limit=limit,
                offset=offset,
                refresh=refresh_index,
            )
            return SearchResultModel(
                success=resp.success,
                results=[
                    CodeChunkResult(
                        file_path=r.file_path,
                        language=r.language,
                        content=r.content,
                        start_line=r.start_line,
                        end_line=r.end_line,
                        score=r.score,
                    )
                    for r in resp.results
                ],
                total_returned=resp.total_returned,
                offset=resp.offset,
                message=resp.message,
            )
        except Exception as e:
            return SearchResultModel(success=False, message=f"Query failed: {e!s}")

    return mcp


# Keep the old `mcp` global for backward compatibility in __init__.py
mcp: FastMCP | None = None


# === Backward-compatible entry point ===


def _convert_embedding_model(env_model: str) -> tuple[str, str]:
    """Convert old COCOINDEX_CODE_EMBEDDING_MODEL to (provider, model)."""
    sbert_prefix = "sbert/"
    if env_model.startswith(sbert_prefix):
        return "sentence-transformers", env_model[len(sbert_prefix) :]
    return "litellm", env_model


def main() -> None:
    """Backward-compatible entry point for ``cocoindex-code`` CLI.

    Auto-detects/creates settings from env vars, then delegates to daemon.
    """
    import argparse

    from .settings import (
        EmbeddingSettings,
        LanguageOverride,
        default_project_settings,
        default_user_settings,
        find_legacy_project_root,
        find_project_root,
        project_settings_path,
        save_project_settings,
        save_user_settings,
        user_settings_path,
    )

    parser = argparse.ArgumentParser(
        prog="cocoindex-code",
        description="MCP server for codebase indexing and querying.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the MCP server (default)")
    subparsers.add_parser("index", help="Build/refresh the index and report stats")
    args = parser.parse_args()

    # --- Discover project root ---
    cwd = Path.cwd()
    project_root = find_project_root(cwd)

    if project_root is None:
        # Try env var
        env_root = os.environ.get("COCOINDEX_CODE_ROOT_PATH")
        if env_root:
            project_root = Path(env_root).resolve()
        else:
            # Use marker-based discovery
            legacy_root = find_legacy_project_root(cwd)
            project_root = legacy_root if legacy_root is not None else cwd

    # --- Auto-create project settings if needed ---
    proj_settings_file = project_settings_path(project_root)
    if not proj_settings_file.is_file():
        ps = default_project_settings()

        # Migrate COCOINDEX_CODE_EXCLUDED_PATTERNS
        raw_excluded = os.environ.get("COCOINDEX_CODE_EXCLUDED_PATTERNS", "").strip()
        if raw_excluded:
            try:
                extra_excluded = json.loads(raw_excluded)
                if isinstance(extra_excluded, list):
                    ps.exclude_patterns.extend(
                        p.strip() for p in extra_excluded if isinstance(p, str) and p.strip()
                    )
            except json.JSONDecodeError:
                pass

        # Migrate COCOINDEX_CODE_EXTRA_EXTENSIONS
        raw_extra = os.environ.get("COCOINDEX_CODE_EXTRA_EXTENSIONS", "")
        for token in raw_extra.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                ext, lang = token.split(":", 1)
                ext = ext.strip()
                lang = lang.strip()
                ps.include_patterns.append(f"**/*.{ext}")
                if lang:
                    ps.language_overrides.append(LanguageOverride(ext=ext, lang=lang))
            else:
                ps.include_patterns.append(f"**/*.{token}")

        save_project_settings(project_root, ps)

    # --- Auto-create user settings if needed ---
    user_file = user_settings_path()
    if not user_file.is_file():
        us = default_user_settings()

        # Migrate COCOINDEX_CODE_EMBEDDING_MODEL
        env_model = os.environ.get("COCOINDEX_CODE_EMBEDDING_MODEL", "")
        if env_model:
            provider, model = _convert_embedding_model(env_model)
            us.embedding = EmbeddingSettings(provider=provider, model=model)

        # Migrate COCOINDEX_CODE_DEVICE
        env_device = os.environ.get("COCOINDEX_CODE_DEVICE")
        if env_device:
            us.embedding.device = env_device

        save_user_settings(us)

    # --- Delegate to daemon ---
    from . import client as _client
    from .protocol import IndexingProgress

    if args.command == "index":
        import sys

        from rich.console import Console
        from rich.live import Live
        from rich.spinner import Spinner

        from .cli import _format_progress

        err_console = Console(stderr=True)
        last_progress_line: str | None = None

        with Live(Spinner("dots", "Indexing..."), console=err_console, transient=True) as live:

            def _on_waiting() -> None:
                live.update(
                    Spinner(
                        "dots",
                        "Another indexing is ongoing, waiting for it to finish...",
                    )
                )

            def _on_progress(progress: IndexingProgress) -> None:
                nonlocal last_progress_line
                last_progress_line = f"Indexing: {_format_progress(progress)}"
                live.update(Spinner("dots", last_progress_line))

            resp = _client.index(
                str(project_root), on_progress=_on_progress, on_waiting=_on_waiting
            )

        if last_progress_line is not None:
            print(last_progress_line, file=sys.stderr)

        if resp.success:
            st = _client.project_status(str(project_root))
            print("\nIndex stats:")
            print(f"  Chunks: {st.total_chunks}")
            print(f"  Files:  {st.total_files}")
            print(f"  LoC:    {st.total_loc}")
            if st.languages:
                print("  Languages:")
                for lang, stats in sorted(st.languages.items(), key=lambda x: -x[1].loc):
                    print(f"    {lang}: {stats.chunks} chunks, {stats.loc} LoC")
        else:
            print(f"Indexing failed: {resp.message}")
    else:
        # Default: run MCP server
        mcp_server = create_mcp_server(str(project_root))

        async def _serve() -> None:
            from .cli import _bg_index

            asyncio.create_task(_bg_index(str(project_root)))
            await mcp_server.run_stdio_async()

        asyncio.run(_serve())
