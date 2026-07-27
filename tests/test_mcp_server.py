"""Tests for the MCP tool surface exposed by ``create_mcp_server``.

Drives the FastMCP server with stub backends, so these cover the tool schemas
and argument plumbing without a daemon, an index, or ripgrep.
"""

from __future__ import annotations

from typing import Any

from cocoindex_code.protocol import (
    RipgrepMatch,
    RipgrepResponse,
    SearchResponse,
    SearchResult,
)
from cocoindex_code.server import create_mcp_server


def _server(captured: dict[str, Any]) -> Any:
    async def search_backend(**kwargs: Any) -> SearchResponse:
        captured["search"] = kwargs
        return SearchResponse(
            success=True,
            results=[
                SearchResult(
                    file_path="a.py",
                    language="python",
                    content="def f(): ...",
                    start_line=1,
                    end_line=1,
                    score=0.9,
                )
            ],
            total_returned=1,
        )

    async def ripgrep_backend(**kwargs: Any) -> RipgrepResponse:
        captured["ripgrep"] = kwargs
        return RipgrepResponse(
            success=True,
            matches=[
                RipgrepMatch(
                    file_path="a.py",
                    line_number=3,
                    content="# TODO: fix",
                    start_line=3,
                    end_line=3,
                )
            ],
            total_returned=1,
            truncated=True,
        )

    return create_mcp_server(
        "/tmp/proj", search_backend=search_backend, ripgrep_backend=ripgrep_backend
    )


def _payload(result: Any) -> Any:
    """FastMCP returns (content, structured) — we assert on the structured half."""
    return result[1] if isinstance(result, tuple) else result


async def test_both_tools_are_exposed() -> None:
    tools = await _server({}).list_tools()
    assert [t.name for t in tools] == ["search", "ripgrep"]


async def test_ripgrep_tool_schema_requires_only_the_pattern() -> None:
    tools = await _server({}).list_tools()
    schema = next(t for t in tools if t.name == "ripgrep").inputSchema
    assert schema["required"] == ["pattern"]
    assert sorted(schema["properties"]) == [
        "branch",
        "case_sensitive",
        "context_lines",
        "fixed_strings",
        "globs",
        "limit",
        "pattern",
    ]


async def test_ripgrep_tool_forwards_every_argument() -> None:
    captured: dict[str, Any] = {}
    await _server(captured).call_tool(
        "ripgrep",
        {
            "pattern": "TODO",
            "limit": 7,
            "globs": ["src/**"],
            "case_sensitive": True,
            "fixed_strings": True,
            "context_lines": 2,
            "branch": "feat/x",
        },
    )
    assert captured["ripgrep"] == {
        "pattern": "TODO",
        "limit": 7,
        "globs": ["src/**"],
        "case_sensitive": True,
        "fixed_strings": True,
        "context_lines": 2,
        "branch": "feat/x",
    }


async def test_ripgrep_tool_defaults_match_the_documented_ones() -> None:
    captured: dict[str, Any] = {}
    await _server(captured).call_tool("ripgrep", {"pattern": "TODO"})
    assert captured["ripgrep"] == {
        "pattern": "TODO",
        "limit": 50,
        "globs": None,
        "case_sensitive": False,
        "fixed_strings": False,
        "context_lines": 0,
        "branch": None,
    }


async def test_ripgrep_tool_returns_matches_and_truncation() -> None:
    result = _payload(await _server({}).call_tool("ripgrep", {"pattern": "TODO"}))
    assert result["success"] is True
    assert result["truncated"] is True
    assert result["matches"][0]["file_path"] == "a.py"
    assert result["matches"][0]["line_number"] == 3


async def test_ripgrep_tool_reports_backend_failure_as_a_message() -> None:
    """A daemon-side error becomes a readable message, never an MCP transport error."""

    async def boom(**kwargs: Any) -> RipgrepResponse:
        raise RuntimeError("ripgrep (rg) is not available on the server")

    mcp = create_mcp_server("/tmp/proj", ripgrep_backend=boom)
    result = _payload(await mcp.call_tool("ripgrep", {"pattern": "TODO"}))
    assert result["success"] is False
    assert "not available on the server" in result["message"]
    assert result["matches"] == []


async def test_search_tool_still_works_alongside_ripgrep() -> None:
    captured: dict[str, Any] = {}
    result = _payload(
        await _server(captured).call_tool("search", {"query": "auth", "branch": "feat/x"})
    )
    assert captured["search"]["query"] == "auth"
    assert captured["search"]["branch"] == "feat/x"
    assert result["results"][0]["file_path"] == "a.py"
