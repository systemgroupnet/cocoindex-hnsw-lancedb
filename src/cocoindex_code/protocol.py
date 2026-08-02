"""IPC message types and serialization helpers for daemon communication."""

from __future__ import annotations

import msgspec as _msgspec

# ---------------------------------------------------------------------------
# Requests (tagged union via struct tag)
# ---------------------------------------------------------------------------


class HandshakeRequest(_msgspec.Struct, tag="handshake"):
    version: str


class IndexRequest(_msgspec.Struct, tag="index"):
    project_root: str


class SearchRequest(_msgspec.Struct, tag="search"):
    project_root: str
    query: str
    languages: list[str] | None = None
    paths: list[str] | None = None
    limit: int = 5
    offset: int = 0
    # Git branch/ref to search. None (or the base ref) searches the base index —
    # the default, zero-overhead path. Any other ref searches the base minus the
    # files that ref changed, plus a ripgrep scan of its version of them.
    branch: str | None = None
    # Incrementally refresh the index before searching. Honored only when no
    # index pass is already running; if one is (e.g. an explicit `ccc index`),
    # the refresh is skipped and the current table is read concurrently.
    refresh: bool = True


class RipgrepRequest(_msgspec.Struct, tag="ripgrep"):
    """Literal text/regex search via the ``rg`` binary — no index involved."""

    project_root: str
    pattern: str
    limit: int = 50
    # Glob filters passed to rg (``src/**``, ``!**/*_test.go``, ...).
    globs: list[str] | None = None
    case_sensitive: bool = False
    fixed_strings: bool = False
    context_lines: int = 0
    # Git branch/ref to search, same semantics as SearchRequest.branch: the base
    # working tree minus the files the branch touched, plus the branch's version
    # of the files it added or modified.
    branch: str | None = None


class ProjectStatusRequest(_msgspec.Struct, tag="project_status"):
    project_root: str


class DaemonStatusRequest(_msgspec.Struct, tag="daemon_status"):
    pass


class RemoveProjectRequest(_msgspec.Struct, tag="remove_project"):
    project_root: str


class StopRequest(_msgspec.Struct, tag="stop"):
    pass


class DoctorRequest(_msgspec.Struct, tag="doctor"):
    project_root: str | None = None


class DaemonEnvRequest(_msgspec.Struct, tag="daemon_env"):
    pass


class CompactRequest(_msgspec.Struct, tag="compact"):
    """Aggressively reclaim disk: compact files and prune all superseded versions."""

    project_root: str


class PushMetricsRequest(_msgspec.Struct, tag="push_metrics"):
    """Push a stats snapshot to the configured MySQL target on demand."""

    project_root: str


class PullRequest(_msgspec.Struct, tag="pull"):
    """Fetch and hard-reset the workspace to its git upstream on demand."""

    project_root: str


Request = (
    HandshakeRequest
    | IndexRequest
    | SearchRequest
    | RipgrepRequest
    | ProjectStatusRequest
    | DaemonStatusRequest
    | RemoveProjectRequest
    | StopRequest
    | DoctorRequest
    | DaemonEnvRequest
    | CompactRequest
    | PushMetricsRequest
    | PullRequest
)

# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class HandshakeResponse(_msgspec.Struct, tag="handshake"):
    ok: bool
    daemon_version: str
    global_settings_mtime_us: int | None = None
    # Non-fatal daemon-side warnings surfaced to the client on every handshake.
    # The client dedupes and prints them to stderr (see client._print_handshake_warnings).
    warnings: list[str] = []


class IndexResponse(_msgspec.Struct, tag="index"):
    success: bool
    message: str | None = None


class IndexingProgress(_msgspec.Struct):
    """Indexing stats snapshot, shared between progress updates and status responses."""

    num_execution_starts: int
    num_unchanged: int
    num_adds: int
    num_deletes: int
    num_reprocesses: int
    num_errors: int


class IndexProgressUpdate(_msgspec.Struct, tag="index_progress"):
    """Streamed during indexing — one per stats change, before the final IndexResponse."""

    progress: IndexingProgress


class IndexWaitingNotice(_msgspec.Struct, tag="index_waiting"):
    """Sent when another indexing is already in progress and the client must wait."""

    pass


class SearchResult(_msgspec.Struct):
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    score: float
    # Which search produced this row: "semantic" (vector search over the base
    # index) or "lexical" (ripgrep scan of a branch's changed files). Lets a
    # client render the two branch-search sections.
    source: str = "semantic"


class SearchResponse(_msgspec.Struct, tag="search"):
    success: bool
    results: list[SearchResult] = []
    total_returned: int = 0
    offset: int = 0
    message: str | None = None


class RipgrepMatch(_msgspec.Struct):
    """One matching line. ``content`` widens to the context window when asked."""

    file_path: str
    line_number: int
    content: str
    start_line: int
    end_line: int


class RipgrepResponse(_msgspec.Struct, tag="ripgrep"):
    success: bool
    matches: list[RipgrepMatch] = []
    total_returned: int = 0
    # True when the limit or the rg timeout cut the scan short — there are more
    # matches than these.
    truncated: bool = False
    message: str | None = None


class LanguageStats(_msgspec.Struct):
    """Per-language breakdown in a project status response."""

    chunks: int
    loc: int


class ProjectStatusResponse(_msgspec.Struct, tag="project_status"):
    indexing: bool
    total_chunks: int
    total_files: int
    total_loc: int
    languages: dict[str, LanguageStats]
    progress: IndexingProgress | None = None
    index_exists: bool = True


class DaemonProjectInfo(_msgspec.Struct):
    project_root: str
    indexing: bool


class DaemonStatusResponse(_msgspec.Struct, tag="daemon_status"):
    version: str
    uptime_seconds: float
    projects: list[DaemonProjectInfo]


class RemoveProjectResponse(_msgspec.Struct, tag="remove_project"):
    ok: bool


class StopResponse(_msgspec.Struct, tag="stop"):
    ok: bool


class DoctorCheckResult(_msgspec.Struct):
    name: str
    ok: bool
    details: list[str]
    errors: list[str]
    # Full formatted traceback for a failed check, shown by `ccc doctor` to aid
    # debugging of daemon-side exceptions (e.g. a failing model check).
    traceback: str | None = None


class DoctorResponse(_msgspec.Struct, tag="doctor"):
    result: DoctorCheckResult
    final: bool = False


class DbPathMappingEntry(_msgspec.Struct):
    source: str
    target: str


class DaemonEnvResponse(_msgspec.Struct, tag="daemon_env"):
    env_names: list[str]
    settings_env_names: list[str]
    db_path_mappings: list[DbPathMappingEntry] = []
    host_path_mappings: list[DbPathMappingEntry] = []


class CompactResponse(_msgspec.Struct, tag="compact"):
    ok: bool
    # On-disk size of the LanceDB store directory before/after compaction, in
    # bytes. Lets the CLI report how much space the prune reclaimed.
    bytes_before: int = 0
    bytes_after: int = 0
    message: str | None = None


class PushMetricsResponse(_msgspec.Struct, tag="push_metrics"):
    ok: bool  # False only on an actual failure (unreachable DB, missing driver)
    pushed: bool = False  # True when a snapshot row was written
    message: str | None = None


class PullResponse(_msgspec.Struct, tag="pull"):
    ok: bool  # True only when the working tree was actually updated
    status: str = "error"  # "updated" | "skipped" | "error"
    message: str | None = None


class ErrorResponse(_msgspec.Struct, tag="error"):
    message: str
    # Full formatted traceback from the daemon, when the error originates from an
    # unhandled exception. Surfaced by the CLI so daemon-side failures are debuggable.
    traceback: str | None = None


Response = (
    HandshakeResponse
    | IndexResponse
    | IndexProgressUpdate
    | IndexWaitingNotice
    | SearchResponse
    | RipgrepResponse
    | ProjectStatusResponse
    | DaemonStatusResponse
    | RemoveProjectResponse
    | StopResponse
    | DoctorResponse
    | DaemonEnvResponse
    | CompactResponse
    | PushMetricsResponse
    | PullResponse
    | ErrorResponse
)

IndexStreamResponse = IndexProgressUpdate | IndexWaitingNotice | IndexResponse | ErrorResponse
SearchStreamResponse = IndexWaitingNotice | SearchResponse | ErrorResponse
DoctorStreamResponse = DoctorResponse | ErrorResponse

# ---------------------------------------------------------------------------
# Encode / decode helpers (msgpack binary)
# ---------------------------------------------------------------------------

_request_encoder = _msgspec.msgpack.Encoder()
_request_decoder = _msgspec.msgpack.Decoder(Request)

_response_encoder = _msgspec.msgpack.Encoder()
_response_decoder = _msgspec.msgpack.Decoder(Response)


def encode_request(req: Request) -> bytes:
    return _request_encoder.encode(req)


def decode_request(data: bytes) -> Request:
    result: Request = _request_decoder.decode(data)
    return result


def encode_response(resp: Response) -> bytes:
    return _response_encoder.encode(resp)


def decode_response(data: bytes) -> Response:
    result: Response = _response_decoder.decode(data)
    return result
