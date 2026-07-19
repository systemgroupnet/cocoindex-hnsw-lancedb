---
name: run-cocoindex-code
description: Build, run, and drive cocoindex-code (the `ccc` CLI + MCP server). Use when asked to start, build, smoke-test, index, search, or otherwise exercise the running app and its daemon.
---

cocoindex-code is a Python CLI (`ccc`) + MCP server that indexes a codebase into a
LanceDB vector store and answers semantic search queries. It runs as a background
**daemon** the CLI talks to over a socket (POSIX `AF_UNIX`) / named pipe (Windows
`AF_PIPE`). The daemon can also serve a **streamable-HTTP MCP server in-process**
(enabled by `COCOINDEX_CODE_MCP_PORT`) — the `search` tool is exposed at
`http://<host>:<port>/mcp` with no external proxy. Drive all of it with the committed
smoke driver `.Codex/skills/run-cocoindex-code/driver.py`, which spins up an isolated
daemon (with the HTTP MCP server on) and runs the real app through
`index → search → mcp-http → status → doctor → compact → daemon stop`.

All paths below are relative to the repo root. Verified on Windows (win32) with
PowerShell; the commands are `uv`-based and cross-platform (the one OS-specific
detail is the `PYTHONUTF8` gotcha below, which the driver handles itself).

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (project + env manager). No system packages
  needed — it's a pure-Python project; `uv` provisions Python 3.11+ and all deps.
- **Network on first run only:** the first `index`/`search` downloads the local
  embedding model `sentence-transformers/paraphrase-MiniLM-L3-v2` (~tens of MB)
  from HuggingFace, then caches it. Subsequent runs are offline.

## Setup / Build

Installs the project plus the `dev` group (which includes `sentence-transformers`
for local embeddings — required to actually index):

```bash
uv sync
```

## Run (agent path)

One command runs the whole lifecycle against a throwaway sample repo and a private
daemon (isolated via its own `COCOINDEX_CODE_DIR`, so it never touches a real daemon,
index, or your global settings):

```bash
uv run python .Codex/skills/run-cocoindex-code/driver.py
```

Prints each step's output and ends with `SMOKE PASSED` (exit 0) or `SMOKE FAILED`
(exit 1, listing the failing steps). It always stops its daemon, even on failure.
Pass `--keep` to leave the temp dir (printed as `[kept] <path>`) for inspection.

What it exercises (each asserts exit code + expected output):

| step | command | checks |
|---|---|---|
| index | `ccc index` | `Chunks:`, `Files:` |
| search | `ccc search authentication` | hit in `auth.py` |
| search | `ccc search "fibonacci number"` | hit in `math_utils.py` |
| mcp-http | `POST /mcp` initialize + `tools/call search` | `success=True`, hit in `auth.py` |
| status | `ccc status` | `Chunks:` |
| doctor | `ccc doctor` | `Model Check (indexing)`, `Embedding dimension` |
| compact | `ccc compact` | `Compaction complete.`, `Reclaimed:` |
| daemon | `ccc daemon status` | `Daemon version:` |

The `mcp-http` step is the harness for the in-daemon MCP server: the driver picks a
free port, sets `COCOINDEX_CODE_MCP_PORT` / `COCOINDEX_CODE_MCP_HOST` /
`COCOINDEX_CODE_ROOT_PATH` in the daemon's env, then does the full transport
handshake (`initialize` → `notifications/initialized` → `tools/call`) over HTTP and
asserts the `search` tool returns a hit. See `mcp_http_search()` in `driver.py` for
the exact request shape (note the SSE `data:` framing the responses use).

The driver writes settings files directly because the real onboarding command
`ccc init` is an interactive wizard (questionary) that can't run headless.

## Direct invocation (single command, isolated)

To drive one command by hand without the full driver, isolate the daemon with a
private home dir so you don't disturb a real one:

```bash
uv run ccc --help          # list commands (no daemon needed)
```

For stateful commands (`index`/`search`/`status`/`compact`), set
`COCOINDEX_CODE_DIR` to a scratch dir first and `cd` into a project that has
`.cocoindex_code/settings.yml` (the driver shows how to create one). There's also a
daemon-free maintenance script for the disk-reclaim path:

```bash
uv run python scripts/lance_compact.py <project-root> --inspect   # report rows/versions/size
```

## MCP server (HTTP)

The daemon serves the streamable-HTTP MCP server in-process when
`COCOINDEX_CODE_MCP_PORT` is set; the `search` tool queries the project registry
directly (no `ccc mcp` subprocess, no proxy). Config env vars, read by the daemon at
launch (set them before the daemon auto-starts):

| env var | effect |
|---|---|
| `COCOINDEX_CODE_MCP_PORT` | port to serve `/mcp` on (unset → HTTP server off) |
| `COCOINDEX_CODE_MCP_HOST` | bind host (default `127.0.0.1`; the Docker image uses `0.0.0.0`) |
| `COCOINDEX_CODE_ROOT_PATH` | project the endpoint serves (HTTP carries no cwd, so pin it) |
| `COCOINDEX_CODE_MCP_DISABLE` | `1`/`true`/`yes`/`on` → kill switch, wins over the port |
| `COCOINDEX_CODE_MCP_ALLOWED_HOSTS` | allow proxied `Host` headers behind a reverse proxy (else `421`) |

To exercise it, just run the driver — its `mcp-http` step is the verified harness.
Confirm a manually started daemon is serving by tailing its log for:

```
MCP HTTP server listening on http://127.0.0.1:<port>/mcp (project: <root>)
```

(`uv run ccc daemon restart` after enabling the env var, then look in the daemon log
— path is printed by `ccc doctor` under "Log Files".)

The stdio variant — `ccc mcp` — is a separate, non-HTTP entry point for clients that
spawn an MCP server as a subprocess; it is not exercised headless here.

## Run (human path)

Verified entry points (write to your real `~/.cocoindex_code`):

```bash
uv run ccc --help
```

On Windows, prefix rich-output commands with UTF-8 mode (see Gotchas):

```bash
PYTHONUTF8=1 uv run ccc doctor   # PowerShell: $env:PYTHONUTF8=1; uv run ccc doctor
```

The normal interactive flow is `ccc init` (an interactive wizard — pick embedding
provider/model), then `ccc index`, `ccc search "<query>"`, and `ccc mcp` (stdio MCP
server). The driver exercises index/search/doctor/compact non-interactively; `init`
and the stdio MCP server are not run headless here.

## Test

```bash
uv run pytest tests/test_e2e_daemon.py -q   # real CLI↔daemon E2E (7 tests, incl. HTTP MCP)
uv run mypy src/                            # type check (clean)
```

Full suite: `uv run pytest tests/ -q` — currently 1 pre-existing failure unrelated
to runtime (`test_dockerfile_install_line_uses_full_extra`, a Docker-packaging
assertion). `uv run mypy .` also reports pre-existing errors in `scripts/` and one
test; `mypy src/` is the clean target.

## Gotchas

- **The daemon caches code at launch.** After editing any `src/cocoindex_code/`
  file, a running daemon keeps the OLD code until restarted: `uv run ccc daemon
  restart`. The smoke driver sidesteps this by starting a fresh isolated daemon.
- **`ccc doctor` (and rich box-drawing output) crashes on a Windows cp1252 console**
  with `UnicodeEncodeError: '─'`. Fix: `PYTHONUTF8=1` (PowerShell:
  `$env:PYTHONUTF8=1`). The driver sets this in its subprocess env automatically.
- **`ccc init` is interactive** — it can't run in a script/CI. Write
  `global_settings.yml` + project `settings.yml` directly instead (see the driver's
  `save_user_settings` / `save_project_settings` usage).
- **DB location is controlled by `COCOINDEX_CODE_DB_PATH_MAPPING`**, not
  `COCOINDEX_CODE_DIR`. Unset → the index lands in `<project>/.cocoindex_code/`
  (inside the repo). The Docker image sets the mapping to redirect it to a volume.
- **First index downloads the embedding model** — a cold run blocks on network;
  budget for it (the daemon's own startup has a 20s socket budget but model
  download happens during indexing, not startup).
- **Daemons are isolated by `COCOINDEX_CODE_DIR`** — the socket/pipe name is derived
  from the runtime dir, so a unique dir gives a unique daemon. Always set it for
  test runs or you'll attach to (and stop) the user's real daemon.
- **MCP HTTP responses are SSE-framed**, not plain JSON. A `POST /mcp` reply comes
  back as `event: message\ndata: {...json...}`, so strip the `data:` prefix before
  parsing (the driver's `_mcp_post` does this). It's also stateful: capture the
  `Mcp-Session-Id` header from `initialize` and echo it back on every later call.
- **The MCP env vars are read once at daemon launch.** Setting
  `COCOINDEX_CODE_MCP_PORT` (or `..._HOST` / `..._ROOT_PATH` / `..._DISABLE`) against
  an already-running daemon does nothing until `ccc daemon restart`.
- **The HTTP port binds ~1s *after* the daemon socket.** Don't assume `/mcp` is up
  the instant `ccc` commands work — poll `initialize` with a short retry (the driver
  waits up to 30s).
- **The HTTP MCP server only accepts a localhost `Host` by default.** FastMCP's
  DNS-rebinding protection returns `421 Invalid Host header` for any other `Host`
  (e.g. behind a reverse proxy with a public hostname). The driver hits `127.0.0.1`
  so it's fine; for a proxied deployment set `COCOINDEX_CODE_MCP_ALLOWED_HOSTS`
  (comma-separated; `host:*` matches any port; literal `*` disables the check) and
  optionally `COCOINDEX_CODE_MCP_ALLOWED_ORIGINS`.

## Troubleshooting

- **`Daemon has no global settings loaded. Run \`ccc init\`...`**: no
  `global_settings.yml` under the active `COCOINDEX_CODE_DIR`. Write one (driver
  does this) or run `ccc init`.
- **Search returns nothing / blocks right after `ccc index` started**: a cold first
  index has no committed rows yet; let it finish. Once rows are committed, searches
  read the current index concurrently with an in-flight index.
- **Stray daemon left running** (e.g. after `ccc doctor` in your real home): stop it
  with `uv run ccc daemon stop`.
- **`/mcp` returns `421 Invalid Host header`**: the request's `Host` isn't localhost
  (typically a reverse proxy forwarding a public hostname). Set
  `COCOINDEX_CODE_MCP_ALLOWED_HOSTS` to that hostname (or `*`) and restart the daemon.
