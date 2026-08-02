# SG Cocoindex-Code — long-running code-search service with Lance/HNSW support

A **fork of [cocoindex-io/cocoindex-code](https://github.com/cocoindex-io/cocoindex-code)** that turns it from a one-shot CLI into a **long-running service**: a background daemon that keeps the index and embedding model warm and serves AST-based semantic code search to your coding agents over **MCP** (stdio *and* HTTP) and the CLI.

Built on the [CocoIndex](https://github.com/cocoindex-io/cocoindex) Rust indexing engine, with an embedded [LanceDB](https://lancedb.github.io/lancedb/) + HNSW vector store.

> This is an independent fork maintained for internal use. It is **not** distributed on PyPI and is **not** the upstream package — install it from source or build the Docker image yourself (see [Install & run](#install--run)). It is not affiliated with or endorsed by the upstream maintainers.

[![License](https://img.shields.io/badge/license-Apache%202.0-5B5BD6?logoColor=white)](https://opensource.org/licenses/Apache-2.0)

## What this fork focuses on

Everything below is what this fork adds or emphasizes on top of upstream's semantic-search core:

- **Long-running daemon.** A background service keeps the embedding model resident and the index open, so searches are fast and the model loads only once. Connections are handled concurrently.
- **MCP server, stdio and HTTP.** Run `ccc mcp` for a stdio server, or expose a streamable-HTTP MCP endpoint from the daemon (`COCOINDEX_CODE_MCP_PORT`) that many clients share against one warm process.
- **Branch search.** Search *arbitrary git branches* — not just the checked-out one — by combining the base index with a ripgrep scan of the branch's diff. Nothing per-branch is ever indexed or embedded. See [Branch search](#branch-search).
- **`ripgrep` MCP tool.** Exact text/regex search alongside semantic search, branch-aware and index-free. See the [MCP tool reference](#coding-agent-integration-mcp).
- **LanceDB + HNSW backend** with automatic disk compaction and recovery tooling. See [Vector search backend](#vector-search-backend-lancedb--hnsw).
- **Container-aware memory governor** that sizes indexing concurrency to the cgroup limit, bounds concurrent text scans, and throttles both under pressure. See [Limiting memory](#limiting-memory).
- **Scheduled maintenance & git sync** — a daily `git pull → index → push metrics` workflow, built for repos kept as read-only mirrors. See [Scheduled maintenance & git sync](#scheduled-maintenance--git-sync).
- **DevLake metrics push** — optional index-stats snapshots to MySQL.

## Install & run

This fork is used two ways: **from source** (development / native install) or as a **locally built Docker image** (the primary deployment path). There is no `pipx`/`pip`/PyPI install.

### From source (uv)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://mars-gitlab.systemgroup.net/aid/cocoindex-code-lance-hnsw.git
cd cocoindex-code-lance-hnsw
uv sync                                  # install into a local .venv

uv run ccc init                          # initialize the current project (creates settings)
uv run ccc index                         # build the index
uv run ccc search "authentication logic" # search!
```

To get `ccc` on your `PATH` (installed from this local checkout, not PyPI):

```bash
uv tool install .            # from the repo root; add '.[full]' for local embeddings
```

The background daemon starts automatically on first use.

> **`[full]` vs slim.** `uv tool install '.[full]'` pulls in `sentence-transformers` so local embeddings (no API key) work out of the box; plain `.` is LiteLLM-only (cloud provider + API key). These mirror the Docker build variants of the same names.

### Docker (build locally)

```bash
# slim (default) — LiteLLM cloud embeddings, ~450 MB
docker build -t cocoindex-code:local -f docker/Dockerfile .

# full — adds sentence-transformers for local embeddings, larger image
docker build -t cocoindex-code:full --build-arg CCC_VARIANT=full -f docker/Dockerfile .
```

Then run it as a persistent container and use `docker exec` to drive it — see [Docker](#docker).

## Coding agent integration (MCP)

Run this fork as an MCP server so your agent uses semantic code search automatically — finding code by description, exploring unfamiliar codebases, and locating implementations without knowing exact names.

<details>
<summary>Claude Code</summary>

```bash
claude mcp add cocoindex-code -- ccc mcp
```
</details>

<details>
<summary>Codex</summary>

```bash
codex mcp add cocoindex-code -- ccc mcp
```
</details>

<details>
<summary>OpenCode</summary>

`opencode.json`:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cocoindex-code": {
      "type": "local",
      "command": ["ccc", "mcp"]
    }
  }
}
```
</details>

> The `cocoindex-code` command (without a subcommand) still works as a stdio MCP server for backward compatibility, auto-creating settings from environment variables on first run.

<details>
<summary>MCP tool reference</summary>

When running as an MCP server (`ccc mcp`, or the HTTP endpoint), two tools are exposed:

**`search`** — semantic code search across the codebase.

```
search(
    query: str,                          # Natural language query or code snippet
    limit: int = 5,                      # Maximum results (1-100)
    offset: int = 0,                     # Pagination offset
    refresh_index: bool = False,         # Incrementally update the index before searching
    languages: list[str] | None = None,  # Filter by language (e.g. ["python", "typescript"])
    paths: list[str] | None = None,      # Filter by path glob (e.g. ["src/utils/*"])
    branch: str | None = None,           # Git branch/ref to search (default: the base branch)
)
```

Returns matching code chunks with file path, language, code content, line numbers, a similarity `score`, and a `source` field (`"semantic"` or `"lexical"` — see [Branch search](#branch-search)).

`refresh_index` defaults to **False**: the index is expected to be refreshed out of band (e.g. the scheduled `ccc index`), so searches read the current table directly. Set it True to force an incremental update before the query.

**`ripgrep`** — exact text and regex search, powered by [ripgrep](https://github.com/BurntSushi/ripgrep).

```
ripgrep(
    pattern: str,                        # Rust-regex pattern (or a literal, with fixed_strings)
    limit: int = 50,                     # Maximum matches (1-1000)
    globs: list[str] | None = None,      # rg glob filters; '!' excludes (e.g. ["src/**", "!**/tests/**"])
    case_sensitive: bool = False,        # Default is case-insensitive
    fixed_strings: bool = False,         # Treat the pattern as a literal string
    context_lines: int = 0,              # Lines of context on each side (0-20)
    branch: str | None = None,           # Git branch/ref to search (default: the checked-out base)
)
```

Returns matches with `file_path`, `line_number`, `content` (the matching line, widened to the context window when `context_lines` is set), `start_line`/`end_line`, and a `truncated` flag telling you whether more matches exist beyond `limit`.

Use `ripgrep` when you know the literal text — a symbol name, an error string, a config key — and `search` when you're looking for code by meaning. It reads the working tree directly, so it needs no index (and never triggers one), and it finds matches in files the index skips. It honors the repo's `.gitignore` and skips `.git` and `.cocoindex_code`. Files over 16 MB are skipped and very long lines are cut (marked with `…`) — see [Limiting memory](#limiting-memory). Requires the `rg` binary on the server (included in the Docker image).

With `branch` set, it searches that ref's view of the tree using the same decomposition as [branch search](#branch-search): the base checkout minus the files the branch touched, plus the branch's own version of the files it added or modified — read from git, never checked out.
</details>

## Manual CLI usage

```bash
ccc init                                # initialize project (creates settings)
ccc index                               # build the index
ccc search "authentication logic"       # search!
```

> **Tip:** `ccc index` auto-initializes if you haven't run `ccc init` yet, so you can skip straight to indexing. (From a source checkout, prefix commands with `uv run`.)

### CLI reference

| Command | Description |
|---------|-------------|
| `ccc init` | Initialize a project — creates settings files, adds `.cocoindex_code/` to `.gitignore` |
| `ccc index` | Build or update the index (auto-inits if needed). Shows streaming progress. |
| `ccc pull` | Pull latest changes from the git upstream (fetch + hard-reset). Discards local changes to tracked files; built for repos kept as mirrors. See [Scheduled maintenance & git sync](#scheduled-maintenance--git-sync). |
| `ccc search <query>` | Semantic search across the codebase (see [Search options](#search-options)) |
| `ccc grep <pattern>` | Exact text/regex search via ripgrep — no index needed (see [Grep options](#grep-options)) |
| `ccc status` | Show index stats (chunk count, file count, language breakdown) |
| `ccc mcp` | Run as MCP server in stdio mode |
| `ccc doctor` | Run diagnostics — checks settings, daemon, model, file matching, and index health |
| `ccc compact` | Reclaim disk — compact index files and prune all superseded versions (see [Disk usage and compaction](#disk-usage-and-compaction)) |
| `ccc reset` | Delete index databases. `--all` also removes settings. `-f` skips confirmation. |
| `ccc daemon status` | Show daemon version, uptime, and loaded projects |
| `ccc daemon restart` | Restart the background daemon |
| `ccc daemon stop` | Stop the daemon |

### Search options

```bash
ccc search database schema                           # basic search
ccc search --lang python --lang markdown schema      # filter by language
ccc search --path 'src/utils/*' query handler        # filter by path
ccc search --offset 10 --limit 5 database schema     # pagination
ccc search --refresh database schema                 # update index first, then search
ccc search --branch feature/login "auth flow"        # search a git branch (see Branch search)
```

By default, `ccc search` scopes results to your current working directory (relative to the project root). Use `--path` to override.

### Grep options

`ccc grep` is exact text/regex search powered by [ripgrep](https://github.com/BurntSushi/ripgrep) — use it when you know the literal string, and `ccc search` when you're looking for code by meaning.

```bash
ccc grep 'def create_mcp_server'                     # regex search (case-insensitive)
ccc grep -F 'cfg["retries"]'                         # literal string, metacharacters and all
ccc grep -s TODO                                     # case-sensitive
ccc grep -C 3 'raise RuntimeError'                   # 3 lines of context around each match
ccc grep -g 'src/**/*.py' -g '!**/tests/**' NEEDLE   # include/exclude globs (repeatable)
ccc grep --limit 200 COCOINDEX_CODE_                 # raise the 50-match default
ccc grep --branch feature/login 'session_token'      # grep a git branch (see Branch search)
```

It reads the working tree directly, so it needs **no index** — it works before the first index pass, never triggers one, and finds matches in files the index skips. It honors the repo's `.gitignore` and skips `.git` and `.cocoindex_code`. Output is the familiar `file:line: text` (with `-C`, each context line is numbered and the matching line marked `>`). If more matches exist than `--limit`, a note goes to stderr so piped output stays clean.

Two bounds apply to every scan, so one query can't blow the daemon's memory budget: files over 16 MB aren't searched, and a line longer than 2000 characters is cut at that point and marked with `…`. See [Limiting memory](#limiting-memory).

Requires the `rg` binary on the machine running the daemon — included in the Docker image; on a source checkout install it yourself (`apt-get install ripgrep`, `brew install ripgrep`, `winget install BurntSushi.ripgrep.MSVC`).

## Branch search

The index is built from the **checked-out working tree** — normally the branch a deployment keeps reset to (the *base*). Branch search lets you query **any other git branch** without maintaining a full index per branch, and without indexing or embedding anything about that branch: it searches the base index with the branch's changed files hidden, then scans the branch's own version of those files with ripgrep.

- **Omit `branch`** (or pass the base ref) → searches the base index, exactly as before, zero overhead.
- **Any other ref** → two sections in one result set:
  - `source: "semantic"` — vector search over the base index *minus the files the branch touched*, so a stale base version of a modified file never surfaces.
  - `source: "lexical"` — a ripgrep scan of the branch's own version of the files it added or modified, ranked by how many of the query's terms each hit contains.

This is the same for every branch, however divergent. Nothing is embedded on the request path, so a branch search costs no model calls and leaves no per-branch state behind.

**`grep` is branch-aware too.** `ccc grep --branch` (and the `ripgrep` MCP tool's `branch`) resolves the ref through exactly the same path and applies the same decomposition: the base working tree minus the files the branch touched, plus rg over the branch's own version of the files it added or modified. The difference from `search` is only what the base side does — a vector query there, a text scan here — so the two tools can never disagree about what "the branch" contains.

Both paths read the branch's files in memory-budgeted batches (see [Limiting memory](#limiting-memory)), so an arbitrarily divergent branch costs more passes, not more RAM. There is no cap on how many files a branch may change.

```bash
ccc search --branch feature/login "session handling"
ccc grep --branch feature/login 'session_token'
```

**Finding the branch.** The server's clone usually has only the base branch checked out, so `branch` is resolved in three steps: the ref as given (local branch, tag, or SHA), then each remote's `refs/remotes/<remote>/<name>` (a branch that was fetched but never checked out), then — if it's still missing — an on-demand `git fetch` of that one branch. The fetch reuses the `COCOINDEX_CODE_GIT_USERNAME` / `COCOINDEX_CODE_GIT_PASSWORD` credentials of the [scheduled pull](#scheduled-maintenance--git-sync), so a branch pushed minutes ago is searchable without waiting for the next pull.

**Every branch search refreshes the clone first**, so a branch pushed moments ago is searchable without waiting for the next scheduled pull. It's best-effort — a failure is logged and the search continues on what the clone already has — and throttled to one refresh per `COCOINDEX_CODE_BRANCH_REFRESH_SECONDS` (default 60) so a burst of searches costs one round-trip. What it runs follows `COCOINDEX_CODE_GIT_PULL_ENABLED`: with the pull step **off** (the default) it's `git fetch --prune`, which leaves the checkout untouched; with it **on** it's the full pull (`fetch` + `reset --hard`), which also advances the base ref and working tree so the diff base is current.

**Nothing is ever checked out** — by `search` or by `grep`. Branch content is read straight out of the git object database (`git show <sha>:<path>`); no code path runs `checkout`, `switch`, or `reset` for a search. HEAD, the index, the working tree, and the local branch list come out byte-identical, verified in the test suite for both tools. The checkout stays on the base branch, so parallel searches and greps of *different* branches can't conflict with each other, with an in-flight index pass, or with anything else using the repo.

The one writer is the pre-search refresh above: it always updates `refs/remotes/*`, and — only when you've set `COCOINDEX_CODE_GIT_PULL_ENABLED` — hard-resets the tree. Even then it moves the **base** branch forward to its own upstream; it never switches to the branch being searched. With the pull gate off (the default), a branch search or grep cannot modify the working tree at all.

| Env var | Default | Effect |
|---|---|---|
| `COCOINDEX_CODE_BASE_REF` | auto (`HEAD`) | The ref the base index represents / the diff base. Auto-detected from the checked-out branch; override when it differs. |
| `COCOINDEX_CODE_BRANCH_FETCH_ENABLED` | on | Set falsy (`0`/`false`/`no`/`off`) to forbid the on-demand fetch, restricting search to refs already in the clone. |
| `COCOINDEX_CODE_BRANCH_REFRESH_SECONDS` | `60` | Minimum seconds between pre-search clone refreshes; `0` refreshes on every search. |

> **Current limitations.** The hardened read-only git guarantee layer is still deferred — tracked in [`docs/branch-search.md`](./docs/branch-search.md) (Future work), which documents the full design.

## Docker

A Docker image gives a reproducible, dependency-free setup — no Python, `uv`, or system deps on the host. Build it locally first (see [Install & run](#install--run)); this fork does not publish images.

The recommended approach is a **persistent container**: start it once and use `docker exec` to run CLI commands or connect MCP sessions. The daemon inside stays warm across sessions, so the embedding model loads only once.

### Build variants

Chosen at build time via the `CCC_VARIANT` build arg:

| Variant | Size | Embedding backends | When to pick |
|---|---|---|---|
| slim (default) | ~450 MB | LiteLLM (cloud: OpenAI, Voyage, Gemini, Ollama, …) | Cloud-backed embeddings, smaller image, fast builds. |
| full (`--build-arg CCC_VARIANT=full`) | ~5 GB | sentence-transformers (local) + LiteLLM | Local embeddings without an API key, or an offline-ready container. Heavier (torch + transformers). |

> **Mac users running the full variant:** local embedding inference is CPU-only inside Docker (Docker on macOS can't access Apple's Metal/MPS GPU). For fast local embeddings, install natively instead (`uv tool install '.[full]'`). The slim variant is unaffected — LiteLLM runs the model provider-side.

### Run with `docker compose`

Grab [`docker/docker-compose.yml`](./docker/docker-compose.yml) and point it at the image you built:

```bash
# macOS / Windows
COCOINDEX_CODE_IMAGE=cocoindex-code:local docker compose up -d

# Linux (aligns file ownership on bind-mounted paths with your host user)
PUID=$(id -u) PGID=$(id -g) COCOINDEX_CODE_IMAGE=cocoindex-code:local docker compose up -d
```

By default your home directory is mounted into the container (set `COCOINDEX_HOST_WORKSPACE` to narrow this to a specific code folder). Index data and the embedding model cache persist in a Docker volume across restarts. Your global settings file at `$HOME/.cocoindex_code/global_settings.yml` is visible and editable on the host; edits take effect on your next `ccc` command.

### Or: `docker run`

<details>
<summary>Docker Desktop (macOS / Windows)</summary>

```bash
docker run -d --name cocoindex-code \
  --volume "$HOME:/workspace" \
  --volume cocoindex-data:/var/cocoindex \
  -e COCOINDEX_CODE_HOST_PATH_MAPPING="/workspace=$HOME" \
  cocoindex-code:local
```
</details>

<details>
<summary>Linux (with <code>PUID</code>/<code>PGID</code>)</summary>

```bash
docker run -d --name cocoindex-code \
  -e PUID=$(id -u) -e PGID=$(id -g) \
  --volume "$HOME:/workspace" \
  --volume cocoindex-data:/var/cocoindex \
  -e COCOINDEX_CODE_HOST_PATH_MAPPING="/workspace=$HOME" \
  cocoindex-code:local
```
</details>

### Shell wrapper for `ccc` commands

Paste this into `~/.bashrc` / `~/.zshrc` so `ccc` feels native on the host and picks up the right project based on your current directory:

```bash
ccc() {
  docker exec -it -e COCOINDEX_CODE_HOST_CWD="$PWD" cocoindex-code ccc "$@"
}
```

Now `cd` into any project under your workspace and run `ccc init`, `ccc index`, `ccc search ...`, `ccc status`, etc.

### Connect your coding agent

<details>
<summary>Claude Code</summary>

Register MCP from inside the target project so `$PWD` points there:

```bash
claude mcp add cocoindex-code -- docker exec -i \
  -e COCOINDEX_CODE_HOST_CWD="$PWD" cocoindex-code ccc mcp
```

> Use `-i` (not `-it`). The `-t` flag allocates a terminal, which interferes with MCP's JSON messaging over stdin/stdout — only add it for interactive `ccc` commands like `ccc init`.
</details>

<details>
<summary>Codex</summary>

```bash
codex mcp add cocoindex-code -- docker exec -i \
  -e COCOINDEX_CODE_HOST_CWD="$PWD" cocoindex-code ccc mcp
```
</details>

### Configuration via environment variables

Pass configuration to `docker run` / compose with `-e`:

```bash
# Extra extensions (e.g. Typesafe Config, SBT build files)
-e COCOINDEX_CODE_EXTRA_EXTENSIONS="conf,sbt"

# Exclude build artefacts (Scala/SBT example)
-e COCOINDEX_CODE_EXCLUDE_PATTERNS='["**/target/**","**/.bloop/**","**/.metals/**"]'

# Set an API key
-e VOYAGE_API_KEY=your-key
```

> **Security note:** mounting `$HOME` gives the container read/write access to everything under it. If that's too broad, bind-mount a narrower directory instead (`COCOINDEX_HOST_WORKSPACE=/path/to/code`).

### Limiting memory

Indexing is memory-hungry: the engine keeps many files in flight at once (each holding its text, chunks, and embedding vectors) and — with local embeddings — the model weights are resident too. In a memory-capped container this can trip the kernel's OOM killer.

The daemon **detects the container's memory limit and sizes itself to fit**: it reads the cgroup limit at startup, caps how many files it indexes concurrently so the working set stays within budget, and throttles that concurrency further in real time as usage approaches the limit. So you just set a limit the normal Docker way — no app-specific configuration required.

**Text scans are governed too.** `ccc grep` / the `ripgrep` MCP tool spawn `rg` subprocesses, which are charged to the same cgroup as everything else. Three bounds keep them inside it:

- **At most 4 scans run at once** (a worker pool, like any connection pool). Further requests queue rather than forking more processes.
- **Files over 16 MB aren't searched**, and match lines are cut at 2000 characters. One checked-in dump or minified bundle would otherwise be buffered whole.
- **A branch scan reads the branch's changed files in batches**, sized from the memory limit, instead of loading all of them at once. Grepping a branch that rewrote 5,000 files costs one batch of resident text, not 5,000 files — it just takes more passes.

Under hard memory pressure the live monitor halves scan concurrency alongside the indexing gate; under soft pressure only indexing eases off, so an interactive grep still gets served.

**Excess requests queue — they are never rejected.** A scan past the pool size waits for a permit and is then served; nothing is dropped, errored, or killed. A waiting request costs almost nothing (it holds no `rg` process, no worker thread, no file handles — the permit is taken *before* any of that is allocated), so the queue is cheap but it does add latency. Watch it with:

```bash
docker exec cocoindex-code ccc doctor   # "Memory" check
```

```
Text scan queue: 4 running, 7 queued (peak queued: 12)
Scans delayed over 2s: 9 (longest wait: 6.4s). Raise COCOINDEX_CODE_MAX_CONCURRENT_SCANS if there's memory headroom.
```

Any scan that waits more than 2 seconds is also logged with the current queue state. A rising delayed count means the pool is your bottleneck, not a hang — raise `COCOINDEX_CODE_MAX_CONCURRENT_SCANS` if the container has room. Note that clients apply their own tool-call timeouts: the daemon will keep a queued request forever, but an MCP client may give up on it first.

**Docker Compose** — set `COCOINDEX_MEM_LIMIT` (the bundled compose file wires it into `mem_limit` / `memswap_limit`, defaulting to `2g`):

```bash
COCOINDEX_MEM_LIMIT=4g COCOINDEX_CODE_IMAGE=cocoindex-code:local docker compose up -d
```

**`docker run`** — use the standard `--memory` flag (match `--memory-swap` to it so the cap can't be masked by swap):

```bash
docker run --memory 4g --memory-swap 4g ... cocoindex-code:local
```

**Verify** what the daemon detected and how it budgeted:

```bash
docker exec cocoindex-code ccc doctor   # see the "Memory" check
```

**Fine-tuning** (env vars, all optional — the cgroup limit is used by default):

```bash
# Plan for a smaller budget than the hard cap (leave headroom for sidecars),
# or set an explicit limit when no cgroup limit is present:
-e COCOINDEX_CODE_MEMORY_LIMIT_MB=1500

# Pin the max concurrent in-flight files, bypassing the auto heuristic:
-e COCOINDEX_CODE_MAX_INFLIGHT_FILES=32

# Pin how many `rg` scans may run at once (default 4). Raise it if many agents
# grep concurrently and you have the headroom; lower it in a tight container:
-e COCOINDEX_CODE_MAX_CONCURRENT_SCANS=2
```

> **What's sized vs. what's fixed.** The indexing fan-out and the branch-scan batch size are derived from the detected limit. Scan concurrency and the 16 MB file cut are *fixed policy* — deliberately, because no one has measured what one `rg` process costs on your workload, and a made-up per-scan byte estimate would look more precise than it is. Both are overridable, and `ccc doctor` prints what's in effect.

The images also set `MALLOC_ARENA_MAX=2` to keep native (torch / Arrow / Lance) allocations from fragmenting resident memory upward over time. If you need to squeeze RSS further, preloading a `jemalloc`/`tcmalloc` allocator via `LD_PRELOAD` is the next lever.

> If no memory limit is detected (e.g. an unconstrained container or a non-Linux host), indexing falls back to the engine default with no runtime throttling, and text scans use fixed defaults rather than a sized budget — they stay bounded, just not fitted to anything. `ccc doctor` flags this so you can set `COCOINDEX_CODE_MEMORY_LIMIT_MB` to re-enable the guard.

### Scheduled maintenance & git sync

The daemon runs one **daily maintenance workflow** that, for each target repo, does these steps in order — each **best-effort**, so a failure in one is logged and the next still runs:

1. **git pull** — fetch and hard-reset the working tree to its upstream (opt-in; skipped for non-git directories),
2. **index** — an incremental index pass over the refreshed tree,
3. **push metrics** — write an index-stats snapshot to MySQL (only when the `COCOINDEX_CODE_METRICS_*` target is configured),

In the Docker image it targets the mounted repo (`/workspace`) at 03:00 local by default. Run steps on demand with `ccc pull` (step 1) and `ccc index` (step 2).

| Env var | Default | Effect |
|---|---|---|
| `COCOINDEX_CODE_SCHEDULE_ENABLED` | on | Set falsy (`0`/`false`/`no`/`off`) to disable the whole workflow |
| `COCOINDEX_CODE_SCHEDULE_TIME` | `03:00` | Local `HH:MM` (24-hour) to run |
| `COCOINDEX_CODE_SCHEDULE_WORKSPACES` | `/workspace` (Docker) | Comma-separated repo roots to process (union'd with loaded projects) |
| `COCOINDEX_CODE_GIT_PULL_ENABLED` | **off** | Set truthy to enable the git-pull step |
| `COCOINDEX_CODE_GIT_USERNAME` | — | Optional HTTPS username (for a token, any non-empty value, e.g. `x-access-token`) |
| `COCOINDEX_CODE_GIT_PASSWORD` | — | Optional HTTPS password / personal-access token |

The two `GIT_` credentials are also used by [branch search](#branch-search)'s pre-search refresh and on-demand fetch. `COCOINDEX_CODE_GIT_PULL_ENABLED` gates the destructive part there too: with it off, branch search only ever fetches (no working-tree write); with it on, its pre-search refresh is a full pull.

> **The git-pull step is destructive by design.** It runs `git fetch` then `git reset --hard` to the upstream branch, discarding any local changes to **tracked** files — it's built for repos kept as read-only mirrors. Untracked and ignored files (including `.cocoindex_code/`) are never touched. It's **off by default**; enable it only where that's what you want.

**Authentication.** For SSH remotes, host auth is used (SSH key / agent). For HTTPS, set `COCOINDEX_CODE_GIT_USERNAME` / `COCOINDEX_CODE_GIT_PASSWORD` — they're injected via an inline git credential helper scoped to the fetch, so the token is **never written to disk, never placed in the remote URL, and never in the process argv**. (A container has no access to host credential helpers like the macOS Keychain, so supplying a token this way, or an SSH key/agent, is required.)

**Diagnostics.** When git pull is enabled, `ccc doctor` runs a read-only `git ls-remote` **connectivity probe** ("Git pull" check) so a broken remote or bad credentials surfaces before the next scheduled pull. Any credentials embedded in a remote URL are masked in the output.

## Features

- **Semantic code search**: find relevant code using natural language queries when grep doesn't work well, and save tokens immediately.
- **Long-running service**: a warm daemon serves the CLI and MCP (stdio + HTTP) with the model loaded once.
- **Branch search**: query arbitrary git branches with no per-branch index (see [Branch search](#branch-search)).
- **Semantic *and* exact search**: `search` finds code by meaning; `ccc grep` / the `ripgrep` MCP tool finds literal text and regexes, index-free and branch-aware (see [Grep options](#grep-options)).
- **Ultra performant**: ⚡ built on the [CocoIndex](https://github.com/cocoindex-io/cocoindex) Rust indexing engine. Only re-indexes changed files for fast updates.
- **Multi-language support**: Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, C#, SQL, Shell, and more (see [Supported languages](#supported-languages)).
- **Built for concurrency**: an embedded [LanceDB](https://lancedb.github.io/lancedb/) + HNSW vector store serves searches concurrently — even while an index pass is writing — so multiple agents (or MCP sessions) query in parallel without blocking. See [Vector search backend](#vector-search-backend-lancedb--hnsw).
- **Flexible embeddings**: local SentenceTransformers via the `[full]` extra (free, no API key) or 100+ cloud providers via LiteLLM.

## Configuration

For a detailed guide on choosing and configuring embedding models, see [EMBEDDINGS.md](EMBEDDINGS.md).

Configuration lives in two YAML files, both created automatically by `ccc init`.

### User settings (`~/.cocoindex_code/global_settings.yml`)

Shared across all projects. Controls the embedding model and environment variables for the daemon.

```yaml
embedding:
  provider: sentence-transformers                    # or "litellm"
  model: Snowflake/snowflake-arctic-embed-xs
  device: mps                                        # optional: cpu, cuda, mps (auto-detected if omitted)
  min_interval_ms: 300                               # optional: pace LiteLLM embedding requests to reduce 429s; defaults to 5 for LiteLLM

  # Optional extra kwargs passed to the embedder, separately for indexing vs query.
  # `ccc init` auto-populates these for known models (e.g. Cohere, Voyage, Nvidia NIM,
  # nomic-ai code-retrieval models, Snowflake arctic-embed).
  # indexing_params:
  #   input_type: search_document        # litellm: input_type
  # query_params:
  #   input_type: search_query           # sentence-transformers: prompt_name

envs:                                                # extra environment variables for the daemon
  OPENAI_API_KEY: your-key                           # only needed if not already in your shell environment
```

> **Note:** The daemon inherits your shell environment. If an API key (e.g. `OPENAI_API_KEY`) is already set as an environment variable, you don't need to duplicate it in `envs`. The `envs` field is only for values that aren't in your environment.

> **Custom location:** set `COCOINDEX_CODE_DIR` to place `global_settings.yml` somewhere other than `~/.cocoindex_code/` — useful if you want the file to live alongside your projects (e.g. on a synced folder).

#### `indexing_params` / `query_params`

Some embedding models expose different modes for documents vs queries (asymmetric retrieval). For example, Cohere's v3 models want `input_type: search_document` when embedding corpus content and `input_type: search_query` when embedding a user query; several SentenceTransformers models use `prompt_name: passage` / `prompt_name: query` for the same purpose. These knobs live under `indexing_params` and `query_params`:

```yaml
embedding:
  provider: litellm
  model: cohere/embed-english-v3.0
  indexing_params:
    input_type: search_document
  query_params:
    input_type: search_query
```

`ccc init` populates these automatically for models it recognizes — including all Cohere v3, Voyage, Nvidia NIM, Gemini embedding (`gemini/gemini-embedding-*`, `gemini/text-embedding-*`, `gemini/embedding-*` — LiteLLM auto-maps `input_type` to Gemini's `task_type`), `nomic-ai/CodeRankEmbed`, `nomic-ai/nomic-embed-code`, `nomic-ai/nomic-embed-text-v1`/`v1.5`, `mixedbread-ai/mxbai-embed-large-v1`, and the `Snowflake/snowflake-arctic-embed-*` family — and prints the chosen defaults. For other models, it leaves a commented-out template under `embedding:` so you can fill it in by hand.

OpenAI embeddings (`text-embedding-3-*`, `text-embedding-ada-002`) are intentionally not in the list: they're symmetric and have no equivalent knob.

**Accepted keys:** `prompt_name` (sentence-transformers) and `input_type` (litellm). Other keys are rejected at daemon startup with a clear error. Note: `dimensions` is intentionally not exposed here — output dimension must be identical for indexing and query, so it's a model-wide setting rather than a per-side knob.

**Doctor checks both sides.** `ccc doctor` exercises the model once with `indexing_params` and once with `query_params`, reporting each as a separate `Model Check (indexing)` / `Model Check (query)` entry — so a misconfiguration on one side is diagnosable without hiding behind the other.

**Legacy-bridge warning:** if you're upgrading from an earlier version and your `global_settings.yml` uses `nomic-ai/CodeRankEmbed` or `nomic-ai/nomic-embed-code` without `indexing_params` / `query_params`, the daemon continues to apply the previous behavior (`prompt_name: query` at query time) and prints a one-time warning asking you to make the setting explicit. You can silence the warning by adding an empty block such as `query_params: {}`.

### Project settings (`<project>/.cocoindex_code/settings.yml`)

Per-project. Controls which files to index.

```yaml
include_patterns:
  - "**/*.py"
  - "**/*.js"
  - "**/*.ts"
  - "**/*.rs"
  - "**/*.go"
  # ... (sensible defaults for 28+ file types)

exclude_patterns:
  - "**/.*"                # hidden directories
  - "**/__pycache__"
  - "**/node_modules"
  - "**/dist"
  # ...

language_overrides:
  - ext: inc               # treat .inc files as PHP
    lang: php

chunkers:
  - ext: toml              # use a custom chunker for .toml files
    module: example_toml_chunker:toml_chunker
```

> `.cocoindex_code/` is automatically added to `.gitignore` during init.

Use `chunkers` when you want to control how a file type is split into chunks before indexing.

`module: example_toml_chunker:toml_chunker` means:
- `example_toml_chunker` is a local Python module
- `toml_chunker` is the function inside that module

In practice, this usually means:
- you create a Python file in your project, for example `example_toml_chunker.py`
- you add a function in that file
- you point `settings.yml` at it with `module.path:function_name`

The function should use this signature:

```python
from pathlib import Path
from cocoindex_code.chunking import Chunk

def my_chunker(path: Path, content: str) -> tuple[str | None, list[Chunk]]:
    ...
```

- `path` is the file being indexed
- `content` is the full text of that file
- return `language_override` as a string like `"toml"` if you want to override language detection
- return `None` as `language_override` if you want to keep the detected language
- return a `list[Chunk]` with the chunks you want stored in the index

See [`src/cocoindex_code/chunking.py`](./src/cocoindex_code/chunking.py) for the public types and [`tests/example_toml_chunker.py`](./tests/example_toml_chunker.py) for a complete example.

## Embedding models

With the `[full]` extra installed, `ccc init` defaults to a local SentenceTransformers model ([Snowflake/snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs)) — no API key required. To use a different model, edit `~/.cocoindex_code/global_settings.yml`.

> The `envs` entries below are only needed if the key isn't already in your shell environment — the daemon inherits your environment automatically.

<details>
<summary>Ollama (local)</summary>

```yaml
embedding:
  model: ollama/nomic-embed-text
```

Set `OLLAMA_API_BASE` in `envs:` if your Ollama server is not at `http://localhost:11434`.

</details>

<details>
<summary>OpenAI</summary>

```yaml
embedding:
  model: text-embedding-3-small
  min_interval_ms: 300                               # optional: override the 5ms LiteLLM default
envs:
  OPENAI_API_KEY: your-api-key
```

</details>

<details>
<summary>OpenAI-compatible (custom endpoint)</summary>

Many providers (vLLM, LM Studio, LocalAI, Together, Fireworks, DeepInfra, …) expose an OpenAI-compatible embedding API. Use the `openai/` prefix and point `OPENAI_BASE_URL` at your endpoint:

```yaml
embedding:
  model: openai/your-model-name
envs:
  OPENAI_BASE_URL: https://your-endpoint/v1
  OPENAI_API_KEY: your-api-key
```

Don't append `/embeddings` to the base URL — LiteLLM handles that.

</details>

<details>
<summary>Azure OpenAI</summary>

```yaml
embedding:
  model: azure/your-deployment-name
envs:
  AZURE_API_KEY: your-api-key
  AZURE_API_BASE: https://your-resource.openai.azure.com
  AZURE_API_VERSION: "2024-06-01"
```

</details>

<details>
<summary>Gemini</summary>

```yaml
embedding:
  model: gemini/gemini-embedding-001
envs:
  GEMINI_API_KEY: your-api-key
```

</details>

<details>
<summary>Mistral</summary>

```yaml
embedding:
  model: mistral/mistral-embed
envs:
  MISTRAL_API_KEY: your-api-key
```

</details>

<details>
<summary>Voyage (code-optimized)</summary>

```yaml
embedding:
  model: voyage/voyage-code-3
envs:
  VOYAGE_API_KEY: your-api-key
```

</details>

<details>
<summary>Cohere</summary>

```yaml
embedding:
  model: cohere/embed-v4.0
envs:
  COHERE_API_KEY: your-api-key
```

</details>

<details>
<summary>AWS Bedrock</summary>

```yaml
embedding:
  model: bedrock/amazon.titan-embed-text-v2:0
envs:
  AWS_ACCESS_KEY_ID: your-access-key
  AWS_SECRET_ACCESS_KEY: your-secret-key
  AWS_REGION_NAME: us-east-1
```

</details>

<details>
<summary>Nebius</summary>

```yaml
embedding:
  model: nebius/BAAI/bge-en-icl
envs:
  NEBIUS_API_KEY: your-api-key
```

</details>

Any [LiteLLM-supported model](https://docs.litellm.ai/docs/embedding/supported_embedding) works. When using a LiteLLM model, set `provider: litellm` (or omit `provider` — LiteLLM is the default for non-`sentence-transformers` models). For the full list of env vars each provider reads (API keys, base URLs, regions, …), see LiteLLM's [Setting API Keys](https://docs.litellm.ai/docs/set_keys).

### Local SentenceTransformers models

Set `provider: sentence-transformers` and use any [SentenceTransformers](https://www.sbert.net/) model (no API key required).

**Example — general purpose text model:**
```yaml
embedding:
  provider: sentence-transformers
  model: nomic-ai/nomic-embed-text-v1.5
```

**GPU-optimised code retrieval:**

[`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed) delivers significantly better code retrieval than the default model. It is 137M parameters, requires ~1 GB VRAM, and has an 8192-token context window.

```yaml
embedding:
  provider: sentence-transformers
  model: nomic-ai/CodeRankEmbed
```

**Note:** Switching models requires re-indexing your codebase (`ccc reset && ccc index`) since the vector dimensions differ.

## Supported languages

| Language | Aliases | File Extensions |
|----------|---------|-----------------|
| c | | `.c` |
| cpp | c++ | `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp` |
| csharp | csharp, cs | `.cs` |
| css | | `.css`, `.scss` |
| dtd | | `.dtd` |
| fortran | f, f90, f95, f03 | `.f`, `.f90`, `.f95`, `.f03` |
| go | golang | `.go` |
| html | | `.html`, `.htm` |
| java | | `.java` |
| javascript | js | `.js` |
| json | | `.json` |
| kotlin | | `.kt`, `.kts` |
| lua | | `.lua` |
| markdown | md | `.md`, `.mdx` |
| pascal | pas, dpr, delphi | `.pas`, `.dpr` |
| php | | `.php` |
| python | | `.py` |
| r | | `.r` |
| ruby | | `.rb` |
| rust | rs | `.rs` |
| scala | | `.scala` |
| solidity | | `.sol` |
| sql | | `.sql` |
| svelte | | `.svelte` |
| swift | | `.swift` |
| toml | | `.toml` |
| tsx | | `.tsx` |
| typescript | ts | `.ts` |
| vue | | `.vue` |
| xml | | `.xml` |
| yaml | | `.yaml`, `.yml` |

### Custom database location

By default, the index databases live alongside settings in `<project>/.cocoindex_code/`: `cocoindex.db` (the LMDB incremental-indexing state) and `lancedb/` (the LanceDB vector store directory). When running in Docker, you may want the databases on the container's native filesystem for performance (LMDB doesn't work well on mounted volumes) while keeping the source code and settings on a mounted volume.

Set `COCOINDEX_CODE_DB_PATH_MAPPING` to remap database locations by path prefix:

```bash
COCOINDEX_CODE_DB_PATH_MAPPING=/workspace=/db-files
```

With this mapping, a project at `/workspace/myrepo` stores its databases in `/db-files/myrepo/` instead of `/workspace/myrepo/.cocoindex_code/`. Settings files remain in the original location.

Multiple mappings are comma-separated and resolved in order (first match wins):

```bash
COCOINDEX_CODE_DB_PATH_MAPPING=/workspace=/db-files,/workspace2=/db-files2
```

Both source and target must be absolute paths. If no mapping matches, the default location is used.

## Vector search backend (LanceDB + HNSW)

Chunk embeddings are stored in an embedded **[LanceDB](https://lancedb.github.io/lancedb/)** table (`lancedb/code_chunks.lance` under your index directory). LanceDB runs in-process — no separate server — so the tool stays zero-config, while supporting concurrent reads for the MCP server. It is the only table: branch search adds none.

**Approximate nearest-neighbour (HNSW).** Once a codebase grows past a few hundred chunks, the embedding column is indexed with an **HNSW graph using cosine distance**. HNSW makes query latency *sublinear* in codebase size, versus an exact brute-force scan that grows linearly. The index is built automatically after indexing and maintained incrementally on subsequent updates.

- **Small indexes** (below 256 chunks) skip the ANN index entirely and use LanceDB's exact flat scan — it's already fast at that size and avoids approximation.
- **HNSW is approximate**, so recall is < 100%. The defaults (`m=20`, `ef_construction=300`, query-time `ef=256`) favor recall for typical top-k code search. These live in `cocoindex_code/lancedb_store.py`; raise `ef` for more recall, lower it for less latency.

**Scoring.** Semantic results carry a `score` that is cosine similarity (`1 − cosine_distance`), in a `0..1`-is-better range (1.0 = identical).

**Filtering parity and one delta.** Language filters (`--lang`) are exact matches. Path filters (`--path`) accept `*` / `?` GLOB wildcards, translated to LanceDB's SQL `LIKE` (`*`→`%`, `?`→`_`). The one behavioral delta from the old sqlite-vec backend: GLOB character classes (e.g. `[abc]`) are **not** supported and are treated as literal text.

**Re-indexing.** Switching embedding models (different vector dimensions) requires a rebuild: `ccc reset && ccc index`. Vectors are re-exported from source — there's no in-place migration of raw vectors.

### Concurrency and high-throughput reads

The tool is designed for high-concurrency workloads — several coding agents, or many MCP sessions, hitting the same index at once.

- **Reads never block on writes.** LanceDB is a multi-version store: a search opens a consistent snapshot and reads it independently of any writer. The daemon handles each connection as its own task, so concurrent searches run in parallel — they don't queue behind one another or behind an in-flight index pass.
- **Searchable mid-index.** LanceDB commits chunks incrementally, so rows become queryable *while* indexing is still running — including the very first index pass. A search only waits when the table is genuinely empty; otherwise it reads whatever has been committed so far.
- **Smart refresh.** `ccc search --refresh` (and the MCP `search` tool's `refresh_index`) only kicks off a fresh index pass when none is already in flight. If one is running, the search reads the current table concurrently instead of blocking behind the index lock.
- **Optional HTTP MCP endpoint.** Set `COCOINDEX_CODE_MCP_PORT` to expose a streamable-HTTP MCP server from the daemon (host via `COCOINDEX_CODE_MCP_HOST`, default `127.0.0.1`; disable with `COCOINDEX_CODE_MCP_DISABLE`). It queries the same in-process registry, so many clients can share one warm daemon and embedding model.

### Disk usage and compaction

LanceDB is append-only: every index pass (and every refresh-on-search) writes new data files and keeps the superseded ones as historical versions. Its built-in prune only reclaims versions older than 7 days, so a high-churn index can balloon to tens of GB if left unmanaged.

- **Automatic upkeep.** After each index pass the daemon compacts small fragments and prunes versions older than a short retention window (10 minutes). That window is deliberately kept — it's the safety margin that lets in-flight concurrent reads finish against a snapshot without it being deleted out from under them. This keeps the store lean during normal use with no action on your part.
- **Manual reclaim — `ccc compact`.** For a one-time aggressive reclaim (e.g. after a big re-index), run `ccc compact`. It compacts fragments and prunes *every* version but the latest. The daemon holds the index lock for the duration, so it waits for any in-flight indexing to finish and no write races the prune. It reports bytes before/after and how much was reclaimed.

  There is also a standalone [`scripts/lance_compact.py`](./scripts/lance_compact.py) that talks to the store directly (no daemon needed) — useful for inspection (`--inspect`) or reclaiming when the daemon is stopped.
- **Recovery.** If a write is killed partway (OOM, `docker stop` mid-merge, disk full), the latest table version can reference a truncated file and become unreadable. [`scripts/lance_recover.py`](./scripts/lance_recover.py) rolls back to the most recent version that reads cleanly and prunes the corrupt files; re-run `ccc index` afterward to re-apply any dropped changes incrementally. Stop the daemon first (`ccc daemon stop`).

## Troubleshooting

Run `ccc doctor` to diagnose common issues. It checks your settings, daemon health, embedding model, file matching, and index status — all in one command.

### `MDB_MAP_FULL: Environment mapsize limit reached`

The index is stored in an LMDB database whose maximum size is fixed when the daemon starts. The default ceiling is **4 GiB**, which is plenty for most projects but can be exhausted by very large codebases (tens of thousands of files), especially with high-dimensional embedding models like `nomic-ai/CodeRankEmbed`.

Raise the ceiling with the `COCOINDEX_LMDB_MAP_SIZE` environment variable (value in **bytes**). LMDB only grows the file as data is written, so a high limit doesn't pre-allocate disk — it's safe to set it generously:

```yaml
# ~/.cocoindex_code/global_settings.yml
envs:
  COCOINDEX_LMDB_MAP_SIZE: "34359738368"   # 32 GiB (= 32 * 1024^3)
```

Or, if you prefer to set it in your shell environment (the daemon inherits it):

```bash
export COCOINDEX_LMDB_MAP_SIZE=$((32 * 1024 * 1024 * 1024))   # 32 GiB
```

The map size is read when the daemon starts, so restart it to pick up the change, then re-index:

```bash
ccc daemon restart
ccc index
```

## Legacy: environment variables

If you previously configured `cocoindex-code` via environment variables, the `cocoindex-code` MCP command still reads them and auto-migrates to YAML settings on first run. We recommend switching to the YAML settings for new setups.

| Environment Variable | YAML Equivalent |
|---------------------|-----------------|
| `COCOINDEX_CODE_EMBEDDING_MODEL` | `embedding.model` in `global_settings.yml` |
| `COCOINDEX_CODE_DEVICE` | `embedding.device` in `global_settings.yml` |
| `COCOINDEX_CODE_ROOT_PATH` | Run `ccc init` in your project root instead |
| `COCOINDEX_CODE_EXCLUDED_PATTERNS` | `exclude_patterns` in project `settings.yml` |
| `COCOINDEX_CODE_EXTRA_EXTENSIONS` | `include_patterns` + `language_overrides` in project `settings.yml` |

## Telemetry

The underlying [CocoIndex](https://github.com/cocoindex-io/cocoindex) engine sends anonymous usage telemetry so aggregate usage can be understood. The events identify themselves as `application: cocoindex-code`. **No** source code, file paths, queries, search results, embeddings, or settings are collected.

To opt out, set:

```bash
export COCOINDEX_DISABLE_USAGE_TRACKING=1
```

## Contributing

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for development. Before opening a PR, run the same checks CI runs:

```bash
uv sync                          # install dev dependencies (Ruff, mypy, pytest, prek)
uv run prek run --all-files      # trailing-whitespace/EOF, Ruff lint+format, mypy, pytest
```

Or run the individual checks directly:

```bash
uv run mypy .            # type check
uv run pytest tests/     # test suite
```

To have the checks run on every `git commit`, install the hook once with `uv run prek install`.

## License

Apache-2.0. This is a fork of [cocoindex-io/cocoindex-code](https://github.com/cocoindex-io/cocoindex-code); upstream copyright and license are retained.
