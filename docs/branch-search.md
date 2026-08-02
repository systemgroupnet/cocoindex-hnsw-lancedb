# Branch search

Search arbitrary git branches without maintaining a full index per branch.

## Motivation

The indexer walks the **working tree on disk** (`indexer_main` →
`localfs.walk_dir(CODEBASE_DIR)`) and stores every chunk in one LanceDB table
(`code_chunks`) keyed by a content-derived id. Whatever is checked out — in a
deployment, the branch the daemon keeps `git reset`-ing to — is the only thing
searchable. That blocks a common workflow: semantic code search / review of a
*feature branch* through the MCP `search` tool.

Re-indexing the whole tree per branch is the obvious fix and the wrong one: it
re-embeds the entire codebase for every branch, most of which is identical to
the base. The branch's *novelty* is exactly the files it changed, which is
usually small. Branch search exploits that.

## Model

There is exactly one index — the **base**. Nothing per-branch is indexed,
embedded, or persisted: a branch's changed files are read out of the object
database and scanned with ripgrep on the request path.

> **Removed: semantic overlays.** Earlier versions embedded a branch's changed
> files into an ephemeral `overlay_<sha>` LanceDB table when the diff was under
> a threshold, falling back to a lexical scan above it. That path is gone. It
> embedded on the request path (model calls inside a search), it created a table
> per branch *commit* that was only evicted after a 7-day TTL, and it held every
> changed file's chunks and vectors in memory at once while building. The
> lexical path it used to fall back to is now the only path, for every branch.
> `drop_legacy_overlays` removes the tables left behind, on first open.

### Base ref

The base is whatever ref the persistent `code_chunks` table represents. It is
auto-detected at index time from `HEAD` (`git rev-parse --abbrev-ref HEAD`) and
can be overridden with `COCOINDEX_CODE_BASE_REF`. Hardcoding `main` would break
any deployment tracking `develop` / `release/*`; detecting `HEAD` stays correct
by construction.

> **Known staleness window.** The diff is computed against the base ref's
> *current* tip, which can be newer than the commit actually indexed into
> `code_chunks` (the base index only advances when a scheduled index pass runs).
> A file changed on the base *after* its last index pass but *not* on the branch
> can therefore be missed by the shadow set. Tolerable for review; a future
> refinement pins the diff base to the indexed commit SHA (see Future work).
> The [pre-search refresh](#pre-search-refresh) widens this window when the pull
> gate is on, since it advances the base ref without re-indexing.

### What "search branch X" means

The correct result set is:

```
base chunks
  MINUS chunks for files X modified or deleted   (the "shadow set")
  PLUS  X's version of files it added or modified
```

The shadow set is exactly `git diff --name-status <merge-base>..X`. It drives
**both** what is scanned from the branch (added + modified) *and* what is hidden
from the base index (modified + deleted). Skipping the subtraction returns stale base
content for every modified file at a misleadingly high score — that omission is
the single most important correctness point.

## Storage layout

```
.cocoindex_code/lancedb/
  code_chunks.lance/            # the base index - the only table
```

Branch search writes nothing. The branch side is read from git and scanned in
memory, one memory-budgeted batch at a time.

## Pre-search refresh

A branch is usually searched moments after it is pushed, so serving whatever the
last scheduled pull left behind is routinely stale. Every branch search therefore
refreshes the clone first. It is best-effort: any failure is logged and the
search continues against whatever the clone already has.

What "refresh" means follows `COCOINDEX_CODE_GIT_PULL_ENABLED`, the same gate the
daily maintenance workflow uses — a pre-search refresh must not quietly turn on
the destructive step an operator deliberately left off:

| Gate | Action | Effect |
| --- | --- | --- |
| **on** | `git fetch --prune` + `git reset --hard @{u}` | Base ref and working tree advance, so the diff base is current. |
| **off** (default) | `git fetch --prune` | Every remote branch and its newest commits become searchable; HEAD, index, and working tree untouched. |

With the gate **on**, note the interaction with the staleness window above: the
refresh advances the base ref but the base *index* only catches up on the next
index pass, so the gap between them widens. It also rewrites the working tree,
which can shift under a concurrent index pass — the same exposure the scheduled
pull already carries, now reachable from the query path.

Refreshes are throttled to one per `COCOINDEX_CODE_BRANCH_REFRESH_SECONDS`
(default 60; `0` refreshes on every search). Agents fire searches in bursts, and
a round-trip per search buys nothing. The throttle timestamp is stamped on
failure too, so an unreachable remote costs one timeout per interval rather than
one per search.

## Ref resolution

The daemon's clone only ever checks out the base branch, so a feature branch
normally exists there as a *remote-tracking* ref — or not at all, if it was
pushed after the last scheduled pull. `git rev-parse` does none of the DWIM that
`git checkout` does, so resolving `X` walks three steps and stops at the first
hit:

1. **`X` as given** — a local branch, a tag, or a SHA.
2. **`refs/remotes/<remote>/X`**, for each configured remote (`origin` first) —
   the common case for a branch that has been fetched but never checked out. No
   network.
3. **On-demand fetch** — `git fetch <remote> +refs/heads/X:refs/remotes/<remote>/X`,
   then resolve again. Writes only objects and the remote-tracking ref; never the
   working tree, the index, or a local branch. Because the fetch uses an explicit
   refspec (rather than leaving the result in `FETCH_HEAD`), the branch becomes a
   normal remote-tracking ref and later searches resolve it with no network at
   all. Controlled by `COCOINDEX_CODE_BRANCH_FETCH_ENABLED` (on by default) and
   authenticated with the same `COCOINDEX_CODE_GIT_USERNAME` /
   `COCOINDEX_CODE_GIT_PASSWORD` credentials the scheduled pull uses, injected
   via the same inline credential helper (never on disk, never in argv). Only
   plain branch names are fetchable; tags and SHAs must already be in the clone.

The resolved SHA — not the caller's ref string — is what `git diff` and
`git show` are given downstream, since a bare `X` would not resolve when the
branch lives only under `refs/remotes/`.

**No checkout, ever.** Branch content is read out of the object database with
`git show <sha>:<path>`; nothing runs `checkout`/`switch`/`reset`. HEAD, the
index, the working tree, and the local branch list are byte-identical before and
after a branch search (regression-guarded in
`test_branch_search_git_calls_leave_the_checkout_alone`). That is what lets the
daemon serve parallel searches for different branches against one clone whose
working tree stays on the base. The single write is the fetch's remote-tracking
ref: two searches racing to fetch the *same* new branch contend on git's per-ref
lock, so the loser re-checks locally — by then the winner's ref is there — rather
than failing.

Caller-supplied refs are rejected before reaching a command line if they could be
read as a git option (leading `-`) or contain whitespace; a ref that will be
fetched must additionally look like a plain branch name.

## Query path

One path, for every branch regardless of divergence.

### Steps

1. Resolve `X` to a commit SHA (see [Ref resolution](#ref-resolution)). Every
   later git call addresses the branch by SHA, so a ref that exists only as
   `refs/remotes/<remote>/<name>` still works.
2. `git diff --name-status <merge-base>..X` gives the added/modified/deleted
   sets, filtered by the same include/exclude/gitignore matchers the base
   indexer uses.
3. Serve **two labeled sections**:
   - **`source = "semantic"`** - `code_chunks` with
     `WHERE file_path NOT IN (<shadow set>)`: the base, with the branch's
     touched files hidden.
   - **`source = "lexical"`** - a ripgrep scan (with an in-process Python
     fallback when the `rg` binary is absent) of the branch's version of the
     files it added or modified.

Lexical hits carry no cosine score, so they are a separate section rather than
being merged into the semantic ranking with a fabricated score. Each result
carries its `source` so the caller/agent knows which is which.

### Bounding the diff scan

There is no cap on how many files a branch may change, so the scan is bounded by
memory rather than by file count. `ripgrep.blob_batches` reads the changed files
from git in batches that stay inside the memory governor's
`ScanBudget.blob_batch_bytes`, skipping any single blob over
`max_filesize_bytes` before it is read (sizes come from one `git cat-file`
pass). Each batch is scored and only the running top-`limit` is kept.

That batching is exact, not an approximation: a lexical hit's score is the
fraction of query terms present in its own snippet, independent of every other
hit, so merging per-batch winners yields the same ranking as scoring the whole
diff at once. The whole scan holds one text-scan permit from the governor
(`scan_slot`), the same gate the `ripgrep` tool uses - a branch search spawns rg
exactly like a grep does.

## MCP surface

The `search` tool gains one optional parameter:

- **`branch`** — the git branch (or ref/SHA) to search. Omitted, or equal to the
  base ref, means "search the base" — today's behavior, zero overhead. Backward
  compatible.

Each result gains a `source` field (`"semantic"` | `"lexical"`, default
`"semantic"`) so existing clients keep working and branch-aware clients can
render the two sections.

The `ripgrep` tool takes the same `branch` parameter and resolves it through the
same path (`BranchSearch.resolve_branch`: refresh → resolve → diff → filter),
then applies the same decomposition to a text scan — the base working tree minus
the shadow set, plus rg over the branch's version of the files it changed. The
only difference from `search` is what the base side does: a vector query there,
a text scan here. No index is needed either way.

## Eviction

Nothing to evict. Branch search creates no per-branch state.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `COCOINDEX_CODE_BASE_REF` | auto (`HEAD`) | Ref the base index represents / diff base. |
| `COCOINDEX_CODE_BRANCH_FETCH_ENABLED` | on | Set falsy to forbid the on-demand fetch, restricting search to refs already in the clone. |
| `COCOINDEX_CODE_BRANCH_REFRESH_SECONDS` | `60` | Minimum seconds between pre-search refreshes; `0` refreshes on every search. |
| `COCOINDEX_CODE_GIT_PULL_ENABLED` | off | When on, the pre-search refresh is a full pull (`reset --hard`) instead of a fetch. |
| `COCOINDEX_CODE_GIT_USERNAME` / `COCOINDEX_CODE_GIT_PASSWORD` | unset | HTTPS credentials for the pre-search refresh and the on-demand fetch (shared with the scheduled pull). |

## Future work (deferred, tracked)

The current implementation drives git with plumbing that reads
(`rev-parse`, `diff`, `show`, `remote`) plus the one narrowly scoped fetch above,
and does **not** yet include the hardened read-only guarantee layer. Before this
is exposed to untrusted branch input in production, add:

1. **Enforced read-only git layer.** An allowlist wrapper permitting only
   read-only subcommands (plus the fetch), every invocation run with
   `GIT_INDEX_FILE` pointed at a temp path so the real index/working tree cannot
   be mutated even by a mis-issued command. This is the guarantee that branch
   search never disturbs the base index's source tree; today the ref validation
   in [Ref resolution](#ref-resolution) is what stands in for it.
2. **SHA-pinned diff base.** Pin the diff base to the commit actually indexed
   into `code_chunks` (closing the staleness window above).
3. **Worktree option.** For very large diffs, a read-only sparse worktree that
   reuses the full CocoIndex incremental pipeline instead of the in-memory blob
   path.
