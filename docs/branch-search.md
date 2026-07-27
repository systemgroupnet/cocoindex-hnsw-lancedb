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

There is one persistent index — the **base** — plus small, ephemeral
**overlays**, one per branch commit.

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

### What "search branch X" means

The correct result set is:

```
base chunks
  MINUS chunks for files X modified or deleted   (the "shadow set")
  PLUS  X's version of files it added or modified
```

The shadow set is exactly `git diff --name-status <merge-base>..X`. It drives
**both** what the overlay embeds (added + modified) *and* what is hidden from
the base index (modified + deleted). Skipping the subtraction returns stale base
content for every modified file at a misleadingly high score — that omission is
the single most important correctness point.

## Storage layout

Overlays live in the project's LanceDB directory alongside the base table:

```
.cocoindex_code/lancedb/
  code_chunks.lance/            # the base index
  overlay_<sha12>.lance/        # one per indexed branch commit
  overlays.json                 # sidecar: table -> {branch, sha, last_access, created}
```

An overlay table has the **same physical schema** as `code_chunks`
(`id, file_path, language, content, start_line, end_line, embedding`) so the
existing query path reads it unchanged. Overlays are small, so they stay below
`INDEX_MIN_ROWS` and use LanceDB's exact flat scan — no HNSW build needed.

Overlay rows are embedded with the **indexing** embedder and
`INDEXING_EMBED_PARAMS` (not the query embedder), so their vectors are
"passage"-style and directly comparable to base rows. (Trade-off: a first-time
overlay build can briefly serialize behind an in-flight index pass on the
indexing embedder's lock. Overlays are cached, so this is a one-time cost per
branch commit.)

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

## Query paths

A branch's divergence from the base decides the path, gated by
`COCOINDEX_CODE_BRANCH_MAX_CHANGED_FILES` (default 50).

### Low divergence — semantic overlay

1. Resolve `X` → commit SHA (see [Ref resolution](#ref-resolution)). The SHA is
   the overlay cache key, so a new commit / force-push transparently invalidates
   the old overlay, and every later git call addresses the branch by SHA.
2. `git diff --name-status <merge-base>..X` → added/modified/deleted sets,
   filtered by the same include/exclude/gitignore matchers the base indexer uses.
3. Build (or reuse) `overlay_<sha>`: read each added/modified file via
   `git show X:<path>`, chunk it (shared `chunk_file_content`), embed, write.
4. Query = two vector searches merged by cosine score:
   - `overlay_<sha>` — unfiltered.
   - `code_chunks` — `WHERE file_path NOT IN (<shadow set>)`.

   Both use the same embedder + metric, so scores are directly comparable. All
   results are one unified semantic ranking (`source = "semantic"`).

### High divergence — lexical fallback

When the changed-file count exceeds the threshold, embedding the diff is too
expensive. Instead of rejecting the branch, serve **two labeled sections**:

- **`source = "semantic"`** — `code_chunks` minus the shadow set (the base, with
  the branch's touched files hidden).
- **`source = "lexical"`** — a ripgrep (with in-process Python fallback when the
  `rg` binary is absent) scan of the branch's version of the changed files.

Lexical hits carry no cosine score, so they are a separate section rather than
being merged into the semantic ranking with a fabricated score. Each result
carries its `source` so the caller/agent knows which is which.

## MCP surface

The `search` tool gains one optional parameter:

- **`branch`** — the git branch (or ref/SHA) to search. Omitted, or equal to the
  base ref, means "search the base" — today's behavior, zero overhead. Backward
  compatible.

Each result gains a `source` field (`"semantic"` | `"lexical"`, default
`"semantic"`) so existing clients keep working and branch-aware clients can
render the two sections.

## Eviction

Overlays are disk. Each is dropped when it hasn't been *searched* within
`COCOINDEX_CODE_BRANCH_OVERLAY_TTL_DAYS` (default 7). Last-access is tracked in
the `overlays.json` sidecar. A sweep runs lazily on each branch search and again
in the daily maintenance workflow. Eviction is a `drop_table` + sidecar prune.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `COCOINDEX_CODE_BASE_REF` | auto (`HEAD`) | Ref the base index represents / diff base. |
| `COCOINDEX_CODE_BRANCH_MAX_CHANGED_FILES` | `50` | Above this, use the lexical fallback instead of a semantic overlay. |
| `COCOINDEX_CODE_BRANCH_OVERLAY_TTL_DAYS` | `7` | Evict an overlay not searched within this many days. |
| `COCOINDEX_CODE_BRANCH_FETCH_ENABLED` | on | Set falsy to forbid the on-demand fetch, restricting search to refs already in the clone. |
| `COCOINDEX_CODE_GIT_USERNAME` / `COCOINDEX_CODE_GIT_PASSWORD` | unset | HTTPS credentials for the on-demand fetch (shared with the scheduled pull). |

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
