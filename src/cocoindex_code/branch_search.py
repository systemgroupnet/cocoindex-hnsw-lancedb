"""Branch search: the base index, minus the branch's changes, plus a scan of them.

See ``docs/branch-search.md`` for the full design. In short: the persistent
``code_chunks`` table is the *base*, and a branch search returns

* semantic results over the base with the files the branch touched hidden (the
  "shadow set"), so a stale base copy of a modified file never surfaces, and
* a ripgrep scan of the branch's own version of the files it added or modified,
  read straight out of the object database and returned as a distinct
  ``source="lexical"`` section.

Nothing about the branch is embedded or indexed. Earlier versions built an
ephemeral ``overlay_<sha>`` table per branch commit when the diff was small
enough; that path is gone — it embedded on the request path, accumulated a
table per commit, and held every changed file in memory at once.
:func:`drop_legacy_overlays` cleans up the tables it left behind.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, NamedTuple

from cocoindex.ops.text import detect_code_language
from cocoindex.resources.file import PatternFilePathMatcher

from . import git_ops, ripgrep
from .indexer import GitignoreAwareMatcher
from .lexical import LexicalFile, LexicalHit, lexical_search
from .memory import ScanBudget
from .protocol import SearchResult
from .query import open_table, query_codebase
from .schedule import git_hard_reset_sync, git_pull_enabled
from .schema import QueryResult
from .settings import lancedb_dir_path, load_gitignore_spec, load_project_settings
from .shared import LANCE_DB, MEMORY_GOVERNOR

if TYPE_CHECKING:
    import cocoindex as coco

logger = logging.getLogger(__name__)

# --- Config knobs (single source of truth) ----------------------------------

ENV_REFRESH_SECONDS = "COCOINDEX_CODE_BRANCH_REFRESH_SECONDS"

_DEFAULT_REFRESH_SECONDS = 60.0

# Written by the removed overlay path; deleted by `drop_legacy_overlays`.
_LEGACY_SIDECAR_NAME = "overlays.json"
_LEGACY_TABLE_PREFIX = "overlay_"


def _float_env(name: str, default: float, *, allow_zero: bool = False) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if value > 0 or (allow_zero and value == 0):
        return value
    return default


class BranchView(NamedTuple):
    """A branch resolved against the base: what to read from it, what to hide.

    Shared by both branch-aware paths (this module's search and the ``ripgrep``
    tool) so they can never disagree about what "the branch" contains.
    """

    sha: str
    # Files to read from the branch — added + modified, filtered to what the
    # base indexer would include.
    branch_paths: list[str]
    # Base paths to hide — modified + deleted, same filtering.
    shadow_paths: list[str]


class BranchSearch:
    """Resolves branches and searches them against one project's base index.

    One instance per :class:`~cocoindex_code.project.Project`. Holds no
    per-branch state: every search resolves the ref, diffs it, and scans the
    diff. Nothing is cached, built, or evicted.
    """

    def __init__(self, env: coco.Environment, project_root: Path) -> None:
        self._env = env
        self._root = project_root
        self._refresh_lock = asyncio.Lock()
        self._last_refresh: float | None = None

    # -- public API ----------------------------------------------------------

    async def search(
        self,
        *,
        query: str,
        base_ref: str,
        branch_ref: str,
        languages: list[str] | None,
        paths: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[SearchResult]:
        """Search *branch_ref* against the base index.

        Raises ``RuntimeError`` when the branch can't be resolved (locally or by
        fetching it) or the diff can't be computed.
        """
        view = await self.resolve_branch(base_ref=base_ref, branch_ref=branch_ref)
        main_table = await open_table(self._env)

        # Branch identical to base (or only touches ignored files): plain search.
        if not view.branch_paths and not view.shadow_paths:
            rows = await query_codebase(
                query, main_table, self._env,
                limit=limit, offset=offset, languages=languages, paths=paths,
            )
            return [_to_result(r, "semantic") for r in rows]

        # Section 1: semantic over the base, with the branch's touched files hidden.
        main_rows = await query_codebase(
            query, main_table, self._env,
            limit=limit, offset=offset, languages=languages, paths=paths,
            exclude_paths=view.shadow_paths or None,
        )
        results = [_to_result(r, "semantic") for r in main_rows]

        # Section 2: ripgrep over the branch's version of the files it changed.
        for hit in await self._scan_diff(view, query, limit):
            results.append(
                SearchResult(
                    file_path=hit.file_path, language=hit.language, content=hit.content,
                    start_line=hit.start_line, end_line=hit.end_line, score=hit.score,
                    source="lexical",
                )
            )
        return results

    async def resolve_branch(self, *, base_ref: str, branch_ref: str) -> BranchView:
        """Refresh the clone, resolve *branch_ref*, and diff it against the base.

        The shared front half of every branch-aware search. Raises
        ``RuntimeError`` when the ref can't be resolved (locally or by fetching
        it) or the diff can't be computed.
        """
        await self._refresh_clone()

        # Resolution can fetch, so it runs off the event loop. Everything after
        # this point addresses the branch by SHA: the ref may live only as
        # `refs/remotes/<remote>/<branch>`, which `git diff`/`git show` would not
        # find under its bare name.
        branch_sha = await asyncio.to_thread(
            git_ops.resolve_commit,
            self._root,
            branch_ref,
            allow_fetch=git_ops.fetch_enabled(),
            credentials=git_ops.load_credentials(),
        )
        if branch_sha is None:
            raise RuntimeError(_unresolved_message(branch_ref))

        diff = git_ops.branch_diff(self._root, base_ref, branch_sha)
        if diff is None:
            raise RuntimeError(
                f"could not diff {branch_ref!r} against base {base_ref!r} "
                f"(is {base_ref!r} a valid ref?)"
            )

        branch_paths, shadow_paths = self._filter(diff)
        return BranchView(
            sha=branch_sha, branch_paths=branch_paths, shadow_paths=shadow_paths
        )

    # -- diff scan -------------------------------------------------------------

    async def _scan_diff(
        self, view: BranchView, query: str, limit: int
    ) -> list[LexicalHit]:
        """Rank the branch's changed files against *query* with ripgrep.

        Runs under the memory governor's scan gate, on the same budget the
        ``ripgrep`` tool uses — this spawns rg exactly like a grep does, and
        there is no divergence ceiling any more, so a branch that rewrote
        thousands of files has to cost bounded memory rather than being refused
        a semantic path.
        """
        if not view.branch_paths:
            return []
        governor = self._env.get_context(MEMORY_GOVERNOR)
        ext_lang_map = self._language_overrides()
        async with governor.scan_slot():
            # git blob reads and rg are both blocking; keep the loop free.
            return await asyncio.to_thread(
                _scan_diff_sync,
                self._root, view.sha, view.branch_paths, query, limit,
                governor.scan_budget, ext_lang_map,
            )

    # -- pre-search clone refresh ---------------------------------------------

    async def _refresh_clone(self) -> None:
        """Bring the clone up to date before searching. Best-effort, never raises.

        A branch is usually searched moments after it is pushed, so a search that
        only reads whatever the last scheduled pull left behind is routinely
        stale. Refreshing first means both the branch *and* the base it is diffed
        against reflect the remote.

        Throttled to one refresh per ``COCOINDEX_CODE_BRANCH_REFRESH_SECONDS``
        (default 60, ``0`` to refresh on every search): an agent typically fires a
        burst of searches, and each one paying a network round-trip buys nothing.
        The timestamp is stamped on failure too, so an unreachable remote costs
        one timeout per interval rather than one per search.
        """
        interval = _float_env(ENV_REFRESH_SECONDS, _DEFAULT_REFRESH_SECONDS, allow_zero=True)
        async with self._refresh_lock:
            now = time.monotonic()
            if self._last_refresh is not None and now - self._last_refresh < interval:
                return
            try:
                outcome = await asyncio.to_thread(_refresh_clone_sync, self._root)
            except Exception:
                # _refresh_clone_sync is already non-raising; this is the last
                # guard that a refresh problem can never fail a search.
                logger.exception("Pre-search refresh of %s failed", self._root)
            else:
                logger.info("Pre-search refresh of %s: %s", self._root, outcome)
            self._last_refresh = time.monotonic()

    # -- filtering -----------------------------------------------------------

    def _filter(self, diff: git_ops.BranchDiff) -> tuple[list[str], list[str]]:
        """Apply the base indexer's include/exclude/gitignore rules to the diff.

        Returns ``(branch_paths, shadow_paths)`` — files to read from the branch
        (added+modified) and base paths to hide (modified+deleted), each
        restricted to what the base index would actually contain.
        """
        matcher = self._matcher()
        branch = [p for p in diff.to_scan if matcher.is_file_included(PurePath(p))]
        shadow = [p for p in diff.shadow if matcher.is_file_included(PurePath(p))]
        return branch, shadow

    def _matcher(self) -> GitignoreAwareMatcher:
        ps = load_project_settings(self._root)
        base = PatternFilePathMatcher(
            included_patterns=ps.include_patterns, excluded_patterns=ps.exclude_patterns
        )
        return GitignoreAwareMatcher(base, load_gitignore_spec(self._root), self._root)

    def _language_overrides(self) -> dict[str, str]:
        ps = load_project_settings(self._root)
        return {f".{lo.ext}": lo.lang for lo in ps.language_overrides}


def _scan_diff_sync(
    root: Path,
    branch_sha: str,
    branch_paths: list[str],
    query: str,
    limit: int,
    budget: ScanBudget,
    ext_lang_map: dict[str, str],
) -> list[LexicalHit]:
    """Blocking half of :meth:`BranchSearch._scan_diff`.

    Files are read from git in memory-budgeted batches and scored batch by
    batch, keeping only the running top *limit*. That is exact rather than an
    approximation: a hit's score is the fraction of query terms in its own
    snippet, independent of every other hit, so merging per-batch winners
    yields the same ranking as scoring the whole diff at once — at the cost of
    one batch resident instead of the entire changed file set.
    """
    hits: list[LexicalHit] = []
    for batch in ripgrep.blob_batches(root, branch_sha, branch_paths, budget):
        files = [
            LexicalFile(path, content, _detect_language(path, ext_lang_map))
            for path, content in batch.items()
        ]
        batch_hits = lexical_search(files, query, limit=limit, budget=budget)
        if not batch_hits:
            continue
        hits.extend(batch_hits)
        hits.sort(key=lambda h: (-h.score, h.file_path, h.start_line))
        del hits[limit:]
    return hits


async def drop_legacy_overlays(env: coco.Environment, project_root: Path) -> None:
    """Remove the ``overlay_<sha>`` tables the old semantic-overlay path left.

    Runs once per project when it is opened. Overlays were evicted only after a
    TTL, so a deployment upgrading into this version can be carrying a week of
    them; nothing reads them any more, so they are pure disk. Best-effort —
    failing to clean up must never stop a project from loading.
    """
    try:
        conn: Any = env.get_context(LANCE_DB)
        # Paginated deliberately: the deployments that need this cleanup are the
        # ones carrying a TTL window's worth of overlays, which is exactly when
        # one page isn't all of them.
        names: list[str] = []
        page_token: str | None = None
        while True:
            page = await conn.list_tables(page_token=page_token)
            names.extend(page.tables)
            page_token = page.page_token
            if not page_token:
                break

        stale = [n for n in names if n.startswith(_LEGACY_TABLE_PREFIX)]
        for name in stale:
            await conn.drop_table(name, ignore_missing=True)
        sidecar = lancedb_dir_path(project_root) / _LEGACY_SIDECAR_NAME
        sidecar.unlink(missing_ok=True)
        if stale:
            logger.info(
                "Dropped %d obsolete branch overlay table(s) from %s",
                len(stale), project_root,
            )
    except Exception:
        logger.exception("Could not clean up obsolete branch overlays in %s", project_root)


def _refresh_clone_sync(root: Path) -> str:
    """Pull (or fetch) *root*; return a one-line outcome for the log. Never raises.

    Which one depends on ``COCOINDEX_CODE_GIT_PULL_ENABLED``, the same gate the
    daily maintenance workflow uses:

    * **on** — a full pull (``fetch --prune`` + ``reset --hard @{u}``), so the
      base ref and working tree advance and the diff base is current. This
      rewrites the working tree, which is why it stays behind the operator's
      opt-in; note the base *index* only catches up on the next index pass, so a
      refresh here widens the staleness window documented in
      ``docs/branch-search.md``.
    * **off** — ``fetch --prune`` only. Every remote branch and its newest
      commits become searchable with no working-tree impact; the diff base stays
      wherever the last real pull left it.
    """
    if not git_pull_enabled():
        error = git_ops.fetch_all(root, credentials=git_ops.load_credentials())
        return f"fetch failed: {error}" if error else "fetched"
    result = git_hard_reset_sync(root, git_ops.load_credentials())
    return f"{result.status}: {result.message}"


def _unresolved_message(branch_ref: str) -> str:
    """Why *branch_ref* couldn't be resolved — the fix differs by fetch setting."""
    if git_ops.fetch_enabled():
        return (
            f"ref {branch_ref!r} could not be resolved: no local branch, tag, or "
            f"remote-tracking ref matches it, and fetching it from the remote failed "
            f"(check the branch name, the remote, and the git credentials)"
        )
    return (
        f"ref {branch_ref!r} not found in the local clone and on-demand fetch is "
        f"disabled ({git_ops.ENV_FETCH_ENABLED}); fetch it first or re-enable fetching"
    )


def _detect_language(path: str, ext_lang_map: dict[str, str]) -> str:
    suffix = PurePath(path).suffix
    return ext_lang_map.get(suffix) or detect_code_language(filename=PurePath(path).name) or "text"


def _to_result(row: QueryResult, source: str) -> SearchResult:
    return SearchResult(
        file_path=row.file_path, language=row.language, content=row.content,
        start_line=row.start_line, end_line=row.end_line, score=row.score, source=source,
    )
