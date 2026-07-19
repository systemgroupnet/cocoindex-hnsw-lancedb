"""Configuration and helpers for the daemon's daily maintenance workflow.

The daemon runs one scheduled workflow per day that, for each target project:

1. **git pull** — hard-resets the working tree to its upstream so indexing sees
   the latest committed code (opt-in; skipped for non-git directories),
2. **index** — runs an incremental index pass over the refreshed tree, and
3. **push metrics** — writes an index-stats snapshot to MySQL for DevLake.

Each step is best-effort: a failure is logged and the workflow continues to the
next step (and the next project), so a broken git remote never blocks indexing
and a failed index never blocks the metrics snapshot.

Configuration (all via environment variables):

* ``COCOINDEX_CODE_SCHEDULE_ENABLED``    — set to a falsy value
  (``0``/``false``/``no``/``off``) to disable the whole workflow. On by default.
* ``COCOINDEX_CODE_SCHEDULE_TIME``       — local time of day to run, ``HH:MM``
  (24-hour). Default ``03:00``.
* ``COCOINDEX_CODE_SCHEDULE_WORKSPACES`` — comma-separated project roots to
  process each run, so the workflow bootstraps a repo even if nothing has
  queried it yet. Union'd with the projects already loaded in the daemon. In the
  Docker image this defaults to ``/workspace`` (the mounted repo).
* ``COCOINDEX_CODE_GIT_PULL_ENABLED``    — set to a truthy value
  (``1``/``true``/``yes``/``on``) to enable the git-pull step. **Off by
  default**: the step hard-resets the working tree to its upstream, discarding
  any local changes, so it is opt-in. Directories without a ``.git`` are skipped
  regardless.
* ``COCOINDEX_CODE_GIT_USERNAME`` / ``COCOINDEX_CODE_GIT_PASSWORD`` — optional
  HTTPS credentials (username + password or personal-access token) for the fetch.
  When a password is set, they are injected via an inline git credential helper
  scoped to that single fetch — never written to disk, never placed in the remote
  URL, and never in the process argv (the helper reads them from the environment).
  Leave unset to rely on whatever auth the host already provides (SSH key, a
  token baked into the remote URL, etc.). For token auth the username is usually
  any non-empty value (e.g. the token itself, or ``x-access-token`` on GitHub);
  supply both to be safe. Ignored for SSH remotes, which don't use credential
  helpers.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# --- Environment knobs (single source of truth) ----------------------------

ENV_ENABLED = "COCOINDEX_CODE_SCHEDULE_ENABLED"
ENV_TIME = "COCOINDEX_CODE_SCHEDULE_TIME"
ENV_WORKSPACES = "COCOINDEX_CODE_SCHEDULE_WORKSPACES"
ENV_GIT_PULL_ENABLED = "COCOINDEX_CODE_GIT_PULL_ENABLED"
ENV_GIT_USERNAME = "COCOINDEX_CODE_GIT_USERNAME"
ENV_GIT_PASSWORD = "COCOINDEX_CODE_GIT_PASSWORD"

_DEFAULT_TIME = time(3, 0)

# Hard ceiling on a single git fetch/reset so a hung remote can't wedge the
# scheduled workflow forever. On timeout the step is reported as an error and
# the workflow moves on.
_GIT_TIMEOUT_SECONDS = 300

# Environment names the inline credential helper reads. Distinct from the
# user-facing ENV_GIT_* knobs: we set these on the git subprocess env explicitly
# so the helper's value can stay a fixed, secret-free string.
_CRED_ENV_USERNAME = "CCC_GIT_CRED_USERNAME"
_CRED_ENV_PASSWORD = "CCC_GIT_CRED_PASSWORD"

# Inline git credential helper: on a `get`, echo the credentials from the
# environment in git's credential-protocol format. The password never appears in
# argv (only this fixed string does) nor on disk. A leading `!` makes git run it
# through a shell; the helper ignores git's `get`/`store`/`erase` argument.
_CREDENTIAL_HELPER = (
    f'!f() {{ echo "username=${_CRED_ENV_USERNAME}"; '
    f'echo "password=${_CRED_ENV_PASSWORD}"; }}; f'
)

_FALSY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GitCredentials:
    """HTTPS username + password/token for the git-pull step."""

    username: str
    password: str


@dataclass(frozen=True)
class ScheduleConfig:
    """Resolved configuration for the daily maintenance workflow."""

    enabled: bool
    run_time: time
    workspaces: tuple[Path, ...]
    git_pull_enabled: bool
    git_credentials: GitCredentials | None


# --- Config -----------------------------------------------------------------


def _is_falsy(raw: str) -> bool:
    return raw.strip().lower() in _FALSY


def _is_truthy(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY


def _parse_time(raw: str | None) -> time:
    """Parse ``HH:MM`` (24-hour). Falls back to the default on anything invalid."""
    if not raw or not raw.strip():
        return _DEFAULT_TIME
    text = raw.strip()
    parts = text.split(":")
    if len(parts) == 2:
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except ValueError:
            hour = minute = -1
        if 0 <= hour < 24 and 0 <= minute < 60:
            return time(hour, minute)
    logger.warning(
        "%s=%r is not a valid HH:MM time; using %s",
        ENV_TIME,
        raw,
        _DEFAULT_TIME.strftime("%H:%M"),
    )
    return _DEFAULT_TIME


def _parse_workspaces(raw: str | None) -> tuple[Path, ...]:
    """Parse the comma-separated workspace list, keeping only existing dirs.

    A configured root that isn't an existing directory is dropped with a warning
    rather than silently created — creating a project under a typo'd path would
    scaffold a bogus ``.cocoindex_code`` tree.
    """
    if not raw or not raw.strip():
        return ()
    roots: list[Path] = []
    seen: set[str] = set()
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        path = Path(candidate).resolve()
        key = str(path)
        if key in seen:
            continue
        if not path.is_dir():
            logger.warning("%s entry %r is not a directory; skipping", ENV_WORKSPACES, candidate)
            continue
        seen.add(key)
        roots.append(path)
    return tuple(roots)


def load_config() -> ScheduleConfig:
    """Build a :class:`ScheduleConfig` from the environment.

    Always returns a config (never ``None``); check ``.enabled`` to decide
    whether to start the loop.
    """
    enabled_raw = os.environ.get(ENV_ENABLED)
    enabled = not (enabled_raw is not None and _is_falsy(enabled_raw))

    git_raw = os.environ.get(ENV_GIT_PULL_ENABLED)
    git_pull_enabled = git_raw is not None and _is_truthy(git_raw)

    # Credentials are active only when a password/token is present; the username
    # is optional (some hosts accept any value for token auth). Don't strip the
    # password — a token is used verbatim.
    password = os.environ.get(ENV_GIT_PASSWORD) or ""
    git_credentials = (
        GitCredentials(username=(os.environ.get(ENV_GIT_USERNAME) or "").strip(), password=password)
        if password
        else None
    )

    return ScheduleConfig(
        enabled=enabled,
        run_time=_parse_time(os.environ.get(ENV_TIME)),
        workspaces=_parse_workspaces(os.environ.get(ENV_WORKSPACES)),
        git_pull_enabled=git_pull_enabled,
        git_credentials=git_credentials,
    )


def describe_config(config: ScheduleConfig) -> str:
    """One-line summary for the daemon startup log. Never includes the password."""
    workspaces = ", ".join(str(p) for p in config.workspaces) or "(loaded projects only)"
    return (
        f"time={config.run_time.strftime('%H:%M')} "
        f"git_pull={'on' if config.git_pull_enabled else 'off'} "
        f"git_creds={'set' if config.git_credentials else 'unset'} "
        f"workspaces=[{workspaces}]"
    )


# --- Timing -----------------------------------------------------------------


def seconds_until_next_run(now: datetime, run_time: time) -> float:
    """Seconds from *now* until the next local occurrence of *run_time*. Always >= 1.

    Local (not UTC) so the run time matches the operator's wall clock. Used by
    the daemon's scheduler, which caps a single sleep well below a day and
    re-checks the wall clock on each wake, so this need not be exact across DST /
    clock steps — it only paces the polling.
    """
    today_run = datetime.combine(now.date(), run_time)
    target = today_run if today_run > now else datetime.combine(
        now.date() + timedelta(days=1), run_time
    )
    return max(1.0, (target - now).total_seconds())


# --- Git update -------------------------------------------------------------

GitUpdateStatus = Literal["updated", "skipped", "error"]


@dataclass(frozen=True)
class GitUpdateResult:
    """Outcome of the git-update step for one project."""

    status: GitUpdateStatus
    message: str


def _credential_git_args(credentials: GitCredentials | None) -> list[str]:
    """The ``-c credential.helper=...`` args that inject *credentials*, if any."""
    if credentials is None:
        return []
    return ["-c", f"credential.helper={_CREDENTIAL_HELPER}"]


def _git_env(credentials: GitCredentials | None) -> dict[str, str]:
    # GIT_TERMINAL_PROMPT=0 makes git fail fast instead of blocking on a
    # credential prompt in the non-interactive daemon. When credentials are
    # configured, expose them under the names the inline helper reads.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if credentials is not None:
        env[_CRED_ENV_USERNAME] = credentials.username
        env[_CRED_ENV_PASSWORD] = credentials.password
    return env


def _run_git(
    root: Path, *args: str, credentials: GitCredentials | None = None
) -> subprocess.CompletedProcess[str]:
    # -c safe.directory=<root>: a bind-mounted repo is often owned by a
    # different UID than the daemon process (host user vs. the container's coco
    # user), which otherwise makes git refuse with "detected dubious ownership".
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            f"safe.directory={root}",
            *_credential_git_args(credentials),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        env=_git_env(credentials),
        check=False,
    )


def git_hard_reset_sync(
    root: Path, credentials: GitCredentials | None = None
) -> GitUpdateResult:
    """Fetch and hard-reset *root* to its upstream. Never raises.

    *credentials* (optional) are injected into the fetch via an inline credential
    helper for HTTPS remotes; SSH remotes ignore them and use their own auth.

    Returns a :class:`GitUpdateResult` describing what happened:

    * ``skipped`` — not a git repo, or git isn't installed,
    * ``error``   — fetch failed, no upstream configured, or reset failed,
    * ``updated`` — working tree now matches the upstream branch.
    """
    if not (root / ".git").exists():
        return GitUpdateResult("skipped", f"{root} is not a git repository")
    if shutil.which("git") is None:
        return GitUpdateResult("error", "git is not installed")

    try:
        # Only the fetch talks to the remote, so credentials are needed there.
        fetch = _run_git(root, "fetch", "--prune", "--quiet", credentials=credentials)
        if fetch.returncode != 0:
            return GitUpdateResult("error", f"git fetch failed: {fetch.stderr.strip()}")

        upstream = _run_git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream.returncode != 0:
            return GitUpdateResult(
                "error", "no upstream branch configured (git rev-parse @{u} failed)"
            )
        upstream_ref = upstream.stdout.strip()

        reset = _run_git(root, "reset", "--hard", upstream_ref, "--quiet")
        if reset.returncode != 0:
            return GitUpdateResult("error", f"git reset failed: {reset.stderr.strip()}")
    except subprocess.TimeoutExpired:
        return GitUpdateResult("error", f"git timed out after {_GIT_TIMEOUT_SECONDS}s")
    except OSError as e:
        return GitUpdateResult("error", f"git could not be run: {e}")

    head = _run_git(root, "rev-parse", "--short", "HEAD")
    sha = head.stdout.strip() if head.returncode == 0 else "?"
    return GitUpdateResult("updated", f"reset to {upstream_ref} @ {sha}")


# --- Connectivity check (for `ccc doctor`) ----------------------------------

# Mask any ``user:pass@`` / ``user@`` credentials embedded in a URL before it
# reaches logs or `ccc doctor` output.
_URL_CRED_RE = re.compile(r"(https?://)[^/@\s]+@")


def _sanitize(text: str) -> str:
    return _URL_CRED_RE.sub(r"\1***@", text)


@dataclass(frozen=True)
class GitCheckResult:
    """Outcome of the git connectivity check: human-readable details + an error."""

    details: list[str]
    error: str | None


def check_connection_sync(
    root: Path, credentials: GitCredentials | None = None
) -> GitCheckResult:
    """Probe that the workspace's git upstream is reachable (auth + network).

    Runs ``git ls-remote`` against the upstream's remote — a read-only probe that
    fetches nothing — so ``ccc doctor`` can flag a broken remote or bad
    credentials before the scheduled pull silently fails. Never raises; returns a
    :class:`GitCheckResult` whose ``error`` is ``None`` when the remote is
    reachable. Credentials (if any) are injected exactly as the pull does.
    """
    if not (root / ".git").exists():
        return GitCheckResult(
            details=[f"{root} is not a git repository"],
            error="git pull is enabled but the workspace is not a git repository",
        )
    if shutil.which("git") is None:
        return GitCheckResult(details=[], error="git is not installed")

    try:
        upstream = _run_git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream.returncode != 0:
            return GitCheckResult(
                details=[], error="no upstream branch configured (git rev-parse @{u} failed)"
            )
        upstream_ref = upstream.stdout.strip()
        remote_name = upstream_ref.split("/", 1)[0]

        url = _run_git(root, "remote", "get-url", remote_name)
        details = [f"Upstream: {upstream_ref}"]
        if url.returncode == 0 and url.stdout.strip():
            details.append(f"Remote: {remote_name} ({_sanitize(url.stdout.strip())})")
        details.append(f"Credentials: {'set' if credentials else 'from host'}")

        probe = _run_git(root, "ls-remote", "--quiet", remote_name, credentials=credentials)
        if probe.returncode != 0:
            return GitCheckResult(
                details=details,
                error=f"cannot reach {remote_name}: {_sanitize(probe.stderr.strip())}",
            )
    except subprocess.TimeoutExpired:
        return GitCheckResult(details=[], error=f"git timed out after {_GIT_TIMEOUT_SECONDS}s")
    except OSError as e:
        return GitCheckResult(details=[], error=f"git could not be run: {e}")

    return GitCheckResult(details=details, error=None)
