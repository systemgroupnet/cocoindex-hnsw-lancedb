"""Push index statistics to MySQL for Apache DevLake dashboards.

Optional and off by default. When a MySQL target is configured through the
environment, the daemon writes a stats snapshot after every index pass:

* one row of repo totals (chunks / files / LoC) in ``ccc_repo_stats``, and
* one row per language (chunks / LoC) in ``ccc_language_stats``,

both stamped with the same UTC ``collected_at`` timestamp and a shared
``snapshot_id`` so DevLake (Grafana on top of MySQL) can build time-series and
per-snapshot breakdowns. Run ``init-stat-table-script.sql`` once to create the
tables.

Configuration (all via environment variables):

* ``COCOINDEX_CODE_METRICS_MYSQL_HOST``     — target host (enables the feature)
* ``COCOINDEX_CODE_METRICS_MYSQL_DATABASE`` — target database (also required)
* ``COCOINDEX_CODE_METRICS_MYSQL_PORT``     — default 3306
* ``COCOINDEX_CODE_METRICS_MYSQL_USER``     — default ``root``
* ``COCOINDEX_CODE_METRICS_MYSQL_PASSWORD`` — default empty
* ``COCOINDEX_CODE_METRICS_REPO``           — override the repo identifier
  (defaults to the project's host path; set this in single-repo deployments)
* ``COCOINDEX_CODE_METRICS_ENABLED``        — set to a falsy value
  (``0``/``false``/``no``/``off``) to disable even when a target is configured

The feature is inactive unless a host *and* database are set, so an install
that never opts in does no work and prints no warnings. Everything here is
best-effort: a misconfigured or unreachable database logs a warning and is
skipped — it never interrupts indexing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import ProjectStatusResponse

logger = logging.getLogger(__name__)

# --- Environment knobs (single source of truth) ----------------------------

ENV_ENABLED = "COCOINDEX_CODE_METRICS_ENABLED"
ENV_HOST = "COCOINDEX_CODE_METRICS_MYSQL_HOST"
ENV_PORT = "COCOINDEX_CODE_METRICS_MYSQL_PORT"
ENV_USER = "COCOINDEX_CODE_METRICS_MYSQL_USER"
ENV_PASSWORD = "COCOINDEX_CODE_METRICS_MYSQL_PASSWORD"
ENV_DATABASE = "COCOINDEX_CODE_METRICS_MYSQL_DATABASE"
ENV_REPO = "COCOINDEX_CODE_METRICS_REPO"

# Table names — must match init-stat-table-script.sql.
REPO_STATS_TABLE = "ccc_repo_stats"
LANGUAGE_STATS_TABLE = "ccc_language_stats"

_DEFAULT_PORT = 3306
_DEFAULT_USER = "root"
_CONNECT_TIMEOUT_SECONDS = 10

# Values that count as "off" for COCOINDEX_CODE_METRICS_ENABLED.
_FALSY = {"0", "false", "no", "off"}

# Set once we've warned about enabled-but-unconfigured, so an actively-indexing
# daemon doesn't repeat the same warning on every pass.
_warned_unconfigured = False


@dataclass(frozen=True)
class MetricsConfig:
    """Resolved MySQL target for the metrics push."""

    host: str
    port: int
    user: str
    password: str
    database: str
    repo_override: str | None


# --- Config -----------------------------------------------------------------


def _is_falsy(raw: str) -> bool:
    return raw.strip().lower() in _FALSY


def _parse_port(raw: str | None) -> int:
    if not raw or not raw.strip():
        return _DEFAULT_PORT
    try:
        port = int(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", ENV_PORT, raw, _DEFAULT_PORT)
        return _DEFAULT_PORT
    if not (0 < port < 65536):
        logger.warning("%s=%r out of range; using %d", ENV_PORT, raw, _DEFAULT_PORT)
        return _DEFAULT_PORT
    return port


def load_config() -> MetricsConfig | None:
    """Build a :class:`MetricsConfig` from the environment.

    Returns ``None`` when metrics is disabled — either explicitly
    (``COCOINDEX_CODE_METRICS_ENABLED`` is falsy) or implicitly (no MySQL host
    and database configured, so there is nothing to push to).
    """
    global _warned_unconfigured  # noqa: PLW0603

    enabled_raw = os.environ.get(ENV_ENABLED)
    explicitly_disabled = enabled_raw is not None and _is_falsy(enabled_raw)
    if explicitly_disabled:
        return None

    host = (os.environ.get(ENV_HOST) or "").strip()
    database = (os.environ.get(ENV_DATABASE) or "").strip()
    if not host or not database:
        # An operator who set ENABLED=true but forgot the target gets one
        # warning; a plain install that never opted in stays silent.
        if enabled_raw is not None and not _warned_unconfigured:
            logger.warning(
                "%s is set but %s / %s are not — metrics push is inactive",
                ENV_ENABLED,
                ENV_HOST,
                ENV_DATABASE,
            )
            _warned_unconfigured = True
        return None

    return MetricsConfig(
        host=host,
        port=_parse_port(os.environ.get(ENV_PORT)),
        user=(os.environ.get(ENV_USER) or _DEFAULT_USER).strip(),
        password=os.environ.get(ENV_PASSWORD) or "",
        database=database,
        repo_override=(os.environ.get(ENV_REPO) or "").strip() or None,
    )


def describe_config(config: MetricsConfig) -> str:
    """One-line ``user@host:port/database`` summary for ``ccc doctor``."""
    return f"{config.user}@{config.host}:{config.port}/{config.database}"


def resolve_repo(config: MetricsConfig, default_repo: str) -> str:
    """The repo identifier stored in both tables — the override if set."""
    return config.repo_override or default_repo


# --- Connection -------------------------------------------------------------

# A factory that opens a DB-API connection for a config. Injected in tests.
Connect = Callable[[MetricsConfig], Any]


class MetricsDriverMissing(RuntimeError):
    """Raised when metrics is configured but the MySQL driver isn't installed.

    Kept distinct from connection errors so callers can surface an actionable
    "install the driver" message instead of a misleading "cannot connect".
    """


_DRIVER_HINT = (
    "the MySQL driver (PyMySQL) is not installed. Install it with "
    "`pip install 'cocoindex-code[metrics]'` (or the `[full]` extra / a `:full` "
    "Docker image), then restart the daemon (`ccc daemon restart`)"
)


def _connect_default(config: MetricsConfig) -> Any:
    # PyMySQL is an optional dependency (the ``metrics`` extra); import lazily so
    # a base install that never enables metrics doesn't need it.
    try:
        import pymysql
    except ImportError as e:
        raise MetricsDriverMissing(_DRIVER_HINT) from e

    return pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        autocommit=False,
    )


# --- Write ------------------------------------------------------------------


def _write_snapshot(
    conn: Any,
    repo: str,
    status: ProjectStatusResponse,
    collected_at: datetime,
) -> str:
    """Insert one repo row plus per-language rows in a single transaction.

    Returns the ``snapshot_id`` correlating the rows. Raises on any DB error
    (the caller turns that into a logged, non-fatal skip).
    """
    snapshot_id = uuid.uuid4().hex
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"INSERT INTO {REPO_STATS_TABLE} "
            "(snapshot_id, repo, collected_at, total_chunks, total_files, total_loc) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                snapshot_id,
                repo,
                collected_at,
                status.total_chunks,
                status.total_files,
                status.total_loc,
            ),
        )
        if status.languages:
            cursor.executemany(
                f"INSERT INTO {LANGUAGE_STATS_TABLE} "
                "(snapshot_id, repo, collected_at, language, chunks, loc) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (snapshot_id, repo, collected_at, lang, stats.chunks, stats.loc)
                    for lang, stats in sorted(status.languages.items())
                ],
            )
        conn.commit()
    finally:
        cursor.close()
    return snapshot_id


def _utc_now() -> datetime:
    # Naive UTC — MySQL DATETIME has no timezone; storing UTC keeps DevLake
    # time-series consistent regardless of the daemon host's local zone.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def seconds_until_next_midnight(now: datetime) -> float:
    """Seconds from *now* until the next local 00:00. Always >= 1.

    Used by the daemon's daily-push scheduler. Local (not UTC) so "midnight"
    matches the operator's wall clock; the stored ``collected_at`` stays UTC.
    """
    next_day = (now + timedelta(days=1)).date()
    next_midnight = datetime.combine(next_day, datetime.min.time())
    return max(1.0, (next_midnight - now).total_seconds())


def push_snapshot_sync(
    default_repo: str,
    status: ProjectStatusResponse,
    *,
    config: MetricsConfig,
    connect: Connect = _connect_default,
) -> str:
    """Write one stats snapshot and return its ``snapshot_id``.

    Raises on any failure — :class:`MetricsDriverMissing` when the driver is
    absent, or the driver's own exception when the DB is unreachable / the write
    fails. Used by the on-demand ``ccc push-metrics`` path, which surfaces the
    error to the user. Callers ensure the index exists first.
    """
    repo = resolve_repo(config, default_repo)
    collected_at = _utc_now()
    conn = connect(config)
    try:
        snapshot_id = _write_snapshot(conn, repo, status, collected_at)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    logger.info(
        "Pushed metrics snapshot %s for %s (chunks=%d, files=%d, loc=%d, langs=%d)",
        snapshot_id,
        repo,
        status.total_chunks,
        status.total_files,
        status.total_loc,
        len(status.languages),
    )
    return snapshot_id


def push_status_sync(
    default_repo: str,
    status: ProjectStatusResponse,
    *,
    config: MetricsConfig | None = None,
    connect: Connect = _connect_default,
) -> bool:
    """Push one stats snapshot to MySQL. Returns True iff a row was written.

    Best-effort: returns False (after logging) when metrics is disabled, the
    index is empty, the database is unreachable, or the write fails. Never
    raises — callers on the indexing path must not be interrupted.
    """
    cfg = config if config is not None else load_config()
    if cfg is None:
        return False
    if not status.index_exists:
        return False
    try:
        push_snapshot_sync(default_repo, status, config=cfg, connect=connect)
    except MetricsDriverMissing as e:
        logger.warning("Metrics push skipped: %s", e)
        return False
    except Exception:
        logger.warning(
            "Metrics push skipped: cannot reach MySQL at %s:%d or write failed",
            cfg.host,
            cfg.port,
            exc_info=True,
        )
        return False
    return True


async def push_status(
    default_repo: str,
    status: ProjectStatusResponse,
    *,
    config: MetricsConfig | None = None,
    connect: Connect = _connect_default,
) -> bool:
    """Async wrapper around :func:`push_status_sync` (runs the blocking DB I/O
    in a worker thread so it never stalls the daemon event loop)."""
    return await asyncio.to_thread(
        push_status_sync, default_repo, status, config=config, connect=connect
    )


async def push_snapshot(
    default_repo: str,
    status: ProjectStatusResponse,
    *,
    config: MetricsConfig,
    connect: Connect = _connect_default,
) -> str:
    """Async wrapper around :func:`push_snapshot_sync` (raising, for on-demand
    pushes) — runs the blocking DB I/O off the event loop."""
    return await asyncio.to_thread(
        push_snapshot_sync, default_repo, status, config=config, connect=connect
    )


# --- Doctor -----------------------------------------------------------------


def check_connection_sync(config: MetricsConfig, connect: Connect = _connect_default) -> str | None:
    """Open and immediately close a connection. Returns an actionable error
    string on failure, or ``None`` when the target is reachable.

    A missing driver and an unreachable database are distinct messages so
    ``ccc doctor`` doesn't report "cannot connect" when the real fix is to
    install the driver.
    """
    try:
        conn = connect(config)
    except MetricsDriverMissing as e:
        return str(e)
    except Exception as e:
        return f"cannot connect to {describe_config(config)}: {e}"
    try:
        conn.close()
    except Exception:
        pass
    return None
