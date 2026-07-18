"""Runtime memory awareness.

The daemon indexes with a large fan-out (the CocoIndex engine keeps up to
``max_inflight_components`` files in flight, each holding file text + chunks +
embedding arrays + pyarrow write buffers). With no notion of how much RAM it
actually has, the process scales to a host-sized workload and gets OOM-killed
inside a memory-capped container.

This module gives it that notion:

* :func:`detect_memory_limit_bytes` reads the *real* limit — the cgroup cap
  when containerized (``psutil``/``/proc/meminfo`` report the host total, not
  the container's slice), with an explicit env override on top.
* :class:`MemoryGovernor` turns the limit into a static concurrency cap
  (:attr:`~MemoryGovernor.max_inflight`) and, while indexing runs, throttles an
  adaptive gate as live usage approaches the limit — shrinking concurrency
  under pressure and recovering when it eases.

Everything degrades gracefully off Linux / without a cgroup: detection returns
``None`` and the governor falls back to the engine default with no runtime
throttling, so behavior on a dev machine is unchanged.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Configuration knobs (single source of truth) --------------------------

# Explicit overrides. The limit override wins over cgroup detection entirely;
# the inflight override pins the static concurrency cap (bypassing the budget
# heuristic) for operators who want a fixed value.
ENV_MEMORY_LIMIT_MB = "COCOINDEX_CODE_MEMORY_LIMIT_MB"
ENV_MAX_INFLIGHT_FILES = "COCOINDEX_CODE_MAX_INFLIGHT_FILES"

# Mirror of the CocoIndex engine default (``_DEFAULT_MAX_INFLIGHT_COMPONENTS``).
# Used as the concurrency ceiling when memory is unknown or plentiful, so we
# never make the app *slower* than upstream's default on an unconstrained host.
ENGINE_DEFAULT_MAX_INFLIGHT = 1024

# Rough resident cost of one in-flight file: its text, the overlapping chunk
# copies, one float32 embedding per chunk, and the pyarrow buffers staged for
# the merge_insert. A deliberately conservative estimate — the runtime monitor
# corrects for reality, this only sizes the initial static cap.
PER_FILE_COST_BYTES = 8 * 1024 * 1024

# Fraction of the limit kept permanently free (kernel page cache, allocator
# fragmentation, transient native spikes we can't see coming).
SAFETY_MARGIN_FRACTION = 0.15

# Never throttle below this — a floor keeps indexing making progress even under
# sustained pressure (better slow than stalled).
MIN_INFLIGHT = 4

# Live-usage pressure thresholds, as a fraction of the limit.
SOFT_PRESSURE = 0.80  # ease off: shed one slot per tick
HARD_PRESSURE = 0.92  # emergency: gc + halve the gate
RECOVER_BELOW = 0.68  # grow back one slot per tick when comfortably under

MONITOR_INTERVAL_SECONDS = 1.0

# cgroup control-file locations.
_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V1_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CGROUP_V1_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
_PROC_MEMINFO = Path("/proc/meminfo")
_PROC_STATM = Path("/proc/self/statm")

# cgroup v1 encodes "no limit" as a near-``i64`` sentinel (page-aligned). Any
# value at or above this is treated as unlimited rather than a real cap.
_CGROUP_V1_UNLIMITED_MIN = 1 << 62


# --- Low-level reads (all return None when unavailable) ---------------------


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except (OSError, ValueError):
        return None
    if not text or text == "max":  # cgroup v2 uses the literal "max"
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _host_total_bytes() -> int | None:
    """Total physical RAM from ``/proc/meminfo`` (Linux only)."""
    try:
        for line in _PROC_MEMINFO.read_text().splitlines():
            if line.startswith("MemTotal:"):
                # "MemTotal:   16307128 kB"
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _page_size() -> int:
    sysconf = getattr(os, "sysconf", None)  # absent on Windows
    if sysconf is None:
        return 4096
    try:
        size = int(sysconf("SC_PAGE_SIZE"))
        return size if size > 0 else 4096
    except (ValueError, OSError):
        return 4096


def _env_limit_bytes() -> int | None:
    raw = os.environ.get(ENV_MEMORY_LIMIT_MB)
    if not raw or not raw.strip():
        return None
    try:
        mb = int(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring", ENV_MEMORY_LIMIT_MB, raw)
        return None
    if mb <= 0:
        return None
    return mb * 1024 * 1024


def detect_memory_limit_bytes() -> tuple[int | None, str]:
    """Return ``(limit_bytes, source)`` — the memory ceiling this process must
    stay under, and where it came from (for logging / ``ccc doctor``).

    Precedence: explicit env override → cgroup v2 → cgroup v1 → host total.
    A cgroup cap is intersected with the host total (a cap larger than physical
    RAM is meaningless). Returns ``(None, "undetected")`` when nothing is
    readable (e.g. Windows/macOS dev machines), which callers treat as
    "unconstrained, no runtime throttling".
    """
    override = _env_limit_bytes()
    if override is not None:
        return override, f"env {ENV_MEMORY_LIMIT_MB}"

    host = _host_total_bytes()

    v2 = _read_int(_CGROUP_V2_MAX)
    if v2 is not None:
        return (min(v2, host) if host else v2), "cgroup v2"

    v1 = _read_int(_CGROUP_V1_LIMIT)
    if v1 is not None and v1 < _CGROUP_V1_UNLIMITED_MIN:
        return (min(v1, host) if host else v1), "cgroup v1"

    if host is not None:
        return host, "host total"

    return None, "undetected"


def current_usage_bytes() -> int | None:
    """Current resident memory of this process's cgroup (preferred) or the
    process itself. ``None`` when nothing is readable."""
    v2 = _read_int(_CGROUP_V2_CURRENT)
    if v2 is not None:
        return v2
    v1 = _read_int(_CGROUP_V1_USAGE)
    if v1 is not None:
        return v1
    # Fall back to this process's RSS from /proc/self/statm (field 2 = resident
    # pages). Coarser than the cgroup (excludes child processes) but enough to
    # drive throttling on a cgroup-less host.
    try:
        fields = _PROC_STATM.read_text().split()
        return int(fields[1]) * _page_size()
    except (OSError, ValueError, IndexError):
        return None


def format_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


# --- Budget sizing ----------------------------------------------------------


def recommend_max_inflight(
    limit_bytes: int | None,
    baseline_bytes: int | None,
    ceiling: int,
) -> int:
    """Static concurrency cap that fits the memory budget.

    ``ceiling`` bounds the result (and is returned as-is when the limit is
    unknown). ``baseline_bytes`` is the resident footprint at idle — model
    weights, torch, the interpreter — which is subtracted before dividing the
    remaining budget by :data:`PER_FILE_COST_BYTES`.
    """
    if limit_bytes is None:
        return ceiling
    work = limit_bytes * (1.0 - SAFETY_MARGIN_FRACTION) - (baseline_bytes or 0)
    n = int(work // PER_FILE_COST_BYTES)
    # Floor at MIN_INFLIGHT, but never above the ceiling (a user-pinned ceiling
    # below MIN_INFLIGHT still wins).
    floor = min(MIN_INFLIGHT, ceiling)
    return max(floor, min(ceiling, n))


def resolve_ceiling() -> int:
    """Concurrency ceiling: the ``COCOINDEX_CODE_MAX_INFLIGHT_FILES`` override
    if set (pins the cap, bypassing the budget heuristic), else the engine
    default."""
    raw = os.environ.get(ENV_MAX_INFLIGHT_FILES)
    if raw and raw.strip():
        try:
            value = int(raw.strip())
            if value >= 1:
                return value
            logger.warning("%s=%r must be >= 1; ignoring", ENV_MAX_INFLIGHT_FILES, raw)
        except ValueError:
            logger.warning("%s=%r is not an integer; ignoring", ENV_MAX_INFLIGHT_FILES, raw)
    return ENGINE_DEFAULT_MAX_INFLIGHT


# --- Status snapshot (for doctor / status surfaces) -------------------------


@dataclass(frozen=True)
class MemoryStatus:
    limit_bytes: int | None
    source: str
    baseline_bytes: int | None
    current_bytes: int | None
    peak_bytes: int | None
    max_inflight: int
    current_capacity: int
    throttle_events: int

    @property
    def usage_fraction(self) -> float | None:
        if self.limit_bytes and self.current_bytes is not None:
            return self.current_bytes / self.limit_bytes
        return None


# --- The governor -----------------------------------------------------------


class MemoryGovernor:
    """Sizes indexing concurrency to the memory limit and throttles it live.

    One instance per daemon (memory is a process-global resource). Wire-up:

    #. Construct with the detected limit.
    #. :meth:`calibrate` once the embedding model(s) are loaded — records the
       idle footprint and computes :attr:`max_inflight`.
    #. Pass :attr:`max_inflight` to ``coco.AppConfig(max_inflight_components=)``
       so the engine's fan-out matches the budget.
    #. Guard each in-flight file with :meth:`slot` inside ``process_file``.
    #. :meth:`start_monitor` on the running event loop so live pressure shrinks
       and grows the gate.

    The gate's capacity only ever moves *between* :data:`MIN_INFLIGHT` and
    :attr:`max_inflight`; the static cap is the ceiling, pressure is the floor.
    """

    def __init__(self, limit_bytes: int | None, source: str, ceiling: int) -> None:
        self._limit = limit_bytes
        self._source = source
        self._ceiling = ceiling
        self._baseline: int | None = None
        self._max_inflight = ceiling
        self._capacity = ceiling
        self._in_use = 0
        # Loop-bound; created lazily on first use inside the daemon event loop
        # (mirrors PacedLiteLLMEmbedder's lazy lock) so construction off-loop is
        # safe.
        self._cond: asyncio.Condition | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._peak_usage = 0
        self._throttle_events = 0

    # -- setup --

    def calibrate(self) -> None:
        """Record idle resident memory and derive the static concurrency cap.

        Call after the embedders are constructed and before indexing starts, so
        the baseline captures model weights (loaded twice for ST) + torch + the
        interpreter — everything resident before the fan-out allocates anything.
        """
        self._baseline = current_usage_bytes()
        self._max_inflight = recommend_max_inflight(self._limit, self._baseline, self._ceiling)
        self._capacity = self._max_inflight
        logger.info(
            "Memory budget: limit=%s (%s), baseline=%s, max_inflight_files=%d",
            format_bytes(self._limit),
            self._source,
            format_bytes(self._baseline),
            self._max_inflight,
        )

    @property
    def max_inflight(self) -> int:
        return self._max_inflight

    @property
    def limit_bytes(self) -> int | None:
        return self._limit

    def _get_cond(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    # -- the gate --

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire one in-flight-file permit for the duration of the block.

        Blocks while the gate is full; capacity is what the monitor has set it
        to, so under memory pressure new files wait here instead of piling on.
        """
        cond = self._get_cond()
        async with cond:
            await cond.wait_for(lambda: self._in_use < self._capacity)
            self._in_use += 1
        try:
            yield
        finally:
            async with cond:
                self._in_use -= 1
                cond.notify_all()

    async def _set_capacity(self, new_capacity: int) -> None:
        floor = min(MIN_INFLIGHT, self._max_inflight)
        new_capacity = max(floor, min(self._max_inflight, new_capacity))
        if new_capacity == self._capacity:
            return
        cond = self._get_cond()
        async with cond:
            grew = new_capacity > self._capacity
            self._capacity = new_capacity
            if grew:
                cond.notify_all()
        logger.debug("Memory gate capacity -> %d", new_capacity)

    # -- runtime monitor --

    def _can_monitor(self) -> bool:
        return self._limit is not None and current_usage_bytes() is not None

    def start_monitor(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the background pressure monitor on *loop* (no-op when live
        usage or the limit can't be read — static cap only)."""
        if self._monitor_task is not None:
            return
        if not self._can_monitor():
            logger.info("Memory monitor disabled (limit or live usage unreadable)")
            return
        self._monitor_task = loop.create_task(self._monitor_loop())

    async def _monitor_loop(self) -> None:
        assert self._limit is not None
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            usage = current_usage_bytes()
            if usage is None:
                continue
            self._peak_usage = max(self._peak_usage, usage)
            fraction = usage / self._limit
            if fraction >= HARD_PRESSURE:
                self._throttle_events += 1
                gc.collect()
                logger.warning(
                    "Memory pressure %.0f%% (%s/%s) — halving indexing concurrency",
                    fraction * 100,
                    format_bytes(usage),
                    format_bytes(self._limit),
                    )
                await self._set_capacity(self._capacity // 2)
            elif fraction >= SOFT_PRESSURE:
                await self._set_capacity(self._capacity - 1)
            elif fraction < RECOVER_BELOW and self._capacity < self._max_inflight:
                await self._set_capacity(self._capacity + 1)

    def stop_monitor(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None

    # -- reporting --

    def snapshot(self) -> MemoryStatus:
        return MemoryStatus(
            limit_bytes=self._limit,
            source=self._source,
            baseline_bytes=self._baseline,
            current_bytes=current_usage_bytes(),
            peak_bytes=self._peak_usage or None,
            max_inflight=self._max_inflight,
            current_capacity=self._capacity,
            throttle_events=self._throttle_events,
        )
