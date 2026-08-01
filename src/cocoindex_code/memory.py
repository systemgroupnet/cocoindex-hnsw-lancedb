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
* :func:`recommend_scan_budget` does the same job for text scans (the ``rg``
  subprocesses behind ``ccc grep`` / the ``ripgrep`` MCP tool). Those need no
  index, but they aren't free: each forks a process, and a *branch* scan reads
  the branch's changed files out of git into memory. The budget bounds how many
  scans run at once and how much each may hold.

Everything degrades gracefully off Linux / without a cgroup: detection returns
``None`` and the governor falls back to the engine default with no runtime
throttling, so behavior on a dev machine is unchanged.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Configuration knobs (single source of truth) --------------------------

# Explicit overrides. The limit override wins over cgroup detection entirely;
# the inflight and scan overrides pin the two static concurrency caps
# (bypassing the budget heuristics) for operators who want a fixed value.
ENV_MEMORY_LIMIT_MB = "COCOINDEX_CODE_MEMORY_LIMIT_MB"
ENV_MAX_INFLIGHT_FILES = "COCOINDEX_CODE_MAX_INFLIGHT_FILES"
ENV_MAX_CONCURRENT_SCANS = "COCOINDEX_CODE_MAX_CONCURRENT_SCANS"

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

# --- Text-scan (ripgrep) bounds --------------------------------------------
#
# Two of these three are *policy*, not memory arithmetic — deliberately, so
# they don't pretend to a precision no one has measured. Only the blob batch is
# derived from the limit.

# Worker-pool size: how many rg processes may run at once. A concurrency
# policy, the same kind of number as a connection pool — not an estimate of
# what a scan costs. Overridable via ENV_MAX_CONCURRENT_SCANS.
DEFAULT_MAX_CONCURRENT_SCANS = 4

# Code search doesn't read files this large (editors draw the same line — VS
# Code stops at 20 MB). Doubles as the bound on what rg buffers for one file
# and what a branch scan will pull out of git.
MAX_SCAN_FILESIZE_BYTES = 16 * 1024 * 1024

# Peak branch-blob text one scan may hold: the branch's changed files are read
# in batches that stay under this, so a branch that rewrote 5,000 files costs
# one batch of resident text instead of all 5,000 at once. This is the one
# scan bound that scales with the memory limit — a share of the same headroom
# indexing draws from, split across the scan pool. The share is a judgement
# call; what matters is that a bound exists and is enforced by the batching.
SCAN_SHARE_FRACTION = 0.25
MIN_BLOB_BATCH_BYTES = 8 * 1024 * 1024
MAX_BLOB_BATCH_BYTES = 256 * 1024 * 1024
DEFAULT_BLOB_BATCH_BYTES = 64 * 1024 * 1024

# A scan that waits this long for a permit is logged and counted. Queueing is
# the intended behavior under load — never rejection — but an unobservable
# queue is indistinguishable from a hang, and the operator needs to know
# whether to raise the scan pool.
SCAN_QUEUE_WARN_SECONDS = 2.0

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


def _env_positive_int(name: str) -> int | None:
    """Value of env var *name* when it's an integer >= 1, else ``None``."""
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring", name, raw)
        return None
    if value < 1:
        logger.warning("%s=%r must be >= 1; ignoring", name, raw)
        return None
    return value


def resolve_ceiling() -> int:
    """Concurrency ceiling: the ``COCOINDEX_CODE_MAX_INFLIGHT_FILES`` override
    if set (pins the cap, bypassing the budget heuristic), else the engine
    default."""
    return _env_positive_int(ENV_MAX_INFLIGHT_FILES) or ENGINE_DEFAULT_MAX_INFLIGHT


def resolve_scan_concurrency() -> int | None:
    """The ``COCOINDEX_CODE_MAX_CONCURRENT_SCANS`` override, or ``None`` to let
    :func:`recommend_scan_budget` size it from the memory budget."""
    return _env_positive_int(ENV_MAX_CONCURRENT_SCANS)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ScanBudget:
    """Resource bounds for text scans.

    One instance per daemon, computed at calibration. ``max_concurrent`` is
    enforced by :meth:`MemoryGovernor.scan_slot`; the rest is handed to
    :mod:`cocoindex_code.ripgrep`, which turns them into rg flags and its
    branch-blob batching.
    """

    # Simultaneous rg scans allowed across all projects.
    max_concurrent: int
    # Peak branch-blob text one scan may hold (and materialize) at a time.
    blob_batch_bytes: int
    # Files above this are not searched (rg ``--max-filesize``).
    max_filesize_bytes: int


def recommend_scan_budget(
    limit_bytes: int | None,
    baseline_bytes: int | None,
    *,
    concurrency_override: int | None = None,
) -> ScanBudget:
    """Bounds for text scans in a process capped at *limit_bytes*.

    Concurrency and the file-size cut are fixed policy (see the constants
    above). Only the blob batch is sized: scans share
    :data:`SCAN_SHARE_FRACTION` of the headroom left after the idle baseline
    and the safety margin, split across the pool. With no limit detected the
    batch falls back to :data:`DEFAULT_BLOB_BATCH_BYTES` — still bounded, just
    not sized to anything.
    """
    concurrent = concurrency_override or DEFAULT_MAX_CONCURRENT_SCANS
    if limit_bytes is None:
        blob_batch = DEFAULT_BLOB_BATCH_BYTES
    else:
        work = limit_bytes * (1.0 - SAFETY_MARGIN_FRACTION) - (baseline_bytes or 0)
        share = max(0.0, work) * SCAN_SHARE_FRACTION / concurrent
        blob_batch = _clamp(int(share), MIN_BLOB_BATCH_BYTES, MAX_BLOB_BATCH_BYTES)
    return ScanBudget(
        max_concurrent=concurrent,
        blob_batch_bytes=blob_batch,
        max_filesize_bytes=MAX_SCAN_FILESIZE_BYTES,
    )


# Bounds used when no governor is in play (standalone helpers, tests).
DEFAULT_SCAN_BUDGET = recommend_scan_budget(None, None)


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
    scan_budget: ScanBudget
    current_scan_capacity: int
    # Scan queue: what's running, what's waiting, and whether waits have been
    # long enough to matter. `delayed_scans` counts waits past
    # SCAN_QUEUE_WARN_SECONDS — none of them are rejections.
    scans_running: int
    scans_queued: int
    peak_scans_queued: int
    delayed_scans: int
    max_scan_wait_seconds: float

    @property
    def usage_fraction(self) -> float | None:
        if self.limit_bytes and self.current_bytes is not None:
            return self.current_bytes / self.limit_bytes
        return None


# --- The governor -----------------------------------------------------------


class _Gate:
    """A concurrency gate whose capacity can move at runtime.

    Permits are handed out up to :attr:`capacity`, which the pressure monitor
    shrinks and grows between :attr:`floor` and :attr:`maximum`.
    """

    def __init__(self, maximum: int, floor: int) -> None:
        self._maximum = maximum
        self._floor = min(floor, maximum)
        self._capacity = maximum
        self._in_use = 0
        self._waiting = 0
        self._peak_waiting = 0
        # Loop-bound; created lazily on first use inside the daemon event loop
        # (mirrors PacedLiteLLMEmbedder's lazy lock) so construction off-loop is
        # safe.
        self._cond: asyncio.Condition | None = None

    def resize(self, maximum: int, floor: int) -> None:
        """Set a new maximum (at calibration, before any permit is held)."""
        self._maximum = maximum
        self._floor = min(floor, maximum)
        self._capacity = maximum

    @property
    def maximum(self) -> int:
        return self._maximum

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        """Permits held right now."""
        return self._in_use

    @property
    def waiting(self) -> int:
        """Callers queued for a permit right now."""
        return self._waiting

    @property
    def peak_waiting(self) -> int:
        """Deepest the queue has been since startup."""
        return self._peak_waiting

    def _get_cond(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one permit for the duration of the block.

        A caller that has to wait is counted while it waits, so queue depth is
        observable — a full gate and a hung scan look identical from outside
        otherwise.
        """
        cond = self._get_cond()
        async with cond:
            if self._in_use >= self._capacity:
                self._waiting += 1
                self._peak_waiting = max(self._peak_waiting, self._waiting)
                try:
                    await cond.wait_for(lambda: self._in_use < self._capacity)
                finally:
                    # `wait_for` re-acquires the lock before returning *or*
                    # raising, so this stays consistent under cancellation.
                    self._waiting -= 1
            self._in_use += 1
        try:
            yield
        finally:
            async with cond:
                self._in_use -= 1
                cond.notify_all()

    async def set_capacity(self, new_capacity: int) -> None:
        new_capacity = max(self._floor, min(self._maximum, new_capacity))
        if new_capacity == self._capacity:
            return
        cond = self._get_cond()
        async with cond:
            grew = new_capacity > self._capacity
            self._capacity = new_capacity
            if grew:
                cond.notify_all()


class MemoryGovernor:
    """Sizes work to the memory limit and throttles it live.

    Two kinds of work are gated, because both scale with request volume and
    both are charged to the same cgroup: indexing (the engine's in-flight file
    fan-out) and text scans (rg subprocesses behind grep). One instance per
    daemon — memory is a process-global resource. Wire-up:

    #. Construct with the detected limit.
    #. :meth:`calibrate` once the embedding model(s) are loaded — records the
       idle footprint and computes :attr:`max_inflight` and :attr:`scan_budget`.
    #. Pass :attr:`max_inflight` to ``coco.AppConfig(max_inflight_components=)``
       so the engine's fan-out matches the budget.
    #. Guard each in-flight file with :meth:`slot` inside ``process_file``, and
       each rg scan with :meth:`scan_slot`.
    #. :meth:`start_monitor` on the running event loop so live pressure shrinks
       and grows the gates.

    A gate's capacity only ever moves *between* its floor (:data:`MIN_INFLIGHT`
    for indexing, 1 for scans) and its calibrated maximum; the static cap is the
    ceiling, pressure is the floor.
    """

    def __init__(self, limit_bytes: int | None, source: str, ceiling: int) -> None:
        self._limit = limit_bytes
        self._source = source
        self._ceiling = ceiling
        self._baseline: int | None = None
        self._index_gate = _Gate(ceiling, MIN_INFLIGHT)
        self._scan_budget = DEFAULT_SCAN_BUDGET
        self._scan_gate = _Gate(self._scan_budget.max_concurrent, 1)
        self._monitor_task: asyncio.Task[None] | None = None
        self._peak_usage = 0
        self._throttle_events = 0
        self._delayed_scans = 0
        self._max_scan_wait = 0.0

    # -- setup --

    def calibrate(self) -> None:
        """Record idle resident memory and derive the static budgets.

        Call after the embedders are constructed and before indexing starts, so
        the baseline captures model weights (loaded twice for ST) + torch + the
        interpreter — everything resident before the fan-out allocates anything.
        """
        self._baseline = current_usage_bytes()
        self._index_gate.resize(
            recommend_max_inflight(self._limit, self._baseline, self._ceiling), MIN_INFLIGHT
        )
        self._scan_budget = recommend_scan_budget(
            self._limit, self._baseline, concurrency_override=resolve_scan_concurrency()
        )
        self._scan_gate.resize(self._scan_budget.max_concurrent, 1)
        logger.info(
            "Memory budget: limit=%s (%s), baseline=%s, max_inflight_files=%d, "
            "max_concurrent_scans=%d (blob batch %s)",
            format_bytes(self._limit),
            self._source,
            format_bytes(self._baseline),
            self._index_gate.maximum,
            self._scan_budget.max_concurrent,
            format_bytes(self._scan_budget.blob_batch_bytes),
        )

    @property
    def max_inflight(self) -> int:
        return self._index_gate.maximum

    @property
    def limit_bytes(self) -> int | None:
        return self._limit

    @property
    def scan_budget(self) -> ScanBudget:
        """Per-scan resource bounds to hand to :mod:`cocoindex_code.ripgrep`."""
        return self._scan_budget

    # -- the gates --

    def slot(self) -> AbstractAsyncContextManager[None]:
        """Acquire one in-flight-file permit for the duration of the block.

        Blocks while the gate is full; capacity is what the monitor has set it
        to, so under memory pressure new files wait here instead of piling on.
        """
        return self._index_gate.slot()

    @asynccontextmanager
    async def scan_slot(self) -> AsyncIterator[None]:
        """Acquire one text-scan permit for the duration of the block.

        Bounds how many rg processes (and, for a branch scan, blob batches) are
        resident at once, so a burst of greps can't outrun the memory budget the
        way an ungated fan-out of subprocesses would.

        Waiting callers are never rejected — they queue here until a permit
        frees. Because that makes a saturated gate indistinguishable from a hung
        scan, a wait past :data:`SCAN_QUEUE_WARN_SECONDS` is logged and counted
        for ``ccc doctor``.
        """
        started = time.monotonic()
        async with self._scan_gate.slot():
            waited = time.monotonic() - started
            if waited >= SCAN_QUEUE_WARN_SECONDS:
                self._delayed_scans += 1
                self._max_scan_wait = max(self._max_scan_wait, waited)
                logger.warning(
                    "Text scan queued %.1fs for a permit (%d running, %d still queued, "
                    "cap %d) — raise %s if this is routine and there's memory headroom",
                    waited,
                    self._scan_gate.in_use,
                    self._scan_gate.waiting,
                    self._scan_gate.capacity,
                    ENV_MAX_CONCURRENT_SCANS,
                )
            yield

    async def _set_capacity(self, new_capacity: int) -> None:
        before = self._index_gate.capacity
        await self._index_gate.set_capacity(new_capacity)
        # Only on a real move: the monitor calls this every tick while usage is
        # comfortable, and an unconditional log would be one line per second.
        if self._index_gate.capacity != before:
            logger.debug("Memory gate capacity -> %d", self._index_gate.capacity)

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
                    "Memory pressure %.0f%% (%s/%s) — halving indexing and scan concurrency",
                    fraction * 100,
                    format_bytes(usage),
                    format_bytes(self._limit),
                    )
                await self._set_capacity(self._index_gate.capacity // 2)
                # Scans get halved too: their gate is small, so shedding one
                # in-flight rg is a meaningful fraction of the scan budget.
                await self._scan_gate.set_capacity(self._scan_gate.capacity // 2)
            elif fraction >= SOFT_PRESSURE:
                # Soft pressure is routine mid-index; only the fan-out eases
                # off, so an interactive grep still gets served.
                await self._set_capacity(self._index_gate.capacity - 1)
            elif fraction < RECOVER_BELOW:
                await self._set_capacity(self._index_gate.capacity + 1)
                await self._scan_gate.set_capacity(self._scan_gate.capacity + 1)

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
            max_inflight=self._index_gate.maximum,
            current_capacity=self._index_gate.capacity,
            throttle_events=self._throttle_events,
            scan_budget=self._scan_budget,
            current_scan_capacity=self._scan_gate.capacity,
            scans_running=self._scan_gate.in_use,
            scans_queued=self._scan_gate.waiting,
            peak_scans_queued=self._scan_gate.peak_waiting,
            delayed_scans=self._delayed_scans,
            max_scan_wait_seconds=self._max_scan_wait,
        )
