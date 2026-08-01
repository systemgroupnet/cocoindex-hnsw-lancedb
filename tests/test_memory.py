"""Tests for runtime memory awareness (detection, budget sizing, governor).

The detection tests point the module's cgroup/meminfo path constants at temp
files via monkeypatch, so they exercise the real parsing/precedence logic
without depending on the host's actual cgroup layout (and run identically on
non-Linux CI).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cocoindex_code import memory
from cocoindex_code.daemon import _second_embedder_fits
from cocoindex_code.memory import (
    DEFAULT_MAX_CONCURRENT_SCANS,
    ENGINE_DEFAULT_MAX_INFLIGHT,
    MAX_BLOB_BATCH_BYTES,
    MAX_SCAN_FILESIZE_BYTES,
    MIN_BLOB_BATCH_BYTES,
    MIN_INFLIGHT,
    MemoryGovernor,
    detect_memory_limit_bytes,
    recommend_max_inflight,
    recommend_scan_budget,
    resolve_ceiling,
    resolve_scan_concurrency,
)

MB = 1024 * 1024
GB = 1024 * MB


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure detection env overrides never leak in from the ambient shell."""
    monkeypatch.delenv(memory.ENV_MEMORY_LIMIT_MB, raising=False)
    monkeypatch.delenv(memory.ENV_MAX_INFLIGHT_FILES, raising=False)
    monkeypatch.delenv(memory.ENV_MAX_CONCURRENT_SCANS, raising=False)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# --- detect_memory_limit_bytes ---------------------------------------------


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(memory.ENV_MEMORY_LIMIT_MB, "512")
    limit, source = detect_memory_limit_bytes()
    assert limit == 512 * MB
    assert "env" in source


def test_cgroup_v2_intersected_with_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(memory, "_CGROUP_V2_MAX", _write(tmp_path / "v2", str(2 * GB)))
    monkeypatch.setattr(memory, "_host_total_bytes", lambda: 8 * GB)
    limit, source = detect_memory_limit_bytes()
    assert limit == 2 * GB
    assert source == "cgroup v2"


def test_cgroup_v2_capped_at_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A cgroup cap larger than physical RAM is meaningless — intersect with host.
    monkeypatch.setattr(memory, "_CGROUP_V2_MAX", _write(tmp_path / "v2", str(64 * GB)))
    monkeypatch.setattr(memory, "_host_total_bytes", lambda: 8 * GB)
    limit, _ = detect_memory_limit_bytes()
    assert limit == 8 * GB


def test_cgroup_v2_max_literal_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(memory, "_CGROUP_V2_MAX", _write(tmp_path / "v2", "max"))
    monkeypatch.setattr(memory, "_CGROUP_V1_LIMIT", tmp_path / "missing")
    monkeypatch.setattr(memory, "_host_total_bytes", lambda: 8 * GB)
    limit, source = detect_memory_limit_bytes()
    assert limit == 8 * GB
    assert source == "host total"


def test_cgroup_v1_unlimited_sentinel_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(memory, "_CGROUP_V2_MAX", tmp_path / "missing_v2")
    monkeypatch.setattr(
        memory, "_CGROUP_V1_LIMIT", _write(tmp_path / "v1", str(9223372036854771712))
    )
    monkeypatch.setattr(memory, "_host_total_bytes", lambda: 4 * GB)
    limit, source = detect_memory_limit_bytes()
    assert limit == 4 * GB
    assert source == "host total"


def test_nothing_detectable_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(memory, "_CGROUP_V2_MAX", tmp_path / "nope_v2")
    monkeypatch.setattr(memory, "_CGROUP_V1_LIMIT", tmp_path / "nope_v1")
    monkeypatch.setattr(memory, "_host_total_bytes", lambda: None)
    limit, source = detect_memory_limit_bytes()
    assert limit is None
    assert source == "undetected"


# --- recommend_max_inflight / resolve_ceiling ------------------------------


def test_recommend_scales_with_budget() -> None:
    # 2 GiB, ~250 MiB baseline: (2Gi*0.85 - 250Mi) / 8Mi ≈ 190 files.
    n = recommend_max_inflight(2 * GB, 250 * MB, ENGINE_DEFAULT_MAX_INFLIGHT)
    assert MIN_INFLIGHT < n < ENGINE_DEFAULT_MAX_INFLIGHT
    # Halving the budget roughly halves the cap.
    n_half = recommend_max_inflight(1 * GB, 250 * MB, ENGINE_DEFAULT_MAX_INFLIGHT)
    assert n_half < n


def test_recommend_floors_at_min_when_tiny() -> None:
    # A budget the baseline nearly consumes still leaves the floor.
    n = recommend_max_inflight(300 * MB, 290 * MB, ENGINE_DEFAULT_MAX_INFLIGHT)
    assert n == MIN_INFLIGHT


def test_recommend_returns_ceiling_when_limit_unknown() -> None:
    assert recommend_max_inflight(None, None, 777) == 777


def test_recommend_never_exceeds_ceiling() -> None:
    assert recommend_max_inflight(1024 * GB, 0, 32) == 32


def test_resolve_ceiling_default() -> None:
    assert resolve_ceiling() == ENGINE_DEFAULT_MAX_INFLIGHT


def test_resolve_ceiling_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(memory.ENV_MAX_INFLIGHT_FILES, "17")
    assert resolve_ceiling() == 17


def test_resolve_ceiling_ignores_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(memory.ENV_MAX_INFLIGHT_FILES, "-3")
    assert resolve_ceiling() == ENGINE_DEFAULT_MAX_INFLIGHT


# --- recommend_scan_budget --------------------------------------------------
#
# These pin the *bounds*, not the sizing: concurrency and the file cut are
# fixed policy, and the blob batch is only asserted to stay inside its clamp
# and to shrink with the budget. The exact byte count is a judgement call, so
# no test enshrines it.


def test_scan_budget_is_bounded_without_a_limit() -> None:
    budget = recommend_scan_budget(None, None)
    assert budget.max_concurrent == DEFAULT_MAX_CONCURRENT_SCANS
    assert budget.max_filesize_bytes == MAX_SCAN_FILESIZE_BYTES
    assert MIN_BLOB_BATCH_BYTES <= budget.blob_batch_bytes <= MAX_BLOB_BATCH_BYTES


def test_scan_budget_blob_batch_stays_within_its_clamp() -> None:
    for limit, baseline in [(512 * MB, 400 * MB), (2 * GB, 250 * MB), (256 * GB, 0)]:
        budget = recommend_scan_budget(limit, baseline)
        assert MIN_BLOB_BATCH_BYTES <= budget.blob_batch_bytes <= MAX_BLOB_BATCH_BYTES


def test_scan_budget_blob_batch_shrinks_with_the_budget() -> None:
    roomy = recommend_scan_budget(8 * GB, 250 * MB).blob_batch_bytes
    tight = recommend_scan_budget(1 * GB, 250 * MB).blob_batch_bytes
    assert tight < roomy


def test_scan_budget_file_cut_is_policy_not_sized() -> None:
    # A tiny container and a huge one skip the same files: this bound exists to
    # keep code search off multi-megabyte blobs, not to fit a budget.
    assert (
        recommend_scan_budget(256 * MB, 200 * MB).max_filesize_bytes
        == recommend_scan_budget(64 * GB, 0).max_filesize_bytes
        == MAX_SCAN_FILESIZE_BYTES
    )


def test_scan_concurrency_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(memory.ENV_MAX_CONCURRENT_SCANS, "2")
    assert resolve_scan_concurrency() == 2
    budget = recommend_scan_budget(2 * GB, 0, concurrency_override=2)
    assert budget.max_concurrent == 2


def test_scan_concurrency_ignores_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(memory.ENV_MAX_CONCURRENT_SCANS, "0")
    assert resolve_scan_concurrency() is None
    assert recommend_scan_budget(2 * GB, 0).max_concurrent == DEFAULT_MAX_CONCURRENT_SCANS


# --- _second_embedder_fits (daemon helper) ---------------------------------


def test_second_embedder_fits_unknown_limit_keeps_separate() -> None:
    assert _second_embedder_fits(None, 100 * MB, 900 * MB) is True


def test_second_embedder_shares_when_second_copy_would_overflow() -> None:
    # First model took 700 MiB (100 -> 800). A second copy would project to
    # ~1.5 GiB, over 85% of a 1 GiB limit → share.
    assert _second_embedder_fits(1 * GB, 100 * MB, 800 * MB) is False


def test_second_embedder_fits_when_cheap() -> None:
    # LiteLLM-like: no measurable model cost → a second instance is free.
    assert _second_embedder_fits(2 * GB, 300 * MB, 300 * MB) is True


# --- MemoryGovernor gate ----------------------------------------------------


def _unconstrained_governor(ceiling: int) -> MemoryGovernor:
    gov = MemoryGovernor(None, "test", ceiling)
    gov.calibrate()  # limit None -> max_inflight == ceiling, no monitor
    return gov


async def test_slot_bounds_concurrency() -> None:
    gov = _unconstrained_governor(ceiling=2)
    assert gov.max_inflight == 2

    active = 0
    peak = 0
    release = asyncio.Event()

    async def worker() -> None:
        nonlocal active, peak
        async with gov.slot():
            active += 1
            peak = max(peak, active)
            await release.wait()
            active -= 1

    tasks = [asyncio.create_task(worker()) for _ in range(5)]
    await asyncio.sleep(0.05)
    # Only 2 may hold a slot at once despite 5 tasks queued.
    assert peak == 2
    release.set()
    await asyncio.gather(*tasks)


async def test_shrinking_capacity_blocks_new_slots() -> None:
    # Ceiling 8 gives headroom above MIN_INFLIGHT (4) so a shrink is meaningful.
    gov = _unconstrained_governor(ceiling=8)
    proceed = asyncio.Event()
    entered = 0

    async def worker() -> None:
        nonlocal entered
        async with gov.slot():
            entered += 1
            await proceed.wait()

    # Fill 5 of 8 slots, then shrink capacity to the floor (4).
    workers = [asyncio.create_task(worker()) for _ in range(5)]
    await asyncio.sleep(0.02)
    assert entered == 5
    await gov._set_capacity(4)

    # A 6th worker must not enter: 5 held >= capacity 4.
    extra = asyncio.create_task(worker())
    await asyncio.sleep(0.02)
    assert entered == 5  # still blocked

    proceed.set()
    await asyncio.gather(*workers, extra)


async def test_capacity_clamped_to_max_inflight() -> None:
    gov = _unconstrained_governor(ceiling=8)
    await gov._set_capacity(100)  # requested above the cap
    snap = gov.snapshot()
    assert snap.current_capacity == gov.max_inflight == 8


async def test_scan_slot_bounds_concurrent_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound that matters: N greps in flight, never N+1.

    Holds whatever the memory numbers are — before the gate existed, the only
    cap on simultaneous rg processes was the default thread pool.
    """
    monkeypatch.setenv(memory.ENV_MAX_CONCURRENT_SCANS, "2")
    gov = MemoryGovernor(None, "test", ENGINE_DEFAULT_MAX_INFLIGHT)
    gov.calibrate()
    assert gov.scan_budget.max_concurrent == 2

    active = 0
    peak = 0
    release = asyncio.Event()

    async def scan() -> None:
        nonlocal active, peak
        async with gov.scan_slot():
            active += 1
            peak = max(peak, active)
            await release.wait()
            active -= 1

    tasks = [asyncio.create_task(scan()) for _ in range(6)]
    await asyncio.sleep(0.05)
    assert peak == 2
    release.set()
    await asyncio.gather(*tasks)


async def test_queued_scans_are_counted_while_they_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue depth has to be visible, or a full gate looks like a hang."""
    monkeypatch.setenv(memory.ENV_MAX_CONCURRENT_SCANS, "1")
    gov = MemoryGovernor(None, "test", ENGINE_DEFAULT_MAX_INFLIGHT)
    gov.calibrate()
    release = asyncio.Event()

    async def scan() -> None:
        async with gov.scan_slot():
            await release.wait()

    tasks = [asyncio.create_task(scan()) for _ in range(4)]
    await asyncio.sleep(0.05)

    mid = gov.snapshot()
    assert mid.scans_running == 1
    assert mid.scans_queued == 3
    assert mid.peak_scans_queued == 3

    release.set()
    await asyncio.gather(*tasks)

    after = gov.snapshot()
    assert (after.scans_running, after.scans_queued) == (0, 0)
    assert after.peak_scans_queued == 3  # retained for `ccc doctor`


async def test_every_queued_scan_is_served_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate must never drop a request — waiting is the whole contract."""
    monkeypatch.setenv(memory.ENV_MAX_CONCURRENT_SCANS, "2")
    gov = MemoryGovernor(None, "test", ENGINE_DEFAULT_MAX_INFLIGHT)
    gov.calibrate()
    served = 0

    async def scan() -> None:
        nonlocal served
        async with gov.scan_slot():
            await asyncio.sleep(0.01)
            served += 1

    await asyncio.gather(*[scan() for _ in range(20)])
    assert served == 20
    assert gov.snapshot().scans_queued == 0


async def test_long_waits_are_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "SCAN_QUEUE_WARN_SECONDS", 0.02)
    monkeypatch.setenv(memory.ENV_MAX_CONCURRENT_SCANS, "1")
    gov = MemoryGovernor(None, "test", ENGINE_DEFAULT_MAX_INFLIGHT)
    gov.calibrate()

    async def scan(hold: float) -> None:
        async with gov.scan_slot():
            await asyncio.sleep(hold)

    # The second scan waits out the first, well past the threshold.
    await asyncio.gather(scan(0.08), scan(0.0))

    snap = gov.snapshot()
    assert snap.delayed_scans == 1
    assert snap.max_scan_wait_seconds >= 0.02


async def test_uncontended_scans_are_not_counted_as_delayed() -> None:
    gov = _unconstrained_governor(ceiling=8)
    async with gov.scan_slot():
        pass
    snap = gov.snapshot()
    assert snap.delayed_scans == 0
    assert snap.peak_scans_queued == 0


async def test_scan_gate_is_independent_of_the_indexing_gate() -> None:
    """A full indexing gate must not stall an interactive grep, or vice versa."""
    gov = _unconstrained_governor(ceiling=1)
    hold = asyncio.Event()

    async def index_file() -> None:
        async with gov.slot():
            await hold.wait()

    indexing = asyncio.create_task(index_file())
    await asyncio.sleep(0.02)

    async with gov.scan_slot():  # would hang if the gates shared permits
        pass

    hold.set()
    await indexing


async def test_scan_capacity_recovers_after_pressure() -> None:
    gov = _unconstrained_governor(ceiling=8)
    await gov._scan_gate.set_capacity(1)
    assert gov.snapshot().current_scan_capacity == 1
    await gov._scan_gate.set_capacity(gov.scan_budget.max_concurrent)
    assert gov.snapshot().current_scan_capacity == gov.scan_budget.max_concurrent


async def test_scan_gate_never_drops_below_one() -> None:
    gov = _unconstrained_governor(ceiling=8)
    await gov._scan_gate.set_capacity(0)
    assert gov.snapshot().current_scan_capacity == 1


async def test_min_inflight_floor_not_exceeding_small_ceiling() -> None:
    # A user-pinned ceiling below MIN_INFLIGHT must be honored, not floored up.
    gov = _unconstrained_governor(ceiling=2)
    assert gov.max_inflight == 2
    await gov._set_capacity(1)  # would floor to MIN_INFLIGHT (4) without the fix
    assert gov.snapshot().current_capacity == 2
