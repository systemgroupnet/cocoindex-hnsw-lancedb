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
    ENGINE_DEFAULT_MAX_INFLIGHT,
    MIN_INFLIGHT,
    MemoryGovernor,
    detect_memory_limit_bytes,
    recommend_max_inflight,
    resolve_ceiling,
)

MB = 1024 * 1024
GB = 1024 * MB


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure detection env overrides never leak in from the ambient shell."""
    monkeypatch.delenv(memory.ENV_MEMORY_LIMIT_MB, raising=False)
    monkeypatch.delenv(memory.ENV_MAX_INFLIGHT_FILES, raising=False)


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


async def test_min_inflight_floor_not_exceeding_small_ceiling() -> None:
    # A user-pinned ceiling below MIN_INFLIGHT must be honored, not floored up.
    gov = _unconstrained_governor(ceiling=2)
    assert gov.max_inflight == 2
    await gov._set_capacity(1)  # would floor to MIN_INFLIGHT (4) without the fix
    assert gov.snapshot().current_capacity == 2
