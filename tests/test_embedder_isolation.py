"""Query embedding must not serialize behind indexing (slim / LiteLLM path).

``PacedLiteLLMEmbedder`` gates every request through a per-instance lock (+ pacing),
so a shared instance makes a search's embedding wait for all in-flight indexing
embeds. The daemon now uses a *separate* embedder instance for the query path; these
tests pin both the bug (shared instance blocks) and the fix (separate instance does
not). ``litellm.aembedding`` is mocked, so no network/API key is needed.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import cocoindex_code.litellm_embedder as _mod
from cocoindex_code.litellm_embedder import PacedLiteLLMEmbedder

_LATENCY = 0.3  # simulated per-request API latency
_FLOOD = 8  # concurrent "indexing" requests saturating an instance's lock


@pytest.fixture
def _fake_aembedding(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(*, model: str, input: list[str], **kwargs: object) -> object:
        await asyncio.sleep(_LATENCY)
        return SimpleNamespace(data=[{"embedding": [0.0, 0.1, 0.2]} for _ in input])

    monkeypatch.setattr(_mod.litellm, "aembedding", fake)


async def _flood(embedder: PacedLiteLLMEmbedder) -> list[asyncio.Task[object]]:
    tasks = [
        asyncio.create_task(embedder.run_embedding_request(input=[f"chunk {i}"]))
        for i in range(_FLOOD)
    ]
    await asyncio.sleep(0.05)  # let the flood grab the lock before we time the query
    return tasks


@pytest.mark.asyncio
async def test_separate_instance_query_not_blocked(_fake_aembedding: None) -> None:
    """A query on its own instance returns in ~one request, not behind the flood."""
    indexing = PacedLiteLLMEmbedder("openai/text-embedding-3-small", min_interval_ms=0)
    query = PacedLiteLLMEmbedder("openai/text-embedding-3-small", min_interval_ms=0)

    idx_tasks = await _flood(indexing)
    start = time.monotonic()
    await query.run_embedding_request(input=["my search query"])
    elapsed = time.monotonic() - start

    # ~one request of latency; well under the flood's serialized total (~8 * 0.3s).
    assert elapsed < _LATENCY * 3, f"query waited {elapsed:.2f}s on a separate instance"
    await asyncio.gather(*idx_tasks)


@pytest.mark.asyncio
async def test_shared_instance_query_blocks(_fake_aembedding: None) -> None:
    """Documents the bug: on a shared instance the query waits behind indexing."""
    shared = PacedLiteLLMEmbedder("openai/text-embedding-3-small", min_interval_ms=0)

    idx_tasks = await _flood(shared)
    start = time.monotonic()
    await shared.run_embedding_request(input=["my search query"])
    elapsed = time.monotonic() - start

    # The query is queued behind the flood on the shared lock.
    assert elapsed > _LATENCY * 3, f"expected query to block, but it returned in {elapsed:.2f}s"
    await asyncio.gather(*idx_tasks)
