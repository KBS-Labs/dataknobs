"""Tests for DedupChecker."""

from __future__ import annotations

import numpy as np
import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig, DedupResult
from dataknobs_data.vector.stores.memory import MemoryVectorStore


@pytest.fixture
def db() -> AsyncMemoryDatabase:
    """Create a fresh in-memory database."""
    return AsyncMemoryDatabase()


@pytest.fixture
def config() -> DedupConfig:
    """Default dedup config."""
    return DedupConfig(hash_fields=["content"])


@pytest.fixture
def checker(db: AsyncMemoryDatabase, config: DedupConfig) -> DedupChecker:
    """Create a DedupChecker with default config."""
    return DedupChecker(db=db, config=config)


@pytest.mark.asyncio
async def test_exact_duplicate_detected(checker: DedupChecker) -> None:
    """Registering content and checking the same content yields exact duplicate."""
    content = {"content": "What is 2+2?"}
    await checker.register(content, record_id="q-1")

    result = await checker.check(content)

    assert result.is_exact_duplicate is True
    assert result.exact_match_id == "q-1"
    assert result.recommendation == "exact_duplicate"
    assert result.content_hash != ""


@pytest.mark.asyncio
async def test_unique_content_passes(checker: DedupChecker) -> None:
    """Different content is not detected as duplicate."""
    await checker.register({"content": "What is 2+2?"}, record_id="q-1")

    result = await checker.check({"content": "What is the capital of France?"})

    assert result.is_exact_duplicate is False
    assert result.recommendation == "unique"
    assert result.similar_items == []


@pytest.mark.asyncio
async def test_semantic_similarity_detected(db: AsyncMemoryDatabase) -> None:
    """Semantic similarity is detected when embeddings are close."""
    vector_store = MemoryVectorStore({"dimensions": 4})
    await vector_store.initialize()

    # Deterministic embeddings: very similar vectors
    embeddings: dict[str, list[float]] = {
        "original question": [1.0, 0.0, 0.0, 0.0],
        "similar question": [0.99, 0.1, 0.0, 0.0],
    }

    async def fake_embed(text: str) -> list[float]:
        return embeddings.get(text, [0.0, 0.0, 0.0, 0.0])

    config = DedupConfig(
        hash_fields=["content"],
        semantic_check=True,
        similarity_threshold=0.9,
    )
    checker = DedupChecker(
        db=db,
        config=config,
        vector_store=vector_store,
        embedding_fn=fake_embed,
    )

    await checker.register({"content": "original question"}, record_id="q-1")
    result = await checker.check({"content": "similar question"})

    assert result.is_exact_duplicate is False
    assert result.recommendation == "possible_duplicate"
    assert len(result.similar_items) >= 1
    assert result.similar_items[0].record_id == "q-1"
    assert result.similar_items[0].score >= 0.9


@pytest.mark.asyncio
async def test_semantic_below_threshold(db: AsyncMemoryDatabase) -> None:
    """Sufficiently different embeddings are not flagged."""
    vector_store = MemoryVectorStore({"dimensions": 4})
    await vector_store.initialize()

    # Very different vectors
    embeddings: dict[str, list[float]] = {
        "math question": [1.0, 0.0, 0.0, 0.0],
        "cooking recipe": [0.0, 0.0, 0.0, 1.0],
    }

    async def fake_embed(text: str) -> list[float]:
        return embeddings.get(text, [0.0, 0.0, 0.0, 0.0])

    config = DedupConfig(
        hash_fields=["content"],
        semantic_check=True,
        similarity_threshold=0.92,
    )
    checker = DedupChecker(
        db=db,
        config=config,
        vector_store=vector_store,
        embedding_fn=fake_embed,
    )

    await checker.register({"content": "math question"}, record_id="q-1")
    result = await checker.check({"content": "cooking recipe"})

    assert result.is_exact_duplicate is False
    assert result.recommendation == "unique"
    assert result.similar_items == []


@pytest.mark.asyncio
async def test_register_then_check(checker: DedupChecker) -> None:
    """Multiple registrations, each detected as exact match."""
    items = {
        "q-1": {"content": "alpha"},
        "q-2": {"content": "beta"},
        "q-3": {"content": "gamma"},
    }

    for record_id, content in items.items():
        await checker.register(content, record_id=record_id)

    # Each registered item is found
    for record_id, content in items.items():
        result = await checker.check(content)
        assert result.is_exact_duplicate is True
        assert result.exact_match_id == record_id

    # New item is not found
    result = await checker.check({"content": "delta"})
    assert result.is_exact_duplicate is False
    assert result.recommendation == "unique"


@pytest.mark.asyncio
async def test_hash_field_selection(db: AsyncMemoryDatabase) -> None:
    """Hash only considers configured fields."""
    config = DedupConfig(hash_fields=["stem"])
    checker = DedupChecker(db=db, config=config)

    # Same stem, different options
    await checker.register(
        {"stem": "What is 2+2?", "options": ["3", "4"]}, record_id="q-1"
    )
    result = await checker.check(
        {"stem": "What is 2+2?", "options": ["5", "6"]}
    )

    assert result.is_exact_duplicate is True
    assert result.exact_match_id == "q-1"


@pytest.mark.asyncio
async def test_hash_algorithm_sha256(db: AsyncMemoryDatabase) -> None:
    """SHA-256 produces 64-char hex string."""
    config = DedupConfig(hash_fields=["content"], hash_algorithm="sha256")
    checker = DedupChecker(db=db, config=config)

    content_hash = checker.compute_hash({"content": "test"})

    assert len(content_hash) == 64
    # Verify it's valid hex
    int(content_hash, 16)


@pytest.mark.asyncio
async def test_check_without_prior_registration(checker: DedupChecker) -> None:
    """Checking against an empty database returns unique."""
    result = await checker.check({"content": "anything"})

    assert result.is_exact_duplicate is False
    assert result.recommendation == "unique"
    assert result.content_hash != ""


@pytest.mark.asyncio
async def test_semantic_without_vector_store(db: AsyncMemoryDatabase) -> None:
    """Semantic check is skipped gracefully when vector_store is None."""
    config = DedupConfig(
        hash_fields=["content"],
        semantic_check=True,
    )
    checker = DedupChecker(db=db, config=config, vector_store=None)

    await checker.register({"content": "hello"}, record_id="q-1")
    result = await checker.check({"content": "similar hello"})

    # No crash, just hash-based check
    assert result.is_exact_duplicate is False
    assert result.recommendation == "unique"
    assert result.similar_items == []
