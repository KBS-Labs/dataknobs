"""Tests for corpus transform helpers."""

from __future__ import annotations

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_bots.artifacts import (
    ArtifactCorpus,
    ArtifactRegistry,
    ArtifactStatus,
    TransformContext,
    add_to_corpus,
    create_corpus,
    finalize_corpus,
)


@pytest.fixture
async def registry() -> ArtifactRegistry:
    """Create a registry with an in-memory database."""
    db = AsyncMemoryDatabase()
    return ArtifactRegistry(db=db)


@pytest.fixture
def context(registry: ArtifactRegistry) -> TransformContext:
    """Create a transform context with the registry."""
    return TransformContext(artifact_registry=registry)


@pytest.mark.asyncio
async def test_create_corpus_transform(context: TransformContext) -> None:
    """create_corpus sets _corpus_id, _corpus, and _corpus_item_count."""
    data: dict = {}
    config = {
        "corpus_type": "quiz_bank",
        "item_type": "quiz_question",
        "name_field": "topic",
    }
    data["topic"] = "Chapter 1 Quiz"

    await create_corpus(data, context, config=config)

    assert data["_corpus_id"].startswith("art_")
    assert isinstance(data["_corpus"], ArtifactCorpus)
    assert data["_corpus_item_count"] == 0


@pytest.mark.asyncio
async def test_add_to_corpus_transform(context: TransformContext) -> None:
    """add_to_corpus creates an item and updates count."""
    data: dict = {}
    data["topic"] = "Math Quiz"

    # Create corpus first
    await create_corpus(data, context, config={
        "corpus_type": "quiz_bank",
        "item_type": "quiz_question",
        "name_field": "topic",
    })

    # Add an item
    data["_current_item"] = {"stem": "What is 2+2?", "answer": "4"}
    await add_to_corpus(data, context, config={
        "content_key": "_current_item",
        "tags": ["math"],
    })

    assert data["_last_added_artifact_id"].startswith("art_")
    assert data["_corpus_item_count"] == 1
    assert data["_dedup_result"] is None

    # Verify artifact exists in registry
    artifact = await context.artifact_registry.get(data["_last_added_artifact_id"])
    assert artifact is not None
    assert artifact.content["stem"] == "What is 2+2?"
    assert "math" in artifact.tags


@pytest.mark.asyncio
async def test_finalize_corpus_transform(
    context: TransformContext,
    registry: ArtifactRegistry,
) -> None:
    """finalize_corpus sets _corpus_summary and approves the corpus."""
    data: dict = {"topic": "Science Quiz"}

    await create_corpus(data, context, config={
        "corpus_type": "quiz_bank",
        "item_type": "quiz_question",
        "name_field": "topic",
    })

    data["_current_item"] = {"stem": "Q1"}
    await add_to_corpus(data, context)
    data["_current_item"] = {"stem": "Q2"}
    await add_to_corpus(data, context)

    await finalize_corpus(data, context)

    summary = data["_corpus_summary"]
    assert summary["total_items"] == 2
    assert summary["corpus_type"] == "quiz_bank"
    assert summary["corpus_status"] == "approved"


@pytest.mark.asyncio
async def test_full_flow(context: TransformContext) -> None:
    """Full flow: create_corpus -> add_to_corpus x3 -> finalize_corpus."""
    data: dict = {"topic": "History Quiz"}

    await create_corpus(data, context, config={
        "corpus_type": "quiz_bank",
        "item_type": "quiz_question",
        "name_field": "topic",
    })

    for i in range(3):
        data["_current_item"] = {"stem": f"Question {i+1}", "answer": f"Answer {i+1}"}
        await add_to_corpus(data, context)

    assert data["_corpus_item_count"] == 3

    await finalize_corpus(data, context)

    summary = data["_corpus_summary"]
    assert summary["total_items"] == 3

    # Verify all items link to the corpus
    corpus = data["_corpus"]
    items = await corpus.get_items()
    assert len(items) == 3
    assert all(item.content["corpus_id"] == data["_corpus_id"] for item in items)


@pytest.mark.asyncio
async def test_add_to_corpus_with_dedup(context: TransformContext) -> None:
    """Dedup config in create_corpus prevents duplicate items."""
    data: dict = {"topic": "Dedup Quiz"}

    await create_corpus(data, context, config={
        "corpus_type": "quiz_bank",
        "item_type": "quiz_question",
        "name_field": "topic",
        "dedup": {
            "hash_fields": ["stem"],
        },
    })

    # First add succeeds
    data["_current_item"] = {"stem": "What is 2+2?"}
    await add_to_corpus(data, context)
    assert data["_corpus_item_count"] == 1
    assert data["_dedup_result"]["is_exact_duplicate"] is False

    # Second add with same content is detected as duplicate
    data["_current_item"] = {"stem": "What is 2+2?"}
    await add_to_corpus(data, context)
    assert data["_corpus_item_count"] == 1
    assert data["_dedup_result"]["is_exact_duplicate"] is True


@pytest.mark.asyncio
async def test_add_to_corpus_reload_from_id(context: TransformContext) -> None:
    """add_to_corpus reconstructs corpus from _corpus_id when _corpus is missing."""
    data: dict = {"topic": "Reload Quiz"}

    await create_corpus(data, context, config={
        "corpus_type": "quiz_bank",
        "item_type": "quiz_question",
        "name_field": "topic",
    })

    # Simulate session reload: remove transient _corpus, keep _corpus_id
    corpus_id = data["_corpus_id"]
    del data["_corpus"]

    data["_current_item"] = {"stem": "Q1"}
    await add_to_corpus(data, context)

    assert data["_last_added_artifact_id"].startswith("art_")
    assert data["_corpus_item_count"] == 1
    assert data["_corpus_id"] == corpus_id
    # Corpus was reconstructed
    assert isinstance(data["_corpus"], ArtifactCorpus)


@pytest.mark.asyncio
async def test_create_corpus_missing_registry() -> None:
    """create_corpus raises ValueError when artifact_registry is None."""
    context = TransformContext(artifact_registry=None)
    data: dict = {"topic": "Test"}

    with pytest.raises(ValueError, match="artifact_registry"):
        await create_corpus(data, context, config={
            "corpus_type": "quiz_bank",
            "item_type": "quiz_question",
        })
