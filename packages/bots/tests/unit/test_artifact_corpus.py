"""Tests for ArtifactCorpus."""

from __future__ import annotations

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig

from dataknobs_bots.artifacts import (
    ArtifactCorpus,
    ArtifactRegistry,
    ArtifactStatus,
)
from dataknobs_bots.artifacts.corpus import CorpusConfig


@pytest.fixture
async def registry() -> ArtifactRegistry:
    """Create a registry with an in-memory database."""
    db = AsyncMemoryDatabase()
    return ArtifactRegistry(db=db)


@pytest.fixture
def config() -> CorpusConfig:
    """Default corpus config for testing."""
    return CorpusConfig(
        corpus_type="quiz_bank",
        item_type="quiz_question",
        name="Chapter 1 Quiz",
    )


@pytest.mark.asyncio
async def test_create_corpus(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """Creating a corpus creates a parent artifact in the registry."""
    corpus = await ArtifactCorpus.create(registry, config)

    assert corpus.id.startswith("art_")
    parent = await registry.get(corpus.id)
    assert parent is not None
    assert parent.type == "quiz_bank"
    assert parent.name == "Chapter 1 Quiz"
    assert parent.content["item_type"] == "quiz_question"
    assert "corpus" in parent.tags


@pytest.mark.asyncio
async def test_add_item(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """Adding an item creates an artifact linked to the corpus."""
    corpus = await ArtifactCorpus.create(registry, config)

    artifact, dedup_result = await corpus.add_item(
        content={"stem": "What is 2+2?", "answer": "4"},
        tags=["math"],
    )

    assert artifact.type == "quiz_question"
    assert artifact.content["corpus_id"] == corpus.id
    assert artifact.content["stem"] == "What is 2+2?"
    assert "math" in artifact.tags
    assert dedup_result is None  # No dedup checker configured

    # Verify queryable via get_items
    items = await corpus.get_items()
    assert len(items) == 1
    assert items[0].id == artifact.id


@pytest.mark.asyncio
async def test_get_items(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """get_items returns all items; status filter works."""
    corpus = await ArtifactCorpus.create(registry, config)

    for i in range(3):
        await corpus.add_item(content={"stem": f"Q{i}"})

    items = await corpus.get_items()
    assert len(items) == 3

    draft_items = await corpus.get_items(status=ArtifactStatus.DRAFT)
    assert len(draft_items) == 3

    approved_items = await corpus.get_items(status=ArtifactStatus.APPROVED)
    assert len(approved_items) == 0


@pytest.mark.asyncio
async def test_get_items_isolation(registry: ArtifactRegistry) -> None:
    """Items are scoped to their corpus, not shared across corpora."""
    config_a = CorpusConfig(
        corpus_type="quiz_bank", item_type="quiz_question", name="Quiz A"
    )
    config_b = CorpusConfig(
        corpus_type="quiz_bank", item_type="quiz_question", name="Quiz B"
    )

    corpus_a = await ArtifactCorpus.create(registry, config_a)
    corpus_b = await ArtifactCorpus.create(registry, config_b)

    await corpus_a.add_item(content={"stem": "A1"})
    await corpus_a.add_item(content={"stem": "A2"})
    await corpus_b.add_item(content={"stem": "B1"})

    items_a = await corpus_a.get_items()
    items_b = await corpus_b.get_items()

    assert len(items_a) == 2
    assert len(items_b) == 1
    assert all(i.content["corpus_id"] == corpus_a.id for i in items_a)
    assert items_b[0].content["corpus_id"] == corpus_b.id


@pytest.mark.asyncio
async def test_count(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """count returns correct item count with optional status filter."""
    corpus = await ArtifactCorpus.create(registry, config)

    await corpus.add_item(content={"stem": "Q1"})
    await corpus.add_item(content={"stem": "Q2"})

    assert await corpus.count() == 2
    assert await corpus.count(status=ArtifactStatus.DRAFT) == 2
    assert await corpus.count(status=ArtifactStatus.APPROVED) == 0


@pytest.mark.asyncio
async def test_finalize(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """Finalizing a corpus records item summary and approves it."""
    corpus = await ArtifactCorpus.create(registry, config)

    a1, _ = await corpus.add_item(content={"stem": "Q1"})
    a2, _ = await corpus.add_item(content={"stem": "Q2"})

    finalized = await corpus.finalize()

    assert finalized.content["finalized"] is True
    assert finalized.content["item_count"] == 2
    assert set(finalized.content["item_ids"]) == {a1.id, a2.id}
    assert finalized.status == ArtifactStatus.APPROVED


@pytest.mark.asyncio
async def test_load_existing(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """Loading an existing corpus restores access to its items."""
    corpus = await ArtifactCorpus.create(registry, config)
    await corpus.add_item(content={"stem": "Q1"})
    corpus_id = corpus.id

    loaded = await ArtifactCorpus.load(registry, corpus_id)

    assert loaded.id == corpus_id
    items = await loaded.get_items()
    assert len(items) == 1
    assert items[0].content["stem"] == "Q1"


@pytest.mark.asyncio
async def test_add_with_dedup(registry: ArtifactRegistry) -> None:
    """Dedup checker prevents adding exact duplicates."""
    dedup_db = AsyncMemoryDatabase()
    dedup_checker = DedupChecker(
        db=dedup_db,
        config=DedupConfig(hash_fields=["stem"]),
    )
    config = CorpusConfig(
        corpus_type="quiz_bank",
        item_type="quiz_question",
        name="Dedup Quiz",
    )
    corpus = await ArtifactCorpus.create(registry, config, dedup_checker=dedup_checker)

    # First add succeeds
    a1, result1 = await corpus.add_item(content={"stem": "What is 2+2?"})
    assert result1 is not None
    assert result1.is_exact_duplicate is False

    # Second add with same content is detected as duplicate
    a2, result2 = await corpus.add_item(content={"stem": "What is 2+2?"})
    assert result2 is not None
    assert result2.is_exact_duplicate is True
    assert a2.id == a1.id  # Returns existing artifact

    # Only one item in corpus
    assert await corpus.count() == 1


@pytest.mark.asyncio
async def test_remove_item(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """Removing an item archives it."""
    corpus = await ArtifactCorpus.create(registry, config)
    artifact, _ = await corpus.add_item(content={"stem": "Q1"})

    await corpus.remove_item(artifact.id)

    # Item is archived
    removed = await registry.get(artifact.id)
    assert removed is not None
    assert removed.status == ArtifactStatus.ARCHIVED

    # Archived items still show in unfiltered query
    all_items = await corpus.get_items()
    assert len(all_items) == 1

    # But not in draft-filtered query
    draft_items = await corpus.get_items(status=ArtifactStatus.DRAFT)
    assert len(draft_items) == 0


@pytest.mark.asyncio
async def test_get_summary(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """Summary includes correct counts and metadata."""
    corpus = await ArtifactCorpus.create(registry, config)
    await corpus.add_item(content={"stem": "Q1"})
    await corpus.add_item(content={"stem": "Q2"})

    summary = await corpus.get_summary()

    assert summary["corpus_id"] == corpus.id
    assert summary["corpus_name"] == "Chapter 1 Quiz"
    assert summary["corpus_type"] == "quiz_bank"
    assert summary["item_type"] == "quiz_question"
    assert summary["total_items"] == 2
    assert summary["status_breakdown"]["draft"] == 2
    assert summary["corpus_status"] == "draft"


@pytest.mark.asyncio
async def test_end_to_end_flow(registry: ArtifactRegistry, config: CorpusConfig) -> None:
    """Full flow: create corpus, add items, finalize."""
    corpus = await ArtifactCorpus.create(registry, config)

    a1, _ = await corpus.add_item(content={"stem": "Q1", "answer": "A1"})
    a2, _ = await corpus.add_item(content={"stem": "Q2", "answer": "A2"})
    a3, _ = await corpus.add_item(content={"stem": "Q3", "answer": "A3"})

    assert await corpus.count() == 3

    summary = await corpus.get_summary()
    assert summary["total_items"] == 3

    finalized = await corpus.finalize()
    assert finalized.status == ArtifactStatus.APPROVED
    assert finalized.content["item_count"] == 3
    assert len(finalized.content["item_ids"]) == 3
