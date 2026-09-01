"""Semantic dedup notices when its stored vectors came from another model.

Swap the embedding model and the vectors already in the store sit in a
different space from the ones queries are now embedded into. Similarity
between the two is meaningless, and ``check`` used to answer from it anyway:
``recommendation="unique"`` with an empty ``similar_items``, which is exactly
what genuinely new content returns. It fails **open** --- duplicates are
admitted --- and an admitted duplicate is silent by construction, because
nobody goes looking for the record that was correctly not flagged.

**Why the exact-hash lane does not cover this, and why the cheap detector
does not either.** ``check`` runs :meth:`_find_exact_match` first, so
byte-identical content never reaches the semantic pass at all. What reaches
it is the *near*-duplicate --- a paraphrase, a re-edit --- whose hash
genuinely differs. So a detector keyed on "equal ``content_hash``, low
similarity" cannot fire on the case that matters: by construction the hashes
differ there. Measured, not assumed --- see
``test_the_content_hash_lane_never_reaches_the_semantic_pass`` below. The
guard has to be the model identity itself.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector.content import MODEL_NAME_KEY
from dataknobs_data.vector.stores.memory import MemoryVectorStore

DIMENSIONS = 8


async def _checker(embedder: DeterministicEmbedder, store: MemoryVectorStore) -> DedupChecker:
    db = AsyncMemoryDatabase()
    await db.connect()
    return DedupChecker(
        db=db,
        config=DedupConfig(semantic_check=True),
        vector_store=store,
        embedder=embedder,
    )


class TestAModelSwapIsVisible:
    """Registering under one model and checking under another is reported."""

    async def test_the_mismatched_model_is_named_on_the_result(self) -> None:
        store = MemoryVectorStore(dimensions=DIMENSIONS)
        original = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-a")
        replacement = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-b")

        await (await _checker(original, store)).register({"body": "alpha beta"}, "r-1")
        # A *near* duplicate: a different hash, so it reaches the semantic
        # pass rather than being caught by the exact lane.
        result = await (await _checker(replacement, store)).check({"body": "alpha beta gamma"})

        assert result.mismatched_model_ids == ["model-a"]

    async def test_it_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A caller reading only logs still learns the answers are meaningless."""
        store = MemoryVectorStore(dimensions=DIMENSIONS)
        original = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-a")
        replacement = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-b")

        await (await _checker(original, store)).register({"body": "alpha beta"}, "r-1")
        with caplog.at_level(logging.WARNING, logger="dataknobs_data.dedup"):
            await (await _checker(replacement, store)).check({"body": "alpha beta gamma"})

        assert "model-a" in caplog.text
        assert "model-b" in caplog.text

    async def test_the_matching_model_reports_nothing(self) -> None:
        """No false alarm when the store and the checker agree."""
        store = MemoryVectorStore(dimensions=DIMENSIONS)
        embedder = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-a")

        await (await _checker(embedder, store)).register({"body": "alpha beta"}, "r-1")
        result = await (await _checker(embedder, store)).check({"body": "alpha beta gamma"})

        assert result.mismatched_model_ids == []


class TestVectorsWrittenBeforeThisKeyExisted:
    """A store predating the key must read as *unknown*, never as a mismatch.

    The same rule the vector lane follows, and for the same reason: an
    upgrade must not turn every existing vector into a warning. A vector
    written through the ``embedding_fn`` lane has no knowable identity either,
    so its key is **absent** rather than ``None``.
    """

    async def test_a_vector_with_no_model_recorded_is_not_a_mismatch(self) -> None:
        store = MemoryVectorStore(dimensions=DIMENSIONS)
        embedder = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-a")
        checker = await _checker(embedder, store)

        # What `register` wrote before this key existed.
        vectors = await embedder.embed(["alpha beta"])
        await store.add_vectors(
            vectors=[vectors[0]],
            ids=["r-1"],
            metadata=[{"text": "alpha beta", "content_hash": "whatever"}],
        )

        result = await checker.check({"body": "alpha beta gamma"})

        assert result.mismatched_model_ids == []

    async def test_the_callable_lane_records_no_model(self) -> None:
        """``embedding_fn`` carries no identity, so none is written."""
        store = MemoryVectorStore(dimensions=DIMENSIONS)
        embedder = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-a")

        async def embedding_fn(text: str) -> list[float]:
            return (await embedder.embed([text]))[0]

        db = AsyncMemoryDatabase()
        await db.connect()
        checker = DedupChecker(
            db=db,
            config=DedupConfig(semantic_check=True),
            vector_store=store,
            embedding_fn=embedding_fn,
        )
        await checker.register({"body": "alpha beta"}, "r-1")

        [(_, _, metadata)] = await store.search(
            query_vector=np.array((await embedder.embed(["alpha beta"]))[0], dtype=np.float32),
            k=1,
        )
        assert MODEL_NAME_KEY not in metadata


class TestWhyTheCheapDetectorWasRejected:
    """The measurement behind this module's docstring, pinned as a test.

    An equal ``content_hash`` with a low similarity score would be
    self-evident proof of two vector spaces, and needs no stored key --- but
    it can never be observed, because ``check`` returns on the exact-hash
    lane before the semantic pass runs. If that order ever changes, this
    fails and the cheap detector becomes worth revisiting.
    """

    async def test_the_content_hash_lane_never_reaches_the_semantic_pass(self) -> None:
        store = MemoryVectorStore(dimensions=DIMENSIONS)
        embedder = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="model-a")
        db = AsyncMemoryDatabase()
        await db.connect()
        checker = DedupChecker(
            db=db,
            config=DedupConfig(semantic_check=True),
            vector_store=store,
            embedder=embedder,
        )

        await checker.register({"body": "alpha beta"}, "r-1")
        result = await checker.check({"body": "alpha beta"})

        assert result.recommendation == "exact_duplicate"
        assert result.similar_items == []
        assert result.mismatched_model_ids == []
