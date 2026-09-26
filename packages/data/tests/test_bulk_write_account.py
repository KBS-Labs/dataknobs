"""A bulk write that fails partway says what reached the store.

Every ``bulk_embed_and_store`` commits in pieces. The vector store slices its
input by its own ``batch_size`` and calls ``add_vectors`` once per slice; the
two database lanes create or update one record at a time. A raise from the
embedder or the backend partway through leaves the earlier pieces committed,
and the method's return value --- the only account it gave --- never arrives.

``SemanticIndex.build()`` promised an account anyway, and computed it one
level too high: it counted its own flushes of ``BUILD_BATCH_SIZE`` items, each
of which is one ``bulk_embed_and_store`` call and several commits. So a build
failing on the 221st of 250 items reported ``0`` written over a store holding 200,
and its message said the partial batch in hand "was never sent" --- when most
of it had been.

The fix is a keyword-only ``on_stored`` callback on all three lanes, called
once per commit with the ids that commit made durable. ``build()`` counts
those instead of its flushes.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import AsyncIterator, Sequence
from typing import Any

import numpy as np
import pytest

from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.index import CallableSource, IndexItem
from dataknobs_common.testing import assert_twins_agree

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import SemanticIndex
from dataknobs_data.vector.bulk_embed_mixin import AsyncBulkEmbedMixin, BulkEmbedMixin
from dataknobs_data.vector.mixins import AsyncVectorOperationsMixin, SyncVectorOperationsMixin
from dataknobs_data.vector.stores.base import VectorStore
from dataknobs_data.vector.stores.memory import MemoryVectorStore

DIMENSIONS = 8


def _id(n: int) -> str:
    return f"item-{n:04d}"


def _text(n: int) -> str:
    return f"text {n:04d}"


class _FailsOn(DeterministicEmbedder):
    """A real embedder that refuses one named text, the way a provider does.

    A provider refuses a whole request when any text in it is too long, so
    the batch containing the poisoned text fails and every batch before it
    succeeds --- which is the shape the defect needs.
    """

    def __init__(self, poison: str, error: Exception | None = None) -> None:
        super().__init__(dimensions=DIMENSIONS)
        self.poison = poison
        self.error = error if error is not None else ValidationError("input too long")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.poison in texts:
            raise self.error
        return await super().embed(texts)


def _items(count: int) -> CallableSource:
    return CallableSource(lambda: [IndexItem(id=_id(n), text=_text(n)) for n in range(count)])


async def _store() -> MemoryVectorStore:
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    return store


# --------------------------------------------------------------------- #
# SemanticIndex.build(): the account a failed build gives
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("items", "fails_on", "stored"),
    [
        # Inside the first flush: two of the store's sub-batches committed
        # before the third failed. It reported 0.
        (250, 220, 200),
        # Inside a later flush, with the real flush size rather than a patched
        # one, because the defect lives in the interaction of the two sizes.
        # It reported 1000: the completed flush, and none of the second.
        (1250, 1120, 1100),
    ],
    ids=["first-flush", "later-flush"],
)
async def test_a_failed_build_reports_what_the_store_holds(
    items: int, fails_on: int, stored: int
) -> None:
    store = await _store()
    try:
        index = SemanticIndex(_items(items), _FailsOn(_text(fails_on)), store)
        with pytest.raises(OperationError) as failed:
            await index.build()

        assert await store.count() == stored
        assert failed.value.context["written"] == stored
    finally:
        await store.close()


async def test_a_failed_write_names_what_did_not_arrive_and_where_to_resume() -> None:
    """The ids ``build()`` held that the store does not have.

    They are a contiguous run starting at the first unstored id, because the
    store commits a prefix of what it was handed. The ids are the source's and
    the store upserts, so resuming or rebuilding from there duplicates nothing.
    """
    store = await _store()
    try:
        with pytest.raises(OperationError) as failed:
            await SemanticIndex(_items(250), _FailsOn(_text(220)), store).build()

        context = failed.value.context
        assert context["unstored"] == 50
        assert context["first_unstored"] == _id(200)
        assert context["last_unstored"] == _id(249)
        assert _id(200) in str(failed.value)
        assert "never sent" not in str(failed.value)

        written = await SemanticIndex(
            _items(250), DeterministicEmbedder(dimensions=DIMENSIONS), store
        ).build()
        assert written == 250
        assert await store.count() == 250
    finally:
        await store.close()


async def test_a_failed_source_counts_what_it_read_and_never_sent() -> None:
    """The one case where the old sentence was true, and still is."""

    async def five_then_trouble() -> AsyncIterator[IndexItem]:
        for n in range(5):
            yield IndexItem(id=_id(n), text=_text(n))
        raise OperationError("the source's backend went away mid-stream")

    store = await _store()
    try:
        index = SemanticIndex(
            CallableSource(five_then_trouble), DeterministicEmbedder(dimensions=DIMENSIONS), store
        )
        with pytest.raises(OperationError, match="never sent") as failed:
            await index.build()

        context = failed.value.context
        assert context["written"] == 0
        assert context["unstored"] == 5
        assert context["first_unstored"] == _id(0)
        assert context["last_unstored"] == _id(4)
        assert await store.count() == 0
    finally:
        await store.close()


async def test_a_build_that_succeeds_names_nothing_unstored() -> None:
    """The count on success is unchanged: what was handed to the store."""
    store = await _store()
    try:
        assert await SemanticIndex(_items(250), _FailsOn("absent"), store).build() == 250
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# VectorStore.bulk_embed_and_store: the seam itself
# --------------------------------------------------------------------- #


async def test_the_store_reports_each_commit_and_raises_the_original_error() -> None:
    """One call per ``add_vectors``, and the exception is not rewrapped.

    ``ValidationError`` is what a context-length refusal is (the provider
    package's ``ContextLengthExceededError`` subclasses it), so a caller
    catching it has to keep catching it.
    """
    store = await _store()
    error = ValidationError("input too long")
    reported: list[list[str]] = []
    try:
        with pytest.raises(ValidationError) as failed:
            await store.bulk_embed_and_store(
                [_text(n) for n in range(250)],
                ids=[_id(n) for n in range(250)],
                embedder=_FailsOn(_text(220), error),
                on_stored=reported.append,
            )

        assert failed.value is error
        assert reported == [
            [_id(n) for n in range(100)],
            [_id(n) for n in range(100, 200)],
        ]
        assert await store.count() == 200
    finally:
        await store.close()


async def test_the_reported_ids_are_a_prefix_of_the_ids_given_in_order() -> None:
    store = await _store()
    ids = [_id(n) for n in reversed(range(230))]
    reported: list[str] = []

    async def record(stored: list[str]) -> None:
        reported.extend(stored)

    try:
        returned = await store.bulk_embed_and_store(
            [_text(n) for n in range(230)],
            ids=ids,
            embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
            batch_size=40,
            on_stored=record,
        )
        assert reported == ids
        assert returned == ids
    finally:
        await store.close()


async def test_an_empty_input_reports_nothing() -> None:
    store = await _store()
    reported: list[list[str]] = []
    try:
        await store.bulk_embed_and_store(
            [],
            embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
            on_stored=reported.append,
        )
        assert reported == []
    finally:
        await store.close()


async def test_an_async_callback_runs_on_the_loop_and_a_sync_one_off_it() -> None:
    """A per-commit hook is the consumer's chance to *do* something.

    A checkpoint write is the obvious use, so a synchronous callback runs on a
    worker thread rather than stalling the loop, and an ``async def`` is
    awaited where it is, with no hop.
    """
    store = await _store()
    loop_thread = threading.get_ident()
    seen: dict[str, set[int]] = {"sync": set(), "async": set()}

    def sync_callback(stored: list[str]) -> None:
        seen["sync"].add(threading.get_ident())

    async def async_callback(stored: list[str]) -> None:
        seen["async"].add(threading.get_ident())

    try:
        for callback in (sync_callback, async_callback):
            await store.bulk_embed_and_store(
                [_text(n) for n in range(150)],
                ids=[_id(n) for n in range(150)],
                embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
                on_stored=callback,
            )

        assert seen["async"] == {loop_thread}
        assert seen["sync"] and loop_thread not in seen["sync"]
    finally:
        await store.close()


async def test_a_callback_that_raises_propagates_and_its_commit_stands() -> None:
    """The commit happened before the callback was told; the failure is the caller's."""
    store = await _store()
    told: list[list[str]] = []

    def refuse(stored: list[str]) -> None:
        told.append(stored)
        raise RuntimeError("checkpoint sink unavailable")

    try:
        with pytest.raises(RuntimeError, match="checkpoint sink"):
            await store.bulk_embed_and_store(
                [_text(n) for n in range(250)],
                ids=[_id(n) for n in range(250)],
                embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
                on_stored=refuse,
            )

        assert told == [[_id(n) for n in range(100)]]
        assert await store.count() == 100
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# The database lanes: one commit per record
# --------------------------------------------------------------------- #


def _embedding_fn_failing_on(poison: str) -> Any:
    def embed(texts: list[str]) -> np.ndarray:
        if poison in texts:
            raise ValidationError("input too long")
        return np.ones((len(texts), DIMENSIONS), dtype=np.float32)

    return embed


def _records(count: int) -> list[Record]:
    return [Record(data={"body": _text(n)}) for n in range(count)]


def test_the_sync_database_lane_reports_each_record_it_wrote() -> None:
    db = SyncMemoryDatabase()
    reported: list[list[str]] = []

    with pytest.raises(ValidationError):
        db.bulk_embed_and_store(
            _records(250),
            "body",
            embedding_fn=_embedding_fn_failing_on(_text(220)),
            on_stored=reported.append,
        )

    assert all(len(one) == 1 for one in reported), "one call per record"
    stored = [one[0] for one in reported]
    assert len(stored) == 200
    assert sorted(stored) == sorted(record.id for record in db.all())
    assert all(db.read(record_id).fields.get("embedding") for record_id in stored)


async def test_the_async_database_lane_reports_each_record_it_wrote() -> None:
    db = AsyncMemoryDatabase()
    reported: list[list[str]] = []

    async def record(stored: list[str]) -> None:
        reported.append(stored)

    with pytest.raises(ValidationError):
        await db.bulk_embed_and_store(
            _records(250),
            "body",
            embedding_fn=_embedding_fn_failing_on(_text(220)),
            on_stored=record,
        )

    assert all(len(one) == 1 for one in reported), "one call per record"
    stored = [one[0] for one in reported]
    assert len(stored) == 200
    assert sorted(stored) == sorted(record.id for record in await db.all())
    for record_id in stored:
        read = await db.read(record_id)
        assert read is not None and read.fields.get("embedding")


# --------------------------------------------------------------------- #
# The shape of the seam, on every lane
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method",
    [
        VectorStore.bulk_embed_and_store,
        BulkEmbedMixin.bulk_embed_and_store,
        AsyncBulkEmbedMixin.bulk_embed_and_store,
        SyncVectorOperationsMixin.bulk_embed_and_store,
        AsyncVectorOperationsMixin.bulk_embed_and_store,
    ],
    ids=lambda method: method.__qualname__,
)
def test_on_stored_is_keyword_only_and_defaults_to_none(method: Any) -> None:
    parameter = inspect.signature(method).parameters.get("on_stored")
    assert parameter is not None, f"{method.__qualname__} takes no on_stored"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_the_two_database_lanes_are_twins() -> None:
    """The async lane adds ``embedder``; the callback is flavoured on each."""
    assert_twins_agree(
        BulkEmbedMixin.bulk_embed_and_store,
        AsyncBulkEmbedMixin.bulk_embed_and_store,
        async_only=["embedder"],
        flavour_typed=["embedding_fn", "on_stored"],
    )
