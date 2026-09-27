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
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.index import CallableSource, IndexItem
from dataknobs_common.testing import assert_twins_agree

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import SemanticIndex
from dataknobs_data.vector.semantic_index import BUILD_BATCH_SIZE
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
    # Each case is only the case its id names while the flush size sits
    # between the two failure points; past either, it silently tests the other.
    assert 220 < BUILD_BATCH_SIZE <= 1120 < 2 * BUILD_BATCH_SIZE
    store = await _store()
    try:
        index = SemanticIndex(_items(items), _FailsOn(_text(fails_on)), store)
        with pytest.raises(OperationError) as failed:
            await index.build()

        assert await store.count() == stored
        assert failed.value.context["written"] == stored
    finally:
        await store.close()


async def test_a_failed_write_names_what_did_not_arrive() -> None:
    """The ids ``build()`` held that the store does not have.

    They are a contiguous run starting at the first unstored id, because the
    store commits a prefix of what it was handed. The ids are the source's and
    the store upserts, so a rebuild duplicates nothing.
    """
    store = await _store()
    try:
        with pytest.raises(OperationError) as failed:
            await SemanticIndex(_items(250), _FailsOn(_text(220)), store).build()

        context = failed.value.context
        assert context["unstored"] == 50
        assert context["first_unstored"] == _id(200)
        assert context["last_unstored"] == _id(249)
        message = str(failed.value)
        assert _id(200) in message
        assert "never sent" not in message
        # `unstored` is what the build held, not the whole shortfall: nothing
        # past the last id it held was read at all.
        assert f"not read past {_id(249)!r}" in message

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


async def test_a_source_that_fails_before_yielding_does_not_claim_a_partial_build() -> None:
    """Nothing was read, so nothing was written: the store holds no part of it."""

    async def trouble_at_once() -> AsyncIterator[IndexItem]:
        raise OperationError("the source's backend was not there")
        yield  # pragma: no cover -- makes this an async generator

    store = await _store()
    try:
        index = SemanticIndex(
            CallableSource(trouble_at_once), DeterministicEmbedder(dimensions=DIMENSIONS), store
        )
        with pytest.raises(OperationError) as failed:
            await index.build()

        context = failed.value.context
        assert context["written"] == 0
        assert context["unstored"] == 0
        assert await store.count() == 0
        message = str(failed.value)
        assert "partial" not in message
        assert "before yielding anything" in message
    finally:
        await store.close()


async def test_a_source_that_fails_between_flushes_names_the_partial_build() -> None:
    """Everything read was stored, but the source stopped short of its end."""

    async def one_flush_then_trouble() -> AsyncIterator[IndexItem]:
        for n in range(BUILD_BATCH_SIZE):
            yield IndexItem(id=_id(n), text=_text(n))
        raise OperationError("the source's backend went away mid-stream")

    store = await _store()
    try:
        index = SemanticIndex(
            CallableSource(one_flush_then_trouble),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
        )
        with pytest.raises(OperationError) as failed:
            await index.build()

        context = failed.value.context
        assert context["written"] == BUILD_BATCH_SIZE
        assert context["unstored"] == 0
        assert await store.count() == BUILD_BATCH_SIZE
        assert "partial build" in str(failed.value)
    finally:
        await store.close()


async def test_a_failed_build_offers_a_rebuild_not_a_resume() -> None:
    """``build()`` cannot start partway, so its error must not say it can.

    It takes no starting point, and a resume point is only a point in an order
    the source repeats --- ``RecordFieldSource`` streams ``stream_read`` with
    no sort, so a second run need not yield the ids in the first run's order,
    and skipping to ``first_unstored`` there skips items that were never
    stored. The remedy ``build()`` supports is a rebuild.
    """
    store = await _store()
    try:
        with pytest.raises(OperationError) as failed:
            await SemanticIndex(_items(250), _FailsOn(_text(220)), store).build()

        message = str(failed.value)
        assert "resume" not in message
        assert "rebuild" in message
    finally:
        await store.close()


class _ClosesBadly:
    """A structural source iterator whose close fails after a clean stream.

    ``IndexSource.stream_items`` is declared to return an ``AsyncIterator``,
    and ``aclosing_iter`` closes anything that has an ``aclose``. So a source
    can yield everything, have every item stored, and still fail the build on
    the way out.
    """

    def __init__(self, count: int) -> None:
        self._items = iter([IndexItem(id=_id(n), text=_text(n)) for n in range(count)])

    def __aiter__(self) -> _ClosesBadly:
        return self

    async def __anext__(self) -> IndexItem:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        raise OperationError("the source's connection failed to close")


class _ClosingSource:
    def __init__(self, count: int) -> None:
        self.count = count

    def stream_items(self) -> _ClosesBadly:
        return _ClosesBadly(self.count)

    def declares(self) -> frozenset[str]:
        return frozenset()


async def test_a_failure_closing_a_drained_source_does_not_claim_a_partial_build() -> None:
    store = await _store()
    try:
        index = SemanticIndex(
            _ClosingSource(250), DeterministicEmbedder(dimensions=DIMENSIONS), store
        )
        with pytest.raises(OperationError) as failed:
            await index.build()

        assert await store.count() == 250
        assert failed.value.context["written"] == 250
        assert failed.value.context["unstored"] == 0
        message = str(failed.value)
        assert "partial" not in message
        assert "closing the source" in message
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


def test_the_abstract_declarations_take_what_their_lanes_take() -> None:
    """A subclass author reads the abstract signature, so it must be the real one.

    Both concrete lanes take ``field_separator``, and the async lane's
    ``embedding_fn`` may be async; the declarations said neither.
    """
    for declared, lane in (
        (SyncVectorOperationsMixin.bulk_embed_and_store, BulkEmbedMixin.bulk_embed_and_store),
        (AsyncVectorOperationsMixin.bulk_embed_and_store, AsyncBulkEmbedMixin.bulk_embed_and_store),
    ):
        shape = [
            (p.name, p.kind, p.default, p.annotation)
            for p in inspect.signature(declared).parameters.values()
        ]
        assert shape == [
            (p.name, p.kind, p.default, p.annotation)
            for p in inspect.signature(lane).parameters.values()
        ], declared.__qualname__


# --------------------------------------------------------------------- #
# The database lanes: what they report, and what they refuse
# --------------------------------------------------------------------- #


class _AsyncCallable:
    """A stateful callback, written the way anything holding a handle is."""

    async def __call__(self, stored: list[str]) -> None:
        return None


async def _async_function(stored: list[str]) -> None:
    return None


@pytest.mark.parametrize(
    "callback", [_async_function, _AsyncCallable()], ids=["function", "object"]
)
def test_the_sync_database_lane_refuses_an_async_callback(callback: Any) -> None:
    """This lane cannot await, so an async callback would never run.

    Called without an ``await`` it returns a coroutine and raises nothing,
    so every report would be dropped with only a ``RuntimeWarning`` to show
    for it. Refused before anything is written.
    """
    db = SyncMemoryDatabase()

    with pytest.raises(TypeError, match="on_stored"):
        db.bulk_embed_and_store(
            _records(3),
            "body",
            embedding_fn=_embedding_fn_failing_on("absent"),
            on_stored=callback,
        )

    assert db.all() == []


async def test_the_async_database_lane_runs_a_sync_callback_off_the_loop() -> None:
    """The per-record thread hop the docstring prices is real, and on a worker."""
    db = AsyncMemoryDatabase()
    loop_thread = threading.get_ident()
    threads: set[int] = set()

    def record(stored: list[str]) -> None:
        threads.add(threading.get_ident())

    await db.bulk_embed_and_store(
        _records(3), "body", embedding_fn=_embedding_fn_failing_on("absent"), on_stored=record
    )

    assert threads and loop_thread not in threads


def test_a_raising_callback_on_the_sync_database_lane_propagates_and_its_record_stands() -> None:
    db = SyncMemoryDatabase()
    told: list[list[str]] = []

    def refuse(stored: list[str]) -> None:
        told.append(stored)
        raise RuntimeError("checkpoint sink unavailable")

    with pytest.raises(RuntimeError, match="checkpoint sink"):
        db.bulk_embed_and_store(
            _records(3), "body", embedding_fn=_embedding_fn_failing_on("absent"), on_stored=refuse
        )

    assert len(told) == 1
    assert [record.id for record in db.all()] == told[0]


async def test_a_raising_callback_on_the_async_database_lane_propagates_and_its_record_stands() -> (
    None
):
    db = AsyncMemoryDatabase()
    told: list[list[str]] = []

    async def refuse(stored: list[str]) -> None:
        told.append(stored)
        raise RuntimeError("checkpoint sink unavailable")

    with pytest.raises(RuntimeError, match="checkpoint sink"):
        await db.bulk_embed_and_store(
            _records(3), "body", embedding_fn=_embedding_fn_failing_on("absent"), on_stored=refuse
        )

    assert len(told) == 1
    assert [record.id for record in await db.all()] == told[0]


# A record deleted between the lane's existence check and its update.
#
# The lanes used to check `exists` and then `update`, ignoring update()'s
# `False` for a record that had gone, and reported the record written. They
# now call `upsert`, which the database already had for exactly this.
#
# The memory backends take their lock once per call and never yield inside
# one, so no second task or thread can be scheduled into the gap between two
# calls deterministically. These subclasses put the delete there instead:
# `exists` answers truthfully, then the record is gone, which is exactly what a
# concurrent writer does to a check-then-act. Everything else is the real
# backend.


class _SyncDeletedAfterCheck(SyncMemoryDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.raced: set[str] = set()

    def exists(self, id: str) -> bool:
        answer = super().exists(id)
        if answer and id not in self.raced:
            self.raced.add(id)
            self.delete(id)
        return answer


class _AsyncDeletedAfterCheck(AsyncMemoryDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.raced: set[str] = set()

    async def exists(self, id: str) -> bool:
        answer = await super().exists(id)
        if answer and id not in self.raced:
            self.raced.add(id)
            await self.delete(id)
        return answer


def _existing(count: int) -> list[Record]:
    return [Record(data={"body": _text(n)}, storage_id=_id(n)) for n in range(count)]


def test_the_sync_database_lane_never_reports_a_record_it_did_not_write() -> None:
    """``update()`` answers ``False`` for a record that has gone; that is not a write."""
    db = _SyncDeletedAfterCheck()
    for record in _existing(3):
        db.create(record)
    reported: list[str] = []

    db.bulk_embed_and_store(
        _existing(3),
        "body",
        embedding_fn=_embedding_fn_failing_on("absent"),
        on_stored=reported.extend,
    )

    # Every record is written and reported, whether or not the write went
    # through the check the delete races: a backend whose upsert is atomic
    # never asks `exists` at all, which closes the gap rather than surviving it.
    assert sorted(reported) == [_id(n) for n in range(3)]
    for record_id in reported:
        read = db.read(record_id)
        assert read is not None and read.fields.get("embedding"), record_id


async def test_the_async_database_lane_never_reports_a_record_it_did_not_write() -> None:
    db = _AsyncDeletedAfterCheck()
    for record in _existing(3):
        await db.create(record)
    reported: list[str] = []

    async def record(stored: list[str]) -> None:
        reported.extend(stored)

    await db.bulk_embed_and_store(
        _existing(3),
        "body",
        embedding_fn=_embedding_fn_failing_on("absent"),
        on_stored=record,
    )

    # Every record is written and reported, whether or not the write went
    # through the check the delete races: a backend whose upsert is atomic
    # never asks `exists` at all, which closes the gap rather than surviving it.
    assert sorted(reported) == [_id(n) for n in range(3)]
    for record_id in reported:
        read = await db.read(record_id)
        assert read is not None and read.fields.get("embedding"), record_id


# The memory backends' upsert never asks `exists`, so the two tests above show
# the gap closed, not survived. SQLite takes the base-class upsert, which does
# check and then update, so there the delete lands between the two calls inside
# upsert itself -- the race the lane has to come through, happening for real.


class _SyncSQLiteDeletedAfterCheck(SyncSQLiteDatabase):
    raced: set[str]

    def exists(self, id: str) -> bool:
        answer = super().exists(id)
        if answer and id not in self.raced:
            self.raced.add(id)
            self.delete(id)
        return answer


class _AsyncSQLiteDeletedAfterCheck(AsyncSQLiteDatabase):
    raced: set[str]

    async def exists(self, id: str) -> bool:
        answer = await super().exists(id)
        if answer and id not in self.raced:
            self.raced.add(id)
            await self.delete(id)
        return answer


def test_the_sync_database_lane_survives_a_delete_between_check_and_update(
    tmp_path: Path,
) -> None:
    db = _SyncSQLiteDeletedAfterCheck(config={"path": str(tmp_path / "records.db")})
    db.raced = set()
    db.connect()
    try:
        for record in _existing(3):
            db.create(record)
        reported: list[str] = []

        db.bulk_embed_and_store(
            _existing(3),
            "body",
            embedding_fn=_embedding_fn_failing_on("absent"),
            on_stored=reported.extend,
        )

        assert db.raced == {_id(n) for n in range(3)}, "the race did not happen"
        assert sorted(reported) == [_id(n) for n in range(3)]
        for record_id in reported:
            read = db.read(record_id)
            assert read is not None and read.fields.get("embedding"), record_id
    finally:
        db.close()


async def test_the_async_database_lane_survives_a_delete_between_check_and_update(
    tmp_path: Path,
) -> None:
    db = _AsyncSQLiteDeletedAfterCheck(config={"path": str(tmp_path / "records.db")})
    db.raced = set()
    await db.connect()
    try:
        for record in _existing(3):
            await db.create(record)
        reported: list[str] = []

        async def record(stored: list[str]) -> None:
            reported.extend(stored)

        await db.bulk_embed_and_store(
            _existing(3),
            "body",
            embedding_fn=_embedding_fn_failing_on("absent"),
            on_stored=record,
        )

        assert db.raced == {_id(n) for n in range(3)}, "the race did not happen"
        assert sorted(reported) == [_id(n) for n in range(3)]
        for record_id in reported:
            read = await db.read(record_id)
            assert read is not None and read.fields.get("embedding"), record_id
    finally:
        await db.close()
