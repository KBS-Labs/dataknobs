"""An ``embedding_fn`` may be any callable shape, including an object.

An embedder holds a model handle, so the natural way to write one is a class
with an ``async def __call__``. Every embedding site in this package asked
:func:`asyncio.iscoroutinefunction`, which answers for *functions* and reports
such an object as **sync** --- and then handed it to :func:`asyncio.to_thread`,
which called it in a worker thread and returned the coroutine.

Nothing raised anywhere along that path. Measured before the fix:

* ``VectorTextSynchronizer`` logged "Embedding function returned unexpected
  type: <class 'coroutine'>" and returned ``(False, [])``. Wrong, but loud
  enough to notice, and nothing was stored.
* ``IncrementalVectorizer`` had no such check. It wrote the **coroutine
  object** into the record as the vector value and persisted it.

Seven copies of the branch, one wrong answer in all seven. The branch now
lives in ``vector/embedding_fn.py`` over
:func:`~dataknobs_common.callbacks.is_async_callable`, and the sync/async
classification is made in one place.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.vector.embedding_fn import call_embedding_fn
from dataknobs_data.vector.migration import IncrementalVectorizer
from dataknobs_data.vector.sync import VectorTextSynchronizer


def _vector_of(text: str) -> np.ndarray:
    """First component is the text length, so a vector names its own source."""
    return np.array([float(len(text)), 1.0, 2.0])


class AsyncEmbedderObject:
    """The shape the classification got wrong."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, text: str) -> np.ndarray:
        self.calls += 1
        return _vector_of(text)


class SyncEmbedderObject:
    """A companion shape: an object whose ``__call__`` is an ordinary def."""

    def __init__(self) -> None:
        self.calls = 0
        self.threads: set[int] = set()

    def __call__(self, text: str) -> np.ndarray:
        self.calls += 1
        self.threads.add(threading.get_ident())
        return _vector_of(text)


async def _async_embed(text: str) -> np.ndarray:
    return _vector_of(text)


def _sync_embed(text: str) -> np.ndarray:
    return _vector_of(text)


def _sync_def_returning_coroutine(text: str) -> object:
    """The fifth shape, and the one this file's own enumeration was missing.

    A plain ``def`` that returns a coroutine rather than awaiting it --- a
    lambda over an async embedder, a thin wrapper someone forgot to mark
    ``async``. ``is_async_callable`` answers for the *callable*, and this
    callable really is synchronous, so it correctly says sync and the offload
    correctly runs it on a worker thread. What comes back is a coroutine.

    Which is the same garbage value described at the top of this file, reached
    by a different route: the classification is right and the *result* still
    has to be re-examined. Four shapes were enumerated here and this was not
    one of them, which is how the batch dispatch came to handle it while the
    per-text one did not.
    """
    return _async_embed(text)


async def _corpus() -> AsyncMemoryDatabase:
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    await db.create(Record(data={"content": "the document"}))
    return db


class TestTheSharedCaller:
    """``call_embedding_fn`` on each shape, driven directly."""

    @pytest.mark.parametrize(
        "embedding_fn",
        [
            _async_embed,
            _sync_embed,
            AsyncEmbedderObject(),
            SyncEmbedderObject(),
            _sync_def_returning_coroutine,
        ],
        ids=[
            "async-function",
            "sync-function",
            "async-object",
            "sync-object",
            "sync-function-returning-coroutine",
        ],
    )
    async def test_every_shape_yields_the_embedding(self, embedding_fn: object) -> None:
        result = await call_embedding_fn(embedding_fn, "abcd")

        assert isinstance(result, np.ndarray), f"got {type(result).__name__}"
        assert result[0] == pytest.approx(4.0)

    async def test_a_sync_callable_runs_off_the_event_loop(self) -> None:
        """The reason the sync branch offloads: embedding is not free.

        Pins that the shared caller kept the offload rather than collapsing
        both branches into a direct call.
        """
        embedder = SyncEmbedderObject()

        await call_embedding_fn(embedder, "abcd")

        assert embedder.threads and threading.get_ident() not in embedder.threads

    async def test_an_async_callable_honours_its_timeout(self) -> None:
        async def slow(text: str) -> np.ndarray:
            await asyncio.sleep(10.0)
            return _vector_of(text)

        with pytest.raises(TimeoutError):
            await call_embedding_fn(slow, "abcd", timeout=0.05)


class TestTheSynchronizerAcceptsAnObject:
    """``VectorTextSynchronizer`` returned ``(False, [])`` on this shape."""

    async def test_an_async_callable_object_produces_a_vector(self) -> None:
        db = await _corpus()
        embedder = AsyncEmbedderObject()
        synchronizer = VectorTextSynchronizer(
            database=db,
            embedding_fn=embedder,
            text_fields=["content"],
            vector_field="embedding",
        )

        result = await synchronizer.sync_all()

        assert embedder.calls == 1
        assert result["updated"] == 1, f"sync_all reported {result}"
        stored = (await db.all())[0]
        assert next(iter(stored.fields["embedding"].value)) == pytest.approx(len("the document"))

    async def test_an_async_function_still_works(self) -> None:
        """A companion: the shape that already worked."""
        db = await _corpus()
        synchronizer = VectorTextSynchronizer(
            database=db,
            embedding_fn=_async_embed,
            text_fields=["content"],
            vector_field="embedding",
        )

        result = await synchronizer.sync_all()

        assert result["updated"] == 1


class TestTheVectorizerStoresAVectorRatherThanACoroutine:
    """The corrupting case: no type check stood between it and the database."""

    async def test_the_stored_value_is_numeric(self) -> None:
        db = await _corpus()
        embedder = AsyncEmbedderObject()
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=embedder,
            text_fields=["content"],
            vector_field="embedding",
        )

        record = (await db.all())[0]
        await vectorizer._process_record(record)

        stored = (await db.all())[0]
        value = stored.get_value("embedding")
        assert not asyncio.iscoroutine(value), "a coroutine object was stored as the vector"
        assert value is not None
        assert all(isinstance(component, float) for component in value)
        assert value[0] == pytest.approx(len("the document"))
