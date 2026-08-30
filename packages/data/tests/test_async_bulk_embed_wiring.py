"""``bulk_embed_and_store`` on an async backend actually stores something.

``AsyncBulkEmbedMixin`` existed and was mixed into nothing. All four async
backends inherited the **sync** ``BulkEmbedMixin`` instead, so the method was
not a coroutine function and its ``self.exists`` / ``self.update`` /
``self.create`` calls were made without ``await``.

That fails in the quietest way available. A coroutine object is truthy, so
``if record.id and self.exists(record.id)`` takes the update branch on a record
that does not exist; ``self.update(...)`` returns another coroutine that is
never awaited and never runs; and ``self.create(...)`` returns a coroutine that
is appended to the result list in place of an id. Measured on
``AsyncMemoryDatabase`` before the fix: the call returns
``['coroutine', 'coroutine']`` and the database holds **zero** records. No
exception is raised at any point.

The two mixins were ~100-line near-copies differing only in their ``await``s,
which is why nothing looked wrong at the import site. The shared body now lives
in module-level helpers both call, so the remaining difference between them is
the awaiting --- and a future divergence has nowhere to hide.
"""

from __future__ import annotations

import inspect
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.file import AsyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.vector.content import (
    CONTENT_HASH_KEY,
    FIELD_SEPARATOR_KEY,
    SOURCE_FIELDS_KEY,
    compute_content_hash,
)

ASYNC_BACKENDS = ["memory", "file", "sqlite"]


def _embed(texts: list[str]) -> np.ndarray:
    """A deterministic embedding whose first component is the text length."""
    return np.array([[float(len(t)), 1.0, 2.0] for t in texts])


async def _aembed(texts: list[str]) -> np.ndarray:
    """The async form, which only the async mixin can drive."""
    return _embed(texts)


async def _make_async_db(kind: str, root: Path) -> Any:
    if kind == "memory":
        return AsyncMemoryDatabase()
    if kind == "file":
        return AsyncFileDatabase({"path": str(root / "records.json")})
    db = AsyncSQLiteDatabase({"path": str(root / "records.db")})
    await db.connect()
    return db


@pytest.fixture(params=ASYNC_BACKENDS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    with tempfile.TemporaryDirectory() as d:
        db = await _make_async_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


class TestTheAsyncMethodIsAwaitable:
    """The structural half: an async backend's method is a coroutine function.

    This is what the wiring bug actually was, and it is checkable without
    running anything --- which is the point. It failed for every async backend
    while every behavioural vector test in the suite went on passing, because
    nothing called this method on an async backend.
    """

    @pytest.mark.parametrize(
        "cls",
        [AsyncMemoryDatabase, AsyncFileDatabase, AsyncSQLiteDatabase],
        ids=lambda c: c.__name__,
    )
    def test_resolves_to_a_coroutine_function(self, cls: type) -> None:
        assert inspect.iscoroutinefunction(cls.bulk_embed_and_store), (
            f"{cls.__name__}.bulk_embed_and_store is not awaitable; it resolves to "
            f"{next(k.__name__ for k in cls.__mro__ if 'bulk_embed_and_store' in k.__dict__)}"
        )


class TestTheRecordsAreActuallyStored:
    """The behavioural half: the write lands, and the ids come back."""

    async def test_new_records_are_created(self, async_db: Any) -> None:
        records = [Record(data={"title": "alpha"}), Record(data={"title": "bravo!"})]

        ids = await async_db.bulk_embed_and_store(records, "title", embedding_fn=_embed)

        assert all(isinstance(i, str) for i in ids), f"not ids: {[type(i).__name__ for i in ids]}"
        stored = await async_db.all()
        assert len(stored) == 2, "bulk_embed_and_store stored nothing"

    async def test_the_stored_vector_is_the_embedding(self, async_db: Any) -> None:
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedding_fn=_embed
        )

        stored = (await async_db.all())[0]
        field = stored.fields["embedding"]
        # First component is the text length, which pins that the vector came
        # from this record's text rather than from anywhere else.
        assert next(iter(field.value)) == pytest.approx(len("alpha"))

    async def test_an_async_embedding_fn_is_awaited(self, async_db: Any) -> None:
        """The capability the sync mixin cannot provide at all.

        Handed to the sync mixin, an async ``embedding_fn`` returns a coroutine
        that is indexed rather than awaited.
        """
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedding_fn=_aembed
        )

        stored = (await async_db.all())[0]
        assert next(iter(stored.fields["embedding"].value)) == pytest.approx(len("alpha"))

    async def test_an_existing_record_is_updated_not_duplicated(self, async_db: Any) -> None:
        """The ``exists`` branch, which the un-awaited call took unconditionally."""
        rid = await async_db.create(Record(data={"title": "alpha"}))
        stored = await async_db.read(rid)
        assert stored is not None

        ids = await async_db.bulk_embed_and_store([stored], "title", embedding_fn=_embed)

        assert ids == [rid]
        assert len(await async_db.all()) == 1, "the update stored a duplicate"


class TestTheDigestSurvivesTheSharedBody:
    """Extracting the shared body must not drop what the sync path records.

    A companion: it passes both before and after, and would fail if the
    refactor lost the content-hash metadata that makes a bulk-embedded vector
    comparable by a synchronizer.
    """

    async def test_metadata_describes_the_assembly(self, async_db: Any) -> None:
        await async_db.bulk_embed_and_store(
            [Record(data={"title": "alpha", "body": "bravo"})],
            ["title", "body"],
            embedding_fn=_embed,
            field_separator=" | ",
        )

        stored = (await async_db.all())[0]
        metadata = stored.fields["embedding"].metadata
        assert metadata[SOURCE_FIELDS_KEY] == ["title", "body"]
        assert metadata[FIELD_SEPARATOR_KEY] == " | "
        assert metadata[CONTENT_HASH_KEY] == compute_content_hash("alpha | bravo")


class TestTheSyncMixinStillWorks:
    """A companion for the extraction: the sync path is unchanged.

    Driven directly rather than through a backend, so it pins the mixin body
    and not a backend's storage conventions.
    """

    def test_sync_bulk_embed_attaches_a_described_vector(self) -> None:
        from dataknobs_data.vector.bulk_embed_mixin import BulkEmbedMixin

        class _Store(BulkEmbedMixin):
            def __init__(self) -> None:
                self.written: dict[str, Record] = {}

            def exists(self, id: str) -> bool:
                return id in self.written

            def update(self, id: str, record: Record) -> bool:
                self.written[id] = record
                return True

            def create(self, record: Record) -> str:
                self.written["minted"] = record
                return "minted"

        store = _Store()
        ids = store.bulk_embed_and_store(
            [Record(data={"title": "alpha"})], "title", embedding_fn=_embed
        )

        assert ids == ["minted"]
        field = store.written["minted"].fields["embedding"]
        assert isinstance(field, VectorField)
        assert field.metadata[CONTENT_HASH_KEY] == compute_content_hash("alpha")
