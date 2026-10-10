"""A backend instance can refuse an operation before the shared body of it runs.

The bases and the vector and bulk-embed mixins hold one body for each of these
operations, and a backend inherits it rather than restating it -- the parity
guards forbid a backend's own copy. So a backend that cannot perform one (a
Postgres table read in place, which nothing may write) cannot refuse it by
overriding the method. It overrides ``_refuse_operation`` instead, which every
shared body calls before anything else: before a row is read, and before a
caller's embedding function is spent on records that will never be stored.

The memory backends are the subject because they inherit the most shared
bodies, and because nothing in them refuses anything until told to.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from dataknobs_common.exceptions import OperationError
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.database import AsyncDatabase, SyncDatabase
from dataknobs_data.records import Record
from dataknobs_data.vector.bulk_embed_mixin import AsyncBulkEmbedMixin, BulkEmbedMixin
from dataknobs_data.vector.mixins import AsyncVectorOperationsMixin, SyncVectorOperationsMixin

#: The layer whose bodies a backend inherits.
SHARED = (
    AsyncDatabase,
    SyncDatabase,
    AsyncVectorOperationsMixin,
    SyncVectorOperationsMixin,
    AsyncBulkEmbedMixin,
    BulkEmbedMixin,
)

#: The operations whose shared bodies ask first: every write, what creates or
#: drops an index, and the vector surface.
GATED = frozenset(
    {
        "create", "update", "delete", "upsert", "clear",
        "create_batch", "upsert_batch", "delete_batch", "update_batch",
        "stream_write", "bulk_embed_and_store", "update_vector", "delete_from_index",
        "transaction", "begin_transaction",
        "enable_vector_support", "create_vector_index", "drop_vector_index",
        "vector_search", "hybrid_search", "get_vector_index_stats",
    }
)  # fmt: skip


class _Refusing:
    def _refuse_operation(self, operation: str) -> None:
        raise OperationError(f"{operation} is refused here")


class RefusingAsync(_Refusing, AsyncMemoryDatabase):
    pass


class RefusingSync(_Refusing, SyncMemoryDatabase):
    pass


def _shared_bodies(cls: type) -> list[str]:
    """The gated operations ``cls`` takes from the shared layer, unrestated."""
    names = []
    for name in sorted(GATED):
        owner = next((k for k in cls.__mro__ if name in k.__dict__), None)
        if owner in SHARED and not getattr(owner.__dict__[name], "__isabstractmethod__", False):
            names.append(name)
    return names


def _call(method: Any) -> None:
    """Call with a placeholder per required argument, and drive what comes back."""
    arguments = [
        None
        for p in inspect.signature(method).parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    result = method(*arguments)
    if inspect.isawaitable(result):
        asyncio.run(_await(result))
    elif inspect.isasyncgen(result):
        asyncio.run(result.__anext__())
    elif hasattr(result, "__aenter__"):
        asyncio.run(_enter(result))
    elif hasattr(result, "__enter__"):
        with result:
            pass


async def _await(result: Any) -> Any:
    return await result


async def _enter(manager: Any) -> None:
    async with manager:
        pass


@pytest.mark.parametrize("cls", [RefusingAsync, RefusingSync], ids=lambda c: c.__name__)
def test_the_memory_backends_inherit_shared_bodies_to_test(cls: type) -> None:
    """A floor, so the sweep below cannot pass by finding nothing."""
    assert {"bulk_embed_and_store", "update_vector", "vector_search"} <= set(_shared_bodies(cls))


@pytest.mark.parametrize("cls", [RefusingAsync, RefusingSync], ids=lambda c: c.__name__)
def test_every_shared_body_asks_before_it_runs(cls: type) -> None:
    """Called with placeholders, a body that ran first would fail on them instead."""
    db = cls()
    for name in _shared_bodies(cls):
        with pytest.raises(OperationError, match=rf"^{name} is refused here$"):
            _call(getattr(db, name))


@pytest.mark.parametrize("cls", [RefusingAsync, RefusingSync], ids=lambda c: c.__name__)
def test_a_refused_bulk_embed_spends_no_embedding(cls: type) -> None:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    db = cls()
    with pytest.raises(OperationError, match="bulk_embed_and_store"):
        _call_with(db.bulk_embed_and_store, [Record({"text": "a"})], "text", embedding_fn=embed)
    assert calls == []


@pytest.mark.parametrize("cls", [AsyncMemoryDatabase, SyncMemoryDatabase], ids=lambda c: c.__name__)
def test_by_default_nothing_is_refused(cls: type) -> None:
    db = cls()
    stored = _call_with(
        db.bulk_embed_and_store,
        [Record({"text": "a"})],
        "text",
        embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    assert len(stored) == 1


def _call_with(method: Any, *args: Any, **kwargs: Any) -> Any:
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(_await(result))
    return result
