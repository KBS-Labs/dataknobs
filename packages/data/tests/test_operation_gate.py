"""A backend instance can refuse an operation before any body of it runs.

The bases and the vector and bulk-embed mixins hold one body for each of these
operations, which a backend inherits rather than restating -- the parity
guards forbid a backend's own copy of those -- and a backend also defines
bodies of its own (the memory backends' ``create``, ``update``, ``delete``).
An instance that cannot perform an operation (a Postgres table read in place,
which nothing may write) overrides ``_refuse_operation`` instead, and every
body asks it before anything else, wherever the body is defined: before a row
is read, and before a caller's embedding function is spent on records that
will never be stored.

The memory backends are the subject because they inherit the most shared
bodies and define several of their own, and because nothing in them refuses
anything until told to.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from dataknobs_common.exceptions import OperationError
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.database import AsyncDatabase, SyncDatabase
from dataknobs_data.operation_gate import GATED_OPERATIONS
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

#: The operations whose bodies ask first: every write, what creates or drops an
#: index, and the vector surface. Restated here, so a change to the gate's set
#: is a change to this test too.
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


# --------------------------------------------------------------------------
# A body a backend defines itself asks too
# --------------------------------------------------------------------------


def _gated_surface(cls: type) -> list[str]:
    """Every gated operation ``cls`` has a concrete body for, wherever it came from."""
    names = []
    for name in sorted(GATED):
        owner = next((k for k in cls.__mro__ if name in k.__dict__), None)
        if owner is not None and not getattr(owner.__dict__[name], "__isabstractmethod__", False):
            names.append(name)
    return names


def test_the_gate_names_the_operations_this_module_sweeps() -> None:
    assert GATED_OPERATIONS == GATED


@pytest.mark.parametrize("cls", [RefusingAsync, RefusingSync], ids=lambda c: c.__name__)
def test_the_memory_backends_define_bodies_of_their_own_to_test(cls: type) -> None:
    """A floor: the memory backends write through bodies the shared layer does not hold."""
    own = set(_gated_surface(cls)) - set(_shared_bodies(cls))
    assert {"create", "update", "delete", "stream_write"} <= own


@pytest.mark.parametrize("cls", [RefusingAsync, RefusingSync], ids=lambda c: c.__name__)
def test_every_body_asks_before_it_runs_whoever_defines_it(cls: type) -> None:
    """Bug: only the shared bodies asked. A backend's own ``create`` ran, so an
    instance refusing writes stored a record through the method it had not
    inherited.
    """
    db = cls()
    for name in _gated_surface(cls):
        with pytest.raises(OperationError, match=rf"^{name} is refused here$"):
            _call(getattr(db, name))


def test_an_override_that_skips_super_still_asks() -> None:
    """The gate is installed on every class that defines a body, not inherited
    from one that applied it, so an override reaching no other body is gated.
    """
    ran: list[str] = []

    class Overriding(RefusingSync):
        def create(self, record: Record) -> str:
            ran.append("create")
            return "id"

    with pytest.raises(OperationError, match=r"^create is refused here$"):
        Overriding().create(Record({"a": 1}))
    assert ran == []


def test_a_gated_body_keeps_its_flavour() -> None:
    assert inspect.iscoroutinefunction(AsyncMemoryDatabase.create)
    assert not inspect.iscoroutinefunction(SyncMemoryDatabase.create)
    assert inspect.signature(AsyncMemoryDatabase.create) == inspect.signature(
        inspect.unwrap(AsyncMemoryDatabase.create)
    )


def test_a_gated_operation_defined_as_an_async_generator_is_refused_at_definition() -> None:
    """A wrapper cannot ask before an async generator's body runs without
    becoming one itself; the class is refused rather than left ungated.
    """
    with pytest.raises(TypeError, match="vector_search"):

        class Streaming(AsyncMemoryDatabase):
            async def vector_search(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                yield None
