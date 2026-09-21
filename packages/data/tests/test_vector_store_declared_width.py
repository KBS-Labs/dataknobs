"""A store's declared ``dimensions`` is compared to the vectors written to it.

An ``index:`` block's ``store:`` must carry a width, and the store refuses one
that is not positive or exceeds 65536 --- and until this guard, **nothing
compared it to a vector**. Measured before the fix, on the memory backend:
768-wide vectors into a store declaring 32, and 32-wide into one declaring
768, both built fifteen rows, both resolved, neither raised. The searches even
answered *correctly*, because both sides of the comparison use the same
wrong-width vectors. So the declaration was not merely unchecked, it was
**invisible** --- a required field whose wrongness could not be observed from
inside the deployment that wrote it.

``pgvector`` does check a width at initialize, and what it checks is the
*table's* declared column against the store's configuration: a different
comparison, in a different place. That is what made one document
portable-looking and not portable --- silent on the backend everyone develops
against and fatal on the one they deploy to.

The comparison runs on every backend, so it is asserted on every backend.
"""

from __future__ import annotations

import traceback
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest
from dataknobs_common.testing import (
    postgres_dsn,
    postgres_env_params,
    requires_chromadb,
    requires_faiss,
    requires_real_postgres,
)

from dataknobs_data.vector.stores.memory import MemoryVectorStore

if TYPE_CHECKING:
    from typing import Any

    from dataknobs_data.vector.stores.base import VectorStore

#: What every store below declares. Small, and the point is only that the
#: vectors written disagree with it.
DECLARED = 8


def _pg_connection_string() -> str:
    return postgres_dsn(postgres_env_params())


async def _make_memory(dimensions: int) -> VectorStore:
    return MemoryVectorStore({"dimensions": dimensions})


async def _make_faiss(dimensions: int) -> VectorStore:
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

    return FaissVectorStore({"dimensions": dimensions, "index_type": "flat"})


async def _make_chroma(dimensions: int) -> VectorStore:
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

    return ChromaVectorStore(
        {"dimensions": dimensions, "collection_name": f"width_{uuid.uuid4().hex[:8]}"}
    )


async def _make_pgvector(dimensions: int) -> VectorStore:
    from dataknobs_data.vector.stores.pgvector import PgVectorStore

    return PgVectorStore(
        {
            "connection_string": _pg_connection_string(),
            "dimensions": dimensions,
            "table_name": f"width_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
        }
    )


_BACKENDS = [
    pytest.param(_make_memory, id="memory"),
    pytest.param(_make_faiss, id="faiss", marks=requires_faiss),
    pytest.param(_make_chroma, id="chroma", marks=requires_chromadb),
    pytest.param(_make_pgvector, id="pgvector", marks=requires_real_postgres),
]

_Factory = "Callable[[int], Coroutine[Any, Any, VectorStore]]"


@pytest.fixture(params=_BACKENDS)
async def store_factory(request: pytest.FixtureRequest) -> Any:
    """The backend's constructor, un-called, so a test picks the width."""
    return request.param


async def _closing(store: VectorStore) -> VectorStore:
    await store.initialize()
    return store


@pytest.mark.parametrize("written", [4, 16], ids=["narrower", "wider"])
async def test_a_batch_that_disagrees_with_the_declaration_is_refused(
    store_factory: Any, written: int
) -> None:
    """Both directions, because both were measured and neither raised.

    Narrower and wider are one defect and not two: the store's declaration
    said one thing and the rows say another, and which side is larger is an
    accident of which process was reconfigured.
    """
    store = await _closing(await store_factory(DECLARED))
    try:
        vectors = np.eye(3, written, dtype=np.float32)
        with pytest.raises(ValueError, match="dimension"):
            await store.add_vectors(vectors, ids=["a", "b", "c"])
    finally:
        await store.close()


async def test_a_batch_that_agrees_is_written(store_factory: Any) -> None:
    """The positive control, and the reason the check is on the first vector.

    A guard that refused everything would pass the test above and break every
    store. This is what says the comparison is a comparison.
    """
    store = await _closing(await store_factory(DECLARED))
    try:
        written = await store.add_vectors(
            np.eye(3, DECLARED, dtype=np.float32), ids=["a", "b", "c"]
        )
        assert len(written) == 3
    finally:
        await store.close()


async def test_an_empty_batch_is_still_a_no_op(store_factory: Any) -> None:
    """The check sits **after** the emptiness guard, not in front of it.

    An empty batch has no first vector to measure, and it is a no-op rather
    than an error --- a contract four backends already implement through
    ``VectorStoreBase._is_empty_batch``. A width check reaching it first would
    turn *a comprehension filtered everything out* into a raise.
    """
    store = await _closing(await store_factory(DECLARED))
    try:
        assert await store.add_vectors([]) == []
        assert await store.add_vectors(np.array([])) == []
    finally:
        await store.close()


async def test_one_un_nested_vector_is_measured_as_one_vector(store_factory: Any) -> None:
    """A 1-D input is a single row, and its length is its width.

    ``_is_empty_batch`` already documents this shape as *correctly not empty*
    --- ``ndim == 1`` with ``len == dimensions`` --- and every backend
    reshapes it to ``(1, -1)``. So the width of a 1-D input is its own
    length, and reading ``len(vectors[0])`` there would measure a scalar.
    """
    store = await _closing(await store_factory(DECLARED))
    try:
        assert len(await store.add_vectors(np.ones(DECLARED, dtype=np.float32), ids=["a"])) == 1
        with pytest.raises(ValueError, match="dimension"):
            await store.add_vectors(np.ones(DECLARED + 1, dtype=np.float32), ids=["b"])
    finally:
        await store.close()


async def test_a_list_of_vectors_is_measured_the_same_way(store_factory: Any) -> None:
    """The list spelling, which is what a caller assembling a batch produces.

    ``add_vectors`` is annotated ``np.ndarray | list[np.ndarray]`` and every
    backend accepts a list of plain lists too. A check reading ``shape`` would
    see none of them.
    """
    store = await _closing(await store_factory(DECLARED))
    try:
        rows = [np.ones(DECLARED, dtype=np.float32), np.zeros(DECLARED, dtype=np.float32)]
        assert len(await store.add_vectors(rows, ids=["a", "b"])) == 2
        with pytest.raises(ValueError, match="dimension"):
            await store.add_vectors([[1.0] * (DECLARED + 2)], ids=["c"])
    finally:
        await store.close()


async def test_a_batch_the_guard_cannot_measure_is_left_to_the_backend(
    store_factory: Any,
) -> None:
    """Two shapes that are errors, and neither of them is a *width* error.

    A **ragged** batch does not convert to a rectangular array at all, and a
    **0-d** input is not a batch in any reading. Both raise, and both must
    raise from the backend's own conversion rather than from this guard:
    ``_is_empty_batch`` answers ``False`` for a 0-d input deliberately, *"so
    that the caller sees the backend's dimension error, which can say what
    shape was expected"*, and a guard pre-empting that with *the batch would
    not convert* would undo it.

    **Asserted on the traceback, not on the message.** That both inputs
    raise is true whether or not the guard steps aside --- the whole
    question is *who* raised, and only the frame list answers it. Removing
    either of the guard's two escape hatches puts ``_check_batch_width`` in
    the traceback, which is what this fails on.
    """
    store = await _closing(await store_factory(DECLARED))
    unmeasurable: list[Any] = [
        [[1.0] * DECLARED, [1.0] * (DECLARED + 1)],  # ragged
        np.float32(1.0),  # 0-d
    ]
    try:
        for payload in unmeasurable:
            with pytest.raises(Exception) as caught:
                await store.add_vectors(payload, ids=["a"])
            frames = [frame.name for frame in traceback.extract_tb(caught.value.__traceback__)]
            assert "_check_batch_width" not in frames, (
                f"the width guard raised for {payload!r}, which is not a width error; "
                f"the backend's own conversion says which row is wrong. Frames: {frames}"
            )
    finally:
        await store.close()
