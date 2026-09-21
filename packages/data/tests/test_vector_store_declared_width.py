"""A store's declared ``dimensions`` is compared to the vectors written to it.

A store refuses a declared width that is negative or exceeds 65536, and one
that is absent on a backend needing the number up front --- and until this
guard, **nothing compared a stated width to a vector**. Measured before the
fix, on the memory backend:
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


# --- A width that was never declared -----------------------------------
#
# ``dimensions`` defaults to ``0`` and has since the backend was written
# (``self.config.get("dimensions", 0)``, with the comment *"required for
# most stores"*). Nothing has ever refused that value: the method written
# to do it, ``VectorStoreBase._validate_dimensions``, is called from
# nowhere in the repository and has been dead since the commit that
# introduced it. So ``0`` is not a declaration of zero width, it is the
# absence of a declaration --- exactly as ``ChromaVectorStoreConfig``
# already documents it for its own subclass, where it resolves to 384.
#
# A guard that compares a batch to *that* is not comparing it to a
# declaration. It reports ``expected 0``, which names neither the config
# key nor the cause, and it refuses every write to a store whose width
# nobody stated --- including the one ``RAGKnowledgeBaseConfig``
# documents as supported, where *"an empty section does not fail: the
# factory falls back to an in-process store."*


async def test_a_store_that_declares_no_width_accepts_a_write() -> None:
    """No declaration is not a declaration of zero, and has never been one.

    The memory backend is where this is reachable: it is the only one that
    never reads ``dimensions`` for anything but ``get_stats``, which is why
    an undeclared width has always worked there and why it is the store the
    RAG path falls back to. ``faiss`` and ``pgvector`` both need the number
    to build a fixed-width structure and are covered below.
    """
    store = MemoryVectorStore({})
    assert store.dimensions == 0, "the sentinel this test is about"
    await store.initialize()
    try:
        written = await store.add_vectors(np.eye(3, 768, dtype=np.float32), ids=["a", "b", "c"])
        assert len(written) == 3
    finally:
        await store.close()


async def test_an_undeclared_store_is_not_held_to_the_sentinel(store_factory: Any) -> None:
    """Whatever a backend does with ``0``, it must not compare a batch to it.

    Faiss and pgvector refuse the sentinel outright, so memory and chroma
    are the two that reach a write. The assertion is written for all four
    anyway: it says *no store reports ``expected 0``*, which is the message
    that named neither the key nor the cause, and it stays true however a
    backend chooses to answer.

    Chroma used to be the interesting case here, resolving the sentinel to
    384 in ``__post_init__`` --- which passed this assertion while refusing
    the write for a different number the caller had also not typed. That
    resolution is gone; see
    ``test_a_chroma_store_that_states_no_width_does_not_invent_one``.
    """
    try:
        store = await store_factory(0)
    except ValueError as exc:
        assert "dimensions" in str(exc), "a backend that needs the width must name the key"
        return
    await store.initialize()
    try:
        await store.add_vectors(
            np.eye(2, store.dimensions or 768, dtype=np.float32), ids=["a", "b"]
        )
    except ValueError as exc:  # pragma: no cover - the regression this pins
        assert "expected 0" not in str(exc), f"the sentinel was compared to a vector: {exc}"
        raise
    finally:
        await store.close()


@pytest.mark.parametrize("declared", [-1, 65537], ids=["negative", "over-maximum"])
async def test_a_declaration_that_is_not_a_width_is_refused_at_construction(
    store_factory: Any, declared: int
) -> None:
    """The range check exists, names these two bounds, and runs nowhere.

    ``_validate_dimensions`` has been dead since it was written, so a
    negative width is accepted at construction and then compared against ---
    ``expected -1`` --- and one above the 65536 the method names reaches the
    backend. Neither value is a sentinel under any reading; both are wrong
    at the moment the config is read, which is where they should be refused.
    """
    with pytest.raises(ValueError, match=r"[Dd]imensions"):
        await store_factory(declared)


@pytest.mark.parametrize("factory", [pytest.param(_make_faiss, id="faiss", marks=requires_faiss)])
async def test_a_backend_that_needs_the_width_refuses_the_sentinel(factory: Any) -> None:
    """``faiss`` cannot defer the question, so it must ask it in words.

    The index is built at ``initialize`` from ``self.dimensions``, and
    ``faiss.IndexFlatL2(0)`` constructs happily --- it is the first ``add``
    that fails, with a bare ``AssertionError`` carrying no message at all.
    ``pgvector`` is the same shape one layer down: it writes
    ``vector(0)`` into a ``CREATE TABLE`` and Postgres refuses the type.
    A backend that needs the number states that when it reads the config.
    """
    with pytest.raises(ValueError, match="dimensions"):
        await factory(0)


# ---------------------------------------------------------------------------
# A width the caller did not state is not a width to hold them to
# ---------------------------------------------------------------------------
#
# ``ChromaVectorStoreConfig.__post_init__`` resolved the ``0`` sentinel to
# ``384``, documented as *"use the 384-dimension sentence-transformers
# default"* and introduced *"matching the legacy backend"*. The legacy
# backend did the same thing --- and in the legacy backend the value was
# **inert**: ``dimensions`` appeared three times in that whole file, all
# three of them in the assignment itself, and nothing ever compared it to a
# vector.
#
# The guard is what turns an inert default into a refusal. Measured: a caller
# who states no width at all and writes their own 768-wide vectors is now
# told ``expected 384, got 768`` --- citing a number nobody typed, on the
# config that states the least. That is the third appearance of one shape in
# this area, after the ``expected 0`` refusal and the phantom 1536 in
# ``VectorMemoryConfig``: a spelled default becoming a claim the consumer is
# then held to.
#
# So preserving the legacy value does not preserve the legacy behaviour --- it
# inverts it. ``0`` stays ``0`` here as everywhere else, and a chroma store
# that wants 384 enforced says 384.


@requires_chromadb
async def test_a_chroma_store_that_states_no_width_does_not_invent_one() -> None:
    """The reproducer: an unstated width used to become an enforced 384."""
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

    store = ChromaVectorStore({"collection_name": f"nowidth_{uuid.uuid4().hex[:8]}"})
    await store.initialize()
    try:
        assert store.dimensions == 0, "nothing was declared, so nothing is declared"
        written = await store.add_vectors(np.ones((2, 768), dtype=np.float32), ids=["a", "b"])
        assert len(written) == 2
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# The other door into the same collection
# ---------------------------------------------------------------------------
#
# Every case above writes through ``add_vectors``, where the caller supplies
# the vectors and the guard measures what they supplied. ``ChromaVectorStore``
# has a second write door --- ``add_documents`` --- where **the embedding
# function chooses the width** and the caller never sees a vector. It is the
# one backend that can write a row the caller did not shape, so it is the one
# backend where a declaration can be overridden rather than contradicted.
#
# Measured against the unguarded path, on one store declaring 8 whose
# embedding function makes 384:
#
#   add_vectors at 384    -> REFUSED, "expected 8, got 384"
#   add_documents         -> ACCEPTED, two rows written
#   width actually stored -> 384, while the store still declares 8
#
# That asymmetry is worse than no guard: a caller who has watched
# ``add_vectors`` refuse has every reason to believe the declaration is
# enforced, and the other door writes past it in silence.
#
# **Chroma pins a collection's width at its first write.** Measured: a second
# write at another width is refused by chroma itself, across both doors --- so
# ``documents`` after an 8-wide vector is refused with *"Collection expecting
# embedding with dimension of 8, got 384"*. The hole is therefore exactly *a
# collection whose first write arrives as documents*: chroma then enforces the
# width it inferred, for that collection's whole life, and the store's
# ``dimensions`` is a number nothing will ever honour.
#
# Chroma offers no way to declare the width up front --- its
# ``CreateCollectionConfiguration`` takes ``hnsw``, ``spann`` and
# ``embedding_function``, and nothing else --- and asking an embedding
# function its width means running it, which for a hosted one is a billable
# call. So the comparison happens where the width first becomes knowable for
# free: against a row the collection already holds.

#: Wide enough to differ from ``DECLARED`` and small enough to stay cheap.
EF_WIDTH = 16


def _chroma_docs_store(**overrides: object) -> object:
    """A chroma store whose embedding function is real, local and sized here.

    Never chromadb's ``"default"``: that fetches ~166 MB of ONNX weights on
    first use, so a suite using it passes on a warm developer machine and
    *fails* --- not skips --- on a cold runner.
    """
    from dataknobs_data.testing import chroma_embedding_function
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

    config: dict[str, object] = {
        "collection_name": f"docs_{uuid.uuid4().hex[:8]}",
        "embedding_function": chroma_embedding_function(EF_WIDTH),
    }
    config.update(overrides)
    return ChromaVectorStore(config)


@requires_chromadb
async def test_documents_embedded_at_another_width_are_refused() -> None:
    """The reproducer: the second door used to write past the declaration."""
    store = _chroma_docs_store(dimensions=DECLARED)
    await store.initialize()
    try:
        with pytest.raises(ValueError, match=rf"{EF_WIDTH}"):
            await store.add_documents(["the sky is blue"], ids=["d1"])
    finally:
        await store.close()


@requires_chromadb
async def test_a_collection_whose_rows_disagree_is_refused_at_initialize() -> None:
    """Reopening is the other way in, and the one ``pgvector`` already guards.

    ``pgvector`` compares the **table's** declared column against the store's
    configuration at initialize, and that difference is exactly what made one
    document portable-looking and not portable. A chroma collection's width is
    just as readable --- one row --- so the same disagreement is now caught in
    the same place on both backends.
    """
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

    name = f"reopen_{uuid.uuid4().hex[:8]}"
    honest = ChromaVectorStore({"dimensions": EF_WIDTH, "collection_name": name})
    await honest.initialize()
    try:
        await honest.add_vectors(np.ones((1, EF_WIDTH), dtype=np.float32), ids=["a"])
    finally:
        await honest.close()

    lying = ChromaVectorStore({"dimensions": DECLARED, "collection_name": name})
    try:
        with pytest.raises(ValueError, match=rf"{EF_WIDTH}"):
            await lying.initialize()
    finally:
        await lying.close()


@requires_chromadb
async def test_documents_at_the_declared_width_are_written() -> None:
    """The positive control: a declaration that matches is not in the way."""
    store = _chroma_docs_store(dimensions=EF_WIDTH)
    await store.initialize()
    try:
        assert await store.add_documents(["the sky is blue"], ids=["d1"]) == ["d1"]
        assert await store.count() == 1
    finally:
        await store.close()


@requires_chromadb
async def test_an_undeclared_store_accepts_documents_of_any_width() -> None:
    """The sentinel rule reaches this door too.

    ``0`` is the absence of a declaration, not a declaration of zero. A
    second door that read it as a claim would refuse every document written
    to the store that states the least.
    """
    store = _chroma_docs_store()
    await store.initialize()
    try:
        assert store.dimensions == 0
        assert await store.add_documents(["the sky is blue"], ids=["d1"]) == ["d1"]
        assert await store.count() == 1
    finally:
        await store.close()


@requires_chromadb
async def test_an_empty_document_batch_is_still_a_no_op() -> None:
    """The ordering the vector door already keeps: emptiness before width."""
    store = _chroma_docs_store(dimensions=DECLARED)
    await store.initialize()
    try:
        assert await store.add_documents([]) == []
        assert await store.count() == 0
    finally:
        await store.close()
