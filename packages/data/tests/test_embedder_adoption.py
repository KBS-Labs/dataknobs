"""How a site chooses between an ``embedder`` and an ``embedding_fn``.

Twenty-five parameter declarations across this package name a thing that
turns text into vectors, and each is gaining an ``embedder`` alongside its
existing callable. If each also wrote its own three-line choice between the
two, that would be the duplication that produced the eight incompatible
spellings in the first place --- so the choice is :func:`embed_texts` and
:func:`embed_text`, and this is where it is pinned.

The two rules worth stating outright, because both are refusals rather than
behaviours and neither is visible from a call site:

* **Neither source is an error**, and it must be raised *before* the loop.
  A bulk method that only discovers it inside the loop returns a
  successful-looking ``[]`` for an empty input.
* **Both sources is an error too.** Resolving it by precedence means one of
  the two silently does not run, and the caller cannot tell which --- the
  same class of error ``model_id`` exists to close.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.query import Query
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.sync import VectorTextSynchronizer
from dataknobs_data.vector import (
    SyncTextEmbedder,
    TextEmbedder,
    embed_text,
    embed_texts,
    require_embedding_source,
)


def _batch_sync(texts: list[str]) -> np.ndarray:
    """The commonest legacy shape: batch in, ``np.ndarray`` out, synchronous."""
    return np.array([[float(len(t)), 1.0] for t in texts])


async def _batch_async(texts: list[str]) -> np.ndarray:
    """The same, async --- the shape only some sites' annotations admit."""
    return _batch_sync(texts)


class _AsyncCallableObject:
    """A callable *object* whose ``__call__`` is ``async def``.

    The shape an embedder holding a model handle naturally takes, and the one
    :func:`inspect.iscoroutinefunction` gets wrong --- it reports this as
    synchronous, after which it is handed to a worker thread that returns the
    coroutine instead of a vector, with nothing raised.
    """

    async def __call__(self, texts: list[str]) -> np.ndarray:
        return _batch_sync(texts)


class _SyncFunctionReturningACoroutine:
    """A plain ``def`` that hands back an awaitable.

    Classifying the *callable* says sync and classifying the *result* says
    async. Only the second is right here, which is why both are asked.
    """

    def __call__(self, texts: list[str]) -> Any:
        return _batch_async(texts)


# --------------------------------------------------------------------------
# embed_texts — the batch choice
# --------------------------------------------------------------------------


async def test_an_embedder_is_awaited_directly() -> None:
    embedder = DeterministicEmbedder(dimensions=8)

    vectors = await embed_texts(["a", "b"], embedder=embedder)

    assert vectors == await embedder.embed(["a", "b"])


async def test_a_sync_callable_is_called() -> None:
    result = await embed_texts(["alpha"], embedding_fn=_batch_sync)

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_an_async_callable_is_awaited() -> None:
    result = await embed_texts(["alpha"], embedding_fn=_batch_async)

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_an_async_callable_object_is_awaited() -> None:
    """The case seven hand-rolled copies of this branch got wrong."""
    result = await embed_texts(["alpha"], embedding_fn=_AsyncCallableObject())

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_a_sync_callable_returning_an_awaitable_is_awaited() -> None:
    """Neither predicate alone covers this; asking both does.

    One of the two dispatch copies this replaces classified the callable and
    the other classified the result, so each handled a case the other missed.
    """
    result = await embed_texts(["alpha"], embedding_fn=_SyncFunctionReturningACoroutine())

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_the_callables_return_value_is_not_converted() -> None:
    """The callable path must reach a store exactly as it did before.

    Normalizing it to ``list[list[float]]`` for symmetry with the embedder
    path would change the dtype an existing caller's array arrives with, and
    this seam is additive.
    """
    result = await embed_texts(["alpha"], embedding_fn=_batch_sync)

    assert isinstance(result, np.ndarray)


async def test_neither_source_raises() -> None:
    with pytest.raises(ValueError, match="embedder is required"):
        await embed_texts(["alpha"])


async def test_both_sources_raise() -> None:
    with pytest.raises(ValueError, match="not both"):
        await embed_texts(["alpha"], embedder=DeterministicEmbedder(), embedding_fn=_batch_sync)


def test_the_guard_is_callable_on_its_own() -> None:
    """Bulk sites apply it before their loop, so an empty input still reports."""
    require_embedding_source(DeterministicEmbedder(), None)
    require_embedding_source(None, _batch_sync)

    with pytest.raises(ValueError, match="embedder is required"):
        require_embedding_source(None, None)
    with pytest.raises(ValueError, match="not both"):
        require_embedding_source(DeterministicEmbedder(), _batch_sync)


# --------------------------------------------------------------------------
# embed_text — the per-text choice
# --------------------------------------------------------------------------


async def test_embed_text_takes_the_first_of_a_batch_of_one() -> None:
    embedder = DeterministicEmbedder(dimensions=8)

    vector = await embed_text("alpha", embedder=embedder)

    assert vector == (await embedder.embed(["alpha"]))[0]


async def test_embed_text_offloads_a_sync_callable() -> None:
    """Routed through ``call_embedding_fn``, which offloads rather than inlines.

    A synchronous embed run on the event loop stalls every other task on it,
    which is invisible in a single-request test and catastrophic under
    concurrency.
    """
    loop_thread = threading.current_thread()
    seen: list[threading.Thread] = []

    def one(text: str) -> list[float]:
        seen.append(threading.current_thread())
        return [float(len(text))]

    vector = await embed_text("alpha", embedding_fn=one)

    assert vector == [5.0]
    assert seen == [seen[0]] and seen[0] is not loop_thread, (
        "a synchronous embedding function ran on the event loop"
    )


async def test_embed_text_awaits_an_async_callable() -> None:
    async def one(text: str) -> list[float]:
        return [float(len(text))]

    assert await embed_text("alpha", embedding_fn=one) == [5.0]


async def test_embed_text_refuses_neither_and_both() -> None:
    async def one(text: str) -> list[float]:
        return [1.0]

    with pytest.raises(ValueError, match="embedder is required"):
        await embed_text("alpha")
    with pytest.raises(ValueError, match="not both"):
        await embed_text("alpha", embedder=DeterministicEmbedder(), embedding_fn=one)


# --------------------------------------------------------------------------
# SyncTextEmbedder — how a `def` site reaches an async embedder
# --------------------------------------------------------------------------


def test_sync_embedder_runs_from_plain_sync_code() -> None:
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=8)) as sync:
        vectors = sync.embed(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)


def test_sync_embedder_answers_what_the_async_one_would() -> None:
    embedder = DeterministicEmbedder(dimensions=8)
    expected = asyncio.run(embedder.embed(["alpha", "bravo"]))

    with SyncTextEmbedder(embedder) as sync:
        assert sync.embed(["alpha", "bravo"]) == expected
        assert sync.embed_one("alpha") == expected[0]


async def test_sync_embedder_runs_from_inside_a_running_loop() -> None:
    """The case that makes the bridge necessary rather than convenient.

    ``asyncio.run`` and ``run_until_complete`` both fail when a loop is
    already running on the calling thread, which is exactly what happens when
    a sync wrapper is reached from async code. The bridge owns its own loop on
    its own thread, so it cannot deadlock against the caller's.

    Called **directly in the coroutine body**, which is the whole claim. An
    earlier version of this test reached it through ``asyncio.to_thread``,
    which moves the call off the loop thread and so tested a case no loop was
    ever running on --- passing identically against a bridge and against a
    bare ``run_until_complete`` that the claim says would raise.
    """
    sync = SyncTextEmbedder(DeterministicEmbedder(dimensions=4))
    try:
        vectors = sync.embed(["alpha"])
    finally:
        sync.close()

    assert len(vectors[0]) == 4


async def test_sync_embedder_called_on_a_loop_blocks_that_loop() -> None:
    """The cost of the line above, which is not free and was not written down.

    "Callable from inside a running loop" means *does not deadlock*. It does
    not mean *does not block*: the caller's thread sits in the bridge's
    ``Future.result()`` for the whole embedding, so every other task on the
    caller's loop is stalled for a network round trip. From inside async
    code the answer is to ``await`` the embedder directly --- the bridge is
    for the five synchronous sites that cannot.

    Pinned as a test rather than left to the docstring because it is the
    kind of claim a later change makes quietly false, and because the
    docstring is where it was missing.
    """
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    assert ticks > 0, "the co-tenant task should be running before the blocking call"

    sync = SyncTextEmbedder(_SlowEmbedder(seconds=0.05))
    try:
        before = ticks
        sync.embed(["alpha"])
        after = ticks
    finally:
        sync.close()
        task.cancel()

    assert after == before, (
        "a co-tenant task advanced during the blocking call, so this no "
        "longer demonstrates the cost the docs now warn about"
    )


class _SlowEmbedder:
    """An embedder that takes measurable time, without blocking its own loop.

    ``asyncio.sleep`` rather than ``time.sleep``: the point is that the
    *caller's* loop stalls while the *bridge's* loop is perfectly free, which
    a blocking sleep would confound.
    """

    def __init__(self, *, seconds: float, dimensions: int = 4) -> None:
        self._seconds = seconds
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return "slow"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        await asyncio.sleep(self._seconds)
        return [[1.0] * self.dimensions for _ in texts]


def test_sync_embedder_forwards_identity_unchanged() -> None:
    """A vector stored through the bridge must look like one stored without it."""
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=16, model_id="m")) as sync:
        assert sync.model_id == "m"
        assert sync.dimensions == 16


def test_sync_embedder_close_is_idempotent() -> None:
    sync = SyncTextEmbedder(DeterministicEmbedder())
    sync.close()
    sync.close()


def test_sync_embedders_methods_fit_the_legacy_parameters() -> None:
    """The reason no sync signature had to change to reach the seam.

    The annotations are the assertion here, and they are deliberately the
    *parameter* types rather than ``Any``: an ``Any`` on either side admits
    anything, so the earlier form of this test passed against a mismatch and
    would have passed against any mismatch. It was written while the sync
    sites declared ``np.ndarray`` returns, which a ``SyncTextEmbedder`` does
    not produce --- so the very snippet the docs publish was an ``arg-type``
    error at three sites, and this test reported nothing.

    Those parameters now admit the list shape they already accepted at
    runtime, and these two lines fail type checking if that regresses.
    """
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=4)) as sync:
        batch_param: Callable[[list[str]], np.ndarray | list[list[float]]] = sync.embed
        single_param: Callable[[str], np.ndarray | list[float]] = sync.embed_one

        assert len(batch_param(["a", "b"])) == 2
        assert len(single_param("a")) == 4


def test_the_published_sync_adoption_snippet_type_checks() -> None:
    """The doc's own example, executed --- and annotated as the docs present it.

    ``packages/data/docs/text-embedder.md`` tells a synchronous caller to hand
    ``sync.embed_one`` to :meth:`Query.near_text`. That is the seam's only
    documented sync adoption path, so if it does not type-check the sync half
    of the seam is unreachable for a typed consumer whatever it does at runtime.
    """
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=4)) as sync:
        query = Query().near_text("some text", sync.embed_one)

    assert query.vector_query is not None
    assert len(query.vector_query.vector) == 4


def test_sync_embedder_is_not_a_text_embedder_but_isinstance_says_it_is() -> None:
    """The protocol's runtime check cannot see the difference, and this proves it.

    ``TextEmbedder.embed`` is ``async def`` and this one is a plain ``def``, so
    a ``SyncTextEmbedder`` is not an embedder in any usable sense. But
    ``isinstance`` against a runtime-checkable protocol checks that the three
    members are *present* and nothing about their signatures, so it answers
    ``True`` --- which is the limitation the protocol's docstring names, met
    here by the one class in this package shaped to trip it.

    Pinned rather than fixed --- but not because the fix is hard. An earlier
    version of this docstring said renaming ``embed`` would break the property
    the class exists for, and that is wrong: the five synchronous sites take a
    *bound method* as a plain ``Callable``, never a ``SyncTextEmbedder`` as a
    protocol, so the name is free to change and only the package docs follow
    it. Distinct member names are exactly what make ``ResourceFactory.create``
    and ``AsyncResourceFactory.create_async`` the one twin pair in this tree
    that ``isinstance`` can tell apart. The rename is deferred because it is an
    API decision of its own, not because it is unavailable.

    Until then the annotation is what stops the mistake; the ``isinstance`` is
    a smoke test and this is the smoke.
    """
    with SyncTextEmbedder(DeterministicEmbedder()) as sync:
        assert not asyncio.iscoroutinefunction(sync.embed)
        assert isinstance(sync, TextEmbedder), (
            "if this ever answers False the protocol grew signature checking, "
            "and the docs saying otherwise are now wrong"
        )


async def test_a_sync_embedder_used_as_an_async_one_fails_loudly() -> None:
    """The consolation for the check above: the mistake cannot be silent.

    A non-vector must never reach a store. It does not: the plain ``def``
    returns a list where the caller awaits, which raises immediately rather
    than producing something vector-shaped enough to persist.
    """
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=4)) as sync:
        with pytest.raises(TypeError, match="can't be used in 'await' expression"):
            await embed_texts(["alpha"], embedder=sync)  # type: ignore[arg-type]


class TestTheVectorStoreFamily:
    """``VectorStore.bulk_embed_and_store`` --- the fourth site, and the one
    with no ``embedder`` coverage of any kind before this class.

    Its three siblings (``AsyncBulkEmbedMixin``, ``VectorSyncMixin`` and
    ``AsyncPostgresDatabase``) call :func:`require_embedding_source` before
    their loop. This one does not, so the module docstring's first rule ---
    *neither source is an error, and it must be raised before the loop* --- was
    stated for four sites and pinned for one.

    Inherited unchanged by ``MemoryVectorStore``, ``ChromaVectorStore``,
    ``FaissVectorStore`` and ``PgVectorStore``, so the gap is the whole family's.
    """

    @staticmethod
    def _store() -> MemoryVectorStore:
        return MemoryVectorStore({"dimensions": 8})

    async def test_neither_source_is_an_error_even_with_no_texts(self) -> None:
        """The empty batch is the whole point: with texts, the loop runs and
        ``embed_texts`` raises from inside it, so only this input can tell
        whether the guard is where the contract says it is.
        """
        store = self._store()
        await store.initialize()
        with pytest.raises(ValueError, match="embedder is required"):
            await store.bulk_embed_and_store([])

    async def test_both_sources_is_an_error_even_with_no_texts(self) -> None:
        """The other half of the same guard, and the more dangerous one.

        A caller passing both has two models in play and no way to learn which
        produced the vectors; returning ``[]`` tells them it went fine.
        """
        store = self._store()
        await store.initialize()
        with pytest.raises(ValueError, match="not both"):
            await store.bulk_embed_and_store(
                [], embedding_fn=_batch_sync, embedder=DeterministicEmbedder(dimensions=8)
            )

    async def test_the_embedder_path_stores_the_embedder_s_vectors(self) -> None:
        """Coverage this family had none of: ``embedder=`` end to end."""
        store = self._store()
        await store.initialize()
        embedder = DeterministicEmbedder(dimensions=8)

        ids = await store.bulk_embed_and_store(["alpha", "beta"], embedder=embedder)

        assert len(ids) == 2
        [expected_alpha, _] = await embedder.embed(["alpha", "beta"])
        stored = store.vectors[ids[0]]
        np.testing.assert_allclose(stored, np.asarray(expected_alpha, dtype=np.float32), rtol=1e-6)

    async def test_the_embedder_path_converts_to_float32(self) -> None:
        """The ``np.asarray(..., dtype=np.float32)`` branch fires only under
        ``embedder=``, and was reachable by no test.

        A ``TextEmbedder`` returns ``list[list[float]]`` by design --- the shape
        that needs no conversion at the ``llm`` boundary --- so this is the one
        place that converts, and the dtype is what the store searches on.
        """
        store = self._store()
        await store.initialize()

        [vector_id] = await store.bulk_embed_and_store(
            ["alpha"], embedder=DeterministicEmbedder(dimensions=8)
        )

        assert store.vectors[vector_id].dtype == np.float32


class TestTheStalenessKeyTheEmbedderDefaults:
    """``model_id`` is documented as removing a class of error. It does not yet.

    The seam's stated reason for carrying an identity is that a stored vector's
    staleness key should come from the thing that produced it rather than from a
    parameter a caller keeps in step by hand. Both write sites honour that ---
    they default ``model_name`` from ``embedder.model_id``.

    But ``model_name`` is written by two sites and compared by none.
    ``VectorTextSynchronizer._has_current_vector`` decides currency on
    ``model_version``, a separate free-text constructor parameter the embedder
    does not default and which disables the check entirely when left ``None``.
    So the mismatch the identity exists to close is still reachable, and this
    is where that is pinned.
    """

    @staticmethod
    async def _store_with(db: Any, embedder: DeterministicEmbedder) -> Record:
        [stored_id] = await db.bulk_embed_and_store(
            [Record(data={"body": "the source text"})], "body", embedder=embedder
        )
        record = await db.read(stored_id)
        assert record is not None
        return record

    async def test_the_defaulted_identity_is_written(self) -> None:
        """The half that works, and the reason the gap is easy to miss."""
        db = AsyncMemoryDatabase()
        record = await self._store_with(db, DeterministicEmbedder(dimensions=8, model_id="v1"))

        field = record.get_field("embedding")
        assert isinstance(field, VectorField)
        assert field.model_name == "v1"

    async def test_a_model_swap_is_not_reported_current(self) -> None:
        """The half that does not.

        A corpus embedded by ``v1`` and then read by a synchronizer configured
        for ``v2`` holds vectors from a different vector space. Nothing about
        the source text changed, so the content digest matches --- and the
        identity that *did* change is in a key the synchronizer never reads.

        The vectors are therefore served forever against the new model's
        queries, silently, which is the exact failure ``model_id`` is
        documented as closing.
        """
        db = AsyncMemoryDatabase()
        record = await self._store_with(db, DeterministicEmbedder(dimensions=8, model_id="v1"))

        v2 = DeterministicEmbedder(dimensions=8, model_id="v2")
        synchronizer = VectorTextSynchronizer(
            database=db,
            embedding_fn=lambda text: np.asarray(v2._vector(text), dtype=np.float32),
            text_fields=["body"],
            model_name=v2.model_id,
        )

        assert not synchronizer._has_current_vector(record, "embedding"), (
            "a vector produced by a different model than the synchronizer is "
            "configured for must not be reported current"
        )

    async def test_a_vector_with_no_recorded_name_is_left_alone(self) -> None:
        """The upgrade clause, and it is a deliberate asymmetry.

        A vector written before anything recorded a model name carries
        ``None``, which is an absence of information rather than evidence of a
        different model. Treating it as a mismatch would re-embed every corpus
        predating the seam on the first sweep after upgrading --- an expensive
        answer to a question nothing asked.
        """
        db = AsyncMemoryDatabase()
        [stored_id] = await db.bulk_embed_and_store(
            [Record(data={"body": "the source text"})],
            "body",
            embedding_fn=lambda texts: np.asarray(
                [[float(len(t))] * 8 for t in texts], dtype=np.float32
            ),
        )
        record = await db.read(stored_id)
        assert record is not None
        field = record.get_field("embedding")
        assert isinstance(field, VectorField)
        assert field.model_name is None, "precondition: nothing recorded a name"

        synchronizer = VectorTextSynchronizer(
            database=db,
            embedding_fn=lambda text: np.asarray([1.0] * 8, dtype=np.float32),
            text_fields=["body"],
            model_name="v2",
        )

        assert synchronizer._has_current_vector(record, "embedding")

    async def test_the_upgrade_clause_holds_in_both_lanes(self) -> None:
        """And it has to be asserted in both, because the check is written twice.

        ``_has_current_vector`` branches on whether the field is a
        ``VectorField`` or a plain list, and each branch reads the model
        identity from a different place --- the field's own attribute, or a
        ``{field}_metadata`` sidecar. Two readers of one concept is how the
        first version of this comparison came to exist in one lane and not the
        other, so a clause stated for one lane is not evidence about the other.

        The clause: a vector carrying no recorded name is an absence of
        information, not a mismatch. An unpaired sidecar must therefore not
        mean stale by itself --- otherwise turning the check on re-embeds every
        plain-value corpus that never wrote one.
        """
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            synchronizer = VectorTextSynchronizer(
                database=db,
                embedding_fn=lambda text: np.asarray([1.0] * 8, dtype=np.float32),
                text_fields=["body"],
                model_name="v2",
            )

            vector_field_lane = Record(data={"body": "the source text"})
            vector_field_lane.fields["embedding"] = VectorField(name="embedding", value=[1.0] * 8)

            plain_lane = Record(data={"body": "the source text", "embedding": [1.0] * 8})
            assert not isinstance(plain_lane.fields["embedding"], VectorField), (
                "precondition: this record must exercise the plain-value branch"
            )

            assert synchronizer._has_current_vector(
                plain_lane, "embedding"
            ) == synchronizer._has_current_vector(vector_field_lane, "embedding"), (
                "the two lanes must agree about a vector that recorded no model name"
            )
        finally:
            await db.close()
