"""Capability advertisement across the vector-store backends.

Several of this family's public methods exist on some backends and not
others — ``save``/``load`` on two, ``create_index`` on one. A consumer
holding the abstraction previously had no way to ask which, so reaching
one meant an ``isinstance`` downcast, and a consumer that downcasts has
stopped being portable. The family now speaks ``CapabilityContract``, so
the question has an answer: ``store.supports(...)``.

Six things are pinned here:

1. **Conformance** — every backend class is a capability-contract host,
   checkable without an instance and without a service connection.
2. **Shadow guard** — no backend drops the ABC's set from
   ``instance_capabilities()``. ``CapabilityMixin`` does NOT auto-union
   across the MRO, so a backend declaring its own
   ``SUPPORTED_CAPABILITIES`` without unioning would silently lose
   whatever the ABC declares. The ABC's set is empty today; this fails
   the moment that stops being true and a backend has not been revisited.
3. **Instance, not class** — ``VECTOR_PERSIST`` is a property of
   configuration. The same class answers differently with and without a
   ``persist_path``, and a store that cannot persist at all answers no
   even when handed one. A ``ClassVar`` declaration passes cells 1 and 2
   and fails this one, which is why it is here.
4. **Truth of advertisement** — the bit is tied to behaviour, not to an
   aspiration. A store advertising ``VECTOR_PERSIST`` round-trips its
   corpus through disk; one that does not advertise it writes nothing
   when asked to save.
5. **Reachable and refusing** — the guarded methods are on the ABC, so a
   consumer can call what it just confirmed. A backend without the
   capability answers with ``CapabilityNotSupportedError`` rather than
   ``AttributeError``, which is the difference between "unsupported
   operation" and "you misspelled something".
6. **The published matrix** is regenerated from the classes and from
   live instances, so the document cannot drift from the code.

No mocks: real in-process backends. pgvector's class-level advertisement
is asserted here without a connection; its behaviour lives with the
suites that have one.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dataknobs_common import (
    Capability,
    CapabilityContract,
    CapabilityMixin,
    CapabilityNotSupportedError,
)
from dataknobs_common.testing import (
    is_chromadb_available,
    is_faiss_available,
    is_package_available,
)

from dataknobs_data.vector.stores.base import VectorStore
from dataknobs_data.vector.stores.memory import MemoryVectorStore

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

if is_package_available("asyncpg"):
    from dataknobs_data.vector.stores.pgvector import PgVectorStore


def _backend_classes() -> list[Any]:
    """Every in-tree store class that can be imported in this environment.

    Class-level advertisement needs no driver and no connection, but the
    optional backends cannot be imported at all without their package —
    so availability gates the import, not the assertion.
    """
    classes: list[Any] = [MemoryVectorStore]
    if is_faiss_available():
        classes.append(FaissVectorStore)
    if is_chromadb_available():
        classes.append(ChromaVectorStore)
    if is_package_available("asyncpg"):
        classes.append(PgVectorStore)
    return classes


_BACKEND_CLASSES = _backend_classes()


# ---------------------------------------------------------------------------
# 1. Conformance — class level, no instance, no service.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("store_cls", _BACKEND_CLASSES, ids=lambda c: c.__name__)
def test_backend_class_is_capability_contract_host(store_cls: type) -> None:
    """Class-level capability-contract host guarantee.

    ``issubclass`` against ``CapabilityContract`` is unavailable — it is a
    ``runtime_checkable`` Protocol with a data member, and Python forbids
    ``issubclass`` on those. So the class-level guarantee is pinned two
    ways: the backend inherits the contract implementation, and the three
    contract methods resolve on the class. Runtime-checkable ``isinstance``
    conformance is asserted on a live instance below.
    """
    assert issubclass(store_cls, CapabilityMixin)
    for name in ("supported_capabilities", "instance_capabilities", "supports"):
        assert callable(getattr(store_cls, name, None))


# ---------------------------------------------------------------------------
# 2. Shadow guard — the MRO no-auto-union caveat.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_instance_is_contract_and_does_not_shadow_the_abc_set(
    initialized_vector_store: Any,
) -> None:
    """A live instance speaks the contract and keeps the ABC's guarantees.

    Two assertions, and only the first is load-bearing today. ``isinstance``
    against the runtime-checkable ``CapabilityContract`` is the conformance
    check cell 1 defers to here, a live instance being the only way to ask it.

    The second arms later, and says so rather than implying otherwise. If a
    backend ever declares ``SUPPORTED_CAPABILITIES`` without unioning
    ``VectorStore.SUPPORTED_CAPABILITIES``, everything the ABC declares drops
    out of that backend silently, because ``CapabilityMixin`` does not union
    across the MRO. But the ABC's set is empty today, so every set satisfies
    the superset test and no backend can fail it. It becomes a real guard on
    the commit that gives the ABC its first capability — which is exactly the
    commit that would otherwise have to remember this hazard unaided, and the
    reason to write the assertion while it still cannot fail.
    """
    assert isinstance(initialized_vector_store, CapabilityContract)
    advertised = initialized_vector_store.instance_capabilities()
    assert advertised >= VectorStore.SUPPORTED_CAPABILITIES


# ---------------------------------------------------------------------------
# 3. Instance, not class — VECTOR_PERSIST follows configuration.
# ---------------------------------------------------------------------------
def _persisting_store(kind: str, path: Path) -> Any:
    if kind == "memory":
        return MemoryVectorStore({"dimensions": 4, "persist_path": str(path)})
    return FaissVectorStore({"dimensions": 4, "metric": "cosine", "persist_path": str(path)})


_PERSISTABLE = ["memory"] + (["faiss"] if is_faiss_available() else [])


@pytest.mark.parametrize("kind", _PERSISTABLE)
def test_persist_capability_follows_the_configured_path(kind: str) -> None:
    """The same class answers differently with and without a path.

    ``save()`` and ``load()`` return early when ``persist_path`` is unset,
    so a store built without one persists exactly as much as a
    server-backed store does. Advertising from a ``ClassVar`` would claim
    otherwise, and would be wrong in the worst available way: the methods
    are present, so a caller who checked by ``hasattr`` — or by reading
    the class — gets a silent no-op rather than an error.
    """
    with tempfile.TemporaryDirectory() as d:
        configured = _persisting_store(kind, Path(d) / "store.pkl")
        assert configured.supports(Capability.VECTOR_PERSIST)
        assert Capability.VECTOR_PERSIST in configured.instance_capabilities()

    unconfigured = (
        MemoryVectorStore({"dimensions": 4})
        if kind == "memory"
        else FaissVectorStore({"dimensions": 4, "metric": "cosine"})
    )
    assert not unconfigured.supports(Capability.VECTOR_PERSIST)

    # Class-level advertisement stays empty for both: the class is not
    # where the answer lives, and a consumer reading it gets no false yes.
    assert Capability.VECTOR_PERSIST not in type(configured).supported_capabilities()


@pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed")
def test_a_path_on_a_store_that_cannot_persist_advertises_nothing() -> None:
    """``persist_path`` is on the shared config, so every store accepts it.

    Chroma persists through its own client and has no ``save``/``load``,
    so the field reaching its config must not be mistaken for the
    capability. Answering yes here would be a portability claim the store
    cannot honour on the one call the consumer would then make.
    """
    store = ChromaVectorStore(
        {
            "dimensions": 4,
            "collection_name": "test_caps_no_persist",
            "persist_path": "/tmp/should-not-matter",
        }
    )
    assert not store.supports(Capability.VECTOR_PERSIST)


# ---------------------------------------------------------------------------
# 4. Truth of advertisement — the bit is backed by behaviour.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _PERSISTABLE)
async def test_advertised_persist_actually_round_trips(kind: str) -> None:
    """A store advertising VECTOR_PERSIST really restores from disk."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "store.pkl"
        writer = _persisting_store(kind, path)
        assert writer.supports(Capability.VECTOR_PERSIST)
        await writer.initialize()
        try:
            await writer.add_vectors(
                np.eye(2, 4, dtype=np.float32), ids=["a", "b"], metadata=[{"n": 1}, {"n": 2}]
            )
            await writer.save(force=True)
        finally:
            await writer.close()

        reader = _persisting_store(kind, path)
        await reader.initialize()
        try:
            await reader.load()
            assert await reader.count() == 2
        finally:
            await reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _PERSISTABLE)
async def test_unadvertised_persist_refuses_and_writes_nothing(kind: str) -> None:
    """A store that does not advertise it saves nothing when asked.

    The negative half of the same claim, and the reason the capability is
    worth querying: ``save()`` on an unconfigured store is not an error —
    it returns, having done nothing. A consumer branching on the
    capability learns that before losing the data; one calling the method
    learns it later, or never.
    """
    with tempfile.TemporaryDirectory() as d:
        store = (
            MemoryVectorStore({"dimensions": 4})
            if kind == "memory"
            else FaissVectorStore({"dimensions": 4, "metric": "cosine"})
        )
        assert not store.supports(Capability.VECTOR_PERSIST)
        await store.initialize()
        try:
            await store.add_vectors(np.eye(1, 4, dtype=np.float32), ids=["a"])
            # One rule, whatever the reason: a store that does not
            # advertise VECTOR_PERSIST refuses. These two *classes* can
            # persist and these two *instances* cannot, which used to be
            # the case that returned quietly — the caller asked for a
            # snapshot, got a successful-looking call, and had nothing on
            # disk to restore from.
            for call in (store.save(force=True), store.load()):
                with pytest.raises(CapabilityNotSupportedError) as excinfo:
                    await call
                assert "vector_persist" in str(excinfo.value)
        finally:
            await store.close()
        # Offloaded rather than called inline: this is an ``async def`` on a
        # live loop, and the repo's async-transport rule holds inside tests
        # too. ``asyncio.to_thread`` is the dependency-free form.
        written = await asyncio.to_thread(lambda: sorted(p.name for p in Path(d).iterdir()))
        assert written == []


@pytest.mark.skipif(not is_package_available("asyncpg"), reason="asyncpg not installed")
def test_index_tuning_is_advertised_by_pgvector_alone() -> None:
    """``create_index`` is genuinely backend-specific, and says so.

    IVFFlat/HNSW tuning has no analogue in a brute-force store, so this
    capability closes by being declared rather than implemented
    everywhere. Class-level, because it does not depend on configuration:
    a store configured ``index_type: none`` still honours an explicit
    argument.
    """
    assert Capability.VECTOR_INDEX_TUNING in PgVectorStore.supported_capabilities()
    for other in _BACKEND_CLASSES:
        if other is PgVectorStore:
            continue
        assert Capability.VECTOR_INDEX_TUNING not in other.supported_capabilities()
        # ``hasattr`` is no longer the question. The method is on the ABC
        # so that a consumer can call what it just confirmed; what a
        # backend without the capability owes is a refusal, not an
        # ``AttributeError`` that reads as a missing attribute rather
        # than an unsupported operation.
        assert hasattr(other, "create_index")


# ---------------------------------------------------------------------------
# 5. Reachable, and refusing — the half a declaration alone does not buy.
# ---------------------------------------------------------------------------
def _unconfigured(name: str) -> Any:
    """One constructed store per backend, no service, no persist path."""
    plain, _ = _probe_pair(name)
    return plain


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ChromaVectorStore", "PgVectorStore"])
async def test_a_backend_that_cannot_persist_refuses_rather_than_vanishing(
    name: str,
) -> None:
    """``save``/``load`` reach every backend and refuse where unsupported.

    Declaring the capability without declaring the method left the check
    reachable and the call not: ``supports`` answered truthfully and the
    only way to act on the answer was to downcast. These two backends
    keep their rows in a service, so the refusal is the honest response —
    and it is a refusal, naming the capability, rather than the
    ``AttributeError`` a missing method used to raise.

    No service is contacted: the guard runs before any transport would.
    """
    store = _unconfigured(name)
    assert not store.supports(Capability.VECTOR_PERSIST)

    for call in (store.save(), store.load()):
        with pytest.raises(CapabilityNotSupportedError) as excinfo:
            await call
        assert "vector_persist" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["MemoryVectorStore", "FaissVectorStore", "ChromaVectorStore"])
async def test_a_backend_without_index_tuning_refuses_create_index(name: str) -> None:
    """``create_index`` is reachable on all four and honoured by one.

    pgvector is excluded because it *implements* the method; its own
    behaviour is asserted by the suites that have a server.
    """
    store = _unconfigured(name)
    assert not store.supports(Capability.VECTOR_INDEX_TUNING)

    with pytest.raises(CapabilityNotSupportedError) as excinfo:
        await store.create_index()
    assert "vector_index_tuning" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_store_that_advertises_persist_and_does_not_implement_it_is_caught() -> None:
    """The other direction: declared, inherited, and therefore unimplemented.

    ``require_capability`` passing is not the end of the base method. A
    backend that advertises ``VECTOR_PERSIST`` and inherits the default
    would otherwise persist nothing in silence — a consumer that checked,
    was told yes, and lost its rows. That is the precise failure the
    advertisement exists to prevent, so the default refuses loudly
    instead of returning.
    """

    class _Liar(ChromaVectorStore):
        SUPPORTED_CAPABILITIES = ChromaVectorStore.SUPPORTED_CAPABILITIES | {
            Capability.VECTOR_PERSIST
        }

    store = _Liar({"dimensions": 4, "collection_name": "capability_liar_probe"})
    assert store.supports(Capability.VECTOR_PERSIST)

    with pytest.raises(NotImplementedError, match="does not implement save"):
        await store.save()


# ---------------------------------------------------------------------------
# 6. The document API is a different bargain, not a missing feature.
# ---------------------------------------------------------------------------
def test_document_api_is_chroma_alone_and_the_portable_path_is_everywhere() -> None:
    """``VECTOR_DOCUMENT_API`` marks server-side embedding, not "can store text".

    The distinction is the whole reason to advertise it. Every backend
    stores text through ``bulk_embed_and_store``, which takes the
    caller's embedder and records the model identity in each row — so a
    vector's provenance survives. Chroma's document API embeds with the
    *store's* model instead, which is convenient and drops that record.
    A consumer that reads the capability as "only Chroma handles
    documents" would reach for the one backend that cannot tell it what
    embedded its rows.
    """
    assert Capability.VECTOR_DOCUMENT_API in ChromaVectorStore.supported_capabilities()
    for other in _BACKEND_CLASSES:
        if other is ChromaVectorStore:
            continue
        assert Capability.VECTOR_DOCUMENT_API not in other.supported_capabilities()
        assert not hasattr(other, "add_documents")

    # The portable path, which is what a consumer should reach for first.
    for cls in _BACKEND_CLASSES:
        assert callable(getattr(cls, "bulk_embed_and_store", None))


# ---------------------------------------------------------------------------
# 7. The published matrix is generated, not maintained.
# ---------------------------------------------------------------------------
_DOC = Path(__file__).parents[3] / "docs" / "vector-store-capabilities.md"
_MATRIX_START = "<!-- capability-matrix:start -->"
_MATRIX_END = "<!-- capability-matrix:end -->"

# Column order is the document's; a capability added to the family adds a
# column here and the test then requires the document to grow one too.
_MATRIX_COLUMNS = (
    Capability.VECTOR_PERSIST,
    Capability.VECTOR_INDEX_TUNING,
    Capability.VECTOR_DOCUMENT_API,
)

# Row order is the document's, and the roster is deliberately written out:
# a backend added to the family without a row is the divergence this whole
# suite exists to catch, and an auto-discovered roster would quietly grow
# one instead of failing.
_MATRIX_ROWS = ("MemoryVectorStore", "FaissVectorStore", "ChromaVectorStore", "PgVectorStore")


def _probe_pair(name: str) -> tuple[Any, Any]:
    """One store built without a ``persist_path`` and one built with.

    Construction only — no ``initialize``, so no backend needs a running
    service to appear in the matrix.
    """
    path = {"persist_path": "/tmp/capability-matrix-probe"}
    if name == "MemoryVectorStore":
        base: dict[str, Any] = {"dimensions": 4}
        return MemoryVectorStore(base), MemoryVectorStore({**base, **path})
    if name == "FaissVectorStore":
        base = {"dimensions": 4, "metric": "cosine"}
        return FaissVectorStore(base), FaissVectorStore({**base, **path})
    if name == "ChromaVectorStore":
        base = {"dimensions": 4, "collection_name": "capability_matrix_probe"}
        return ChromaVectorStore(base), ChromaVectorStore({**base, **path})
    base = {
        "dimensions": 4,
        "connection_string": "postgresql://u:p@localhost:5432/db",
        "table": "capability_matrix_probe",
    }
    return PgVectorStore(base), PgVectorStore({**base, **path})


def _cell(name: str, capability: Capability) -> str:
    """The matrix cell for one backend and one capability.

    Three answers, and the middle one is why the matrix is worth
    generating: a capability can be a property of configuration rather
    than of type, and a table that only had "yes"/"no" would have to pick
    one of them and be wrong half the time.
    """
    cls = {c.__name__: c for c in _BACKEND_CLASSES}[name]
    if capability in cls.supported_capabilities():
        return "yes"
    plain, configured = _probe_pair(name)
    if configured.supports(capability) and not plain.supports(capability):
        return "with `persist_path`"
    assert not configured.supports(capability), (
        f"{name} advertises {capability} per instance on some axis this "
        f"matrix does not model — add the axis rather than the exception"
    )
    return "—"


def _generate_matrix() -> str:
    header = "| Backend | " + " | ".join(f"`{c.name}`" for c in _MATRIX_COLUMNS) + " |"
    rule = "|---|" + "---|" * len(_MATRIX_COLUMNS)
    rows = [
        "| `" + name + "` | " + " | ".join(_cell(name, c) for c in _MATRIX_COLUMNS) + " |"
        for name in _MATRIX_ROWS
    ]
    return "\n".join([header, rule, *rows])


@pytest.mark.skipif(
    not (is_faiss_available() and is_chromadb_available() and is_package_available("asyncpg")),
    reason="the published matrix covers all four backends; verifying a subset would "
    "let an unchecked row drift while the suite reported green",
)
def test_published_capability_matrix_matches_the_code() -> None:
    """The doc's matrix is rebuilt from the classes and compared.

    A cross-backend promise stated in prose is what let four divergences
    accumulate underneath one — the table is only worth having if
    something fails when it stops being true. Regenerate by running this
    test and pasting the diff; do not hand-edit the block.
    """
    text = _DOC.read_text(encoding="utf-8")
    start = text.index(_MATRIX_START) + len(_MATRIX_START)
    published = text[start : text.index(_MATRIX_END)].strip()
    assert published == _generate_matrix()
