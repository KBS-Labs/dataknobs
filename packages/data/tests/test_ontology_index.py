# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A configured ontology's semantic index, reached through the registry.

Two of the configuration contract's acceptance cases live here --- that the
enumeration **is** the vocabulary, and that every row it writes carries an id
a caller can read back --- plus what this work builds that nothing in that
suite exercises.

The store is real and in memory, and the embedder is a real ``TextEmbedder``
with no model behind it. Neither is a mock: the code under test runs its own
path, and the vectors are reproducible across processes so a second test can
assert on what the first wrote.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.index import AliasSource, AsyncIndexSource, CallableSource, IndexItem
from dataknobs_common.ontology import (
    ALIAS_FORMS_KEY,
    ONTOLOGY_ID_KEY,
    OntologyConfig,
)
from dataknobs_common.testing import requires_chromadb, requires_faiss

from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import SemanticIndex
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.types import DistanceMetric

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

#: Sixteen is enough for the forms in this file to separate and small enough
#: that a failure prints legibly.
DIMENSIONS = 16

#: The vocabulary the enumeration case asks for: **an entity of every
#: declared type and at least one type holding none.**
#:
#: The empty type is what makes that assertable. Over a vocabulary
#: whose every declared type holds something, *enumerate the schema* and
#: *enumerate what is held* agree, and the test cannot tell which one ran ---
#: and they are different answers over a real vocabulary, where a schema is a
#: declaration and a source's contents are a fact.
#:
#: Authored rather than live, because a live binding declares one type by
#: construction (its projection's ``type:`` is a constant) and the registry
#: refuses two live bindings over one store. The multi-type half therefore
#: needs an authored document; the live half is asserted elsewhere.
CATALOG: dict[str, Any] = {
    "id": "catalog",
    "version": "1.0",
    "entity_types": [{"id": "Product"}, {"id": "Brand"}, {"id": "Discontinued"}],
    "entities": [
        {
            "id": "sku-4471",
            "type": "Product",
            "name": "Acme Widget",
            "description": "a widget",
            "aliases": ["Widget", "ACME widget"],
        },
        {"id": "sku-8802", "type": "Product", "name": "Bolt"},
        {"id": "acme", "type": "Brand", "name": "Acme Corp", "aliases": ["ACME"]},
    ],
}


def _document(**extra: Any) -> dict[str, Any]:
    return {**CATALOG, **extra}


def _indexed(**extra: Any) -> dict[str, Any]:
    """:data:`CATALOG` declaring a store the test process can actually open."""
    return _document(index={"store": {"backend": "memory", "dimensions": DIMENSIONS}}, **extra)


async def _registry(document: dict[str, Any], **components: Any) -> OntologyRegistry:
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document),
        embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
        **components,
    )
    await registry.load()
    return registry


# --------------------------------------------------------------------------
# The enumeration is the vocabulary
# --------------------------------------------------------------------------


async def test_the_index_holds_every_entity_the_vocabulary_holds() -> None:
    """Reached through ``registry.index("catalog")``, and it is the whole vocabulary.

    Three entities under two of three declared types. The third type holds
    nothing and contributes nothing, without an error --- a declared type with
    no members is a vocabulary saying so, not a fault.

    The registry door is the one this is asserted through. ``from_components`` is a
    registry door as much as the configured one is, and the two are required
    to reach the same object; what this asserts is that an ``index:`` section
    produces an index reachable by id, which is true through either.
    """
    registry = await _registry(_indexed())
    try:
        index = registry.index("catalog")
        assert index is not None

        assert await index.build() == 3

        hits = await index.search("Acme Widget", k=10)
        assert {hit.record.id for hit in hits} == {
            "catalog:sku-4471",
            "catalog:sku-8802",
            "catalog:acme",
        }
    finally:
        await registry.close()


async def test_an_ontology_with_no_index_section_still_answers_none() -> None:
    """The shipped contract, and it must survive this leg.

    Absence is a configuration answer rather than an error. What the leg owes
    beside it is the **positive** twin above: after this leg the absent-section
    branch is only half the member, and nothing would notice the other half
    going back to answering ``None``.
    """
    registry = await _registry(_document())
    try:
        assert registry.index("catalog") is None
    finally:
        await registry.close()


async def test_unloading_drops_the_index_and_reloading_rebuilds_it() -> None:
    """A per-id thing is dropped where the other per-id things are dropped.

    ``unload`` releases no handle --- that is its stated contract and this
    does not change it --- but an index it keeps for an id it no longer holds
    is a value answering for a vocabulary that has departed.
    """
    registry = await _registry(_indexed())
    try:
        assert registry.index("catalog") is not None
        assert await registry.unload("catalog") is True
        assert registry.index("catalog") is None

        await registry.load(_indexed(), replace=True)
        assert registry.index("catalog") is not None
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# What every row carries
# --------------------------------------------------------------------------


async def test_every_row_carries_a_qualified_id_and_the_ontology_key() -> None:
    """The id round-trips through ``localize``, and the key is on every row.

    **This is the first caller to take the id fallback.** A row written
    without a ``record_id`` comes back carrying the vector id, and every
    shipped writer before this one wrote both from the same value --- so the
    fallback existed, was documented, and had never been exercised in the case
    it was written for.
    """
    registry = await _registry(_indexed())
    try:
        index = registry.index("catalog")
        assert index is not None
        await index.build()
        ontology = registry.get("catalog")
        assert ontology is not None

        hits = await index.search("Acme", k=10)
        assert hits

        for hit in hits:
            assert hit.record.id is not None
            assert hit.record.id.startswith("catalog:")
            assert hit.metadata[ONTOLOGY_ID_KEY] == "catalog"
            assert ontology.qualify(ontology.localize(hit.record.id)) == hit.record.id
    finally:
        await registry.close()


async def test_a_row_carries_the_source_field_beside_the_source_text() -> None:
    """Otherwise an index row and any other caller's row are indistinguishable.

    Both answer ``vector_field=None`` at the published read door, so ``None``
    would have to mean both *written through a door that records no source
    field* and *composed by this index* --- and a consumer cannot act on a
    value carrying two meanings.

    The pair is written **by the store**, from a call argument, rather than by
    the index reaching into the metadata dict: the invariant is the store's
    and a caller writing half of it by hand is the rule gaining a silent
    exception at the site that most needed it to hold.
    """
    registry = await _registry(_indexed())
    try:
        index = registry.index("catalog")
        assert index is not None
        await index.build()

        [hit] = [
            hit for hit in await index.search("Bolt", k=10) if hit.record.id == "catalog:sku-8802"
        ]
        assert hit.source_text == "Bolt"
        assert hit.vector_field == "name"
        assert hit.metadata[ALIAS_FORMS_KEY] == []
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The key prefix, across the backends this suite can run
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    [
        "memory",
        pytest.param("faiss", marks=requires_faiss),
        pytest.param("chroma", marks=requires_chromadb),
    ],
)
async def test_the_dk_prefix_and_a_list_value_survive_a_round_trip(backend: str) -> None:
    """The falsifier the key family carries: *if a backend rejects ``dk_``, it is wrong.*

    Run rather than quoted, and run on the increment that first holds vector
    stores --- the increment that owns these keys is service-free and cannot
    check it.

    **The list-valued half is the part the original question did not ask.**
    Two of the four keys are list-valued because a row may be about several
    things at once, and a backend that flattens or rejects a list would break
    them while leaving the string keys intact.

    ``pgvector`` is not here: it needs a running service, and it is owed under
    the optional-dependency rule rather than skipped silently.
    """
    import numpy as np

    from dataknobs_common.ontology.tags import NODE_ID_KEY, TAXONOMY_ID_KEY

    from dataknobs_data.vector.stores.factory import VectorStoreFactory

    store = VectorStoreFactory().create(backend=backend, dimensions=DIMENSIONS)
    await store.initialize()
    try:
        written = {
            ONTOLOGY_ID_KEY: "catalog",
            TAXONOMY_ID_KEY: "categories",
            NODE_ID_KEY: ["tools", "hardware"],
            ALIAS_FORMS_KEY: ["Widget", "ACME widget"],
        }
        await store.add_vectors(
            np.ones((1, DIMENSIONS), dtype=np.float32),
            ids=["catalog:sku-4471"],
            metadata=[dict(written)],
        )

        [(_vector, metadata)] = await store.get_vectors(["catalog:sku-4471"])
        assert metadata is not None
        for key, value in written.items():
            assert metadata[key] == value, f"{backend} did not round-trip {key}"
    finally:
        await store.close()


# --------------------------------------------------------------------------
# What the registry refuses, and why
# --------------------------------------------------------------------------


async def test_a_configured_embedder_with_nothing_injected_is_refused_by_name() -> None:
    """**Refused, not dropped**, and the message names what to pass.

    A section parsed into a field and discarded leaves a registry reporting
    success while holding no store, no embedder and no index --- and no error.
    The refusal is that failure with its silence removed.

    It costs something real and the cost is stated rather than buried: the
    published call site's ``embedder:`` block does not build an index by
    itself, so *typical usage needs no Python* fails for one line of one
    section.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "embedder": {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}},
        }
    )
    registry = OntologyRegistry.from_components(config=OntologyConfig(**document))
    try:
        with pytest.raises(ValidationError, match="embedder") as refused:
            await registry.load()
        assert "dataknobs-llm" in str(refused.value)
    finally:
        await registry.close()


async def test_an_index_section_with_no_store_block_is_refused() -> None:
    """There is no default store, and defaulting to one would be worse."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document(index={"embedder": {}})),
        embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
    )
    try:
        with pytest.raises(ValidationError, match="store"):
            await registry.load()
    finally:
        await registry.close()


async def test_the_store_the_registry_opened_is_closed_and_the_index_is_not_reopened() -> None:
    """What it opened, it closes. The embedder is not a handle and is untouched.

    An embedder has no ``close()`` at all --- three members, none of them a
    lifecycle --- so a registry recording ownership of one would record a
    responsibility it could not discharge and would say so only at DEBUG.
    """
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
    registry = await _registry(_indexed())
    registry._injected_embedder = embedder

    index = registry.index("catalog")
    assert index is not None
    store = index.store
    assert (store, True) in registry._handles

    await registry.close()
    assert store._initialized is False or not registry._handles
    assert not hasattr(embedder, "close")


# --------------------------------------------------------------------------
# The index itself, and the family it is built over
# --------------------------------------------------------------------------


async def _store() -> MemoryVectorStore:
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    return store


async def test_search_batch_carries_searchs_keywords_with_searchs_defaults() -> None:
    """Parity asserted over the signatures, not over the results.

    This pair has lost a keyword from one side already. A batch member that
    quietly drops an option a caller passed is a divergence nothing reports:
    the call still succeeds and the option is simply not applied.
    """
    import inspect

    one = inspect.signature(SemanticIndex.search).parameters
    many = inspect.signature(SemanticIndex.search_batch).parameters

    keywords = {
        name: parameter.default
        for name, parameter in one.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert keywords == {
        name: parameter.default
        for name, parameter in many.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert keywords, "an empty comparison would pass over two members with no keywords at all"


async def test_the_pair_this_class_does_not_ship_is_absent_rather_than_ignored() -> None:
    """An empty answer to a policy a caller asked for is worse than a ``TypeError``.

    The provenance report and its policy keyword arrive together, on both
    members, when the store has the surface to answer them. Until then a
    caller naming one finds out.
    """
    import inspect

    assert not hasattr(SemanticIndex, "observed_embedders")
    for member in (SemanticIndex.search, SemanticIndex.search_batch):
        assert "on_incompatible" not in inspect.signature(member).parameters


async def test_a_metric_the_store_is_not_serving_is_refused_at_construction() -> None:
    """The parameter is a claim the index checks, not a setting it applies.

    The store owns the metric --- it resolves one at construction and its
    search takes no such argument --- so a value here cannot change what the
    store does. Refusing a disagreement is the one honest thing it can do.
    """
    store = await _store()
    try:
        with pytest.raises(ValueError, match="euclidean"):
            SemanticIndex(
                CallableSource(lambda: []),
                DeterministicEmbedder(dimensions=DIMENSIONS),
                store,
                metric=DistanceMetric.EUCLIDEAN,
            )
    finally:
        await store.close()


async def test_two_spellings_of_one_metric_agree() -> None:
    """Compared in canonical form, because six member names name four metrics.

    An assertion firing on the spelling would refuse a store the caller
    configured correctly, while reading as the index catching a mistake ---
    and validating where the caller still holds the mistake depends on the
    mistake being real.
    """
    store = MemoryVectorStore({"dimensions": DIMENSIONS, "metric": "l2"})
    await store.initialize()
    try:
        index = SemanticIndex(
            CallableSource(lambda: []),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
            metric=DistanceMetric.EUCLIDEAN,
        )
        assert index.metric is DistanceMetric.EUCLIDEAN
    finally:
        await store.close()


async def test_no_metric_checks_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """``None`` is the only safe default: cosine would refuse every euclidean store."""
    store = MemoryVectorStore({"dimensions": DIMENSIONS, "metric": "euclidean"})
    await store.initialize()
    try:
        index = SemanticIndex(
            CallableSource(lambda: []), DeterministicEmbedder(dimensions=DIMENSIONS), store
        )
        assert index.metric is None
    finally:
        await store.close()


async def test_a_build_over_an_empty_source_writes_nothing_and_says_zero() -> None:
    """A vocabulary that holds nothing is a vocabulary, not a failure."""
    store = await _store()
    try:
        index = SemanticIndex(
            CallableSource(lambda: []), DeterministicEmbedder(dimensions=DIMENSIONS), store
        )
        assert await index.build() == 0
        assert await store.count() == 0
    finally:
        await store.close()


async def test_the_build_streams_rather_than_collecting_the_source() -> None:
    """A source exists to be larger than memory, and the one call site that
    consumes them all must not take that away.

    Asserted by counting writes at a batch size of one: a build that collected
    first would make exactly one write however the batch size was set.
    """
    from dataknobs_data.vector import semantic_index as module

    store = await _store()
    writes: list[int] = []
    original_write = store.bulk_embed_and_store

    async def counting(texts: list[str], **kwargs: Any) -> list[str]:
        writes.append(len(texts))
        return await original_write(texts, **kwargs)

    store.bulk_embed_and_store = counting  # type: ignore[method-assign]
    original = module.BUILD_BATCH_SIZE
    module.BUILD_BATCH_SIZE = 1
    try:
        index = SemanticIndex(_three_items(), DeterministicEmbedder(dimensions=DIMENSIONS), store)
        assert await index.build() == 3
    finally:
        module.BUILD_BATCH_SIZE = original
        await store.close()

    assert writes == [1, 1, 1]


def _three_items() -> AsyncIndexSource:
    async def rows() -> AsyncIterator[IndexItem]:
        for key, text in (("a", "alpha"), ("b", "beta"), ("c", "gamma")):
            yield IndexItem(id=key, text=text)

    return CallableSource(rows)


async def test_a_decorator_forwards_identity_and_scope_and_decorates_only_text() -> None:
    """One entity, several surface forms, one id --- and nothing else moved.

    A decorator that re-declares is a second place a scope filter is
    translated, and two translation sites start disagreeing; one that rewrites
    an id breaks the only guarantee a hit carries.
    """

    async def one() -> AsyncIterator[IndexItem]:
        yield IndexItem(
            id="catalog:acme",
            text="Acme Corp",
            metadata={ONTOLOGY_ID_KEY: "catalog", ALIAS_FORMS_KEY: ["ACME", "Acme Corporation"]},
        )

    decorated = AliasSource(CallableSource(one, frozenset({"catalog"})), ALIAS_FORMS_KEY)

    items = [item async for item in decorated.stream_items()]

    assert [item.text for item in items] == ["Acme Corp", "ACME", "Acme Corporation"]
    assert {item.id for item in items} == {"catalog:acme"}
    assert all(item.metadata[ONTOLOGY_ID_KEY] == "catalog" for item in items)
    assert decorated.declares() == frozenset({"catalog"})


async def test_an_entity_with_no_aliases_yields_its_own_text_alone() -> None:
    """An entity nobody has another name for is not an error."""

    async def one() -> AsyncIterator[IndexItem]:
        yield IndexItem(id="catalog:sku-8802", text="Bolt", metadata={ALIAS_FORMS_KEY: []})

    decorated = AliasSource(CallableSource(one), ALIAS_FORMS_KEY)
    assert [item.text async for item in decorated.stream_items()] == ["Bolt"]


async def test_the_forms_collide_in_a_store_keyed_on_id() -> None:
    """The measured consequence, asserted so it is visible rather than latent.

    The decorator produces one item per surface form by design, all carrying
    the entity's id; a vector store upserts on id conflict. Three items
    therefore leave **one** row. The canonical text is yielded first, so a
    store that kept the first would keep the entity's own name --- these keep
    the last, which is whichever alias the source happened to list last.

    Which vector id the extra rows should take, and under what grammar, is not
    settled by anything this class can read. This test is what keeps the gap
    from being discovered by a consumer instead.
    """

    async def one() -> AsyncIterator[IndexItem]:
        yield IndexItem(
            id="catalog:acme",
            text="Acme Corp",
            metadata={ALIAS_FORMS_KEY: ["ACME", "Acme Corporation"]},
        )

    store = await _store()
    try:
        index = SemanticIndex(
            AliasSource(CallableSource(one), ALIAS_FORMS_KEY),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
        )
        assert await index.build() == 3
        assert await store.count() == 1
    finally:
        await store.close()


# --------------------------------------------------------------------------
# What the `index:` block reads, and what it refuses
# --------------------------------------------------------------------------


async def test_a_declared_embedder_beside_an_injected_one_is_refused() -> None:
    """The refusal's other configuration, which was not refused at all.

    ``block["embedder"]`` was read only inside ``and self._injected_embedder
    is None``, so with an embedder injected the document's ``embedder:``
    section was never read, never compared and never logged. The comment two
    lines above names that exact failure --- *"a section parsed into a field
    and quietly dropped is the failure that produces a registry reporting
    success while holding no store, no embedder and no index"* --- and the
    block did it in the other branch.

    What makes it invisible rather than merely wrong: every row written
    records the **injected** model under ``MODEL_NAME_KEY``, so the staleness
    contract stays self-consistent while the document, the runbook and every
    reader of the configuration name a different model.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "embedder": {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}},
        }
    )

    with pytest.raises(ValidationError, match="embedder") as refused:
        await _registry(document)

    assert "injected" in str(refused.value)


async def test_a_key_the_index_block_does_not_declare_is_refused() -> None:
    """A typo built an index with no metric check and reported nothing.

    The same silent drop the ``embedder:`` refusal exists for, one key along:
    ``_index_from`` read three names off the block and ignored the rest, so
    ``metirc:`` was configuration a consumer wrote, the registry accepted, and
    nothing acted on.
    """
    document = _document(
        index={"store": {"backend": "memory", "dimensions": DIMENSIONS}, "metirc": "l2"}
    )

    with pytest.raises(ValidationError, match="metirc"):
        await _registry(document)


async def test_the_index_block_configures_the_source_it_builds() -> None:
    """``AliasSource`` shipped reachable only from Python, which is half a feature.

    The block hard-coded ``EntitySourceIndexSource(ontology)`` with its
    default single field, so a document could not ask for ``description`` in
    the embedded text, could not set the separator, and could not reach the
    surface-form decorator at all --- the headline of the layer above it.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "fields": ["name", "description"],
            "join": " / ",
            "aliases": True,
        }
    )
    registry = await _registry(document)
    try:
        index = registry.index("catalog")
        assert index is not None

        # Collected as a list per id rather than a mapping: the decorator
        # yields several items under one id by design, and a dict keeps
        # whichever came last -- which is the collision this composition is
        # documented to have, not something to assert around by accident.
        widget = [
            item.text async for item in index.source.stream_items() if item.id == "catalog:sku-4471"
        ]

        # `fields:` and `join:` both reached the leaf, and the canonical text
        # is first, which is the decorator's own ordering guarantee.
        assert widget[0] == "Acme Widget / a widget"
        # `aliases: true` reached the decorator, so the forms are items of
        # their own rather than sitting unread in metadata.
        assert widget[1:] == ["Widget", "ACME widget"]
    finally:
        await registry.close()


async def test_a_document_that_cannot_build_an_index_leaves_no_store_open() -> None:
    """The comment promises this; two of the three refusals happened after the open.

    ``_vector_store_handle`` ran before ``EntitySourceIndexSource(ontology)``
    and before the metric was parsed, so a document failing either left a
    built, connected, cached store behind proving it tried --- which is what
    the comment above the embedder refusal says must not happen.

    The cache is the only surface that can answer this: nothing public reports
    what a failed load opened, which is itself why the ordering has to be
    right rather than merely tidy.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "fields": ["latin_name"],
        }
    )
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document),
        embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
    )

    with pytest.raises(ValidationError, match="latin_name"):
        await registry.load()

    assert registry._vector_store_cache == {}


async def test_a_metric_the_document_misspells_is_refused_as_a_validation_error() -> None:
    """The block's ``Raises:`` names one exception; a bad spelling raised another.

    Every sibling refusal in this block is a ``ValidationError`` --- it is a
    document being validated --- and a malformed ``metric:`` reached
    ``DistanceMetric.resolve`` and came back as a bare ``ValueError``, so a
    caller catching what the docstring named did not catch this.
    """
    document = _document(
        index={"store": {"backend": "memory", "dimensions": DIMENSIONS}, "metric": "nearest-ish"}
    )

    with pytest.raises(ValidationError, match="nearest-ish"):
        await _registry(document)


async def test_a_build_that_fails_partway_says_how_far_it_got() -> None:
    """The docstring promised an all-or-nothing build and the method is not one.

    *"A full build, not an update: what the source streams now is what the
    store holds after"* --- but batches are written as they fill, so a raise
    from the embedder or the store leaves batches 1..n-1 committed. The count
    lived in a local and the exception carried nothing, so a caller saw a
    failure, read the docstring, and could not tell a store holding nothing
    from one holding seven thousand rows. Both retrying from scratch and
    reporting the index unbuilt are wrong over the second.

    The number is this layer's to report: no store can say how much of one
    build reached it, because the store never saw the stream.
    """
    from dataknobs_data.vector import semantic_index as module

    async def two_then_trouble() -> AsyncIterator[IndexItem]:
        yield IndexItem(id="a", text="alpha")
        yield IndexItem(id="b", text="beta")
        raise OperationError("the source's backend went away mid-stream")

    store = await _store()
    original = module.BUILD_BATCH_SIZE
    module.BUILD_BATCH_SIZE = 1
    try:
        index = SemanticIndex(
            CallableSource(two_then_trouble),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
        )
        with pytest.raises(OperationError, match="wrote 2") as failed:
            await index.build()

        assert failed.value.context["written"] == 2
        assert isinstance(failed.value.__cause__, OperationError)
        assert "went away" in str(failed.value.__cause__)

        # Partial rather than nothing, which is the fact the count is for.
        assert await store.count() == 2
    finally:
        module.BUILD_BATCH_SIZE = original
        await store.close()


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({"fields": "name"}, "fields"),
        ({"fields": 5}, "fields"),
        ({"join": 5}, "join"),
        ({"aliases": "yes"}, "aliases"),
    ],
)
async def test_a_malformed_source_key_is_refused_as_a_validation_error(
    block: dict[str, Any], match: str
) -> None:
    """The block's guarantee is that every refusal in it names a document.

    Which was true of the keys it read and not of the ones it had just gained.
    ``fields: name`` is the shape a consumer writes by hand --- YAML makes a
    bare scalar a string --- and ``tuple("name")`` is four one-character field
    names, so the refusal that follows names ``'a', 'e', 'm', 'n'`` rather than
    the mistake. ``fields: 5`` and ``join: 5`` raised ``TypeError`` and
    ``AttributeError`` instead, the second of them not until the first read.

    ``aliases:`` is checked for being a boolean rather than for truthiness:
    ``aliases: "no"`` is truthy, so the value that most obviously means *off*
    turned it on.
    """
    document = _document(index={"store": {"backend": "memory", "dimensions": DIMENSIONS}, **block})

    with pytest.raises(ValidationError, match=match):
        await _registry(document)
