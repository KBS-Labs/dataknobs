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
# A row built through the decorator says what its text came from
# --------------------------------------------------------------------------


async def _hits_by_id(index: SemanticIndex) -> dict[str, Any]:
    """One hit per row, keyed by id --- the store holds one row per id."""
    hits = await index.search("Acme", k=10)
    return {hit.record.id: hit for hit in hits}


async def test_every_row_an_aliased_index_writes_says_what_its_text_came_from() -> None:
    """``aliases: true`` wrapped the leaf in a decorator that could not say.

    The index reads ``source_field`` off its source with a fall-back, and the
    decorator had none, so every row an ``aliases: true`` document built
    recorded ``None`` --- the value this pair exists to distinguish from.
    The row with no aliases is the plain case: its text is the leaf's
    composition, and it says so in the leaf's own spelling.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "fields": ["name", "description"],
            "aliases": True,
        }
    )
    registry = await _registry(document)
    try:
        index = registry.index("catalog")
        assert index is not None
        await index.build()

        hits = await _hits_by_id(index)

        bolt = hits["catalog:sku-8802"]
        assert bolt.source_text == "Bolt"
        assert bolt.vector_field == "name,description"
    finally:
        await registry.close()


async def test_the_row_a_collapse_leaves_names_the_field_its_alias_came_from() -> None:
    """The surviving row holds an alias, and its pair has to say so.

    The forms collide in a store keyed on id and the last one survives, so
    an aliased entity's only row holds an alias. Forwarding the leaf's answer
    alone would label that row ``("Acme Corporation", "name,description")``
    --- a text that is neither the name nor the description, attributed to
    both. So each form states its own field, and the check here is the one a
    reader can make from the hit alone: the text is among the values of the
    field the row names.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "fields": ["name", "description"],
            "aliases": True,
        }
    )
    registry = await _registry(document)
    try:
        index = registry.index("catalog")
        assert index is not None
        await index.build()

        hits = await _hits_by_id(index)

        for entity_id, last_form in (("catalog:acme", "ACME"), ("catalog:sku-4471", "ACME widget")):
            hit = hits[entity_id]
            assert hit.source_text == last_form
            assert hit.vector_field == ALIAS_FORMS_KEY
            assert hit.source_text in hit.metadata[hit.vector_field]
    finally:
        await registry.close()


async def test_an_item_s_own_source_field_outranks_the_one_its_metadata_carries() -> None:
    """``item.source_field`` > a metadata key the inner wrote > the batch value.

    The decorator copies the inner item's metadata onto every form, so an
    inner source that wrote ``source_field`` into its own metadata would
    otherwise label each alias with the canonical text's field --- the false
    pair above, arriving by another route. The middle rung still holds for a
    row whose item says nothing: the store's per-row route is unchanged.

    The same field gives the escape hatch a way to state what it indexed,
    per item, with no constructor parameter to add.
    """

    async def inner() -> AsyncIterator[IndexItem]:
        yield IndexItem(
            id="acme",
            text="Acme Corp",
            metadata={"source_field": "name", ALIAS_FORMS_KEY: ["ACME"]},
        )
        yield IndexItem(id="bolt", text="Bolt", metadata={"source_field": "name"})

    async def titled() -> AsyncIterator[IndexItem]:
        yield IndexItem(id="guide", text="Acme guide", source_field="title")

    store = await _store()
    try:
        embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
        aliased = SemanticIndex(
            AliasSource(CallableSource(inner), ALIAS_FORMS_KEY), embedder, store
        )
        await aliased.build()
        await SemanticIndex(CallableSource(titled), embedder, store).build()

        hits = await _hits_by_id(aliased)

        assert (hits["acme"].source_text, hits["acme"].vector_field) == ("ACME", ALIAS_FORMS_KEY)
        assert (hits["bolt"].source_text, hits["bolt"].vector_field) == ("Bolt", "name")
        assert (hits["guide"].source_text, hits["guide"].vector_field) == ("Acme guide", "title")
    finally:
        await store.close()


# --------------------------------------------------------------------------
# What the `index:` block reads, and what it refuses
# --------------------------------------------------------------------------


async def _loaded_with(document: dict[str, Any], embedder: Any) -> OntologyRegistry:
    """A loaded registry whose embedder the caller chose.

    ``_registry`` above injects one of its own, so a test about *which*
    embedder was injected cannot go through it.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document), embedder=embedder
    )
    await registry.load()
    return registry


def _declaring(model: str, provider: str | None = None) -> dict[str, Any]:
    """:data:`CATALOG` with an ``index:`` section naming a model."""
    embedding: dict[str, Any] = {"model": model}
    if provider is not None:
        embedding["provider"] = provider
    return _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "embedder": {"embedding": embedding},
        }
    )


async def test_a_declared_embedder_that_disagrees_with_the_injected_one_is_refused() -> None:
    """The document and the process name different models, and it is said.

    ``block["embedder"]`` was once read only inside ``and
    self._injected_embedder is None``, so with an embedder injected the
    document's ``embedder:`` section was never read, never compared and never
    logged --- *"a section parsed into a field and quietly dropped is the
    failure that produces a registry reporting success while holding no
    store, no embedder and no index"*, in the other branch. What made it
    invisible rather than merely wrong: every row written records the
    **injected** model, so the staleness contract stays self-consistent while
    the document, the runbook and every reader of the configuration name a
    different model.
    """
    with pytest.raises(ValidationError, match="embedder") as refused:
        await _loaded_with(
            _declaring("mxbai-embed-large"),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text:latest"),
        )

    message = str(refused.value)
    assert "mxbai-embed-large" in message and "nomic-embed-text" in message


async def test_a_declared_embedder_that_agrees_is_a_claim_the_document_may_make() -> None:
    """The repair, and the whole of what changed: agreement builds the index.

    The refusal this replaces was added to *make the disagreement visible*,
    and a comparison serves that intent more completely than a refusal does:
    refusal left the document unable to state the model at all, while the
    number a ``kind: semantic`` rung's ``threshold:`` means something against
    is a property of that model. The refusal's own stated reason --- *"cannot
    build the declared one to compare"* --- is about **building**, and the
    injected embedder already publishes ``model_id``.

    **The tag rule, measured.** A document declares ``nomic-embed-text`` and
    an Ollama-backed embedder publishes ``ollama:nomic-embed-text:latest``:
    the provider is a prefix the document may omit and the tag is a suffix it
    may omit, and requiring either would refuse every correctly configured
    deployment. Both spellings below name one model.
    """
    for document in (
        _declaring("nomic-embed-text"),
        _declaring("nomic-embed-text", provider="ollama"),
        _declaring("nomic-embed-text:latest", provider="ollama"),
    ):
        registry = await _loaded_with(
            document,
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text:latest"),
        )
        try:
            assert registry.index("catalog") is not None
        finally:
            await registry.close()


async def test_two_stated_tags_that_differ_are_two_models() -> None:
    """The omission rule is *may omit*, which is not *is always ignored*.

    ``_untagged`` stripped the tag from **both** sides before comparing, so
    a document declaring ``nomic-embed-text:v1.5`` agreed with an embedder
    publishing ``nomic-embed-text:latest``. Those are different weights, and
    a version bump is the one change that most reliably invalidates a
    calibrated ``threshold:`` --- which is this comparison's stated reason
    for existing. The rule the docstring gives is that the tag is *"a suffix
    either side may omit"*; stripping a tag the other side also states
    discards the only thing distinguishing them.

    Where one side omits it there is nothing to compare and the base names
    decide, which is the case the test above pins and this must not break.
    """
    with pytest.raises(ValidationError, match="embedder") as refused:
        await _loaded_with(
            _declaring("nomic-embed-text:v1.5", provider="ollama"),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text:latest"),
        )

    message = str(refused.value)
    assert "v1.5" in message and "latest" in message, (
        "the message must show both tags, since the base names are identical "
        f"and the tags are the whole disagreement: {message}"
    )


async def test_two_stated_tags_that_match_are_one_model() -> None:
    """The positive control: comparing tags is not refusing them."""
    registry = await _loaded_with(
        _declaring("nomic-embed-text:v1.5", provider="ollama"),
        DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text:v1.5"),
    )
    try:
        assert registry.index("catalog") is not None
    finally:
        await registry.close()


async def test_an_identity_named_verbatim_is_the_model_it_names() -> None:
    """The remedy the refusal prints has to clear the refusal.

    The message names both sides --- *"naming model X, and the injected
    embedder publishes Y"* --- so the obvious repair is to write Y into the
    document. That was refused, for every ``model_id`` carrying a colon: the
    published side was split on its first colon as ``provider:model`` and the
    document's side was not, so the two halves of an identical pair were
    compared against each other and found to differ. The refusal then said
    *"one of them is wrong about what this index holds"* of two identical
    strings.

    Nothing promises the split. ``TextEmbedder.model_id`` says in those words
    that it does not promise a format, and only one of the three shipped
    implementations spells ``provider:model`` --- the bots knowledge-base
    adapter publishes ``kb:<name>`` and ``DeterministicEmbedder`` publishes a
    bare word.
    """
    for published in (
        "nomic-embed-text:latest",
        "ollama:nomic-embed-text",
        "ollama:nomic-embed-text:latest",
        "kb:my-model",
        "deterministic",
    ):
        registry = await _loaded_with(
            _declaring(published),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id=published),
        )
        try:
            assert registry.index("catalog") is not None
        finally:
            await registry.close()


async def test_an_embedder_publishing_no_provider_can_still_be_named() -> None:
    """A colon is not a provider, and a tagged bare name is the common case.

    An embedder publishing ``nomic-embed-text:latest`` --- no provider prefix,
    which the protocol permits and a consumer implementation wrapping an
    Ollama client naturally produces --- was read as provider
    ``nomic-embed-text`` and model ``latest``. A document declaring
    ``nomic-embed-text`` was then refused, with a message asserting the
    embedder publishes a different model.

    The provider is a prefix **either side may omit**, exactly as the tag is a
    suffix either side may omit. Where the published side omits it there is no
    provider to compare, so a provider the document states is not contradicted
    by one the embedder never published.
    """
    for document in (
        _declaring("nomic-embed-text"),
        _declaring("nomic-embed-text:latest"),
        _declaring("nomic-embed-text", provider="ollama"),
    ):
        registry = await _loaded_with(
            document,
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="nomic-embed-text:latest"),
        )
        try:
            assert registry.index("catalog") is not None
        finally:
            await registry.close()


async def test_a_declared_provider_the_injected_embedder_is_not_is_refused() -> None:
    """Right model name, wrong provider, is still two different geometries.

    ``model_id`` is ``provider:model`` for the reason its own docstring gives
    --- *"so that two embedders reaching the same model agree, and two
    reaching different models do not"*. Two providers serving a same-named
    model are not guaranteed to be the same weights, and a ``threshold:``
    calibrated against one means nothing against the other.
    """
    with pytest.raises(ValidationError, match="embedder") as refused:
        await _loaded_with(
            _declaring("nomic-embed-text", provider="openai"),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text"),
        )

    assert "openai" in str(refused.value)


async def test_a_key_the_embedder_block_does_not_read_is_refused() -> None:
    """The same silent drop the block's own no-``model:`` refusal exists for.

    ``embedder:`` is a **claim**, not a spec: the registry cannot build an
    embedder and does not try, so ``dimensions:`` and ``api_base:`` written
    here configure nothing. Reading two keys and passing over the rest is
    *"configuration the registry accepted and acted on in no way"* --- the
    words the no-``model:`` refusal two tests down uses about itself, and the
    reason ``metirc:`` is refused one section up.

    Partially reading the block is what makes the remainder surprising: a
    reader who sees ``model:`` checked has every reason to think the keys
    beside it are too.
    """
    for block, named in (
        (
            {"model": "nomic-embed-text", "dimensions": 256, "api_base": "http://x"},
            ["api_base", "dimensions"],
        ),
        ({"embedding": {"model": "nomic-embed-text", "base_url": "http://x"}}, ["base_url"]),
    ):
        document = _document(
            index={
                "store": {"backend": "memory", "dimensions": DIMENSIONS},
                "embedder": block,
            }
        )
        with pytest.raises(ValidationError, match="does not read") as refused:
            await _loaded_with(
                document,
                DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text"),
            )
        assert str(named) in str(refused.value), refused.value


async def test_the_claim_is_not_read_out_of_whatever_mapping_happens_to_carry_a_model() -> None:
    """One nesting is supported, and it is named rather than searched for.

    ``dataknobs-llm`` spells its own configuration ``embedder: {embedding:
    {...}}``, so that one level is read. Scanning *every* nested mapping for
    a ``model`` key instead made any sub-block the claim --- measured, a
    ``retry:`` block naming a model refused a correctly configured
    deployment, reporting a model the document never declared as the model
    the document declares.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "embedder": {"retry": {"model": "a-model-nobody-declared"}},
        }
    )

    with pytest.raises(ValidationError, match="does not read") as refused:
        await _loaded_with(
            document,
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text"),
        )

    message = str(refused.value)
    assert "retry" in message
    assert "a-model-nobody-declared" not in message, (
        "the diagnosis is the unread key, not a model disagreement the document "
        f"never stated: {message}"
    )

    # Pinned at the reader too, not only through the key check in front of it.
    # Widening `EMBEDDER_CLAIM_KEYS` one day must not quietly restore the scan:
    # the refusal above would stop firing and a scanning reader would answer
    # again, with nothing in between to say so.
    from dataknobs_data.ontology.registry import _stated_model

    assert _stated_model({"retry": {"model": "a-model-nobody-declared"}}) is None
    assert _stated_model({"embedding": {"model": "nomic-embed-text"}}) == (
        None,
        "nomic-embed-text",
    )


async def test_a_claim_split_across_the_two_levels_is_refused() -> None:
    """Half a claim at each level is half a claim read.

    The block states the model and the provider at **one** level. Written at
    two, one of them is dropped, and both drops are the failure this section
    refuses elsewhere:

    * ``{provider: openai, embedding: {model: X}}`` loaded clean with the
      provider unread --- and ``openai`` against an ``ollama`` embedder is
      exactly the disagreement the sibling test below has refused all along;
    * ``{model: X, embedding: {model: Y}}`` loaded clean on ``X``, with a
      second, contradictory model claim in the same block read by nothing.
    """
    for block in (
        {"provider": "openai", "embedding": {"model": "nomic-embed-text"}},
        {"model": "nomic-embed-text", "embedding": {"model": "mxbai-embed-large"}},
    ):
        document = _document(
            index={
                "store": {"backend": "memory", "dimensions": DIMENSIONS},
                "embedder": block,
            }
        )
        with pytest.raises(ValidationError, match="does not read"):
            await _loaded_with(
                document,
                DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text"),
            )


async def test_the_three_shapes_a_claim_is_written_in_still_load() -> None:
    """The positive control: refusing a key is not refusing the block.

    Flat, flat with a provider, and the ``embedding:`` nesting
    ``dataknobs-llm``'s own configuration uses. All three are what the guide
    publishes and all three must keep loading.
    """
    for block in (
        {"model": "nomic-embed-text"},
        {"model": "nomic-embed-text", "provider": "ollama"},
        {"embedding": {"model": "nomic-embed-text", "provider": "ollama"}},
    ):
        document = _document(
            index={
                "store": {"backend": "memory", "dimensions": DIMENSIONS},
                "embedder": block,
            }
        )
        registry = await _loaded_with(
            document,
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="ollama:nomic-embed-text"),
        )
        try:
            assert registry.index("catalog") is not None
        finally:
            await registry.close()


async def test_an_embedder_block_naming_no_model_is_refused_as_an_unreadable_claim() -> None:
    """A claim the registry cannot read is not a claim it may pass over.

    This is the ``metric:`` precedent: a spelling the library cannot resolve
    is refused rather than ignored, because accepting a block and acting on
    it in no way is the silent drop the whole section exists to prevent.
    """
    document = _document(
        index={
            "store": {"backend": "memory", "dimensions": DIMENSIONS},
            "embedder": {"embedding": {"provider": "ollama"}},
        }
    )

    with pytest.raises(ValidationError, match="embedder") as refused:
        await _loaded_with(document, DeterministicEmbedder(dimensions=DIMENSIONS))

    # Asserted on the *reason*, not on the word "model": the refusal this
    # replaced also said "model", so a looser match would pass against the
    # unfixed code and pin nothing.
    assert "names no `model:`" in str(refused.value)


async def test_an_embedder_with_no_identity_is_reported_rather_than_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No ``model_id`` is no claim to check, and the document is not refused for it.

    ``model_id`` is optional on the ``TextEmbedder`` protocol, and an
    embedder publishing none leaves the comparison with one side. Refusing
    would make a legitimate embedder unusable with a legitimate document;
    saying nothing would make an unchecked claim look checked. So it is
    reported and the load continues --- which is the one place this check
    warns rather than raising, because the document is not what is wrong.
    """
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="")

    with caplog.at_level("WARNING"):
        registry = await _loaded_with(_declaring("nomic-embed-text"), embedder)
    try:
        assert registry.index("catalog") is not None
        reports = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(reports) == 1
        assert "nomic-embed-text" in reports[0].getMessage()
    finally:
        await registry.close()


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


@pytest.mark.parametrize("block", [["store"], "memory"], ids=["a-list", "a-string"])
async def test_an_index_block_that_is_not_a_mapping_is_refused(block: Any) -> None:
    """The key check ran ``set(block)`` on whatever it was handed.

    A string was refused for *declaring* its own letters as keys, and a list
    of mappings raised a bare ``TypeError``. The refusal says what the block
    has to be instead.
    """
    with pytest.raises(ValidationError) as refused:
        await _registry(_document(index=block))

    message = str(refused.value)
    assert "`index:` must be a mapping" in message
    assert type(block).__name__ in message


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


# --------------------------------------------------------------------------
# The staleness key, compared
# --------------------------------------------------------------------------


def _breeds() -> AsyncIndexSource:
    """A corpus small enough to search exhaustively and real enough to rank."""

    async def rows() -> AsyncIterator[IndexItem]:
        for key, text in (
            ("a", "a spaniel is a gundog"),
            ("b", "a beagle is a hound"),
            ("c", "a hound tracks by scent"),
        ):
            yield IndexItem(id=key, text=text)

    return CallableSource(rows)


async def test_a_hit_written_by_another_model_is_reported_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A build job and a service with a configuration change between them.

    ``SemanticIndex`` writes the embedder's ``model_id`` on every row and
    says what for --- *"which is what makes a stored vector's staleness
    judgeable by something that never saw this object"*. Measured before
    this report: a store built under one model and searched through
    another returned three ranked hits, raised nothing, and logged nothing
    at warning or above. The datum was recorded and there was nowhere to
    stand to read it --- the key is legible in the metadata *of a hit*, so
    the check a consumer would write can only run after the query it would
    have invalidated, and only on a query that returned something.

    Reported once per index instance, on the first hit that disagrees. The
    read path had no report of any kind before this one: the empty-source
    ``logger.info`` is on the **build** path and fires once per build, so
    this is the first thing ``_search_many`` has ever said.
    """
    store = await _store()
    try:
        builder = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="build-v1")
        await SemanticIndex(_breeds(), builder, store).build()

        searcher = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="serve-v2")
        index = SemanticIndex(_breeds(), searcher, store)
        with caplog.at_level("WARNING"):
            first = await index.search("hound", k=3)
            second = await index.search("gundog", k=3)

        assert first and second, "the search still answers; the report is beside it"
        reports = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(reports) == 1, "once per index, not once per hit and not once per query"
        message = reports[0].getMessage()
        assert "build-v1" in message and "serve-v2" in message

    finally:
        await store.close()


async def test_a_hit_written_by_the_same_model_is_not_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The positive control: agreement is silent.

    Without this, a report that fired unconditionally would pass the test
    above and cry wolf on every correctly configured deployment.
    """
    store = await _store()
    try:
        embedder = DeterministicEmbedder(dimensions=DIMENSIONS, model_id="v1")
        await SemanticIndex(_breeds(), embedder, store).build()

        index = SemanticIndex(
            _breeds(),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="v1"),
            store,
        )
        with caplog.at_level("WARNING"):
            assert await index.search("hound", k=3)

        assert [record for record in caplog.records if record.levelname == "WARNING"] == []
    finally:
        await store.close()


async def test_a_row_carrying_no_model_name_is_not_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Absent is not disagreement.

    ``add_records`` omits ``model_name`` entirely when the field carries no
    name, so a store written before the key existed --- or through the raw
    ``embedding_fn`` path, which has no identity to default from --- holds
    rows with nothing to compare. Reporting there would fire on every
    pre-existing store and say only that it is old.

    The rows are written straight through ``add_vectors`` rather than through
    a build, because every shipped embedder names itself:
    ``DeterministicEmbedder`` defaults ``model_id`` to ``"deterministic"``
    even when the caller passes none, so a build cannot produce this state.
    """
    import numpy as np

    store = await _store()
    try:
        await store.add_vectors(np.ones((3, DIMENSIONS), dtype=np.float32), ids=["a", "b", "c"])

        index = SemanticIndex(
            _breeds(), DeterministicEmbedder(dimensions=DIMENSIONS, model_id="serve-v2"), store
        )
        with caplog.at_level("WARNING"):
            assert await index.search("hound", k=3)

        assert [record for record in caplog.records if record.levelname == "WARNING"] == []
    finally:
        await store.close()


# --------------------------------------------------------------------------
# The comparison runs on what the store answered, not on what survived
# the caller's threshold
# --------------------------------------------------------------------------
#
# ``D338`` ruled that ``SemanticIndex`` *"reports once, the first time a search
# returns a hit whose ``model_name`` is not its own embedder's ``model_id``"*,
# and the limit it accepted is the one the key's placement forces: the name is
# legible in the metadata **of a hit**, so nothing can be said about a query
# the store answered with nothing.
#
# A threshold is not that. Measured on the unfixed code: the store answered a
# cross-model query with **three** rows, every one carrying ``build-v1``, and
# ``_search_many`` dropped them before comparing. The evidence was in hand and
# was discarded with the answer.
#
# Why that is the *likeliest* presentation is reasoning rather than a
# measurement here, and the distinction matters: vectors from two embedding
# spaces score against each other arbitrarily, so a threshold tuned under one
# model has no meaning under two and will tend to keep nothing --- leaving a
# caller an empty list and no account of it. ``D338`` recorded the same limit
# on its own evidence (*"under a hash embedder the wrong answers look wrong;
# under two real models of the same width they would look ordinary"*), and
# ``DeterministicEmbedder`` is a hash embedder, so these tests can pin where
# the comparison runs but not how a real pair of models would score.
#
# The sibling implementation gets this right. ``DedupChecker._find_similar``
# compares the stored name against its own on every candidate the store
# returned and applies ``similarity_threshold`` afterwards, in the same loop.


async def test_a_mismatch_is_reported_even_when_the_threshold_keeps_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reproducer: a filtered-away answer is not absent evidence.

    ``threshold=2.0`` keeps nothing, which is the shape a real mismatch
    tends to produce for a threshold tuned under one model. What this pins
    is narrower and is the part that is measurable here: the store answered
    with three rows carrying ``build-v1``, so the comparison had everything
    it needed, and the filter is the caller's view of the answer rather than
    a limit on what can be compared.
    """
    store = await _store()
    try:
        await SemanticIndex(
            _breeds(),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="build-v1"),
            store,
        ).build()

        index = SemanticIndex(
            _breeds(),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="serve-v2"),
            store,
        )
        with caplog.at_level("WARNING"):
            assert await index.search("hound", k=3, threshold=2.0) == []

        reports = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(reports) == 1, (
            "the store returned three rows carrying 'build-v1'; the threshold "
            "removed them from the answer, not from the evidence"
        )
        assert "build-v1" in reports[0].getMessage()
    finally:
        await store.close()


async def test_a_search_the_store_answers_with_nothing_reports_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The limit that is real, expressed the way it is actually reached.

    This is what ``D338`` accepted: the model name lives in a hit's metadata,
    so a query the **store** answered with nothing leaves nothing to compare.
    An empty store is how that happens; a threshold is not.
    """
    store = await _store()
    try:
        index = SemanticIndex(
            _breeds(),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="serve-v2"),
            store,
        )
        with caplog.at_level("WARNING"):
            assert await index.search("hound", k=3) == []

        assert [record for record in caplog.records if record.levelname == "WARNING"] == []
    finally:
        await store.close()


# --------------------------------------------------------------------------
# The fact is handed back, not only logged
# --------------------------------------------------------------------------
#
# A log line is not an answer a program can act on, and this one is emitted
# once per instance --- so a service that starts before its log sink, or reads
# its logs nowhere, has the datum pass by a second time. ``D338`` declined
# option (b), *"publish the stored key as a member, so an index or a store
# answers WHAT MODEL WROTE THESE ROWS without a search"*, because it *"is a
# scan on some backends and undefined over a store several vocabularies
# share"*.
#
# Publishing what the searches already compared is neither of those things.
# It reads no row the search did not already read, and over a store several
# vocabularies share it answers the only question that is well-posed there:
# which foreign models this index has actually been ranking against.
#
# The name is the sibling's. ``DedupResult.mismatched_model_ids`` carries the
# same fact for the same reason --- *"because every candidate is still the
# best answer available; what changes is that the caller can now tell the
# answer is untrustworthy"* --- and two names for one fact is how a consumer
# who found one fails to find the other.


async def test_the_mismatched_model_is_handed_back_not_only_logged() -> None:
    """The consumer's standing place: a member, not a log line."""
    store = await _store()
    try:
        await SemanticIndex(
            _breeds(),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="build-v1"),
            store,
        ).build()

        index = SemanticIndex(
            _breeds(),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="serve-v2"),
            store,
        )
        assert index.mismatched_model_ids == [], "nothing searched, nothing seen"

        await index.search("hound", k=3)

        assert index.mismatched_model_ids == ["build-v1"]
    finally:
        await store.close()


async def test_every_foreign_model_is_named_though_only_one_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The division of labour between the log and the member.

    The log is a one-shot alert and stays one, as ruled. The member is the
    complete fact, so a store holding rows from two earlier models names both
    --- which is what a caller needs to decide what to re-embed, and exactly
    what a report that stops at the first cannot say.
    """
    store = await _store()
    try:
        for model, source in (
            ("build-v1", _breeds()),
            ("build-v2", CallableSource(_one_more)),
        ):
            await SemanticIndex(
                source, DeterministicEmbedder(dimensions=DIMENSIONS, model_id=model), store
            ).build()

        index = SemanticIndex(
            _breeds(),
            DeterministicEmbedder(dimensions=DIMENSIONS, model_id="serve-v3"),
            store,
        )
        with caplog.at_level("WARNING"):
            await index.search("hound", k=10)

        assert index.mismatched_model_ids == ["build-v1", "build-v2"]
        reports = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(reports) == 1, "the alert is once per index; the member holds the rest"
    finally:
        await store.close()


async def _one_more() -> AsyncIterator[IndexItem]:
    """A fourth row, written under a second model, sharing no id with the first three."""
    yield IndexItem(id="d", text="a terrier goes to ground")


async def test_a_matching_model_leaves_the_member_empty() -> None:
    """The positive control: agreement publishes nothing, as it logs nothing."""
    store = await _store()
    try:
        await SemanticIndex(
            _breeds(), DeterministicEmbedder(dimensions=DIMENSIONS, model_id="v1"), store
        ).build()

        index = SemanticIndex(
            _breeds(), DeterministicEmbedder(dimensions=DIMENSIONS, model_id="v1"), store
        )
        await index.search("hound", k=3)

        assert index.mismatched_model_ids == []
    finally:
        await store.close()


# --------------------------------------------------------------------------
# A build that stops partway closes what it opened
# --------------------------------------------------------------------------


class _HoldingSource:
    """A source that holds something for as long as its stream is open.

    The shape two shipped backends have under
    :class:`~dataknobs_data.vector.RecordFieldSource`: ``stream_read`` on
    PostgreSQL yields from inside ``pool.acquire()`` and an open
    ``conn.transaction()``, and on Elasticsearch from inside a scroll
    context cleared in a ``finally``. Neither releases until the generator
    is **closed**, which is the property asserted here; a memory backend
    holds nothing, so nothing in-tree can stand in for them.

    Not a mock. It is a real async generator with a real acquire/release
    pair, and the pair is the whole subject.
    """

    def __init__(self) -> None:
        self.held = 0
        self.releases = 0

    def declares(self) -> frozenset[str]:
        return frozenset()

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        self.held += 1
        try:
            for key in ("a", "b", "c"):
                yield IndexItem(id=key, text=f"row {key}")
        finally:
            self.held -= 1
            self.releases += 1


async def test_a_build_that_fails_partway_closes_the_stream_it_opened() -> None:
    """The failure path abandoned the generator and left its cleanup to the collector.

    ``build`` opens the source's stream and, on any failure, raises out of
    the ``async for`` without closing it. An abandoned async generator runs
    its ``finally`` when the interpreter finalizes it --- a later turn of the
    loop --- so whatever the source was holding was still held when the
    caller got the error and started deciding what to do about it.

    Asserted at the moment the error arrives rather than afterwards, because
    "eventually released" is what the unfixed code already does.

    The width mismatch is a real in-tree failure rather than a raising
    stand-in: a store declaring one width handed vectors of another is what
    ``_check_batch_width`` exists to refuse.
    """
    from dataknobs_data.vector import semantic_index as module

    store = await _store()
    source = _HoldingSource()
    original = module.BUILD_BATCH_SIZE
    module.BUILD_BATCH_SIZE = 1
    try:
        index = SemanticIndex(source, DeterministicEmbedder(dimensions=DIMENSIONS * 2), store)
        with pytest.raises(OperationError) as failed:
            await index.build()

        assert source.held == 0, "the stream was still open when the caller got the error"
        assert source.releases == 1, "closed once, not left to the collector and closed twice"
        assert failed.value.context == {"written": 0}
    finally:
        module.BUILD_BATCH_SIZE = original
        await store.close()


#: :data:`CATALOG` with the one description removed, so ``fields:
#: ["description"]`` composes empty text for every entity rather than some.
NO_DESCRIPTIONS: dict[str, Any] = {
    **CATALOG,
    "entities": [
        {key: value for key, value in entity.items() if key != "description"}
        for entity in CATALOG["entities"]
    ],
}


async def test_the_report_a_closed_stream_makes_reaches_the_failing_build(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The two halves of the close meet here, and nothing else asserts that.

    ``EntitySourceIndexSource`` reports an all-empty stream from a
    ``finally``, and ``build`` closes the stream it opened. Each is testable
    alone in its own package; that a real vocabulary's report reaches a real
    failing build crosses the boundary between them, which is where a fix
    written in two halves comes apart.

    A build failing on the store's width is also the case where the report
    is worth reading: the caller is holding an error about the store and the
    stream is telling it the text was empty anyway, which changes what a
    retry should change.
    """
    from dataknobs_data.vector import semantic_index as module

    registry = await _registry(
        {
            **NO_DESCRIPTIONS,
            "index": {
                "store": {"backend": "memory", "dimensions": DIMENSIONS},
                "fields": ["description"],
            },
        }
    )
    store = await _store()
    original = module.BUILD_BATCH_SIZE
    module.BUILD_BATCH_SIZE = 1
    try:
        configured = registry.index("catalog")
        assert configured is not None
        index = SemanticIndex(
            configured.source, DeterministicEmbedder(dimensions=DIMENSIONS * 2), store
        )

        caplog.clear()
        with caplog.at_level("WARNING"), pytest.raises(OperationError):
            await index.build()

        empty_text = [
            record
            for record in caplog.records
            if record.levelname == "WARNING" and "composed empty text" in record.getMessage()
        ]
        assert len(empty_text) == 1, "the stream's report arrived with the build's failure"
        assert "did not finish" in empty_text[0].getMessage()
    finally:
        module.BUILD_BATCH_SIZE = original
        await store.close()
        await registry.close()
