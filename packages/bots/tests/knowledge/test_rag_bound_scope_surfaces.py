"""Every store-touching surface honours the knowledge base's identity binding.

:class:`RAGKnowledgeBase` binds an identity — ``tenant_id``,
``domain_id``, or both — and its class docstring claims that binding
composes onto reads. It did so for ``query`` and ``hybrid_query`` and
for nothing else: ``count`` reported every scope's rows, and ``clear``
and ``update_metadata_where`` acted on them. On a shared store that
made ``clear()`` cross-scope destruction and made the
skip-if-populated check answer for the wrong scope, so a second tenant
over a store the first had populated was told it was already ingested
and never got any chunks of its own.

Reads compose with **explicit-filter-wins**, so admin tooling can
still read across scopes by naming the key. The two filter-driven
mutations compose with **bound-wins**, mirroring
:meth:`KnowledgeIngestionManager._scope_for_tenant`: a caller cannot
widen a destructive operation past the scope it is bound to. An
*unbound* knowledge base keeps today's semantics on all three — that
is the escape hatch, and the last test pins it.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_bots.knowledge.service import KnowledgeIngestionService
from dataknobs_bots.providers import build_embedding_config, create_embedding_provider
from dataknobs_data.vector.stores import VectorStoreFactory

# Both bindings, so neither can regress on its own. The store stays
# unscoped in every case: these surfaces are about what the *knowledge
# base* composes, and an unscoped store composes nothing of its own.
BINDINGS = ["tenant_id", "domain_id"]


async def _shared_store() -> Any:
    store = VectorStoreFactory().create(backend="memory", dimensions=384)
    await store.initialize()
    return store


async def _bound_kb(store: Any, binding: str | None, value: str | None) -> RAGKnowledgeBase:
    embedder = await create_embedding_provider(
        build_embedding_config(embedding_provider="echo", embedding_model="test")
    )
    config = {binding: value} if binding else None
    return RAGKnowledgeBase.from_components(config, vector_store=store, embedding_provider=embedder)


async def _seed(kb: RAGKnowledgeBase, source: str, body: str) -> None:
    await kb.load_markdown_text(f"# Heading\n\n{body}\n", source=source)


@pytest.mark.parametrize("binding", BINDINGS)
async def test_count_is_scoped_to_the_binding(binding: str) -> None:
    """``count`` reports the bound scope's rows, not the whole store."""
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha content.")
    await _seed(kb_a, "guide.md", "Alpha guide.")
    await _seed(kb_b, "brief.md", "Beta content.")

    assert await kb_a.count() == 2
    assert await kb_b.count() == 1
    assert await kb_a.count(include_stale=True) == 2


@pytest.mark.parametrize("binding", BINDINGS)
async def test_count_subtracts_only_its_own_stale_rows(binding: str) -> None:
    """The ``_stale`` subtraction stays inside the scope it counts.

    ``count`` is ``count(filter) - count(filter ∧ _stale)``. Scoping
    only the first term would subtract another scope's tombstones from
    this scope's total and under-report — negatively, given enough of
    them.
    """
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha content.")
    await _seed(kb_b, "brief.md", "Beta one.")
    await _seed(kb_b, "notes.md", "Beta two.")
    await kb_b.update_metadata_where(None, {"_stale": True})

    assert await kb_a.count() == 1
    assert await kb_b.count() == 0
    assert await kb_b.count(include_stale=True) == 2


@pytest.mark.parametrize("binding", BINDINGS)
async def test_check_needs_ingestion_is_true_for_the_second_scope(binding: str) -> None:
    """A scope with no rows of its own is not skipped as already-populated.

    ``check_needs_ingestion`` is exactly ``count() < min_chunks``, so an
    unscoped ``count`` made the second scope permanently invisible to
    the ingest path — told it was populated, never given anything to
    read.
    """
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")
    service = KnowledgeIngestionService()

    await _seed(kb_a, "overview.md", "Alpha content.")

    assert await service.check_needs_ingestion(kb_a) is False
    assert await service.check_needs_ingestion(kb_b) is True


@pytest.mark.parametrize("binding", BINDINGS)
async def test_clear_leaves_the_other_scope_intact(binding: str) -> None:
    """An unfiltered ``clear`` on a bound KB stays inside its own scope."""
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha content.")
    await _seed(kb_b, "brief.md", "Beta content.")
    assert len(store.metadata_store) == 2

    await kb_b.clear()

    assert await kb_a.count() == 1
    assert await kb_b.count() == 0
    assert len(store.metadata_store) == 1


@pytest.mark.parametrize("binding", BINDINGS)
async def test_clear_with_a_filter_narrows_within_the_scope(binding: str) -> None:
    """A supplied filter narrows further; it cannot widen past the binding."""
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha overview.")
    await _seed(kb_a, "guide.md", "Alpha guide.")
    await _seed(kb_b, "brief.md", "Beta content.")

    await kb_a.clear(filter={"source": "overview.md"})
    assert await kb_a.count() == 1
    assert await kb_b.count() == 1

    # Naming the *other* scope does not reach it: on a destructive
    # surface the binding wins, the inverse of the read side.
    await kb_a.clear(filter={binding: "umbrella"})
    assert await kb_b.count() == 1
    assert await kb_a.count() == 1, "the refused clear fell back on its own scope"


@pytest.mark.parametrize("binding", BINDINGS)
async def test_update_metadata_where_is_scoped(binding: str) -> None:
    """The tombstone-swap primitive does not reach another scope's rows."""
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha content.")
    await _seed(kb_b, "brief.md", "Beta content.")

    updated = await kb_b.update_metadata_where(None, {"_stale": True})

    assert updated == 1
    assert await kb_a.count() == 1
    assert await kb_b.count() == 0


async def test_unbound_kb_retains_unscoped_semantics() -> None:
    """No binding: every surface behaves exactly as it did before.

    The escape hatch. An unbound knowledge base is how an admin
    legitimately counts, clears, or re-tags across every scope in a
    shared store, so ``clear()`` with no filter still wipes everything.
    """
    store = await _shared_store()
    kb_a = await _bound_kb(store, "tenant_id", "acme")
    kb_b = await _bound_kb(store, "domain_id", "bot-b")
    admin = await _bound_kb(store, None, None)

    await _seed(kb_a, "overview.md", "Alpha content.")
    await _seed(kb_b, "brief.md", "Beta content.")

    assert await admin.count() == 2

    touched = await admin.update_metadata_where(None, {"reviewed": True})
    assert touched == 2

    await admin.clear()
    assert len(store.metadata_store) == 0


@pytest.mark.parametrize("binding", BINDINGS)
async def test_clear_naming_another_scope_spares_this_one_too(binding: str) -> None:
    """Naming another scope destroys nothing — not even the caller's own rows.

    Bound-wins was spelled as an unconditional overwrite, so a filter
    naming another scope did not narrow to nothing: it was rewritten
    into the binding's own value and the operation widened from "no
    rows" to "every row in this scope". ``kb_a.clear(domain_id=B)``
    deleted all of A. The caller asked to reach outside the scope and
    got the one outcome worse than being refused — the inverse of what
    they asked for, silently, on a destructive surface.

    The store layer already had the answer: an out-of-scope request
    resolves to the empty-list value ``_match_metadata_filter``
    documents as unsatisfiable on every backend.
    """
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha overview.")
    await _seed(kb_a, "guide.md", "Alpha guide.")
    await _seed(kb_b, "brief.md", "Beta content.")

    await kb_a.clear(filter={binding: "umbrella"})

    assert await kb_a.count() == 2, "clear() naming another scope wiped its own"
    assert await kb_b.count() == 1, "clear() reached across the binding"
    assert len(store.metadata_store) == 3


@pytest.mark.parametrize("binding", BINDINGS)
async def test_update_metadata_where_naming_another_scope_is_a_no_op(binding: str) -> None:
    """The tombstone primitive refuses the same request the same way."""
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha content.")
    await _seed(kb_b, "brief.md", "Beta content.")

    updated = await kb_a.update_metadata_where({binding: "umbrella"}, {"_stale": True})

    assert updated == 0, "an out-of-scope filter matched rows"
    assert await kb_a.count() == 1, "tombstoned its own scope instead"
    assert await kb_b.count() == 1, "reached across the binding"


@pytest.mark.parametrize("binding", BINDINGS)
async def test_a_filter_naming_this_scope_still_matches(binding: str) -> None:
    """Refusing the *other* scope must not refuse this one.

    The unsatisfiable value is reserved for a genuine disagreement; a
    caller naming the value it is already bound to is asking for what
    it would have been given anyway, and the operation proceeds.
    """
    store = await _shared_store()
    kb_a = await _bound_kb(store, binding, "acme")
    kb_b = await _bound_kb(store, binding, "umbrella")

    await _seed(kb_a, "overview.md", "Alpha content.")
    await _seed(kb_b, "brief.md", "Beta content.")

    await kb_a.clear(filter={binding: "acme"})

    assert await kb_a.count() == 0, "naming its own scope was refused"
    assert await kb_b.count() == 1
