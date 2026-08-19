"""A manager and a bound destination must not both claim the domain.

:class:`KnowledgeIngestionManager` threads ``domain_id`` per call and
documents that value as authoritative — "a ``domain_id`` carried in the
config is overridden, so the config never re-targets identity". A
destination :class:`RAGKnowledgeBase` that carries a binding of its own
re-targets it anyway, and every surface disagrees in a different way:

* writes are stamped with the destination's domain, not the call's, so
  the chunks land in a scope the caller never asked for;
* the swap's ``clear`` and its tombstone/rollback filters are scoped to
  the destination's binding, so they no longer name the rows the swap is
  replacing.

Neither is recoverable by the manager, and the pairing is a
configuration error rather than a runtime condition — so it is refused
at the first call that reveals it, in the same fail-fast posture the
constructor already takes for a tenant-requiring context shape on an
unbound manager. A destination bound to the *same* domain is not a
conflict and is left alone: it says the same thing the call does.
"""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_common.exceptions import ConfigurationError

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_bots.knowledge.ingestion import KnowledgeIngestionManager
from dataknobs_bots.knowledge.storage.memory import InMemoryKnowledgeBackend
from dataknobs_bots.providers import build_embedding_config, create_embedding_provider
from dataknobs_data.vector.stores import VectorStoreFactory


async def _kb(config: dict[str, Any] | None, *, store_domain: str | None = None) -> Any:
    kwargs: dict[str, Any] = {"backend": "memory", "dimensions": 384}
    if store_domain is not None:
        kwargs["domain_id"] = store_domain
    store = VectorStoreFactory().create(**kwargs)
    await store.initialize()
    embedder = await create_embedding_provider(
        build_embedding_config(embedding_provider="echo", embedding_model="test")
    )
    return RAGKnowledgeBase.from_components(config, vector_store=store, embedding_provider=embedder)


async def _backend(*domains: str) -> InMemoryKnowledgeBackend:
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    for domain in domains:
        await backend.create_kb(domain)
        await backend.put_file(domain, "overview.md", b"# Overview\n\nBody.\n")
    return backend


async def test_ingest_refuses_a_destination_bound_to_another_domain() -> None:
    """The conflict is refused at the call, naming both values."""
    kb = await _kb({"domain_id": "acme"})
    manager = KnowledgeIngestionManager(source=await _backend("umbrella"), destination=kb)

    with pytest.raises(ConfigurationError) as excinfo:
        await manager.ingest("umbrella")

    message = str(excinfo.value)
    assert "umbrella" in message and "acme" in message


async def test_a_store_scoped_destination_conflicts_the_same_way() -> None:
    """The binding need not be configured on the knowledge base itself.

    A destination over a domain-scoped *store* resolves the same
    binding, stamps it onto every chunk, and mis-tags the ingest just as
    thoroughly — so the refusal follows the resolved value rather than
    the configured one.
    """
    kb = await _kb(None, store_domain="acme")
    manager = KnowledgeIngestionManager(source=await _backend("umbrella"), destination=kb)

    with pytest.raises(ConfigurationError):
        await manager.ingest("umbrella")


async def test_ingest_changes_and_reconcile_refuse_it_too() -> None:
    """Every per-domain entry point that reaches the destination."""
    kb = await _kb({"domain_id": "acme"})
    manager = KnowledgeIngestionManager(source=await _backend("umbrella"), destination=kb)

    with pytest.raises(ConfigurationError):
        await manager.ingest_changes("umbrella", since_version=None)
    with pytest.raises(ConfigurationError):
        await manager.reconcile("umbrella")
    with pytest.raises(ConfigurationError):
        await manager.ingest_if_changed("umbrella")


async def test_a_matching_binding_is_not_a_conflict() -> None:
    """One domain per destination is a legitimate, unchanged shape."""
    kb = await _kb({"domain_id": "acme"})
    manager = KnowledgeIngestionManager(source=await _backend("acme"), destination=kb)

    result = await manager.ingest("acme")

    assert result.success
    assert await kb.count() == 1


async def test_an_unbound_destination_is_untouched() -> None:
    """The multi-domain shape the manager exists for."""
    kb = await _kb(None)
    manager = KnowledgeIngestionManager(source=await _backend("bot-a", "bot-b"), destination=kb)

    for domain in ("bot-a", "bot-b"):
        assert (await manager.ingest(domain)).success
    assert await kb.count() == 2
