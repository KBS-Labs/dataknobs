"""Domain identity on the direct :class:`RAGKnowledgeBase` ingest path.

Two knowledge bases over one physical vector store, each scoped to its
own domain, must not collide. The chunk-id derivation already folds
``domain_id`` when it is present in chunk metadata
(:attr:`RAGKnowledgeBase._CHUNK_ID_PREFIX_KEYS`) — what was missing is
the binding that supplies the value, so a KB driven without a
:class:`KnowledgeIngestionManager` derived every id from the source
stem alone and the second domain's ``overview.md`` landed on the first
domain's ``overview_0``.

The binding resolves from the KB's own config when set and otherwise
from the bound vector store, which already carries ``domain_id`` for
its own read/write scoping. Deriving it means a consumer who scoped the
*store* — the population this defect actually reaches — gets correct
namespacing without configuring the same value twice, and the chunk-id
namespace cannot disagree with the tag the store stamps on the row.

An unbound KB over an unscoped store is untouched: its ids keep the
historical ``<stem>_<index>`` shape, which test 4 pins byte-for-byte.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dataknobs_bots.knowledge import KnowledgeIngestionManager, RAGKnowledgeBase
from dataknobs_bots.knowledge.service import KnowledgeIngestionService
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend
from dataknobs_bots.providers import build_embedding_config, create_embedding_provider
from dataknobs_data.vector.stores import VectorStoreFactory

_STORE = {"backend": "memory", "dimensions": 384}


async def _make_store(domain_id: str | None = None) -> Any:
    """Build and initialize a real ``MemoryVectorStore``, optionally scoped."""
    kwargs: dict[str, Any] = dict(_STORE)
    if domain_id is not None:
        kwargs["domain_id"] = domain_id
    store = VectorStoreFactory().create(**kwargs)
    await store.initialize()
    return store


def _share_physical_rows(primary: Any, *views: Any) -> None:
    """Point several domain-scoped stores at one physical row set.

    A real shared store — pgvector, chroma — is one table that many
    scoped clients open, each carrying its own ``domain_id``. The
    in-memory store keeps its rows in instance dicts, so the same shape
    is expressed by binding the views' dicts to the primary's. Every
    other code path stays real: writes still go through ``add_vectors``
    and its cross-domain guard, reads through ``search`` / ``count`` and
    ``_effective_filter``.
    """
    for view in views:
        view.vectors = primary.vectors
        view.metadata_store = primary.metadata_store
        view.timestamps = primary.timestamps


async def _make_kb(store: Any, **config: Any) -> RAGKnowledgeBase:
    """A KB over a pre-built store, with a deterministic embedder."""
    embedder = await create_embedding_provider(
        build_embedding_config(embedding_provider="echo", embedding_model="test")
    )
    return RAGKnowledgeBase.from_components(
        config or None, vector_store=store, embedding_provider=embedder
    )


def _write_corpus(root: Path, domain: str, body: str) -> Path:
    """One ``overview.md`` per domain — the same stem, deliberately."""
    directory = root / domain
    directory.mkdir()
    (directory / "overview.md").write_text(f"# Overview\n\n{body}\n")
    return directory


async def test_two_domains_over_one_store_both_survive(tmp_path: Path) -> None:
    """Two domain-scoped KBs ingesting the same filename both keep their chunks.

    The reproducer. Each domain has an ``overview.md``, so both derive
    the chunk id ``overview_0`` while the KB has no domain binding, and
    the second ingest lands on the first's row. What the store does
    about that has changed — it now refuses a write that would capture
    another domain's row rather than performing it silently — so the
    symptom today is that the second domain ingests *nothing*, with the
    refusal collected into ``errors`` while ``error`` stays ``None`` and
    the caller is told 0 files. Either way the second domain has no
    chunks; the assertions below are on the end state, not the symptom.
    """
    store_a = await _make_store("bot-a")
    store_b = await _make_store("bot-b")
    _share_physical_rows(store_a, store_b)

    docs_a = _write_corpus(tmp_path, "bot-a", "Alpha content about photosynthesis.")
    docs_b = _write_corpus(tmp_path, "bot-b", "Beta content about the French revolution.")

    kb_a = await _make_kb(store_a)
    kb_b = await _make_kb(store_b)
    service = KnowledgeIngestionService()

    result_a = await service.ingest_from_config(
        kb_a, {"enabled": True, "documents_path": str(docs_a)}
    )
    result_b = await service.ingest_from_config(
        kb_b, {"enabled": True, "documents_path": str(docs_b)}
    )

    assert result_a.errors == []
    assert result_b.errors == []
    assert (result_a.total_files, result_a.total_chunks) == (1, 1)
    assert (result_b.total_files, result_b.total_chunks) == (1, 1)

    # Two physical rows, disjoint id namespaces, each visible only
    # through the scoped view that wrote it — a scoped store answers 0
    # for another domain's filter rather than reaching across.
    assert len(store_a.metadata_store) == 2
    assert await store_a.count() == 1
    assert await store_b.count() == 1

    by_domain = {meta["domain_id"]: meta for meta in store_a.metadata_store.values()}
    assert "photosynthesis" in by_domain["bot-a"]["text"]
    assert "French revolution" in by_domain["bot-b"]["text"]


async def test_chunk_id_carries_the_store_derived_domain(tmp_path: Path) -> None:
    """A KB with no ``domain_id`` of its own adopts the bound store's."""
    store = await _make_store("bot-a")
    kb = await _make_kb(store)

    assert kb._domain_id == "bot-a"

    docs = _write_corpus(tmp_path, "bot-a", "Alpha content.")
    await kb.load_documents_from_directory(docs)

    assert sorted(store.metadata_store) == ["bot-a\x1foverview\x1f0"]
    assert store.metadata_store["bot-a\x1foverview\x1f0"]["domain_id"] == "bot-a"


async def test_explicit_config_domain_beats_the_store(caplog: Any) -> None:
    """An explicitly configured ``domain_id`` wins, and says so.

    Config-wins is deliberate — it is the shape that serves an
    unscoped store whose domains are distinguished only at the chunk
    layer. Against a store scoped to something *else* it is a
    misconfiguration with an invisible consequence: the row is written
    carrying the KB's domain and the store's own read filter then
    requires its own, so the chunk is stored and can never be read
    back. That warrants a WARNING rather than silence.
    """
    store = await _make_store("store-domain")
    with caplog.at_level(logging.WARNING, logger="dataknobs_bots.knowledge.rag"):
        kb = await _make_kb(store, domain_id="config-domain")
        assert kb._domain_id == "config-domain"

    assert any(
        "config-domain" in record.message and "store-domain" in record.message
        for record in caplog.records
    ), caplog.text


async def test_unbound_kb_keeps_the_historical_chunk_id(tmp_path: Path) -> None:
    r"""No binding, unscoped store: the id is byte-identical to before.

    The no-regression guard for every single-domain consumer. The
    historical shape is ``<stem>_<index>`` with an underscore separator;
    a change here silently doubles up every stored chunk on the next
    ingest, because ``stem_0`` and ``stem\x1f0`` are different keys and
    an UPSERT inserts rather than overwrites.
    """
    store = await _make_store()
    kb = await _make_kb(store)

    assert kb._domain_id is None

    docs = _write_corpus(tmp_path, "plain", "Content.")
    await kb.load_documents_from_directory(docs)

    assert sorted(store.metadata_store) == ["overview_0"]


async def test_bound_domain_scopes_reads_on_an_unscoped_store() -> None:
    """The read filter composes the bound domain; an explicit filter still wins."""
    store = await _make_store()
    kb_a = await _make_kb(store, domain_id="bot-a")
    kb_b = await _make_kb(store, domain_id="bot-b")

    await kb_a.load_markdown_text("# Overview\n\nAlpha content.\n", source="overview.md")
    await kb_b.load_markdown_text("# Overview\n\nBeta content.\n", source="overview.md")

    assert len(store.metadata_store) == 2

    # ``min_similarity=-1.0`` because what is under test is the filter,
    # not the ranking: the deterministic test embedder scores unrelated
    # short strings near zero from either side, so a similarity floor
    # would decide the assertion instead of the binding.
    rows_a = await kb_a.query("Alpha content", k=10, min_similarity=-1.0)
    assert [row["metadata"]["domain_id"] for row in rows_a] == ["bot-a"]

    # Admin escape hatch: an explicit filter overrides the binding, the
    # same inversion ``tenant_id`` already has on the read side.
    crossed = await kb_a.query(
        "Beta content", k=10, filter_metadata={"domain_id": "bot-b"}, min_similarity=-1.0
    )
    assert [row["metadata"]["domain_id"] for row in crossed] == ["bot-b"]


async def test_manager_per_call_domain_is_unaffected() -> None:
    """An unbound KB driven by a manager keeps its per-call ``domain_id``.

    The manager threads ``domain_id`` per ingest call, which is the
    shape that already worked. Its store is unscoped by construction —
    it holds many domains — so the new binding resolves to ``None`` and
    this path is untouched.
    """
    store = await _make_store()
    kb = await _make_kb(store)

    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    for domain, body in (("bot-a", "Alpha."), ("bot-b", "Beta.")):
        await backend.create_kb(domain)
        await backend.put_file(domain, "overview.md", f"# Overview\n\n{body}\n".encode())

    manager = KnowledgeIngestionManager(source=backend, destination=kb)
    for domain in ("bot-a", "bot-b"):
        await manager.ingest(domain)

    assert sorted(store.metadata_store) == [
        "bot-a\x1foverview\x1f0",
        "bot-b\x1foverview\x1f0",
    ]
