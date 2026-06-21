"""KnowledgeIngestionManager per-tenant ingest-state routing.

A tenant-bound manager (constructed with ``tenant_id``) routes every
backend state operation (``set_ingestion_status`` / ``get_info`` /
``get_checksum`` / ``has_changes_since`` / ``list_changes_since``)
through a :class:`~dataknobs_common.tenancy.BoundTenantContext`, so on a
*shared* backend it isolates its per-tenant ingestion **status** under
the tenant's state-key prefix. An unbound manager passes ``ctx=None``,
which every backend treats as the single-tenant case — byte-identical
state paths to the pre-routing behavior.

Snapshot lineage is shared domain *content* state, NOT per-tenant
isolated: a tenant's change detection resolves against the shared
domain-keyed content lineage and stays minimal (it does not force a
full re-ingest just because the tenant recorded no snapshot of its
own). Only ingest **status** is genuinely per-tenant.

These exercise the manager wrapping real components — an
``InMemoryKnowledgeBackend`` source + a ``RAGKnowledgeBase`` (memory
vector store, echo embedder) destination — so no services and no mocks
are needed. Construction directly (rather than via ``BotTestHarness``)
is legitimate: these test the manager/backend coordination internals,
not a bot conversation flow.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge import (
    InMemoryKnowledgeBackend,
    IngestSwapMode,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.storage import IngestionStatus
from dataknobs_common.capabilities import Capability
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.tenancy import (
    BoundTenantContext,
    PrefixedTenantContext,
    SharedCorpusTenantContext,
)


async def _make_kb() -> RAGKnowledgeBase:
    """A memory-backed RAG knowledge base with a deterministic embedder."""
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


async def _seed_backend(*files: tuple[str, bytes]) -> InMemoryKnowledgeBackend:
    """A shared backend with ``shared_kb`` created and ``files`` written."""
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    await backend.create_kb("shared_kb")
    for path, body in files:
        await backend.put_file("shared_kb", path, body)
    return backend


# --- T1: per-tenant ingest-status isolation on a shared backend ---


async def test_tenant_bound_manager_isolates_ingest_status() -> None:
    """A tenant-bound manager's status write is visible only under its
    own tenant context — not to another tenant, and not to the shared
    domain view."""
    backend = await _seed_backend(
        ("doc.md", b"# Heading\n\nBody content for the shared kb.\n")
    )
    acme_kb = await _make_kb()

    acme = KnowledgeIngestionManager(
        source=backend, destination=acme_kb, tenant_id="acme"
    )
    result = await acme.ingest("shared_kb")
    assert result.success

    acme_ctx = BoundTenantContext("acme", "shared_kb")
    beta_ctx = BoundTenantContext("beta", "shared_kb")

    acme_info = await backend.get_info("shared_kb", ctx=acme_ctx)
    beta_info = await backend.get_info("shared_kb", ctx=beta_ctx)
    domain_info = await backend.get_info("shared_kb")  # ctx=None
    assert acme_info is not None
    assert beta_info is not None
    assert domain_info is not None

    # acme's successful ingest landed READY under acme's prefix...
    assert acme_info.ingestion_status is IngestionStatus.READY
    # ...and did NOT leak to a different tenant's fresh view...
    assert beta_info.ingestion_status is IngestionStatus.PENDING
    # ...nor to the shared domain (ctx=None) view — acme routed every
    # state write through its tenant context, never the domain store.
    assert domain_info.ingestion_status is IngestionStatus.PENDING

    assert acme_info.ingestion_status is not beta_info.ingestion_status

    await acme_kb.close()
    await backend.close()


async def test_two_tenants_track_status_independently() -> None:
    """Two managers bound to distinct tenants, sharing one backend,
    track independent ingestion status."""
    backend = await _seed_backend(("doc.md", b"# H\n\nShared body.\n"))
    acme_kb = await _make_kb()
    beta_kb = await _make_kb()

    acme = KnowledgeIngestionManager(
        source=backend, destination=acme_kb, tenant_id="acme"
    )
    beta = KnowledgeIngestionManager(
        source=backend, destination=beta_kb, tenant_id="beta"
    )

    await acme.ingest("shared_kb")
    await beta.ingest("shared_kb")

    acme_info = await backend.get_info(
        "shared_kb", ctx=BoundTenantContext("acme", "shared_kb")
    )
    beta_info = await backend.get_info(
        "shared_kb", ctx=BoundTenantContext("beta", "shared_kb")
    )
    assert acme_info is not None and beta_info is not None
    # Both succeeded → both READY, each under its own prefix. The point
    # is independence: neither manager wrote the other's state doc.
    assert acme_info.ingestion_status is IngestionStatus.READY
    assert beta_info.ingestion_status is IngestionStatus.READY

    await acme_kb.close()
    await beta_kb.close()
    await backend.close()


# --- T2: change detection resolves against the shared content lineage ---


async def test_tenant_change_detection_is_minimal_via_shared_lineage() -> None:
    """A tenant-bound incremental ingest sees a *minimal* diff against
    the shared domain-keyed content lineage — it does not force a full
    re-ingest just because the tenant recorded no snapshot of its own.

    Snapshot lineage is shared content state (recorded on content
    mutations, which are domain-keyed); only ingest status is per-tenant.
    """
    backend = await _seed_backend(("a.md", b"# A\n\nFirst doc.\n"))
    kb = await _make_kb()
    acme = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="acme"
    )

    # Full ingest, then capture the content-identity version.
    await acme.ingest("shared_kb")
    v1 = await acme.get_current_version("shared_kb")
    assert v1 is not None

    # A content mutation advances the shared (domain-keyed) lineage.
    await backend.put_file("shared_kb", "b.md", b"# B\n\nSecond doc.\n")

    # Incremental ingest from v1. The tenant never recorded a
    # tenant-scoped snapshot for v1 (only the domain-keyed content
    # mutation did) — the backend falls back to the shared content
    # lineage, so the diff is minimal (just b.md), NOT a full re-ingest.
    result = await acme.ingest_changes(
        "shared_kb", since_version=v1, swap_mode=IngestSwapMode.APPEND
    )
    assert result.success
    # Only the one added file was re-embedded. A forced full re-ingest
    # (the bug this guards against) would report 2 files processed.
    assert result.files_processed == 1
    assert result.files_deleted == 0

    await kb.close()
    await backend.close()


async def test_fresh_tenant_get_info_is_default_view_through_manager() -> None:
    """A tenant that has never ingested observes a fresh DEFAULT view
    (PENDING, no in-flight generation token) — not the shared domain
    view, nor another tenant's in-flight state."""
    backend = await _seed_backend(("a.md", b"# A\n\nDoc.\n"))
    kb = await _make_kb()

    # An existing tenant ingests, leaving READY under its own prefix.
    acme = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="acme"
    )
    await acme.ingest("shared_kb")

    # A brand-new tenant's manager reads version → get_info under the
    # newcomer context returns a fresh default, not acme's / the
    # domain's state.
    newcomer = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="newcomer"
    )
    info = await backend.get_info(
        "shared_kb", ctx=BoundTenantContext("newcomer", "shared_kb")
    )
    assert info is not None
    assert info.ingestion_status is IngestionStatus.PENDING
    assert info.generation is None

    # The manager's own version read for the newcomer still resolves the
    # shared content identity (content is shared by domain_id).
    version = await newcomer.get_current_version("shared_kb")
    assert version is not None

    await kb.close()
    await backend.close()


# --- T3: unbound manager byte-identity ---


async def test_unbound_manager_resolves_none_and_ingests_unchanged() -> None:
    """An unbound manager resolves ``None`` for every domain and its
    state calls are unchanged (ctx=None == single-tenant)."""
    backend = await _seed_backend(("doc.md", b"# H\n\nBody.\n"))
    kb = await _make_kb()
    mgr = KnowledgeIngestionManager(source=backend, destination=kb)

    assert mgr._resolve_context("shared_kb") is None

    result = await mgr.ingest("shared_kb")
    assert result.success

    # Status is observable on the shared (domain-keyed) store, and the
    # tenant-prefixed view is irrelevant for an unbound manager.
    info = await backend.get_info("shared_kb")
    assert info is not None
    assert info.ingestion_status is IngestionStatus.READY

    version = await mgr.get_current_version("shared_kb")
    assert version is not None

    await kb.close()
    await backend.close()


# --- T4: _resolve_context behavior ---


async def test_resolve_context_bound_vs_unbound() -> None:
    backend = await _seed_backend()
    kb = await _make_kb()

    bound = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="acme"
    )
    assert bound._resolve_context("kb") == BoundTenantContext("acme", "kb")
    # A different domain yields a context over that domain.
    assert bound._resolve_context("other") == BoundTenantContext(
        "acme", "other"
    )

    unbound = KnowledgeIngestionManager(source=backend, destination=kb)
    assert unbound._resolve_context("kb") is None

    await kb.close()
    await backend.close()


# --- T4b: tenant_context_config shape selection ---
#
# The default (no-config) byte-identity guard — bound resolves
# ``BoundTenantContext`` and unbound resolves ``None`` — is already pinned by
# ``test_resolve_context_bound_vs_unbound`` above; it is not duplicated here.


async def test_context_config_selects_shared_corpus() -> None:
    """A ``shared_corpus`` config keeps per-tenant state but locks/matches
    on the corpus, so two domain views of one corpus are the same scope."""
    backend = await _seed_backend()
    kb = await _make_kb()

    mgr = KnowledgeIngestionManager(
        source=backend,
        destination=kb,
        tenant_id="acme",
        tenant_context_config={
            "kind": "shared_corpus",
            "shared_corpus_id": "corpus-1",
        },
    )
    ctx = mgr._resolve_context("kb_a")
    assert isinstance(ctx, SharedCorpusTenantContext)
    assert ctx.tenant_id == "acme"
    assert ctx.shared_corpus_id == "corpus-1"
    assert ctx.domain_id == "kb_a"
    # The meaningful delta: the lock key is on the corpus, not the domain.
    assert ctx.lock_key("ingest") == "ingest:acme:corpus-1"
    # Two domain views of the same corpus are the same scope — this is what
    # distinguishes shared_corpus from bound (which would be False here).
    assert ctx.matches(mgr._resolve_context("kb_b")) is True

    await kb.close()
    await backend.close()


async def test_context_config_selects_prefixed_changes_state_prefix() -> None:
    """A ``prefixed`` config redirects where the backend writes state —
    the seam changes backend behavior, not just the returned type."""
    backend = await _seed_backend()
    kb = await _make_kb()

    prefixed = KnowledgeIngestionManager(
        source=backend,
        destination=kb,
        tenant_id="acme",
        tenant_context_config={
            "kind": "prefixed",
            "prefix_pattern": "t-{tenant_id}/{domain_id}/",
        },
    )
    ctx = prefixed._resolve_context("kb")
    assert isinstance(ctx, PrefixedTenantContext)
    assert ctx.state_key_prefix() == "t-acme/kb/"

    # Contrast: a bound manager's state prefix is the default convention.
    bound = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="acme"
    )
    bound_ctx = bound._resolve_context("kb")
    assert bound_ctx is not None
    assert bound_ctx.state_key_prefix() == "tenants/acme/_state/"

    await kb.close()
    await backend.close()


async def test_context_config_identity_is_authoritative() -> None:
    """The manager's bound tenant and per-call domain override any identity
    a config tries to smuggle in."""
    backend = await _seed_backend()
    kb = await _make_kb()

    mgr = KnowledgeIngestionManager(
        source=backend,
        destination=kb,
        tenant_id="acme",
        tenant_context_config={
            "kind": "bound",
            "tenant_id": "intruder",
            "domain_id": "other",
        },
    )
    assert mgr._resolve_context("real_kb") == BoundTenantContext(
        "acme", "real_kb"
    )

    await kb.close()
    await backend.close()


async def test_unbound_manager_with_non_single_config_raises() -> None:
    """A tenant-requiring config on an unbound manager fails fast at
    construction."""
    backend = await _seed_backend()
    kb = await _make_kb()

    with pytest.raises(ConfigurationError):
        KnowledgeIngestionManager(
            source=backend,
            destination=kb,
            tenant_context_config={
                "kind": "shared_corpus",
                "shared_corpus_id": "c",
            },
        )

    await kb.close()
    await backend.close()


# --- T5: capability advertisement ---


async def test_capability_advertisement() -> None:
    """Bound or unbound, the manager advertises the two state
    capabilities (they are structural) but never the locking
    capabilities (locking is the orchestrator's contract)."""
    backend = await _seed_backend()
    kb = await _make_kb()

    for tenant_id in (None, "acme"):
        mgr = KnowledgeIngestionManager(
            source=backend, destination=kb, tenant_id=tenant_id
        )
        # New in this layer:
        assert mgr.supports(Capability.TENANT_SCOPED_STATE)
        assert mgr.supports(Capability.SNAPSHOT_ISOLATION)
        # Pre-existing capabilities survive the union:
        assert mgr.supports(Capability.TENANT_SCOPED_CHUNKS)
        assert mgr.supports(Capability.CALLBACK_REGISTRY)
        assert mgr.supports(Capability.INGEST_EVENT_PUBLICATION)
        # Locking is the orchestrator's contract, NOT the manager's:
        assert not mgr.supports(Capability.TENANT_SCOPED_LOCKS)
        assert not mgr.supports(Capability.TRANSACTIONAL_METADATA)

    await kb.close()
    await backend.close()


async def test_event_bus_emission_still_config_dependent() -> None:
    """Adding the static state capabilities must not disturb the
    config-dependent EVENT_BUS_EMISSION advertisement."""
    from dataknobs_common.events import InMemoryEventBus

    backend = await _seed_backend()
    kb = await _make_kb()

    busless = KnowledgeIngestionManager(source=backend, destination=kb)
    assert not busless.supports(Capability.EVENT_BUS_EMISSION)

    with_bus = KnowledgeIngestionManager(
        source=backend, destination=kb, event_bus=InMemoryEventBus()
    )
    assert with_bus.supports(Capability.EVENT_BUS_EMISSION)
    # The state capabilities are present regardless of the bus.
    assert busless.supports(Capability.TENANT_SCOPED_STATE)
    assert with_bus.supports(Capability.TENANT_SCOPED_STATE)

    await kb.close()
    await backend.close()
