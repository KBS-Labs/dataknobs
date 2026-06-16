# Multi-tenant knowledge bases

`RAGKnowledgeBase` and `KnowledgeIngestionManager` support two
deployment shapes for multi-tenant workloads:

| Shape | Topology | When to use |
|---|---|---|
| **Per-tenant store** | One `RAGKnowledgeBase` (and one vector store) per tenant | Strict isolation; per-tenant capacity / cost accounting; tenants with very different corpora |
| **Shared store, per-tenant chunks** | One shared `RAGKnowledgeBase`; per-tenant identity stamped onto every chunk | Many tenants with similar corpora; consolidated operations; cross-tenant search permitted by admin tooling |

Per-tenant stores need no DK-level support — wire each tenant's
`KnowledgeIngestionManager` to a distinct `RAGKnowledgeBase` instance
in your application's resolver. The rest of this page documents the
shared-store shape.

## Shared store, per-tenant chunks

Two surfaces compose to produce the shared-store shape: a per-call
`tenant_id` carried via `extra_metadata`, or a bound `tenant_id` on
the manager / KB that auto-stamps every write.

### Bound-tenant manager (recommended)

The natural single-tenant-per-manager pattern. Build one
`KnowledgeIngestionManager(tenant_id=…)` per tenant, all pointing at
the same shared `RAGKnowledgeBase`. Every write through that manager
stamps the bound `tenant_id` onto chunk metadata AND folds it into the
chunk-id prefix; every filter-based clear/tombstone scopes by tenant.

```python
from dataknobs_bots.knowledge import (
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.storage import S3KnowledgeBackend

shared_kb = await RAGKnowledgeBase.from_config({
    "vector_store": {"backend": "pgvector", "dimensions": 1536, ...},
    "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
})

def build_manager_for_tenant(tenant_id: str) -> KnowledgeIngestionManager:
    backend = S3KnowledgeBackend.from_config({
        "bucket": f"acme-knowledge-{tenant_id}",
        "prefix": "knowledge/",
    })
    return KnowledgeIngestionManager(
        source=backend,
        destination=shared_kb,
        tenant_id=tenant_id,
    )

# Two tenants ingesting the same domain_id coexist without collision:
mgr_a = build_manager_for_tenant("acme")
mgr_b = build_manager_for_tenant("umbrella")
await mgr_a.ingest(domain_id="prompts")
await mgr_b.ingest(domain_id="prompts")   # tenant A's chunks untouched
```

The `MappingResolver`-driven `build_manager_for_tenant` shape composes
with the generic `dataknobs_common.resolver.ResourceResolver` Protocol
— a downstream router can dispatch by tenant without coupling to a
bespoke dict-of-builders.

### Bound-tenant KB

Bind the tenant on the KB itself when reads should also be scoped to
the tenant by default. Useful for per-tenant query APIs where the
authenticated tenant is a session-level fact.

```python
kb = await RAGKnowledgeBase.from_config({
    "vector_store": {...},
    "embedding": {...},
    "tenant_id": "acme",            # NEW — bound at construction
})

# Reads automatically AND-compose {"tenant_id": "acme"} into the
# vector-store search filter:
results = await kb.query("How do I configure …?")

# Admin tooling can still read across tenants by passing the explicit
# tenant_id filter (explicit-filter-wins on the read path):
all_tenants = await kb.query(
    "How do I configure …?",
    filter_metadata={"tenant_id": "umbrella"},
)
```

The write-side precedence is the inverse — auto-derived bound tenant
wins on collision with caller-supplied `extra_metadata={"tenant_id":
...}` so identity cannot be silently re-tagged. The asymmetry
(write: auto-derived wins; read: explicit wins) reflects different
threat models on the two sides and is documented on
`_resolve_read_filter`.

### Per-call `extra_metadata` (consumer-driven routing)

When a single `RAGKnowledgeBase` and `KnowledgeIngestionManager`
should serve many tenants without binding (e.g., a stateless ingest
worker pool routing by request payload), use the `extra_metadata`
parameter on every direct entry point uniformly:

```python
# K1-K8 all accept `extra_metadata=` and `tenant_id=` keyword-only.
await kb.load_markdown_text(text, source="snippet.md", tenant_id="acme")
await kb.load_from_directory("./docs", tenant_id="umbrella")
await kb.ingest_from_backend(backend, "prompts", tenant_id="initech")

# Manager entry points symmetric:
await manager.ingest(
    domain_id="prompts",
    extra_metadata={"tenant_id": "acme", "cohort": "beta"},
)
```

The `tenant_id=` kwarg is a convenience shortcut for
`extra_metadata={"tenant_id": tenant_id}`; both routes converge on
the same auto-derived-wins composition.

## Reserved vs. Consumer-Extensible Metadata Keys

The KB owns three identity tags at the write boundary:

```
RAGKnowledgeBase._RESERVED_METADATA_KEYS == frozenset({
    "tenant_id",
    "domain_id",
    "_generation",
})
```

These tags are auto-derived from the bound `tenant_id` / per-call
`domain_id` / TOMBSTONE-swap generation token. A caller-supplied
`extra_metadata` entry with any of these keys is **shadowed** by the
auto-derived value — identity is sacred at the write boundary, and a
caller cannot silently re-tag chunks for another tenant or domain
through the consumer-facing metadata channel.

**All other keys flow through unchanged.** Per-document tags (`region`,
`cohort`, `language`, custom tags) belong here; they remain visible
to the metadata-filter surface on every read path and to any
downstream metadata-aware consumer.

```python
await kb.load_markdown_text(
    text,
    source="snippet.md",
    extra_metadata={
        "tenant_id": "evil",      # shadowed by bound tenant
        "domain_id": "hijacked",   # shadowed by per-call domain
        "region": "us-west",       # preserved
        "cohort": "beta",          # preserved
    },
)
```

Subclasses needing a deeper fold (e.g., per-region partitioning that
becomes part of the chunk-id namespace) override
`_CHUNK_ID_PREFIX_KEYS` to add fold positions without forking the
derivation:

```python
class RegionAwareRAG(RAGKnowledgeBase):
    _CHUNK_ID_PREFIX_KEYS = (
        "tenant_id",
        "domain_id",
        "region",          # new fold position
        "_generation",
    )
```

The default `_RESERVED_METADATA_KEYS` deliberately does NOT include
caller-augmented fold keys (`region` above) — a subclass that wants
to reserve them additionally overrides both ClassVars in lockstep.

## Capability advertisement

`RAGKnowledgeBase` and `KnowledgeIngestionManager` declare
`Capability.TENANT_SCOPED_CHUNKS` through the
`CapabilityContract` protocol from `dataknobs_common`. Consumers
fail-fast at config-load time:

```python
from dataknobs_common.capabilities import (
    Capability,
    require_capability,
)

# Before wiring a multi-tenant manager, verify the destination has
# the chunk-layer tenant-scoping code path:
require_capability(shared_kb, Capability.TENANT_SCOPED_CHUNKS)
```

Advertisement is **structural** ("the class HAS the chunk-layer code
path"), not activation-state — an unbound `RAGKnowledgeBase` still
advertises `TENANT_SCOPED_CHUNKS` because the class implements the
code path; whether a specific instance is currently tenant-scoping is
the natural `kb._tenant_id is not None` binding check.

### Tenancy-family three-layer split

The Tenancy family separates three layers that real systems activate
independently:

| Capability | Layer | Activated where |
|---|---|---|
| `TENANT_SCOPED_CHUNKS` | Content / chunk storage | `RAGKnowledgeBase` + `KnowledgeIngestionManager` (this page) |
| `TENANT_SCOPED_STATE` | Backend bookkeeping (ingestion status, snapshots, checksums) | `KnowledgeResourceBackend` (future deliverable) |
| `TENANT_SCOPED_LOCKS` | Cross-replica concurrency control | `DistributedLock` (future deliverable) |

Declaring these as distinct identifiers avoids over-claiming.
A `RAGKnowledgeBase` declaring `TENANT_SCOPED_CHUNKS` authentically
supports per-chunk tenant scoping; the same class declaring
`TENANT_SCOPED_STATE` would over-claim because the underlying
`KnowledgeResourceBackend.set_ingestion_status` is still per-domain.

The in-tree backends (`FileKnowledgeBackend`,
`InMemoryKnowledgeBackend`, `S3KnowledgeBackend`) inherit
`CapabilityMixin` today via the shared
`KnowledgeResourceBackendMixin` so the contract surface is uniformly
present, but declare an empty `SUPPORTED_CAPABILITIES` set —
per-backend widening (e.g. `STREAMING_READS`, `CHANGE_SUBSCRIPTION`,
`KEY_PATTERN_FILTERING`, `TENANT_SCOPED_STATE`) is captured in the
roadmap rather than advertised speculatively.

## Tombstone swap interaction

A bound-tenant `KnowledgeIngestionManager` running an
`IngestSwapMode.TOMBSTONE` re-ingest applies the same scope discipline
to every step of the swap:

1. The tombstone scope (`update_metadata_where(scope, {"_stale": True})`)
   includes the bound `tenant_id`, so only the manager's own tenant's
   chunks are marked stale.
2. The new generation's chunks carry the bound `tenant_id` AND a
   fresh `_generation` token; the chunk-id prefix folds tenant
   BEFORE generation (declared `_CHUNK_ID_PREFIX_KEYS` order:
   `tenant_id`, `domain_id`, `_generation`).
3. The rollback path (`_rollback_swap`) and reconcile path
   (`_reconcile_interrupted_swap`) also tenant-scope their filters,
   so a crash mid-swap of tenant A does not affect tenant B's rows.

The cross-tenant isolation pin is exercised end-to-end in
`tests/knowledge/test_rag_multi_tenant_isolation.py`.
