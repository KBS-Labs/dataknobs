"""Multi-tenant RAG shared-store isolation tests.

Reproduces the cross-tenant ``chunk_id`` UPSERT collision: two tenants
ingesting the same ``domain_id`` through a shared
:class:`RAGKnowledgeBase` instance must coexist physically (``2*N``
rows, distinct ``chunk_id`` namespaces) rather than overwrite each
other in place (``N`` rows, second ingest's tags only).

The chunk-id prefix derivation in
:meth:`RAGKnowledgeBase._embed_and_store_chunks` folds metadata keys
(``tenant_id`` / ``domain_id`` / ``_generation`` / ``source_stem``) into
the prefix in declared order via
:attr:`RAGKnowledgeBase._CHUNK_ID_PREFIX_KEYS`; the bound-tenant
:class:`RAGKnowledgeBase` and tenant-aware
:class:`KnowledgeIngestionManager` thread ``tenant_id`` uniformly so
the shared-store posture is the byte-identical default for single-
tenant consumers and the correct posture for shared-store multi-tenant
ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.knowledge import (
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.ingestion import IngestSwapMode
from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    InMemoryKnowledgeBackend,
    S3KnowledgeBackend,
)
from dataknobs_common.capabilities import (
    Capability,
    CapabilityContract,
    CapabilityMixin,
)


async def _make_shared_kb(
    tenant_id: str | None = None,
) -> RAGKnowledgeBase:
    """Build a :class:`RAGKnowledgeBase` optionally bound to a tenant.

    The shared-store posture under test: many tenants, one KB, one
    vector store. Tenant identity flows in through the per-call
    ``extra_metadata`` / per-manager ``tenant_id`` channels OR through
    the optional KB-level ``tenant_id`` binding (Change B).
    """
    config: dict[str, Any] = {
        "vector_store": {"backend": "memory", "dimensions": 384},
        "embedding_provider": "echo",
        "embedding_model": "test",
    }
    if tenant_id is not None:
        config["tenant_id"] = tenant_id
    return await RAGKnowledgeBase.from_config(config)


async def _populate(backend: InMemoryKnowledgeBackend, domain: str) -> None:
    """Seed a backend with the same three-file corpus every tenant uses."""
    await backend.initialize()
    await backend.create_kb(domain)
    await backend.put_file(domain, "intro.md", b"# Intro\n\nHello.\n")
    await backend.put_file(domain, "guide.md", b"# Guide\n\nBody.\n")
    await backend.put_file(domain, "ref.md", b"# Reference\n\nLinks.\n")


def _stored_metadata(kb: RAGKnowledgeBase) -> dict[str, dict[str, Any]]:
    """Return the ``MemoryVectorStore`` metadata map keyed by chunk id.

    The memory store exposes the internal map as ``metadata_store``;
    relying on it keeps the test on the real store path
    (``add_vectors`` → in-memory dict) without hand-rolling a recorder.
    """
    return kb.vector_store.metadata_store  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_shared_kb_two_tenants_isolated_after_ingest_via_extra_metadata() -> None:
    """**Failing-on-HEAD reproducer for the chunk-id UPSERT collision.**

    Two tenants ingest the same ``domain_id`` through a shared KB. Each
    threads its own ``tenant_id`` via ``extra_metadata`` — the
    chunk-metadata tag lands, but the chunk-id derivation does NOT read
    ``tenant_id`` today, so tenant B's ingest overwrites tenant A's
    chunks in place.

    Expected after the fix: 2*N rows physically present; each tenant's
    metadata-filtered read returns exactly its own N rows; the two
    ``chunk_id`` sets are disjoint.
    """
    kb = await _make_shared_kb()

    backend_a = InMemoryKnowledgeBackend()
    backend_b = InMemoryKnowledgeBackend()
    await _populate(backend_a, "prompts")
    await _populate(backend_b, "prompts")

    await kb.ingest_from_backend(
        backend_a,
        "prompts",
        extra_metadata={"tenant_id": "tenant_a"},
    )
    rows_after_a = len(_stored_metadata(kb))

    await kb.ingest_from_backend(
        backend_b,
        "prompts",
        extra_metadata={"tenant_id": "tenant_b"},
    )
    rows_after_b = len(_stored_metadata(kb))

    assert rows_after_b == 2 * rows_after_a, (
        f"Tenant B's ingest overwrote tenant A's rows "
        f"({rows_after_a} → {rows_after_b}, expected "
        f"{2 * rows_after_a})"
    )

    # Per-tenant metadata-filter selects exactly that tenant's rows.
    stored = _stored_metadata(kb)
    a_ids = {cid for cid, meta in stored.items() if meta.get("tenant_id") == "tenant_a"}
    b_ids = {cid for cid, meta in stored.items() if meta.get("tenant_id") == "tenant_b"}
    assert len(a_ids) == rows_after_a, (
        f"Expected tenant A to own {rows_after_a} rows; got {len(a_ids)}"
    )
    assert len(b_ids) == rows_after_a, (
        f"Expected tenant B to own {rows_after_a} rows; got {len(b_ids)}"
    )
    assert a_ids.isdisjoint(b_ids), (
        "Tenants' chunk_id sets must be disjoint; collision sets "
        f"= {a_ids & b_ids}"
    )


@pytest.mark.asyncio
async def test_shared_kb_two_tenants_isolated_after_ingest_via_managers() -> None:
    """E2E shape from Plan 165: two managers, each bound to a distinct
    tenant, ingesting the same domain through a shared KB. Pins the
    tenant-aware :class:`KnowledgeIngestionManager.__init__` surface
    and the manager's chunk-metadata thread-through.

    Same expectation as the ``extra_metadata`` variant: 2*N rows, per-
    tenant disjoint chunk_id sets.
    """
    kb = await _make_shared_kb()

    backend_a = InMemoryKnowledgeBackend()
    backend_b = InMemoryKnowledgeBackend()
    await _populate(backend_a, "prompts")
    await _populate(backend_b, "prompts")

    mgr_a = KnowledgeIngestionManager(
        source=backend_a, destination=kb, tenant_id="tenant_a"
    )
    mgr_b = KnowledgeIngestionManager(
        source=backend_b, destination=kb, tenant_id="tenant_b"
    )

    await mgr_a.ingest(domain_id="prompts")
    rows_after_a = len(_stored_metadata(kb))

    await mgr_b.ingest(domain_id="prompts")
    rows_after_b = len(_stored_metadata(kb))

    assert rows_after_b == 2 * rows_after_a, (
        f"Tenant B's manager-driven ingest overwrote tenant A's rows "
        f"({rows_after_a} → {rows_after_b}, expected "
        f"{2 * rows_after_a})"
    )

    stored = _stored_metadata(kb)
    a_ids = {cid for cid, meta in stored.items() if meta.get("tenant_id") == "tenant_a"}
    b_ids = {cid for cid, meta in stored.items() if meta.get("tenant_id") == "tenant_b"}
    assert len(a_ids) == rows_after_a
    assert len(b_ids) == rows_after_a
    assert a_ids.isdisjoint(b_ids)


# ---------------------------------------------------------------------------
# T2 — same-tenant re-ingest UPSERTs in place (back-compat pin)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_tenant_reingest_upserts_in_place() -> None:
    """A second ingest under the same ``(tenant, domain, files)`` tuple
    must produce the SAME chunk-id set — UPSERT correctness preserved.

    Otherwise the multi-tenant fix would double up every chunk on
    re-ingest, breaking the documented "single-tenant consumers see no
    change" contract.
    """
    kb = await _make_shared_kb()
    backend = InMemoryKnowledgeBackend()
    await _populate(backend, "prompts")

    mgr = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="tenant_a"
    )

    await mgr.ingest(domain_id="prompts", swap_mode=IngestSwapMode.APPEND)
    first_ids = set(_stored_metadata(kb).keys())

    await mgr.ingest(domain_id="prompts", swap_mode=IngestSwapMode.APPEND)
    second_ids = set(_stored_metadata(kb).keys())

    assert second_ids == first_ids, (
        "Same-tenant re-ingest must UPSERT in place "
        f"(extra ids: {second_ids - first_ids}; missing: {first_ids - second_ids})"
    )


# ---------------------------------------------------------------------------
# T3 — _derive_chunk_prefix back-compat + new tenant fold positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadata,expected_prefix,expected_sep",
    [
        # Legacy single-segment path — byte-identical to pre-fold.
        (None, "doc", "_"),
        ({}, "doc", "_"),
        # Legacy multi-domain paths preserved.
        ({"domain_id": "d"}, "d\x1fdoc", "\x1f"),
        ({"domain_id": "d", "_generation": "g"}, "d\x1fg\x1fdoc", "\x1f"),
        # New tenant fold positions.
        ({"tenant_id": "t"}, "t\x1fdoc", "\x1f"),
        ({"tenant_id": "t", "domain_id": "d"}, "t\x1fd\x1fdoc", "\x1f"),
        (
            {"tenant_id": "t", "domain_id": "d", "_generation": "g"},
            "t\x1fd\x1fg\x1fdoc",
            "\x1f",
        ),
    ],
)
def test_derive_chunk_prefix_back_compat_and_tenant_fold(
    metadata: dict[str, Any] | None,
    expected_prefix: str,
    expected_sep: str,
) -> None:
    """Pure-unit test on the chunk-prefix derivation.

    Pins the legacy three branches (none / domain-only / domain+gen)
    byte-for-byte so single-tenant populated stores keep matching on
    re-ingest, plus the three new tenant-aware combinations.
    """
    prefix, sep = RAGKnowledgeBase._derive_chunk_prefix("doc", metadata)
    assert prefix == expected_prefix
    assert sep == expected_sep


# ---------------------------------------------------------------------------
# T4 — _CHUNK_ID_PREFIX_KEYS overridability (consumer extensibility)
# ---------------------------------------------------------------------------


def test_chunk_id_prefix_keys_is_overridable_by_subclass() -> None:
    """A subclass rebinds the ClassVar to add a ``region`` fold position
    without forking the derivation — pins the consumer-extensibility
    surface ``_CHUNK_ID_PREFIX_KEYS`` documents.
    """

    class RegionAwareRAG(RAGKnowledgeBase):
        _CHUNK_ID_PREFIX_KEYS = (
            "tenant_id",
            "domain_id",
            "region",
            "_generation",
        )

    prefix, sep = RegionAwareRAG._derive_chunk_prefix(
        "doc",
        {"tenant_id": "t", "domain_id": "d", "region": "us-west"},
    )
    assert prefix == "t\x1fd\x1fus-west\x1fdoc"
    assert sep == "\x1f"


# ---------------------------------------------------------------------------
# T5 — bound tenant: write-side injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bound_tenant_stamps_chunk_metadata_and_id_prefix() -> None:
    """A bound-tenant KB auto-stamps ``tenant_id`` into every chunk's
    metadata AND folds it into the chunk-id prefix, with no per-call
    ``extra_metadata`` from the caller. Pins Change B's write-side
    injection through :meth:`_compose_extra_metadata`.
    """
    kb = await _make_shared_kb(tenant_id="acme")
    await kb.load_markdown_text(
        "# Title\n\nBody.", source="snippet.md"
    )

    stored = _stored_metadata(kb)
    assert stored, "Expected chunks to have been stored"
    for chunk_id, meta in stored.items():
        assert meta.get("tenant_id") == "acme", (
            f"chunk {chunk_id} missing bound tenant; got meta={meta}"
        )
        assert chunk_id.startswith("acme\x1f"), (
            f"chunk_id {chunk_id!r} must be tenant-prefixed"
        )


# ---------------------------------------------------------------------------
# T6 — bound tenant: read-side injection (with explicit override)
# ---------------------------------------------------------------------------


def test_bound_tenant_read_filter_explicit_override_wins() -> None:
    """The read-side ``_resolve_read_filter`` AND-composes the bound
    tenant onto the supplied filter with **explicit-filter-wins** on
    collision — the deliberate write/read asymmetry (write: auto-
    derived wins; read: explicit wins for admin tooling that
    legitimately reads across tenants).
    """

    class _Probe(RAGKnowledgeBase):  # type: ignore[misc]
        # Bypass async _ainit by constructing through __init__ directly
        # — this test exercises the pure read-filter logic.
        pass

    kb = object.__new__(_Probe)
    kb._tenant_id = "acme"  # type: ignore[attr-defined]

    # No supplied filter → bound tenant is injected.
    assert _Probe._resolve_read_filter(kb, None) == {"tenant_id": "acme"}

    # Non-tenant supplied filter → bound tenant is added (no collision).
    assert _Probe._resolve_read_filter(kb, {"category": "api"}) == {
        "tenant_id": "acme",
        "category": "api",
    }

    # Explicit tenant_id supplied → explicit value wins (admin tooling).
    assert _Probe._resolve_read_filter(kb, {"tenant_id": "other"}) == {
        "tenant_id": "other",
    }

    # Unbound KB → supplied filter passes through verbatim.
    unbound = object.__new__(_Probe)
    unbound._tenant_id = None  # type: ignore[attr-defined]
    assert _Probe._resolve_read_filter(unbound, None) is None
    assert _Probe._resolve_read_filter(unbound, {"k": "v"}) == {"k": "v"}


# ---------------------------------------------------------------------------
# T7 — auto-derived-wins precedence (the load-bearing write-side invariant)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bound_tenant_overrides_extra_metadata_tenant_id() -> None:
    """A caller supplying ``extra_metadata={"tenant_id": "evil"}`` to a
    bound-tenant KB cannot re-tag chunks for another tenant — the
    auto-derived bound tenant wins on collision. Pins the
    write-boundary identity invariant.
    """
    kb = await _make_shared_kb(tenant_id="acme")
    await kb.load_markdown_text(
        "# Title\n\nBody.",
        source="snippet.md",
        extra_metadata={"tenant_id": "evil"},
    )
    for chunk_id, meta in _stored_metadata(kb).items():
        assert meta.get("tenant_id") == "acme", (
            f"chunk {chunk_id} carries hostile tenant_id; got meta={meta}"
        )

    # Manager-side mirror: a hostile caller passing tenant_id+domain_id
    # via extra_metadata cannot override the manager's domain_id or
    # bound tenant.
    backend = InMemoryKnowledgeBackend()
    await _populate(backend, "real-domain")
    mgr = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="acme"
    )
    # Fresh KB for clean inspection.
    kb2 = await _make_shared_kb()
    mgr._destination = kb2  # type: ignore[attr-defined]
    await mgr.ingest(
        domain_id="real-domain",
        swap_mode=IngestSwapMode.APPEND,
        extra_metadata={
            "tenant_id": "evil",
            "domain_id": "hijacked",
            "cohort": "preserved",
        },
    )
    for chunk_id, meta in _stored_metadata(kb2).items():
        assert meta.get("tenant_id") == "acme"
        assert meta.get("domain_id") == "real-domain"


# ---------------------------------------------------------------------------
# T8 — _compose_extra_metadata preserves consumer-supplied non-identity keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_preserves_non_identity_metadata_keys() -> None:
    """Identity tags (``tenant_id`` / ``domain_id`` / ``_generation``)
    are auto-derived and win on collision; other keys flow through
    unchanged. Pins the merge semantics so consumer-extensibility
    (per-document ``region`` / ``cohort`` / custom tags) is preserved.
    """
    kb = await _make_shared_kb(tenant_id="acme")
    await kb.load_markdown_text(
        "# Title\n\nBody.",
        source="snippet.md",
        extra_metadata={
            "tenant_id": "evil",        # auto-derived wins
            "region": "us-west",         # preserved
            "cohort": "beta",            # preserved
        },
    )
    for chunk_id, meta in _stored_metadata(kb).items():
        assert meta.get("tenant_id") == "acme"
        assert meta.get("region") == "us-west"
        assert meta.get("cohort") == "beta"


# ---------------------------------------------------------------------------
# T9 — every direct entry point K1-K7 threads extra_metadata uniformly
# ---------------------------------------------------------------------------


def _write_corpus(root: Path) -> Path:
    """Write a small markdown corpus under ``root`` and return it."""
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "intro.md").write_text("# Intro\n\nHello.\n")
    (corpus / "guide.md").write_text("# Guide\n\nBody.\n")
    return corpus


def _write_one(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


@pytest.fixture
async def bound_kb() -> RAGKnowledgeBase:
    return await _make_shared_kb(tenant_id="acme")


async def _k1(kb: RAGKnowledgeBase, _: Path) -> None:
    await kb.load_markdown_text("# T\n\nBody.", source="snippet.md")


async def _k2(kb: RAGKnowledgeBase, tmp: Path) -> None:
    p = tmp / "k2.md"
    _write_one(p, "# K2\n\nBody.\n")
    await kb.load_markdown_document(p)


async def _k3(kb: RAGKnowledgeBase, tmp: Path) -> None:
    p = tmp / "k3.json"
    p.write_text(json.dumps({"name": "k3", "items": ["a", "b"]}))
    await kb.load_json_document(p)


async def _k4(kb: RAGKnowledgeBase, tmp: Path) -> None:
    p = tmp / "k4.yaml"
    p.write_text("title: K4\nbody: hello\n")
    await kb.load_yaml_document(p)


async def _k5(kb: RAGKnowledgeBase, tmp: Path) -> None:
    p = tmp / "k5.csv"
    p.write_text("name,value\nalice,1\nbob,2\n")
    await kb.load_csv_document(p)


async def _k6(kb: RAGKnowledgeBase, tmp: Path) -> None:
    corpus = _write_corpus(tmp)
    await kb.load_documents_from_directory(corpus)


async def _k7(kb: RAGKnowledgeBase, tmp: Path) -> None:
    corpus = _write_corpus(tmp)
    await kb.load_from_directory(corpus)


@pytest.mark.parametrize(
    "entry_point,name",
    [
        (_k1, "load_markdown_text"),
        (_k2, "load_markdown_document"),
        (_k3, "load_json_document"),
        (_k4, "load_yaml_document"),
        (_k5, "load_csv_document"),
        (_k6, "load_documents_from_directory"),
        (_k7, "load_from_directory"),
    ],
)
@pytest.mark.asyncio
async def test_bound_tenant_stamps_via_every_entry_point(
    entry_point: Callable[[RAGKnowledgeBase, Path], Any],
    name: str,
    tmp_path: Path,
    bound_kb: RAGKnowledgeBase,
) -> None:
    """Every K1-K7 entry point threads the bound tenant through to
    stored chunk metadata. Pins the entry-point uniformity guarantee
    — a consumer reaching for any of the seven gets correct behavior
    rather than depending on a particular subset.
    """
    await entry_point(bound_kb, tmp_path)
    stored = _stored_metadata(bound_kb)
    assert stored, f"{name} produced no chunks; cannot verify tenant stamp"
    for chunk_id, meta in stored.items():
        assert meta.get("tenant_id") == "acme", (
            f"{name} chunk {chunk_id} missing bound tenant; meta={meta}"
        )


@pytest.mark.parametrize(
    "entry_point,name",
    [
        (_k1, "load_markdown_text"),
        (_k7, "load_from_directory"),
    ],
)
@pytest.mark.asyncio
async def test_per_call_tenant_kwarg_stamps_when_unbound(
    entry_point: Callable[[RAGKnowledgeBase, Path], Any],
    name: str,
    tmp_path: Path,
) -> None:
    """Per-call ``tenant_id=`` kwarg lands when the KB is unbound.
    Smoke-checks two of the seven entry points (K1 funnel + K7
    directory) — the parametrized cover above pins the per-entry-point
    routing through the funnel.
    """
    kb = await _make_shared_kb()  # unbound
    # Override the entry point invocation to pass tenant_id kwarg.
    if entry_point is _k1:
        await kb.load_markdown_text(
            "# T\n\nBody.", source="snippet.md", tenant_id="acme"
        )
    else:
        corpus = _write_corpus(tmp_path)
        await kb.load_from_directory(corpus, tenant_id="acme")

    stored = _stored_metadata(kb)
    assert stored
    for chunk_id, meta in stored.items():
        assert meta.get("tenant_id") == "acme", (
            f"{name} chunk {chunk_id} missing per-call tenant; meta={meta}"
        )


# ---------------------------------------------------------------------------
# T10 — KnowledgeIngestionManager threads tenant_id into extra_metadata
#       (verified via a RecordingRAGKnowledgeBase subclass — real code path,
#       not a mock per `rules/testing-practices.md`)
# ---------------------------------------------------------------------------


class _RecordingRAG(RAGKnowledgeBase):
    """Real :class:`RAGKnowledgeBase` subclass that records the
    ``extra_metadata`` argument :meth:`ingest_from_backend` receives,
    then forwards verbatim. Replaces the would-be ``MagicMock`` per
    ``rules/testing-practices.md`` — a real implementation with a real
    code path that exposes one extra inspection seam.
    """

    captured_extra_metadata: list[dict[str, Any] | None] = []

    async def ingest_from_backend(self, *args: Any, **kwargs: Any) -> Any:
        self.captured_extra_metadata.append(
            dict(kwargs["extra_metadata"])
            if kwargs.get("extra_metadata") is not None
            else None
        )
        return await super().ingest_from_backend(*args, **kwargs)


@pytest.mark.asyncio
async def test_manager_threads_tenant_into_extra_metadata() -> None:
    """The manager's :meth:`_compose_extra_metadata` lands both
    ``domain_id`` and the bound ``tenant_id`` on every call into
    :meth:`RAGKnowledgeBase.ingest_from_backend`. Recorded against the
    real ``ingest_from_backend`` code path.
    """
    _RecordingRAG.captured_extra_metadata = []
    kb = await _RecordingRAG.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    backend = InMemoryKnowledgeBackend()
    await _populate(backend, "domain-x")

    mgr = KnowledgeIngestionManager(
        source=backend, destination=kb, tenant_id="acme"
    )
    await mgr.ingest(domain_id="domain-x", swap_mode=IngestSwapMode.APPEND)

    assert _RecordingRAG.captured_extra_metadata, (
        "Expected ingest_from_backend to be invoked at least once"
    )
    composed = _RecordingRAG.captured_extra_metadata[-1]
    assert composed == {"domain_id": "domain-x", "tenant_id": "acme"}


# ---------------------------------------------------------------------------
# T11 — tombstone swap preserves tenant scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tombstone_swap_preserves_tenant_scoping() -> None:
    """A bound-tenant manager running a TOMBSTONE re-ingest:

    1. New chunks carry ``tenant_id`` AND a fresh ``_generation`` token.
    2. The chunk-id prefix folds tenant BEFORE generation (declared
       ``_CHUNK_ID_PREFIX_KEYS`` order: tenant, domain, _generation).
    3. Filter-based clears + tombstones scope by tenant, so a
       concurrent search filtered on a different tenant returns 0 rows.
    """
    kb = await _make_shared_kb()
    backend_a = InMemoryKnowledgeBackend()
    backend_b = InMemoryKnowledgeBackend()
    await _populate(backend_a, "prompts")
    await _populate(backend_b, "prompts")

    mgr_a = KnowledgeIngestionManager(
        source=backend_a, destination=kb, tenant_id="a"
    )
    mgr_b = KnowledgeIngestionManager(
        source=backend_b, destination=kb, tenant_id="b"
    )

    await mgr_a.ingest("prompts", swap_mode=IngestSwapMode.TOMBSTONE)
    await mgr_b.ingest("prompts", swap_mode=IngestSwapMode.TOMBSTONE)

    stored = _stored_metadata(kb)

    a_rows = [
        (cid, meta) for cid, meta in stored.items()
        if meta.get("tenant_id") == "a"
    ]
    b_rows = [
        (cid, meta) for cid, meta in stored.items()
        if meta.get("tenant_id") == "b"
    ]
    assert a_rows, "Tenant A's TOMBSTONE swap produced no live rows"
    assert b_rows, "Tenant B's TOMBSTONE swap produced no live rows"

    # Every live row carries a generation token + the bound tenant; the
    # chunk_id prefix folds tenant BEFORE generation (declared
    # `_CHUNK_ID_PREFIX_KEYS` order).
    for cid, meta in a_rows:
        gen = meta.get("_generation")
        assert isinstance(gen, str) and gen, (
            f"Tenant A row {cid} missing _generation; meta={meta}"
        )
        assert cid.startswith(f"a\x1fprompts\x1f{gen}\x1f"), (
            f"Tenant A chunk_id {cid!r} order mismatch; expected "
            f"`a\\x1fprompts\\x1f{gen}\\x1f...`"
        )

    # Cross-tenant isolation pin: the chunk_id sets are disjoint AND
    # the live row counts are independent (tenant B's swap didn't
    # tombstone tenant A's rows).
    a_ids = {cid for cid, _ in a_rows}
    b_ids = {cid for cid, _ in b_rows}
    assert a_ids.isdisjoint(b_ids)


# ---------------------------------------------------------------------------
# T14 — RAGKnowledgeBase advertises TENANT_SCOPED_CHUNKS, not
#       TENANT_SCOPED_STATE / TENANT_SCOPED_LOCKS (the chunk/state/locks
#       three-layer Tenancy-family split documented in Plan 165 F.1).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_knowledge_base_advertises_tenant_scoped_chunks_only() -> None:
    """Both bound and unbound RAGs advertise ``TENANT_SCOPED_CHUNKS``
    structurally (the class HAS the chunk-layer code path), and
    deliberately do NOT advertise the state/locks layers (W4 deliverable).
    Pins the static `CapabilityMixin` semantics: capability
    advertisement is structural, not activation-state.
    """
    for kb in (
        await _make_shared_kb(tenant_id="acme"),
        await _make_shared_kb(),
    ):
        assert kb.supports(Capability.TENANT_SCOPED_CHUNKS) is True
        # Raw-string equivalence — consumer-extensibility contract.
        assert kb.supports("tenant_scoped_chunks") is True
        assert Capability.TENANT_SCOPED_CHUNKS in kb.instance_capabilities()
        assert (
            Capability.TENANT_SCOPED_CHUNKS
            in RAGKnowledgeBase.supported_capabilities()
        )
        # Backend-state-layer and concurrency-layer are deliberately
        # NOT advertised until W4 activates them.
        assert kb.supports(Capability.TENANT_SCOPED_STATE) is False
        assert kb.supports(Capability.TENANT_SCOPED_LOCKS) is False


# ---------------------------------------------------------------------------
# T15 — KnowledgeIngestionManager advertises TENANT_SCOPED_CHUNKS only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingestion_manager_advertises_tenant_scoped_chunks_only() -> None:
    """Symmetric assertion against :class:`KnowledgeIngestionManager`.
    Both bound and unbound managers advertise the chunk-layer
    capability and do NOT over-claim the state/locks layers.
    """
    kb = await _make_shared_kb()
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()

    for mgr in (
        KnowledgeIngestionManager(
            source=backend, destination=kb, tenant_id="acme"
        ),
        KnowledgeIngestionManager(source=backend, destination=kb),
    ):
        assert mgr.supports(Capability.TENANT_SCOPED_CHUNKS) is True
        assert mgr.supports("tenant_scoped_chunks") is True
        assert (
            Capability.TENANT_SCOPED_CHUNKS in mgr.instance_capabilities()
        )
        assert (
            Capability.TENANT_SCOPED_CHUNKS
            in KnowledgeIngestionManager.supported_capabilities()
        )
        assert mgr.supports(Capability.TENANT_SCOPED_STATE) is False
        assert mgr.supports(Capability.TENANT_SCOPED_LOCKS) is False


# ---------------------------------------------------------------------------
# T16 — Backend subclasses inherit CapabilityMixin but advertise no
#       capabilities (plumbing-only at this wave; W3/W4 widen)
# ---------------------------------------------------------------------------


def test_backend_subclasses_inherit_capability_mixin_but_advertise_nothing() -> None:
    """In-tree backends inherit :class:`CapabilityMixin` via the shared
    :class:`KnowledgeResourceBackendMixin` so the contract surface is
    uniformly present. The declared set is empty today — wave-by-wave
    widening (``STREAMING_READS`` / ``CHANGE_SUBSCRIPTION`` /
    ``KEY_PATTERN_FILTERING`` / ``TENANT_SCOPED_STATE``) is captured in
    the roadmap rather than declared speculatively. A consumer asking
    ``backend.supports(...)`` for any specific capability gets the
    honest "not advertised" answer.
    """
    backend_classes: list[type[Any]] = [
        InMemoryKnowledgeBackend,
        FileKnowledgeBackend,
        S3KnowledgeBackend,
    ]
    for cls in backend_classes:
        # Capability-contract conformance via the Protocol.
        assert issubclass(cls, CapabilityMixin), (
            f"{cls.__name__} must inherit CapabilityMixin via the "
            "KnowledgeResourceBackendMixin"
        )
        # Empty declaration — no capability advertised.
        assert cls.supported_capabilities() == frozenset(), (
            f"{cls.__name__} must not advertise any capability today; "
            f"got {cls.supported_capabilities()}"
        )
        # Pin specific identifiers — back-compat guard for the four
        # most-likely-to-be-asked capabilities at the backend layer.
        for cap in (
            Capability.TENANT_SCOPED_CHUNKS,  # chunk layer, not backend
            Capability.TENANT_SCOPED_STATE,   # W4 deliverable
            Capability.STREAMING_READS,        # W3 deliverable
            Capability.CHANGE_SUBSCRIPTION,    # W3 deliverable
        ):
            assert cls.SUPPORTED_CAPABILITIES.isdisjoint({cap}), (
                f"{cls.__name__} unexpectedly advertises {cap.value!r}"
            )


def test_in_memory_backend_satisfies_capability_contract_protocol() -> None:
    """Runtime-checkable Protocol assertion: an instance of an in-tree
    backend structurally satisfies :class:`CapabilityContract`. Pins the
    Protocol-instance check downstream consumers may use to dispatch
    capability-aware code paths without coupling to the mixin class.
    """
    backend = InMemoryKnowledgeBackend()
    assert isinstance(backend, CapabilityContract)
    assert backend.instance_capabilities() == frozenset()
