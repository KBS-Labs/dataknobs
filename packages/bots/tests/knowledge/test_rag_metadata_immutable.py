"""Item 118 scope expansion 1: RAG chunk-metadata system-field protection.

Pre-fix, ``RAGKnowledgeBase._embed_and_store_chunks`` built a
chunk-metadata dict with system-controlled fields (``text``,
``source``, ``chunk_index``, ``document_type``, ``source_path``)
first, then caller-supplied ``metadata`` last.  A caller doing
``kb.load_markdown_text(text, source="real", metadata={"source": "fake"})``
silently corrupted the stored source attribution.  Reachable through
every public ingest entry point.

Post-fix, those system fields are immutable: caller-supplied values
for them are dropped with a warning, and non-conflicting caller keys
flow through unaffected.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase


async def _make_kb() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


@pytest.mark.asyncio
async def test_caller_metadata_cannot_overwrite_system_fields(caplog):
    """Caller-supplied system-field keys are blocked with a warning."""
    kb = await _make_kb()

    with caplog.at_level(logging.WARNING):
        await kb.load_markdown_text(
            "# Real heading\n\nReal body.\n",
            source="real_source.md",
            metadata={
                # Attempted overrides for every system field.
                "text": "TAMPERED",
                "source": "FAKE_SOURCE",
                "chunk_index": 999,
                "document_type": "ATTACK",
                "source_path": "/etc/passwd",
                # Legitimate caller key: must flow through.
                "category": "support",
            },
        )

    # Inspect what was stored.
    results = await kb.query("Real")
    assert results, "Expected at least one chunk stored"
    md = results[0]["metadata"]

    # System fields preserved.
    assert md["source"] == "real_source.md"
    assert md["document_type"] == "markdown"
    # ``text`` and ``chunk_index`` come from the chunker / loop, not
    # caller — must not have been clobbered.
    assert md["text"] != "TAMPERED"
    assert md["chunk_index"] != 999

    # Non-conflicting caller key flows through.
    assert md.get("category") == "support"

    # Warning must be emitted naming at least one of the blocked keys.
    blocked_mentions = [
        record.message
        for record in caplog.records
        if "immutable" in record.message.lower()
    ]
    assert blocked_mentions, "expected an immutable-key warning"
    # At least one of the system keys must appear in the warning(s).
    blocked_text = " ".join(blocked_mentions)
    assert any(
        key in blocked_text
        for key in ("text", "source", "chunk_index", "document_type")
    )


@pytest.mark.asyncio
async def test_caller_metadata_non_system_keys_flow_through(caplog):
    """Caller keys that don't collide with system fields are preserved silently."""
    kb = await _make_kb()

    with caplog.at_level(logging.WARNING):
        await kb.load_markdown_text(
            "# Heading\n\nBody.\n",
            source="src.md",
            metadata={"domain_id": "docs", "category": "guide"},
        )

    results = await kb.query("Heading")
    assert results
    md = results[0]["metadata"]
    assert md.get("domain_id") == "docs"
    assert md.get("category") == "guide"

    # No immutable-key warnings for keys that don't collide.
    assert not any(
        "immutable" in record.message.lower() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_load_markdown_text_with_redundant_source_emits_no_warning(caplog):
    """Direct callers passing ``source`` in metadata see no spurious warning.

    ``KnowledgeBaseConfig.get_metadata`` adds ``source`` (relative path)
    legitimately, and direct callers of ``load_markdown_text`` may also
    pass ``source`` in caller metadata as a hint. The system-field path
    re-derives ``source`` from the explicit ``source_file`` argument
    anyway, so a caller-supplied ``source`` is redundant — not an
    attack — and should NOT trigger an immutable-key warning.

    Pre-fix, only ``ingest_from_backend`` stripped redundant source/
    filename keys; direct callers of ``load_markdown_text(metadata=
    {"source": ...})`` still triggered a warning. This test pins the
    fix at the shared layer.
    """
    kb = await _make_kb()

    with caplog.at_level(logging.WARNING):
        await kb.load_markdown_text(
            "# Heading\n\nBody.\n",
            # ``source_file`` is the full URI (display path).
            source="s3://bucket/docs/src.md",
            # Caller passes the relative-path ``source`` and ``filename``
            # — exactly the shape ``KnowledgeBaseConfig.get_metadata``
            # produces. The values legitimately DIFFER from the
            # explicit ``source_file`` argument because they're
            # different views of the same file.
            metadata={
                "source": "docs/src.md",  # relative path
                "filename": "src.md",
                "domain_id": "docs",
            },
        )

    # No immutable-key warning — these are redundant copies that the
    # ingest path will recompute, not caller-as-attacker overrides.
    assert not any(
        "immutable" in record.message.lower() for record in caplog.records
    ), (
        "Direct caller of load_markdown_text with redundant source/filename "
        "in metadata should not trigger an immutable-key warning."
    )

    results = await kb.query("Heading")
    assert results
    md = results[0]["metadata"]
    # source still resolves to the explicit source_file argument.
    assert md["source"] == "s3://bucket/docs/src.md"


@pytest.mark.asyncio
async def test_chunk_id_does_not_collide_for_snake_case_domains():
    """Snake-case domain IDs sharing a stem produce distinct chunk IDs.

    Pre-fix, the chunk-id prefix used ``_`` to separate ``domain_id``
    and ``stem``, so ``domain_id="my"`` + file ``team_doc.md`` and
    ``domain_id="my_team"`` + file ``doc.md`` both produced
    ``my_team_doc_0`` — a real footgun for snake_case tenants. Post-
    fix uses a non-printable record separator that cannot appear in
    either component.
    """
    kb = await _make_kb()

    # Two tenants whose (domain_id, file) pairs collide under
    # underscore separation.
    await kb.load_markdown_text(
        "# Heading 1\n\nBody A.\n",
        source="team_doc.md",
        metadata={"domain_id": "my"},
    )
    await kb.load_markdown_text(
        "# Heading 2\n\nBody B.\n",
        source="doc.md",
        metadata={"domain_id": "my_team"},
    )

    # If chunk IDs collided, the second ingest would have upserted
    # over the first → only one record would survive.
    domain_my = await kb.vector_store.count(filter={"domain_id": "my"})
    domain_my_team = await kb.vector_store.count(
        filter={"domain_id": "my_team"}
    )
    assert domain_my > 0, "Expected at least one chunk for domain_id=my"
    assert domain_my_team > 0, (
        "Expected at least one chunk for domain_id=my_team — collision "
        "with domain_id=my would have upserted over it."
    )


@pytest.mark.asyncio
async def test_single_domain_chunk_id_uses_underscore_separator():
    """Single-domain consumers' chunk-id format is unchanged on re-ingest.

    Pre-PR (main): ``chunk_id = f"{stem}_{chunk_index}"``.

    The bots CHANGELOG for this PR explicitly states "Single-domain
    consumers see no change." Honoring that contract requires keeping
    the historical ``_`` separator when no ``domain_id`` is threaded
    through. Otherwise existing populated stores re-ingested under
    the new code would silently double up: every old chunk_id
    ``stem_index`` would fail to upsert against the new ``stem\\x1findex``
    and insert as a new row.
    """
    kb = await _make_kb()

    await kb.load_markdown_text(
        "# Heading\n\nBody.\n",
        source="my_doc.md",
        # No domain_id in metadata — the single-domain branch.
        metadata=None,
    )

    stored_ids = list(kb.vector_store.metadata_store.keys())
    assert stored_ids, "expected at least one stored chunk"
    for chunk_id in stored_ids:
        assert "\x1f" not in chunk_id, (
            "single-domain chunk_id must not contain the record-separator "
            f"(historical format used '_'); got {chunk_id!r}"
        )
        assert chunk_id.startswith("my_doc_"), (
            "single-domain chunk_id must keep the historical "
            f"'<stem>_<index>' shape; got {chunk_id!r}"
        )


@pytest.mark.asyncio
async def test_warning_emitted_once_per_offense_not_per_chunk(caplog):
    """One bad metadata blob → one warning, not N (one per chunk).

    Pre-fix, ``enforce_immutable_keys`` was invoked inside the per-chunk
    loop, so a 5-chunk document with caller-supplied ``text=...``
    emitted 5 identical warnings. Hoisting the warning emission above
    the loop produces one warning per offense.
    """
    kb = await _make_kb()

    # A document long enough to produce multiple chunks.
    long_md = "# Section " + "\n\n# Section ".join(
        f"{i}\n\n{'word ' * 40}" for i in range(10)
    )
    with caplog.at_level(logging.WARNING):
        await kb.load_markdown_text(
            long_md,
            source="multi.md",
            metadata={"text": "TAMPERED"},  # one offense
        )

    text_warnings = [
        record.message
        for record in caplog.records
        if "immutable" in record.message.lower() and "'text'" in record.message
    ]
    # Exactly one warning, not one per chunk.
    assert len(text_warnings) == 1, (
        f"expected 1 warning for the single offense, got {len(text_warnings)}: "
        f"{text_warnings}"
    )
