"""Item 118 scope expansion 2: markdown chunker node_type defense-in-depth.

Pre-fix, ``MarkdownChunker._create_chunk`` built
``custom_metadata = {"node_type": node_type}`` then unconditionally
applied ``custom_metadata.update(metadata)``, allowing the optional
``metadata`` parameter to overwrite the chunker-supplied
``node_type``.  The path is practically unreachable today (md_parser
controls the metadata dicts and never sets ``node_type``), but
fixing it as defense-in-depth is zero marginal cost once the
``enforce_immutable_keys`` helper exists.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_xization.markdown.md_chunker import MarkdownChunker


def test_create_chunk_preserves_chunker_node_type(caplog):
    """Caller-supplied ``metadata={"node_type": "WRONG"}`` cannot win.

    The chunker passes ``node_type`` as an explicit argument; that
    value must end up in the chunk's custom metadata regardless of
    any colliding entry in the ``metadata`` dict.
    """
    chunker = MarkdownChunker()
    with caplog.at_level(logging.WARNING):
        chunk = chunker._create_chunk(
            text="example body text",
            headings=["Heading"],
            heading_levels=[1],
            line_number=1,
            metadata={"node_type": "WRONG", "language": "python"},
            node_type="code",
        )

    # Chunker-supplied node_type wins.
    assert chunk.metadata.custom["node_type"] == "code"
    # Non-conflicting metadata key is preserved.
    assert chunk.metadata.custom.get("language") == "python"
    # Warning emitted naming node_type.
    assert any(
        "node_type" in record.message
        and "immutable" in record.message.lower()
        for record in caplog.records
    )


def test_to_dict_custom_cannot_overwrite_structured_fields():
    """``ChunkMetadata.custom`` cannot overwrite structured fields.

    Pre-fix, ``ChunkMetadata.to_dict()`` ended with ``**self.custom``,
    so a custom entry sharing a key with a structured field
    (``headings``, ``chunk_index``, ``chunk_size``, etc.) would
    silently overwrite the structured value in the serialized dict —
    same vulnerability class as the ``_create_chunk`` ``node_type``
    defense, but covering the entire system-field surface.
    """
    from dataknobs_xization.markdown.md_chunker import ChunkMetadata

    md = ChunkMetadata(
        headings=["Real Heading"],
        heading_levels=[1],
        line_number=10,
        char_start=0,
        char_end=100,
        chunk_index=5,
        chunk_size=42,
        content_length=42,
        heading_display="Real Heading",
        # Custom is the attack vector — every structured field is
        # given a colliding entry here. Post-fix, ALL of these must
        # be ignored.
        custom={
            "headings": ["TAMPERED"],
            "heading_levels": [99],
            "line_number": 999,
            "char_start": -1,
            "char_end": -1,
            "chunk_index": 999,
            "chunk_size": -1,
            "content_length": -1,
            "heading_display": "TAMPERED",
            # Non-conflicting custom entry must flow through.
            "node_type": "body",
        },
    )

    d = md.to_dict()

    # Structured fields preserved.
    assert d["headings"] == ["Real Heading"]
    assert d["heading_levels"] == [1]
    assert d["line_number"] == 10
    assert d["chunk_index"] == 5
    assert d["chunk_size"] == 42
    assert d["content_length"] == 42
    assert d["heading_display"] == "Real Heading"
    assert d["char_start"] == 0
    assert d["char_end"] == 100
    # Non-conflicting custom entry flows through.
    assert d["node_type"] == "body"
