"""Tests for the pluggable chunking abstraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_xization.chunking import (
    ChunkTransform,
    Chunker,
    CompositeChunker,
    DocumentInfo,
    MarkdownTreeChunker,
    MergeSmallChunks,
    QualityFilterTransform,
    SplitLargeChunks,
    chunker_registry,
    create_chunker,
    register_chunker,
    register_transform,
    split_text,
    transform_registry,
)
from dataknobs_xization.markdown.md_chunker import Chunk, ChunkMetadata
from dataknobs_xization.markdown.md_parser import MarkdownParser


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class PlaintextChunker(Chunker):
    """Simple test chunker that splits on double newlines."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk(
        self,
        content: str,
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return [
            Chunk(
                text=para,
                metadata=ChunkMetadata(
                    chunk_index=i,
                    chunk_size=len(para),
                    content_length=len(para),
                ),
            )
            for i, para in enumerate(paragraphs)
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PlaintextChunker:
        return cls(max_chunk_size=config.get("max_chunk_size", 1000))


SAMPLE_MARKDOWN = """\
# Heading One

First paragraph under heading one.

Second paragraph under heading one.

## Sub-heading

Content under sub-heading.
"""


# ---------------------------------------------------------------------------
# Chunker ABC
# ---------------------------------------------------------------------------

class TestChunkerABC:
    """Tests for the Chunker ABC contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Chunker()  # type: ignore[abstract]

    def test_subclass_must_implement_chunk(self):
        class IncompleteChunker(Chunker):
            pass

        with pytest.raises(TypeError):
            IncompleteChunker()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        chunker = PlaintextChunker()
        chunks = chunker.chunk("para one\n\npara two")
        assert len(chunks) == 2
        assert chunks[0].text == "para one"
        assert chunks[1].text == "para two"


# ---------------------------------------------------------------------------
# DocumentInfo
# ---------------------------------------------------------------------------

class TestDocumentInfo:
    def test_defaults(self):
        info = DocumentInfo()
        assert info.source == ""
        assert info.content_type == "text/markdown"
        assert info.metadata == {}

    def test_custom_values(self):
        info = DocumentInfo(source="rfc791.txt", content_type="text/plain", metadata={"rfc": 791})
        assert info.source == "rfc791.txt"
        assert info.content_type == "text/plain"
        assert info.metadata["rfc"] == 791


# ---------------------------------------------------------------------------
# MarkdownTreeChunker
# ---------------------------------------------------------------------------

class TestMarkdownTreeChunker:
    def test_is_chunker_subclass(self):
        assert issubclass(MarkdownTreeChunker, Chunker)

    def test_chunks_markdown_content(self):
        chunker = MarkdownTreeChunker(max_chunk_size=500)
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        assert len(chunks) > 0
        # All returned objects are Chunk instances
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_default_heading_inclusion_is_metadata(self):
        chunker = MarkdownTreeChunker()
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        # With IN_METADATA, headings should appear in metadata not in text
        for c in chunks:
            assert c.metadata.headings  # should have heading info

    def test_from_config_defaults(self):
        chunker = MarkdownTreeChunker.from_config({})
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        assert len(chunks) > 0

    def test_from_config_custom_max_size(self):
        chunker = MarkdownTreeChunker.from_config({"max_chunk_size": 50})
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        # Smaller max should produce more chunks (body text may split)
        assert len(chunks) > 0

    def test_from_config_quality_filter_dict(self):
        chunker = MarkdownTreeChunker.from_config({
            "quality_filter": {"min_content_chars": 10, "min_words": 2},
        })
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        for c in chunks:
            assert c.metadata.content_length >= 10

    def test_from_config_heading_inclusion_string(self):
        chunker = MarkdownTreeChunker.from_config({
            "heading_inclusion": "both",
        })
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        assert len(chunks) > 0

    def test_from_config_heading_inclusion_invalid(self):
        """Invalid heading_inclusion value produces a clear error."""
        with pytest.raises(ValueError, match="Invalid 'heading_inclusion'.*include_metadata"):
            MarkdownTreeChunker.from_config({"heading_inclusion": "include_metadata"})

    def test_document_info_passed_through(self):
        chunker = MarkdownTreeChunker()
        info = DocumentInfo(source="test.md", metadata={"version": "1.0"})
        chunks = chunker.chunk(SAMPLE_MARKDOWN, document_info=info)
        # Should still produce chunks — document_info doesn't affect markdown chunker
        assert len(chunks) > 0

    def test_same_output_as_direct_chunk_markdown_tree(self):
        """MarkdownTreeChunker must produce identical output to the
        legacy inline chunk_markdown_tree() call."""
        from dataknobs_xization.markdown import (
            HeadingInclusion,
            chunk_markdown_tree,
            parse_markdown,
        )

        tree = parse_markdown(SAMPLE_MARKDOWN)
        direct = chunk_markdown_tree(
            tree,
            max_chunk_size=500,
            heading_inclusion=HeadingInclusion.IN_METADATA,
            combine_under_heading=True,
            generate_embeddings=True,
        )

        via_chunker = MarkdownTreeChunker(
            max_chunk_size=500,
            combine_under_heading=True,
            generate_embeddings=True,
        ).chunk(SAMPLE_MARKDOWN)

        assert len(direct) == len(via_chunker)
        for d, v in zip(direct, via_chunker):
            assert d.text == v.text
            assert d.metadata.headings == v.metadata.headings
            assert d.metadata.embedding_text == v.metadata.embedding_text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestChunkerRegistry:
    def test_default_markdown_tree_registered(self):
        assert chunker_registry.is_registered("markdown_tree")

    def test_case_insensitive_lookup(self):
        assert chunker_registry.is_registered("MARKDOWN_TREE")

    def test_create_default(self):
        chunker = create_chunker()
        assert isinstance(chunker, MarkdownTreeChunker)

    def test_create_with_explicit_key(self):
        chunker = create_chunker({"chunker": "markdown_tree", "max_chunk_size": 300})
        assert isinstance(chunker, MarkdownTreeChunker)

    def test_create_without_chunker_key_defaults(self):
        chunker = create_chunker({"max_chunk_size": 800})
        assert isinstance(chunker, MarkdownTreeChunker)

    def test_register_and_create_custom_chunker(self):
        register_chunker("plaintext_test", PlaintextChunker, override=True)
        try:
            chunker = create_chunker({"chunker": "plaintext_test"})
            assert isinstance(chunker, PlaintextChunker)
            chunks = chunker.chunk("A\n\nB\n\nC")
            assert len(chunks) == 3
        finally:
            chunker_registry.unregister("plaintext_test")

    def test_dotted_import_path(self):
        """Dotted import path resolves and registers the class."""
        path = f"{PlaintextChunker.__module__}.PlaintextChunker"
        try:
            chunker = create_chunker({"chunker": path})
            assert isinstance(chunker, PlaintextChunker)
            # Should now be registered
            assert chunker_registry.is_registered(path)
        finally:
            if chunker_registry.is_registered(path):
                chunker_registry.unregister(path)

    def test_dotted_import_invalid_path(self):
        with pytest.raises(Exception):
            create_chunker({"chunker": "no_such_module.NoSuchClass"})

    def test_dotted_import_not_chunker_subclass(self):
        # int is not a Chunker subclass
        with pytest.raises(Exception):
            create_chunker({"chunker": "builtins.int"})

    def test_dotted_import_race_safe(self):
        """Concurrent dotted-import registration should not raise."""
        path = f"{PlaintextChunker.__module__}.PlaintextChunker"
        try:
            # Simulate the race: register first, then create_chunker
            # should succeed even though the key is already registered
            # (override=True prevents OperationError).
            chunker_registry.register(path, PlaintextChunker, override=True)
            chunker = create_chunker({"chunker": path})
            assert isinstance(chunker, PlaintextChunker)
        finally:
            if chunker_registry.is_registered(path):
                chunker_registry.unregister(path)


# ---------------------------------------------------------------------------
# End-to-end: custom chunker produces expected chunks
# ---------------------------------------------------------------------------

class TestCustomChunkerEndToEnd:
    def test_plaintext_chunker_via_registry(self):
        register_chunker("plaintext_e2e", PlaintextChunker, override=True)
        try:
            chunker = create_chunker({
                "chunker": "plaintext_e2e",
                "max_chunk_size": 500,
            })
            chunks = chunker.chunk(
                "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
                DocumentInfo(source="test.txt", content_type="text/plain"),
            )
            assert len(chunks) == 3
            assert chunks[0].text == "First paragraph."
            assert chunks[2].text == "Third paragraph."
        finally:
            chunker_registry.unregister("plaintext_e2e")


# ---------------------------------------------------------------------------
# DirectoryProcessor with custom chunker
# ---------------------------------------------------------------------------

class TestDirectoryProcessorCustomChunker:
    """Test that DirectoryProcessor dispatches to a custom chunker."""

    def test_custom_chunker_in_default_chunking(self, tmp_path: Path):
        """DirectoryProcessor with a custom chunker key in default_chunking
        dispatches to the registered chunker for markdown files."""
        from dataknobs_xization.ingestion import (
            DirectoryProcessor,
            KnowledgeBaseConfig,
        )

        register_chunker("dp_test_plain", PlaintextChunker, override=True)
        try:
            # Write a markdown file
            md_file = tmp_path / "doc.md"
            md_file.write_text("Para one.\n\nPara two.\n\nPara three.")

            config = KnowledgeBaseConfig(
                name="test",
                default_chunking={"chunker": "dp_test_plain"},
            )
            processor = DirectoryProcessor(config, tmp_path)

            docs = list(processor.process())
            assert len(docs) == 1
            doc = docs[0]
            assert doc.document_type == "markdown"
            # PlaintextChunker splits on double newlines → 3 chunks
            assert doc.chunk_count == 3
            assert doc.chunks[0]["text"] == "Para one."
            assert doc.chunks[2]["text"] == "Para three."
        finally:
            chunker_registry.unregister("dp_test_plain")

    def test_default_chunker_reused_across_files(self, tmp_path: Path):
        """When all files share the same config, the default chunker
        instance is reused (not re-created per file)."""
        from dataknobs_xization.ingestion import (
            DirectoryProcessor,
            KnowledgeBaseConfig,
        )

        (tmp_path / "a.md").write_text("# A\nContent A.")
        (tmp_path / "b.md").write_text("# B\nContent B.")

        config = KnowledgeBaseConfig(
            name="test",
            default_chunking={"max_chunk_size": 500},
        )
        processor = DirectoryProcessor(config, tmp_path)

        # The processor should have built a default chunker at init
        assert isinstance(processor._default_chunker, MarkdownTreeChunker)

        docs = list(processor.process())
        assert len(docs) == 2
        # Both should produce chunks (verifies the reuse path works)
        for doc in docs:
            assert doc.chunk_count > 0

    def test_quality_filter_forwarded_to_custom_chunker(self, tmp_path: Path):
        """default_quality_filter is forwarded in the config dict to
        from_config(). Custom chunkers should silently ignore unknown
        keys."""
        from dataknobs_xization.ingestion import (
            DirectoryProcessor,
            KnowledgeBaseConfig,
        )

        register_chunker("dp_test_qf", PlaintextChunker, override=True)
        try:
            md_file = tmp_path / "doc.md"
            md_file.write_text("Para one.\n\nPara two.")

            config = KnowledgeBaseConfig(
                name="test",
                default_chunking={"chunker": "dp_test_qf"},
                default_quality_filter={"min_content_chars": 5},
            )
            processor = DirectoryProcessor(config, tmp_path)

            # Should not raise even though PlaintextChunker doesn't know
            # about quality_filter — from_config ignores unknown keys.
            docs = list(processor.process())
            assert len(docs) == 1
            assert docs[0].chunk_count == 2
        finally:
            chunker_registry.unregister("dp_test_qf")

    def test_chunk_positions_forwarded_in_metadata(self, tmp_path: Path):
        """DirectoryProcessor includes char_start/char_end in chunk metadata."""
        from dataknobs_xization.ingestion import (
            DirectoryProcessor,
            KnowledgeBaseConfig,
        )

        md_file = tmp_path / "doc.md"
        md_file.write_text("# Title\n\nSome content here.")

        config = KnowledgeBaseConfig(name="test", default_chunking={"max_chunk_size": 500})
        processor = DirectoryProcessor(config, tmp_path)

        docs = list(processor.process())
        assert len(docs) == 1
        for chunk_dict in docs[0].chunks:
            assert "char_start" in chunk_dict["metadata"]
            assert "char_end" in chunk_dict["metadata"]


# ---------------------------------------------------------------------------
# Parser character positions
# ---------------------------------------------------------------------------

class TestParserCharPositions:
    """Tests for character offset tracking in MarkdownParser."""

    def test_single_body_line(self):
        source = "Hello world."
        parser = MarkdownParser()
        tree = parser.parse(source)
        nodes = tree.collect_terminal_nodes()
        assert len(nodes) == 1
        node = nodes[0].data
        assert node.char_start == 0
        assert node.char_end == len(source)
        assert source[node.char_start:node.char_end] == source

    def test_heading_positions(self):
        source = "# Heading\n\nBody text."
        parser = MarkdownParser()
        tree = parser.parse(source)
        # Heading is at line 0: "# Heading" (9 chars)
        heading = tree.children[0].data
        assert heading.node_type == "heading"
        assert heading.char_start == 0
        assert heading.char_end == 9
        assert source[heading.char_start:heading.char_end] == "# Heading"

    def test_multiline_positions(self):
        source = "Line one.\nLine two.\nLine three."
        parser = MarkdownParser()
        tree = parser.parse(source)
        nodes = tree.collect_terminal_nodes()
        # Three body lines
        assert len(nodes) == 3
        for node in nodes:
            span = source[node.data.char_start:node.data.char_end]
            assert node.data.text in span

    def test_code_block_spans_full_block(self):
        source = "```python\nprint('hi')\n```"
        parser = MarkdownParser()
        tree = parser.parse(source)
        nodes = tree.collect_terminal_nodes()
        assert len(nodes) == 1
        node = nodes[0].data
        assert node.node_type == "code"
        # char span covers from ``` to ```
        assert node.char_start == 0
        assert node.char_end == len(source)

    def test_positions_after_empty_lines(self):
        source = "First.\n\nSecond."
        parser = MarkdownParser()
        tree = parser.parse(source)
        nodes = tree.collect_terminal_nodes()
        assert len(nodes) == 2
        first, second = nodes[0].data, nodes[1].data
        assert first.char_start == 0
        assert first.char_end == 6  # "First."
        assert second.char_start == 8  # after "First.\n\n"
        assert second.char_end == 15  # "Second."
        assert source[second.char_start:second.char_end] == "Second."

    def test_crlf_line_endings(self):
        """Character positions must be correct for \\r\\n line endings."""
        source = "# Title\r\n\r\nBody text."
        parser = MarkdownParser()
        tree = parser.parse(source)
        # Heading
        heading = tree.children[0].data
        assert heading.node_type == "heading"
        assert source[heading.char_start:heading.char_end] == "# Title"
        # Body
        body_nodes = tree.collect_terminal_nodes(
            accept_node_fn=lambda n: n.data.node_type == "body"
        )
        assert len(body_nodes) == 1
        body = body_nodes[0].data
        assert source[body.char_start:body.char_end] == "Body text."

    def test_mixed_line_endings(self):
        """Handles mixed \\n and \\r\\n in same document."""
        source = "Line one.\r\nLine two.\nLine three."
        parser = MarkdownParser()
        tree = parser.parse(source)
        nodes = tree.collect_terminal_nodes()
        assert len(nodes) == 3
        for node in nodes:
            span = source[node.data.char_start:node.data.char_end]
            assert node.data.text == span


# ---------------------------------------------------------------------------
# split_text utility
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_short_text_returns_single(self):
        result = split_text("Hello world.", max_size=100)
        assert len(result) == 1
        text, start, end = result[0]
        assert text == "Hello world."
        assert start == 0
        assert end == 12

    def test_split_returns_positions(self):
        long_text = "First sentence. Second sentence. Third sentence."
        result = split_text(long_text, max_size=20)
        assert len(result) > 1
        for text, start, end in result:
            assert start < end
            assert text  # Non-empty

    def test_positions_cover_full_text(self):
        long_text = "A" * 50 + " " + "B" * 50
        result = split_text(long_text, max_size=60)
        assert len(result) == 2
        # First chunk starts at 0
        assert result[0][1] == 0
        # Last chunk ends at text length
        assert result[-1][2] == len(long_text)


# ---------------------------------------------------------------------------
# Chunk positions through chunker pipeline
# ---------------------------------------------------------------------------

class TestChunkPositions:
    def test_single_section_positions(self):
        source = "# Title\n\nBody text here."
        chunker = MarkdownTreeChunker(max_chunk_size=500)
        chunks = chunker.chunk(source)
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata.char_start >= 0
            assert c.metadata.char_end > c.metadata.char_start

    def test_multiple_sections_positions(self):
        source = "# A\n\nFirst.\n\n# B\n\nSecond."
        chunker = MarkdownTreeChunker(max_chunk_size=500)
        chunks = chunker.chunk(source)
        assert len(chunks) == 2
        # First chunk should start before second
        assert chunks[0].metadata.char_start < chunks[1].metadata.char_start

    def test_positions_in_to_dict(self):
        source = "# Heading\n\nContent."
        chunker = MarkdownTreeChunker(max_chunk_size=500)
        chunks = chunker.chunk(source)
        d = chunks[0].metadata.to_dict()
        assert "char_start" in d
        assert "char_end" in d

    def test_atomic_construct_positions(self):
        source = "# Code\n\n```python\nx = 1\n```"
        chunker = MarkdownTreeChunker(
            max_chunk_size=500,
            combine_under_heading=True,
        )
        chunks = chunker.chunk(source)
        code_chunks = [c for c in chunks if c.metadata.custom.get("node_type") == "code"]
        assert len(code_chunks) == 1
        assert code_chunks[0].metadata.char_start > 0
        assert code_chunks[0].metadata.char_end > code_chunks[0].metadata.char_start


# ---------------------------------------------------------------------------
# ChunkTransform ABC
# ---------------------------------------------------------------------------

class TestChunkTransformABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ChunkTransform()  # type: ignore[abstract]

    def test_subclass_must_implement_transform(self):
        class Incomplete(ChunkTransform):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class NoOpTransform(ChunkTransform):
            def transform(self, chunks, document_info=None):
                return chunks

        t = NoOpTransform()
        chunks = [Chunk(text="test", metadata=ChunkMetadata())]
        assert t.transform(chunks) == chunks


# ---------------------------------------------------------------------------
# CompositeChunker
# ---------------------------------------------------------------------------

class TestCompositeChunker:
    def test_applies_transforms_in_order(self):
        """Transforms are applied sequentially."""
        class AppendSuffix(ChunkTransform):
            def __init__(self, suffix: str):
                self.suffix = suffix

            def transform(self, chunks, document_info=None):
                return [
                    Chunk(
                        text=c.text + self.suffix,
                        metadata=c.metadata,
                    )
                    for c in chunks
                ]

        inner = MarkdownTreeChunker(max_chunk_size=500)
        composite = CompositeChunker(
            inner=inner,
            transforms=[AppendSuffix("_A"), AppendSuffix("_B")],
        )
        chunks = composite.chunk("# Title\n\nContent.")
        assert len(chunks) > 0
        assert chunks[0].text.endswith("_A_B")

    def test_reindexes_after_transforms(self):
        """chunk_index is re-numbered 0..N after transforms."""
        class DropFirst(ChunkTransform):
            def transform(self, chunks, document_info=None):
                return chunks[1:] if len(chunks) > 1 else chunks

        inner = MarkdownTreeChunker(max_chunk_size=500)
        composite = CompositeChunker(inner=inner, transforms=[DropFirst()])
        chunks = composite.chunk("# A\n\nFirst.\n\n# B\n\nSecond.")
        # After dropping first, remaining chunk(s) start at index 0
        if chunks:
            assert chunks[0].metadata.chunk_index == 0

    def test_no_transforms_passthrough(self):
        """CompositeChunker with empty transforms is equivalent to inner."""
        inner = MarkdownTreeChunker(max_chunk_size=500)
        composite = CompositeChunker(inner=inner, transforms=[])
        source = "# Title\n\nBody."
        direct = inner.chunk(source)
        via_composite = composite.chunk(source)
        assert len(direct) == len(via_composite)
        for d, c in zip(direct, via_composite):
            assert d.text == c.text


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------

class TestMergeSmallChunks:
    def _make_chunk(self, text: str, headings: list[str] | None = None,
                    char_start: int = 0, char_end: int = 0) -> Chunk:
        return Chunk(
            text=text,
            metadata=ChunkMetadata(
                headings=headings or [],
                chunk_size=len(text),
                content_length=len(text),
                char_start=char_start,
                char_end=char_end,
            ),
        )

    def test_merges_adjacent_small_chunks(self):
        chunks = [
            self._make_chunk("Short.", char_start=0, char_end=6),
            self._make_chunk("Also short.", char_start=7, char_end=18),
        ]
        result = MergeSmallChunks(min_size=200).transform(chunks)
        assert len(result) == 1
        assert "Short." in result[0].text
        assert "Also short." in result[0].text

    def test_preserves_heading_boundaries(self):
        chunks = [
            self._make_chunk("Short.", headings=["A"]),
            self._make_chunk("Short.", headings=["B"]),
        ]
        result = MergeSmallChunks(min_size=200).transform(chunks)
        # Different headings → not merged
        assert len(result) == 2

    def test_maintains_position_invariants(self):
        chunks = [
            self._make_chunk("First.", char_start=0, char_end=6),
            self._make_chunk("Second.", char_start=8, char_end=15),
        ]
        result = MergeSmallChunks(min_size=200).transform(chunks)
        assert len(result) == 1
        assert result[0].metadata.char_start == 0
        assert result[0].metadata.char_end == 15

    def test_max_size_prevents_cascade(self):
        """max_size prevents merging into unboundedly large chunks."""
        chunks = [
            self._make_chunk("A" * 150, char_start=0, char_end=150),
            self._make_chunk("B" * 30, char_start=151, char_end=181),
            self._make_chunk("C" * 30, char_start=182, char_end=212),
        ]
        # Without max_size, all would merge (150 < 200, 30 < 200)
        result_unbounded = MergeSmallChunks(min_size=200).transform(list(chunks))
        assert len(result_unbounded) == 1

        # With max_size=200, first merge (150+30) succeeds but
        # second (182+30) would exceed 200 → not merged
        result_bounded = MergeSmallChunks(min_size=200, max_size=200).transform(list(chunks))
        assert len(result_bounded) == 2

    def test_from_config(self):
        t = MergeSmallChunks.from_config({"min_size": 300, "max_size": 500})
        assert t.min_size == 300
        assert t.max_size == 500


class TestSplitLargeChunks:
    def _make_chunk(self, text: str, char_start: int = 0, char_end: int = 0) -> Chunk:
        return Chunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_size=len(text),
                content_length=len(text),
                char_start=char_start,
                char_end=char_end or len(text),
            ),
        )

    def test_splits_oversized_chunk(self):
        text = "Word " * 100  # ~500 chars
        chunks = [self._make_chunk(text)]
        result = SplitLargeChunks(max_size=100).transform(chunks)
        assert len(result) > 1
        for c in result:
            assert c.metadata.chunk_size <= 100

    def test_leaves_small_chunks_alone(self):
        chunks = [self._make_chunk("Small.")]
        result = SplitLargeChunks(max_size=100).transform(chunks)
        assert len(result) == 1
        assert result[0].text == "Small."

    def test_from_config(self):
        t = SplitLargeChunks.from_config({"max_size": 500})
        assert t.max_size == 500


class TestQualityFilterTransform:
    def _make_chunk(self, text: str) -> Chunk:
        return Chunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_size=len(text),
                content_length=len(text),
            ),
        )

    def test_filters_small_chunks(self):
        short = self._make_chunk("Hi")
        long_enough = self._make_chunk(
            "This sentence has more than enough words and characters to pass all filters."
        )
        result = QualityFilterTransform({
            "min_content_chars": 10,
            "min_words": 2,
        }).transform([short, long_enough])
        assert len(result) == 1
        assert result[0].text == long_enough.text

    def test_from_config(self):
        t = QualityFilterTransform.from_config({"min_content_chars": 5})
        assert t._filter is not None


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

class TestTransformRegistry:
    def test_builtin_transforms_registered(self):
        assert transform_registry.is_registered("merge_small")
        assert transform_registry.is_registered("split_large")
        assert transform_registry.is_registered("quality_filter")

    def test_register_and_use_custom_transform(self):
        class UpperTransform(ChunkTransform):
            def transform(self, chunks, document_info=None):
                return [
                    Chunk(text=c.text.upper(), metadata=c.metadata)
                    for c in chunks
                ]

            @classmethod
            def from_config(cls, config):
                return cls()

        register_transform("upper_test", UpperTransform, override=True)
        try:
            assert transform_registry.is_registered("upper_test")
        finally:
            transform_registry.unregister("upper_test")


# ---------------------------------------------------------------------------
# Config-driven pipeline
# ---------------------------------------------------------------------------

class TestConfigDrivenPipeline:
    def test_no_transforms_returns_plain_chunker(self):
        chunker = create_chunker({"max_chunk_size": 500})
        assert isinstance(chunker, MarkdownTreeChunker)

    def test_transforms_returns_composite(self):
        chunker = create_chunker({
            "max_chunk_size": 500,
            "transforms": [
                {"merge_small": {"min_size": 100}},
            ],
        })
        assert isinstance(chunker, CompositeChunker)

    def test_empty_transforms_returns_plain_chunker(self):
        chunker = create_chunker({
            "max_chunk_size": 500,
            "transforms": [],
        })
        # Empty transforms list → no wrapper
        assert isinstance(chunker, MarkdownTreeChunker)

    def test_pipeline_produces_chunks(self):
        chunker = create_chunker({
            "max_chunk_size": 500,
            "transforms": [
                {"merge_small": {"min_size": 50}},
            ],
        })
        chunks = chunker.chunk("# Title\n\nContent here.\n\n## Sub\n\nMore content.")
        assert len(chunks) > 0
        # chunk_index should be sequential
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_multiple_transforms_in_pipeline(self):
        chunker = create_chunker({
            "max_chunk_size": 500,
            "transforms": [
                {"split_large": {"max_size": 50}},
                {"merge_small": {"min_size": 10}},
            ],
        })
        assert isinstance(chunker, CompositeChunker)
        chunks = chunker.chunk("# A\n\n" + "Word " * 30)
        assert len(chunks) > 0

    def test_invalid_transform_entry_raises(self):
        with pytest.raises(ValueError, match="exactly one key"):
            create_chunker({
                "transforms": [
                    {"a": {}, "b": {}},  # Two keys — invalid
                ],
            })
