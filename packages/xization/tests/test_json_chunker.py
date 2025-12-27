"""Tests for JSON chunker module."""

import json
import tempfile
from pathlib import Path

import pytest

from dataknobs_xization.json.json_chunker import (
    JSONChunk,
    JSONChunkConfig,
    JSONChunker,
)


class TestJSONChunkConfig:
    """Tests for JSONChunkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = JSONChunkConfig()
        assert config.max_chunk_size == 1000
        assert config.text_template is None
        assert config.text_fields is None
        assert config.nested_separator == "."
        assert config.array_handling == "expand"
        assert config.include_field_names is True
        assert config.skip_technical_fields is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = JSONChunkConfig(
            max_chunk_size=500,
            text_template="{{ title }}: {{ description }}",
            text_fields=["title", "description"],
            nested_separator="/",
            array_handling="join",
            include_field_names=False,
            skip_technical_fields=False,
        )
        assert config.max_chunk_size == 500
        assert config.text_template == "{{ title }}: {{ description }}"
        assert config.text_fields == ["title", "description"]
        assert config.nested_separator == "/"
        assert config.array_handling == "join"


class TestJSONChunk:
    """Tests for JSONChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a JSON chunk."""
        chunk = JSONChunk(
            text="Hello World",
            metadata={"title": "Test", "category": "demo"},
            source_path="[0]",
            source_file="data.json",
            embedding_text="[DEMO] Hello World",
            chunk_index=0,
        )
        assert chunk.text == "Hello World"
        assert chunk.metadata["title"] == "Test"
        assert chunk.source_path == "[0]"
        assert chunk.source_file == "data.json"
        assert chunk.chunk_index == 0

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = JSONChunk(
            text="Test content",
            metadata={"key": "value"},
            source_path="[1]",
            chunk_index=5,
        )
        data = chunk.to_dict()
        assert data["text"] == "Test content"
        assert data["metadata"] == {"key": "value"}
        assert data["source_path"] == "[1]"
        assert data["chunk_index"] == 5


class TestJSONChunker:
    """Tests for JSONChunker class."""

    def test_chunk_simple_object(self):
        """Test chunking a simple JSON object."""
        chunker = JSONChunker()
        data = {"title": "Hello", "content": "World"}

        chunks = chunker.chunk(data)

        assert len(chunks) == 1
        assert "Hello" in chunks[0].text
        assert "World" in chunks[0].text
        assert chunks[0].metadata["title"] == "Hello"
        assert chunks[0].metadata["content"] == "World"

    def test_chunk_array_of_objects(self):
        """Test chunking an array of objects."""
        chunker = JSONChunker()
        data = [
            {"title": "First", "description": "Item 1"},
            {"title": "Second", "description": "Item 2"},
            {"title": "Third", "description": "Item 3"},
        ]

        chunks = chunker.chunk(data)

        assert len(chunks) == 3
        assert "First" in chunks[0].text
        assert "Second" in chunks[1].text
        assert "Third" in chunks[2].text
        assert chunks[0].source_path == "[0]"
        assert chunks[1].source_path == "[1]"
        assert chunks[2].source_path == "[2]"

    def test_chunk_with_nested_objects(self):
        """Test chunking with nested objects."""
        chunker = JSONChunker()
        data = {
            "title": "Document",
            "metadata": {
                "author": "John",
                "date": "2024-01-01",
            },
        }

        chunks = chunker.chunk(data)

        assert len(chunks) == 1
        # Nested fields should be flattened in metadata
        assert chunks[0].metadata["metadata.author"] == "John"
        assert chunks[0].metadata["metadata.date"] == "2024-01-01"

    def test_chunk_with_arrays(self):
        """Test chunking with array fields."""
        chunker = JSONChunker()
        data = {
            "title": "Article",
            "tags": ["python", "testing", "json"],
        }

        chunks = chunker.chunk(data)

        assert len(chunks) == 1
        assert "python" in chunks[0].text
        assert "testing" in chunks[0].text
        assert chunks[0].metadata["tags"] == ["python", "testing", "json"]

    def test_auto_generate_text_prioritizes_text_fields(self):
        """Test that auto-generation prioritizes known text fields."""
        chunker = JSONChunker()
        data = {
            "id": "abc123",
            "title": "Important Title",
            "description": "A detailed description",
            "status": "active",
        }

        chunks = chunker.chunk(data)

        text = chunks[0].text
        # Title and description should appear
        assert "Important Title" in text
        assert "detailed description" in text

    def test_skip_technical_fields(self):
        """Test that technical fields are skipped in text generation."""
        config = JSONChunkConfig(skip_technical_fields=True)
        chunker = JSONChunker(config)
        data = {
            "title": "Document",
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2024-01-01T00:00:00Z",
            "content": "Actual content here",
        }

        chunks = chunker.chunk(data)
        text = chunks[0].text

        # UUID should be skipped
        assert "550e8400" not in text
        # Content should appear
        assert "Actual content" in text

    def test_include_field_names(self):
        """Test field name inclusion in generated text."""
        config = JSONChunkConfig(include_field_names=True)
        chunker = JSONChunker(config)
        data = {"category": "testing", "value": 42}

        chunks = chunker.chunk(data)
        text = chunks[0].text

        assert "category:" in text.lower() or "category" in text.lower()

    def test_specific_text_fields(self):
        """Test specifying which fields to use for text."""
        config = JSONChunkConfig(text_fields=["summary"])
        chunker = JSONChunker(config)
        data = {
            "title": "Ignored Title",
            "summary": "This is the summary",
            "details": "These are details",
        }

        chunks = chunker.chunk(data)
        text = chunks[0].text

        assert "This is the summary" in text
        # Other fields might not appear depending on implementation

    def test_max_chunk_size_truncation(self):
        """Test that text is truncated when exceeding max size."""
        config = JSONChunkConfig(max_chunk_size=50)
        chunker = JSONChunker(config)
        data = {
            # Use realistic text that won't be detected as technical (base64)
            "content": "This is a very long content string that should be truncated. " * 5,
        }

        chunks = chunker.chunk(data)

        assert len(chunks[0].text) <= 50
        assert chunks[0].text.endswith("...")

    def test_array_handling_join(self):
        """Test array handling with 'join' mode."""
        config = JSONChunkConfig(array_handling="join")
        chunker = JSONChunker(config)
        data = {"tags": ["a", "b", "c"]}

        chunks = chunker.chunk(data)
        text = chunks[0].text

        # Should be comma-joined
        assert "a" in text and "b" in text and "c" in text

    def test_array_handling_first(self):
        """Test array handling with 'first' mode."""
        config = JSONChunkConfig(array_handling="first")
        chunker = JSONChunker(config)
        data = {"items": ["first_item", "second_item", "third_item"]}

        chunks = chunker.chunk(data)
        text = chunks[0].text

        assert "first_item" in text

    def test_embedding_text_includes_context(self):
        """Test that embedding text includes contextual information."""
        chunker = JSONChunker()
        data = {
            "type": "tutorial",
            "title": "Learn Python",
            "tags": ["python", "programming"],
        }

        chunks = chunker.chunk(data)
        embedding_text = chunks[0].embedding_text

        # Type should be included as context
        assert "TUTORIAL" in embedding_text.upper()
        # Tags should be included
        assert "python" in embedding_text.lower()

    def test_chunk_index_increments(self):
        """Test that chunk indices increment correctly."""
        chunker = JSONChunker()
        data = [{"title": f"Item {i}"} for i in range(5)]

        chunks = chunker.chunk(data)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_source_file_preserved(self):
        """Test that source file is preserved in chunks."""
        chunker = JSONChunker()
        data = {"title": "Test"}

        chunks = chunker.chunk(data, source="/path/to/file.json")

        assert chunks[0].source_file == "/path/to/file.json"

    def test_invalid_input_type(self):
        """Test that invalid input type raises error."""
        chunker = JSONChunker()

        with pytest.raises(ValueError, match="Expected dict or list"):
            chunker.chunk("not a dict or list")

    def test_flatten_nested_structure(self):
        """Test flattening of deeply nested structures."""
        chunker = JSONChunker()
        data = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }

        chunks = chunker.chunk(data)

        assert "level1.level2.level3" in chunks[0].metadata
        assert chunks[0].metadata["level1.level2.level3"] == "deep_value"

    def test_custom_nested_separator(self):
        """Test custom nested key separator."""
        config = JSONChunkConfig(nested_separator="/")
        chunker = JSONChunker(config)
        data = {
            "config": {
                "database": {
                    "host": "localhost"
                }
            }
        }

        chunks = chunker.chunk(data)

        assert "config/database/host" in chunks[0].metadata


class TestJSONChunkerStreaming:
    """Tests for JSONChunker streaming functionality."""

    def test_stream_jsonl_file(self):
        """Test streaming from a JSONL file."""
        chunker = JSONChunker()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write('{"title": "First"}\n')
            f.write('{"title": "Second"}\n')
            f.write('{"title": "Third"}\n')
            temp_path = f.name

        try:
            chunks = list(chunker.stream_chunks(temp_path))
            assert len(chunks) == 3
            assert "First" in chunks[0].text
            assert "Second" in chunks[1].text
            assert "Third" in chunks[2].text
        finally:
            Path(temp_path).unlink()

    def test_stream_jsonl_skips_empty_lines(self):
        """Test that streaming skips empty lines in JSONL."""
        chunker = JSONChunker()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write('{"title": "First"}\n')
            f.write('\n')  # Empty line
            f.write('{"title": "Second"}\n')
            temp_path = f.name

        try:
            chunks = list(chunker.stream_chunks(temp_path))
            assert len(chunks) == 2
        finally:
            Path(temp_path).unlink()

    def test_stream_jsonl_skips_malformed_lines(self):
        """Test that streaming skips malformed JSON lines."""
        chunker = JSONChunker()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write('{"title": "Valid"}\n')
            f.write('not valid json\n')
            f.write('{"title": "Also Valid"}\n')
            temp_path = f.name

        try:
            chunks = list(chunker.stream_chunks(temp_path))
            assert len(chunks) == 2
        finally:
            Path(temp_path).unlink()

    def test_is_jsonl_file_detection(self):
        """Test JSONL file detection by extension."""
        chunker = JSONChunker()

        assert chunker._is_jsonl_file("data.jsonl") is True
        assert chunker._is_jsonl_file("data.jsonl.gz") is True
        assert chunker._is_jsonl_file("data.ndjson") is True
        assert chunker._is_jsonl_file("data.ndjson.gz") is True
        assert chunker._is_jsonl_file("data.json") is False
        assert chunker._is_jsonl_file("data.json.gz") is False


class TestJSONChunkerTemplate:
    """Tests for template-based text generation."""

    def test_jinja_template_rendering(self):
        """Test Jinja2 template rendering."""
        pytest.importorskip("jinja2")

        config = JSONChunkConfig(
            text_template="Title: {{ title }}\nDescription: {{ description }}"
        )
        chunker = JSONChunker(config)
        data = {"title": "Hello", "description": "World"}

        chunks = chunker.chunk(data)

        assert "Title: Hello" in chunks[0].text
        assert "Description: World" in chunks[0].text

    def test_template_with_missing_field(self):
        """Test template with missing field uses empty string."""
        pytest.importorskip("jinja2")

        config = JSONChunkConfig(
            text_template="{{ title }} - {{ missing_field }}"
        )
        chunker = JSONChunker(config)
        data = {"title": "Present"}

        chunks = chunker.chunk(data)
        # Jinja2 renders missing as empty string by default
        assert "Present" in chunks[0].text
