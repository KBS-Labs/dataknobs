"""Tests for ingestion module."""

import json
import tempfile
from pathlib import Path

import pytest

from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    FilePatternConfig,
    IngestionConfigError,
    KnowledgeBaseConfig,
    ProcessedDocument,
    process_directory,
)


class TestFilePatternConfig:
    """Tests for FilePatternConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FilePatternConfig(pattern="**/*.json")
        assert config.pattern == "**/*.json"
        assert config.enabled is True
        assert config.chunking is None
        assert config.text_template is None
        assert config.text_fields is None
        assert config.metadata_fields is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FilePatternConfig(
            pattern="api/**/*.json",
            enabled=True,
            chunking={"max_chunk_size": 800},
            text_template="{{ title }}: {{ description }}",
            text_fields=["title", "description"],
            metadata_fields=["author", "date"],
        )
        assert config.pattern == "api/**/*.json"
        assert config.chunking == {"max_chunk_size": 800}
        assert config.text_template == "{{ title }}: {{ description }}"
        assert config.text_fields == ["title", "description"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = FilePatternConfig(
            pattern="**/*.md",
            chunking={"max_chunk_size": 500},
        )
        data = config.to_dict()
        assert data["pattern"] == "**/*.md"
        assert data["chunking"] == {"max_chunk_size": 500}
        assert "enabled" not in data  # Default not included

    def test_to_dict_disabled(self):
        """Test that disabled flag is included when False."""
        config = FilePatternConfig(pattern="**/*.tmp", enabled=False)
        data = config.to_dict()
        assert data["enabled"] is False

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "pattern": "docs/**/*.md",
            "chunking": {"chunk_overlap": 100},
            "text_fields": ["content"],
        }
        config = FilePatternConfig.from_dict(data)
        assert config.pattern == "docs/**/*.md"
        assert config.chunking == {"chunk_overlap": 100}
        assert config.text_fields == ["content"]
        assert config.enabled is True  # Default


class TestKnowledgeBaseConfig:
    """Tests for KnowledgeBaseConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = KnowledgeBaseConfig(name="test-kb")
        assert config.name == "test-kb"
        assert config.default_chunking == {"max_chunk_size": 500, "chunk_overlap": 50}
        assert config.default_quality_filter is None
        assert config.patterns == []
        assert config.exclude_patterns == []
        assert config.default_metadata == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = KnowledgeBaseConfig(
            name="product-docs",
            default_chunking={"max_chunk_size": 800},
            exclude_patterns=["**/drafts/**"],
            default_metadata={"version": "1.0"},
        )
        assert config.name == "product-docs"
        assert config.default_chunking == {"max_chunk_size": 800}
        assert config.exclude_patterns == ["**/drafts/**"]
        assert config.default_metadata == {"version": "1.0"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "api-docs",
            "default_chunking": {"max_chunk_size": 600},
            "patterns": [
                {"pattern": "**/*.json", "text_fields": ["title"]},
            ],
            "exclude_patterns": ["**/.git/**"],
        }
        config = KnowledgeBaseConfig.from_dict(data)
        assert config.name == "api-docs"
        assert config.default_chunking == {"max_chunk_size": 600}
        assert len(config.patterns) == 1
        assert config.patterns[0].pattern == "**/*.json"
        assert config.exclude_patterns == ["**/.git/**"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = KnowledgeBaseConfig(
            name="test",
            default_chunking={"max_chunk_size": 500},
            patterns=[FilePatternConfig(pattern="**/*.md")],
        )
        data = config.to_dict()
        assert data["name"] == "test"
        assert data["default_chunking"] == {"max_chunk_size": 500}
        assert len(data["patterns"]) == 1

    def test_load_from_json_file(self):
        """Test loading config from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "knowledge_base.json"
            config_data = {
                "name": "test-kb",
                "default_chunking": {"max_chunk_size": 400},
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            config = KnowledgeBaseConfig.load(tmpdir)
            assert config.name == "test-kb"
            assert config.default_chunking == {"max_chunk_size": 400}

    def test_load_from_yaml_file(self):
        """Test loading config from YAML file."""
        pytest.importorskip("yaml")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "knowledge_base.yaml"
            config_content = """
name: yaml-kb
default_chunking:
  max_chunk_size: 700
patterns:
  - pattern: "**/*.md"
    chunking:
      max_chunk_size: 900
"""
            with open(config_path, "w") as f:
                f.write(config_content)

            config = KnowledgeBaseConfig.load(tmpdir)
            assert config.name == "yaml-kb"
            assert config.default_chunking == {"max_chunk_size": 700}
            assert len(config.patterns) == 1

    def test_load_missing_config_uses_defaults(self):
        """Test that missing config file uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KnowledgeBaseConfig.load(tmpdir)
            assert config.name == Path(tmpdir).name
            assert config.default_chunking == {"max_chunk_size": 500, "chunk_overlap": 50}

    def test_load_invalid_json_raises_error(self):
        """Test that invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "knowledge_base.json"
            with open(config_path, "w") as f:
                f.write("not valid json")

            with pytest.raises(IngestionConfigError, match="Failed to load"):
                KnowledgeBaseConfig.load(tmpdir)

    def test_get_pattern_config_matches(self):
        """Test pattern matching for files."""
        config = KnowledgeBaseConfig(
            name="test",
            patterns=[
                FilePatternConfig(pattern="api/**/*.json", text_fields=["title"]),
                FilePatternConfig(pattern="**/*.md"),
            ],
        )

        # Should match first pattern
        pattern = config.get_pattern_config("api/v1/endpoints.json")
        assert pattern is not None
        assert pattern.text_fields == ["title"]

        # Should match second pattern
        pattern = config.get_pattern_config("docs/guide.md")
        assert pattern is not None
        assert pattern.pattern == "**/*.md"

        # Should not match any pattern
        pattern = config.get_pattern_config("data/file.csv")
        assert pattern is None

    def test_get_pattern_config_disabled_pattern(self):
        """Test that disabled patterns are not matched."""
        config = KnowledgeBaseConfig(
            name="test",
            patterns=[
                FilePatternConfig(pattern="**/*.json", enabled=False),
            ],
        )
        pattern = config.get_pattern_config("data.json")
        assert pattern is None

    def test_is_excluded(self):
        """Test file exclusion checking."""
        config = KnowledgeBaseConfig(
            name="test",
            exclude_patterns=["**/drafts/**", "**/.git/**", ".git/**", "*.tmp"],
        )

        assert config.is_excluded("docs/drafts/readme.md") is True
        assert config.is_excluded("repo/.git/config") is True  # With prefix
        assert config.is_excluded(".git/config") is True  # At root (needs .git/** pattern)
        assert config.is_excluded("backup.tmp") is True
        assert config.is_excluded("docs/guide.md") is False

    def test_get_chunking_config_default(self):
        """Test getting default chunking config."""
        config = KnowledgeBaseConfig(
            name="test",
            default_chunking={"max_chunk_size": 500, "chunk_overlap": 50},
        )
        chunking = config.get_chunking_config("any/file.md")
        assert chunking == {"max_chunk_size": 500, "chunk_overlap": 50}

    def test_get_chunking_config_with_override(self):
        """Test chunking config with pattern override."""
        config = KnowledgeBaseConfig(
            name="test",
            default_chunking={"max_chunk_size": 500, "chunk_overlap": 50},
            patterns=[
                FilePatternConfig(
                    pattern="api/**/*.json",
                    chunking={"max_chunk_size": 800},
                ),
            ],
        )
        chunking = config.get_chunking_config("api/v1/endpoints.json")
        assert chunking["max_chunk_size"] == 800
        assert chunking["chunk_overlap"] == 50  # From default

    def test_get_metadata(self):
        """Test getting metadata for a file."""
        config = KnowledgeBaseConfig(
            name="test",
            default_metadata={"version": "1.0", "author": "test"},
        )
        metadata = config.get_metadata("docs/guide.md")
        assert metadata["version"] == "1.0"
        assert metadata["author"] == "test"
        assert metadata["source"] == "docs/guide.md"
        assert metadata["filename"] == "guide.md"


class TestProcessedDocument:
    """Tests for ProcessedDocument dataclass."""

    def test_basic_creation(self):
        """Test creating a processed document."""
        doc = ProcessedDocument(
            source_file="/path/to/file.md",
            document_type="markdown",
            chunks=[{"text": "Hello", "chunk_index": 0}],
            metadata={"source": "file.md"},
        )
        assert doc.source_file == "/path/to/file.md"
        assert doc.document_type == "markdown"
        assert doc.chunk_count == 1
        assert doc.has_errors is False

    def test_chunk_count(self):
        """Test chunk count property."""
        doc = ProcessedDocument(
            source_file="test.md",
            document_type="markdown",
            chunks=[{"text": f"Chunk {i}"} for i in range(5)],
        )
        assert doc.chunk_count == 5

    def test_has_errors(self):
        """Test error detection property."""
        doc_no_errors = ProcessedDocument(
            source_file="test.md",
            document_type="markdown",
            chunks=[],
        )
        assert doc_no_errors.has_errors is False

        doc_with_errors = ProcessedDocument(
            source_file="test.md",
            document_type="markdown",
            chunks=[],
            errors=["Failed to parse"],
        )
        assert doc_with_errors.has_errors is True


class TestDirectoryProcessor:
    """Tests for DirectoryProcessor class."""

    def test_process_markdown_file(self):
        """Test processing a markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create markdown file
            md_file = Path(tmpdir) / "guide.md"
            md_file.write_text("# Introduction\n\nThis is the guide content.")

            config = KnowledgeBaseConfig(name="test")
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 1
            assert docs[0].document_type == "markdown"
            assert docs[0].chunk_count >= 1
            assert docs[0].has_errors is False

    def test_process_json_file(self):
        """Test processing a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON file
            json_file = Path(tmpdir) / "data.json"
            json_data = [
                {"title": "Item 1", "description": "First item"},
                {"title": "Item 2", "description": "Second item"},
            ]
            with open(json_file, "w") as f:
                json.dump(json_data, f)

            config = KnowledgeBaseConfig(name="test")
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 1
            assert docs[0].document_type == "json"
            assert docs[0].chunk_count == 2  # One per array item
            assert docs[0].has_errors is False

    def test_process_jsonl_file(self):
        """Test processing a JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSONL file
            jsonl_file = Path(tmpdir) / "data.jsonl"
            with open(jsonl_file, "w") as f:
                f.write('{"title": "Line 1"}\n')
                f.write('{"title": "Line 2"}\n')
                f.write('{"title": "Line 3"}\n')

            config = KnowledgeBaseConfig(name="test")
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 1
            assert docs[0].document_type == "jsonl"
            assert docs[0].chunk_count == 3
            assert docs[0].has_errors is False

    def test_process_excludes_files(self):
        """Test that excluded files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            (Path(tmpdir) / "guide.md").write_text("# Guide")
            drafts = Path(tmpdir) / "drafts"
            drafts.mkdir()
            (drafts / "draft.md").write_text("# Draft")

            # Use both patterns: drafts/** for root, **/drafts/** for nested
            config = KnowledgeBaseConfig(
                name="test",
                exclude_patterns=["drafts/**", "**/drafts/**"],
            )
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 1
            assert "guide.md" in docs[0].source_file

    def test_process_uses_pattern_config(self):
        """Test that pattern-specific config is applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON file
            json_file = Path(tmpdir) / "api.json"
            json_data = {"title": "API Doc", "method": "GET", "path": "/users"}
            with open(json_file, "w") as f:
                json.dump(json_data, f)

            config = KnowledgeBaseConfig(
                name="test",
                patterns=[
                    FilePatternConfig(
                        pattern="*.json",
                        text_fields=["title", "method"],
                    ),
                ],
            )
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 1
            assert docs[0].chunk_count == 1

    def test_process_multiple_file_types(self):
        """Test processing multiple file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            (Path(tmpdir) / "guide.md").write_text("# Guide\n\nContent here.")
            with open(Path(tmpdir) / "data.json", "w") as f:
                json.dump({"title": "Data"}, f)

            config = KnowledgeBaseConfig(name="test")
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 2
            doc_types = {d.document_type for d in docs}
            assert "markdown" in doc_types
            assert "json" in doc_types

    def test_process_empty_directory(self):
        """Test processing an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KnowledgeBaseConfig(name="test")
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 0

    def test_process_nested_directories(self):
        """Test processing nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            api_dir = docs_dir / "api"
            api_dir.mkdir()

            (docs_dir / "intro.md").write_text("# Intro")
            (api_dir / "endpoints.md").write_text("# Endpoints")

            config = KnowledgeBaseConfig(name="test")
            processor = DirectoryProcessor(config, tmpdir)

            docs = list(processor.process())

            assert len(docs) == 2


class TestProcessDirectory:
    """Tests for process_directory convenience function."""

    def test_process_directory_with_config(self):
        """Test process_directory with explicit config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test")

            config = KnowledgeBaseConfig(name="custom")
            docs = list(process_directory(tmpdir, config))

            assert len(docs) == 1

    def test_process_directory_auto_loads_config(self):
        """Test process_directory auto-loads config from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_data = {"name": "auto-loaded"}
            with open(Path(tmpdir) / "knowledge_base.json", "w") as f:
                json.dump(config_data, f)

            (Path(tmpdir) / "test.md").write_text("# Test")

            docs = list(process_directory(tmpdir))

            assert len(docs) == 1
