"""Tests for testing utilities."""

import json
from pathlib import Path

import pytest

from dataknobs_common.testing import (
    create_test_json_files,
    create_test_markdown_files,
    get_test_bot_config,
    get_test_rag_config,
    is_chromadb_available,
    is_faiss_available,
    is_ollama_available,
    is_ollama_model_available,
    is_package_available,
    is_redis_available,
    requires_chromadb,
    requires_faiss,
    requires_ollama,
    requires_ollama_model,
    requires_package,
    requires_redis,
)


class TestServiceAvailability:
    """Tests for service availability checks."""

    def test_is_ollama_available_returns_bool(self):
        """Test that is_ollama_available returns a boolean."""
        result = is_ollama_available()
        assert isinstance(result, bool)

    def test_is_ollama_model_available_returns_bool(self):
        """Test that is_ollama_model_available returns a boolean."""
        result = is_ollama_model_available("nomic-embed-text")
        assert isinstance(result, bool)

    def test_is_ollama_model_available_returns_false_if_ollama_unavailable(self):
        """Test that model check returns False if Ollama is not available."""
        # If Ollama is not available, model check should also return False
        if not is_ollama_available():
            assert is_ollama_model_available("any-model") is False

    def test_is_faiss_available_returns_bool(self):
        """Test that is_faiss_available returns a boolean."""
        result = is_faiss_available()
        assert isinstance(result, bool)

    def test_is_chromadb_available_returns_bool(self):
        """Test that is_chromadb_available returns a boolean."""
        result = is_chromadb_available()
        assert isinstance(result, bool)

    def test_is_redis_available_returns_bool(self):
        """Test that is_redis_available returns a boolean."""
        result = is_redis_available()
        assert isinstance(result, bool)

    def test_is_redis_available_with_custom_host_port(self):
        """Test is_redis_available with custom host and port."""
        # Test with unlikely port - should return False
        result = is_redis_available(host="localhost", port=65432)
        assert result is False

    def test_is_package_available_returns_true_for_installed(self):
        """Test that is_package_available returns True for installed packages."""
        # pytest is definitely installed since we're running tests
        assert is_package_available("pytest") is True

    def test_is_package_available_returns_false_for_missing(self):
        """Test that is_package_available returns False for missing packages."""
        assert is_package_available("nonexistent_package_xyz") is False


class TestPytestMarkers:
    """Tests for pytest markers."""

    def test_requires_ollama_is_marker(self):
        """Test that requires_ollama is a valid marker."""
        assert requires_ollama is not None
        # It should be a pytest.mark object
        assert hasattr(requires_ollama, "mark")

    def test_requires_faiss_is_marker(self):
        """Test that requires_faiss is a valid marker."""
        assert requires_faiss is not None
        assert hasattr(requires_faiss, "mark")

    def test_requires_chromadb_is_marker(self):
        """Test that requires_chromadb is a valid marker."""
        assert requires_chromadb is not None
        assert hasattr(requires_chromadb, "mark")

    def test_requires_redis_is_marker(self):
        """Test that requires_redis is a valid marker."""
        assert requires_redis is not None
        assert hasattr(requires_redis, "mark")

    def test_requires_package_returns_marker(self):
        """Test that requires_package returns a marker."""
        marker = requires_package("pytest")
        assert marker is not None
        assert hasattr(marker, "mark")

    def test_requires_ollama_model_returns_marker(self):
        """Test that requires_ollama_model returns a marker."""
        marker = requires_ollama_model("nomic-embed-text")
        assert marker is not None
        assert hasattr(marker, "mark")


class TestBotConfigFactory:
    """Tests for get_test_bot_config factory."""

    def test_default_config(self):
        """Test default configuration."""
        config = get_test_bot_config()

        assert "llm" in config
        assert config["llm"]["provider"] == "echo"
        assert config["llm"]["model"] == "test"
        assert "conversation_storage" in config
        assert config["conversation_storage"]["backend"] == "memory"

    def test_with_real_llm(self):
        """Test configuration with real LLM."""
        config = get_test_bot_config(use_echo_llm=False)

        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["model"] == "gpt-4o-mini"

    def test_with_memory(self):
        """Test configuration with memory enabled."""
        config = get_test_bot_config(include_memory=True)

        assert "memory" in config
        assert config["memory"]["type"] == "buffer"
        assert config["memory"]["max_messages"] == 10

    def test_without_memory(self):
        """Test configuration without memory."""
        config = get_test_bot_config(include_memory=False)

        assert "memory" not in config

    def test_with_system_prompt(self):
        """Test configuration with system prompt."""
        prompt = "You are a helpful assistant."
        config = get_test_bot_config(system_prompt=prompt)

        assert "system_prompt" in config
        assert config["system_prompt"] == prompt

    def test_without_system_prompt(self):
        """Test configuration without system prompt."""
        config = get_test_bot_config()

        assert "system_prompt" not in config

    def test_file_storage(self):
        """Test configuration with file storage."""
        config = get_test_bot_config(use_in_memory_storage=False)

        assert config["conversation_storage"]["backend"] == "file"


class TestRAGConfigFactory:
    """Tests for get_test_rag_config factory."""

    def test_default_config(self):
        """Test default RAG configuration."""
        config = get_test_rag_config()

        assert config["type"] == "rag"
        assert config["vector_store"]["backend"] == "memory"
        assert config["embedding_provider"] == "ollama"
        assert config["embedding_model"] == "nomic-embed-text"
        assert "chunking" in config
        assert "retrieval" in config

    def test_with_faiss_backend(self):
        """Test RAG configuration with FAISS backend."""
        config = get_test_rag_config(use_in_memory_store=False)

        assert config["vector_store"]["backend"] == "faiss"

    def test_with_custom_embedding(self):
        """Test RAG configuration with custom embedding."""
        config = get_test_rag_config(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        assert config["embedding_provider"] == "openai"
        assert config["embedding_model"] == "text-embedding-3-small"

    def test_chunking_config(self):
        """Test that chunking configuration is present."""
        config = get_test_rag_config()

        assert config["chunking"]["max_chunk_size"] == 800
        assert config["chunking"]["chunk_overlap"] == 100

    def test_retrieval_config(self):
        """Test that retrieval configuration is present."""
        config = get_test_rag_config()

        assert config["retrieval"]["top_k"] == 5
        assert config["retrieval"]["score_threshold"] == 0.7


class TestFileHelpers:
    """Tests for file creation helpers."""

    def test_create_test_markdown_files(self, tmp_path: Path):
        """Test creating test markdown files."""
        files = create_test_markdown_files(tmp_path)

        assert len(files) == 2
        assert all(Path(f).exists() for f in files)
        assert all(f.endswith(".md") for f in files)

        # Check content
        for file_path in files:
            content = Path(file_path).read_text()
            assert len(content) > 0
            assert "# " in content  # Contains headers

    def test_create_test_json_files(self, tmp_path: Path):
        """Test creating test JSON files."""
        files = create_test_json_files(tmp_path)

        assert len(files) == 2
        assert all(Path(f).exists() for f in files)
        assert all(f.endswith(".json") for f in files)

        # Check content is valid JSON
        for file_path in files:
            content = Path(file_path).read_text()
            data = json.loads(content)
            assert "title" in data
            assert "items" in data
            assert "metadata" in data

    def test_markdown_files_in_correct_directory(self, tmp_path: Path):
        """Test that markdown files are created in the correct directory."""
        files = create_test_markdown_files(tmp_path)

        for file_path in files:
            assert Path(file_path).parent == tmp_path

    def test_json_files_in_correct_directory(self, tmp_path: Path):
        """Test that JSON files are created in the correct directory."""
        files = create_test_json_files(tmp_path)

        for file_path in files:
            assert Path(file_path).parent == tmp_path
