"""Tests for knowledge resource storage backends."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dataknobs_bots.knowledge.storage import (
    FileKnowledgeBackend,
    IngestionStatus,
    InMemoryKnowledgeBackend,
    KnowledgeBaseInfo,
    KnowledgeFile,
    create_knowledge_backend,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def memory_backend():
    """Create an in-memory backend for testing."""
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def file_backend(tmp_path: Path):
    """Create a file-based backend for testing."""
    backend = FileKnowledgeBackend(base_path=tmp_path / "knowledge")
    await backend.initialize()
    yield backend
    await backend.close()


# ============================================================================
# KnowledgeFile Model Tests
# ============================================================================


class TestKnowledgeFile:
    """Tests for KnowledgeFile dataclass."""

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        now = datetime.now(timezone.utc)
        file = KnowledgeFile(
            path="content/intro.md",
            content_type="text/markdown",
            size_bytes=100,
            checksum="abc123",
            uploaded_at=now,
            metadata={"author": "test"},
        )

        data = file.to_dict()
        assert data["path"] == "content/intro.md"
        assert data["content_type"] == "text/markdown"
        assert data["size_bytes"] == 100
        assert data["checksum"] == "abc123"
        assert data["metadata"] == {"author": "test"}

        restored = KnowledgeFile.from_dict(data)
        assert restored.path == file.path
        assert restored.content_type == file.content_type
        assert restored.size_bytes == file.size_bytes
        assert restored.checksum == file.checksum
        assert restored.metadata == file.metadata


class TestKnowledgeBaseInfo:
    """Tests for KnowledgeBaseInfo dataclass."""

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        now = datetime.now(timezone.utc)
        info = KnowledgeBaseInfo(
            domain_id="test-domain",
            file_count=5,
            total_size_bytes=1000,
            last_updated=now,
            version="1",
            ingestion_status=IngestionStatus.READY,
            metadata={"description": "Test KB"},
        )

        data = info.to_dict()
        assert data["domain_id"] == "test-domain"
        assert data["file_count"] == 5
        assert data["ingestion_status"] == "ready"

        restored = KnowledgeBaseInfo.from_dict(data)
        assert restored.domain_id == info.domain_id
        assert restored.file_count == info.file_count
        assert restored.ingestion_status == IngestionStatus.READY

    def test_increment_version(self):
        """Test version incrementing."""
        info = KnowledgeBaseInfo(
            domain_id="test",
            file_count=0,
            total_size_bytes=0,
            last_updated=datetime.now(timezone.utc),
            version="1",
        )

        info.increment_version()
        assert info.version == "2"

        info.increment_version()
        assert info.version == "3"


# ============================================================================
# InMemoryKnowledgeBackend Tests
# ============================================================================


class TestInMemoryKnowledgeBackend:
    """Tests for InMemoryKnowledgeBackend."""

    async def test_create_kb(self, memory_backend: InMemoryKnowledgeBackend):
        """Test creating a knowledge base."""
        info = await memory_backend.create_kb("test-domain", metadata={"key": "value"})

        assert info.domain_id == "test-domain"
        assert info.file_count == 0
        assert info.total_size_bytes == 0
        assert info.version == "1"
        assert info.ingestion_status == IngestionStatus.PENDING
        assert info.metadata == {"key": "value"}

    async def test_create_kb_duplicate_raises(self, memory_backend: InMemoryKnowledgeBackend):
        """Test that creating duplicate KB raises error."""
        await memory_backend.create_kb("test-domain")

        with pytest.raises(ValueError, match="already exists"):
            await memory_backend.create_kb("test-domain")

    async def test_get_info(self, memory_backend: InMemoryKnowledgeBackend):
        """Test getting KB info."""
        await memory_backend.create_kb("test-domain")

        info = await memory_backend.get_info("test-domain")
        assert info is not None
        assert info.domain_id == "test-domain"

        # Non-existent KB returns None
        info = await memory_backend.get_info("non-existent")
        assert info is None

    async def test_list_kbs(self, memory_backend: InMemoryKnowledgeBackend):
        """Test listing all knowledge bases."""
        await memory_backend.create_kb("domain-a")
        await memory_backend.create_kb("domain-b")
        await memory_backend.create_kb("domain-c")

        kbs = await memory_backend.list_kbs()
        assert len(kbs) == 3
        # Should be sorted by domain_id
        assert [kb.domain_id for kb in kbs] == ["domain-a", "domain-b", "domain-c"]

    async def test_delete_kb(self, memory_backend: InMemoryKnowledgeBackend):
        """Test deleting a knowledge base."""
        await memory_backend.create_kb("test-domain")
        await memory_backend.put_file("test-domain", "test.md", b"# Hello")

        result = await memory_backend.delete_kb("test-domain")
        assert result is True

        info = await memory_backend.get_info("test-domain")
        assert info is None

        # Deleting non-existent returns False
        result = await memory_backend.delete_kb("non-existent")
        assert result is False

    async def test_put_and_get_file(self, memory_backend: InMemoryKnowledgeBackend):
        """Test uploading and retrieving files."""
        await memory_backend.create_kb("test-domain")

        # Upload file with explicit content type
        content = b"# Hello World\n\nThis is a test."
        file_info = await memory_backend.put_file(
            "test-domain",
            "content/intro.md",
            content,
            content_type="text/markdown",
            metadata={"author": "test"},
        )

        assert file_info.path == "content/intro.md"
        assert file_info.content_type == "text/markdown"
        assert file_info.size_bytes == len(content)
        assert file_info.checksum is not None
        assert file_info.metadata == {"author": "test"}

        # Retrieve file
        retrieved = await memory_backend.get_file("test-domain", "content/intro.md")
        assert retrieved == content

        # Non-existent file returns None
        result = await memory_backend.get_file("test-domain", "non-existent.md")
        assert result is None

    async def test_put_file_without_kb_raises(self, memory_backend: InMemoryKnowledgeBackend):
        """Test that putting file without KB raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            await memory_backend.put_file("non-existent", "test.md", b"content")

    async def test_list_files(self, memory_backend: InMemoryKnowledgeBackend):
        """Test listing files in a KB."""
        await memory_backend.create_kb("test-domain")
        await memory_backend.put_file("test-domain", "a.md", b"A")
        await memory_backend.put_file("test-domain", "b.md", b"B")
        await memory_backend.put_file("test-domain", "subdir/c.md", b"C")

        files = await memory_backend.list_files("test-domain")
        assert len(files) == 3
        paths = [f.path for f in files]
        assert "a.md" in paths
        assert "b.md" in paths
        assert "subdir/c.md" in paths

        # Filter by prefix
        files = await memory_backend.list_files("test-domain", prefix="subdir/")
        assert len(files) == 1
        assert files[0].path == "subdir/c.md"

    async def test_file_exists(self, memory_backend: InMemoryKnowledgeBackend):
        """Test checking file existence."""
        await memory_backend.create_kb("test-domain")
        await memory_backend.put_file("test-domain", "exists.md", b"content")

        assert await memory_backend.file_exists("test-domain", "exists.md") is True
        assert await memory_backend.file_exists("test-domain", "not-exists.md") is False

    async def test_delete_file(self, memory_backend: InMemoryKnowledgeBackend):
        """Test deleting a file."""
        await memory_backend.create_kb("test-domain")
        await memory_backend.put_file("test-domain", "test.md", b"content")

        result = await memory_backend.delete_file("test-domain", "test.md")
        assert result is True
        assert await memory_backend.file_exists("test-domain", "test.md") is False

        # KB info should be updated
        info = await memory_backend.get_info("test-domain")
        assert info.file_count == 0

        # Deleting non-existent returns False
        result = await memory_backend.delete_file("test-domain", "non-existent.md")
        assert result is False

    async def test_stream_file(self, memory_backend: InMemoryKnowledgeBackend):
        """Test streaming file content."""
        await memory_backend.create_kb("test-domain")
        content = b"A" * 1000
        await memory_backend.put_file("test-domain", "large.bin", content)

        stream = await memory_backend.stream_file("test-domain", "large.bin", chunk_size=100)
        assert stream is not None

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert b"".join(chunks) == content
        assert len(chunks) == 10  # 1000 bytes / 100 chunk_size

    async def test_set_ingestion_status(self, memory_backend: InMemoryKnowledgeBackend):
        """Test updating ingestion status."""
        await memory_backend.create_kb("test-domain")

        await memory_backend.set_ingestion_status("test-domain", "ingesting")
        info = await memory_backend.get_info("test-domain")
        assert info.ingestion_status == IngestionStatus.INGESTING

        await memory_backend.set_ingestion_status("test-domain", "error", "Something failed")
        info = await memory_backend.get_info("test-domain")
        assert info.ingestion_status == IngestionStatus.ERROR
        assert info.ingestion_error == "Something failed"

    async def test_checksum_and_change_detection(
        self, memory_backend: InMemoryKnowledgeBackend
    ):
        """Test checksum calculation and change detection."""
        await memory_backend.create_kb("test-domain")

        # Empty KB has empty checksum
        checksum = await memory_backend.get_checksum("test-domain")
        assert checksum == ""

        # Add file
        await memory_backend.put_file("test-domain", "test.md", b"content")
        checksum1 = await memory_backend.get_checksum("test-domain")
        assert checksum1 != ""

        # Get version
        info = await memory_backend.get_info("test-domain")
        version = info.version

        # No changes since current version
        has_changes = await memory_backend.has_changes_since("test-domain", version)
        assert has_changes is False

        # Add another file
        await memory_backend.put_file("test-domain", "test2.md", b"content2")

        # Changes since previous version
        has_changes = await memory_backend.has_changes_since("test-domain", version)
        assert has_changes is True

        # Checksum changed
        checksum2 = await memory_backend.get_checksum("test-domain")
        assert checksum2 != checksum1


# ============================================================================
# FileKnowledgeBackend Tests
# ============================================================================


class TestFileKnowledgeBackend:
    """Tests for FileKnowledgeBackend."""

    async def test_create_kb(self, file_backend: FileKnowledgeBackend):
        """Test creating a knowledge base."""
        info = await file_backend.create_kb("test-domain")

        assert info.domain_id == "test-domain"
        assert info.file_count == 0

        # Directory should be created
        kb_path = file_backend._kb_path("test-domain")
        assert kb_path.exists()
        assert (kb_path / "_metadata.json").exists()
        assert (kb_path / "content").exists()

    async def test_put_and_get_file(self, file_backend: FileKnowledgeBackend):
        """Test uploading and retrieving files."""
        await file_backend.create_kb("test-domain")

        content = b"# Test Content"
        file_info = await file_backend.put_file(
            "test-domain",
            "docs/intro.md",
            content,
        )

        assert file_info.path == "docs/intro.md"
        assert file_info.size_bytes == len(content)

        # File should exist on disk
        file_path = file_backend._file_path("test-domain", "docs/intro.md")
        assert file_path.exists()
        assert file_path.read_bytes() == content

        # Retrieve via API
        retrieved = await file_backend.get_file("test-domain", "docs/intro.md")
        assert retrieved == content

    async def test_delete_file(self, file_backend: FileKnowledgeBackend):
        """Test deleting a file."""
        await file_backend.create_kb("test-domain")
        await file_backend.put_file("test-domain", "test.md", b"content")

        result = await file_backend.delete_file("test-domain", "test.md")
        assert result is True

        # File should be removed from disk
        file_path = file_backend._file_path("test-domain", "test.md")
        assert not file_path.exists()

    async def test_list_files(self, file_backend: FileKnowledgeBackend):
        """Test listing files."""
        await file_backend.create_kb("test-domain")
        await file_backend.put_file("test-domain", "a.md", b"A")
        await file_backend.put_file("test-domain", "subdir/b.md", b"B")

        files = await file_backend.list_files("test-domain")
        assert len(files) == 2
        paths = [f.path for f in files]
        assert "a.md" in paths
        assert "subdir/b.md" in paths

    async def test_delete_kb(self, file_backend: FileKnowledgeBackend):
        """Test deleting entire KB."""
        await file_backend.create_kb("test-domain")
        await file_backend.put_file("test-domain", "test.md", b"content")

        kb_path = file_backend._kb_path("test-domain")
        assert kb_path.exists()

        result = await file_backend.delete_kb("test-domain")
        assert result is True
        assert not kb_path.exists()

    async def test_list_kbs(self, file_backend: FileKnowledgeBackend):
        """Test listing all KBs."""
        await file_backend.create_kb("kb-a")
        await file_backend.create_kb("kb-b")

        kbs = await file_backend.list_kbs()
        assert len(kbs) == 2
        domain_ids = [kb.domain_id for kb in kbs]
        assert "kb-a" in domain_ids
        assert "kb-b" in domain_ids


# ============================================================================
# Factory Tests
# ============================================================================


class TestCreateKnowledgeBackend:
    """Tests for create_knowledge_backend factory."""

    def test_create_memory_backend(self):
        """Test creating memory backend."""
        backend = create_knowledge_backend("memory")
        assert isinstance(backend, InMemoryKnowledgeBackend)

    def test_create_file_backend(self, tmp_path: Path):
        """Test creating file backend."""
        backend = create_knowledge_backend("file", {"path": str(tmp_path / "kb")})
        assert isinstance(backend, FileKnowledgeBackend)

    def test_case_insensitive(self):
        """Test that type is case-insensitive."""
        backend1 = create_knowledge_backend("MEMORY")
        backend2 = create_knowledge_backend("Memory")
        assert isinstance(backend1, InMemoryKnowledgeBackend)
        assert isinstance(backend2, InMemoryKnowledgeBackend)

    def test_unknown_type_raises(self):
        """Test that unknown type raises error."""
        with pytest.raises(ValueError, match="Unknown knowledge backend type"):
            create_knowledge_backend("unknown")
